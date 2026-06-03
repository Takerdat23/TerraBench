import hashlib
import json
import os
import secrets
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Optional, Union


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UploadManager:
    """Stores uploaded files on disk and exposes Docker volume bindings."""

    def __init__(
        self,
        storage_dir: Union[Path, str] = "uploads",
        container_mount: str = "/uploads",
        *,
        ttl_seconds: Optional[int] = 86_400,
        max_file_size_bytes: int = 512 * 1024 * 1024,
        delete_after_use: bool = False,
    ):
        self.storage_dir = Path(storage_dir).expanduser().resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.container_mount = PurePosixPath(container_mount)
        if not str(self.container_mount):
            raise ValueError("container_mount cannot be empty")
        self.ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None
        self.max_file_size_bytes = max_file_size_bytes
        self.delete_after_use = delete_after_use
        self.chunk_size = 1024 * 1024  # 1MB default chunks
        self._records: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_existing_records()

    @property
    def volume_mapping(self) -> Dict[str, Dict[str, str]]:
        """Docker volume mapping to mount uploads inside the execution container."""
        return {str(self.storage_dir): {"bind": str(self.container_mount), "mode": "rw"}}

    def save_stream(
        self,
        stream: BinaryIO,
        *,
        filename: Optional[str],
        content_type: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Persist an uploaded file from a binary stream.

        Returns a record containing identifiers and metadata for later lookups.
        """
        safe_name = Path(filename or "upload.bin").name
        file_id = secrets.token_hex(16)
        target_dir = self.storage_dir / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        sha256 = hashlib.sha256()
        total_bytes = 0
        # reset pointer if possible
        try:
            stream.seek(0)
        except (AttributeError, OSError):
            pass

        try:
            with target_path.open("wb") as destination:
                while True:
                    chunk = stream.read(self.chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if self.max_file_size_bytes and total_bytes > self.max_file_size_bytes:
                        raise ValueError(
                            f"Upload exceeds max size of {self.max_file_size_bytes} bytes"
                        )
                    sha256.update(chunk)
                    destination.write(chunk)
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise

        created_at = _now()
        expires_at = created_at + self.ttl if self.ttl else None
        record = {
            "file_id": file_id,
            "filename": safe_name,
            "content_type": content_type,
            "size_bytes": total_bytes,
            "sha256": sha256.hexdigest(),
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "metadata": metadata or {},
            "host_path": str(target_path),
            "container_path": str(self.container_mount / file_id / safe_name),
            "last_accessed": created_at.isoformat(),
        }
        with self._lock:
            self._records[file_id] = record
            self._write_metadata(record)
        return record.copy()

    def get_record(self, file_id: str) -> Dict[str, Any]:
        """Return a copy of the upload record, ensuring it has not expired."""
        with self._lock:
            record = self._records.get(file_id)
            if not record:
                raise KeyError(f"Unknown upload ID: {file_id}")
            if self._is_expired(record):
                self._delete_locked(file_id)
                raise KeyError(f"Upload {file_id} has expired")
            record["last_accessed"] = _now().isoformat()
            self._write_metadata(record)
            return record.copy()

    def delete(self, file_id: str) -> None:
        with self._lock:
            self._delete_locked(file_id)

    def purge_expired(self) -> int:
        """Remove expired uploads; returns number of deletions."""
        removed = 0
        with self._lock:
            for file_id in list(self._records.keys()):
                if self._is_expired(self._records[file_id]):
                    self._delete_locked(file_id)
                    removed += 1
        return removed

    def _is_expired(self, record: Dict[str, Any]) -> bool:
        if not self.ttl or not record.get("expires_at"):
            return False
        expires_at = datetime.fromisoformat(record["expires_at"])
        return expires_at < _now()

    def _delete_locked(self, file_id: str) -> None:
        record = self._records.pop(file_id, None)
        if not record:
            return
        target_dir = Path(record["host_path"]).parent
        shutil.rmtree(target_dir, ignore_errors=True)

    def _load_existing_records(self) -> None:
        for meta_path in self.storage_dir.glob("*/metadata.json"):
            try:
                with meta_path.open("r") as handle:
                    record = json.load(handle)
                file_id = record.get("file_id")
                if file_id:
                    self._records[file_id] = record
            except (OSError, json.JSONDecodeError):
                continue
        self.purge_expired()

    def _write_metadata(self, record: Dict[str, Any]) -> None:
        meta_path = Path(record["host_path"]).parent / "metadata.json"
        tmp_path = meta_path.with_suffix(".tmp")
        with tmp_path.open("w") as handle:
            json.dump(record, handle, indent=2)
        os.replace(tmp_path, meta_path)

    def _hash_file(self, path: Path) -> tuple[int, str]:
        """Compute size and sha256 for an existing file."""
        sha256 = hashlib.sha256()
        total_bytes = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(self.chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                sha256.update(chunk)
        return total_bytes, sha256.hexdigest()

    def register_existing_file(
        self,
        host_path: Union[Path, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a file that already exists under the uploads directory.
        The file is moved into a new record directory and assigned a file_id so it can be served.
        """
        path = Path(host_path).resolve()
        try:
            path.relative_to(self.storage_dir)
        except ValueError as exc:
            raise ValueError(f"Path must be inside uploads directory: {host_path}") from exc
        if not path.is_file():
            raise ValueError(f"Path must be a file: {host_path}")

        file_id = secrets.token_hex(16)
        target_dir = self.storage_dir / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name

        if path != target_path:
            shutil.move(str(path), target_path)

        size_bytes, sha = self._hash_file(target_path)
        created_at = _now()
        expires_at = created_at + self.ttl if self.ttl else None
        record = {
            "file_id": file_id,
            "filename": target_path.name,
            "content_type": None,
            "size_bytes": size_bytes,
            "sha256": sha,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "metadata": metadata or {},
            "host_path": str(target_path),
            "container_path": str(self.container_mount / file_id / target_path.name),
            "last_accessed": created_at.isoformat(),
        }
        with self._lock:
            self._records[file_id] = record
            self._write_metadata(record)
        return record.copy()
