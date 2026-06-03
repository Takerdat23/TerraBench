"""
Web utilities (search, crawl, summarize) consolidated.
"""

import json
import os
import hashlib
import mimetypes
import time
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import certifi
import requests
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from dotenv import load_dotenv
from terra_agent.full_agent.utils import (
    _is_unsupported_openai_reasoning_error,
    _resolve_openai_reasoning_effort,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
_COMBINED_BUNDLE_PATH = Path("/tmp/serper_combined.pem")
_CA_ENV_KEYS: tuple[str, ...] = ("WEB_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
_VERIFY_FLAG_ENV = "WEB_VERIFY"

SERPER_ENDPOINTS: Dict[str, str] = {
    "search": "https://google.serper.dev/search",
    "news": "https://google.serper.dev/news",
    "places": "https://google.serper.dev/places",
}

DEFAULT_MAX_PAGES = 3
DEFAULT_MAX_CHARS_PER_PAGE = 6000
DEFAULT_MAX_PDF_PAGES = 10
DEFAULT_MAX_PDF_CHARS = 8000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_SUMMARY_MODEL_ENV = "WEB_SUMMARY_MODEL"
DEFAULT_MODEL_FALLBACK = "gpt-4o-mini"
DEFAULT_CLAUDE_FALLBACK = "claude-3-5-sonnet-20241022"
SUMMARY_INSTRUCTIONS_ENV = "WEB_SUMMARY_INSTRUCTIONS"
SUMMARY_PROMPT_PATH_ENV = "WEB_SUMMARY_PROMPT_PATH"
SUMMARY_OPENAI_BASE_URL_ENV = "WEB_SUMMARY_OPENAI_BASE_URL"
SUMMARY_ANTHROPIC_BASE_URL_ENV = "WEB_SUMMARY_ANTHROPIC_BASE_URL"
SUMMARY_BASE_URL_ENV = "WEB_SUMMARY_BASE_URL"
SUMMARY_USE_CRAWLER_ENV = "WEB_SUMMARY_USE_CRAWLER"
SUMMARY_CRAWL_MAX_PAGES_ENV = "WEB_SUMMARY_CRAWL_MAX_PAGES"
SUMMARY_CRAWL_MAX_CHARS_PER_PAGE_ENV = "WEB_SUMMARY_CRAWL_MAX_CHARS_PER_PAGE"
SUMMARY_CRAWL_SAME_DOMAIN_ENV = "WEB_SUMMARY_CRAWL_SAME_DOMAIN"
SUMMARY_FALLBACK_RAW_URL_ENV = "WEB_SUMMARY_FALLBACK_RAW_URL"
SUMMARY_USE_JINA_FALLBACK_ENV = "WEB_SUMMARY_USE_JINA_FALLBACK"
WEB_ARTIFACT_ROOT_ENV = "WEB_ARTIFACT_ROOT"
WEB_ARCHIVE_DEFAULT_ENV = "WEB_ARCHIVE_DEFAULT"
WEB_ARCHIVE_MAX_BYTES_ENV = "WEB_ARCHIVE_MAX_BYTES"
WEB_ARCHIVE_TIMEOUT_ENV = "WEB_ARCHIVE_TIMEOUT"
WEB_ARTIFACT_MAX_TEXT_CHARS_ENV = "WEB_ARTIFACT_MAX_TEXT_CHARS"
WEB_ARTIFACT_MAX_PDF_PAGES_ENV = "WEB_ARTIFACT_MAX_PDF_PAGES"
DEFAULT_SUMMARY_MAX_TOKENS = 500
DEFAULT_SUMMARY_INSTRUCTIONS = (
    "Summarize only the key information needed to answer the user's question. "
    "Do not provide a full document summary. Use concise bullet points and "
    "prioritize facts, figures, definitions, and constraints that affect the answer. "
    "If the page does not contain relevant information, say so."
)
SUMMARY_FOCUS_PREFIX = (
    "Summarize only the key information needed to answer the user's question. "
    "Do not provide a full document summary."
)
DEFAULT_WEB_ARTIFACT_ROOT = BASE_DIR / "dataset" / "web_artifacts"
DEFAULT_WEB_ARCHIVE_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_WEB_ARCHIVE_TIMEOUT = 20
DEFAULT_WEB_ARTIFACT_MAX_TEXT_CHARS = 200_000
DEFAULT_WEB_ARTIFACT_MAX_PDF_PAGES = 200
DEFAULT_SEARCH_OVERFETCH_FACTOR = 2
_HIGH_SIGNAL_URL_HINTS: tuple[str, ...] = (
    ".pdf",
    "/report",
    "/reports",
    "/chapter",
    "/article",
    "/paper",
    "/study",
    "/publication",
    "/publications",
    "/bulletin",
    "/outlook",
    "/analysis",
    "/docs",
    "/download",
)
_LOW_SIGNAL_TEXT_HINTS: tuple[str, ...] = (
    "home",
    "homepage",
    "main menu",
    "navigation",
    "sign in",
    "log in",
    "subscribe",
    "cookie policy",
    "privacy policy",
    "terms of use",
    "about us",
    "contact us",
)
_NAV_ATTR_HINTS: tuple[str, ...] = (
    "nav",
    "navbar",
    "navigation",
    "menu",
    "footer",
    "header",
    "breadcrumb",
    "cookie",
    "consent",
    "topbar",
    "site-header",
    "site-footer",
)
_NAV_ROLES: tuple[str, ...] = ("navigation", "banner", "contentinfo", "complementary", "search")


def _resolve_verify_setting() -> Any:
    """
    Build a verify setting for requests, merging the default certifi bundle
    with any user-provided CA files if present.
    """
    flag = os.getenv(_VERIFY_FLAG_ENV)
    if isinstance(flag, str) and flag.lower() in {"0", "false", "no", "off"}:
        return False

    env_values: List[str] = []
    ca_paths: List[Path] = []
    for env_key in _CA_ENV_KEYS:
        value = os.getenv(env_key)
        if not value:
            continue
        env_values.append(value)
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            candidate = BASE_DIR / expanded
            if candidate.exists():
                expanded = candidate
        if expanded.exists():
            ca_paths.append(expanded)

    if ca_paths:
        try:
            combined_parts = [Path(certifi.where()).read_text()]
            for path in ca_paths:
                try:
                    combined_parts.append(path.read_text())
                except FileNotFoundError:
                    continue
            _COMBINED_BUNDLE_PATH.write_text("\n".join(combined_parts))
            return str(_COMBINED_BUNDLE_PATH)
        except Exception:
            return str(ca_paths[0])

    if env_values:
        return env_values[0]

    return True


def _error(message: str, *, status_code: Optional[int] = None) -> ToolResponse:
    metadata: Dict[str, Any] = {"error": True, "message": message}
    if status_code is not None:
        metadata["status_code"] = status_code
    return ToolResponse(
        content=[TextBlock(type="text", text=f"Error: {message}")],
        metadata=metadata,
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_artifact_root(artifact_root: Optional[str]) -> Path:
    root_value = (
        artifact_root
        or os.getenv(WEB_ARTIFACT_ROOT_ENV)
        or str(DEFAULT_WEB_ARTIFACT_ROOT)
    )
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _env_int_with_cap(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _resolve_archive_settings(
    *,
    request_timeout: Optional[int] = None,
) -> tuple[int, int, int, int]:
    archive_timeout = request_timeout or _env_int_with_cap(
        WEB_ARCHIVE_TIMEOUT_ENV, DEFAULT_WEB_ARCHIVE_TIMEOUT
    )
    max_bytes = _env_int_with_cap(
        WEB_ARCHIVE_MAX_BYTES_ENV, DEFAULT_WEB_ARCHIVE_MAX_BYTES
    )
    max_text_chars = _env_int_with_cap(
        WEB_ARTIFACT_MAX_TEXT_CHARS_ENV, DEFAULT_WEB_ARTIFACT_MAX_TEXT_CHARS
    )
    max_pdf_pages = _env_int_with_cap(
        WEB_ARTIFACT_MAX_PDF_PAGES_ENV, DEFAULT_WEB_ARTIFACT_MAX_PDF_PAGES
    )
    return archive_timeout, max_bytes, max_text_chars, max_pdf_pages


def _guess_extension(url: str, content_type: Optional[str]) -> str:
    path_suffix = Path(urlparse(url).path).suffix.strip().lower()
    if path_suffix and len(path_suffix) <= 10:
        return path_suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type) or ""
        if guessed:
            return guessed
    return ".bin"


def _content_addressed_path(root: Path, kind: str, digest: str, suffix: str) -> Path:
    return root / kind / digest[:2] / f"{digest}{suffix}"


def _write_if_missing(path: Path, data: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_json_if_missing(path: Path, payload: Dict[str, Any]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_manifest(root: Path, entry: Dict[str, Any]) -> None:
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False))
        handle.write("\n")


def _persist_text_artifact(
    text: str,
    *,
    artifact_root: Optional[str],
    kind: str,
) -> Dict[str, Any]:
    root = _resolve_artifact_root(artifact_root)
    data = (text or "").encode("utf-8")
    digest = _sha256_bytes(data)
    path = _content_addressed_path(root, kind, digest, ".txt")
    _write_if_missing(path, data)
    return {
        "path": str(path),
        "sha256": digest,
        "chars": len(text or ""),
    }


def _extract_plain_text_from_payload(
    *,
    url: str,
    content_type: Optional[str],
    data: bytes,
    encoding: Optional[str] = None,
    max_chars: int = DEFAULT_WEB_ARTIFACT_MAX_TEXT_CHARS,
    max_pdf_pages: int = DEFAULT_WEB_ARTIFACT_MAX_PDF_PAGES,
) -> tuple[str, bool, Optional[str]]:
    normalized_type = _normalize_content_type(content_type or "")
    if _looks_like_pdf(url, normalized_type, data):
        text, truncated, _, error = _extract_pdf_text(
            data,
            max_pages=max_pdf_pages,
            max_chars=max_chars,
        )
        return text, truncated, error

    decoded = ""
    if _is_html_type(normalized_type) or url.lower().endswith((".html", ".htm")):
        html = data.decode(encoding or "utf-8", errors="ignore")
        parser = _PlainTextParser()
        parser.feed(html)
        parser.close()
        decoded = parser.get_text()
    elif _is_text_type(normalized_type) or url.lower().endswith(
        (".txt", ".csv", ".json", ".xml", ".md")
    ):
        decoded = data.decode(encoding or "utf-8", errors="ignore")
    elif not normalized_type:
        decoded = data.decode(encoding or "utf-8", errors="ignore")

    if not decoded:
        return "", False, None
    truncated = len(decoded) > max_chars
    return decoded[:max_chars], truncated, None


def _archive_payload(
    *,
    source_url: str,
    final_url: str,
    status_code: int,
    content_type: Optional[str],
    data: bytes,
    artifact_root: Optional[str],
    extracted_text: Optional[str] = None,
    text_truncated: bool = False,
    text_error: Optional[str] = None,
    encoding: Optional[str] = None,
    max_text_chars: Optional[int] = None,
    max_pdf_pages: Optional[int] = None,
) -> Dict[str, Any]:
    root = _resolve_artifact_root(artifact_root)
    _, _, resolved_max_text_chars, resolved_max_pdf_pages = _resolve_archive_settings()
    max_chars = max_text_chars or resolved_max_text_chars
    pdf_pages = max_pdf_pages or resolved_max_pdf_pages
    normalized_type = _normalize_content_type(content_type or "")

    file_digest = _sha256_bytes(data)
    suffix = _guess_extension(final_url, normalized_type)
    raw_path = _content_addressed_path(root, "raw", file_digest, suffix)
    _write_if_missing(raw_path, data)

    text_path: Optional[Path] = None
    text_digest: Optional[str] = None
    text_size_bytes: Optional[int] = None
    text_len = 0

    plain_text = extracted_text
    plain_text_truncated = text_truncated
    plain_text_error = text_error
    if plain_text is None and not plain_text_error:
        plain_text, plain_text_truncated, plain_text_error = _extract_plain_text_from_payload(
            url=final_url,
            content_type=normalized_type,
            data=data,
            encoding=encoding,
            max_chars=max_chars,
            max_pdf_pages=pdf_pages,
        )

    if plain_text:
        text_bytes = plain_text.encode("utf-8")
        text_digest = _sha256_bytes(text_bytes)
        text_path = _content_addressed_path(root, "text", text_digest, ".txt")
        _write_if_missing(text_path, text_bytes)
        text_size_bytes = len(text_bytes)
        text_len = len(plain_text)

    entry: Dict[str, Any] = {
        "source_url": source_url,
        "final_url": final_url,
        "retrieved_at": _utc_now(),
        "status_code": int(status_code),
        "content_type": normalized_type or None,
        "size_bytes": len(data),
        "sha256": file_digest,
        "artifact_path": str(raw_path),
        "text_path": str(text_path) if text_path else None,
        "text_sha256": text_digest,
        "text_size_bytes": text_size_bytes,
        "text_length_chars": text_len,
        "text_truncated": bool(plain_text_truncated),
        "text_error": plain_text_error,
    }

    metadata_path = _content_addressed_path(root, "meta", file_digest, ".json")
    _write_json_if_missing(metadata_path, entry)
    entry["metadata_path"] = str(metadata_path)
    _append_manifest(root, entry)
    return entry


def _archive_url(
    url: str,
    *,
    artifact_root: Optional[str] = None,
    request_timeout: Optional[int] = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    timeout, max_bytes, max_text_chars, max_pdf_pages = _resolve_archive_settings(
        request_timeout=request_timeout
    )
    normalized = _normalize_url(url)
    if not normalized:
        return {"source_url": url, "error": "URL is empty."}
    if not _is_http_url(normalized):
        return {"source_url": normalized, "error": "URL must start with http:// or https://"}

    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(
            normalized,
            headers=headers,
            timeout=timeout,
            verify=_resolve_verify_setting(),
        )
    except requests.RequestException as exc:
        return {"source_url": normalized, "error": f"Request failed: {exc}"}

    if resp.status_code >= 400:
        return {
            "source_url": normalized,
            "final_url": resp.url or normalized,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code} error.",
        }

    data = resp.content or b""
    if len(data) > max_bytes:
        return {
            "source_url": normalized,
            "final_url": resp.url or normalized,
            "status_code": resp.status_code,
            "size_bytes": len(data),
            "error": f"Artifact exceeds max size ({max_bytes} bytes).",
        }

    return _archive_payload(
        source_url=normalized,
        final_url=resp.url or normalized,
        status_code=resp.status_code,
        content_type=resp.headers.get("Content-Type"),
        data=data,
        artifact_root=artifact_root,
        encoding=resp.encoding,
        max_text_chars=max_text_chars,
        max_pdf_pages=max_pdf_pages,
    )


def _collect_serper_urls(payload: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()

    def add(candidate: Any) -> None:
        if not isinstance(candidate, str):
            return
        value = candidate.strip()
        if not value or value in seen or not _is_http_url(value):
            return
        seen.add(value)
        urls.append(value)

    answer_box = payload.get("answerBox")
    if isinstance(answer_box, dict):
        add(answer_box.get("link"))

    for list_key in ("organic", "news", "places", "images", "peopleAlsoAsk"):
        entries = payload.get(list_key)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if isinstance(item, dict):
                add(item.get("link"))

    knowledge_graph = payload.get("knowledgeGraph")
    if isinstance(knowledge_graph, dict):
        add(knowledge_graph.get("website"))

    return urls


def _looks_like_landing_result(entry: Dict[str, Any]) -> bool:
    link = str(entry.get("link") or "").strip()
    if not link or not _is_http_url(link):
        return False

    parsed = urlparse(link)
    path = (parsed.path or "/").lower().strip()
    if path.endswith("/"):
        path = path[:-1] or "/"

    text_blob = " ".join(
        str(part).strip().lower()
        for part in (entry.get("title"), entry.get("snippet"))
        if isinstance(part, str) and part.strip()
    )

    # Keep clearly substantive document/report URLs.
    if any(hint in link.lower() for hint in _HIGH_SIGNAL_URL_HINTS):
        return False

    score = 0
    if path in {"/", "/home", "/index", "/index.html", "/about", "/contact"}:
        score += 2
    if path.count("/") <= 1 and "." not in path.replace("/", ""):
        score += 1
    if len(str(entry.get("snippet") or "").strip()) < 80:
        score += 1
    for hint in _LOW_SIGNAL_TEXT_HINTS:
        if hint in text_blob:
            score += 1
            if score >= 3:
                break
    return score >= 3


def _filter_serper_results(payload: Dict[str, Any], *, requested_count: int) -> Dict[str, Any]:
    filtered_payload = dict(payload)
    filtered_out: List[Dict[str, str]] = []

    for key in ("organic", "news"):
        items = filtered_payload.get(key)
        if not isinstance(items, list):
            continue

        kept: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if _looks_like_landing_result(item):
                filtered_out.append(
                    {
                        "section": key,
                        "title": str(item.get("title") or ""),
                        "link": str(item.get("link") or ""),
                    }
                )
                continue
            kept.append(item)

        filtered_payload[key] = kept[:requested_count]

    filtered_payload["filtered_out_results"] = filtered_out
    filtered_payload["filtered_out_count"] = len(filtered_out)
    return filtered_payload


# ----- Web search ------------------------------------------------------------
def web_search_serper(
    query: str,
    *,
    search_type: str = "search",
    num_results: int = 10,
    language: Optional[str] = None,
    country: Optional[str] = None,
    filter_landing_pages: bool = True,
    overfetch_factor: int = DEFAULT_SEARCH_OVERFETCH_FACTOR,
    archive_results: Optional[bool] = None,
    archive_limit: int = 5,
    artifact_root: Optional[str] = None,
    archive_timeout: Optional[int] = None,
) -> ToolResponse:
    """
    Run a Serper API web search and return structured results.
    Optionally archive result URLs into local immutable artifacts.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return _error("SERPER_API_KEY is not set in the environment")

    endpoint = SERPER_ENDPOINTS.get(search_type.lower())
    if not endpoint:
        valid = ", ".join(sorted(SERPER_ENDPOINTS))
        return _error(f"Unsupported search_type '{search_type}'. Valid options: {valid}")

    clean_query = (query or "").strip()
    if not clean_query:
        return _error("Query must be a non-empty string")

    capped_results = max(1, min(int(num_results), 20))
    fetch_results = capped_results
    if filter_landing_pages:
        fetch_results = max(
            capped_results,
            min(20, capped_results * max(1, int(overfetch_factor))),
        )
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    body: Dict[str, Any] = {"q": clean_query, "num": fetch_results}
    if language:
        body["hl"] = language
    if country:
        body["gl"] = country

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=15,
            verify=_resolve_verify_setting(),
        )
    except requests.RequestException as exc:
        return _error(f"Serper request failed: {exc}")

    if response.status_code != 200:
        detail = response.text[:500] if response.text else "no response body"
        return _error(
            f"Serper returned status {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        payload = response.json()
    except ValueError:
        return _error("Failed to parse Serper response as JSON", status_code=response.status_code)

    if filter_landing_pages:
        payload = _filter_serper_results(payload, requested_count=capped_results)
    payload["landing_page_filter_enabled"] = bool(filter_landing_pages)
    payload["requested_num_results"] = capped_results
    payload["fetched_num_results"] = fetch_results

    should_archive = archive_results
    if should_archive is None:
        should_archive = _env_flag(WEB_ARCHIVE_DEFAULT_ENV, True)

    archive_entries: List[Dict[str, Any]] = []
    if should_archive:
        urls = _collect_serper_urls(payload)
        capped = max(0, int(archive_limit))
        for url_value in urls[:capped]:
            archive_entries.append(
                _archive_url(
                    url_value,
                    artifact_root=artifact_root,
                    request_timeout=archive_timeout,
                )
            )
        payload["archived_artifacts"] = archive_entries
        payload["archive_root"] = str(_resolve_artifact_root(artifact_root))
        payload["archive_count"] = len(archive_entries)
    payload["archive_enabled"] = bool(should_archive)

    return ToolResponse(
        content=[TextBlock(type="text", text=json.dumps(payload))],
        metadata=payload,
    )


# ----- Web crawler -----------------------------------------------------------
@dataclass
class PageRecord:
    url: str
    text: str
    text_length: int
    truncated: bool
    content_length: int
    content_type: Optional[str] = None
    artifact: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class _PlainTextParser(HTMLParser):
    """Convert HTML to plain text by collecting textual chunks."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip_stack: List[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - basic accumulation
        if self._skip_stack:
            return
        if data and not data.isspace():
            self._chunks.append(data.strip())

    def _should_skip_tag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> bool:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript", "nav", "header", "footer", "aside", "menu"}:
            return True

        attr_map = {str(k).lower(): str(v or "").lower() for k, v in attrs}
        role = attr_map.get("role", "")
        if any(nav_role == role for nav_role in _NAV_ROLES):
            return True

        attr_blob = " ".join(
            attr_map.get(key, "")
            for key in ("id", "class", "aria-label", "data-testid")
        )
        return any(hint in attr_blob for hint in _NAV_ATTR_HINTS)

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if self._should_skip_tag(tag, attrs):
            self._skip_stack.append(tag.lower())

    def handle_endtag(self, tag: str) -> None:
        if not self._skip_stack:
            return
        tag_lower = tag.lower()
        if tag_lower == self._skip_stack[-1]:
            self._skip_stack.pop()
            return
        for idx in range(len(self._skip_stack) - 1, -1, -1):
            if self._skip_stack[idx] == tag_lower:
                del self._skip_stack[idx:]
                return

    def get_text(self) -> str:
        return " ".join(self._chunks)


class _LinkParser(HTMLParser):
    """Collect href attributes from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self._links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str]]) -> None:
        if tag.lower() != "a":
            return
        for attr, value in attrs:
            if attr.lower() == "href" and value:
                self._links.append(value)

    def get_links(self) -> List[str]:
        return self._links


def _normalize_url(url: str) -> str:
    return url.strip()


def _normalize_content_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}


def _strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.fragment:
        return url
    return parsed._replace(fragment="").geturl()


def _should_skip_link(link: str) -> bool:
    lowered = link.strip().lower()
    return lowered.startswith(("mailto:", "javascript:", "tel:", "data:"))


def _normalize_link(link: str, base_url: str) -> str:
    link = link.strip()
    if not link or link.startswith("#") or _should_skip_link(link):
        return ""
    return _strip_fragment(urljoin(base_url, link))


def _is_http3_downgrade_error(exc: Exception) -> bool:
    """
    Detect the urllib3 MustDowngradeError / Alt-Svc HTTP/3 handshake issue so we can fail fast.
    """
    msg = str(exc).lower()
    return "mustdowngradeerror" in msg or "http/3" in msg or "altsvc" in msg


def _same_domain(url: str, root: str) -> bool:
    parsed = urlparse(url)
    root_parsed = urlparse(root)
    return parsed.netloc == root_parsed.netloc


def _looks_like_pdf(url: str, content_type: str | None, data: bytes | None = None) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    if content_type and content_type.lower().startswith("application/pdf"):
        return True
    if data and data[:4] == b"%PDF":
        return True
    return False


def _is_html_type(content_type: str) -> bool:
    return content_type in {"text/html", "application/xhtml+xml"}


def _is_text_type(content_type: str) -> bool:
    return content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xml",
        "text/xml",
    }


def _build_error_record(url: str, message: str, content_type: str | None = None) -> PageRecord:
    return PageRecord(
        url=url,
        text="",
        text_length=0,
        truncated=False,
        content_length=0,
        content_type=content_type or None,
        error=message,
    )


def _extract_pdf_text(
    data: bytes,
    *,
    max_pages: int,
    max_chars: int,
) -> tuple[str, bool, int, str | None]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return "", False, 0, "pypdf is not installed (pip install pypdf)."

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:  # pragma: no cover - runtime dependency issues
        return "", False, 0, f"Failed to read PDF: {exc}"

    parts: List[str] = []
    for idx, page in enumerate(reader.pages):
        if idx >= max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            parts.append(text)

    combined = "\n".join(parts).strip()
    full_length = len(combined)
    truncated = full_length > max_chars or len(reader.pages) > max_pages
    return combined[:max_chars], truncated, full_length, None


def _fetch_via_jina_reader(url: str, *, max_chars: int, timeout: int = 20) -> tuple[str | None, bool, int, str | None]:
    """
    Use the public jina.ai reader proxy to fetch page text, which often bypasses 403/anti-bot blocks.
    Returns (text_or_none, truncated, content_length, error_or_none).
    """
    if not _is_http_url(url):
        return None, False, 0, "URL must start with http/https for Jina reader."
    proxy_url = f"https://r.jina.ai/{url.strip()}"
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        resp = requests.get(
            proxy_url,
            headers=headers,
            timeout=timeout,
            verify=_resolve_verify_setting(),
        )
    except requests.RequestException as exc:
        return None, False, 0, f"Jina reader request failed: {exc}"

    if resp.status_code >= 400:
        return None, False, 0, f"Jina reader HTTP {resp.status_code} error."

    text = resp.text or ""
    truncated = len(text) > max_chars
    return text[:max_chars], truncated, len(text), None


def crawl_web_site(
    url: str,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    same_domain: bool = True,
    max_chars_per_page: int = DEFAULT_MAX_CHARS_PER_PAGE,
    request_timeout: int = 10,
    archive_pages: Optional[bool] = None,
    artifact_root: Optional[str] = None,
) -> ToolResponse:
    """Retrieve plain text from a small crawl starting at ``url``."""
    start_url = _normalize_url(url)
    if not start_url:
        return _error("URL must be provided for crawl_web_site")
    if not _is_http_url(start_url):
        return _error("URL must start with http:// or https://")

    should_archive = archive_pages
    if should_archive is None:
        should_archive = _env_flag(WEB_ARCHIVE_DEFAULT_ENV, True)
    _, max_archive_bytes, max_archive_text_chars, max_archive_pdf_pages = _resolve_archive_settings(
        request_timeout=request_timeout
    )

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    visited: Set[str] = set()
    queue = deque([start_url])
    records: List[PageRecord] = []
    archived_artifacts: List[Dict[str, Any]] = []

    while queue and len(records) < max_pages:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if not _is_http_url(current):
            records.append(_build_error_record(current, "Skipped non-HTTP URL."))
            continue
        try:
            resp = requests.get(
                current,
                headers=headers,
                timeout=request_timeout,
                verify=_resolve_verify_setting(),
            )
        except requests.RequestException as exc:
            message = f"Request failed: {exc}"
            if _is_http3_downgrade_error(exc):
                message = "HTTP/3 Alt-Svc downgrade issue; skipping (will try fallback if enabled)."
            records.append(_build_error_record(current, message))
            continue

        content_type = _normalize_content_type(resp.headers.get("Content-Type", ""))
        final_url = resp.url or current
        if same_domain and not _same_domain(final_url, start_url):
            records.append(
                _build_error_record(final_url, "Redirected outside the start domain.", content_type)
            )
            continue
        if resp.status_code >= 400:
            records.append(
                _build_error_record(final_url, f"HTTP {resp.status_code} error.", content_type)
            )
            continue

        text = ""
        plain_text_for_archive: Optional[str] = None
        plain_text_archive_truncated = False
        if _looks_like_pdf(final_url, content_type, resp.content):
            pdf_text, truncated, full_length, error = _extract_pdf_text(
                resp.content,
                max_pages=DEFAULT_MAX_PDF_PAGES,
                max_chars=DEFAULT_MAX_PDF_CHARS,
            )
            if error:
                records.append(
                    _build_error_record(final_url, f"PDF parsing failed: {error}", content_type)
                )
                continue
            trimmed = pdf_text
            content_length = full_length
        else:
            if _is_html_type(content_type) or final_url.lower().endswith((".html", ".htm")):
                text = resp.text or ""
                parser = _PlainTextParser()
                parser.feed(text)
                parser.close()
                plain_text = parser.get_text()
            elif _is_text_type(content_type) or final_url.lower().endswith(
                (".txt", ".csv", ".json", ".xml")
            ):
                plain_text = resp.text or ""
            elif not content_type and resp.content:
                plain_text = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
            else:
                records.append(
                    _build_error_record(
                        final_url,
                        f"Unsupported content type '{content_type or 'unknown'}'.",
                        content_type,
                    )
                )
                continue

            truncated = len(plain_text) > max_chars_per_page
            plain_text_for_archive = plain_text
            plain_text_archive_truncated = len(plain_text) > max_archive_text_chars
            trimmed = plain_text[:max_chars_per_page]
            content_length = len(plain_text)

        artifact_entry: Optional[Dict[str, Any]] = None
        if should_archive:
            if len(resp.content) > max_archive_bytes:
                artifact_entry = {
                    "source_url": current,
                    "final_url": final_url,
                    "status_code": resp.status_code,
                    "size_bytes": len(resp.content),
                    "error": f"Artifact exceeds max size ({max_archive_bytes} bytes).",
                }
            else:
                artifact_entry = _archive_payload(
                    source_url=current,
                    final_url=final_url,
                    status_code=resp.status_code,
                    content_type=content_type,
                    data=resp.content,
                    artifact_root=artifact_root,
                    extracted_text=plain_text_for_archive,
                    text_truncated=plain_text_archive_truncated,
                    encoding=resp.encoding,
                    max_text_chars=max_archive_text_chars,
                    max_pdf_pages=max_archive_pdf_pages,
                )
            archived_artifacts.append(artifact_entry)

        records.append(
            PageRecord(
                url=final_url,
                text=trimmed,
                text_length=len(trimmed),
                truncated=truncated,
                content_length=content_length,
                content_type=content_type or None,
                artifact=artifact_entry,
            )
        )

        if len(records) >= max_pages:
            break

        if text:
            link_parser = _LinkParser()
            link_parser.feed(text)
            link_parser.close()
            for link in link_parser.get_links():
                abs_link = _normalize_link(link, final_url)
                if not abs_link:
                    continue
                if same_domain and not _same_domain(abs_link, start_url):
                    continue
                if abs_link not in visited:
                    queue.append(abs_link)

    aggregated = " ".join(rec.text for rec in records)
    payload = {
        "pages_crawled": len(records),
        "records": [rec.__dict__ for rec in records],
        "aggregated_text": aggregated,
        "archive_enabled": bool(should_archive),
        "archive_root": str(_resolve_artifact_root(artifact_root)) if should_archive else None,
        "archived_artifacts": archived_artifacts,
    }
    return ToolResponse(content=[TextBlock(type="text", text=json.dumps(payload))], metadata=payload)


# ----- Web summarizer --------------------------------------------------------


def _pick_llm_provider(model_name: Optional[str]) -> tuple[str, str]:
    web_openai_key = (
        os.environ.get("WEB_SUMMARY_OPENAI_API_KEY")
        or os.environ.get("WEB_SUMMARY_API_KEY")
        or ""
    ).strip()
    web_anthropic_key = (
        os.environ.get("WEB_SUMMARY_ANTHROPIC_API_KEY")
        or os.environ.get("WEB_SUMMARY_API_KEY")
        or ""
    ).strip()
    openai_api_key = web_openai_key or os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_api_key = web_anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    provider_hint = (
        os.environ.get("WEB_SUMMARY_MODEL_PROVIDER")
        or os.environ.get("TERRABENCH_WEB_SUMMARY_MODEL_PROVIDER")
        or os.environ.get("CLIMATE_AGENT_WEB_SUMMARY_MODEL_PROVIDER")
        or os.environ.get("TERRABENCH_MODEL_PROVIDER")
        or os.environ.get("CLIMATE_AGENT_MODEL_PROVIDER")
        or ""
    ).strip().lower()
    
    if provider_hint == "anthropic" and anthropic_api_key:
        return "anthropic", anthropic_api_key
    if provider_hint == "openai" and openai_api_key:
        return "openai", openai_api_key
    return "", ""


def _normalize_max_tokens(value: int) -> int:
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_SUMMARY_MAX_TOKENS
    return max(50, min(tokens, 1000))


def _load_summary_prompt_text(path: str) -> tuple[Optional[str], Optional[str]]:
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (Path.cwd() / prompt_path).resolve()
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"Failed to read {SUMMARY_PROMPT_PATH_ENV} file '{path}': {exc}"
    if not text.strip():
        return None, f"{SUMMARY_PROMPT_PATH_ENV} file '{path}' is empty."
    return text.strip(), None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_summary_base_url(provider: str) -> Optional[str]:
    if provider == "openai":
        base_url = (
            os.getenv(SUMMARY_OPENAI_BASE_URL_ENV)
            or os.getenv(SUMMARY_BASE_URL_ENV)
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("TERRABENCH_OPENAI_BASE_URL")
            or os.getenv("CLIMATE_AGENT_OPENAI_BASE_URL")
        )
    elif provider == "anthropic":
        base_url = (
            os.getenv(SUMMARY_ANTHROPIC_BASE_URL_ENV)
            or os.getenv(SUMMARY_BASE_URL_ENV)
            or os.getenv("ANTHROPIC_BASE_URL")
            or os.getenv("TERRABENCH_ANTHROPIC_BASE_URL")
            or os.getenv("CLIMATE_AGENT_ANTHROPIC_BASE_URL")
        )
    else:
        base_url = None
    if base_url and base_url.strip():
        return base_url.strip()
    return None


def summarize_web_page(
    url: Optional[str] = None,
    page_text: Optional[str] = None,
    question: Optional[str] = None,
    instructions: Optional[str] = None,
    model_name: Optional[str] = None,
    max_chars: int = 1000,
    max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    request_timeout: int = 20,
    archive_source: Optional[bool] = None,
    artifact_root: Optional[str] = None,
) -> ToolResponse:
    """
    Summarize a web page URL or supplied text via OpenAI/Anthropic chat models,
    optionally focusing on a specific question. URLs are fetched with a lightweight
    crawler (HTML + PDF support) before summarization. Requires the corresponding
    API key in the environment.
    """
    text_source = page_text
    content_url = url
    truncated = False
    crawl_used = False
    crawl_pages = None
    crawl_error: Optional[str] = None
    fallback_raw_url = _env_flag(SUMMARY_FALLBACK_RAW_URL_ENV, True)
    use_jina_fallback = _env_flag(SUMMARY_USE_JINA_FALLBACK_ENV, True)
    jina_used = False
    jina_error: Optional[str] = None
    should_archive = archive_source
    if should_archive is None:
        should_archive = _env_flag(WEB_ARCHIVE_DEFAULT_ENV, True)
    archive_entries: List[Dict[str, Any]] = []
    archive_root_path = str(_resolve_artifact_root(artifact_root)) if should_archive else None

    if not text_source:
        if not url:
            return _error("Either url or page_text must be provided")

        crawl_used = True
        crawl_max_pages = _env_int(SUMMARY_CRAWL_MAX_PAGES_ENV, DEFAULT_MAX_PAGES)
        crawl_max_chars = _env_int(
            SUMMARY_CRAWL_MAX_CHARS_PER_PAGE_ENV,
            DEFAULT_MAX_CHARS_PER_PAGE,
        )
        crawl_same_domain = _env_flag(SUMMARY_CRAWL_SAME_DOMAIN_ENV, True)
        crawl_response = crawl_web_site(
            url,
            max_pages=crawl_max_pages,
            same_domain=crawl_same_domain,
            max_chars_per_page=crawl_max_chars,
            request_timeout=min(request_timeout, 12),
            archive_pages=should_archive,
            artifact_root=artifact_root,
        )
        crawl_meta = crawl_response.metadata or {}
        if isinstance(crawl_meta.get("archived_artifacts"), list):
            archive_entries = list(crawl_meta.get("archived_artifacts") or [])
        if crawl_meta.get("error"):
            message = crawl_meta.get("message", "Unknown crawl failure.")
            crawl_error = message
        aggregated = (crawl_meta.get("aggregated_text") or "").strip()
        if not aggregated:
            crawl_error = crawl_error or "Crawler returned no text to summarize."

            # Try Jina reader proxy as a second attempt for 403/blocked pages.
            if use_jina_fallback:
                jina_text, jina_truncated, jina_len, jina_err = _fetch_via_jina_reader(
                    url, max_chars=max_chars, timeout=request_timeout
                )
                if jina_text:
                    text_source = jina_text
                    truncated = truncated or jina_truncated
                    crawl_used = False  # we bypassed the crawler
                    jina_used = True
                    if should_archive:
                        archive_entries.append(
                            _archive_url(
                                url,
                                artifact_root=artifact_root,
                                request_timeout=request_timeout,
                            )
                        )
                else:
                    jina_error = jina_err

            # Final fallback: hand raw URL to the model if allowed
            if not text_source:
                if not fallback_raw_url:
                    return _error(crawl_error)
                text_source = f"URL: {url.strip()}"
                crawl_used = False
        else:
            crawl_pages = crawl_meta.get("pages_crawled")
            if len(aggregated) > max_chars:
                aggregated = aggregated[:max_chars]
                truncated = True
            text_source = aggregated
    else:
        if len(text_source) > max_chars:
            text_source = text_source[:max_chars]
            truncated = True
        if should_archive and content_url:
            archive_entries.append(
                _archive_url(
                    content_url,
                    artifact_root=artifact_root,
                    request_timeout=request_timeout,
                )
            )

    provider, api_key = _pick_llm_provider(model_name or os.getenv(DEFAULT_SUMMARY_MODEL_ENV))

    if not provider or not api_key:
        return _error("No OpenAI or Anthropic API key configured for summarization")

    base_url = _resolve_summary_base_url(provider)
    resolved_model = (
        model_name
        or os.getenv(DEFAULT_SUMMARY_MODEL_ENV)
        or (DEFAULT_CLAUDE_FALLBACK if provider == "anthropic" else DEFAULT_MODEL_FALLBACK)
    )
    max_tokens = _normalize_max_tokens(max_tokens)
    prompt_path = os.getenv(SUMMARY_PROMPT_PATH_ENV)
    prompt_text = None
    if not instructions and prompt_path:
        prompt_text, error = _load_summary_prompt_text(prompt_path)
        if error:
            return _error(error)
    base_instructions = (
        instructions
        or prompt_text
        or os.getenv(SUMMARY_INSTRUCTIONS_ENV)
        or DEFAULT_SUMMARY_INSTRUCTIONS
    )
    prompt = f"{SUMMARY_FOCUS_PREFIX}\n\n{instructions}" if instructions else base_instructions
    if question:
        prompt = f"{prompt}\n\nFocus question:\n{question.strip()}"
    if content_url:
        prompt = f"{prompt}\n\nSource URL:\n{content_url.strip()}"
    prompt = f"{prompt}\n\nKeep the response under about {max_tokens} tokens."
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        request_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_source},
            ],
            "max_tokens": max_tokens,
        }
        reasoning_effort = _resolve_openai_reasoning_effort(resolved_model)
        inserted_reasoning_effort = False
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
            inserted_reasoning_effort = True
        try:
            completion = client.chat.completions.create(**request_kwargs)
            summary_text = completion.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - API failure paths
            if inserted_reasoning_effort and _is_unsupported_openai_reasoning_error(exc):
                request_kwargs.pop("reasoning_effort", None)
                try:
                    completion = client.chat.completions.create(**request_kwargs)
                    summary_text = completion.choices[0].message.content or ""
                except Exception as retry_exc:  # pragma: no cover - API failure paths
                    return _error(f"OpenAI summarization failed: {retry_exc}")
            else:
                return _error(f"OpenAI summarization failed: {exc}")
    else:
        import anthropic

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**client_kwargs)
        try:
            response = client.messages.create(
                model=resolved_model,
                system=prompt,
                messages=[{"role": "user", "content": text_source}],
                max_tokens=max_tokens,
            )
            summary_text = response.content[0].text if response.content else ""
        except Exception as exc:  # pragma: no cover - API failure paths
            return _error(f"Anthropic summarization failed: {exc}")

    payload = {
        "source_url": content_url,
        "truncated": truncated,
        "chars_input": len(text_source),
        "max_tokens": max_tokens,
        "prompt_path": prompt_path if prompt_text else None,
        "base_url": base_url,
        "crawl_used": crawl_used,
        "crawl_pages": crawl_pages,
        "crawl_error": crawl_error,
        "fallback_raw_url": fallback_raw_url,
        "jina_fallback_used": jina_used,
        "jina_fallback_error": jina_error,
        "provider": provider,
        "model": resolved_model,
        "archive_enabled": bool(should_archive),
        "archive_root": archive_root_path,
        "archived_artifacts": archive_entries,
    }
    source_text_paths = [
        str(entry.get("text_path"))
        for entry in archive_entries
        if isinstance(entry, dict) and entry.get("text_path")
    ]
    if source_text_paths:
        primary_text_path = source_text_paths[0]
        primary_text_sha256 = next(
            (
                str(entry.get("text_sha256"))
                for entry in archive_entries
                if isinstance(entry, dict) and entry.get("text_path") == primary_text_path
            ),
            None,
        )
    else:
        text_artifact = _persist_text_artifact(
            text_source or "",
            artifact_root=artifact_root,
            kind="summary_input",
        )
        primary_text_path = text_artifact["path"]
        primary_text_sha256 = text_artifact["sha256"]

    summary_artifact = _persist_text_artifact(
        summary_text or "",
        artifact_root=artifact_root,
        kind="summaries",
    )
    payload["summary_path"] = summary_artifact["path"]
    payload["summary_sha256"] = summary_artifact["sha256"]
    payload["summary_chars"] = summary_artifact["chars"]
    payload["primary_text_path"] = primary_text_path
    payload["primary_text_sha256"] = primary_text_sha256
    payload["source_text_paths"] = source_text_paths

    # Keep tool result compact while still surfacing the summary content.
    return ToolResponse(
        content=[
            TextBlock(type="text", text=summary_text),
            TextBlock(type="text", text=primary_text_path),
        ],
        metadata=payload,
    )
