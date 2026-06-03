"""Artifact and trace sanitization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SENSITIVE_PATTERNS = {
    "/home/": "<HOME>/",
    "/Users/": "<USERS>/",
    "ClimatePilotBench": "TerraBench",
    "ClimateCoPilotBench": "TerraBench",
    "ClimateAgent": "TerraAgent",
    "ClimaAgent": "TerraAgent",
    "EarthReasonBench": "TerraBench",
    "EarthReasonAgent": "TerraAgent",
    "EarthReason": "TerraBench",
}

SECRET_REGEX = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?[^'\"\s,}]+")


@dataclass
class SanitizationReport:
    files_scanned: int = 0
    patterns_found: dict[str, int] = field(default_factory=dict)
    replacements_made: int = 0
    files_skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "patterns_found": dict(self.patterns_found),
            "replacements_made": self.replacements_made,
            "files_skipped": list(self.files_skipped),
        }


def sanitize_text(text: str, report: SanitizationReport | None = None) -> str:
    output = text
    for pattern, replacement in SENSITIVE_PATTERNS.items():
        count = output.count(pattern)
        if count:
            output = output.replace(pattern, replacement)
            if report is not None:
                report.patterns_found[pattern] = report.patterns_found.get(pattern, 0) + count
                report.replacements_made += count
    output, count = SECRET_REGEX.subn(r"\1=<REDACTED>", output)
    if count and report is not None:
        report.patterns_found["secret-like assignment"] = report.patterns_found.get("secret-like assignment", 0) + count
        report.replacements_made += count
    return output


def sanitize_tree(input_dir: str | Path, output_dir: str | Path) -> SanitizationReport:
    source = Path(input_dir)
    target = Path(output_dir)
    report = SanitizationReport()
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            report.files_skipped.append(str(relative))
            continue
        report.files_scanned += 1
        destination.write_text(sanitize_text(text, report), encoding="utf-8")
    return report
