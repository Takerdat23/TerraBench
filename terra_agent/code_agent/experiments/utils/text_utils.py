import re


def normalize_text(text: str) -> str:
    """Collapse whitespace into single spaces and trim the string."""
    return re.sub(r"\s+", " ", text.strip())


def clean_completion_text(text: str) -> str:
    """Preserve newlines while trimming leading/trailing whitespace."""
    if text is None:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
