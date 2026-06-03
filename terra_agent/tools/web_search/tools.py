"""TerraAgent wrappers for Serper search and web-page summarization."""

from __future__ import annotations

from typing import Any

from terra_agent.tools.base import BaseTool, ToolResult
from terra_agent.tools.registry import register_tool
from terra_agent.tools.web_search.web_tools import summarize_web_page, web_search_serper


@register_tool
class WebSearchSerperTool(BaseTool):
    name = "web_search_serper"
    group = "web_search"
    description = "Search Google through the Serper API and optionally archive result pages."
    requires_credentials = True
    credential_env_vars = ("SERPER_API_KEY",)
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        query = inputs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("web_search_serper requires a non-empty string input named 'query'.")

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        response = web_search_serper(
            query=inputs["query"],
            search_type=inputs.get("search_type", "search"),
            num_results=inputs.get("num_results", 10),
            language=inputs.get("language"),
            country=inputs.get("country"),
            filter_landing_pages=inputs.get("filter_landing_pages", True),
            overfetch_factor=inputs.get("overfetch_factor", 2),
            archive_results=inputs.get("archive_results"),
            archive_limit=inputs.get("archive_limit", 5),
            artifact_root=inputs.get("artifact_root") or context.get("artifact_root"),
            archive_timeout=inputs.get("archive_timeout"),
        )
        return _to_tool_result(response, default_summary="Serper search completed.")


@register_tool
class SummarizeWebPageTool(BaseTool):
    name = "summarize_web_page"
    group = "web_search"
    description = "Fetch and summarize a URL or supplied page text with a configured LLM."
    requires_credentials = True
    credential_env_any = (
        "WEB_SUMMARY_API_KEY",
        "WEB_SUMMARY_OPENAI_API_KEY",
        "WEB_SUMMARY_ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    )
    deterministic = False

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        url = inputs.get("url")
        page_text = inputs.get("page_text")
        if not (isinstance(url, str) and url.strip()) and not (
            isinstance(page_text, str) and page_text.strip()
        ):
            raise ValueError("summarize_web_page requires either 'url' or 'page_text'.")

    def run(self, inputs: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        self.validate_inputs(inputs)
        response = summarize_web_page(
            url=inputs.get("url"),
            page_text=inputs.get("page_text"),
            question=inputs.get("question"),
            instructions=inputs.get("instructions"),
            model_name=inputs.get("model_name"),
            max_chars=inputs.get("max_chars", 1000),
            max_tokens=inputs.get("max_tokens", 500),
            request_timeout=inputs.get("request_timeout", 20),
            archive_source=inputs.get("archive_source"),
            artifact_root=inputs.get("artifact_root") or context.get("artifact_root"),
        )
        return _to_tool_result(response, default_summary="Web page summarization completed.")


def _to_tool_result(response: Any, *, default_summary: str) -> ToolResult:
    metadata = dict(getattr(response, "metadata", None) or {})
    status = "error" if metadata.get("error") else "success"
    text_blocks = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            text_blocks.append(str(text))
    summary = text_blocks[0] if text_blocks else metadata.get("message") or default_summary
    return ToolResult(status=status, summary=summary, data=metadata, metadata=metadata)
