# TerraAgent Prompts

This directory contains the prompt assets needed by the TerraBench/TerraAgent runtime.

## Runtime Prompts

- `runtime/Inference_prompt.yaml`: main inference-running prompt.
- `runtime/Annotating_prompt.yaml`: general agent/tool-use prompt preserved from the internal workflow.
- `runtime/history_compression_prompt.md`: long-history compression prompt.
- `runtime/web_summary_prompt.yaml`: web summarizer prompt used through `WEB_SUMMARY_PROMPT_PATH`.
- `math_agent/claude_prompt.yaml`: legacy copy of the InterCode code-agent prompt template.
- `math_agent/instruct.txt`: compatibility launch notes for the packaged code-agent service.
- `../code_agent/claude_prompt.yaml`: prompt template used by the integrated code-agent service.

## Local-Only Annotation Prompts

The following annotation prompts are present locally but intentionally ignored by `.gitignore` so they are not pushed to GitHub:

- `local_only/Annotating_csv_prompt.yaml`
- `local_only/Annotate_doc_veri.yaml`

They are copied for internal annotation workflows only. Do not move them into `runtime/` unless you also decide they should be public.

## Environment Variables

Recommended prompt path variables for full runs:

```bash
TERRA_AGENT_INFERENCE_PROMPT_PATH=terra_agent/prompts/runtime/Inference_prompt.yaml
TERRA_AGENT_PROMPT_PATH=terra_agent/prompts/runtime/Annotating_prompt.yaml
TERRA_AGENT_COMPRESS_HISTORY_PROMPT_PATH=terra_agent/prompts/runtime/history_compression_prompt.md
WEB_SUMMARY_PROMPT_PATH=terra_agent/prompts/runtime/web_summary_prompt.yaml
CODE_AGENT_PROMPT_TEMPLATE_PATH=terra_agent/code_agent/claude_prompt.yaml
MATH_AGENT_PROMPT_TEMPLATE_PATH=terra_agent/code_agent/claude_prompt.yaml
```

The original internal code used legacy environment variable names. New public code should prefer the `TERRA_AGENT_*` names above.
