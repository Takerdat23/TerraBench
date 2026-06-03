<system-hint>
You have been working on the task described above but have not yet completed it.
Write a continuation summary that will replace older conversation history in a future context window.

Your summary must be optimized for resuming execution, not for storytelling.
Summarize the old history as a gap analysis against the visible task requirements or CSV instruction fields.

Requirements:
1. Be concise, structured, and directly actionable.
2. Focus on what has been completed, what is still missing, and what the next required action is.
3. Include grounded numeric results only when they were actually produced by kept tool outputs.
4. Preserve every concrete artifact reference present in the compressed history.
5. Ignore tool-switching housekeeping such as `reset_equipped_tools` unless it explains a blocker or an inactive-tool error.
6. Do not restate long deleted chatter or low-value retries unless they matter for the next step.
7. Do not invent progress, artifact paths, files, or resolved results.

When filling the structured response:
- `task_overview`: restate the task, success criteria, constraints, date window, AOI, thresholds, and required deliverables.
- `gap_analysis`: write short bullet lines that begin with `DONE:`, `OUTSTANDING:`, `BLOCKED:`, or `CHECK:`. This section should read like an execution gap analysis.
- `next_required_step`: name the immediate next action, including the tool(s) and artifact(s) that should be used next.
- `artifact_paths`: separate artifact references by location and machine.
  Include two labeled subsections when applicable:
  `Local paths:` for files on the current ClimateAgent machine, using exact absolute local filesystem paths.
  `Math-agent / remote paths:` for files, upload paths, container paths, working-directory paths, file_id values, or artifact identifiers that exist on the math-agent server or another remote execution environment.
  List every concrete artifact reference seen in the compressed history, one per line, under the correct subsection.
- `important_discoveries`: preserve key findings, errors, failed approaches, and technical constraints that matter for continuation.
- `context_to_preserve`: preserve formatting requirements, output schema requirements, benchmark rules, and user preferences.

Prefer exact path strings and exact file identifiers over paraphrases.
Do not mix local-machine paths with math-agent-server paths. If both exist for the same logical artifact, preserve both and label them clearly.
The result should help a future agent continue from the compressed summary without forgetting unfinished deliverables or losing artifact provenance.
</system-hint>
