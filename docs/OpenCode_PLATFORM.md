# OpenCode

- **Creates conda env `insulin-ai-sim`** — RDKit, OpenMM, openmmforcefields, OpenFF Toolkit, pdbfixer, packmol, psmiles, mcp, paper-qa, `-e .`

## Chat transcripts vs `runs/` session folders

**OpenCode does not automatically copy** the assistant chat into `runs/<session_id>/`. Discovery artifacts (`agent_iteration_*.json`, `SUMMARY_REPORT.md`, etc.) are written by MCP tools because they know `INSULIN_AI_SESSION_DIR`; the IDE keeps conversation history in its **own** store.

Typical location for **parent** chat transcripts (JSONL) on disk:

- `~/.cursor/projects/<project-id>/agent-transcripts/<uuid>.jsonl`

(Subagent transcripts may live alongside; naming depends on Cursor/OpenCode version.)

## Archiving chat into the session (required by default)

**Every materials discovery iteration must** end with a transcript under the same `runs/<session_id>/` as the rest of the run. This is **not** optional unless the user explicitly opted out.

1. **Prefer** **`import_chat_transcript_file`** — pass the **absolute path** to the current parent chat JSONL under `~/.cursor/.../agent-transcripts/`. Copies into the session (e.g. `CHAT_<uuid>.jsonl`).
2. **If** the path is unknown or the copy fails, **use** **`save_session_transcript`** — pass a **complete** Markdown recap of this session (tool calls, decisions, results). Do not skip this step.

Call **after** `start_discovery_session` so `INSULIN_AI_SESSION_DIR` matches, or pass **`run_dir`** to the same folder as `save_discovery_state`.

There is no extra dependency beyond the MCP server; the tools only read/write files you authorize.
