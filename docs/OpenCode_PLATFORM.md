# OpenCode

- **Creates conda env `insulin-ai-sim`** — RDKit, OpenMM, openmmforcefields, OpenFF Toolkit, pdbfixer, packmol, psmiles, mcp, paper-qa, `-e .`

## Chat transcripts vs `runs/` session folders

**OpenCode does not automatically copy** the assistant chat into `runs/<session_id>/`. Discovery artifacts (`agent_iteration_*.json`, `SUMMARY_REPORT.md`, optional **`discovery_world.json`** structured rollup, etc.) are written by MCP tools because they know `INSULIN_AI_SESSION_DIR`; the IDE keeps conversation history in its **own** store. The world file accumulates objectives, literature claims, simulation summaries, hypotheses, and human steering notes across iterations; see **`patch_discovery_world`** / **`discovery_world_planning_context`** in [MCP_SERVERS.md](MCP_SERVERS.md).

### Where the session archive must live (canonical)

- **Always** persist the iteration’s chat archive **in the same folder as the rest of that run** — i.e. `runs/<session_id>/` next to `SUMMARY_REPORT.md`, `structures/`, and other iteration outputs (or the explicit `run_dir` / `INSULIN_AI_SESSION_DIR` for that job).
- **Never** use `~/.cursor/` (or any path under **`.cursor/`**) as the **destination** for a session transcript. Do **not** write copies there, do **not** leave the only archive there, and do **not** treat the IDE store as the project’s record of the run.

The IDE may still keep a **source** JSONL on disk (often under `~/.cursor/projects/<project-id>/agent-transcripts/<uuid>.jsonl`) — that path is only for **reading** when calling `import_chat_transcript_file`, which **copies** into `runs/<session_id>/`. Subagent JSONL may sit alongside; naming depends on the OpenCode/Cursor version.

## Archiving chat into the session (required by default)

**Every materials discovery iteration must** end with a transcript **file under** the same `runs/<session_id>/` as the rest of the run. This is **not** optional unless the user explicitly opted out.

1. **Prefer** **`import_chat_transcript_file`** — pass the **absolute path** to the current parent chat JSONL (often under `~/.cursor/.../agent-transcripts/` **as the read-only source**). The tool **writes** into the session folder (e.g. `CHAT_<uuid>.jsonl` under `runs/<session_id>/`).
2. **If** the path is unknown or the copy fails, **use** **`save_session_transcript`** — pass a **complete** Markdown recap of this session (tool calls, decisions, results) so it is stored **only** under `runs/<session_id>/`. Do not skip this step.

Call **after** `start_discovery_session` so `INSULIN_AI_SESSION_DIR` matches, or pass **`run_dir`** to the same folder as `save_discovery_state`.

There is no extra dependency beyond the MCP server; the tools only read/write files you authorize.
