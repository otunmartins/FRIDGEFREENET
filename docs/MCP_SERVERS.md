# MCP Servers for Insulin AI

## Prerequisites

| Capability | Required |
|------------|----------|
| **`mine_literature`** | **Asta MCP** when `ASTA_API_KEY` set; else Semantic Scholar. |
| **`evaluate_psmiles`**, **`run_autonomous_discovery`** | **OpenMM** stack (`openmm`, `openmmforcefields`, `openff-toolkit`, `rdkit`, `pdbfixer`), `data/4F1C.pdb` or `ensure_insulin_pdb`. Merged minimize screening. |
| **Mutation** | psmiles / extras per `pyproject.toml`. |
| **`validate_psmiles(..., crosscheck_web=true)`** | Optional **DuckDuckGo** snippets when `material_name` is set (requires `duckduckgo-search`). Heuristic only. **PSMILES primer:** [PSMILES_GUIDE.md](PSMILES_GUIDE.md). |
| **`render_psmiles_png`** | [psmiles](https://github.com/FermiQ/psmiles) ``PolymerSmiles.savefig`` — 2D monomer/repeat-unit PNG under ``<session>/structures/``. |
| **`compile_discovery_markdown_to_pdf`** | **Agent-authored** ``SUMMARY_REPORT.md`` → **SUMMARY_REPORT.pdf** (``markdown`` + ``fpdf2``). |
| **`write_discovery_summary_report`** | *Optional batch:* from ``agent_iteration_*.json`` only—skeleton MD + PNG + PDF without narrative. Prefer agent-written MD + ``compile_discovery_markdown_to_pdf``. |
| **`save_session_transcript`** | Write text (e.g. full session recap) into ``runs/<session>/``. **Required** each run if JSONL import is not used. |
| **`import_chat_transcript_file`** | Copy OpenCode JSONL from ``~/.cursor/.../agent-transcripts/`` (or repo path) into the session. **Preferred** default to archive chat. |

See [DEPENDENCIES.md](DEPENDENCIES.md) (MCP — discovery figures & PDF reports). Chat is **not** mirrored into `runs/` automatically; agents **must** call one of the tools above every iteration — see [OpenCode_PLATFORM.md](OpenCode_PLATFORM.md).

## insulin-ai

| Command | `bash scripts/run_mcp_server.sh` |
|---------|----------------------------------|

**Session folder:** `runs/<id>/`. **Screening:** [OPENMM_SCREENING.md](OPENMM_SCREENING.md).
