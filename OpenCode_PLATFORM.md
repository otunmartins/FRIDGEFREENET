# OpenCode as the Materials Discovery Platform

**Insulin AI** uses **OpenCode** ([github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)) as the basis. The coding agent becomes a **materials discovery agent**. Run `opencode` from this repo – it loads `.opencode/` with the materials-discovery agent and MCP tools.

## Why CLI / OpenCode?

- **OpenCode** = terminal-based AI coding agent ([opencode.ai](https://opencode.ai))
- Interface is conversational in the **terminal**, like Claude Code
- Materials discovery happens through **code invocation**, not web clicks
- The AI agent runs commands, interprets output, and iterates

## Primary Interface: CLI

```bash
# Full active learning feedback loop
insulin-ai discover --iterations 2

# Literature-only (no MD)
insulin-ai discover --no-md -n 1

# Literature mining only
insulin-ai mine

# Evaluate PSMILES
insulin-ai evaluate "[*]OCC[*]" "[*]CC[*]"

# System status
insulin-ai status
```

## Running Without Installing

From the project root:

```bash
python insulin_ai_cli.py discover --iterations 2
```

Or install in development mode:

```bash
pip install -e .
insulin-ai discover
```

## OpenCode Workflow

1. You (or OpenCode) run `insulin-ai discover` in the terminal
2. Literature mining → material candidates
3. MD / RDKit proxy evaluation → feedback
4. Next iteration with refined queries
5. Results in `cycle_results/`, `iterative_results/`
6. OpenCode can inspect JSON, modify parameters, re-run

The **code** is the platform. The terminal is the interface. OpenCode operates on this codebase directly.

## Web App (Optional)

`app.py` provides an optional web UI for chat, literature, PSMILES. It is **not** the primary discovery platform. Use it for demos or manual exploration; the CLI is for systematic discovery with AI agents.
