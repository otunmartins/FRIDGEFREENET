# Insulin AI Revamp: OpenCode as the Materials Discovery Platform

**Goal**: Use [OpenCode](https://github.com/anomalyco/opencode) source code as the **basis** of our platform. The coding agent becomes a **materials discovery agent**.

## Architecture

```
insulin-ai/                          (workspace root)
├── opencode_src/                    ← OpenCode source (cloned)
│   └── packages/opencode/           ← Core CLI, TUI, server
├── .opencode/                       ← Our config (agent, skills, MCP)
│   ├── opencode.jsonc              ← Config + MCP
│   ├── agent/
│   │   └── materials-discovery.md  ← Primary agent
│   ├── skills/
│   └── command/
├── insulin_ai/                      ← Our Python backend (unchanged)
│   ├── simulation/
│   ├── cli.py
│   └── ...
├── literature_mining_system.py
├── iterative_literature_mining.py
├── insulin_ai_cli.py               ← Called by MCP + bash
└── insulin_ai_mcp_server.py        ← MCP server for OpenCode
```

**Removed**: `app.py`, `templates/`, `static/` (Flask web UI) – **DONE**

## How It Works

1. **User runs OpenCode** from this repo: `opencode` (or `bun run dev` from opencode_src)
2. OpenCode loads `.opencode/` config; default agent = **materials-discovery**
3. Agent has access to:
   - **Bash** → `insulin_ai_cli.py discover`, `mine`, `evaluate`
   - **MCP** → `insulin_ai_mcp_server` exposing `mine_literature`, `evaluate_psmiles`, `run_discover`
   - **Skills** → workflow docs (literature, PSMILES, active learning)
4. **Materials discovery** happens in OpenCode's TUI – the agent runs our tools, interprets results, iterates.

## Integration Steps

1. Create `.opencode/` with materials-discovery agent
2. Create `insulin_ai_mcp_server.py` (MCP) wrapping our tools
3. Remove Flask web app
4. Document: run OpenCode from this repo for materials discovery
