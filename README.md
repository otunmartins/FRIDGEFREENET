# Insulin AI: OpenCode as the Materials Discovery Platform

**OpenCode** ([github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)) is the basis of this platform. The coding agent becomes a **materials discovery agent** for insulin patch polymer design.

## Architecture

```
insulin-ai/
├── opencode_src/              ← OpenCode source (from anomalyco/opencode)
├── .opencode/                 ← Our config
│   ├── opencode.jsonc        ← MCP, default agent
│   ├── agent/
│   │   └── materials-discovery.md
│   └── skills/
├── insulin_ai_mcp_server.py   ← MCP server: mine, evaluate, discover
├── insulin_ai_cli.py          ← CLI (used by MCP + bash)
├── src/python/insulin_ai/     ← Simulation, CLI
├── literature_mining_system.py
├── iterative_literature_mining.py
└── ...
```

**Removed**: Flask web app (`app.py`, `templates/`, `static/`). The interface is **OpenCode's TUI**.

## Quick Start

### 1. Install OpenCode

```bash
curl -fsSL https://opencode.ai/install | bash
# or: npm i -g opencode-ai
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run OpenCode from This Repo

```bash
cd /path/to/insulin-ai
opencode
```

OpenCode loads `.opencode/` – default agent is **materials-discovery**. Use MCP tools (`mine_literature`, `evaluate_psmiles`, `run_discover_cycle`) or bash to run `python insulin_ai_cli.py discover`.

### 4. Or Run OpenCode from Source (Development)

```bash
git clone https://github.com/anomalyco/opencode.git opencode_src
cd opencode_src
bun install
bun run dev
```

Start OpenCode from the **insulin-ai repo root** (parent of opencode_src) so it finds `.opencode/`.

## MCP Tools (insulin-ai server)

| Tool | Description |
|------|-------------|
| `mine_literature` | Mine Semantic Scholar for insulin delivery materials |
| `evaluate_psmiles` | Evaluate PSMILES (MD or RDKit proxy) |
| `run_discover_cycle` | Run active learning feedback loop |
| `get_materials_status` | System status |

## CLI (Standalone)

```bash
python insulin_ai_cli.py discover --iterations 2
python insulin_ai_cli.py mine
python insulin_ai_cli.py evaluate "[*]OCC[*]" "[*]CC[*]"
python insulin_ai_cli.py status
```

## Prerequisites

- Python 3.8+
- Ollama (for literature mining LLM)
- OpenCode (global install or from opencode_src)
- Optional: `openmm`, `openmmforcefields`, `rdkit` for full MD (else RDKit proxy)

## Output Directories

- `cycle_results/` – Active learning cycle outputs
- `iterative_results/` – Per-iteration mining
- `mining_results/` – Literature mining

## References

- [OpenCode](https://opencode.ai) | [GitHub](https://github.com/anomalyco/opencode)
- [REVAMP_OPENCODE_BASE.md](REVAMP_OPENCODE_BASE.md) – Integration details
