# Insulin AI: OpenCode + MCP for materials discovery

OpenCode is the chat UI; **this repo is the whole backend**: literature mining, PSMILES, optional MD, and mutation—not “only” an MCP server file.

## Why so many folders?

| If you only care about… | What you touch |
|-------------------------|----------------|
| **Chat / MCP** | `.opencode/`, `insulin_ai_mcp_server.py`, `scripts/run_mcp_server.sh` |
| **Agents** | `.opencode/agent/*.md`, `.opencode/skills/*.md` |
| **Science (MD, polymers)** | `src/python/insulin_ai/simulation/` |
| **Mining + APIs** | Root `*_mining*.py`, `semantic_scholar_client.py`, `paper_qa_config.py` (legacy layout so MCP `sys.path` stays simple) |
| **PDF deep reading** | `papers/`, `scripts/index_papers.sh` |
| **Tests / benchmarks** | `tests/`, `benchmarks/` |
| **Write-up** | `docs/` |

The MCP server is **thin**: it imports the discovery stack. That stack is multi-folder by design (simulation vs literature vs mutation). Optional **Karpathy autoresearch** (GPU LLM training) is **not** included—clone it elsewhere if you want that boilerplate; this repo ignores `autoresearch/`.

## Layout (trimmed)

```
insulin-ai/
├── .opencode/                 # Agents, MCP config, skills
├── insulin_ai_mcp_server.py   # MCP tool surface
├── insulin_ai_cli.py          # CLI
├── iterative_literature_mining.py, literature_mining_system.py, …  # Mining (root = import path)
├── src/python/insulin_ai/     # Simulation, mutation, autonomous_discovery, CLI module
├── scripts/                   # run_mcp_server.sh, index_papers.sh, run_autonomous_discovery.py
├── papers/                    # Your PDFs (gitignored)
├── tests/                     # pytest
├── benchmarks/                # MD benchmarks (dev)
└── docs/                      # MCP_SERVERS.md, proposal, method notes
```

## Quick start

```bash
./install
pip install -r requirements.txt
insulin-ai
```

Conda for MD: `mamba env create -f environment-simulation.yml` → `scripts/run_mcp_server.sh` uses **insulin-ai-sim**.

## MCP tools (insulin-ai server)

| Tool | Role |
|------|------|
| `mine_literature`, `paper_qa`, `index_papers` | Literature + PaperQA2 |
| `semantic_scholar_search`, `pubmed_search`, `arxiv_search`, `web_search`, `lookup_material` | Search |
| `validate_psmiles`, `evaluate_psmiles`, `mutate_psmiles`, PSMILES helpers | Chemistry + MD |
| `run_autonomous_discovery` | Time-budget autoresearch-style loop (background) |
| `save_discovery_state`, `load_discovery_state`, `get_materials_status` | State |

Full list: [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md).

## CLI

```bash
python insulin_ai_cli.py discover --iterations 2 --mutate
python scripts/run_autonomous_discovery.py --budget-minutes 60
```

## Outputs (gitignored)

`discovery_state/`, `cycle_results/`, `iterative_results/`, `mining_results/*.json`

## References

[OpenCode](https://opencode.ai) · [OpenCode GitHub](https://github.com/anomalyco/opencode)
