# Repository layout

This project is **materials-discovery software** exposed through one MCP server. Folders exist because the problem splits into literature, simulation, and mutation—not because we like nesting.

## Top level

| Path | Purpose |
|------|---------|
| `insulin_ai_mcp_server.py` | FastMCP app; calls into everything below |
| `insulin_ai_cli.py` | Same stack, terminal entry |
| `iterative_literature_mining.py`, `literature_mining_system.py`, `semantic_scholar_client.py`, `paper_qa_config.py`, `ollama_client.py`, `psmiles_generator.py` | Mining + LLM + PaperQA; kept at repo root for stable imports |
| `src/python/insulin_ai/` | Packaged code: MD, mutation, scoring, autonomous loop |
| `.opencode/` | OpenCode agents + MCP JSON |
| `scripts/` | Shell/Python launchers |
| `papers/` | User PDFs (not committed) |
| `tests/`, `benchmarks/` | Quality / perf |
| `docs/` | MCP docs, proposal, methods |

## Optional external repos

- **autoresearch/** (Karpathy) — not vendored; add to `.gitignore` if you clone it locally for unrelated LLM experiments.

## Runtime dirs (not in git)

`discovery_state/`, `cycle_results/`, `iterative_results/`
