# Insulin AI / FridgeFreeNet Project Structure

## Layout

```
insulin-ai/
├── install                    # Main install script (OpenCode + env)
├── run                        # Quick launcher for insulin-ai
├── insulin_ai_cli.py          # CLI: discover, mine, evaluate, status
├── insulin_ai_mcp_server.py   # MCP server for OpenCode tools
├── literature_mining_system.py # Literature mining + Semantic Scholar
├── psmiles_generator.py       # PSMILES generation
├── ollama_client.py           # Ollama LLM client
├── paper_qa_config.py         # PaperQA2 deep-reading config
│
├── .opencode/                 # OpenCode config
│   ├── opencode.jsonc         # MCP, default agent
│   ├── agent/                 # Agent definitions
│   └── skills/                # Active-learning, literature-mining
│
├── src/python/insulin_ai/
│   ├── cli.py                 # CLI entrypoints
│   ├── simulation/             # MD simulation
│   │   ├── md_simulator.py
│   │   ├── openmm_runner.py
│   │   ├── insulin_polymer_system.py
│   │   ├── property_extractor.py
│   │   └── psmiles_to_openmm.py
│   └── mutation/              # Feedback-guided mutation
│
├── docs/                      # Documentation
│   ├── archive/               # Archived plans and summaries
│   ├── api_reference.md
│   ├── MCP_SERVERS.md
│   └── proposal.tex
│
├── papers/                    # PDFs for PaperQA2 (user-provided)
├── scripts/                   # Shell helpers (index_papers, run_mcp_server)
├── tests/                     # Pytest tests
└── environment-simulation.yml # Conda env for MD simulation
```

## Outputs (not committed)

- `chat_memory/`, `cycle_results/`, `iterative_results/` – runtime outputs
- `discovery_state/` – discovery checkpoint state
- `mining_results/*.json` – literature mining results
