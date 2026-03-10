# Insulin AI: OpenCode for Materials Discovery

Materials discovery via [OpenCode](https://github.com/anomalyco/opencode) – our project provides the agent config and MCP servers; OpenCode supplies the chat/TUI.

## Architecture

```
insulin-ai/
├── install                    ← Install script
├── .opencode/                 ← Our config (agent + MCP)
│   ├── opencode.jsonc        ← MCP, default agent
│   ├── agent/
│   │   └── materials-discovery.md
│   └── skills/
├── insulin_ai_mcp_server.py   ← MCP server: mine, evaluate, discover
├── insulin_ai_cli.py          ← CLI (used by MCP + bash)
├── src/python/insulin_ai/     ← Simulation, mutation, CLI
├── papers/                    ← PDFs for PaperQA2 deep reading
├── scripts/
│   ├── index_papers.sh       ← Build PaperQA2 search index
│   └── run_mcp_server.sh     ← MCP server launcher (uses insulin-ai-sim)
└── ...
```

## Quick Start

### 1. Install

```bash
cd /path/to/insulin-ai
./install
```

The install script installs OpenCode (upstream) and creates the `insulin-ai` launcher.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
insulin-ai
```

Opens the chat/TUI. Default agent is **materials-discovery**. Type e.g. "Discover materials for fridge-free insulin patches".

## MCP Tools (single insulin-ai server)

All tools are in one MCP server. The agent orchestrates the discovery loop step by step.

| Tool | Description |
|------|-------------|
| `mine_literature` | Literature mining + PaperQA2 synthesis when indexed |
| `paper_qa`, `paper_qa_index_status`, `index_papers` | Deep reading over PDFs |
| `semantic_scholar_search`, `pubmed_search`, `arxiv_search`, `web_search` | Literature search |
| `lookup_material` | PubMed structure lookup |
| `validate_psmiles`, `psmiles_canonicalize`, `psmiles_dimerize`, `psmiles_fingerprint`, `psmiles_similarity` | PSMILES utilities |
| `evaluate_psmiles`, `mutate_psmiles` | MD evaluation, mutation |
| `save_discovery_state`, `load_discovery_state`, `get_materials_status` | State management |

## CLI (Standalone)

```bash
python insulin_ai_cli.py discover --iterations 2
python insulin_ai_cli.py mine
python insulin_ai_cli.py evaluate "[*]OCC[*]" "[*]CC[*]"
python insulin_ai_cli.py status
```

## Prerequisites

- **Python 3.8+** – for materials discovery backend
- **Ollama** – optional, for LLM-based literature extraction (else keyword-based)
- **OpenAI API key** – optional, for PaperQA2 deep reading (vector embeddings + synthesis)
- **MD simulation** – optional, for insulin+polymer MD screening (see below)

### MD Simulation (Insulin + Polymer)

MD requires OpenMM, openmmforcefields, openff-toolkit, RDKit, and psmiles. The conda env includes **Packmol** for realistic insulin+polymer packing (overlap-free); without it, naive placement is used. **On macOS ARM64 (Apple Silicon), pip often fails for openff-toolkit.** Use conda/mamba:

```bash
# Create env with simulation deps (recommended on ARM Mac)
mamba env create -f environment-simulation.yml
mamba activate insulin-ai-sim

# Or with conda:
conda env create -f environment-simulation.yml
conda activate insulin-ai-sim
```

On x86_64 Linux/macOS, you can try:
```bash
pip install insulin-ai[simulation]
# or: pip install openmm openmmforcefields openff-toolkit rdkit psmiles
```

## Output Directories

- `discovery_state/` – Per-iteration state (agent-orchestrated loop)
- `cycle_results/` – Batch CLI cycle outputs
- `iterative_results/` – Per-iteration mining
- `mining_results/` – Literature mining

### PaperQA2 (Deep Reading + RAG)

PaperQA2 is built into insulin-ai. Add papers to `papers/`, build the index, and `mine_literature` includes synthesis from your PDFs automatically.

```bash
# Add PDFs, then build index (OpenAI or Ollama):
cp ~/Downloads/insulin-paper.pdf papers/
OPENAI_API_KEY=sk-... ./scripts/index_papers.sh
# Or Ollama: PQA_EMBEDDING=ollama/nomic-embed-text ./scripts/index_papers.sh

insulin-ai
```

## MCP Servers

Single server: **insulin-ai**. See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md) for all tools and env vars.

## References

- [OpenCode](https://opencode.ai) | [GitHub](https://github.com/anomalyco/opencode)
