# MCP Servers for Insulin AI

OpenCode uses a **single MCP server** (insulin-ai) that exposes all materials discovery tools.

## insulin-ai (Consolidated)

| Command | Config |
|---------|--------|
| `bash scripts/run_mcp_server.sh` | `.opencode/opencode.jsonc` (uses insulin-ai-sim env) |

**Tools:**

| Category | Tools |
|----------|-------|
| **Discovery** | `mine_literature` (includes PaperQA2 when indexed), `paper_qa`, `paper_qa_index_status`, `index_papers` |
| **Literature** | `semantic_scholar_search`, `pubmed_search`, `arxiv_search`, `web_search`, `lookup_material` |
| **PSMILES** | `validate_psmiles`, `psmiles_canonicalize`, `psmiles_dimerize`, `psmiles_fingerprint`, `psmiles_similarity` |
| **Evaluation** | `evaluate_psmiles`, `mutate_psmiles` |
| **Autoresearch** | `run_autonomous_discovery` (time-budget loop → TSV + background subprocess) |
| **State** | `save_discovery_state`, `load_discovery_state`, `get_materials_status` |

**Autoresearch mode:** `run_autonomous_discovery(budget_minutes=480, run_in_background=true)` spawns overnight literature + mutation + MD cycles; logs to `discovery_state/autoresearch_results.tsv`. Use agent **autoresearch-materials** in OpenCode for instructions. CLI: `python scripts/run_autonomous_discovery.py --budget-minutes 480`.

## PaperQA2 (Deep Reading + RAG)

PaperQA2 is built into insulin-ai. Add PDFs to `papers/`, build the index, and `mine_literature` will automatically include synthesis from your PDFs when indexed.

**Setup:**

```bash
# 1. Add PDFs to papers/
cp ~/Downloads/insulin-paper.pdf papers/

# 2. Build the search index (one-time, re-run when adding papers)
# OpenAI:
OPENAI_API_KEY=sk-... ./scripts/index_papers.sh
# Or Ollama (no API key):
PQA_EMBEDDING=ollama/nomic-embed-text ./scripts/index_papers.sh
# (run: ollama pull nomic-embed-text first)

# 3. Launch
insulin-ai
```

**Environment variables** (optional, in `.opencode/opencode.jsonc` or shell):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PAPER_DIRECTORY` | `./papers` | PDF directory |
| `PQA_LLM` | `gpt-4o-mini` | LLM for reasoning (use `ollama/llama3.2` for local) |
| `PQA_SUMMARY_LLM` | `gpt-4o-mini` | LLM for summarizing |
| `PQA_EMBEDDING` | `text-embedding-3-small` | Embeddings (use `ollama/nomic-embed-text` for local) |
| `OPENAI_API_KEY` | — | Required only when using OpenAI models |

## API Keys (Runtime)

```bash
export OPENAI_API_KEY=sk-...               # For PaperQA2 (or use Ollama)
export SEMANTIC_SCHOLAR_API_KEY=your_key   # Higher Semantic Scholar rate limit
export NCBI_API_KEY=your_key               # Higher PubMed rate limit (free from ncbi.nlm.nih.gov)
```

## MCP Server Environment

The MCP server runs via `scripts/run_mcp_server.sh`, which uses the `insulin-ai-sim` conda environment when mamba/conda is available. Create it with `mamba env create -f environment-simulation.yml`.
