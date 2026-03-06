# MCP Servers for Insulin AI

OpenCode supports multiple MCP servers. All listed here can be enabled in `.opencode/opencode.jsonc`.

## Literature Mining (No API Key Required)

| Server | Command | API Key | Description |
|--------|---------|---------|-------------|
| **lit-semantic-scholar** | `mcp_servers/mcp_lit_semantic_scholar.py` | Optional (`SEMANTIC_SCHOLAR_API_KEY`) | Semantic Scholar search |
| **lit-pubmed** | `mcp_servers/mcp_lit_pubmed.py` | Optional (`NCBI_API_KEY`) | PubMed/NCBI search |
| **lit-arxiv** | `mcp_servers/mcp_lit_arxiv.py` | None | arXiv preprint search |

## Literature (API Key / Ollama)

| Server | Command | Requirements | Description |
|--------|---------|--------------|-------------|
| **lit-semantic-scholar-full** | `semantic_scholar_server.py` | Ollama | Full analysis, LLM relevance, material extraction |

## PSMILES (Ramprasad Group)

| Server | Command | Install | Description |
|--------|---------|---------|-------------|
| **psmiles-ramprasad** | `mcp_servers/mcp_psmiles_ramprasad.py` | `pip install 'psmiles[mordred,polyBERT]'` | Canonicalize, dimerize, fingerprints, similarity |

## Materials Discovery (Combined)

| Server | Command | Description |
|--------|---------|-------------|
| **insulin-ai** | `insulin_ai_mcp_server.py` | mine_literature, evaluate_psmiles, run_discover_cycle, get_materials_status |

## API Keys (Runtime)

Set via environment before starting OpenCode:

```bash
export SEMANTIC_SCHOLAR_API_KEY=your_key   # Higher Semantic Scholar rate limit
export NCBI_API_KEY=your_key               # Higher PubMed rate limit (free from ncbi.nlm.nih.gov)
```

## 5-Iteration Test Results (No API Keys)

Run: `python run_discover_no_ollama.py`

**Materials obtained** (24 unique):

1. hydrogel, 2. alginate, 3. polymer, 4. chitosan, 5. hyaluronic acid, 6. pva, 7. cellulose, 8. composite, 9. collagen, 10. peg, 11. plga, 12. copolymer, 13. pmma, 14. polyurethane-based, 15. polypyrrole-based, 16. polyamine-based, 17. peg-based, ...

Sources: Semantic Scholar, PubMed, arXiv (direct APIs, no LLM).
