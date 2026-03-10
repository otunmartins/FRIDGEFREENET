# Insulin AI API Reference

## CLI

The primary interface for materials discovery. Run from the project root.

```bash
# Full feedback loop with MD and mutation
python insulin_ai_cli.py discover --iterations 2 --mutate

# Full feedback loop (MD only)
python insulin_ai_cli.py discover --iterations 2

# Literature-only (no MD)
python insulin_ai_cli.py discover --no-md -n 1

# Literature mining only
python insulin_ai_cli.py mine

# Evaluate PSMILES
python insulin_ai_cli.py evaluate "[*]OCC[*]" "[*]CC[*]"

# System status
python insulin_ai_cli.py status
```

When installed via `pip install -e .`, use the `insulin-ai` entry point:

```bash
insulin-ai discover --iterations 2 --mutate
insulin-ai mine
insulin-ai evaluate "[*]OCC[*]" "[*]CC[*]"
insulin-ai status
```

---

## MCP Tools (insulin-ai server)

Exposed via `insulin_ai_mcp_server.py` when OpenCode loads this project. Single server, all tools consolidated.

| Tool | Description |
|------|-------------|
| `mine_literature` | Literature mining + PaperQA2 synthesis when papers indexed |
| `paper_qa`, `paper_qa_index_status`, `index_papers` | PaperQA2 deep reading over PDFs |
| `semantic_scholar_search`, `pubmed_search`, `arxiv_search`, `web_search` | Literature search |
| `lookup_material` | PubMed structure lookup |
| `validate_psmiles`, `psmiles_canonicalize`, `psmiles_dimerize`, `psmiles_fingerprint`, `psmiles_similarity` | PSMILES utilities |
| `evaluate_psmiles`, `mutate_psmiles` | MD evaluation, mutation |
| `save_discovery_state`, `load_discovery_state`, `get_materials_status` | State management |

---

## System Status

`get_materials_status` (MCP) or `insulin_ai_cli.py status` (CLI) returns:

- `MD Simulation`: insulin + polymer (implicit solvent) or unavailable (CPU)
- `Mutation`: available (cheminformatics) or unavailable
- `Literature Mining`: available or import error
- `PaperQA2`: index status (e.g. "5/10 indexed" or "paper-qa not installed")

---

## Python API

### insulin_ai.simulation.MDSimulator

```python
from insulin_ai.simulation import MDSimulator

sim = MDSimulator(n_steps=50000, temperature=298.0)
feedback = sim.evaluate_candidates(candidates, max_candidates=10)
# feedback: high_performers, effective_mechanisms, problematic_features, property_analysis
```

### insulin_ai.simulation.OpenMMRunner

```python
from insulin_ai.simulation import OpenMMRunner

runner = OpenMMRunner(platform_name="CPU", temperature=298.0)
result = runner.run("[*]OCC[*]", n_steps=50000)
# result: initial_energy_kj_mol, final_energy_kj_mol, ...
```

---

## Benchmarks

```bash
python benchmarks/md_benchmark.py
```
