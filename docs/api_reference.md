# Insulin AI API Reference

## Active Learning

### POST /api/active-learning/run

Run the active learning feedback loop: literature mining → MD evaluation → feedback → next iteration.

**Request body (JSON):**
```json
{
  "iterations": 2,
  "use_md": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| iterations | int | 1 | Number of cycles (1-5) |
| use_md | bool | true | Use MD simulation for evaluation; if false, literature-only |

**Response:**
```json
{
  "success": true,
  "iterations": 2,
  "summary": [
    {"iteration": 1, "candidates": 15, "high_performers": 3},
    {"iteration": 2, "candidates": 12, "high_performers": 4}
  ],
  "cycle_results_dir": "cycle_results"
}
```

**Status codes:** 200 (success), 500 (error), 503 (literature miner not initialized)

---

## System Status

### GET /api/status

Returns health of all components including:
- `literature_mining`
- `chatbot`
- `psmiles_generator`
- `mcp_literature_mining`
- `md_simulation` (status: active/proxy_only, mode: OpenMM+PME or RDKit proxy, platform: CPU)

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

## CLI

```bash
# Full feedback loop with MD
python run_active_learning.py --iterations 2

# Literature-only (no MD)
python run_active_learning.py --no-md --iterations 1
```

## Benchmarks

```bash
python benchmarks/md_benchmark.py
```
