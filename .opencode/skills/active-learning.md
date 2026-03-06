---
description: Active learning feedback loop for materials discovery
---

# Active Learning Cycle Skill

1. Mine literature → material candidates
2. Evaluate candidates (MD or RDKit proxy)
3. Extract feedback (high performers, mechanisms, limitations)
4. Refine queries for next iteration
5. Repeat

## Run

- MCP: `run_discover_cycle(iterations=2, use_md=true)`
- Bash: `python insulin_ai_cli.py discover --iterations 2`

## Results

- `cycle_results/complete_cycle_*.json` – Full cycle
- `iterative_results/iteration_*_*.json` – Per-iteration
