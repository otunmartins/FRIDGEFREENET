---
description: Active learning feedback loop for materials discovery
---

# Active Learning Cycle Skill

The discovery loop is agent-orchestrated: you drive each step through individual tool calls, enabling human steering between iterations.

## Loop

1. **Mine** – `mine_literature(query=..., iteration=N, ...)` (includes PaperQA2 when indexed)
2. **Translate** – Convert material names to PSMILES (polymer chemistry knowledge)
3. **Validate** – `validate_psmiles(psmiles)` for each
4. **Evaluate** – `evaluate_psmiles(psmiles_list)` via GROMACS
5. **Mutate** – `mutate_psmiles(feedback_json=...)` with high performers and problematic PSMILES from evaluation
6. **Save** – `save_discovery_state(iteration=N, feedback_json=..., query_used=..., notes=...)`
7. **Report** – Summarize to user; wait for input before next iteration
8. **Refine** – Use feedback to build a better query for the next iteration
9. Repeat from step 1

## Feedback Flow

MD evaluation returns:
- `high_performers` -- materials with good stability metrics
- `effective_mechanisms` -- what's working (hydrogen bonding, hydrophobic interactions, etc.)
- `problematic_features` -- what to avoid (high crystallinity, poor water retention, etc.)

Feed these back into `mine_literature` and `mutate_psmiles` to narrow the search.

## Bash Fallback

Use MCP `run_autonomous_discovery` or orchestrate `mine_literature` / `evaluate_psmiles` / `mutate_psmiles`.

## Results

- `discovery_state/` -- per-iteration state (agent loop)
- `cycle_results/` -- batch CLI results
- `iterative_results/` -- per-iteration mining
