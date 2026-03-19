---
description: Run materials discovery autonomously, step by step
---

# Autonomous Discovery Skill

When the user asks to discover materials (e.g. "discover materials for fridge-free insulin patches"), **start immediately without asking for confirmation**. You orchestrate the loop yourself through individual tool calls.

## Protocol

### Each iteration

1. `mine_literature(query=..., iteration=N, top_candidates=..., stability_mechanisms=..., limitations=...)` — includes PaperQA2 synthesis when papers indexed
2. Translate returned material names to PSMILES (use chemistry knowledge, `lookup_material`, or `web_search`)
3. `validate_psmiles(psmiles)` for each; fix failures
4. `evaluate_psmiles(psmiles_list)` -- MD evaluation
5. `mutate_psmiles(feedback_json=...)` -- generate variants of high performers; evaluate those too
6. `save_discovery_state(iteration=N, feedback_json=..., query_used=..., notes=...)`
7. Report to user: high performers, mechanisms, problems, next plan. **Wait for user input** before proceeding.

### Iteration 1

Use broad queries: `"hydrogels insulin delivery transdermal"`, `"polymer protein stabilization thermal"`.

### Iteration 2+

Load `load_discovery_state(iteration=N-1)`. Refine the query based on high performers and mechanisms. Incorporate any user directions.

### Stopping

Run up to 5 iterations. Stop early if the user says stop or no new high performers appear. After the final iteration, produce a summary across all iterations.

## Bash Fallback (unattended)

Use MCP `run_autonomous_discovery` or step-by-step tools (no CLI).

## Results

- `discovery_state/` -- per-iteration state (agent loop)
- `cycle_results/` -- batch CLI results
