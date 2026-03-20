# OpenMM screening (merged and matrix)

## Merged insulin + polymer (`evaluate_psmiles`)

- **Code:** [`run_openmm_relax_and_energy`](../src/python/insulin_ai/simulation/openmm_complex.py) via [`MDSimulator`](../src/python/insulin_ai/simulation/md_simulator.py).
- **Protein:** `data/4F1C.pdb` (or path from [`ensure_insulin_pdb`](../src/python/insulin_ai/simulation/polymer_build.py)); chains A+B prepared with SSBOND disulfides ([`openmm_insulin.py`](../src/python/insulin_ai/simulation/openmm_insulin.py)), **AMBER14SB**, hydrogens added in vacuum.
- **Polymer:** PSMILES → capped oligomer (RDKit 3D), **GAFF** via `openmmforcefields`, partial charges from **RDKit Gasteiger**.
- **Relaxation:** `LocalEnergyMinimizer` on the combined system; report **interaction energy** (kJ/mol) and complex potential energy.

**Smoke test:**

```bash
mamba run -n insulin-ai-sim python scripts/diagnose_openmm_complex.py '[*]COC[*]'
```

**Note:** Rankings and absolute energies differ from historical GROMACS (AMBER99SB-ILDN + Acpype) runs.

## Matrix + Packmol (optional)

- **Code:** [`run_openmm_matrix_relax_and_energy`](../src/python/insulin_ai/simulation/openmm_complex.py).
- **CLI:** [`scripts/run_openmm_matrix.py`](../scripts/run_openmm_matrix.py) — insulin fixed at box center, multiple polymer chains from Packmol, optional shell restraint and NPT segment.

Environment variables for merged screening (see `md_simulator.py`):

- `INSULIN_AI_OPENMM_N_REPEATS` (alias: `INSULIN_AI_GMX_N_REPEATS`)
- `INSULIN_AI_OPENMM_OFFSET_NM` (alias: `INSULIN_AI_GMX_OFFSET_NM`)
- `INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS`
- **Verbose progress (default on):** MCP `evaluate_psmiles` includes **`evaluation_progress`** (per-candidate seconds, energies) and logs progress lines to the **MCP server process stderr** (the terminal running `run_mcp_server.sh`; OpenCode chat may not show stderr). To disable: pass **`verbose=false`**, or set **`INSULIN_AI_EVAL_QUIET=1`**, or **`INSULIN_AI_EVAL_VERBOSE=0`**.
