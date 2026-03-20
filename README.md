# Insulin AI

1. **Install** via **`./install`** or **`mamba env create -f environment-simulation.yml`**. Stack: **RDKit**, **OpenMM**, **openmmforcefields**, **OpenFF Toolkit**, **pdbfixer**, MCP, psmiles.
2. **MCP**: `bash scripts/run_mcp_server.sh` (env `insulin-ai-sim`).
3. **`evaluate_psmiles`**: OpenMM merged minimize (AMBER14SB + GAFF + Gasteiger) + interaction energy; needs **`data/4F1C.pdb`** (or bundled insulin PDB via `ensure_insulin_pdb`).

See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md), [docs/OPENMM_SCREENING.md](docs/OPENMM_SCREENING.md), [docs/PSMILES_GUIDE.md](docs/PSMILES_GUIDE.md), [docs/SUMMARY_REPORT_STYLE.md](docs/SUMMARY_REPORT_STYLE.md) (agent-authored session summaries), [docs/THIRD_PARTY_BENCHMARKS.md](docs/THIRD_PARTY_BENCHMARKS.md) (optional Wisconsin + IBM clones under `extern/benchmarks/`).

**Diagnose OpenMM screening (needs `insulin-ai-sim`, not plain `(base)`):**  
`mamba run -n insulin-ai-sim python scripts/diagnose_openmm_complex.py '[*]COC[*]'`

**Optuna PSMILES benchmark (agent-free; same env — OpenMM + Optuna):**  
`mamba run -n insulin-ai-sim python benchmarks/optuna_psmiles_discovery.py --seed '[*]OCC[*]' --n-trials 5`

**Matrix (Packmol + OpenMM):** `python scripts/run_openmm_matrix.py '[*]CC[*]'`
