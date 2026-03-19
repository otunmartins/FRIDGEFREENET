# Insulin AI

1. **Install** via **`./install`** or **`mamba env create -f environment-simulation.yml`**. Stack: **RDKit**, **GROMACS**, **AmberTools**, **acpype** (pip), MCP, psmiles.
2. **MCP**: `bash scripts/run_mcp_server.sh` (env `insulin-ai-sim`).
3. **`evaluate_psmiles`**: **gmx** + **acpype** on PATH + **`data/4F1C.pdb`**; merged insulin (AMBER99SB-ILDN) + polymer (GAFF) energy minimization.

See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md), [docs/GROMACS_TOPOLOGY.md](docs/GROMACS_TOPOLOGY.md).

**Diagnose GROMACS (needs `insulin-ai-sim`, not plain `(base)`):**  
`mamba run -n insulin-ai-sim python scripts/diagnose_gromacs_complex.py '[*]COC[*]'`
