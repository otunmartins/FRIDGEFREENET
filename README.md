# Insulin AI

AI-driven design of fridge-free insulin delivery patches. An AI assistant uses an MCP server to search literature, validate polymer structures (PSMILES), and run physics-based screening—so you can explore patch materials from inside an editor like Cursor.

---

## What you need

- **Conda or Mamba** — to install the simulation stack (OpenMM, RDKit, Packmol, psmiles). [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda works.
- **An IDE with MCP** — Cursor is the main target; others that support MCP will work too.
- **Optional:** API keys for literature search (see [docs/SECURITY.md](docs/SECURITY.md)).

---

## Quick start

1. Clone this repo and run the installer from the repo root:

   ```bash
   ./install
   ```

2. Copy the MCP config and fix the paths:

   ```bash
   cp .cursor/mcp.json.example .cursor/mcp.json
   # Edit .cursor/mcp.json and replace /ABSOLUTE/PATH/TO/insulin-ai with your real path
   ```

3. Restart Cursor so it loads the MCP server.

For step-by-step instructions (including Windows), see [docs/MCP_GETTING_STARTED.md](docs/MCP_GETTING_STARTED.md).

---

## Install by platform

### macOS and Linux

From the repo root:

```bash
./install
```

Or create the conda env manually:

```bash
mamba env create -f environment-simulation.yml
```

The environment is named **`insulin-ai-sim`**. You don't need to activate it to use MCP—the launcher script does that for you.

### Windows

The screening stack (OpenMM, Packmol, bash scripts) does **not** run on native Windows. Use **WSL2** (Windows Subsystem for Linux):

1. Install [WSL2 and Ubuntu](https://apps.microsoft.com/store/detail/ubuntu/9PDXGNCFSCZV).
2. Open a Ubuntu terminal and clone the repo inside Linux (e.g. `~/insulin-ai`).
3. Follow the same steps as macOS/Linux. In `.cursor/mcp.json`, use paths like `/home/yourname/insulin-ai/...`, not `C:\...`.

Full details: [docs/MCP_GETTING_STARTED.md](docs/MCP_GETTING_STARTED.md#windows-users-use-wsl2).

---

## Connect the MCP server in Cursor

1. Copy `.cursor/mcp.json.example` to `.cursor/mcp.json`.
2. Replace every `/ABSOLUTE/PATH/TO/insulin-ai` with the absolute path to your clone (e.g. `/Users/jane/insulin-ai` or `/home/jane/insulin-ai`).
3. Set `PAPER_DIRECTORY` and `PYTHONPATH` in the `env` block to point at your repo (the example shows the pattern).
4. Restart Cursor (Cmd/Ctrl+Shift+P → "Reload Window").

The MCP server runs via `scripts/run_mcp_server.sh`, which uses the `insulin-ai-sim` conda env automatically.

---

## Verify the setup

From a terminal:

```bash
mamba run -n insulin-ai-sim python scripts/diagnose_openmm_complex.py '[*]COC[*]'
```

You should see energies printed. If this fails, Packmol or OpenMM may be missing—see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

---

## Documentation

| Document | What it covers |
|----------|----------------|
| [MCP Getting Started](docs/MCP_GETTING_STARTED.md) | Full setup, WSL, Cursor config, troubleshooting. |
| [MCP Tool Reference](docs/MCP_SERVERS.md) | What each MCP tool does and what it needs. |
| [PSMILES Guide](docs/PSMILES_GUIDE.md) | Polymer notation and writing valid PSMILES. |
| [OpenMM Screening](docs/OPENMM_SCREENING.md) | How screening works, env vars, structure outputs. |
| [Dependencies](docs/DEPENDENCIES.md) | Conda env contents, Packmol, PyMOL, reporting libs. |
| [Security](docs/SECURITY.md) | API keys, where to put secrets, rotation. |
| [Project Structure](docs/PROJECT_STRUCTURE.md) | Repo layout, where code and config live. |
| [Summary Report Style](docs/SUMMARY_REPORT_STYLE.md) | How agent-written discovery reports are formatted. |
| [Third-Party Benchmarks](docs/THIRD_PARTY_BENCHMARKS.md) | Wisconsin and IBM benchmark adapters in `extern/`. |

---

## Other commands

**Optuna benchmark** (no MCP, same env):

```bash
mamba run -n insulin-ai-sim python benchmarks/optuna_psmiles_discovery.py --seed '[*]OCC[*]' --n-trials 5
```

**Matrix screening** (Packmol + OpenMM):

```bash
mamba run -n insulin-ai-sim python scripts/run_openmm_matrix.py '[*]CC[*]'
```
