# Dependencies

| File | Role |
|------|------|
| **`environment-simulation.yml`** | **conda env `insulin-ai-sim`:** Python, **rdkit**, **openmm**, **pdbfixer**, pip (openmmforcefields, openff-toolkit, packmol, psmiles, **optuna**, **fpdf2**, **markdown**, mcp, paper-qa, `-e .`) |
| **`requirements.txt`** | Pointer to conda env |

## Simulation / evaluate

- **OpenMM**, **openmmforcefields**, **OpenFF Toolkit**, **RDKit**, **pdbfixer**: merged insulin + polymer minimize and interaction energy (see `docs/OPENMM_SCREENING.md`).
- **`INSULIN_AI_OPENMM_N_REPEATS`** (alias `INSULIN_AI_GMX_N_REPEATS`), **`INSULIN_AI_OPENMM_OFFSET_NM`** (alias `INSULIN_AI_GMX_OFFSET_NM`), **`INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS`** optional.

```bash
pytest tests/test_simulation.py tests/test_openmm_complex.py tests/test_material_mappings.py -v
```

## Benchmark (Optuna PSMILES discovery)

- **Preferred:** use conda env **`insulin-ai-sim`** (same as OpenMM screening and MCP). After `mamba env update -f environment-simulation.yml` or `./install`, **Optuna** is installed with the rest of the pip stack.
- **Without conda:** `pip install -e ".[benchmark]"` adds **Optuna** only; you still need the OpenMM/RDKit stack for real evaluation (see `docs/OPENMM_SCREENING.md`).
- **Run (real physics):** `mamba run -n insulin-ai-sim python benchmarks/optuna_psmiles_discovery.py --seed '[*]OCC[*]' --n-trials 5`
- **Does not** require MCP, LLMs, or literature tools.

## External third-party benchmarks (non-BO; MCP-independent)

Clones live under **`extern/benchmarks/`** (gitignored); see **[`docs/THIRD_PARTY_BENCHMARKS.md`](THIRD_PARTY_BENCHMARKS.md)** and **`scripts/clone_external_benchmarks.sh`**.

| System | Upstream deps | insulin-ai `pyproject.toml` |
|--------|----------------|-----------------------------|
| **Polymer Generative Models Benchmark** (Wisconsin) | PyTorch / MOSES / upstream Zenodo per [ytl0410](https://github.com/ytl0410/Polymer-Generative-Models-Benchmark) | Not pinned; install in a separate venv when running full training |
| **IBM logical-agent polymer RL** | `pip install -e md-envs` + data zip per [IBM repo](https://github.com/IBM/logical-agent-driven-polymer-discovery) | Not pinned |

Thin smoke scripts: `python benchmarks/polymer_generative_models_benchmark.py`, `python benchmarks/ibm_polymer_rl_benchmark.py` (no extra insulin-ai deps beyond Python).

## MCP — discovery figures & PDF reports

These power **insulin-ai** MCP tools; they are **not** used by the Optuna benchmark unless you import reporting helpers.

| Dependency | Role | If missing |
|------------|------|------------|
| **[psmiles](https://github.com/FermiQ/psmiles)** (git in `pyproject.toml`) | `PolymerSmiles.savefig` — 2D PNG of repeat units; `render_psmiles_png` | Tool returns install hint; use `insulin-ai-sim` |
| **fpdf2** | PDF output (`compile_discovery_markdown_to_pdf`, batch `write_discovery_summary_report`) | PDF step fails; error JSON lists `pip install fpdf2` |
| **markdown** | MD → HTML before PDF (`compile_discovery_markdown_to_pdf`) | Tool error; `pip install markdown` |
| **duckduckgo-search** | `validate_psmiles(..., crosscheck_web=true)` snippets | Cross-check disabled or error per tool |

**AI-driven reporting (preferred):** the agent writes `SUMMARY_REPORT.md`, calls `render_psmiles_png` for figures, then `compile_discovery_markdown_to_pdf`. No extra packages beyond the table above for that path.

**Optional:** `write_discovery_summary_report` rebuilds a minimal MD+PDF from `agent_iteration_*.json` only (quick skeleton, no narrative)—same dependencies.

**Beyond this repo’s defaults:** if you add Pandoc, LaTeX, or WeasyPrint yourself, document them in your environment; they are **not** required by the shipped tools.
