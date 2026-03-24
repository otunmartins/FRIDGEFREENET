# OpenMM screening (MCP `evaluate_psmiles` = matrix / Packmol)

## Packmol matrix (default for `evaluate_psmiles`)

MCP **`evaluate_psmiles`** and [`MDSimulator.evaluate_candidates`](../src/python/insulin_ai/simulation/md_simulator.py) use **only** the matrix path:

- **Code:** [`run_openmm_matrix_relax_and_energy`](../src/python/insulin_ai/simulation/openmm_complex.py).
- **Requirement:** the **`packmol` binary on PATH**. If Packmol is missing, evaluation **raises** (no fallback). Install via conda-forge (`packmol`) or `pip install packmol` (see [`DEPENDENCIES.md`](DEPENDENCIES.md)).
- **Geometry:** Insulin (chains A+B) fixed at the box center; **N polymer chains** from **Packmol** in a periodic cube.
  - **`INSULIN_AI_OPENMM_MATRIX_PACKING_MODE=bulk` (default):** space-filling **bulk-in-cell** — no `outside sphere`; chains fill the box (overlap with insulin avoided by Packmol `tolerance`). Density-driven chain counts use **full-cell** volume (see [`matrix_density.py`](../src/python/insulin_ai/simulation/matrix_density.py)).
  - **`shell`:** annulus / encapsulation — polymers constrained with **`outside sphere`** (shell around insulin). Density-driven chain counts use **shell** volume (box minus inner sphere).
- **Relaxation:** `LocalEnergyMinimizer`, optional short **NPT** segment (`INSULIN_AI_OPENMM_MATRIX_NPT`), then **interaction energy** (kJ/mol). Spherical **shell restraints** during minimize apply only in **shell** mode unless you set a shell radius and enable restraint.

**CLI (same physics, more options):** [`scripts/run_openmm_matrix.py`](../scripts/run_openmm_matrix.py).

### Environment variables (matrix evaluation)

| Variable | Default (typical) | Role |
|----------|-------------------|------|
| `INSULIN_AI_OPENMM_N_REPEATS` (`INSULIN_AI_GMX_N_REPEATS`) | `4` | Repeat units per polymer chain |
| `INSULIN_AI_OPENMM_MATRIX_PACKING_MODE` | `bulk` | `bulk` = full-cell bulk packing (no `outside sphere`); `shell` = annulus (`outside sphere`) |
| `INSULIN_AI_OPENMM_MATRIX_FIXED_MODE` | `0` | If `1`, use fixed `N_POLYMERS` + `SHELL_A` below. If `0` (default), use **default density** unless overridden. |
| `INSULIN_AI_OPENMM_MATRIX_DEFAULT_DENSITY_G_CM3` | `0.52` | When not in fixed mode and `TARGET_DENSITY` is unset: derive `n_polymers` (and shell radius in **shell** mode) from this target polymer density (g/cm³). |
| `INSULIN_AI_OPENMM_MATRIX_N_POLYMERS` | `8` | Chain count when **fixed mode** (or explicit override when not using density) |
| `INSULIN_AI_OPENMM_MATRIX_BOX_NM` | `7.5` | Cubic box edge (nm). Larger cells need more chains to look “filled” at a given target density; old default **9.0** often looked sparse when *n* was capped. |
| `INSULIN_AI_OPENMM_MATRIX_DENSITY_N_MIN` | `4` | Lower clamp for density-derived chain count |
| `INSULIN_AI_OPENMM_MATRIX_DENSITY_N_MAX` | `100` | Upper clamp for density-derived chain count (lower values + large box ⇒ sparse visuals). Runtime is bounded by Packmol/OpenMM timeouts and agent budgets. |
| `INSULIN_AI_OPENMM_MATRIX_SHELL_A` | `14.0` | Shell inner radius (Å) for Packmol `outside sphere` (**shell** mode; fixed / non–density-driven) |
| `INSULIN_AI_OPENMM_MATRIX_TARGET_DENSITY_G_CM3` | *(optional)* | **Explicit** density driver; overrides default density when set (see `matrix_density.py`) |
| `INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS` | `2000` | Minimizer iterations |
| `INSULIN_AI_OPENMM_MATRIX_NPT` | `0` (off) | Set `1` to run short NPT after minimize |
| `INSULIN_AI_OPENMM_MATRIX_NPT_PS` | `0.5` | NPT length (ps) when NPT on |
| `INSULIN_AI_OPENMM_MATRIX_WALL_CLOCK_S` | `180` | Stop NPT when wall-clock exceeds this (seconds) |
| `INSULIN_AI_OPENMM_MATRIX_RESTRAIN_SHELL` | *(unset)* | If **unset**, minimize uses shell restraint only in **shell** mode (off for **bulk**). If set to `0`/`1`, forces off/on (on only applies when a shell radius exists). |
| `INSULIN_AI_OPENMM_MATRIX_BAROSTAT_INTERVAL_FS` | `10` | Barostat interval when NPT on |
| `INSULIN_AI_OPENMM_MATRIX_PROGRESSIVE_PACK` | `0` | If `1`, after the initial *n_polymers* (density or fixed), **greedily add chains** until the next Packmol run fails or times out, or limits below are hit. JSON includes **`packmol_progressive`**. |
| `INSULIN_AI_OPENMM_MATRIX_PACK_PER_ATTEMPT_TIMEOUT_S` | `120` | Per Packmol subprocess timeout (seconds) during progressive packing. |
| `INSULIN_AI_OPENMM_MATRIX_PACK_MAX_TOTAL_S` | *(unset)* | Optional **cumulative** wall-clock budget for all progressive Packmol attempts (unset = no total cap). |
| `INSULIN_AI_OPENMM_MATRIX_PROGRESSIVE_N_MAX` | *(unset)* | Optional **maximum** chain count (cap progressive growth). |
| **Verbose / quiet:** `INSULIN_AI_EVAL_QUIET=1` or `INSULIN_AI_EVAL_VERBOSE=0` | | Smaller JSON / no stderr progress |

**Note:** NPT is **off** by default so MCP runs finish in minutes; turn on for sampling-averaged interaction energy at the cost of runtime.

**Packing quality:** Each result includes **`packing_metrics`** (nearest protein–polymer heavy-atom distances in nm, fractions within 0.5 / 0.8 / 1.2 nm). Use **`min_polymer_protein_distance_nm`** and **`fraction_polymer_within_0.80_nm`** to spot sparse or disconnected polymer relative to insulin (e.g. very large min distance or low fraction within 0.8 nm after minimization).

## Fast merged insulin + single oligomer (diagnostics only)

**Not** used by `evaluate_psmiles`. For a quick vacuum merge of insulin + **one** offset oligomer (no Packmol):

- **Code:** [`run_openmm_relax_and_energy`](../src/python/insulin_ai/simulation/openmm_complex.py)
- **CLI:** [`scripts/diagnose_openmm_complex.py`](../scripts/diagnose_openmm_complex.py)

```bash
mamba run -n insulin-ai-sim python scripts/diagnose_openmm_complex.py '[*]COC[*]'
```

Use this only for debugging or comparisons; it does **not** model encapsulation.

## Session structure artifacts (reports)

When **`INSULIN_AI_SESSION_DIR`** points at the active run folder (or you pass MCP **`run_dir`** / **`artifacts_dir`**), each successful candidate gets files under **`<session>/structures/`**:

| File pattern | Content |
|--------------|---------|
| `<slug>_monomer.png` | 2D repeat unit (psmiles `savefig`) |
| `<slug>_complex_minimized.pdb` | Minimized **matrix** complex (insulin + many chains, periodic image) |
| `<slug>_complex_preview.png` | Matplotlib 3D scatter preview of that PDB |
| `<slug>_complex_chemviz.png` | PyMOL ray-traced insulin cartoon + polymer sticks (`pymol_complex_viz`); see `scripts/render_complex_chemviz.py` |

Disable with **`INSULIN_AI_EVAL_NO_STRUCTURE_ARTIFACTS=1`**. Override directory with **`INSULIN_AI_EVAL_ARTIFACTS_DIR`**. Requires **matplotlib** for monomer + `*_complex_preview.png`; **PyMOL on PATH** for `*_complex_chemviz.png`.

The MCP response includes **`structure_artifacts_dir`** and **`structure_artifact_paths`** when artifacts are written. See **`docs/SUMMARY_REPORT_STYLE.md`** for embedding in `SUMMARY_REPORT.md`.

Rankings and absolute energies differ from historical GROMACS (AMBER99SB-ILDN + Acpype) runs.
