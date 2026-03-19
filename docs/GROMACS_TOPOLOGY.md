# GROMACS insulin topology

`evaluate_psmiles` runs **`gmx pdb2gmx`** on **`data/4F1C.pdb`** at evaluation time (amber99sb-ildn, tip3p, `-ignh`). Disulfide prompts are answered with **`y`** via stdin (see [`gromacs_complex.py`](../src/python/insulin_ai/simulation/gromacs_complex.py)).

To debug SS handling offline:

```bash
bash scripts/prepare_gromacs_insulin_topology.sh
```

Outputs under `src/python/insulin_ai/simulation/data/gromacs/` (optional commit for air-gapped runs).

Polymer topology is generated per candidate with **Acpype** (GAFF) from an RDKit **SDF**.

Committed **protein.gro / protein.top** (optional): run `bash scripts/prepare_gromacs_insulin_topology.sh` and commit `simulation/data/gromacs/` to skip interactive pdb2gmx in air-gapped runs (not required by default).
</think>


<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
Shell
