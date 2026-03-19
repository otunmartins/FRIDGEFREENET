# Dependencies

| File | Role |
|------|------|
| **`environment-simulation.yml`** | **conda:** Python, **rdkit**, **gromacs**, **ambertools**, pip (**acpype**, psmiles, mcp, paper-qa, `-e .`) |
| **`requirements.txt`** | Pointer to conda env |

## Simulation / evaluate

- **gmx** on PATH, **acpype**, **antechamber** (AmberTools): merged **pdb2gmx** (amber99sb-ildn) + **GAFF** polymer → **mdrun** EM.
- **`INSULIN_AI_GMX_N_REPEATS`**, **`INSULIN_AI_GMX_OFFSET_NM`** optional.

```bash
pytest tests/test_simulation.py tests/test_gromacs_complex.py tests/test_material_mappings.py -v
```
