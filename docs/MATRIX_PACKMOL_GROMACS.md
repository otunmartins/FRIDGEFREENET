# Insulin in a polymer matrix (Packmol + GROMACS)

## Flow

1. **Protein** — `pdb2gmx` on insulin → `protein.gro` / `protein.top`.
2. **Packmol insulin template** — same atom order as `protein.gro` via `gro_to_pdb(protein.gro)`.
3. **Polymer** — RDKit oligomer PDB (default) or optional **PSP** `MoleculeBuilder` (`psp_polymer_build.build_polymer_pdb_for_packmol(..., prefer_psp=True)`).
4. **Acpype** — single-chain SDF → `LIG_GMX.itp` + `LIG_GMX.gro`; Packmol polymer template from `gro_to_pdb(LIG_GMX.gro)`.
5. **Packmol** — insulin fixed at origin; `N` polymer chains inside the box (optional **`outside sphere`** shell around origin).
6. **GROMACS** — `write_matrix_gro_from_packmol` + `system.top` with **N** ligand molecules → `grompp` / `mdrun` EM.

## Scripts

```bash
# Single oligomer next to insulin (legacy)
python scripts/diagnose_gromacs_complex.py '[*]CC[*]' --save

# Enough polymer around insulin in PBC (~9 nm box, shell outside 14 Å, ~10–24 chains)
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --around-insulin --save
# Preview counts only:
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --suggest-around-insulin

# Matrix: fixed insulin + N chains (manual)
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --n-polymers 6 --box-nm 10 --save
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --shell-angstrom 15 --box-nm 12 --save
```

## PSP (optional)

PSP must be on `PYTHONPATH` (e.g. repo `PSP/`). Heavy deps (OpenBabel, etc.). If PSP fails, RDKit oligomer PDB is used automatically.

## Long chains that “fill” insulin — what is actually possible

| Goal | Packmol + EM | Reality |
|------|----------------|--------|
| **Long chains** | Yes — raise `--n-repeats` (RDKit may fail for very long; use **PSP** `--psp-polymer`) | Acpype once per chemistry; topology replicated **N** times (identical chains). |
| **Dense / space-filling** | Yes — many chains: `--suggest-n` estimates **n_polymers** from box volume and ~0.85 g/cm³; lower `--tolerance` (e.g. 1.6 Å) packs tighter. | Packmol can fail if the box is too full; increase box or reduce **N** or tolerance. |
| **Entangled melt** | **No** — initial geometry is non-overlapping rigid placements. | **Entanglement = MD**: run **NVT then NPT** (often 10 ns–µs scale depending on MW and viscosity). No shortcut from Packmol alone. |

So: you *can* build a **long, dense polymer bath around insulin**; you **cannot** get a true entangled melt until you **equilibrate** that system.

```bash
# Suggested chain count (rough melt density; insulin steals some volume)
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --n-repeats 32 --suggest-n --box-nm 14

# Heavy run: long chains + many copies (long Packmol; use screen/tmux)
python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --n-repeats 32 --n-polymers 120 \
  --box-nm 14 --tolerance 1.7 --packmol-timeout 7200 --save
```

Then: **NPT equilibration** (not included in this repo’s one-shot EM) to let chains reptate and entangle.

## acpype “Semi-QM” abort (~3 h)

Matrix uses **long oligomers**. Acpype default **bcc** runs antechamber **AM1-BCC** (semi-empirical QM) on the **whole** chain — hours on large molecules. Acpype **kills antechamber after ~10800 s** regardless of `--acpype-no-timeout`.

- **Default matrix charge model:** `--acpype-charge gas` (Gasteiger, fast) so you reach **Packmol + EM**.
- For publication-quality charges on long chains, parameterize a **short** fragment or use another workflow; not full-chain AM1-BCC.

## Limits

- Large **N** or long chains: Packmol runtime and acpype still single topology replicated **N** times (identical chains only).
- Post-EM you still want **NVT/NPT** for a realistic melt and entanglement.
