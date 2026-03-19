#!/usr/bin/env python3
"""Packmol + GROMACS: insulin fixed at center, N polymer chains, then EM."""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "python"))

_HINT = """
Requires: gmx, acpype, packmol (pip install packmol), insulin-ai-sim env.
  mamba activate insulin-ai-sim

Suggest chain count (no gmx/packmol; safe for large --n-repeats):
  python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --n-repeats 32 --suggest-n --box-nm 14

Just enough polymer around insulin (PBC), not a full melt:
  python scripts/diagnose_gromacs_matrix.py '[*]CC[*]' --around-insulin --save -v
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("psmiles", nargs="?", default="[*]CC[*]")
    ap.add_argument("--n-repeats", type=int, default=4, help="Repeat units per chain (longer = slower Packmol/acpype)")
    ap.add_argument("--n-polymers", type=int, default=4, help="Number of chains (Packmol)")
    ap.add_argument(
        "--suggest-n",
        action="store_true",
        help="Print suggested n_polymers from box + chain MW @ ~0.85 g/cm3, then exit (no RDKit embed)",
    )
    ap.add_argument(
        "--suggest-around-insulin",
        action="store_true",
        help="Print modest n_polymers + box for surrounding insulin only, then exit",
    )
    ap.add_argument(
        "--around-insulin",
        action="store_true",
        help="Preset: ~9 nm box, shell around insulin, ~10 repeats, n_polymers auto (then run GROMACS)",
    )
    ap.add_argument("--box-nm", type=float, default=8.0, help="Cubic box edge (nm)")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Packmol min intermolecular distance (Å); lower = denser, harder to pack",
    )
    ap.add_argument(
        "--packmol-timeout",
        type=int,
        default=None,
        help="Packmol subprocess seconds (default: min(3600, --timeout))",
    )
    ap.add_argument(
        "--acpype-timeout",
        type=int,
        default=7200,
        metavar="SEC",
        help="acpype max seconds (default 7200; max ~1.5e6; larger values overflow Python)",
    )
    ap.add_argument(
        "--acpype-no-timeout",
        action="store_true",
        help="No acpype time limit (risk: hung job runs forever)",
    )
    ap.add_argument(
        "--acpype-charge",
        choices=("gas", "bcc"),
        default="gas",
        help="gas=Gasteiger fast (default matrix); bcc=AM1-BCC accurate but slow + acpype 3h antechamber cap",
    )
    ap.add_argument(
        "--shell-angstrom",
        type=float,
        default=None,
        metavar="R",
        help="Optional: polymers only outside sphere radius R (Å) around origin",
    )
    ap.add_argument("--psp-polymer", action="store_true", help="Prefer PSP MoleculeBuilder PDB")
    ap.add_argument(
        "--save",
        nargs="?",
        const=os.path.join(REPO, "runs", "gmx_matrix_last"),
        default=None,
        metavar="DIR",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Progress per stage (pdb2gmx, acpype, packmol, grompp, mdrun) + timings",
    )
    args = ap.parse_args()

    # Preset: enough polymer around insulin (not melt-scale)
    if args.around_insulin:
        args.box_nm = 9.0
        args.shell_angstrom = 14.0
        # Short enough chains to avoid psmiles/RDKit segfault on embed
        args.n_repeats = min(max(args.n_repeats, 8), 12)
        from insulin_ai.simulation.matrix_density import (
            estimate_chain_mw_g_mol,
            suggest_n_polymer_around_insulin,
        )

        mw = estimate_chain_mw_g_mol(args.psmiles, args.n_repeats)
        args.n_polymers = suggest_n_polymer_around_insulin(
            args.box_nm, mw, args.n_repeats
        )
        print(
            f"--around-insulin preset: box={args.box_nm} nm, n_repeats={args.n_repeats}, "
            f"n_polymers={args.n_polymers}, shell={args.shell_angstrom} Å"
        )

    # Early exit: no psmiles dimer stack, no RDKit embed (prevents segfault on large n)
    if args.suggest_n:
        from insulin_ai.simulation.matrix_density import (
            estimate_chain_mw_g_mol,
            suggest_n_chains_for_density,
        )

        mw = estimate_chain_mw_g_mol(args.psmiles, args.n_repeats)
        n = suggest_n_chains_for_density(args.box_nm, mw)
        print(f"Chain MW ~{mw:.1f} g/mol (estimate; no 3D build)")
        print(f"Box {args.box_nm} nm -> suggest n_polymers ≈ {n}")
        print("(Entanglement still needs NPT MD after Packmol+EM.)")
        sys.exit(0)

    if args.suggest_around_insulin:
        from insulin_ai.simulation.matrix_density import (
            estimate_chain_mw_g_mol,
            suggest_n_polymer_around_insulin,
        )

        nr = min(max(args.n_repeats, 8), 12)
        box = 9.0
        mw = estimate_chain_mw_g_mol(args.psmiles, nr)
        n = suggest_n_polymer_around_insulin(box, mw, nr)
        print("Modest setup to surround insulin in PBC (not full-melt density):")
        print(f"  box_nm={box}  n_repeats={nr}  n_polymers={n}  shell_angstrom=14")
        print(f"  chain MW ~{mw:.0f} g/mol")
        print("Run:")
        print(
            f"  python scripts/diagnose_gromacs_matrix.py '{args.psmiles}' "
            f"--around-insulin --save"
        )
        sys.exit(0)

    from insulin_ai.simulation.gromacs_complex import gmx_available, run_gromacs_matrix_em
    from insulin_ai.simulation.packmol_packer import _packmol_available

    acpype_to = None if args.acpype_no_timeout else args.acpype_timeout

    if not gmx_available() or not _packmol_available():
        print("gmx:", gmx_available(), "packmol:", _packmol_available())
        print(_HINT)
        sys.exit(1)

    r = run_gromacs_matrix_em(
        args.psmiles,
        n_repeats=args.n_repeats,
        n_polymers=args.n_polymers,
        box_size_nm=args.box_nm,
        timeout_s=args.packmol_timeout or 3600,
        save_structures_dir=args.save,
        prefer_psp_polymer=args.psp_polymer,
        shell_only_angstrom=args.shell_angstrom,
        packmol_tolerance_angstrom=args.tolerance,
        packmol_timeout_s=args.packmol_timeout,
        acpype_timeout_s=acpype_to,
        acpype_charge=args.acpype_charge,
        verbose=args.verbose,
        log=(lambda m: print(m, flush=True)) if args.verbose else None,
    )
    print(json.dumps(r, indent=2) if r else "run_gromacs_matrix_em returned None")
    if r and r.get("saved_structures_dir"):
        print("Open:", os.path.join(r["saved_structures_dir"], "em.gro"))


if __name__ == "__main__":
    main()
