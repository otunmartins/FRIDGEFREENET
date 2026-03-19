#!/usr/bin/env python3
"""Smoke test: gmx + acpype + merged EM for one PSMILES (same path as evaluate_psmiles)."""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "python"))

from insulin_ai.simulation.gromacs_complex import gmx_available, run_gromacs_merged_em

_HINT = """
gmx not on PATH. Use: mamba activate insulin-ai-sim
  (or) mamba run -n insulin-ai-sim python scripts/diagnose_gromacs_complex.py 'PSMILES'
Create env: mamba env create -f environment-simulation.yml
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("psmiles", nargs="?", default="[*]COC[*]")
    ap.add_argument(
        "--save",
        nargs="?",
        const=os.path.join(REPO, "runs", "gmx_last_structure"),
        default=None,
        metavar="DIR",
        help="Export gro/top for VMD/PyMOL. Default DIR: runs/gmx_last_structure",
    )
    args = ap.parse_args()

    if not gmx_available():
        print("gmx on PATH: False")
        for runner in ("mamba", "conda"):
            if shutil.which(runner):
                try:
                    r = subprocess.run(
                        [runner, "run", "-n", "insulin-ai-sim", "which", "gmx"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        print("insulin-ai-sim has gmx:", r.stdout.strip())
                except Exception:
                    pass
                break
        print(_HINT)
        sys.exit(1)

    print("gmx on PATH: True")
    save_dir = args.save  # None, or path from --save [DIR]
    r = run_gromacs_merged_em(args.psmiles, n_repeats=2, save_structures_dir=save_dir)
    print(json.dumps(r, indent=2) if r else "run_gromacs_merged_em returned None")
    if r and r.get("saved_structures_dir"):
        print()
        print("Open structure:")
        print("  vmd", os.path.join(r["saved_structures_dir"], "em.gro"))


if __name__ == "__main__":
    main()
