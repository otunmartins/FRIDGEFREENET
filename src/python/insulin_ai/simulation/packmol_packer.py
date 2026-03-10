#!/usr/bin/env python3
"""
Packmol integration for insulin + polymer matrix packing.

Packs insulin (1 copy, fixed) and polymer chains (N copies, inside box) with
no overlaps. Uses Packmol binary (pip install packmol).
"""

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path


def _packmol_available() -> bool:
    """Check if packmol binary is on PATH."""
    return shutil.which("packmol") is not None


def pack_insulin_polymers(
    insulin_pdb_path: str,
    polymer_pdb_path: str,
    n_polymers: int,
    output_path: str,
    box_size_nm: float = 6.0,
    tolerance_angstrom: float = 2.0,
    seed: int = 42,
) -> bool:
    """
    Pack insulin and N polymer chains into a box using Packmol.

    Insulin is fixed at origin. Polymers are placed inside the box with no overlaps.

    Args:
        insulin_pdb_path: Path to insulin PDB (with hydrogens).
        polymer_pdb_path: Path to single polymer chain PDB.
        n_polymers: Number of polymer chains to pack.
        output_path: Path for packed output PDB.
        box_size_nm: Cubic box edge length in nm. Converted to Angstroms for Packmol.
        tolerance_angstrom: Min distance between atoms from different molecules (Angstroms).
        seed: Random seed for packmol.

    Returns:
        True on success, False on failure (packmol not found, error, etc.).
    """
    if not _packmol_available():
        warnings.warn("packmol not found. Install via: pip install packmol")
        return False

    insulin_pdb_path = Path(insulin_pdb_path).resolve()
    polymer_pdb_path = Path(polymer_pdb_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not insulin_pdb_path.is_file():
        warnings.warn(f"Insulin PDB not found: {insulin_pdb_path}")
        return False
    if not polymer_pdb_path.is_file():
        warnings.warn(f"Polymer PDB not found: {polymer_pdb_path}")
        return False

    # Packmol uses Angstroms
    box_angstrom = box_size_nm * 10.0
    half = box_angstrom / 2.0
    xmin, ymin, zmin = -half, -half, -half
    xmax, ymax, zmax = half, half, half

    inp_content = f"""tolerance {tolerance_angstrom}
filetype pdb
output {output_path}
seed {seed}

structure {insulin_pdb_path}
  number 1
  center
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure {polymer_pdb_path}
  number {n_polymers}
  inside box {xmin} {ymin} {zmin} {xmax} {ymax} {zmax}
end structure
"""

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inp", delete=False
        ) as f:
            f.write(inp_content)
            inp_path = f.name

        packmol_exe = shutil.which("packmol")
        with open(inp_path) as inp_file:
            result = subprocess.run(
                [packmol_exe],
                stdin=inp_file,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(output_path.parent) if output_path.parent else ".",
            )
        os.unlink(inp_path)

        if result.returncode != 0:
            warnings.warn(
                f"Packmol failed (exit {result.returncode}): {result.stderr or result.stdout}"
            )
            return False

        return output_path.is_file()
    except subprocess.TimeoutExpired:
        warnings.warn("Packmol timed out")
        return False
    except Exception as e:
        warnings.warn(f"Packmol error: {e}")
        return False
