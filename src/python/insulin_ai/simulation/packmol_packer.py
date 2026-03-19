#!/usr/bin/env python3
"""
Packmol integration for insulin + polymer matrix packing.

Packs insulin (1 copy, fixed) and polymer chains (N copies, inside box or
outside sphere shell). Uses Packmol binary (pip install packmol).
"""

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional


def _packmol_available() -> bool:
    """Check if packmol binary is on PATH."""
    return shutil.which("packmol") is not None


def build_packmol_inp_content(
    insulin_pdb_path: str,
    polymer_pdb_path: str,
    n_polymers: int,
    output_path: str,
    box_size_nm: float,
    tolerance_angstrom: float,
    seed: int,
    shell_only_angstrom: Optional[float] = None,
) -> str:
    """Build Packmol input text (for tests and inspection)."""
    box_angstrom = box_size_nm * 10.0
    half = box_angstrom / 2.0
    xmin, ymin, zmin = -half, -half, -half
    xmax, ymax, zmax = half, half, half
    polymer_constraints = f"  inside box {xmin} {ymin} {zmin} {xmax} {ymax} {zmax}\n"
    if shell_only_angstrom is not None and shell_only_angstrom > 0:
        if shell_only_angstrom < half - 1.0:
            polymer_constraints += (
                f"  outside sphere 0. 0. 0. {shell_only_angstrom}\n"
            )
    return f"""tolerance {tolerance_angstrom}
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
{polymer_constraints}end structure
"""


def pack_insulin_polymers(
    insulin_pdb_path: str,
    polymer_pdb_path: str,
    n_polymers: int,
    output_path: str,
    box_size_nm: float = 6.0,
    tolerance_angstrom: float = 2.0,
    seed: int = 42,
    timeout_s: int = 300,
    shell_only_angstrom: Optional[float] = None,
) -> bool:
    """
    Pack insulin and N polymer chains into a box using Packmol.

    Insulin is fixed at origin. Polymers are placed inside the box with no overlaps.
    If shell_only_angstrom is set (e.g. 15), each polymer is constrained with
    ``outside sphere 0. 0. 0. R`` so chains lie in a spherical shell (matrix
    shell around insulin); R must be less than half the box edge in Å.

    Args:
        insulin_pdb_path: Path to insulin PDB (with hydrogens).
        polymer_pdb_path: Path to single polymer chain PDB.
        n_polymers: Number of polymer chains to pack.
        output_path: Path for packed output PDB.
        box_size_nm: Cubic box edge length in nm. Converted to Angstroms for Packmol.
        tolerance_angstrom: Min distance between atoms from different molecules (Angstroms).
        seed: Random seed for packmol.
        timeout_s: Subprocess timeout (large N may need more time).
        shell_only_angstrom: If set, polymers only outside sphere of this radius (Å).

    Returns:
        True on success, False on failure (packmol not found, error, etc.).
    """
    if not _packmol_available():
        warnings.warn("packmol not found. Install via: pip install packmol")
        return False

    insulin_pdb_path = str(Path(insulin_pdb_path).resolve())
    polymer_pdb_path = str(Path(polymer_pdb_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(insulin_pdb_path).is_file():
        warnings.warn(f"Insulin PDB not found: {insulin_pdb_path}")
        return False
    if not Path(polymer_pdb_path).is_file():
        warnings.warn(f"Polymer PDB not found: {polymer_pdb_path}")
        return False

    if shell_only_angstrom is not None and shell_only_angstrom > 0:
        half = box_size_nm * 5.0
        if shell_only_angstrom >= half - 1.0:
            warnings.warn(
                "shell_only_angstrom should be < half box (Å); disabling shell constraint"
            )
            shell_only_angstrom = None

    inp_content = build_packmol_inp_content(
        insulin_pdb_path,
        polymer_pdb_path,
        n_polymers,
        output_path,
        box_size_nm,
        tolerance_angstrom,
        seed,
        shell_only_angstrom,
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".inp", delete=False, encoding="utf-8"
        ) as f:
            f.write(inp_content)
            inp_path = f.name

        packmol_exe = shutil.which("packmol")
        with open(inp_path, encoding="utf-8") as inp_file:
            result = subprocess.run(
                [packmol_exe],
                stdin=inp_file,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(Path(output_path).parent),
            )
        os.unlink(inp_path)

        if result.returncode != 0:
            warnings.warn(
                f"Packmol failed (exit {result.returncode}): {result.stderr or result.stdout}"
            )
            return False

        return Path(output_path).is_file()
    except subprocess.TimeoutExpired:
        warnings.warn("Packmol timed out")
        return False
    except Exception as e:
        warnings.warn(f"Packmol error: {e}")
        return False
