"""Matrix GRO/top helpers; no GROMACS required."""

from pathlib import Path

import pytest

from insulin_ai.simulation.gromacs_complex import (
    _fix_system_top_n_ligands,
    _write_gro,
    write_matrix_gro_from_packmol,
)
from insulin_ai.simulation.gro_pdb_io import gro_to_pdb, read_pdb_coords_nm


def test_gro_to_pdb_roundtrip_count(tmp_path):
    g = tmp_path / "a.gro"
    _write_gro(str(g), "x", [(1, "ALA", "CA", 1.0, 0.0, 0.0), (1, "ALA", "CB", 1.1, 0.0, 0.0)], (5, 5, 5))
    p = tmp_path / "a.pdb"
    gro_to_pdb(str(g), str(p))
    c = read_pdb_coords_nm(str(p))
    assert len(c) == 2
    assert abs(c[0][2] - 1.0) < 1e-5  # nm


def test_fix_system_top_n_ligands(tmp_path):
    prot = tmp_path / "protein.top"
    prot.write_text(
        """#include "amber99sb-ildn.ff/forcefield.itp"
[ system ]
X
[ molecules ]
Protein_chain_A     1
"""
    )
    lig = tmp_path / "lig.itp"
    lig.write_text(
        """[ moleculetype ]
LIG              3
[ atoms ]
1   C   1  LIG  C1  1   0.0  12.01
"""
    )
    out = _fix_system_top_n_ligands(str(tmp_path), str(prot), str(lig), 5)
    t = Path(out).read_text()
    assert "ligand.itp" in t
    assert "LIG              5" in t


def test_write_matrix_gro_from_packmol(tmp_path):
    _write_gro(
        str(tmp_path / "prot.gro"),
        "p",
        [(1, "ALA", "CA", 0.0, 0.0, 0.0)],
        (4, 4, 4),
    )
    _write_gro(
        str(tmp_path / "lig.gro"),
        "l",
        [(1, "LIG", "C1", 2.0, 0.0, 0.0), (1, "LIG", "C2", 2.2, 0.0, 0.0)],
        (2, 2, 2),
    )
    # packed.pdb: 1 prot + 2 lig copies = 5 atoms; coords arbitrary (nm in read = Å/10)
    pdb = tmp_path / "packed.pdb"
    pdb.write_text(
        """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C1  LIG A   2       3.000   0.000   0.000  1.00  0.00           C
ATOM      3  C2  LIG A   2       3.200   0.000   0.000  1.00  0.00           C
ATOM      4  C1  LIG A   3       5.000   1.000   0.000  1.00  0.00           C
ATOM      5  C2  LIG A   3       5.200   1.000   0.000  1.00  0.00           C
END
"""
    )
    out = tmp_path / "sys.gro"
    write_matrix_gro_from_packmol(
        str(tmp_path / "prot.gro"),
        str(tmp_path / "lig.gro"),
        str(pdb),
        2,
        str(out),
    )
    text = out.read_text()
    assert "5\n" in text  # atom count line


def test_suggest_n_chains_for_density():
    from insulin_ai.simulation.matrix_density import suggest_n_chains_for_density

    # ~28 g/mol per PE repeat, 10 repeats -> ~280 g/mol per chain
    n = suggest_n_chains_for_density(10.0, 280.0, target_density_g_cm3=0.85)
    assert n >= 10


def test_suggest_n_around_insulin_sane_range():
    from insulin_ai.simulation.matrix_density import suggest_n_polymer_around_insulin

    n = suggest_n_polymer_around_insulin(9.0, 280.0, 10)
    assert 10 <= n <= 28


def test_estimate_chain_mw_pe_no_crash():
    from insulin_ai.simulation.matrix_density import estimate_chain_mw_g_mol

    mw32 = estimate_chain_mw_g_mol("[*]CC[*]", 32)
    assert 850 < mw32 < 920  # ~28*32 + 2


def test_rdkit_polymer_pdb():
    from insulin_ai.simulation.psp_polymer_build import rdkit_polymer_pdb

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.pdb"
        ok = rdkit_polymer_pdb("[*]CC[*]", 2, str(p))
        assert ok and p.is_file()
