"""Tests for matplotlib ribbon + ball-stick complex visualization."""

import pytest


MINIMAL_PDB = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       7.600   0.500   0.000  1.00  0.00           C
HETATM    4  C1  UNK C   1      20.000   0.000   0.000  1.00  0.00           C
HETATM    5  O1  UNK C   1      21.200   0.000   0.000  1.00  0.00           O
CONECT    4  5
END
"""


def test_write_complex_chemviz_png(tmp_path):
    pytest.importorskip("matplotlib")
    from insulin_ai.simulation.chemviz_complex import write_complex_chemviz_png

    pdb = tmp_path / "c.pdb"
    pdb.write_text(MINIMAL_PDB, encoding="utf-8")
    png = tmp_path / "out.png"
    r = write_complex_chemviz_png(str(pdb), str(png), protein_chains=("A",))
    assert r.get("ok") is True, r
    assert png.is_file()
    assert r.get("n_ca_segments") == 1
    assert r.get("n_polymer_heavy") == 2
    assert r.get("n_bonds_drawn") >= 1
