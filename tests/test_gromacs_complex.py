"""GROMACS complex helpers."""

from pathlib import Path

from insulin_ai.simulation.gromacs_complex import (
    _fix_system_top,
    merge_gro_translate_ligand,
    _write_gro,
    _read_gro,
)


def test_fix_system_top_includes_ligand(tmp_path):
    prot = tmp_path / "protein.top"
    prot.write_text(
        """#include "amber99sb-ildn.ff/forcefield.itp"
[ system ]
Insulin
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
    out = _fix_system_top(str(tmp_path), str(prot), str(lig))
    text = Path(out).read_text()
    assert "ligand.itp" in text
    # ligand must follow forcefield (atomtypes before moleculetype rule)
    assert text.index("ligand.itp") < text.index("[ system ]")
    assert "LIG" in text.split("[ molecules ]")[-1]


def test_merge_two_gros(tmp_path):
    p1 = tmp_path / "a.gro"
    p2 = tmp_path / "b.gro"
    _write_gro(str(p1), "a", [(1, "A", "C", 0, 0, 0)], (5, 5, 5))
    _write_gro(str(p2), "b", [(1, "B", "C", 5, 0, 0)], (2, 2, 2))
    out = tmp_path / "m.gro"
    merge_gro_translate_ligand(str(p1), str(p2), str(out), (1.0, 0, 0))
    _, atoms = _read_gro(str(out))
    assert len(atoms) == 2
