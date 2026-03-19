"""Packmol import; no OpenMM."""

import pytest


def test_packmol_packer_import():
    from insulin_ai.simulation.packmol_packer import pack_insulin_polymers, _packmol_available

    assert callable(pack_insulin_polymers)
    assert isinstance(_packmol_available(), bool)


def test_packmol_inp_contains_shell_when_requested():
    from insulin_ai.simulation.packmol_packer import build_packmol_inp_content

    s = build_packmol_inp_content(
        "/a/ins.pdb",
        "/b/poly.pdb",
        3,
        "/c/out.pdb",
        box_size_nm=10.0,
        tolerance_angstrom=2.0,
        seed=1,
        shell_only_angstrom=12.0,
    )
    assert "outside sphere" in s
    assert "12.0" in s


def test_polymer_build_ensure_pdb_or_skip():
    from insulin_ai.simulation.polymer_build import ensure_insulin_pdb

    try:
        p = ensure_insulin_pdb()
        assert p.endswith(".pdb")
    except FileNotFoundError:
        pytest.skip("4F1C.pdb not present")
