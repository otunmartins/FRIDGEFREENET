"""Simulation stack: RDKit + GROMACS (gmx optional on PATH)."""

import pytest


def test_mdsimulator_requires_gmx():
    from insulin_ai.simulation.gromacs_complex import gmx_available
    from insulin_ai.simulation import MDSimulator

    if not gmx_available():
        pytest.skip("gmx not on PATH")
    MDSimulator(n_steps=100)


def test_gmx_available_is_bool():
    from insulin_ai.simulation.gromacs_complex import gmx_available

    assert isinstance(gmx_available(), bool)


def test_merge_gro_roundtrip():
    from insulin_ai.simulation.gromacs_complex import _write_gro, _read_gro
    import tempfile
    import os

    atoms = [(1, "MOL", "C1", 0.0, 0.0, 0.0), (1, "MOL", "C2", 0.1, 0.0, 0.0)]
    box = (2.0, 2.0, 2.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gro", delete=False) as f:
        p = f.name
    try:
        _write_gro(p, "t", atoms, box)
        title, read = _read_gro(p)
        assert len(read) == 2
        assert read[0][3] == pytest.approx(0.0)
    finally:
        os.unlink(p)
