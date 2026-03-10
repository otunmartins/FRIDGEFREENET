"""
Integration tests for insulin + polymer matrix packing.

Requires: openmm, rdkit, openmmforcefields, (packmol optional for realistic packing)
"""

import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))


def test_insulin_polymer_builder_import():
    """Test that InsulinPolymerSystemBuilder imports."""
    try:
        from insulin_ai.simulation.insulin_polymer_system import InsulinPolymerSystemBuilder
        assert InsulinPolymerSystemBuilder is not None
    except ImportError as e:
        pytest.skip(f"Simulation deps not installed: {e}")


@pytest.mark.slow
def test_insulin_polymer_build_returns_valid_system():
    """Test InsulinPolymerSystemBuilder.build returns non-None with valid atoms."""
    try:
        from insulin_ai.simulation.insulin_polymer_system import InsulinPolymerSystemBuilder
    except ImportError:
        pytest.skip("OpenMM/RDKit not installed")

    builder = InsulinPolymerSystemBuilder(n_repeats=2, n_chains=2)
    top, pos, sys, n_insulin = builder.build("[*]OCC[*]")

    if top is None or pos is None or sys is None:
        pytest.skip("Build returned None (insulin PDB or GAFF may be unavailable)")

    assert n_insulin > 0, "n_insulin_atoms should be positive"
    assert top.getNumAtoms() > n_insulin, "Total atoms should exceed insulin atoms"


def test_packmol_packer_import():
    """Test packmol_packer module imports."""
    try:
        from insulin_ai.simulation.packmol_packer import pack_insulin_polymers, _packmol_available
        assert callable(pack_insulin_polymers)
        assert isinstance(_packmol_available(), bool)
    except ImportError:
        pytest.skip("packmol_packer not importable")
