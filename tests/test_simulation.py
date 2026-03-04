"""
Unit tests for CPU-only MD simulation pipeline.

Requires: openmm, rdkit, openmmforcefields (optional)
"""

import os
import sys
import pytest

# Add src/python to path for insulin_ai package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))


def test_psmiles_to_openmm_import():
    """Test that simulation module imports correctly."""
    try:
        from insulin_ai.simulation import PSMILestoOpenMM, MDSimulator
        assert PSMILestoOpenMM is not None
        assert MDSimulator is not None
    except ImportError:
        pytest.skip("Simulation dependencies not installed (openmm, rdkit)")


def test_psmiles_cap():
    """Test PSMILES capping replaces [*] with [H]."""
    try:
        from insulin_ai.simulation.psmiles_to_openmm import PSMILestoOpenMM
        conv = PSMILestoOpenMM()
        capped = conv._cap_psmiles("[*]OCC[*]")
        assert "[*]" not in capped
        assert "[H]" in capped
    except ImportError:
        pytest.skip("RDKit not installed")


def test_psmiles_to_mol():
    """Test PSMILES to RDKit Mol conversion."""
    try:
        from insulin_ai.simulation.psmiles_to_openmm import PSMILestoOpenMM
        conv = PSMILestoOpenMM()
        mol = conv.psmiles_to_mol("[*]OCC[*]")
        assert mol is not None
    except ImportError:
        pytest.skip("RDKit not installed")


@pytest.mark.slow
def test_build_openmm_system():
    """Test full PSMILES to OpenMM system conversion."""
    try:
        from insulin_ai.simulation.psmiles_to_openmm import PSMILestoOpenMM
        conv = PSMILestoOpenMM()
        top, pos, sys = conv.build_openmm_system("[*]OCC[*]")
        assert top is not None
        assert pos is not None
        assert sys is not None
    except ImportError:
        pytest.skip("OpenMM/RDKit not installed")


def test_property_extractor():
    """Test PropertyExtractor feedback conversion."""
    from insulin_ai.simulation.property_extractor import PropertyExtractor
    ext = PropertyExtractor()
    md_results = [
        {"initial_energy_kj_mol": -100, "final_energy_kj_mol": -120, "psmiles": "[*]OCC[*]"},
        {"initial_energy_kj_mol": -50, "final_energy_kj_mol": 100, "psmiles": "[*]CC[*]"},
    ]
    feedback = ext.extract_feedback(md_results)
    assert "high_performers" in feedback
    assert "problematic_features" in feedback
    assert "effective_mechanisms" in feedback
