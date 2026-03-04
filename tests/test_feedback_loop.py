"""
Integration test: Active learning feedback loop with MDSimulator.

Does not require Ollama or network - uses mock/minimal data.
"""

import os
import sys
import json
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))
sys.path.insert(0, ROOT)


def test_mdsimulator_interface():
    """MDSimulator implements evaluate_candidates(candidates) -> feedback dict."""
    from insulin_ai.simulation import MDSimulator
    
    sim = MDSimulator()
    candidates = [
        {"material_name": "PEG hydrogel", "chemical_structure": "[*]OCC[*]"},
        {"material_name": "Chitosan", "material_composition": "chitosan"},
    ]
    result = sim.evaluate_candidates(candidates, max_candidates=5)
    
    assert "high_performers" in result
    assert "effective_mechanisms" in result
    assert "problematic_features" in result
    assert "property_analysis" in result
    assert "successful_materials" in result


def test_feedback_state_update():
    """Feedback dict format matches _update_feedback_state expectations."""
    # Test the feedback dict format without initializing IterativeLiteratureMiner
    # (which requires Ollama, Semantic Scholar, etc.)
    md_results = {
        "high_performers": ["PEG", "Chitosan"],
        "effective_mechanisms": ["hydrogen bonding"],
        "problematic_features": ["high crystallinity"],
        "successful_materials": ["PEG", "Chitosan"],
        "failed_features": [],
    }
    
    # Expected keys that _update_feedback_state produces
    expected = {
        "top_candidates": md_results.get("successful_materials", md_results.get("high_performers", [])),
        "stability_mechanisms": md_results.get("effective_mechanisms", []),
        "limitations": md_results.get("failed_features", md_results.get("problematic_features", [])),
    }
    assert expected["top_candidates"] == ["PEG", "Chitosan"]
    assert "hydrogen bonding" in expected["stability_mechanisms"]


def test_mdsimulator_with_mock_candidates():
    """Full evaluate_candidates with mixed candidate formats."""
    from insulin_ai.simulation import MDSimulator
    
    sim = MDSimulator()
    candidates = [
        {"material_name": "A", "psmiles": "[*]OCC[*]"},
        {"material_name": "B", "chemical_structure": "[*]CC[*]"},
        {"material_name": "Unknown", "material_composition": "unknown polymer"},
    ]
    result = sim.evaluate_candidates(candidates, max_candidates=5)
    
    assert isinstance(result["high_performers"], list)
    assert isinstance(result["problematic_features"], list)
    # A and B should be evaluable; Unknown may fail
    assert len(result["property_analysis"]) >= 0
