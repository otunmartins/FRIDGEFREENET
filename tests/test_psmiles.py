"""
Smoke tests for PSMILES generator.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_psmiles_generator_import():
    """PSMILESGenerator can be imported."""
    try:
        from psmiles_generator import PSMILESGenerator
        assert PSMILESGenerator is not None
    except ImportError:
        import pytest
        pytest.skip("PSMILES dependencies not installed")


def test_psmiles_direct_mapping():
    """Direct PSMILES mapping returns correct strings."""
    try:
        from psmiles_generator import PSMILESGenerator
        gen = PSMILESGenerator(ollama_model="llama3.2")
        result = gen._direct_psmiles_mapping("peg")
        assert result is not None
        assert result["psmiles"] == "[*]OCC[*]"
        
        result2 = gen._direct_psmiles_mapping("polyethylene")
        assert result2["psmiles"] == "[*]CC[*]"
    except ImportError:
        import pytest
        pytest.skip("PSMILES dependencies not installed")


def test_psmiles_basic_syntax_check():
    """Basic syntax check validates PSMILES."""
    try:
        from psmiles_generator import PSMILESGenerator
        gen = PSMILESGenerator()
        valid = gen._basic_syntax_check("[*]OCC[*]")
        assert valid["valid"] is True
        assert valid["connection_count"] == 2
        
        invalid = gen._basic_syntax_check("C C C")  # spaces
        assert invalid["valid"] is False
        assert any("space" in str(e).lower() for e in invalid["errors"])
    except ImportError:
        import pytest
        pytest.skip("PSMILES dependencies not installed")
