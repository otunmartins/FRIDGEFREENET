"""
Smoke tests for literature mining components.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_semantic_scholar_client_import():
    """SemanticScholarClient can be imported."""
    from semantic_scholar_client import SemanticScholarClient
    assert SemanticScholarClient is not None


def test_semantic_scholar_client_init():
    """SemanticScholarClient initializes without API key."""
    from semantic_scholar_client import SemanticScholarClient
    client = SemanticScholarClient(api_key=None)
    assert client.base_url == "https://api.semanticscholar.org/graph/v1"
    assert client.rate_limit_delay > 0


def test_materials_literature_miner_import():
    """MaterialsLiteratureMiner can be imported."""
    try:
        from literature_mining_system import MaterialsLiteratureMiner
        assert MaterialsLiteratureMiner is not None
    except ImportError as e:
        if "ollama" in str(e).lower():
            import pytest
            pytest.skip("Ollama not installed")
        raise
