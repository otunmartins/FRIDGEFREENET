"""Tests for discovery_score (autoresearch-style scoring)."""

import pytest

from insulin_ai.simulation.scoring import discovery_score


def test_discovery_score_empty():
    assert discovery_score({}) == 0.0
    assert discovery_score({"high_performers": [], "effective_mechanisms": [], "problematic_features": []}) == 0.0


def test_discovery_score_positive():
    fb = {
        "high_performers": ["a", "b"],
        "effective_mechanisms": ["hydrogen_bonding"],
        "problematic_features": [],
    }
    # 2*2 + 1*0.5 - 0 = 4.5
    assert discovery_score(fb) == pytest.approx(4.5)


def test_discovery_score_penalty():
    fb = {
        "high_performers": ["x"],
        "effective_mechanisms": [],
        "problematic_features": ["bad1", "bad2"],
    }
    # 1*2 - 2*1 = 0
    assert discovery_score(fb) == pytest.approx(0.0)


def test_discovery_score_custom_weights():
    fb = {"high_performers": [1], "effective_mechanisms": [], "problematic_features": []}
    assert discovery_score(fb, high_performer_weight=10.0) == 10.0
