"""
Tests for benchmarks/optuna_psmiles_discovery.py (Optuna + PSMILES; no MCP / LLM).

Install: pip install -e ".[benchmark]"
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))
sys.path.insert(0, ROOT)

optuna = pytest.importorskip("optuna")

from benchmarks.optuna_psmiles_discovery import (  # noqa: E402
    _update_feedback_from_md,
    run_optuna_benchmark,
)


def _mock_evaluate(candidates, max_candidates):
    """Minimal feedback dict compatible with discovery_score."""
    names = [c.get("material_name", f"c{i}") for i, c in enumerate(candidates[:max_candidates])]
    name = names[0] if names else "c0"
    return {
        "high_performers": [name],
        "effective_mechanisms": ["OpenMM_merged_screening"],
        "problematic_features": [],
        "successful_materials": [name],
        "property_analysis": {
            name: {
                "interaction_energy_kj_mol": -120.0,
                "insulin_rmsd_to_initial_nm": 0.06,
            }
        },
    }


def test_update_feedback_from_md_maps_names_to_psmiles():
    md = {
        "high_performers": ["A1"],
        "problematic_features": ["bad"],
        "effective_mechanisms": [],
        "property_analysis": {},
    }
    cands = [{"material_name": "A1", "chemical_structure": "[*]CC[*]"}]
    st = _update_feedback_from_md(md, cands)
    assert "[*]CC[*]" in st["high_performer_psmiles"]


def test_run_optuna_benchmark_mocked_completes_study():
    """Fast path: no OpenMM; inject evaluate_candidates_fn."""
    out = run_optuna_benchmark(
        "[*]OCC[*]",
        n_trials=3,
        library_size_per_trial=2,
        random_seed=0,
        mutator_seed_high=5000,
        evaluate_candidates_fn=_mock_evaluate,
    )
    assert "error" not in out
    assert out["best_value"] > float("-inf")
    assert out["best_trial"]["number"] >= 0
    assert out["n_trials"] == 3


@pytest.mark.slow
def test_run_optuna_benchmark_one_trial_openmm():
    """Optional integration: one trial with real MDSimulator."""
    from insulin_ai.simulation.openmm_compat import openmm_available

    if not openmm_available():
        pytest.skip("OpenMM stack required")

    out = run_optuna_benchmark(
        "[*]OCC[*]",
        n_trials=1,
        library_size_per_trial=1,
        random_seed=1,
        mutator_seed_high=100,
        md_steps=100,
        verbose_eval=False,
    )
    assert "error" not in out
    assert out["n_trials"] == 1
    assert out["best_value"] > float("-inf")
