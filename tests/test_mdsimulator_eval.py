"""MDSimulator.evaluate_candidates: Packmol matrix path and error handling."""

import pytest


def test_evaluate_candidates_raises_without_packmol(monkeypatch):
    """Matrix evaluation requires packmol; no silent fallback."""
    from insulin_ai.simulation import MDSimulator
    from insulin_ai.simulation.openmm_compat import openmm_available

    if not openmm_available():
        pytest.skip("OpenMM stack required")

    monkeypatch.setattr(
        "insulin_ai.simulation.packmol_packer._packmol_available",
        lambda: False,
    )
    sim = MDSimulator(n_steps=100)
    with pytest.raises(RuntimeError, match="Packmol"):
        sim.evaluate_candidates(
            [{"material_name": "t", "chemical_structure": "[*]CC[*]"}],
            max_candidates=1,
            verbose=False,
        )


def test_evaluate_candidates_matrix_smoke(tmp_path):
    """Full matrix path when packmol + OpenMM available (slow)."""
    from insulin_ai.simulation import MDSimulator
    from insulin_ai.simulation.openmm_compat import openmm_available
    from insulin_ai.simulation.packmol_packer import _packmol_available

    if not openmm_available():
        pytest.skip("OpenMM stack required")
    if not _packmol_available():
        pytest.skip("packmol binary required")

    import os

    os.environ["INSULIN_AI_OPENMM_MATRIX_NPT"] = "0"
    os.environ["INSULIN_AI_OPENMM_MATRIX_FIXED_MODE"] = "1"
    os.environ["INSULIN_AI_OPENMM_MATRIX_N_POLYMERS"] = "2"
    os.environ["INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS"] = "300"
    os.environ["INSULIN_AI_OPENMM_N_REPEATS"] = "2"
    try:
        sim = MDSimulator(n_steps=100, random_seed=42)
        ad = str(tmp_path / "structures")
        r = sim.evaluate_candidates(
            [{"material_name": "smoke", "chemical_structure": "[*]CC[*]"}],
            max_candidates=1,
            verbose=False,
            artifacts_dir=ad,
        )
    finally:
        for k in (
            "INSULIN_AI_OPENMM_MATRIX_NPT",
            "INSULIN_AI_OPENMM_MATRIX_FIXED_MODE",
            "INSULIN_AI_OPENMM_MATRIX_N_POLYMERS",
            "INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS",
            "INSULIN_AI_OPENMM_N_REPEATS",
        ):
            os.environ.pop(k, None)

    assert r.get("md_results_raw")
    raw = r["md_results_raw"][0]
    assert raw is not None
    assert raw.get("method", "").startswith("OpenMM_matrix")
    assert raw.get("n_polymer_chains") == 2
    assert raw.get("n_polymer_atoms") == raw.get("n_polymer_atoms_per_chain", 0) * 2
    pdb = (tmp_path / "structures" / "smoke_complex_minimized.pdb")
    assert pdb.is_file()
    pm = raw.get("packing_metrics") or {}
    assert pm.get("ok") is True
    assert "min_polymer_protein_distance_nm" in pm
