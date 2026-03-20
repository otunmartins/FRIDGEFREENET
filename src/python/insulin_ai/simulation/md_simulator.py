#!/usr/bin/env python3
"""Evaluate PSMILES via OpenMM merged minimize (AMBER14SB + GAFF + Gasteiger)."""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .openmm_compat import openmm_available
from .property_extractor import PropertyExtractor


def _env_int(primary: str, fallback: str, default: str) -> int:
    v = os.environ.get(primary) or os.environ.get(fallback) or default
    return int(v)


def _env_float(primary: str, fallback: str, default: str) -> float:
    v = os.environ.get(primary) or os.environ.get(fallback) or default
    return float(v)


def _eval_quiet() -> bool:
    """
    Suppress per-candidate progress (JSON + stderr) when user opts out.

    INSULIN_AI_EVAL_QUIET=1, or INSULIN_AI_EVAL_VERBOSE=0/false/no.
    """
    if os.environ.get("INSULIN_AI_EVAL_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    v = os.environ.get("INSULIN_AI_EVAL_VERBOSE", "").strip().lower()
    if v in ("0", "false", "no"):
        return True
    return False


def _progress_log(msg: str) -> None:
    """Visible in MCP server stderr (terminal running the server), not in tool return."""
    print(msg, file=sys.stderr, flush=True)


class MDSimulator:
    def __init__(
        self,
        n_steps: int = 50000,
        temperature: float = 298.0,
        random_seed: int = 42,
    ):
        if not openmm_available():
            raise RuntimeError(
                "OpenMM screening stack not importable. Install with: "
                "pip install -e '.[openmm]' (or conda: openmm, pip: openmmforcefields, openff-toolkit, pdbfixer, rdkit)."
            )
        self.extractor = PropertyExtractor()
        self.n_steps = n_steps
        self.random_seed = random_seed

    def _get_psmiles(self, candidate: Dict[str, Any]) -> Optional[str]:
        if isinstance(candidate, str):
            return candidate
        p = candidate.get("psmiles") or candidate.get("chemical_structure")
        if p:
            return p
        m = candidate.get("material_name", "")
        return m if m and "[*]" in str(m) else None

    def evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        from .openmm_complex import run_openmm_relax_and_energy

        if _eval_quiet():
            verbose = False
        to_eval = candidates[:max_candidates]
        if not to_eval:
            raise ValueError("empty candidates")
        md_results = []
        material_names = []
        progress: List[Dict[str, Any]] = []
        n_repeats = _env_int("INSULIN_AI_OPENMM_N_REPEATS", "INSULIN_AI_GMX_N_REPEATS", "2")
        offset_x = _env_float("INSULIN_AI_OPENMM_OFFSET_NM", "INSULIN_AI_GMX_OFFSET_NM", "2.5")
        ligand_offset_nm: Tuple[float, float, float] = (offset_x, 0.0, 0.0)
        max_minimize = int(os.environ.get("INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS", "5000"))
        n_total = len(to_eval)
        msg = (
            f"[insulin-ai] OpenMM screening: {n_total} candidate(s) "
            f"(energy minimization + interaction energy, not a long MD trajectory)"
        )
        print(f"  Evaluating {n_total} via OpenMM merged minimize...")
        if verbose:
            _progress_log(msg)

        for i, cand in enumerate(to_eval):
            psmiles = self._get_psmiles(cand)
            if not psmiles or "[*]" not in str(psmiles):
                md_results.append(None)
                material_names.append(cand.get("material_name", f"candidate_{i}"))
                if verbose:
                    entry = {
                        "index": i,
                        "total": n_total,
                        "status": "skipped",
                        "reason": "no valid PSMILES with [*]",
                        "material_name": cand.get("material_name", f"candidate_{i}"),
                    }
                    progress.append(entry)
                    _progress_log(f"[insulin-ai] {i + 1}/{n_total} skipped (no valid PSMILES)")
                continue
            name = cand.get("material_name", psmiles)
            material_names.append(name)
            preview = str(psmiles)[:60] + ("…" if len(str(psmiles)) > 60 else "")
            t0 = time.perf_counter()
            if verbose:
                _progress_log(
                    f"[insulin-ai] {i + 1}/{n_total} minimize+energy: {preview} "
                    f"(max {max_minimize} minimizer steps)"
                )
            res = run_openmm_relax_and_energy(
                psmiles,
                n_repeats=n_repeats,
                random_seed=self.random_seed,
                ligand_offset_nm=ligand_offset_nm,
                max_minimize_steps=max_minimize,
            )
            elapsed = time.perf_counter() - t0
            if res is None:
                raise RuntimeError(f"OpenMM evaluate failed for {name[:40]}")
            md_results.append(res)
            if verbose:
                entry = {
                    "index": i,
                    "total": n_total,
                    "status": "completed",
                    "material_name": name,
                    "psmiles_preview": preview,
                    "seconds": round(elapsed, 3),
                    "method": res.get("method"),
                    "interaction_energy_kj_mol": res.get("interaction_energy_kj_mol"),
                    "potential_energy_complex_kj_mol": res.get("potential_energy_complex_kj_mol"),
                    "n_insulin_atoms": res.get("n_insulin_atoms"),
                    "n_polymer_atoms": res.get("n_polymer_atoms"),
                }
                progress.append(entry)
                _progress_log(
                    f"[insulin-ai] {i + 1}/{n_total} done in {elapsed:.1f}s "
                    f"E_int={res.get('interaction_energy_kj_mol')} kJ/mol"
                )

        feedback = self.extractor.extract_feedback(md_results, material_names)
        out: Dict[str, Any] = {
            "high_performers": feedback["high_performers"],
            "effective_mechanisms": feedback["effective_mechanisms"],
            "problematic_features": feedback["problematic_features"],
            "property_analysis": feedback["property_analysis"],
            "successful_materials": feedback["high_performers"],
            "md_results_raw": md_results,
        }
        if verbose:
            out["evaluation_progress"] = progress
            out["evaluation_note"] = (
                "Each candidate: LocalEnergyMinimizer on insulin+oligomer in vacuum, "
                "then interaction energy (kJ/mol). Not molecular dynamics (no ns-scale trajectory)."
            )
        return out
