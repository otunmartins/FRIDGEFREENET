#!/usr/bin/env python3
"""Evaluate PSMILES via GROMACS merged EM (AMBER99SB-ILDN + GAFF)."""

import os
from typing import Any, Dict, List, Optional

from .gromacs_complex import gmx_available, run_gromacs_merged_em
from .property_extractor import PropertyExtractor


class MDSimulator:
    def __init__(
        self,
        n_steps: int = 50000,
        temperature: float = 298.0,
        random_seed: int = 42,
    ):
        if not gmx_available():
            raise RuntimeError(
                "gmx required on PATH (conda-forge: mamba install gromacs acpype ambertools)"
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
    ) -> Dict[str, Any]:
        to_eval = candidates[:max_candidates]
        if not to_eval:
            raise ValueError("empty candidates")
        md_results = []
        material_names = []
        print(f"  Evaluating {len(to_eval)} via GROMACS merged EM...")

        for i, cand in enumerate(to_eval):
            psmiles = self._get_psmiles(cand)
            if not psmiles or "[*]" not in str(psmiles):
                md_results.append(None)
                material_names.append(cand.get("material_name", f"candidate_{i}"))
                continue
            name = cand.get("material_name", psmiles)
            material_names.append(name)
            res = run_gromacs_merged_em(
                psmiles,
                n_repeats=int(os.environ.get("INSULIN_AI_GMX_N_REPEATS", "2")),
                offset_nm=float(os.environ.get("INSULIN_AI_GMX_OFFSET_NM", "2.5")),
            )
            if res is None:
                raise RuntimeError(f"GROMACS evaluate failed for {name[:40]}")
            md_results.append(res)

        feedback = self.extractor.extract_feedback(md_results, material_names)
        return {
            "high_performers": feedback["high_performers"],
            "effective_mechanisms": feedback["effective_mechanisms"],
            "problematic_features": feedback["problematic_features"],
            "property_analysis": feedback["property_analysis"],
            "successful_materials": feedback["high_performers"],
            "md_results_raw": md_results,
        }
