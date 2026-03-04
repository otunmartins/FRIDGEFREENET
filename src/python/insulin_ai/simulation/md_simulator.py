#!/usr/bin/env python3
"""
MD Simulator - Evaluation Interface for Active Learning

Implements the evaluate_candidates() interface expected by
IterativeLiteratureMiner.run_active_learning_cycle(md_simulator=...).
Returns feedback dict compatible with _process_md_feedback().
"""

from typing import List, Dict, Any, Optional

from .property_extractor import PropertyExtractor
from .rdkit_proxy import evaluate_candidates_proxy

try:
    from .openmm_runner import OpenMMRunner
    OPENMM_AVAILABLE = True
except ImportError:
    OpenMMRunner = None
    OPENMM_AVAILABLE = False


class MDSimulator:
    """
    CPU-only MD evaluator for polymer candidates.
    
    Accepts material candidates (with PSMILES or chemical_structure),
    runs short MD screens, and returns feedback for the active learning loop.
    """
    
    def __init__(
        self,
        n_steps: int = 50000,
        temperature: float = 298.0,
        random_seed: int = 42,
    ):
        """
        Args:
            n_steps: MD steps per candidate (50000 ≈ 100 ps at 2 fs)
            temperature: Kelvin
            random_seed: For reproducibility
        """
        self.runner = None
        if OPENMM_AVAILABLE and OpenMMRunner:
            self.runner = OpenMMRunner(
                temperature=temperature,
                pressure=1.01325,
                timestep_fs=2.0,
                platform_name="CPU",
            )
        self.extractor = PropertyExtractor()
        self.n_steps = n_steps
        self.random_seed = random_seed
    
    def _get_psmiles(self, candidate: Dict[str, Any]) -> Optional[str]:
        """Extract PSMILES from candidate dict."""
        if isinstance(candidate, str):
            return candidate
        psmiles = candidate.get("psmiles") or candidate.get("chemical_structure")
        if psmiles:
            return psmiles
        mat_name = candidate.get("material_name", "")
        if mat_name:
            return mat_name
        return None
    
    def _psmiles_from_material_name(self, name: str) -> Optional[str]:
        """Map common material names to PSMILES for evaluation."""
        mapping = {
            "peg": "[*]OCC[*]",
            "polyethylene glycol": "[*]OCC[*]",
            "peg-based": "[*]OCC[*]",
            "polyethylene": "[*]CC[*]",
            "chitosan": "[*]CC([*])OC1OC(C)C(O)C(O)C1O",
            "plga": "[*]CC(=O)O[*]",
            "pva": "[*]CC([*])O",
            "pmma": "[*]CC([*])(C)C(=O)OC",
        }
        name_lower = name.lower()
        for key, psmiles in mapping.items():
            if key in name_lower:
                return psmiles
        return None
    
    def evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate material candidates via CPU-only MD screening.
        
        Args:
            candidates: List of material dicts (from literature mining)
            max_candidates: Max number to simulate (for runtime control)
            
        Returns:
            Feedback dict: high_performers, effective_mechanisms,
            problematic_features, property_analysis
        """
        md_results = []
        material_names = []
        
        for i, cand in enumerate(candidates[:max_candidates]):
            psmiles = self._get_psmiles(cand)
            if not psmiles and isinstance(cand, dict):
                psmiles = self._psmiles_from_material_name(
                    cand.get("material_name", "") or cand.get("material_composition", "")
                )
            if not psmiles:
                md_results.append(None)
                material_names.append(cand.get("material_name", f"candidate_{i}"))
                continue
            
            name = cand.get("material_name", psmiles) if isinstance(cand, dict) else psmiles
            material_names.append(name)
            
            res = None
            if self.runner:
                try:
                    res = self.runner.run(
                        psmiles,
                        n_steps=self.n_steps,
                        random_seed=self.random_seed + i,
                    )
                except Exception:
                    pass
            md_results.append(res)
        
        # Use MD feedback if we got any valid results; else fall back to RDKit proxy
        n_valid = sum(1 for r in md_results if r is not None)
        if n_valid > 0:
            feedback = self.extractor.extract_feedback(md_results, material_names)
        else:
            feedback = evaluate_candidates_proxy(candidates, max_candidates=max_candidates)
        
        return {
            "high_performers": feedback["high_performers"],
            "effective_mechanisms": feedback["effective_mechanisms"],
            "problematic_features": feedback["problematic_features"],
            "property_analysis": feedback["property_analysis"],
            "successful_materials": feedback["high_performers"],
            "md_results_raw": md_results,
        }
