#!/usr/bin/env python3
"""
Property Extraction from MD Results

Maps raw MD simulation output to feedback quantities for the active learning loop:
- high_performers: materials with favorable stability metrics
- effective_mechanisms: inferred stabilization mechanisms
- problematic_features: features associated with poor performance
"""

from typing import List, Dict, Any, Optional


class PropertyExtractor:
    """
    Extracts feedback-ready properties from MD simulation results.
    
    For thermal stability screening:
    - Energy drift (large increase = instability)
    - RMSD (conformational change) if positions saved
    """
    
    def __init__(
        self,
        energy_increase_threshold_kj: float = 50.0,
        min_energy_threshold_kj: float = -500.0,
    ):
        """
        Args:
            energy_increase_threshold_kj: Max acceptable potential energy increase
            min_energy_threshold_kj: Minimum final energy to be considered stable
        """
        self.energy_increase_threshold = energy_increase_threshold_kj
        self.min_energy_threshold = min_energy_threshold_kj
    
    def extract_feedback(
        self,
        md_results: List[Dict[str, Any]],
        material_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert list of MD results to feedback dict for iterative mining.
        
        Args:
            md_results: List of dicts from OpenMMRunner.run()
            material_names: Optional names for each candidate
            
        Returns:
            Dict with high_performers, effective_mechanisms, problematic_features
        """
        high_performers = []
        effective_mechanisms = []
        problematic_features = []
        property_analysis = {}
        
        for i, res in enumerate(md_results):
            if res is None:
                problematic_features.append("md_conversion_failed")
                continue
            
            name = (material_names[i] if material_names and i < len(material_names) 
                    else res.get("psmiles", f"candidate_{i}"))
            
            initial = res.get("initial_energy_kj_mol", 0)
            final = res.get("final_energy_kj_mol", 0)
            delta = final - initial
            
            # Screening heuristics
            if delta > self.energy_increase_threshold:
                problematic_features.append(f"high_energy_drift_{name[:20]}")
            elif final < self.min_energy_threshold and delta < 20:
                high_performers.append(name)
                effective_mechanisms.append("low_energy_conformation")
            
            property_analysis[name] = {
                "initial_energy_kj_mol": initial,
                "final_energy_kj_mol": final,
                "energy_change_kj_mol": delta,
            }
        
        return {
            "high_performers": high_performers[:5],
            "effective_mechanisms": list(set(effective_mechanisms))[:3],
            "problematic_features": list(set(problematic_features))[:5],
            "property_analysis": property_analysis,
        }
