#!/usr/bin/env python3
"""
Property Extraction from MD Results

Maps raw MD simulation output to feedback quantities for the active learning loop.
Per proposal.tex Section 6: thermal stability, insulin protection,
insulin-material interactions.

- high_performers: materials with favorable stability metrics
- effective_mechanisms: inferred stabilization mechanisms
- problematic_features: features associated with poor performance
"""

from typing import List, Dict, Any, Optional


class PropertyExtractor:
    """
    Extracts feedback-ready properties from MD simulation results.
    
    Proposal metrics:
    - Thermal stability: energy drift (large increase = instability)
    - Insulin protection: insulin backbone RMSD from initial structure
    - Insulin-material interactions: insulin-polymer contact count
    """
    
    def __init__(
        self,
        energy_increase_threshold_kj: float = 50.0,
        min_energy_threshold_kj: float = -500.0,
        insulin_rmsd_threshold_nm: float = 0.5,
        min_insulin_polymer_contacts: int = 5,
    ):
        """
        Args:
            energy_increase_threshold_kj: Max acceptable potential energy increase
            min_energy_threshold_kj: Minimum final energy to be considered stable
            insulin_rmsd_threshold_nm: Max acceptable insulin RMSD (nm); higher = problematic
            min_insulin_polymer_contacts: Min contacts for favorable interaction
        """
        self.energy_increase_threshold = energy_increase_threshold_kj
        self.min_energy_threshold = min_energy_threshold_kj
        self.insulin_rmsd_threshold = insulin_rmsd_threshold_nm
        self.min_insulin_polymer_contacts = min_insulin_polymer_contacts
    
    def extract_feedback(
        self,
        md_results: List[Dict[str, Any]],
        material_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert list of MD results to feedback dict for iterative mining.
        
        Args:
            md_results: List of dicts from OpenMMRunner.run() (insulin+polymer)
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
            delta = res.get("energy_drift_kj_mol", final - initial)
            insulin_rmsd = res.get("insulin_rmsd_nm")
            insulin_contacts = res.get("insulin_polymer_contacts")
            
            # Thermal stability: energy drift
            if delta > self.energy_increase_threshold:
                problematic_features.append(f"high_energy_drift_{name[:20]}")
            elif final < self.min_energy_threshold and delta < 20:
                high_performers.append(name)
                effective_mechanisms.append("low_energy_conformation")
            
            # Insulin protection: RMSD (high RMSD = poor protection)
            if insulin_rmsd is not None and insulin_rmsd > self.insulin_rmsd_threshold:
                problematic_features.append(f"high_insulin_rmsd_{name[:20]}")
            elif insulin_rmsd is not None and insulin_rmsd < 0.2:
                effective_mechanisms.append("insulin_structure_retained")
            
            # Insulin-material interactions: contact count
            if insulin_contacts is not None:
                if insulin_contacts >= self.min_insulin_polymer_contacts:
                    effective_mechanisms.append("insulin_polymer_interaction")
                elif insulin_contacts < 2:
                    problematic_features.append(f"low_insulin_contacts_{name[:20]}")
            
            property_analysis[name] = {
                "initial_energy_kj_mol": initial,
                "final_energy_kj_mol": final,
                "energy_change_kj_mol": delta,
                "insulin_rmsd_nm": insulin_rmsd,
                "insulin_polymer_contacts": insulin_contacts,
            }
        
        return {
            "high_performers": high_performers[:5],
            "effective_mechanisms": list(set(effective_mechanisms))[:3],
            "problematic_features": list(set(problematic_features))[:5],
            "property_analysis": property_analysis,
        }
