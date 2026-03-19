#!/usr/bin/env python3
"""
Property extraction after GROMACS EM on merged insulin + polymer.

Ranks by potential_energy_complex_kj_mol (lower = better MM energy).
"""

from typing import Any, Dict, List, Optional

from insulin_ai.simulation.scoring import composite_screening_score


class PropertyExtractor:
    """Maps GROMACS (and legacy) results to feedback + composite score."""

    def __init__(
        self,
        interaction_favorable_max_kj: float = -5.0,
        interaction_unfavorable_min_kj: float = 50.0,
        min_insulin_polymer_contacts: int = 5,
        insulin_rmsd_problematic_nm: float = 0.45,
    ):
        self.interaction_favorable_max_kj = interaction_favorable_max_kj
        self.interaction_unfavorable_min_kj = interaction_unfavorable_min_kj
        self.min_insulin_polymer_contacts = min_insulin_polymer_contacts
        self.insulin_rmsd_problematic_nm = insulin_rmsd_problematic_nm

    def extract_feedback(
        self,
        md_results: List[Dict[str, Any]],
        material_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        high_performers: List[str] = []
        effective_mechanisms: List[str] = []
        problematic_features: List[str] = []
        property_analysis: Dict[str, Any] = {}

        # GROMACS batch: lower E_pot kJ/mol = better
        gmx_rows: List[tuple[float, str]] = []
        for i, res in enumerate(md_results):
            if res and res.get("gromacs_only") and res.get("potential_energy_complex_kj_mol") is not None:
                name = (
                    material_names[i]
                    if material_names and i < len(material_names)
                    else res.get("psmiles", f"candidate_{i}")
                )
                gmx_rows.append((float(res["potential_energy_complex_kj_mol"]), name))
        if gmx_rows:
            effective_mechanisms.append("GROMACS_EM_merged_screening")
            gmx_rows.sort(key=lambda t: t[0])
            median_e = gmx_rows[len(gmx_rows) // 2][0]
            for e, name in gmx_rows:
                if e <= median_e:
                    high_performers.append(name)
            for i, res in enumerate(md_results):
                if res is None:
                    problematic_features.append("gromacs_failed")
                    continue
                name = (
                    material_names[i]
                    if material_names and i < len(material_names)
                    else res.get("psmiles", f"candidate_{i}")
                )
                property_analysis[name] = {
                    "potential_energy_complex_kj_mol": res.get("potential_energy_complex_kj_mol"),
                    "method": res.get("method"),
                    "gromacs_only": True,
                }
            return {
                "high_performers": high_performers[:5],
                "effective_mechanisms": list(dict.fromkeys(effective_mechanisms))[:5],
                "problematic_features": list(dict.fromkeys(problematic_features))[:5],
                "property_analysis": property_analysis,
            }

        for i, res in enumerate(md_results):
            if res is None:
                problematic_features.append("evaluation_failed")
                continue
            name = (
                material_names[i]
                if material_names and i < len(material_names)
                else res.get("psmiles", f"candidate_{i}")
            )
            e_int = res.get("interaction_energy_kj_mol")
            contacts = res.get("insulin_polymer_contacts")
            e_complex = res.get("potential_energy_complex_kj_mol")
            rmsd = res.get("insulin_rmsd_to_initial_nm")
            composite = None
            if e_int is not None and rmsd is not None:
                try:
                    composite = composite_screening_score(float(e_int), float(rmsd))
                except (TypeError, ValueError):
                    pass
            if e_int is not None:
                if e_int <= self.interaction_favorable_max_kj:
                    high_performers.append(name)
                    effective_mechanisms.append("favorable_interaction_energy")
                if e_int >= self.interaction_unfavorable_min_kj:
                    problematic_features.append(f"high_interaction_energy_{name[:20]}")
            if rmsd is not None and rmsd == rmsd:
                if rmsd <= 0.15:
                    effective_mechanisms.append("insulin_structure_preserved")
                if rmsd >= self.insulin_rmsd_problematic_nm:
                    problematic_features.append(f"high_insulin_distortion_{name[:20]}")
            if contacts is not None:
                if contacts >= self.min_insulin_polymer_contacts:
                    effective_mechanisms.append("insulin_polymer_contacts")
                elif contacts < 2:
                    problematic_features.append(f"low_insulin_contacts_{name[:20]}")
            property_analysis[name] = {
                "interaction_energy_kj_mol": e_int,
                "insulin_rmsd_to_initial_nm": rmsd,
                "composite_screening_score": composite,
                "potential_energy_complex_kj_mol": e_complex,
                "potential_energy_insulin_kj_mol": res.get("potential_energy_insulin_kj_mol"),
                "potential_energy_polymer_kj_mol": res.get("potential_energy_polymer_kj_mol"),
                "insulin_polymer_contacts": contacts,
            }
        scored = [
            (property_analysis[n].get("composite_screening_score") or -1e9, n)
            for n in high_performers
        ]
        scored.sort(key=lambda x: -x[0])
        high_performers = [n for _, n in scored[:5]] if scored else high_performers[:5]
        return {
            "high_performers": high_performers[:5],
            "effective_mechanisms": list(dict.fromkeys(effective_mechanisms))[:5],
            "problematic_features": list(dict.fromkeys(problematic_features))[:5],
            "property_analysis": property_analysis,
        }
