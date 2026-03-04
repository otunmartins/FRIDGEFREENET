#!/usr/bin/env python3
"""
RDKit Proxy Evaluator (Fallback when OpenMM/GAFF unavailable)

Uses RDKit descriptors as a stability proxy when full MD parameterization fails
(e.g., openff-toolkit unavailable on platform). Keeps the feedback loop closed.
"""

from typing import List, Dict, Any, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def _cap_psmiles(psmiles: str) -> str:
    """Replace [*] with [H] for RDKit."""
    return psmiles.replace("[*]", "[H]") if "[*]" in psmiles else psmiles


def _psmiles_from_candidate(cand) -> Optional[str]:
    """Extract PSMILES from candidate dict."""
    if isinstance(cand, str):
        return cand
    return (cand.get("psmiles") or cand.get("chemical_structure") or 
            cand.get("material_composition"))


def _material_name_from_candidate(cand) -> str:
    if isinstance(cand, str):
        return cand
    return cand.get("material_name", "unknown")


# Map common material names to PSMILES for proxy evaluation
MATERIAL_PSMILES = {
    "peg": "[*]OCC[*]",
    "polyethylene glycol": "[*]OCC[*]",
    "peg-based": "[*]OCC[*]",
    "polyethylene": "[*]CC[*]",
    "chitosan": "[*]CC([*])OC1OC(C)C(O)C(O)C1O",
    "plga": "[*]CC(=O)O[*]",
    "pva": "[*]CC([*])O",
}


def evaluate_candidates_proxy(
    candidates: List[Dict[str, Any]],
    max_candidates: int = 10,
) -> Dict[str, Any]:
    """
    RDKit-based proxy evaluation when MD is unavailable.
    
    Uses MW, LogP, rotatable bonds as heuristic "stability" indicators.
    Lower |LogP| and moderate MW often correlate with biocompatibility.
    """
    if not RDKIT_AVAILABLE:
        return {
            "high_performers": [],
            "effective_mechanisms": ["rdkit_proxy_unavailable"],
            "problematic_features": ["rdkit_not_installed"],
            "property_analysis": {},
        }
    
    high_performers = []
    problematic_features = []
    property_analysis = {}
    
    for i, cand in enumerate(candidates[:max_candidates]):
        psmiles = _psmiles_from_candidate(cand)
        if not psmiles or (psmiles and not any(c in psmiles for c in "CNOS[]()=")):
            # Not valid SMILES - try material name mapping
            name = str(cand.get("material_name", "") or cand.get("material_composition", "") if isinstance(cand, dict) else "").lower()
            for key, val in MATERIAL_PSMILES.items():
                if key in name:
                    psmiles = val
                    break
        if not psmiles:
            problematic_features.append(f"no_psmiles_{i}")
            continue
        
        name = _material_name_from_candidate(cand)
        capped = _cap_psmiles(psmiles)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            problematic_features.append(f"invalid_smiles_{name[:20]}")
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        property_analysis[name] = {
            "mol_weight": mw,
            "logP": logp,
            "rotatable_bonds": rotatable,
            "proxy_score": -abs(logp) * 0.5 + min(mw / 500, 2),  # Heuristic
        }
        
        # Heuristic: moderate MW (100-2000), |LogP| < 4
        if 100 < mw < 2000 and abs(logp) < 4:
            high_performers.append(name)
        elif mw > 5000 or abs(logp) > 6:
            problematic_features.append(f"extreme_props_{name[:15]}")
    
    return {
        "high_performers": high_performers[:5],
        "effective_mechanisms": ["rdkit_descriptor_proxy"] if high_performers else [],
        "problematic_features": list(set(problematic_features))[:5],
        "property_analysis": property_analysis,
        "successful_materials": high_performers,
    }
