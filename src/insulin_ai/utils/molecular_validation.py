#!/usr/bin/env python3
"""
Molecular Validation Utilities for Insulin-AI

This module provides comprehensive validation functions for molecular structures,
with special focus on detecting and preventing radical species that cause
OpenFF Toolkit failures in molecular dynamics simulations.

Author: AI-Driven Material Discovery Team
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for RDKit availability
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdmolops
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - some validation functions will be limited")

@dataclass
class ValidationResult:
    """Result of molecular validation"""
    is_valid: bool
    has_radicals: bool
    has_problematic_elements: bool
    error_message: str
    warnings: List[str]
    molecule_info: Dict[str, Any]
    corrected_smiles: Optional[str] = None

class MolecularValidator:
    """
    Comprehensive molecular validation system for insulin delivery materials.
    
    Focuses on detecting and preventing radical species and other problematic
    chemical structures that cause simulation failures.
    
    **UPDATED PHILOSOPHY: Allow all elements when in stable configurations**
    """
    
    # Elements that commonly form radicals or cause OpenFF issues - FOR REFERENCE ONLY
    # These are NOT automatically banned but require stability checks
    POTENTIALLY_PROBLEMATIC_ELEMENTS = {
        'B',   # Boron - radical-prone when not properly coordinated
        'Al',  # Aluminum - can form radicals in incomplete coordination
        'Si',  # Silicon - in complex environments can form radicals  
        'Ge',  # Germanium - similar issues to silicon
        'Sn',  # Tin - heavy metal with electron issues
        'Pb',  # Lead - heavy metal
        'Ti',  # Titanium - transition metal
        'V',   # Vanadium - transition metal
        'Cr',  # Chromium - transition metal
        'Mn',  # Manganese - transition metal
        'Fe',  # Iron - transition metal
        'Co',  # Cobalt - transition metal
        'Ni',  # Nickel - transition metal
        'Cu',  # Copper - transition metal
        'Zn',  # Zinc - divalent metal
    }
    
    # Safe elements for polymer chemistry
    SAFE_ELEMENTS = {
        'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'
    }
    
    # Stable coordination patterns for potentially problematic elements
    STABLE_COORDINATION_PATTERNS = {
        'B': {
            'min_bonds': 3,  # Boron stable with 3+ bonds (BCl3, BF3, etc.)
            'max_bonds': 4,  # Can coordinate to 4 (BF4-)
            'stable_patterns': ['B(F)(F)F', 'B(Cl)(Cl)Cl', 'B(O)', 'B(N)'],
            'description': 'Boron stable with 3-4 coordination'
        },
        'Si': {
            'min_bonds': 4,  # Silicon stable with 4 bonds (tetrahedral)
            'max_bonds': 6,  # Can expand to 6 in some cases
            'stable_patterns': ['Si(O)(O)(O)O', 'Si(C)(C)(C)C', 'Si(O)(O)'],
            'description': 'Silicon stable with 4+ bonds, tetrahedral geometry'
        },
        'Al': {
            'min_bonds': 3,  # Aluminum stable with 3 bonds (AlCl3)
            'max_bonds': 6,  # Can coordinate to 6 (AlF6-)
            'stable_patterns': ['Al(Cl)(Cl)Cl', 'Al(F)(F)F', 'Al(O)'],
            'description': 'Aluminum stable with 3+ coordination'
        }
    }

    def __init__(self):
        """Initialize the molecular validator."""
        self.rdkit_available = RDKIT_AVAILABLE
        if not self.rdkit_available:
            logger.warning("⚠️ RDKit not available - validation will be limited to string-based checks")
    
    def validate_smiles(self, smiles: str, molecule_type: str = "polymer") -> ValidationResult:
        """
        Validate a SMILES string with sophisticated stability assessment.
        
        NEW APPROACH: Check for actual radicals and stability rather than blanket element bans.
        """
        if not smiles or not smiles.strip():
            return ValidationResult(
                is_valid=False,
                has_radicals=False,
                has_problematic_elements=False,
                error_message="Empty SMILES string",
                warnings=[],
                molecule_info={}
            )
        
        smiles = smiles.strip()
        
        # **PRIMARY VALIDATION: Use RDKit if available**
        if self.rdkit_available:
            rdkit_result = self._validate_with_rdkit_stability_focused(smiles)
            if rdkit_result is not None:
                return ValidationResult(
                    is_valid=rdkit_result["valid"],
                    has_radicals=rdkit_result.get("has_radicals", False),
                    has_problematic_elements=rdkit_result.get("has_potentially_problematic_elements", False),
                    error_message=rdkit_result.get("error", ""),
                    warnings=rdkit_result.get("warnings", []),
                    molecule_info=rdkit_result.get("molecule_info", {})
                )
        
        # **FALLBACK: String-based validation with stability assessment**
        string_result = self._validate_smiles_string_stability_focused(smiles)
        
        return ValidationResult(
            is_valid=string_result["valid"],
            has_radicals=string_result.get("has_radicals", False),
            has_problematic_elements=string_result.get("has_potentially_problematic_elements", False),
            error_message=string_result.get("error", ""),
            warnings=string_result.get("warnings", []),
            molecule_info=string_result.get("molecule_info", {})
        )
    
    def _validate_smiles_string_stability_focused(self, smiles: str) -> Dict[str, Any]:
        """Enhanced string-based SMILES validation with stability assessment."""
        
        # Check for potentially problematic elements and assess their stability
        potentially_problematic_found = []
        stability_warnings = []
        
        for element in self.POTENTIALLY_PROBLEMATIC_ELEMENTS:
            if element in smiles:
                # More sophisticated check to avoid false positives
                if element == 'B':
                    if 'Br' not in smiles and ('B(' in smiles or ')B' in smiles or '[B' in smiles or f'{element}' in smiles):
                        # Check if boron appears to be in stable coordination
                        stability_assessment = self._assess_element_stability_string(smiles, element)
                        if not stability_assessment['appears_stable']:
                            potentially_problematic_found.append(element)
                            stability_warnings.append(f"Boron may not be in stable coordination: {stability_assessment['reason']}")
                        else:
                            stability_warnings.append(f"Boron appears stable: {stability_assessment['reason']}")
                            
                elif element in ['Si', 'Al'] and element in smiles:
                    stability_assessment = self._assess_element_stability_string(smiles, element)
                    if not stability_assessment['appears_stable']:
                        potentially_problematic_found.append(element)
                        stability_warnings.append(f"{element} may not be in stable coordination: {stability_assessment['reason']}")
                    else:
                        stability_warnings.append(f"{element} appears stable: {stability_assessment['reason']}")
                        
                elif len(element) == 1 and element in smiles:
                    # Single character elements need more careful checking
                    import re
                    pattern = rf'[^A-Za-z]{element}[^a-z]|^{element}[^a-z]|[^A-Za-z]{element}$|^{element}$'
                    if re.search(pattern, smiles):
                        stability_assessment = self._assess_element_stability_string(smiles, element)
                        if not stability_assessment['appears_stable']:
                            potentially_problematic_found.append(element)
                            stability_warnings.append(f"{element} may not be in stable coordination: {stability_assessment['reason']}")
        
        warnings = stability_warnings
        
        # **NEW LOGIC: Only reject if elements appear unstable, not just present**
        if potentially_problematic_found:
            return {
                "valid": False,
                "error": f"Elements appear to be in unstable configurations: {', '.join(potentially_problematic_found)}",
                "has_potentially_problematic_elements": True,
                "potentially_problematic_elements": potentially_problematic_found,
                "warnings": warnings,
                "has_radicals": True  # Assume radical-prone if unstable
            }
        
        # Check for basic SMILES syntax issues
        if smiles.count('(') != smiles.count(')'):
            return {
                "valid": False,
                "error": "Unmatched parentheses in SMILES",
                "warnings": warnings
            }
        
        if smiles.count('[') != smiles.count(']'):
            return {
                "valid": False,
                "error": "Unmatched square brackets in SMILES",
                "warnings": warnings
            }
        
        # **SUCCESS: Molecule appears stable**
        return {
            "valid": True,
            "has_radicals": False,
            "has_potentially_problematic_elements": len(potentially_problematic_found) > 0,
            "warnings": warnings,
            "molecule_info": {
                "elements_detected": list(set(char for char in smiles if char.isalpha() and char.isupper())),
                "stability_assessment": "String-based analysis suggests stable configuration"
            }
        }
    
    def _assess_element_stability_string(self, smiles: str, element: str) -> Dict[str, Any]:
        """Assess whether an element appears to be in a stable configuration using string analysis."""
        
        if element not in self.STABLE_COORDINATION_PATTERNS:
            return {"appears_stable": False, "reason": f"No stability patterns defined for {element}"}
        
        patterns = self.STABLE_COORDINATION_PATTERNS[element]
        
        # Simple heuristics for string-based stability assessment
        if element == 'B':
            # Look for coordination patterns
            if any(pattern in smiles for pattern in ['B(F)', 'B(Cl)', 'B(O)', 'B(N)']):
                return {"appears_stable": True, "reason": "Appears coordinated with electronegative atoms"}
            elif f'{element}(' in smiles:
                return {"appears_stable": True, "reason": "Appears to have coordination bonds"}
            else:
                return {"appears_stable": False, "reason": "No clear coordination pattern detected"}
                
        elif element == 'Si':
            # Look for tetrahedral coordination
            if any(pattern in smiles for pattern in ['Si(O)', 'Si(C)', '[Si]']):
                return {"appears_stable": True, "reason": "Appears in coordination environment"}
            elif f'{element}(' in smiles:
                return {"appears_stable": True, "reason": "Appears to have coordination bonds"}
            else:
                return {"appears_stable": False, "reason": "May lack proper tetrahedral coordination"}
                
        elif element == 'Al':
            # Look for coordination patterns
            if any(pattern in smiles for pattern in ['Al(Cl)', 'Al(F)', 'Al(O)', '[Al]']):
                return {"appears_stable": True, "reason": "Appears coordinated"}
            elif f'{element}(' in smiles:
                return {"appears_stable": True, "reason": "Appears to have coordination bonds"}
            else:
                return {"appears_stable": False, "reason": "May lack proper coordination"}
        
        # Default: assume potentially problematic
        return {"appears_stable": False, "reason": f"Unknown coordination pattern for {element}"}
    
    def _validate_with_rdkit_stability_focused(self, smiles: str) -> Dict[str, Any]:
        """Comprehensive validation using RDKit with stability-focused assessment."""
        
        try:
            # Parse molecule with RDKit
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            
            if mol is None:
                return {
                    "valid": False,
                    "error": "Invalid SMILES - RDKit cannot parse the structure",
                    "warnings": []
                }
            
            # Try to sanitize the molecule
            try:
                rdmolops.SanitizeMol(mol)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"SMILES sanitization failed: {str(e)}",
                    "warnings": []
                }
            
            # **PRIMARY CHECK: Actual radical detection**
            radical_check = self._check_for_radicals(mol)
            
            # Check molecular properties
            molecule_info = self._get_molecule_info(mol)
            
            # **SOPHISTICATED: Check element stability using molecular structure**
            element_stability_check = self._check_element_stability_rdkit(mol)
            
            warnings = []
            
            # Add warnings based on molecular properties
            if molecule_info.get("molecular_weight", 0) > 10000:
                warnings.append("Very high molecular weight - may cause simulation issues")
            
            if molecule_info.get("num_atoms", 0) > 1000:
                warnings.append("Very large molecule - may cause performance issues")
            
            # Add element stability warnings
            warnings.extend(element_stability_check.get("warnings", []))
            
            # Get corrected SMILES if possible
            corrected_smiles = None
            try:
                corrected_smiles = Chem.MolToSmiles(mol)
            except:
                pass
            
            # **DECISION LOGIC: Reject only if actually problematic**
            is_valid = True
            error_message = ""
            
            # Reject if has actual radicals
            if radical_check.get("has_radicals", False):
                is_valid = False
                error_message = f"Contains actual radical electrons: {radical_check.get('error', '')}"
            
            # Reject if elements are in clearly unstable configurations
            elif element_stability_check.get("has_unstable_elements", False):
                is_valid = False
                error_message = f"Elements in unstable configurations: {element_stability_check.get('error', '')}"
            
            return {
                "valid": is_valid,
                "error": error_message,
                "has_radicals": radical_check.get("has_radicals", False),
                "has_potentially_problematic_elements": element_stability_check.get("has_potentially_problematic_elements", False),
                "warnings": warnings,
                "molecule_info": molecule_info,
                "corrected_smiles": corrected_smiles,
                "radical_check": radical_check,
                "element_stability": element_stability_check
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"RDKit validation error: {str(e)}",
                "warnings": []
            }
    
    def _check_element_stability_rdkit(self, mol) -> Dict[str, Any]:
        """Check element stability using RDKit molecular structure analysis."""
        
        try:
            potentially_problematic_elements = []
            unstable_elements = []
            warnings = []
            
            for atom in mol.GetAtoms():
                element = atom.GetSymbol()
                
                if element in self.POTENTIALLY_PROBLEMATIC_ELEMENTS:
                    potentially_problematic_elements.append(element)
                    
                    # **SOPHISTICATED STABILITY ANALYSIS**
                    stability_analysis = self._analyze_atom_stability(atom, mol)
                    
                    if not stability_analysis["is_stable"]:
                        unstable_elements.append(element)
                        warnings.append(f"{element} (atom {atom.GetIdx()}): {stability_analysis['reason']}")
                    else:
                        warnings.append(f"{element} (atom {atom.GetIdx()}): Stable - {stability_analysis['reason']}")
            
            return {
                "has_potentially_problematic_elements": len(potentially_problematic_elements) > 0,
                "has_unstable_elements": len(unstable_elements) > 0,
                "potentially_problematic_elements": potentially_problematic_elements,
                "unstable_elements": unstable_elements,
                "warnings": warnings,
                "error": f"Unstable elements detected: {', '.join(unstable_elements)}" if unstable_elements else ""
            }
            
        except Exception as e:
            return {
                "has_potentially_problematic_elements": False,
                "has_unstable_elements": False,
                "error": f"Could not check element stability: {str(e)}"
            }
    
    def _analyze_atom_stability(self, atom, mol) -> Dict[str, Any]:
        """Analyze individual atom stability based on bonding environment."""
        
        element = atom.GetSymbol()
        degree = atom.GetDegree()  # Number of bonds
        total_hs = atom.GetTotalNumHs()  # Including implicit hydrogens
        formal_charge = atom.GetFormalCharge()
        
        if element == 'B':
            # Boron stable with 3-4 bonds
            total_bonds = degree + total_hs
            if total_bonds >= 3:
                return {"is_stable": True, "reason": f"Boron with {total_bonds} bonds (stable coordination)"}
            else:
                return {"is_stable": False, "reason": f"Boron with only {total_bonds} bonds (incomplete valence)"}
                
        elif element == 'Si':
            # Silicon stable with 4+ bonds (tetrahedral)
            total_bonds = degree + total_hs
            if total_bonds >= 4:
                return {"is_stable": True, "reason": f"Silicon with {total_bonds} bonds (tetrahedral/stable)"}
            else:
                return {"is_stable": False, "reason": f"Silicon with only {total_bonds} bonds (incomplete tetrahedron)"}
                
        elif element == 'Al':
            # Aluminum stable with 3+ bonds
            total_bonds = degree + total_hs
            if total_bonds >= 3:
                return {"is_stable": True, "reason": f"Aluminum with {total_bonds} bonds (stable coordination)"}
            else:
                return {"is_stable": False, "reason": f"Aluminum with only {total_bonds} bonds (incomplete coordination)"}
        
        # For other potentially problematic elements, use general rules
        elif element in self.POTENTIALLY_PROBLEMATIC_ELEMENTS:
            # General heuristic: if it has reasonable bonding, assume stable
            total_bonds = degree + total_hs
            if total_bonds >= 2:
                return {"is_stable": True, "reason": f"{element} with {total_bonds} bonds (appears coordinated)"}
            else:
                return {"is_stable": False, "reason": f"{element} with only {total_bonds} bonds (may be isolated)"}
        
        # Should not reach here for elements not in the problematic list
        return {"is_stable": True, "reason": f"{element} not in problematic elements list"}
    
    def _check_for_radicals(self, mol) -> Dict[str, Any]:
        """Check for radical electrons in the molecule."""
        
        try:
            # Check each atom for unpaired electrons
            radical_atoms = []
            total_radical_electrons = 0
            
            for atom in mol.GetAtoms():
                num_radical_electrons = atom.GetNumRadicalElectrons()
                if num_radical_electrons > 0:
                    radical_atoms.append({
                        "atom_idx": atom.GetIdx(),
                        "element": atom.GetSymbol(),
                        "num_radical_electrons": num_radical_electrons
                    })
                    total_radical_electrons += num_radical_electrons
            
            has_radicals = len(radical_atoms) > 0
            
            error_msg = ""
            if has_radicals:
                error_msg = f"Molecule contains {total_radical_electrons} radical electron(s) on {len(radical_atoms)} atom(s)"
            
            return {
                "has_radicals": has_radicals,
                "total_radical_electrons": total_radical_electrons,
                "radical_atoms": radical_atoms,
                "error": error_msg
            }
            
        except Exception as e:
            return {
                "has_radicals": False,
                "error": f"Could not check for radicals: {str(e)}"
            }
    
    def _get_molecule_info(self, mol) -> Dict[str, Any]:
        """Extract useful information about the molecule."""
        
        try:
            info = {
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "molecular_weight": Descriptors.MolWt(mol),
                "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "num_rings": Descriptors.RingCount(mol),
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "tpsa": Descriptors.TPSA(mol),  # Topological polar surface area
                "logp": Descriptors.MolLogP(mol),  # Lipophilicity
            }
            
            return info
            
        except Exception as e:
            logger.warning(f"Could not extract molecule info: {e}")
            return {}
    
    def correct_common_issues(self, smiles: str) -> Tuple[str, List[str]]:
        """
        Attempt to correct common issues in SMILES strings.
        
        Args:
            smiles: Original SMILES string
            
        Returns:
            Tuple of (corrected_smiles, list_of_corrections_made)
        """
        
        corrections = []
        corrected = smiles
        
        # Remove problematic elements by replacement
        element_replacements = {
            'B': 'C',   # Replace boron with carbon
            'Si': 'C',  # Replace silicon with carbon
            'Al': 'C',  # Replace aluminum with carbon
        }
        
        for problematic, replacement in element_replacements.items():
            if problematic in corrected:
                corrected = corrected.replace(problematic, replacement)
                corrections.append(f"Replaced {problematic} with {replacement}")
        
        # Remove explicit hydrogens
        if '[H]' in corrected:
            corrected = corrected.replace('[H]', '')
            corrections.append("Removed explicit hydrogens")
        
        return corrected, corrections
    
    def validate_psmiles(self, psmiles: str) -> ValidationResult:
        """
        Validate a PSMILES string (polymer SMILES with [*] connection points).
        
        Args:
            psmiles: PSMILES string to validate
            
        Returns:
            ValidationResult with validation information
        """
        
        if not psmiles or not psmiles.strip():
            return ValidationResult(
                is_valid=False,
                has_radicals=False,
                has_problematic_elements=False,
                error_message="Empty PSMILES string",
                warnings=[],
                molecule_info={}
            )
        
        psmiles = psmiles.strip()
        
        # Check for exactly two [*] connection points
        connection_count = psmiles.count('[*]')
        if connection_count != 2:
            return ValidationResult(
                is_valid=False,
                has_radicals=False,
                has_problematic_elements=False,
                error_message=f"PSMILES must have exactly 2 [*] connection points, found {connection_count}",
                warnings=[],
                molecule_info={}
            )
        
        # Convert PSMILES to SMILES for validation (replace [*] with dummy atoms)
        validation_smiles = psmiles.replace('[*]', 'C')
        
        # Validate the core structure
        result = self.validate_smiles(validation_smiles, "monomer")
        
        # Update result to indicate this was a PSMILES validation
        result.molecule_info['original_psmiles'] = psmiles
        result.molecule_info['validation_smiles'] = validation_smiles
        result.molecule_info['connection_points'] = connection_count
        
        return result


# Convenience functions for easy use
def validate_smiles_for_simulation(smiles: str) -> ValidationResult:
    """
    Quick validation function for SMILES before simulation.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        ValidationResult
    """
    validator = MolecularValidator()
    return validator.validate_smiles(smiles)

def validate_psmiles_for_simulation(psmiles: str) -> ValidationResult:
    """
    Quick validation function for PSMILES before simulation.
    
    Args:
        psmiles: PSMILES string to validate
        
    Returns:
        ValidationResult
    """
    validator = MolecularValidator()
    return validator.validate_psmiles(psmiles)

def is_safe_for_openmm(smiles: str) -> bool:
    """
    Quick check if a SMILES is safe for OpenMM/OpenFF simulation.
    
    Args:
        smiles: SMILES string to check
        
    Returns:
        True if safe for simulation, False otherwise
    """
    result = validate_smiles_for_simulation(smiles)
    return result.is_valid and not result.has_radicals and not result.has_problematic_elements 