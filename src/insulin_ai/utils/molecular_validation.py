#!/usr/bin/env python3
"""
Molecular Validation Utilities for Insulin-AI

This module provides comprehensive validation functions for molecular structures,
with special focus on detecting and preventing radical species that cause
OpenFF Toolkit failures in molecular dynamics simulations.

Author: AI-Driven Material Discovery Team
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path

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
        Correct common chemical issues in SMILES strings.
        
        ENHANCED: Now includes silicon-specific corrections for valency completion.
        """
        corrected = smiles
        corrections = []
        
        # Check for unpaired brackets
        if corrected.count('(') != corrected.count(')'):
            bracket_diff = corrected.count('(') - corrected.count(')')
            if bracket_diff > 0:
                corrected += ')' * bracket_diff
                corrections.append(f"Added {bracket_diff} closing parentheses")
            else:
                # Remove extra closing parentheses
                for _ in range(abs(bracket_diff)):
                    last_paren = corrected.rfind(')')
                    if last_paren != -1:
                        corrected = corrected[:last_paren] + corrected[last_paren+1:]
                corrections.append(f"Removed {abs(bracket_diff)} extra closing parentheses")
        
        # Check for unpaired square brackets
        if corrected.count('[') != corrected.count(']'):
            bracket_diff = corrected.count('[') - corrected.count(']')
            if bracket_diff > 0:
                corrected += ']' * bracket_diff
                corrections.append(f"Added {bracket_diff} closing square brackets")
            else:
                # Remove extra closing square brackets
                for _ in range(abs(bracket_diff)):
                    last_bracket = corrected.rfind(']')
                    if last_bracket != -1:
                        corrected = corrected[:last_bracket] + corrected[last_bracket+1:]
                corrections.append(f"Removed {abs(bracket_diff)} extra closing square brackets")
        
        # **NEW: Silicon-specific valency correction**
        if '[Si]' in corrected or 'Si' in corrected:
            silicon_corrected, silicon_corrections = self._correct_silicon_valency(corrected)
            if silicon_corrected != corrected:
                corrected = silicon_corrected
                corrections.extend(silicon_corrections)
        
        return corrected, corrections
    
    def _correct_silicon_valency(self, smiles: str) -> Tuple[str, List[str]]:
        """
        Correct silicon valency issues in SMILES strings using intelligent bond counting.
        
        Silicon atoms need exactly 4 bonds to be stable. This method analyzes existing
        bonds and adds only the minimum substituents needed to reach exactly 4 bonds.
        
        Args:
            smiles: Original SMILES string
            
        Returns:
            Tuple of (corrected_smiles, list_of_corrections_made)
        """
        corrections = []
        
        # Import regex
        import re
        
        # Use RDKit to parse and analyze the molecule if available
        if self.rdkit_available:
            try:
                from rdkit import Chem
                
                # Parse the original molecule to understand current bonding
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is None:
                    # Fallback to string-based correction
                    return self._fallback_silicon_correction(smiles)
                
                # Analyze each silicon atom and its current bonds
                silicon_fixes = []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'Si':
                        current_bonds = atom.GetDegree()  # Number of explicit bonds
                        bonds_needed = 4 - current_bonds
                        
                        if bonds_needed > 0:
                            # We need to add substituents
                            silicon_fixes.append({
                                'atom_idx': atom.GetIdx(),
                                'current_bonds': current_bonds,
                                'bonds_needed': bonds_needed
                            })
                
                if not silicon_fixes:
                    # No silicon atoms need fixing
                    return smiles, corrections
                
                # Apply fixes using string replacement patterns
                corrected_smiles = self._apply_smart_silicon_fixes(smiles, silicon_fixes)
                
                if corrected_smiles != smiles:
                    corrections.append(f"Fixed {len(silicon_fixes)} silicon atoms by adding appropriate substituents")
                    
                    # Validate the result
                    test_mol = Chem.MolFromSmiles(corrected_smiles, sanitize=False)
                    if test_mol is not None:
                        try:
                            Chem.SanitizeMol(test_mol)
                            corrections.append("✅ All silicon atoms now have proper tetrahedral coordination")
                        except:
                            corrections.append("⚠️ Warning: Corrected structure may need additional validation")
                    else:
                        corrections.append("⚠️ Warning: Correction may have created parsing issues")
                
                return corrected_smiles, corrections
                
            except Exception as e:
                corrections.append(f"⚠️ RDKit analysis failed: {str(e)}, using fallback method")
                return self._fallback_silicon_correction(smiles)
        else:
            # RDKit not available, use fallback
            return self._fallback_silicon_correction(smiles)
    
    def _apply_smart_silicon_fixes(self, smiles: str, silicon_fixes: List[Dict]) -> str:
        """Apply intelligent silicon fixes based on bond analysis"""
        
        # Strategy: Use common silicon coordination patterns
        corrected = smiles
        
        # Pattern 1: [Si]O[Si] → [Si](C)(C)O[Si](C)(C) (each Si gets 2 methyls, has 2 bonds: O and next Si)
        if '[Si]O[Si]' in corrected:
            corrected = corrected.replace('[Si]O[Si]', '[Si](C)(C)O[Si](C)(C)')
        
        # Pattern 2: Remaining [Si]O → [Si](C)(C)(C)O (Si has 1 bond to O, needs 3 more)
        if '[Si]O' in corrected and '[Si](C)(C)O' not in corrected:
            corrected = corrected.replace('[Si]O', '[Si](C)(C)(C)O')
        
        # Pattern 3: [Si]C → [Si](C)(C)(C)C (Si has 1 bond to C, needs 3 more)
        if re.search(r'\[Si\][CN]', corrected):
            corrected = re.sub(r'\[Si\]([CN])', r'[Si](C)(C)(C)\1', corrected)
        
        # Pattern 4: Isolated [Si] → [Si](C)(C)(C)(C) (Si has 0 bonds, needs 4)
        if '[Si]' in corrected:
            corrected = corrected.replace('[Si]', '[Si](C)(C)(C)(C)')
        
        return corrected
    
    def _fallback_silicon_correction(self, smiles: str) -> Tuple[str, List[str]]:
        """Fallback string-based silicon correction when RDKit is not available"""
        
        corrections = []
        corrected = smiles
        
        # Strategy: Apply corrections in order of specificity, ensuring no double-correction
        
        # 1. Fix Si-O-Si patterns first (most specific) - each Si gets exactly 2 methyls
        if '[Si]O[Si]' in corrected:
            pattern_count = corrected.count('[Si]O[Si]')
            corrected = corrected.replace('[Si]O[Si]', '[Si](C)(C)O[Si](C)(C)')
            corrections.append(f"Fixed {pattern_count} Si-O-Si patterns with dimethyl substitution")
        
        # 2. Fix remaining Si-O patterns (Si has 1 bond to O, needs 3 more methyls)
        # But only if we haven't already fixed them above
        remaining_si_o = corrected.count('[Si]O')
        if remaining_si_o > 0:
            corrected = corrected.replace('[Si]O', '[Si](C)(C)(C)O')
            corrections.append(f"Fixed {remaining_si_o} remaining Si-O patterns with trimethyl substitution")
        
        # 3. Fix Si-C or Si-N patterns (Si has 1 bond, needs 3 more methyls)
        import re
        si_cn_matches = re.findall(r'\[Si\][CN]', corrected)
        if si_cn_matches:
            corrected = re.sub(r'\[Si\]([CN])', r'[Si](C)(C)(C)\1', corrected)
            corrections.append(f"Fixed {len(si_cn_matches)} Si-C/N patterns with trimethyl substitution")
        
        # 4. Finally, fix any completely isolated [Si] atoms (no bonds, needs 4 methyls)
        remaining_isolated = corrected.count('[Si]')
        if remaining_isolated > 0:
            corrected = corrected.replace('[Si]', '[Si](C)(C)(C)(C)')
            corrections.append(f"Fixed {remaining_isolated} isolated Si atoms with tetramethyl substitution")
        
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
        
        # **NEW: Apply silicon correction to PSMILES before validation**
        corrected_psmiles = psmiles
        silicon_corrections = []
        
        if '[Si]' in psmiles or 'Si' in psmiles:
            # Extract the monomer part (remove [*] symbols)
            monomer_smiles = psmiles.replace('[*]', 'C')  # Replace with dummy carbons for analysis
            
            # Apply silicon corrections
            corrected_monomer, corrections = self.correct_common_issues(monomer_smiles)
            
            if corrections:
                # Apply the same corrections to the original PSMILES
                corrected_psmiles = self._apply_corrections_to_psmiles(psmiles, corrections)
                silicon_corrections = [c for c in corrections if 'silicon' in c.lower() or 'Si' in c]
        
        # Convert PSMILES to SMILES for validation (replace [*] with dummy atoms)
        validation_smiles = corrected_psmiles.replace('[*]', 'C')
        
        # Validate the core structure
        result = self.validate_smiles(validation_smiles, "monomer")
        
        # Update result to indicate this was a PSMILES validation
        result.molecule_info['original_psmiles'] = psmiles
        result.molecule_info['corrected_psmiles'] = corrected_psmiles
        result.molecule_info['validation_smiles'] = validation_smiles
        result.molecule_info['connection_points'] = connection_count
        
        # Add silicon correction info if any were applied
        if silicon_corrections:
            result.warnings.extend(silicon_corrections)
            result.molecule_info['silicon_corrections_applied'] = True
        
        return result
    
    def _apply_corrections_to_psmiles(self, psmiles: str, corrections: List[str]) -> str:
        """Apply silicon corrections to a PSMILES string"""
        
        corrected = psmiles
        
        # Simple approach: Apply the most common silicon corrections to PSMILES format
        # [*]...[Si]O...  → [*]...[Si](C)(C)O...
        if 'Si-O patterns' in str(corrections):
            corrected = corrected.replace('[Si]O', '[Si](C)(C)O')
        
        # [*]...[Si]C... or [*]...[Si]N... → [*]...[Si](C)(C)C/N...
        if 'Si-C/N patterns' in str(corrections):
            import re
            corrected = re.sub(r'\[Si\]([CN])', r'[Si](C)(C)\1', corrected)
        
        # Isolated [Si] → [Si](C)(C)(C)
        if 'isolated Si atoms' in str(corrections):
            corrected = corrected.replace('[Si]', '[Si](C)(C)(C)')
        
        return corrected


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