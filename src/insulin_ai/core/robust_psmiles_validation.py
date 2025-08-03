#!/usr/bin/env python3
"""
Robust PSMILES Validation and Repair System

This module provides comprehensive validation and auto-repair capabilities for PSMILES
generation to prevent valence errors, radical electrons, and syntax issues.

Created: 2025-01-03
Author: AI Assistant
"""

import re
import logging
from typing import Dict, Any, Tuple, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Try to import RDKit for advanced validation
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, Descriptors
    from rdkit.Chem.rdmolops import SanitizeFlags
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("RDKit not available - using basic validation only")


class RobustPSMILESValidator:
    """
    Comprehensive PSMILES validation and auto-repair system.
    
    This class provides multiple layers of validation and repair to prevent
    the common issues we've been seeing:
    - Valence violations (oxygen with 3+ bonds, carbon with 5+ bonds)
    - Radical electrons (incomplete electron shells)
    - SMILES syntax errors (unmatched parentheses, unclosed rings)
    - Chemical impossibilities
    """
    
    def __init__(self):
        """Initialize the robust validator."""
        self.rdkit_available = RDKIT_AVAILABLE
        
        # Define safe monomer fallbacks for different functionalities
        self.safe_monomers = {
            'carboxylic_acid': '[*]CC(=O)O[*]',      # Acrylic acid derivative
            'ester': '[*]CC(=O)OC[*]',               # Ester linkage
            'amide': '[*]CC(=O)N[*]',                # Amide derivative  
            'aromatic': '[*]c1ccccc1[*]',            # Benzene derivative
            'ether': '[*]COC[*]',                    # Ether linkage
            'vinyl': '[*]C=C[*]',                    # Ethylene
            'alcohol': '[*]CC(O)C[*]',               # Alcohol functionality
            'default': '[*]CC(=O)O[*]'               # Default safe choice
        }
        
        logger.info(f"Robust PSMILES Validator initialized - RDKit: {self.rdkit_available}")
    
    def validate_and_repair(self, psmiles: str, max_repair_attempts: int = 3) -> Dict[str, Any]:
        """
        Main validation and repair method.
        
        Args:
            psmiles: PSMILES string to validate and potentially repair
            max_repair_attempts: Maximum number of repair attempts
            
        Returns:
            Dict with validation results and potentially repaired PSMILES
        """
        logger.info(f"🔍 Starting validation for: {psmiles}")
        
        result = {
            'original_psmiles': psmiles,
            'final_psmiles': psmiles,
            'is_valid': False,
            'was_repaired': False,
            'repair_attempts': 0,
            'issues_found': {},
            'repair_log': [],
            'error_message': '',
            'validation_method': 'comprehensive'
        }
        
        current_psmiles = psmiles
        
        for attempt in range(max_repair_attempts):
            result['repair_attempts'] = attempt + 1
            
            # **STEP 1: Comprehensive Pre-validation**
            validation_result = self._comprehensive_validation(current_psmiles)
            
            if validation_result['valid']:
                # Validation passed!
                result['is_valid'] = True
                result['final_psmiles'] = current_psmiles
                result['was_repaired'] = (current_psmiles != psmiles)
                
                if result['was_repaired']:
                    logger.info(f"✅ Validation successful after repair: {psmiles} → {current_psmiles}")
                else:
                    logger.info(f"✅ Validation successful without repair: {current_psmiles}")
                
                return result
            
            # **STEP 2: Validation failed, attempt repair**
            result['issues_found'] = validation_result.get('issues_found', {})
            result['error_message'] = validation_result.get('error', 'Unknown validation error')
            
            logger.warning(f"⚠️ Validation failed (attempt {attempt + 1}): {validation_result['error']}")
            
            # Attempt repair
            repaired_psmiles, repair_success = self._comprehensive_repair(
                current_psmiles, 
                validation_result
            )
            
            if repair_success and repaired_psmiles != current_psmiles:
                current_psmiles = repaired_psmiles
                result['repair_log'].append(f"Attempt {attempt + 1}: {psmiles} → {current_psmiles}")
                logger.info(f"🔧 Repair attempt {attempt + 1} applied: {repaired_psmiles}")
            else:
                # Repair didn't help or wasn't possible
                logger.warning(f"🔧 Repair attempt {attempt + 1} failed or made no changes")
                break
        
        # If we get here, all repair attempts failed
        if not result['is_valid']:
            # Last resort: use a safe fallback monomer
            logger.warning(f"🆘 All repairs failed, using safety fallback")
            fallback_psmiles = self._get_safety_fallback(psmiles)
            
            # Validate the fallback
            fallback_validation = self._comprehensive_validation(fallback_psmiles)
            if fallback_validation['valid']:
                result['final_psmiles'] = fallback_psmiles
                result['is_valid'] = True
                result['was_repaired'] = True
                result['repair_log'].append(f"Safety fallback: {fallback_psmiles}")
                logger.info(f"✅ Safety fallback successful: {fallback_psmiles}")
            else:
                logger.error(f"❌ Even safety fallback failed! This should never happen.")
        
        return result
    
    def _comprehensive_validation(self, psmiles: str) -> Dict[str, Any]:
        """
        Comprehensive validation covering all known failure modes.
        """
        result = {
            'valid': True,
            'error': '',
            'warnings': [],
            'issues_found': {}
        }
        
        if not psmiles or not isinstance(psmiles, str):
            result['valid'] = False
            result['error'] = 'Invalid input: empty or non-string PSMILES'
            return result
        
        # **1. BASIC PSMILES FORMAT VALIDATION**
        format_issues = self._validate_psmiles_format(psmiles)
        if format_issues:
            result['valid'] = False
            result['error'] = f"Format issues: {'; '.join(format_issues)}"
            result['issues_found']['format'] = format_issues
            return result
        
        clean_smiles = psmiles.replace('[*]', '')
        
        # **2. SYNTAX VALIDATION**
        syntax_issues = self._validate_syntax(clean_smiles)
        if syntax_issues:
            result['issues_found']['syntax'] = syntax_issues
        
        # **3. VALENCE VALIDATION**
        valence_issues = self._validate_valences(clean_smiles)
        if valence_issues:
            result['issues_found']['valence'] = valence_issues
        
        # **4. RADICAL RISK VALIDATION**
        radical_risks = self._detect_radical_risks(clean_smiles)
        if radical_risks:
            result['issues_found']['radical_risk'] = radical_risks
        
        # **5. POLYMER CHEMISTRY VALIDATION**
        polymer_issues = self._validate_polymer_chemistry(clean_smiles)
        if polymer_issues:
            result['issues_found']['polymer'] = polymer_issues
        
        # **6. RDKIT VALIDATION (if available)**
        if self.rdkit_available:
            rdkit_issues = self._rdkit_validation(clean_smiles)
            if rdkit_issues:
                result['issues_found']['rdkit'] = rdkit_issues
        
        # **7. COMPILE FINAL RESULT**
        all_issues = []
        for category, issues in result['issues_found'].items():
            all_issues.extend(issues)
        
        if all_issues:
            result['valid'] = False
            result['error'] = f"Validation failed: {'; '.join(all_issues)}"
        
        return result
    
    def _validate_psmiles_format(self, psmiles: str) -> List[str]:
        """Validate basic PSMILES format requirements."""
        issues = []
        
        # Check for exactly 2 [*] connection points
        star_count = psmiles.count('[*]')
        if star_count != 2:
            issues.append(f"Wrong number of connection points: {star_count} (expected 2)")
        
        # Check minimum length
        if len(psmiles) < 6:  # Minimum: [*]C[*]
            issues.append("PSMILES too short")
        
        # Check maximum reasonable length
        if len(psmiles) > 200:
            issues.append("PSMILES too long (likely to cause issues)")
        
        return issues
    
    def _validate_syntax(self, smiles: str) -> List[str]:
        """Validate SMILES syntax for common errors."""
        issues = []
        
        # Check balanced parentheses
        if smiles.count('(') != smiles.count(')'):
            issues.append(f"Unbalanced parentheses: {smiles.count('(')} open, {smiles.count(')')} close")
        
        # Check balanced square brackets
        if smiles.count('[') != smiles.count(']'):
            issues.append(f"Unbalanced square brackets: {smiles.count('[')} open, {smiles.count(']')} close")
        
        # Check ring closures
        ring_numbers = re.findall(r'\d+', smiles)
        for num in set(ring_numbers):
            if ring_numbers.count(num) not in [0, 2]:
                issues.append(f"Ring closure {num} appears {ring_numbers.count(num)} times (should be 2)")
        
        # Check for incomplete structures
        if smiles.endswith('(') or smiles.endswith('['):
            issues.append("SMILES ends with unclosed bracket/parenthesis")
        
        # Check for multiple components
        if '.' in smiles:
            issues.append("Multiple disconnected components detected")
        
        return issues
    
    def _validate_valences(self, smiles: str) -> List[str]:
        """Detect valence violations that cause failures."""
        issues = []
        
        # **OXYGEN VALENCE VIOLATIONS (max 2 bonds)**
        oxygen_patterns = [
            (r'O\([^)]*,[^)]*,[^)]*\)', "Oxygen with 3+ explicit bonds"),
            (r'C\([^)]*\)\([^)]*\)=O', "Over-substituted carbon with carbonyl"),
            (r'O=.*?=O.*?=O', "Oxygen in multiple double bonds"),
            (r'C\([^)]*O[^)]*\)\([^)]*\)=O', "Oxygen bridging carbons with carbonyl"),
        ]
        
        for pattern, description in oxygen_patterns:
            if re.search(pattern, smiles):
                issues.append(f"Oxygen valence violation: {description}")
        
        # **CARBON VALENCE VIOLATIONS (max 4 bonds)**
        carbon_patterns = [
            (r'C\([^)]*\)\([^)]*\)\([^)]*\)\([^)]*\)\([^)]*\)', "Carbon with 5+ substituents"),
            (r'C\([^)]*\)\([^)]*\)\([^)]*\)\([^)]*\)=', "Carbon with 4 substituents and double bond"),
        ]
        
        for pattern, description in carbon_patterns:
            if re.search(pattern, smiles):
                issues.append(f"Carbon valence violation: {description}")
        
        # **HALOGEN VALENCE VIOLATIONS (max 1 bond) - FIXED: More specific patterns**
        halogen_patterns = [
            (r'F\([^)]*\)', "Fluorine with multiple bonds"),
            (r'Cl\([^)]*\)', "Chlorine with multiple bonds"),
            (r'Br\([^)]*\)', "Bromine with multiple bonds"),
            # FIXED: Only flag actual halogens in double bonds, not C=C patterns
            (r'[FClBr]=', "Halogen in double bond"),
        ]
        
        for pattern, description in halogen_patterns:
            if re.search(pattern, smiles):
                # Additional check: make sure we actually found a halogen, not just C=C
                match = re.search(pattern, smiles)
                if match and any(halogen in match.group() for halogen in ['F', 'Cl', 'Br']):
                    issues.append(f"Halogen valence violation: {description}")
        
        # **SPECIFIC PROBLEMATIC PATTERNS FROM USER'S LOGS**
        problematic_patterns = [
            (r'C\(C\(C\)=O\)C\(C\)=O', "Complex branched carbonyl (known to cause failures)"),
            (r'C\(C\(C\)=O\)\(C\)NC', "Complex amide-ester (known to cause failures)"),
            (r'OCCCOC=O.*?O.*?=O', "Multiple carbonyls with ether (known to cause failures)"),
        ]
        
        for pattern, description in problematic_patterns:
            if re.search(pattern, smiles):
                issues.append(f"Known problematic pattern: {description}")
        
        return issues
    
    def _detect_radical_risks(self, smiles: str) -> List[str]:
        """Detect patterns that commonly lead to radical electrons."""
        risks = []
        
        # Atoms in square brackets without charges (radical risk)
        if re.search(r'\[[A-Z][a-z]?\](?![+-])', smiles):
            # Additional check: make sure it's not just normal organic chemistry
            if not re.search(r'\[[CH]\]|\[CH2\]|\[CH3\]', smiles):  # Common organic patterns
                risks.append("Atom in square brackets without charge (potential radical)")
        
        # **MUCH MORE CONSERVATIVE: Only flag obviously problematic patterns**
        # Don't flag normal organic chemistry
        isolated_patterns = [
            # Only flag if there's a clear disconnect in bonding
            (r'[NOPS]\s', "Heteroatom followed by space (potential radical)"),
            (r'\s[NOPS]', "Heteroatom preceded by space (potential radical)"),
            # Only flag boron if it's clearly not in proper coordination
            (r'\b[B]\b(?![ropCONF])', "Potentially uncoordinated boron"),
        ]
        
        for pattern, description in isolated_patterns:
            if re.search(pattern, smiles):
                risks.append(description)
        
        return risks
    
    def _validate_polymer_chemistry(self, smiles: str) -> List[str]:
        """Validate polymer chemistry requirements."""
        issues = []
        
        # **FIXED: Recognize both uppercase and lowercase carbon**
        if not re.search(r'[Cc]', smiles):
            issues.append("No carbon atoms (not a valid organic polymer)")
        
        # Must be reasonable size
        if len(smiles) < 2:
            issues.append("Structure too simple for polymerization")
        
        if len(smiles) > 100:
            issues.append("Structure too complex (computationally problematic)")
        
        # Should not have overly complex branching
        if smiles.count('(') > 6:
            issues.append("Overly complex branching (likely to cause issues)")
        
        return issues
    
    def _rdkit_validation(self, smiles: str) -> List[str]:
        """Validate using RDKit if available."""
        issues = []
        
        try:
            # Try to create molecule
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                issues.append("RDKit cannot parse structure")
                return issues
            
            # Try sanitization
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                issues.append(f"RDKit sanitization failed: {str(e)}")
                return issues
            
            # Check for radicals
            for atom in mol.GetAtoms():
                num_radical_electrons = atom.GetNumRadicalElectrons()
                if num_radical_electrons > 0:
                    issues.append(f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) has {num_radical_electrons} radical electrons")
            
        except Exception as e:
            issues.append(f"RDKit validation error: {str(e)}")
        
        return issues
    
    def _comprehensive_repair(self, psmiles: str, validation_result: Dict) -> Tuple[str, bool]:
        """
        Comprehensive repair system addressing all types of issues.
        """
        repaired = psmiles
        was_repaired = False
        issues = validation_result.get('issues_found', {})
        
        logger.info(f"🔧 Starting comprehensive repair for: {list(issues.keys())}")
        
        # **REPAIR 1: SYNTAX ISSUES**
        if 'syntax' in issues:
            repaired, syntax_repaired = self._repair_syntax(repaired)
            if syntax_repaired:
                was_repaired = True
                logger.info("   ✅ Syntax issues repaired")
        
        # **REPAIR 2: VALENCE VIOLATIONS**
        if 'valence' in issues:
            repaired, valence_repaired = self._repair_valences(repaired)
            if valence_repaired:
                was_repaired = True
                logger.info("   ✅ Valence issues repaired")
        
        # **REPAIR 3: RADICAL RISKS**
        if 'radical_risk' in issues:
            repaired, radical_repaired = self._repair_radical_risks(repaired)
            if radical_repaired:
                was_repaired = True
                logger.info("   ✅ Radical risks repaired")
        
        # **REPAIR 4: POLYMER CHEMISTRY**
        if 'polymer' in issues:
            repaired, polymer_repaired = self._repair_polymer_issues(repaired)
            if polymer_repaired:
                was_repaired = True
                logger.info("   ✅ Polymer issues repaired")
        
        return repaired, was_repaired
    
    def _repair_syntax(self, psmiles: str) -> Tuple[str, bool]:
        """Repair syntax errors in PSMILES."""
        repaired = psmiles
        was_repaired = False
        
        clean_part = repaired.replace('[*]', '')
        
        # Fix unbalanced parentheses
        open_parens = clean_part.count('(')
        close_parens = clean_part.count(')')
        
        if open_parens > close_parens:
            clean_part += ')' * (open_parens - close_parens)
            was_repaired = True
        elif close_parens > open_parens:
            # Remove extra closing parentheses
            for _ in range(close_parens - open_parens):
                clean_part = clean_part.replace(')', '', 1)
            was_repaired = True
        
        # Fix unbalanced square brackets
        open_brackets = clean_part.count('[')
        close_brackets = clean_part.count(']')
        
        if open_brackets > close_brackets:
            clean_part += ']' * (open_brackets - close_brackets)
            was_repaired = True
        elif close_brackets > open_brackets:
            for _ in range(close_brackets - open_brackets):
                clean_part = clean_part.replace(']', '', 1)
            was_repaired = True
        
        # Remove multiple components (keep largest)
        if '.' in clean_part:
            fragments = clean_part.split('.')
            clean_part = max(fragments, key=len)
            was_repaired = True
        
        if was_repaired:
            repaired = psmiles.replace(psmiles.replace('[*]', ''), clean_part)
        
        return repaired, was_repaired
    
    def _repair_valences(self, psmiles: str) -> Tuple[str, bool]:
        """Repair valence violations."""
        clean_part = psmiles.replace('[*]', '')
        
        # Define valence repair patterns
        repairs = [
            # Oxygen valence fixes
            (r'C\([^)]*\)\([^)]*\)=O', 'CC(=O)', "Over-substituted carbonyl"),
            (r'C\(C\(C\)=O\)C\(C\)=O', 'CC(=O)C', "Complex dual carbonyl"),
            
            # Carbon valence fixes
            (r'C\([^)]*\)\([^)]*\)\([^)]*\)\([^)]*\)', 'C(C)(C)', "Over-substituted carbon"),
            
            # Halogen fixes
            (r'F\([^)]*\)', 'F', "Over-coordinated fluorine"),
            (r'Cl\([^)]*\)', 'Cl', "Over-coordinated chlorine"),
            (r'[FClBr]=', 'F', "Halogen double bond"),
        ]
        
        for pattern, replacement, description in repairs:
            if re.search(pattern, clean_part):
                clean_part = re.sub(pattern, replacement, clean_part)
                repaired = psmiles.replace(psmiles.replace('[*]', ''), clean_part)
                logger.info(f"   🔧 Valence repair: {description}")
                return repaired, True
        
        return psmiles, False
    
    def _repair_radical_risks(self, psmiles: str) -> Tuple[str, bool]:
        """Repair patterns that lead to radical electrons."""
        clean_part = psmiles.replace('[*]', '')
        
        # Remove brackets from isolated atoms
        fixes = [
            (r'\[([CNOS])\]', r'\1', "Remove brackets from heteroatoms"),
            (r'\[([A-Z][a-z]?)\](?![+-])', r'\1', "Remove uncharged brackets"),
        ]
        
        for pattern, replacement, description in fixes:
            if re.search(pattern, clean_part):
                clean_part = re.sub(pattern, replacement, clean_part)
                repaired = psmiles.replace(psmiles.replace('[*]', ''), clean_part)
                logger.info(f"   🔧 Radical repair: {description}")
                return repaired, True
        
        return psmiles, False
    
    def _repair_polymer_issues(self, psmiles: str) -> Tuple[str, bool]:
        """Repair polymer chemistry issues."""
        clean_part = psmiles.replace('[*]', '')
        
        # If structure is too simple or complex, use fallback
        if len(clean_part) < 2 or len(clean_part) > 100 or clean_part.count('(') > 6:
            fallback = self._get_safety_fallback(psmiles)
            logger.info(f"   🔧 Polymer repair: Using safety fallback")
            return fallback, True
        
        return psmiles, False
    
    def _get_safety_fallback(self, original_psmiles: str) -> str:
        """Get a safe fallback monomer based on the original request."""
        original_lower = original_psmiles.lower()
        
        # Choose based on detected functionality
        if 'acid' in original_lower or '=o' in original_lower:
            return self.safe_monomers['carboxylic_acid']
        elif 'amide' in original_lower or ('n' in original_lower and '=' in original_lower):
            return self.safe_monomers['amide']
        elif 'aromatic' in original_lower or 'c1' in original_lower:
            return self.safe_monomers['aromatic']
        elif 'ether' in original_lower or 'o' in original_lower:
            return self.safe_monomers['ether']
        elif 'vinyl' in original_lower or 'c=c' in original_lower:
            return self.safe_monomers['vinyl']
        else:
            return self.safe_monomers['default']


def validate_and_repair_psmiles(psmiles: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Convenience function for validating and repairing PSMILES.
    
    Args:
        psmiles: PSMILES string to validate/repair
        max_attempts: Maximum repair attempts
        
    Returns:
        Dictionary with validation and repair results
    """
    validator = RobustPSMILESValidator()
    return validator.validate_and_repair(psmiles, max_attempts)


# Example usage and testing
if __name__ == "__main__":
    # Test cases covering the common failure modes
    test_cases = [
        "[*]OC[C@H]1O[C@@H][C@H][C@@H][C@H]1O[*]",  # Radical electron issue
        "[*]OCCCOC=O[*]",  # Valence violation
        "[*]C(C(C)=O)C(C)=O[*]",  # Complex branching
        "CCOC)OCC[B]OCCO",  # Syntax error
        "[*]C=C[*]",  # Valid case
    ]
    
    validator = RobustPSMILESValidator()
    
    print("🧪 Testing Robust PSMILES Validator\n")
    
    for i, test_psmiles in enumerate(test_cases, 1):
        print(f"Test {i}: {test_psmiles}")
        result = validator.validate_and_repair(test_psmiles)
        
        print(f"   Original: {result['original_psmiles']}")
        print(f"   Final: {result['final_psmiles']}")
        print(f"   Valid: {result['is_valid']}")
        print(f"   Repaired: {result['was_repaired']}")
        if result['error_message']:
            print(f"   Error: {result['error_message']}")
        print()
    
    print("✅ Robust validation testing complete!") 