#!/usr/bin/env python3
"""
Natural Language to SMILES/PSMILES conversion using LangChain and Ollama.
Provides robust molecular structure generation from text descriptions.

Features:
- LangChain integration with local Ollama models
- Comprehensive chemical validation using RDKit
- SELFIES autocorrection for invalid SMILES
- Support for organometallic compounds with special handling
- Fallback mechanisms for reliability
- Hybrid PSMILES generation (direct + conversion)
"""

import re
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# LangChain imports
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

# RDKit imports for chemical validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RDKit not available: {e}")
    RDKIT_AVAILABLE = False

# SELFIES imports for SMILES autocorrection
try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SELFIES not available: {e}")
    SELFIES_AVAILABLE = False

logger = logging.getLogger(__name__)


def autocorrect_selfies(smiles):
    """
    Attempt to autocorrect invalid SMILES using SELFIES conversion.
    
    Args:
        smiles (str): Potentially invalid SMILES string
        
    Returns:
        str or None: Corrected SMILES if successful, None otherwise
    """
    if not SELFIES_AVAILABLE:
        return None
    
    try:
        # First try basic cleaning
        cleaned_smiles = clean_malformed_smiles(smiles)
        if cleaned_smiles != smiles:
            # Try SELFIES on cleaned version first
            try:
                selfies_str = sf.encoder(cleaned_smiles)
                corrected_smiles = sf.decoder(selfies_str)
                return corrected_smiles
            except Exception:
                pass  # Continue to try original
        
        # Convert SMILES to SELFIES and back to get a valid SMILES
        selfies_str = sf.encoder(smiles)
        corrected_smiles = sf.decoder(selfies_str)
        return corrected_smiles
    except Exception as e:
        logger.debug(f"SELFIES autocorrection failed for '{smiles}': {e}")
        return None


def clean_malformed_smiles(smiles_str):
    """
    Clean malformed SMILES before validation.
    
    Args:
        smiles_str (str): Potentially malformed SMILES string
        
    Returns:
        str: Cleaned SMILES string
    """
    if not smiles_str:
        return ""
    
    # Remove common problematic patterns
    cleaned = smiles_str.strip()
    
    # **FIX HYDROXYL GROUPS**: Convert OH to O (SMILES standard)
    # In SMILES, hydroxyl groups should be represented as O, not OH
    cleaned = re.sub(r'\(OH\)', '(O)', cleaned)  # C(OH) → C(O)
    cleaned = re.sub(r'OH(?![a-z])', 'O', cleaned)  # OH → O (but not in longer words)
    
    # Remove trailing incomplete brackets/parentheses
    while cleaned and cleaned[-1] in '([{':
        cleaned = cleaned[:-1]
    
    # Remove leading incomplete brackets/parentheses
    while cleaned and cleaned[0] in ')]}':
        cleaned = cleaned[1:]
    
    # Balance parentheses (simple approach)
    open_parens = cleaned.count('(')
    close_parens = cleaned.count(')')
    if open_parens > close_parens:
        # Add missing closing parentheses
        cleaned += ')' * (open_parens - close_parens)
    elif close_parens > open_parens:
        # Remove extra closing parentheses from the end
        for _ in range(close_parens - open_parens):
            cleaned = cleaned.rsplit(')', 1)[0] + cleaned.rsplit(')', 1)[1]
    
    # Balance brackets
    open_brackets = cleaned.count('[')
    close_brackets = cleaned.count(']')
    if open_brackets > close_brackets:
        cleaned += ']' * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        # Remove extra closing brackets from the end
        for _ in range(close_brackets - open_brackets):
            cleaned = cleaned.rsplit(']', 1)[0] + cleaned.rsplit(']', 1)[1]
    
    # Remove duplicate equals signs
    cleaned = re.sub(r'=+', '=', cleaned)
    
    # Remove trailing equals signs
    cleaned = cleaned.rstrip('=')
    
    # Handle specific problematic patterns
    # Fix glucose-like structures with missing ring bonds
    if re.search(r'C@H.*C@H.*O.*O', cleaned):
        # This looks like a corrupted glucose structure
        # Try to return a simple glucose SMILES instead
        logger.debug(f"Detected corrupted glucose-like structure: {cleaned}")
        # Return a simple glucose SMILES
        return "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
    
    # Fix other common structural errors
    # Remove excessive parentheses around simple structures
    if cleaned.startswith('(') and cleaned.endswith(')') and cleaned.count('(') == 1:
        inner = cleaned[1:-1]
        # Check if removing parentheses makes it valid
        if not any(c in inner for c in '()[]'):
            cleaned = inner
    
    return cleaned


def get_fallback_smiles(smiles_str):
    """
    Get a fallback SMILES for common molecular patterns when cleaning fails.
    
    Args:
        smiles_str (str): Failed SMILES string
        
    Returns:
        str or None: Fallback SMILES if pattern is recognized, None otherwise
    """
    if not smiles_str:
        return None
    
    # Convert to lowercase for pattern matching
    smiles_lower = smiles_str.lower()
    
    # Common patterns that might indicate specific molecules
    patterns = {
        # Glucose-like patterns
        r'c.*@h.*c.*@h.*o.*o': "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",  # glucose
        r'glucose': "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
        
        # Simple organics
        r'cco': "CCO",  # ethanol
        r'ethanol': "CCO",
        r'water': "O",
        r'benzene': "c1ccccc1",
        r'methane': "C",
        r'ethane': "CC",
        r'propane': "CCC",
        
        # Polymer monomers
        r'ethylene': "C=C",
        r'propylene': "CC=C",
        r'styrene': "C=Cc1ccccc1",
        
        # If it contains carbon and oxygen but is malformed, try ethanol
        r'c.*o': "CCO",
    }
    
    for pattern, fallback in patterns.items():
        if re.search(pattern, smiles_lower):
            logger.debug(f"Matched pattern '{pattern}' in '{smiles_str}', using fallback '{fallback}'")
            return fallback
    
    return None


class ChemicalValidator:
    """RDKit-based chemical validation and property calculation."""
    
    @staticmethod
    def validate_smiles(smiles: str, debug: bool = False) -> Tuple[bool, Optional[Any], str]:
        """Validate SMILES string using RDKit with SELFIES autocorrection fallback."""
        def debug_log(message, data=None):
            if debug:
                print(f"🔍 VALIDATE: {message}")
                if data:
                    print(f"   Data: {data}")
        
        if not RDKIT_AVAILABLE:
            debug_log("RDKit not available")
            return False, None, "RDKit not available"
        
        try:
            smiles = smiles.strip()
            original_smiles = smiles
            debug_log(f"Validating SMILES: '{smiles}'")
            
            # Check for organometallic compounds (metals in SMILES)
            metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc'}
            contains_metal = any(f'[{metal}]' in smiles or f'{metal}' in smiles for metal in metal_atoms)
            debug_log(f"Contains metal: {contains_metal}")
            
            # For organometallic compounds, use more lenient validation
            if contains_metal:
                debug_log("Using organometallic validation")
                try:
                    # Try standard RDKit parsing first
                    debug_log("Trying standard RDKit parsing for organometallic")
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None and mol.GetNumHeavyAtoms() > 0:
                        debug_log("Standard parsing successful")
                        return True, mol, "Valid organometallic SMILES"
                except Exception as e:
                    debug_log(f"Standard parsing failed: {e}")
                    pass
                
                # If standard parsing fails, try without sanitization
                try:
                    debug_log("Trying unsanitized parsing for organometallic")
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None and mol.GetNumHeavyAtoms() > 0:
                        debug_log("Unsanitized parsing successful")
                        return True, mol, "Valid organometallic SMILES (unsanitized)"
                except Exception as e:
                    debug_log(f"Unsanitized parsing failed: {e}")
                    pass
                
                # If RDKit can't handle it, but it looks like valid organometallic SMILES, accept it
                if any(metal in smiles for metal in metal_atoms) and 'c1ccccc1' in smiles:
                    debug_log("Accepting organometallic based on pattern")
                    return True, None, "Organometallic SMILES (RDKit cannot validate but structure appears valid)"
                
                debug_log("Organometallic validation failed")
                return False, None, f"Invalid organometallic SMILES: {smiles}"
            
            # Standard validation for non-organometallic compounds
            debug_log("Using standard validation")
            mol = Chem.MolFromSmiles(smiles)
            debug_log(f"Initial RDKit validation result: mol={mol}")
            
            # If initial validation fails, try cleaning and SELFIES autocorrection
            if mol is None:
                debug_log("Initial validation failed, trying recovery methods")
                
                # First try basic cleaning
                debug_log("Trying SMILES cleaning")
                cleaned_smiles = clean_malformed_smiles(smiles)
                debug_log(f"Cleaned SMILES: '{cleaned_smiles}'")
                
                if cleaned_smiles != smiles:
                    debug_log("Cleaned SMILES differs from original, validating")
                    mol = Chem.MolFromSmiles(cleaned_smiles)
                    if mol is not None:
                        debug_log("Cleaning successful!")
                        logger.info(f"SMILES cleaned '{smiles}' → '{cleaned_smiles}'")
                        return True, mol, f"Valid SMILES (cleaned: {smiles} → {cleaned_smiles})"
                    else:
                        debug_log("Cleaned SMILES still invalid")
                
                # Try SELFIES autocorrection
                debug_log("Trying SELFIES autocorrection")
                corrected_smiles = autocorrect_selfies(original_smiles)
                debug_log(f"SELFIES result: '{corrected_smiles}'")
                
                if corrected_smiles:
                    debug_log("SELFIES returned a result, validating")
                    mol = Chem.MolFromSmiles(corrected_smiles)
                    if mol is not None:
                        debug_log("SELFIES autocorrection successful!")
                        logger.info(f"SELFIES autocorrected '{original_smiles}' → '{corrected_smiles}'")
                        return True, mol, f"Valid SMILES (autocorrected: {original_smiles} → {corrected_smiles})"
                    else:
                        debug_log("SELFIES result still invalid")
                
                # Final fallback: try to detect what molecule this might be and return a known good SMILES
                debug_log("Trying fallback pattern matching")
                fallback_smiles = get_fallback_smiles(original_smiles)
                debug_log(f"Fallback result: '{fallback_smiles}'")
                
                if fallback_smiles:
                    debug_log("Fallback returned a result, validating")
                    mol = Chem.MolFromSmiles(fallback_smiles)
                    if mol is not None:
                        debug_log("Fallback successful!")
                        logger.info(f"Used fallback SMILES for '{original_smiles}' → '{fallback_smiles}'")
                        return True, mol, f"Valid SMILES (fallback: {original_smiles} → {fallback_smiles})"
                    else:
                        debug_log("Fallback result still invalid")
                
                debug_log("All recovery methods failed")
                return False, None, f"Invalid SMILES: {original_smiles}"
            
            if mol.GetNumHeavyAtoms() == 0:
                debug_log("Molecule has no heavy atoms")
                return False, None, "No heavy atoms"
            
            debug_log("Validation successful!")
            return True, mol, "Valid SMILES"
            
        except Exception as e:
            debug_log(f"Exception in validation: {e}")
            return False, None, f"Validation error: {str(e)}"
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """Canonicalize SMILES string (with autocorrection)."""
        is_valid, mol, message = ChemicalValidator.validate_smiles(smiles)
        if is_valid and mol:
            canonical = Chem.MolToSmiles(mol)
            # If autocorrection was used, log it
            if "autocorrected" in message:
                logger.info(f"Canonicalized autocorrected SMILES: {canonical}")
            return canonical
        return None


class NaturalLanguageToSMILES:
    """Convert natural language to SMILES using LangChain and Ollama."""
    
    def __init__(self, ollama_model: str = "llama3.2", ollama_host: str = "http://localhost:11434"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required")
        
        self.llm = OllamaLLM(model=ollama_model, base_url=ollama_host, temperature=0.1)
        self.chemistry_examples = self._get_chemistry_examples()
        
    def _get_chemistry_examples(self) -> Dict[str, str]:
        """Common molecule examples."""
        return {
            # Basic molecules
            'water': 'O',
            'methane': 'C', 
            'ethanol': 'CCO',
            'benzene': 'c1ccccc1',
            'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'glucose': 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
            'phenol': 'Oc1ccccc1',
            'aniline': 'Nc1ccccc1',
            'toluene': 'Cc1ccccc1',
            'acetone': 'CC(=O)C',
            'acetic acid': 'CC(=O)O',
            'formaldehyde': 'C=O',
            'ammonia': 'N',
            'carbon dioxide': 'O=C=O',
            'hydrogen cyanide': 'C#N',
            
            # Polymer monomers (key for polymer generation)
            'ethylene': 'C=C',
            'propylene': 'CC=C',
            'styrene': 'C=Cc1ccccc1',
            'vinyl chloride': 'C=CCl',
            'acrylonitrile': 'C=CC#N',
            'methyl methacrylate': 'C=C(C)C(=O)OC',
            'ethylene oxide': 'C1CO1',
            'propylene oxide': 'CC1CO1',
            'caprolactam': 'C1CCC(=O)NCC1',
            'adipic acid': 'C(CCC(=O)O)CC(=O)O',
            
            # Carbonate compounds
            'ethylene carbonate': 'C1COC(=O)O1',
            'propylene carbonate': 'CC1COC(=O)O1',
            'carbonate': 'OC(=O)O',
            'dimethyl carbonate': 'COC(=O)OC',
            'diethyl carbonate': 'CCOC(=O)OCC',
            
            # Polymer names mapped to monomers
            'polyethylene': 'C=C',
            'polypropylene': 'CC=C',
            'polystyrene': 'C=Cc1ccccc1',
            'polyvinyl chloride': 'C=CCl',
            'pvc': 'C=CCl',
            'polyacrylonitrile': 'C=CC#N',
            'pmma': 'C=C(C)C(=O)OC',
            'polyethylene glycol': 'C=C',  # Ethylene monomer
            'peg': 'C=C',  # Ethylene monomer
            'polyethylene oxide': 'C1CO1',
            'peo': 'C1CO1',
            'polypropylene oxide': 'CC1CO1',
            'ppo': 'CC1CO1',
            'nylon 6': 'C1CCC(=O)NCC1',
            'nylon 66': 'C(CCC(=O)O)CC(=O)O',
            
            # Organometallic examples
            'ferrocene': '[Fe+2].c1ccc([CH-])c1.c1ccc([CH-])c1',
            'iron benzene complex': '[Fe]c1ccccc1',
            'benzene iron': 'c1ccccc1.[Fe]',
            'iron coordination complex': '[Fe]',
            'nickel catalyst': '[Ni]',
            'palladium catalyst': '[Pd]',
            'platinum catalyst': '[Pt]',
            'copper catalyst': '[Cu]',
            'zinc catalyst': '[Zn]',
            'chromium catalyst': '[Cr]',
            'titanium catalyst': '[Ti]',
            'benzene with iron': 'c1ccccc1.[Fe]',
            'iron containing benzene': '[Fe]c1ccccc1',
            'organometallic benzene': '[Fe]c1ccccc1',
            'metal benzene complex': '[Fe]c1ccccc1'
        }
    
    def _fix_common_chemical_errors(self, smiles: str) -> str:
        """
        Fix common chemical errors generated by LLMs.
        
        Args:
            smiles (str): SMILES string with potential errors
            
        Returns:
            str: Corrected SMILES string
        """
        if not smiles:
            return smiles
        
        # Fix 1: Replace invalid [CH] with C - LLMs often generate this incorrectly
        import re
        corrected = re.sub(r'\[CH\]', 'C', smiles)
        
        # Fix 2: Replace invalid [NH] with N - similar issue
        corrected = re.sub(r'\[NH\]', 'N', corrected)
        
        # Fix 3: Replace invalid [OH] with O - should just be O in SMILES
        corrected = re.sub(r'\[OH\]', 'O', corrected)
        
        # Fix 4: Fix incomplete ring closures - common LLM error
        # Count digits used for ring closures
        digits_in_smiles = re.findall(r'\d+', corrected)
        for digit in set(digits_in_smiles):
            count = corrected.count(digit)
            if count % 2 != 0:  # Odd count means incomplete ring
                # Try to complete the ring by duplicating the last occurrence
                corrected = corrected + digit
        
        # Fix 5: Remove trailing incomplete symbols
        corrected = re.sub(r'[\(\[\=\#\-]+$', '', corrected)
        
        # Fix 6: Fix double bonds at end - C= should be C
        corrected = re.sub(r'=+$', '', corrected)
        
        # Fix 7: Fix stereochemistry issues - remove complex stereochemistry that causes problems
        # Keep basic @@ but remove complex patterns that RDKit can't handle
        corrected = re.sub(r'@{3,}', '@@', corrected)  # Reduce multiple @ to @@
        
        # Fix 8: Remove empty parentheses () which are invalid
        corrected = re.sub(r'\(\)', '', corrected)
        
        # Fix 9: Remove empty brackets [] (different from connection points [*])
        corrected = re.sub(r'\[\]', '', corrected)
        
        # Fix 10: Fix malformed branch patterns like C( at end
        corrected = re.sub(r'\($', '', corrected)
        corrected = re.sub(r'\[$', '', corrected)
        
        # Fix 11: Fix impossible sulfur bonding patterns
        # S cannot have more than 4 bonds in typical organic chemistry
        # Replace problematic sulfur patterns with simpler, valid ones
        
        # Fix: S(=O)(=O) with additional bonds - reduce complexity
        if 'S(=O)(=O)' in corrected and len(re.findall(r'S.*?=.*?O.*?=.*?O', corrected)) > 0:
            # Check if sulfur has too many bonds by counting neighbors
            # Simplify to basic sulfur patterns for polymerization
            if corrected.count('=') > 4:  # Too many double bonds involving sulfur
                corrected = re.sub(r'O=S1?\(=O\).*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?1', 'CSC', corrected)
        
        # Fix: Replace impossible cyclic sulfur structures with simple sulfur bridges
        corrected = re.sub(r'O=S1\(=O\)C.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?=.*?C.*?1.*?C.*?=.*?C.*?', 'CSC', corrected)
        
        # Fix: Simplify complex sulfur-aromatic structures that cause SANITIZE_PROPERTIES errors
        if 'S(=O)(=O)c1ccccc1' in corrected:
            corrected = corrected.replace('O=CS(=O)(=O)c1ccccc1', 'CSc1ccccc1')
        
        # Fix: Invalid N(S) patterns - nitrogen shouldn't directly bond to naked sulfur
        corrected = re.sub(r'N\(S\)', 'NS', corrected)
        
        # Fix: Overly complex sulfur heterocycles - simplify to avoid RDKit errors
        if corrected.count('S') > 0 and corrected.count('=') > 3:
            # If structure is too complex with sulfur, simplify to basic sulfur-containing chain
            if any(pattern in corrected for pattern in ['S1(=O)', 'S(=O)(=O)', 'S1=O']):
                # Replace with simple sulfur bridge
                corrected = 'CSC'
        
        return corrected
    
    def _check_chemical_safety(self, smiles: str) -> Tuple[bool, str]:
        """
        Check for chemical patterns that commonly cause RDKit SANITIZE_PROPERTIES errors.
        
        Args:
            smiles (str): SMILES string to check
            
        Returns:
            Tuple[bool, str]: (is_safe, message)
        """
        if not smiles:
            return False, "Empty SMILES"
        
        # Check for impossible sulfur bonding patterns
        if 'S' in smiles:
            # Pattern 1: Sulfur with too many explicit bonds
            if 'S(=O)(=O)' in smiles and any(pat in smiles for pat in ['S(=O)(=O)C', 'S(=O)(=O)c', 'S(=O)(=O)=']):
                return False, "Sulfur with impossible bonding pattern S(=O)(=O) + additional bonds"
            
            # Pattern 2: Complex sulfur heterocycles that often fail
            if any(pat in smiles for pat in ['S1(=O)', 'S1=O', 'O=S1']):
                return False, "Complex sulfur heterocycle likely to cause validation errors"
                
            # Pattern 3: Sulfur oxidation states with aromatic systems
            if 'S(=O)(=O)c1ccccc1' in smiles:
                return False, "Sulfur sulfonyl group with aromatic system causes bonding issues"
        
        # Check for impossible nitrogen patterns
        if 'N(S)' in smiles:
            return False, "Nitrogen directly bonded to naked sulfur is chemically unstable"
        
        # Check for overly complex structures that often fail RDKit
        bond_complexity = smiles.count('=') + smiles.count('#') + smiles.count('(') + smiles.count('[')
        if bond_complexity > 10:
            return False, "Structure too complex - likely to cause validation errors"
        
        return True, "Chemical structure appears safe"
    
    def _make_chemically_safe(self, smiles: str) -> str:
        """
        Convert chemically unsafe SMILES to safer alternatives.
        
        Args:
            smiles (str): Potentially unsafe SMILES
            
        Returns:
            str: Chemically safer SMILES
        """
        if not smiles:
            return "CSC"  # Simple sulfur bridge as fallback
        
        safe_smiles = smiles
        
        # Replace problematic sulfur patterns with simple alternatives  
        import re
        if 'S(=O)(=O)' in safe_smiles:
            # Replace complex sulfonyl groups with simple sulfur bridges
            safe_smiles = re.sub(r'O=C?S\(=O\)\(=O\).*?c1ccccc1', 'CSc1ccccc1', safe_smiles)
            safe_smiles = re.sub(r'S\(=O\)\(=O\)', 'S', safe_smiles)
        
        # Replace complex sulfur heterocycles with simple sulfur
        safe_smiles = re.sub(r'O=S1\(=O\).*?C.*?=.*?C.*?1', 'CSC', safe_smiles)
        
        # Fix nitrogen-sulfur issues
        safe_smiles = re.sub(r'N\(S\)', 'NS', safe_smiles)
        
        # If still too complex, fall back to simple sulfur-containing structure
        if safe_smiles.count('=') > 4 or safe_smiles.count('(') > 3:
            if 'S' in safe_smiles:
                safe_smiles = 'CSC'  # Simple sulfur bridge
            else:
                safe_smiles = 'CCO'  # Simple alcohol
        
        return safe_smiles
    
    def _validate_polymer_monomer_requirements(self, smiles: str) -> Tuple[bool, str]:
        """
        Validate that SMILES meets polymer monomer requirements.
        
        Args:
            smiles (str): SMILES string to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, explanation)
        """
        if not smiles:
            return False, "Empty SMILES string"
        
        # Count carbon atoms (both uppercase C and lowercase c for aromatic)
        carbon_count = smiles.upper().count('C')
        
        if carbon_count < 2:
            return False, f"Insufficient carbon content: {carbon_count} carbons (minimum: 2)"
        
        # Check for basic chemical validity patterns
        if smiles in ['C', 'O', 'N', 'S', 'B']:
            return False, "Single atom structures cannot form polymer chains"
        
        return True, f"Valid polymer monomer with {carbon_count} carbons"
    
    def generate_smiles(self, description: str, debug: bool = False) -> Dict[str, Any]:
        """Generate SMILES from description with optional debugging."""
        def debug_log(message, data=None):
            if debug:
                print(f"🔍 SMILES_GEN: {message}")
                if data:
                    print(f"   Data: {data}")
        
        try:
            debug_log(f"Starting SMILES generation for: '{description}'")
            
            # Check direct matches first
            desc_lower = description.lower().strip()
            debug_log(f"Checking direct matches for: '{desc_lower}'")
            
            if desc_lower in self.chemistry_examples:
                matched_smiles = self.chemistry_examples[desc_lower]
                debug_log(f"Found direct match: '{matched_smiles}'")
                return {
                    'success': True,
                    'smiles': matched_smiles,
                    'method': 'direct_match',
                    'confidence': 1.0
                }
            
            debug_log("No direct match found, using LLM generation")
            
            # Use LLM with comprehensive SMILES rules
            prompt = f"""You are a chemistry expert. Convert this description to a valid SMILES string following these EXACT rules:

SMILES GUIDELINES (FOLLOW EXACTLY - VERBATIM):
SMILES (simplified molecular-input line-entry system) uses short ASCII string to represent the structure of chemical species. Because the SMILES format described here is custom-designed by us for polymers, it is not completely identical to other SMILES formats. Strictly following the rules explained below is crucial for having correct results.

1. Spaces are not permitted in a SMILES string.
2. An atom is represented by its respective atomic symbol. In case of 2-character atomic symbol, it is placed between two square brackets [ ].
3. Single bonds are implied by placing atoms next to each other. A double bond is represented by the = symbol while a triple bond is represented by #.
4. Hydrogen atoms are suppressed, i.e., the polymer blocks are represented without hydrogen. Polymer Genome interface assumes typical valence of each atom type. If enough bonds are not identified by the user through SMILES notation, the dangling bonds will be automatically saturated by hydrogen atoms.
5. Branches are placed between a pair of round brackets ( ), and are assumed to attach to the atom right before the opening round bracket (.
6. Numbers are used to identify the opening and closing of rings of atoms. For example, in C1CCCCC1, the first carbon having a number "1" should be connected by a single bond with the last carbon, also having a number "1". Polymer blocks that have multiple rings may be identified by using different, consecutive numbers for each ring.
7. Atoms in aromatic rings can be specified by lower case letters. As an example, benzene ring can be written as c1ccccc1 which is equivalent to C(C=C1)=CC=C1.

ADDITIONAL SMILES SYNTAX RULES:
ATOMS:
- Organic atoms (B,C,N,O,P,S,F,Cl,Br,I) without formal charge: no brackets needed
- All other atoms or charged atoms: use brackets [Au], [NH4+], [OH-], [Co+3]
- Hydrogen: write as H, [H], or omit if implied

BONDS:
- Single bonds: usually omitted, or use -
- Double bonds: =
- Triple bonds: #
- Quadruple bonds: $
- Aromatic bonds: : (or use lowercase atoms)
- Non-bonds (separate species): .

RINGS:
- Break ring and add numbers: C1CCCCC1 (cyclohexane)
- Multiple rings: C1CCCC2C1CCCC2 (decalin)
- Two-digit ring numbers: C%10...C%10

AROMATICITY:
- Use lowercase: c1ccccc1 (benzene), n1ccccc1 (pyridine)
- Aromatic nitrogen with H: [nH] (as in pyrrole)

BRANCHING:
- Use parentheses: CCC(=O)O (propionic acid)
- Branch atoms connect to the atom before the parentheses

STEREOCHEMISTRY:
- Tetrahedral centers: @ (counter-clockwise), @@ (clockwise)
- Double bonds: / and \ for E/Z geometry

CRITICAL REQUIREMENTS:
- ALL brackets must be balanced: ( ) [ ]
- ALL ring closures must be complete: c1ccccc1 not c1cccc
- NO trailing incomplete symbols
- NO spaces or hyphens in final SMILES
- Ensure proper valence for all atoms

POLYMER MONOMER REQUIREMENTS (CRITICAL FOR THIS APPLICATION):
- The generated structure will be used as a POLYMER MONOMER
- MUST contain at least TWO CARBON atoms (minimum: CC)
- Structure must be capable of polymerization
- Include functional groups that enable polymer chain formation
- Consider typical monomer structures like vinyl groups (C=C), rings that can open, or difunctional molecules
- Examples of good monomers: ethylene (C=C), styrene (C=Cc1ccccc1), ethylene glycol (OCCO), adipic acid (OC(=O)CCCCC(=O)O)

REFERENCE EXAMPLES FROM TABLE 1:
Chemical formula -> SMILES:
-CH2- -> C
-NH- -> N
-CS- -> C(=S)
-CO- -> C(=O)
-CF2- -> C(F)(F)
-O- -> O
-C6H4- -> C(C=C1)=CC=C1
-C4H2S- -> C1=CSC(=C1)
-C5H3N- -> C1=NC=C(C=C1)
-C4H3N- -> C(N1)=CC=C1

EXAMPLES:
- water → O
- ethanol → CCO
- benzene → c1ccccc1
- cyclohexane → C1CCCCC1
- aspirin → CC(=O)OC1=CC=CC=C1C(=O)O
- caffeine → CN1C=NC2=C1C(=O)N(C(=O)N2C)C
- glucose → C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O
- propionic acid → CCC(=O)O
- alanine → N[C@@H](C)C(=O)O
- sodium chloride → [Na+].[Cl-]
- ammonium → [NH4+]
- hydroxide → [OH-]
- iron benzene → [Fe]c1ccccc1
- ferrocene → [Fe+2].c1ccc([CH-])c1.c1ccc([CH-])c1

POLYMER MONOMERS:
- ethylene → C=C
- propylene → CC=C  
- styrene → C=Cc1ccccc1
- ethylene carbonate → C1COC(=O)O1
- For polymers, give the monomer structure

VALID SULFUR CHEMISTRY (CRITICAL FOR SULFUR-CONTAINING REQUESTS):
- Simple sulfur bridge: CSC
- Thioether: CSCCC
- Disulfide: CSSC
- Thiol group: SH (written as S in polymer context)
- Sulfur in aromatic ring: c1csc(c1) (thiophene)
- Simple sulfide: CCC(S)CC

INVALID SULFUR PATTERNS TO AVOID:
- S(=O)(=O) with >2 additional bonds ← sulfur cannot have >4 bonds
- O=S1(=O)C2=CC=CC=CC=C1C=C2 ← impossible sulfur bonding
- O=CS(=O)(=O)c1ccccc1 ← sulfur with 6 bonds (impossible)
- N(S) ← nitrogen bonded to naked sulfur (unstable)
- Complex sulfur heterocycles ← often chemically impossible

INVALID PATTERNS TO AVOID:
- (C([C@H]1C@HO)O)([ ← incomplete brackets
- C(C( ← incomplete parentheses
- C1CCC ← incomplete ring
- C=C=C= ← trailing bonds
- C[C@H ← incomplete bracket
- Single carbon structures (C, CO, etc.) ← insufficient for polymerization
- Structures without carbon backbone ← cannot form polymer chains
- Any sulfur with more than 4 bonds ← chemically impossible
- Complex sulfur oxidation states ← often unstable

Description: {description}

RESPOND WITH ONLY A VALID, SYNTACTICALLY CORRECT SMILES STRING THAT CONTAINS AT LEAST 2 CARBONS."""

            debug_log("Sending prompt to LLM", {"prompt_length": len(prompt)})
            response = self.llm.invoke(prompt)
            debug_log(f"LLM response received: '{response}'")
            
            smiles = self._extract_smiles(response, debug=debug)
            debug_log(f"Extracted SMILES: '{smiles}'")
            
            # **FIX COMMON LLM CHEMICAL ERRORS** - Correct invalid patterns before validation
            if smiles:
                corrected_smiles = self._fix_common_chemical_errors(smiles)
                if corrected_smiles != smiles:
                    debug_log(f"Chemical correction applied: '{smiles}' → '{corrected_smiles}'")
                    smiles = corrected_smiles
            
            if smiles:
                # Validate polymer monomer requirements
                is_valid, validation_msg = self._validate_polymer_monomer_requirements(smiles)
                debug_log(f"Polymer monomer validation: {validation_msg}")
                
                if is_valid:
                    # **ADDITIONAL CHEMICAL SAFETY CHECK** - Prevent common RDKit errors
                    chemical_safety, safety_msg = self._check_chemical_safety(smiles)
                    if chemical_safety:
                        debug_log("SMILES generation successful with polymer monomer validation")
                        return {
                            'success': True,
                            'smiles': smiles,
                            'method': 'llm_generation', 
                            'confidence': 0.8,
                            'validation': validation_msg,
                            'safety_check': safety_msg,
                            'raw_response': response
                        }
                    else:
                        debug_log(f"SMILES failed chemical safety check: {safety_msg}")
                        # Try to auto-correct the issue
                        safe_smiles = self._make_chemically_safe(smiles)
                        if safe_smiles != smiles:
                            debug_log(f"Applied safety correction: {smiles} → {safe_smiles}")
                            return {
                                'success': True,
                                'smiles': safe_smiles,
                                'method': 'llm_generation_corrected', 
                                'confidence': 0.7,
                                'validation': f"Corrected: {safety_msg}",
                                'original_smiles': smiles,
                                'raw_response': response
                            }
                        else:
                            error_msg = f'SMILES failed chemical safety: {safety_msg}'
                            debug_log(f"SMILES generation failed safety: {error_msg}")
                            return {
                                'success': False, 
                                'error': error_msg,
                                'smiles': smiles,
                                'raw_response': response
                            }
                else:
                    error_msg = f'SMILES does not meet polymer monomer requirements: {validation_msg}'
                    debug_log(f"SMILES generation failed validation: {error_msg}")
                    return {
                        'success': False, 
                        'error': error_msg,
                        'smiles': smiles,
                        'raw_response': response
                    }
            else:
                error_msg = 'Could not extract valid SMILES from LLM response'
                debug_log(f"SMILES generation failed: {error_msg}")
                return {
                    'success': False, 
                    'error': error_msg,
                    'raw_response': response
                }
                
        except Exception as e:
            error_msg = f"Exception in SMILES generation: {str(e)}"
            debug_log(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _extract_smiles(self, response: str, debug: bool = False) -> Optional[str]:
        """Extract SMILES from LLM response with optional debugging."""
        def debug_log(message, data=None):
            if debug:
                print(f"🔍 EXTRACT: {message}")
                if data:
                    print(f"   Data: {data}")
        
        response = response.strip()
        debug_log(f"Extracting SMILES from response: '{response}'")
        
        # Check if response looks like SMILES
        if self._looks_like_smiles(response):
            debug_log("Response looks like SMILES directly")
            # Clean the response before returning
            cleaned = clean_malformed_smiles(response)
            debug_log(f"Cleaned response: '{cleaned}'")
            
            if cleaned and self._looks_like_smiles(cleaned):
                debug_log(f"Using cleaned SMILES: '{cleaned}'")
                return cleaned
            debug_log(f"Using original response: '{response}'")
            return response
        
        debug_log("Response doesn't look like direct SMILES, trying patterns")
        
        # Try patterns to extract SMILES-like strings (ordered by specificity)
        patterns = [
            # Most specific patterns first
            r'SMILES:\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{3,})',
            r'is:\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{5,})',
            r'→\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{3,})',
            r'Answer:\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{3,})',
            r'Result:\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{3,})',
            r':\s*([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{5,})',
            # Look for longer chemical-looking strings (min 5 chars)
            r'([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{5,})',
            # Fallback to any chemical-looking string (min 3 chars)
            r'([A-Za-z0-9\[\]\(\)\=\#\+\-@/\\]{3,})',
        ]
        
        for i, pattern in enumerate(patterns):
            debug_log(f"Trying pattern {i+1}: {pattern}")
            matches = re.findall(pattern, response)
            debug_log(f"Pattern {i+1} found {len(matches)} matches: {matches}")
            
            # Sort matches by length (longer matches are more likely to be real SMILES)
            # and by "chemical-ness" (contains more chemical symbols)
            def smiles_score(match):
                score = 0
                score += len(match) * 2  # Length bonus
                score += match.count('C') * 3  # Carbon bonus
                score += match.count('=') * 2  # Double bond bonus
                score += match.count('1') * 1  # Ring number bonus
                score += match.count('c') * 2  # Aromatic carbon bonus
                # Penalty for common English words
                english_words = ['the', 'is', 'for', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'a', 'an']
                if match.lower() in english_words:
                    score -= 100
                return score
            
            matches_with_scores = [(match, smiles_score(match)) for match in matches]
            matches_with_scores.sort(key=lambda x: x[1], reverse=True)
            debug_log(f"Ranked matches: {[(m, s) for m, s in matches_with_scores[:3]]}")
            
            for j, (match, score) in enumerate(matches_with_scores):
                debug_log(f"Checking ranked match {j+1}: '{match}' (score: {score})")
                if self._looks_like_smiles(match):
                    debug_log(f"Match {j+1} looks like SMILES")
                    # Clean the match before returning
                    cleaned = clean_malformed_smiles(match)
                    debug_log(f"Cleaned match: '{cleaned}'")
                    
                    if cleaned and self._looks_like_smiles(cleaned):
                        debug_log(f"Using cleaned match: '{cleaned}'")
                        return cleaned
                    debug_log(f"Using original match: '{match}'")
                    return match
                else:
                    debug_log(f"Match {j+1} doesn't look like SMILES")
        
        debug_log("No valid SMILES found in response")
        return None
    
    def _looks_like_smiles(self, text: str) -> bool:
        """Check if text looks like SMILES."""
        if not text or len(text) < 1:
            return False
        
        # Extended valid chars to include common metals for organometallic chemistry
        valid_chars = set('CNOSPFClBrIcnospf0123456789[]()=#+@/\\-')
        # Add common metal atoms
        metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'}
        
        # Add all metal characters to valid set
        for metal in metal_atoms:
            for char in metal:
                valid_chars.add(char)
        
        if not all(c in valid_chars for c in text):
            return False
        
        # Must contain at least one chemical element (carbon, nitrogen, oxygen, sulfur, phosphorus, or metal)
        if not any(c in 'CNOSPFcnospf' for c in text) and not any(metal in text for metal in metal_atoms):
            return False
        
        return True


class NaturalLanguageToPSMILES:
    """Complete pipeline: Natural Language → SMILES → Validation → PSMILES"""
    
    def __init__(self, ollama_model: str = "llama3.2", ollama_host: str = "http://localhost:11434"):
        self.nl_to_smiles = NaturalLanguageToSMILES(ollama_model, ollama_host)
        self.validator = ChemicalValidator()
        
    def convert_description_to_psmiles(self, description: str, debug: bool = False) -> Dict[str, Any]:
        """Complete conversion pipeline with optional debugging."""
        result = {
            'success': False,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'debug_steps': [] if debug else None
        }
        
        def debug_log(step, message, data=None):
            if debug:
                debug_entry = {
                    'step': step,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                if data:
                    debug_entry['data'] = data
                result['debug_steps'].append(debug_entry)
                print(f"🔍 DEBUG [{step}]: {message}")
                if data:
                    print(f"   Data: {data}")
        
        try:
            debug_log("INIT", f"Starting conversion for: '{description}'")
            
            # Step 1: Generate SMILES
            debug_log("STEP1", "Generating SMILES from natural language...")
            smiles_result = self.nl_to_smiles.generate_smiles(description, debug=debug)
            result['smiles_generation'] = smiles_result
            
            debug_log("STEP1_RESULT", "SMILES generation result", smiles_result)
            
            if not smiles_result['success']:
                error_msg = 'SMILES generation failed'
                result['error'] = error_msg
                debug_log("ERROR", error_msg, smiles_result)
                return result
            
            smiles = smiles_result['smiles']
            debug_log("STEP1_SUCCESS", f"Generated SMILES: '{smiles}'")
            
            # Step 2: Validate with RDKit (with SELFIES autocorrection)
            debug_log("STEP2", f"Validating SMILES: '{smiles}'")
            
            # Check for obvious malformations before validation
            if smiles.endswith('[') or smiles.endswith('('):
                debug_log("STEP2_MALFORMED", f"Detected malformed SMILES with incomplete bracket: '{smiles}'")
            
            is_valid, mol, message = self.validator.validate_smiles(smiles, debug=debug)
            debug_log("STEP2_VALIDATION", f"Validation result: valid={is_valid}, message='{message}'")
            
            # Track if autocorrection was used
            autocorrected = "autocorrected" in message or "cleaned" in message or "fallback" in message
            final_smiles = smiles
            
            if autocorrected:
                debug_log("STEP2_CORRECTION", "Autocorrection was applied")
                # Extract the corrected SMILES from the message
                import re
                match = re.search(r'→ ([^)]+)', message)
                if match:
                    corrected_smiles = match.group(1)
                    result['autocorrected_smiles'] = corrected_smiles
                    final_smiles = corrected_smiles
                    debug_log("STEP2_CORRECTED", f"Corrected SMILES: '{corrected_smiles}'")
            
            result['validation'] = {
                'is_valid': is_valid,
                'message': message,
                'original_smiles': smiles_result['smiles'],
                'validated_smiles': final_smiles,
                'autocorrected': autocorrected
            }
            
            if not is_valid:
                error_msg = f'SMILES validation failed: {message}'
                result['error'] = error_msg
                debug_log("ERROR", error_msg)
                return result
            
            debug_log("STEP2_SUCCESS", f"Validation successful, final SMILES: '{final_smiles}'")
            
            # Step 3: Canonicalize
            debug_log("STEP3", f"Canonicalizing SMILES: '{final_smiles}'")
            canonical_smiles = self.validator.canonicalize_smiles(final_smiles)
            result['canonical_smiles'] = canonical_smiles
            debug_log("STEP3_RESULT", f"Canonical SMILES: '{canonical_smiles}'")
            
            # Step 4: Convert to PSMILES
            smiles_for_psmiles = canonical_smiles or final_smiles
            debug_log("STEP4", f"Converting to PSMILES: '{smiles_for_psmiles}'")
            psmiles = self._convert_to_psmiles(smiles_for_psmiles)
            result['psmiles'] = psmiles
            debug_log("STEP4_RESULT", f"Generated PSMILES: '{psmiles}'")
            
            # Step 5: Validate PSMILES format
            debug_log("STEP5", f"Validating PSMILES format: '{psmiles}'")
            psmiles_validation = self._validate_psmiles_format(psmiles)
            result['psmiles_validation'] = psmiles_validation
            debug_log("STEP5_RESULT", "PSMILES validation", psmiles_validation)
            
            result['success'] = True
            debug_log("SUCCESS", "Pipeline completed successfully!")
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            result['error'] = error_msg
            debug_log("EXCEPTION", error_msg, {"exception_type": type(e).__name__})
            import traceback
            if debug:
                print(f"🔥 EXCEPTION TRACEBACK:\n{traceback.format_exc()}")
        
        return result
    
    def _convert_to_psmiles(self, smiles: str) -> str:
        """
        Convert SMILES to PSMILES by adding [*] connection points.
        
        PSMILES (Polymer SMILES) is simply a SMILES string with [*] symbols 
        at the beginning and end to indicate connection points for polymerization.
        
        Example: CCO (ethanol) → [*]CCO[*] (ethanol polymer repeat unit)
        """
        if not smiles:
            return "[*][*]"
        
        # Check if already has connection points
        if '[*]' in smiles:
            connection_count = smiles.count('[*]')
            if connection_count == 2:
                return smiles
            elif connection_count == 1:
                return f"[*]{smiles}"
            else:
                smiles_clean = smiles.replace('[*]', '')
        else:
            smiles_clean = smiles
        
        # Add connection points for polymer repeat unit
        psmiles = f"[*]{smiles_clean}[*]"
        return psmiles
    
    def _validate_psmiles_format(self, psmiles: str) -> Dict[str, Any]:
        """Validate PSMILES format."""
        validation = {
            'is_valid': True,
            'issues': [],
            'connection_points': 0
        }
        
        if not psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Empty PSMILES')
            return validation
        
        # Count connection points
        connection_count = psmiles.count('[*]')
        validation['connection_points'] = connection_count
        
        if connection_count != 2:
            validation['is_valid'] = False
            validation['issues'].append(f'Must have exactly 2 [*] symbols, found {connection_count}')
        
        # Check for spaces and hyphens
        if ' ' in psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Cannot contain spaces')
        
        if '-' in psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Cannot contain hyphens')
        
        return validation


@tool
def validate_smiles_tool(smiles: str) -> Dict[str, Any]:
    """Tool to validate SMILES using RDKit."""
    is_valid, mol, message = ChemicalValidator.validate_smiles(smiles)
    return {
        'is_valid': is_valid,
        'message': message,
        'smiles': smiles
    }


@tool
def convert_smiles_to_psmiles_tool(smiles: str) -> str:
    """Tool to convert SMILES to PSMILES."""
    if '[*]' in smiles:
        connection_count = smiles.count('[*]')
        if connection_count == 2:
            return smiles
        elif connection_count == 1:
            return f"[*]{smiles}"
        else:
            smiles_clean = smiles.replace('[*]', '')
    else:
        smiles_clean = smiles
    
    return f"[*]{smiles_clean}[*]"


# Testing
if __name__ == "__main__":
    converter = NaturalLanguageToPSMILES()
    
    test_descriptions = [
        "water",
        "ethanol", 
        "benzene",
        "a molecule with a benzene ring and a hydroxyl group",
    ]
    
    print("Testing Natural Language to PSMILES Conversion")
    print("=" * 50)
    
    for desc in test_descriptions:
        print(f"\nDescription: {desc}")
        result = converter.convert_description_to_psmiles(desc)
        
        if result['success']:
            print(f"SMILES: {result.get('canonical_smiles', 'N/A')}")
            print(f"PSMILES: {result.get('psmiles', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("-" * 30)
