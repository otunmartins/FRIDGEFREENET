"""
SMILES Self-Correcting System with Error Feedback Loop

This module implements a self-correcting system that uses error tracebacks
in a feedback loop with prompt engineering using LLM agents. It loops up to 
100 times until the SMILES string is valid, attempting automatic repairs
using the full SMILES specification rules.
"""

import re
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import requests
    import json
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Full SMILES specification rules (verbatim from the tutorial)
SMILES_RULES = """
SMILES Tutorial - Complete Specification Rules

SMILES (Simplified Molecular Input Line Entry System) is a chemical notation that allows a user to represent a chemical structure in a way that can be used by the computer. SMILES is an easily learned and flexible notation.

SMILES has five basic syntax rules which must be observed:

Rule One: Atoms and Bonds
- SMILES supports all elements in the periodic table
- An atom is represented using its respective atomic symbol
- Upper case letters refer to non-aromatic atoms; lower case letters refer to aromatic atoms
- If the atomic symbol has more than one letter the second letter must be lower case

Bond symbols:
- (no symbol) = Single bond (default)
= = Double bond
# = Triple bond
* = Aromatic bond (rarely used explicitly)
. = Disconnected structures

Examples:
- CC = CH3CH3 (Ethane)
- C=C = CH2CH2 (Ethene) 
- CBr = CH3Br (Bromomethane)
- C#N = C≡N (Hydrocyanic acid)
- Na.Cl = NaCl (Sodium chloride)

Rule Two: Simple Chains
- Structures are hydrogen-suppressed (hydrogens not shown explicitly)
- System automatically assumes other connections are satisfied by hydrogen bonds
- Can explicitly identify hydrogens: HC(H)=C(H)(H) for Ethene

Rule Three: Branches
- A branch from a chain is specified by placing the SMILES symbol(s) for the branch between parentheses
- The string in parentheses is placed directly after the symbol for the atom to which it is connected
- If connected by double or triple bond, the bond symbol immediately follows the left parenthesis

Examples:
- CC(O)C = 2-Propanol
- CC(=O)C = 2-Propanone
- CC(CC)C = 2-Methylbutane
- CC(C)(C)CC = 2,2-Dimethylbutane

Rule Four: Rings
- Ring structures identified by using numbers to identify opening and closing ring atom
- Same number connects atoms in ring closure
- Different numbers for multiple rings
- Bond symbol placed before ring closure number if needed

Examples:
- C1CCCCC1 = Cyclohexane
- C=1CCCCC1 = Cyclohexene  
- c1ccccc1 = Benzene
- c1cc2ccccc2cc1 = Naphthalene

Rule Five: Charged Atoms
- Charges override built-in valence knowledge
- Format: atom followed by brackets enclosing charge
- Charge number may be explicit [+1] or implicit [+]

Examples:
- CCC(=O)O[-1] = Ionized propanoic acid
- c1ccccn[+1]1CC(=O)O = 1-Carboxylmethyl pyridinium

Critical Validation Rules:
1. Balanced parentheses for branches
2. Balanced brackets for charges/atoms
3. Valid ring closure numbers (must be paired)
4. Valid atomic symbols (case-sensitive)
5. No impossible valencies
6. Proper bond syntax
7. No orphaned ring numbers
"""

@dataclass
class SMILESError:
    """Container for SMILES validation errors"""
    error_type: str
    error_message: str
    position: Optional[int]
    suggested_fix: Optional[str]
    severity: str  # 'critical', 'warning', 'info'

@dataclass
class CorrectionAttempt:
    """Container for correction attempt results"""
    attempt_number: int
    original_smiles: str
    corrected_smiles: str
    errors_found: List[SMILESError]
    correction_method: str
    success: bool
    llm_reasoning: Optional[str]

class SMILESSelfCorrector:
    """
    Self-correcting SMILES system using error feedback loops and LLM agents
    """
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 ollama_model: str = "llama3.2",
                 max_iterations: int = 100,
                 debug: bool = False):
        """
        Initialize the SMILES self-corrector
        
        Args:
            ollama_host: Ollama server URL
            ollama_model: Model name to use
            max_iterations: Maximum correction attempts (default: 100)
            debug: Enable debug logging
        """
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.max_iterations = max_iterations
        self.debug = debug
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Correction history
        self.correction_history: List[CorrectionAttempt] = []
        
        # Validate dependencies
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES validation")
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests is required for LLM communication")
    
    def validate_smiles(self, smiles: str) -> Tuple[bool, List[SMILESError]]:
        """
        Validate SMILES string and capture detailed error information
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Basic syntax validation first
            syntax_errors = self._check_syntax_errors(smiles)
            errors.extend(syntax_errors)
            
            # RDKit validation
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                # Try to get more specific error information
                rdkit_errors = self._extract_rdkit_errors(smiles)
                errors.extend(rdkit_errors)
                return False, errors
            
            # Additional chemical validity checks
            validity_errors = self._check_chemical_validity(mol, smiles)
            errors.extend(validity_errors)
            
            # If we have any critical errors, it's invalid
            critical_errors = [e for e in errors if e.severity == 'critical']
            
            return len(critical_errors) == 0, errors
            
        except Exception as e:
            errors.append(SMILESError(
                error_type="validation_exception",
                error_message=f"Validation failed with exception: {str(e)}",
                position=None,
                suggested_fix="Check SMILES syntax",
                severity="critical"
            ))
            return False, errors
    
    def _check_syntax_errors(self, smiles: str) -> List[SMILESError]:
        """Check for basic SMILES syntax errors"""
        errors = []
        
        # Check balanced parentheses
        paren_count = 0
        for i, char in enumerate(smiles):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    errors.append(SMILESError(
                        error_type="unbalanced_parentheses",
                        error_message="Closing parenthesis without opening parenthesis",
                        position=i,
                        suggested_fix="Remove extra ')' or add missing '('",
                        severity="critical"
                    ))
        
        if paren_count > 0:
            errors.append(SMILESError(
                error_type="unbalanced_parentheses", 
                error_message=f"Missing {paren_count} closing parenthesis/parentheses",
                position=len(smiles),
                suggested_fix="Add missing ')' characters",
                severity="critical"
            ))
        
        # Check balanced brackets
        bracket_count = 0
        for i, char in enumerate(smiles):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count < 0:
                    errors.append(SMILESError(
                        error_type="unbalanced_brackets",
                        error_message="Closing bracket without opening bracket",
                        position=i,
                        suggested_fix="Remove extra ']' or add missing '['",
                        severity="critical"
                    ))
        
        if bracket_count > 0:
            errors.append(SMILESError(
                error_type="unbalanced_brackets",
                error_message=f"Missing {bracket_count} closing bracket(s)",
                position=len(smiles),
                suggested_fix="Add missing ']' characters",
                severity="critical"
            ))
        
        # Check ring closures
        ring_numbers = re.findall(r'(\d+)', smiles)
        ring_counts = {}
        for num in ring_numbers:
            ring_counts[num] = ring_counts.get(num, 0) + 1
        
        for ring_num, count in ring_counts.items():
            if count != 2:
                errors.append(SMILESError(
                    error_type="invalid_ring_closure",
                    error_message=f"Ring number {ring_num} appears {count} times (should be exactly 2)",
                    position=None,
                    suggested_fix=f"Ensure ring number {ring_num} appears exactly twice",
                    severity="critical"
                ))
        
        # Check for invalid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]+-=#.*/@%')
        for i, char in enumerate(smiles):
            if char not in valid_chars:
                errors.append(SMILESError(
                    error_type="invalid_character",
                    error_message=f"Invalid character '{char}' at position {i}",
                    position=i,
                    suggested_fix=f"Remove or replace invalid character '{char}'",
                    severity="critical"
                ))
        
        return errors
    
    def _extract_rdkit_errors(self, smiles: str) -> List[SMILESError]:
        """Extract specific error information from RDKit"""
        errors = []
        
        try:
            # Try to capture RDKit error messages
            import io
            import sys
            from contextlib import redirect_stderr
            
            # Capture stderr to get RDKit error messages
            error_capture = io.StringIO()
            with redirect_stderr(error_capture):
                mol = Chem.MolFromSmiles(smiles)
            
            error_output = error_capture.getvalue()
            
            if error_output:
                # Parse common RDKit error patterns
                if "Explicit valence" in error_output:
                    errors.append(SMILESError(
                        error_type="valence_error",
                        error_message="Explicit valence error - atom has too many/few bonds",
                        position=None,
                        suggested_fix="Check atom valencies and bond counts",
                        severity="critical"
                    ))
                
                if "ring closure" in error_output.lower():
                    errors.append(SMILESError(
                        error_type="ring_closure_error",
                        error_message="Invalid ring closure",
                        position=None,
                        suggested_fix="Check ring closure numbers are paired correctly",
                        severity="critical"
                    ))
                
                if "aromatic" in error_output.lower():
                    errors.append(SMILESError(
                        error_type="aromaticity_error",
                        error_message="Aromatic atom in non-aromatic environment",
                        position=None,
                        suggested_fix="Check aromatic atoms (lowercase) are in aromatic rings",
                        severity="critical"
                    ))
                
                # Generic RDKit error
                if not errors:
                    errors.append(SMILESError(
                        error_type="rdkit_parse_error",
                        error_message=f"RDKit parsing failed: {error_output.strip()}",
                        position=None,
                        suggested_fix="Check SMILES syntax according to SMILES rules",
                        severity="critical"
                    ))
            
        except Exception as e:
            self.logger.debug(f"Error extracting RDKit errors: {e}")
        
        return errors
    
    def _check_chemical_validity(self, mol, smiles: str) -> List[SMILESError]:
        """Check chemical validity of the molecule"""
        errors = []
        
        try:
            # Check for unrealistic molecular weight
            mw = Descriptors.MolWt(mol)
            if mw > 2000:
                errors.append(SMILESError(
                    error_type="high_molecular_weight",
                    error_message=f"Molecular weight {mw:.1f} is very high",
                    position=None,
                    suggested_fix="Consider if this molecular weight is realistic",
                    severity="warning"
                ))
            
            # Check number of atoms
            num_atoms = mol.GetNumAtoms()
            if num_atoms > 200:
                errors.append(SMILESError(
                    error_type="large_molecule",
                    error_message=f"Molecule has {num_atoms} atoms (very large)",
                    position=None,
                    suggested_fix="Consider if this molecule size is realistic",
                    severity="warning"
                ))
            
            # Check for common chemical validity
            # This could be expanded with more sophisticated checks
            
        except Exception as e:
            self.logger.debug(f"Error checking chemical validity: {e}")
        
        return errors
    
    def _query_llm(self, prompt: str) -> Optional[str]:
        """
        Query the LLM with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response or None if failed
        """
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for consistency
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                self.logger.error(f"LLM request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            return None
    
    def _create_correction_prompt(self, smiles: str, errors: List[SMILESError], attempt_number: int, original_request: Optional[str] = None) -> str:
        """Create a detailed correction prompt with SMILES rules and anti-repetition measures"""
        
        # Add variation strategies for higher attempt numbers
        temperature_instruction = ""
        if attempt_number > 3:
            temperature_instruction = "\n🔥 IMPORTANT: Previous attempts failed. Try a COMPLETELY DIFFERENT approach. Be more creative and consider alternative molecular structures."
        elif attempt_number > 6:
            temperature_instruction = "\n🚨 CRITICAL: Many attempts failed. Consider simplifying the molecule or using basic functional groups only. Avoid complex ring systems or charged species."
        
        error_details = "\n".join([
            f"- {error.error_type}: {error.error_message}" + 
            (f" (suggested fix: {error.suggested_fix})" if error.suggested_fix else "")
            for error in errors
        ])
        
        # Anti-repetition measures
        previous_attempts_warning = ""
        if attempt_number > 1:
            previous_attempts_warning = f"""
🔄 AVOID REPETITION: This is attempt #{attempt_number}. 
- DO NOT repeat previous failed attempts
- Try a completely different molecular structure
- Simplify if previous attempts were too complex
- Focus on basic, valid SMILES syntax
"""

        # **NEW: Include original user request as context to preserve intent**
        original_context = ""
        if original_request:
            original_context = f"""
🎯 ORIGINAL USER REQUEST: "{original_request}"
🔗 CRITICAL: While fixing syntax errors, PRESERVE the user's intent!
- If they asked for sulfur atoms, keep sulfur (S) in the corrected SMILES
- If they asked for specific functional groups, maintain those groups
- If they asked for aromatic compounds, keep aromatic rings
- Match the molecular requirements while fixing syntax only
"""

        prompt = f"""You are a SMILES notation expert. Fix the invalid SMILES string following these rules:

{original_context}

📋 SMILES RULES (CRITICAL - MUST FOLLOW):
1. Atoms: C, N, O, S, P, F, Cl, Br, I, B (uppercase for non-aromatic, lowercase for aromatic)
2. Bonds: - (single, default), = (double), # (triple), : (aromatic)
3. Branches: Use parentheses () for branches
4. Rings: Use numbers 1-9, each number must appear EXACTLY twice
5. Charges: Use [C+] or [N-] format (brackets required for charged atoms)
6. NO invalid characters like @, &, or standalone charges like [-1]

❌ INVALID SMILES: {smiles}

🐛 ERRORS FOUND:
{error_details}

{previous_attempts_warning}

{temperature_instruction}

✅ TASK: Return ONLY a valid SMILES string that fixes syntax errors while preserving the original intent. No explanation, no text, just the corrected SMILES.

EXAMPLES OF VALID FIXES:
- Ring error C1CCCC → C1CCCCC1 (complete the ring)
- Charge error CC(=O)O[-1] → CC(=O)[O-] (proper charge notation)
- Branch error C(C)C))C → C(C)C(C)C (fix unbalanced parentheses)
- Invalid char C@C&C → CCC (remove invalid characters)

CORRECTED SMILES:"""

        return prompt

    def _create_initial_generation_prompt(self, description: str) -> str:
        """Create a prompt for initial SMILES generation with strict format requirements"""
        
        prompt = f"""You are a molecular structure expert. Generate a SMILES string for: {description}

📋 SMILES RULES (MUST FOLLOW):
1. Atoms: Use correct atomic symbols (C, N, O, S, P, etc.)
2. Bonds: - (single, can omit), = (double), # (triple)  
3. Branches: Use parentheses () correctly
4. Rings: Each ring number (1-9) must appear EXACTLY twice
5. Charges: Use [atom+charge] format, e.g., [NH3+], [O-]
6. NO invalid syntax like CH3 (use CCC instead), [-1] charges, or @ symbols

🎯 REQUEST: {description}

✅ RESPOND WITH: Only the SMILES string, nothing else.

EXAMPLES:
- "alcohol" → CCO
- "benzene" → c1ccccc1  
- "carboxylic acid" → CC(=O)O
- "amine" → CCN

SMILES:"""

        return prompt
    
    def correct_smiles(self, smiles: str, original_request: Optional[str] = None) -> Dict[str, Any]:
        """
        Main correction method - iteratively correct SMILES until valid
        
        Args:
            smiles: Invalid SMILES string to correct
            original_request: Original user request/description to preserve intent
            
        Returns:
            Dictionary with correction results
        """
        self.correction_history = []
        original_smiles = smiles
        current_smiles = smiles
        
        for attempt in range(1, self.max_iterations + 1):
            self.logger.debug(f"Correction attempt {attempt}/{self.max_iterations}")
            
            # Validate current SMILES
            is_valid, errors = self.validate_smiles(current_smiles)
            
            if is_valid:
                self.logger.info(f"SMILES corrected successfully in {attempt} attempts")
                return {
                    'success': True,
                    'original_smiles': original_smiles,
                    'corrected_smiles': current_smiles,
                    'attempts': attempt,
                    'correction_history': self.correction_history,
                    'final_errors': errors,  # May contain warnings
                    'original_request': original_request
                }
            
            # Create correction prompt with original request context
            prompt = self._create_correction_prompt(current_smiles, errors, attempt, original_request)
            
            # Query LLM for correction
            llm_response = self._query_llm(prompt)
            
            if llm_response is None:
                self.logger.error(f"LLM query failed on attempt {attempt}")
                break
            
            # Extract corrected SMILES from response
            corrected_smiles = self._extract_smiles_from_response(llm_response)
            
            # Record this attempt
            correction_attempt = CorrectionAttempt(
                attempt_number=attempt,
                original_smiles=current_smiles,
                corrected_smiles=corrected_smiles,
                errors_found=errors,
                correction_method="llm_feedback_loop",
                success=False,  # Will be updated if validation passes
                llm_reasoning=llm_response
            )
            
            self.correction_history.append(correction_attempt)
            
            # Check if correction actually changed anything
            if corrected_smiles == current_smiles:
                self.logger.warning(f"LLM returned same SMILES on attempt {attempt}")
                # Try basic automatic repair
                corrected_smiles = self._apply_basic_repairs(current_smiles, errors)
            
            current_smiles = corrected_smiles
            
            self.logger.debug(f"Attempt {attempt}: {original_smiles} → {corrected_smiles}")
        
        # Failed to correct after max iterations
        self.logger.error(f"Failed to correct SMILES after {self.max_iterations} attempts")
        return {
            'success': False,
            'original_smiles': original_smiles,
            'final_smiles': current_smiles,
            'attempts': self.max_iterations,
            'correction_history': self.correction_history,
            'error': f"Failed to correct after {self.max_iterations} attempts"
        }
    
    def generate_and_correct_smiles(self, description: str) -> Dict[str, Any]:
        """
        Generate SMILES from description and auto-correct if needed
        
        Args:
            description: Description of desired molecule
            
        Returns:
            Dictionary with generation and correction results
        """
        # Generate initial SMILES
        prompt = self._create_initial_generation_prompt(description)
        llm_response = self._query_llm(prompt)
        
        if llm_response is None:
            return {
                'success': False,
                'error': 'Failed to generate initial SMILES'
            }
        
        # Extract SMILES from response
        generated_smiles = self._extract_smiles_from_response(llm_response)
        
        # Validate and correct if needed
        is_valid, errors = self.validate_smiles(generated_smiles)
        
        if is_valid:
            return {
                'success': True,
                'description': description,
                'generated_smiles': generated_smiles,
                'corrected_smiles': generated_smiles,
                'correction_needed': False,
                'attempts': 1,
                'errors': errors  # May contain warnings
            }
        else:
            # Need correction - pass the original description to preserve user intent
            correction_result = self.correct_smiles(generated_smiles, description)
            
            result = {
                'success': correction_result['success'],
                'description': description,
                'generated_smiles': generated_smiles,
                'correction_needed': True,
                'attempts': correction_result.get('attempts', 0),
                'correction_history': correction_result.get('correction_history', [])
            }
            
            if correction_result['success']:
                result['corrected_smiles'] = correction_result['corrected_smiles']
                result['final_errors'] = correction_result.get('final_errors', [])
            else:
                result['error'] = correction_result.get('error', 'Correction failed')
                result['final_smiles'] = correction_result.get('final_smiles', generated_smiles)
            
            return result
    
    def _extract_smiles_from_response(self, response: str) -> str:
        """Extract SMILES from LLM response with improved pattern matching"""
        if not response:
            return ""
        
        # Clean the response
        response = response.strip()
        
        # If the response is very short and looks like a SMILES, return it
        if len(response) < 100 and not any(word in response.lower() for word in ['based', 'the', 'this', 'here', 'is', 'molecule']):
            # Basic SMILES character check
            valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]=#+-.:')
            if all(c in valid_chars for c in response):
                return response
        
        # Try to find SMILES patterns in the response
        patterns = [
            r'^([A-Za-z0-9\(\)\[\]=#\+\-\.:]+)$',  # Full line is SMILES
            r'SMILES:?\s*([A-Za-z0-9\(\)\[\]=#\+\-\.:]+)',  # After "SMILES:"
            r'→\s*([A-Za-z0-9\(\)\[\]=#\+\-\.:]+)',  # After arrow
            r'corrected:?\s*([A-Za-z0-9\(\)\[\]=#\+\-\.:]+)',  # After "corrected"
            r'^[^A-Za-z]*([A-Za-z0-9\(\)\[\]=#\+\-\.:]{3,}).*$',  # SMILES at start
            r'([A-Za-z0-9\(\)\[\]=#\+\-\.:]{6,})',  # Any longish SMILES-like string
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip()
                # Additional validation - avoid obvious non-SMILES
                if not any(word in candidate.lower() for word in ['based', 'after', 'analyzing', 'error', 'becomes']):
                    return candidate
        
        # If no pattern matches, try to clean common prefixes
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 50:
                # Remove common prefixes
                for prefix in ['SMILES:', 'corrected:', 'fixed:', 'result:', '→', '->', 'answer:']:
                    if line.lower().startswith(prefix.lower()):
                        line = line[len(prefix):].strip()
                        break
                
                # Basic validation
                if len(line) > 2 and not any(word in line.lower() for word in ['based', 'the', 'this', 'here', 'is']):
                    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]=#+-.:')
                    if all(c in valid_chars for c in line):
                        return line
        
        # Fallback: return the first word that looks like SMILES
        words = response.split()
        for word in words:
            if len(word) > 3 and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]=#+-.:' for c in word):
                return word
        
        return response.strip()
    
    def _apply_basic_repairs(self, smiles: str, errors: List[SMILESError]) -> str:
        """Apply basic repair strategies for common SMILES errors"""
        repaired = smiles
        
        for error in errors:
            if error.error_type == "unbalanced_parentheses":
                # Try to balance parentheses by removing extras or adding missing ones
                open_count = repaired.count('(')
                close_count = repaired.count(')')
                
                if close_count > open_count:
                    # Remove extra closing parentheses from the end
                    excess = close_count - open_count
                    for _ in range(excess):
                        idx = repaired.rfind(')')
                        if idx != -1:
                            repaired = repaired[:idx] + repaired[idx+1:]
                elif open_count > close_count:
                    # Add missing closing parentheses at the end
                    missing = open_count - close_count
                    repaired += ')' * missing
            
            elif error.error_type == "unbalanced_brackets":
                # Fix unbalanced square brackets
                open_count = repaired.count('[')
                close_count = repaired.count(']')
                
                if close_count > open_count:
                    # Remove extra closing brackets
                    excess = close_count - open_count
                    for _ in range(excess):
                        idx = repaired.rfind(']')
                        if idx != -1:
                            repaired = repaired[:idx] + repaired[idx+1:]
                elif open_count > close_count:
                    # Add missing closing brackets
                    missing = open_count - close_count
                    repaired += ']' * missing
            
            elif error.error_type == "invalid_character":
                # Remove common invalid characters
                invalid_chars = ['@', '&', '$', '%', '^', '`', '~']
                for char in invalid_chars:
                    repaired = repaired.replace(char, '')
                
                # Fix common charge notation errors
                repaired = re.sub(r'\[-?\+?\d*\]', '', repaired)  # Remove standalone charges
                repaired = re.sub(r'O\[-\]', '[O-]', repaired)    # Fix O[-] to [O-]
                repaired = re.sub(r'N\[\+\]', '[NH3+]', repaired) # Fix N[+] to [NH3+]
            
            elif error.error_type == "invalid_ring_closure":
                # Simple ring closure fixes
                # Find incomplete rings and either complete them or remove ring numbers
                ring_numbers = re.findall(r'\d', repaired)
                ring_counts = {}
                for num in ring_numbers:
                    ring_counts[num] = ring_counts.get(num, 0) + 1
                
                # Remove ring numbers that appear only once
                for num, count in ring_counts.items():
                    if count == 1:
                        repaired = repaired.replace(num, '', 1)
                    elif count > 2:
                        # Remove extra occurrences
                        excess = count - 2
                        for _ in range(excess):
                            idx = repaired.rfind(num)
                            if idx != -1:
                                repaired = repaired[:idx] + repaired[idx+1:]
        
        # Final cleanup
        repaired = re.sub(r'\s+', '', repaired)  # Remove spaces
        repaired = re.sub(r'[()]+$', '', repaired)  # Remove trailing parentheses
        
        return repaired


def create_smiles_self_corrector(ollama_host: str = "http://localhost:11434",
                                ollama_model: str = "llama3.2",
                                max_iterations: int = 100,
                                debug: bool = False) -> SMILESSelfCorrector:
    """
    Factory function to create a SMILES self-corrector instance
    
    Args:
        ollama_host: Ollama server URL
        ollama_model: Model name to use
        max_iterations: Maximum correction attempts
        debug: Enable debug logging
        
    Returns:
        SMILESSelfCorrector instance
    """
    return SMILESSelfCorrector(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        max_iterations=max_iterations,
        debug=debug
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the self-corrector
    corrector = create_smiles_self_corrector(debug=True)
    
    # Test with an invalid SMILES
    test_smiles = "C(C(C)C))C"  # Unbalanced parentheses
    print(f"Testing correction of: {test_smiles}")
    
    result = corrector.correct_smiles(test_smiles)
    print(f"Result: {result}")
    
    # Test generation and correction
    test_description = "simple alcohol with 3 carbons"
    print(f"\nTesting generation of: {test_description}")
    
    result = corrector.generate_and_correct_smiles(test_description)
    print(f"Result: {result}") 