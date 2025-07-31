#!/usr/bin/env python3
"""
Enhanced PSMILES Auto-Corrector System
Automatically corrects malformed PSMILES strings using OpenAI ChatGPT models.
Integrates with the enhanced chemical repair system for comprehensive validation.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# **UPDATED: OpenAI imports instead of Ollama**
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Enhanced chemical repair system
try:
    import sys
    import os
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from insulin_ai.core.enhanced_chemical_repair import EnhancedChemicalRepair, repair_chemical_structure
    ENHANCED_REPAIR_AVAILABLE = True
except ImportError as e:
    ENHANCED_REPAIR_AVAILABLE = False
    logging.warning(f"Enhanced Chemical Repair not available: {e}")

# RDKit for validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available - using basic validation")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSMILESAutoCorrector:
    """
    Enhanced PSMILES Auto-Corrector using OpenAI ChatGPT models.
    Provides intelligent correction of malformed PSMILES strings with comprehensive validation.
    """
    
    def __init__(self, 
                 model_type: str = "openai",
                 openai_model: str = "gpt-4o",
                 temperature: float = 0.3):
        """
        Initialize the PSMILES Auto-Corrector with OpenAI models.
        
        Args:
            model_type (str): Type of model ('openai')
            openai_model (str): OpenAI model name (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
            temperature (float): Model temperature (0.0-2.0, lower for more precise corrections)
        """
        
        self.model_type = model_type
        self.openai_model = openai_model
        self.temperature = temperature
        
        # **UPDATED: Initialize OpenAI ChatGPT model**
        try:
            # Verify API key is available
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                openai_api_key=api_key
            )
            logger.info(f"✅ PSMILES Auto-Corrector initialized with OpenAI {openai_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI PSMILES Auto-Corrector: {e}")
            raise
        
        # Initialize enhanced repair system if available
        if ENHANCED_REPAIR_AVAILABLE:
            try:
                self.enhanced_repairer = EnhancedChemicalRepair()
                logger.info("✅ Enhanced Chemical Repair System integrated")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced repair integration failed: {e}")
                self.enhanced_repairer = None
        else:
            self.enhanced_repairer = None
        
        # Setup correction prompts
        self.prompts = self._setup_correction_prompts()
        
        logger.info(f"✅ PSMILES Auto-Corrector initialized with {openai_model}")
    
    def _setup_correction_prompts(self) -> Dict:
        """Setup prompt templates for PSMILES correction."""
        
        correction_system_prompt = """You are an expert computational chemist specializing in PSMILES (Polymer SMILES) correction and validation.

PSMILES RULES:
1. MUST have exactly TWO [*] connection points (no more, no less)
2. Must follow proper SMILES syntax (valid atoms, bonds, brackets, parentheses)
3. Must represent chemically valid polymer structures
4. Connection points mark where polymer units connect during polymerization
5. Structure between [*] symbols should be chemically reasonable

COMMON ERRORS TO FIX:
- Wrong number of [*] symbols (not exactly 2)
- Invalid SMILES syntax (unbalanced brackets, invalid atoms)
- Chemical impossibilities (wrong valences, impossible bonds)
- Malformed connection points

EXAMPLES OF CORRECTIONS:
- [*]C(=O)O[O-][*] → [*]C(=O)O[*] (remove invalid O-)
- [*]CC(=O)[N+]-[N+][*] → [*]CC(=O)N[*] (fix invalid charges)
- C1CCCCC1 → [*]C1CCCCC1[*] (add missing connection points)
- [*]C[*]C[*] → [*]CC[*] (remove extra connection point)

Your task: Fix the malformed PSMILES to make it chemically valid with exactly 2 [*] connection points."""

        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", correction_system_prompt),
            ("human", """Please correct this malformed PSMILES string:

MALFORMED PSMILES: {malformed_psmiles}

ERRORS DETECTED: {error_description}

Provide the corrected PSMILES as a single line with exactly 2 [*] connection points and valid chemistry.

CORRECTED PSMILES:"""),
        ])
        
        return {
            'correction': correction_prompt
        }
    
    def correct_psmiles(self, 
                       malformed_psmiles: str, 
                       context: Optional[str] = None,
                       max_attempts: int = 3) -> Dict:
        """
        Correct a malformed PSMILES string using OpenAI.
        
        Args:
            malformed_psmiles (str): The malformed PSMILES to correct
            context (str, optional): Additional context about desired structure
            max_attempts (int): Maximum correction attempts
            
        Returns:
            Dict: Correction results with success status and corrected PSMILES
        """
        try:
            logger.info(f"🔧 Correcting malformed PSMILES: {malformed_psmiles}")
            
            # First, try enhanced repair system if available
            if self.enhanced_repairer:
                logger.info("🚀 Attempting enhanced chemical repair first...")
                enhanced_result = self.enhanced_repairer.repair_psmiles_structure(malformed_psmiles, context)
                
                if enhanced_result['success']:
                    logger.info("✅ Enhanced repair successful!")
                    return {
                        'success': True,
                        'original_psmiles': malformed_psmiles,
                        'corrected_psmiles': enhanced_result['repaired_psmiles'],
                        'correction_method': f"enhanced_{enhanced_result['repair_strategy']}",
                        'validation_result': enhanced_result.get('validation_result', {}),
                        'attempts': 1,
                        'model': self.openai_model,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    logger.warning(f"Enhanced repair failed: {enhanced_result['error']}")
                    logger.info("🔄 Falling back to LLM-based correction...")
            
            # LLM-based correction as fallback or primary method
            for attempt in range(max_attempts):
                try:
                    logger.info(f"🤖 LLM correction attempt {attempt + 1}/{max_attempts}")
                    
                    # Analyze errors in the malformed PSMILES
                    error_description = self._analyze_psmiles_errors(malformed_psmiles)
                    
                    # **UPDATED: Use OpenAI ChatGPT with proper message format**
                    messages = self.prompts['correction'].format_messages(
                        malformed_psmiles=malformed_psmiles,
                        error_description=error_description
                    )
                    
                    response = self.llm.invoke(messages)
                    corrected_psmiles = response.content if hasattr(response, 'content') else str(response)
                    
                    # Clean the response
                    corrected_psmiles = self._clean_correction_response(corrected_psmiles)
                    
                    # Validate the correction
                    is_valid, validation_message = self._validate_corrected_psmiles(corrected_psmiles)
                    
                    if is_valid:
                        logger.info(f"✅ LLM correction successful: {corrected_psmiles}")
                        return {
                            'success': True,
                            'original_psmiles': malformed_psmiles,
                            'corrected_psmiles': corrected_psmiles,
                            'correction_method': f'llm_correction_attempt_{attempt + 1}',
                            'error_analysis': error_description,
                            'validation_message': validation_message,
                            'attempts': attempt + 1,
                            'model': self.openai_model,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        logger.warning(f"❌ Attempt {attempt + 1} validation failed: {validation_message}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"⚠️ LLM correction attempt {attempt + 1} failed: {e}")
                    continue
            
            # All attempts failed
            return {
                'success': False,
                'error': f'Failed to correct PSMILES after {max_attempts} attempts',
                'original_psmiles': malformed_psmiles,
                'attempts': max_attempts,
                'model': self.openai_model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ PSMILES correction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_psmiles': malformed_psmiles,
                'model': self.openai_model,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_psmiles_errors(self, psmiles: str) -> str:
        """Analyze the errors in a PSMILES string."""
        errors = []
        
        # Check for unbalanced parentheses
        if psmiles.count('(') != psmiles.count(')'):
            errors.append("Unbalanced parentheses detected.")
        
        # Check for invalid SMILES syntax (e.g., unbalanced brackets)
        if psmiles.count('[') != psmiles.count(']'):
            errors.append("Unbalanced brackets detected.")
        
        # Check for invalid charges (e.g., [N+]-[N+])
        if re.search(r'\[\+\]-[\+\]', psmiles):
            errors.append("Invalid charge notation detected.")
        
        # Check for invalid aromatic atoms (lowercase) directly connected to [*]
        if re.search(r'c\[\*\]|n\[\*\]|o\[\*\]|s\[\*\]', psmiles):
            errors.append("Aromatic atoms (lowercase) directly connected to [*] detected.")
        
        # Check for invalid valency (e.g., C=C=C)
        if re.search(r'=[=\#]', psmiles):
            errors.append("Invalid bond notation detected (double or triple bond with same atom).")
        
        # Check for invalid atom types (e.g., C(=O)O[O-])
        if re.search(r'\[\-\]', psmiles):
            errors.append("Invalid charge notation detected (O-).")
        
        # Check for invalid connection points (e.g., [*]C[*]C[*])
        if psmiles.count('[*]') > 2:
            errors.append("Multiple connection points detected.")
        
        return "; ".join(errors) if errors else "No specific errors detected."
    
    def _clean_correction_response(self, response: str) -> str:
        """Clean the response from the LLM to ensure it's a single line PSMILES."""
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # Remove any extra newlines or carriage returns
        response = re.sub(r'\n\s*', ' ', response)
        
        # Ensure it's a single line PSMILES
        response = re.sub(r'\s+', ' ', response)
        
        # Remove any remaining brackets if they are not part of a valid SMILES
        response = re.sub(r'\[?\*\]?\[?\*\]?', '[*]', response)
        
        return response
    
    def _validate_corrected_psmiles(self, corrected_psmiles: str) -> Tuple[bool, str]:
        """Validate if the corrected PSMILES is chemically valid and has exactly 2 [*]."""
        if corrected_psmiles.count('[*]') != 2:
            return False, "Correction failed: PSMILES does not have exactly 2 [*] connection points."
        
        # Basic RDKit validation (optional, but good for chemical validity)
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(corrected_psmiles)
                if mol is None:
                    return False, "Correction failed: The corrected PSMILES is not a valid SMILES string."
                # Check for common chemical issues (e.g., invalid valency, aromatic atoms)
                # This is a simplified check, a full RDKit validation would be better
                if Descriptors.MolWt(mol) == 0: # Check for a valid molecule
                    return False, "Correction failed: The corrected PSMILES is not a valid chemical structure."
            except Exception as e:
                logger.warning(f"RDKit validation failed: {e}")
                return False, "Correction failed: Could not validate chemical validity with RDKit."
        
        return True, "Correction is chemically valid and has exactly 2 [*]."


# Convenience function for integration
def create_psmiles_auto_corrector(model_type: str = "openai", 
                                 openai_model: str = "gpt-4o",
                                 temperature: float = 0.3) -> PSMILESAutoCorrector:
    """Create and return a PSMILES auto-corrector instance."""
    return PSMILESAutoCorrector(model_type=model_type, openai_model=openai_model, temperature=temperature)


if __name__ == "__main__":
    # Test the system with the provided examples
    corrector = create_psmiles_auto_corrector()
    
    test_structures = [
        "C(O)C(=O)N[*]CC(=O)[*]",  # Structure 4
        "C(O)CO[*]NC(=O)COC(=O)C[*]S"  # Structure 5
    ]
    
    for psmiles in test_structures:
        print(f"\n🔧 Testing correction for: {psmiles}")
        result = corrector.correct_psmiles(psmiles)
        
        if result['success']:
            print(f"✅ Generated {result['correction_method']} correction:")
            print(f"  Original: {result['original_psmiles']}")
            print(f"  Corrected: {result['corrected_psmiles']}")
            print(f"  Validation: {result['validation_message']}")
        else:
            print(f"❌ Correction failed after {result['attempts']} attempts: {result['error']}") 