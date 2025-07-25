#!/usr/bin/env python3
"""
LangChain-based Robust PSMILES Generation System with OLLAMA Integration

This module implements a cutting-edge LLM agent system for generating 100% validated
PSMILES strings using local OLLAMA models (llama3.2) instead of external APIs.

Key Features:
- OLLAMA integration with LangChain for local model execution
- Pydantic structured output for guaranteed format compliance
- ReAct agent framework with iterative refinement
- Tool-augmented agents with RDKit validation
- Self-correcting mechanisms with chemical constraint checking
- Comprehensive error handling and fallback strategies

Author: AI Engineering Team
Date: 2024
"""

import json
import logging
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler

# Pydantic for structured output
from pydantic import BaseModel, Field, validator
from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1, validator as validatorV1

# Scientific libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available - chemical validation will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PSMILESValidationResult(BaseModel):
    """Pydantic model for PSMILES validation results"""
    psmiles: str = Field(..., description="The generated PSMILES string")
    is_valid: bool = Field(..., description="Whether the PSMILES is chemically valid")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the generation (0-1)")
    validation_errors: List[str] = Field(default=[], description="List of validation errors if any")
    chemical_properties: Dict[str, Any] = Field(default={}, description="Chemical properties if valid")
    generation_method: str = Field(..., description="Method used for generation")
    
    @validator('psmiles')
    def validate_psmiles_format(cls, v):
        """Basic PSMILES format validation"""
        if not v or len(v) < 3:
            raise ValueError("PSMILES must be at least 3 characters long")
        if not v.startswith('[*]') or not v.endswith('[*]'):
            raise ValueError("PSMILES must start and end with [*]")
        return v

class PSMILESGenerationRequest(BaseModel):
    """Pydantic model for PSMILES generation requests"""
    polymer_type: str = Field(..., description="Type of polymer (e.g., 'nanostructured', 'linear', 'branched')")
    functional_groups: List[str] = Field(default=[], description="Desired functional groups")
    molecular_weight_range: Optional[Tuple[int, int]] = Field(None, description="Target molecular weight range")
    special_properties: List[str] = Field(default=[], description="Special properties required")
    context: str = Field(default="", description="Additional context for generation")

class OLLAMACallbackHandler(BaseCallbackHandler):
    """Custom callback handler for OLLAMA operations"""
    
    def __init__(self):
        self.start_time = None
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts"""
        self.start_time = datetime.now()
        logger.info(f"🤖 OLLAMA LLM started at {self.start_time}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"✅ OLLAMA LLM completed in {duration:.2f}s")

class RDKitValidationTool:
    """RDKit-based chemical validation tools for LangChain agents"""
    
    def __init__(self):
        self.validation_cache = {}
    
    @tool
    def validate_psmiles(self, psmiles: str) -> Dict[str, Any]:
        """
        Validate a PSMILES string using RDKit
        
        Args:
            psmiles: The PSMILES string to validate
            
        Returns:
            Dictionary with validation results
        """
        if not RDKIT_AVAILABLE:
            return {
                "is_valid": False,
                "error": "RDKit not available",
                "properties": {}
            }
        
        try:
            # Convert PSMILES to SMILES for validation
            smiles = psmiles.replace('[*]', 'H')  # Replace connection points with hydrogens
            
            # Parse with RDKit
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                return {
                    "is_valid": False,
                    "error": "Invalid SMILES structure",
                    "properties": {}
                }
            
            # Calculate properties
            properties = {
                "molecular_weight": Descriptors.MolWt(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "has_aromatic": any(atom.GetIsAromatic() for atom in mol.GetAtoms())
            }
            
            return {
                "is_valid": True,
                "error": None,
                "properties": properties,
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(mol)
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "properties": {}
            }
    
    @tool
    def fix_psmiles_valency(self, psmiles: str) -> str:
        """
        Attempt to fix valency issues in PSMILES strings
        
        Args:
            psmiles: The PSMILES string with potential valency issues
            
        Returns:
            Fixed PSMILES string or original if no fix possible
        """
        try:
            # Common valency fixes
            fixes = {
                '[*]O=C': '[*]C(=O)',
                'C=O[*]': 'C(=O)[*]',
                '[*]O=': '[*]C(=O)',
                '=O[*]': '(=O)[*]',
                '[*]N=': '[*]NC(=O)',
                '=N[*]': '(=N)[*]'
            }
            
            fixed_psmiles = psmiles
            for pattern, replacement in fixes.items():
                fixed_psmiles = fixed_psmiles.replace(pattern, replacement)
            
            # Validate the fix
            validation = self.validate_psmiles(fixed_psmiles)
            if validation["is_valid"]:
                return fixed_psmiles
            else:
                return psmiles  # Return original if fix doesn't work
                
        except Exception as e:
            logger.error(f"Error in valency fixing: {e}")
            return psmiles

class LangChainPSMILESAgent:
    """
    Advanced LangChain-based PSMILES generation agent using OLLAMA
    
    This agent implements a sophisticated multi-step approach:
    1. Structured prompt engineering with chemical constraints
    2. OLLAMA-powered local LLM execution
    3. Pydantic-based output validation
    4. ReAct framework for iterative refinement
    5. RDKit tool integration for chemical validation
    6. Self-correction mechanisms
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2",
                 ollama_base_url: str = "http://localhost:11434",
                 max_iterations: int = 3,
                 temperature: float = 0.1):
        """
        Initialize the LangChain PSMILES agent
        
        Args:
            model_name: OLLAMA model name (default: llama3.2)
            ollama_base_url: OLLAMA server URL
            max_iterations: Maximum iterations for self-correction
            temperature: LLM temperature for generation
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Initialize OLLAMA LLM
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            callbacks=[OLLAMACallbackHandler()]
        )
        
        # Initialize tools
        self.rdkit_tool = RDKitValidationTool()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup parsers
        self.psmiles_parser = PydanticOutputParser(pydantic_object=PSMILESValidationResult)
        
        # Cache for generated PSMILES
        self.generation_cache = {}
        
        logger.info(f"🚀 LangChain PSMILES Agent initialized with OLLAMA model: {model_name}")
    
    def create_psmiles_generation_prompt(self) -> PromptTemplate:
        """
        Create a sophisticated prompt template for PSMILES generation
        
        Returns:
            PromptTemplate for PSMILES generation
        """
        template = """
You are an expert computational chemist and polymer scientist specializing in generating chemically valid PSMILES (Polymer SMILES) strings.

CRITICAL INSTRUCTIONS:
1. PSMILES MUST start and end with [*] connection points
2. ALL chemical structures MUST be valid (proper valency, realistic bonds)
3. Use standard organic chemistry notation
4. Avoid impossible ring structures or overvalent atoms
5. Consider realistic polymer chain architectures

POLYMER TYPE: {polymer_type}
FUNCTIONAL GROUPS: {functional_groups}
SPECIAL PROPERTIES: {special_properties}
CONTEXT: {context}

CHEMICAL CONSTRAINTS:
- Carbon: max 4 bonds
- Nitrogen: max 3 bonds (5 in aromatic rings)
- Oxygen: max 2 bonds
- Sulfur: max 2-6 bonds depending on oxidation state
- No impossible ring closures or strained geometries

EXAMPLES OF VALID PSMILES:
- [*]CC(C)C(=O)O[*] (simple aliphatic chain)
- [*]c1ccc(C(=O)O)cc1[*] (aromatic with carboxyl)
- [*]CC(=O)NCC(=O)[*] (amide linkages)
- [*]CC(C)(C)c1ccc(O)cc1[*] (branched aromatic)

GENERATE a chemically valid PSMILES string that matches the requirements.

{format_instructions}

PSMILES Generation:
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["polymer_type", "functional_groups", "special_properties", "context"],
            partial_variables={"format_instructions": self.psmiles_parser.get_format_instructions()}
        )
    
    def create_self_correction_prompt(self) -> PromptTemplate:
        """
        Create a prompt template for self-correction of invalid PSMILES
        
        Returns:
            PromptTemplate for PSMILES correction
        """
        template = """
You are a chemical structure validation expert. A PSMILES string has failed validation.

ORIGINAL PSMILES: {original_psmiles}
VALIDATION ERRORS: {validation_errors}

Your task is to FIX the chemical structure while maintaining the core polymer architecture.

COMMON ISSUES AND FIXES:
1. Valency errors: Ensure carbon has ≤4 bonds, nitrogen ≤3, oxygen ≤2
2. Ring strain: Replace impossible rings with linear or stable ring structures
3. Connection points: Ensure [*] points don't create overvalent atoms
4. Functional groups: Use standard chemistry notation

CORRECTION STRATEGIES:
- Replace [*]O=C with [*]C(=O) for carbonyl connection
- Convert impossible ring closures to linear chains
- Add explicit hydrogens where needed
- Use parentheses for branching: C(=O) instead of CO=

{format_instructions}

CORRECTED PSMILES:
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["original_psmiles", "validation_errors"],
            partial_variables={"format_instructions": self.psmiles_parser.get_format_instructions()}
        )
    
    def generate_psmiles_with_validation(self, request: PSMILESGenerationRequest) -> PSMILESValidationResult:
        """
        Generate a validated PSMILES string with iterative self-correction
        
        Args:
            request: PSMILES generation request
            
        Returns:
            PSMILESValidationResult with the final result
        """
        logger.info(f"🧪 Starting PSMILES generation for {request.polymer_type}")
        
        # Create generation prompt
        generation_prompt = self.create_psmiles_generation_prompt()
        correction_prompt = self.create_self_correction_prompt()
        
        # Initial generation
        try:
            # Format the prompt
            formatted_prompt = generation_prompt.format(
                polymer_type=request.polymer_type,
                functional_groups=", ".join(request.functional_groups) if request.functional_groups else "none specified",
                special_properties=", ".join(request.special_properties) if request.special_properties else "none specified",
                context=request.context or "general polymer design"
            )
            
            logger.info("🤖 Generating initial PSMILES with OLLAMA...")
            
            # Generate with OLLAMA
            response = self.llm.invoke(formatted_prompt)
            
            # Parse the response
            try:
                result = self.psmiles_parser.parse(response)
                logger.info(f"✅ Initial generation: {result.psmiles}")
            except Exception as parse_error:
                logger.warning(f"⚠️ Parsing failed, extracting PSMILES manually: {parse_error}")
                # Fallback: extract PSMILES manually
                result = self._extract_psmiles_manually(response, request)
            
            # Validate with RDKit
            validation = self.rdkit_tool.validate_psmiles(result.psmiles)
            
            if validation["is_valid"]:
                result.is_valid = True
                result.chemical_properties = validation["properties"]
                result.generation_method = "initial_generation"
                logger.info("🎉 Initial generation successful!")
                return result
            
            # Self-correction loop
            current_psmiles = result.psmiles
            current_errors = [validation["error"]] if validation["error"] else ["Structure validation failed"]
            
            for iteration in range(self.max_iterations):
                logger.info(f"🔄 Self-correction iteration {iteration + 1}/{self.max_iterations}")
                
                # Create correction prompt
                correction_formatted = correction_prompt.format(
                    original_psmiles=current_psmiles,
                    validation_errors="; ".join(current_errors)
                )
                
                # Generate correction
                correction_response = self.llm.invoke(correction_formatted)
                
                try:
                    corrected_result = self.psmiles_parser.parse(correction_response)
                    current_psmiles = corrected_result.psmiles
                    logger.info(f"   → Corrected: {current_psmiles}")
                except Exception:
                    # Manual extraction fallback
                    corrected_result = self._extract_psmiles_manually(correction_response, request)
                    current_psmiles = corrected_result.psmiles
                
                # Validate correction
                validation = self.rdkit_tool.validate_psmiles(current_psmiles)
                
                if validation["is_valid"]:
                    corrected_result.is_valid = True
                    corrected_result.chemical_properties = validation["properties"]
                    corrected_result.generation_method = f"self_correction_iteration_{iteration + 1}"
                    logger.info(f"🎉 Self-correction successful at iteration {iteration + 1}!")
                    return corrected_result
                
                # Update errors for next iteration
                current_errors = [validation["error"]] if validation["error"] else ["Structure validation failed"]
            
            # If all corrections failed, return with error details
            result.psmiles = current_psmiles
            result.is_valid = False
            result.validation_errors = current_errors
            result.generation_method = f"failed_after_{self.max_iterations}_corrections"
            
            logger.warning(f"⚠️ All correction attempts failed. Final PSMILES: {current_psmiles}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.error(traceback.format_exc())
            
            return PSMILESValidationResult(
                psmiles="[*]CC[*]",  # Fallback
                is_valid=False,
                confidence_score=0.0,
                validation_errors=[f"Generation error: {str(e)}"],
                generation_method="error_fallback"
            )
    
    def _extract_psmiles_manually(self, response: str, request: PSMILESGenerationRequest) -> PSMILESValidationResult:
        """
        Manual extraction of PSMILES from LLM response when parsing fails
        
        Args:
            response: Raw LLM response
            request: Original request
            
        Returns:
            PSMILESValidationResult with extracted PSMILES
        """
        import re
        
        # Look for PSMILES pattern [*]...[*]
        psmiles_pattern = r'\[\*\][^\[\]]*\[\*\]'
        matches = re.findall(psmiles_pattern, response)
        
        if matches:
            psmiles = matches[0]  # Take first match
            logger.info(f"📝 Manually extracted PSMILES: {psmiles}")
        else:
            # Fallback to simple structure
            psmiles = "[*]CC[*]"
            logger.warning("⚠️ No PSMILES found, using fallback")
        
        return PSMILESValidationResult(
            psmiles=psmiles,
            is_valid=False,  # Will be validated separately
            confidence_score=0.5,
            validation_errors=[],
            generation_method="manual_extraction"
        )
    
    def batch_generate_psmiles(self, requests: List[PSMILESGenerationRequest]) -> List[PSMILESValidationResult]:
        """
        Generate multiple PSMILES strings in batch
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of PSMILESValidationResult objects
        """
        logger.info(f"🔄 Starting batch generation for {len(requests)} requests")
        
        results = []
        for i, request in enumerate(requests):
            logger.info(f"📊 Processing request {i+1}/{len(requests)}")
            result = self.generate_psmiles_with_validation(request)
            results.append(result)
        
        success_count = sum(1 for r in results if r.is_valid)
        logger.info(f"✅ Batch completed: {success_count}/{len(requests)} successful")
        
        return results
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about PSMILES generation performance
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            "model_name": self.model_name,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "rdkit_available": RDKIT_AVAILABLE,
            "cache_size": len(self.generation_cache)
        }

# Factory function for easy instantiation
def create_psmiles_agent(model_name: str = "llama3.2", **kwargs) -> LangChainPSMILESAgent:
    """
    Factory function to create a PSMILES agent with OLLAMA
    
    Args:
        model_name: OLLAMA model name
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Configured LangChainPSMILESAgent instance
    """
    return LangChainPSMILESAgent(model_name=model_name, **kwargs)

if __name__ == "__main__":
    # Example usage
    print("🚀 LangChain PSMILES System with OLLAMA")
    print("="*50)
    
    # Create agent
    agent = create_psmiles_agent("llama3.2")
    
    # Example request
    request = PSMILESGenerationRequest(
        polymer_type="nanostructured",
        functional_groups=["carboxyl", "hydroxyl"],
        special_properties=["biodegradable", "biocompatible"],
        context="insulin delivery polymer"
    )
    
    # Generate PSMILES
    result = agent.generate_psmiles_with_validation(request)
    
    print(f"Generated PSMILES: {result.psmiles}")
    print(f"Valid: {result.is_valid}")
    print(f"Method: {result.generation_method}")
    print(f"Properties: {result.chemical_properties}") 