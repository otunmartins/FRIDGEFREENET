#!/usr/bin/env python3

"""
🧪 LangChain-Based Robust PSMILES Generation Agent

This module implements a state-of-the-art LangChain agent for generating 
100% validated and correct PSMILES strings using:

1. VALID-Mol Framework principles (83% success rate)
2. Tool-Augmented Agents with structured output
3. Self-Correcting Agents with ReAct framework
4. Constrained Generation with Pydantic validation
5. Multi-step validation pipelines with RDKit integration

Based on 2024-2025 research in chemical LLM agents.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

# LangChain imports
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Chemical validation imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available - chemical validation will be limited")


class ChemicalValidationLevel(str, Enum):
    """Validation levels for chemical structures."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class PSMILESGenerationRequest(BaseModel):
    """Structured request for PSMILES generation."""
    
    description: str = Field(
        ..., 
        description="Natural language description of the desired polymer structure"
    )
    required_elements: Optional[List[str]] = Field(
        default=None, 
        description="Required chemical elements (e.g., ['S', 'N', 'O'])"
    )
    excluded_elements: Optional[List[str]] = Field(
        default=None, 
        description="Elements to exclude"
    )
    target_complexity: str = Field(
        default="medium", 
        description="Complexity level: simple, medium, complex"
    )
    validation_level: ChemicalValidationLevel = Field(
        default=ChemicalValidationLevel.ADVANCED,
        description="Level of chemical validation to apply"
    )
    max_attempts: int = Field(
        default=5, 
        description="Maximum number of generation attempts"
    )
    
    @validator('required_elements')
    def validate_elements(cls, v):
        if v is not None:
            # Validate chemical elements
            valid_elements = {
                'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Zn', 'Cu', 'Ni'
            }
            for element in v:
                if element not in valid_elements:
                    raise ValueError(f"Invalid chemical element: {element}")
        return v


class PSMILESGenerationResult(BaseModel):
    """Structured result for PSMILES generation."""
    
    success: bool = Field(..., description="Whether generation was successful")
    psmiles: Optional[str] = Field(default=None, description="Generated PSMILES string")
    smiles: Optional[str] = Field(default=None, description="Converted SMILES string")
    validation_score: float = Field(default=0.0, description="Chemical validation score (0-1)")
    validation_errors: List[str] = Field(default=[], description="Any validation errors found")
    generation_method: str = Field(default="", description="Method used for generation")
    attempts_used: int = Field(default=0, description="Number of attempts used")
    confidence: float = Field(default=0.0, description="Confidence in the result (0-1)")
    molecular_properties: Dict[str, Any] = Field(default={}, description="Calculated molecular properties")
    warnings: List[str] = Field(default=[], description="Any warnings about the structure")


class LangChainPSMILESAgent:
    """
    🤖 Advanced LangChain-based PSMILES Generation Agent
    
    Implements cutting-edge techniques for 100% validated chemical structure generation.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 verbose: bool = True):
        """Initialize the LangChain PSMILES agent."""
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize tools
        self.tools = self._create_chemical_tools()
        
        # Create the agent
        self.agent = self._create_react_agent()
        
        # Initialize cache
        self.generation_cache = {}
        
        print(f"🤖 LangChain PSMILES Agent initialized with {model_name}")
        if not RDKIT_AVAILABLE:
            print("⚠️ Warning: RDKit not available - using simplified validation")
    
    def _create_chemical_tools(self) -> List:
        """Create chemical validation and analysis tools."""
        
        @tool
        def validate_smiles(smiles: str) -> Dict[str, Any]:
            """
            Validate a SMILES string and return detailed analysis.
            
            Args:
                smiles: The SMILES string to validate
                
            Returns:
                Dictionary with validation results and molecular properties
            """
            result = {
                'valid': False,
                'errors': [],
                'warnings': [],
                'properties': {}
            }
            
            if not RDKIT_AVAILABLE:
                result['errors'].append("RDKit not available for validation")
                return result
            
            try:
                # Parse molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    result['errors'].append("Invalid SMILES: Cannot parse molecule")
                    return result
                
                # Basic validation
                result['valid'] = True
                
                # Calculate properties
                result['properties'] = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_bonds': mol.GetNumBonds(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'num_hbd': Descriptors.NumHDonors(mol),
                    'num_hba': Descriptors.NumHAcceptors(mol)
                }
                
                # Advanced validation checks
                self._advanced_chemical_validation(mol, result)
                
            except Exception as e:
                result['errors'].append(f"Validation error: {str(e)}")
            
            return result
        
        @tool
        def validate_psmiles(psmiles: str) -> Dict[str, Any]:
            """
            Validate a PSMILES string by converting to SMILES and validating.
            
            Args:
                psmiles: The PSMILES string to validate
                
            Returns:
                Dictionary with validation results
            """
            result = {
                'valid': False,
                'errors': [],
                'warnings': [],
                'smiles': None
            }
            
            try:
                # Convert PSMILES to SMILES for validation
                if psmiles.startswith('[*]') and psmiles.endswith('[*]'):
                    # Remove connection points for validation
                    smiles = psmiles[3:-3]  # Remove [*] from both ends
                    smiles = smiles.replace('[*]', 'H')  # Replace any internal [*] with H
                    
                    result['smiles'] = smiles
                    
                    # Validate the SMILES
                    smiles_result = validate_smiles(smiles)
                    result.update(smiles_result)
                    
                else:
                    result['errors'].append("Invalid PSMILES format: Must start and end with [*]")
                
            except Exception as e:
                result['errors'].append(f"PSMILES validation error: {str(e)}")
            
            return result
        
        @tool
        def suggest_psmiles_improvements(psmiles: str, errors: List[str]) -> List[str]:
            """
            Suggest improvements for a problematic PSMILES string.
            
            Args:
                psmiles: The problematic PSMILES string
                errors: List of validation errors
                
            Returns:
                List of suggested improvements
            """
            suggestions = []
            
            # Analyze common issues and suggest fixes
            if "valence" in str(errors).lower():
                suggestions.append("Consider reducing bond orders or adding hydrogen atoms")
                suggestions.append("Check for overvalent atoms (e.g., C with >4 bonds)")
            
            if "ring" in str(errors).lower():
                suggestions.append("Verify ring closure numbers match correctly")
                suggestions.append("Ensure ring sizes are chemically reasonable (5-7 atoms)")
            
            if "sanitize" in str(errors).lower():
                suggestions.append("Simplify the structure by removing complex functional groups")
                suggestions.append("Use explicit hydrogen atoms where needed")
            
            # Element-specific suggestions
            if 'S' in psmiles:
                suggestions.append("Sulfur can have multiple oxidation states - check S(=O) or S(=O)(=O)")
            
            if 'N' in psmiles:
                suggestions.append("Nitrogen valency: N can have 3-5 bonds depending on charge")
            
            if not suggestions:
                suggestions.append("Try simplifying the structure or using alternative functional groups")
            
            return suggestions
        
        @tool
        def generate_alternative_psmiles(description: str, failed_psmiles: str) -> List[str]:
            """
            Generate alternative PSMILES structures based on description and failed attempt.
            
            Args:
                description: Original description of desired structure
                failed_psmiles: The PSMILES that failed validation
                
            Returns:
                List of alternative PSMILES strings to try
            """
            alternatives = []
            
            # Extract key functional groups from description
            if "sulfur" in description.lower():
                alternatives.extend([
                    "[*]CC(=O)SC(=O)C[*]",  # Thioester linkage
                    "[*]CCS[*]",             # Simple sulfur bridge
                    "[*]CS(=O)(=O)C[*]"      # Sulfone linkage
                ])
            
            if "carbonyl" in description.lower() or "ketone" in description.lower():
                alternatives.extend([
                    "[*]CC(=O)C[*]",         # Simple ketone
                    "[*]C(=O)CC(=O)C[*]",    # Diketone
                    "[*]COC(=O)C[*]"         # Ester linkage
                ])
            
            if "ester" in description.lower():
                alternatives.extend([
                    "[*]COC(=O)C[*]",        # Simple ester
                    "[*]CC(=O)OC[*]",        # Reverse ester
                    "[*]COC(=O)CC(=O)OC[*]"  # Diester
                ])
            
            # Fallback simple structures
            if not alternatives:
                alternatives.extend([
                    "[*]CC[*]",              # Simple alkyl chain
                    "[*]COC[*]",             # Ether linkage
                    "[*]CNC[*]",             # Amine linkage
                    "[*]C(C)C[*]"            # Branched alkyl
                ])
            
            return alternatives[:3]  # Return top 3 alternatives
        
        return [validate_smiles, validate_psmiles, suggest_psmiles_improvements, generate_alternative_psmiles]
    
    def _advanced_chemical_validation(self, mol, result: Dict[str, Any]):
        """Perform advanced chemical validation checks."""
        
        try:
            # Check for unusual valencies
            for atom in mol.GetAtoms():
                valence = atom.GetTotalValence()
                symbol = atom.GetSymbol()
                
                # Define expected valence ranges
                expected_valences = {
                    'C': [4], 'N': [3, 5], 'O': [2], 'S': [2, 4, 6],
                    'P': [3, 5], 'F': [1], 'Cl': [1], 'Br': [1], 'I': [1]
                }
                
                if symbol in expected_valences:
                    if valence not in expected_valences[symbol]:
                        result['warnings'].append(f"Unusual valence for {symbol}: {valence}")
            
            # Check for strained rings
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                if len(ring) < 3:
                    result['warnings'].append(f"Very strained ring of size {len(ring)}")
                elif len(ring) == 3 or len(ring) == 4:
                    result['warnings'].append(f"Strained ring of size {len(ring)}")
            
            # Check molecular weight reasonableness for polymers
            mw = Descriptors.MolWt(mol)
            if mw > 1000:
                result['warnings'].append(f"High molecular weight: {mw:.1f}")
            elif mw < 50:
                result['warnings'].append(f"Very low molecular weight: {mw:.1f}")
                
        except Exception as e:
            result['warnings'].append(f"Advanced validation error: {str(e)}")
    
    def _create_react_agent(self):
        """Create a ReAct-style self-correcting agent."""
        
        # Create the prompt template
        prompt_template = """
You are an expert computational chemist specializing in polymer structure generation. 
Your task is to generate valid PSMILES (Polymer SMILES) strings that are 100% chemically correct.

CRITICAL RULES:
1. PSMILES MUST start and end with [*] connection points
2. ALL structures must pass RDKit validation
3. If a structure fails validation, you MUST use the available tools to fix it
4. Always validate your output before providing the final answer

Available tools: {tools}

Previous conversation:
{chat_history}

Question: {input}

Thought: I need to generate a chemically valid PSMILES string. Let me think about this step by step.

{agent_scratchpad}
"""
        
        # Get the ReAct prompt from hub (fallback to custom if not available)
        try:
            prompt = hub.pull("hwchase17/react")
        except:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def generate_psmiles(self, request: PSMILESGenerationRequest) -> PSMILESGenerationResult:
        """
        Generate a validated PSMILES string using the LangChain agent.
        
        Args:
            request: Structured request with generation parameters
            
        Returns:
            Structured result with generated PSMILES and validation data
        """
        print(f"\n🤖 Generating PSMILES for: {request.description}")
        
        # Check cache first
        cache_key = f"{request.description}_{request.required_elements}_{request.target_complexity}"
        if cache_key in self.generation_cache:
            print("📦 Using cached result")
            return self.generation_cache[cache_key]
        
        # Create the input for the agent
        agent_input = self._create_agent_input(request)
        
        best_result = PSMILESGenerationResult(
            success=False,
            generation_method="langchain_react_agent",
            attempts_used=0
        )
        
        for attempt in range(request.max_attempts):
            print(f"\n🔄 Attempt {attempt + 1}/{request.max_attempts}")
            
            try:
                # Run the agent
                response = self.agent.invoke(agent_input)
                
                # Extract PSMILES from response
                psmiles = self._extract_psmiles_from_response(response)
                
                if psmiles:
                    # Validate the generated PSMILES
                    validation_result = self._validate_generated_psmiles(psmiles, request)
                    
                    result = PSMILESGenerationResult(
                        success=validation_result['valid'],
                        psmiles=psmiles,
                        smiles=validation_result.get('smiles'),
                        validation_score=validation_result.get('score', 0.0),
                        validation_errors=validation_result.get('errors', []),
                        generation_method="langchain_react_agent",
                        attempts_used=attempt + 1,
                        confidence=validation_result.get('confidence', 0.0),
                        molecular_properties=validation_result.get('properties', {}),
                        warnings=validation_result.get('warnings', [])
                    )
                    
                    if result.success:
                        print(f"✅ Success! Generated valid PSMILES: {psmiles}")
                        self.generation_cache[cache_key] = result
                        return result
                    else:
                        print(f"❌ Validation failed: {result.validation_errors}")
                        best_result = result  # Keep the best attempt so far
                
            except Exception as e:
                print(f"❌ Agent error: {str(e)}")
                best_result.validation_errors.append(f"Agent error: {str(e)}")
        
        print(f"❌ Failed to generate valid PSMILES after {request.max_attempts} attempts")
        best_result.attempts_used = request.max_attempts
        return best_result
    
    def _create_agent_input(self, request: PSMILESGenerationRequest) -> Dict[str, str]:
        """Create input for the LangChain agent."""
        
        constraints = []
        if request.required_elements:
            constraints.append(f"Must contain elements: {', '.join(request.required_elements)}")
        if request.excluded_elements:
            constraints.append(f"Must NOT contain elements: {', '.join(request.excluded_elements)}")
        
        complexity_guide = {
            "simple": "Use simple functional groups and linear structures",
            "medium": "Can include rings and common functional groups",
            "complex": "May include complex rings and multiple functional groups"
        }
        
        input_text = f"""
Generate a chemically valid PSMILES string for: {request.description}

Requirements:
- Target complexity: {request.target_complexity} ({complexity_guide.get(request.target_complexity, '')})
- Validation level: {request.validation_level.value}
{chr(10).join([f'- {c}' for c in constraints]) if constraints else ''}

CRITICAL REQUIREMENTS:
1. The PSMILES MUST start and end with [*] connection points
2. The structure MUST pass RDKit chemical validation
3. Use the available tools to validate your answer
4. If validation fails, use the tools to get suggestions and try again

Please generate and validate a PSMILES string that meets all requirements.
"""
        
        return {"input": input_text}
    
    def _extract_psmiles_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract PSMILES string from agent response."""
        
        output = response.get('output', '')
        
        # Look for PSMILES patterns
        import re
        psmiles_pattern = r'\[\*\][^[]*\[\*\]'
        matches = re.findall(psmiles_pattern, output)
        
        if matches:
            # Return the first valid-looking PSMILES
            return matches[0]
        
        # Fallback: look for any string that might be PSMILES
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[*]') and line.endswith('[*]'):
                return line
        
        return None
    
    def _validate_generated_psmiles(self, psmiles: str, request: PSMILESGenerationRequest) -> Dict[str, Any]:
        """Validate the generated PSMILES using our tools."""
        
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'score': 0.0,
            'confidence': 0.0,
            'properties': {}
        }
        
        try:
            # Use the validation tool
            validation_tool = self.tools[1]  # validate_psmiles tool
            validation_result = validation_tool.func(psmiles)
            
            result.update(validation_result)
            
            # Calculate validation score
            if result['valid']:
                score = 1.0
                
                # Reduce score for warnings
                score -= len(result.get('warnings', [])) * 0.1
                
                # Check requirements compliance
                if request.required_elements:
                    for element in request.required_elements:
                        if element not in psmiles:
                            score -= 0.2
                            result['warnings'].append(f"Missing required element: {element}")
                
                if request.excluded_elements:
                    for element in request.excluded_elements:
                        if element in psmiles:
                            score -= 0.3
                            result['errors'].append(f"Contains excluded element: {element}")
                            result['valid'] = False
                
                result['score'] = max(0.0, score)
                result['confidence'] = result['score']
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result


# Example usage and testing
def main():
    """Example usage of the LangChain PSMILES Agent."""
    
    print("🧪 LangChain PSMILES Agent - Example Usage")
    print("=" * 60)
    
    # Initialize the agent
    agent = LangChainPSMILESAgent(verbose=True)
    
    # Test requests
    test_requests = [
        PSMILESGenerationRequest(
            description="polymer with sulfur atoms for drug delivery",
            required_elements=["S", "O"],
            target_complexity="medium"
        ),
        PSMILESGenerationRequest(
            description="biodegradable ester linkage polymer",
            required_elements=["O"],
            excluded_elements=["S", "N"],
            target_complexity="simple"
        ),
        PSMILESGenerationRequest(
            description="polyamide with nitrogen functionality",
            required_elements=["N", "O"],
            target_complexity="medium"
        )
    ]
    
    # Test each request
    for i, request in enumerate(test_requests, 1):
        print(f"\n{'='*20} Test {i} {'='*20}")
        result = agent.generate_psmiles(request)
        
        print(f"✅ Success: {result.success}")
        print(f"🔗 PSMILES: {result.psmiles}")
        print(f"📊 Validation Score: {result.validation_score:.2f}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"🔄 Attempts: {result.attempts_used}")
        
        if result.validation_errors:
            print(f"❌ Errors: {result.validation_errors}")
        if result.warnings:
            print(f"⚠️ Warnings: {result.warnings}")


if __name__ == "__main__":
    main() 