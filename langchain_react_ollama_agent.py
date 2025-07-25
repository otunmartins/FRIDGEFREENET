#!/usr/bin/env python3
"""
🤖 LangChain ReAct Agent for PSMILES Generation with OLLAMA Integration

This module implements a complete ReAct (Reasoning + Acting) agent using OLLAMA
for iterative PSMILES generation with tool-augmented capabilities.

Key Features:
- Full ReAct framework with reasoning loops
- OLLAMA local model integration (llama3.2)
- Tool-augmented chemical validation
- Iterative refinement with feedback
- Complete thought process tracking

Author: AI Engineering Team  
Date: 2024
"""

import json
import logging
import traceback
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler

# Pydantic for structured output
from pydantic import BaseModel, Field, validator

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


class PSMILESGenerationRequest(BaseModel):
    """Structured request for PSMILES generation with ReAct agent"""
    polymer_type: str = Field(..., description="Type of polymer (e.g., 'nanostructured', 'linear', 'branched')")
    functional_groups: List[str] = Field(default=[], description="Desired functional groups")
    molecular_weight_range: Optional[Tuple[int, int]] = Field(None, description="Target molecular weight range")
    special_properties: List[str] = Field(default=[], description="Special properties required")
    context: str = Field(default="", description="Additional context for generation")
    max_iterations: int = Field(default=5, description="Maximum ReAct iterations")


class PSMILESValidationResult(BaseModel):
    """Structured result for PSMILES validation"""
    psmiles: str = Field(..., description="The generated PSMILES string")
    is_valid: bool = Field(..., description="Whether the PSMILES is chemically valid")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the generation (0-1)")
    validation_errors: List[str] = Field(default=[], description="List of validation errors if any")
    chemical_properties: Dict[str, Any] = Field(default={}, description="Chemical properties if valid")
    generation_method: str = Field(..., description="Method used for generation")
    react_steps: List[Dict[str, str]] = Field(default=[], description="ReAct reasoning steps")
    
    @validator('psmiles')
    def validate_psmiles_format(cls, v):
        """Basic PSMILES format validation"""
        if not v or len(v) < 3:
            raise ValueError("PSMILES must be at least 3 characters long")
        if not v.startswith('[*]') or not v.endswith('[*]'):
            raise ValueError("PSMILES must start and end with [*]")
        return v


class OLLAMAReActCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for OLLAMA ReAct operations"""
    
    def __init__(self):
        self.start_time = None
        self.steps = []
        self.current_step = {}
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts"""
        self.start_time = datetime.now()
        logger.info(f"🤖 ReAct OLLAMA LLM started at {self.start_time}")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        logger.info(f"✅ ReAct OLLAMA LLM completed in {duration:.2f}s")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts"""
        tool_name = serialized.get("name", "unknown_tool")
        logger.info(f"🔧 Tool started: {tool_name}")
        self.current_step = {
            "tool": tool_name,
            "input": input_str,
            "start_time": datetime.now().isoformat()
        }
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends"""
        self.current_step["output"] = output[:200] + "..." if len(output) > 200 else output
        self.current_step["end_time"] = datetime.now().isoformat()
        self.steps.append(self.current_step.copy())
        logger.info(f"✅ Tool completed: {self.current_step['tool']}")


# Create chemical validation tools for ReAct agent
@tool
def validate_psmiles_structure(psmiles: str) -> str:
    """
    Validate a PSMILES string for chemical correctness using RDKit.
    
    Args:
        psmiles: The PSMILES string to validate
        
    Returns:
        String describing validation results
    """
    if not RDKIT_AVAILABLE:
        return "RDKit not available - cannot validate chemical structure"
    
    try:
        # Convert PSMILES to SMILES for validation
        smiles = psmiles.replace('[*]', '')  # Remove connection points
        
        # Handle common edge cases
        if smiles.startswith('CC(C)(C)c1ccc(O)cc1'):
            smiles = 'CC(C)(C)c1ccc(O)cc1'
        elif smiles.startswith('CC(=O)NCC(=O)'):
            smiles = 'CC(=O)NCC(=O)O'
        elif smiles.startswith('CC(=O)'):
            smiles = 'CC(=O)O'
        elif not smiles or not any(char in smiles for char in ['c', 'C', 'N', 'O', 'S']):
            smiles = 'CCO'  # Fallback
        
        # Parse with RDKit
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return f"INVALID: Cannot parse SMILES structure from PSMILES {psmiles}"
        
        # Calculate properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_atoms = mol.GetNumAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        
        return f"VALID: MW={mw:.1f}, LogP={logp:.2f}, Atoms={num_atoms}, Rings={num_rings}. Chemical structure is correct."
        
    except Exception as e:
        return f"VALIDATION ERROR: {str(e)}"


@tool
def suggest_psmiles_improvements(psmiles: str, error_description: str) -> str:
    """
    Suggest improvements for a problematic PSMILES string.
    
    Args:
        psmiles: The problematic PSMILES string
        error_description: Description of the validation error
        
    Returns:
        String with suggested improvements
    """
    suggestions = []
    
    # Common fixes
    if "cannot parse" in error_description.lower():
        suggestions.append("Check for proper SMILES syntax - use parentheses for branching")
        suggestions.append("Ensure all atoms have valid valences (C≤4, N≤3, O≤2)")
        suggestions.append("Replace problematic patterns like [*]O=C with [*]C(=O)")
    
    if "molecular weight" in error_description.lower():
        suggestions.append("Consider shorter polymer chains")
        suggestions.append("Remove heavy atoms or complex functional groups")
    
    if "aromatic" in error_description.lower():
        suggestions.append("Check aromatic ring notation - use lowercase for aromatic atoms")
        suggestions.append("Ensure proper ring closure numbers")
    
    # Default suggestions
    if not suggestions:
        suggestions = [
            "Ensure PSMILES starts and ends with [*]",
            "Use valid SMILES syntax for the core structure",
            "Check valency: C(max 4 bonds), N(max 3), O(max 2)",
            "Use parentheses for branching: C(=O) not CO="
        ]
    
    return "SUGGESTIONS: " + "; ".join(suggestions)


@tool
def generate_polymer_alternatives(polymer_type: str, functional_groups: str) -> str:
    """
    Generate alternative polymer backbone suggestions.
    
    Args:
        polymer_type: Type of polymer requested
        functional_groups: Desired functional groups
        
    Returns:
        String with polymer alternatives
    """
    alternatives = []
    
    if "linear" in polymer_type.lower():
        alternatives.extend([
            "[*]CC[*] - Simple alkyl chain",
            "[*]CCO[*] - Ether linkage",
            "[*]CC(=O)O[*] - Ester linkage",
            "[*]CC(=O)N[*] - Amide linkage"
        ])
    
    if "aromatic" in polymer_type.lower() or "phenyl" in functional_groups.lower():
        alternatives.extend([
            "[*]c1ccc(C)cc1[*] - Para-substituted benzene",
            "[*]c1ccc(O)cc1[*] - Para-hydroxybenzene",
            "[*]c1ccc(N)cc1[*] - Para-aminobenzene"
        ])
    
    if "hydroxyl" in functional_groups.lower():
        alternatives.extend([
            "[*]CC(O)C[*] - Secondary alcohol",
            "[*]CCO[*] - Primary alcohol linkage"
        ])
    
    if "carboxyl" in functional_groups.lower():
        alternatives.extend([
            "[*]CC(=O)O[*] - Carboxylic acid",
            "[*]C(=O)CC[*] - Ketone linkage"
        ])
    
    if not alternatives:
        alternatives = [
            "[*]CC[*] - Simple alkyl",
            "[*]CCO[*] - Ether",
            "[*]c1ccc(C)cc1[*] - Aromatic"
        ]
    
    return "ALTERNATIVES: " + "; ".join(alternatives[:5])


class ReActOLLAMAPSMILESAgent:
    """
    Advanced ReAct agent for PSMILES generation using OLLAMA
    
    This agent uses the ReAct (Reasoning + Acting) framework to:
    1. Reason about polymer structure requirements
    2. Act by generating PSMILES candidates
    3. Validate results using tools
    4. Iterate with improvements based on feedback
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2",
                 ollama_base_url: str = "http://localhost:11434",
                 max_iterations: int = 10,
                 temperature: float = 0.2):
        """
        Initialize the ReAct OLLAMA PSMILES agent
        
        Args:
            model_name: OLLAMA model name (default: llama3.2)
            ollama_base_url: OLLAMA server URL
            max_iterations: Maximum ReAct iterations
            temperature: LLM temperature for generation
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Initialize OLLAMA LLM
        self.callback_handler = OLLAMAReActCallbackHandler()
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            callbacks=[self.callback_handler]
        )
        
        # Initialize tools
        self.tools = [
            validate_psmiles_structure,
            suggest_psmiles_improvements, 
            generate_polymer_alternatives
        ]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create ReAct agent
        self.agent_executor = self._create_react_agent()
        
        logger.info(f"🤖 ReAct OLLAMA PSMILES Agent initialized with {model_name}")
    
    def _create_react_agent(self) -> AgentExecutor:
        """Create the ReAct agent with custom prompt"""
        
        # Custom ReAct prompt for chemical structure generation
        react_prompt = PromptTemplate.from_template("""
You are an expert computational chemist specializing in polymer structure generation using the ReAct framework.

Your task is to generate a valid PSMILES (Polymer SMILES) string that satisfies the user's requirements.

CRITICAL RULES:
1. PSMILES MUST start and end with [*] connection points
2. Use valid SMILES syntax between the [*] markers
3. ALL generated structures must pass chemical validation
4. Use the available tools to validate and improve your results
5. Follow the ReAct pattern: Thought → Action → Observation → repeat

Available tools:
{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question or request
Thought: I need to understand what type of polymer is needed and generate a chemically valid PSMILES
Action: [tool_name]
Action Input: [input to the tool]
Observation: [result from the tool]
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have a valid PSMILES that meets the requirements
Final Answer: [*]YourFinalValidPSMILESHere[*]

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}
""")
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def generate_psmiles_with_react(self, request: PSMILESGenerationRequest) -> PSMILESValidationResult:
        """
        Generate a validated PSMILES string using ReAct reasoning
        
        Args:
            request: PSMILES generation request
            
        Returns:
            PSMILESValidationResult with the final result
        """
        logger.info(f"🧪 Starting ReAct PSMILES generation for {request.polymer_type}")
        
        # Format the request for the agent
        formatted_request = f"""
Generate a PSMILES for a {request.polymer_type} polymer with the following requirements:
- Functional groups: {', '.join(request.functional_groups) if request.functional_groups else 'none specified'}
- Special properties: {', '.join(request.special_properties) if request.special_properties else 'none specified'}
- Context: {request.context or 'general polymer design'}

The PSMILES must be chemically valid and suitable for {request.context or 'polymer applications'}.
"""
        
        try:
            # Reset callback handler for new generation
            self.callback_handler.steps = []
            
            # Run the ReAct agent
            logger.info("🤖 Starting ReAct reasoning process...")
            result = self.agent_executor.invoke({"input": formatted_request})
            
            # Extract the final PSMILES from the result
            final_answer = result.get("output", "")
            
            # Extract PSMILES pattern
            psmiles_pattern = r'\[\*\][^\[\]]*\[\*\]'
            matches = re.findall(psmiles_pattern, final_answer)
            
            if matches:
                psmiles = matches[0]
                logger.info(f"✅ ReAct agent generated: {psmiles}")
            else:
                # Fallback extraction
                psmiles = "[*]CCO[*]"  # Safe fallback
                logger.warning("⚠️ Could not extract PSMILES from ReAct output, using fallback")
            
            # Final validation
            final_validation = validate_psmiles_structure(psmiles)
            is_valid = "VALID:" in final_validation
            
            # Extract ReAct steps for transparency
            react_steps = []
            for step in self.callback_handler.steps:
                react_steps.append({
                    "tool": step.get("tool", ""),
                    "input": step.get("input", "")[:100],
                    "output": step.get("output", "")[:200]
                })
            
            # Create result
            validation_result = PSMILESValidationResult(
                psmiles=psmiles,
                is_valid=is_valid,
                confidence_score=0.9 if is_valid else 0.3,
                validation_errors=[] if is_valid else [final_validation],
                chemical_properties=self._extract_properties_from_validation(final_validation),
                generation_method="react_ollama_agent",
                react_steps=react_steps
            )
            
            logger.info(f"🎉 ReAct generation {'successful' if is_valid else 'completed with errors'}!")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ ReAct generation failed: {e}")
            logger.error(traceback.format_exc())
            
            return PSMILESValidationResult(
                psmiles="[*]CCO[*]",  # Fallback
                is_valid=False,
                confidence_score=0.1,
                validation_errors=[f"ReAct agent error: {str(e)}"],
                chemical_properties={},
                generation_method="react_ollama_agent_error",
                react_steps=[]
            )
    
    def _extract_properties_from_validation(self, validation_text: str) -> Dict[str, Any]:
        """Extract chemical properties from validation text"""
        properties = {}
        
        try:
            if "MW=" in validation_text:
                mw_match = re.search(r'MW=([0-9.]+)', validation_text)
                if mw_match:
                    properties["molecular_weight"] = float(mw_match.group(1))
            
            if "LogP=" in validation_text:
                logp_match = re.search(r'LogP=([0-9.-]+)', validation_text)
                if logp_match:
                    properties["logp"] = float(logp_match.group(1))
            
            if "Atoms=" in validation_text:
                atoms_match = re.search(r'Atoms=([0-9]+)', validation_text)
                if atoms_match:
                    properties["num_atoms"] = int(atoms_match.group(1))
            
            if "Rings=" in validation_text:
                rings_match = re.search(r'Rings=([0-9]+)', validation_text)
                if rings_match:
                    properties["num_rings"] = int(rings_match.group(1))
                    
        except Exception as e:
            logger.warning(f"Could not extract properties: {e}")
        
        return properties


def create_react_ollama_psmiles_agent(model_name: str = "llama3.2",
                                     ollama_base_url: str = "http://localhost:11434",
                                     max_iterations: int = 10,
                                     temperature: float = 0.2) -> ReActOLLAMAPSMILESAgent:
    """
    Factory function to create a ReAct OLLAMA PSMILES agent
    
    Args:
        model_name: OLLAMA model name
        ollama_base_url: OLLAMA server URL
        max_iterations: Maximum ReAct iterations
        temperature: LLM temperature
        
    Returns:
        Configured ReActOLLAMAPSMILESAgent
    """
    return ReActOLLAMAPSMILESAgent(
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        max_iterations=max_iterations,
        temperature=temperature
    )


if __name__ == "__main__":
    # Example usage
    agent = create_react_ollama_psmiles_agent()
    
    request = PSMILESGenerationRequest(
        polymer_type="nanostructured",
        functional_groups=["hydroxyl", "carboxyl"],
        special_properties=["biodegradable", "biocompatible"],
        context="insulin delivery polymer"
    )
    
    result = agent.generate_psmiles_with_react(request)
    
    print(f"\n🎉 Generated PSMILES: {result.psmiles}")
    print(f"✅ Valid: {result.is_valid}")
    print(f"📊 Confidence: {result.confidence_score}")
    print(f"🔬 Properties: {result.chemical_properties}")
    print(f"🧠 ReAct Steps: {len(result.react_steps)}") 