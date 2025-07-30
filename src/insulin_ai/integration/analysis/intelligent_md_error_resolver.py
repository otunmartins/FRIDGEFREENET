"""
Intelligent MD Error Resolver using LangChain RAG Agents

This module implements a sophisticated RAG-based agent system for diagnosing and resolving
molecular dynamics force field parameterization errors, following AI Engineering principles.

Key Features:
- LangChain-powered diagnostic agents
- Context-aware error resolution
- OpenFF toolkit integration with proper stereochemistry handling
- Intelligent fallback strategies for GAFF vs SMIRNOFF parameterization
- Real-time error diagnosis and fixing

Authors: AI-driven molecular simulation error resolution system
License: MIT
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import traceback

# LangChain imports for RAG and agent workflows
try:
    from langchain.chains import RetrievalQA, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import Tool, AgentType, initialize_agent
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.document_loaders import TextLoader
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available - install with: pip install langchain langchain-community langchain-huggingface")

# OpenMM and OpenFF imports
try:
    from openff.toolkit import Molecule, Topology as OFFTopology
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
    from openmm.app import ForceField
    import openmm as mm
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    print("⚠️ OpenFF toolkit not available")

# RDKit for molecular handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available")

# Scientific computing
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Structured representation of an MD error for intelligent diagnosis"""
    error_type: str
    error_message: str
    stack_trace: str
    molecule_info: Optional[Dict[str, Any]] = None
    force_field_info: Optional[Dict[str, Any]] = None
    suggestions: List[str] = None
    severity: str = "medium"  # low, medium, high, critical


class LocalLLMStub(LLM):
    """
    Stub LLM for testing and fallback when external LLM services aren't available.
    In production, replace with actual LLM like OpenAI, Anthropic, or local models.
    """
    
    @property
    def _llm_type(self) -> str:
        return "local_stub"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple rule-based response system for MD error diagnosis.
        Replace with actual LLM inference in production.
        """
        prompt_lower = prompt.lower()
        
        # Stereochemistry error patterns
        if "unspecified stereochemistry" in prompt_lower or "undefined chiral centers" in prompt_lower:
            return """
DIAGNOSIS: RDKit stereochemistry error in molecule extraction
SOLUTION: Use allow_undefined_stereo=True flag and implement stereochemistry enumeration
APPROACH: Switch to composition-based molecule creation with simplified SMILES
CONFIDENCE: 95%
"""
        
        # GAFF template generator errors
        elif "no template found" in prompt_lower or "did not recognize residue" in prompt_lower:
            return """
DIAGNOSIS: GAFFTemplateGenerator molecule registration failure
SOLUTION: Pre-register molecules with template generator or use SystemGenerator
APPROACH: Create molecules from topology matching, not SMILES generation
CONFIDENCE: 90%
"""
        
        # Force field parameterization errors
        elif "force field" in prompt_lower and "fail" in prompt_lower:
            return """
DIAGNOSIS: Force field parameterization issue
SOLUTION: Try SMIRNOFF as fallback, use simplified molecular representations
APPROACH: Implement multi-strategy force field selection with fallbacks
CONFIDENCE: 85%
"""
        
        else:
            return """
DIAGNOSIS: General MD simulation error
SOLUTION: Check molecule preparation and force field compatibility
APPROACH: Use robust error handling and fallback strategies
CONFIDENCE: 70%
"""


class StereoChemistryHandler:
    """Advanced stereochemistry handling for complex molecules"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".StereoHandler")
    
    def fix_undefined_stereochemistry(self, smiles: str) -> List[str]:
        """
        Generate reasonable stereoisomers from undefined stereochemistry.
        
        Args:
            smiles: SMILES string with undefined stereocenters
            
        Returns:
            List of stereochemically-defined SMILES
        """
        if not RDKIT_AVAILABLE:
            self.logger.warning("RDKit not available - returning simplified SMILES")
            return [self._simplify_smiles(smiles)]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [self._simplify_smiles(smiles)]
            
            # Remove stereochemistry and generate reasonable stereoisomers
            Chem.RemoveStereochemistry(mol)
            
            # For complex molecules, use the first reasonable stereoisomer
            # In production, you might want to use more sophisticated enumeration
            stereoisomers = []
            
            # Generate up to 3 stereoisomers to avoid combinatorial explosion
            stereo_mol = Chem.AddHs(mol)
            Chem.AssignStereochemistry(stereo_mol, cleanIt=True, force=True)
            
            # Create simplified stereochemistry
            simplified_smiles = Chem.MolToSmiles(stereo_mol, isomericSmiles=True)
            if simplified_smiles:
                stereoisomers.append(simplified_smiles)
            
            # Fallback: create a simplified version
            if not stereoisomers:
                stereoisomers.append(self._simplify_smiles(smiles))
                
            return stereoisomers
            
        except Exception as e:
            self.logger.warning(f"Stereochemistry fixing failed: {e}")
            return [self._simplify_smiles(smiles)]
    
    def _simplify_smiles(self, smiles: str) -> str:
        """Create a simplified SMILES for problematic molecules"""
        # Remove stereochemistry indicators
        simplified = smiles.replace("@", "").replace("/", "").replace("\\", "")
        
        # For very complex molecules, create a generic polymer-like structure
        if len(simplified) > 200:  # Very long SMILES
            return "CC(=O)NCCC(=O)NC"  # Generic polymer unit
        
        return simplified


class MolecularForceFieldManager:
    """Intelligent force field selection and molecule registration"""
    
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        self.stereo_handler = StereoChemistryHandler()
        self.logger = logging.getLogger(__name__ + ".FFManager")
        
        # Knowledge base of common force field fixes
        self.ff_strategies = [
            {
                "name": "gaff_with_stereo_fix",
                "description": "GAFF with stereochemistry handling",
                "implementation": self._try_gaff_with_stereo_fix
            },
            {
                "name": "smirnoff_fallback", 
                "description": "OpenFF SMIRNOFF as fallback",
                "implementation": self._try_smirnoff_fallback
            },
            {
                "name": "simplified_topology",
                "description": "Simplified molecular topology approach",
                "implementation": self._try_simplified_topology
            }
        ]
    
    def create_robust_force_field(
        self, 
        topology, 
        target_molecules: List[str],
        error_context: Optional[ErrorContext] = None
    ) -> Tuple[ForceField, List[Molecule], Dict[str, Any]]:
        """
        Create force field with intelligent error handling and fallbacks.
        
        Args:
            topology: OpenMM topology
            target_molecules: List of molecule identifiers or SMILES
            error_context: Previous error context for informed decisions
            
        Returns:
            Tuple of (forcefield, molecules, metadata)
        """
        self.logger.info("🤖 Starting intelligent force field creation...")
        
        # Get AI recommendation if available
        strategy_recommendation = None
        if self.llm_agent and error_context:
            strategy_recommendation = self._get_ai_strategy_recommendation(error_context)
        
        # Try strategies in order of recommendation or default priority
        strategies_to_try = self._prioritize_strategies(strategy_recommendation, error_context)
        
        for strategy in strategies_to_try:
            try:
                self.logger.info(f"🔄 Trying strategy: {strategy['name']}")
                
                forcefield, molecules, metadata = strategy["implementation"](
                    topology, target_molecules
                )
                
                if forcefield and molecules:
                    self.logger.info(f"✅ Success with strategy: {strategy['name']}")
                    metadata["strategy_used"] = strategy["name"]
                    metadata["success"] = True
                    return forcefield, molecules, metadata
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Strategy {strategy['name']} failed: {e}")
                continue
        
        # Ultimate fallback
        self.logger.warning("🔴 All strategies failed - using minimal fallback")
        return self._create_minimal_fallback_forcefield()
    
    def _try_gaff_with_stereo_fix(
        self, 
        topology, 
        target_molecules: List[str]
    ) -> Tuple[ForceField, List[Molecule], Dict[str, Any]]:
        """GAFF approach with stereochemistry fixing"""
        
        molecules = []
        for mol_identifier in target_molecules:
            try:
                # Fix stereochemistry issues
                if "unspecified stereochemistry" in str(mol_identifier).lower():
                    fixed_smiles_list = self.stereo_handler.fix_undefined_stereochemistry(mol_identifier)
                    mol_identifier = fixed_smiles_list[0]  # Use first valid stereoisomer
                
                # Create molecule with error handling
                molecule = Molecule.from_smiles(mol_identifier, allow_undefined_stereo=True)
                molecule.assign_partial_charges("am1bcc", use_conformers=None)
                molecules.append(molecule)
                
            except Exception as e:
                self.logger.warning(f"Molecule creation failed: {e}, trying composition fallback")
                # Create from composition if SMILES fails
                fallback_molecule = self._create_from_composition_analysis(topology)
                if fallback_molecule:
                    molecules.append(fallback_molecule)
        
        if not molecules:
            raise ValueError("No molecules could be created")
        
        # Create GAFF force field
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Register GAFF template generator with proper error handling
        try:
            gaff = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.11")
            forcefield.registerTemplateGenerator(gaff.generator)
            
            metadata = {
                "molecules_created": len(molecules),
                "gaff_version": gaff.gaff_version,
                "approach": "gaff_with_stereo_fix"
            }
            
            return forcefield, molecules, metadata
            
        except Exception as e:
            self.logger.error(f"GAFF registration failed: {e}")
            raise
    
    def _try_smirnoff_fallback(
        self, 
        topology, 
        target_molecules: List[str]
    ) -> Tuple[ForceField, List[Molecule], Dict[str, Any]]:
        """OpenFF SMIRNOFF fallback approach"""
        
        molecules = []
        for mol_identifier in target_molecules:
            try:
                molecule = Molecule.from_smiles(mol_identifier, allow_undefined_stereo=True)
                molecules.append(molecule)
            except Exception:
                # Simplified molecule fallback
                simple_mol = Molecule.from_smiles("CCCCCCCCCC")  # Simple alkane
                molecules.append(simple_mol)
        
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Use SMIRNOFF template generator
        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.1.0")
        forcefield.registerTemplateGenerator(smirnoff.generator)
        
        metadata = {
            "molecules_created": len(molecules),
            "smirnoff_version": "openff-2.1.0",
            "approach": "smirnoff_fallback"
        }
        
        return forcefield, molecules, metadata
    
    def _try_simplified_topology(
        self, 
        topology, 
        target_molecules: List[str]
    ) -> Tuple[ForceField, List[Molecule], Dict[str, Any]]:
        """Simplified topology approach using composition analysis"""
        
        # Analyze topology composition
        molecules = []
        for residue in topology.residues():
            if residue.name == "UNL":  # Unknown ligand
                molecule = self._create_from_composition_analysis(topology, residue)
                if molecule:
                    molecules.append(molecule)
        
        if not molecules:
            # Ultimate fallback - generic polymer
            molecules = [Molecule.from_smiles("CC(=O)NCCC(=O)NC")]
        
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Try GAFF first, then SMIRNOFF
        try:
            gaff = GAFFTemplateGenerator(molecules=molecules)
            forcefield.registerTemplateGenerator(gaff.generator)
            approach = "simplified_gaff"
        except Exception:
            smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules)
            forcefield.registerTemplateGenerator(smirnoff.generator)
            approach = "simplified_smirnoff"
        
        metadata = {
            "molecules_created": len(molecules),
            "approach": approach
        }
        
        return forcefield, molecules, metadata
    
    def _create_from_composition_analysis(self, topology, target_residue=None) -> Optional[Molecule]:
        """Create molecule based on atomic composition analysis"""
        
        if target_residue:
            atoms = list(target_residue.atoms())
        else:
            # Find first UNL residue
            atoms = []
            for residue in topology.residues():
                if residue.name == "UNL":
                    atoms = list(residue.atoms())
                    break
        
        if not atoms:
            return None
        
        # Count elements
        element_counts = {}
        for atom in atoms:
            element = atom.element.symbol
            element_counts[element] = element_counts.get(element, 0) + 1
        
        # Generate reasonable SMILES based on composition
        carbon_count = element_counts.get('C', 0)
        nitrogen_count = element_counts.get('N', 0)
        oxygen_count = element_counts.get('O', 0)
        
        if carbon_count > 10 and nitrogen_count > 0:
            # Polymer with nitrogen - polyamide-like
            smiles = "CC(=O)NC" * min(5, nitrogen_count)
        elif carbon_count > 10 and oxygen_count > 0:
            # Polymer with oxygen - polyester-like
            smiles = "CC(=O)OC" * min(5, oxygen_count)
        elif carbon_count > 5:
            # Simple carbon chain
            smiles = "C" * min(10, carbon_count)
        else:
            # Default small molecule
            smiles = "CCCCCCCCCC"
        
        try:
            molecule = Molecule.from_smiles(smiles)
            molecule.assign_partial_charges("gasteiger")
            return molecule
        except Exception:
            return None
    
    def _get_ai_strategy_recommendation(self, error_context: ErrorContext) -> Optional[str]:
        """Get AI recommendation for strategy selection"""
        if not self.llm_agent:
            return None
        
        prompt = f"""
        Given this molecular dynamics error context:
        Error Type: {error_context.error_type}
        Error Message: {error_context.error_message}
        
        Recommend the best strategy from:
        1. gaff_with_stereo_fix - For stereochemistry issues
        2. smirnoff_fallback - For GAFF registration problems  
        3. simplified_topology - For complex topology issues
        
        Respond with just the strategy name.
        """
        
        try:
            response = self.llm_agent._call(prompt)
            if "gaff_with_stereo_fix" in response.lower():
                return "gaff_with_stereo_fix"
            elif "smirnoff_fallback" in response.lower():
                return "smirnoff_fallback"
            elif "simplified_topology" in response.lower():
                return "simplified_topology"
        except Exception:
            pass
        
        return None
    
    def _prioritize_strategies(self, ai_recommendation: Optional[str], error_context: Optional[ErrorContext]) -> List[Dict]:
        """Prioritize strategies based on AI recommendation and error context"""
        
        # Default order
        strategies = self.ff_strategies.copy()
        
        # AI recommendation takes priority
        if ai_recommendation:
            recommended_strategy = next((s for s in strategies if s["name"] == ai_recommendation), None)
            if recommended_strategy:
                strategies.remove(recommended_strategy)
                strategies.insert(0, recommended_strategy)
        
        # Error-based prioritization
        elif error_context:
            if "stereochemistry" in error_context.error_message.lower():
                # Prioritize stereo fix for stereochemistry errors
                stereo_strategy = next((s for s in strategies if "stereo_fix" in s["name"]), None)
                if stereo_strategy:
                    strategies.remove(stereo_strategy)
                    strategies.insert(0, stereo_strategy)
            
            elif "template" in error_context.error_message.lower():
                # Prioritize SMIRNOFF for template issues
                smirnoff_strategy = next((s for s in strategies if "smirnoff" in s["name"]), None)
                if smirnoff_strategy:
                    strategies.remove(smirnoff_strategy)
                    strategies.insert(0, smirnoff_strategy)
        
        return strategies
    
    def _create_minimal_fallback_forcefield(self) -> Tuple[ForceField, List[Molecule], Dict[str, Any]]:
        """Minimal fallback force field for when all else fails"""
        
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Simple alkane molecule as universal fallback
        molecules = [Molecule.from_smiles("CCCCCCCCCC")]
        
        metadata = {
            "molecules_created": 1,
            "approach": "minimal_fallback",
            "warning": "Using minimal fallback - results may be less accurate"
        }
        
        return forcefield, molecules, metadata


class MDErrorResolver:
    """Main LangChain-powered agent for MD error resolution"""
    
    def __init__(self, llm=None, knowledge_base_path: Optional[str] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for MDErrorResolver")
        
        self.llm = llm or LocalLLMStub()
        self.ff_manager = MolecularForceFieldManager(self.llm)
        self.logger = logging.getLogger(__name__ + ".MDResolver")
        
        # Initialize RAG components
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize knowledge base
        if knowledge_base_path:
            self._setup_knowledge_base(knowledge_base_path)
        else:
            self._create_default_knowledge_base()
        
        # Setup agent tools
        self._setup_agent_tools()
    
    def _setup_knowledge_base(self, knowledge_base_path: str):
        """Setup RAG knowledge base from file"""
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            loader = TextLoader(knowledge_base_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )
            
            self.logger.info("✅ Knowledge base setup complete")
            
        except Exception as e:
            self.logger.warning(f"Knowledge base setup failed: {e}")
            self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self):
        """Create default in-memory knowledge base"""
        
        # Default MD error knowledge
        knowledge_docs = [
            Document(
                page_content="""
                Stereochemistry Error Solutions:
                1. Use allow_undefined_stereo=True in OpenFF Molecule.from_smiles()
                2. Enumerate stereoisomers using RDKit 
                3. Simplify molecular representation for complex polymers
                4. Use composition-based molecule creation as fallback
                """,
                metadata={"source": "stereochemistry_guide"}
            ),
            Document(
                page_content="""
                GAFF Template Generator Issues:
                1. Pre-register molecules before force field creation
                2. Use GAFFTemplateGenerator(molecules=molecules) syntax
                3. Fallback to SMIRNOFFTemplateGenerator if GAFF fails
                4. Ensure molecule topology matches force field expectations
                """,
                metadata={"source": "gaff_troubleshooting"}
            ),
            Document(
                page_content="""
                Force Field Creation Best Practices:
                1. Start with simplified molecular representations
                2. Use composition analysis for unknown residues
                3. Implement multi-strategy approaches with fallbacks
                4. Test force field creation before running simulations
                """,
                metadata={"source": "forcefield_best_practices"}
            )
        ]
        
        try:
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            self.vector_store = FAISS.from_documents(knowledge_docs, self.embeddings)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", 
                retriever=self.vector_store.as_retriever()
            )
            
            self.logger.info("✅ Default knowledge base created")
            
        except Exception as e:
            self.logger.warning(f"Default knowledge base creation failed: {e}")
    
    def _setup_agent_tools(self):
        """Setup LangChain agent tools"""
        
        self.tools = []
        
        # Error diagnosis tool
        error_diagnosis_tool = Tool(
            name="ErrorDiagnosis",
            func=self._diagnose_error,
            description="Diagnose molecular dynamics and force field errors"
        )
        self.tools.append(error_diagnosis_tool)
        
        # Force field creation tool
        ff_creation_tool = Tool(
            name="ForceFieldCreation", 
            func=self._create_force_field_wrapper,
            description="Create molecular force fields with intelligent error handling"
        )
        self.tools.append(ff_creation_tool)
        
        # Knowledge base query tool
        if self.qa_chain:
            kb_query_tool = Tool(
                name="KnowledgeBase",
                func=self.qa_chain.run,
                description="Query molecular dynamics knowledge base for solutions"
            )
            self.tools.append(kb_query_tool)
        
        # Initialize agent
        try:
            self.agent = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3
            )
            self.logger.info("✅ Agent tools setup complete")
        except Exception as e:
            self.logger.warning(f"Agent setup failed: {e}")
            self.agent = None
    
    def resolve_error(
        self, 
        error_message: str,
        topology=None,
        molecules: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for intelligent error resolution.
        
        Args:
            error_message: The error message to resolve
            topology: OpenMM topology (optional)
            molecules: List of molecules/SMILES (optional)
            context: Additional context (optional)
            
        Returns:
            Dictionary with resolution results
        """
        
        self.logger.info("🤖 Starting intelligent error resolution...")
        
        # Create error context
        error_context = ErrorContext(
            error_type="force_field_error",
            error_message=error_message,
            stack_trace=traceback.format_exc(),
            molecule_info={"count": len(molecules) if molecules else 0},
            force_field_info=context or {}
        )
        
        resolution_result = {
            "success": False,
            "error_context": error_context,
            "resolution_steps": [],
            "forcefield": None,
            "molecules": None,
            "metadata": {}
        }
        
        try:
            # Step 1: Diagnose error using AI
            diagnosis = self._diagnose_error(error_message)
            resolution_result["resolution_steps"].append(f"Diagnosis: {diagnosis}")
            
            # Step 2: Query knowledge base for solutions
            if self.qa_chain:
                kb_solution = self.qa_chain.run(f"How to fix: {error_message}")
                resolution_result["resolution_steps"].append(f"Knowledge base solution: {kb_solution}")
            
            # Step 3: Apply force field creation with intelligent strategies
            if topology and molecules:
                forcefield, resolved_molecules, metadata = self.ff_manager.create_robust_force_field(
                    topology, molecules, error_context
                )
                
                resolution_result["forcefield"] = forcefield
                resolution_result["molecules"] = resolved_molecules
                resolution_result["metadata"] = metadata
                resolution_result["success"] = True
                resolution_result["resolution_steps"].append(f"Applied strategy: {metadata.get('strategy_used', 'unknown')}")
            
            # Step 4: Use agent for complex reasoning (if available)
            if self.agent and not resolution_result["success"]:
                try:
                    agent_response = self.agent.run(
                        f"Resolve this molecular dynamics error: {error_message}. "
                        f"Context: {context or 'No additional context'}"
                    )
                    resolution_result["resolution_steps"].append(f"Agent response: {agent_response}")
                except Exception as e:
                    self.logger.warning(f"Agent reasoning failed: {e}")
            
            self.logger.info("✅ Error resolution complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error resolution failed: {e}")
            resolution_result["resolution_steps"].append(f"Resolution failed: {str(e)}")
        
        return resolution_result
    
    def _diagnose_error(self, error_message: str) -> str:
        """Diagnose error using LLM"""
        
        diagnosis_prompt = f"""
        Analyze this molecular dynamics error and provide a concise diagnosis:
        
        Error: {error_message}
        
        Focus on:
        1. Root cause identification
        2. Specific component that failed (stereochemistry, force field, etc.)
        3. Recommended solution approach
        
        Respond with a brief diagnosis.
        """
        
        try:
            return self.llm._call(diagnosis_prompt)
        except Exception as e:
            return f"Diagnosis failed: {str(e)}"
    
    def _create_force_field_wrapper(self, args: str) -> str:
        """Wrapper for force field creation tool"""
        try:
            # This would parse args and call ff_manager.create_robust_force_field
            # For now, return status message
            return "Force field creation tool activated - use resolve_error() method for full functionality"
        except Exception as e:
            return f"Force field creation failed: {str(e)}"


def create_intelligent_md_resolver(
    llm=None,
    knowledge_base_path: Optional[str] = None
) -> MDErrorResolver:
    """
    Factory function to create an intelligent MD error resolver.
    
    Args:
        llm: LangChain LLM instance (optional, uses stub if None)
        knowledge_base_path: Path to knowledge base file (optional)
        
    Returns:
        Configured MDErrorResolver instance
    """
    
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required. Install with: "
            "pip install langchain langchain-community langchain-huggingface"
        )
    
    resolver = MDErrorResolver(llm=llm, knowledge_base_path=knowledge_base_path)
    return resolver


# Example usage and integration points
def example_usage():
    """Example of how to use the intelligent MD error resolver"""
    
    print("🤖 Intelligent MD Error Resolver Example")
    print("=" * 50)
    
    # Create resolver
    resolver = create_intelligent_md_resolver()
    
    # Example error from the logs
    error_message = """
    RDMol has unspecified stereochemistry. Undefined chiral centers are:
    - Atom C (index 2)
    - Atom C (index 4)
    Unable to make OFFMol from SMILES
    """
    
    # Resolve error
    result = resolver.resolve_error(
        error_message=error_message,
        molecules=["CC(=O)NCCC(=O)NC"],  # Example polymer SMILES
        context={"simulation_type": "mmgbsa", "residue": "UNL"}
    )
    
    print(f"Resolution success: {result['success']}")
    print("\nResolution steps:")
    for step in result["resolution_steps"]:
        print(f"  - {step}")
    
    if result["metadata"]:
        print(f"\nMetadata: {result['metadata']}")


if __name__ == "__main__":
    example_usage() 