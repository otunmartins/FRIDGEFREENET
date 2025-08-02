# Core infrastructure confirms rules are active
"""
LLM Decision Engine for Active Learning System

Provides automated decision making capabilities using LLM agents for various
choice points throughout the active learning loop.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions the engine can make."""
    LITERATURE_SEARCH_QUERY = "literature_search_query"
    PAPER_RELEVANCE = "paper_relevance" 
    MONOMER_SELECTION = "monomer_selection"
    REACTION_CONDITIONS = "reaction_conditions"
    FORCE_FIELD_SELECTION = "force_field_selection"
    SIMULATION_PARAMETERS = "simulation_parameters"
    PROPERTY_CALCULATION = "property_calculation"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    NEXT_ITERATION_PROMPT = "next_iteration_prompt"


class DecisionContext(BaseModel):
    """Context for decision making."""
    decision_type: str = Field(description="Type of decision to make")
    current_iteration: int = Field(description="Current iteration number")
    available_options: List[str] = Field(description="Available choices")
    context_data: Dict[str, Any] = Field(description="Relevant context information")
    constraints: Dict[str, Any] = Field(description="Decision constraints")
    objectives: List[str] = Field(description="Current objectives")


class DecisionOutput(BaseModel):
    """Output from decision making."""
    chosen_option: Union[str, Dict[str, Any]] = Field(description="Selected option or configuration")
    confidence: float = Field(description="Confidence score (0-1)")
    reasoning: str = Field(description="Explanation for the decision")
    alternative_options: List[str] = Field(description="Other viable options considered")


@dataclass 
class DecisionRecord:
    """Record of a decision made by the engine."""
    decision_type: DecisionType
    context: DecisionContext
    output: DecisionOutput
    timestamp: datetime
    iteration_number: int
    execution_time: float


class LLMDecisionEngine:
    """LLM-powered decision engine for active learning automation."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the decision engine.
        
        Args:
            model_name: LLM model to use for decisions
            temperature: Temperature for LLM generation
        """
        try:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            self.llm_available = True
        except Exception as e:
            logger.warning(f"LLM not available, will use fallback decisions: {e}")
            self.llm = None
            self.llm_available = False
        
        self.decision_history: List[DecisionRecord] = []
        
        # Set up parsers
        self.json_parser = JsonOutputParser(pydantic_object=DecisionOutput)
        self.str_parser = StrOutputParser()
        
        # Decision prompts
        self._setup_decision_prompts()
        
        logger.info(f"LLMDecisionEngine initialized with model {model_name}, LLM available: {self.llm_available}")
    
    def _setup_decision_prompts(self) -> None:
        """Set up decision-specific prompts."""
        
        # Literature search query generation
        self.literature_search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in scientific literature search and materials science.
            Your task is to generate optimal search queries for finding relevant papers about material discovery.
            
            Consider:
            - Specific material properties of interest
            - Synthesis methods and conditions
            - Application domains
            - Key scientific terms and synonyms
            
            Provide a search query that will find the most relevant and high-quality papers."""),
            ("human", """Current context:
            Iteration: {current_iteration}
            Target properties: {target_properties}
            Previous queries: {previous_queries}
            Objectives: {objectives}
            
            Generate an optimal search query for literature mining.
            
            {format_instructions}""")
        ])
        
        # Monomer selection prompt
        self.monomer_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert polymer chemist specializing in material design.
            Your task is to select optimal monomers for polymer synthesis based on target properties
            and previous iteration results.
            
            CRITICAL ELEMENT EXCLUSIONS:
            - NEVER recommend monomers containing silicon (Si), boron (B), aluminum (Al), or germanium (Ge)
            - EXCLUDE all silicone, siloxane, organosilicon, or silicon-based monomers
            - Use ONLY biocompatible elements: C, N, O, S, P, F, Cl, Br
            - Focus on carbon-based organic chemistry for safe biomedical applications
            
            Consider:
            - Structure-property relationships (silicon-free only)
            - Synthesis feasibility (using permitted elements only)
            - Biocompatibility (for medical applications)
            - Degradation characteristics
            - Processing conditions"""),
            ("human", """Current context:
            Iteration: {current_iteration}
            Target properties: {target_properties}
            Available monomers: {available_monomers}
            Previous results: {previous_results}
            Constraints: {constraints}
            
            Select the best monomer combination and explain your reasoning.
            IMPORTANT: Only recommend monomers using C, N, O, S, P, F, Cl, Br elements.
            
            {format_instructions}""")
        ])
        
        # Force field selection prompt
        self.forcefield_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in molecular dynamics simulations and force field selection.
            Your task is to choose the most appropriate force field for MD simulations based on
            the molecular system and target properties.
            
            Consider:
            - Molecule composition and functional groups
            - Intended application (biological, material, etc.)
            - Accuracy vs computational cost trade-offs
            - Validation and benchmarking data"""),
            ("human", """Current context:
            Iteration: {current_iteration}
            Molecules: {molecule_structures}
            Available force fields: {available_forcefields}
            Target properties: {target_properties}
            Computational resources: {computational_resources}
            
            Select the best force field and simulation parameters.
            
            {format_instructions}""")
        ])
        
        # Performance evaluation prompt
        self.performance_evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in materials science and performance evaluation.
            Your task is to assess the performance of discovered materials against target properties
            and suggest improvements for the next iteration.
            
            Consider:
            - Quantitative metrics vs targets
            - Property trade-offs and optimization
            - Feasibility and practical considerations
            - Novel insights from current results"""),
            ("human", """Current context:
            Iteration: {current_iteration}
            Computed properties: {computed_properties}
            Target properties: {target_properties}
            Previous iterations: {previous_results}
            
            Evaluate performance and suggest next iteration approach.
            
            {format_instructions}""")
        ])
    
    def make_decision(self, 
                     decision_type: DecisionType,
                     context_data: Dict[str, Any],
                     available_options: List[str] = None,
                     constraints: Dict[str, Any] = None,
                     objectives: List[str] = None) -> DecisionOutput:
        """Make a decision using the LLM.
        
        Args:
            decision_type: Type of decision to make
            context_data: Relevant context information
            available_options: List of available choices
            constraints: Decision constraints
            objectives: Current objectives
            
        Returns:
            DecisionOutput with chosen option and reasoning
        """
        start_time = datetime.now()
        
        # Create context
        context = DecisionContext(
            decision_type=decision_type.value,
            current_iteration=context_data.get("iteration", 0),
            available_options=available_options or [],
            context_data=context_data,
            constraints=constraints or {},
            objectives=objectives or []
        )
        
        # Select appropriate prompt and make decision
        if decision_type == DecisionType.LITERATURE_SEARCH_QUERY:
            output = self._make_literature_search_decision(context)
        elif decision_type == DecisionType.MONOMER_SELECTION:
            output = self._make_monomer_selection_decision(context)
        elif decision_type == DecisionType.FORCE_FIELD_SELECTION:
            output = self._make_forcefield_selection_decision(context)
        elif decision_type == DecisionType.PERFORMANCE_EVALUATION:
            output = self._make_performance_evaluation_decision(context)
        else:
            output = self._make_generic_decision(context)
        
        # Record the decision
        execution_time = (datetime.now() - start_time).total_seconds()
        record = DecisionRecord(
            decision_type=decision_type,
            context=context,
            output=output,
            timestamp=start_time,
            iteration_number=context.current_iteration,
            execution_time=execution_time
        )
        self.decision_history.append(record)
        
        logger.info(f"Made {decision_type.value} decision: {output.chosen_option} (confidence: {output.confidence})")
        return output
    
    def _make_literature_search_decision(self, context: DecisionContext) -> DecisionOutput:
        """Make literature search query decision."""
        if not self.llm_available:
            # Fallback logic when LLM is not available
            target_props = context.context_data.get("target_properties", {})
            prop_terms = " ".join(target_props.keys()) if target_props else "biocompatible"
            fallback_query = f"polymer materials {prop_terms} drug delivery"
            return DecisionOutput(
                chosen_option=fallback_query,
                confidence=0.6,
                reasoning="Generated fallback search query based on target properties",
                alternative_options=["biodegradable polymers", "drug delivery systems"]
            )
        
        try:
            prompt = self.literature_search_prompt.partial(
                format_instructions=self.json_parser.get_format_instructions()
            )
            
            chain = prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "current_iteration": context.current_iteration,
                "target_properties": context.context_data.get("target_properties", {}),
                "previous_queries": context.context_data.get("previous_queries", []),
                "objectives": context.objectives
            })
            
            return DecisionOutput(**result)
        
        except Exception as e:
            logger.error(f"Error in literature search decision: {e}")
            return DecisionOutput(
                chosen_option="polymer materials biocompatible delivery",
                confidence=0.5,
                reasoning=f"Fallback search query due to error: {e}",
                alternative_options=[]
            )
    
    def _make_monomer_selection_decision(self, context: DecisionContext) -> DecisionOutput:
        """Make monomer selection decision."""
        if not self.llm_available:
            # Fallback logic when LLM is not available
            if context.available_options:
                # Simple heuristic: prefer biocompatible monomers
                biocompatible_monomers = ["lactic_acid", "glycolic_acid", "ethylene_glycol"]
                for preferred in biocompatible_monomers:
                    if preferred in context.available_options:
                        alternatives = [opt for opt in context.available_options if opt != preferred][:2]
                        return DecisionOutput(
                            chosen_option=preferred,
                            confidence=0.7,
                            reasoning=f"Selected {preferred} - known biocompatible monomer for drug delivery",
                            alternative_options=alternatives
                        )
                # If no preferred found, use first option
                chosen = context.available_options[0]
                alternatives = context.available_options[1:3]
            else:
                chosen = "lactic_acid"
                alternatives = ["glycolic_acid", "ethylene_glycol"]
            
            return DecisionOutput(
                chosen_option=chosen,
                confidence=0.6,
                reasoning="Fallback monomer selection based on biocompatibility heuristics",
                alternative_options=alternatives
            )
        
        try:
            prompt = self.monomer_selection_prompt.partial(
                format_instructions=self.json_parser.get_format_instructions()
            )
            
            chain = prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "current_iteration": context.current_iteration,
                "target_properties": context.context_data.get("target_properties", {}),
                "available_monomers": context.available_options,
                "previous_results": context.context_data.get("previous_results", {}),
                "constraints": context.constraints
            })
            
            return DecisionOutput(**result)
        
        except Exception as e:
            logger.error(f"Error in monomer selection decision: {e}")
            # Fallback to first available option
            fallback_option = context.available_options[0] if context.available_options else "ethylene_glycol"
            return DecisionOutput(
                chosen_option=fallback_option,
                confidence=0.3,
                reasoning=f"Fallback selection due to error: {e}",
                alternative_options=context.available_options[1:3] if len(context.available_options) > 1 else []
            )
    
    def _make_forcefield_selection_decision(self, context: DecisionContext) -> DecisionOutput:
        """Make force field selection decision."""
        if not self.llm_available:
            # Fallback logic when LLM is not available
            if context.available_options:
                # Prefer GAFF for organic molecules, OpenFF for general use
                preferred_ffs = ["gaff-2.2.20", "openff-2.1.0", "amber/protein.ff14SB.xml"]
                for preferred in preferred_ffs:
                    if preferred in context.available_options:
                        alternatives = [opt for opt in context.available_options if opt != preferred][:2]
                        reasoning = f"Selected {preferred} - good general-purpose force field for organic molecules"
                        return DecisionOutput(
                            chosen_option=preferred,
                            confidence=0.8,
                            reasoning=reasoning,
                            alternative_options=alternatives
                        )
                # If no preferred found, use first option
                chosen = context.available_options[0]
                alternatives = context.available_options[1:3]
            else:
                chosen = "gaff-2.2.20"
                alternatives = ["openff-2.1.0", "amber/protein.ff14SB.xml"]
            
            return DecisionOutput(
                chosen_option=chosen,
                confidence=0.7,
                reasoning="Fallback force field selection for organic molecules",
                alternative_options=alternatives
            )
        
        try:
            prompt = self.forcefield_selection_prompt.partial(
                format_instructions=self.json_parser.get_format_instructions()
            )
            
            chain = prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "current_iteration": context.current_iteration,
                "molecule_structures": context.context_data.get("molecules", []),
                "available_forcefields": context.available_options,
                "target_properties": context.context_data.get("target_properties", {}),
                "computational_resources": context.context_data.get("computational_resources", {})
            })
            
            return DecisionOutput(**result)
        
        except Exception as e:
            logger.error(f"Error in force field selection decision: {e}")
            # Fallback to GAFF for organic molecules
            return DecisionOutput(
                chosen_option="gaff-2.2.20",
                confidence=0.6,
                reasoning=f"Fallback to GAFF 2.2.20 for organic molecules due to error: {e}",
                alternative_options=["openff-2.1.0", "amber/protein.ff14SB.xml"]
            )
    
    def _make_performance_evaluation_decision(self, context: DecisionContext) -> DecisionOutput:
        """Make performance evaluation decision."""
        if not self.llm_available:
            # Fallback logic when LLM is not available
            computed_props = context.context_data.get("computed_properties", {})
            target_props = context.context_data.get("target_properties", {})
            
            # Simple heuristic evaluation
            if computed_props and target_props:
                # Check if we're meeting targets
                meeting_targets = True
                for prop, target in target_props.items():
                    if prop in computed_props and abs(computed_props[prop] - target) > target * 0.2:
                        meeting_targets = False
                        break
                
                if meeting_targets:
                    chosen = "optimize_further"
                    reasoning = "Properties close to targets, continue optimization"
                    confidence = 0.7
                else:
                    chosen = "modify_structure"
                    reasoning = "Properties not meeting targets, structural modifications needed"
                    confidence = 0.6
            else:
                chosen = "continue_optimization"
                reasoning = "Insufficient data for evaluation, continue optimization"
                confidence = 0.5
            
            return DecisionOutput(
                chosen_option=chosen,
                confidence=confidence,
                reasoning=reasoning,
                alternative_options=["change_conditions", "try_different_monomers"]
            )
        
        try:
            prompt = self.performance_evaluation_prompt.partial(
                format_instructions=self.json_parser.get_format_instructions()
            )
            
            chain = prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "current_iteration": context.current_iteration,
                "computed_properties": context.context_data.get("computed_properties", {}),
                "target_properties": context.context_data.get("target_properties", {}),
                "previous_results": context.context_data.get("previous_results", [])
            })
            
            return DecisionOutput(**result)
        
        except Exception as e:
            logger.error(f"Error in performance evaluation decision: {e}")
            return DecisionOutput(
                chosen_option="continue_optimization",
                confidence=0.4,
                reasoning=f"Generic optimization recommendation due to error: {e}",
                alternative_options=["modify_structure", "change_conditions"]
            )
    
    def _make_generic_decision(self, context: DecisionContext) -> DecisionOutput:
        """Make a generic decision for unsupported decision types."""
        # Simple logic for unsupported decision types
        if context.available_options:
            chosen = context.available_options[0]
            alternatives = context.available_options[1:3]
        else:
            chosen = "default_option"
            alternatives = []
        
        return DecisionOutput(
            chosen_option=chosen,
            confidence=0.5,
            reasoning=f"Generic decision for {context.decision_type} - selected first available option",
            alternative_options=alternatives
        )
    
    def get_decision_history(self, decision_type: DecisionType = None) -> List[DecisionRecord]:
        """Get decision history, optionally filtered by type.
        
        Args:
            decision_type: Filter by specific decision type
            
        Returns:
            List of decision records
        """
        if decision_type:
            return [record for record in self.decision_history 
                   if record.decision_type == decision_type]
        return self.decision_history.copy()
    
    def get_decision_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for decision making.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.decision_history:
            return {"total_decisions": 0, "average_confidence": 0.0, "average_execution_time": 0.0}
        
        confidences = [record.output.confidence for record in self.decision_history]
        execution_times = [record.execution_time for record in self.decision_history]
        
        decision_type_counts = {}
        for record in self.decision_history:
            decision_type_counts[record.decision_type.value] = decision_type_counts.get(record.decision_type.value, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "average_confidence": sum(confidences) / len(confidences),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "decision_type_distribution": decision_type_counts,
            "recent_decisions": [
                {
                    "type": record.decision_type.value,
                    "confidence": record.output.confidence,
                    "timestamp": record.timestamp.isoformat()
                }
                for record in self.decision_history[-5:]  # Last 5 decisions
            ]
        }
    
    def export_decision_log(self, filepath: str) -> None:
        """Export decision history to JSON file.
        
        Args:
            filepath: Path to save the decision log
        """
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_decisions": len(self.decision_history)
            },
            "decisions": [
                {
                    "decision_type": record.decision_type.value,
                    "timestamp": record.timestamp.isoformat(),
                    "iteration": record.iteration_number,
                    "execution_time": record.execution_time,
                    "context": record.context.dict(),
                    "output": record.output.dict()
                }
                for record in self.decision_history
            ],
            "performance_metrics": self.get_decision_performance_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Decision log exported to {filepath}")


# Test functionality
def test_decision_engine():
    """Test the LLMDecisionEngine functionality."""
    print("Testing LLMDecisionEngine...")
    
    # Create decision engine
    engine = LLMDecisionEngine()
    
    # Test literature search decision
    print("\n1. Testing literature search decision...")
    lit_decision = engine.make_decision(
        decision_type=DecisionType.LITERATURE_SEARCH_QUERY,
        context_data={
            "iteration": 1,
            "target_properties": {"biocompatibility": 0.9, "degradation_rate": 0.5},
            "previous_queries": []
        },
        objectives=["Find biodegradable polymers for drug delivery"]
    )
    print(f"Literature search query: {lit_decision.chosen_option}")
    print(f"Reasoning: {lit_decision.reasoning}")
    print(f"Confidence: {lit_decision.confidence}")
    
    # Test monomer selection decision
    print("\n2. Testing monomer selection decision...")
    monomer_decision = engine.make_decision(
        decision_type=DecisionType.MONOMER_SELECTION,
        context_data={
            "iteration": 1,
            "target_properties": {"biocompatibility": 0.9},
            "previous_results": {}
        },
        available_options=["ethylene_glycol", "lactic_acid", "glycolic_acid", "caprolactone"],
        constraints={"biocompatible": True, "biodegradable": True}
    )
    print(f"Selected monomer: {monomer_decision.chosen_option}")
    print(f"Reasoning: {monomer_decision.reasoning}")
    print(f"Alternatives: {monomer_decision.alternative_options}")
    
    # Test force field selection decision
    print("\n3. Testing force field selection decision...")
    ff_decision = engine.make_decision(
        decision_type=DecisionType.FORCE_FIELD_SELECTION,
        context_data={
            "iteration": 1,
            "molecules": [{"smiles": "CCOC(=O)C", "name": "ethyl_acetate"}],
            "target_properties": {"density": 1.0},
            "computational_resources": {"max_atoms": 10000}
        },
        available_options=["gaff-2.2.20", "openff-2.1.0", "amber/protein.ff14SB.xml"]
    )
    print(f"Selected force field: {ff_decision.chosen_option}")
    print(f"Reasoning: {ff_decision.reasoning}")
    
    # Test decision history and metrics
    print("\n4. Testing decision tracking...")
    history = engine.get_decision_history()
    print(f"Total decisions made: {len(history)}")
    
    metrics = engine.get_decision_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Export decision log
    engine.export_decision_log("test_decision_log.json")
    print("Decision log exported to test_decision_log.json")
    
    print("\nLLMDecisionEngine test completed successfully!")


if __name__ == "__main__":
    test_decision_engine() 