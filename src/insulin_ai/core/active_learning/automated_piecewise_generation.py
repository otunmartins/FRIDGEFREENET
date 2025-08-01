#!/usr/bin/env python3
"""
AutomatedPiecewiseGeneration - Phase 2 Implementation

This module provides automated piecewise generation with LLM-powered decision making
for the active learning material discovery system. It integrates existing
PSMILES generation components with intelligent automation.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Import existing piecewise generation systems
try:
    from ..psmiles_generator import PSMILESGenerator
    PSMILES_GENERATOR_AVAILABLE = True
except ImportError:
    PSMILES_GENERATOR_AVAILABLE = False
    logging.warning("PSMILESGenerator not available")

try:
    from ..psmiles_processor import PSMILESProcessor
    PSMILES_PROCESSOR_AVAILABLE = True
except ImportError:
    PSMILES_PROCESSOR_AVAILABLE = False
    logging.warning("PSMILESProcessor not available")

try:
    from ..psmiles_diversification import CandidateOrchestrator
    PSMILES_DIVERSIFIER_AVAILABLE = True
except ImportError:
    PSMILES_DIVERSIFIER_AVAILABLE = False
    logging.warning("CandidateOrchestrator not available")

# Import active learning infrastructure
from .state_manager import IterationState, LiteratureResults, GeneratedMolecules
from .decision_engine import LLMDecisionEngine, DecisionType

logger = logging.getLogger(__name__)


class GenerationContext:
    """Context data for piecewise generation decisions."""
    
    def __init__(self, iteration: int, target_properties: Dict[str, float],
                 literature_results: Optional[LiteratureResults] = None,
                 previous_molecules: List[Dict] = None):
        self.iteration = iteration
        self.target_properties = target_properties or {}
        self.literature_results = literature_results
        self.previous_molecules = previous_molecules or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for decision engine."""
        return {
            "iteration": self.iteration,
            "target_properties": self.target_properties,
            "literature_insights": self._extract_literature_insights(),
            "previous_molecules_count": len(self.previous_molecules),
            "material_candidates": self.literature_results.material_candidates if self.literature_results else [],
            "synthesis_routes": self.literature_results.synthesis_routes if self.literature_results else [],
            "timestamp": self.timestamp.isoformat()
        }
    
    def _extract_literature_insights(self) -> Dict[str, Any]:
        """Extract key insights from literature results."""
        if not self.literature_results:
            return {}
        
        return {
            "relevant_papers": self.literature_results.relevant_papers,
            "extracted_properties": list(self.literature_results.extracted_properties.keys()),
            "synthesis_routes": self.literature_results.synthesis_routes,
            "material_candidates": self.literature_results.material_candidates or []
        }


class AutomatedPiecewiseGeneration:
    """
    Automated piecewise generation with LLM-powered decision making.
    
    This class integrates existing PSMILES generation systems with intelligent
    automation for monomer selection, reaction condition optimization,
    molecular weight targeting, and functional group optimization.
    """
    
    def __init__(self, storage_path: str = "automated_piecewise_generation"):
        """Initialize automated piecewise generation system.
        
        Args:
            storage_path: Path to store generation data and results
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize PSMILES generation systems
        self._initialize_generation_systems()
        
        # Cache for results and decisions
        self._results_cache = {}
        self._decision_cache = {}
        
        logger.info("AutomatedPiecewiseGeneration initialized")
    
    def _initialize_generation_systems(self):
        """Initialize available PSMILES generation systems."""
        # Initialize PSMILES Generator
        if PSMILES_GENERATOR_AVAILABLE:
            try:
                self.psmiles_generator = PSMILESGenerator(
                    model_type="openai",
                    openai_model="gpt-4o-mini",  # More cost-effective
                    temperature=0.7  # Higher temperature for creativity
                )
                logger.info("PSMILESGenerator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PSMILESGenerator: {e}")
                self.psmiles_generator = None
        else:
            self.psmiles_generator = None
        
        # Initialize PSMILES Processor
        if PSMILES_PROCESSOR_AVAILABLE:
            try:
                self.psmiles_processor = PSMILESProcessor()
                logger.info("PSMILESProcessor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PSMILESProcessor: {e}")
                self.psmiles_processor = None
        else:
            self.psmiles_processor = None
        
        # Initialize PSMILES Diversifier
        if PSMILES_DIVERSIFIER_AVAILABLE:
            try:
                self.psmiles_diversifier = CandidateOrchestrator()
                logger.info("CandidateOrchestrator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CandidateOrchestrator: {e}")
                self.psmiles_diversifier = None
        else:
            self.psmiles_diversifier = None
    
    async def run_automated_generation(self, state: IterationState, 
                                     decision_engine: LLMDecisionEngine) -> GeneratedMolecules:
        """
        Run automated piecewise generation with LLM decision making.
        
        Args:
            state: Current iteration state
            decision_engine: LLM decision engine for automation
            
        Returns:
            GeneratedMolecules: Comprehensive generation results
        """
        logger.info(f"Starting automated piecewise generation for iteration {state.iteration_number}")
        
        try:
            # Step 1: Analyze literature results and create generation strategy
            generation_context = GenerationContext(
                iteration=state.iteration_number,
                target_properties=state.target_properties,
                literature_results=state.literature_results,
                previous_molecules=self._extract_previous_molecules(state)
            )
            
            # Step 2: Select optimal monomer combinations
            monomer_strategy = await self._select_monomer_strategy(generation_context, decision_engine)
            
            # Step 3: Optimize reaction conditions and synthesis parameters
            synthesis_parameters = await self._optimize_synthesis_parameters(
                generation_context, monomer_strategy, decision_engine
            )
            
            # Step 4: Generate polymer structures
            generated_structures = await self._generate_polymer_structures(
                generation_context, monomer_strategy, synthesis_parameters
            )
            
            # Step 5: Apply diversification and optimization
            optimized_molecules = await self._optimize_and_diversify(
                generated_structures, generation_context, decision_engine
            )
            
            # Step 6: Validate and score generated molecules
            final_results = await self._validate_and_score_molecules(
                optimized_molecules, generation_context, decision_engine
            )
            
            # Step 7: Save results and update cache
            self._save_results(state.iteration_number, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in automated piecewise generation: {e}")
            # Return minimal results to prevent pipeline failure
            return GeneratedMolecules(
                monomers_generated=["error"],
                psmiles_strings=["[C][C][O]"],  # Basic polymer as fallback
                generation_method=f"Error fallback: {str(e)}",
                diversity_score=0.0,
                validity_score=0.0,
                execution_time=0.0
            )
    
    async def _select_monomer_strategy(self, context: GenerationContext, 
                                     decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Select optimal monomer combinations using LLM decision making."""
        
        # Extract material candidates from literature
        literature_materials = context.literature_results.material_candidates if context.literature_results else []
        
        # Define monomer categories based on literature and target properties
        monomer_categories = [
            "biocompatible_polymers",  # PEG, PLGA, chitosan
            "biodegradable_polymers",  # PLA, PCL, PDLLA
            "hydrophilic_monomers",    # acrylic acid, HEMA
            "hydrophobic_monomers",    # styrene, MMA
            "functional_monomers"      # maleic anhydride, vinyl acetate
        ]
        
        # Generate monomer selection decision
        monomer_decision = decision_engine.make_decision(
            decision_type=DecisionType.MONOMER_SELECTION,
            context_data=context.to_dict(),
            available_options=monomer_categories,
            objectives=["Select monomers that optimize target properties"],
            constraints={"biocompatible": True, "literature_informed": True}
        )
        
        # Map decision to specific monomers
        monomer_mapping = {
            "biocompatible_polymers": ["ethylene_glycol", "lactic_acid", "glycolic_acid"],
            "biodegradable_polymers": ["lactic_acid", "caprolactone", "glycolic_acid"],
            "hydrophilic_monomers": ["acrylic_acid", "methacrylic_acid", "vinyl_alcohol"],
            "hydrophobic_monomers": ["styrene", "methyl_methacrylate", "vinyl_acetate"],
            "functional_monomers": ["maleic_anhydride", "vinyl_acetate", "acrylamide"]
        }
        
        selected_monomers = monomer_mapping.get(monomer_decision.chosen_option, ["ethylene_glycol", "lactic_acid"])
        
        # Incorporate literature-suggested materials
        if literature_materials:
            selected_monomers.extend(literature_materials[:2])  # Add top 2 from literature
        
        return {
            "strategy": monomer_decision.chosen_option,
            "primary_monomers": selected_monomers,
            "reasoning": monomer_decision.reasoning,
            "literature_influence": len(literature_materials) > 0
        }
    
    async def _optimize_synthesis_parameters(self, context: GenerationContext,
                                           monomer_strategy: Dict[str, Any],
                                           decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Optimize reaction conditions and synthesis parameters."""
        
        # Generate synthesis optimization decision
        synthesis_decision = decision_engine.make_decision(
            decision_type=DecisionType.SYNTHESIS_CONDITIONS,
            context_data={
                **context.to_dict(),
                "monomer_strategy": monomer_strategy
            },
            available_options=["mild_conditions", "controlled_conditions", "optimized_conditions"],
            objectives=["Ensure successful polymerization while maintaining properties"],
            constraints={"temperature_limit": 80, "biocompatible_solvents": True}
        )
        
        # Map decision to specific parameters
        parameter_mapping = {
            "mild_conditions": {
                "temperature": 25,
                "pressure": 1.0,
                "catalyst": "organic",
                "molecular_weight_target": "medium"
            },
            "controlled_conditions": {
                "temperature": 60,
                "pressure": 1.0,
                "catalyst": "metal_organic",
                "molecular_weight_target": "high"
            },
            "optimized_conditions": {
                "temperature": 80,
                "pressure": 1.2,
                "catalyst": "advanced",
                "molecular_weight_target": "targeted"
            }
        }
        
        base_params = parameter_mapping.get(synthesis_decision.chosen_option, parameter_mapping["controlled_conditions"])
        
        # Adjust based on target properties
        if "degradation_rate" in context.target_properties:
            target_degradation = context.target_properties["degradation_rate"]
            if target_degradation > 0.7:  # Fast degradation
                base_params["molecular_weight_target"] = "low"
            elif target_degradation < 0.3:  # Slow degradation
                base_params["molecular_weight_target"] = "high"
        
        return {
            **base_params,
            "synthesis_strategy": synthesis_decision.chosen_option,
            "reasoning": synthesis_decision.reasoning
        }
    
    async def _generate_polymer_structures(self, context: GenerationContext,
                                         monomer_strategy: Dict[str, Any],
                                         synthesis_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate polymer structures using available generation systems."""
        
        generated_structures = []
        
        # Try PSMILES Generator first (most advanced)
        if self.psmiles_generator:
            try:
                for monomer in monomer_strategy["primary_monomers"][:3]:  # Limit to 3 monomers
                    polymer_description = self._create_polymer_description(
                        monomer, context.target_properties, synthesis_parameters
                    )
                    
                    # Generate polymer SMILES
                    generation_result = await self._generate_with_psmiles_generator(
                        polymer_description, synthesis_parameters
                    )
                    
                    if generation_result:
                        generated_structures.extend(generation_result)
                
                logger.info(f"PSMILES generator produced {len(generated_structures)} structures")
            except Exception as e:
                logger.warning(f"PSMILES generation failed: {e}")
        
        # Fallback to basic generation if needed
        if len(generated_structures) < 3:
            fallback_structures = self._generate_fallback_structures(
                monomer_strategy, synthesis_parameters
            )
            generated_structures.extend(fallback_structures)
        
        return generated_structures[:5]  # Limit to 5 structures
    
    def _create_polymer_description(self, monomer: str, target_properties: Dict[str, float],
                                  synthesis_parameters: Dict[str, Any]) -> str:
        """Create natural language description for polymer generation."""
        
        description = f"Design a polymer from {monomer} monomer"
        
        # Add property requirements
        if target_properties:
            prop_descriptions = []
            for prop, value in target_properties.items():
                if prop == "biocompatibility" and value > 0.8:
                    prop_descriptions.append("highly biocompatible")
                elif prop == "degradation_rate" and value > 0.6:
                    prop_descriptions.append("fast degrading")
                elif prop == "degradation_rate" and value < 0.4:
                    prop_descriptions.append("stable")
                elif "mechanical" in prop.lower() and value > 0.7:
                    prop_descriptions.append("mechanically strong")
            
            if prop_descriptions:
                description += f" that is {', '.join(prop_descriptions)}"
        
        # Add synthesis context
        molecular_weight = synthesis_parameters.get("molecular_weight_target", "medium")
        description += f" with {molecular_weight} molecular weight"
        
        # Add application context
        description += " suitable for insulin delivery applications"
        
        return description
    
    async def _generate_with_psmiles_generator(self, description: str, 
                                             parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate polymers using the enhanced PSMILES generator."""
        try:
            # Generate PSMILES
            result = self.psmiles_generator.generate_polymer_smiles(
                natural_language_description=description,
                num_variants=3
            )
            
            structures = []
            if result and "generated_polymers" in result:
                for i, polymer in enumerate(result["generated_polymers"][:3]):
                    structures.append({
                        "id": f"gen_{i+1}",
                        "psmiles": polymer.get("psmiles", ""),
                        "description": description,
                        "generation_method": "psmiles_generator",
                        "parameters": parameters,
                        "confidence": polymer.get("confidence", 0.5)
                    })
            
            return structures
        except Exception as e:
            logger.warning(f"PSMILES generator failed for description '{description}': {e}")
            return []
    
    def _generate_fallback_structures(self, monomer_strategy: Dict[str, Any],
                                    synthesis_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback polymer structures when main generator fails."""
        
        # Basic polymer SMILES for common monomers
        basic_polymers = {
            "ethylene_glycol": "[C][C][O]",
            "lactic_acid": "[C][C](=O)[O]",
            "glycolic_acid": "[C](=O)[O]",
            "acrylic_acid": "[C][C](=O)[O]",
            "styrene": "c1ccccc1[C][C]",
            "vinyl_acetate": "[C][C](=O)O[C]"
        }
        
        structures = []
        for i, monomer in enumerate(monomer_strategy["primary_monomers"][:3]):
            psmiles = basic_polymers.get(monomer.lower(), "[C][C][O]")  # Default to PEG-like
            
            structures.append({
                "id": f"fallback_{i+1}",
                "psmiles": psmiles,
                "description": f"Basic polymer from {monomer}",
                "generation_method": "fallback",
                "parameters": synthesis_parameters,
                "confidence": 0.3
            })
        
        return structures
    
    async def _optimize_and_diversify(self, structures: List[Dict[str, Any]],
                                    context: GenerationContext,
                                    decision_engine: LLMDecisionEngine) -> List[Dict[str, Any]]:
        """Apply diversification and optimization to generated structures."""
        
        # Generate diversification strategy decision
        diversification_decision = decision_engine.make_decision(
            decision_type=DecisionType.MOLECULAR_DIVERSIFICATION,
            context_data=context.to_dict(),
            available_options=["structural_diversity", "functional_diversity", "balanced_diversity"],
            objectives=["Increase molecular diversity while maintaining target properties"]
        )
        
        optimized_structures = []
        
        # Try PSMILES Diversifier if available
        if self.psmiles_diversifier and len(structures) > 0:
            try:
                for structure in structures:
                    psmiles = structure.get("psmiles", "")
                    if psmiles and psmiles != "[C][C][O]":  # Skip basic fallback
                        diversified = self.psmiles_diversifier.diversify_structure(
                            psmiles, num_variants=2
                        )
                        
                        if diversified:
                            for i, variant in enumerate(diversified):
                                optimized_structures.append({
                                    **structure,
                                    "id": f"{structure['id']}_div_{i+1}",
                                    "psmiles": variant,
                                    "generation_method": f"{structure['generation_method']}_diversified",
                                    "diversification_strategy": diversification_decision.chosen_option
                                })
                        else:
                            optimized_structures.append(structure)
                    else:
                        optimized_structures.append(structure)
                
                logger.info(f"Diversification produced {len(optimized_structures)} structures")
            except Exception as e:
                logger.warning(f"Diversification failed: {e}")
                optimized_structures = structures
        else:
            optimized_structures = structures
        
        return optimized_structures[:8]  # Limit to 8 structures
    
    async def _validate_and_score_molecules(self, structures: List[Dict[str, Any]],
                                          context: GenerationContext,
                                          decision_engine: LLMDecisionEngine) -> GeneratedMolecules:
        """Validate and score generated molecules."""
        
        valid_structures = []
        total_validity_score = 0.0
        total_diversity_score = 0.0
        
        # Validate each structure
        for structure in structures:
            try:
                # Basic validation
                psmiles = structure.get("psmiles", "")
                if self._validate_psmiles(psmiles):
                    validity_score = structure.get("confidence", 0.5)
                    total_validity_score += validity_score
                    
                    # Calculate diversity score (simplified)
                    diversity_score = self._calculate_diversity_score(structure, valid_structures)
                    total_diversity_score += diversity_score
                    
                    structure["validity_score"] = validity_score
                    structure["diversity_score"] = diversity_score
                    valid_structures.append(structure)
                    
            except Exception as e:
                logger.warning(f"Failed to validate structure {structure.get('id', 'unknown')}: {e}")
        
        # Calculate overall scores
        num_valid = len(valid_structures)
        avg_validity = total_validity_score / num_valid if num_valid > 0 else 0.0
        avg_diversity = total_diversity_score / num_valid if num_valid > 0 else 0.0
        
        # Extract results
        monomers_generated = list(set([
            structure.get("description", "").split(" from ")[-1].split(" ")[0]
            for structure in valid_structures
        ]))
        
        psmiles_strings = [structure.get("psmiles", "") for structure in valid_structures]
        
        # Determine generation method
        methods = [structure.get("generation_method", "") for structure in valid_structures]
        primary_method = max(set(methods), key=methods.count) if methods else "automated"
        
        return GeneratedMolecules(
            monomers_generated=monomers_generated,
            psmiles_strings=psmiles_strings,
            generation_method=primary_method,
            diversity_score=avg_diversity,
            validity_score=avg_validity,
            execution_time=0.0,  # Would be measured in real implementation
            # Additional fields for Phase 2
            molecules=[{
                "id": struct.get("id", ""),
                "structure": struct.get("psmiles", ""),
                "confidence": struct.get("confidence", 0.5),
                "generation_method": struct.get("generation_method", "")
            } for struct in valid_structures],
            generation_parameters={
                "iteration": context.iteration,
                "target_properties": context.target_properties,
                "literature_informed": bool(context.literature_results)
            },
            success_rate=len(valid_structures) / len(structures) if structures else 0.0
        )
    
    def _validate_psmiles(self, psmiles: str) -> bool:
        """Basic validation of PSMILES string."""
        if not psmiles or len(psmiles) < 3:
            return False
        
        # Check for balanced brackets
        if psmiles.count('[') != psmiles.count(']'):
            return False
        
        if psmiles.count('(') != psmiles.count(')'):
            return False
        
        # Check for minimum polymer structure
        if '[C]' not in psmiles and 'C' not in psmiles:
            return False
        
        return True
    
    def _calculate_diversity_score(self, structure: Dict[str, Any], 
                                 existing_structures: List[Dict[str, Any]]) -> float:
        """Calculate diversity score compared to existing structures."""
        if not existing_structures:
            return 1.0
        
        current_psmiles = structure.get("psmiles", "")
        similarities = []
        
        for existing in existing_structures:
            existing_psmiles = existing.get("psmiles", "")
            # Simple string similarity (would use chemical similarity in real implementation)
            similarity = self._string_similarity(current_psmiles, existing_psmiles)
            similarities.append(similarity)
        
        # Diversity is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple string similarity."""
        if not s1 or not s2:
            return 0.0
        
        # Simple character overlap metric
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_previous_molecules(self, state: IterationState) -> List[Dict]:
        """Extract previous molecules from state history."""
        # This would extract molecules from previous iterations
        return []
    
    def _save_results(self, iteration: int, results: GeneratedMolecules):
        """Save results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_generation.json"
            with open(results_file, 'w') as f:
                # Create serializable dict
                results_dict = {
                    "monomers_generated": results.monomers_generated,
                    "psmiles_strings": results.psmiles_strings,
                    "generation_method": results.generation_method,
                    "diversity_score": results.diversity_score,
                    "validity_score": results.validity_score,
                    "execution_time": results.execution_time
                }
                if hasattr(results, 'molecules'):
                    results_dict["molecules"] = results.molecules
                if hasattr(results, 'generation_parameters'):
                    results_dict["generation_parameters"] = results.generation_parameters
                
                json.dump(results_dict, f, indent=2)
            logger.info(f"Generation results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save generation results: {e}")


# Test functionality
async def test_automated_piecewise_generation():
    """Test the AutomatedPiecewiseGeneration functionality."""
    print("Testing AutomatedPiecewiseGeneration...")
    
    # Import required components
    from .state_manager import StateManager
    from .decision_engine import LLMDecisionEngine
    
    # Create test components
    state_manager = StateManager("test_automated_generation")
    decision_engine = LLMDecisionEngine()
    piecewise_generator = AutomatedPiecewiseGeneration("test_generation_output")
    
    # Create test iteration state with literature results
    state = state_manager.create_new_iteration(
        initial_prompt="Design a biodegradable polymer for insulin delivery",
        target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
    )
    
    # Add mock literature results
    from .state_manager import LiteratureResults
    state.literature_results = LiteratureResults(
        papers_found=25,
        relevant_papers=8,
        extracted_properties={"biocompatibility": [0.8, 0.9], "degradation_rate": [0.3, 0.5]},
        synthesis_routes=["polymerization", "grafting"],
        query_used="insulin delivery polymers",
        material_candidates=["PEG", "PLGA", "chitosan"]
    )
    
    print(f"Created test iteration {state.iteration_number}")
    
    # Run automated piecewise generation
    results = await piecewise_generator.run_automated_generation(state, decision_engine)
    
    print(f"Piecewise generation results:")
    print(f"- Monomers generated: {len(results.monomers_generated)}")
    print(f"- PSMILES strings: {len(results.psmiles_strings)}")
    print(f"- Generation method: {results.generation_method}")
    print(f"- Diversity score: {results.diversity_score:.3f}")
    print(f"- Validity score: {results.validity_score:.3f}")
    print(f"- First PSMILES: {results.psmiles_strings[0] if results.psmiles_strings else 'None'}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_automated_piecewise_generation()) 