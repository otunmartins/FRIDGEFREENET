# Core infrastructure confirms rules are active
"""
Active Learning Orchestrator

Main coordination class that orchestrates the complete active learning loop,
managing all components and their interactions.
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

# Import core active learning components
from .state_manager import (
    StateManager, IterationState, IterationStatus, 
    LiteratureResults, GeneratedMolecules, SimulationResults, 
    ComputedProperties, RAGAnalysis
)
from .decision_engine import LLMDecisionEngine, DecisionType, DecisionOutput
from .loop_controller import (
    LoopController, LoopStatus, StopReason,
    ConvergenceConfig, ResourceLimits, QualityGates
)

# Import existing components (will be properly integrated in Phase 2)
# from ..literature_mining_system import MaterialsLiteratureMiner
# from ..psmiles_generator import PSMILESGenerator  
# from ..psmiles_processor import PSMILESProcessor

# Set up logging
logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    """Main orchestrator for the active learning material discovery system."""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 storage_path: str = "active_learning_data",
                 convergence_config: Optional[ConvergenceConfig] = None,
                 resource_limits: Optional[ResourceLimits] = None,
                 quality_gates: Optional[QualityGates] = None):
        """Initialize the active learning orchestrator.
        
        Args:
            max_iterations: Maximum number of iterations to run
            storage_path: Path to store active learning data
            convergence_config: Configuration for convergence detection
            resource_limits: Resource usage limits
            quality_gates: Quality validation gates
        """
        self.max_iterations = max_iterations
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.state_manager = StateManager(self.storage_path / "states")
        self.decision_engine = LLMDecisionEngine()
        self.loop_controller = LoopController(
            max_iterations=max_iterations,
            convergence_config=convergence_config,
            resource_limits=resource_limits,
            quality_gates=quality_gates
        )
        
        # Initialize existing components with automation wrappers
        self.literature_mining = AutomatedLiteratureMining()
        self.psmiles_generation = AutomatedPiecewiseGeneration() 
        self.md_simulation = AutomatedMDSimulation()
        self.post_processing = AutomatedPostProcessing()
        self.rag_analyzer = RAGPropertyAnalyzer()
        
        # Callbacks for monitoring
        self.iteration_callbacks: List[Callable[[IterationState], None]] = []
        self.completion_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info(f"ActiveLearningOrchestrator initialized - Max iterations: {max_iterations}")
    
    async def run_active_learning_loop(self, 
                                     initial_prompt: str,
                                     target_properties: Dict[str, float] = None) -> Dict[str, Any]:
        """Run the complete active learning loop.
        
        Args:
            initial_prompt: Initial prompt to start the process
            target_properties: Target material properties for optimization
            
        Returns:
            Dictionary with complete results and metrics
        """
        logger.info("Starting active learning loop")
        self.loop_controller.start_loop()
        
        # Create initial iteration state
        current_state = self.state_manager.create_new_iteration(
            initial_prompt=initial_prompt,
            target_properties=target_properties or {}
        )
        
        try:
            while True:
                # Check if we should continue
                should_continue, stop_reason = self.loop_controller.should_continue(self.state_manager)
                
                if not should_continue:
                    self.loop_controller.stop_loop(stop_reason)
                    logger.info(f"Loop stopped: {stop_reason.value}")
                    break
                
                # Begin new iteration
                self.loop_controller.begin_iteration(current_state.iteration_number)
                current_state.add_reasoning("Loop Control", f"Starting iteration {current_state.iteration_number}")
                
                # Execute iteration steps
                try:
                    await self._execute_iteration(current_state)
                    
                    # Complete iteration
                    current_state.update_status(IterationStatus.COMPLETED, "All steps completed successfully")
                    self.loop_controller.complete_iteration(current_state)
                    
                    # Notify callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(current_state)
                        except Exception as e:
                            logger.error(f"Error in iteration callback: {e}")
                    
                    # Prepare next iteration if needed
                    if current_state.rag_analysis and current_state.rag_analysis.next_iteration_prompt:
                        next_state = self.state_manager.create_new_iteration(
                            initial_prompt=current_state.rag_analysis.next_iteration_prompt,
                            target_properties=current_state.target_properties
                        )
                        # Calculate improvement over previous iteration
                        if len(self.state_manager.get_completed_states()) > 1:
                            previous_state = self.state_manager.get_completed_states()[-2]
                            next_state.improvement_over_previous = current_state.overall_score - previous_state.overall_score
                        
                        current_state = next_state
                    else:
                        break
                
                except Exception as e:
                    logger.error(f"Error in iteration {current_state.iteration_number}: {e}")
                    current_state.add_error(f"Iteration failed: {str(e)}")
                    current_state.update_status(IterationStatus.FAILED, f"Error: {str(e)}")
                    self.loop_controller.complete_iteration(current_state)
                    
                    # Decide whether to continue or stop
                    error_rate = len([s for s in self.state_manager.get_all_states() if s.status == IterationStatus.FAILED]) / len(self.state_manager.get_all_states())
                    if error_rate > 0.5:  # Stop if more than 50% of iterations failed
                        self.loop_controller.stop_loop(StopReason.ERROR_THRESHOLD)
                        break
                    
                    # Create recovery iteration
                    recovery_prompt = await self._generate_recovery_prompt(current_state, str(e))
                    current_state = self.state_manager.create_new_iteration(
                        initial_prompt=recovery_prompt,
                        target_properties=current_state.target_properties
                    )
        
        except Exception as e:
            logger.critical(f"Critical error in active learning loop: {e}")
            logger.critical(traceback.format_exc())
            self.loop_controller.stop_loop(StopReason.ERROR_THRESHOLD)
        
        # Compile final results
        final_results = self._compile_final_results()
        
        # Notify completion callbacks
        for callback in self.completion_callbacks:
            try:
                callback(final_results)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
        
        logger.info("Active learning loop completed")
        return final_results
    
    async def _execute_iteration(self, state: IterationState) -> None:
        """Execute a single iteration of the active learning loop.
        
        Args:
            state: Current iteration state
        """
        # Step 1: Literature Mining
        state.update_status(IterationStatus.LITERATURE_MINING, "Starting literature mining")
        state.literature_results = await self.literature_mining.run_automated_mining(
            state, self.decision_engine
        )
        
        # Step 2: Piecewise Generation
        state.update_status(IterationStatus.PIECEWISE_GENERATION, "Starting molecule generation")
        state.generated_molecules = await self.psmiles_generation.run_automated_generation(
            state, self.decision_engine
        )
        
        # Step 3: MD Simulation
        state.update_status(IterationStatus.MD_SIMULATION, "Starting MD simulation")
        state.simulation_results = await self.md_simulation.run_automated_simulation(
            state, self.decision_engine
        )
        
        # Step 4: Post-Processing
        state.update_status(IterationStatus.POST_PROCESSING, "Starting post-processing")
        state.computed_properties = await self.post_processing.run_automated_processing(
            state, self.decision_engine
        )
        
        # Step 5: RAG Analysis
        state.update_status(IterationStatus.RAG_ANALYSIS, "Starting RAG analysis")
        state.rag_analysis = await self.rag_analyzer.run_automated_analysis(
            state, self.decision_engine
        )
        
        # Calculate overall score
        state.overall_score = self._calculate_overall_score(state)
        
        # Save updated state
        self.state_manager.save_state(state)
    
    def _calculate_overall_score(self, state: IterationState) -> float:
        """Calculate overall performance score for the iteration.
        
        Args:
            state: Iteration state to score
            
        Returns:
            Overall performance score (0-1)
        """
        scores = []
        
        # Literature mining score
        if state.literature_results:
            lit_score = min(1.0, state.literature_results.relevant_papers / 10.0)
            scores.append(lit_score * 0.15)
        
        # Generation success score
        if state.generated_molecules:
            gen_score = state.generated_molecules.success_rate
            scores.append(gen_score * 0.2)
        
        # Simulation convergence score
        if state.simulation_results:
            sim_score = 1.0 if state.simulation_results.convergence_achieved else 0.5
            scores.append(sim_score * 0.25)
        
        # Properties score (based on target achievement)
        if state.computed_properties and state.target_properties:
            prop_scores = []
            for prop, target in state.target_properties.items():
                if prop in state.computed_properties.mechanical_properties:
                    actual = state.computed_properties.mechanical_properties[prop]
                    prop_score = 1.0 - abs(actual - target) / max(target, 1.0)
                    prop_scores.append(max(0.0, prop_score))
            
            if prop_scores:
                avg_prop_score = sum(prop_scores) / len(prop_scores)
                scores.append(avg_prop_score * 0.3)
        
        # RAG analysis confidence score
        if state.rag_analysis:
            rag_score = state.rag_analysis.confidence_score
            scores.append(rag_score * 0.1)
        
        return sum(scores) if scores else 0.0
    
    async def _generate_recovery_prompt(self, failed_state: IterationState, error: str) -> str:
        """Generate a recovery prompt after iteration failure.
        
        Args:
            failed_state: The failed iteration state
            error: Error message
            
        Returns:
            Recovery prompt for next iteration
        """
        recovery_decision = self.decision_engine.make_decision(
            decision_type=DecisionType.NEXT_ITERATION_PROMPT,
            context_data={
                "iteration": failed_state.iteration_number,
                "error": error,
                "target_properties": failed_state.target_properties,
                "previous_prompt": failed_state.initial_prompt
            },
            objectives=["Recover from error", "Simplify approach", "Ensure stability"]
        )
        
        return recovery_decision.chosen_option
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile final results from all iterations.
        
        Returns:
            Dictionary with comprehensive results
        """
        all_states = self.state_manager.get_all_states()
        completed_states = self.state_manager.get_completed_states()
        
        best_iteration = None
        if completed_states:
            best_iteration = max(completed_states, key=lambda s: s.overall_score)
        
        return {
            "summary": {
                "total_iterations": len(all_states),
                "completed_iterations": len(completed_states),
                "success_rate": len(completed_states) / len(all_states) if all_states else 0.0,
                "best_score": best_iteration.overall_score if best_iteration else 0.0,
                "total_runtime": self.loop_controller.metrics.total_runtime,
                "stop_reason": self.loop_controller.stop_reason.value if self.loop_controller.stop_reason else None
            },
            "best_result": {
                "iteration": best_iteration.iteration_number if best_iteration else None,
                "score": best_iteration.overall_score if best_iteration else 0.0,
                "properties": best_iteration.computed_properties.to_dict() if best_iteration and best_iteration.computed_properties else {},
                "molecules": best_iteration.generated_molecules.psmiles_strings if best_iteration and best_iteration.generated_molecules else []
            },
            "progress_metrics": self.state_manager.calculate_progress_metrics(),
            "loop_metrics": self.loop_controller.get_loop_status(),
            "decision_metrics": self.decision_engine.get_decision_performance_metrics(),
            "iteration_history": [
                {
                    "iteration": state.iteration_number,
                    "status": state.status.value,
                    "score": state.overall_score,
                    "prompt": state.initial_prompt[:100] + "..." if len(state.initial_prompt) > 100 else state.initial_prompt,
                    "errors": len(state.errors)
                }
                for state in all_states
            ]
        }
    
    def add_iteration_callback(self, callback: Callable[[IterationState], None]) -> None:
        """Add callback for iteration completion.
        
        Args:
            callback: Function to call after each iteration
        """
        self.iteration_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for loop completion.
        
        Args:
            callback: Function to call when loop completes
        """
        self.completion_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the active learning system.
        
        Returns:
            Dictionary with current status information
        """
        return {
            "loop_status": self.loop_controller.get_loop_status(),
            "current_iteration": self.state_manager.current_iteration,
            "total_states": len(self.state_manager.get_all_states()),
            "completed_states": len(self.state_manager.get_completed_states()),
            "storage_path": str(self.storage_path),
            "components_status": {
                "state_manager": "active",
                "decision_engine": "active", 
                "loop_controller": self.loop_controller.status.value,
                "literature_mining": "available",
                "psmiles_generation": "available",
                "md_simulation": "available",
                "post_processing": "available",
                "rag_analyzer": "available"
            }
        }
    
    def export_results(self, format: str = "json") -> str:
        """Export all results to specified format.
        
        Args:
            format: Export format ('json', 'pickle')
            
        Returns:
            Path to exported file
        """
        # Export state manager results
        state_export = self.state_manager.export_results(format)
        
        # Export decision engine log
        decision_export = self.storage_path / f"decision_log.json"
        self.decision_engine.export_decision_log(str(decision_export))
        
        logger.info(f"Results exported - States: {state_export}, Decisions: {decision_export}")
        return state_export


# Placeholder classes for automated components (to be implemented in Phase 2)
class AutomatedLiteratureMining:
    """Automated wrapper for literature mining with LLM decision making."""
    
    async def run_automated_mining(self, state: IterationState, decision_engine: LLMDecisionEngine) -> LiteratureResults:
        """Run automated literature mining with decision engine."""
        # Mock implementation for Phase 1
        logger.info(f"Running automated literature mining for iteration {state.iteration_number}")
        
        # Generate search query using decision engine
        query_decision = decision_engine.make_decision(
            decision_type=DecisionType.LITERATURE_SEARCH_QUERY,
            context_data={
                "iteration": state.iteration_number,
                "target_properties": state.target_properties,
                "previous_queries": []
            },
            objectives=["Find relevant materials research"]
        )
        
        # Simulate literature results
        return LiteratureResults(
            papers_found=20,
            relevant_papers=5,
            extracted_properties={"biocompatibility": [0.8, 0.9], "degradation_rate": [0.3, 0.5]},
            synthesis_routes=["polymerization", "grafting"],
            query_used=str(query_decision.chosen_option)
        )


class AutomatedPiecewiseGeneration:
    """Automated wrapper for piecewise generation with LLM decision making."""
    
    async def run_automated_generation(self, state: IterationState, decision_engine: LLMDecisionEngine) -> GeneratedMolecules:
        """Run automated piecewise generation with decision engine."""
        logger.info(f"Running automated piecewise generation for iteration {state.iteration_number}")
        
        # Generate monomer selection using decision engine
        monomer_decision = decision_engine.make_decision(
            decision_type=DecisionType.MONOMER_SELECTION,
            context_data={
                "iteration": state.iteration_number,
                "target_properties": state.target_properties,
                "literature_results": state.literature_results
            },
            available_options=["ethylene_glycol", "lactic_acid", "glycolic_acid"],
            constraints={"biocompatible": True}
        )
        
        # Simulate generation results
        return GeneratedMolecules(
            molecules=[{"id": 1, "structure": "mock_polymer"}],
            psmiles_strings=["[C][C][O]"],
            generation_parameters={"monomer": monomer_decision.chosen_option},
            success_rate=0.8
        )


class AutomatedMDSimulation:
    """Automated wrapper for MD simulation with LLM decision making."""
    
    async def run_automated_simulation(self, state: IterationState, decision_engine: LLMDecisionEngine) -> SimulationResults:
        """Run automated MD simulation with decision engine."""
        logger.info(f"Running automated MD simulation for iteration {state.iteration_number}")
        
        # Select force field using decision engine
        ff_decision = decision_engine.make_decision(
            decision_type=DecisionType.FORCE_FIELD_SELECTION,
            context_data={
                "iteration": state.iteration_number,
                "molecules": state.generated_molecules.molecules if state.generated_molecules else [],
                "target_properties": state.target_properties
            },
            available_options=["gaff-2.2.20", "openff-2.1.0", "amber/protein.ff14SB.xml"]
        )
        
        # Simulate simulation results
        return SimulationResults(
            simulation_files=["trajectory.dcd", "topology.pdb"],
            energy_data={"potential": [100, 95, 90], "kinetic": [50, 55, 60]},
            trajectory_length=1000.0,
            force_field_used=str(ff_decision.chosen_option),
            convergence_achieved=True
        )


class AutomatedPostProcessing:
    """Automated wrapper for post-processing with LLM decision making."""
    
    async def run_automated_processing(self, state: IterationState, decision_engine: LLMDecisionEngine) -> ComputedProperties:
        """Run automated post-processing with decision engine."""
        logger.info(f"Running automated post-processing for iteration {state.iteration_number}")
        
        # Simulate property calculation
        return ComputedProperties(
            mechanical_properties={"young_modulus": 2.5, "tensile_strength": 45.0},
            thermal_properties={"glass_transition": 60.0, "melting_point": 180.0},
            transport_properties={"diffusion_coefficient": 1e-8},
            stability_metrics={"degradation_rate": 0.4},
            performance_score=0.75
        )


class RAGPropertyAnalyzer:
    """RAG-powered analysis and feedback system."""
    
    async def run_automated_analysis(self, state: IterationState, decision_engine: LLMDecisionEngine) -> RAGAnalysis:
        """Run automated RAG analysis with decision engine."""
        logger.info(f"Running automated RAG analysis for iteration {state.iteration_number}")
        
        # Generate performance evaluation using decision engine
        eval_decision = decision_engine.make_decision(
            decision_type=DecisionType.PERFORMANCE_EVALUATION,
            context_data={
                "iteration": state.iteration_number,
                "computed_properties": state.computed_properties.to_dict() if state.computed_properties else {},
                "target_properties": state.target_properties,
                "previous_results": []
            }
        )
        
        # Simulate RAG analysis results
        return RAGAnalysis(
            similar_materials=[{"name": "PEG", "similarity": 0.8}],
            property_benchmarks={"biocompatibility": 0.85},
            improvement_suggestions=["Increase molecular weight", "Add hydrophilic groups"],
            next_iteration_prompt="Design a polymer with higher molecular weight for improved mechanical properties",
            confidence_score=0.7
        )


# Test functionality
def test_orchestrator():
    """Test the ActiveLearningOrchestrator functionality."""
    print("Testing ActiveLearningOrchestrator...")
    
    async def run_test():
        # Create orchestrator
        orchestrator = ActiveLearningOrchestrator(
            max_iterations=3,
            storage_path="test_active_learning"
        )
        
        # Add monitoring callbacks
        def iteration_callback(state: IterationState):
            print(f"Iteration {state.iteration_number} completed with score: {state.overall_score}")
        
        def completion_callback(results: Dict[str, Any]):
            print(f"Loop completed - Best score: {results['summary']['best_score']}")
        
        orchestrator.add_iteration_callback(iteration_callback)
        orchestrator.add_completion_callback(completion_callback)
        
        # Run active learning loop
        results = await orchestrator.run_active_learning_loop(
            initial_prompt="Design a biodegradable polymer for insulin delivery",
            target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
        )
        
        print("Results summary:")
        print(f"- Total iterations: {results['summary']['total_iterations']}")
        print(f"- Success rate: {results['summary']['success_rate']:.2f}")
        print(f"- Best score: {results['summary']['best_score']:.2f}")
        print(f"- Runtime: {results['summary']['total_runtime']:.1f}s")
        
        # Export results
        export_path = orchestrator.export_results()
        print(f"Results exported to: {export_path}")
        
        return results
    
    # Run the async test
    results = asyncio.run(run_test())
    print("ActiveLearningOrchestrator test completed successfully!")
    return results


if __name__ == "__main__":
    test_orchestrator() 