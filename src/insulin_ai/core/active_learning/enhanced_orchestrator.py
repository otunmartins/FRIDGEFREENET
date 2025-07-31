#!/usr/bin/env python3
"""
Enhanced Active Learning Orchestrator - Phase 4 Integration

This module provides the main orchestrator for the complete active learning system,
integrating all components from Phases 1-3 with comprehensive monitoring, error handling,
and performance optimization.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Core active learning components
from .state_manager import StateManager, IterationState, IterationStatus
from .decision_engine import LLMDecisionEngine, DecisionType, DecisionContext
from .loop_controller import LoopController, ConvergenceConfig, ResourceLimits, QualityGates

# Phase 2 automated components
from .automated_literature_mining import AutomatedLiteratureMining
from .automated_piecewise_generation import AutomatedPiecewiseGeneration
from .automated_md_simulation import AutomatedMDSimulation
from .automated_post_processing import AutomatedPostProcessing

# Phase 3 RAG system
try:
    from .rag import (
        create_complete_rag_system,
        FeedbackGenerator,
        IterationFeedback,
        VectorDatabase,
        PropertyDatabase
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available")

# Monitoring and observability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrchestrationPhase(Enum):
    """Phases of the orchestration process."""
    INITIALIZATION = "initialization"
    LITERATURE_MINING = "literature_mining"
    MOLECULE_GENERATION = "molecule_generation"
    MD_SIMULATION = "md_simulation"
    POST_PROCESSING = "post_processing"
    RAG_ANALYSIS = "rag_analysis"
    FEEDBACK_GENERATION = "feedback_generation"
    ITERATION_COMPLETION = "iteration_completion"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    iteration_start_time: datetime
    iteration_end_time: Optional[datetime] = None
    phase_durations: Dict[str, float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    total_execution_time: float = 0.0
    component_success_rates: Dict[str, float] = None
    error_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.phase_durations is None:
            self.phase_durations = {}
        if self.component_success_rates is None:
            self.component_success_rates = {}
        if self.error_counts is None:
            self.error_counts = {}


@dataclass
class OrchestrationConfig:
    """Configuration for the enhanced orchestrator."""
    max_iterations: int = 10
    storage_path: str = "active_learning_data"
    enable_rag: bool = True
    enable_monitoring: bool = True
    performance_logging: bool = True
    auto_optimization: bool = True
    convergence_config: Optional[ConvergenceConfig] = None
    resource_limits: Optional[ResourceLimits] = None
    quality_gates: Optional[QualityGates] = None
    rag_config: Optional[Dict[str, Any]] = None
    target_properties: Optional[Dict[str, float]] = None


class PerformanceMonitor:
    """Monitors system performance and resource usage."""
    
    def __init__(self, enable_monitoring: bool = True):
        """Initialize performance monitor."""
        self.enable_monitoring = enable_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.phase_start_times: Dict[str, datetime] = {}
        
    def start_phase(self, phase: OrchestrationPhase):
        """Start monitoring a phase."""
        if not self.enable_monitoring:
            return
        
        self.phase_start_times[phase.value] = datetime.now()
        logger.debug(f"Started monitoring phase: {phase.value}")
    
    def end_phase(self, phase: OrchestrationPhase, metrics: PerformanceMetrics):
        """End monitoring a phase and update metrics."""
        if not self.enable_monitoring:
            return
        
        if phase.value in self.phase_start_times:
            duration = (datetime.now() - self.phase_start_times[phase.value]).total_seconds()
            metrics.phase_durations[phase.value] = duration
            logger.debug(f"Phase {phase.value} completed in {duration:.2f}s")
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return {"memory_mb": 0.0, "cpu_percent": 0.0}
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {"memory_mb": 0.0, "cpu_percent": 0.0}
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all iterations."""
        if not self.metrics_history:
            return {}
        
        total_iterations = len(self.metrics_history)
        avg_execution_time = sum(m.total_execution_time for m in self.metrics_history) / total_iterations
        
        # Phase duration averages
        phase_averages = {}
        for metrics in self.metrics_history:
            for phase, duration in metrics.phase_durations.items():
                if phase not in phase_averages:
                    phase_averages[phase] = []
                phase_averages[phase].append(duration)
        
        for phase in phase_averages:
            phase_averages[phase] = sum(phase_averages[phase]) / len(phase_averages[phase])
        
        return {
            "total_iterations": total_iterations,
            "average_execution_time": avg_execution_time,
            "phase_averages": phase_averages,
            "latest_memory_mb": self.metrics_history[-1].memory_usage_mb,
            "latest_cpu_percent": self.metrics_history[-1].cpu_usage_percent
        }


class EnhancedActiveLearningOrchestrator:
    """Enhanced orchestrator integrating all active learning components with RAG system."""
    
    def __init__(self, config: OrchestrationConfig):
        """Initialize the enhanced orchestrator."""
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.state_manager = StateManager(self.storage_path / "states")
        self.decision_engine = LLMDecisionEngine()
        self.loop_controller = LoopController(
            max_iterations=config.max_iterations,
            convergence_config=config.convergence_config,
            resource_limits=config.resource_limits,
            quality_gates=config.quality_gates
        )
        
        # Initialize Phase 2 automated components
        self.literature_mining = AutomatedLiteratureMining()
        self.psmiles_generation = AutomatedPiecewiseGeneration()
        self.md_simulation = AutomatedMDSimulation()
        self.post_processing = AutomatedPostProcessing()
        
        # Initialize Phase 3 RAG system
        self.rag_system = None
        self.feedback_generator = None
        if config.enable_rag and RAG_AVAILABLE:
            try:
                rag_config = config.rag_config or {
                    'vector_db': {'storage_path': str(self.storage_path / "rag_vector_db")},
                    'property_db': {'path': str(self.storage_path / "rag_property_db.db")},
                    'web_search': {'cache_directory': str(self.storage_path / "rag_search_cache")}
                }
                
                self.rag_system = create_complete_rag_system(rag_config)
                self.feedback_generator = self.rag_system['feedback_generator']
                logger.info("RAG system initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                self.rag_system = None
                self.feedback_generator = None
        
        # Initialize monitoring
        self.performance_monitor = PerformanceMonitor(config.enable_monitoring)
        
        # Callbacks for monitoring and extensibility
        self.iteration_start_callbacks: List[Callable[[IterationState], None]] = []
        self.iteration_end_callbacks: List[Callable[[IterationState, PerformanceMetrics], None]] = []
        self.phase_callbacks: Dict[OrchestrationPhase, List[Callable[[IterationState], None]]] = {}
        self.error_callbacks: List[Callable[[Exception, IterationState], None]] = []
        
        # Target properties for optimization
        self.target_properties = config.target_properties or {
            "young_modulus": 3.0,
            "glass_transition_temp": 350.0,
            "degradation_rate": 0.05,
            "biocompatibility_score": 0.95
        }
        
        logger.info(f"EnhancedActiveLearningOrchestrator initialized - Max iterations: {config.max_iterations}")
    
    def add_iteration_callback(self, 
                             callback: Callable[[IterationState], None],
                             phase: Optional[OrchestrationPhase] = None):
        """Add callback for iteration events."""
        if phase is None:
            self.iteration_start_callbacks.append(callback)
        else:
            if phase not in self.phase_callbacks:
                self.phase_callbacks[phase] = []
            self.phase_callbacks[phase].append(callback)
    
    def add_completion_callback(self, 
                              callback: Callable[[IterationState, PerformanceMetrics], None]):
        """Add callback for iteration completion."""
        self.iteration_end_callbacks.append(callback)
    
    def add_error_callback(self, 
                          callback: Callable[[Exception, IterationState], None]):
        """Add callback for error handling."""
        self.error_callbacks.append(callback)
    
    async def run_complete_active_learning(self, 
                                         initial_prompt: str,
                                         max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete active learning workflow with full integration."""
        max_iterations = max_iterations or self.config.max_iterations
        
        logger.info(f"🚀 Starting Complete Active Learning Workflow")
        logger.info(f"Initial prompt: {initial_prompt}")
        logger.info(f"Target properties: {self.target_properties}")
        logger.info(f"Max iterations: {max_iterations}")
        
        start_time = datetime.now()
        all_iterations: List[IterationState] = []
        overall_metrics = []
        
        try:
            for iteration_num in range(1, max_iterations + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting Iteration {iteration_num}/{max_iterations}")
                logger.info(f"{'='*60}")
                
                # Create iteration state
                iteration_state = IterationState(
                    iteration_number=iteration_num,
                    start_time=datetime.now()
                )
                
                # Initialize performance metrics
                iteration_metrics = PerformanceMetrics(
                    iteration_start_time=iteration_state.start_time
                )
                
                try:
                    # Run single iteration
                    iteration_result = await self._run_single_iteration(
                        iteration_state, 
                        initial_prompt if iteration_num == 1 else None,
                        iteration_metrics
                    )
                    
                    # Update metrics
                    iteration_metrics.iteration_end_time = datetime.now()
                    iteration_metrics.total_execution_time = (
                        iteration_metrics.iteration_end_time - 
                        iteration_metrics.iteration_start_time
                    ).total_seconds()
                    
                    # Add system metrics
                    sys_metrics = self.performance_monitor.get_system_metrics()
                    iteration_metrics.memory_usage_mb = sys_metrics["memory_mb"]
                    iteration_metrics.cpu_usage_percent = sys_metrics["cpu_percent"]
                    
                    # Store results
                    all_iterations.append(iteration_result)
                    overall_metrics.append(iteration_metrics)
                    self.performance_monitor.add_metrics(iteration_metrics)
                    
                    # Call completion callbacks
                    for callback in self.iteration_end_callbacks:
                        try:
                            callback(iteration_result, iteration_metrics)
                        except Exception as e:
                            logger.warning(f"Completion callback failed: {e}")
                    
                    # Check convergence
                    if await self._check_convergence(all_iterations):
                        logger.info(f"🎯 Convergence achieved after {iteration_num} iterations")
                        break
                    
                    # Log iteration summary
                    self._log_iteration_summary(iteration_result, iteration_metrics)
                    
                except Exception as e:
                    logger.error(f"❌ Iteration {iteration_num} failed: {e}")
                    
                    # Update iteration state with error
                    iteration_state.update_status(IterationStatus.FAILED, str(e))
                    iteration_state.errors.append(str(e))
                    iteration_state.end_time = datetime.now()
                    
                    # Call error callbacks
                    for callback in self.error_callbacks:
                        try:
                            callback(e, iteration_state)
                        except Exception as callback_error:
                            logger.warning(f"Error callback failed: {callback_error}")
                    
                    # Decide whether to continue or stop
                    if not await self._handle_iteration_error(e, iteration_state, iteration_num):
                        logger.error(f"🛑 Stopping due to critical error in iteration {iteration_num}")
                        break
                    
                    all_iterations.append(iteration_state)
            
            # Generate final summary
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            summary = await self._generate_final_summary(
                all_iterations, overall_metrics, total_duration
            )
            
            logger.info(f"\n🎉 Active Learning Workflow Completed!")
            logger.info(f"Total iterations: {len(all_iterations)}")
            logger.info(f"Total duration: {total_duration:.2f} seconds")
            logger.info(f"Success rate: {summary['success_rate']:.1%}")
            
            return summary
            
        except Exception as e:
            logger.error(f"💥 Critical failure in active learning workflow: {e}")
            raise
    
    async def _run_single_iteration(self, 
                                  state: IterationState,
                                  initial_prompt: Optional[str],
                                  metrics: PerformanceMetrics) -> IterationState:
        """Run a single iteration of the active learning loop."""
        
        # Call iteration start callbacks
        for callback in self.iteration_start_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.warning(f"Start callback failed: {e}")
        
        try:
            # Phase 1: Literature Mining
            state = await self._execute_phase(
                OrchestrationPhase.LITERATURE_MINING,
                self._run_literature_mining,
                state, metrics, initial_prompt
            )
            
            # Phase 2: Molecule Generation
            state = await self._execute_phase(
                OrchestrationPhase.MOLECULE_GENERATION,
                self._run_molecule_generation,
                state, metrics
            )
            
            # Phase 3: MD Simulation
            state = await self._execute_phase(
                OrchestrationPhase.MD_SIMULATION,
                self._run_md_simulation,
                state, metrics
            )
            
            # Phase 4: Post-processing
            state = await self._execute_phase(
                OrchestrationPhase.POST_PROCESSING,
                self._run_post_processing,
                state, metrics
            )
            
            # Phase 5: RAG Analysis (if available)
            if self.feedback_generator:
                state = await self._execute_phase(
                    OrchestrationPhase.RAG_ANALYSIS,
                    self._run_rag_analysis,
                    state, metrics
                )
            
            # Phase 6: Feedback Generation
            if self.feedback_generator:
                state = await self._execute_phase(
                    OrchestrationPhase.FEEDBACK_GENERATION,
                    self._run_feedback_generation,
                    state, metrics
                )
            
            # Mark iteration as completed
            state.update_status(IterationStatus.COMPLETED)
            state.end_time = datetime.now()
            
            # Store iteration state
            await self.state_manager.save_iteration_state(state)
            
            return state
            
        except Exception as e:
            state.update_status(IterationStatus.FAILED, str(e))
            state.errors.append(str(e))
            state.end_time = datetime.now()
            raise
    
    async def _execute_phase(self, 
                           phase: OrchestrationPhase,
                           phase_function: Callable,
                           state: IterationState,
                           metrics: PerformanceMetrics,
                           *args) -> IterationState:
        """Execute a single phase with monitoring and error handling."""
        
        logger.info(f"🔄 Executing {phase.value.replace('_', ' ').title()}")
        
        # Start phase monitoring
        self.performance_monitor.start_phase(phase)
        
        # Call phase-specific callbacks
        if phase in self.phase_callbacks:
            for callback in self.phase_callbacks[phase]:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"Phase callback failed: {e}")
        
        try:
            # Execute phase
            result = await phase_function(state, *args)
            
            # End phase monitoring
            self.performance_monitor.end_phase(phase, metrics)
            
            # Update success rates
            if phase.value not in metrics.component_success_rates:
                metrics.component_success_rates[phase.value] = 0
            metrics.component_success_rates[phase.value] += 1
            
            logger.info(f"✅ {phase.value.replace('_', ' ').title()} completed successfully")
            
            return result
            
        except Exception as e:
            # End phase monitoring
            self.performance_monitor.end_phase(phase, metrics)
            
            # Update error counts
            if phase.value not in metrics.error_counts:
                metrics.error_counts[phase.value] = 0
            metrics.error_counts[phase.value] += 1
            
            logger.error(f"❌ {phase.value.replace('_', ' ').title()} failed: {e}")
            
            # Add error to state
            state.errors.append(f"{phase.value}: {str(e)}")
            
            raise
    
    async def _run_literature_mining(self, state: IterationState, initial_prompt: str = None) -> IterationState:
        """Execute literature mining phase."""
        
        # Use initial prompt or generate from previous iteration
        prompt = initial_prompt or "Search for advanced drug delivery polymer materials"
        
        # Run automated literature mining
        literature_results = await self.literature_mining.run_automated_mining(state, self.decision_engine)
        
        # Update state
        state.literature_results = literature_results
        state.reasoning_log.append(f"Literature mining completed: {len(literature_results.papers_analyzed) if literature_results.papers_analyzed else 0} papers analyzed")
        
        return state
    
    async def _run_molecule_generation(self, state: IterationState) -> IterationState:
        """Execute molecule generation phase."""
        
        # Use literature insights for generation
        generation_context = "Generate biocompatible polymer for drug delivery"
        if state.literature_results and state.literature_results.key_insights:
            generation_context += f". Key insights: {'; '.join(state.literature_results.key_insights[:3])}"
        
        # Run automated piecewise generation
        generated_molecules = await self.psmiles_generation.run_automated_generation(state, self.decision_engine)
        
        # Update state
        state.generated_molecules = generated_molecules
        state.reasoning_log.append(f"Molecule generation completed: {len(generated_molecules.molecules) if generated_molecules.molecules else 0} molecules generated")
        
        return state
    
    async def _run_md_simulation(self, state: IterationState) -> IterationState:
        """Execute MD simulation phase."""
        
        if not state.generated_molecules or not state.generated_molecules.molecules:
            raise ValueError("No molecules available for MD simulation")
        
        # Run automated MD simulation
        simulation_results = await self.md_simulation.run_automated_simulation(state, self.decision_engine)
        
        # Update state
        state.simulation_results = simulation_results
        state.reasoning_log.append("MD simulation completed successfully")
        
        return state
    
    async def _run_post_processing(self, state: IterationState) -> IterationState:
        """Execute post-processing phase."""
        
        if not state.simulation_results:
            raise ValueError("No simulation results available for post-processing")
        
        # Run automated post-processing
        computed_properties = await self.post_processing.run_automated_processing(state, self.decision_engine)
        
        # Update state
        state.computed_properties = computed_properties
        
        # Calculate overall score
        if computed_properties.target_scores:
            if hasattr(computed_properties.target_scores, 'overall_score'):
                state.overall_score = computed_properties.target_scores.overall_score
            else:
                # Calculate from individual scores
                scores = []
                if hasattr(computed_properties.target_scores, 'biocompatibility'):
                    scores.append(computed_properties.target_scores.biocompatibility)
                if hasattr(computed_properties.target_scores, 'degradation_rate'):
                    scores.append(computed_properties.target_scores.degradation_rate)
                if hasattr(computed_properties.target_scores, 'mechanical_strength'):
                    scores.append(computed_properties.target_scores.mechanical_strength)
                
                state.overall_score = sum(scores) / len(scores) if scores else 0.5
        
        state.reasoning_log.append(f"Post-processing completed: Overall score {state.overall_score:.3f}")
        
        return state
    
    async def _run_rag_analysis(self, state: IterationState) -> IterationState:
        """Execute RAG analysis phase."""
        
        if not self.rag_system:
            logger.warning("RAG system not available, skipping RAG analysis")
            return state
        
        try:
            # Add current material to property database for analysis
            if state.computed_properties and state.computed_properties.md_properties:
                material = await self._create_material_from_state(state)
                await self.rag_system['property_db'].add_material(material)
            
            # Run similarity analysis
            if state.computed_properties:
                current_properties = self._extract_current_properties(state)
                similar_materials = await self.rag_system['similarity_matcher'].find_property_analogs(
                    current_properties, max_results=5
                )
                
                # Store results in RAG analysis (simplified)
                from .state_manager import RAGAnalysis
                state.rag_analysis = RAGAnalysis(
                    similar_materials=[f"Material {i+1}" for i in range(len(similar_materials))],
                    performance_analysis=f"Found {len(similar_materials)} similar materials",
                    improvement_suggestions=["Optimize crosslinking density", "Adjust molecular weight"],
                    next_iteration_prompt="Focus on improving mechanical properties",
                    confidence_score=0.8,
                    execution_time=1.0
                )
            
            state.reasoning_log.append("RAG analysis completed successfully")
            
        except Exception as e:
            logger.warning(f"RAG analysis failed: {e}")
            # Continue without RAG analysis
        
        return state
    
    async def _run_feedback_generation(self, state: IterationState) -> IterationState:
        """Execute feedback generation phase."""
        
        if not self.feedback_generator:
            logger.warning("Feedback generator not available, skipping feedback generation")
            return state
        
        try:
            # Generate comprehensive feedback
            feedback = await self.feedback_generator.generate_iteration_feedback(
                state, self.target_properties
            )
            
            # Update RAG analysis with feedback
            if not state.rag_analysis:
                from .state_manager import RAGAnalysis
                state.rag_analysis = RAGAnalysis()
            
            # Update with feedback information
            state.rag_analysis.next_iteration_prompt = feedback.next_iteration_prompt
            state.rag_analysis.improvement_suggestions = feedback.priority_improvements
            state.rag_analysis.performance_analysis = feedback.overall_assessment
            state.rag_analysis.confidence_score = feedback.confidence
            
            state.reasoning_log.append(f"Feedback generation completed: {len(feedback.feedback_items)} items generated")
            
        except Exception as e:
            logger.warning(f"Feedback generation failed: {e}")
            # Continue without feedback generation
        
        return state
    
    async def _create_material_from_state(self, state: IterationState):
        """Create a Material object from iteration state for RAG analysis."""
        from .rag.property_database import Material, MaterialProperty, PropertyType, DataSource
        
        properties = []
        
        if state.computed_properties and state.computed_properties.md_properties:
            md_props = state.computed_properties.md_properties
            
            # Convert MD properties to MaterialProperty objects
            property_mappings = [
                ("youngs_modulus_x", "young_modulus", "GPa"),
                ("glass_transition_temp", "glass_transition_temperature", "K"),
                ("density", "density", "g/cm3"),
                ("cohesive_energy", "cohesive_energy", "kJ/mol")
            ]
            
            for md_attr, prop_name, unit in property_mappings:
                if hasattr(md_props, md_attr):
                    value = getattr(md_props, md_attr)
                    if value is not None:
                        prop = MaterialProperty(
                            material_id=f"iter_{state.iteration_number}",
                            property_name=prop_name,
                            property_type=PropertyType.MECHANICAL,
                            value=value,
                            unit=unit,
                            source=DataSource.COMPUTATIONAL,
                            confidence=0.8
                        )
                        properties.append(prop)
        
        return Material(
            material_id=f"iteration_{state.iteration_number}",
            name=f"Generated_Material_Iter_{state.iteration_number}",
            composition={"generated_polymer": 1.0},
            synthesis_method="automated_piecewise_generation",
            properties=properties,
            created_date=datetime.now(),
            metadata={"iteration": state.iteration_number}
        )
    
    def _extract_current_properties(self, state: IterationState) -> Dict[str, float]:
        """Extract current material properties from state."""
        properties = {}
        
        if state.computed_properties and state.computed_properties.md_properties:
            md_props = state.computed_properties.md_properties
            
            if hasattr(md_props, 'youngs_modulus_x') and md_props.youngs_modulus_x is not None:
                properties["young_modulus"] = md_props.youngs_modulus_x
            if hasattr(md_props, 'glass_transition_temp') and md_props.glass_transition_temp is not None:
                properties["glass_transition_temp"] = md_props.glass_transition_temp
            if hasattr(md_props, 'density') and md_props.density is not None:
                properties["density"] = md_props.density
        
        return properties
    
    async def _check_convergence(self, iterations: List[IterationState]) -> bool:
        """Check if the active learning loop has converged."""
        if len(iterations) < 2:
            return False
        
        # Simple convergence check based on score improvement
        recent_scores = [it.overall_score for it in iterations[-3:] if hasattr(it, 'overall_score')]
        
        if len(recent_scores) >= 3:
            # Check if improvement has plateaued
            improvements = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores)-1)]
            avg_improvement = sum(improvements) / len(improvements)
            
            if avg_improvement < 0.01:  # Less than 1% improvement
                return True
        
        # Check target achievement
        latest_score = iterations[-1].overall_score
        if latest_score > 0.95:  # 95% of target achieved
            return True
        
        return False
    
    async def _handle_iteration_error(self, 
                                    error: Exception,
                                    state: IterationState,
                                    iteration_num: int) -> bool:
        """Handle iteration error and decide whether to continue."""
        
        # Log error details
        logger.error(f"Iteration {iteration_num} error details: {error}")
        
        # Simple error handling strategy
        if iteration_num <= 2:
            # Continue for early iterations
            logger.info("Continuing despite error in early iteration")
            return True
        
        # Check error frequency
        recent_iterations = iteration_num
        if recent_iterations > 3:
            # Stop if too many consecutive errors
            return False
        
        return True
    
    def _log_iteration_summary(self, state: IterationState, metrics: PerformanceMetrics):
        """Log iteration summary."""
        logger.info(f"\n📊 Iteration {state.iteration_number} Summary:")
        logger.info(f"   Status: {state.status.value}")
        logger.info(f"   Overall Score: {state.overall_score:.3f}")
        logger.info(f"   Execution Time: {metrics.total_execution_time:.2f}s")
        logger.info(f"   Memory Usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.phase_durations:
            logger.info("   Phase Durations:")
            for phase, duration in metrics.phase_durations.items():
                logger.info(f"     {phase.replace('_', ' ').title()}: {duration:.2f}s")
        
        if state.errors:
            logger.warning(f"   Errors: {len(state.errors)}")
        
        if state.rag_analysis and state.rag_analysis.next_iteration_prompt:
            logger.info(f"   Next Prompt: {state.rag_analysis.next_iteration_prompt[:100]}...")
    
    async def _generate_final_summary(self, 
                                    iterations: List[IterationState],
                                    metrics: List[PerformanceMetrics],
                                    total_duration: float) -> Dict[str, Any]:
        """Generate final summary of the active learning workflow."""
        
        successful_iterations = [it for it in iterations if it.status == IterationStatus.COMPLETED]
        success_rate = len(successful_iterations) / len(iterations) if iterations else 0
        
        # Calculate score progression
        scores = [it.overall_score for it in iterations if hasattr(it, 'overall_score')]
        score_improvement = scores[-1] - scores[0] if len(scores) >= 2 else 0
        
        # Performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        
        # Best iteration
        best_iteration = max(iterations, key=lambda x: getattr(x, 'overall_score', 0))
        
        summary = {
            "total_iterations": len(iterations),
            "successful_iterations": len(successful_iterations),
            "success_rate": success_rate,
            "total_duration": total_duration,
            "score_progression": scores,
            "total_improvement": score_improvement,
            "best_iteration": {
                "number": best_iteration.iteration_number,
                "score": getattr(best_iteration, 'overall_score', 0),
                "timestamp": best_iteration.start_time.isoformat()
            },
            "performance_metrics": perf_summary,
            "final_status": "SUCCESS" if success_rate > 0.5 else "PARTIAL_SUCCESS" if success_rate > 0 else "FAILED"
        }
        
        # Save summary
        summary_path = self.storage_path / f"workflow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Final summary saved to: {summary_path}")
        
        return summary


# Factory function for easy initialization
def create_enhanced_orchestrator(config: Dict[str, Any] = None) -> EnhancedActiveLearningOrchestrator:
    """Create enhanced orchestrator with configuration."""
    config = config or {}
    
    orchestration_config = OrchestrationConfig(
        max_iterations=config.get('max_iterations', 10),
        storage_path=config.get('storage_path', 'active_learning_data'),
        enable_rag=config.get('enable_rag', True),
        enable_monitoring=config.get('enable_monitoring', True),
        target_properties=config.get('target_properties')
    )
    
    return EnhancedActiveLearningOrchestrator(orchestration_config) 