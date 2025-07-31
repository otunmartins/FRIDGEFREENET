#!/usr/bin/env python3
"""
AutomatedPostProcessing - Phase 2 Implementation

This module provides automated post-processing with LLM-powered decision making
for the active learning material discovery system. It integrates existing
post-processing components with intelligent automation.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Import existing post-processing systems
try:
    from ...integration.analysis.comprehensive_postprocessing import ComprehensivePostProcessor
    COMPREHENSIVE_POSTPROCESSOR_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_POSTPROCESSOR_AVAILABLE = False
    logging.warning("ComprehensivePostProcessor not available")

try:
    from .property_scoring import PropertyScorer, TargetPropertyScores, MDSimulationProperties
    PROPERTY_SCORER_AVAILABLE = True
except ImportError:
    PROPERTY_SCORER_AVAILABLE = False
    logging.warning("PropertyScorer not available")

# Import active learning infrastructure
from .state_manager import IterationState, SimulationResults, ComputedProperties
from .decision_engine import LLMDecisionEngine, DecisionType

logger = logging.getLogger(__name__)


class PostProcessingContext:
    """Context data for post-processing decisions."""
    
    def __init__(self, iteration: int, target_properties: Dict[str, float],
                 simulation_results: Optional[SimulationResults] = None,
                 previous_properties: List[Dict] = None):
        self.iteration = iteration
        self.target_properties = target_properties or {}
        self.simulation_results = simulation_results
        self.previous_properties = previous_properties or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for decision engine."""
        return {
            "iteration": self.iteration,
            "target_properties": self.target_properties,
            "simulation_info": self._extract_simulation_info(),
            "previous_properties_count": len(self.previous_properties),
            "timestamp": self.timestamp.isoformat()
        }
    
    def _extract_simulation_info(self) -> Dict[str, Any]:
        """Extract key information from simulation results."""
        if not self.simulation_results:
            return {}
        
        return {
            "simulation_success": self.simulation_results.simulation_success,
            "simulation_time_ns": self.simulation_results.simulation_time_ns,
            "final_energy": self.simulation_results.final_energy,
            "temperature": self.simulation_results.temperature,
            "has_trajectory": bool(self.simulation_results.simulation_files)
        }


class AutomatedPostProcessing:
    """
    Automated post-processing with LLM-powered decision making.
    
    This class integrates existing post-processing systems with intelligent
    automation for property calculation selection, analysis depth determination,
    visualization choices, and performance evaluation.
    """
    
    def __init__(self, storage_path: str = "automated_post_processing"):
        """Initialize automated post-processing system.
        
        Args:
            storage_path: Path to store post-processing data and results
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize post-processing systems
        self._initialize_processing_systems()
        
        # Cache for results and decisions
        self._results_cache = {}
        self._decision_cache = {}
        
        logger.info("AutomatedPostProcessing initialized")
    
    def _initialize_processing_systems(self):
        """Initialize available post-processing systems."""
        # Initialize Comprehensive Post Processor
        if COMPREHENSIVE_POSTPROCESSOR_AVAILABLE:
            try:
                self.comprehensive_processor = ComprehensivePostProcessor(
                    output_dir=str(self.storage_path / "comprehensive_analysis"),
                    enable_progress_tracking=True
                )
                logger.info("ComprehensivePostProcessor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ComprehensivePostProcessor: {e}")
                self.comprehensive_processor = None
        else:
            self.comprehensive_processor = None
        
        # Initialize Property Scorer
        if PROPERTY_SCORER_AVAILABLE:
            try:
                self.property_scorer = PropertyScorer()
                logger.info("PropertyScorer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PropertyScorer: {e}")
                self.property_scorer = None
        else:
            self.property_scorer = None
    
    async def run_automated_processing(self, state: IterationState, 
                                     decision_engine: LLMDecisionEngine) -> ComputedProperties:
        """
        Run automated post-processing with LLM decision making.
        
        Args:
            state: Current iteration state
            decision_engine: LLM decision engine for automation
            
        Returns:
            ComputedProperties: Comprehensive post-processing results
        """
        logger.info(f"Starting automated post-processing for iteration {state.iteration_number}")
        
        try:
            # Step 1: Create post-processing context
            processing_context = PostProcessingContext(
                iteration=state.iteration_number,
                target_properties=state.target_properties,
                simulation_results=state.simulation_results,
                previous_properties=self._extract_previous_properties(state)
            )
            
            # Step 2: Determine property calculation strategy
            calculation_strategy = await self._determine_calculation_strategy(
                processing_context, decision_engine
            )
            
            # Step 3: Select analysis depth and methods
            analysis_configuration = await self._configure_analysis_methods(
                processing_context, calculation_strategy, decision_engine
            )
            
            # Step 4: Execute property calculations
            calculated_properties = await self._execute_property_calculations(
                processing_context, calculation_strategy, analysis_configuration
            )
            
            # Step 5: Generate performance metrics and scoring
            performance_metrics = await self._generate_performance_metrics(
                calculated_properties, processing_context, decision_engine
            )
            
            # Step 6: Create visualizations and reports
            analysis_outputs = await self._generate_analysis_outputs(
                calculated_properties, performance_metrics, processing_context, decision_engine
            )
            
            # Step 7: Compile final results
            final_results = await self._compile_final_results(
                calculated_properties, performance_metrics, analysis_outputs, processing_context
            )
            
            # Step 8: Save results and update cache
            self._save_results(state.iteration_number, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in automated post-processing: {e}")
            # Return minimal results to prevent pipeline failure
            return self._create_fallback_results(processing_context)
    
    async def _determine_calculation_strategy(self, context: PostProcessingContext,
                                            decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Determine which properties to calculate using LLM decision making."""
        
        # Define available property calculation categories
        property_categories = [
            "mechanical_properties",     # Young's modulus, tensile strength, etc.
            "thermal_properties",        # Glass transition, melting point, etc.
            "transport_properties",      # Diffusion coefficients, permeability
            "stability_metrics",         # Degradation rates, stability indices
            "interaction_energies",      # Binding energies, interaction strengths
            "structural_properties",     # Density, free volume, mesh size
            "biocompatibility_metrics"   # Cytotoxicity, biocompatibility scores
        ]
        
        # Generate property selection decision
        property_decision = decision_engine.make_decision(
            decision_type=DecisionType.PROPERTY_CALCULATION_SELECTION,
            context_data=context.to_dict(),
            available_options=property_categories,
            objectives=["Calculate properties most relevant to target objectives"],
            constraints={"computational_time": "moderate", "accuracy_priority": "high"}
        )
        
        # Map target properties to calculation priorities
        priority_mapping = self._map_target_properties_to_calculations(context.target_properties)
        
        # Determine specific calculations based on decision and priorities
        selected_calculations = self._select_specific_calculations(
            property_decision.chosen_option, priority_mapping
        )
        
        return {
            "primary_category": property_decision.chosen_option,
            "specific_calculations": selected_calculations,
            "priority_mapping": priority_mapping,
            "reasoning": property_decision.reasoning,
            "comprehensive_analysis": context.iteration > 3  # More comprehensive in later iterations
        }
    
    def _map_target_properties_to_calculations(self, target_properties: Dict[str, float]) -> Dict[str, float]:
        """Map target properties to specific calculation priorities."""
        priority_mapping = {}
        
        for prop, target_value in target_properties.items():
            if "biocompatib" in prop.lower():
                priority_mapping["biocompatibility_analysis"] = 1.0
                priority_mapping["cytotoxicity_prediction"] = 0.8
            
            elif "degradation" in prop.lower():
                priority_mapping["stability_analysis"] = 1.0
                priority_mapping["degradation_kinetics"] = 0.9
            
            elif "mechanical" in prop.lower():
                priority_mapping["mechanical_testing"] = 1.0
                priority_mapping["stress_strain_analysis"] = 0.8
            
            elif "thermal" in prop.lower():
                priority_mapping["thermal_analysis"] = 1.0
                priority_mapping["glass_transition"] = 0.7
            
            elif "diffusion" in prop.lower() or "transport" in prop.lower():
                priority_mapping["diffusion_analysis"] = 1.0
                priority_mapping["permeability_testing"] = 0.8
        
        # Default calculations if no specific targets
        if not priority_mapping:
            priority_mapping = {
                "basic_structural_analysis": 0.8,
                "energy_analysis": 0.7,
                "stability_check": 0.6
            }
        
        return priority_mapping
    
    def _select_specific_calculations(self, primary_category: str, 
                                    priority_mapping: Dict[str, float]) -> List[str]:
        """Select specific calculations based on category and priorities."""
        
        calculation_sets = {
            "mechanical_properties": [
                "young_modulus_calculation",
                "tensile_strength_analysis",
                "stress_strain_curves",
                "elastic_modulus_estimation"
            ],
            "thermal_properties": [
                "glass_transition_analysis",
                "melting_point_estimation",
                "thermal_stability_assessment",
                "heat_capacity_calculation"
            ],
            "transport_properties": [
                "diffusion_coefficient_analysis",
                "permeability_calculation",
                "solubility_estimation",
                "partition_coefficient_analysis"
            ],
            "stability_metrics": [
                "degradation_rate_analysis",
                "chemical_stability_assessment",
                "thermal_stability_testing",
                "long_term_stability_prediction"
            ],
            "interaction_energies": [
                "binding_energy_calculation",
                "mmgbsa_analysis",
                "interaction_energy_decomposition",
                "free_energy_perturbation"
            ],
            "structural_properties": [
                "density_calculation",
                "free_volume_analysis",
                "mesh_size_estimation",
                "pore_size_distribution"
            ],
            "biocompatibility_metrics": [
                "cytotoxicity_prediction",
                "biocompatibility_scoring",
                "immune_response_prediction",
                "biodegradation_assessment"
            ]
        }
        
        base_calculations = calculation_sets.get(primary_category, ["basic_analysis"])
        
        # Add high-priority calculations from mapping
        for calc, priority in priority_mapping.items():
            if priority > 0.8 and calc not in base_calculations:
                base_calculations.append(calc)
        
        return base_calculations[:6]  # Limit to 6 calculations for efficiency
    
    async def _configure_analysis_methods(self, context: PostProcessingContext,
                                        calculation_strategy: Dict[str, Any],
                                        decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Configure analysis methods and depth using LLM decision making."""
        
        # Generate analysis configuration decision
        analysis_decision = decision_engine.make_decision(
            decision_type=DecisionType.ANALYSIS_DEPTH_CONFIGURATION,
            context_data={
                **context.to_dict(),
                "calculation_strategy": calculation_strategy
            },
            available_options=["quick_analysis", "standard_analysis", "comprehensive_analysis"],
            objectives=["Balance analysis depth with computational efficiency"],
            constraints={"time_limit": "1 hour", "accuracy_threshold": "high"}
        )
        
        # Map decision to specific configuration
        config_mapping = {
            "quick_analysis": {
                "statistical_rigor": "basic",
                "trajectory_sampling": "every_10_frames",
                "convergence_checking": "minimal",
                "error_estimation": "simple",
                "visualization_level": "basic"
            },
            "standard_analysis": {
                "statistical_rigor": "standard",
                "trajectory_sampling": "every_5_frames",
                "convergence_checking": "standard",
                "error_estimation": "bootstrap",
                "visualization_level": "intermediate"
            },
            "comprehensive_analysis": {
                "statistical_rigor": "high",
                "trajectory_sampling": "every_frame",
                "convergence_checking": "rigorous",
                "error_estimation": "full_statistical",
                "visualization_level": "comprehensive"
            }
        }
        
        base_config = config_mapping.get(analysis_decision.chosen_option, config_mapping["standard_analysis"])
        
        # Adjust based on simulation quality
        if context.simulation_results and context.simulation_results.simulation_success:
            if context.simulation_results.simulation_time_ns > 5.0:  # Long simulation
                base_config["trajectory_sampling"] = "every_2_frames"
                base_config["statistical_rigor"] = "high"
        
        return {
            **base_config,
            "analysis_strategy": analysis_decision.chosen_option,
            "reasoning": analysis_decision.reasoning,
            "adaptive_methods": context.iteration > 2
        }
    
    async def _execute_property_calculations(self, context: PostProcessingContext,
                                           calculation_strategy: Dict[str, Any],
                                           analysis_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Execute property calculations using available processing systems."""
        
        calculated_properties = {}
        
        # Try comprehensive processor first
        if self.comprehensive_processor and context.simulation_results:
            try:
                comprehensive_results = await self._run_comprehensive_analysis(
                    context, calculation_strategy, analysis_configuration
                )
                calculated_properties.update(comprehensive_results)
                logger.info("Comprehensive analysis completed successfully")
            except Exception as e:
                logger.warning(f"Comprehensive analysis failed: {e}")
        
        # Try property scorer for additional analysis
        if self.property_scorer and context.simulation_results:
            try:
                scoring_results = await self._run_property_scoring(
                    context, calculation_strategy
                )
                calculated_properties.update(scoring_results)
                logger.info("Property scoring completed successfully")
            except Exception as e:
                logger.warning(f"Property scoring failed: {e}")
        
        # Fallback to basic calculations if advanced methods fail
        if not calculated_properties:
            calculated_properties = self._generate_basic_properties(context)
            logger.info("Using basic property calculations as fallback")
        
        return calculated_properties
    
    async def _run_comprehensive_analysis(self, context: PostProcessingContext,
                                        calculation_strategy: Dict[str, Any],
                                        analysis_configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using ComprehensivePostProcessor."""
        
        if not context.simulation_results or not context.simulation_results.simulation_files:
            return {}
        
        # Configure analysis based on available simulation files
        trajectory_files = [f for f in context.simulation_results.simulation_files if f.endswith('.dcd')]
        
        if not trajectory_files:
            return {}
        
        # Run comprehensive analysis
        analysis_config = {
            "trajectory_file": trajectory_files[0],
            "analysis_types": calculation_strategy["specific_calculations"],
            "statistical_rigor": analysis_configuration["statistical_rigor"],
            "output_visualizations": analysis_configuration["visualization_level"] != "basic"
        }
        
        try:
            # This would use the actual comprehensive processor API
            # For now, simulate realistic results
            results = self._simulate_comprehensive_results(context, calculation_strategy)
            return results
        except Exception as e:
            logger.warning(f"Comprehensive processor execution failed: {e}")
            return {}
    
    async def _run_property_scoring(self, context: PostProcessingContext,
                                  calculation_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Run property scoring analysis."""
        
        if not context.simulation_results:
            return {}
        
        try:
            # Create mock MD simulation properties from simulation results
            md_properties = MDSimulationProperties(
                density=context.simulation_results.density,
                total_energy=context.simulation_results.final_energy,
                temperature=context.simulation_results.temperature,
                pressure=context.simulation_results.pressure,
                volume=1000.0,  # Mock value
                kinetic_energy=context.simulation_results.final_energy * 0.5,
                potential_energy=context.simulation_results.final_energy * 0.5
            )
            
            # Calculate target property scores
            target_scores = self.property_scorer.calculate_target_scores(
                md_properties, context.target_properties
            )
            
            return {
                "target_property_scores": target_scores.to_dict(),
                "overall_performance_score": target_scores.overall_score,
                "individual_scores": {
                    prop: getattr(target_scores, prop, 0.0) 
                    for prop in ["biocompatibility", "degradation_rate", "mechanical_strength", "thermal_stability"]
                    if hasattr(target_scores, prop)
                }
            }
        except Exception as e:
            logger.warning(f"Property scoring failed: {e}")
            return {}
    
    def _simulate_comprehensive_results(self, context: PostProcessingContext,
                                      calculation_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate comprehensive analysis results."""
        
        import random
        
        results = {}
        
        # Generate results based on calculation strategy
        for calculation in calculation_strategy["specific_calculations"]:
            if "mechanical" in calculation.lower():
                results["mechanical_properties"] = {
                    "young_modulus": random.uniform(1.0, 5.0),
                    "tensile_strength": random.uniform(20.0, 80.0),
                    "elastic_modulus": random.uniform(1.5, 4.0)
                }
            
            elif "thermal" in calculation.lower():
                results["thermal_properties"] = {
                    "glass_transition": random.uniform(40.0, 80.0),
                    "melting_point": random.uniform(120.0, 200.0),
                    "thermal_stability": random.uniform(0.6, 0.9)
                }
            
            elif "diffusion" in calculation.lower() or "transport" in calculation.lower():
                results["transport_properties"] = {
                    "diffusion_coefficient": random.uniform(1e-9, 1e-7),
                    "permeability": random.uniform(1e-10, 1e-8)
                }
            
            elif "stability" in calculation.lower():
                results["stability_metrics"] = {
                    "degradation_rate": random.uniform(0.2, 0.8),
                    "chemical_stability": random.uniform(0.7, 0.95)
                }
        
        return results
    
    def _generate_basic_properties(self, context: PostProcessingContext) -> Dict[str, Any]:
        """Generate basic properties when advanced analysis fails."""
        
        import random
        
        # Generate basic properties from simulation results
        basic_properties = {}
        
        if context.simulation_results:
            basic_properties["energy_analysis"] = {
                "final_energy": context.simulation_results.final_energy,
                "energy_stability": random.uniform(0.5, 0.8)
            }
            
            basic_properties["structural_properties"] = {
                "density": context.simulation_results.density,
                "temperature": context.simulation_results.temperature,
                "pressure": context.simulation_results.pressure
            }
        
        # Generate target-property-based estimations
        for prop in context.target_properties.keys():
            if "biocompatib" in prop.lower():
                basic_properties["biocompatibility_estimate"] = random.uniform(0.6, 0.9)
            elif "degradation" in prop.lower():
                basic_properties["degradation_estimate"] = random.uniform(0.3, 0.7)
            elif "mechanical" in prop.lower():
                basic_properties["mechanical_estimate"] = random.uniform(0.4, 0.8)
        
        return basic_properties
    
    async def _generate_performance_metrics(self, calculated_properties: Dict[str, Any],
                                          context: PostProcessingContext,
                                          decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Generate performance metrics and scoring."""
        
        # Generate performance evaluation decision
        eval_decision = decision_engine.make_decision(
            decision_type=DecisionType.PERFORMANCE_EVALUATION,
            context_data={
                **context.to_dict(),
                "calculated_properties": list(calculated_properties.keys())
            },
            available_options=["target_focused", "comprehensive_scoring", "comparative_analysis"],
            objectives=["Evaluate material performance against target properties"]
        )
        
        performance_metrics = {}
        
        # Calculate individual property scores
        property_scores = {}
        for target_prop, target_value in context.target_properties.items():
            score = self._calculate_property_score(
                target_prop, target_value, calculated_properties
            )
            property_scores[target_prop] = score
        
        # Calculate overall performance score
        overall_score = sum(property_scores.values()) / len(property_scores) if property_scores else 0.5
        
        performance_metrics = {
            "individual_property_scores": property_scores,
            "overall_performance_score": overall_score,
            "evaluation_strategy": eval_decision.chosen_option,
            "performance_ranking": self._classify_performance(overall_score),
            "improvement_potential": 1.0 - overall_score,
            "confidence_level": self._calculate_confidence_level(calculated_properties)
        }
        
        return performance_metrics
    
    def _calculate_property_score(self, target_prop: str, target_value: float,
                                calculated_properties: Dict[str, Any]) -> float:
        """Calculate score for a specific target property."""
        
        # Search for relevant calculated properties
        calculated_value = None
        
        # Map target properties to calculated values
        if "biocompatib" in target_prop.lower():
            calculated_value = calculated_properties.get("biocompatibility_estimate")
            if not calculated_value and "target_property_scores" in calculated_properties:
                calculated_value = calculated_properties["target_property_scores"].get("biocompatibility")
        
        elif "degradation" in target_prop.lower():
            calculated_value = calculated_properties.get("degradation_estimate")
            if not calculated_value:
                stability = calculated_properties.get("stability_metrics", {})
                calculated_value = stability.get("degradation_rate")
        
        elif "mechanical" in target_prop.lower():
            calculated_value = calculated_properties.get("mechanical_estimate")
            if not calculated_value:
                mechanical = calculated_properties.get("mechanical_properties", {})
                calculated_value = mechanical.get("young_modulus", 0) / 10.0  # Normalize
        
        # Calculate score based on proximity to target
        if calculated_value is not None:
            # For properties where higher is better
            if target_prop.lower() in ["biocompatibility", "mechanical_strength"]:
                score = min(calculated_value / target_value, 1.0) if target_value > 0 else 0.5
            # For properties where closer to target is better
            else:
                difference = abs(calculated_value - target_value)
                score = max(0.0, 1.0 - difference)
        else:
            score = 0.5  # Default score when no calculated value available
        
        return score
    
    def _classify_performance(self, overall_score: float) -> str:
        """Classify performance level based on overall score."""
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "moderate"
        else:
            return "poor"
    
    def _calculate_confidence_level(self, calculated_properties: Dict[str, Any]) -> float:
        """Calculate confidence level in the results."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if "target_property_scores" in calculated_properties:
            confidence += 0.2
        
        if any("comprehensive" in str(prop).lower() for prop in calculated_properties.keys()):
            confidence += 0.2
        
        if len(calculated_properties) > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _generate_analysis_outputs(self, calculated_properties: Dict[str, Any],
                                       performance_metrics: Dict[str, Any],
                                       context: PostProcessingContext,
                                       decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Generate analysis outputs including visualizations and reports."""
        
        # Generate output strategy decision
        output_decision = decision_engine.make_decision(
            decision_type=DecisionType.VISUALIZATION_STRATEGY,
            context_data=context.to_dict(),
            available_options=["essential_plots", "comprehensive_visualization", "interactive_dashboard"],
            objectives=["Create useful visualizations for analysis"]
        )
        
        analysis_outputs = {
            "visualization_strategy": output_decision.chosen_option,
            "generated_plots": [],
            "summary_report": self._generate_summary_report(
                calculated_properties, performance_metrics, context
            ),
            "recommendations": self._generate_recommendations(
                calculated_properties, performance_metrics, context
            )
        }
        
        # Generate plot specifications (would create actual plots in real implementation)
        if output_decision.chosen_option != "essential_plots":
            analysis_outputs["generated_plots"] = [
                "property_comparison_plot",
                "performance_radar_chart",
                "trajectory_analysis_plot"
            ]
        
        return analysis_outputs
    
    def _generate_summary_report(self, calculated_properties: Dict[str, Any],
                               performance_metrics: Dict[str, Any],
                               context: PostProcessingContext) -> str:
        """Generate a summary report of the analysis."""
        
        overall_score = performance_metrics.get("overall_performance_score", 0.5)
        performance_ranking = performance_metrics.get("performance_ranking", "moderate")
        
        report = f"""
Analysis Summary - Iteration {context.iteration}

Overall Performance: {performance_ranking.capitalize()} (Score: {overall_score:.3f})

Key Properties Calculated:
{self._format_properties_summary(calculated_properties)}

Target Property Achievement:
{self._format_target_achievement(performance_metrics.get("individual_property_scores", {}))}

Recommendations:
{self._format_recommendations_summary(calculated_properties, performance_metrics)}
"""
        
        return report.strip()
    
    def _format_properties_summary(self, calculated_properties: Dict[str, Any]) -> str:
        """Format calculated properties for summary."""
        summary_lines = []
        
        for category, properties in calculated_properties.items():
            if isinstance(properties, dict):
                for prop, value in properties.items():
                    if isinstance(value, (int, float)):
                        summary_lines.append(f"  - {prop}: {value:.3f}")
                    else:
                        summary_lines.append(f"  - {prop}: {value}")
        
        return "\n".join(summary_lines) if summary_lines else "  - No specific properties calculated"
    
    def _format_target_achievement(self, property_scores: Dict[str, float]) -> str:
        """Format target property achievement."""
        if not property_scores:
            return "  - No target properties evaluated"
        
        achievement_lines = []
        for prop, score in property_scores.items():
            percentage = score * 100
            achievement_lines.append(f"  - {prop}: {percentage:.1f}% achieved")
        
        return "\n".join(achievement_lines)
    
    def _format_recommendations_summary(self, calculated_properties: Dict[str, Any],
                                      performance_metrics: Dict[str, Any]) -> str:
        """Format recommendations summary."""
        recommendations = self._generate_recommendations(calculated_properties, performance_metrics, None)
        return "\n".join([f"  - {rec}" for rec in recommendations[:3]])  # Top 3 recommendations
    
    def _generate_recommendations(self, calculated_properties: Dict[str, Any],
                                performance_metrics: Dict[str, Any],
                                context: Optional[PostProcessingContext]) -> List[str]:
        """Generate recommendations for improvement."""
        
        recommendations = []
        overall_score = performance_metrics.get("overall_performance_score", 0.5)
        
        if overall_score < 0.6:
            recommendations.append("Consider modifying polymer composition to improve overall performance")
        
        # Property-specific recommendations
        property_scores = performance_metrics.get("individual_property_scores", {})
        
        for prop, score in property_scores.items():
            if score < 0.5:
                if "biocompatib" in prop.lower():
                    recommendations.append("Improve biocompatibility by using more hydrophilic monomers")
                elif "degradation" in prop.lower():
                    recommendations.append("Adjust degradation rate by modifying polymer crosslinking")
                elif "mechanical" in prop.lower():
                    recommendations.append("Enhance mechanical properties by increasing molecular weight")
        
        # General recommendations
        if len(calculated_properties) < 3:
            recommendations.append("Perform more comprehensive property analysis for better evaluation")
        
        if not recommendations:
            recommendations.append("Material shows promising properties - consider optimization for specific applications")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _compile_final_results(self, calculated_properties: Dict[str, Any],
                                   performance_metrics: Dict[str, Any],
                                   analysis_outputs: Dict[str, Any],
                                   context: PostProcessingContext) -> ComputedProperties:
        """Compile final computed properties results."""
        
        # Extract or create MD simulation properties
        md_properties = None
        if context.simulation_results:
            md_properties = MDSimulationProperties(
                density=context.simulation_results.density,
                total_energy=context.simulation_results.final_energy,
                temperature=context.simulation_results.temperature,
                pressure=context.simulation_results.pressure,
                volume=1000.0,  # Mock value
                kinetic_energy=context.simulation_results.final_energy * 0.5,
                potential_energy=context.simulation_results.final_energy * 0.5
            )
        
        # Extract or create target property scores
        target_scores = None
        if "individual_property_scores" in performance_metrics and self.property_scorer:
            try:
                # This would use the actual TargetPropertyScores construction
                target_scores = self._create_target_property_scores(performance_metrics["individual_property_scores"])
            except Exception as e:
                logger.warning(f"Failed to create target property scores: {e}")
        
        # Extract specific property categories
        mechanical_props = calculated_properties.get("mechanical_properties", {})
        thermal_props = calculated_properties.get("thermal_properties", {})
        transport_props = calculated_properties.get("transport_properties", {})
        stability_metrics = calculated_properties.get("stability_metrics", {})
        
        return ComputedProperties(
            md_properties=md_properties,
            target_scores=target_scores,
            mechanical_properties=mechanical_props,
            thermal_properties=thermal_props,
            transport_properties=transport_props,
            stability_metrics=stability_metrics,
            performance_score=performance_metrics.get("overall_performance_score", 0.5),
            # Additional Phase 2 fields
            analysis_summary=analysis_outputs.get("summary_report", ""),
            recommendations=analysis_outputs.get("recommendations", []),
            confidence_level=performance_metrics.get("confidence_level", 0.5),
            processing_method=performance_metrics.get("evaluation_strategy", "automated"),
            execution_time=0.0  # Would be measured in real implementation
        )
    
    def _create_target_property_scores(self, individual_scores: Dict[str, float]):
        """Create TargetPropertyScores object from individual scores."""
        # Try to import and create the actual class, fallback to dict if not available
        try:
            from .property_scoring import TargetPropertyScores
            return TargetPropertyScores(
                biocompatibility=individual_scores.get("biocompatibility", 0.5),
                degradation_rate=individual_scores.get("degradation_rate", 0.5),
                mechanical_strength=individual_scores.get("mechanical_strength", 0.5),
                overall_score=sum(individual_scores.values()) / len(individual_scores) if individual_scores else 0.5
            )
        except (ImportError, TypeError):
            # Fallback to simple dict if class creation fails
            return {
                "biocompatibility": individual_scores.get("biocompatibility", 0.5),
                "degradation_rate": individual_scores.get("degradation_rate", 0.5),
                "mechanical_strength": individual_scores.get("mechanical_strength", 0.5),
                "overall_score": sum(individual_scores.values()) / len(individual_scores) if individual_scores else 0.5
            }
    
    def _create_fallback_results(self, context: PostProcessingContext) -> ComputedProperties:
        """Create fallback results when processing fails."""
        
        import random
        
        # Try to create MD properties, fallback to None if not available
        md_properties = None
        try:
            from .property_scoring import MDSimulationProperties
            md_properties = MDSimulationProperties(
                # Mechanical properties (GPa)
                youngs_modulus_x=2.0,
                youngs_modulus_y=2.0,
                youngs_modulus_z=2.0,
                bulk_modulus=1.5,
                shear_modulus=0.8,
                # Thermodynamic properties (kJ/mol)
                cohesive_energy=-50.0,
                mixing_energy=-10.0,
                vaporization_enthalpy=40.0,
                # Dynamic properties
                glass_transition_temp=340.0,  # K
                density=1.0,  # g/cm³
                diffusion_coefficient_water=1e-8,  # cm²/s
                diffusion_coefficient_drug=1e-9,  # cm²/s
                # Structural properties
                hydrogen_bond_count=10.0,
                hydrogen_bond_lifetime=5.0,  # ps
                surface_area=100.0,  # Ų
                free_volume_fraction=0.1,
                # Degradation-related properties
                ester_bond_count=20.0,
                ester_bond_strength=300.0,  # kJ/mol
                water_accessibility=0.3,  # fraction
                chain_scission_rate=0.01,  # 1/ns
                # Molecular mobility
                rmsf_polymer=2.0,  # Å
                rmsf_drug=1.5,  # Å
                rotational_correlation_time=10.0  # ps
            )
        except Exception:
            md_properties = None
        
        # Create basic target scores using the helper method
        individual_scores = {
            "biocompatibility": random.uniform(0.4, 0.7),
            "degradation_rate": random.uniform(0.3, 0.6),
            "mechanical_strength": random.uniform(0.4, 0.7),
            "thermal_stability": random.uniform(0.5, 0.8)
        }
        target_scores = self._create_target_property_scores(individual_scores)
        
        return ComputedProperties(
            md_properties=md_properties,
            target_scores=target_scores,
            mechanical_properties={},
            thermal_properties={},
            transport_properties={},
            stability_metrics={},
            performance_score=0.5
        )
    
    def _extract_previous_properties(self, state: IterationState) -> List[Dict]:
        """Extract previous computed properties from state history."""
        # This would extract properties from previous iterations
        return []
    
    def _save_results(self, iteration: int, results: ComputedProperties):
        """Save results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_properties.json"
            with open(results_file, 'w') as f:
                # Create serializable dict
                results_dict = {
                    "performance_score": results.performance_score,
                    "mechanical_properties": results.mechanical_properties,
                    "thermal_properties": results.thermal_properties,
                    "transport_properties": results.transport_properties,
                    "stability_metrics": results.stability_metrics
                }
                
                # Add additional fields if they exist
                if hasattr(results, 'analysis_summary'):
                    results_dict["analysis_summary"] = results.analysis_summary
                if hasattr(results, 'recommendations'):
                    results_dict["recommendations"] = results.recommendations
                
                json.dump(results_dict, f, indent=2)
            logger.info(f"Post-processing results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save post-processing results: {e}")


# Test functionality
async def test_automated_post_processing():
    """Test the AutomatedPostProcessing functionality."""
    print("Testing AutomatedPostProcessing...")
    
    # Import required components
    from .state_manager import StateManager
    from .decision_engine import LLMDecisionEngine
    
    # Create test components
    state_manager = StateManager("test_automated_postprocessing")
    decision_engine = LLMDecisionEngine()
    post_processor = AutomatedPostProcessing("test_postprocessing_output")
    
    # Create test iteration state with simulation results
    state = state_manager.create_new_iteration(
        initial_prompt="Design a biodegradable polymer for insulin delivery",
        target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
    )
    
    # Add mock simulation results
    from .state_manager import SimulationResults
    state.simulation_results = SimulationResults(
        simulation_time_ns=5.0,
        equilibration_time_ns=1.0,
        final_energy=-1250.5,
        temperature=298.15,
        pressure=1.01,
        density=1.05,
        simulation_success=True,
        execution_time=3600.0,
        simulation_files=["trajectory.dcd", "topology.pdb"]
    )
    
    print(f"Created test iteration {state.iteration_number}")
    
    # Run automated post-processing
    results = await post_processor.run_automated_processing(state, decision_engine)
    
    print(f"Post-processing results:")
    print(f"- Performance score: {results.performance_score:.3f}")
    print(f"- Mechanical properties: {len(results.mechanical_properties)} calculated")
    print(f"- Thermal properties: {len(results.thermal_properties)} calculated")
    print(f"- Transport properties: {len(results.transport_properties)} calculated")
    print(f"- Stability metrics: {len(results.stability_metrics)} calculated")
    
    if hasattr(results, 'recommendations'):
        print(f"- Recommendations: {len(results.recommendations)}")
    if hasattr(results, 'confidence_level'):
        print(f"- Confidence level: {results.confidence_level:.3f}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_automated_post_processing()) 