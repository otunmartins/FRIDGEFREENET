# Core infrastructure confirms rules are active
"""
State Management for Active Learning Material Discovery

This module manages the state of active learning iterations, tracking progress through
literature mining, molecular generation, MD simulation, and property computation stages.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid

from .property_scoring import PropertyScoring, MDSimulationProperties, TargetPropertyScores

logger = logging.getLogger(__name__)


class IterationStatus(Enum):
    """Status of an active learning iteration"""
    INITIALIZED = "initialized"
    LITERATURE_MINING = "literature_mining"
    PIECEWISE_GENERATION = "piecewise_generation"
    MD_SIMULATION = "md_simulation" 
    POST_PROCESSING = "post_processing"
    RAG_ANALYSIS = "rag_analysis"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LiteratureResults:
    """Results from literature mining stage"""
    papers_found: int
    relevant_papers: int
    extracted_properties: Dict[str, List[float]]
    synthesis_routes: List[str]
    query_used: str
    material_candidates: Optional[List[str]] = None
    extraction_strategy: Optional[str] = None
    papers_analyzed: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None
    
    # Legacy fields for backward compatibility
    query: Optional[str] = None
    key_insights: Optional[List[str]] = None
    material_suggestions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "papers_found": self.papers_found,
            "relevant_papers": self.relevant_papers,
            "extracted_properties": self.extracted_properties,
            "synthesis_routes": self.synthesis_routes,
            "query_used": self.query_used,
            "material_candidates": self.material_candidates or [],
            "extraction_strategy": self.extraction_strategy,
            "papers_analyzed": self.papers_analyzed or [],
            "execution_time": self.execution_time
        }


@dataclass
class GeneratedMolecules:
    """Results from piecewise generation stage"""
    monomers_generated: List[str]
    psmiles_strings: List[str]
    generation_method: str
    diversity_score: float
    validity_score: float
    execution_time: float
    # Additional Phase 2 fields
    molecules: Optional[List[Dict[str, Any]]] = None
    generation_parameters: Optional[Dict[str, Any]] = None
    success_rate: Optional[float] = None


@dataclass
class SimulationResults:
    """Results from MD simulation stage"""
    simulation_time_ns: float
    equilibration_time_ns: float
    final_energy: float
    temperature: float
    pressure: float
    density: float
    simulation_success: bool
    execution_time: float
    # Raw MD properties for property scoring
    md_properties: Optional[MDSimulationProperties] = None
    # Additional Phase 2 fields
    simulation_files: Optional[List[str]] = None
    energy_data: Optional[Dict[str, List[float]]] = None
    trajectory_length: Optional[float] = None
    force_field_used: Optional[str] = None
    convergence_achieved: Optional[bool] = None


@dataclass 
class ComputedProperties:
    """Computed material properties from post-processing"""
    # Raw MD simulation properties
    md_properties: Optional[MDSimulationProperties]
    
    # Target property scores (0-1 scale)
    target_scores: Optional[TargetPropertyScores]
    
    # Specific property categories
    mechanical_properties: Dict[str, float]
    thermal_properties: Dict[str, float]
    transport_properties: Dict[str, float]
    stability_metrics: Dict[str, float]
    performance_score: float
    
    # Additional Phase 2 fields
    analysis_summary: Optional[str] = None
    recommendations: Optional[List[str]] = None
    confidence_level: Optional[float] = None
    processing_method: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Legacy fields for backward compatibility
    property_details: Optional[Dict[str, Any]] = None
    computation_method: Optional[str] = None
    confidence_score: Optional[float] = None


@dataclass
class RAGAnalysis:
    """Results from RAG analysis and feedback generation"""
    property_analysis: str
    similar_materials: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    next_iteration_prompt: str
    confidence_score: float
    # Additional Phase 2 fields
    benchmark_comparison: Optional[str] = None
    research_insights: Optional[List[Dict[str, Any]]] = None
    iteration_strategy: Optional[str] = None
    expected_improvements: Optional[Dict[str, float]] = None
    competitive_analysis: Optional[Dict[str, Any]] = None
    # Legacy fields for backward compatibility
    online_search_results: Optional[List[str]] = None
    reasoning_chain: Optional[List[str]] = None
    execution_time: Optional[float] = None


@dataclass
class IterationState:
    """Complete state of a single active learning iteration"""
    
    # Basic metadata
    iteration_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: IterationStatus = IterationStatus.INITIALIZED
    
    # Component results
    literature_results: Optional[LiteratureResults] = None
    generated_molecules: Optional[GeneratedMolecules] = None
    simulation_results: Optional[SimulationResults] = None
    computed_properties: Optional[ComputedProperties] = None
    rag_analysis: Optional[RAGAnalysis] = None
    
    # Performance metrics
    overall_score: float = 0.0
    improvement_over_previous: float = 0.0
    
    # Tracking and logging
    errors: List[str] = None
    warnings: List[str] = None
    reasoning_log: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists to avoid mutable default arguments"""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.reasoning_log is None:
            self.reasoning_log = []
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def update_status(self, new_status: IterationStatus, message: str = ""):
        """Update iteration status with logging"""
        old_status = self.status
        self.status = new_status
        
        log_message = f"Iteration {self.iteration_number} - [{datetime.now().isoformat()}] Status Change to {new_status.value}"
        if message:
            log_message += f": {message}"
            
        self.reasoning_log.append(log_message)
        logger.info(log_message)
    
    def add_error(self, error_message: str):
        """Add an error to the iteration"""
        self.errors.append(f"[{datetime.now().isoformat()}] {error_message}")
        logger.error(f"Iteration {self.iteration_number}: {error_message}")
    
    def add_warning(self, warning_message: str):
        """Add a warning to the iteration"""
        self.warnings.append(f"[{datetime.now().isoformat()}] {warning_message}")
        logger.warning(f"Iteration {self.iteration_number}: {warning_message}")
    
    def calculate_score(self) -> float:
        """Calculate overall iteration score based on property scores"""
        if self.computed_properties and self.computed_properties.target_scores:
            self.overall_score = self.computed_properties.target_scores.overall_score
        else:
            # Fallback scoring based on completion
            completion_score = 0.0
            if self.literature_results:
                completion_score += 0.2
            if self.generated_molecules:
                completion_score += 0.2
            if self.simulation_results:
                completion_score += 0.3
            if self.computed_properties:
                completion_score += 0.2
            if self.rag_analysis:
                completion_score += 0.1
            
            self.overall_score = completion_score
        
        return self.overall_score
    
    def is_complete(self) -> bool:
        """Check if iteration has completed all stages"""
        return self.status == IterationStatus.COMPLETED
    
    def get_completion_percentage(self) -> float:
        """Get completion percentage based on completed stages"""
        stages = [
            self.literature_results is not None,
            self.generated_molecules is not None, 
            self.simulation_results is not None,
            self.computed_properties is not None,
            self.rag_analysis is not None
        ]
        return sum(stages) / len(stages) * 100


class StateManager:
    """
    Manages state persistence and retrieval for active learning iterations
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize state manager
        
        Args:
            storage_path: Directory path for storing iteration states
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize property scoring system
        self.property_scorer = PropertyScoring()
        
        # Load existing states
        self.states: Dict[int, IterationState] = {}
        self._load_existing_states()
        
        logger.info(f"StateManager initialized with storage at {storage_path}")
    
    def _load_existing_states(self):
        """Load existing iteration states from storage"""
        state_files = list(self.storage_path.glob("iteration_*.json"))
        
        for state_file in state_files:
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to IterationState object
                state = self._dict_to_iteration_state(data)
                self.states[state.iteration_number] = state
                
            except Exception as e:
                logger.error(f"Failed to load state from {state_file}: {e}")
        
        if self.states:
            max_iteration = max(self.states.keys())
            logger.info(f"Loaded {len(self.states)} existing states. Current iteration: {max_iteration}")
        else:
            logger.info("No existing states found")
    
    def _dict_to_iteration_state(self, data: Dict[str, Any]) -> IterationState:
        """Convert dictionary back to IterationState object"""
        
        # Handle datetime fields
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Handle enum field
        if 'status' in data:
            data['status'] = IterationStatus(data['status'])
        
        # Reconstruct nested dataclass objects
        if 'literature_results' in data and data['literature_results']:
            data['literature_results'] = LiteratureResults(**data['literature_results'])
        
        if 'generated_molecules' in data and data['generated_molecules']:
            data['generated_molecules'] = GeneratedMolecules(**data['generated_molecules'])
        
        if 'simulation_results' in data and data['simulation_results']:
            sim_data = data['simulation_results']
            # Handle nested md_properties
            if 'md_properties' in sim_data and sim_data['md_properties']:
                sim_data['md_properties'] = MDSimulationProperties(**sim_data['md_properties'])
            data['simulation_results'] = SimulationResults(**sim_data)
        
        if 'computed_properties' in data and data['computed_properties']:
            comp_data = data['computed_properties']
            # Reconstruct nested objects
            if 'md_properties' in comp_data:
                comp_data['md_properties'] = MDSimulationProperties(**comp_data['md_properties'])
            if 'target_scores' in comp_data:
                comp_data['target_scores'] = TargetPropertyScores(**comp_data['target_scores'])
            data['computed_properties'] = ComputedProperties(**comp_data)
        
        if 'rag_analysis' in data and data['rag_analysis']:
            data['rag_analysis'] = RAGAnalysis(**data['rag_analysis'])
        
        return IterationState(**data)
    
    def _iteration_state_to_dict(self, state: IterationState) -> Dict[str, Any]:
        """Convert IterationState to dictionary for JSON serialization"""
        def convert_value(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return asdict(obj)
            else:
                return obj
        
        # Use asdict but handle special cases
        data = asdict(state)
        
        # Convert datetime and enum fields
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        
        return data
    
    def create_new_iteration(self, initial_prompt: str = "") -> IterationState:
        """
        Create a new iteration state
        
        Args:
            initial_prompt: Initial prompt for the iteration
            
        Returns:
            New IterationState object
        """
        next_iteration = max(self.states.keys(), default=0) + 1
        
        state = IterationState(
            iteration_number=next_iteration,
            start_time=datetime.now()
        )
        
        state.update_status(IterationStatus.INITIALIZED, f"Created new iteration {next_iteration}")
        if initial_prompt:
            state.reasoning_log.append(f"Initial prompt: {initial_prompt}")
        
        self.states[next_iteration] = state
        logger.info(f"Created new iteration {next_iteration}")
        
        return state
    
    def save_state(self, state: IterationState):
        """
        Save iteration state to storage
        
        Args:
            state: IterationState to save
        """
        state_file = self.storage_path / f"iteration_{state.iteration_number}.json"
        
        try:
            # Convert to dictionary for JSON serialization
            data = self._iteration_state_to_dict(state)
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Saved state for iteration {state.iteration_number}")
            
        except Exception as e:
            logger.error(f"Failed to save state for iteration {state.iteration_number}: {e}")
            raise
    
    def get_state(self, iteration_number: int) -> Optional[IterationState]:
        """
        Get iteration state by number
        
        Args:
            iteration_number: Iteration number to retrieve
            
        Returns:
            IterationState if found, None otherwise
        """
        return self.states.get(iteration_number)
    
    def get_latest_state(self) -> Optional[IterationState]:
        """Get the most recent iteration state"""
        if not self.states:
            return None
        
        latest_iteration = max(self.states.keys())
        return self.states[latest_iteration]
    
    def get_all_states(self) -> List[IterationState]:
        """Get all iteration states sorted by iteration number"""
        return [self.states[i] for i in sorted(self.states.keys())]
    
    def update_properties_with_scoring(self, state: IterationState, md_properties: MDSimulationProperties):
        """
        Update iteration state with computed properties using the new scoring system
        
        Args:
            state: IterationState to update
            md_properties: MD simulation properties to score
        """
        
        # Calculate target property scores
        target_scores = self.property_scorer.score_material_properties(md_properties)
        
        # Create detailed property information
        property_details = {
            "biocompatibility_factors": {
                "glass_transition_temp": md_properties.glass_transition_temp,
                "hydrogen_bonding": md_properties.hydrogen_bond_count,
                "density": md_properties.density,
                "water_interaction": md_properties.diffusion_coefficient_water
            },
            "degradation_factors": {
                "chain_scission_rate": md_properties.chain_scission_rate,
                "water_diffusion": md_properties.diffusion_coefficient_water,
                "ester_bond_strength": md_properties.ester_bond_strength,
                "molecular_mobility": (md_properties.rmsf_polymer + md_properties.rmsf_drug) / 2
            },
            "mechanical_factors": {
                "youngs_modulus_avg": (md_properties.youngs_modulus_x + md_properties.youngs_modulus_y + md_properties.youngs_modulus_z) / 3,
                "bulk_modulus": md_properties.bulk_modulus,
                "shear_modulus": md_properties.shear_modulus,
                "cohesive_energy": md_properties.cohesive_energy
            }
        }
        
        # Create ComputedProperties object
        computed_props = ComputedProperties(
            md_properties=md_properties,
            target_scores=target_scores,
            property_details=property_details,
            computation_method="MD_simulation_with_literature_scoring",
            execution_time=5.0,  # Mock execution time
            confidence_score=0.85  # Mock confidence
        )
        
        # Update state
        state.computed_properties = computed_props
        state.calculate_score()  # This will use the new target_scores.overall_score
        
        logger.info(f"Updated iteration {state.iteration_number} with scored properties: "
                   f"Biocomp={target_scores.biocompatibility:.3f}, "
                   f"Degrd={target_scores.degradation_rate:.3f}, "
                   f"Mech={target_scores.mechanical_strength:.3f}, "
                   f"Overall={target_scores.overall_score:.3f}")
    
    def calculate_progress_metrics(self) -> Dict[str, Any]:
        """Calculate overall progress metrics across all iterations"""
        if not self.states:
            return {
                'total_iterations': 0,
                'completed_iterations': 0,
                'success_rate': 0.0,
                'average_score': 0.0,
                'best_score': 0.0,
                'improvement_trend': [],
                'current_iteration': 0
            }
        
        all_states = self.get_all_states()
        completed_states = [s for s in all_states if s.is_complete()]
        scores = [s.overall_score for s in all_states if s.overall_score > 0]
        
        # Calculate improvement trend (last 5 iterations)
        recent_scores = scores[-5:] if len(scores) >= 5 else scores
        improvement_trend = []
        for i in range(1, len(recent_scores)):
            improvement_trend.append(recent_scores[i] - recent_scores[i-1])
        
        return {
            'total_iterations': len(all_states),
            'completed_iterations': len(completed_states),
            'success_rate': len(completed_states) / len(all_states) if all_states else 0.0,
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'best_score': max(scores) if scores else 0.0,
            'improvement_trend': improvement_trend,
            'current_iteration': max(self.states.keys()) if self.states else 0
        }
    
    def export_results(self, export_path: str) -> Dict[str, Any]:
        """
        Export all results to a comprehensive summary
        
        Args:
            export_path: Path to save export file
            
        Returns:
            Dictionary containing all results
        """
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_iterations': len(self.states),
                'storage_path': str(self.storage_path)
            },
            'progress_metrics': self.calculate_progress_metrics(),
            'iterations': []
        }
        
        # Add all iteration data
        for state in self.get_all_states():
            iteration_data = self._iteration_state_to_dict(state)
            export_data['iterations'].append(iteration_data)
        
        # Save to file
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported results to {export_path}")
        return export_data 