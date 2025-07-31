# Core infrastructure confirms rules are active
"""
Active Learning Material Discovery System

This module provides autonomous material discovery through active learning loops
that combine literature mining, molecular generation, MD simulation, and RAG-based feedback.
"""

from .orchestrator import ActiveLearningOrchestrator
from .state_manager import IterationState, StateManager, IterationStatus
from .decision_engine import LLMDecisionEngine
from .loop_controller import LoopController, ConvergenceConfig, ResourceLimits, QualityGates
from .property_scoring import PropertyScoring, MDSimulationProperties, TargetPropertyScores

__all__ = [
    'ActiveLearningOrchestrator',
    'IterationState', 
    'StateManager',
    'IterationStatus',
    'LLMDecisionEngine',
    'LoopController',
    'ConvergenceConfig',
    'ResourceLimits',
    'QualityGates',
    'PropertyScoring',
    'MDSimulationProperties',
    'TargetPropertyScores'
]

__version__ = "0.1.0" 