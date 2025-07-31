#!/usr/bin/env python3
"""
Standardized Interfaces - Phase 2 Implementation

This module provides standardized interfaces, error handling, and base classes
for all automated components in the active learning material discovery system.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from .state_manager import IterationState
from .decision_engine import LLMDecisionEngine

logger = logging.getLogger(__name__)


class ActiveLearningError(Exception):
    """Base exception for active learning system errors."""
    pass


class ComponentInitializationError(ActiveLearningError):
    """Raised when a component fails to initialize properly."""
    pass


class ComponentExecutionError(ActiveLearningError):
    """Raised when a component fails during execution."""
    pass


class DecisionEngineError(ActiveLearningError):
    """Raised when decision engine fails to make a decision."""
    pass


class DataValidationError(ActiveLearningError):
    """Raised when input data validation fails."""
    pass


class ComponentStatus:
    """Status tracking for automated components."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.status = "initialized"
        self.last_execution = None
        self.error_count = 0
        self.success_count = 0
        self.last_error = None
        self.execution_history = []
    
    def record_execution_start(self, operation: str):
        """Record the start of an execution."""
        execution_record = {
            "operation": operation,
            "start_time": datetime.now(),
            "status": "running"
        }
        self.execution_history.append(execution_record)
        self.status = "running"
        logger.info(f"{self.component_name}: Started {operation}")
    
    def record_execution_success(self, operation: str, duration: float = None):
        """Record successful execution."""
        if self.execution_history:
            self.execution_history[-1]["status"] = "success"
            self.execution_history[-1]["end_time"] = datetime.now()
            if duration:
                self.execution_history[-1]["duration"] = duration
        
        self.success_count += 1
        self.status = "idle"
        self.last_execution = datetime.now()
        logger.info(f"{self.component_name}: Successfully completed {operation}")
    
    def record_execution_error(self, operation: str, error: Exception, duration: float = None):
        """Record execution error."""
        if self.execution_history:
            self.execution_history[-1]["status"] = "error"
            self.execution_history[-1]["error"] = str(error)
            self.execution_history[-1]["end_time"] = datetime.now()
            if duration:
                self.execution_history[-1]["duration"] = duration
        
        self.error_count += 1
        self.status = "error"
        self.last_error = error
        logger.error(f"{self.component_name}: Failed {operation} - {error}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        total_executions = self.success_count + self.error_count
        success_rate = self.success_count / total_executions if total_executions > 0 else 1.0
        
        return {
            "component_name": self.component_name,
            "current_status": self.status,
            "success_rate": success_rate,
            "total_executions": total_executions,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_error": str(self.last_error) if self.last_error else None
        }


class BaseAutomatedComponent(ABC):
    """Base class for all automated components with standardized interface."""
    
    def __init__(self, component_name: str, storage_path: Optional[str] = None):
        """Initialize base component.
        
        Args:
            component_name: Name of the component for logging and tracking
            storage_path: Optional path for component-specific storage
        """
        self.component_name = component_name
        self.storage_path = Path(storage_path) if storage_path else Path(f"component_{component_name}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Status tracking
        self.status = ComponentStatus(component_name)
        
        # Configuration
        self.config = self._load_default_config()
        
        # Initialize component-specific resources
        try:
            self._initialize_resources()
            logger.info(f"{self.component_name} initialized successfully")
        except Exception as e:
            raise ComponentInitializationError(f"Failed to initialize {self.component_name}: {e}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for the component."""
        return {
            "retry_attempts": 3,
            "timeout_seconds": 300,
            "enable_caching": True,
            "log_level": "INFO"
        }
    
    @abstractmethod
    def _initialize_resources(self):
        """Initialize component-specific resources. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_core_logic(self, state: IterationState, 
                                decision_engine: LLMDecisionEngine, 
                                **kwargs) -> Any:
        """Execute core component logic. Must be implemented by subclasses."""
        pass
    
    async def execute_with_error_handling(self, state: IterationState,
                                        decision_engine: LLMDecisionEngine,
                                        **kwargs) -> Any:
        """Execute component with standardized error handling and retry logic."""
        
        operation_name = f"execute_{self.component_name}"
        self.status.record_execution_start(operation_name)
        start_time = datetime.now()
        
        # Validate inputs
        try:
            self._validate_inputs(state, decision_engine, **kwargs)
        except Exception as e:
            self.status.record_execution_error(operation_name, e)
            raise DataValidationError(f"Input validation failed for {self.component_name}: {e}")
        
        # Execute with retry logic
        last_error = None
        for attempt in range(self.config["retry_attempts"]):
            try:
                # Execute core logic
                result = await asyncio.wait_for(
                    self._execute_core_logic(state, decision_engine, **kwargs),
                    timeout=self.config["timeout_seconds"]
                )
                
                # Validate output
                self._validate_output(result)
                
                # Record success
                duration = (datetime.now() - start_time).total_seconds()
                self.status.record_execution_success(operation_name, duration)
                
                # Cache result if enabled
                if self.config["enable_caching"]:
                    self._cache_result(state.iteration_number, result)
                
                return result
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"{self.component_name} attempt {attempt + 1} timed out")
                
            except Exception as e:
                last_error = e
                logger.warning(f"{self.component_name} attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, break
                if attempt == self.config["retry_attempts"] - 1:
                    break
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
        
        # All attempts failed
        duration = (datetime.now() - start_time).total_seconds()
        self.status.record_execution_error(operation_name, last_error, duration)
        
        # Try to return fallback result
        try:
            fallback_result = self._create_fallback_result(state)
            logger.warning(f"{self.component_name} using fallback result after {self.config['retry_attempts']} failed attempts")
            return fallback_result
        except Exception as fallback_error:
            raise ComponentExecutionError(
                f"{self.component_name} failed after {self.config['retry_attempts']} attempts. "
                f"Last error: {last_error}. Fallback failed: {fallback_error}"
            )
    
    def _validate_inputs(self, state: IterationState, decision_engine: LLMDecisionEngine, **kwargs):
        """Validate component inputs. Can be overridden by subclasses."""
        if not isinstance(state, IterationState):
            raise ValueError("state must be an IterationState instance")
        
        if not isinstance(decision_engine, LLMDecisionEngine):
            raise ValueError("decision_engine must be an LLMDecisionEngine instance")
        
        if not state.target_properties:
            raise ValueError("state must have target_properties defined")
    
    def _validate_output(self, result: Any):
        """Validate component output. Can be overridden by subclasses."""
        if result is None:
            raise ValueError("Component result cannot be None")
    
    @abstractmethod
    def _create_fallback_result(self, state: IterationState) -> Any:
        """Create fallback result when execution fails. Must be implemented by subclasses."""
        pass
    
    def _cache_result(self, iteration: int, result: Any):
        """Cache component result. Can be overridden by subclasses."""
        try:
            cache_file = self.storage_path / f"iteration_{iteration}_cache.json"
            # Simple caching implementation would go here
            # For now, just log that caching was attempted
            logger.debug(f"Caching result for {self.component_name} iteration {iteration}")
        except Exception as e:
            logger.warning(f"Failed to cache result for {self.component_name}: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        return self.status.get_health_status()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update component configuration."""
        self.config.update(new_config)
        logger.info(f"Updated configuration for {self.component_name}")


class ComponentRegistry:
    """Registry for managing all automated components."""
    
    def __init__(self):
        self.components: Dict[str, BaseAutomatedComponent] = {}
        self.initialization_order = []
    
    def register_component(self, component: BaseAutomatedComponent):
        """Register a component."""
        self.components[component.component_name] = component
        self.initialization_order.append(component.component_name)
        logger.info(f"Registered component: {component.component_name}")
    
    def get_component(self, name: str) -> Optional[BaseAutomatedComponent]:
        """Get a component by name."""
        return self.components.get(name)
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all components."""
        return {
            name: component.get_health_status() 
            for name, component in self.components.items()
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        all_status = self.get_all_health_status()
        
        total_components = len(all_status)
        healthy_components = sum(1 for status in all_status.values() 
                               if status["current_status"] not in ["error", "failed"])
        
        overall_success_rate = sum(status["success_rate"] for status in all_status.values()) / total_components if total_components > 0 else 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_components": total_components,
            "healthy_components": healthy_components,
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0.0,
            "overall_success_rate": overall_success_rate,
            "system_status": "healthy" if healthy_components == total_components else "degraded"
        }


class ValidationUtils:
    """Utility functions for data validation across components."""
    
    @staticmethod
    def validate_target_properties(target_properties: Dict[str, float]) -> bool:
        """Validate target properties format and values."""
        if not isinstance(target_properties, dict):
            return False
        
        for prop, value in target_properties.items():
            if not isinstance(prop, str) or not isinstance(value, (int, float)):
                return False
            
            if value < 0 or value > 1:  # Assuming normalized values
                return False
        
        return True
    
    @staticmethod
    def validate_iteration_state(state: IterationState) -> List[str]:
        """Validate iteration state and return list of issues."""
        issues = []
        
        if not isinstance(state, IterationState):
            issues.append("state is not an IterationState instance")
            return issues
        
        if state.iteration_number < 1:
            issues.append("iteration_number must be >= 1")
        
        if not state.target_properties:
            issues.append("target_properties cannot be empty")
        elif not ValidationUtils.validate_target_properties(state.target_properties):
            issues.append("target_properties format is invalid")
        
        if not state.initial_prompt or len(state.initial_prompt.strip()) < 10:
            issues.append("initial_prompt must be a meaningful description")
        
        return issues
    
    @staticmethod
    def validate_decision_engine(decision_engine: LLMDecisionEngine) -> List[str]:
        """Validate decision engine and return list of issues."""
        issues = []
        
        if not isinstance(decision_engine, LLMDecisionEngine):
            issues.append("decision_engine is not an LLMDecisionEngine instance")
        
        # Additional validation could be added here
        
        return issues


# Global component registry instance
component_registry = ComponentRegistry()


def register_component(component: BaseAutomatedComponent):
    """Convenience function to register a component globally."""
    component_registry.register_component(component)


def get_system_health() -> Dict[str, Any]:
    """Convenience function to get system health."""
    return component_registry.get_system_health_summary()


# Example usage and testing
if __name__ == "__main__":
    # This would demonstrate how to use the interfaces
    print("Active Learning Interfaces Module")
    print("This module provides standardized interfaces for automated components.")
    
    # Test validation utilities
    valid_props = {"biocompatibility": 0.9, "degradation_rate": 0.5}
    invalid_props = {"biocompatibility": 1.5, "degradation_rate": "fast"}
    
    print(f"Valid properties check: {ValidationUtils.validate_target_properties(valid_props)}")
    print(f"Invalid properties check: {ValidationUtils.validate_target_properties(invalid_props)}")
    
    # Test component registry
    print(f"System health: {get_system_health()}") 