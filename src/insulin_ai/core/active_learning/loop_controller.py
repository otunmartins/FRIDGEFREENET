# Core infrastructure confirms rules are active
"""
Loop Controller for Active Learning System

Manages iteration control, safety mechanisms, convergence detection, 
and early stopping criteria for the active learning loop.
"""

import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .state_manager import IterationState, IterationStatus, StateManager

# Set up logging
logger = logging.getLogger(__name__)


class LoopStatus(Enum):
    """Status of the active learning loop."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class StopReason(Enum):
    """Reasons for stopping the loop."""
    MAX_ITERATIONS = "max_iterations_reached"
    CONVERGENCE = "convergence_achieved"
    HUMAN_INTERVENTION = "human_intervention"
    RESOURCE_LIMIT = "resource_limit_exceeded"
    ERROR_THRESHOLD = "error_threshold_exceeded"
    QUALITY_GATE = "quality_gate_failure"
    TIME_LIMIT = "time_limit_exceeded"
    USER_REQUEST = "user_request"


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""
    property_tolerance: float = 0.01  # Tolerance for property improvement
    min_improvement: float = 0.05     # Minimum improvement required
    patience: int = 3                 # Number of iterations without improvement
    score_threshold: float = 0.95     # Target score threshold
    enable_early_stopping: bool = True


@dataclass 
class ResourceLimits:
    """Resource usage limits for safety."""
    max_memory_mb: int = 16000        # Maximum memory usage in MB
    max_cpu_percent: float = 90.0     # Maximum CPU usage percentage
    max_disk_gb: float = 100.0        # Maximum disk usage in GB
    max_time_hours: float = 24.0      # Maximum total execution time
    check_interval: int = 30          # Resource check interval in seconds


@dataclass
class QualityGates:
    """Quality gates for iteration validation."""
    min_confidence_score: float = 0.3  # Minimum confidence for decisions
    max_error_rate: float = 0.2         # Maximum error rate per iteration
    min_success_rate: float = 0.5       # Minimum success rate for components
    enable_validation: bool = True


@dataclass
class LoopMetrics:
    """Metrics for loop performance tracking."""
    total_runtime: float = 0.0
    iterations_completed: int = 0
    iterations_failed: int = 0
    average_iteration_time: float = 0.0
    best_score_achieved: float = 0.0
    convergence_progress: List[float] = field(default_factory=list)
    resource_usage_history: List[Dict[str, float]] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)


class LoopController:
    """Controls the active learning loop execution and monitoring."""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 convergence_config: ConvergenceConfig = None,
                 resource_limits: ResourceLimits = None,
                 quality_gates: QualityGates = None):
        """Initialize the loop controller.
        
        Args:
            max_iterations: Maximum number of iterations to run
            convergence_config: Configuration for convergence detection
            resource_limits: Resource usage limits
            quality_gates: Quality validation gates
        """
        self.max_iterations = max_iterations
        self.convergence_config = convergence_config or ConvergenceConfig()
        self.resource_limits = resource_limits or ResourceLimits()
        self.quality_gates = quality_gates or QualityGates()
        
        # Loop state
        self.status = LoopStatus.IDLE
        self.start_time: Optional[datetime] = None
        self.stop_reason: Optional[StopReason] = None
        self.current_iteration = 0
        
        # Monitoring
        self.metrics = LoopMetrics()
        self.human_intervention_callbacks: List[Callable] = []
        self.convergence_tracker = ConvergenceTracker(self.convergence_config)
        
        # Safety monitoring
        self.last_resource_check = datetime.now()
        self.resource_warnings = []
        
        logger.info(f"LoopController initialized - Max iterations: {max_iterations}")
    
    def should_continue(self, state_manager: StateManager) -> tuple[bool, Optional[StopReason]]:
        """Check if the loop should continue running.
        
        Args:
            state_manager: StateManager instance for accessing states
            
        Returns:
            Tuple of (should_continue, stop_reason)
        """
        # Check maximum iterations
        if self.current_iteration >= self.max_iterations:
            return False, StopReason.MAX_ITERATIONS
        
        # Check time limit
        if self.start_time:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours > self.resource_limits.max_time_hours:
                return False, StopReason.TIME_LIMIT
        
        # Check resource limits
        if not self._check_resource_limits():
            return False, StopReason.RESOURCE_LIMIT
        
        # Check quality gates
        current_state = state_manager.get_current_state()
        if current_state and not self._check_quality_gates(current_state):
            return False, StopReason.QUALITY_GATE
        
        # Check convergence
        completed_states = state_manager.get_completed_states()
        if len(completed_states) >= 2:  # Need at least 2 iterations for convergence
            if self.convergence_tracker.check_convergence(completed_states):
                return False, StopReason.CONVERGENCE
        
        # Check human intervention
        if self._check_human_intervention():
            return False, StopReason.HUMAN_INTERVENTION
        
        return True, None
    
    def start_loop(self) -> None:
        """Start the active learning loop."""
        if self.status == LoopStatus.RUNNING:
            logger.warning("Loop is already running")
            return
        
        self.status = LoopStatus.RUNNING
        self.start_time = datetime.now()
        self.current_iteration = 0
        self.stop_reason = None
        
        logger.info("Active learning loop started")
    
    def pause_loop(self) -> None:
        """Pause the active learning loop."""
        if self.status == LoopStatus.RUNNING:
            self.status = LoopStatus.PAUSED
            logger.info("Active learning loop paused")
    
    def resume_loop(self) -> None:
        """Resume the active learning loop."""
        if self.status == LoopStatus.PAUSED:
            self.status = LoopStatus.RUNNING
            logger.info("Active learning loop resumed")
    
    def stop_loop(self, reason: StopReason = StopReason.USER_REQUEST) -> None:
        """Stop the active learning loop.
        
        Args:
            reason: Reason for stopping the loop
        """
        self.status = LoopStatus.STOPPED
        self.stop_reason = reason
        
        if self.start_time:
            self.metrics.total_runtime = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"Active learning loop stopped - Reason: {reason.value}")
    
    def begin_iteration(self, iteration_number: int) -> None:
        """Begin a new iteration.
        
        Args:
            iteration_number: The iteration number being started
        """
        self.current_iteration = iteration_number
        self._update_resource_usage()
        logger.info(f"Beginning iteration {iteration_number}")
    
    def complete_iteration(self, state: IterationState) -> None:
        """Complete an iteration and update metrics.
        
        Args:
            state: The completed iteration state
        """
        if state.status == IterationStatus.COMPLETED:
            self.metrics.iterations_completed += 1
            
            # Update best score
            if state.overall_score > self.metrics.best_score_achieved:
                self.metrics.best_score_achieved = state.overall_score
            
            # Update convergence tracking
            self.convergence_tracker.add_iteration_score(state.overall_score)
            
        else:
            self.metrics.iterations_failed += 1
            self.metrics.error_history.extend(state.errors)
        
        # Update timing metrics
        if state.start_time and state.end_time:
            iteration_time = (state.end_time - state.start_time).total_seconds()
            total_time = self.metrics.iterations_completed * self.metrics.average_iteration_time + iteration_time
            self.metrics.average_iteration_time = total_time / (self.metrics.iterations_completed + 1)
        
        logger.info(f"Completed iteration {state.iteration_number} - Score: {state.overall_score}")
    
    def _check_resource_limits(self) -> bool:
        """Check if resource usage is within limits.
        
        Returns:
            True if within limits, False otherwise
        """
        now = datetime.now()
        
        # Check if it's time for resource monitoring
        if (now - self.last_resource_check).seconds < self.resource_limits.check_interval:
            return True
        
        self.last_resource_check = now
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            if memory_mb > self.resource_limits.max_memory_mb:
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.resource_limits.max_memory_mb}MB")
                return False
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.resource_limits.max_cpu_percent:
                logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds limit {self.resource_limits.max_cpu_percent}%")
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_gb = disk.used / (1024**3)
            if disk_gb > self.resource_limits.max_disk_gb:
                logger.warning(f"Disk usage {disk_gb:.1f}GB exceeds limit {self.resource_limits.max_disk_gb}GB")
                return False
            
            # Record resource usage
            self.metrics.resource_usage_history.append({
                "timestamp": now.timestamp(),
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "disk_gb": disk_gb
            })
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return True  # Continue on error
    
    def _check_quality_gates(self, state: IterationState) -> bool:
        """Check if iteration meets quality gates.
        
        Args:
            state: Iteration state to validate
            
        Returns:
            True if quality gates pass, False otherwise
        """
        if not self.quality_gates.enable_validation:
            return True
        
        # Check error rate
        total_steps = 5  # Number of main steps in iteration
        error_rate = len(state.errors) / total_steps
        if error_rate > self.quality_gates.max_error_rate:
            logger.warning(f"Error rate {error_rate:.2f} exceeds limit {self.quality_gates.max_error_rate}")
            return False
        
        # Check overall score (if available)
        if state.overall_score < self.quality_gates.min_success_rate:
            logger.warning(f"Overall score {state.overall_score:.2f} below minimum {self.quality_gates.min_success_rate}")
            return False
        
        return True
    
    def _check_human_intervention(self) -> bool:
        """Check if human intervention has been requested.
        
        Returns:
            True if intervention requested, False otherwise
        """
        for callback in self.human_intervention_callbacks:
            try:
                if callback():
                    logger.info("Human intervention requested")
                    return True
            except Exception as e:
                logger.error(f"Error in human intervention callback: {e}")
        
        return False
    
    def _update_resource_usage(self) -> None:
        """Update current resource usage metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            
            usage = {
                "timestamp": datetime.now().timestamp(),
                "memory_mb": memory.used / (1024 * 1024),
                "cpu_percent": cpu,
                "disk_gb": disk.used / (1024**3)
            }
            
            self.metrics.resource_usage_history.append(usage)
            
            # Keep only recent history (last 100 records)
            if len(self.metrics.resource_usage_history) > 100:
                self.metrics.resource_usage_history = self.metrics.resource_usage_history[-100:]
        
        except Exception as e:
            logger.error(f"Error updating resource usage: {e}")
    
    def add_human_intervention_callback(self, callback: Callable[[], bool]) -> None:
        """Add a callback for human intervention detection.
        
        Args:
            callback: Function that returns True if intervention is requested
        """
        self.human_intervention_callbacks.append(callback)
    
    def get_loop_status(self) -> Dict[str, Any]:
        """Get comprehensive loop status information.
        
        Returns:
            Dictionary with loop status and metrics
        """
        runtime = 0.0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "runtime_seconds": runtime,
            "metrics": {
                "iterations_completed": self.metrics.iterations_completed,
                "iterations_failed": self.metrics.iterations_failed,
                "success_rate": self.metrics.iterations_completed / max(1, self.current_iteration),
                "average_iteration_time": self.metrics.average_iteration_time,
                "best_score": self.metrics.best_score_achieved,
                "convergence_progress": self.convergence_tracker.get_progress()
            },
            "resource_usage": self.metrics.resource_usage_history[-1] if self.metrics.resource_usage_history else None,
            "convergence_status": self.convergence_tracker.get_status()
        }


class ConvergenceTracker:
    """Tracks convergence of the active learning process."""
    
    def __init__(self, config: ConvergenceConfig):
        """Initialize convergence tracker.
        
        Args:
            config: Convergence configuration
        """
        self.config = config
        self.scores_history: List[float] = []
        self.improvements: List[float] = []
        self.stagnation_count = 0
        self.is_converged = False
    
    def add_iteration_score(self, score: float) -> None:
        """Add a new iteration score.
        
        Args:
            score: Performance score for the iteration
        """
        self.scores_history.append(score)
        
        # Calculate improvement
        if len(self.scores_history) > 1:
            improvement = score - self.scores_history[-2]
            self.improvements.append(improvement)
            
            # Check for stagnation
            if improvement < self.config.min_improvement:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
    
    def check_convergence(self, completed_states: List[IterationState]) -> bool:
        """Check if convergence has been achieved.
        
        Args:
            completed_states: List of completed iteration states
            
        Returns:
            True if converged, False otherwise
        """
        if not self.config.enable_early_stopping:
            return False
        
        if len(completed_states) < 2:
            return False
        
        # Update scores from states
        recent_scores = [state.overall_score for state in completed_states[-5:]]
        
        # Check score threshold
        if recent_scores[-1] >= self.config.score_threshold:
            logger.info(f"Convergence achieved - Score threshold {self.config.score_threshold} reached")
            self.is_converged = True
            return True
        
        # Check stagnation
        if self.stagnation_count >= self.config.patience:
            logger.info(f"Convergence achieved - No improvement for {self.config.patience} iterations")
            self.is_converged = True
            return True
        
        # Check improvement tolerance
        if len(recent_scores) >= 3:
            recent_variance = max(recent_scores) - min(recent_scores)
            if recent_variance <= self.config.property_tolerance:
                logger.info(f"Convergence achieved - Variance {recent_variance:.4f} within tolerance")
                self.is_converged = True
                return True
        
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get convergence progress information.
        
        Returns:
            Dictionary with convergence progress metrics
        """
        return {
            "scores_history": self.scores_history[-10:],  # Last 10 scores
            "improvements": self.improvements[-5:],        # Last 5 improvements
            "stagnation_count": self.stagnation_count,
            "is_converged": self.is_converged,
            "current_score": self.scores_history[-1] if self.scores_history else 0.0,
            "best_score": max(self.scores_history) if self.scores_history else 0.0
        }
    
    def get_status(self) -> str:
        """Get current convergence status.
        
        Returns:
            String describing convergence status
        """
        if self.is_converged:
            return "converged"
        elif self.stagnation_count > 0:
            return f"stagnating ({self.stagnation_count}/{self.config.patience})"
        else:
            return "improving"


# Test functionality
def test_loop_controller():
    """Test the LoopController functionality."""
    print("Testing LoopController...")
    
    # Create test configuration
    conv_config = ConvergenceConfig(
        min_improvement=0.05,
        patience=2,
        score_threshold=0.9
    )
    
    resource_limits = ResourceLimits(
        max_memory_mb=8000,
        max_time_hours=1.0
    )
    
    quality_gates = QualityGates(
        min_confidence_score=0.5,
        max_error_rate=0.1
    )
    
    # Create loop controller
    controller = LoopController(
        max_iterations=5,
        convergence_config=conv_config,
        resource_limits=resource_limits,
        quality_gates=quality_gates
    )
    
    print(f"Initial status: {controller.get_loop_status()}")
    
    # Start the loop
    controller.start_loop()
    print(f"Started - Status: {controller.status.value}")
    
    # Simulate some iterations
    from .state_manager import StateManager
    state_manager = StateManager("test_loop_states")
    
    for i in range(1, 4):
        # Check if should continue
        should_continue, stop_reason = controller.should_continue(state_manager)
        print(f"Iteration {i} - Continue: {should_continue}, Reason: {stop_reason}")
        
        if not should_continue:
            controller.stop_loop(stop_reason)
            break
        
        # Begin iteration
        controller.begin_iteration(i)
        
        # Create mock iteration state
        state = state_manager.create_new_iteration(f"Test iteration {i}")
        state.overall_score = 0.6 + (i * 0.1)  # Gradually improving score
        state.update_status(IterationStatus.COMPLETED)
        
        # Complete iteration
        controller.complete_iteration(state)
        
        time.sleep(0.1)  # Small delay
    
    # Get final status
    final_status = controller.get_loop_status()
    print(f"Final status: {final_status}")
    
    # Test convergence
    tracker = ConvergenceTracker(conv_config)
    tracker.add_iteration_score(0.7)
    tracker.add_iteration_score(0.75)
    tracker.add_iteration_score(0.76)  # Small improvement
    
    print(f"Convergence progress: {tracker.get_progress()}")
    print(f"Convergence status: {tracker.get_status()}")
    
    print("LoopController test completed successfully!")


if __name__ == "__main__":
    test_loop_controller() 