"""
Configuration to Use Simple Literature Mining

This file shows how to replace the complex AutomatedLiteratureMining
with the simple, working SimpleLiteratureMining system.

To use this approach, update your orchestrator initialization.
"""

from .simple_literature_mining import SimpleLiteratureMining
from .automated_piecewise_generation import AutomatedPiecewiseGeneration
from .automated_md_simulation import AutomatedMDSimulation
from .automated_post_processing import AutomatedPostProcessing
from .decision_engine import LLMDecisionEngine


def create_simple_active_learning_orchestrator():
    """
    Create an orchestrator with the simple, working literature mining system.
    
    This replaces the complex AutomatedLiteratureMining with SimpleLiteratureMining
    that uses your working MaterialsLiteratureMiner.
    """
    
    # Use simple literature mining instead of complex automated version
    literature_mining = SimpleLiteratureMining()
    
    # Keep the other components (they can be improved later)
    psmiles_generation = AutomatedPiecewiseGeneration()
    md_simulation = AutomatedMDSimulation()
    post_processing = AutomatedPostProcessing()
    decision_engine = LLMDecisionEngine()
    
    return {
        "literature_mining": literature_mining,
        "psmiles_generation": psmiles_generation,
        "md_simulation": md_simulation,
        "post_processing": post_processing,
        "decision_engine": decision_engine
    }


def update_orchestrator_to_use_simple_literature_mining(orchestrator):
    """
    Update an existing orchestrator to use simple literature mining.
    
    Args:
        orchestrator: Your existing active learning orchestrator
    """
    
    # Replace the complex literature mining with simple version
    simple_literature_mining = SimpleLiteratureMining()
    
    # Update the orchestrator's literature mining component
    if hasattr(orchestrator, 'literature_mining'):
        orchestrator.literature_mining = simple_literature_mining
        print("✅ Updated orchestrator to use simple literature mining")
    else:
        print("⚠️ Orchestrator doesn't have literature_mining attribute")
    
    return orchestrator


# Example usage in your orchestrator initialization:
"""
# Instead of:
# from .automated_literature_mining import AutomatedLiteratureMining
# literature_mining = AutomatedLiteratureMining()

# Use:
from .simple_literature_mining import SimpleLiteratureMining
literature_mining = SimpleLiteratureMining()

# Or update existing orchestrator:
orchestrator = update_orchestrator_to_use_simple_literature_mining(your_orchestrator)
""" 