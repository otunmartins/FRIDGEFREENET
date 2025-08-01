"""
Simple Literature Mining for Active Learning

This module provides a simplified literature mining approach that leverages
the existing working MaterialsLiteratureMiner instead of the complex automated system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Import the working literature mining system
from ..literature_mining_system import MaterialsLiteratureMiner
from .state_manager import IterationState, LiteratureResults

logger = logging.getLogger(__name__)


class SimpleLiteratureMining:
    """
    Simplified literature mining that uses the working MaterialsLiteratureMiner.
    
    This replaces the complex AutomatedLiteratureMining system with a simpler
    approach that actually works reliably.
    """
    
    def __init__(self, storage_path: str = "simple_literature_results"):
        """Initialize simple literature mining."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the working literature miner
        try:
            self.literature_miner = MaterialsLiteratureMiner()
            logger.info("✅ Simple literature mining initialized with working MaterialsLiteratureMiner")
        except Exception as e:
            logger.error(f"Failed to initialize MaterialsLiteratureMiner: {e}")
            self.literature_miner = None
    
    async def run_simple_mining(self, state: IterationState) -> LiteratureResults:
        """
        Run simple literature mining using the working system.
        
        Args:
            state: Current iteration state
            
        Returns:
            LiteratureResults: Literature mining results
        """
        logger.info(f"Starting simple literature mining for iteration {state.iteration_number}")
        
        if self.literature_miner is None:
            return self._create_fallback_results(state)
        
        try:
            # Create a user request based on target properties
            user_request = self._create_user_request(state)
            
            # Use the working intelligent mining method
            mining_results = self.literature_miner.intelligent_mining(
                user_request=user_request,
                max_papers=20,
                recent_only=True,
                progress_callback=self._progress_callback
            )
            
            # Convert to LiteratureResults format
            literature_results = self._convert_to_literature_results(mining_results, state)
            
            # Save results
            self._save_results(state.iteration_number, literature_results)
            
            return literature_results
            
        except Exception as e:
            logger.error(f"Error in simple literature mining: {e}")
            return self._create_fallback_results(state)
    
    def _create_user_request(self, state: IterationState) -> str:
        """Create a user request string from the iteration state."""
        target_props = state.target_properties or {}
        
        # Build request based on target properties
        request_parts = [
            "Find materials for insulin delivery patches with the following properties:"
        ]
        
        for prop, value in target_props.items():
            request_parts.append(f"- {prop}: {value}")
        
        if not target_props:
            request_parts.append("- Biocompatible polymer materials for transdermal drug delivery")
            request_parts.append("- Thermal stability for insulin preservation")
            request_parts.append("- Controlled release properties")
        
        return " ".join(request_parts)
    
    def _progress_callback(self, message: str, step_type: str = "info"):
        """Progress callback for the literature mining."""
        logger.info(f"Literature mining: {message}")
    
    def _convert_to_literature_results(self, mining_results: Dict, state: IterationState) -> LiteratureResults:
        """Convert MaterialsLiteratureMiner results to LiteratureResults format."""
        
        # Extract material candidates
        material_candidates = mining_results.get('material_candidates', [])
        
        # Extract key insights
        key_insights = []
        if mining_results.get('analysis_summary'):
            key_insights.append(mining_results['analysis_summary'])
        
        # Extract relevant papers (use first 10)
        relevant_papers = mining_results.get('papers_analyzed', [])[:10]
        
        # Create synthesis routes from materials found
        synthesis_routes = []
        for material in material_candidates[:5]:
            if material.get('material_composition'):
                synthesis_routes.append({
                    "material": material.get('material_name', 'Unknown'),
                    "synthesis": material.get('material_composition', 'Not specified')
                })
        
        # Create extracted properties
        extracted_properties = {}
        for i, material in enumerate(material_candidates[:5]):
            material_key = f"material_{i+1}"
            extracted_properties[material_key] = {
                "thermal_stability": material.get('thermal_stability_temp_range', 'Not specified'),
                "biocompatibility": material.get('biocompatibility_data', 'Not specified'),
                "delivery_properties": material.get('delivery_properties', 'Not specified')
            }
        
        return LiteratureResults(
            papers_found=len(relevant_papers),
            relevant_papers=relevant_papers,
            material_candidates=material_candidates,
            key_insights=key_insights,
            synthesis_routes=synthesis_routes,
            extracted_properties=extracted_properties,
            search_queries=[self._create_user_request(state)],
            execution_time=mining_results.get('execution_time', 0.0)
        )
    
    def _create_fallback_results(self, state: IterationState) -> LiteratureResults:
        """Create fallback results when mining fails."""
        return LiteratureResults(
            papers_found=0,
            relevant_papers=[],
            material_candidates=[],
            key_insights=["Literature mining system unavailable - using fallback"],
            synthesis_routes=[],
            extracted_properties={},
            search_queries=[self._create_user_request(state)],
            execution_time=0.0
        )
    
    def _save_results(self, iteration: int, results: LiteratureResults):
        """Save results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_simple_literature.json"
            
            # Convert to dict for JSON serialization
            results_dict = {
                "papers_found": results.papers_found,
                "material_candidates": results.material_candidates,
                "key_insights": results.key_insights,
                "synthesis_routes": results.synthesis_routes,
                "extracted_properties": results.extracted_properties,
                "search_queries": results.search_queries,
                "execution_time": results.execution_time
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
                
            logger.info(f"Simple literature results saved to {results_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save simple literature results: {e}") 