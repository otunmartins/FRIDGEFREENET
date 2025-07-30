# MILESTONE 4: Active Learning Framework & Feedback Integration
# This file contains the iterative functionality for the complete active learning system
# Implementation for Week 4 - when integrating MD simulation feedback

import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime

from .literature_mining_system import MaterialsLiteratureMiner


class IterativeLiteratureMiner(MaterialsLiteratureMiner):
    """
    Advanced iterative literature mining system with MD simulation feedback integration.
    Extends the basic MaterialsLiteratureMiner for Milestone 4.
    
    Features for Milestone 4:
    - Dynamic prompt evolution based on MD simulation results
    - Iterative refinement of search queries
    - Feedback integration from UMA-ASE MD simulations
    - Active learning cycle implementation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("🔄 Iterative Literature Mining System (Milestone 4) initialized!")
    
    def mine_with_feedback(self, 
                          iteration: int = 1,
                          top_candidates: List[str] = None,
                          stability_mechanisms: List[str] = None,
                          target_properties: Dict[str, Any] = None,
                          limitations: List[str] = None,
                          md_simulation_results: Dict = None,
                          num_candidates: int = 15) -> Dict:
        """
        Iterative literature mining with MD simulation feedback.
        
        This is the core function for Milestone 4 - integrates with:
        - UMA-ASE MD simulation results
        - Generative model outputs
        - Dynamic prompt evolution
        
        Args:
            iteration (int): Current iteration in active learning cycle
            top_candidates (List[str]): High-performing materials from previous iterations
            stability_mechanisms (List[str]): Successful mechanisms from MD simulations
            target_properties (Dict): Properties to optimize based on simulation results
            limitations (List[str]): Failed approaches to avoid
            md_simulation_results (Dict): Results from UMA-ASE MD simulations
            num_candidates (int): Number of new candidates to generate
        
        Returns:
            Dict: Comprehensive results for feeding back into active learning cycle
        """
        print(f"\n🔬 Iterative Literature Mining - Iteration {iteration}")
        
        # Process MD simulation feedback (Milestone 4 integration)
        if md_simulation_results:
            feedback = self._process_md_feedback(md_simulation_results)
            top_candidates = feedback.get('successful_materials', top_candidates)
            stability_mechanisms = feedback.get('working_mechanisms', stability_mechanisms)
            limitations = feedback.get('failed_approaches', limitations)
        
        # Generate dynamic search queries
        search_queries = self._generate_dynamic_queries(
            iteration, top_candidates, stability_mechanisms, 
            target_properties, limitations
        )
        
        # Search and extract with iterative prompting
        all_papers = []
        for query in search_queries:
            print(f"📚 Searching: {query}")
            papers = self.scholar.search_papers_by_topic(
                topic=query,
                max_results=20,
                recent_years_only=True
            )
            all_papers.extend(papers)
        
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Extract with dynamic prompting
        material_candidates = self._extract_with_dynamic_prompts(
            unique_papers, iteration, top_candidates, 
            stability_mechanisms, target_properties, limitations, num_candidates
        )
        
        # Compile iterative results
        results = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "search_queries": search_queries,
            "papers_analyzed": len(unique_papers),
            "material_candidates": material_candidates,
            "feedback_metadata": {
                "top_candidates": top_candidates,
                "stability_mechanisms": stability_mechanisms,
                "target_properties": target_properties,
                "limitations": limitations,
                "md_results_processed": md_simulation_results is not None
            }
        }
        
        self._save_iterative_results(results)
        return results
    
    def _process_md_feedback(self, md_results: Dict) -> Dict:
        """
        Process MD simulation results to extract feedback for next iteration.
        Integration point for Milestone 3 (UMA-ASE MD Pipeline).
        """
        # This will integrate with the MD simulation pipeline
        # For now, return structured feedback format
        return {
            "successful_materials": md_results.get("high_performers", []),
            "working_mechanisms": md_results.get("effective_mechanisms", []),
            "failed_approaches": md_results.get("problematic_features", []),
            "property_insights": md_results.get("property_analysis", {})
        }
    
    def _generate_dynamic_queries(self, iteration: int, top_candidates: List[str], 
                                 stability_mechanisms: List[str], target_properties: Dict,
                                 limitations: List[str]) -> List[str]:
        """
        Generate search queries that evolve based on iteration and feedback.
        Implements the dynamic prompt evolution strategy from the proposal.
        """
        base_queries = [
            "hydrogels insulin delivery transdermal patch",
            "polymer protein stabilization thermal",
            "biocompatible materials drug delivery skin",
            "nanomaterials insulin encapsulation controlled release"
        ]
        
        if iteration == 1:
            # Initial broad exploration
            return base_queries + [
                "protein stabilization polymers temperature",
                "peptide delivery hydrogels biocompatible",
                "insulin stability materials room temperature",
                "transdermal drug delivery patches"
            ]
        
        elif iteration <= 3:
            # Incorporate initial insights
            refined_queries = base_queries.copy()
            
            if top_candidates:
                for material in top_candidates[:3]:
                    refined_queries.append(f"{material} insulin stabilization")
                    refined_queries.append(f"{material} protein drug delivery")
            
            if stability_mechanisms:
                for mechanism in stability_mechanisms[:2]:
                    refined_queries.append(f"protein stabilization {mechanism}")
            
            return refined_queries
        
        else:
            # Advanced targeted exploration
            targeted_queries = []
            
            if top_candidates and stability_mechanisms:
                for material in top_candidates[:2]:
                    for mechanism in stability_mechanisms[:2]:
                        targeted_queries.append(f"{material} {mechanism} insulin")
            
            if target_properties:
                for prop, value in target_properties.items():
                    targeted_queries.append(f"materials {prop} insulin delivery")
            
            if limitations:
                avoid_terms = " ".join([f"-{limit}" for limit in limitations[:2]])
                targeted_queries.append(f"insulin delivery materials {avoid_terms}")
            
            return targeted_queries + base_queries[:2]
    
    def _extract_with_dynamic_prompts(self, papers: List[Dict], iteration: int,
                                     top_candidates: List[str], stability_mechanisms: List[str],
                                     target_properties: Dict, limitations: List[str],
                                     num_candidates: int) -> List[Dict]:
        """
        Extract material data using prompts that evolve based on feedback.
        """
        extraction_prompt = self._build_dynamic_prompt(
            iteration, top_candidates, stability_mechanisms, 
            target_properties, limitations, num_candidates
        )
        
        papers_context = self._prepare_papers_context(papers[:10])
        full_prompt = f"{extraction_prompt}\n\nPAPERS TO ANALYZE:\n{papers_context}"
        
        try:
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={'temperature': 0.3, 'num_predict': 4000}
            )
            
            return self._parse_llm_response(response['message']['content'])
            
        except Exception as e:
            print(f"Error in iterative extraction: {e}")
            return []
    
    def _build_dynamic_prompt(self, iteration: int, top_candidates: List[str],
                             stability_mechanisms: List[str], target_properties: Dict,
                             limitations: List[str], num_candidates: int) -> str:
        """
        Build prompts that evolve based on iteration and feedback.
        Implements the prompt template architecture from the proposal.
        """
        base_prompt = f"""# Iterative Materials Extraction - Iteration {iteration}

EXTRACTION TASK: Identify {num_candidates} materials with potential for fridge-free insulin delivery patches.

INPUT CONTEXT:
- Iteration Number: {iteration}"""

        if top_candidates:
            base_prompt += f"""
- Previous High-performing Materials: {', '.join(top_candidates)}"""
        
        if stability_mechanisms:
            base_prompt += f"""
- Observed Stability Mechanisms: {', '.join(stability_mechanisms)}"""
        
        if target_properties:
            base_prompt += f"""
- Target Properties: {', '.join([f'{k}: {v}' for k, v in target_properties.items()])}"""
        
        if limitations:
            base_prompt += f"""
- Current Performance Limitations: {', '.join(limitations)}"""

        # Build requirements based on iteration
        if iteration == 1:
            requirements = """
MATERIAL REQUIREMENTS:
1. Demonstrated protein or peptide stabilization capability
2. Thermal stability at temperatures 25-40°C
3. Biocompatible for transdermal application
4. Controlled release properties for drug delivery"""
        
        elif iteration <= 3:
            requirements = """
MATERIAL REQUIREMENTS:
1. Similar stabilization mechanisms to successful candidates
2. Enhanced thermal stability compared to previous materials
3. Improved biocompatibility profiles
4. Optimized release kinetics for insulin delivery"""
            
            if top_candidates:
                requirements += f"""
5. Build upon successful features from: {', '.join(top_candidates[:2])}"""
        
        else:
            requirements = """
MATERIAL REQUIREMENTS:
1. Incorporate proven beneficial structural motifs
2. Address identified performance limitations
3. Target specific property improvements
4. Avoid problematic features from previous iterations"""
            
            if limitations:
                requirements += f"""
5. Specifically avoid: {', '.join(limitations[:2])}"""

        output_format = """
OUTPUT FORMAT: For each material, provide JSON-formatted data:
{
  "material_name": "Name of the material",
  "material_composition": "Chemical composition and formula",
  "chemical_structure": "Structural information (backbone, side chains, crosslinking)",
  "thermal_stability_temp_range": "Temperature stability range",
  "insulin_stability_duration": "Reported insulin stability duration",
  "biocompatibility_data": "Biocompatibility information",
  "release_kinetics": "Release kinetics and delivery mechanism",
  "delivery_efficiency": "Delivery efficiency data",
  "stabilization_mechanism": "Mechanism of protein stabilization",
  "literature_references": ["Reference 1", "Reference 2"],
  "confidence_score": "Score 1-10 based on evidence quality",
  "iteration_relevance": "How this material addresses current iteration goals"
}

Extract information ONLY if supported by the provided papers."""

        return f"{base_prompt}\n\n{requirements}\n\n{output_format}"
    
    def _save_iterative_results(self, results: Dict):
        """Save iterative mining results."""
        os.makedirs("iterative_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iterative_results/iteration_{results['iteration']}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Iterative results saved to: {filename}")
    
    def run_active_learning_cycle(self, max_iterations: int = 5, 
                                 md_simulator=None, 
                                 generative_model=None) -> List[Dict]:
        """
        Run complete active learning cycle for Milestone 4.
        
        This orchestrates the full pipeline:
        1. Literature mining
        2. Generative model candidate generation
        3. MD simulation evaluation
        4. Feedback integration
        5. Next iteration
        
        Args:
            max_iterations (int): Maximum number of learning cycles
            md_simulator: MD simulation component (from Milestone 3)
            generative_model: Generative model component (from Milestone 2)
        
        Returns:
            List[Dict]: Results from all iterations
        """
        all_results = []
        feedback_state = {
            "top_candidates": [],
            "stability_mechanisms": [],
            "target_properties": {},
            "limitations": []
        }
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ACTIVE LEARNING CYCLE - ITERATION {iteration}")
            print(f"{'='*60}")
            
            # Step 1: Literature mining with current feedback
            mining_results = self.mine_with_feedback(
                iteration=iteration,
                top_candidates=feedback_state["top_candidates"],
                stability_mechanisms=feedback_state["stability_mechanisms"],
                target_properties=feedback_state["target_properties"],
                limitations=feedback_state["limitations"]
            )
            
            # Step 2: Generate candidates with generative model (Milestone 2)
            if generative_model:
                generated_candidates = generative_model.generate_candidates(
                    base_materials=mining_results["material_candidates"]
                )
                mining_results["generated_candidates"] = generated_candidates
            
            # Step 3: Evaluate with MD simulations (Milestone 3)
            if md_simulator:
                md_results = md_simulator.evaluate_candidates(
                    mining_results["material_candidates"]
                )
                mining_results["md_evaluation"] = md_results
                
                # Step 4: Update feedback state
                feedback_state = self._update_feedback_state(md_results, feedback_state)
            
            all_results.append(mining_results)
            
            print(f"Iteration {iteration} complete. Found {len(mining_results['material_candidates'])} candidates.")
        
        self._save_complete_cycle_results(all_results)
        return all_results
    
    def _update_feedback_state(self, md_results: Dict, current_state: Dict) -> Dict:
        """Update feedback state based on MD simulation results."""
        # This will be implemented when integrating with MD simulations
        # For now, return updated state structure
        return {
            "top_candidates": md_results.get("successful_materials", []),
            "stability_mechanisms": md_results.get("effective_mechanisms", []),
            "target_properties": md_results.get("target_improvements", {}),
            "limitations": md_results.get("failed_features", [])
        }
    
    def _save_complete_cycle_results(self, all_results: List[Dict]):
        """Save complete active learning cycle results."""
        os.makedirs("cycle_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cycle_results/complete_cycle_{timestamp}.json"
        
        cycle_summary = {
            "total_iterations": len(all_results),
            "timestamp": datetime.now().isoformat(),
            "iterations": all_results,
            "performance_progression": self._analyze_performance_progression(all_results)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cycle_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Complete cycle results saved to: {filename}")
    
    def _analyze_performance_progression(self, all_results: List[Dict]) -> Dict:
        """Analyze how performance improved across iterations."""
        progression = {
            "candidate_count_per_iteration": [len(r["material_candidates"]) for r in all_results],
            "query_evolution": [r["search_queries"] for r in all_results],
            "feedback_evolution": [r.get("feedback_metadata", {}) for r in all_results]
        }
        return progression


# Demo for Milestone 4 testing
if __name__ == "__main__":
    print("🚀 MILESTONE 4: Iterative Literature Mining Demo")
    print("This will be used when integrating with MD simulations and generative models")
    
    # Initialize iterative system
    iterative_miner = IterativeLiteratureMiner()
    
    # Example: Simulated active learning cycle
    print("\n📊 Simulated 3-iteration cycle:")
    
    # Mock MD simulation results for demonstration
    mock_md_results = {
        "high_performers": ["PEG-based hydrogels", "chitosan derivatives"],
        "effective_mechanisms": ["hydrogen bonding", "hydrophobic interactions"],
        "problematic_features": ["high crystallinity", "poor water retention"]
    }
    
    results = iterative_miner.mine_with_feedback(
        iteration=3,
        md_simulation_results=mock_md_results,
        num_candidates=8
    )
    
    print(f"Found {len(results['material_candidates'])} candidates with feedback integration") 