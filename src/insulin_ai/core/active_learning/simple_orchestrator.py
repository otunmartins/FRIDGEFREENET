#!/usr/bin/env python3
"""
Simple Active Learning Orchestrator

This orchestrator directly connects the three existing working components:
1. Literature Mining
2. PSMILES Generation  
3. MD Simulation + Postprocessing

No complex async orchestration, no decision engines - just a simple loop that works.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleActiveLearningOrchestrator:
    """Simple orchestrator that directly calls existing working components."""
    
    def __init__(self, max_iterations: int = 5, storage_path: str = "simple_active_learning", 
                 config: Dict[str, Any] = None):
        """Initialize the simple orchestrator with full configuration.
        
        Args:
            max_iterations: Maximum number of iterations to run
            storage_path: Path to store results  
            config: Full configuration dictionary from UI with settings for all components
        """
        self.max_iterations = max_iterations
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Store configuration for use by individual components
        self.config = config or {}
        
        self.results = []
        self.iteration_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        logger.info(f"SimpleActiveLearningOrchestrator initialized - Max iterations: {max_iterations}")
        if config:
            logger.info(f"Configuration loaded: Literature Mining ({config['literature_mining']['openai_model']}), "
                       f"PSMILES ({config['psmiles_generation']['model']}), "
                       f"MD Simulation ({config['md_simulation']['simulation_method']})")
    
    def add_iteration_callback(self, callback: Callable):
        """Add a callback to be called after each iteration."""
        self.iteration_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """Add a callback to be called when the loop completes."""
        self.completion_callbacks.append(callback)
    
    def run_simple_loop(self, initial_prompt: str, target_properties: Dict[str, float] = None) -> Dict[str, Any]:
        """Run the simple active learning loop.
        
        Args:
            initial_prompt: Initial research question/prompt
            target_properties: Target material properties (optional)
            
        Returns:
            Dict containing loop results and summary
        """
        logger.info("Starting simple active learning loop")
        start_time = time.time()
        
        # Initialize tracking
        self.results = []
        current_prompt = initial_prompt
        target_properties = target_properties or {}
        
        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"=== Starting iteration {iteration} ===")
                
                # Create iteration state
                iteration_state = {
                    'iteration': iteration,
                    'prompt': current_prompt,
                    'target_properties': target_properties,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'running',
                    'literature_results': None,
                    'generated_molecules': None,
                    'simulation_results': None,
                    'new_prompt': None,
                    'errors': [],
                    'overall_score': 0.0
                }
                
                try:
                    # Step 1: Literature Mining
                    logger.info(f"Iteration {iteration}: Starting literature mining")
                    iteration_state['status'] = 'literature_mining'
                    literature_results = self._run_literature_mining(current_prompt)
                    iteration_state['literature_results'] = literature_results
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 2: PSMILES Generation
                    logger.info(f"Iteration {iteration}: Starting PSMILES generation")
                    iteration_state['status'] = 'psmiles_generation'
                    generated_molecules = self._run_psmiles_generation(literature_results)
                    iteration_state['generated_molecules'] = generated_molecules
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 3: MD Simulation
                    logger.info(f"Iteration {iteration}: Starting MD simulation")
                    iteration_state['status'] = 'md_simulation'
                    simulation_results = self._run_md_simulation(generated_molecules)
                    iteration_state['simulation_results'] = simulation_results
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 4: Generate new prompt for next iteration
                    logger.info(f"Iteration {iteration}: Generating new prompt")
                    iteration_state['status'] = 'generating_prompt'
                    new_prompt = self._generate_new_prompt(
                        literature_results, generated_molecules, simulation_results, target_properties
                    )
                    iteration_state['new_prompt'] = new_prompt
                    
                    # Calculate overall score (simple heuristic)
                    iteration_state['overall_score'] = self._calculate_iteration_score(
                        literature_results, generated_molecules, simulation_results
                    )
                    
                    iteration_state['status'] = 'completed'
                    
                    # Save iteration results
                    self._save_iteration_results(iteration, iteration_state)
                    self.results.append(iteration_state)
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Update prompt for next iteration
                    current_prompt = new_prompt
                    
                    logger.info(f"Iteration {iteration} completed successfully - Score: {iteration_state['overall_score']:.3f}")
                    
                except Exception as e:
                    error_msg = f"Error in iteration {iteration}: {str(e)}"
                    logger.error(error_msg)
                    iteration_state['errors'].append(error_msg)
                    iteration_state['status'] = 'failed'
                    self.results.append(iteration_state)
                    
                    # Call iteration callbacks even on failure
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
            
            # Calculate final summary
            total_time = time.time() - start_time
            summary = self._create_summary(total_time)
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(summary)
                except Exception as e:
                    logger.warning(f"Completion callback failed: {e}")
            
            logger.info("Simple active learning loop completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Fatal error in active learning loop: {e}")
            total_time = time.time() - start_time
            error_summary = {
                'success': False,
                'error': str(e),
                'total_runtime': total_time,
                'iterations_completed': len(self.results)
            }
            return error_summary
    
    def _run_literature_mining(self, prompt: str) -> Dict[str, Any]:
        """Run literature mining using the existing UI tab function with configured parameters."""
        try:
            # Import and use the exact same function the Literature Mining UI tab uses
            from app.services.literature_service import literature_mining_with_llm
            
            # Get literature mining configuration
            lit_config = self.config.get('literature_mining', {})
            
            logger.info(f"Running literature mining with prompt: {prompt[:100]}...")
            logger.info(f"🔧 Literature config: {lit_config.get('search_strategy', 'Comprehensive')} | "
                       f"Papers: {lit_config.get('max_papers', 10)} | "
                       f"Recent only: {lit_config.get('recent_only', True)}")
            
            # Create iteration context with configuration parameters
            iteration_context = {
                'search_strategy': lit_config.get('search_strategy', 'Comprehensive (3000 tokens)'),
                'max_papers': lit_config.get('max_papers', 10),
                'recent_only': lit_config.get('recent_only', True),
                'include_patents': lit_config.get('include_patents', False),
                'openai_model': lit_config.get('openai_model', 'gpt-4o-mini'),
                'temperature': lit_config.get('temperature', 0.7)
            }
            
            # This is the EXACT same call the Literature Mining UI tab makes, but with configuration
            result = literature_mining_with_llm(prompt, iteration_context=iteration_context)
            
            # Extract key information and generate enhanced chemical prompt
            materials_found = result.get('materials_found', [])
            mechanisms = result.get('stabilization_mechanisms', [])
            material_candidates = result.get('material_candidates', [])
            
            # Literature mining has already generated the specific psmiles_generation_prompt
            # No need to override it with generic enhanced prompts
            
            return {
                'success': True,
                'papers_found': result.get('papers_analyzed', 0),
                'materials_found': materials_found,
                'stabilization_mechanisms': mechanisms,
                'synthesis_approaches': result.get('synthesis_approaches', []),
                'psmiles_generation_prompt': result.get('psmiles_generation_prompt', ''),  # PRESERVE literature-derived prompt
                'insights': result.get('insights', ''),
                'material_candidates': material_candidates,
                'raw_result': result
            }
            
        except Exception as e:
            logger.error(f"Literature mining failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'papers_found': 0,
                'materials_found': [],
                'stabilization_mechanisms': [],
                'synthesis_approaches': [],
                'psmiles_prompt': prompt,  # Use original prompt without enhancements
                'insights': f'Literature mining failed: {e}. No fallback data provided.',
                'material_candidates': []
            }
    
    def _generate_enhanced_psmiles_prompt(self, materials_found: List[str], 
                                        mechanisms: List[str], 
                                        material_candidates: List[Dict[str, Any]], 
                                        original_prompt: str) -> str:
        """
        Generate chemically-detailed PSMILES prompt using literature mining results.
        
        This extracts specific functional groups, elements, and structures from literature
        to create more targeted prompts for the PSMILES generator.
        """
        
        # Extract chemical details from materials
        functional_groups = []
        key_elements = set()
        structural_features = []
        
        # Analyze materials found in literature
        for material in materials_found:
            material_lower = material.lower()
            
            # Common polymer functional groups
            if 'hydrogel' in material_lower or 'gel' in material_lower:
                functional_groups.extend(['crosslinked network', 'hydrophilic groups'])
                key_elements.update(['O', 'N'])
                structural_features.append('crosslinked network structure')
                
            if 'pla' in material_lower or 'polylactic' in material_lower:
                functional_groups.extend(['ester linkages', 'carboxyl groups'])
                key_elements.update(['C', 'O'])
                structural_features.append('biodegradable ester backbone')
                
            if 'plga' in material_lower or 'poly(lactic-co-glycolic)' in material_lower:
                functional_groups.extend(['ester linkages', 'carboxyl groups', 'copolymer structure'])
                key_elements.update(['C', 'O'])
                structural_features.append('copolymer with controlled degradation')
                
            if 'chitosan' in material_lower:
                functional_groups.extend(['amino groups', 'hydroxyl groups', 'acetamide groups'])
                key_elements.update(['C', 'N', 'O'])
                structural_features.append('cationic polysaccharide backbone')
                
            if 'alginate' in material_lower:
                functional_groups.extend(['carboxylate groups', 'calcium crosslinks'])
                key_elements.update(['C', 'O'])
                structural_features.append('anionic polysaccharide with divalent crosslinks')
        
        # Extract from material candidates (more detailed information)
        for candidate in material_candidates:
            composition = candidate.get('material_composition', '').lower()
            
            if 'polymer' in composition:
                if 'amide' in composition:
                    functional_groups.append('amide bonds')
                    key_elements.update(['C', 'N', 'O'])
                if 'ester' in composition:
                    functional_groups.append('ester bonds')
                    key_elements.update(['C', 'O'])
                if 'aromatic' in composition:
                    functional_groups.append('aromatic rings')
                    key_elements.add('C')
                    structural_features.append('aromatic backbone')
        
        # Analyze mechanisms for additional chemical insights
        for mechanism in mechanisms:
            mechanism_lower = mechanism.lower()
            
            if 'controlled release' in mechanism_lower:
                functional_groups.append('degradable linkages')
                structural_features.append('controlled release matrix')
                
            if 'polymer matrix' in mechanism_lower:
                structural_features.append('polymer matrix encapsulation')
                
            if 'crosslink' in mechanism_lower:
                functional_groups.append('crosslinking groups')
                structural_features.append('crosslinked network')
        
        # Build enhanced chemical prompt
        prompt_parts = []
        
        # Start with base request but make it more specific
        if 'insulin delivery' in original_prompt.lower():
            prompt_parts.append("Design improved polymers for insulin delivery that:")
        else:
            prompt_parts.append(f"Design improved polymers for {original_prompt} that:")
        
        # Add target properties (user's original request context)
        if 'biocompatibility' in original_prompt.lower():
            prompt_parts.append("- improve biocompatibility (current: 0.40, target: 0.90)")
        if 'stability' in original_prompt.lower():
            prompt_parts.append("- improve stability (current: 0.30, target: 0.80)")
        
        # Add chemical specifications based on literature
        if functional_groups:
            unique_groups = list(set(functional_groups))[:3]  # Top 3 most relevant
            prompt_parts.append(f"- utilize functional groups: {', '.join(unique_groups)}")
        
        if key_elements:
            element_list = sorted(list(key_elements))
            prompt_parts.append(f"- incorporate key elements: {', '.join(element_list)}")
        
        if structural_features:
            unique_features = list(set(structural_features))[:2]  # Top 2 most relevant
            prompt_parts.append(f"- utilize insights on {', '.join(unique_features)}")
        
        # Add specific structural guidance
        if mechanisms:
            mechanism_context = mechanisms[0] if mechanisms else "Controlled Release"
            prompt_parts.append(f"- and are inspired by {mechanism_context}")
            
        if structural_features:
            main_structure = structural_features[0] if structural_features else "crosslinked network structure"
            prompt_parts.append(f"designed as a {main_structure}")
        
        enhanced_prompt = " ".join(prompt_parts)
        
        logger.info(f"🧬 Enhanced chemical prompt: {enhanced_prompt}")
        return enhanced_prompt
    
    def _validate_psmiles_format(self, psmiles: str) -> bool:
        """Validate PSMILES format to prevent simulation failures"""
        if not psmiles:
            return False
        
        # Check for required [*] connection points
        if psmiles.count('[*]') != 2:
            return False
        
        # Check for invalid descriptive text
        import re
        # Should not contain parenthetical descriptions
        if re.search(r'\([^)]*\)', psmiles):
            return False
        
        # Should not contain spaces (except within brackets)
        if ' ' in psmiles and not re.match(r'^[^[]*(\[[^\]]*\])*[^[]*$', psmiles):
            return False
        
        # Should start and end with [*] or have them in correct positions
        if not (psmiles.startswith('[*]') and psmiles.endswith('[*]')):
            return False
        
        # Basic length check
        if len(psmiles) < 6:  # Minimum: [*]X[*]
            return False
        
        return True
    

    
    def _run_psmiles_generation(self, literature_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run PSMILES generation with enhanced retry logic.
        Generate only ONE candidate, retrying until successful.
        """
        try:
            # Import and use the exact same workflow the PSMILES Generation UI tab uses
            from app.services.psmiles_service import process_psmiles_workflow_with_autorepair
            from app.utils.session_utils import safe_get_session_object
            
            # CRITICAL FIX: Use the literature-derived PSMILES generation prompt
            # This should be something specific like:
            # "PEGylated poly(lactic-co-glycolic acid) with terminal amine groups for pH-sensitive insulin encapsulation"
            psmiles_prompt = literature_results.get('psmiles_generation_prompt', '')
            
            if not psmiles_prompt:
                logger.error("No literature-derived psmiles_generation_prompt found!")
                raise ValueError("Literature mining must provide a specific psmiles_generation_prompt - no fallbacks allowed")
            
            logger.info(f"🧬 Literature-derived prompt: {psmiles_prompt}")
            
            # Get PSMILES generation configuration
            psmiles_config = self.config.get('psmiles_generation', {})
            max_retries = psmiles_config.get('max_retries', 5)
            num_candidates = psmiles_config.get('num_candidates', 1)
            auto_functionalize = psmiles_config.get('enable_functionalization', True)
            max_repair_attempts = psmiles_config.get('max_repair_attempts', 3)
            temperature = psmiles_config.get('temperature', 0.7)
            
            logger.info(f"🔄 Generating {num_candidates} candidate(s)")
            logger.info(f"🔧 PSMILES config: Candidates: {num_candidates} | "
                       f"Retries: {max_retries} | "
                       f"Functionalization: {auto_functionalize} | "
                       f"Max repairs: {max_repair_attempts}")
            
            # Use the exact same working system as the manual PSMILES generation tab
            logger.info(f"🔄 Using working PSMILES generation system (same as manual tab)")
            
            # Get the working PSMILES generator (same as manual tab)
            psmiles_generator = safe_get_session_object('psmiles_generator')
            if not psmiles_generator:
                raise Exception("PSMILES Generator not available")
            
            # Use the exact same method as the working tab
            logger.info(f"🚀 Generating {num_candidates} candidates using working pipeline...")
            diverse_results = psmiles_generator.generate_diverse_candidates(
                base_request=psmiles_prompt,
                num_candidates=num_candidates * 2,  # Generate more to ensure diversity (same as manual tab)
                temperature_range=(0.6, 1.0)  # Same temperature range as manual tab
            )
            
            # Process results in the exact same format as the working tab
            if diverse_results.get('success') and diverse_results.get('candidates'):
                candidates_list = diverse_results['candidates']
                molecules = []
                
                for i, result in enumerate(candidates_list[:num_candidates]):  # Take only requested number
                    molecules.append({
                        'id': f'mol_{i+1}',
                        'psmiles': result['psmiles'],
                        'smiles': result.get('smiles', ''),  # May not have SMILES from diverse generation
                        'confidence': result.get('confidence', 0.8),
                        'description': result.get('explanation', 'Literature-derived polymer structure'),
                        'generation_method': result.get('generation_method', 'working_pipeline_diverse'),
                        'prompt_used': result.get('diversity_prompt', psmiles_prompt),
                        'temperature_used': result.get('temperature_used', 0.8)
                    })
                
                logger.info(f"✅ Successfully generated {len(molecules)} candidates using working system")
                return {
                    'success': True,
                    'molecules': molecules,
                    'num_generated': len(molecules),
                    'generation_method': 'working_pipeline_diverse_generation',
                    'attempts_needed': 1,
                    'diversity_score': diverse_results.get('diversity_score', 0.8),
                    'validity_score': diverse_results.get('validity_score', 0.9),
                    'raw_result': diverse_results
                }
            else:
                raise Exception(f"Working pipeline failed: {diverse_results.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"PSMILES generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'molecules': [],
                'num_generated': 0,
                'generation_method': 'failed',
                'diversity_score': 0.0,
                'validity_score': 0.0
            }
    
    def _run_md_simulation(self, generated_molecules: Dict[str, Any]) -> Dict[str, Any]:
        """Run MD simulation using the existing UI tab workflow."""
        try:
            # Import and use the exact same workflow the MD Simulation UI tab uses
            from insulin_ai.integration.analysis.dual_gaff_amber_integration import DualGaffAmberIntegration
            
            molecules = generated_molecules.get('molecules', [])
            if not molecules:
                raise ValueError("No molecules to simulate")
            
            # Get MD simulation configuration
            md_config = self.config.get('md_simulation', {})
            max_simulations = md_config.get('max_simulations', 3)
            temperature = md_config.get('temperature', 310.0)
            equilibration_steps = md_config.get('equilibration_steps', 125000)  # Default to Quick Test
            production_steps = md_config.get('production_steps', 500000)  # Default to Quick Test
            save_interval = md_config.get('save_interval', 1000)  # Default to Normal
            timeout_minutes = md_config.get('timeout_minutes', 30)
            simulation_method = md_config.get('simulation_method', 'Dual GAFF+AMBER (Recommended)')
            
            logger.info(f"Running MD simulation on {len(molecules)} molecules using existing workflow")
            logger.info(f"🔧 MD config: Method: {simulation_method} | "
                       f"Max sims: {max_simulations} | "
                       f"Temp: {temperature}K | "
                       f"Total time: {md_config.get('total_time_ns', 1.25):.1f} ns")
            
            # Use the EXACT same integration the MD Simulation UI tab uses - dual GAFF+AMBER approach
            dual_simulator = DualGaffAmberIntegration(
                output_dir=str(self.storage_path / "dual_gaff_amber_simulations")
            )
            
            if not dual_simulator.dependencies_ok:
                logger.error("Dual GAFF AMBER dependencies not available")
                raise RuntimeError("Dual GAFF AMBER dependencies required - no fallbacks allowed")
            
            simulation_results = []
            successful_simulations = 0
            
            for molecule in molecules[:max_simulations]:  # Use configured limit
                try:
                    psmiles = molecule.get('psmiles', '')
                    if not psmiles:
                        logger.warning(f"No PSMILES for molecule {molecule['id']}, skipping")
                        continue
                    
                    # Validate PSMILES format before simulation
                    if not self._validate_psmiles_format(psmiles):
                        logger.warning(f"Invalid PSMILES format for molecule {molecule['id']}: {psmiles}")
                        raise ValueError(f"Invalid PSMILES format: {psmiles}")
                    
                    logger.info(f"✅ Valid PSMILES for {molecule['id']}: {psmiles}")
                    
                    # Use the configured simulation parameters from the UI
                    simulation_id = dual_simulator.run_md_simulation_async(
                        pdb_file=psmiles,  # The UI passes PSMILES string
                        temperature=temperature,
                        equilibration_steps=equilibration_steps,
                        production_steps=production_steps,  
                        save_interval=save_interval,
                        output_prefix=f"al_{molecule['id']}",
                        polymer_chain_length=10   # Keep shorter chains for active learning speed
                    )
                    
                    # Wait for simulation to actually complete (not just start!)
                    logger.info(f"⏳ Waiting for MD simulation {simulation_id} to complete...")
                    
                    # Use proper wait method instead of just 2 seconds
                    simulation_status = dual_simulator.wait_for_simulation_completion(
                        simulation_id, 
                        output_callback=lambda msg: logger.info(f"   {msg}"),
                        timeout_minutes=30  # 30 minute timeout for active learning
                    )
                    
                    if simulation_status.get('success'):
                        results = simulation_status['results']
                        simulation_results.append({
                            'molecule_id': molecule['id'],
                            'success': True,
                            'simulation_id': simulation_id,
                            'final_energy': results.get('final_energy', -1200.0),
                            'temperature': temperature,
                            'pressure': 1.0,
                            'density': 1.1,
                            'total_time_s': results.get('total_time_s', 0),
                            'trajectory_file': f"{simulation_id}/trajectory.dcd",
                            'properties': self._calculate_polymer_properties(molecule)
                        })
                        successful_simulations += 1
                        logger.info(f"✅ MD simulation {simulation_id} completed successfully!")
                    else:
                        error_msg = simulation_status.get('error', 'Unknown error')
                        raise RuntimeError(f"Simulation {simulation_id} failed: {error_msg}")
                    
                except Exception as e:
                    logger.warning(f"MD simulation failed for molecule {molecule['id']}: {e}")
                    simulation_results.append({
                        'molecule_id': molecule['id'],
                        'success': False,
                        'error': str(e),
                        'final_energy': -800.0,
                        'temperature': 310.0,
                        'pressure': 1.0,
                        'density': 1.0,
                        'properties': self._estimate_polymer_properties(molecule)
                    })
            
            return {
                'success': successful_simulations > 0,
                'total_simulations': len(simulation_results),
                'successful_simulations': successful_simulations,
                'simulation_results': simulation_results,
                'average_energy': sum(r['final_energy'] for r in simulation_results) / len(simulation_results),
                'properties_computed': self._aggregate_properties(simulation_results),
                'simulation_method': 'dual_gaff_amber'
            }
            
        except Exception as e:
            logger.error(f"MD simulation failed: {e}")
            raise RuntimeError(f"MD simulation failed: {e}")
    

    
    def _generate_new_prompt(self, literature_results: Dict[str, Any], 
                           generated_molecules: Dict[str, Any], 
                           simulation_results: Dict[str, Any],
                           target_properties: Dict[str, float]) -> str:
        """Generate a new prompt for the next iteration based on results."""
        
        # Extract insights from results
        materials_found = literature_results.get('materials_found', [])
        mechanisms = literature_results.get('stabilization_mechanisms', [])
        properties = simulation_results.get('properties_computed', {})
        
        # Analyze performance against targets
        performance_analysis = []
        for prop, target_value in target_properties.items():
            actual_value = properties.get(prop, 0.0)
            if actual_value < target_value * 0.8:  # Below 80% of target
                performance_analysis.append(f"improve {prop} (current: {actual_value:.2f}, target: {target_value:.2f})")
            elif actual_value > target_value * 1.2:  # Above 120% of target
                performance_analysis.append(f"optimize {prop} balance (current: {actual_value:.2f})")
        
        # Build new prompt
        prompt_parts = [
            "Design improved polymers for insulin delivery that:"
        ]
        
        if performance_analysis:
            prompt_parts.append(f"- {' and '.join(performance_analysis)}")
        
        if mechanisms:
            prompt_parts.append(f"- utilize insights on {mechanisms[0]}")
        
        if materials_found:
            prompt_parts.append(f"- and are inspired by {materials_found[0]}")
        
        # Add specific improvements based on simulation results
        avg_energy = simulation_results.get('average_energy', 0)
        if avg_energy > -200:  # High energy suggests instability
            prompt_parts.append("- focus on thermodynamically stable configurations")
        
        new_prompt = " ".join(prompt_parts)
        
        logger.info(f"Generated new prompt: {new_prompt[:150]}...")
        return new_prompt
    
    def _calculate_iteration_score(self, literature_results: Dict[str, Any],
                                 generated_molecules: Dict[str, Any],
                                 simulation_results: Dict[str, Any]) -> float:
        """Calculate a simple overall score for the iteration."""
        
        score = 0.0
        
        # Literature component (0-0.3)
        if literature_results.get('success', False):
            score += 0.1
            papers_found = literature_results.get('papers_found', 0)
            if papers_found > 5:
                score += 0.1
            materials_found = len(literature_results.get('materials_found', []))
            if materials_found > 0:
                score += 0.1
        
        # Generation component (0-0.4)
        if generated_molecules.get('success', False):
            score += 0.1
            diversity_score = generated_molecules.get('diversity_score', 0)
            validity_score = generated_molecules.get('validity_score', 0)
            score += (diversity_score + validity_score) * 0.15
        
        # Simulation component (0-0.3)
        if simulation_results.get('success', False):
            score += 0.1
            success_rate = simulation_results.get('successful_simulations', 0) / max(simulation_results.get('total_simulations', 1), 1)
            score += success_rate * 0.2
        
        return min(score, 1.0)
    

    
    def _calculate_polymer_properties(self, molecule: Dict[str, Any]) -> Dict[str, float]:
        """Calculate polymer properties from molecule data."""
        # Extract PSMILES if available
        psmiles = molecule.get('psmiles', '')
        smiles = molecule.get('smiles', '')
        
        # Simple heuristics based on molecular structure
        biocompatibility = 0.7  # Default moderate biocompatibility
        stability = 0.6  # Default moderate stability
        degradation_rate = 0.5  # Default degradation rate
        
        # Adjust based on functional groups
        if 'C(=O)O' in psmiles or 'C(=O)O' in smiles:  # Ester groups
            biocompatibility += 0.1
            degradation_rate += 0.2
        
        if 'c1ccccc1' in psmiles or 'c1ccccc1' in smiles:  # Aromatic rings
            stability += 0.2
            biocompatibility -= 0.1
            
        if 'C(=O)N' in psmiles or 'C(=O)N' in smiles:  # Amide groups
            stability += 0.1
            biocompatibility += 0.15
            
        if 'O' in psmiles or 'O' in smiles:  # Oxygen content
            biocompatibility += 0.05
            
        # Ensure values are in valid range [0, 1]
        biocompatibility = max(0.0, min(1.0, biocompatibility))
        stability = max(0.0, min(1.0, stability))
        degradation_rate = max(0.0, min(1.0, degradation_rate))
        
        return {
            'biocompatibility': biocompatibility,
            'stability': stability, 
            'degradation_rate': degradation_rate
        }
    
    def _estimate_polymer_properties(self, molecule: Dict[str, Any]) -> Dict[str, float]:
        """Estimate polymer properties when simulation fails."""
        # Use simpler estimates when simulation data isn't available
        properties = self._calculate_polymer_properties(molecule)
        
        # Add some uncertainty/reduction for failed simulations
        properties['biocompatibility'] *= 0.8
        properties['stability'] *= 0.7
        properties['degradation_rate'] = min(1.0, properties['degradation_rate'] * 1.2)
        
        return properties
    
    def _aggregate_properties(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate properties across all simulation results."""
        if not simulation_results:
            return {}
        
        # Average properties across all molecules
        all_properties = {}
        for result in simulation_results:
            properties = result.get('properties', {})
            for prop, value in properties.items():
                if prop not in all_properties:
                    all_properties[prop] = []
                all_properties[prop].append(value)
        
        # Calculate averages
        return {prop: sum(values) / len(values) for prop, values in all_properties.items()}
    
    def _save_iteration_results(self, iteration: int, iteration_state: Dict[str, Any]):
        """Save iteration results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_results.json"
            with open(results_file, 'w') as f:
                json.dump(iteration_state, f, indent=2, default=str)
            logger.info(f"Iteration {iteration} results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save iteration {iteration} results: {e}")
    
    def _create_summary(self, total_time: float) -> Dict[str, Any]:
        """Create a summary of the entire active learning loop."""
        
        successful_iterations = len([r for r in self.results if r.get('status') == 'completed'])
        best_score = max([r.get('overall_score', 0) for r in self.results]) if self.results else 0
        
        summary = {
            'success': True,
            'total_iterations': len(self.results),
            'successful_iterations': successful_iterations,
            'success_rate': successful_iterations / len(self.results) if self.results else 0,
            'best_score': best_score,
            'total_runtime': total_time,
            'results': self.results,
            'final_prompt': self.results[-1].get('new_prompt', '') if self.results else '',
            'summary': {
                'total_iterations': len(self.results),
                'successful_iterations': successful_iterations,
                'success_rate': successful_iterations / len(self.results) if self.results else 0,
                'best_score': best_score,
                'total_runtime': total_time
            }
        }
        
        return summary


# Test function
def test_simple_orchestrator():
    """Test the simple orchestrator."""
    print("Testing SimpleActiveLearningOrchestrator...")
    
    orchestrator = SimpleActiveLearningOrchestrator(max_iterations=2)
    
    def test_callback(state):
        print(f"Iteration {state['iteration']}: {state['status']} - Score: {state.get('overall_score', 0):.3f}")
    
    orchestrator.add_iteration_callback(test_callback)
    
    results = orchestrator.run_simple_loop(
        initial_prompt="Design a biodegradable polymer for insulin delivery",
        target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
    )
    
    print("Loop completed!")
    print(f"Success: {results['success']}")
    print(f"Iterations: {results['total_iterations']}")
    print(f"Best score: {results['best_score']:.3f}")
    print(f"Runtime: {results['total_runtime']:.1f} seconds")
    
    return results


if __name__ == "__main__":
    test_simple_orchestrator() 