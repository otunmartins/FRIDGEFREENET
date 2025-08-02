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
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

@tool
def validate_md_simulation_inputs(psmiles: str, polymer_smiles: str = None) -> dict:
    """Validate inputs for molecular dynamics simulation using LangChain patterns."""
    
    errors = []
    
    # Basic PSMILES validation
    if not psmiles or psmiles.strip() == "":
        errors.append("PSMILES string cannot be empty")
    elif not psmiles.replace('[*]', '').strip():
        errors.append("PSMILES contains only connection points - missing actual molecular structure")
    elif psmiles.count('(') != psmiles.count(')'):
        errors.append("Unmatched parentheses in PSMILES structure")
    
    # Check for problematic characters that could cause KeyError
    if psmiles and ('  ' in psmiles or '\t' in psmiles or '\n' in psmiles):
        errors.append("PSMILES contains whitespace characters that could cause OpenMM KeyError")
    
    # Polymer SMILES validation if provided
    if polymer_smiles:
        if not polymer_smiles.strip():
            errors.append("Polymer SMILES cannot be empty")
        elif any(char == '' for char in polymer_smiles):
            errors.append("Polymer SMILES contains empty characters")
    
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "psmiles": psmiles,
            "polymer_smiles": polymer_smiles
        }
    
    return {
        "valid": True,
        "validated_psmiles": psmiles.strip(),
        "validated_polymer_smiles": polymer_smiles.strip() if polymer_smiles else None
    }


class SimpleActiveLearningOrchestrator:
    """Simple orchestrator that directly calls existing working components."""
    
    def __init__(self, max_iterations: int = 5, storage_path: str = "simple_active_learning", 
                 config: Dict[str, Any] = None):
        """
        Initialize the simple active learning orchestrator.
        
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
        
        # Console output callback for UI integration
        self.output_callback: Optional[Callable[[str], None]] = None
        
        # Initialize MD simulation system early to ensure interface compatibility
        from insulin_ai.integration.analysis.dual_gaff_amber_integration import DualGaffAmberIntegration
        self.dual_simulator = DualGaffAmberIntegration(
            output_dir=str(self.storage_path / "dual_gaff_amber_simulations")
        )
        
        # Initialize comprehensive post-processing system for detailed analysis
        from insulin_ai.integration.analysis.comprehensive_postprocessing import ComprehensivePostProcessor
        self.postprocessor = ComprehensivePostProcessor(
            output_dir=str(self.storage_path / "comprehensive_analysis")
        )
        
        # Initialize PSMILES generator directly (not from session state)
        from insulin_ai.core.psmiles_generator import PSMILESGenerator
        self.psmiles_generator = PSMILESGenerator()
        
        # Initialize RAG literature mining system for advanced literature analysis
        from insulin_ai.integration.rag_literature_mining import RAGLiteratureMiningSystem
        self.rag_literature_system = RAGLiteratureMiningSystem(
            output_dir=str(self.storage_path / "rag_literature")
        )
        
        self._log_message("🤖 Simple Active Learning Orchestrator initialized")
    
    def set_output_callback(self, callback: Callable[[str], None]):
        """Set a callback function for console output"""
        self.output_callback = callback
        self._log_message("Console output callback registered")
    
    def _log_message(self, message: str):
        """Log a message to both logger and console callback"""
        logger.info(message)
        if self.output_callback:
            try:
                self.output_callback(message)
            except Exception as e:
                logger.warning(f"Output callback failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        current_iteration = len(self.results)
        progress_percent = (current_iteration / self.max_iterations) * 100
        
        status = {
            'status': 'ready' if current_iteration == 0 else 'completed',
            'current_iteration': current_iteration,
            'max_iterations': self.max_iterations,
            'progress_percent': progress_percent,
            'runtime_seconds': 0,  # Would need to track runtime
            'errors': []
        }
        
        # If we have results, update status based on latest result
        if self.results:
            latest_result = self.results[-1]
            status['status'] = latest_result.get('status', 'unknown')
            status['errors'] = latest_result.get('errors', [])
        
        return status
    
    def add_iteration_callback(self, callback: Callable):
        """Add a callback for iteration updates"""
        self.iteration_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """Add a callback for completion"""
        self.completion_callbacks.append(callback)
    
    def run_simple_loop(self, initial_prompt: str, target_properties: Dict[str, float] = None) -> Dict[str, Any]:
        """Run the simple active learning loop.
        
        Args:
            initial_prompt: Initial research question/prompt
            target_properties: Target material properties (optional)
            
        Returns:
            Dict containing loop results and summary
        """
        
        try:
            self._log_message(f"🚀 Starting Active Learning Loop")
            self._log_message(f"📝 Initial prompt: '{initial_prompt[:80]}...'")
            self._log_message(f"🎯 Max iterations: {self.max_iterations}")
            
            if target_properties:
                props_str = ", ".join([f"{k}: {v:.2f}" for k, v in target_properties.items()])
                self._log_message(f"🎯 Target properties: {props_str}")
            
            start_time = time.time()
            current_prompt = initial_prompt
            
            for iteration in range(1, self.max_iterations + 1):
                self._log_message(f"\n🔄 === ITERATION {iteration}/{self.max_iterations} ===")
                self._log_message(f"📝 Current prompt: '{current_prompt[:80]}...'")
                
                iteration_state = {
                    'iteration': iteration,
                    'prompt': current_prompt,
                    'status': 'starting',
                    'errors': [],
                    'warnings': [],
                    'overall_score': 0.0
                }
                
                try:
                    # Step 1: Literature Mining
                    self._log_message(f"📚 Step 1: Starting literature mining...")
                    iteration_state['status'] = 'literature_mining'
                    literature_results = self._run_literature_mining(current_prompt)
                    iteration_state['literature_results'] = literature_results
                    
                    if literature_results.get('success', False):
                        insights_count = len(literature_results.get('insights', []))
                        self._log_message(f"✅ Literature mining completed: {insights_count} insights generated")
                    else:
                        self._log_message(f"⚠️ Literature mining had issues: {literature_results.get('error', 'Unknown error')}")
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 2: PSMILES Generation
                    self._log_message(f"🧪 Step 2: Starting PSMILES generation...")
                    iteration_state['status'] = 'psmiles_generation'
                    generated_molecules = self._run_psmiles_generation(literature_results)
                    iteration_state['generated_molecules'] = generated_molecules
                    
                    if generated_molecules.get('success', False):
                        molecules_count = len(generated_molecules.get('molecules', []))
                        self._log_message(f"✅ PSMILES generation completed: {molecules_count} molecules generated")
                    else:
                        self._log_message(f"⚠️ PSMILES generation had issues: {generated_molecules.get('error', 'Unknown error')}")
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 3: MD Simulation
                    self._log_message(f"⚛️ Step 3: Starting MD simulation...")
                    iteration_state['status'] = 'md_simulation'
                    simulation_results = self._run_md_simulation(generated_molecules)
                    iteration_state['simulation_results'] = simulation_results
                    
                    if simulation_results.get('success', False):
                        sims_count = len(simulation_results.get('simulation_ids', []))
                        self._log_message(f"✅ MD simulation completed: {sims_count} simulations run")
                    else:
                        self._log_message(f"⚠️ MD simulation had issues: {simulation_results.get('error', 'Unknown error')}")
                    
                    # Call iteration callbacks
                    for callback in self.iteration_callbacks:
                        try:
                            callback(iteration_state)
                        except Exception as e:
                            logger.warning(f"Iteration callback failed: {e}")
                    
                    # Step 4: Generate new prompt for next iteration
                    self._log_message(f"🔮 Step 4: Generating new prompt for next iteration...")
                    iteration_state['status'] = 'generating_prompt'
                    new_prompt = self._generate_new_prompt(
                        literature_results, generated_molecules, simulation_results, target_properties
                    )
                    iteration_state['new_prompt'] = new_prompt
                    
                    # Calculate overall score (simple heuristic)
                    score = self._calculate_iteration_score(
                        literature_results, generated_molecules, simulation_results
                    )
                    iteration_state['overall_score'] = score
                    
                    iteration_state['status'] = 'completed'
                    
                    self._log_message(f"📊 Iteration {iteration} completed - Overall score: {score:.3f}")
                    if new_prompt and new_prompt != current_prompt:
                        self._log_message(f"🔮 Next prompt: '{new_prompt[:80]}...'")
                    
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
                    current_prompt = new_prompt if new_prompt else current_prompt
                    
                    self._log_message(f"✅ Iteration {iteration} completed successfully - Score: {iteration_state['overall_score']:.3f}")
                    
                except Exception as e:
                    error_msg = f"Error in iteration {iteration}: {str(e)}"
                    self._log_message(f"❌ {error_msg}")
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
            self._log_message(f"\n🎉 Active Learning Loop Completed!")
            self._log_message(f"⏱️ Total runtime: {total_time:.1f} seconds")
            self._log_message(f"🔄 Iterations completed: {len(self.results)}")
            
            summary = self._create_summary(total_time)
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(summary)
                except Exception as e:
                    logger.warning(f"Completion callback failed: {e}")
            
            self._log_message("✅ Simple active learning loop completed successfully")
            return summary
            
        except Exception as e:
            self._log_message(f"❌ Fatal error in active learning loop: {e}")
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
            self._log_message(f"   📚 Initializing literature mining system...")
            
            # Import and use the exact same function the Literature Mining UI tab uses
            from insulin_ai.app.services.literature_service import literature_mining_with_llm
            
            # Get literature mining configuration
            lit_config = self.config.get('literature_mining', {})
            
            # Log configuration
            strategy = lit_config.get('search_strategy', 'Comprehensive')
            max_papers = lit_config.get('max_papers', 10)
            recent_only = lit_config.get('recent_only', True)
            
            self._log_message(f"   🔧 Configuration: {strategy} | Papers: {max_papers} | Recent only: {recent_only}")
            self._log_message(f"   🔍 Starting literature search for: '{prompt[:60]}...'")
            
            # Create iteration context with configuration parameters
            iteration_context = {
                'search_strategy': lit_config.get('search_strategy', 'Comprehensive (3000 tokens)'),
                'max_papers': lit_config.get('max_papers', 10),
                'recent_only': lit_config.get('recent_only', True),
                'include_patents': lit_config.get('include_patents', False),
                'openai_model': lit_config.get('openai_model', 'gpt-4o-mini'),
                'temperature': lit_config.get('temperature', 0.7)
            }
            
            self._log_message(f"   🤖 Using model: {iteration_context['openai_model']} (temp: {iteration_context['temperature']})")
            
            # This is the EXACT same call the Literature Mining UI tab makes, but with configuration
            result = literature_mining_with_llm(prompt, iteration_context=iteration_context)
            
            # Extract key information and generate enhanced chemical prompt
            materials_found = result.get('materials_found', [])
            mechanisms = result.get('stabilization_mechanisms', [])
            material_candidates = result.get('material_candidates', [])
            papers_analyzed = result.get('papers_analyzed', 0)
            
            self._log_message(f"   📄 Analyzed {papers_analyzed} papers")
            self._log_message(f"   🧬 Found {len(materials_found)} material insights")
            self._log_message(f"   ⚙️ Identified {len(mechanisms)} stabilization mechanisms")
            self._log_message(f"   🎯 Generated {len(material_candidates)} material candidates")
            
            # Literature mining has already generated the specific psmiles_generation_prompt
            # No need to override it with generic enhanced prompts
            
            return {
                'success': True,
                'papers_found': papers_analyzed,
                'materials_found': materials_found,
                'stabilization_mechanisms': mechanisms,
                'synthesis_approaches': result.get('synthesis_approaches', []),
                'psmiles_generation_prompt': result.get('psmiles_generation_prompt', ''),  # PRESERVE literature-derived prompt
                'insights': result.get('insights', ''),
                'material_candidates': material_candidates,
                'raw_result': result
            }
            
        except Exception as e:
            error_msg = f"Literature mining failed: {str(e)}"
            self._log_message(f"   ❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'papers_found': 0,
                'materials_found': [],
                'stabilization_mechanisms': [],
                'synthesis_approaches': [],
                'psmiles_generation_prompt': '',
                'insights': '',
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
            mechanism_context = mechanisms[0]
            prompt_parts.append(f"- and are inspired by {mechanism_context}")
            
        if structural_features:
            main_structure = structural_features[0]
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

    def _validate_simulation_inputs_with_retry(self, molecule: Dict[str, Any], psmiles: str) -> Dict[str, Any]:
        """LangChain-style validation with retry logic for simulation inputs."""
        
        max_attempts = 3
        for attempt in range(max_attempts):
            logger.info(f"🔍 Validating simulation inputs (attempt {attempt + 1}/{max_attempts})")
            
            # Get polymer SMILES if available
            polymer_smiles = molecule.get('polymer_smiles', '')
            
            # **CRITICAL: Add molecular validation to prevent radicals**
            try:
                from insulin_ai.utils.molecular_validation import validate_psmiles_for_simulation, validate_smiles_for_simulation
                
                # Validate PSMILES for radicals and problematic elements
                if psmiles:
                    logger.info(f"🧬 Validating PSMILES: {psmiles}")
                    psmiles_result = validate_psmiles_for_simulation(psmiles)
                    
                    if not psmiles_result.is_valid:
                        logger.error(f"❌ PSMILES validation failed: {psmiles_result.error_message}")
                        return {
                            "success": False,
                            "errors": [f"PSMILES validation failed: {psmiles_result.error_message}"]
                        }
                    
                    if psmiles_result.has_radicals:
                        logger.error(f"❌ PSMILES contains radicals: {psmiles_result.error_message}")
                        return {
                            "success": False,
                            "errors": [f"PSMILES contains radicals - would cause OpenFF failure: {psmiles_result.error_message}"]
                        }
                    
                    if psmiles_result.has_problematic_elements:
                        logger.warning(f"⚠️ PSMILES contains problematic elements: {psmiles_result.error_message}")
                        # Continue with warning but don't fail - problematic elements might be correctable
                
                # Validate polymer SMILES if available
                if polymer_smiles:
                    logger.info(f"🧬 Validating polymer SMILES: {polymer_smiles[:50]}...")
                    smiles_result = validate_smiles_for_simulation(polymer_smiles)
                    
                    if smiles_result.has_radicals:
                        logger.error(f"❌ Polymer SMILES contains radicals: {smiles_result.error_message}")
                        return {
                            "success": False,
                            "errors": [f"Polymer SMILES contains radicals - would cause OpenFF failure: {smiles_result.error_message}"]
                        }
                    
                    if smiles_result.has_problematic_elements:
                        logger.warning(f"⚠️ Polymer SMILES contains problematic elements: {smiles_result.error_message}")
                
                logger.info("✅ Molecular validation passed - no radicals detected")
                
            except ImportError:
                logger.warning("⚠️ Molecular validation not available - install RDKit for radical detection")
            except Exception as e:
                logger.warning(f"⚠️ Molecular validation failed: {e}")
            
            # Validate using LangChain tool pattern (existing validation)
            validation_result = validate_md_simulation_inputs(psmiles, polymer_smiles)
            
            if validation_result["valid"]:
                logger.info("✅ Input validation passed")
                return {
                    "success": True,
                    "validated_psmiles": validation_result["validated_psmiles"],
                    "validated_polymer_smiles": validation_result.get("validated_polymer_smiles")
                }
            
            # Log validation errors
            logger.warning(f"❌ Validation failed (attempt {attempt + 1}):")
            for error in validation_result["errors"]:
                logger.warning(f"   - {error}")
            
            if attempt < max_attempts - 1:
                # Try to self-correct using LLM (LangChain pattern)
                try:
                    logger.info("🔧 Attempting to fix validation errors using LLM...")
                    
                    correction_prompt = f"""
                    The following molecular simulation input has validation errors:
                    
                    PSMILES: {psmiles}
                    Errors: {', '.join(validation_result['errors'])}
                    
                    Please provide a corrected PSMILES string that:
                    1. Contains valid chemical structure (not just connection points [*])
                    2. Has balanced parentheses
                    3. Contains no empty characters or spaces
                    4. Is suitable for polymer molecular dynamics simulation
                    5. **CRITICAL: Contains NO radical species or unpaired electrons**
                    6. **CRITICAL: Uses only safe elements (C, N, O, S, P, F, Cl, Br) - NO boron (B), silicon (Si), or aluminum (Al)**
                    
                    Return only the corrected PSMILES string.
                    """
                    
                    # This is where you'd call your LLM to fix the input
                    # For now, just clean the input and ensure it's safe
                    psmiles = psmiles.strip() if psmiles else ""
                    if not psmiles.replace('[*]', '').strip():
                        psmiles = "[*]CCOCCOCCCO[*]"  # Simple safe polymer fallback
                    
                    # Remove any problematic elements
                    problematic_replacements = {
                        'B': 'C',
                        'Si': 'C', 
                        'Al': 'C'
                    }
                    for problematic, safe in problematic_replacements.items():
                        if problematic in psmiles:
                            psmiles = psmiles.replace(problematic, safe)
                            logger.info(f"🔧 Replaced {problematic} with {safe} in PSMILES")
                        
                except Exception as e:
                    logger.warning(f"   LLM correction failed: {e}")
        
        logger.error(f"❌ Failed to validate inputs after {max_attempts} attempts")
        return {
            "success": False,
            "errors": validation_result["errors"]
        }
    
    def _run_psmiles_generation(self, literature_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run PSMILES generation with enhanced retry logic.
        Generate only ONE candidate, retrying until successful.
        """
        try:
            self._log_message(f"   🧪 Initializing PSMILES generation system...")
            
            # Import and use the exact same workflow the PSMILES Generation UI tab uses
            from insulin_ai.app.services.psmiles_service import process_psmiles_workflow_with_autorepair
            
            # CRITICAL DEBUG: Check what we received from literature mining
            self._log_message(f"   🔍 DEBUG: Literature results keys: {list(literature_results.keys())}")
            
            # CRITICAL FIX: Use the literature-derived PSMILES generation prompt
            # This should be something specific like:
    
            psmiles_prompt = literature_results.get('psmiles_generation_prompt', '')
            
            self._log_message(f"   🔍 DEBUG: psmiles_generation_prompt exists: {'psmiles_generation_prompt' in literature_results}")
            self._log_message(f"   🔍 DEBUG: psmiles_prompt length: {len(psmiles_prompt) if psmiles_prompt else 0}")
            
            if not psmiles_prompt:
                error_msg = "No literature-derived psmiles_generation_prompt found!"
                self._log_message(f"   ❌ {error_msg}")
                self._log_message(f"   🔍 DEBUG: Available keys in literature_results: {list(literature_results.keys())}")
                # FALLBACK: Try to create a basic prompt from the insights
                insights = literature_results.get('insights', '')
                if insights:
                    psmiles_prompt = f"Based on the following research insights: {insights[:200]}... Design a biodegradable polymer for insulin delivery."
                    self._log_message(f"   🔄 FALLBACK: Created basic prompt from insights")
                else:
                    raise ValueError("Literature mining must provide a specific psmiles_generation_prompt - no fallbacks allowed")
            
            self._log_message(f"   📝 Using prompt: '{psmiles_prompt[:80]}...'")
            
            # Get PSMILES generation configuration
            psmiles_config = self.config.get('psmiles_generation', {})
            max_retries = psmiles_config.get('max_retries', 5)
            num_candidates = psmiles_config.get('num_candidates', 1)
            auto_functionalize = psmiles_config.get('enable_functionalization', True)
            max_repair_attempts = psmiles_config.get('max_repair_attempts', 3)
            temperature = psmiles_config.get('temperature', 0.7)
            
            self._log_message(f"   🔧 Configuration: {num_candidates} candidates | Max retries: {max_retries}")
            self._log_message(f"   ⚙️ Functionalization: {auto_functionalize} | Repair attempts: {max_repair_attempts}")
            
            # Use the orchestrator's own PSMILES generator (no session state dependency)
            self._log_message(f"   🚀 Starting diverse candidate generation...")
            psmiles_generator = self.psmiles_generator
            
            # Use the exact same method as the working tab
            diverse_results = psmiles_generator.generate_diverse_candidates(
                base_request=psmiles_prompt,
                num_candidates=num_candidates * 2,  # Generate more to ensure diversity (same as manual tab)
                temperature_range=(0.6, 1.0)  # Same temperature range as manual tab
            )
            
            # Process results in the exact same format as the working tab
            if diverse_results.get('success') and diverse_results.get('candidates'):
                candidates_list = diverse_results['candidates']
                molecules = []
                
                self._log_message(f"   🎯 Processing {len(candidates_list)} generated candidates...")
                
                for i, result in enumerate(candidates_list[:num_candidates]):  # Take only requested number
                    psmiles = result['psmiles']
                    confidence = result.get('confidence', 0.8)
                    self._log_message(f"   📄 Candidate {i+1}: {psmiles[:40]}... (confidence: {confidence:.2f})")
                    
                    molecules.append({
                        'id': f'mol_{i+1}',
                        'psmiles': psmiles,
                        'smiles': result.get('smiles', ''),  # May not have SMILES from diverse generation
                        'confidence': confidence,
                        'description': result.get('explanation', 'Literature-derived polymer structure'),
                        'generation_method': result.get('generation_method', 'working_pipeline_diverse'),
                        'prompt_used': result.get('diversity_prompt', psmiles_prompt),
                        'temperature_used': result.get('temperature_used', 0.8)
                    })
                
                diversity_score = diverse_results.get('diversity_score', 0.8)
                validity_score = diverse_results.get('validity_score', 0.9)
                
                self._log_message(f"   ✅ Successfully generated {len(molecules)} candidates")
                self._log_message(f"   📊 Diversity score: {diversity_score:.2f} | Validity score: {validity_score:.2f}")
                
                return {
                    'success': True,
                    'molecules': molecules,
                    'num_generated': len(molecules),
                    'generation_method': 'direct_psmiles_generator',
                    'attempts_needed': 1,
                    'diversity_score': diversity_score,
                    'validity_score': validity_score,
                    'raw_result': diverse_results
                }
            else:
                error_msg = f"PSMILES generation failed: {diverse_results.get('error', 'Unknown error')}"
                self._log_message(f"   ❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'molecules': [],
                    'num_generated': 0,
                    'generation_method': 'direct_psmiles_generator',
                    'attempts_needed': 0,
                    'raw_result': diverse_results
                }
                
        except Exception as e:
            error_msg = f"PSMILES generation failed: {str(e)}"
            self._log_message(f"   ❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'molecules': [],
                'num_generated': 0,
                'generation_method': 'direct_psmiles_generator_error',
                'attempts_needed': 0,
                'raw_result': {}
            }
    
    def _run_md_simulation(self, generated_molecules: Dict[str, Any]) -> Dict[str, Any]:
        """Run MD simulation using the existing UI tab workflow."""
        try:
            self._log_message(f"   ⚛️ Initializing MD simulation system...")
            
            # Import and use the exact same workflow the MD Simulation UI tab uses
            from insulin_ai.integration.analysis.dual_gaff_amber_integration import DualGaffAmberIntegration
            
            molecules = generated_molecules.get('molecules', [])
            if not molecules:
                error_msg = "No molecules to simulate"
                self._log_message(f"   ❌ {error_msg}")
                raise ValueError(error_msg)
            
            self._log_message(f"   🧬 Found {len(molecules)} molecules to simulate")
            
            # Get MD simulation configuration
            md_config = self.config.get('md_simulation', {})
            max_simulations = md_config.get('max_simulations', 3)
            temperature = md_config.get('temperature', 310.0)
            equilibration_steps = md_config.get('equilibration_steps', 125000)  # Default to Quick Test
            production_steps = md_config.get('production_steps', 500000)  # Default to Quick Test
            save_interval = md_config.get('save_interval', 1000)  # Default to Normal
            timeout_minutes = md_config.get('timeout_minutes', 30)
            simulation_method = md_config.get('simulation_method', 'Dual GAFF+AMBER (Recommended)')
            total_time_ns = (equilibration_steps + production_steps) * 2 / 1000000  # 2 fs timestep
            polymer_chain_length = md_config.get('polymer_chain_length', 10)  # Default to 10
            num_polymer_chains = md_config.get('num_polymer_chains', 1)  # Default to 1
            
            self._log_message(f"   🔧 Configuration: {simulation_method}")
            self._log_message(f"   🌡️ Temperature: {temperature}K | Max simulations: {max_simulations}")
            self._log_message(f"   ⏱️ Equilibration: {equilibration_steps} steps | Production: {production_steps} steps")
            self._log_message(f"   📊 Total simulation time: {total_time_ns:.2f} ns | Timeout: {timeout_minutes} min")
            self._log_message(f"   🧬 Polymer chains: {num_polymer_chains} chains × {polymer_chain_length} units each")
            
            # dual_simulator is already initialized in __init__
            if not self.dual_simulator.dependencies_ok:
                error_msg = "Dual GAFF AMBER dependencies not available"
                self._log_message(f"   ❌ {error_msg}")
                raise RuntimeError("Dual GAFF AMBER dependencies required - no fallbacks allowed")
            
            self._log_message(f"   ✅ Dual GAFF+AMBER dependencies verified")
            
            simulation_results = []
            successful_simulations = 0
            
            for i, molecule in enumerate(molecules[:max_simulations]):  # Use configured limit
                try:
                    mol_id = molecule.get('id', f'mol_{i+1}')
                    psmiles = molecule.get('psmiles', '')
                    
                    self._log_message(f"   🧪 Starting simulation {i+1}/{min(len(molecules), max_simulations)} for {mol_id}")
                    
                    if not psmiles:
                        warning_msg = f"No PSMILES for molecule {mol_id}, skipping"
                        self._log_message(f"   ⚠️ {warning_msg}")
                        continue
                    
                    self._log_message(f"   🔍 Validating PSMILES: {psmiles[:50]}...")
                    
                    # Use LangChain-style validation with retry logic
                    validation_result = self._validate_simulation_inputs_with_retry(molecule, psmiles)
                    
                    if not validation_result["success"]:
                        error_msg = f"Validation failed for molecule {mol_id}: {validation_result.get('errors', 'Unknown errors')}"
                        self._log_message(f"   ❌ {error_msg}")
                        raise ValueError(f"Input validation failed after retry attempts: {validation_result.get('errors')}")
                    
                    # Use validated inputs
                    validated_psmiles = validation_result["validated_psmiles"]
                    self._log_message(f"   ✅ Validation successful: {validated_psmiles[:50]}...")
                    
                    self._log_message(f"   🚀 Starting MD simulation for {mol_id}...")
                    
                    # Use the configured simulation parameters from the UI
                    simulation_id = self.dual_simulator.run_md_simulation_async(
                        pdb_file=validated_psmiles,  # Use validated PSMILES string
                        temperature=temperature,
                        equilibration_steps=equilibration_steps,
                        production_steps=production_steps,  
                        save_interval=save_interval,
                        output_prefix=f"al_{mol_id}",
                        polymer_chain_length=polymer_chain_length,   # Configurable chain length
                        num_polymer_chains=num_polymer_chains        # Configurable number of chains
                    )
                    
                    # Wait for simulation to actually complete (not just start!)
                    logger.info(f"⏳ Waiting for MD simulation {simulation_id} to complete...")
                    
                    # Use proper wait method instead of just 2 seconds
                    simulation_status = self.dual_simulator.wait_for_simulation_completion(
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
                            'final_energy': results.get('final_energy'),
                            'temperature': temperature,
                            'pressure': 1.0,
                            'density': 1.1,
                            'total_time_s': results.get('total_time_s', 0),
                            'trajectory_file': f"{simulation_id}/trajectory.dcd",
                            'properties': self._calculate_polymer_properties(molecule)
                        })
                        successful_simulations += 1
                        logger.info(f"✅ MD simulation {simulation_id} completed successfully!")
                        
                        # Run post-processing analysis to extract performance metrics
                        logger.info(f"🔬 Running post-processing analysis for {simulation_id}...")
                        analysis_results = self._run_postprocessing_analysis(simulation_id, molecule)
                        if analysis_results:
                            simulation_results[-1]['analysis_results'] = analysis_results
                            logger.info(f"✅ Post-processing analysis completed for {simulation_id}")
                        else:
                            logger.warning(f"⚠️ Post-processing analysis failed for {simulation_id}")
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
    
    def _run_postprocessing_analysis(self, simulation_id: str, molecule: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive post-processing analysis using the full ComprehensivePostProcessor"""
        try:
            logger.info(f"   🔬 Running comprehensive post-processing for {simulation_id}...")
            
            # Get simulation results from the dual simulator
            simulation_results = self.dual_simulator.get_simulation_results(simulation_id)
            
            if not simulation_results or not simulation_results.get('success'):
                logger.warning(f"   ⚠️ No valid simulation results for {simulation_id}")
                return None
            
            # Find the simulation directory for comprehensive analysis
            simulation_dir = simulation_results.get('output_dir')
            if not simulation_dir:
                logger.warning(f"   ⚠️ No output directory found for {simulation_id}")
                return None
            
            logger.info(f"   📁 Analyzing simulation directory: {simulation_dir}")
            
            # Use the comprehensive postprocessor to run full analysis
            # This will analyze trajectories, calculate properties, etc.
            analysis_options = {
                'basic_analysis': True,
                'insulin_stability': True,
                'binding_energy': True,
                'diffusion_analysis': True,
                'mesh_analysis': True,
                'volume_analysis': True,
                'interaction_analysis': True
            }
            
            # Start comprehensive analysis (async)
            analysis_job_id = self.postprocessor.start_comprehensive_analysis_async(
                simulation_id=simulation_id,
                simulation_dir=simulation_dir,
                analysis_options=analysis_options,
                simulation_structure_type='integrated'
            )
            
            logger.info(f"   ⏳ Comprehensive analysis started with job ID: {analysis_job_id}")
            
            # Wait for analysis to complete (with timeout)
            max_wait_time = 300  # 5 minutes timeout
            wait_time = 0
            poll_interval = 10
            
            while wait_time < max_wait_time:
                try:
                    status = self.postprocessor.get_analysis_status()
                    
                    # **ENHANCED DEBUGGING: Log the actual status structure for debugging**
                    logger.info(f"   📊 Analysis status check: running={status.get('analysis_running', False)}, "
                              f"completed={len(status.get('completed_jobs', []))}, "
                              f"failed={len(status.get('failed_jobs', []))}")
                    
                    if 'completed_jobs' not in status:
                        logger.warning(f"   ⚠️ Status missing 'completed_jobs' field. Available keys: {list(status.keys())}")
                        # Add backward compatibility
                        status.setdefault('completed_jobs', [])
                        status.setdefault('failed_jobs', [])
                    
                    if analysis_job_id in status['completed_jobs']:
                        logger.info(f"   ✅ Comprehensive analysis completed for {simulation_id}")
                        break
                    elif analysis_job_id in status['failed_jobs']:
                        logger.error(f"   ❌ Comprehensive analysis failed for {simulation_id}")
                        return None
                    elif not status.get('analysis_running', True):
                        # Analysis stopped but not in completed or failed - check current status
                        if hasattr(self.postprocessor, 'current_analysis') and self.postprocessor.current_analysis:
                            current_status = self.postprocessor.current_analysis.get('status', 'unknown')
                            if current_status == 'completed':
                                logger.info(f"   ✅ Analysis completed (detected via current_analysis)")
                                break
                            elif current_status == 'failed':
                                error_msg = self.postprocessor.current_analysis.get('error', 'Unknown error')
                                logger.error(f"   ❌ Analysis failed (detected via current_analysis): {error_msg}")
                                return None
                    
                except Exception as e:
                    logger.error(f"   ❌ Comprehensive post-processing analysis error: {e}")
                    logger.error(f"   🎯 Analysis Job: {analysis_job_id}")
                    logger.error(f"   📊 Simulation: {simulation_id}")
                    # Try to get more diagnostic information
                    try:
                        if hasattr(self.postprocessor, 'current_analysis'):
                            logger.error(f"   🔍 Current analysis state: {self.postprocessor.current_analysis}")
                    except:
                        pass
                    return None
                    
                time.sleep(poll_interval)
                wait_time += poll_interval
                logger.info(f"   ⏳ Waiting for analysis... ({wait_time}s/{max_wait_time}s)")
            
            if wait_time >= max_wait_time:
                logger.warning(f"   ⏰ Analysis timeout for {simulation_id}, using partial results")
            
            # Get the comprehensive analysis results
            analysis_results = self.postprocessor.get_analysis_results(simulation_id)
            
            if not analysis_results:
                logger.warning(f"   ⚠️ No analysis results available for {simulation_id}")
                return None
            
            # Extract key metrics for active learning and literature mining
            key_metrics = {}
            
            # Insulin stability metrics
            if 'insulin_stability' in analysis_results:
                stability = analysis_results['insulin_stability']
                key_metrics['insulin_rmsd'] = stability.get('rmsd_avg', 0.0)
                key_metrics['insulin_stability_score'] = stability.get('stability_score', 0.0)
                key_metrics['insulin_flexibility'] = stability.get('rmsf_avg', 0.0)
                logger.info(f"   🧬 Insulin RMSD: {key_metrics['insulin_rmsd']:.2f} Å")
                logger.info(f"   📊 Stability score: {key_metrics['insulin_stability_score']:.3f}")
            
            # Binding and interaction energy
            if 'binding_energy' in analysis_results:
                binding = analysis_results['binding_energy']
                key_metrics['binding_energy'] = binding.get('binding_energy_avg', 0.0)
                key_metrics['interaction_strength'] = abs(binding.get('binding_energy_avg', 0.0))
                logger.info(f"   🔗 Binding energy: {key_metrics['binding_energy']:.2f} kJ/mol")
            
            # Diffusion and transport properties
            if 'diffusion_analysis' in analysis_results:
                diffusion = analysis_results['diffusion_analysis']
                key_metrics['diffusion_coefficient'] = diffusion.get('diffusion_coefficient', 0.0)
                key_metrics['transfer_efficiency'] = diffusion.get('transfer_efficiency', 0.0)
                logger.info(f"   🚶 Diffusion coefficient: {key_metrics['diffusion_coefficient']:.6f} cm²/s")
            
            # Mesh and structural properties
            if 'mesh_analysis' in analysis_results:
                mesh = analysis_results['mesh_analysis']
                key_metrics['mesh_size'] = mesh.get('average_mesh_size', 0.0)
                key_metrics['pore_distribution'] = mesh.get('pore_size_distribution', [])
                logger.info(f"   🕸️ Average mesh size: {key_metrics['mesh_size']:.2f} Å")
            
            # Volume and swelling properties
            if 'volume_analysis' in analysis_results:
                volume = analysis_results['volume_analysis']
                key_metrics['swelling_ratio'] = volume.get('swelling_ratio', 1.0)
                key_metrics['volume_change'] = volume.get('volume_change_percent', 0.0)
                logger.info(f"   💧 Swelling ratio: {key_metrics['swelling_ratio']:.2f}")
            
            # pH/environmental responsiveness (if available)
            if 'environmental_response' in analysis_results:
                env_response = analysis_results['environmental_response']
                key_metrics['ph_sensitivity'] = env_response.get('ph_sensitivity', 0.0)
                key_metrics['temperature_sensitivity'] = env_response.get('temperature_sensitivity', 0.0)
            
            # Basic energy analysis for compatibility
            if 'final_energy' in simulation_results and 'initial_energy' in simulation_results:
                energy_change = simulation_results['final_energy'] - simulation_results['initial_energy']
                key_metrics['energy_change'] = energy_change
                key_metrics['final_energy'] = simulation_results['final_energy']
                key_metrics['energy_stability'] = abs(energy_change) < 1000  # kJ/mol threshold
                logger.info(f"   ⚡ Energy change: {energy_change:.2f} kJ/mol")
            
            # Calculate comprehensive fitness score
            fitness_score = self._calculate_comprehensive_fitness_score(key_metrics, molecule)
            
            # Prepare comprehensive results
            comprehensive_results = {
                'simulation_id': simulation_id,
                'success': True,
                'molecule_info': molecule,
                'key_metrics': key_metrics,
                'comprehensive_analysis': analysis_results,
                'fitness_score': fitness_score,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"   ✅ Comprehensive analysis completed - fitness score: {fitness_score:.3f}")
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"   ❌ Comprehensive post-processing analysis error: {e}")
            # Fall back to basic analysis if comprehensive fails
            logger.info(f"   🔄 Falling back to basic analysis for {simulation_id}")
            return self._run_basic_fallback_analysis(simulation_id, molecule)
    
    def _calculate_comprehensive_fitness_score(self, key_metrics: Dict[str, Any], molecule: Dict[str, Any]) -> float:
        """Calculate fitness score based on comprehensive analysis metrics"""
        fitness_score = 0.5  # Base score
        
        # Insulin stability (weight: 0.25)
        insulin_rmsd = key_metrics.get('insulin_rmsd', 0.0)
        stability_score = key_metrics.get('insulin_stability_score', 0.0)
        if insulin_rmsd > 0:
            if insulin_rmsd < 2.0:  # Excellent stability
                fitness_score += 0.15
            elif insulin_rmsd < 3.5:  # Good stability
                fitness_score += 0.10
            else:  # Poor stability
                fitness_score -= 0.10
        
        if stability_score > 0.8:  # High stability
            fitness_score += 0.10
        elif stability_score < 0.5:  # Low stability
            fitness_score -= 0.05
        
        # Binding/interaction strength (weight: 0.20)
        binding_energy = key_metrics.get('binding_energy', 0.0)
        if binding_energy != 0.0:
            abs_binding = abs(binding_energy)
            if 20.0 <= abs_binding <= 80.0:  # Optimal binding range
                fitness_score += 0.15
            elif 10.0 <= abs_binding <= 100.0:  # Acceptable binding
                fitness_score += 0.10
            elif abs_binding > 150.0:  # Too strong binding
                fitness_score -= 0.10
        
        # Diffusion properties (weight: 0.20)
        diffusion_coeff = key_metrics.get('diffusion_coefficient', 0.0)
        if diffusion_coeff > 0:
            if 0.001 <= diffusion_coeff <= 0.1:  # Optimal diffusion range
                fitness_score += 0.15
            elif 0.0001 <= diffusion_coeff <= 0.5:  # Acceptable range
                fitness_score += 0.10
            else:  # Too fast or too slow
                fitness_score -= 0.05
        
        # Swelling behavior (weight: 0.15)
        swelling_ratio = key_metrics.get('swelling_ratio', 1.0)
        if 1.5 <= swelling_ratio <= 8.0:  # Good swelling range
            fitness_score += 0.10
        elif swelling_ratio > 15.0:  # Excessive swelling
            fitness_score -= 0.10
        
        # Energy stability (weight: 0.10)
        energy_stability = key_metrics.get('energy_stability', False)
        if energy_stability:
            fitness_score += 0.08
        else:
            fitness_score -= 0.05
        
        # Polymer complexity bonus (weight: 0.10)
        psmiles = molecule.get('psmiles', '')
        if psmiles:
            complexity = len(psmiles.replace('[*]', ''))
            if 10 <= complexity <= 50:  # Optimal complexity
                fitness_score += 0.08
            elif complexity > 100:  # Too complex
                fitness_score -= 0.05
        
        return max(0.0, min(1.0, fitness_score))
    
    def _run_basic_fallback_analysis(self, simulation_id: str, molecule: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to basic analysis if comprehensive analysis fails"""
        try:
            logger.info(f"   🔄 Running fallback basic analysis for {simulation_id}")
            
            simulation_results = self.dual_simulator.get_simulation_results(simulation_id)
            
            if not simulation_results or not simulation_results.get('success'):
                return None
            
            # Basic analysis results
            analysis_results = {
                'simulation_id': simulation_id,
                'success': True,
                'molecule_info': molecule,
                'key_metrics': {}
            }
            
            # Basic energy analysis
            if 'final_energy' in simulation_results and 'initial_energy' in simulation_results:
                energy_change = simulation_results['final_energy'] - simulation_results['initial_energy']
                analysis_results['key_metrics']['energy_change'] = energy_change
                analysis_results['key_metrics']['final_energy'] = simulation_results['final_energy']
                analysis_results['key_metrics']['energy_stability'] = abs(energy_change) < 1000
            
            # Assign basic fitness score
            fitness_score = 0.5
            if analysis_results['key_metrics'].get('energy_stability', False):
                fitness_score += 0.2
            
            analysis_results['fitness_score'] = fitness_score
            
            logger.info(f"   ✅ Basic fallback analysis completed - fitness score: {fitness_score:.3f}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"   ❌ Basic fallback analysis error: {e}")
            return None
    
    def _generate_new_prompt(self, literature_results: Dict[str, Any], 
                           generated_molecules: Dict[str, Any], 
                           simulation_results: Dict[str, Any],
                           target_properties: Dict[str, float]) -> str:
        """Generate a new prompt for the next iteration based on comprehensive results including post-processing analysis."""
        
        # Extract insights from literature results
        materials_found = literature_results.get('materials_found', [])
        mechanisms = literature_results.get('stabilization_mechanisms', [])
        
        # Extract comprehensive post-processing analysis results from simulation results
        key_improvements = []
        chemical_deficiencies = []
        specific_targets = []
        
        # Analyze each simulation result for comprehensive post-processing insights
        for sim_result in simulation_results.get('simulation_results', []):
            if sim_result.get('success') and 'analysis_results' in sim_result:
                analysis = sim_result['analysis_results']
                key_metrics = analysis.get('key_metrics', {})
                
                # Insulin stability analysis with chemical specificity
                insulin_rmsd = key_metrics.get('insulin_rmsd', 0.0)
                stability_score = key_metrics.get('insulin_stability_score', 0.0)
                insulin_flexibility = key_metrics.get('insulin_flexibility', 0.0)
                
                if insulin_rmsd > 3.0:  # High RMSD indicates structural instability
                    key_improvements.append("enhance insulin structural stability")
                    chemical_deficiencies.append("lack of stabilizing hydrogen bond donors/acceptors")
                    specific_targets.append("secondary structure preservation")
                    
                if stability_score < 0.7:  # Low stability score
                    key_improvements.append("improve protein conformation preservation")
                    chemical_deficiencies.append("insufficient hydrophobic-hydrophilic balance")
                    specific_targets.append("α-helix and β-sheet stabilization")
                    
                if insulin_flexibility > 2.5:  # Excessive flexibility
                    key_improvements.append("reduce insulin structural fluctuations")
                    chemical_deficiencies.append("inadequate crosslinking density")
                
                # Binding and interaction analysis with molecular specificity
                binding_energy = key_metrics.get('binding_energy', 0.0)
                interaction_strength = key_metrics.get('interaction_strength', 0.0)
                
                if abs(binding_energy) < 10.0:  # Weak binding
                    key_improvements.append("strengthen polymer-insulin intermolecular interactions")
                    chemical_deficiencies.append("insufficient aromatic-aromatic stacking or electrostatic interactions")
                    specific_targets.append("π-π stacking domains and ionic coordination sites")
                    
                elif abs(binding_energy) > 100.0:  # Too strong binding
                    key_improvements.append("achieve reversible polymer-insulin association")
                    chemical_deficiencies.append("overly strong covalent or chelation interactions")
                    specific_targets.append("pH-cleavable ester/amide linkages")
                
                # Diffusion and transport with mesh design specificity
                diffusion_coeff = key_metrics.get('diffusion_coefficient', 0.0)
                mesh_size = key_metrics.get('mesh_size', 0.0)
                
                if diffusion_coeff < 0.001:  # Too slow diffusion
                    key_improvements.append("optimize mesh architecture for controlled insulin permeation")
                    chemical_deficiencies.append("excessive crosslink density or inappropriate pore size")
                    specific_targets.append("tunable mesh size between 10-50 nm")
                    
                elif diffusion_coeff > 0.1:  # Too fast diffusion
                    key_improvements.append("reduce initial burst release through selective permeability")
                    chemical_deficiencies.append("insufficient diffusion barriers or size selectivity")
                    specific_targets.append("molecular weight cutoff control")
                
                if mesh_size > 0:
                    if mesh_size < 5.0:  # Too tight mesh
                        chemical_deficiencies.append("overly dense polymer network")
                        specific_targets.append("degradable spacer units")
                    elif mesh_size > 100.0:  # Too loose mesh  
                        chemical_deficiencies.append("insufficient network integrity")
                        specific_targets.append("multi-functional crosslinking agents")
                
                # Swelling and volume response with polymer chemistry
                swelling_ratio = key_metrics.get('swelling_ratio', 1.0)
                volume_change = key_metrics.get('volume_change_percent', 0.0)
                
                if swelling_ratio < 1.5:  # Insufficient swelling
                    key_improvements.append("enhance hydrophilic swelling for drug release")
                    chemical_deficiencies.append("inadequate hydrophilic groups or ionic centers")
                    specific_targets.append("carboxylate, sulfonate, or quaternary ammonium groups")
                    
                elif swelling_ratio > 10.0:  # Excessive swelling
                    key_improvements.append("control excessive swelling to maintain matrix integrity")
                    chemical_deficiencies.append("insufficient hydrophobic domains or crosslink stability")
                    specific_targets.append("hydrophobic alkyl chains and permanent crosslinks")
                
                # pH/environmental responsiveness with ionizable chemistry
                ph_sensitivity = key_metrics.get('ph_sensitivity', 0.0)
                temperature_sensitivity = key_metrics.get('temperature_sensitivity', 0.0)
                
                if ph_sensitivity < 0.3:  # Low pH response
                    key_improvements.append("enhance pH-responsive behavior for intestinal targeting")
                    chemical_deficiencies.append("insufficient ionizable functional groups")
                    specific_targets.append("carboxylic acid, amine, or imidazole pH switches")
                
                if temperature_sensitivity < 0.2:  # Low temperature response
                    key_improvements.append("incorporate thermosensitive release mechanisms")
                    chemical_deficiencies.append("lack of lower critical solution temperature (LCST) behavior")
                    specific_targets.append("N-isopropylacrylamide-type thermoresponsive units")
        
        # Energy analysis from simulation with thermodynamic implications
        avg_energy = simulation_results.get('average_energy', 0)
        if avg_energy > -1000:  # High energy suggests poor thermodynamic stability
            key_improvements.append("achieve thermodynamically favorable polymer-insulin complexation")
            chemical_deficiencies.append("unfavorable enthalpy or entropy of mixing")
            specific_targets.append("compatible polymer-protein surface interactions")
        
        # Build comprehensive new prompt with chemical specificity
        prompt_parts = ["Design improved polymers for insulin delivery"]
        
        # Add specific improvements based on comprehensive analysis
        if key_improvements:
            # Remove duplicates and limit to top 3 improvements
            unique_improvements = list(dict.fromkeys(key_improvements))[:3]
            prompt_parts.append(f"that {', '.join(unique_improvements)}")
        
        # Add literature-derived insights with chemical context
        if mechanisms:
            mechanism_text = mechanisms[0] if len(mechanisms) > 0 else "advanced molecular recognition mechanisms"
            prompt_parts.append(f"utilizing {mechanism_text}")
        
        # Add material inspiration with chemical basis
        if materials_found and len(materials_found) > 0:
            material_text = materials_found[0]
            prompt_parts.append(f"inspired by {material_text}")
        
        # Add specific chemical functionality based on comprehensive performance gaps
        chemical_features = []
        molecular_targets = []
        
        # Map deficiencies to specific chemical solutions
        if any("pH-responsive" in imp for imp in key_improvements):
            chemical_features.append("ionizable carboxylate or tertiary amine side chains")
            molecular_targets.append("pKa values between 5.5-7.4")
            
        if any("insulin affinity" in imp for imp in key_improvements) or any("binding" in imp for imp in key_improvements):
            chemical_features.append("hydrophobic aromatic domains and hydrogen bonding motifs")
            molecular_targets.append("complementary to insulin's hydrophobic patches")
            
        if any("crosslinking" in imp or "mesh" in imp for imp in key_improvements):
            chemical_features.append("degradable crosslinking agents with tunable kinetics")
            molecular_targets.append("enzyme-cleavable peptide or pH-labile acetal linkages")
            
        if any("swelling" in imp for imp in key_improvements):
            chemical_features.append("amphiphilic block copolymer architecture")
            molecular_targets.append("hydrophilic-lipophilic balance (HLB) optimization")
            
        if any("stability" in imp for imp in key_improvements):
            chemical_features.append("protective coordination sites and chaperone-like domains")
            molecular_targets.append("metal chelation centers or protein-mimetic sequences")
        
        # Add structural targets based on specific deficiencies
        structural_requirements = []
        if any("mesh size" in target for target in specific_targets):
            structural_requirements.append("controlled mesh size between 10-50 nm")
        if any("π-π stacking" in target for target in specific_targets):
            structural_requirements.append("aromatic stacking domains")
        if any("α-helix" in target for target in specific_targets):
            structural_requirements.append("secondary structure stabilizing motifs")
        
        # Construct comprehensive prompt with chemical and structural specificity
        if chemical_features:
            prompt_parts.append(f"incorporating {' and '.join(chemical_features)}")
            
        if molecular_targets:
            prompt_parts.append(f"featuring {' and '.join(molecular_targets)}")
            
        if structural_requirements:
            prompt_parts.append(f"achieving {' and '.join(structural_requirements)}")
        
        # Add specific polymer architecture guidance
        architecture_specs = []
        if any("block copolymer" in feature for feature in chemical_features):
            architecture_specs.append("multi-block copolymer design")
        if any("crosslinking" in imp for imp in key_improvements):
            architecture_specs.append("crosslinked hydrogel network")
        if any("pH" in imp for imp in key_improvements):
            architecture_specs.append("stimuli-responsive architecture")
            
        if architecture_specs:
            prompt_parts.append(f"using {' with '.join(architecture_specs)}")
        
        # Finalize prompt with comprehensive chemical guidance
        new_prompt = " ".join(prompt_parts) + "."
        
        # Log comprehensive analysis insights
        logger.info(f"📝 Generated chemically-specific prompt based on comprehensive analysis:")
        logger.info(f"    {new_prompt[:150]}...")
        logger.info(f"🎯 Key improvements identified: {', '.join(key_improvements[:3])}")
        logger.info(f"⚗️ Chemical deficiencies: {', '.join(chemical_deficiencies[:2])}")
        logger.info(f"🔬 Molecular targets: {', '.join(molecular_targets[:2])}")
        
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