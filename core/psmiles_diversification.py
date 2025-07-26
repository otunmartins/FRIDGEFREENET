"""
PSMILES Diversification System

This module provides a comprehensive system for generating diverse PSMILES candidates
by combining prompt diversification with chemical functionalization techniques.

Key Components:
1. PromptDiversifier - Creates varied prompts for each candidate
2. FunctionalizationEngine - Adds random functional groups using psmiles package
3. DiversityValidator - Ensures candidates are truly unique
4. CandidateOrchestrator - Coordinates the entire generation process
"""

import random
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
from openff.toolkit import Molecule

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CandidateConfig:
    """Configuration for candidate generation."""
    base_request: str
    num_candidates: int = 5
    temperature_range: Tuple[float, float] = (0.6, 1.0)
    enable_functionalization: bool = True
    diversity_threshold: float = 0.3
    max_functionalization_attempts: int = 3
    prompt_variation_strategies: List[str] = None
    
    def __post_init__(self):
        if self.prompt_variation_strategies is None:
            self.prompt_variation_strategies = [
                'perspective_variation',
                'property_emphasis',
                'structure_focus',
                'application_context',
                'chemical_modification'
            ]


class PromptStrategy(ABC):
    """Abstract base class for prompt diversification strategies."""
    
    @abstractmethod
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate a variant of the base prompt for a specific candidate."""
        pass


class PerspectiveVariationStrategy(PromptStrategy):
    """Varies the perspective or approach to the same material request."""
    
    def __init__(self):
        self.perspectives = [
            "Design a {base_request} with enhanced mechanical properties",
            "Create a {base_request} optimized for biocompatibility",
            "Develop a {base_request} with improved thermal stability",
            "Engineer a {base_request} with better processability",
            "Synthesize a {base_request} with controlled degradation",
            "Formulate a {base_request} with specific surface properties",
            "Design a {base_request} with tunable elasticity",
            "Create a {base_request} with improved barrier properties"
        ]
    
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate perspective-varied prompt."""
        perspective = self.perspectives[candidate_index % len(self.perspectives)]
        variant_prompt = perspective.format(base_request=base_prompt)
        
        return {
            'prompt': variant_prompt,
            'strategy': 'perspective_variation',
            'original_prompt': base_prompt,
            'variation_details': {
                'perspective_used': perspective,
                'candidate_index': candidate_index
            }
        }


class PropertyEmphasisStrategy(PromptStrategy):
    """Emphasizes different properties for the same base material."""
    
    def __init__(self):
        self.property_emphases = [
            "high strength",
            "flexibility and elasticity",
            "chemical resistance",
            "optical transparency",
            "electrical conductivity",
            "biocompatibility",
            "thermal insulation",
            "adhesive properties",
            "barrier properties",
            "self-healing capabilities"
        ]
    
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate property-emphasized prompt."""
        property_focus = self.property_emphases[candidate_index % len(self.property_emphases)]
        variant_prompt = f"{base_prompt} with emphasis on {property_focus}"
        
        return {
            'prompt': variant_prompt,
            'strategy': 'property_emphasis',
            'original_prompt': base_prompt,
            'variation_details': {
                'property_focus': property_focus,
                'candidate_index': candidate_index
            }
        }


class StructureFocusStrategy(PromptStrategy):
    """Focuses on different structural aspects of the polymer."""
    
    def __init__(self):
        self.structural_focuses = [
            "linear polymer structure",
            "branched polymer architecture",
            "crosslinked network structure",
            "block copolymer design",
            "random copolymer structure",
            "alternating copolymer pattern",
            "star polymer topology",
            "dendritic structure",
            "comb-like polymer structure",
            "ladder polymer design"
        ]
    
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate structure-focused prompt."""
        structure_focus = self.structural_focuses[candidate_index % len(self.structural_focuses)]
        variant_prompt = f"{base_prompt} designed as a {structure_focus}"
        
        return {
            'prompt': variant_prompt,
            'strategy': 'structure_focus',
            'original_prompt': base_prompt,
            'variation_details': {
                'structure_focus': structure_focus,
                'candidate_index': candidate_index
            }
        }


class ApplicationContextStrategy(PromptStrategy):
    """Provides different application contexts for the material."""
    
    def __init__(self):
        self.application_contexts = [
            "biomedical applications",
            "packaging industry",
            "automotive components",
            "aerospace materials",
            "electronics and semiconductors",
            "construction materials",
            "textiles and fibers",
            "energy storage systems",
            "environmental applications",
            "food contact materials"
        ]
    
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate application-contextualized prompt."""
        application = self.application_contexts[candidate_index % len(self.application_contexts)]
        variant_prompt = f"{base_prompt} for use in {application}"
        
        return {
            'prompt': variant_prompt,
            'strategy': 'application_context',
            'original_prompt': base_prompt,
            'variation_details': {
                'application_context': application,
                'candidate_index': candidate_index
            }
        }


class ChemicalModificationStrategy(PromptStrategy):
    """Suggests specific chemical modifications or functional groups."""
    
    def __init__(self):
        self.chemical_modifications = [
            "with hydroxyl functional groups",
            "with amine functionalization",
            "with carboxyl groups",
            "with aromatic rings",
            "with fluorinated segments",
            "with sulfur-containing groups",
            "with ester linkages",
            "with amide bonds",
            "with ether linkages",
            "with vinyl groups"
        ]
    
    def generate_variant(self, base_prompt: str, candidate_index: int) -> Dict[str, Any]:
        """Generate chemically modified prompt."""
        modification = self.chemical_modifications[candidate_index % len(self.chemical_modifications)]
        variant_prompt = f"{base_prompt} {modification}"
        
        return {
            'prompt': variant_prompt,
            'strategy': 'chemical_modification',
            'original_prompt': base_prompt,
            'variation_details': {
                'chemical_modification': modification,
                'candidate_index': candidate_index
            }
        }


class PromptDiversifier:
    """
    Generates diverse prompts for PSMILES candidate generation.
    
    This class uses multiple strategies to create varied prompts that still
    follow the original intent but approach it from different angles.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the prompt diversifier."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.strategies = {
            'perspective_variation': PerspectiveVariationStrategy(),
            'property_emphasis': PropertyEmphasisStrategy(),
            'structure_focus': StructureFocusStrategy(),
            'application_context': ApplicationContextStrategy(),
            'chemical_modification': ChemicalModificationStrategy()
        }
        
        logger.info(f"PromptDiversifier initialized with {len(self.strategies)} strategies")
    
    def generate_diverse_prompts(self, config: CandidateConfig) -> List[Dict[str, Any]]:
        """
        Generate diverse prompts based on the configuration.
        
        Args:
            config: Configuration for candidate generation
            
        Returns:
            List of prompt variants with metadata
        """
        diverse_prompts = []
        
        logger.info(f"Generating {config.num_candidates} diverse prompts for: {config.base_request}")
        
        for i in range(config.num_candidates):
            # Choose strategy for this candidate
            strategy_name = config.prompt_variation_strategies[i % len(config.prompt_variation_strategies)]
            strategy = self.strategies.get(strategy_name)
            
            if strategy is None:
                logger.warning(f"Unknown strategy: {strategy_name}. Using original prompt.")
                prompt_data = {
                    'prompt': config.base_request,
                    'strategy': 'original',
                    'original_prompt': config.base_request,
                    'variation_details': {'fallback': True}
                }
            else:
                prompt_data = strategy.generate_variant(config.base_request, i)
            
            # Add metadata
            prompt_data.update({
                'candidate_id': i + 1,
                'generation_timestamp': datetime.now().isoformat(),
                'diversity_target': True
            })
            
            diverse_prompts.append(prompt_data)
            logger.debug(f"Generated prompt {i+1}: {prompt_data['prompt']}")
        
        logger.info(f"Successfully generated {len(diverse_prompts)} diverse prompts")
        return diverse_prompts


class FunctionalizationEngine:
    """
    Handles chemical functionalization of PSMILES using the psmiles package.
    
    This engine adds random functional groups and modifications to generated
    PSMILES to increase chemical diversity.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the functionalization engine."""
        if random_seed is not None:
            random.seed(random_seed)
        
        self.available_functionalizations = self._setup_functionalizations()
        self._check_psmiles_availability()
        
    def _check_psmiles_availability(self):
        """Check if psmiles package is available."""
        try:
            import psmiles
            self.psmiles_available = True
            logger.info("psmiles package available for functionalization")
        except ImportError:
            self.psmiles_available = False
            logger.warning("psmiles package not available. Install with: pip install git+https://github.com/Ramprasad-Group/psmiles.git")
    
    def _setup_functionalizations(self) -> Dict[str, Dict[str, Any]]:
        """Setup available functionalization methods."""
        return {
            'randomize': {
                'description': 'Apply random structural variations to PSMILES',
                'method': self._apply_randomization,
                'parameters': {'max_iterations': 5}
            },
            'dimerize': {
                'description': 'Create dimeric variations of the PSMILES',
                'method': self._apply_dimerization,
                'parameters': {'connection_types': ['head_to_tail', 'head_to_head']}
            },
            'copolymerize': {
                'description': 'Create copolymer variations with common monomers',
                'method': self._apply_copolymerization,
                'parameters': {'common_monomers': ['[*]CC[*]', '[*]CCO[*]', '[*]c1ccccc1[*]']}
            },
            'substitute': {
                'description': 'Add substituent groups to the polymer backbone',
                'method': self._apply_substitution,
                'parameters': {'substituents': ['F', 'Cl', 'Br', 'OH', 'NH2', 'COOH', 'CH3']}
            }
        }
    
    def _apply_randomization(self, psmiles: str, **kwargs) -> List[str]:
        """Apply randomization using psmiles package."""
        if not self.psmiles_available:
            return [psmiles]  # Return original if psmiles not available
        
        try:
            from psmiles import PolymerSmiles as PS
            
            ps = PS(psmiles)
            randomized_variants = []
            
            max_iterations = kwargs.get('max_iterations', 5)
            for _ in range(max_iterations):
                try:
                    # Use randomize method from psmiles
                    randomized = ps.randomize()  # Fixed: Added parentheses to call the method
                    # Convert to string if it's a PolymerSmiles object
                    if hasattr(randomized, 'psmiles'):
                        randomized = randomized.psmiles
                    elif not isinstance(randomized, str):
                        randomized = str(randomized)
                    
                    if randomized != psmiles:  # Only add if different
                        randomized_variants.append(randomized)
                except Exception as e:
                    logger.debug(f"Randomization attempt failed: {e}")
                    continue
            
            return randomized_variants if randomized_variants else [psmiles]
            
        except Exception as e:
            logger.warning(f"Randomization failed: {e}")
            return [psmiles]
    
    def _apply_dimerization(self, psmiles: str, **kwargs) -> List[str]:
        """Apply dimerization using psmiles package."""
        if not self.psmiles_available:
            return [psmiles]
        
        try:
            from psmiles import PolymerSmiles as PS
            
            ps = PS(psmiles)
            dimers = []
            
            # Try dimerization
            try:
                dimerized = ps.dimerize()
                # Convert to string if it's a PolymerSmiles object
                if hasattr(dimerized, 'psmiles'):
                    dimerized = dimerized.psmiles
                elif not isinstance(dimerized, str):
                    dimerized = str(dimerized)
                
                if dimerized != psmiles:
                    dimers.append(dimerized)
            except Exception as e:
                logger.debug(f"Dimerization failed: {e}")
            
            return dimers if dimers else [psmiles]
            
        except Exception as e:
            logger.warning(f"Dimerization failed: {e}")
            return [psmiles]
    
    def _apply_copolymerization(self, psmiles: str, **kwargs) -> List[str]:
        """Create copolymer variations."""
        if not self.psmiles_available:
            return [psmiles]
        
        try:
            from psmiles import PolymerSmiles as PS
            
            common_monomers = kwargs.get('common_monomers', ['[*]CC[*]', '[*]CCO[*]'])
            copolymers = []
            
            for monomer in common_monomers:
                try:
                    # Create alternating copolymer
                    ps1 = PS(psmiles)
                    ps2 = PS(monomer)
                    
                    # Use the alternating copolymer method if available
                    if hasattr(ps1, 'alternating_copolymer'):
                        copolymer = ps1.alternating_copolymer(ps2)
                        if copolymer and copolymer != psmiles:
                            copolymers.append(str(copolymer))
                except Exception as e:
                    logger.debug(f"Copolymerization with {monomer} failed: {e}")
                    continue
            
            return copolymers if copolymers else [psmiles]
            
        except Exception as e:
            logger.warning(f"Copolymerization failed: {e}")
            return [psmiles]
    
    def _apply_substitution(self, psmiles: str, **kwargs) -> List[str]:
        """Apply simple substitutions to add functional groups."""
        substituents = kwargs.get('substituents', ['F', 'Cl', 'OH', 'CH3'])
        substituted = []
        
        for substituent in substituents:
            # Simple substitution by replacing hydrogen positions
            if 'C' in psmiles and '[*]' in psmiles:
                # Add substituent to carbon atoms
                modified = psmiles.replace('C', f'C({substituent})', 1)
                if modified != psmiles:
                    substituted.append(modified)
        
        return substituted if substituted else [psmiles]
    
    def functionalize_candidate(self, candidate_data: Dict[str, Any], 
                              config: CandidateConfig) -> Dict[str, Any]:
        """
        Apply functionalization to a single candidate.
        
        Args:
            candidate_data: Original candidate data with PSMILES
            config: Configuration for functionalization
            
        Returns:
            Enhanced candidate data with functionalized variants
        """
        if not config.enable_functionalization:
            return candidate_data
        
        original_psmiles = candidate_data.get('psmiles')
        if not original_psmiles:
            logger.warning("No PSMILES found in candidate data")
            return candidate_data
        
        logger.debug(f"Functionalizing candidate: {original_psmiles}")
        
        functionalized_variants = []
        
        # Apply different functionalization methods
        for method_name, method_info in self.available_functionalizations.items():
            try:
                variants = method_info['method'](
                    original_psmiles, 
                    **method_info['parameters']
                )
                
                # Debug: Log what we got from the method
                logger.debug(f"Method {method_name} returned: {type(variants)} - {variants}")
                
                # Ensure variants is iterable and contains strings
                if not hasattr(variants, '__iter__') or isinstance(variants, str):
                    # If it's a single string or non-iterable, wrap in list
                    variants = [str(variants)]
                
                for variant in variants:
                    # Ensure variant is a string
                    if hasattr(variant, 'psmiles'):
                        variant = variant.psmiles
                    elif not isinstance(variant, str):
                        variant = str(variant)
                        
                    if variant != original_psmiles:  # Only add if different
                        functionalized_variants.append({
                            'psmiles': variant,
                            'functionalization_method': method_name,
                            'parent_psmiles': original_psmiles,
                            'method_description': method_info['description']
                        })
                        
            except Exception as e:
                logger.error(f"Functionalization method {method_name} failed: {type(e).__name__}: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Add functionalized variants to candidate data
        enhanced_candidate = candidate_data.copy()
        enhanced_candidate['functionalized_variants'] = functionalized_variants
        enhanced_candidate['num_variants'] = len(functionalized_variants)
        
        logger.debug(f"Generated {len(functionalized_variants)} functionalized variants")
        
        return enhanced_candidate


class DiversityValidator:
    """
    Validates and ensures diversity among generated candidates.
    
    This class checks for duplicate PSMILES and calculates diversity metrics
    to ensure candidates are truly unique.
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        """Initialize diversity validator."""
        self.similarity_threshold = similarity_threshold
        self._check_molecular_tools()
    
    def _check_molecular_tools(self):
        """Check availability of molecular similarity tools."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            self.rdkit_available = True
            logger.info("RDKit available for similarity calculations")
        except ImportError:
            self.rdkit_available = False
            logger.warning("RDKit not available. Using basic string similarity.")
    
    def calculate_psmiles_similarity(self, psmiles1: str, psmiles2: str) -> float:
        """Calculate similarity between two PSMILES strings."""
        if psmiles1 == psmiles2:
            return 1.0
        
        # Basic string similarity using Jaccard similarity
        set1 = set(psmiles1)
        set2 = set(psmiles2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def validate_diversity(self, candidates: List[Dict[str, Any]], 
                          config: CandidateConfig) -> Dict[str, Any]:
        """
        Validate diversity among candidates.
        
        Args:
            candidates: List of candidate data
            config: Configuration with diversity settings
            
        Returns:
            Diversity validation results
        """
        logger.info(f"Validating diversity among {len(candidates)} candidates")
        
        psmiles_list = []
        for candidate in candidates:
            psmiles = candidate.get('psmiles')
            if psmiles:
                psmiles_list.append(psmiles)
        
        unique_psmiles = set(psmiles_list)
        duplicate_count = len(psmiles_list) - len(unique_psmiles)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(psmiles_list)):
            for j in range(i + 1, len(psmiles_list)):
                sim = self.calculate_psmiles_similarity(psmiles_list[i], psmiles_list[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        min_similarity = np.min(similarities) if similarities else 0.0
        max_similarity = np.max(similarities) if similarities else 0.0
        
        diversity_score = 1.0 - avg_similarity
        meets_threshold = diversity_score >= config.diversity_threshold
        
        validation_results = {
            'total_candidates': len(candidates),
            'unique_psmiles': len(unique_psmiles),
            'duplicate_count': duplicate_count,
            'diversity_score': diversity_score,
            'average_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'meets_diversity_threshold': meets_threshold,
            'diversity_threshold': config.diversity_threshold,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Diversity validation complete. Score: {diversity_score:.3f}, "
                   f"Threshold: {config.diversity_threshold}, "
                   f"Meets threshold: {meets_threshold}")
        
        return validation_results


class CandidateOrchestrator:
    """
    Main orchestrator for diverse PSMILES candidate generation.
    
    This class coordinates all components to generate truly diverse PSMILES candidates
    by combining prompt diversification, PSMILES generation, and functionalization.
    """
    
    def __init__(self, psmiles_generator, random_seed: Optional[int] = None):
        """
        Initialize the candidate orchestrator.
        
        Args:
            psmiles_generator: The PSMILESGenerator instance to use
            random_seed: Random seed for reproducibility
        """
        self.psmiles_generator = psmiles_generator
        self.prompt_diversifier = PromptDiversifier(random_seed)
        self.functionalization_engine = FunctionalizationEngine(random_seed)
        self.diversity_validator = DiversityValidator()
        
        logger.info("CandidateOrchestrator initialized with all components")
    
    def generate_diverse_candidates(self, config: CandidateConfig) -> Dict[str, Any]:
        """
        Generate diverse PSMILES candidates using the complete workflow.
        
        Args:
            config: Configuration for candidate generation
            
        Returns:
            Complete results with diverse candidates, validation, and metadata
        """
        logger.info(f"🚀 Starting diverse candidate generation for: {config.base_request}")
        start_time = datetime.now()
        
        try:
            # Step 1: Generate diverse prompts
            logger.info("📝 Step 1: Generating diverse prompts")
            diverse_prompts = self.prompt_diversifier.generate_diverse_prompts(config)
            
            # Step 2: Generate PSMILES for each prompt
            logger.info("🧬 Step 2: Generating PSMILES from diverse prompts")
            candidates = []
            generation_stats = {
                'total_attempts': 0,
                'successful_generations': 0,
                'failed_generations': 0,
                'validation_failures': 0
            }
            
            for i, prompt_data in enumerate(diverse_prompts):
                logger.info(f"   Generating candidate {i+1}/{len(diverse_prompts)}")
                generation_stats['total_attempts'] += 1
                
                try:
                    # Generate PSMILES using the diverse prompt
                    result = self.psmiles_generator.generate_psmiles(
                        description=prompt_data['prompt'],
                        num_candidates=1,
                        validate=True,
                        disable_early_chemistry_lookup=True  # Enable diversity by preventing early chemistry lookup stops
                    )
                    
                    if result.get('success') and result.get('candidates'):
                        candidate_data = result['candidates'][0]
                        
                        # Add prompt information to candidate
                        candidate_data.update({
                            'prompt_data': prompt_data,
                            'generation_method': 'diverse_orchestrated',
                            'candidate_index': i + 1
                        })
                        
                        candidates.append(candidate_data)
                        generation_stats['successful_generations'] += 1
                        
                        logger.info(f"   ✅ Generated: {candidate_data.get('psmiles', 'N/A')}")
                    else:
                        logger.warning(f"   ❌ Generation failed: {result.get('error', 'Unknown error')}")
                        generation_stats['failed_generations'] += 1
                
                except Exception as e:
                    logger.error(f"   ❌ Exception during generation: {e}")
                    generation_stats['failed_generations'] += 1
                    continue
            
            # Step 3: Apply functionalization to increase diversity
            if config.enable_functionalization and candidates:
                logger.info("🔬 Step 3: Applying functionalization")
                functionalized_candidates = []
                
                for candidate in candidates:
                    enhanced_candidate = self.functionalization_engine.functionalize_candidate(
                        candidate, config
                    )
                    functionalized_candidates.append(enhanced_candidate)
                    
                    # Add functionalized variants as separate candidates
                    variants = enhanced_candidate.get('functionalized_variants', [])
                    for variant in variants:
                        variant_candidate = enhanced_candidate.copy()
                        variant_candidate.update({
                            'psmiles': variant['psmiles'],
                            'is_functionalized_variant': True,
                            'parent_candidate_index': enhanced_candidate.get('candidate_index'),
                            'functionalization_method': variant['functionalization_method'],
                            'method_description': variant['method_description']
                        })
                        functionalized_candidates.append(variant_candidate)
                
                candidates = functionalized_candidates
                logger.info(f"   Generated {len(candidates)} total candidates (including variants)")
            
            # Step 4: Validate diversity
            logger.info("📊 Step 4: Validating diversity")
            diversity_results = self.diversity_validator.validate_diversity(candidates, config)
            
            # Step 5: Select best candidates if we have too many
            final_candidates = self._select_best_candidates(candidates, config, diversity_results)
            
            # Step 6: Prepare final results
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results = {
                'success': len(final_candidates) > 0,
                'base_request': config.base_request,
                'candidates': final_candidates,
                'num_generated': len(final_candidates),
                'diverse_prompts': diverse_prompts,
                'generation_stats': generation_stats,
                'diversity_validation': diversity_results,
                'processing_time_seconds': processing_time,
                'generation_method': 'diverse_orchestrated_pipeline',
                'config': {
                    'num_candidates_requested': config.num_candidates,
                    'enable_functionalization': config.enable_functionalization,
                    'diversity_threshold': config.diversity_threshold,
                    'temperature_range': config.temperature_range
                },
                'timestamp': end_time.isoformat()
            }
            
            logger.info(f"🎉 Diverse candidate generation complete!")
            logger.info(f"   Generated: {len(final_candidates)} final candidates")
            logger.info(f"   Diversity score: {diversity_results.get('diversity_score', 0):.3f}")
            logger.info(f"   Processing time: {processing_time:.1f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Diverse candidate generation failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'base_request': config.base_request,
                'generation_method': 'diverse_orchestrated_pipeline',
                'timestamp': datetime.now().isoformat(),
                'debug_info': {
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
            }
    
    def _select_best_candidates(self, candidates: List[Dict[str, Any]], 
                               config: CandidateConfig,
                               diversity_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select the best candidates based on diversity and validity.
        
        Args:
            candidates: All generated candidates
            config: Configuration settings
            diversity_results: Results from diversity validation
            
        Returns:
            Selected best candidates
        """
        if len(candidates) <= config.num_candidates:
            return candidates
        
        logger.info(f"Selecting best {config.num_candidates} from {len(candidates)} candidates")
        
        # Filter valid candidates first
        valid_candidates = [c for c in candidates if c.get('valid', False)]
        if not valid_candidates:
            logger.warning("No valid candidates found, using all candidates")
            valid_candidates = candidates
        
        # Prioritize by multiple criteria
        def candidate_score(candidate):
            score = 0.0
            
            # Validity bonus
            if candidate.get('valid', False):
                score += 1.0
            
            # Method bonus (prefer chemistry lookup)
            method = candidate.get('method', '')
            if method == 'chemistry_lookup':
                score += 0.5
            elif method == 'openai_comprehensive':
                score += 0.3
            
            # Confidence bonus
            confidence = candidate.get('confidence', 0.0)
            score += confidence * 0.2
            
            # Functionalization bonus for diversity
            if candidate.get('is_functionalized_variant', False):
                score += 0.1
            
            return score
        
        # Sort by score and take top candidates
        scored_candidates = [(candidate_score(c), c) for c in valid_candidates]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        selected_candidates = [c[1] for c in scored_candidates[:config.num_candidates]]
        
        logger.info(f"Selected {len(selected_candidates)} best candidates")
        return selected_candidates
    
    def generate_with_retry(self, config: CandidateConfig, 
                           max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate diverse candidates with retry logic for improved success rate.
        
        Args:
            config: Configuration for candidate generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Results from generation with retry metadata
        """
        logger.info(f"Starting generation with retry (max_retries: {max_retries})")
        
        best_result = None
        retry_history = []
        
        for attempt in range(max_retries + 1):
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                result = self.generate_diverse_candidates(config)
                retry_history.append({
                    'attempt': attempt + 1,
                    'success': result.get('success', False),
                    'num_candidates': len(result.get('candidates', [])),
                    'diversity_score': result.get('diversity_validation', {}).get('diversity_score', 0.0)
                })
                
                if result.get('success'):
                    diversity_score = result.get('diversity_validation', {}).get('diversity_score', 0.0)
                    
                    if best_result is None or diversity_score > best_result.get('diversity_validation', {}).get('diversity_score', 0.0):
                        best_result = result.copy()
                        best_result['retry_metadata'] = {
                            'successful_attempt': attempt + 1,
                            'total_attempts': attempt + 1,
                            'retry_history': retry_history.copy()
                        }
                    
                    # Check if we meet the diversity threshold
                    if diversity_score >= config.diversity_threshold:
                        logger.info(f"✅ Diversity threshold met on attempt {attempt + 1}")
                        break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                retry_history.append({
                    'attempt': attempt + 1,
                    'success': False,
                    'error': str(e)
                })
        
        if best_result:
            best_result['retry_metadata']['total_attempts'] = len(retry_history)
            best_result['retry_metadata']['retry_history'] = retry_history
            logger.info(f"Best result achieved diversity score: {best_result.get('diversity_validation', {}).get('diversity_score', 0.0):.3f}")
        else:
            logger.error("All retry attempts failed")
            best_result = {
                'success': False,
                'error': 'All retry attempts failed',
                'retry_metadata': {
                    'total_attempts': len(retry_history),
                    'retry_history': retry_history
                }
            }
        
        return best_result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configure candidate generation
    config = CandidateConfig(
        base_request="polymer for drug delivery",
        num_candidates=5,
        enable_functionalization=True,
        diversity_threshold=0.4
    )
    
    # Generate diverse prompts
    prompt_diversifier = PromptDiversifier(random_seed=42)
    diverse_prompts = prompt_diversifier.generate_diverse_prompts(config)
    
    # Print results
    print("\n=== Generated Diverse Prompts ===")
    for i, prompt_data in enumerate(diverse_prompts):
        print(f"\nCandidate {i+1}:")
        print(f"Strategy: {prompt_data['strategy']}")
        print(f"Prompt: {prompt_data['prompt']}")
    
    # Test functionalization engine
    print("\n=== Testing Functionalization Engine ===")
    func_engine = FunctionalizationEngine()
    
    # Mock candidate data
    mock_candidate = {
        'psmiles': '[*]CC(C)[*]',
        'valid': True,
        'method': 'test'
    }
    
    enhanced = func_engine.functionalize_candidate(mock_candidate, config)
    print(f"Original PSMILES: {mock_candidate['psmiles']}")
    print(f"Generated {enhanced.get('num_variants', 0)} functionalized variants")
    
    for variant in enhanced.get('functionalized_variants', []):
        print(f"  - {variant['psmiles']} (method: {variant['functionalization_method']})") 