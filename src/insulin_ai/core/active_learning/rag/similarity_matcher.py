#!/usr/bin/env python3
"""
Similarity Matcher Implementation for RAG System - Phase 3

This module provides sophisticated similarity matching for finding analogous materials
based on composition, structure, properties, and performance characteristics.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import math

# Scientific computing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available")

# Chemical similarity
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Fingerprints
    from rdkit.DataStructs import TanimotoSimilarity
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available for chemical similarity")

# Import database components
from .property_database import PropertyDatabase, Material, MaterialProperty, PropertyType
from .vector_database import VectorDatabase, MaterialDocument, SearchResult

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Types of similarity metrics."""
    COMPOSITION = "composition"
    STRUCTURE = "structure"
    PROPERTY = "property"
    PERFORMANCE = "performance"
    SEMANTIC = "semantic"
    FINGERPRINT = "fingerprint"
    COMBINED = "combined"


@dataclass
class SimilarityResult:
    """Result of similarity matching."""
    target_material: Material
    similar_material: Material
    similarity_score: float
    metric_type: SimilarityMetric
    detailed_scores: Dict[str, float]
    confidence: float
    explanation: str


@dataclass
class SimilarityConfig:
    """Configuration for similarity matching."""
    metrics: List[SimilarityMetric]
    weights: Dict[SimilarityMetric, float]
    thresholds: Dict[SimilarityMetric, float]
    max_results: int = 10
    min_overall_score: float = 0.3
    property_importance: Dict[str, float] = None


class CompositionSimilarity:
    """Calculate similarity based on material composition."""
    
    @staticmethod
    def calculate(comp1: Dict[str, Any], comp2: Dict[str, Any]) -> float:
        """Calculate composition similarity using Jaccard and weighted overlap."""
        if not comp1 or not comp2:
            return 0.0
        
        # Extract elements and their fractions
        elements1 = set(comp1.keys())
        elements2 = set(comp2.keys())
        
        # Jaccard similarity for elements
        intersection = elements1.intersection(elements2)
        union = elements1.union(elements2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Weighted similarity for common elements
        weighted_sim = 0.0
        if intersection:
            total_weight = 0.0
            for element in intersection:
                try:
                    frac1 = float(comp1.get(element, 0))
                    frac2 = float(comp2.get(element, 0))
                    weight = min(frac1, frac2)
                    similarity = 1.0 - abs(frac1 - frac2) / max(frac1 + frac2, 1e-6)
                    weighted_sim += weight * similarity
                    total_weight += weight
                except (ValueError, TypeError):
                    continue
            
            if total_weight > 0:
                weighted_sim /= total_weight
        
        # Combine Jaccard and weighted similarity
        return 0.5 * jaccard + 0.5 * weighted_sim


class PropertySimilarity:
    """Calculate similarity based on material properties."""
    
    @staticmethod
    def calculate(props1: List[MaterialProperty], 
                 props2: List[MaterialProperty],
                 importance_weights: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate property-based similarity."""
        if not props1 or not props2:
            return {"overall": 0.0}
        
        # Create property dictionaries
        prop_dict1 = {}
        prop_dict2 = {}
        
        for prop in props1:
            try:
                prop_dict1[prop.property_name] = float(prop.value)
            except (ValueError, TypeError):
                continue
        
        for prop in props2:
            try:
                prop_dict2[prop.property_name] = float(prop.value)
            except (ValueError, TypeError):
                continue
        
        # Find common properties
        common_props = set(prop_dict1.keys()).intersection(set(prop_dict2.keys()))
        
        if not common_props:
            return {"overall": 0.0}
        
        # Calculate similarity for each property
        property_similarities = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for prop_name in common_props:
            val1 = prop_dict1[prop_name]
            val2 = prop_dict2[prop_name]
            
            # Calculate relative similarity (1 - relative error)
            max_val = max(abs(val1), abs(val2), 1e-6)
            similarity = 1.0 - abs(val1 - val2) / max_val
            
            property_similarities[prop_name] = similarity
            
            # Apply importance weight
            weight = 1.0
            if importance_weights and prop_name in importance_weights:
                weight = importance_weights[prop_name]
            
            weighted_sum += weight * similarity
            total_weight += weight
        
        # Overall similarity
        overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        property_similarities["overall"] = overall_similarity
        
        return property_similarities


class StructureSimilarity:
    """Calculate similarity based on molecular structure."""
    
    @staticmethod
    def calculate_smiles_similarity(smiles1: str, smiles2: str) -> float:
        """Calculate structural similarity using SMILES."""
        if not RDKIT_AVAILABLE:
            return 0.0
        
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            # Calculate Morgan fingerprints
            fp1 = Fingerprints.FingerprintMols.FingerprintMol(mol1)
            fp2 = Fingerprints.FingerprintMols.FingerprintMol(mol2)
            
            # Tanimoto similarity
            return TanimotoSimilarity(fp1, fp2)
            
        except Exception as e:
            logger.warning(f"Failed to calculate SMILES similarity: {e}")
            return 0.0
    
    @staticmethod
    def calculate_descriptor_similarity(smiles1: str, smiles2: str) -> float:
        """Calculate similarity using molecular descriptors."""
        if not RDKIT_AVAILABLE:
            return 0.0
        
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            # Calculate key descriptors
            descriptors = [
                Descriptors.MolWt,
                Descriptors.MolLogP,
                Descriptors.TPSA,
                Descriptors.NumHDonors,
                Descriptors.NumHAcceptors,
                Descriptors.NumRotatableBonds
            ]
            
            desc1 = [desc(mol1) for desc in descriptors]
            desc2 = [desc(mol2) for desc in descriptors]
            
            # Calculate normalized similarity
            similarities = []
            for d1, d2 in zip(desc1, desc2):
                if d1 == 0 and d2 == 0:
                    similarities.append(1.0)
                else:
                    max_val = max(abs(d1), abs(d2), 1e-6)
                    sim = 1.0 - abs(d1 - d2) / max_val
                    similarities.append(sim)
            
            return sum(similarities) / len(similarities)
            
        except Exception as e:
            logger.warning(f"Failed to calculate descriptor similarity: {e}")
            return 0.0


class SemanticSimilarity:
    """Calculate semantic similarity using vector embeddings."""
    
    def __init__(self, vector_db: VectorDatabase):
        """Initialize with vector database."""
        self.vector_db = vector_db
    
    async def calculate(self, material1_desc: str, material2_desc: str) -> float:
        """Calculate semantic similarity between material descriptions."""
        if not self.vector_db or not self.vector_db.embedding_model.embeddings:
            return 0.0
        
        try:
            # Get embeddings
            embedding1 = self.vector_db.embedding_model.embed_text(material1_desc)
            embedding2 = self.vector_db.embedding_model.embed_text(material2_desc)
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # Calculate cosine similarity
            if NUMPY_AVAILABLE:
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                # Normalize vectors
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                return max(0.0, cosine_sim)  # Ensure non-negative
            else:
                # Manual cosine similarity calculation
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                norm1 = math.sqrt(sum(a * a for a in embedding1))
                norm2 = math.sqrt(sum(b * b for b in embedding2))
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                cosine_sim = dot_product / (norm1 * norm2)
                return max(0.0, cosine_sim)
            
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0


class SimilarityMatcher:
    """Advanced similarity matcher for materials discovery."""
    
    def __init__(self, 
                 property_db: PropertyDatabase,
                 vector_db: Optional[VectorDatabase] = None,
                 default_config: Optional[SimilarityConfig] = None):
        """
        Initialize similarity matcher.
        
        Args:
            property_db: Property database for material data
            vector_db: Vector database for semantic similarity
            default_config: Default configuration for similarity matching
        """
        self.property_db = property_db
        self.vector_db = vector_db
        
        # Initialize similarity calculators
        self.composition_sim = CompositionSimilarity()
        self.property_sim = PropertySimilarity()
        self.structure_sim = StructureSimilarity()
        self.semantic_sim = SemanticSimilarity(vector_db) if vector_db else None
        
        # Default configuration
        self.default_config = default_config or SimilarityConfig(
            metrics=[SimilarityMetric.COMPOSITION, SimilarityMetric.PROPERTY, SimilarityMetric.SEMANTIC],
            weights={
                SimilarityMetric.COMPOSITION: 0.3,
                SimilarityMetric.PROPERTY: 0.4,
                SimilarityMetric.SEMANTIC: 0.3
            },
            thresholds={
                SimilarityMetric.COMPOSITION: 0.2,
                SimilarityMetric.PROPERTY: 0.3,
                SimilarityMetric.SEMANTIC: 0.2
            },
            max_results=10,
            min_overall_score=0.3
        )
        
        # Cache for expensive calculations
        self._similarity_cache = {}
        
        logger.info("SimilarityMatcher initialized")
    
    async def find_similar_materials(self, 
                                   target_material: Material,
                                   config: Optional[SimilarityConfig] = None) -> List[SimilarityResult]:
        """Find materials similar to the target material."""
        config = config or self.default_config
        
        try:
            # Get all materials from database
            similar_results = []
            
            # Get candidate materials (this could be optimized with indices)
            import sqlite3
            with sqlite3.connect(self.property_db.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT material_id FROM materials WHERE material_id != ?", 
                             (target_material.material_id,))
                candidate_ids = [row[0] for row in cursor.fetchall()]
            
            # Calculate similarity for each candidate
            for candidate_id in candidate_ids:
                candidate_material = await self.property_db.get_material(candidate_id)
                if not candidate_material:
                    continue
                
                similarity_result = await self._calculate_material_similarity(
                    target_material, candidate_material, config
                )
                
                if similarity_result and similarity_result.similarity_score >= config.min_overall_score:
                    similar_results.append(similarity_result)
            
            # Sort by similarity score
            similar_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return similar_results[:config.max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar materials: {e}")
            return []
    
    async def _calculate_material_similarity(self, 
                                           material1: Material,
                                           material2: Material,
                                           config: SimilarityConfig) -> Optional[SimilarityResult]:
        """Calculate comprehensive similarity between two materials."""
        try:
            detailed_scores = {}
            weighted_sum = 0.0
            total_weight = 0.0
            
            # Calculate each requested metric
            for metric in config.metrics:
                score = 0.0
                
                if metric == SimilarityMetric.COMPOSITION:
                    score = self.composition_sim.calculate(
                        material1.composition, material2.composition
                    )
                    detailed_scores["composition"] = score
                
                elif metric == SimilarityMetric.PROPERTY:
                    prop_scores = self.property_sim.calculate(
                        material1.properties, material2.properties,
                        config.property_importance
                    )
                    score = prop_scores.get("overall", 0.0)
                    detailed_scores.update({f"property_{k}": v for k, v in prop_scores.items()})
                
                elif metric == SimilarityMetric.STRUCTURE:
                    if (hasattr(material1, 'structure') and hasattr(material2, 'structure') and
                        material1.structure and material2.structure):
                        score = self.structure_sim.calculate_smiles_similarity(
                            material1.structure, material2.structure
                        )
                    detailed_scores["structure"] = score
                
                elif metric == SimilarityMetric.SEMANTIC and self.semantic_sim:
                    desc1 = self._create_material_description(material1)
                    desc2 = self._create_material_description(material2)
                    score = await self.semantic_sim.calculate(desc1, desc2)
                    detailed_scores["semantic"] = score
                
                # Apply weight and threshold
                if score >= config.thresholds.get(metric, 0.0):
                    weight = config.weights.get(metric, 1.0)
                    weighted_sum += weight * score
                    total_weight += weight
            
            # Calculate overall similarity
            overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Generate explanation
            explanation = self._generate_similarity_explanation(detailed_scores, config.metrics)
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(material1, material2, detailed_scores)
            
            return SimilarityResult(
                target_material=material1,
                similar_material=material2,
                similarity_score=overall_similarity,
                metric_type=SimilarityMetric.COMBINED,
                detailed_scores=detailed_scores,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate material similarity: {e}")
            return None
    
    def _create_material_description(self, material: Material) -> str:
        """Create a textual description of the material for semantic similarity."""
        description_parts = [
            f"Material: {material.name}",
            f"Composition: {', '.join(f'{k}: {v}' for k, v in material.composition.items())}"
        ]
        
        if material.synthesis_method:
            description_parts.append(f"Synthesis: {material.synthesis_method}")
        
        # Add key properties
        key_properties = []
        for prop in material.properties:
            if prop.property_name in ["young_modulus", "tensile_strength", "glass_transition_temp", 
                                    "degradation_rate", "biocompatibility"]:
                key_properties.append(f"{prop.property_name}: {prop.value} {prop.unit}")
        
        if key_properties:
            description_parts.append(f"Properties: {', '.join(key_properties)}")
        
        return ". ".join(description_parts)
    
    def _generate_similarity_explanation(self, 
                                       scores: Dict[str, float],
                                       metrics: List[SimilarityMetric]) -> str:
        """Generate human-readable explanation of similarity."""
        explanations = []
        
        if "composition" in scores:
            comp_score = scores["composition"]
            if comp_score > 0.8:
                explanations.append("very similar composition")
            elif comp_score > 0.6:
                explanations.append("similar composition")
            elif comp_score > 0.3:
                explanations.append("somewhat similar composition")
        
        if "property_overall" in scores:
            prop_score = scores["property_overall"]
            if prop_score > 0.8:
                explanations.append("very similar properties")
            elif prop_score > 0.6:
                explanations.append("similar properties")
            elif prop_score > 0.3:
                explanations.append("somewhat similar properties")
        
        if "semantic" in scores:
            sem_score = scores["semantic"]
            if sem_score > 0.8:
                explanations.append("very similar functionality")
            elif sem_score > 0.6:
                explanations.append("similar functionality")
        
        if not explanations:
            return "Low overall similarity"
        
        return f"Materials have {' and '.join(explanations)}"
    
    def _calculate_confidence(self, 
                            material1: Material,
                            material2: Material,
                            scores: Dict[str, float]) -> float:
        """Calculate confidence in similarity assessment."""
        confidence_factors = []
        
        # Data completeness
        prop_count1 = len(material1.properties) if material1.properties else 0
        prop_count2 = len(material2.properties) if material2.properties else 0
        data_completeness = min(prop_count1, prop_count2) / max(prop_count1, prop_count2, 1)
        confidence_factors.append(data_completeness)
        
        # Score consistency
        score_values = [v for v in scores.values() if isinstance(v, (int, float))]
        if score_values:
            score_std = np.std(score_values) if NUMPY_AVAILABLE else 0.0
            consistency = 1.0 - min(score_std, 0.5) / 0.5
            confidence_factors.append(consistency)
        
        # Overall confidence
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    async def find_property_analogs(self, 
                                  target_properties: Dict[str, float],
                                  tolerance: Dict[str, float] = None,
                                  max_results: int = 10) -> List[SimilarityResult]:
        """Find materials with analogous properties."""
        try:
            # Use property database to find similar materials
            similar_materials = await self.property_db.find_similar_materials(
                target_properties, 
                tolerance=0.2,  # Default tolerance
                max_results=max_results * 2  # Get more for filtering
            )
            
            results = []
            for material, similarity_score in similar_materials:
                # Create a pseudo-target material for comparison
                target_material = Material(
                    material_id="target",
                    name="Target Material",
                    composition={"target": 1.0},
                    properties=[
                        MaterialProperty(
                            material_id="target",
                            property_name=prop_name,
                            property_type=PropertyType.MECHANICAL,  # Default type
                            value=prop_value,
                            unit="unknown"
                        )
                        for prop_name, prop_value in target_properties.items()
                    ]
                )
                
                result = SimilarityResult(
                    target_material=target_material,
                    similar_material=material,
                    similarity_score=similarity_score,
                    metric_type=SimilarityMetric.PROPERTY,
                    detailed_scores={"property_overall": similarity_score},
                    confidence=0.8,  # High confidence for property-based matching
                    explanation=f"Similar property values within tolerance"
                )
                results.append(result)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find property analogs: {e}")
            return []


# Factory function for easy initialization
def create_similarity_matcher(property_db: PropertyDatabase,
                            vector_db: Optional[VectorDatabase] = None) -> SimilarityMatcher:
    """Create similarity matcher with default configuration."""
    return SimilarityMatcher(property_db, vector_db) 