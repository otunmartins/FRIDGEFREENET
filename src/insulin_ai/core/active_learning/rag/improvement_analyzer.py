#!/usr/bin/env python3
"""
Improvement Analyzer Implementation for RAG System - Phase 3

This module analyzes structure-property relationships and generates specific
improvement suggestions for material discovery based on performance gaps.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import statistics

# Scientific analysis
try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available for advanced statistics")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available for data analysis")

# LLM integration
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for LLM analysis")

# Import database components
from .property_database import PropertyDatabase, Material, MaterialProperty, PropertyType, BenchmarkData
from .similarity_matcher import SimilarityMatcher, SimilarityResult

logger = logging.getLogger(__name__)


class ImprovementType(Enum):
    """Types of material improvements."""
    COMPOSITION_MODIFICATION = "composition_modification"
    STRUCTURAL_CHANGE = "structural_change"
    SYNTHESIS_OPTIMIZATION = "synthesis_optimization"
    PROCESSING_CONDITION = "processing_condition"
    ADDITIVE_INCLUSION = "additive_inclusion"
    MOLECULAR_WEIGHT_ADJUSTMENT = "molecular_weight_adjustment"
    CROSSLINKING_MODIFICATION = "crosslinking_modification"


class AnalysisMethod(Enum):
    """Methods for structure-property analysis."""
    STATISTICAL_CORRELATION = "statistical_correlation"
    MACHINE_LEARNING = "machine_learning"
    LITERATURE_BASED = "literature_based"
    EXPERT_RULES = "expert_rules"
    LLM_ANALYSIS = "llm_analysis"


@dataclass
class PropertyGap:
    """Represents a gap between current and target property values."""
    property_name: str
    current_value: float
    target_value: float
    gap_magnitude: float
    gap_percentage: float
    importance: float
    unit: str


@dataclass
class StructurePropertyRelation:
    """Represents a relationship between structure and property."""
    structural_feature: str
    property_name: str
    correlation_strength: float
    correlation_type: str  # "positive", "negative", "nonlinear"
    confidence: float
    evidence_count: int
    analysis_method: AnalysisMethod


@dataclass
class ImprovementSuggestion:
    """Specific suggestion for material improvement."""
    suggestion_id: str
    improvement_type: ImprovementType
    description: str
    specific_changes: Dict[str, Any]
    expected_benefits: Dict[str, float]
    implementation_difficulty: float  # 0-1 scale
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    estimated_cost: Optional[str] = None
    time_to_implement: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete analysis result with gaps, relations, and suggestions."""
    material_id: str
    property_gaps: List[PropertyGap]
    structure_property_relations: List[StructurePropertyRelation]
    improvement_suggestions: List[ImprovementSuggestion]
    overall_score: float
    analysis_confidence: float
    analysis_timestamp: datetime


class PropertyGapAnalyzer:
    """Analyzes gaps between current and target properties."""
    
    def __init__(self, property_db: PropertyDatabase):
        """Initialize with property database."""
        self.property_db = property_db
    
    async def identify_gaps(self, 
                          material: Material,
                          target_properties: Dict[str, float],
                          importance_weights: Dict[str, float] = None) -> List[PropertyGap]:
        """Identify gaps between current and target properties."""
        gaps = []
        
        # Create property dictionary from material
        current_props = {}
        for prop in material.properties:
            try:
                current_props[prop.property_name] = float(prop.value)
            except (ValueError, TypeError):
                continue
        
        # Calculate gaps for each target property
        for prop_name, target_value in target_properties.items():
            if prop_name in current_props:
                current_value = current_props[prop_name]
                gap_magnitude = abs(target_value - current_value)
                gap_percentage = gap_magnitude / max(abs(target_value), 1e-6) * 100
                
                # Determine importance
                importance = 1.0
                if importance_weights and prop_name in importance_weights:
                    importance = importance_weights[prop_name]
                
                # Get unit from material property
                unit = "unknown"
                for prop in material.properties:
                    if prop.property_name == prop_name:
                        unit = prop.unit
                        break
                
                gap = PropertyGap(
                    property_name=prop_name,
                    current_value=current_value,
                    target_value=target_value,
                    gap_magnitude=gap_magnitude,
                    gap_percentage=gap_percentage,
                    importance=importance,
                    unit=unit
                )
                gaps.append(gap)
        
        # Sort by importance and gap magnitude
        gaps.sort(key=lambda x: x.importance * x.gap_magnitude, reverse=True)
        
        return gaps


class StructurePropertyAnalyzer:
    """Analyzes relationships between structure and properties."""
    
    def __init__(self, property_db: PropertyDatabase, similarity_matcher: SimilarityMatcher):
        """Initialize with databases."""
        self.property_db = property_db
        self.similarity_matcher = similarity_matcher
    
    async def analyze_relationships(self, 
                                  materials: List[Material],
                                  target_properties: List[str]) -> List[StructurePropertyRelation]:
        """Analyze structure-property relationships from material dataset."""
        relationships = []
        
        if not SCIPY_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("Advanced statistical analysis not available")
            return self._basic_relationship_analysis(materials, target_properties)
        
        try:
            # Create dataset for analysis
            data = []
            for material in materials:
                material_data = {
                    'material_id': material.material_id,
                    'composition': material.composition,
                    'structure': material.structure,
                    'synthesis_method': material.synthesis_method
                }
                
                # Add properties
                for prop in material.properties:
                    try:
                        material_data[f"prop_{prop.property_name}"] = float(prop.value)
                    except (ValueError, TypeError):
                        continue
                
                data.append(material_data)
            
            if not data:
                return relationships
            
            df = pd.DataFrame(data)
            
            # Analyze composition relationships
            relationships.extend(await self._analyze_composition_relationships(df, target_properties))
            
            # Analyze synthesis method relationships
            relationships.extend(await self._analyze_synthesis_relationships(df, target_properties))
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to analyze relationships: {e}")
            return []
    
    async def _analyze_composition_relationships(self, 
                                               df: pd.DataFrame,
                                               target_properties: List[str]) -> List[StructurePropertyRelation]:
        """Analyze relationships between composition and properties."""
        relationships = []
        
        try:
            # Extract composition features
            all_elements = set()
            for _, row in df.iterrows():
                if 'composition' in row and row['composition']:
                    if isinstance(row['composition'], str):
                        comp = json.loads(row['composition'])
                    else:
                        comp = row['composition']
                    all_elements.update(comp.keys())
            
            # Create composition matrix
            for element in all_elements:
                element_fractions = []
                property_values = {prop: [] for prop in target_properties}
                
                for _, row in df.iterrows():
                    if 'composition' in row and row['composition']:
                        if isinstance(row['composition'], str):
                            comp = json.loads(row['composition'])
                        else:
                            comp = row['composition']
                        
                        element_fraction = comp.get(element, 0.0)
                        element_fractions.append(float(element_fraction))
                        
                        # Get property values
                        for prop in target_properties:
                            prop_col = f"prop_{prop}"
                            if prop_col in row and pd.notna(row[prop_col]):
                                property_values[prop].append(float(row[prop_col]))
                            else:
                                property_values[prop].append(None)
                
                # Calculate correlations
                for prop in target_properties:
                    prop_vals = property_values[prop]
                    
                    # Filter out None values
                    valid_pairs = [(ef, pv) for ef, pv in zip(element_fractions, prop_vals) 
                                 if pv is not None]
                    
                    if len(valid_pairs) >= 5:  # Minimum samples for correlation
                        element_vals, prop_vals = zip(*valid_pairs)
                        
                        correlation, p_value = stats.pearsonr(element_vals, prop_vals)
                        
                        if abs(correlation) > 0.3 and p_value < 0.05:  # Significant correlation
                            relation = StructurePropertyRelation(
                                structural_feature=f"{element}_content",
                                property_name=prop,
                                correlation_strength=abs(correlation),
                                correlation_type="positive" if correlation > 0 else "negative",
                                confidence=1.0 - p_value,
                                evidence_count=len(valid_pairs),
                                analysis_method=AnalysisMethod.STATISTICAL_CORRELATION
                            )
                            relationships.append(relation)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to analyze composition relationships: {e}")
            return []
    
    async def _analyze_synthesis_relationships(self, 
                                             df: pd.DataFrame,
                                             target_properties: List[str]) -> List[StructurePropertyRelation]:
        """Analyze relationships between synthesis methods and properties."""
        relationships = []
        
        try:
            # Group by synthesis method
            synthesis_groups = df.groupby('synthesis_method')
            
            for prop in target_properties:
                prop_col = f"prop_{prop}"
                if prop_col not in df.columns:
                    continue
                
                # Calculate property means for each synthesis method
                method_means = {}
                for method, group in synthesis_groups:
                    prop_values = group[prop_col].dropna()
                    if len(prop_values) >= 3:  # Minimum samples
                        method_means[method] = prop_values.mean()
                
                if len(method_means) >= 2:
                    # Find methods with significant differences
                    methods = list(method_means.keys())
                    values = list(method_means.values())
                    
                    # Calculate coefficient of variation
                    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    
                    if cv > 0.2:  # Significant variation
                        best_method = max(method_means, key=method_means.get)
                        
                        relation = StructurePropertyRelation(
                            structural_feature=f"synthesis_method_{best_method}",
                            property_name=prop,
                            correlation_strength=cv,
                            correlation_type="method_specific",
                            confidence=0.8,  # High confidence for method effects
                            evidence_count=len(method_means),
                            analysis_method=AnalysisMethod.STATISTICAL_CORRELATION
                        )
                        relationships.append(relation)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to analyze synthesis relationships: {e}")
            return []
    
    def _basic_relationship_analysis(self, 
                                   materials: List[Material],
                                   target_properties: List[str]) -> List[StructurePropertyRelation]:
        """Basic relationship analysis without advanced statistics."""
        relationships = []
        
        # Simple analysis based on material property ranges
        property_ranges = {}
        
        for prop_name in target_properties:
            values = []
            for material in materials:
                for prop in material.properties:
                    if prop.property_name == prop_name:
                        try:
                            values.append(float(prop.value))
                        except (ValueError, TypeError):
                            continue
            
            if len(values) >= 3:
                property_ranges[prop_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # Create basic relationships
        for prop_name, stats in property_ranges.items():
            if stats['std'] > 0.1 * stats['mean']:  # Significant variation
                relation = StructurePropertyRelation(
                    structural_feature="material_variation",
                    property_name=prop_name,
                    correlation_strength=0.5,
                    correlation_type="variable",
                    confidence=0.6,
                    evidence_count=len(materials),
                    analysis_method=AnalysisMethod.EXPERT_RULES
                )
                relationships.append(relation)
        
        return relationships


class LLMImprovementSuggester:
    """Uses LLM to generate intelligent improvement suggestions."""
    
    def __init__(self):
        """Initialize LLM-based suggester."""
        self.llm = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,  # Some creativity for suggestions
                    timeout=60
                )
                logger.info("LLM initialized for improvement suggestions")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for improvement suggestions."""
        self.improvement_prompt = ChatPromptTemplate.from_template("""
        You are a materials science expert analyzing a material for drug delivery applications.
        
        Material Information:
        - Name: {material_name}
        - Composition: {composition}
        - Current Properties: {current_properties}
        
        Performance Gaps:
        {property_gaps}
        
        Structure-Property Relationships:
        {relationships}
        
        Based on this analysis, provide 3-5 specific improvement suggestions to address the property gaps.
        For each suggestion, include:
        1. Type of improvement (composition, structure, synthesis, etc.)
        2. Specific changes to make
        3. Expected benefits
        4. Implementation difficulty (1-10 scale)
        5. Scientific reasoning
        
        Focus on improvements that are scientifically sound and practically implementable.
        
        Format your response as JSON with the following structure:
        {{
            "suggestions": [
                {{
                    "type": "improvement_type",
                    "description": "brief description",
                    "specific_changes": {{"parameter": "value"}},
                    "expected_benefits": {{"property": improvement_amount}},
                    "difficulty": difficulty_score,
                    "reasoning": "scientific justification"
                }}
            ]
        }}
        """)
    
    async def generate_suggestions(self, 
                                 material: Material,
                                 property_gaps: List[PropertyGap],
                                 relationships: List[StructurePropertyRelation]) -> List[ImprovementSuggestion]:
        """Generate LLM-based improvement suggestions."""
        if not self.llm:
            return self._fallback_suggestions(property_gaps)
        
        try:
            # Prepare prompt data
            current_properties = {}
            for prop in material.properties:
                current_properties[prop.property_name] = f"{prop.value} {prop.unit}"
            
            gaps_text = "\n".join([
                f"- {gap.property_name}: Current={gap.current_value}, Target={gap.target_value}, Gap={gap.gap_percentage:.1f}%"
                for gap in property_gaps
            ])
            
            relationships_text = "\n".join([
                f"- {rel.structural_feature} affects {rel.property_name} ({rel.correlation_type}, strength={rel.correlation_strength:.2f})"
                for rel in relationships
            ])
            
            # Generate suggestions
            messages = self.improvement_prompt.invoke({
                "material_name": material.name,
                "composition": json.dumps(material.composition),
                "current_properties": json.dumps(current_properties),
                "property_gaps": gaps_text,
                "relationships": relationships_text
            })
            
            response = await self.llm.ainvoke(messages)
            
            # Parse response
            try:
                response_data = json.loads(response.content)
                suggestions = []
                
                for i, sugg_data in enumerate(response_data.get("suggestions", [])):
                    suggestion = ImprovementSuggestion(
                        suggestion_id=f"{material.material_id}_sugg_{i}",
                        improvement_type=ImprovementType(sugg_data.get("type", "composition_modification")),
                        description=sugg_data.get("description", ""),
                        specific_changes=sugg_data.get("specific_changes", {}),
                        expected_benefits=sugg_data.get("expected_benefits", {}),
                        implementation_difficulty=float(sugg_data.get("difficulty", 5)) / 10,
                        confidence=0.7,  # LLM suggestions have moderate confidence
                        reasoning=sugg_data.get("reasoning", ""),
                        supporting_evidence=["LLM analysis based on materials science principles"]
                    )
                    suggestions.append(suggestion)
                
                return suggestions
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return self._fallback_suggestions(property_gaps)
            
        except Exception as e:
            logger.error(f"Failed to generate LLM suggestions: {e}")
            return self._fallback_suggestions(property_gaps)
    
    def _fallback_suggestions(self, property_gaps: List[PropertyGap]) -> List[ImprovementSuggestion]:
        """Generate fallback suggestions without LLM."""
        suggestions = []
        
        for i, gap in enumerate(property_gaps[:3]):  # Top 3 gaps
            if "mechanical" in gap.property_name.lower():
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"fallback_mech_{i}",
                    improvement_type=ImprovementType.CROSSLINKING_MODIFICATION,
                    description=f"Increase crosslinking to improve {gap.property_name}",
                    specific_changes={"crosslink_density": "increase by 20%"},
                    expected_benefits={gap.property_name: gap.gap_magnitude * 0.3},
                    implementation_difficulty=0.6,
                    confidence=0.5,
                    reasoning="Higher crosslinking typically improves mechanical properties",
                    supporting_evidence=["Standard polymer science principles"]
                )
            elif "degradation" in gap.property_name.lower():
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"fallback_deg_{i}",
                    improvement_type=ImprovementType.COMPOSITION_MODIFICATION,
                    description=f"Modify degradable linkages to adjust {gap.property_name}",
                    specific_changes={"ester_ratio": "adjust based on target rate"},
                    expected_benefits={gap.property_name: gap.gap_magnitude * 0.4},
                    implementation_difficulty=0.7,
                    confidence=0.6,
                    reasoning="Ester linkage density controls degradation rate",
                    supporting_evidence=["Biodegradable polymer literature"]
                )
            else:
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"fallback_gen_{i}",
                    improvement_type=ImprovementType.SYNTHESIS_OPTIMIZATION,
                    description=f"Optimize synthesis conditions for better {gap.property_name}",
                    specific_changes={"reaction_conditions": "optimize temperature and time"},
                    expected_benefits={gap.property_name: gap.gap_magnitude * 0.2},
                    implementation_difficulty=0.4,
                    confidence=0.4,
                    reasoning="Synthesis optimization often improves material properties",
                    supporting_evidence=["General materials engineering principles"]
                )
            
            suggestions.append(suggestion)
        
        return suggestions


class ImprovementAnalyzer:
    """Main improvement analyzer for material discovery."""
    
    def __init__(self, 
                 property_db: PropertyDatabase,
                 similarity_matcher: SimilarityMatcher):
        """Initialize improvement analyzer."""
        self.property_db = property_db
        self.similarity_matcher = similarity_matcher
        
        # Initialize sub-components
        self.gap_analyzer = PropertyGapAnalyzer(property_db)
        self.structure_analyzer = StructurePropertyAnalyzer(property_db, similarity_matcher)
        self.llm_suggester = LLMImprovementSuggester()
        
        logger.info("ImprovementAnalyzer initialized")
    
    async def analyze_material(self, 
                             material: Material,
                             target_properties: Dict[str, float],
                             importance_weights: Dict[str, float] = None) -> AnalysisResult:
        """Perform comprehensive improvement analysis."""
        try:
            # Step 1: Identify property gaps
            property_gaps = await self.gap_analyzer.identify_gaps(
                material, target_properties, importance_weights
            )
            
            # Step 2: Get similar materials for analysis
            similar_results = await self.similarity_matcher.find_similar_materials(material)
            similar_materials = [result.similar_material for result in similar_results[:20]]
            similar_materials.append(material)  # Include target material
            
            # Step 3: Analyze structure-property relationships
            target_prop_names = list(target_properties.keys())
            structure_relations = await self.structure_analyzer.analyze_relationships(
                similar_materials, target_prop_names
            )
            
            # Step 4: Generate improvement suggestions
            improvement_suggestions = await self.llm_suggester.generate_suggestions(
                material, property_gaps, structure_relations
            )
            
            # Step 5: Calculate overall scores
            overall_score = self._calculate_overall_score(property_gaps, improvement_suggestions)
            analysis_confidence = self._calculate_analysis_confidence(
                property_gaps, structure_relations, improvement_suggestions
            )
            
            return AnalysisResult(
                material_id=material.material_id,
                property_gaps=property_gaps,
                structure_property_relations=structure_relations,
                improvement_suggestions=improvement_suggestions,
                overall_score=overall_score,
                analysis_confidence=analysis_confidence,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze material: {e}")
            # Return minimal result
            return AnalysisResult(
                material_id=material.material_id,
                property_gaps=[],
                structure_property_relations=[],
                improvement_suggestions=[],
                overall_score=0.0,
                analysis_confidence=0.0,
                analysis_timestamp=datetime.now()
            )
    
    def _calculate_overall_score(self, 
                               gaps: List[PropertyGap],
                               suggestions: List[ImprovementSuggestion]) -> float:
        """Calculate overall improvement potential score."""
        if not gaps:
            return 1.0  # Perfect if no gaps
        
        # Score based on gap magnitude and suggestion quality
        gap_score = 1.0 - sum(gap.gap_percentage * gap.importance for gap in gaps) / (100 * len(gaps))
        gap_score = max(0.0, gap_score)
        
        # Suggestion quality score
        suggestion_score = 0.5
        if suggestions:
            avg_confidence = sum(sugg.confidence for sugg in suggestions) / len(suggestions)
            avg_difficulty = sum(sugg.implementation_difficulty for sugg in suggestions) / len(suggestions)
            suggestion_score = (avg_confidence + (1.0 - avg_difficulty)) / 2
        
        return (gap_score + suggestion_score) / 2
    
    def _calculate_analysis_confidence(self, 
                                     gaps: List[PropertyGap],
                                     relations: List[StructurePropertyRelation],
                                     suggestions: List[ImprovementSuggestion]) -> float:
        """Calculate confidence in the analysis."""
        confidence_factors = []
        
        # Data availability
        if gaps:
            confidence_factors.append(0.8)  # Have property data
        
        # Structure-property relationships
        if relations:
            avg_relation_confidence = sum(rel.confidence for rel in relations) / len(relations)
            confidence_factors.append(avg_relation_confidence)
        
        # Suggestion quality
        if suggestions:
            avg_suggestion_confidence = sum(sugg.confidence for sugg in suggestions) / len(suggestions)
            confidence_factors.append(avg_suggestion_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.3


# Factory function for easy initialization
def create_improvement_analyzer(property_db: PropertyDatabase,
                              similarity_matcher: SimilarityMatcher) -> ImprovementAnalyzer:
    """Create improvement analyzer with default configuration."""
    return ImprovementAnalyzer(property_db, similarity_matcher) 