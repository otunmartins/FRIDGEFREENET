#!/usr/bin/env python3
"""
Feedback Generator Implementation for RAG System - Phase 3

This module integrates all RAG components to generate intelligent feedback,
next iteration prompts, and specific improvement recommendations for the
active learning material discovery system.

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

# LLM integration
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for feedback generation")

# Import RAG components
from .vector_database import VectorDatabase, MaterialDocument, SearchResult
from .property_database import PropertyDatabase, Material, MaterialProperty, PropertyType, DataSource
from .similarity_matcher import SimilarityMatcher, SimilarityResult
from .improvement_analyzer import ImprovementAnalyzer, AnalysisResult, ImprovementSuggestion
from .enhanced_web_search import EnhancedWebSearchAgent, AggregatedResults, SearchDomain

# Import active learning components
from ..state_manager import IterationState, ComputedProperties, RAGAnalysis

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be generated."""
    NEXT_ITERATION_PROMPT = "next_iteration_prompt"
    IMPROVEMENT_SUGGESTIONS = "improvement_suggestions"
    SYNTHESIS_MODIFICATIONS = "synthesis_modifications"
    PROPERTY_OPTIMIZATION = "property_optimization"
    LITERATURE_INSIGHTS = "literature_insights"
    BENCHMARK_COMPARISON = "benchmark_comparison"


class FeedbackPriority(Enum):
    """Priority levels for feedback items."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FeedbackItem:
    """Individual feedback item with specific recommendations."""
    feedback_type: FeedbackType
    priority: FeedbackPriority
    title: str
    description: str
    specific_actions: List[str]
    expected_impact: Dict[str, float]
    confidence: float
    supporting_evidence: List[str]
    implementation_notes: Optional[str] = None


@dataclass
class IterationFeedback:
    """Complete feedback for a single iteration."""
    iteration_number: int
    material_name: str
    overall_assessment: str
    performance_score: float
    feedback_items: List[FeedbackItem]
    next_iteration_prompt: str
    priority_improvements: List[str]
    literature_insights: List[str]
    benchmark_comparison: Dict[str, Any]
    confidence: float
    generation_timestamp: datetime


class LiteratureInsightExtractor:
    """Extracts insights from literature search results."""
    
    def __init__(self, web_search_agent: EnhancedWebSearchAgent):
        """Initialize with web search agent."""
        self.web_search_agent = web_search_agent
    
    async def extract_insights(self, 
                             material_properties: Dict[str, float],
                             performance_gaps: List[str]) -> List[str]:
        """Extract relevant insights from literature."""
        insights = []
        
        try:
            # Search for similar materials and their improvements
            for gap in performance_gaps[:3]:  # Focus on top 3 gaps
                search_query = f"improve {gap} drug delivery polymer materials"
                
                search_results = await self.web_search_agent.search(
                    search_query,
                    domains=[SearchDomain.SCIENTIFIC_LITERATURE, SearchDomain.PATENTS],
                    max_results=10
                )
                
                # Extract key insights from results
                if search_results.key_findings:
                    insights.extend(search_results.key_findings[:2])  # Top 2 findings per gap
                
                # Add synthesis method insights
                if search_results.synthesis_methods:
                    insights.append(f"Consider synthesis methods: {', '.join(search_results.synthesis_methods[:3])}")
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            logger.error(f"Failed to extract literature insights: {e}")
            return ["Literature search encountered technical difficulties"]


class BenchmarkComparator:
    """Compares material performance against literature benchmarks."""
    
    def __init__(self, property_db: PropertyDatabase, similarity_matcher: SimilarityMatcher):
        """Initialize with databases."""
        self.property_db = property_db
        self.similarity_matcher = similarity_matcher
    
    async def compare_against_benchmarks(self, 
                                       material: Material,
                                       target_properties: Dict[str, float]) -> Dict[str, Any]:
        """Compare material against benchmark performance."""
        comparison = {
            "benchmark_scores": {},
            "ranking": {},
            "improvement_potential": {},
            "best_in_class": {}
        }
        
        try:
            # Get similar materials for comparison
            similar_results = await self.similarity_matcher.find_similar_materials(material)
            
            if not similar_results:
                return comparison
            
            # Extract properties from similar materials
            benchmark_properties = {}
            for result in similar_results[:10]:  # Top 10 similar materials
                for prop in result.similar_material.properties:
                    prop_name = prop.property_name
                    try:
                        prop_value = float(prop.value)
                        if prop_name not in benchmark_properties:
                            benchmark_properties[prop_name] = []
                        benchmark_properties[prop_name].append(prop_value)
                    except (ValueError, TypeError):
                        continue
            
            # Calculate benchmark statistics
            current_properties = {}
            for prop in material.properties:
                try:
                    current_properties[prop.property_name] = float(prop.value)
                except (ValueError, TypeError):
                    continue
            
            for prop_name in target_properties.keys():
                if prop_name in benchmark_properties and prop_name in current_properties:
                    benchmark_values = benchmark_properties[prop_name]
                    current_value = current_properties[prop_name]
                    
                    # Calculate percentile ranking
                    better_count = sum(1 for v in benchmark_values if current_value > v)
                    percentile = (better_count / len(benchmark_values)) * 100
                    
                    comparison["benchmark_scores"][prop_name] = {
                        "current_value": current_value,
                        "benchmark_mean": sum(benchmark_values) / len(benchmark_values),
                        "benchmark_max": max(benchmark_values),
                        "benchmark_min": min(benchmark_values),
                        "percentile": percentile
                    }
                    
                    comparison["ranking"][prop_name] = f"{percentile:.1f}th percentile"
                    
                    # Calculate improvement potential
                    max_value = max(benchmark_values)
                    if max_value > current_value:
                        potential = ((max_value - current_value) / current_value) * 100
                        comparison["improvement_potential"][prop_name] = f"{potential:.1f}% improvement possible"
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare against benchmarks: {e}")
            return comparison


class PromptGenerator:
    """Generates next iteration prompts based on analysis."""
    
    def __init__(self):
        """Initialize prompt generator."""
        self.llm = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.8,  # Higher creativity for prompt generation
                    timeout=60
                )
                logger.info("LLM initialized for prompt generation")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates."""
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert in materials science and drug delivery systems. Generate a creative and specific 
        prompt for the next iteration of material design based on the current analysis.
        
        Current Material: {material_name}
        Current Properties: {current_properties}
        Target Properties: {target_properties}
        Performance Gaps: {performance_gaps}
        Improvement Suggestions: {improvement_suggestions}
        Literature Insights: {literature_insights}
        
        Generate a specific, actionable prompt for the next iteration that:
        1. Addresses the most critical performance gaps
        2. Incorporates the best improvement suggestions
        3. Uses insights from literature
        4. Is creative but scientifically sound
        5. Provides specific guidance for molecular design
        
        Return only the prompt text (no additional explanation):
        """)
    
    async def generate_next_iteration_prompt(self, 
                                           material_name: str,
                                           current_properties: Dict[str, float],
                                           target_properties: Dict[str, float],
                                           improvement_suggestions: List[ImprovementSuggestion],
                                           literature_insights: List[str]) -> str:
        """Generate next iteration prompt."""
        if not self.llm:
            return self._fallback_prompt(material_name, current_properties, target_properties)
        
        try:
            # Calculate performance gaps
            performance_gaps = []
            for prop_name, target_value in target_properties.items():
                if prop_name in current_properties:
                    current_value = current_properties[prop_name]
                    gap = abs(target_value - current_value) / max(abs(target_value), 1e-6) * 100
                    performance_gaps.append(f"{prop_name}: {gap:.1f}% gap")
            
            # Format improvement suggestions
            suggestion_text = "\n".join([
                f"- {sugg.description}: {sugg.reasoning}"
                for sugg in improvement_suggestions[:3]
            ])
            
            messages = self.prompt_template.invoke({
                "material_name": material_name,
                "current_properties": json.dumps(current_properties),
                "target_properties": json.dumps(target_properties),
                "performance_gaps": "\n".join(performance_gaps),
                "improvement_suggestions": suggestion_text,
                "literature_insights": "\n".join(literature_insights[:3])
            })
            
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate LLM prompt: {e}")
            return self._fallback_prompt(material_name, current_properties, target_properties)
    
    def _fallback_prompt(self, 
                        material_name: str,
                        current_properties: Dict[str, float],
                        target_properties: Dict[str, float]) -> str:
        """Generate fallback prompt without LLM."""
        # Find the property with the largest gap
        max_gap = 0
        gap_property = "mechanical_strength"
        
        for prop_name, target_value in target_properties.items():
            if prop_name in current_properties:
                current_value = current_properties[prop_name]
                gap = abs(target_value - current_value) / max(abs(target_value), 1e-6)
                if gap > max_gap:
                    max_gap = gap
                    gap_property = prop_name
        
        return (f"Design an improved version of {material_name} with enhanced {gap_property}. "
               f"Focus on optimizing polymer composition and crosslinking density to achieve "
               f"target properties while maintaining biocompatibility and controlled degradation. "
               f"Consider incorporating functional groups that can improve {gap_property} "
               f"without compromising other essential properties.")


class FeedbackGenerator:
    """Main feedback generator integrating all RAG components."""
    
    def __init__(self, 
                 vector_db: VectorDatabase,
                 property_db: PropertyDatabase,
                 similarity_matcher: SimilarityMatcher,
                 improvement_analyzer: ImprovementAnalyzer,
                 web_search_agent: EnhancedWebSearchAgent):
        """Initialize feedback generator with all RAG components."""
        self.vector_db = vector_db
        self.property_db = property_db
        self.similarity_matcher = similarity_matcher
        self.improvement_analyzer = improvement_analyzer
        self.web_search_agent = web_search_agent
        
        # Initialize sub-components
        self.literature_extractor = LiteratureInsightExtractor(web_search_agent)
        self.benchmark_comparator = BenchmarkComparator(property_db, similarity_matcher)
        self.prompt_generator = PromptGenerator()
        
        # Cache for expensive operations
        self._feedback_cache = {}
        
        logger.info("FeedbackGenerator initialized with all RAG components")
    
    async def generate_iteration_feedback(self, 
                                        state: IterationState,
                                        target_properties: Dict[str, float]) -> IterationFeedback:
        """Generate comprehensive feedback for an iteration."""
        try:
            # Extract material information
            material_name = f"Iteration_{state.iteration_number}_Material"
            
            # Create material object from computed properties
            material = await self._create_material_from_state(state)
            
            # Run comprehensive analysis
            analysis_result = await self.improvement_analyzer.analyze_material(
                material, target_properties
            )
            
            # Extract literature insights
            performance_gaps = [gap.property_name for gap in analysis_result.property_gaps]
            literature_insights = await self.literature_extractor.extract_insights(
                target_properties, performance_gaps
            )
            
            # Compare against benchmarks
            benchmark_comparison = await self.benchmark_comparator.compare_against_benchmarks(
                material, target_properties
            )
            
            # Generate next iteration prompt
            current_properties = {}
            if state.computed_properties and state.computed_properties.md_properties:
                # Extract current properties from MD simulation results
                md_props = state.computed_properties.md_properties
                current_properties = {
                    "young_modulus": getattr(md_props, 'youngs_modulus_x', 0.0),
                    "glass_transition_temp": getattr(md_props, 'glass_transition_temp', 0.0),
                    "density": getattr(md_props, 'density', 0.0)
                }
            
            next_prompt = await self.prompt_generator.generate_next_iteration_prompt(
                material_name, current_properties, target_properties,
                analysis_result.improvement_suggestions, literature_insights
            )
            
            # Create feedback items
            feedback_items = await self._create_feedback_items(analysis_result, literature_insights)
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                current_properties, target_properties, analysis_result
            )
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(
                performance_score, analysis_result, literature_insights
            )
            
            # Extract priority improvements
            priority_improvements = [
                sugg.description for sugg in analysis_result.improvement_suggestions[:3]
            ]
            
            return IterationFeedback(
                iteration_number=state.iteration_number,
                material_name=material_name,
                overall_assessment=overall_assessment,
                performance_score=performance_score,
                feedback_items=feedback_items,
                next_iteration_prompt=next_prompt,
                priority_improvements=priority_improvements,
                literature_insights=literature_insights,
                benchmark_comparison=benchmark_comparison,
                confidence=analysis_result.analysis_confidence,
                generation_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate iteration feedback: {e}")
            return self._create_fallback_feedback(state)
    
    async def _create_material_from_state(self, state: IterationState) -> Material:
        """Create a Material object from iteration state."""
        properties = []
        
        if state.computed_properties:
            # Add computed properties
            if state.computed_properties.md_properties:
                md_props = state.computed_properties.md_properties
                
                # Convert MD properties to MaterialProperty objects
                property_mappings = [
                    ("youngs_modulus_x", "young_modulus", "GPa"),
                    ("glass_transition_temp", "glass_transition_temperature", "K"),
                    ("density", "density", "g/cm3"),
                    ("diffusion_coefficient_water", "water_diffusion", "cm2/s"),
                    ("cohesive_energy", "cohesive_energy", "kJ/mol")
                ]
                
                for md_attr, prop_name, unit in property_mappings:
                    if hasattr(md_props, md_attr):
                        value = getattr(md_props, md_attr)
                        if value is not None:
                            prop = MaterialProperty(
                                material_id=f"iter_{state.iteration_number}",
                                property_name=prop_name,
                                property_type=PropertyType.MECHANICAL,  # Default type
                                value=value,
                                unit=unit,
                                source=DataSource.COMPUTATIONAL,
                                confidence=0.8
                            )
                            properties.append(prop)
        
        # Create material composition from generated molecules
        composition = {"polymer_matrix": 1.0}
        if state.generated_molecules and hasattr(state.generated_molecules, 'molecules'):
            composition = {"generated_polymer": 1.0}
        
        return Material(
            material_id=f"iteration_{state.iteration_number}",
            name=f"Generated_Material_Iter_{state.iteration_number}",
            composition=composition,
            synthesis_method="automated_piecewise_generation",
            properties=properties,
            created_date=datetime.now(),
            metadata={"iteration": state.iteration_number}
        )
    
    async def _create_feedback_items(self, 
                                   analysis_result: AnalysisResult,
                                   literature_insights: List[str]) -> List[FeedbackItem]:
        """Create specific feedback items from analysis."""
        feedback_items = []
        
        # Create feedback for improvement suggestions
        for i, suggestion in enumerate(analysis_result.improvement_suggestions[:5]):
            priority = FeedbackPriority.HIGH if i < 2 else FeedbackPriority.MEDIUM
            
            feedback_item = FeedbackItem(
                feedback_type=FeedbackType.IMPROVEMENT_SUGGESTIONS,
                priority=priority,
                title=f"Improvement: {suggestion.improvement_type.value.replace('_', ' ').title()}",
                description=suggestion.description,
                specific_actions=[
                    f"Implement: {key} = {value}" 
                    for key, value in suggestion.specific_changes.items()
                ],
                expected_impact=suggestion.expected_benefits,
                confidence=suggestion.confidence,
                supporting_evidence=suggestion.supporting_evidence,
                implementation_notes=suggestion.reasoning
            )
            feedback_items.append(feedback_item)
        
        # Create feedback for literature insights
        if literature_insights:
            feedback_item = FeedbackItem(
                feedback_type=FeedbackType.LITERATURE_INSIGHTS,
                priority=FeedbackPriority.MEDIUM,
                title="Literature-Based Recommendations",
                description="Insights from recent scientific literature",
                specific_actions=literature_insights[:3],
                expected_impact={"research_alignment": 0.8},
                confidence=0.7,
                supporting_evidence=["Recent scientific publications"],
                implementation_notes="Consider these findings from current research"
            )
            feedback_items.append(feedback_item)
        
        # Create feedback for property gaps
        for gap in analysis_result.property_gaps[:3]:
            feedback_item = FeedbackItem(
                feedback_type=FeedbackType.PROPERTY_OPTIMIZATION,
                priority=FeedbackPriority.HIGH if gap.importance > 0.7 else FeedbackPriority.MEDIUM,
                title=f"Address {gap.property_name} Gap",
                description=f"Current: {gap.current_value}, Target: {gap.target_value}",
                specific_actions=[f"Improve {gap.property_name} by {gap.gap_percentage:.1f}%"],
                expected_impact={gap.property_name: gap.gap_magnitude},
                confidence=0.9,
                supporting_evidence=["Direct property measurement gap"],
                implementation_notes=f"Priority based on {gap.importance:.1f} importance weight"
            )
            feedback_items.append(feedback_item)
        
        return feedback_items
    
    def _calculate_performance_score(self, 
                                   current_properties: Dict[str, float],
                                   target_properties: Dict[str, float],
                                   analysis_result: AnalysisResult) -> float:
        """Calculate overall performance score."""
        if not current_properties or not target_properties:
            return 0.5  # Default score
        
        # Calculate property achievement score
        property_scores = []
        for prop_name, target_value in target_properties.items():
            if prop_name in current_properties:
                current_value = current_properties[prop_name]
                relative_error = abs(target_value - current_value) / max(abs(target_value), 1e-6)
                score = max(0.0, 1.0 - relative_error)
                property_scores.append(score)
        
        property_score = sum(property_scores) / len(property_scores) if property_scores else 0.5
        
        # Combine with analysis overall score
        combined_score = (property_score + analysis_result.overall_score) / 2
        
        return min(1.0, max(0.0, combined_score))
    
    def _generate_overall_assessment(self, 
                                   performance_score: float,
                                   analysis_result: AnalysisResult,
                                   literature_insights: List[str]) -> str:
        """Generate overall assessment text."""
        if performance_score > 0.8:
            assessment = "Excellent performance achieved. Material meets most target requirements."
        elif performance_score > 0.6:
            assessment = "Good performance with room for targeted improvements."
        elif performance_score > 0.4:
            assessment = "Moderate performance. Several key properties need optimization."
        else:
            assessment = "Performance below targets. Significant improvements needed."
        
        # Add specific insights
        if analysis_result.improvement_suggestions:
            top_suggestion = analysis_result.improvement_suggestions[0]
            assessment += f" Priority focus: {top_suggestion.description}."
        
        if literature_insights:
            assessment += f" Literature suggests: {literature_insights[0][:100]}..."
        
        return assessment
    
    def _create_fallback_feedback(self, state: IterationState) -> IterationFeedback:
        """Create minimal feedback when full analysis fails."""
        return IterationFeedback(
            iteration_number=state.iteration_number,
            material_name=f"Material_Iter_{state.iteration_number}",
            overall_assessment="Analysis incomplete due to technical issues",
            performance_score=0.5,
            feedback_items=[],
            next_iteration_prompt="Continue with current approach and monitor for improvements",
            priority_improvements=["Resolve technical analysis issues"],
            literature_insights=["Technical difficulties prevented literature analysis"],
            benchmark_comparison={},
            confidence=0.1,
            generation_timestamp=datetime.now()
        )


# Factory function for easy initialization
def create_feedback_generator(vector_db: VectorDatabase,
                            property_db: PropertyDatabase,
                            similarity_matcher: SimilarityMatcher,
                            improvement_analyzer: ImprovementAnalyzer,
                            web_search_agent: EnhancedWebSearchAgent) -> FeedbackGenerator:
    """Create feedback generator with all RAG components."""
    return FeedbackGenerator(
        vector_db, property_db, similarity_matcher, 
        improvement_analyzer, web_search_agent
    ) 