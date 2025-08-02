#!/usr/bin/env python3
"""
RAGPropertyAnalyzer - Phase 2 Implementation

This module provides RAG-powered analysis and feedback generation for the 
active learning material discovery system. It includes web search capabilities,
property benchmarking, and next iteration prompt generation.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Import web search and RAG components
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily search not available")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available")

try:
    from ...integration.rag_literature_mining import RAGLiteratureMiningSystem
    RAG_LITERATURE_AVAILABLE = True
except ImportError:
    RAG_LITERATURE_AVAILABLE = False
    logging.warning("RAG Literature Mining system not available")

# Import active learning infrastructure
from .state_manager import IterationState, ComputedProperties, RAGAnalysis
from .decision_engine import LLMDecisionEngine, DecisionType

logger = logging.getLogger(__name__)


class RAGAnalysisContext:
    """Context data for RAG analysis and feedback generation."""
    
    def __init__(self, iteration: int, target_properties: Dict[str, float],
                 computed_properties: Optional[ComputedProperties] = None,
                 previous_analyses: List[Dict] = None):
        self.iteration = iteration
        self.target_properties = target_properties or {}
        self.computed_properties = computed_properties
        self.previous_analyses = previous_analyses or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for decision engine."""
        return {
            "iteration": self.iteration,
            "target_properties": self.target_properties,
            "computed_properties_summary": self._extract_properties_summary(),
            "previous_analyses_count": len(self.previous_analyses),
            "timestamp": self.timestamp.isoformat()
        }
    
    def _extract_properties_summary(self) -> Dict[str, Any]:
        """Extract key information from computed properties."""
        if not self.computed_properties:
            return {}
        
        return {
            "performance_score": self.computed_properties.performance_score,
            "mechanical_properties_count": len(self.computed_properties.mechanical_properties),
            "thermal_properties_count": len(self.computed_properties.thermal_properties),
            "confidence_level": getattr(self.computed_properties, 'confidence_level', 0.5),
            "has_recommendations": bool(getattr(self.computed_properties, 'recommendations', None))
        }


class RAGPropertyAnalyzer:
    """
    RAG-powered analysis and feedback system.
    
    This class provides comprehensive analysis including online property research,
    benchmark comparison, improvement suggestions, and next iteration planning.
    """
    
    def __init__(self, storage_path: str = "rag_property_analysis"):
        """Initialize RAG property analyzer.
        
        Args:
            storage_path: Path to store analysis data and results
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize web search and RAG systems
        self._initialize_rag_systems()
        
        # Cache for results and searches
        self._search_cache = {}
        self._benchmark_cache = {}
        
        logger.info("RAGPropertyAnalyzer initialized")
    
    def _initialize_rag_systems(self):
        """Initialize available RAG and search systems."""
        # Initialize OpenAI for analysis
        if OPENAI_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,  # Lower temperature for more factual analysis
                    timeout=60
                )
                logger.info("OpenAI initialized for RAG analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
                self.llm = None
        else:
            self.llm = None
        
        # Initialize web search
        if TAVILY_AVAILABLE:
            try:
                self.web_searcher = TavilySearchResults(
                    max_results=10,
                    search_depth="advanced"
                )
                logger.info("Tavily web search initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily search: {e}")
                self.web_searcher = None
        else:
            self.web_searcher = None
        
        # Initialize RAG literature mining
        if RAG_LITERATURE_AVAILABLE:
            try:
                self.rag_literature = RAGLiteratureMiningSystem()
                logger.info("RAG Literature Mining initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG Literature Mining: {e}")
                self.rag_literature = None
        else:
            self.rag_literature = None
        
        # Setup analysis prompts
        self._setup_analysis_prompts()
    
    def _setup_analysis_prompts(self):
        """Setup prompt templates for RAG analysis."""
        
        self.property_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert materials scientist specializing in biomedical polymers and drug delivery systems.

Analyze the provided material properties and performance data to:
1. Identify strengths and weaknesses
2. Compare against literature benchmarks
3. Suggest specific improvements
4. Generate next iteration guidance

Be specific, quantitative when possible, and focus on actionable insights."""),
            ("human", """Material Analysis Request:

Target Properties: {target_properties}
Computed Properties: {computed_properties}
Performance Score: {performance_score}

Current Property Values:
- Mechanical: {mechanical_properties}
- Thermal: {thermal_properties}
- Transport: {transport_properties}
- Stability: {stability_metrics}

Please provide a comprehensive analysis with specific recommendations for improvement.""")
        ])
        
        self.benchmark_comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a materials science expert analyzing polymer properties against literature benchmarks.

Compare the provided properties with known materials and:
1. Identify similar materials from literature
2. Benchmark performance against established standards
3. Highlight gaps and opportunities
4. Suggest specific modifications"""),
            ("human", """Benchmark Analysis Request:

Current Material Properties: {current_properties}
Target Application: Insulin delivery polymer
Target Properties: {target_properties}

Literature Context: {literature_context}

Please compare against known insulin delivery polymers and provide specific benchmark comparisons.""")
        ])
        
        self.improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a polymer design expert focused on practical improvements.

Based on the analysis, generate specific, actionable recommendations for:
1. Monomer selection modifications
2. Synthesis parameter adjustments  
3. Processing condition changes
4. Functional group additions/modifications

Prioritize changes that will most effectively address performance gaps."""),
            ("human", """Improvement Generation Request:

Current Performance: {performance_analysis}
Key Weaknesses: {identified_weaknesses}
Target Improvements: {target_improvements}
Previous Iterations: {iteration_history}

Generate 3-5 specific, prioritized recommendations for the next design iteration.""")
        ])
        
        self.next_prompt_generator = ChatPromptTemplate.from_messages([
            ("system", """You are a prompt engineering expert for materials design.

Create a detailed, specific prompt for the next iteration of polymer design that:
1. Incorporates lessons learned from current iteration
2. Addresses specific performance gaps
3. Provides clear design guidance
4. Builds on successful elements

The prompt should be suitable for a piecewise polymer generation system."""),
            ("human", """Next Iteration Prompt Request:

Current Iteration: {iteration_number}
Performance Summary: {performance_summary}
Key Improvements Needed: {improvements_needed}
Successful Elements: {successful_elements}
Target Properties: {target_properties}

Generate a comprehensive prompt for the next polymer design iteration.""")
        ])
    
    async def run_automated_analysis(self, state: IterationState, 
                                   decision_engine: LLMDecisionEngine) -> RAGAnalysis:
        """
        Run automated RAG analysis with decision making.
        
        Args:
            state: Current iteration state
            decision_engine: LLM decision engine for automation
            
        Returns:
            RAGAnalysis: Comprehensive RAG analysis results
        """
        logger.info(f"Starting RAG analysis for iteration {state.iteration_number}")
        
        try:
            # Step 1: Create analysis context
            analysis_context = RAGAnalysisContext(
                iteration=state.iteration_number,
                target_properties=state.target_properties,
                computed_properties=state.computed_properties,
                previous_analyses=self._extract_previous_analyses(state)
            )
            
            # Step 2: Perform online property research
            research_results = await self._perform_online_research(
                analysis_context, decision_engine
            )
            
            # Step 3: Execute benchmark comparison
            benchmark_results = await self._execute_benchmark_comparison(
                analysis_context, research_results, decision_engine
            )
            
            # Step 4: Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(
                analysis_context, benchmark_results, decision_engine
            )
            
            # Step 5: Create next iteration planning
            next_iteration_plan = await self._create_next_iteration_plan(
                analysis_context, improvement_suggestions, decision_engine
            )
            
            # Step 6: Compile final RAG analysis
            final_analysis = await self._compile_rag_analysis(
                research_results, benchmark_results, improvement_suggestions, 
                next_iteration_plan, analysis_context
            )
            
            # Step 7: Save results and update cache
            self._save_results(state.iteration_number, final_analysis)
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}")
            # Return minimal results to prevent pipeline failure
            return self._create_fallback_analysis(analysis_context)
    
    async def _perform_online_research(self, context: RAGAnalysisContext,
                                     decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Perform online research for similar materials and properties."""
        
        # Generate research strategy decision
        research_decision = decision_engine.make_decision(
            decision_type=DecisionType.RESEARCH_STRATEGY,
            context_data=context.to_dict(),
            available_options=["broad_search", "targeted_search", "comparative_search"],
            objectives=["Find relevant materials and benchmark data"],
            constraints={"search_depth": "moderate", "source_quality": "high"}
        )
        
        research_results = {
            "strategy": research_decision.chosen_option,
            "similar_materials": [],
            "benchmark_data": [],
            "literature_insights": [],
            "search_queries_used": []
        }
        
        # Generate search queries based on context
        search_queries = self._generate_search_queries(context, research_decision.chosen_option)
        research_results["search_queries_used"] = search_queries
        
        # Execute web searches
        for query in search_queries:
            try:
                search_results = await self._execute_web_search(query)
                research_results["literature_insights"].extend(search_results)
            except Exception as e:
                logger.warning(f"Web search failed for query '{query}': {e}")
        
        # Try RAG literature mining for additional insights
        if self.rag_literature:
            try:
                rag_results = await self._execute_rag_literature_search(context)
                research_results["similar_materials"].extend(rag_results.get("materials", []))
                research_results["benchmark_data"].extend(rag_results.get("benchmarks", []))
            except Exception as e:
                logger.warning(f"RAG literature search failed: {e}")
        
        # Extract similar materials and benchmarks from search results
        extracted_data = self._extract_materials_and_benchmarks(research_results["literature_insights"])
        research_results["similar_materials"].extend(extracted_data.get("materials", []))
        research_results["benchmark_data"].extend(extracted_data.get("benchmarks", []))
        
        return research_results
    
    def _generate_search_queries(self, context: RAGAnalysisContext, strategy: str) -> List[str]:
        """Generate search queries based on context and strategy."""
        
        base_terms = ["insulin delivery polymer", "biodegradable polymer", "biocompatible polymer"]
        
        queries = []
        
        # Add target property specific queries
        for prop in context.target_properties.keys():
            if "biocompatib" in prop.lower():
                queries.append("biocompatible polymer insulin delivery")
            elif "degradation" in prop.lower():
                queries.append("biodegradable polymer degradation rate")
            elif "mechanical" in prop.lower():
                queries.append("polymer mechanical properties insulin delivery")
        
        # Add strategy-specific queries
        if strategy == "broad_search":
            queries.extend([
                "polymer drug delivery systems review",
                "insulin formulation stability polymers",
                "transdermal drug delivery materials"
            ])
        elif strategy == "targeted_search":
            if context.computed_properties:
                perf_score = context.computed_properties.performance_score
                if perf_score < 0.6:
                    queries.append("improving polymer drug delivery performance")
                queries.append("polymer optimization drug delivery")
        elif strategy == "comparative_search":
            queries.extend([
                "insulin delivery polymer comparison",
                "best polymers for drug delivery",
                "polymer performance benchmarks drug delivery"
            ])
        
        return queries[:5]  # Limit to 5 queries
    
    async def _execute_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute web search for a specific query."""
        
        if not self.web_searcher:
            return []
        
        try:
            search_results = self.web_searcher.invoke({"query": query})
            
            processed_results = []
            for result in search_results:
                processed_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "source": "web_search",
                    "query": query,
                    "relevance": self._calculate_search_relevance(result, query)
                })
            
            # Sort by relevance and return top results
            processed_results.sort(key=lambda x: x["relevance"], reverse=True)
            return processed_results[:5]
            
        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {e}")
            return []
    
    def _calculate_search_relevance(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for search result."""
        
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Check for query terms in title (higher weight)
        query_terms = query_lower.split()
        for term in query_terms:
            if term in title:
                score += 0.3
            if term in content:
                score += 0.1
        
        # Boost for relevant keywords
        relevant_keywords = ["insulin", "polymer", "delivery", "biocompatible", "degradation"]
        for keyword in relevant_keywords:
            if keyword in title:
                score += 0.2
            if keyword in content:
                score += 0.1
        
        return min(score, 1.0)
    
    async def _execute_rag_literature_search(self, context: RAGAnalysisContext) -> Dict[str, Any]:
        """Execute RAG literature search for additional insights."""
        
        # This would use the RAG literature mining system
        # For now, return empty results
        return {"materials": [], "benchmarks": []}
    
    def _extract_materials_and_benchmarks(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract material information and benchmarks from search results."""
        
        materials = []
        benchmarks = []
        
        # Simple extraction based on keywords and patterns
        for result in search_results:
            content = result.get("content", "").lower()
            title = result.get("title", "").lower()
            
            # Extract material names using LLM analysis instead of hardcoded keywords
            try:
                material_extraction_prompt = f"""Extract polymer/material names from this research content:

Title: {result.get("title", "")}
Content: {content[:500]}...

List only the specific polymer/material names mentioned. Return as JSON array of material names only.
Example: ["PEG", "PLGA", "chitosan"]"""

                response = self.llm.invoke(material_extraction_prompt)
                import json
                try:
                    extracted_materials = json.loads(response.content.strip())
                    if isinstance(extracted_materials, list):
                        for material in extracted_materials:
                            materials.append({
                                "name": str(material).upper(),
                                "source": result.get("url", ""),
                                "context": result.get("title", ""),
                                "similarity": 0.9
                            })
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse extracted materials: {response.content}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract materials using LLM: {e}")
            
            # Extract benchmark data (simplified)
            if any(term in content for term in ["mpa", "degradation", "biocompatibility", "mechanical"]):
                benchmarks.append({
                    "property": "general",
                    "value": "literature_reported",
                    "source": result.get("url", ""),
                    "material": "various"
                })
        
        # Remove duplicates
        unique_materials = []
        seen_materials = set()
        for material in materials:
            if material["name"] not in seen_materials:
                unique_materials.append(material)
                seen_materials.add(material["name"])
        
        return {
            "materials": unique_materials[:10],  # Limit to 10
            "benchmarks": benchmarks[:10]
        }
    
    async def _execute_benchmark_comparison(self, context: RAGAnalysisContext,
                                          research_results: Dict[str, Any],
                                          decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Execute benchmark comparison using LLM analysis."""
        
        if not self.llm or not context.computed_properties:
            return {"comparison_summary": "No benchmark comparison available"}
        
        try:
            # Prepare data for comparison
            current_properties = {
                "mechanical": context.computed_properties.mechanical_properties,
                "thermal": context.computed_properties.thermal_properties,
                "transport": context.computed_properties.transport_properties,
                "stability": context.computed_properties.stability_metrics,
                "performance_score": context.computed_properties.performance_score
            }
            
            literature_context = {
                "similar_materials": research_results.get("similar_materials", []),
                "benchmark_data": research_results.get("benchmark_data", []),
                "research_insights": research_results.get("literature_insights", [])[:3]  # Top 3
            }
            
            # Generate benchmark comparison
            comparison_response = await self.llm.ainvoke(
                self.benchmark_comparison_prompt.format_messages(
                    current_properties=json.dumps(current_properties, indent=2),
                    target_properties=json.dumps(context.target_properties, indent=2),
                    literature_context=json.dumps(literature_context, indent=2)
                )
            )
            
            return {
                "comparison_summary": comparison_response.content,
                "benchmark_materials": research_results.get("similar_materials", []),
                "performance_gaps": self._identify_performance_gaps(context, research_results),
                "competitive_analysis": self._generate_competitive_analysis(context, research_results)
            }
            
        except Exception as e:
            logger.warning(f"Benchmark comparison failed: {e}")
            return {"comparison_summary": f"Benchmark comparison failed: {str(e)}"}
    
    def _identify_performance_gaps(self, context: RAGAnalysisContext,
                                 research_results: Dict[str, Any]) -> List[str]:
        """Identify performance gaps compared to literature."""
        
        gaps = []
        
        if context.computed_properties:
            perf_score = context.computed_properties.performance_score
            
            if perf_score < 0.6:
                gaps.append("Overall performance below target threshold")
            
            # Check individual property categories
            if not context.computed_properties.mechanical_properties:
                gaps.append("Mechanical properties not characterized")
            
            if not context.computed_properties.thermal_properties:
                gaps.append("Thermal properties not characterized")
            
            # Target property specific gaps
            for prop, target in context.target_properties.items():
                if "biocompatib" in prop.lower() and target > 0.8:
                    gaps.append("High biocompatibility requirement needs validation")
                elif "degradation" in prop.lower():
                    gaps.append("Degradation rate optimization needed")
        
        return gaps[:5]  # Limit to 5 gaps
    
    def _generate_competitive_analysis(self, context: RAGAnalysisContext,
                                     research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competitive analysis against similar materials."""
        
        similar_materials = research_results.get("similar_materials", [])
        
        if not similar_materials or not context.computed_properties:
            return {"analysis": "No competitive materials identified for comparison"}
        
        analysis = {
            "competing_materials": [m["name"] for m in similar_materials[:5]],
            "competitive_advantages": [],
            "areas_for_improvement": [],
            "market_position": "competitive"  # Default
        }
        
        # Analyze competitive position
        perf_score = context.computed_properties.performance_score
        
        if perf_score > 0.7:
            analysis["competitive_advantages"].append("Strong overall performance")
            analysis["market_position"] = "leading"
        elif perf_score > 0.5:
            analysis["market_position"] = "competitive"
        else:
            analysis["market_position"] = "developing"
            analysis["areas_for_improvement"].append("Overall performance enhancement needed")
        
        return analysis
    
    async def _generate_improvement_suggestions(self, context: RAGAnalysisContext,
                                              benchmark_results: Dict[str, Any],
                                              decision_engine: LLMDecisionEngine) -> List[str]:
        """Generate specific improvement suggestions using LLM analysis."""
        
        if not self.llm:
            return self._generate_basic_suggestions(context)
        
        try:
            # Prepare improvement generation data
            performance_analysis = benchmark_results.get("comparison_summary", "")
            identified_weaknesses = benchmark_results.get("performance_gaps", [])
            
            target_improvements = []
            for prop, target in context.target_properties.items():
                current_achievement = 0.5  # Default
                if context.computed_properties:
                    current_achievement = context.computed_properties.performance_score
                
                if current_achievement < target:
                    target_improvements.append(f"Improve {prop} from {current_achievement:.2f} to {target:.2f}")
            
            # Generate improvements
            improvement_response = await self.llm.ainvoke(
                self.improvement_prompt.format_messages(
                    performance_analysis=performance_analysis,
                    identified_weaknesses=json.dumps(identified_weaknesses),
                    target_improvements=json.dumps(target_improvements),
                    iteration_history=f"Iteration {context.iteration}"
                )
            )
            
            # Parse improvements from response
            improvements = self._parse_improvement_suggestions(improvement_response.content)
            return improvements
            
        except Exception as e:
            logger.warning(f"LLM improvement generation failed: {e}")
            return self._generate_basic_suggestions(context)
    
    def _parse_improvement_suggestions(self, llm_response: str) -> List[str]:
        """Parse improvement suggestions from LLM response."""
        
        # Simple parsing - look for numbered lists or bullet points
        lines = llm_response.split('\n')
        suggestions = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the suggestion
                suggestion = line.lstrip('0123456789.-• ').strip()
                if suggestion and len(suggestion) > 10:  # Meaningful suggestion
                    suggestions.append(suggestion)
        
        # If no structured suggestions found, try to extract key sentences
        if not suggestions:
            sentences = llm_response.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in 
                                            ['increase', 'decrease', 'improve', 'modify', 'add', 'use', 'consider']):
                    suggestions.append(sentence)
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _generate_basic_suggestions(self, context: RAGAnalysisContext) -> List[str]:
        """Generate basic improvement suggestions when LLM is not available."""
        
        suggestions = []
        
        if context.computed_properties:
            perf_score = context.computed_properties.performance_score
            
            if perf_score < 0.6:
                suggestions.append("Consider increasing molecular weight to improve mechanical properties")
                suggestions.append("Add hydrophilic groups to enhance biocompatibility")
                suggestions.append("Optimize crosslinking density for better stability")
        
        # Target property specific suggestions
        for prop in context.target_properties.keys():
            if "biocompatib" in prop.lower():
                suggestions.append("Use biocompatible monomers like ethylene glycol or lactic acid")
            elif "degradation" in prop.lower():
                suggestions.append("Adjust ester linkage density to control degradation rate")
            elif "mechanical" in prop.lower():
                suggestions.append("Increase crystallinity or add reinforcing groups")
        
        return suggestions[:5]
    
    async def _create_next_iteration_plan(self, context: RAGAnalysisContext,
                                        improvement_suggestions: List[str],
                                        decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Create detailed plan for next iteration including prompt generation."""
        
        # Generate next iteration strategy decision
        strategy_decision = decision_engine.make_decision(
            decision_type=DecisionType.NEXT_ITERATION_STRATEGY,
            context_data=context.to_dict(),
            available_options=["incremental_improvement", "targeted_optimization", "major_redesign"],
            objectives=["Plan most effective next iteration approach"],
            constraints={"build_on_successes": True, "address_weaknesses": True}
        )
        
        # Identify successful elements
        successful_elements = self._identify_successful_elements(context)
        
        # Generate next iteration prompt
        next_prompt = await self._generate_next_iteration_prompt(
            context, improvement_suggestions, successful_elements, strategy_decision.chosen_option
        )
        
        return {
            "strategy": strategy_decision.chosen_option,
            "next_prompt": next_prompt,
            "successful_elements": successful_elements,
            "priority_improvements": improvement_suggestions[:3],  # Top 3
            "iteration_goals": self._define_iteration_goals(context, improvement_suggestions),
            "expected_improvements": self._predict_expected_improvements(context, improvement_suggestions)
        }
    
    def _identify_successful_elements(self, context: RAGAnalysisContext) -> List[str]:
        """Identify successful elements from current iteration."""
        
        successful_elements = []
        
        if context.computed_properties:
            perf_score = context.computed_properties.performance_score
            
            if perf_score > 0.6:
                successful_elements.append("Overall design approach shows promise")
            
            # Check individual property successes
            if context.computed_properties.mechanical_properties:
                successful_elements.append("Mechanical properties successfully calculated")
            
            if context.computed_properties.thermal_properties:
                successful_elements.append("Thermal characterization completed")
            
            if getattr(context.computed_properties, 'confidence_level', 0) > 0.7:
                successful_elements.append("High confidence in property predictions")
        
        return successful_elements
    
    def _define_iteration_goals(self, context: RAGAnalysisContext,
                              improvement_suggestions: List[str]) -> List[str]:
        """Define specific goals for next iteration."""
        
        goals = []
        
        # Performance improvement goal
        if context.computed_properties:
            current_score = context.computed_properties.performance_score
            target_score = min(current_score + 0.1, 1.0)  # 10% improvement
            goals.append(f"Achieve performance score of {target_score:.2f} or higher")
        
        # Property-specific goals
        for prop, target in context.target_properties.items():
            goals.append(f"Optimize {prop} closer to target value of {target:.2f}")
        
        # Suggestion-based goals
        for suggestion in improvement_suggestions[:2]:  # Top 2
            if "molecular weight" in suggestion.lower():
                goals.append("Investigate higher molecular weight variants")
            elif "biocompatib" in suggestion.lower():
                goals.append("Enhance biocompatibility through monomer selection")
        
        return goals[:4]  # Limit to 4 goals
    
    def _predict_expected_improvements(self, context: RAGAnalysisContext,
                                     improvement_suggestions: List[str]) -> Dict[str, float]:
        """Predict expected improvements from suggestions."""
        
        expected_improvements = {}
        
        # Base improvement expectations
        base_improvement = 0.05  # 5% base improvement per iteration
        
        for prop in context.target_properties.keys():
            expected_improvements[prop] = base_improvement
        
        # Boost expectations based on specific suggestions
        for suggestion in improvement_suggestions:
            if "molecular weight" in suggestion.lower():
                expected_improvements["mechanical_strength"] = expected_improvements.get("mechanical_strength", base_improvement) + 0.1
            elif "biocompatib" in suggestion.lower():
                expected_improvements["biocompatibility"] = expected_improvements.get("biocompatibility", base_improvement) + 0.1
            elif "degradation" in suggestion.lower():
                expected_improvements["degradation_rate"] = expected_improvements.get("degradation_rate", base_improvement) + 0.1
        
        return expected_improvements
    
    async def _generate_next_iteration_prompt(self, context: RAGAnalysisContext,
                                            improvement_suggestions: List[str],
                                            successful_elements: List[str],
                                            strategy: str) -> str:
        """Generate detailed prompt for next iteration."""
        
        if not self.llm:
            return self._generate_basic_next_prompt(context, improvement_suggestions)
        
        try:
            # Prepare prompt generation data
            performance_summary = f"Iteration {context.iteration} achieved {context.computed_properties.performance_score:.2f} performance score" if context.computed_properties else "Previous iteration completed"
            
            improvements_needed = improvement_suggestions[:3]  # Top 3
            
            # Generate next iteration prompt
            prompt_response = await self.llm.ainvoke(
                self.next_prompt_generator.format_messages(
                    iteration_number=context.iteration + 1,
                    performance_summary=performance_summary,
                    improvements_needed=json.dumps(improvements_needed),
                    successful_elements=json.dumps(successful_elements),
                    target_properties=json.dumps(context.target_properties)
                )
            )
            
            return prompt_response.content
            
        except Exception as e:
            logger.warning(f"LLM prompt generation failed: {e}")
            return self._generate_basic_next_prompt(context, improvement_suggestions)
    
    def _generate_basic_next_prompt(self, context: RAGAnalysisContext,
                                  improvement_suggestions: List[str]) -> str:
        """Generate basic next iteration prompt."""
        
        prompt = f"""Design an improved polymer for insulin delivery (Iteration {context.iteration + 1}):

Target Properties:
{chr(10).join([f"- {prop}: {value:.2f}" for prop, value in context.target_properties.items()])}

Key Improvements Needed:
{chr(10).join([f"- {suggestion}" for suggestion in improvement_suggestions[:3]])}

Focus on creating a polymer that addresses the identified weaknesses while maintaining successful elements from previous iteration.
Consider biocompatible monomers and optimize for the target application.
"""
        
        return prompt.strip()
    
    async def _compile_rag_analysis(self, research_results: Dict[str, Any],
                                  benchmark_results: Dict[str, Any],
                                  improvement_suggestions: List[str],
                                  next_iteration_plan: Dict[str, Any],
                                  context: RAGAnalysisContext) -> RAGAnalysis:
        """Compile final RAG analysis results."""
        
        # Extract key components
        similar_materials = research_results.get("similar_materials", [])
        property_benchmarks = benchmark_results.get("benchmark_materials", [])
        
        # Create property analysis summary
        property_analysis = self._create_property_analysis_summary(
            context, benchmark_results, improvement_suggestions
        )
        
        return RAGAnalysis(
            property_analysis=property_analysis,
            similar_materials=similar_materials[:5],  # Top 5
            improvement_suggestions=improvement_suggestions,
            next_iteration_prompt=next_iteration_plan.get("next_prompt", ""),
            confidence_score=self._calculate_analysis_confidence(
                research_results, benchmark_results, context
            ),
            # Additional Phase 2 fields
            benchmark_comparison=benchmark_results.get("comparison_summary", ""),
            research_insights=research_results.get("literature_insights", [])[:3],
            iteration_strategy=next_iteration_plan.get("strategy", "incremental_improvement"),
            expected_improvements=next_iteration_plan.get("expected_improvements", {}),
            competitive_analysis=benchmark_results.get("competitive_analysis", {})
        )
    
    def _create_property_analysis_summary(self, context: RAGAnalysisContext,
                                        benchmark_results: Dict[str, Any],
                                        improvement_suggestions: List[str]) -> str:
        """Create comprehensive property analysis summary."""
        
        summary_parts = []
        
        # Performance overview
        if context.computed_properties:
            perf_score = context.computed_properties.performance_score
            summary_parts.append(f"Overall Performance: {perf_score:.2f}/1.0")
            
            # Property breakdown
            if context.computed_properties.mechanical_properties:
                summary_parts.append(f"Mechanical properties: {len(context.computed_properties.mechanical_properties)} calculated")
            
            if context.computed_properties.thermal_properties:
                summary_parts.append(f"Thermal properties: {len(context.computed_properties.thermal_properties)} calculated")
        
        # Benchmark insights
        if benchmark_results.get("performance_gaps"):
            summary_parts.append(f"Key gaps: {', '.join(benchmark_results['performance_gaps'][:2])}")
        
        # Top improvements
        if improvement_suggestions:
            summary_parts.append(f"Priority improvements: {improvement_suggestions[0]}")
        
        return "; ".join(summary_parts)
    
    def _calculate_analysis_confidence(self, research_results: Dict[str, Any],
                                     benchmark_results: Dict[str, Any],
                                     context: RAGAnalysisContext) -> float:
        """Calculate confidence score for RAG analysis."""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on available data
        if research_results.get("similar_materials"):
            confidence += 0.1
        
        if research_results.get("literature_insights"):
            confidence += 0.1
        
        if benchmark_results.get("comparison_summary") and "failed" not in benchmark_results["comparison_summary"].lower():
            confidence += 0.2
        
        if context.computed_properties and context.computed_properties.performance_score > 0.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _create_fallback_analysis(self, context: RAGAnalysisContext) -> RAGAnalysis:
        """Create fallback RAG analysis when main analysis fails."""
        
        return RAGAnalysis(
            property_analysis=f"Basic analysis for iteration {context.iteration}",
            similar_materials=[{"name": "PEG", "similarity": 0.5}],
            improvement_suggestions=["Consider optimizing polymer composition"],
            next_iteration_prompt=f"Continue polymer development for iteration {context.iteration + 1}",
            confidence_score=0.3
        )
    
    def _extract_previous_analyses(self, state: IterationState) -> List[Dict]:
        """Extract previous RAG analyses from state history."""
        # This would extract analyses from previous iterations
        return []
    
    def _save_results(self, iteration: int, results: RAGAnalysis):
        """Save RAG analysis results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_rag_analysis.json"
            with open(results_file, 'w') as f:
                # Create serializable dict
                results_dict = {
                    "property_analysis": results.property_analysis,
                    "similar_materials": results.similar_materials,
                    "improvement_suggestions": results.improvement_suggestions,
                    "next_iteration_prompt": results.next_iteration_prompt,
                    "confidence_score": results.confidence_score
                }
                
                # Add additional fields if they exist
                if hasattr(results, 'benchmark_comparison'):
                    results_dict["benchmark_comparison"] = results.benchmark_comparison
                if hasattr(results, 'competitive_analysis'):
                    results_dict["competitive_analysis"] = results.competitive_analysis
                
                json.dump(results_dict, f, indent=2)
            logger.info(f"RAG analysis results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save RAG analysis results: {e}")


# Test functionality
async def test_rag_property_analyzer():
    """Test the RAGPropertyAnalyzer functionality."""
    print("Testing RAGPropertyAnalyzer...")
    
    # Import required components
    from .state_manager import StateManager
    from .decision_engine import LLMDecisionEngine
    
    # Create test components
    state_manager = StateManager("test_rag_analyzer")
    decision_engine = LLMDecisionEngine()
    rag_analyzer = RAGPropertyAnalyzer("test_rag_output")
    
    # Create test iteration state with computed properties
    state = state_manager.create_new_iteration(
        initial_prompt="Design a biodegradable polymer for insulin delivery",
        target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
    )
    
    # Add mock computed properties
    from .state_manager import ComputedProperties
    state.computed_properties = ComputedProperties(
        md_properties=None,
        target_scores=None,
        mechanical_properties={"young_modulus": 2.5, "tensile_strength": 45.0},
        thermal_properties={"glass_transition": 60.0},
        transport_properties={"diffusion_coefficient": 1e-8},
        stability_metrics={"degradation_rate": 0.4},
        performance_score=0.65
    )
    
    print(f"Created test iteration {state.iteration_number}")
    
    # Run RAG analysis
    results = await rag_analyzer.run_automated_analysis(state, decision_engine)
    
    print(f"RAG analysis results:")
    print(f"- Property analysis: {len(results.property_analysis)} characters")
    print(f"- Similar materials: {len(results.similar_materials)}")
    print(f"- Improvement suggestions: {len(results.improvement_suggestions)}")
    print(f"- Next prompt length: {len(results.next_iteration_prompt)} characters")
    print(f"- Confidence score: {results.confidence_score:.3f}")
    
    if hasattr(results, 'competitive_analysis'):
        print(f"- Competitive analysis available: {bool(results.competitive_analysis)}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_rag_property_analyzer()) 