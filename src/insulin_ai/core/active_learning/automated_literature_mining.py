#!/usr/bin/env python3
"""
AutomatedLiteratureMining - Phase 2 Implementation

This module provides automated literature mining with LLM-powered decision making
for the active learning material discovery system. It integrates existing
literature mining components with intelligent automation.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

# Import existing literature mining systems
try:
    from ..literature_mining_system import MaterialsLiteratureMiner
    MATERIALS_MINER_AVAILABLE = True
except ImportError:
    MATERIALS_MINER_AVAILABLE = False
    logging.warning("MaterialsLiteratureMiner not available")

try:
    from ...integration.rag_literature_mining import RAGLiteratureMiningSystem
    RAG_MINER_AVAILABLE = True
except ImportError:
    RAG_MINER_AVAILABLE = False
    logging.warning("RAGLiteratureMiningSystem not available")

try:
    from ..iterative_literature_mining import IterativeLiteratureMiner
    ITERATIVE_MINER_AVAILABLE = True
except ImportError:
    ITERATIVE_MINER_AVAILABLE = False
    logging.warning("IterativeLiteratureMiner not available")

# Import active learning infrastructure
from .state_manager import IterationState, LiteratureResults
from .decision_engine import LLMDecisionEngine, DecisionType

logger = logging.getLogger(__name__)


class LiteratureSearchContext:
    """Context data for literature search decisions."""
    
    def __init__(self, iteration: int, target_properties: Dict[str, float],
                 previous_queries: List[str] = None, previous_results: List[Dict] = None):
        self.iteration = iteration
        self.target_properties = target_properties or {}
        self.previous_queries = previous_queries or []
        self.previous_results = previous_results or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for decision engine."""
        return {
            "iteration": self.iteration,
            "target_properties": self.target_properties,
            "previous_queries": self.previous_queries,
            "previous_results_count": len(self.previous_results),
            "timestamp": self.timestamp.isoformat()
        }


class AutomatedLiteratureMining:
    """
    Automated literature mining with LLM-powered decision making.
    
    This class integrates existing literature mining systems with intelligent
    automation for query generation, database selection, relevance scoring,
    and information extraction.
    """
    
    def __init__(self, storage_path: str = "automated_literature_mining"):
        """Initialize automated literature mining system.
        
        Args:
            storage_path: Path to store literature mining data and results
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize literature mining systems
        self._initialize_mining_systems()
        
        # Cache for results and decisions
        self._results_cache = {}
        self._decision_cache = {}
        
        logger.info("AutomatedLiteratureMining initialized")
    
    def _initialize_mining_systems(self):
        """Initialize available literature mining systems."""
        # Initialize Materials Literature Miner
        if MATERIALS_MINER_AVAILABLE:
            try:
                self.materials_miner = MaterialsLiteratureMiner(
                    model_type="openai",
                    openai_model="gpt-4o-mini",  # More cost-effective for literature mining
                    temperature=0.3  # Lower temperature for more focused results
                )
                logger.info("MaterialsLiteratureMiner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MaterialsLiteratureMiner: {e}")
                self.materials_miner = None
        else:
            self.materials_miner = None
        
        # Initialize RAG Literature Mining System
        if RAG_MINER_AVAILABLE:
            try:
                self.rag_miner = RAGLiteratureMiningSystem()
                logger.info("RAGLiteratureMiningSystem initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAGLiteratureMiningSystem: {e}")
                self.rag_miner = None
        else:
            self.rag_miner = None
        
        # Initialize Iterative Literature Miner
        if ITERATIVE_MINER_AVAILABLE:
            try:
                self.iterative_miner = IterativeLiteratureMiner()
                logger.info("IterativeLiteratureMiner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IterativeLiteratureMiner: {e}")
                self.iterative_miner = None
        else:
            self.iterative_miner = None
    
    async def run_automated_mining(self, state: IterationState, 
                                 decision_engine: LLMDecisionEngine) -> LiteratureResults:
        """
        Run automated literature mining with LLM decision making.
        
        Args:
            state: Current iteration state
            decision_engine: LLM decision engine for automation
            
        Returns:
            LiteratureResults: Comprehensive literature mining results
        """
        logger.info(f"Starting automated literature mining for iteration {state.iteration_number}")
        
        try:
            # Step 1: Generate optimized search queries
            search_context = LiteratureSearchContext(
                iteration=state.iteration_number,
                target_properties=state.target_properties,
                previous_queries=self._extract_previous_queries(state),
                previous_results=self._extract_previous_results(state)
            )
            
            queries = await self._generate_search_queries(search_context, decision_engine)
            
            # Step 2: Select optimal databases and search strategies
            search_strategy = await self._select_search_strategy(search_context, decision_engine)
            
            # Step 3: Execute literature search
            raw_results = await self._execute_literature_search(queries, search_strategy)
            
            # Step 4: Apply automated relevance scoring and filtering
            filtered_results = await self._filter_and_score_results(
                raw_results, search_context, decision_engine
            )
            
            # Step 5: Extract key information and insights
            final_results = await self._extract_key_information(
                filtered_results, search_context, decision_engine
            )
            
            # Step 6: Save results and update cache
            self._save_results(state.iteration_number, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in automated literature mining: {e}")
            # Return minimal results to prevent pipeline failure
            return LiteratureResults(
                papers_found=0,
                relevant_papers=0,
                extracted_properties={},
                synthesis_routes=[],
                query_used=f"Error: {str(e)}"
            )
    
    async def _generate_search_queries(self, context: LiteratureSearchContext, 
                                     decision_engine: LLMDecisionEngine) -> List[str]:
        """Generate optimized search queries using LLM decision making."""
        
        # Prepare query generation options
        query_objectives = [
            "Find materials for insulin delivery",
            "Discover polymer stabilization mechanisms",
            "Identify biocompatible material synthesis",
            "Research thermal stability enhancement",
            "Explore drug delivery system innovations"
        ]
        
        # Generate query decision
        query_decision = decision_engine.make_decision(
            decision_type=DecisionType.LITERATURE_SEARCH_QUERY,
            context_data=context.to_dict(),
            available_options=query_objectives,
            objectives=["Find most relevant literature for current iteration"],
            constraints={"iteration_focus": True, "avoid_repetition": True}
        )
        
        # Use materials miner for advanced query generation if available
        if self.materials_miner:
            try:
                base_query = f"Materials for insulin delivery with properties: {context.target_properties}"
                generated_queries = self.materials_miner.generate_search_queries(
                    user_request=base_query,
                    iteration_context={"iteration": context.iteration}
                )
                logger.info(f"Generated {len(generated_queries)} search queries")
                return generated_queries
            except Exception as e:
                logger.warning(f"Failed to generate queries with MaterialsLiteratureMiner: {e}")
        
        # Fallback to basic query generation
        basic_queries = [
            f"insulin delivery polymer {query_decision.chosen_option}",
            "biocompatible polymer drug delivery",
            "insulin stabilization materials",
            f"polymer {list(context.target_properties.keys())[0] if context.target_properties else 'properties'}"
        ]
        
        return basic_queries[:3]  # Limit to 3 queries for efficiency
    
    async def _select_search_strategy(self, context: LiteratureSearchContext,
                                    decision_engine: LLMDecisionEngine) -> Dict[str, Any]:
        """Select optimal search strategy using LLM decision making."""
        
        strategy_options = [
            "comprehensive_search",  # Search multiple databases thoroughly
            "focused_search",        # Search specific high-quality sources
            "rapid_search",          # Quick search for immediate insights
            "iterative_search"       # Build on previous iteration results
        ]
        
        strategy_decision = decision_engine.make_decision(
            decision_type=DecisionType.LITERATURE_SEARCH_STRATEGY,
            context_data=context.to_dict(),
            available_options=strategy_options,
            objectives=["Balance thoroughness with efficiency"],
            constraints={"time_limit": "moderate", "quality_threshold": "high"}
        )
        
        return {
            "strategy": strategy_decision.chosen_option,
            "databases": ["semantic_scholar", "pubmed"] if "comprehensive" in strategy_decision.chosen_option else ["semantic_scholar"],
            "max_papers": 50 if "comprehensive" in strategy_decision.chosen_option else 20,
            "relevance_threshold": 0.7
        }
    
    async def _execute_literature_search(self, queries: List[str], 
                                       strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute literature search using available mining systems."""
        
        all_results = {
            "papers": [],
            "total_found": 0,
            "search_metadata": {
                "queries_used": queries,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Try RAG mining system first (most advanced)
        if self.rag_miner:
            try:
                for query in queries:
                    rag_results = await self._execute_rag_search(query, strategy)
                    all_results["papers"].extend(rag_results.get("papers", []))
                    all_results["total_found"] += rag_results.get("count", 0)
                logger.info(f"RAG mining found {len(all_results['papers'])} papers")
            except Exception as e:
                logger.warning(f"RAG mining failed: {e}")
        
        # Try materials miner if RAG didn't produce enough results
        if self.materials_miner and len(all_results["papers"]) < 10:
            try:
                for query in queries:
                    materials_results = await self._execute_materials_search(query, strategy)
                    all_results["papers"].extend(materials_results.get("papers", []))
                    all_results["total_found"] += materials_results.get("count", 0)
                logger.info(f"Materials mining found additional {len(materials_results.get('papers', []))} papers")
            except Exception as e:
                logger.warning(f"Materials mining failed: {e}")
        
        # Remove duplicates
        unique_papers = []
        seen_titles = set()
        for paper in all_results["papers"]:
            title = paper.get("title", "").lower().strip()
            if title and title not in seen_titles:
                unique_papers.append(paper)
                seen_titles.add(title)
        
        all_results["papers"] = unique_papers
        all_results["unique_count"] = len(unique_papers)
        
        logger.info(f"Literature search completed: {len(unique_papers)} unique papers found")
        return all_results
    
    async def _execute_rag_search(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search using RAG literature mining system."""
        # This would be implemented based on the RAG system's API
        # For now, return simulated results
        return {
            "papers": [],
            "count": 0
        }
    
    async def _execute_materials_search(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search using materials literature miner."""
        try:
            # Run search with materials miner
            search_results = self.materials_miner.search_papers(
                query=query,
                max_results=strategy.get("max_papers", 20)
            )
            
            return {
                "papers": search_results.get("papers", []),
                "count": len(search_results.get("papers", []))
            }
        except Exception as e:
            logger.warning(f"Materials search failed for query '{query}': {e}")
            return {"papers": [], "count": 0}
    
    async def _filter_and_score_results(self, raw_results: Dict[str, Any],
                                      context: LiteratureSearchContext,
                                      decision_engine: LLMDecisionEngine) -> List[Dict[str, Any]]:
        """Apply automated relevance scoring and filtering."""
        
        papers = raw_results.get("papers", [])
        if not papers:
            return []
        
        # Generate filtering criteria decision
        filter_decision = decision_engine.make_decision(
            decision_type=DecisionType.LITERATURE_FILTERING_CRITERIA,
            context_data=context.to_dict(),
            available_options=["strict", "moderate", "lenient"],
            objectives=["Select high-quality relevant papers"],
            constraints={"min_papers": 5, "max_papers": 20}
        )
        
        relevance_threshold = {
            "strict": 0.8,
            "moderate": 0.6,
            "lenient": 0.4
        }.get(filter_decision.chosen_option, 0.6)
        
        # Score and filter papers
        scored_papers = []
        for paper in papers:
            try:
                # Calculate relevance score
                score = self._calculate_relevance_score(paper, context)
                
                if score >= relevance_threshold:
                    paper["relevance_score"] = score
                    scored_papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to score paper: {e}")
        
        # Sort by relevance score
        scored_papers.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)
        
        # Limit results based on strategy
        max_papers = filter_decision.constraints.get("max_papers", 20)
        return scored_papers[:max_papers]
    
    def _calculate_relevance_score(self, paper: Dict[str, Any], 
                                 context: LiteratureSearchContext) -> float:
        """Calculate relevance score for a paper."""
        
        # Basic scoring based on title and abstract keywords
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Keywords for insulin delivery and polymers
        keywords = {
            "insulin": 0.3,
            "delivery": 0.2,
            "polymer": 0.2,
            "biocompatible": 0.1,
            "drug": 0.1,
            "transdermal": 0.1,
            "patch": 0.1,
            "stabilization": 0.1,
            "hydrogel": 0.1,
            "controlled release": 0.2
        }
        
        score = 0.0
        for keyword, weight in keywords.items():
            if keyword in text:
                score += weight
        
        # Boost score for target properties
        for prop in context.target_properties.keys():
            if prop.lower() in text:
                score += 0.1
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    async def _extract_key_information(self, filtered_papers: List[Dict[str, Any]],
                                     context: LiteratureSearchContext,
                                     decision_engine: LLMDecisionEngine) -> LiteratureResults:
        """Extract key information and create final results."""
        
        # Generate extraction strategy decision
        extraction_decision = decision_engine.make_decision(
            decision_type=DecisionType.INFORMATION_EXTRACTION_PRIORITY,
            context_data=context.to_dict(),
            available_options=["materials_focus", "properties_focus", "synthesis_focus", "comprehensive"],
            objectives=["Extract most relevant information for current iteration"]
        )
        
        # Extract information based on strategy
        extracted_properties = {}
        synthesis_routes = []
        material_candidates = []
        
        for paper in filtered_papers:
            try:
                # Extract properties
                properties = self._extract_properties_from_paper(paper, context.target_properties)
                for prop, values in properties.items():
                    if prop not in extracted_properties:
                        extracted_properties[prop] = []
                    extracted_properties[prop].extend(values)
                
                # Extract synthesis routes
                routes = self._extract_synthesis_routes(paper)
                synthesis_routes.extend(routes)
                
                # Extract material candidates
                materials = self._extract_material_candidates(paper)
                material_candidates.extend(materials)
                
            except Exception as e:
                logger.warning(f"Failed to extract information from paper: {e}")
        
        # Create final results
        return LiteratureResults(
            papers_found=len(filtered_papers),
            relevant_papers=len([p for p in filtered_papers if p.get("relevance_score", 0) > 0.7]),
            extracted_properties=extracted_properties,
            synthesis_routes=list(set(synthesis_routes)),  # Remove duplicates
            query_used="; ".join(context.previous_queries[-3:] if context.previous_queries else ["automated query"]),
            material_candidates=material_candidates[:10],  # Limit to top 10
            extraction_strategy=extraction_decision.chosen_option,
            papers_analyzed=[{
                "title": p.get("title", ""),
                "relevance_score": p.get("relevance_score", 0),
                "year": p.get("year", "unknown")
            } for p in filtered_papers[:10]]
        )
    
    def _extract_properties_from_paper(self, paper: Dict[str, Any], 
                                     target_properties: Dict[str, float]) -> Dict[str, List[float]]:
        """Extract numerical properties from paper."""
        # Simplified property extraction
        # In a real implementation, this would use NLP to extract numerical values
        properties = {}
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Mock property extraction based on keywords
        if "biocompatible" in text:
            properties["biocompatibility"] = [0.8, 0.9]
        if "degradation" in text:
            properties["degradation_rate"] = [0.3, 0.4, 0.5]
        if "mechanical" in text:
            properties["mechanical_strength"] = [2.5, 3.0]
        
        return properties
    
    def _extract_synthesis_routes(self, paper: Dict[str, Any]) -> List[str]:
        """Extract synthesis routes from paper."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        routes = []
        synthesis_keywords = {
            "polymerization": "polymerization",
            "grafting": "grafting",
            "crosslinking": "crosslinking",
            "electrospinning": "electrospinning",
            "sol-gel": "sol-gel synthesis",
            "emulsion": "emulsion polymerization"
        }
        
        for keyword, route in synthesis_keywords.items():
            if keyword in text:
                routes.append(route)
        
        return routes
    
    def _extract_material_candidates(self, paper: Dict[str, Any]) -> List[str]:
        """Extract material candidates from paper."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        materials = []
        material_keywords = [
            "PEG", "poly(ethylene glycol)",
            "PLGA", "poly(lactic-co-glycolic acid)",
            "chitosan", "alginate", "hyaluronic acid",
            "collagen", "gelatin", "fibrin",
            "PDMS", "poly(dimethylsiloxane)"
        ]
        
        for material in material_keywords:
            if material.lower() in text:
                materials.append(material)
        
        return materials
    
    def _extract_previous_queries(self, state: IterationState) -> List[str]:
        """Extract previous search queries from state history."""
        # This would extract queries from previous iterations
        return []
    
    def _extract_previous_results(self, state: IterationState) -> List[Dict]:
        """Extract previous literature results from state history."""
        # This would extract results from previous iterations
        return []
    
    def _save_results(self, iteration: int, results: LiteratureResults):
        """Save results to storage."""
        try:
            results_file = self.storage_path / f"iteration_{iteration}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")


# Test functionality
async def test_automated_literature_mining():
    """Test the AutomatedLiteratureMining functionality."""
    print("Testing AutomatedLiteratureMining...")
    
    # Import required components
    from .state_manager import StateManager
    from .decision_engine import LLMDecisionEngine
    
    # Create test components
    state_manager = StateManager("test_automated_literature")
    decision_engine = LLMDecisionEngine()
    literature_miner = AutomatedLiteratureMining("test_literature_output")
    
    # Create test iteration state
    state = state_manager.create_new_iteration(
        initial_prompt="Design a biodegradable polymer for insulin delivery",
        target_properties={"biocompatibility": 0.9, "degradation_rate": 0.5}
    )
    
    print(f"Created test iteration {state.iteration_number}")
    
    # Run automated literature mining
    results = await literature_miner.run_automated_mining(state, decision_engine)
    
    print(f"Literature mining results:")
    print(f"- Papers found: {results.papers_found}")
    print(f"- Relevant papers: {results.relevant_papers}")
    print(f"- Properties extracted: {len(results.extracted_properties)}")
    print(f"- Synthesis routes: {len(results.synthesis_routes)}")
    print(f"- Query used: {results.query_used}")
    
    return results


if __name__ == "__main__":
    asyncio.run(test_automated_literature_mining()) 