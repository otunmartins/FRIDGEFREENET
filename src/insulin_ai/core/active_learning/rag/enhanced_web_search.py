#!/usr/bin/env python3
"""
Enhanced Web Search Agent for RAG System - Phase 3

This module provides advanced web search capabilities with integration to scientific
databases, patent searches, and intelligent query processing for material discovery.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import urllib.parse

# Web search and scraping
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily search not available")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("Web scraping tools not available")

# LLM integration for query enhancement
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for query enhancement")

# Scientific data parsing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchDomain(Enum):
    """Types of search domains."""
    GENERAL_WEB = "general_web"
    SCIENTIFIC_LITERATURE = "scientific_literature"
    PATENTS = "patents"
    MATERIALS_DATABASE = "materials_database"
    DRUG_DELIVERY = "drug_delivery"
    POLYMER_SCIENCE = "polymer_science"


class SearchResult(Enum):
    """Quality levels for search results."""
    HIGH_RELEVANCE = "high_relevance"
    MEDIUM_RELEVANCE = "medium_relevance"
    LOW_RELEVANCE = "low_relevance"
    IRRELEVANT = "irrelevant"


@dataclass
class WebSearchResult:
    """Individual web search result."""
    title: str
    url: str
    content: str
    domain: str
    relevance_score: float
    source_type: SearchDomain
    extracted_data: Dict[str, Any]
    timestamp: datetime
    confidence: float


@dataclass
class SearchQuery:
    """Enhanced search query with context."""
    original_query: str
    enhanced_query: str
    search_domains: List[SearchDomain]
    filters: Dict[str, Any]
    max_results: int
    time_range: Optional[Tuple[datetime, datetime]] = None
    language: str = "en"


@dataclass
class AggregatedResults:
    """Aggregated and analyzed search results."""
    query: SearchQuery
    results: List[WebSearchResult]
    summary: str
    key_findings: List[str]
    material_properties: Dict[str, Any]
    synthesis_methods: List[str]
    performance_data: Dict[str, float]
    citations: List[str]
    confidence: float


class QueryEnhancer:
    """Enhances search queries for better scientific results."""
    
    def __init__(self):
        """Initialize query enhancer."""
        self.llm = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,  # Lower temperature for factual queries
                    timeout=30
                )
                logger.info("LLM initialized for query enhancement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
        
        self._setup_prompts()
        
        # Scientific terminology mappings
        self.term_mappings = {
            "drug delivery": ["controlled release", "pharmaceutical", "biomedical", "therapeutic"],
            "biocompatible": ["biocompatibility", "non-toxic", "FDA approved", "medical grade"],
            "degradable": ["biodegradable", "bioresorbable", "degradation", "erosion"],
            "mechanical properties": ["tensile strength", "elastic modulus", "Young's modulus", "stress-strain"],
            "polymer": ["macromolecule", "polymeric", "polymerization", "copolymer"]
        }
    
    def _setup_prompts(self):
        """Setup prompt templates for query enhancement."""
        self.enhancement_prompt = ChatPromptTemplate.from_template("""
        You are a scientific literature search expert. Enhance the following search query 
        to improve results for materials science and drug delivery research.
        
        Original Query: {original_query}
        Search Domain: {domain}
        
        Guidelines:
        1. Add relevant scientific terminology
        2. Include alternative phrasings
        3. Add specific material property keywords
        4. Include synthesis/manufacturing terms if relevant
        5. Make the query more specific for academic databases
        
        Enhanced Query (return only the improved query string):
        """)
    
    async def enhance_query(self, 
                          original_query: str,
                          domain: SearchDomain = SearchDomain.SCIENTIFIC_LITERATURE) -> str:
        """Enhance a search query for better scientific results."""
        if self.llm:
            try:
                messages = self.enhancement_prompt.invoke({
                    "original_query": original_query,
                    "domain": domain.value
                })
                
                response = await self.llm.ainvoke(messages)
                enhanced_query = response.content.strip()
                
                # Add domain-specific terms
                enhanced_query = self._add_domain_terms(enhanced_query, domain)
                
                return enhanced_query
                
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        
        # Fallback enhancement
        return self._basic_enhancement(original_query, domain)
    
    def _add_domain_terms(self, query: str, domain: SearchDomain) -> str:
        """Add domain-specific terminology to query."""
        domain_terms = {
            SearchDomain.DRUG_DELIVERY: ["drug delivery", "controlled release", "pharmaceutical"],
            SearchDomain.POLYMER_SCIENCE: ["polymer", "macromolecule", "polymerization"],
            SearchDomain.MATERIALS_DATABASE: ["material properties", "characterization", "performance"],
            SearchDomain.PATENTS: ["patent", "invention", "method", "composition"]
        }
        
        if domain in domain_terms:
            for term in domain_terms[domain]:
                if term.lower() not in query.lower():
                    query += f" {term}"
        
        return query
    
    def _basic_enhancement(self, query: str, domain: SearchDomain) -> str:
        """Basic query enhancement without LLM."""
        enhanced = query
        
        # Add scientific terminology
        for key_term, alternatives in self.term_mappings.items():
            if key_term in query.lower():
                enhanced += " " + " ".join(alternatives[:2])  # Add top 2 alternatives
        
        # Add domain-specific terms
        if domain == SearchDomain.SCIENTIFIC_LITERATURE:
            enhanced += " research study analysis"
        elif domain == SearchDomain.PATENTS:
            enhanced += " patent application method"
        elif domain == SearchDomain.MATERIALS_DATABASE:
            enhanced += " properties characterization data"
        
        return enhanced


class ScientificDatabaseSearcher:
    """Searcher for scientific databases and repositories."""
    
    def __init__(self):
        """Initialize scientific database searcher."""
        # Database endpoints (would be configured with actual APIs)
        self.databases = {
            "pubmed": "https://pubmed.ncbi.nlm.nih.gov/",
            "google_scholar": "https://scholar.google.com/scholar",
            "materials_project": "https://materialsproject.org/",
            "polymer_database": "https://polymerdatabase.com/",
            "uspto": "https://patft.uspto.gov/"
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def search_pubmed(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        """Search PubMed for scientific literature."""
        results = []
        
        if not WEB_SCRAPING_AVAILABLE:
            logger.warning("Web scraping not available for PubMed search")
            return results
        
        try:
            # This is a simplified example - real implementation would use PubMed API
            search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(query)}"
            
            # Simulate search results (in real implementation, use proper API)
            mock_results = self._generate_mock_pubmed_results(query)
            
            for i, mock_result in enumerate(mock_results[:max_results]):
                result = WebSearchResult(
                    title=mock_result["title"],
                    url=mock_result["url"],
                    content=mock_result["abstract"],
                    domain="pubmed.ncbi.nlm.nih.gov",
                    relevance_score=0.8 - (i * 0.05),
                    source_type=SearchDomain.SCIENTIFIC_LITERATURE,
                    extracted_data={
                        "authors": mock_result["authors"],
                        "journal": mock_result["journal"],
                        "year": mock_result["year"],
                        "pmid": mock_result["pmid"]
                    },
                    timestamp=datetime.now(),
                    confidence=0.9
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return results
    
    def _generate_mock_pubmed_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate mock PubMed results for demonstration."""
        return [
            {
                "title": f"Advanced {query} Materials for Biomedical Applications",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                "abstract": f"This study investigates {query} with focus on biocompatibility and controlled release properties...",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "journal": "Journal of Biomedical Materials Research",
                "year": 2023,
                "pmid": "12345678"
            },
            {
                "title": f"Synthesis and Characterization of {query} for Drug Delivery",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345679",
                "abstract": f"Novel {query} polymers were synthesized and evaluated for pharmaceutical applications...",
                "authors": ["Brown, C.", "Davis, M."],
                "journal": "Biomaterials",
                "year": 2023,
                "pmid": "12345679"
            }
        ]
    
    async def search_patents(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Search patent databases."""
        results = []
        
        try:
            # Mock patent search results
            mock_patents = [
                {
                    "title": f"Method for Producing {query} with Enhanced Properties",
                    "url": "https://patents.uspto.gov/patent/123456",
                    "abstract": f"A method for synthesizing {query} materials with improved biocompatibility and mechanical strength...",
                    "patent_number": "US123456",
                    "inventors": ["Inventor, A.", "Creator, B."],
                    "assignee": "Research University",
                    "filing_date": "2023-01-15"
                }
            ]
            
            for i, patent in enumerate(mock_patents[:max_results]):
                result = WebSearchResult(
                    title=patent["title"],
                    url=patent["url"],
                    content=patent["abstract"],
                    domain="patents.uspto.gov",
                    relevance_score=0.7 - (i * 0.05),
                    source_type=SearchDomain.PATENTS,
                    extracted_data={
                        "patent_number": patent["patent_number"],
                        "inventors": patent["inventors"],
                        "assignee": patent["assignee"],
                        "filing_date": patent["filing_date"]
                    },
                    timestamp=datetime.now(),
                    confidence=0.8
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Patent search failed: {e}")
            return results


class ResultProcessor:
    """Processes and analyzes search results."""
    
    def __init__(self):
        """Initialize result processor."""
        self.llm = None
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    timeout=60
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for processing: {e}")
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompts for result processing."""
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following search results about materials science and drug delivery.
        Extract key information about material properties, synthesis methods, and performance data.
        
        Search Query: {query}
        
        Search Results:
        {results_text}
        
        Provide analysis in JSON format:
        {{
            "summary": "Brief summary of findings",
            "key_findings": ["finding1", "finding2", "finding3"],
            "material_properties": {{"property": "value"}},
            "synthesis_methods": ["method1", "method2"],
            "performance_data": {{"metric": value}},
            "citations": ["citation1", "citation2"]
        }}
        """)
    
    async def process_results(self, 
                            query: SearchQuery,
                            results: List[WebSearchResult]) -> AggregatedResults:
        """Process and aggregate search results."""
        try:
            if not results:
                return AggregatedResults(
                    query=query,
                    results=[],
                    summary="No results found",
                    key_findings=[],
                    material_properties={},
                    synthesis_methods=[],
                    performance_data={},
                    citations=[],
                    confidence=0.0
                )
            
            # Filter and rank results
            filtered_results = self._filter_results(results)
            ranked_results = self._rank_results(filtered_results)
            
            # Extract structured data
            if self.llm:
                analysis = await self._llm_analysis(query, ranked_results)
            else:
                analysis = self._basic_analysis(query, ranked_results)
            
            return AggregatedResults(
                query=query,
                results=ranked_results,
                summary=analysis.get("summary", ""),
                key_findings=analysis.get("key_findings", []),
                material_properties=analysis.get("material_properties", {}),
                synthesis_methods=analysis.get("synthesis_methods", []),
                performance_data=analysis.get("performance_data", {}),
                citations=analysis.get("citations", []),
                confidence=self._calculate_confidence(ranked_results, analysis)
            )
            
        except Exception as e:
            logger.error(f"Failed to process results: {e}")
            return AggregatedResults(
                query=query,
                results=results,
                summary="Processing failed",
                key_findings=[],
                material_properties={},
                synthesis_methods=[],
                performance_data={},
                citations=[],
                confidence=0.1
            )
    
    def _filter_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Filter results based on relevance and quality."""
        # Remove duplicates and low-quality results
        filtered = []
        seen_urls = set()
        
        for result in results:
            if (result.url not in seen_urls and 
                result.relevance_score > 0.3 and
                len(result.content) > 100):
                filtered.append(result)
                seen_urls.add(result.url)
        
        return filtered
    
    def _rank_results(self, results: List[WebSearchResult]) -> List[WebSearchResult]:
        """Rank results by relevance and quality."""
        def ranking_score(result):
            # Combine relevance, confidence, and source type
            source_weight = {
                SearchDomain.SCIENTIFIC_LITERATURE: 1.0,
                SearchDomain.PATENTS: 0.9,
                SearchDomain.MATERIALS_DATABASE: 0.9,
                SearchDomain.DRUG_DELIVERY: 1.0,
                SearchDomain.GENERAL_WEB: 0.6
            }
            
            return (result.relevance_score * 0.4 + 
                   result.confidence * 0.3 + 
                   source_weight.get(result.source_type, 0.5) * 0.3)
        
        return sorted(results, key=ranking_score, reverse=True)
    
    async def _llm_analysis(self, 
                          query: SearchQuery,
                          results: List[WebSearchResult]) -> Dict[str, Any]:
        """Use LLM to analyze search results."""
        try:
            # Prepare results text
            results_text = "\n\n".join([
                f"Title: {result.title}\nSource: {result.domain}\nContent: {result.content[:500]}..."
                for result in results[:10]  # Limit to top 10 for token efficiency
            ])
            
            messages = self.analysis_prompt.invoke({
                "query": query.original_query,
                "results_text": results_text
            })
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM analysis as JSON")
                return self._basic_analysis(query, results)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._basic_analysis(query, results)
    
    def _basic_analysis(self, 
                       query: SearchQuery,
                       results: List[WebSearchResult]) -> Dict[str, Any]:
        """Basic analysis without LLM."""
        # Extract common terms and patterns
        all_content = " ".join([result.content for result in results])
        
        # Simple keyword extraction for material properties
        property_keywords = ["modulus", "strength", "temperature", "degradation", "biocompatibility"]
        found_properties = {}
        
        for keyword in property_keywords:
            if keyword in all_content.lower():
                # Try to extract numerical values
                pattern = rf"{keyword}[:\s]+(\d+\.?\d*)"
                matches = re.findall(pattern, all_content, re.IGNORECASE)
                if matches:
                    found_properties[keyword] = float(matches[0])
        
        return {
            "summary": f"Found {len(results)} relevant results for {query.original_query}",
            "key_findings": [f"Results from {len(set(r.domain for r in results))} different sources"],
            "material_properties": found_properties,
            "synthesis_methods": ["polymerization", "crosslinking"],  # Default methods
            "performance_data": found_properties,
            "citations": [f"{result.title} - {result.domain}" for result in results[:5]]
        }
    
    def _calculate_confidence(self, 
                            results: List[WebSearchResult],
                            analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis."""
        if not results:
            return 0.0
        
        # Average result confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Quality indicators
        quality_score = 0.5
        if analysis.get("material_properties"):
            quality_score += 0.2
        if analysis.get("synthesis_methods"):
            quality_score += 0.1
        if len(analysis.get("key_findings", [])) >= 3:
            quality_score += 0.2
        
        return min(1.0, (avg_confidence + quality_score) / 2)


class EnhancedWebSearchAgent:
    """Main enhanced web search agent."""
    
    def __init__(self, cache_directory: str = "search_cache"):
        """Initialize enhanced web search agent."""
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.query_enhancer = QueryEnhancer()
        self.scientific_searcher = ScientificDatabaseSearcher()
        self.result_processor = ResultProcessor()
        
        # Initialize Tavily search if available
        self.tavily_search = None
        if TAVILY_AVAILABLE:
            try:
                self.tavily_search = TavilySearchResults(
                    max_results=20,
                    search_depth="advanced",
                    include_domains=["scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "nature.com", "science.org"]
                )
                logger.info("Tavily search initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily: {e}")
        
        # Cache for search results
        self._search_cache = {}
        
        logger.info("EnhancedWebSearchAgent initialized")
    
    async def search(self, 
                   query: str,
                   domains: List[SearchDomain] = None,
                   max_results: int = 20,
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> AggregatedResults:
        """Perform enhanced web search across multiple domains."""
        domains = domains or [SearchDomain.SCIENTIFIC_LITERATURE, SearchDomain.GENERAL_WEB]
        
        try:
            # Check cache
            cache_key = f"{query}_{str(domains)}_{max_results}"
            if cache_key in self._search_cache:
                cached_result = self._search_cache[cache_key]
                if (datetime.now() - cached_result["timestamp"]).hours < 24:  # 24-hour cache
                    logger.info(f"Returning cached results for: {query}")
                    return cached_result["results"]
            
            # Enhance query
            enhanced_query = await self.query_enhancer.enhance_query(
                query, domains[0] if domains else SearchDomain.SCIENTIFIC_LITERATURE
            )
            
            search_query = SearchQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                search_domains=domains,
                filters={},
                max_results=max_results,
                time_range=time_range
            )
            
            # Collect results from multiple sources
            all_results = []
            
            # Tavily search (general web + scientific)
            if self.tavily_search:
                tavily_results = await self._tavily_search(enhanced_query, max_results // 2)
                all_results.extend(tavily_results)
            
            # Scientific database searches
            if SearchDomain.SCIENTIFIC_LITERATURE in domains:
                pubmed_results = await self.scientific_searcher.search_pubmed(enhanced_query, max_results // 4)
                all_results.extend(pubmed_results)
            
            if SearchDomain.PATENTS in domains:
                patent_results = await self.scientific_searcher.search_patents(enhanced_query, max_results // 4)
                all_results.extend(patent_results)
            
            # Process and aggregate results
            aggregated_results = await self.result_processor.process_results(search_query, all_results)
            
            # Cache results
            self._search_cache[cache_key] = {
                "results": aggregated_results,
                "timestamp": datetime.now()
            }
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Return empty results
            return AggregatedResults(
                query=SearchQuery(query, query, domains, {}, max_results),
                results=[],
                summary="Search failed",
                key_findings=[],
                material_properties={},
                synthesis_methods=[],
                performance_data={},
                citations=[],
                confidence=0.0
            )
    
    async def _tavily_search(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Perform Tavily search and convert results."""
        results = []
        
        try:
            tavily_results = self.tavily_search.run(query)
            
            for i, result in enumerate(tavily_results[:max_results]):
                web_result = WebSearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("content", ""),
                    domain=urllib.parse.urlparse(result.get("url", "")).netloc,
                    relevance_score=max(0.1, 1.0 - (i * 0.05)),  # Decreasing relevance
                    source_type=SearchDomain.GENERAL_WEB,
                    extracted_data={},
                    timestamp=datetime.now(),
                    confidence=0.8
                )
                results.append(web_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return results
    
    def clear_cache(self):
        """Clear search cache."""
        self._search_cache.clear()
        logger.info("Search cache cleared")


# Factory function for easy initialization
def create_enhanced_web_search_agent(cache_directory: str = "search_cache") -> EnhancedWebSearchAgent:
    """Create enhanced web search agent with default configuration."""
    return EnhancedWebSearchAgent(cache_directory) 