#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client for Insulin AI
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import os

# Import our local modules
from .semantic_scholar_client import SemanticScholarClient
from .ollama_client import OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
semantic_scholar_client = SemanticScholarClient()
ollama_client = OllamaClient()

async def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with basic error handling."""
    try:
        result = await asyncio.to_thread(func, *args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise e

async def evaluate_paper_relevance_with_llm(paper: Dict, query: str, timeout: int = 10) -> Dict[str, any]:
    """
    Use LLM to evaluate paper relevance instead of keyword matching.
    
    Args:
        paper: Paper dictionary with title, abstract, etc.
        query: Original search query
        timeout: Timeout in seconds for LLM call
    
    Returns:
        Dict with relevance score and reasoning
    """
    global ollama_client
    
    try:
        # Initialize ollama client if needed
        try:
            if not hasattr(ollama_client, 'client') or ollama_client.client is None:
                logger.warning("Ollama client not properly initialized, reinitializing...")
                ollama_client = OllamaClient()
        except Exception as init_error:
            logger.warning(f"Failed to initialize Ollama client: {init_error}")
            return {
                "relevance_score": 7,
                "reasoning": "LLM client initialization failed, defaulting to relevant",
                "relevant": True
            }
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        year = paper.get('year', 'Unknown')
        
        prompt = f"""
Evaluate the relevance of this research paper to the query: "{query}"

Paper details:
Title: {title}
Year: {year}
Abstract: {abstract[:500]}...

On a scale of 1-10, rate how relevant this paper is to the search query. Consider:
1. Direct relevance to the topic
2. Potential insights for the research area
3. Methodological contributions
4. Novel approaches or findings

Respond with ONLY a JSON object in this exact format:
{{
    "relevance_score": [number from 1-10],
    "reasoning": "[brief explanation of why this score was given]",
    "relevant": [true/false - true if score >= 6]
}}
"""
        
        # Add timeout to LLM call
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama_client.client.chat,
                    model=ollama_client.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3, 'num_predict': 200}
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM evaluation timed out after {timeout}s for paper: {title[:50]}...")
            return {
                "relevance_score": 7,
                "reasoning": f"LLM evaluation timed out after {timeout}s, defaulting to relevant",
                "relevant": True
            }
        except Exception as llm_error:
            logger.warning(f"LLM call failed: {llm_error}")
            return {
                "relevance_score": 7,
                "reasoning": f"LLM call failed: {str(llm_error)}, defaulting to relevant",
                "relevant": True
            }
        
        # Check if response is valid
        if not response or not response.get('message') or not response.get('message', {}).get('content'):
            logger.warning(f"Invalid LLM response for paper: {title[:50]}...")
            return {
                "relevance_score": 7,
                "reasoning": "Invalid LLM response, defaulting to relevant",
                "relevant": True
            }
        
        # Safely extract response text
        try:
            response_text = response['message']['content'].strip()
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to extract response content: {e}")
            return {
                "relevance_score": 7,
                "reasoning": "Failed to extract LLM response content, defaulting to relevant",
                "relevant": True
            }
        
        # Try to parse JSON response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning(f"Failed to parse LLM JSON response: {response_text[:100]}...")
            return {
                "relevance_score": 7,
                "reasoning": "LLM evaluation failed, using moderate relevance",
                "relevant": True
            }
            
    except Exception as e:
        logger.warning(f"LLM relevance evaluation failed: {e}")
        # Fallback to accepting the paper
        return {
            "relevance_score": 7,
            "reasoning": "LLM evaluation unavailable, defaulting to relevant",
            "relevant": True
        }

async def enhanced_search_papers(query: str, max_results: int = 20, recent_only: bool = False, use_llm_evaluation: bool = False) -> Dict[str, Any]:
    """Enhanced paper search using the same efficient approach as the original system."""
    try:
        logger.info(f"Starting enhanced search with query: {query}, max_results: {max_results}, use_llm: {use_llm_evaluation}")
        
        # Use the same efficient search method as the original system
        papers = await asyncio.to_thread(
            semantic_scholar_client.search_papers_by_topic,
            topic=query,
            max_results=max_results,
            recent_years_only=recent_only
        )
        
        logger.info(f"📚 Found {len(papers)} papers from search")
        
        # Apply LLM filtering only if requested
        if use_llm_evaluation and papers:
            filtered_papers = []
            for i, paper in enumerate(papers):
                title = paper.get('title', 'Unknown')[:50]
                logger.info(f"   🤖 Evaluating relevance with LLM: Paper {i+1}: '{title}...'")
                
                relevance_result = await evaluate_paper_relevance_with_llm(paper, query, timeout=10)
                
                if relevance_result.get('relevant', True):
                    score = relevance_result.get('relevance_score', 7)
                    reasoning = relevance_result.get('reasoning', 'Relevant')
                    logger.info(f"   ✅ Paper {i+1}: '{title}...' - Relevant (Score: {score}/10)")
                    
                    paper['llm_relevance'] = relevance_result
                    filtered_papers.append(paper)
                else:
                    score = relevance_result.get('relevance_score', 3)
                    logger.info(f"   ❌ Paper {i+1}: '{title}...' - Not relevant (Score: {score}/10)")
                
                if len(filtered_papers) >= max_results:
                    break
            
            papers = filtered_papers
        else:
            # Fast mode: Accept all papers without LLM evaluation
            for i, paper in enumerate(papers):
                title = paper.get('title', 'Unknown')[:50]
                logger.info(f"   ✅ Paper {i+1}: '{title}...' - Accepted (fast mode, no LLM evaluation)")
        
        search_mode = "Fast mode (no filtering)" if not use_llm_evaluation else "LLM evaluation mode"
        logger.info(f"✅ Search yielded {len(papers)} papers using {search_mode}")
        
        return {
            "papers": papers[:max_results],
            "total_found": len(papers),
            "query_used": query,
            "search_mode": search_mode
        }
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {str(e)}")
        return {"papers": [], "total_found": 0, "error": str(e)}

def analyze_material_properties(papers: List[Dict[str, Any]], focus_area: str = "thermal stability and biocompatibility") -> Dict[str, Any]:
    """Analyze papers to extract material properties."""
    logger.info(f"Analyzing {len(papers)} papers for material properties")
    
    try:
        materials = []
        
        for i, paper in enumerate(papers):
            logger.info(f"Analyzing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
            # Extract material information
            material_info = {
                "material_name": "Unknown Material",
                "material_composition": "Not specified",
                "thermal_stability_temp_range": "Not specified",
                "biocompatibility_data": "Not specified",
                "stabilization_mechanism": "Not specified",
                "confidence_score": 5,
                "source_paper": {
                    "title": paper.get('title', ''),
                    "paperId": paper.get('paperId', ''),
                    "year": paper.get('year', ''),
                    "url": paper.get('url', ''),
                    "citationCount": paper.get('citationCount', 0)
                }
            }
            
            # Enhanced keyword-based extraction
            title = paper.get('title', '') or ''
            abstract = paper.get('abstract', '') or ''
            text = f"{title.lower()} {abstract.lower()}"
            
            # Material identification patterns
            material_patterns = {
                "PLGA": ["plga", "poly(lactic-co-glycolic acid)", "polylactic"],
                "Chitosan": ["chitosan", "chitin"],
                "PEG": ["peg", "polyethylene glycol"],
                "Alginate": ["alginate", "sodium alginate"],
                "Hydrogel": ["hydrogel", "gel matrix"],
                "Nanoparticles": ["nanoparticle", "nano particle", "nanosphere"],
                "Liposomes": ["liposome", "lipid vesicle"],
                "PCL": ["polycaprolactone", "pcl"],
                "PLA": ["polylactic acid", "pla", "poly(lactic acid)"]
            }
            
            for material_name, patterns in material_patterns.items():
                if any(pattern in text for pattern in patterns):
                    material_info["material_name"] = material_name
                    material_info["confidence_score"] = 7
                    break
            
            # Temperature stability extraction
            if any(temp_word in text for temp_word in ["temperature", "thermal", "stability"]):
                if "room temperature" in text or "25°c" in text or "ambient" in text:
                    material_info["thermal_stability_temp_range"] = "Room temperature stable"
                    material_info["confidence_score"] += 1
                elif "40°c" in text or "40 °c" in text:
                    material_info["thermal_stability_temp_range"] = "Stable above 40°C"
                    material_info["confidence_score"] += 2
            
            # Biocompatibility extraction
            if any(bio_word in text for bio_word in ["biocompatible", "non-toxic", "fda approved", "safe"]):
                material_info["biocompatibility_data"] = "Biocompatible"
                material_info["confidence_score"] += 1
            
            materials.append(material_info)
        
        logger.info(f"Extracted {len(materials)} material candidates")
        
        return {
            "materials": materials,
            "focus_area": focus_area,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_material_properties: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}", "materials": []}

class SimplifiedLiteratureMiner:
    """Simplified literature mining system with enhanced MCP functionality."""
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
    
    def _generate_process_explanation(self, user_request: str, phase: str, **kwargs) -> str:
        """Generate explanatory text for different phases."""
        if phase == "start":
            return f"I'm analyzing the literature to find materials relevant to: '{user_request}'. This enhanced search will identify materials mentioned in recent scientific papers and extract their properties from the literature."
        elif phase == "search":
            papers_count = kwargs.get("unique_papers", 0)
            return f"Successfully found {papers_count} relevant research papers. Now extracting material information and properties from the abstracts and titles of these papers."
        elif phase == "results":
            material_count = kwargs.get("material_count", 0)
            return f"Literature analysis complete. Extracted information about {material_count} materials from the scientific papers. These are materials mentioned in the literature with properties relevant to your query - this is raw extracted data without quality filtering."
        else:
            return "Processing your request..."

    async def intelligent_mining_with_mcp(
        self,
        user_request: str,
        max_papers: int = 30,
        recent_only: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform enhanced literature mining with simplified MCP integration."""
        def update_progress(message: str, step_type: str = "info"):
            if progress_callback:
                progress_callback(message, step_type)
            logger.info(message)
        
        try:
            # Phase 1: Introduction
            update_progress("Enhanced Literature Mining System", "start")
            update_progress(f"Research Question: {user_request}")
            
            intro_explanation = self._generate_process_explanation(user_request, "start")
            update_progress(intro_explanation, "explanation")
            
            # Phase 2: Strategy (simplified - use single search instead of multiple queries)
            update_progress("Phase 1: Intelligent Search Strategy", "explanation")
            material_focus = self._determine_material_focus(user_request)
            update_progress(f"Focus Area: {material_focus}")
            
            # Phase 3: Enhanced Search (simplified - single search call like original system)
            update_progress("Phase 2: Enhanced Academic Search", "info")
            update_progress("└─ Using improved filtering and rate limiting...")
            
            # Use single search query (like the original system) instead of multiple queries
            search_results = await enhanced_search_papers(
                query=user_request,
                max_results=max_papers,
                recent_only=recent_only,
                use_llm_evaluation=False  # Keep disabled for efficiency
            )
            
            if "error" in search_results:
                return {"error": f"Search failed: {search_results['error']}"}
            
            papers = search_results.get("papers", [])
            update_progress(f"└─ Found {len(papers)} relevant research papers")
            
            # Handle empty results gracefully
            if not papers:
                update_progress("Search Results Analysis:", "info")
                update_progress("└─ No papers found matching the specific criteria")
                update_progress("Try broader search terms or remove date filters", "info")
                
                return {
                    "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "user_request": user_request,
                    "material_focus": material_focus,
                    "papers_analyzed": 0,
                    "material_candidates": [],
                    "paper_summaries": [],
                    "enhanced_search": True,
                    "status": "no_papers_found",
                    "suggestions": [
                        "Try broader search terms",
                        "Remove date filters (set recent_only=False)",
                        "Use more general keywords like 'polymer' or 'biocompatible'"
                    ]
                }
            
            # Generate search explanation
            search_explanation = self._generate_process_explanation(
                user_request, "search", unique_papers=len(papers)
            )
            update_progress(search_explanation, "explanation")
            
            # Phase 4: Material Analysis
            update_progress("Phase 3: Enhanced Material Analysis", "explanation")
            update_progress("└─ Extracting material properties and data...")
            
            material_analysis = analyze_material_properties(papers[:15], material_focus)
            
            if "error" in material_analysis:
                return {"error": f"Analysis failed: {material_analysis['error']}"}
            
            materials = material_analysis.get("materials", [])
            update_progress(f"└─ Extracted {len(materials)} material candidates")
            
            # Phase 4.5: Generate detailed per-paper summaries
            update_progress("└─ Generating detailed per-paper analysis...")
            paper_summaries = await self._generate_paper_summaries(papers[:10], user_request)
            update_progress(f"└─ Generated {len(paper_summaries)} detailed paper summaries")
            
            # Phase 5: LLM Enhancement
            if self.ollama_client and materials:
                update_progress("Phase 4: AI Enhancement", "explanation")
                try:
                    enhanced_materials = await self._enhance_with_llm(materials, user_request)
                    if enhanced_materials:
                        materials = enhanced_materials
                        update_progress(f"└─ Enhanced analysis for {len(materials)} materials")
                except Exception as e:
                    update_progress(f"└─ Using base analysis: {str(e)}")
            
            # Generate results explanation
            results_explanation = self._generate_process_explanation(
                user_request, "results", material_count=len(materials)
            )
            update_progress(results_explanation, "explanation")
            
            # Compile results
            results = {
                "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_request": user_request,
                "material_focus": material_focus,
                "papers_analyzed": len(papers),
                "material_candidates": materials,
                "paper_summaries": paper_summaries,
                "enhanced_search": True,
                "query_used": search_results.get("query_used", user_request)
            }
            
            update_progress(f"Enhanced Literature Mining Complete! Found {len(materials)} materials from literature", "complete")
            
            return results
            
        except Exception as e:
            error_msg = f"Error during enhanced mining: {str(e)}"
            update_progress(f"Error: {error_msg}")
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _determine_material_focus(self, user_request: str) -> str:
        """Determine material focus from user request."""
        request_lower = user_request.lower()
        
        if any(keyword in request_lower for keyword in ["polymer", "plga", "pla", "peg"]):
            return "biocompatible polymers"
        elif any(keyword in request_lower for keyword in ["hydrogel", "gel", "crosslink"]):
            return "hydrogel systems"
        elif any(keyword in request_lower for keyword in ["nano", "particle", "liposome"]):
            return "nanoparticle delivery"
        elif any(keyword in request_lower for keyword in ["patch", "transdermal", "skin"]):
            return "transdermal patches"
        else:
            return "thermal-stable biomaterials"
    
    async def _enhance_with_llm(self, materials: List[Dict], user_request: str) -> List[Dict]:
        """Enhance material analysis with LLM insights."""
        if not self.ollama_client:
            return materials
        
        try:
            enhanced_materials = []
            
            for material in materials[:10]:
                material_name = material.get("material_name", "Unknown")
                if material_name == "Unknown Material":
                    enhanced_materials.append(material)
                    continue
                
                prompt = f"""Analyze this material for insulin delivery: {material_name}
                
Composition: {material.get('material_composition', 'Not specified')}
Thermal Stability: {material.get('thermal_stability_temp_range', 'Not specified')}
Source: {material.get('source_paper', {}).get('title', 'Unknown')}

For the query "{user_request}", provide a brief insight (2-3 sentences) about this material's suitability."""
                
                try:
                    response = self.ollama_client.ask_question(prompt)
                    material["ai_insight"] = response.get("response", "No insight available")
                    material["enhancement_method"] = "Enhanced Search + Local LLM"
                except Exception as e:
                    logger.warning(f"LLM enhancement failed for {material_name}: {str(e)}")
                
                enhanced_materials.append(material)
            
            return enhanced_materials
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {str(e)}")
            return materials

    async def _generate_paper_summaries(self, papers: List[Dict], user_request: str) -> List[Dict]:
        """Generate detailed summaries for each paper including material candidates discussed."""
        summaries = []
        
        for i, paper in enumerate(papers):
            try:
                title = paper.get('title', 'Unknown Title')
                abstract = paper.get('abstract', '')
                year = paper.get('year', 'Unknown')
                citation_count = paper.get('citationCount', 0)
                url = paper.get('url', '')
                
                # Extract key information
                summary = {
                    "paper_number": i + 1,
                    "title": title,
                    "year": year,
                    "citation_count": citation_count,
                    "url": url,
                    "materials_discussed": [],
                    "key_findings": "",
                    "relevance_to_query": ""
                }
                
                # Analyze abstract for materials and findings
                if abstract:
                    text = f"{title.lower()} {abstract.lower()}"
                    
                    # Identify materials mentioned
                    material_keywords = {
                        "PLGA": ["plga", "poly(lactic-co-glycolic acid)", "polylactic"],
                        "Chitosan": ["chitosan", "chitin"],
                        "PEG": ["peg", "polyethylene glycol"],
                        "Alginate": ["alginate", "sodium alginate"],
                        "Hydrogels": ["hydrogel", "gel matrix"],
                        "Nanoparticles": ["nanoparticle", "nano particle", "nanosphere"],
                        "Liposomes": ["liposome", "lipid vesicle"],
                        "PCL": ["polycaprolactone", "pcl"],
                        "PLA": ["polylactic acid", "pla", "poly(lactic acid)"],
                        "Polymers": ["polymer", "polymeric"],
                        "Insulin": ["insulin", "protein drug"],
                        "Transdermal": ["transdermal", "patch", "skin delivery"]
                    }
                    
                    materials_found = []
                    for material_name, patterns in material_keywords.items():
                        if any(pattern in text for pattern in patterns):
                            materials_found.append(material_name)
                    
                    summary["materials_discussed"] = materials_found
                    
                    # Generate AI summary if OLLAMA is available
                    if self.ollama_client:
                        try:
                            ai_summary = await self._generate_ai_paper_summary(title, abstract, user_request)
                            summary["key_findings"] = ai_summary.get("key_findings", "")
                            summary["relevance_to_query"] = ai_summary.get("relevance", "")
                            summary["material_insights"] = ai_summary.get("material_insights", "")
                        except Exception as e:
                            logger.warning(f"AI summary failed for paper {i+1}: {str(e)}")
                            summary["key_findings"] = self._extract_key_findings_basic(abstract)
                            summary["relevance_to_query"] = "Moderate relevance based on keyword analysis"
                    else:
                        summary["key_findings"] = self._extract_key_findings_basic(abstract)
                        summary["relevance_to_query"] = "Moderate relevance based on keyword analysis"
                
                summaries.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to generate summary for paper {i+1}: {str(e)}")
                summaries.append({
                    "paper_number": i + 1,
                    "title": paper.get('title', 'Unknown Title'),
                    "error": f"Summary generation failed: {str(e)}"
                })
        
        return summaries

    async def _generate_ai_paper_summary(self, title: str, abstract: str, user_request: str) -> Dict:
        """Generate AI-powered summary of a paper's relevance and findings."""
        prompt = f"""Analyze this research paper for the query: "{user_request}"

Paper Title: {title}
Abstract: {abstract}

Provide a structured analysis in JSON format:
{{
    "key_findings": "2-3 sentences summarizing the main findings relevant to the query",
    "material_insights": "specific materials or compounds discussed and their properties",
    "relevance": "1-2 sentences explaining how this paper relates to the user's query",
    "potential_applications": "potential applications for insulin delivery"
}}

Focus on material candidates, thermal stability, biocompatibility, and delivery mechanisms."""
        
        try:
            response = await asyncio.to_thread(
                self.ollama_client.client.chat,
                model=self.ollama_client.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 300}
            )
            
            response_text = response['message']['content'].strip()
            
            # Try to parse JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                return {
                    "key_findings": response_text[:200] + "...",
                    "material_insights": "Analysis available in full response",
                    "relevance": "Relevant to user query",
                    "potential_applications": "Potential for insulin delivery applications"
                }
                
        except Exception as e:
            logger.warning(f"AI paper summary failed: {str(e)}")
            return {
                "key_findings": "AI analysis unavailable",
                "material_insights": "Manual analysis required",
                "relevance": "Requires further analysis",
                "potential_applications": "Unknown"
            }

    def _extract_key_findings_basic(self, abstract: str) -> str:
        """Basic keyword-based extraction of key findings."""
        if not abstract:
            return "No abstract available"
        
        # Look for key result indicators
        result_indicators = [
            "demonstrated", "showed", "found", "revealed", "indicated",
            "confirmed", "achieved", "resulted in", "improved", "enhanced"
        ]
        
        sentences = abstract.split('.')
        key_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in result_indicators):
                key_sentences.append(sentence.strip())
        
        if key_sentences:
            return '. '.join(key_sentences[:2]) + '.'
        else:
            return abstract[:200] + '...'

# Sync wrapper for Flask integration
class SimplifiedMCPLiteratureMinerSync:
    """Synchronous wrapper for the simplified literature miner."""
    
    def __init__(self, ollama_client=None):
        self.async_miner = SimplifiedLiteratureMiner(ollama_client)
        
    def intelligent_mining_with_mcp(self, user_request: str, max_papers: int = 30, 
                                   recent_only: bool = False, progress_callback=None):
        """Sync wrapper for enhanced mining."""
        try:
            return asyncio.run(
                self.async_miner.intelligent_mining_with_mcp(
                    user_request, max_papers, recent_only, progress_callback
                )
            )
        except Exception as e:
            return {"error": f"Enhanced mining error: {str(e)}"}
    
    def cleanup(self):
        """Cleanup (no-op for simplified version)."""
        pass 