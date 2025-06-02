from typing import Any, List, Dict, Optional
import asyncio
import logging
import json
import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from semantic_scholar_search import initialize_client, search_papers, get_paper_details, get_author_details, get_citations_and_references
from ollama_client import OllamaClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastMCP server
mcp = FastMCP("semanticscholar-insulin-ai")

# Initialize SemanticScholar client
client = initialize_client()

# Initialize OLLAMA client for LLM-based relevance evaluation
ollama_client = OllamaClient()

# Rate limiting configuration based on Semantic Scholar API limits
# - Without API key: 100 requests per 5-minute window (1 request every 3-4 seconds)  
# - With API key: 1 request per second
SEMANTIC_SCHOLAR_API_KEY = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')

if SEMANTIC_SCHOLAR_API_KEY:
    RATE_LIMIT_DELAY = 1.2  # 1.2 seconds for authenticated users (slightly conservative)
    MAX_RETRIES = 3
    RETRY_DELAY = 5.0
    logging.info("Using authenticated Semantic Scholar API access")
else:
    RATE_LIMIT_DELAY = 4.0  # 4 seconds for unauthenticated users (respects 100 req/5min limit)
    MAX_RETRIES = 2  # Fewer retries for unauthenticated to avoid hitting limits
    RETRY_DELAY = 15.0  # Longer retry delay for unauthenticated
    logging.info("Using unauthenticated Semantic Scholar API access (rate limited)")

async def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with rate limiting and retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            # Always wait before API calls to respect rate limits
            await asyncio.sleep(RATE_LIMIT_DELAY)
            result = await asyncio.to_thread(func, *args, **kwargs)
            return result
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logging.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                    raise Exception(f"Semantic Scholar API rate limit exceeded. Please wait and try again later.")
            elif "500" in error_str or "internal server error" in error_str.lower():
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logging.warning(f"Server error, waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Server errors persist after {MAX_RETRIES} attempts")
                    raise Exception(f"Semantic Scholar API server errors. Please try again later.")
            else:
                logging.error(f"API call failed: {error_str}")
                raise e
    return None

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
    try:
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
            logging.warning(f"LLM evaluation timed out after {timeout}s for paper: {title[:50]}...")
            return {
                "relevance_score": 7,
                "reasoning": f"LLM evaluation timed out after {timeout}s, defaulting to relevant",
                "relevant": True
            }
        
        response_text = response['message']['content'].strip()
        
        # Try to parse JSON response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "relevance_score": 7,
                "reasoning": "LLM evaluation failed, using moderate relevance",
                "relevant": True
            }
            
    except Exception as e:
        logging.warning(f"LLM relevance evaluation failed: {e}")
        # Fallback to accepting the paper
        return {
            "relevance_score": 7,
            "reasoning": "LLM evaluation unavailable, defaulting to relevant",
            "relevant": True
        }

@mcp.tool()
async def search_insulin_materials(
    query: str, 
    max_results: int = 20,
    recent_only: bool = False,  # Changed default to False
    use_llm_evaluation: bool = False,  # NEW: Make LLM evaluation optional
    **kwargs  # Accept any additional parameters for compatibility
) -> Dict[str, Any]:
    """
    Search for papers with optional LLM-based relevance evaluation.

    Args:
        query: Search query focused on insulin delivery materials
        max_results: Maximum number of results to return (default: 20)
        recent_only: Whether to focus on recent papers (default: False - all years)
        use_llm_evaluation: Whether to use LLM for relevance evaluation (default: False - faster)

    Returns:
        Dictionary containing papers and metadata
    """
    logging.info(f"Starting search_insulin_materials with query: {query}, max_results: {max_results}, use_llm: {use_llm_evaluation}")
    
    # Handle legacy parameter names for compatibility
    if 'num_results' in kwargs:
        max_results = kwargs['num_results']
        logging.info(f"Using legacy parameter num_results: {max_results}")
    
    try:
        # Use the original query without forced enhancement
        enhanced_query = query  # Let users control their own query specificity
        
        logging.info(f"🔍 Executing search: {enhanced_query[:80]}...")
        
        # Single API call with rate limiting
        results = await safe_api_call(search_papers, client, enhanced_query, max_results * 2)  # Get more to compensate for filtering
        
        if not results:
            logging.warning("No results returned from search")
            return {"error": "No results returned from search", "papers": []}
        
        logging.info(f"📚 Found {len(results)} papers from search")
        
        # Apply filtering based on settings
        filtered_results = []
        for i, paper in enumerate(results):
            paper_debug_info = f"Paper {i+1}: '{paper.get('title', 'No title')[:80]}...'"
            
            # Optional year filter (now defaults to False)
            if recent_only and paper.get('year') and paper['year'] < 2020:
                logging.info(f"   ❌ {paper_debug_info} - Filtered out (too old: {paper.get('year')})")
                continue
            
            if use_llm_evaluation:
                # LLM-based relevance evaluation (slower but more intelligent)
                logging.info(f"   🤖 Evaluating relevance with LLM: {paper_debug_info}")
                relevance_result = await evaluate_paper_relevance_with_llm(paper, query, timeout=10)
                
                if relevance_result.get('relevant', True):
                    score = relevance_result.get('relevance_score', 7)
                    reasoning = relevance_result.get('reasoning', 'Relevant')
                    logging.info(f"   ✅ {paper_debug_info} - Relevant (Score: {score}/10, Reason: {reasoning[:50]}...)")
                    
                    # Add relevance metadata to paper
                    paper['llm_relevance'] = relevance_result
                    filtered_results.append(paper)
                else:
                    score = relevance_result.get('relevance_score', 3)
                    reasoning = relevance_result.get('reasoning', 'Not relevant')
                    logging.info(f"   ❌ {paper_debug_info} - Not relevant (Score: {score}/10, Reason: {reasoning[:50]}...)")
            else:
                # Fast mode: Accept all papers (no LLM evaluation)
                logging.info(f"   ✅ {paper_debug_info} - Accepted (fast mode, no LLM evaluation)")
                paper['llm_relevance'] = {
                    "relevance_score": 7,
                    "reasoning": "Fast mode - no LLM evaluation performed",
                    "relevant": True
                }
                filtered_results.append(paper)
            
            # Stop if we have enough relevant papers
            if len(filtered_results) >= max_results:
                break
        
        # Sort by relevance score if LLM evaluation was used
        if use_llm_evaluation:
            filtered_results.sort(key=lambda x: x.get('llm_relevance', {}).get('relevance_score', 0), reverse=True)
        
        evaluation_method = "LLM-based relevance scoring" if use_llm_evaluation else "Fast mode (no filtering)"
        logging.info(f"✅ Search yielded {len(filtered_results)} papers using {evaluation_method}")
        
        return {
            "papers": filtered_results[:max_results],
            "total_found": len(filtered_results),
            "query_used": enhanced_query,
            "evaluation_method": evaluation_method,
            "llm_evaluation_used": use_llm_evaluation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logging.error(f"Error in search_insulin_materials: {error_msg}")
        return {"error": error_msg, "papers": []}

@mcp.tool()
async def analyze_material_properties(
    papers: List[Dict[str, Any]], 
    focus_area: str = "thermal stability and biocompatibility"
) -> Dict[str, Any]:
    """
    Analyze papers to extract material properties relevant to insulin delivery.

    Args:
        papers: List of paper dictionaries to analyze
        focus_area: Specific properties to focus on during extraction

    Returns:
        Dictionary containing extracted material information
    """
    logging.info(f"Analyzing {len(papers)} papers for material properties")
    
    try:
        materials = []
        
        for i, paper in enumerate(papers):
            logging.info(f"Analyzing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
            # Extract basic information
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
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            text = f"{title} {abstract}"
            
            # Material identification
            material_patterns = {
                "PLGA": ["plga", "poly(lactic-co-glycolic acid)", "polylactic"],
                "Chitosan": ["chitosan", "chitin"],
                "PEG": ["peg", "polyethylene glycol"],
                "Alginate": ["alginate", "sodium alginate"],
                "Hydrogel": ["hydrogel", "gel matrix"],
                "Nanoparticles": ["nanoparticle", "nano particle", "nanosphere"],
                "Liposomes": ["liposome", "lipid vesicle"]
            }
            
            for material_name, patterns in material_patterns.items():
                if any(pattern in text for pattern in patterns):
                    material_info["material_name"] = material_name
                    material_info["confidence_score"] = 7
                    break
            
            # Temperature stability extraction
            if any(temp_word in text for temp_word in ["temperature", "thermal", "stability", "room temp"]):
                if "room temperature" in text or "25°c" in text or "ambient" in text:
                    material_info["thermal_stability_temp_range"] = "Room temperature stable"
                    material_info["confidence_score"] += 1
            
            # Biocompatibility extraction
            if any(bio_word in text for bio_word in ["biocompatible", "non-toxic", "fda approved", "safe"]):
                material_info["biocompatibility_data"] = "Biocompatible"
                material_info["confidence_score"] += 1
            
            materials.append(material_info)
        
        logging.info(f"Extracted {len(materials)} material candidates")
        
        return {
            "materials": materials,
            "focus_area": focus_area,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in analyze_material_properties: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}", "materials": []}

@mcp.tool()
async def search_semantic_scholar(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    logging.info(f"Searching for papers with query: {query}, num_results: {num_results}")
    """
    Search for papers on Semantic Scholar using a query string.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        List of dictionaries containing paper information
    """
    try:
        results = await asyncio.to_thread(search_papers, client, query, num_results)
        return results
    except Exception as e:
        return [{"error": f"An error occurred while searching: {str(e)}"}]

@mcp.tool()
async def get_semantic_scholar_paper_details(paper_id: str) -> Dict[str, Any]:
    logging.info(f"Fetching paper details for paper ID: {paper_id}")
    """
    Get details of a specific paper on Semantic Scholar.

    Args:
        paper_id: ID of the paper

    Returns:
        Dictionary containing paper details
    """
    try:
        paper = await asyncio.to_thread(get_paper_details, client, paper_id)
        return {
            "paperId": paper.paperId,
            "title": paper.title,
            "abstract": paper.abstract,
            "year": paper.year,
            "authors": [{"name": author.name, "authorId": author.authorId} for author in paper.authors],
            "url": paper.url,
            "venue": paper.venue,
            "publicationTypes": paper.publicationTypes,
            "citationCount": paper.citationCount
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching paper details: {str(e)}"}

@mcp.tool()
async def get_semantic_scholar_author_details(author_id: str) -> Dict[str, Any]:
    logging.info(f"Fetching author details for author ID: {author_id}")
    """
    Get details of a specific author on Semantic Scholar.

    Args:
        author_id: ID of the author

    Returns:
        Dictionary containing author details
    """
    try:
        author = await asyncio.to_thread(get_author_details, client, author_id)
        return {
            "authorId": author.authorId,
            "name": author.name,
            "url": author.url,
            "affiliations": author.affiliations,
            "paperCount": author.paperCount,
            "citationCount": author.citationCount,
            "hIndex": author.hIndex
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching author details: {str(e)}"}

@mcp.tool()
async def get_semantic_scholar_citations_and_references(paper_id: str) -> Dict[str, List[Dict[str, Any]]]:
    logging.info(f"Fetching citations and references for paper ID: {paper_id}")
    """
    Get citations and references for a specific paper on Semantic Scholar.

    Args:
        paper_id: ID of the paper

    Returns:
        Dictionary containing lists of citations and references
    """
    try:
        paper = await asyncio.to_thread(get_paper_details, client, paper_id)
        citations_refs = await asyncio.to_thread(get_citations_and_references, paper)
        return {
            "citations": [
                {
                    "paperId": citation.paperId,
                    "title": citation.title,
                    "year": citation.year,
                    "authors": [{"name": author.name, "authorId": author.authorId} for author in citation.authors]
                } for citation in citations_refs["citations"]
            ],
            "references": [
                {
                    "paperId": reference.paperId,
                    "title": reference.title,
                    "year": reference.year,
                    "authors": [{"name": author.name, "authorId": author.authorId} for author in reference.authors]
                } for reference in citations_refs["references"]
            ]
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching citations and references: {str(e)}"}

@mcp.tool()
async def save_research_findings(
    user_request: str,
    materials: List[Dict[str, Any]],
    papers_analyzed: int
) -> Dict[str, Any]:
    """
    Save research findings to a file.

    Args:
        user_request: Original user request
        materials: List of material findings
        papers_analyzed: Number of papers analyzed

    Returns:
        Save status information
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mining_results/mcp_enhanced_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs("mining_results", exist_ok=True)
        
        findings = {
            "user_request": user_request,
            "materials": materials,
            "papers_analyzed": papers_analyzed,
            "timestamp": datetime.now().isoformat(),
            "method": "MCP Enhanced"
        }
        
        with open(filename, 'w') as f:
            json.dump(findings, f, indent=2)
        
        logging.info(f"Research findings saved to {filename}")
        
        return {
            "status": "saved",
            "filename": filename,
            "materials_count": len(materials)
        }
        
    except Exception as e:
        logging.error(f"Error saving research findings: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logging.info("Starting Enhanced Semantic Scholar MCP server for Insulin AI")
    # Initialize and run the server
    mcp.run(transport='stdio')