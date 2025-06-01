import json
import os
from typing import List, Dict, Optional
from datetime import datetime

from semantic_scholar_client import SemanticScholarClient
from ollama_client import OllamaClient


class MaterialsLiteratureMiner:
    """
    Literature mining system for discovering materials suitable for fridge-free 
    insulin delivery patches. Milestone 1: Basic LLM-guided literature mining.
    """
    
    def __init__(self, 
                 semantic_scholar_api_key: Optional[str] = None,
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the Materials Literature Mining system.
        
        Args:
            semantic_scholar_api_key (str, optional): Semantic Scholar API key
            ollama_model (str): OLLAMA model to use
            ollama_host (str): OLLAMA server host URL
        """
        self.scholar = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.ollama = OllamaClient(model_name=ollama_model, host=ollama_host)
        
        print("Materials Literature Mining System initialized!")
        print(f"Using OLLAMA model: {ollama_model}")
        print(f"Semantic Scholar API: {'Authenticated' if semantic_scholar_api_key else 'Public (rate limited)'}")
    
    def intelligent_mining(self,
                          user_request: str,
                          max_papers: int = 50,
                          recent_only: bool = True,
                          save_results: bool = True) -> Dict:
        """
        Perform literature mining based on intelligent interpretation of user requests.
        
        Args:
            user_request (str): User's request or question about insulin delivery materials
            max_papers (int): Maximum number of papers to analyze
            recent_only (bool): Whether to focus on recent papers (2020+)
            save_results (bool): Whether to save results to file
        
        Returns:
            Dict: Mining results with material candidates
        """
        print(f"\n🧠 Intelligent Literature Mining")
        print(f"📝 User Request: {user_request}")
        
        # Generate search strategy using LLM
        search_strategy = self._generate_search_strategy(user_request)
        
        if not search_strategy.get('relevant', True):
            return {
                "error": "Request not relevant to insulin delivery materials research",
                "suggestion": search_strategy.get('suggestion', ''),
                "candidates": []
            }
        
        # Use generated search queries
        search_queries = search_strategy['search_queries']
        extraction_focus = search_strategy.get('extraction_focus', '')
        
        print(f"🎯 Generated {len(search_queries)} targeted search queries")
        
        # Search for relevant papers
        all_papers = []
        for query in search_queries:
            print(f"📚 Searching: {query}")
            papers = self.scholar.search_papers_by_topic(
                topic=query,
                max_results=max_papers // len(search_queries),
                recent_years_only=recent_only
            )
            all_papers.extend(papers)
        
        # Remove duplicates
        unique_papers = self._deduplicate_papers(all_papers)
        print(f"✅ Found {len(unique_papers)} unique papers")
        
        if not unique_papers:
            return {"error": "No relevant papers found", "candidates": []}
        
        # Extract material information using customized LLM prompt
        material_candidates = self._extract_material_data_focused(
            unique_papers[:max_papers], 
            extraction_focus
        )
        
        # Compile results
        results = {
            "search_timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "search_strategy": search_strategy,
            "search_queries": search_queries,
            "papers_analyzed": len(unique_papers),
            "material_candidates": material_candidates
        }
        
        # Save results if requested
        if save_results:
            self._save_mining_results(results, prefix="intelligent")
        
        print(f"🎯 Identified {len(material_candidates)} material candidates")
        return results
    
    def _generate_search_strategy(self, user_request: str) -> Dict:
        """
        Generate search strategy based on user request using LLM.
        
        Args:
            user_request (str): User's request or question
            
        Returns:
            Dict: Search strategy with queries and focus areas
        """
        strategy_prompt = f"""# Intelligent Search Strategy Generation

PROJECT CONTEXT: AI-Driven Design of Fridge-Free Insulin Delivery Patches
OBJECTIVE: Discover materials that can maintain insulin stability without refrigeration using transdermal patches

USER REQUEST: "{user_request}"

TASK: Analyze the user request and generate an appropriate search strategy.

EVALUATION CRITERIA:
1. Is this request relevant to insulin delivery, material science, or biomedical applications? 
2. If not directly relevant, can it be reasonably connected to the project objectives?
3. If completely off-topic (e.g., cooking recipes, entertainment), mark as not relevant but suggest a connection if possible.

OUTPUT FORMAT (JSON):
{{
  "relevant": true/false,
  "relevance_score": 1-10,
  "interpretation": "How you interpreted the user's request in context of insulin delivery",
  "search_queries": [
    "search query 1 based on user request",
    "search query 2 based on user request", 
    "search query 3 based on user request",
    "search query 4 based on user request",
    "search query 5 based on user request"
  ],
  "extraction_focus": "What specific aspects to focus on during material extraction",
  "suggestion": "If not relevant, suggest how to make it relevant to the project"
}}

EXAMPLES:
- User: "smart polymers" → Focus on temperature-responsive materials
- User: "green chemistry" → Focus on biocompatible, sustainable materials  
- User: "nanotechnology" → Focus on nanocarriers for insulin delivery
- User: "patches for pain relief" → Focus on transdermal delivery mechanisms
- User: "cooking recipes" → Not relevant, suggest "food-safe materials for biomedical applications"

Generate a thoughtful search strategy that connects the user's interest to insulin delivery materials research."""

        try:
            print("🤖 Generating intelligent search strategy...")
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user',
                    'content': strategy_prompt
                }],
                options={
                    'temperature': 0.2,
                    'num_predict': 1000
                }
            )
            
            # Parse the strategy response
            strategy = self._parse_strategy_response(response['message']['content'])
            return strategy
            
        except Exception as e:
            print(f"Error in strategy generation: {e}")
            # Fallback to default behavior
            return {
                "relevant": True,
                "relevance_score": 5,
                "interpretation": f"Using default search for: {user_request}",
                "search_queries": self._get_default_search_queries(),
                "extraction_focus": "General insulin delivery materials"
            }
    
    def _parse_strategy_response(self, response_text: str) -> Dict:
        """Parse LLM strategy response."""
        try:
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                strategy = json.loads(json_str)
                
                # Validate required fields
                if 'search_queries' in strategy and isinstance(strategy['search_queries'], list):
                    return strategy
        except:
            pass
        
        # Fallback parsing
        return {
            "relevant": True,
            "relevance_score": 5,
            "interpretation": "Fallback interpretation",
            "search_queries": self._get_default_search_queries()[:5],
            "extraction_focus": "General materials research"
        }

    def mine_insulin_delivery_materials(self, 
                                      max_papers: int = 50,
                                      recent_only: bool = True,
                                      save_results: bool = True) -> Dict:
        """
        Mine literature for materials suitable for insulin delivery patches.
        (Legacy method - use intelligent_mining for enhanced functionality)
        
        Args:
            max_papers (int): Maximum number of papers to analyze
            recent_only (bool): Whether to focus on recent papers (2020+)
            save_results (bool): Whether to save results to file
        
        Returns:
            Dict: Mining results with material candidates
        """
        print(f"\n🔬 Starting Materials Literature Mining")
        print(f"📊 Analyzing up to {max_papers} papers...")
        
        # Search for relevant papers
        search_queries = self._get_default_search_queries()
        all_papers = []
        
        for query in search_queries:
            print(f"📚 Searching: {query}")
            papers = self.scholar.search_papers_by_topic(
                topic=query,
                max_results=max_papers // len(search_queries),  # Distribute across queries
                recent_years_only=recent_only
            )
            all_papers.extend(papers)
        
        # Remove duplicates
        unique_papers = self._deduplicate_papers(all_papers)
        print(f"✅ Found {len(unique_papers)} unique papers")
        
        if not unique_papers:
            return {"error": "No relevant papers found", "candidates": []}
        
        # Extract material information using LLM
        material_candidates = self._extract_material_data(unique_papers[:max_papers])
        
        # Compile results
        results = {
            "search_timestamp": datetime.now().isoformat(),
            "search_queries": search_queries,
            "papers_analyzed": len(unique_papers),
            "material_candidates": material_candidates
        }
        
        # Save results if requested
        if save_results:
            self._save_mining_results(results)
        
        print(f"🎯 Identified {len(material_candidates)} material candidates")
        return results
    
    def _get_default_search_queries(self) -> List[str]:
        """
        Get default search queries for insulin delivery materials.
        """
        return [
            "hydrogels insulin delivery transdermal patch",
            "polymer protein stabilization thermal",
            "biocompatible materials drug delivery skin",
            "nanomaterials insulin encapsulation controlled release",
            "protein stabilization polymers temperature",
            "peptide delivery hydrogels biocompatible",
            "insulin stability materials room temperature",
            "transdermal drug delivery patches"
        ]
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers based on title similarity."""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _extract_material_data(self, papers: List[Dict]) -> List[Dict]:
        """
        Extract structured material information from papers using LLM.
        """
        # Prepare papers context (limit to avoid token limits)
        papers_context = self._prepare_papers_context(papers[:15])
        
        extraction_prompt = self._build_extraction_prompt()
        full_prompt = f"{extraction_prompt}\n\nPAPERS TO ANALYZE:\n{papers_context}"
        
        try:
            print("🤖 Extracting material data using LLM...")
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user',
                    'content': full_prompt
                }],
                options={
                    'temperature': 0.3,
                    'num_predict': 4000
                }
            )
            
            # Parse the LLM response
            material_candidates = self._parse_llm_response(response['message']['content'])
            return material_candidates
            
        except Exception as e:
            print(f"Error in material extraction: {e}")
            return []
    
    def _build_extraction_prompt(self) -> str:
        """
        Build the extraction prompt for material data.
        """
        return """# Materials Extraction for Insulin Delivery Patches

EXTRACTION TASK: Identify materials with potential for fridge-free insulin delivery patches.

MATERIAL REQUIREMENTS:
1. Demonstrated protein or peptide stabilization capability
2. Thermal stability at temperatures 25-40°C
3. Biocompatible for transdermal application
4. Controlled release properties for drug delivery

OUTPUT FORMAT: For each material, provide JSON-formatted data:
{
  "material_name": "Name of the material",
  "material_composition": "Chemical composition and formula",
  "chemical_structure": "Structural information (backbone, side chains, crosslinking)",
  "thermal_stability_temp_range": "Temperature stability range",
  "insulin_stability_duration": "Reported insulin stability duration",
  "biocompatibility_data": "Biocompatibility information",
  "release_kinetics": "Release kinetics and delivery mechanism", 
  "delivery_efficiency": "Delivery efficiency data",
  "stabilization_mechanism": "Mechanism of protein stabilization",
  "literature_references": ["Reference 1", "Reference 2"],
  "confidence_score": "Score 1-10 based on evidence quality"
}

Extract information ONLY if supported by the provided papers. If data is not available, use "Not specified" or "Not reported".
Provide up to 10 material candidates."""
    
    def _prepare_papers_context(self, papers: List[Dict]) -> str:
        """Prepare papers context for LLM analysis."""
        context = ""
        for i, paper in enumerate(papers, 1):
            context += f"\nPAPER {i}:\n"
            context += f"Title: {paper.get('title', 'Unknown')}\n"
            context += f"Year: {paper.get('year', 'Unknown')}\n"
            context += f"Journal: {paper.get('journal', 'Unknown')}\n"
            context += f"Abstract: {paper.get('abstract', 'No abstract')}\n"
            context += "-" * 80 + "\n"
        
        return context
    
    def _parse_llm_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response to extract structured material data."""
        materials = []
        
        # Try to find JSON objects in the response
        lines = response_text.split('\n')
        json_buffer = ""
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('{'):
                in_json = True
                json_buffer = line
            elif in_json:
                json_buffer += "\n" + line
                if line.endswith('}'):
                    try:
                        material_data = json.loads(json_buffer)
                        if 'material_name' in material_data:
                            materials.append(material_data)
                    except json.JSONDecodeError:
                        pass
                    in_json = False
                    json_buffer = ""
        
        # If JSON parsing fails, try alternative parsing
        if not materials:
            materials = self._fallback_parse_response(response_text)
        
        return materials
    
    def _fallback_parse_response(self, response_text: str) -> List[Dict]:
        """Fallback parser for when JSON extraction fails."""
        materials = []
        
        sections = response_text.split('\n\n')
        for section in sections:
            if any(keyword in section.lower() for keyword in ['material', 'polymer', 'hydrogel']):
                material = {
                    "material_name": "Extracted from text",
                    "material_composition": "See full text",
                    "extraction_text": section[:500],
                    "confidence_score": 5
                }
                materials.append(material)
        
        return materials[:10]
    
    def _save_mining_results(self, results: Dict, prefix: str = "basic"):
        """Save mining results to file."""
        os.makedirs("mining_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mining_results/{prefix}_mining_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Mining results saved to: {filename}")
    
    def get_material_details(self, material_name: str) -> Dict:
        """
        Get detailed information about a specific material.
        
        Args:
            material_name (str): Name of the material to research
            
        Returns:
            Dict: Detailed material information
        """
        print(f"🔍 Researching material: {material_name}")
        
        papers = self.scholar.search_papers_by_topic(
            topic=f"{material_name} insulin protein stabilization",
            max_results=10,
            recent_years_only=False
        )
        
        if not papers:
            return {"error": f"No papers found for material: {material_name}"}
        
        detail_prompt = f"""Provide detailed information about {material_name} for insulin delivery applications.

Focus on:
1. Chemical composition and molecular structure
2. Thermal stability properties
3. Biocompatibility and safety data
4. Insulin/protein interaction mechanisms
5. Drug delivery performance
6. Manufacturing considerations
7. Regulatory status (if any)

Base your analysis on the following papers:

{self._prepare_papers_context(papers[:5])}

Provide a comprehensive technical summary."""

        try:
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user', 
                    'content': detail_prompt
                }]
            )
            
            return {
                "material_name": material_name,
                "detailed_analysis": response['message']['content'],
                "source_papers": len(papers),
                "papers": papers[:5]
            }
            
        except Exception as e:
            return {"error": f"Error analyzing {material_name}: {e}"}

    def _extract_material_data_focused(self, papers: List[Dict], extraction_focus: str) -> List[Dict]:
        """
        Extract structured material information from papers using LLM with specific focus.
        """
        # Prepare papers context (limit to avoid token limits)
        papers_context = self._prepare_papers_context(papers[:15])
        
        extraction_prompt = self._build_focused_extraction_prompt(extraction_focus)
        full_prompt = f"{extraction_prompt}\n\nPAPERS TO ANALYZE:\n{papers_context}"
        
        try:
            print(f"🤖 Extracting material data with focus: {extraction_focus}")
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user',
                    'content': full_prompt
                }],
                options={
                    'temperature': 0.3,
                    'num_predict': 4000
                }
            )
            
            # Parse the LLM response
            material_candidates = self._parse_llm_response(response['message']['content'])
            return material_candidates
            
        except Exception as e:
            print(f"Error in material extraction: {e}")
            return []
    
    def _build_focused_extraction_prompt(self, extraction_focus: str) -> str:
        """
        Build extraction prompt with specific focus area.
        """
        base_prompt = """# Materials Extraction for Insulin Delivery Patches

EXTRACTION TASK: Identify materials with potential for fridge-free insulin delivery patches.

MATERIAL REQUIREMENTS:
1. Demonstrated protein or peptide stabilization capability
2. Thermal stability at temperatures 25-40°C
3. Biocompatible for transdermal application
4. Controlled release properties for drug delivery"""
        
        if extraction_focus:
            base_prompt += f"\n\nSPECIAL FOCUS: {extraction_focus}"
            base_prompt += "\nPay particular attention to materials and properties related to this focus area."
        
        base_prompt += """

OUTPUT FORMAT: For each material, provide JSON-formatted data:
{
  "material_name": "Name of the material",
  "material_composition": "Chemical composition and formula",
  "chemical_structure": "Structural information (backbone, side chains, crosslinking)",
  "thermal_stability_temp_range": "Temperature stability range",
  "insulin_stability_duration": "Reported insulin stability duration",
  "biocompatibility_data": "Biocompatibility information",
  "release_kinetics": "Release kinetics and delivery mechanism", 
  "delivery_efficiency": "Delivery efficiency data",
  "stabilization_mechanism": "Mechanism of protein stabilization",
  "literature_references": ["Reference 1", "Reference 2"],
  "confidence_score": "Score 1-10 based on evidence quality"
}

Extract information ONLY if supported by the provided papers. If data is not available, use "Not specified" or "Not reported".
Provide up to 10 material candidates."""
        
        return base_prompt


# Example usage
if __name__ == "__main__":
    # Initialize the mining system
    miner = MaterialsLiteratureMiner()
    
    # Run basic literature mining
    results = miner.mine_insulin_delivery_materials(max_papers=30)
    
    print(f"\nFound {len(results.get('material_candidates', []))} candidates")
    
    # Get details about a specific material
    details = miner.get_material_details("chitosan")
    print(f"Details analysis length: {len(details.get('detailed_analysis', ''))}") 