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
                          recent_only: bool = False,
                          save_results: bool = True,
                          progress_callback = None) -> Dict:
        """
        Perform literature mining based on intelligent interpretation of user requests.
        
        Args:
            user_request (str): User's request or question about insulin delivery materials
            max_papers (int): Maximum number of papers to analyze
            recent_only (bool): Whether to focus on recent papers (2020+) - now defaults to False
            save_results (bool): Whether to save results to file
            progress_callback (callable): Optional callback for progress updates
        
        Returns:
            Dict: Mining results with material candidates
        """
        def update_progress(message, step_type="info"):
            """Helper to update progress and call callback if provided."""
            print(message)
            if progress_callback:
                progress_callback(message, step_type)
        
        update_progress(f"\nIntelligent Literature Mining", "start")
        update_progress(f"User Request: {user_request}")
        
        # Generate explanatory introduction
        intro_explanation = self._generate_process_explanation(user_request, "start")
        update_progress(intro_explanation, "explanation")
        
        # Generate search strategy using LLM
        update_progress("Generating intelligent search strategy...")
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
        
        update_progress(f"Generated {len(search_queries)} targeted search queries")
        
        # Show the actual queries generated
        if search_queries:
            update_progress("Phase 1: Intelligent Search Strategy")
            for i, query in enumerate(search_queries, 1):
                update_progress(f"   {i}. {query}")
        
        # Explain the search strategy
        strategy_explanation = self._generate_process_explanation(
            user_request, "strategy", search_queries, extraction_focus
        )
        update_progress(strategy_explanation, "explanation")
        
        # Search for relevant papers
        all_papers = []
        update_progress("Phase 2: Enhanced Academic Search")
        update_progress("└─ Using improved filtering and rate limiting...")
        
        for i, query in enumerate(search_queries, 1):
            update_progress(f"└─ Query {i}/{len(search_queries)}: {query}")
            papers = self.scholar.search_papers_by_topic(
                topic=query,
                max_results=max_papers // len(search_queries),
                recent_years_only=recent_only
            )
            update_progress(f"   ✅ Retrieved {len(papers)} papers for this query")
            
            # DEBUG: Show some paper titles to verify we're getting relevant content
            if papers:
                print(f"🔍 DEBUG: Sample papers for query '{query}':")
                for j, paper in enumerate(papers[:2]):  # Show first 2 papers
                    title = paper.get('title', 'No title')
                    abstract = paper.get('abstract', '')
                    print(f"   Paper {j+1}: {title[:100]}...")
                    if abstract:
                        # Check for material keywords
                        material_keywords = ['hydrogel', 'polymer', 'chitosan', 'drug delivery', 'transdermal']
                        found_keywords = [kw for kw in material_keywords if kw.lower() in abstract.lower()]
                        if found_keywords:
                            print(f"   Keywords found: {found_keywords}")
            
            all_papers.extend(papers)
        
        # Remove duplicates
        unique_papers = self._deduplicate_papers(all_papers)
        update_progress(f"└─ Total unique papers found: {len(unique_papers)} (after deduplication)")
        
        if not unique_papers:
            return {"error": "No relevant papers found", "candidates": []}
        
        # Explain the extraction process
        extraction_explanation = self._generate_process_explanation(
            user_request, "extraction", unique_papers=len(unique_papers)
        )
        update_progress(extraction_explanation, "explanation")
        
        # Extract material information using customized LLM prompt
        update_progress("Phase 3: Enhanced Material Analysis")
        update_progress("└─ Extracting material properties and data...")
        material_candidates = self._extract_material_data_focused(
            unique_papers[:max_papers], 
            extraction_focus,
            progress_callback=update_progress
        )
        
        update_progress(f"└─ Extracted {len(material_candidates)} material candidates")
        
        # Generate final explanation
        final_explanation = self._generate_process_explanation(
            user_request, "results", material_count=len(material_candidates)
        )
        update_progress(final_explanation, "explanation")
        
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
        
        update_progress(f"Phase 4: AI Enhancement")
        update_progress(f"└─ Enhanced analysis for {len(material_candidates)} materials")
        update_progress(final_explanation, "explanation")
        update_progress(f"Enhanced Literature Mining Complete! Found {len(material_candidates)} materials from literature", "complete")
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

You are an expert in materials science and biomedical engineering. Based on the user's request, generate 5 specific, targeted search queries that will find the most relevant research papers.

Consider:
- Key materials mentioned or implied in the request
- Temperature stability requirements (room temperature, 25-40°C)
- Insulin delivery mechanisms (transdermal, controlled release)
- Biocompatibility requirements
- Both experimental and computational research approaches

OUTPUT FORMAT (JSON):
{{
  "relevant": true/false,
  "relevance_score": 1-10,
  "interpretation": "Brief explanation of how you interpreted the user's request",
  "search_queries": [
    "specific search query 1 targeting main material type",
    "specific search query 2 targeting application area", 
    "specific search query 3 targeting stability mechanism",
    "specific search query 4 targeting delivery method",
    "specific search query 5 targeting characterization/testing"
  ],
  "extraction_focus": "What specific material properties and data to prioritize during extraction"
}}

EXAMPLES:
- Request: "smart polymers" → Queries about thermoresponsive polymers, PNIPAM, temperature-controlled release
- Request: "nanotechnology" → Queries about nanoparticles, nanocarriers, encapsulation, targeted delivery
- Request: "synthetic polymers" → Queries about PLGA, PEG, synthetic biodegradable polymers, controlled release

Generate search queries that are specific enough to find high-quality research but broad enough to capture relevant work. Use scientific terminology that researchers would actually use in their papers."""

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
                                      recent_only: bool = False,
                                      save_results: bool = True) -> Dict:
        """
        Mine literature for materials suitable for insulin delivery patches.
        (Legacy method - use intelligent_mining for enhanced functionality)
        
        Args:
            max_papers (int): Maximum number of papers to analyze
            recent_only (bool): Whether to focus on recent papers (2020+) - now defaults to False
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

IMPORTANT: Extract specific material names and properties from the papers. Do NOT provide generic summaries.

OUTPUT FORMAT: Provide ONLY valid JSON objects, one per line. Each material should have this exact structure:

{"material_name": "Specific material name (e.g., Chitosan, PEG-400, Alginate)", "material_composition": "Chemical composition and formula", "chemical_structure": "Structural information", "thermal_stability_temp_range": "Temperature stability range with specific temperatures", "insulin_stability_duration": "Reported insulin stability duration", "biocompatibility_data": "Biocompatibility information", "release_kinetics": "Release kinetics and delivery mechanism", "delivery_efficiency": "Delivery efficiency data", "stabilization_mechanism": "Mechanism of protein stabilization", "literature_references": ["Paper title 1", "Paper title 2"], "confidence_score": 8}

EXAMPLES:
{"material_name": "Chitosan nanoparticles", "material_composition": "Deacetylated chitin polymer", "chemical_structure": "Linear polysaccharide with amino groups", "thermal_stability_temp_range": "Stable up to 60°C for 24h", "insulin_stability_duration": "72 hours at 25°C", "biocompatibility_data": "FDA approved, non-toxic", "release_kinetics": "pH-responsive controlled release", "delivery_efficiency": "85% transdermal permeation", "stabilization_mechanism": "Electrostatic interaction with insulin", "literature_references": ["Chitosan-based insulin delivery systems"], "confidence_score": 9}

{"material_name": "PEG-PLGA copolymer", "material_composition": "Polyethylene glycol-poly(lactic-co-glycolic acid)", "chemical_structure": "Block copolymer with hydrophilic PEG and hydrophobic PLGA", "thermal_stability_temp_range": "Stable 20-40°C for weeks", "insulin_stability_duration": "5 days at room temperature", "biocompatibility_data": "Biodegradable, FDA approved components", "release_kinetics": "Sustained release over 72h", "delivery_efficiency": "60% bioavailability", "stabilization_mechanism": "Encapsulation and reduced aggregation", "literature_references": ["PLGA microspheres for protein delivery"], "confidence_score": 8}

Extract information ONLY if supported by the provided papers. Focus on specific materials mentioned by name in the abstracts and titles. Provide up to 10 material candidates."""
    
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
        
        print("🔍 DEBUG: Starting LLM response parsing...")
        
        # First try to find JSON objects in the response
        lines = response_text.split('\n')
        json_buffer = ""
        in_json = False
        json_found_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('{'):
                in_json = True
                json_buffer = line
                json_found_count += 1
            elif in_json:
                json_buffer += "\n" + line
                if line.endswith('}'):
                    try:
                        material_data = json.loads(json_buffer)
                        print(f"🔍 DEBUG: Successfully parsed JSON {json_found_count}")
                        print(f"   Material name: {material_data.get('material_name', 'NONE')}")
                        
                        # Validate that this is a proper material entry
                        if ('material_name' in material_data and 
                            material_data.get('material_name', '').strip() and
                            material_data.get('material_name') != 'Extracted from text'):
                            materials.append(material_data)
                            print(f"   ✅ Added material: {material_data.get('material_name')}")
                        else:
                            print(f"   ❌ Rejected material (invalid name): {material_data.get('material_name')}")
                    except json.JSONDecodeError as e:
                        print(f"   ❌ JSON decode error: {e}")
                    in_json = False
                    json_buffer = ""
        
        print(f"🔍 DEBUG: Found {json_found_count} JSON objects, {len(materials)} valid materials")
        
        # Try alternative JSON parsing approaches
        if not materials:
            print("🔍 DEBUG: Trying alternative JSON parsing...")
            # Look for JSON objects more flexibly
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            print(f"🔍 DEBUG: Found {len(json_matches)} potential JSON matches with regex")
            
            for i, match in enumerate(json_matches):
                try:
                    material_data = json.loads(match)
                    print(f"   JSON {i+1}: {material_data.get('material_name', 'NONE')}")
                    if ('material_name' in material_data and 
                        material_data.get('material_name', '').strip() and
                        material_data.get('material_name') != 'Extracted from text'):
                        materials.append(material_data)
                        print(f"   ✅ Added material: {material_data.get('material_name')}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"   ❌ JSON error: {e}")
                    continue
        
        # If no valid JSON materials found, analyze papers individually
        if not materials:
            print("⚠️  No valid JSON materials found, analyzing papers individually...")
            materials = self._analyze_papers_individually(response_text)
        else:
            print(f"✅ Successfully extracted {len(materials)} materials from JSON")
        
        return materials[:10]  # Limit to 10 materials

    def _analyze_papers_individually(self, llm_response: str) -> List[Dict]:
        """
        Analyze each paper individually to extract real material information.
        This replaces the terrible fallback that generated fake data.
        """
        materials = []
        
        print("🔍 DEBUG: Starting individual paper analysis...")
        
        # Find paper sections in the LLM response
        paper_sections = self._extract_paper_sections(llm_response)
        
        print(f"🔍 DEBUG: Found {len(paper_sections)} paper sections")
        
        if not paper_sections:
            # Try a different approach - look for materials directly in the response text
            print("🔍 DEBUG: No paper sections found, trying direct material extraction...")
            materials = self._extract_materials_directly(llm_response)
            if materials:
                print(f"✅ Found {len(materials)} materials via direct extraction")
                return materials
            else:
                print("❌ Could not extract material information from papers")
                return []
        
        for i, section in enumerate(paper_sections, 1):
            print(f"🔍 DEBUG: Analyzing section {i}")
            material = self._extract_material_from_section(section, i)
            if material:
                materials.append(material)
                print(f"   ✅ Extracted material: {material.get('material_name')}")
            else:
                print(f"   ❌ No material found in section {i}")
        
        print(f"✅ Individual analysis found {len(materials)} materials")
        return materials

    def _extract_materials_directly(self, text: str) -> List[Dict]:
        """
        Directly extract materials from LLM response text using keyword matching.
        This is a fallback when structured parsing fails.
        """
        materials = []
        
        # Look for common material keywords in the text
        material_keywords = {
            'hydrogel': ['hydrogel', 'gel'],
            'chitosan': ['chitosan'],
            'alginate': ['alginate'],
            'PLGA': ['PLGA', 'poly(lactic-co-glycolic acid)', 'polylactic'],
            'PLA': ['PLA', 'polylactic acid', 'poly(lactic acid)'],
            'PEG': ['PEG', 'polyethylene glycol'],
            'PCL': ['PCL', 'polycaprolactone'],
            'collagen': ['collagen'],
            'gelatin': ['gelatin'],
            'hyaluronic acid': ['hyaluronic acid'],
            'nanoparticle': ['nanoparticle', 'nano particle'],
            'liposome': ['liposome'],
            'microsphere': ['microsphere'],
            'polymer': ['polymer', 'copolymer']
        }
        
        text_lower = text.lower()
        
        for material_name, keywords in material_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # Create a basic material entry
                material = {
                    'material_name': material_name,
                    'material_composition': 'Mentioned in literature analysis',
                    'thermal_stability_temp_range': 'Not specified',
                    'biocompatibility_data': 'Not specified',
                    'stabilization_mechanism': 'Not specified',
                    'confidence_score': 5,
                    'extraction_method': 'Direct keyword extraction',
                    'source': 'LLM analysis text'
                }
                materials.append(material)
        
        return materials
    
    def _extract_paper_sections(self, llm_response: str) -> List[Dict]:
        """Extract individual paper analysis sections from LLM response."""
        sections = []
        
        # Look for paper markers like "PAPER 1:", "Paper #2", etc.
        import re
        paper_pattern = r'(?:PAPER\s*#?\d+:|Paper\s*#?\d+:|\d+\.\s*[A-Z])'
        
        parts = re.split(paper_pattern, llm_response, flags=re.IGNORECASE)
        
        # Skip the first part (usually intro text)
        for i, part in enumerate(parts[1:], 1):
            if part.strip():
                sections.append({
                    'paper_number': i,
                    'content': part.strip()
                })
        
        return sections
    
    def _extract_material_from_section(self, section: Dict, paper_num: int) -> Dict:
        """Extract material information from a specific paper section."""
        content = section['content']
        
        # Look for specific material indicators in the content
        material_data = {
            'paper_number': paper_num,
            'material_name': self._find_material_name(content),
            'citation': self._find_citation(content),
            'composition': self._find_composition(content),
            'properties': self._find_properties(content),
            'confidence_score': self._calculate_confidence(content)
        }
        
        # Only return if we found a real material name
        if material_data['material_name'] and material_data['material_name'] != 'Unknown':
            return material_data
        
        return None
    
    def _find_material_name(self, content: str) -> str:
        """Find the primary material name mentioned in the content."""
        import re
        
        # Common material patterns
        material_patterns = [
            r'(?:poly\([^)]+\))',  # poly(something)
            r'(?:PLGA|PLA|PCL|PEG|PLLA|PDLA)',  # Common polymer abbreviations  
            r'(?:chitosan|alginate|collagen|gelatin|hyaluronic\s+acid)',  # Natural polymers
            r'(?:hydrogel|nanoparticle|microsphere|liposome)',  # Delivery systems
            r'(?:polymer|copolymer|terpolymer)',  # General polymer terms
        ]
        
        content_lower = content.lower()
        for pattern in material_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                return matches[0].title()
        
        return None
    
    def _find_citation(self, content: str) -> str:
        """Extract citation information from the content."""
        # Look for title and authors in the content
        lines = content.split('\n')
        citation_info = []
        
        for line in lines:
            line = line.strip()
            if 'title:' in line.lower():
                citation_info.append(line)
            elif 'author' in line.lower():
                citation_info.append(line)
            elif 'year:' in line.lower():
                citation_info.append(line)
            elif 'journal:' in line.lower():
                citation_info.append(line)
        
        return ' | '.join(citation_info) if citation_info else 'Citation information not extracted'
    
    def _find_composition(self, content: str) -> str:
        """Find composition details in the content."""
        composition_keywords = ['composition', 'ratio', 'molecular weight', 'formula', 'structure']
        
        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in composition_keywords):
                return line.strip()
        
        return 'Composition details not specified in abstract'
    
    def _find_properties(self, content: str) -> Dict:
        """Find material properties in the content."""
        properties = {}
        
        property_keywords = {
            'thermal_stability': ['thermal', 'temperature', 'stability', 'degradation'],
            'biocompatibility': ['biocompatible', 'cytotoxicity', 'toxicity', 'safe'],
            'release_kinetics': ['release', 'controlled', 'sustained', 'kinetics'],
            'mechanical': ['mechanical', 'strength', 'modulus', 'elastic']
        }
        
        content_lower = content.lower()
        for prop_name, keywords in property_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                properties[prop_name] = 'Mentioned in paper'
            else:
                properties[prop_name] = 'Not specified'
        
        return properties
    
    def _calculate_confidence(self, content: str) -> int:
        """Calculate confidence score based on content quality."""
        score = 5  # Base score
        
        # Increase score for specific details
        if 'molecular weight' in content.lower():
            score += 1
        if 'temperature' in content.lower():
            score += 1  
        if 'biocompatible' in content.lower():
            score += 1
        if 'insulin' in content.lower():
            score += 1
        if len(content) > 200:  # More detailed content
            score += 1
            
        return min(score, 10)
    
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

    def _extract_material_data_focused(self, papers: List[Dict], extraction_focus: str, progress_callback = None) -> List[Dict]:
        """
        Extract detailed per-paper material information using LLM analysis.
        This replaces the old JSON-based extraction with a fully LLM-driven approach.
        """
        material_candidates = []
        
        try:
            if progress_callback:
                progress_callback(f"Analyzing papers individually with focus: {extraction_focus}")
                progress_callback("   └─ Generating detailed per-paper material analysis...")
            else:
                print(f"Analyzing papers individually with focus: {extraction_focus}")
            
            print(f"🔍 DEBUG: Analyzing {len(papers)} papers individually")
            
            # Analyze each paper individually for detailed material information
            for i, paper in enumerate(papers[:10], 1):  # Limit to 10 papers for faster web performance
                if progress_callback:
                    progress_callback(f"   └─ Analyzing paper {i}/{min(10, len(papers))}: {paper.get('title', 'Unknown')[:50]}...")
                
                material_analysis = self._analyze_single_paper_for_materials(paper, extraction_focus, i)
                
                if material_analysis:
                    material_candidates.extend(material_analysis)
                    print(f"   ✅ Paper {i}: Found {len(material_analysis)} material(s)")
                else:
                    print(f"   ❌ Paper {i}: No relevant materials found")
            
            if progress_callback:
                progress_callback(f"   └─ Completed individual analysis: {len(material_candidates)} materials found")
            
            print(f"✅ Individual analysis completed: {len(material_candidates)} total materials found")
            return material_candidates
            
        except Exception as e:
            error_msg = f"Error in individual paper analysis: {e}"
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            else:
                print(error_msg)
            return []
    
    def _analyze_single_paper_for_materials(self, paper: Dict, extraction_focus: str, paper_number: int) -> List[Dict]:
        """
        Analyze a single paper for material information using LLM.
        Returns detailed material information with Harvard citations.
        """
        # DEBUG: Print the actual paper structure
        print(f"🔍 DEBUG: Paper {paper_number} structure:")
        print(f"   Type: {type(paper)}")
        if isinstance(paper, dict):
            print(f"   Keys: {list(paper.keys())}")
            print(f"   Title type: {type(paper.get('title'))}")
            print(f"   Authors type: {type(paper.get('authors'))}")
            if paper.get('authors'):
                print(f"   Authors content: {paper.get('authors')}")
        else:
            print(f"   Paper is not a dict: {paper}")
            return []
        
        title = paper.get('title', 'Unknown Title')
        abstract = paper.get('abstract', 'No abstract available')
        authors = paper.get('authors', [])
        year = paper.get('year', 'Unknown Year')
        journal = paper.get('journal', 'Unknown Journal')
        
        # Generate Harvard citation
        harvard_citation = self._generate_harvard_citation(paper)
        
        # Create analysis prompt for this specific paper
        analysis_prompt = f"""# Individual Paper Analysis for Insulin Delivery Materials

RESEARCH FOCUS: {extraction_focus}

PAPER TO ANALYZE:
Title: {title}
Authors: {', '.join([str(author) for author in authors[:3]] if isinstance(authors, list) else [str(authors)])}{'...' if isinstance(authors, list) and len(authors) > 3 else ''}
Year: {year}
Journal: {journal}
Abstract: {abstract}

TASK: Analyze this paper to identify specific materials that could be relevant for insulin stabilization and delivery, especially for room temperature storage applications.

ANALYSIS REQUIREMENTS:
1. Look for specific material names (polymers, nanoparticles, hydrogels, etc.)
2. Extract composition and chemical structure information
3. Find thermal stability data and temperature ranges
4. Identify stabilization mechanisms for proteins/insulin
5. Note biocompatibility and delivery efficiency data

OUTPUT FORMAT: For each relevant material found in this paper, provide a detailed analysis following this structure:

Material: [Specific material name]
Citation: {harvard_citation}
Composition: [Chemical composition, molecular formula, or structural details]
Thermal Stability: [Temperature range, duration, specific stability data]
Stabilization Mechanism: [How the material stabilizes proteins/insulin]
Biocompatibility: [Safety data, toxicity information, regulatory status]
Delivery Properties: [Release kinetics, permeation data, efficiency]
Key Findings: [Most important research findings about this material]
Confidence: [1-10 score based on data quality and relevance]

IMPORTANT INSTRUCTIONS:
- Only analyze materials explicitly mentioned in this paper
- Be specific about temperatures, time periods, and quantitative data
- If no relevant materials are found, respond with "No relevant materials identified in this paper"
- Focus on materials with demonstrated insulin or protein interaction
- Extract exact values and data points when available
- Prioritize materials relevant to: {extraction_focus}

Analyze the paper now:"""
        
        try:
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user',
                    'content': analysis_prompt
                }],
                options={
                    'temperature': 0.2,  # Low temperature for factual analysis
                    'num_predict': 2000
                }
            )
            
            analysis_text = response['message']['content'].strip()
            
            # Parse the analysis into structured material data
            materials = self._parse_single_paper_analysis(analysis_text, harvard_citation, paper_number)
            
            return materials
            
        except Exception as e:
            print(f"   ❌ Error analyzing paper {paper_number}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_harvard_citation(self, paper: Dict) -> str:
        """
        Generate Harvard format citation for a paper.
        Handles the actual data structure from Semantic Scholar API.
        """
        try:
            # Ensure paper is a dictionary
            if not isinstance(paper, dict):
                return "Unknown Author (n.d.). Unknown Title. Unknown Journal."
            
            authors = paper.get('authors', [])
            title = paper.get('title', 'Unknown Title')
            year = paper.get('year', 'n.d.')
            journal = paper.get('journal', 'Unknown Journal')
            
            # Handle different author formats
            author_text = "Unknown Author"
            
            if authors:
                if isinstance(authors, list):
                    # Authors is a list - handle both string items and dict items
                    author_names = []
                    for author in authors:
                        try:
                            if isinstance(author, dict):
                                # Author is a dictionary with 'name' field
                                name = author.get('name', 'Unknown Author')
                            elif isinstance(author, str):
                                # Author is already a string
                                name = author
                            else:
                                # Convert to string as fallback
                                name = str(author)
                            author_names.append(name)
                        except Exception as e:
                            # If anything goes wrong with this author, skip it
                            print(f"Warning: Skipping author due to error: {e}")
                            continue
                    
                    # Format author list
                    if len(author_names) == 0:
                        author_text = "Unknown Author"
                    elif len(author_names) == 1:
                        author_text = author_names[0]
                    elif len(author_names) == 2:
                        author_text = f"{author_names[0]} and {author_names[1]}"
                    else:
                        author_text = f"{author_names[0]} et al."
                        
                elif isinstance(authors, str):
                    # Authors is a single string
                    author_text = authors
                else:
                    # Convert to string as fallback
                    author_text = str(authors)
            
            # Format: Author, A. (Year). Title. Journal.
            citation = f"{author_text} ({year}). {title}. {journal}."
            return citation
            
        except Exception as e:
            print(f"Error generating citation: {e}")
            # Robust fallback that handles any data structure
            try:
                if isinstance(paper, dict):
                    title = str(paper.get('title', 'Unknown Title'))
                    year = str(paper.get('year', 'n.d.'))
                    return f"Unknown Author ({year}). {title}. Unknown Journal."
                else:
                    return "Unknown Author (n.d.). Unknown Title. Unknown Journal."
            except Exception:
                return "Unknown Author (n.d.). Unknown Title. Unknown Journal."
    
    def _parse_single_paper_analysis(self, analysis_text: str, citation: str, paper_number: int) -> List[Dict]:
        """
        Parse the LLM analysis of a single paper into structured material data.
        """
        materials = []
        
        # Check if no materials were found
        if "no relevant materials identified" in analysis_text.lower():
            return []
        
        # Split analysis into sections by "Material:" markers
        import re
        material_sections = re.split(r'\n\s*Material:\s*', analysis_text, flags=re.IGNORECASE)
        
        # Skip the first section (usually intro text)
        for i, section in enumerate(material_sections[1:], 1):
            material_data = self._extract_material_from_analysis_section(section, citation, paper_number, i)
            if material_data:
                materials.append(material_data)
        
        return materials
    
    def _extract_material_from_analysis_section(self, section: str, citation: str, paper_number: int, material_number: int) -> Dict:
        """
        Extract structured material data from a single analysis section.
        """
        try:
            lines = section.strip().split('\n')
            material_name = lines[0].strip() if lines else "Unknown Material"
            
            # Initialize material data structure
            material_data = {
                'material_name': material_name,
                'harvard_citation': citation,
                'paper_number': paper_number,
                'material_composition': 'Not specified',
                'thermal_stability_temp_range': 'Not specified',
                'stabilization_mechanism': 'Not specified',
                'biocompatibility_data': 'Not specified',
                'delivery_properties': 'Not specified',
                'key_findings': 'Not specified',
                'confidence_score': 5,
                'extraction_method': 'Individual LLM Analysis'
            }
            
            # Parse each field from the analysis
            current_field = None
            current_content = []
            
            for line in lines[1:]:
                line = line.strip()
                
                # Check for field markers
                if line.lower().startswith('citation:'):
                    current_field = 'harvard_citation'
                    current_content = [line.replace('Citation:', '').strip()]
                elif line.lower().startswith('composition:'):
                    current_field = 'material_composition'
                    current_content = [line.replace('Composition:', '').strip()]
                elif line.lower().startswith('thermal stability:'):
                    current_field = 'thermal_stability_temp_range'
                    current_content = [line.replace('Thermal Stability:', '').strip()]
                elif line.lower().startswith('stabilization mechanism:'):
                    current_field = 'stabilization_mechanism'
                    current_content = [line.replace('Stabilization Mechanism:', '').strip()]
                elif line.lower().startswith('biocompatibility:'):
                    current_field = 'biocompatibility_data'
                    current_content = [line.replace('Biocompatibility:', '').strip()]
                elif line.lower().startswith('delivery properties:'):
                    current_field = 'delivery_properties'
                    current_content = [line.replace('Delivery Properties:', '').strip()]
                elif line.lower().startswith('key findings:'):
                    current_field = 'key_findings'
                    current_content = [line.replace('Key Findings:', '').strip()]
                elif line.lower().startswith('confidence:'):
                    confidence_text = line.replace('Confidence:', '').strip()
                    try:
                        # Extract number from confidence text
                        import re
                        confidence_match = re.search(r'(\d+)', confidence_text)
                        if confidence_match:
                            material_data['confidence_score'] = int(confidence_match.group(1))
                    except:
                        pass
                elif current_field and line:
                    # Continue previous field
                    current_content.append(line)
                elif current_field and current_content:
                    # Save current field
                    material_data[current_field] = ' '.join(current_content).strip()
                    current_field = None
                    current_content = []
            
            # Save any remaining field
            if current_field and current_content:
                material_data[current_field] = ' '.join(current_content).strip()
            
            # Only return if we have a valid material name
            if material_data['material_name'] and material_data['material_name'] != "Unknown Material":
                return material_data
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error parsing material section: {e}")
            return None

    def _generate_process_explanation(self, user_request: str, phase: str, 
                                    search_queries: List[str] = None, 
                                    extraction_focus: str = None,
                                    unique_papers: int = None,
                                    material_count: int = None) -> str:
        """
        Generate explanatory text about what the system is doing at each phase using LLM.
        
        Args:
            user_request (str): The original user request
            phase (str): Current phase - 'start', 'strategy', 'extraction', 'results'
            search_queries (List[str]): Generated search queries (for strategy phase)
            extraction_focus (str): Focus area for extraction (for strategy phase)
            unique_papers (int): Number of papers found (for extraction phase)
            material_count (int): Number of materials found (for results phase)
        
        Returns:
            str: AI-generated explanatory text for the current phase
        """
        try:
            if phase == "start":
                prompt = f"""You are an AI research assistant about to help with literature mining for insulin delivery materials. 

The user has asked: "{user_request}"

Explain in a conversational, enthusiastic way:
1. What you understand from their request
2. Your general approach to finding relevant materials
3. Why literature mining is valuable for this type of research

Keep it concise (3-4 sentences), friendly, and show your expertise. Start with something like "I'm excited to help you find..." or "Let me dive into..."

Do not use numbered lists or bullet points. Write in flowing paragraphs."""

            elif phase == "strategy":
                query_list = "\n".join([f"• {query}" for query in (search_queries or [])])
                prompt = f"""You just generated these search queries for the user's request "{user_request}":

{query_list}

Focus area: {extraction_focus or 'General materials research'}

Explain your reasoning behind these specific queries:
1. Why you chose these particular search terms
2. What aspects of insulin delivery they target
3. How they complement each other to cover the research space

Be conversational and show your scientific reasoning. Explain like you're talking to a colleague. Keep it 4-5 sentences."""

            elif phase == "extraction":
                prompt = f"""You've found {unique_papers} relevant papers for the user's request about "{user_request}".

Now you're about to analyze these papers to extract material information. Explain:
1. What specific information you're looking for in the papers
2. Why these types of data are important for insulin delivery applications
3. Your approach to ensuring the extracted information is reliable

Be enthusiastic about the scientific process. Keep it conversational and show expertise. 3-4 sentences."""

            elif phase == "results":
                prompt = f"""You've completed analyzing papers for "{user_request}" and found {material_count} material candidates.

Provide a factual summary:
1. Simply report the number of materials extracted from the literature
2. Mention that the materials come from the analyzed scientific papers
3. Remind the user that this is raw extracted data from literature

Be factual and objective. Do NOT assess quality, promise levels, or biocompatibility. Do NOT filter or judge the materials. Simply report what was found in the literature. 2-3 sentences."""

            else:
                return "🔄 Processing your request..."

            # Generate the explanation using LLM
            response = self.ollama.client.chat(
                model=self.ollama.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,  # Higher temperature for more natural language
                    'num_predict': 300   # Shorter responses
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            # Fallback to simple message
            return f"🔄 Working on {phase} phase for your request..."


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