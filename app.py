#!/usr/bin/env python3
"""
Insulin AI Web Application
A ChatGPT-like interface for AI-driven design of fridge-free insulin delivery patches.

This Flask application provides a web interface for literature mining, material discovery,
and interactive chat with the AI system for insulin delivery research.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, session, Response
from flask_cors import CORS
import time
from threading import Thread
import queue
import urllib.parse
import re
import random

from literature_mining_system import MaterialsLiteratureMiner
from chatbot_system import InsulinAIChatbot
from mcp_client import SimplifiedMCPLiteratureMinerSync
from psmiles_generator import PSMILESGenerator
from psmiles_processor import PSMILESProcessor
from llamol_integration import llamol_manager, LLAMOL_AVAILABLE

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'insulin-ai-secret-key-2024')
CORS(app)

# Global variables
literature_miner = None
mcp_literature_miner = None
chatbot = None
psmiles_generator = None
psmiles_processor = None

def initialize_systems():
    """Initialize the literature mining, chatbot, and PSMILES generation systems."""
    global literature_miner, mcp_literature_miner, chatbot, psmiles_generator, psmiles_processor
    
    try:
        # Initialize literature mining system
        semantic_scholar_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        
        literature_miner = MaterialsLiteratureMiner(
            semantic_scholar_api_key=semantic_scholar_key,
            ollama_model=ollama_model,
            ollama_host=ollama_host
        )
        
        # Initialize chatbot system with enhanced memory and model selection
        memory_type = os.environ.get('CHATBOT_MEMORY_TYPE', 'buffer_window')
        memory_dir = os.environ.get('CHATBOT_MEMORY_DIR', 'chat_memory')
        default_model_type = os.environ.get('DEFAULT_MODEL_TYPE', 'ollama')  # 'ollama' or 'llamol'
        llamol_model = os.environ.get('LLAMOL_MODEL', 'osunlp/LlaSMol-Mistral-7B')
        
        chatbot = InsulinAIChatbot(
            model_type=default_model_type,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            llamol_model=llamol_model,
            memory_type=memory_type,
            memory_dir=memory_dir
        )
        
        # Initialize PSMILES generator with model selection support
        try:
            psmiles_model_type = os.environ.get('PSMILES_MODEL_TYPE', 'ollama')
            if psmiles_model_type == 'llamol' and LLAMOL_AVAILABLE:
                # Use LlaSMol for PSMILES generation
                psmiles_generator = PSMILESGenerator(
                    model_type='llamol',
                    llamol_model=llamol_model
                )
            else:
                # Use Ollama for PSMILES generation
                psmiles_generator = PSMILESGenerator(
                    model_type='ollama',
                    ollama_model=ollama_model,
                    ollama_host=ollama_host
                )
            print("✅ PSMILES Generator initialized successfully!")
        except Exception as e:
            print(f"⚠️ PSMILES Generator initialization failed: {e}")
            psmiles_generator = None
        
        # Initialize enhanced PSMILES processor
        try:
            psmiles_processor = PSMILESProcessor()
            print("✅ PSMILES Processor initialized successfully!")
        except Exception as e:
            print(f"⚠️ PSMILES Processor initialization failed: {e}")
            psmiles_processor = None
        
        # Initialize MCP-enhanced literature miner (kept for future use, currently disabled)
        # try:
        #     mcp_literature_miner = SimplifiedMCPLiteratureMinerSync(ollama_client=literature_miner.ollama)
        #     print("✅ MCP Literature Miner initialized successfully!")
        # except Exception as e:
        #     print(f"⚠️ MCP Literature Miner initialization failed: {e}")
        #     mcp_literature_miner = None
        mcp_literature_miner = None  # Disabled for simplification to pure SemanticScholar
        
        print("✅ Systems initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing systems: {e}")
        return False

@app.route('/')
def index():
    """Main page with ChatGPT-like interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the frontend with multi-model support."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        chat_type = data.get('type', 'general')  # 'general', 'literature', 'research', 'mcp', 'psmiles'
        model_type = data.get('model_type')  # Optional: 'ollama' or 'llamol'
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID for conversation history
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Handle model switching if requested
        if model_type and chatbot:
            switch_success = chatbot.switch_model(model_type)
            if not switch_success:
                return jsonify({
                    'error': f'Failed to switch to {model_type} model',
                    'current_model': chatbot.get_model_info()
                }), 400
        
        # Route message based on type
        if chat_type == 'literature':
            response = handle_literature_query(message, session_id)
        elif chat_type == 'research':
            response = handle_research_query(message, session_id)
        elif chat_type == 'psmiles':
            response = handle_psmiles_query(message, session_id)
        else:
            response = handle_general_chat(message, session_id)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def handle_mcp_literature_query(message: str, session_id: str) -> Dict:
    """Handle MCP-powered literature mining queries."""
    try:
        if not mcp_literature_miner:
            return {
                'type': 'error',
                'message': 'MCP Literature mining system not available. Please check MCP server and dependencies.',
                'suggestion': 'You can still use the standard literature mining by selecting "Literature Mining" mode.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Collect progress updates
        progress_messages = []
        
        def progress_callback(msg, step_type="info"):
            """Callback to collect progress messages."""
            progress_messages.append({
                'message': msg,
                'type': step_type,
                'timestamp': datetime.now().isoformat()
            })
        
        # Perform MCP-enhanced intelligent mining with progress tracking
        results = mcp_literature_miner.intelligent_mining_with_mcp(
            user_request=message,
            max_papers=30,
            recent_only=True,
            progress_callback=progress_callback
        )
        
        if 'error' in results:
            return {
                'type': 'error',
                'message': results['error'],
                'suggestion': 'Try using the standard literature mining mode instead.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Format the complete response with progress and results
        material_candidates = results.get('material_candidates', [])
        papers_count = results.get('papers_analyzed', 0)
        material_focus = results.get('material_focus', 'general')
        paper_summaries = results.get('paper_summaries', [])
        
        # Build the progress narrative
        progress_narrative = ""
        for update in progress_messages:
            if update['type'] == 'explanation':
                progress_narrative += f"\n{update['message']}\n"
            elif update['type'] in ['start', 'complete']:
                progress_narrative += f"\n{update['message']}\n"
            else:
                progress_narrative += f"\n{update['message']}\n"
        
        response_message = f"""
## 🚀 MCP-Powered AI Literature Mining

{progress_narrative}

---

## 📊 Enhanced Results Summary

Query: {message}

Material Focus Detected: {material_focus.title()}

Papers Analyzed: {papers_count} research papers

Enhanced Material Insights: {len(material_candidates)} findings

### 🧪 Material Candidates with AI Analysis:
"""
        
        # Handle the enhanced material candidate format
        for i, candidate in enumerate(material_candidates[:8], 1):  # Show top 8
            # Handle both enhanced and standard formats
            if isinstance(candidate, dict):
                name = candidate.get('material_name', 'Unknown Material')
                composition = candidate.get('material_composition', 'Composition not specified')
                thermal_stability = candidate.get('thermal_stability_temp_range', 'Not specified')
                mechanism = candidate.get('stabilization_mechanism', 'Not specified')
                confidence = candidate.get('confidence_score', 'N/A')
                
                # Look for AI insight if available
                ai_insight = candidate.get('ai_insight', '')
                paper_title = candidate.get('source_paper', {}).get('title', '') if isinstance(candidate.get('source_paper'), dict) else ''
                
                response_message += f"""
{i}. {name}
   - Composition: {composition}
   - Thermal Stability: {thermal_stability}
   - Stabilization Mechanism: {mechanism}
   - Confidence Score: {confidence}/10"""
   
                if paper_title:
                    response_message += f"\n   - Source Paper: {paper_title}"
                if ai_insight:
                    response_message += f"\n   - AI Insight: {ai_insight}"
                
                response_message += "\n"
            else:
                # Fallback for unexpected format
                response_message += f"""
{i}. Material Finding {i}
   - Details: {str(candidate)[:200]}...
"""
        
        if len(material_candidates) > 8:
            response_message += f"\n*Showing top 8 of {len(material_candidates)} enhanced insights. Full results saved to files.*"
        
        # Add detailed per-paper summaries
        if paper_summaries:
            response_message += f"""

---

## 📚 Detailed Paper Analysis

Found {len(paper_summaries)} papers with comprehensive analysis:
"""
            
            for summary in paper_summaries[:5]:  # Show top 5 paper summaries
                title = summary.get('title', 'Unknown Title')
                year = summary.get('year', 'Unknown')
                materials = summary.get('materials_discussed', [])
                key_findings = summary.get('key_findings', 'No findings available')
                relevance = summary.get('relevance_to_query', 'Relevance analysis unavailable')
                material_insights = summary.get('material_insights', '')
                citation_count = summary.get('citation_count', 0)
                
                response_message += f"""

{summary.get('paper_number', 'N/A')}. {title} ({year})
   Citations: {citation_count}
   
   Materials Discussed: {', '.join(materials) if materials else 'General materials research'}
   
   Key Findings: {key_findings}
   
   Relevance to Query: {relevance}"""
   
                if material_insights:
                    response_message += f"\n   \n   Material Insights: {material_insights}"
                    
                response_message += "\n"
            
            if len(paper_summaries) > 5:
                response_message += f"\n*Showing 5 of {len(paper_summaries)} detailed paper analyses. Full results include comprehensive summaries for all papers.*"
        
        response_message += f"""

---

## 🎯 MCP Enhancement Benefits:
- Advanced Search Strategy: Intelligent query generation with domain-specific targeting
- Material-Focused Analysis: AI-driven extraction based on detected research focus
- Enhanced AI Insights: Local LLM provides contextual analysis and natural language explanations
- Real-time Progress Updates: Natural language explanations of AI reasoning process
- Detailed Per-Paper Analysis: Comprehensive summaries of material candidates per research paper
- Structured Data Storage: Results saved with metadata for future reference

*This mining session used the Model Context Protocol (MCP) for enhanced Semantic Scholar integration with AI-generated process explanations.*
"""
        
        return {
            'type': 'mcp_literature',
            'message': response_message,
            'data': results,
            'progress': progress_messages,
            'metadata': {
                'papers_analyzed': papers_count,
                'material_focus': material_focus,
                'mcp_enhanced': True,
                'paper_summaries_count': len(paper_summaries)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'MCP literature mining error: {str(e)}',
            'suggestion': 'Try using the standard literature mining mode.',
            'timestamp': datetime.now().isoformat()
        }

def handle_literature_query(message: str, session_id: str) -> Dict:
    """Handle literature mining queries."""
    try:
        if not literature_miner:
            return {
                'type': 'error',
                'message': 'Literature mining system not available. Please check Ollama connection.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Collect progress updates
        progress_messages = []
        
        def progress_callback(msg, step_type="info"):
            """Callback to collect progress messages."""
            progress_messages.append({
                'message': msg,
                'type': step_type,
                'timestamp': datetime.now().isoformat()
            })
        
        # Perform intelligent mining with progress tracking
        results = literature_miner.intelligent_mining(
            user_request=message,
            max_papers=15,  # Reduced from 30 to 15 for faster processing
            recent_only=True,
            save_results=True,
            progress_callback=progress_callback
        )
        
        if 'error' in results:
            return {
                'type': 'error',
                'message': results['error'],
                'suggestion': results.get('suggestion', ''),
                'timestamp': datetime.now().isoformat()
            }
        
        # Format the complete response with progress and results
        material_count = len(results.get('material_candidates', []))
        papers_count = results.get('papers_analyzed', 0)
        
        # Build the progress narrative
        progress_narrative = ""
        for update in progress_messages:
            if update['type'] == 'explanation':
                progress_narrative += f"\n{update['message']}\n"
            elif update['type'] in ['start', 'complete']:
                progress_narrative += f"\n{update['message']}\n"
            else:
                progress_narrative += f"\n{update['message']}\n"
        
        response_message = f"""
## 🧠 AI Literature Mining Process

{progress_narrative}

---

## 📊 Final Results

Query: {message}

Summary: Found {material_count} material candidates from {papers_count} research papers.

Material Candidates:
"""
        
        for i, candidate in enumerate(results.get('material_candidates', [])[:10], 1):
            name = candidate.get('material_name', 'Unknown Material')
            citation = candidate.get('harvard_citation', 'Citation not available')
            composition = candidate.get('material_composition', 'Not specified')
            stability = candidate.get('thermal_stability_temp_range', 'Not specified')
            mechanism = candidate.get('stabilization_mechanism', 'Not specified')
            biocompat = candidate.get('biocompatibility_data', 'Not specified')
            delivery = candidate.get('delivery_properties', 'Not specified')
            findings = candidate.get('key_findings', 'Not specified')
            confidence = candidate.get('confidence_score', 'N/A')
            
            response_message += f"""
{i}. **{name}**

**Citation:** {citation}

**Material Details:**
- **Composition:** {composition}
- **Thermal Stability:** {stability}
- **Stabilization Mechanism:** {mechanism}
- **Biocompatibility:** {biocompat}
- **Delivery Properties:** {delivery}

**Key Research Findings:** {findings}

**Confidence Score:** {confidence}/10

---
"""
        
        if material_count > 10:
            response_message += f"\n*Showing top 10 of {material_count} candidates. Full results saved to files.*"
        
        return {
            'type': 'literature',
            'message': response_message,
            'data': results,
            'progress': progress_messages,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'Literature mining error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def handle_research_query(message: str, session_id: str) -> Dict:
    """Handle research-focused queries with enhanced model support."""
    try:
        if not chatbot:
            return {
                'type': 'error',
                'message': 'Chatbot system not available. Please check system initialization.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Process message with research mode
        result = chatbot.chat(message, session_id, mode='research')
        
        if result['success']:
            response_data = {
                'type': 'research',
                'message': result['response'],
                'model_info': {
                    'type': result['model_type'],
                    'name': result['model_name'],
                    'capabilities': chatbot.get_chemistry_capabilities()
                },
                'session_id': session_id,
                'timestamp': result['timestamp']
            }
            
            # Add chemistry-specific information if available
            if result.get('parsed_chemistry'):
                response_data['chemistry'] = result['parsed_chemistry']
            
            return response_data
        else:
            return {
                'type': 'error',
                'message': f'Research query error: {result["error"]}',
                'timestamp': result['timestamp']
            }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'An error occurred in research mode: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def handle_general_chat(message: str, session_id: str) -> Dict:
    """Handle general chat messages with multi-model support."""
    try:
        if not chatbot:
            return {
                'type': 'error',
                'message': 'Chatbot system not available. Please check system initialization.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Process message with chatbot
        result = chatbot.chat(message, session_id, mode='general')
        
        if result['success']:
            response_data = {
                'type': 'general',
                'message': result['response'],
                'model_info': {
                    'type': result['model_type'],
                    'name': result['model_name'],
                    'memory_type': result['memory_type']
                },
                'session_id': session_id,
                'timestamp': result['timestamp']
            }
            
            # Add chemistry-specific information if available
            if result.get('parsed_chemistry'):
                response_data['chemistry'] = result['parsed_chemistry']
            
            return response_data
        else:
            return {
                'type': 'error',
                'message': f'Chat error: {result["error"]}',
                'model_info': {
                    'type': result['model_type'],
                    'name': result['model_name']
                },
                'timestamp': result['timestamp']
            }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'An error occurred in general chat: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def handle_psmiles_query(message: str, session_id: str) -> Dict:
    """Handle enhanced PSMILES queries with structured workflow."""
    try:
        if not psmiles_generator and not psmiles_processor:
            return {
                'type': 'error',
                'message': 'PSMILES systems not available. Please check Ollama server and psmiles library installation.',
                'suggestion': 'Install psmiles library with: pip install "psmiles[polyBERT,mordred]@git+https://github.com/Ramprasad-Group/psmiles.git"',
                'timestamp': datetime.now().isoformat()
            }
        
        lower_message = message.lower()
        
        # Check if this is a workflow operation request
        if any(keyword in lower_message for keyword in ['dimerize', 'dimer', 'connect star']):
            return handle_dimerization_request(message, session_id)
        
        elif any(keyword in lower_message for keyword in ['copolymer', 'block', 'alternating']):
            return handle_copolymerization_request(message, session_id)
        
        elif any(keyword in lower_message for keyword in ['fingerprint', 'inchi', 'analyze']):
            return handle_analysis_request(message, session_id)
        
        # Enhanced PSMILES detection - more specific patterns
        # Look for patterns that are more likely to be actual PSMILES strings
        psmiles_patterns = [
            r'\[\*\][A-Za-z0-9\[\]\(\)\=\#\-\+\\\/]*\[\*\]',  # Pattern with [*] at both ends
            r'\[\*\][A-Za-z0-9\[\]\(\)\=\#\-\+\\\/]+',        # Pattern starting with [*]
            r'[A-Za-z0-9\[\]\(\)\=\#\-\+\\\/]*\[\*\]',        # Pattern ending with [*]
        ]
        
        potential_psmiles = []
        for pattern in psmiles_patterns:
            matches = re.findall(pattern, message)
            potential_psmiles.extend(matches)
        
        # Additional filtering - must contain key PSMILES elements
        valid_psmiles = []
        for candidate in potential_psmiles:
            # Must contain [*] or be a chemical-looking string with specific indicators
            if '[*]' in candidate:
                valid_psmiles.append(candidate)
            elif (len(candidate) > 3 and 
                  any(indicator in candidate for indicator in ['=', '#', 'C', 'N', 'O']) and
                  not any(word in candidate.lower() for word in ['poly', 'acid', 'alcohol', 'amine', 'ester'])):
                # Exclude common polymer names that might be mistaken for PSMILES
                valid_psmiles.append(candidate)
        
        # Remove duplicates and sort by length (longer likely more complete)
        valid_psmiles = sorted(list(set(valid_psmiles)), key=len, reverse=True)
        
        if any(keyword in lower_message for keyword in ['validate', 'check', 'verify']):
            # Validation request
            if valid_psmiles:
                psmiles_string = valid_psmiles[0]
                return handle_psmiles_validation(psmiles_string, message, session_id)
            else:
                return {
                    'type': 'error',
                    'message': 'Could not find a PSMILES string to validate in your message.',
                    'suggestion': 'Example: "Please validate this PSMILES: [*]CC[*]"',
                    'timestamp': datetime.now().isoformat()
                }
        
        elif any(keyword in lower_message for keyword in ['example', 'show me', 'list']):
            # Examples request
            return handle_psmiles_examples(message)
        
        elif valid_psmiles:
            # Potential PSMILES found - validate before processing
            psmiles_string = valid_psmiles[0]
            
            # Quick validation check using basic PSMILES rules
            if _is_likely_psmiles(psmiles_string):
                return process_psmiles_with_workflow(psmiles_string, session_id, message)
            else:
                # Not a valid PSMILES, treat as material description
                return handle_psmiles_generation(message, session_id)
        
        else:
            # No PSMILES detected - generate from description
            return handle_psmiles_generation(message, session_id)
            
    except Exception as e:
        return {
            'type': 'error',
            'message': f'PSMILES processing error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def _is_likely_psmiles(candidate: str) -> bool:
    """
    Quick validation to determine if a string is likely a valid PSMILES.
    This prevents material names from being processed as PSMILES strings.
    """
    # Must contain [*] symbols for connection points
    if '[*]' not in candidate:
        return False
    
    # Count [*] symbols - should typically have exactly 2
    star_count = candidate.count('[*]')
    if star_count < 1 or star_count > 4:  # Allow some flexibility
        return False
    
    # Should not contain common polymer name words
    exclude_words = ['poly', 'acid', 'alcohol', 'amine', 'ester', 'glycol', 'vinyl', 'methyl', 'ethyl']
    candidate_lower = candidate.lower()
    if any(word in candidate_lower for word in exclude_words):
        return False
    
    # Should contain chemical structure indicators
    chemical_indicators = ['C', 'N', 'O', 'S', '=', '#', '(', ')', '-']
    if not any(indicator in candidate for indicator in chemical_indicators):
        return False
    
    # Check for reasonable length (not too short, not extremely long)
    if len(candidate) < 5 or len(candidate) > 200:
        return False
    
    return True

def process_psmiles_with_workflow(psmiles_string: str, session_id: str, original_message: str) -> Dict:
    """Process PSMILES through the complete workflow with canonicalization and visualization."""
    if not psmiles_processor or not psmiles_processor.available:
        return {
            'type': 'error',
            'message': 'PSMILES processor not available. Please install the psmiles library.',
            'suggestion': 'Install with: pip install "psmiles[polyBERT,mordred]@git+https://github.com/Ramprasad-Group/psmiles.git"',
            'timestamp': datetime.now().isoformat()
        }
    
    # Process through workflow
    result = psmiles_processor.process_psmiles_workflow(psmiles_string, session_id, "initial")
    
    if not result['success']:
        return {
            'type': 'error',
            'message': f"PSMILES processing failed: {result['error']}",
            'timestamp': datetime.now().isoformat()
        }
    
    # Build comprehensive response with interactive options
    workflow_options = result['workflow_options']
    
    response_message = f"""
## 🧪 PSMILES Processing Complete!

**Your Request**: {original_message}

**Original PSMILES**: `{result['original_psmiles']}`
**Canonicalized PSMILES**: `{result['canonical_psmiles']}`

### 📊 Structure Visualization
The polymer structure has been generated and saved as an SVG image.

---

## 🎯 Choose Your Next Action:

Use the interactive buttons below to continue with your polymer analysis!
"""
    
    return {
        'type': 'psmiles_workflow',
        'message': response_message,
        'psmiles_result': result,
        'svg_content': result.get('svg_content', ''),
        'svg_filename': result.get('svg_filename', ''),
        'workflow_options': workflow_options,
        'interactive_buttons': _generate_interactive_buttons(workflow_options),
        'session_count': result['session_count'],
        'timestamp': datetime.now().isoformat()
    }

def handle_dimerization_request(message: str, session_id: str) -> Dict:
    """Handle dimerization requests."""
    if not psmiles_processor or not psmiles_processor.available:
        return {'type': 'error', 'message': 'PSMILES processor not available'}
    
    # Extract star index (0 or 1)
    star_index = 0  # default
    if 'star 1' in message.lower() or 'second star' in message.lower():
        star_index = 1
    
    # Get session PSMILES
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session. Please provide a PSMILES string first.',
            'suggestion': 'Example: "[*]CC[*]" then ask for dimerization'
        }
    
    # Use the most recent PSMILES
    psmiles_index = len(session_psmiles) - 1
    
    result = psmiles_processor.perform_dimerization(session_id, psmiles_index, star_index)
    
    if not result['success']:
        return {'type': 'error', 'message': result['error']}
    
    response_message = f"""
## 🔗 Dimerization Complete!

**Operation**: {result['operation']}
**Parent PSMILES**: `{result['parent_psmiles']}`
**Dimer PSMILES**: `{result['canonical_psmiles']}`

### 📊 Dimer Structure
The dimer structure has been generated and visualized.

**What's next?** You can:
- Perform further dimerization
- Copolymerize with another polymer
- Analyze the structure properties
- Ask for modifications
"""
    
    return {
        'type': 'psmiles_dimer',
        'message': response_message,
        'dimer_result': result,
        'svg_content': result.get('svg_content', ''),
        'timestamp': datetime.now().isoformat()
    }

def handle_copolymerization_request(message: str, session_id: str) -> Dict:
    """Handle copolymerization requests."""
    if not psmiles_processor or not psmiles_processor.available:
        return {'type': 'error', 'message': 'PSMILES processor not available'}
    
    # Extract second PSMILES and connection pattern
    psmiles_pattern = r'(\[[^\]]*\][A-Za-z0-9\[\]\(\)\=\#\*\-\+\\\/]*|\b[A-Za-z0-9\[\]\(\)\=\#\*\-\+\\\/]{5,})'
    potential_psmiles = re.findall(psmiles_pattern, message)
    
    if not potential_psmiles:
        return {
            'type': 'error',
            'message': 'Please provide the second PSMILES string for copolymerization.',
            'suggestion': 'Example: "copolymerize with [*]CC(=O)[*] using pattern [1,1]"'
        }
    
    second_psmiles = potential_psmiles[0]
    
    # Extract connection pattern
    pattern_match = re.search(r'\[(\d+),\s*(\d+)\]', message)
    if pattern_match:
        connection_pattern = [int(pattern_match.group(1)), int(pattern_match.group(2))]
    else:
        connection_pattern = [1, 1]  # default
    
    # Get session PSMILES
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session. Please provide a PSMILES string first.'
        }
    
    # Use the most recent PSMILES
    psmiles_index = len(session_psmiles) - 1
    
    result = psmiles_processor.perform_copolymerization(session_id, psmiles_index, second_psmiles, connection_pattern)
    
    if not result['success']:
        return {'type': 'error', 'message': result['error']}
    
    response_message = f"""
## 🧬 Copolymerization Complete!

**Operation**: {result['operation']}
**First PSMILES**: `{result['parent_psmiles1']}`
**Second PSMILES**: `{result['parent_psmiles2']}`
**Copolymer PSMILES**: `{result['canonical_psmiles']}`

### 📊 Copolymer Structure
The alternating copolymer structure has been generated and visualized.
"""
    
    return {
        'type': 'psmiles_copolymer',
        'message': response_message,
        'copolymer_result': result,
        'svg_content': result.get('svg_content', ''),
        'svg_filename': result.get('svg_filename', ''),
        'interactive_buttons': _generate_interactive_buttons(result['workflow_options']),
        'timestamp': datetime.now().isoformat()
    }

def handle_analysis_request(message: str, session_id: str) -> Dict:
    """Handle analysis requests (fingerprints, InChI, etc.)."""
    if not psmiles_processor or not psmiles_processor.available:
        return {'type': 'error', 'message': 'PSMILES processor not available'}
    
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session for analysis.'
        }
    
    psmiles_index = len(session_psmiles) - 1  # Use most recent
    
    if 'fingerprint' in message.lower():
        # Generate fingerprints
        fingerprint_types = ['ci', 'rdkit', 'polyBERT']
        if 'mordred' in message.lower():
            fingerprint_types.append('mordred')
        
        result = psmiles_processor.get_fingerprints(session_id, psmiles_index, fingerprint_types)
        
        if not result['success']:
            return {'type': 'error', 'message': result['error']}
        
        response_message = f"""
## 🔬 Fingerprint Analysis

**PSMILES**: `{result['psmiles']}`

### Generated Fingerprints:
"""
        
        for fp_type, fp_data in result['fingerprints'].items():
            response_message += f"\n**{fp_type.upper()} Fingerprint:**\n"
            if isinstance(fp_data, dict):
                # Mordred fingerprints (showing first 10)
                for key, value in fp_data.items():
                    response_message += f"- {key}: {value}\n"
            elif isinstance(fp_data, list):
                response_message += f"Vector length: {len(fp_data)} (showing first 10): {fp_data[:10]}\n"
            else:
                response_message += f"{fp_data}\n"
        
        return {
            'type': 'psmiles_fingerprints',
            'message': response_message,
            'fingerprint_result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    elif 'inchi' in message.lower():
        # Generate InChI information
        result = psmiles_processor.get_inchi_info(session_id, psmiles_index)
        
        if not result['success']:
            return {'type': 'error', 'message': result['error']}
        
        response_message = f"""
## 🔬 InChI Analysis

**PSMILES**: `{result['psmiles']}`

**InChI String**: `{result['inchi']}`

**InChI Key**: `{result['inchi_key']}`

The InChI (International Chemical Identifier) provides a unique text string identifier for the polymer structure.
"""
        
        return {
            'type': 'psmiles_inchi',
            'message': response_message,
            'inchi_result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    else:
        return {
            'type': 'error',
            'message': 'Analysis type not recognized. Try "fingerprints" or "inchi".'
        }

def handle_psmiles_validation(psmiles_string: str, original_message: str, session_id: str) -> Dict:
    """Handle PSMILES validation requests."""
    if not psmiles_generator:
        return {'type': 'error', 'message': 'PSMILES generator not available'}
    
    result = psmiles_generator.validate_psmiles(psmiles_string, original_message)
    
    if result['success']:
        basic_val = result['basic_validation']
        detailed_val = result['ai_validation']
        
        response_message = f"""
## 🔍 PSMILES Validation Results

**PSMILES String**: `{psmiles_string}`

### Basic Syntax Check:
- **Status**: {'✅ VALID' if basic_val['valid'] else '❌ INVALID'}
- **Length**: {basic_val['length']} characters
- **Connection Symbols**: {basic_val['connection_symbols'] if basic_val['connection_symbols'] else 'None found'}

"""
        if basic_val['errors']:
            response_message += f"**Errors Found**:\n"
            for error in basic_val['errors']:
                response_message += f"- ❌ {error}\n"
        
        if basic_val['warnings']:
            response_message += f"**Warnings**:\n"
            for warning in basic_val['warnings']:
                response_message += f"- ⚠️ {warning}\n"
        
        response_message += f"""
### AI Expert Analysis:
{detailed_val}

---
**Want to process this PSMILES?** Send it again to start the workflow!
"""
        
        return {
            'type': 'psmiles_validation',
            'message': response_message,
            'validation_result': result,
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'type': 'error',
            'message': f"Validation failed: {result.get('error', 'Unknown error')}",
            'timestamp': datetime.now().isoformat()
        }

def handle_psmiles_examples(message: str) -> Dict:
    """Handle PSMILES examples requests."""
    if not psmiles_generator:
        return {'type': 'error', 'message': 'PSMILES generator not available'}
    
    category = 'all'
    if 'basic' in message.lower():
        category = 'basic'
    elif 'aromatic' in message.lower():
        category = 'aromatic'
    elif 'complex' in message.lower():
        category = 'complex'
    
    examples = psmiles_generator.get_examples(category)
    
    response_message = f"""
## 📚 PSMILES Examples ({category.title()})

Here are some common PSMILES strings and their meanings:

"""
    for name, info in examples.items():
        response_message += f"""
**{name.replace('_', ' ').title()}**
- PSMILES: `{info['psmiles']}`
- Description: {info['description']}
- Formula: {info['formula']}

"""
    
    response_message += """
---
**Want to process any of these?** Copy a PSMILES string and send it to start the workflow!
**Need a custom polymer?** Describe the structure you want!
"""
    
    return {
        'type': 'psmiles_examples',
        'message': response_message,
        'examples': examples,
        'timestamp': datetime.now().isoformat()
    }

def handle_psmiles_generation(message: str, session_id: str) -> Dict:
    """Handle PSMILES generation from description."""
    if not psmiles_generator:
        return {'type': 'error', 'message': 'PSMILES generator not available'}
    
    result = psmiles_generator.interactive_generation(message)
    
    if result['success']:
        if 'generation' in result and 'validation' in result:
            gen_result = result['generation']
            val_result = result['validation']
            
            psmiles = gen_result['psmiles']
            explanation = gen_result['explanation']
            
            if val_result['success']:
                basic_val = val_result['basic_validation']
                
                # Auto-start the workflow with the generated PSMILES
                if psmiles_processor and psmiles_processor.available:
                    try:
                        workflow_result = psmiles_processor.process_psmiles_workflow(psmiles, session_id, "generated")
                        
                        if workflow_result['success']:
                            response_message = f"""
## 🧪 PSMILES Generated & Processed!

**Your Request**: {message}

### Generated PSMILES: `{psmiles}`

### AI Explanation:
{explanation}

### Validation Check:
- **Status**: {'✅ VALID' if basic_val['valid'] else '❌ NEEDS REVIEW'}
- **Length**: {basic_val['length']} characters
- **Connection Symbols**: {basic_val['connection_symbols'] if basic_val['connection_symbols'] else 'None (terminal connections)'}

**Original PSMILES**: `{workflow_result['original_psmiles']}`
**Canonicalized PSMILES**: `{workflow_result['canonical_psmiles']}`

### 📊 Structure Visualization
The polymer structure has been generated and saved as an SVG image.
"""
                            
                            if basic_val['errors']:
                                response_message += f"\n**Issues Found**:\n"
                                for error in basic_val['errors']:
                                    response_message += f"- ❌ {error}\n"
                            
                            if basic_val['warnings']:
                                response_message += f"\n**Notes**:\n"
                                for warning in basic_val['warnings']:
                                    response_message += f"- ⚠️ {warning}\n"
                            
                            # Return the full workflow response with interactive options
                            return {
                                'type': 'psmiles_generated_workflow',
                                'message': response_message,
                                'generated_psmiles': psmiles,
                                'generation_result': result,
                                'psmiles_result': workflow_result,
                                'svg_content': workflow_result.get('svg_content', ''),
                                'svg_filename': workflow_result.get('svg_filename', ''),
                                'workflow_options': workflow_result['workflow_options'],
                                'interactive_buttons': _generate_interactive_buttons(workflow_result['workflow_options']),
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            # Fall through to fallback response
                            pass
                            
                    except Exception as e:
                        # Fall through to the fallback response
                        pass
                
                # Fallback if workflow processor not available
                response_message = f"""
## 🧪 PSMILES Generation Results

**Your Request**: {message}

### Generated PSMILES: `{psmiles}`

### AI Explanation:
{explanation}

### Validation Check:
- **Status**: {'✅ VALID' if basic_val['valid'] else '❌ NEEDS REVIEW'}
- **Length**: {basic_val['length']} characters
- **Connection Symbols**: {basic_val['connection_symbols'] if basic_val['connection_symbols'] else 'None (terminal connections)'}

"""
                if basic_val['errors']:
                    response_message += f"**Issues Found**:\n"
                    for error in basic_val['errors']:
                        response_message += f"- ❌ {error}\n"
                
                if basic_val['warnings']:
                    response_message += f"**Notes**:\n"
                    for warning in basic_val['warnings']:
                        response_message += f"- ⚠️ {warning}\n"
                
                response_message += """
---
**PSMILES processor not available.** Install the psmiles library for advanced workflow features.
"""
                
                return {
                    'type': 'psmiles_generation',
                    'message': response_message,
                    'generated_psmiles': psmiles,
                    'generation_result': result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'type': 'error',
                    'message': f"Generated PSMILES validation failed: {val_result.get('error', 'Unknown error')}",
                    'generation_result': result,
                    'timestamp': datetime.now().isoformat()
                }
        else:
            return {
                'type': 'psmiles_generation',
                'message': f"**Generated PSMILES**: `{result.get('psmiles', 'Not found')}`\n\n{result.get('explanation', '')}",
                'generation_result': result,
                'timestamp': datetime.now().isoformat()
            }
    else:
        return {
            'type': 'error',
            'message': f"PSMILES generation failed: {result.get('error', 'Unknown error')}",
            'timestamp': datetime.now().isoformat()
        }

def _generate_interactive_buttons(workflow_options: Dict) -> Dict:
    """Generate interactive button data for the frontend."""
    return {
        'dimerization': {
            'title': workflow_options['dimerization']['title'],
            'buttons': [
                {
                    'id': 'dimer_star_0',
                    'label': 'Connect First Star [*]',
                    'action': 'dimerize',
                    'params': {'star': 0},
                    'style': 'primary'
                },
                {
                    'id': 'dimer_star_1', 
                    'label': 'Connect Second Star [*]',
                    'action': 'dimerize',
                    'params': {'star': 1},
                    'style': 'primary'
                }
            ]
        },
        'copolymerization': {
            'title': workflow_options['copolymerization']['title'],
            'buttons': [
                {
                    'id': 'copolymer_input',
                    'label': 'Enter PSMILES & Pattern',
                    'action': 'copolymer_input',
                    'style': 'secondary',
                    'input_fields': [
                        {
                            'name': 'second_psmiles',
                            'placeholder': 'Enter second PSMILES (e.g., [*]CC(=O)[*])',
                            'type': 'text'
                        },
                        {
                            'name': 'pattern',
                            'placeholder': 'Connection pattern',
                            'type': 'select',
                            'options': [
                                {'value': '[1,1]', 'label': '[1,1] - Both second stars'},
                                {'value': '[0,1]', 'label': '[0,1] - First→first, Second→second'},
                                {'value': '[1,0]', 'label': '[1,0] - First→second, Second→first'},
                                {'value': '[0,0]', 'label': '[0,0] - Both first stars'}
                            ]
                        }
                    ]
                },
                {
                    'id': 'generate_random_copolymer',
                    'label': 'Generate Random Copolymer',
                    'action': 'random_copolymer',
                    'style': 'outline',
                    'tooltip': 'Generate a random second PSMILES and copolymerize'
                }
            ]
        },
        'addition': {
            'title': workflow_options['addition']['title'],
            'buttons': [
                {
                    'id': 'add_hydroxyl',
                    'label': 'Add Hydroxyl (-OH)',
                    'action': 'addition',
                    'params': {'description': 'add hydroxyl (-OH) groups to the polymer. Found in alcohols and phenols, often involved in hydrogen bonding, affecting polymer solubility and flexibility.'},
                    'style': 'secondary',
                    'tooltip': 'Hydroxyl groups improve polymer solubility and flexibility through hydrogen bonding'
                },
                {
                    'id': 'add_carboxyl',
                    'label': 'Add Carboxyl (-COOH)',
                    'action': 'addition',
                    'params': {'description': 'add carboxyl (-COOH) groups to the polymer. Found in carboxylic acids, can form salts and esters, impacting acidity and reactivity.'},
                    'style': 'secondary',
                    'tooltip': 'Carboxyl groups can form salts and esters, impacting acidity and reactivity'
                },
                {
                    'id': 'add_amine',
                    'label': 'Add Amine (-NH2)',
                    'action': 'addition',
                    'params': {'description': 'add amine (-NH2) groups to the polymer. Found in amines, can act as nucleophiles and participate in reactions like amidation, influencing polymer chain interactions.'},
                    'style': 'secondary',
                    'tooltip': 'Amine groups act as nucleophiles and influence polymer chain interactions'
                },
                {
                    'id': 'add_amide',
                    'label': 'Add Amide (-CONH2)',
                    'action': 'addition',
                    'params': {'description': 'add amide (-CONH2) groups to the polymer. Formed by the reaction of a carboxylic acid and an amine, contributing to polymer strength and rigidity.'},
                    'style': 'secondary',
                    'tooltip': 'Amide groups contribute to polymer strength and rigidity'
                },
                {
                    'id': 'add_carbonyl',
                    'label': 'Add Carbonyl (C=O)',
                    'action': 'addition',
                    'params': {'description': 'add carbonyl (C=O) groups to the polymer. Found in aldehydes, ketones, carboxylic acids, and esters. Can be reactive and influence polymer polarity.'},
                    'style': 'secondary',
                    'tooltip': 'Carbonyl groups are reactive and influence polymer polarity'
                },
                {
                    'id': 'add_ester',
                    'label': 'Add Ester (-COOR)',
                    'action': 'addition',
                    'params': {'description': 'add ester (-COOR) groups to the polymer. Formed from a carboxylic acid and an alcohol, important in polyesters and other polymers.'},
                    'style': 'secondary',
                    'tooltip': 'Ester groups are important in polyesters and other polymers'
                },
                {
                    'id': 'add_ether',
                    'label': 'Add Ether (-ROR)',
                    'action': 'addition',
                    'params': {'description': 'add ether groups to the polymer. Characterized by an oxygen atom bonded to two carbon atoms, can impact polymer flexibility and solubility.'},
                    'style': 'secondary',
                    'tooltip': 'Ether groups impact polymer flexibility and solubility'
                },
                {
                    'id': 'add_alkene',
                    'label': 'Add Alkene (C=C)',
                    'action': 'addition',
                    'params': {'description': 'add alkene (C=C) groups to the polymer. Found in unsaturated polymers, can participate in polymerization reactions.'},
                    'style': 'secondary',
                    'tooltip': 'Alkene groups can participate in polymerization reactions'
                },
                {
                    'id': 'add_alkyne',
                    'label': 'Add Alkyne (C≡C)',
                    'action': 'addition',
                    'params': {'description': 'add alkyne (C≡C) groups to the polymer. Contains a carbon-carbon triple bond, can be used in crosslinking and polymer functionalization.'},
                    'style': 'secondary',
                    'tooltip': 'Alkyne groups can be used in crosslinking and polymer functionalization'
                },
                {
                    'id': 'add_haloalkane',
                    'label': 'Add Haloalkane (-X)',
                    'action': 'addition',
                    'params': {'description': 'add haloalkane (-X, where X is a halogen) groups to the polymer. Reactive groups that can be used for further polymer modification or crosslinking.'},
                    'style': 'secondary',
                    'tooltip': 'Haloalkane groups are reactive and useful for polymer modification'
                },
                {
                    'id': 'add_aromatic',
                    'label': 'Add Aromatic Rings',
                    'action': 'addition',
                    'params': {'description': 'add aromatic rings to the polymer backbone, providing rigidity and thermal stability'},
                    'style': 'secondary',
                    'tooltip': 'Aromatic rings provide rigidity and thermal stability'
                }
            ]
        },
        'analysis': {
            'title': workflow_options['analysis']['title'],
            'buttons': [
                {
                    'id': 'fingerprints',
                    'label': 'Generate Fingerprints',
                    'action': 'fingerprints',
                    'style': 'info',
                    'tooltip': 'Generate CI, RDKit, and polyBERT fingerprints'
                },
                {
                    'id': 'inchi_analysis',
                    'label': 'InChI Analysis',
                    'action': 'inchi',
                    'style': 'info',
                    'tooltip': 'Generate InChI string and key'
                },
                {
                    'id': 'all_analysis',
                    'label': 'Complete Analysis',
                    'action': 'complete_analysis',
                    'style': 'success',
                    'tooltip': 'Run all available analysis tools'
                }
            ]
        }
    }

@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Start a new chat session."""
    try:
        # Clear session and create new one
        old_session_id = session.get('session_id')
        session.clear()
        session['session_id'] = str(uuid.uuid4())
        
        # Clear chatbot history if available
        if chatbot:
            chatbot.clear_history(session['session_id'])
        
        # Clear PSMILES processor session data if available
        if psmiles_processor and old_session_id:
            psmiles_processor.clear_session(old_session_id)
        
        return jsonify({
            'message': 'New chat session started',
            'session_id': session['session_id'],
            'memory_type': chatbot.memory_type if chatbot else 'unknown'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/psmiles/svg/<filename>')
def serve_psmiles_svg(filename):
    """Serve PSMILES SVG files."""
    try:
        if not psmiles_processor:
            return jsonify({'error': 'PSMILES processor not available'}), 404
        
        svg_path = os.path.join(psmiles_processor.temp_dir, filename)
        
        if not os.path.exists(svg_path):
            return jsonify({'error': 'SVG file not found'}), 404
        
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        return Response(svg_content, mimetype='image/svg+xml')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/psmiles/status')
def psmiles_status():
    """Get PSMILES processor status."""
    try:
        if not psmiles_processor:
            return jsonify({
                'available': False,
                'error': 'PSMILES processor not initialized'
            })
        
        status = psmiles_processor.get_status()
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/psmiles/session-history')
def get_psmiles_session_history():
    """Get PSMILES history for current session."""
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        if not psmiles_processor:
            return jsonify({'history': []})
        
        history = psmiles_processor.get_session_psmiles(session['session_id'])
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/summary', methods=['GET'])
def get_memory_summary():
    """Get memory summary for current session."""
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        if not chatbot:
            return jsonify({'error': 'Chatbot not available'}), 500
        
        session_id = session['session_id']
        mode = request.args.get('mode', 'general')
        
        summary = chatbot.get_memory_summary(session_id, mode)
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/clear', methods=['POST'])
def clear_memory():
    """Clear memory for current session."""
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        if not chatbot:
            return jsonify({'error': 'Chatbot not available'}), 500
        
        data = request.get_json() or {}
        session_id = session['session_id']
        mode = data.get('mode')  # If None, clears all modes
        
        chatbot.clear_history(session_id, mode)
        
        return jsonify({
            'message': f'Memory cleared for {"all modes" if not mode else mode} in session {session_id}',
            'session_id': session_id,
            'mode': mode
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/config', methods=['GET'])
def get_memory_config():
    """Get current memory configuration."""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot not available'}), 500
        
        return jsonify({
            'memory_type': chatbot.memory_type,
            'memory_dir': chatbot.memory_dir,
            'available_types': ['buffer', 'summary', 'buffer_window'],
            'description': {
                'buffer': 'Keeps full conversation history',
                'summary': 'Summarizes old conversations to save tokens',
                'buffer_window': 'Keeps only the last N interactions (default: 10)'
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """Get information about available and current models."""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot system not initialized'}), 500
        
        model_info = chatbot.get_model_info()
        chemistry_capabilities = chatbot.get_chemistry_capabilities()
        
        return jsonify({
            'success': True,
            'current_model': {
                'type': model_info['model_type'],
                'name': model_info['model_name'],
                'memory_type': model_info['memory_type']
            },
            'available_models': model_info['available_models'],
            'chemistry_capabilities': chemistry_capabilities,
            'llamol_available': LLAMOL_AVAILABLE,
            'llamol_loaded_models': model_info.get('llamol_loaded_models', []),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/models/switch', methods=['POST'])
def switch_model():
    """Switch between different model types."""
    try:
        if not chatbot:
            return jsonify({'error': 'Chatbot system not initialized'}), 500
        
        data = request.get_json()
        model_type = data.get('model_type')
        model_name = data.get('model_name')
        
        if not model_type:
            return jsonify({'error': 'model_type is required'}), 400
        
        if model_type not in ['ollama', 'llamol']:
            return jsonify({'error': 'model_type must be either "ollama" or "llamol"'}), 400
        
        success = chatbot.switch_model(model_type, model_name)
        
        if success:
            new_info = chatbot.get_model_info()
            return jsonify({
                'success': True,
                'message': f'Successfully switched to {model_type} model',
                'current_model': {
                    'type': new_info['model_type'],
                    'name': new_info['model_name']
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to {model_type} model'
            }), 400
            
    except Exception as e:
        return jsonify({'error': f'Model switching error: {str(e)}'}), 500

@app.route('/api/models/llamol/load', methods=['POST'])
def load_llamol_model():
    """Load a specific LlaSMol model."""
    try:
        if not LLAMOL_AVAILABLE:
            return jsonify({'error': 'LlaSMol not available'}), 400
        
        data = request.get_json()
        model_name = data.get('model_name', 'osunlp/LlaSMol-Mistral-7B')
        device = data.get('device', 'auto')
        
        success = llamol_manager.load_model(model_name, device)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully loaded LlaSMol model: {model_name}',
                'loaded_models': llamol_manager.list_loaded_models(),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to load LlaSMol model: {model_name}'
            }), 400
            
    except Exception as e:
        return jsonify({'error': f'Model loading error: {str(e)}'}), 500

@app.route('/api/models/llamol/unload', methods=['POST'])
def unload_llamol_model():
    """Unload a specific LlaSMol model to free memory."""
    try:
        if not LLAMOL_AVAILABLE:
            return jsonify({'error': 'LlaSMol not available'}), 400
        
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400
        
        success = llamol_manager.unload_model(model_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully unloaded LlaSMol model: {model_name}',
                'loaded_models': llamol_manager.list_loaded_models(),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Model {model_name} was not loaded'
            }), 400
            
    except Exception as e:
        return jsonify({'error': f'Model unloading error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Get system status including all components and models."""
    try:
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'systems': {}
        }
        
        # Literature mining system status
        if literature_miner:
            try:
                test_result = literature_miner.test_connection()
                status_data['systems']['literature_mining'] = {
                    'status': 'active',
                    'test_result': test_result,
                    'ollama_model': literature_miner.ollama_model,
                    'ollama_host': literature_miner.ollama_host
                }
            except Exception as e:
                status_data['systems']['literature_mining'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            status_data['systems']['literature_mining'] = {'status': 'not_initialized'}
        
        # Chatbot system status with model information
        if chatbot:
            try:
                model_info = chatbot.get_model_info()
                chemistry_capabilities = chatbot.get_chemistry_capabilities()
                test_result = chatbot.test_connection()
                
                status_data['systems']['chatbot'] = {
                    'status': 'active',
                    'test_result': test_result,
                    'current_model': {
                        'type': model_info['model_type'],
                        'name': model_info['model_name'],
                        'memory_type': model_info['memory_type']
                    },
                    'available_models': model_info['available_models'],
                    'chemistry_capabilities': chemistry_capabilities,
                    'llamol_available': LLAMOL_AVAILABLE
                }
                
                if LLAMOL_AVAILABLE:
                    status_data['systems']['chatbot']['llamol_loaded_models'] = model_info.get('llamol_loaded_models', [])
                    
            except Exception as e:
                status_data['systems']['chatbot'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            status_data['systems']['chatbot'] = {'status': 'not_initialized'}
        
        # PSMILES generator status
        if psmiles_generator:
            try:
                test_result = psmiles_generator.test_connection()
                status_data['systems']['psmiles_generator'] = {
                    'status': 'active',
                    'test_result': test_result,
                    'model_type': getattr(psmiles_generator, 'model_type', 'unknown'),
                    'model_name': getattr(psmiles_generator, 'model_name', 'unknown')
                }
            except Exception as e:
                status_data['systems']['psmiles_generator'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            status_data['systems']['psmiles_generator'] = {'status': 'not_initialized'}
        
        # MCP literature mining status
        if mcp_literature_miner:
            status_data['systems']['mcp_literature_mining'] = {'status': 'active'}
        else:
            status_data['systems']['mcp_literature_mining'] = {'status': 'disabled'}
        
        # Overall system health
        active_systems = sum(1 for system in status_data['systems'].values() 
                           if system.get('status') == 'active')
        total_systems = len(status_data['systems'])
        
        status_data['overall'] = {
            'healthy': active_systems >= 2,  # At least chatbot and one other system
            'active_systems': active_systems,
            'total_systems': total_systems,
            'llamol_integration': LLAMOL_AVAILABLE
        }
        
        return jsonify(status_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Status check failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/examples')
def examples():
    """Get example prompts for different modes."""
    return jsonify({
        'literature': [
            "Find polymers for insulin stabilization at room temperature",
            "Search for hydrogels used in transdermal drug delivery",
            "What materials are used for protein preservation without refrigeration?",
            "Look for biocompatible patches for hormone delivery",
            "Find smart materials that respond to body temperature",
            "Analyze recent advances in PLGA-based insulin delivery systems",
            "Find biocompatible polymers with thermal stability above 40°C",
            "Search for nanoparticle formulations for protein preservation",
            "Investigate chitosan-based transdermal patches for drug delivery",
            "Explore smart hydrogels for temperature-controlled release"
        ],
        'research': [
            "Explain the mechanism of insulin degradation at room temperature",
            "What are the key challenges in transdermal insulin delivery?",
            "How do hydrogels protect proteins from thermal degradation?",
            "What factors affect insulin permeation through skin?",
            "Compare different approaches to protein stabilization"
        ],
        'general': [
            "What is the goal of this insulin AI project?",
            "How does the active learning framework work?",
            "What are the key milestones for this research?",
            "How can AI help in materials discovery?",
            "Explain the importance of fridge-free insulin delivery"
        ]
    })

@app.route('/api/literature-stream/<message>', methods=['GET'])
def literature_stream(message):
    """Handle literature mining with real-time progress streaming via Server-Sent Events."""
    def generate_progress():
        try:
            if not message:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Message cannot be empty'})}\n\n"
                return
            
            if not literature_miner:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Literature mining system not available'})}\n\n"
                return
            
            # Create a queue for progress updates
            progress_queue = queue.Queue()
            
            def progress_callback(msg, step_type="info"):
                """Callback to send progress updates via SSE."""
                progress_queue.put({
                    'type': 'progress',
                    'step_type': step_type,
                    'message': msg,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting literature mining...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Start mining in background thread
            results = {}
            error = None
            mining_complete = False
            
            def mining_worker():
                nonlocal results, error, mining_complete
                try:
                    # URL decode the message
                    decoded_message = urllib.parse.unquote(message)
                    
                    results = literature_miner.intelligent_mining(
                        user_request=decoded_message,
                        max_papers=15,  # Reduced from 30 to 15 for faster processing
                        recent_only=True,
                        save_results=True,
                        progress_callback=progress_callback
                    )
                except Exception as e:
                    error = str(e)
                finally:
                    mining_complete = True
                    # Signal completion
                    progress_queue.put({'type': 'complete'})
            
            # Start mining thread
            mining_thread = Thread(target=mining_worker)
            mining_thread.start()
            
            # Stream progress updates
            while not mining_complete:
                try:
                    update = progress_queue.get(timeout=1)
                    if update['type'] == 'complete':
                        break
                    yield f"data: {json.dumps(update)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue
            
            # Wait for mining to complete
            mining_thread.join()
            
            # Send final results
            if error:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Mining error: {error}', 'timestamp': datetime.now().isoformat()})}\n\n"
            else:
                # Process results
                material_count = len(results.get('material_candidates', []))
                papers_count = results.get('papers_analyzed', 0)
                
                # Format final response
                response_message = f"""## 📊 Final Results

**Query:** {urllib.parse.unquote(message)}

**Summary:** Found {material_count} material candidates from {papers_count} research papers.

**Per-Paper Material Analysis:**
"""
                
                for i, candidate in enumerate(results.get('material_candidates', [])[:15], 1):
                    name = candidate.get('material_name', 'Unknown Material')
                    citation = candidate.get('harvard_citation', 'Citation not available')
                    composition = candidate.get('material_composition', 'Not specified')
                    stability = candidate.get('thermal_stability_temp_range', 'Not specified')
                    mechanism = candidate.get('stabilization_mechanism', 'Not specified')
                    biocompat = candidate.get('biocompatibility_data', 'Not specified')
                    delivery = candidate.get('delivery_properties', 'Not specified')
                    findings = candidate.get('key_findings', 'Not specified')
                    confidence = candidate.get('confidence_score', 'N/A')
                    
                    response_message += f"""
**{i}. {name}**

**Citation:** {citation}

**Material Details:**
- **Composition:** {composition}
- **Thermal Stability:** {stability}
- **Stabilization Mechanism:** {mechanism}
- **Biocompatibility:** {biocompat}
- **Delivery Properties:** {delivery}

**Key Research Findings:** {findings}

**Confidence Score:** {confidence}/10

---
"""
                
                if material_count > 15:
                    response_message += f"\n*Showing top 15 of {material_count} materials. Full results saved to files.*"
                
                yield f"data: {json.dumps({'type': 'results', 'message': response_message, 'data': results, 'timestamp': datetime.now().isoformat()})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Streaming error: {str(e)}', 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return Response(generate_progress(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@app.route('/api/mcp-literature-stream/<message>', methods=['GET'])
def mcp_literature_stream(message):
    """Handle MCP literature mining with real-time progress streaming via Server-Sent Events."""
    def generate_mcp_progress():
        try:
            if not message:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Message cannot be empty'})}\n\n"
                return
            
            if not mcp_literature_miner:
                yield f"data: {json.dumps({'type': 'error', 'message': 'MCP literature mining system not available'})}\n\n"
                return
            
            # Create a queue for progress updates
            progress_queue = queue.Queue()
            
            def progress_callback(msg, step_type="info"):
                """Callback to send progress updates via SSE."""
                progress_queue.put({
                    'type': 'progress',
                    'step_type': step_type,
                    'message': msg,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting MCP literature mining...', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Start MCP mining in background thread
            results = {}
            error = None
            mining_complete = False
            
            def mcp_mining_worker():
                nonlocal results, error, mining_complete
                try:
                    # URL decode the message
                    decoded_message = urllib.parse.unquote(message)
                    
                    results = mcp_literature_miner.intelligent_mining_with_mcp(
                        user_request=decoded_message,
                        max_papers=20,
                        recent_only=False,
                        progress_callback=progress_callback
                    )
                except Exception as e:
                    error = str(e)
                finally:
                    mining_complete = True
                    # Signal completion
                    progress_queue.put({'type': 'complete'})
            
            # Start MCP mining thread
            mining_thread = Thread(target=mcp_mining_worker)
            mining_thread.start()
            
            # Stream progress updates
            while not mining_complete:
                try:
                    update = progress_queue.get(timeout=1)
                    if update['type'] == 'complete':
                        break
                    yield f"data: {json.dumps(update)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue
            
            # Wait for mining to complete
            mining_thread.join()
            
            # Send final results
            if error:
                yield f"data: {json.dumps({'type': 'error', 'message': f'MCP mining error: {error}', 'timestamp': datetime.now().isoformat()})}\n\n"
            else:
                # Normal results processing
                papers_analyzed = results.get('papers_analyzed', 0)
                material_candidates = results.get('material_candidates', [])
                
                # Format comprehensive results with material list
                response_message = f"""**Literature Mining Results**

**Query:** {urllib.parse.unquote(message)}

**Summary:** Analyzed {papers_analyzed} research papers and found {len(material_candidates)} material candidates.

**Per-Paper Material Analysis:**

"""
                
                # Always show material candidates first - this is the most important output
                if material_candidates:
                    for i, material in enumerate(material_candidates[:15], 1):
                        material_name = material.get('material_name', 'Material not identified')
                        citation = material.get('harvard_citation', 'Citation not available')
                        composition = material.get('material_composition', 'Not specified')
                        thermal_stability = material.get('thermal_stability_temp_range', 'Not specified')
                        biocompatibility = material.get('biocompatibility_data', 'Not specified')
                        mechanism = material.get('stabilization_mechanism', 'Not specified')
                        delivery = material.get('delivery_properties', 'Not specified')
                        findings = material.get('key_findings', 'Not specified')
                        confidence = material.get('confidence_score', 5)
                        
                        response_message += f"""**{i}. {material_name}**

**Citation:** {citation}

**Material Details:**
- **Composition:** {composition}
- **Thermal Stability:** {thermal_stability}
- **Stabilization Mechanism:** {mechanism}
- **Biocompatibility:** {biocompatibility}
- **Delivery Properties:** {delivery}

**Key Research Findings:** {findings}

**Confidence Score:** {confidence}/10

---

"""
                else:
                    response_message += "No specific material candidates extracted from the analyzed papers.\n\n"
                
                # Add source papers information
                response_message += f"""
**Source Papers Analysis:**

Based on {papers_analyzed} research papers, the analysis focused on materials suitable for insulin delivery systems. The materials identified show promise for developing fridge-free insulin delivery patches due to their:

- Biocompatibility for transdermal applications
- Thermal stability at room temperature
- Controlled release properties for sustained drug delivery
- Protein stabilization mechanisms for insulin preservation

**Research Categories Covered:**
- Synthetic polymers: PLGA, PLA, PCL-based systems
- Natural polymers: Chitosan, alginate, collagen-based formulations  
- Hybrid systems: Combinations of synthetic and natural materials
- Delivery mechanisms: Nanoparticles, hydrogels, and transdermal patches

Full analysis powered by Model Context Protocol (MCP) integration.
"""
                
                yield f"data: {json.dumps({'type': 'complete', 'message': response_message, 'data': results, 'timestamp': datetime.now().isoformat()})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'MCP streaming error: {str(e)}', 'timestamp': datetime.now().isoformat()})}\n\n"
    
    return Response(generate_mcp_progress(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@app.route('/api/psmiles/action', methods=['POST'])
def handle_psmiles_action():
    """Handle interactive PSMILES workflow actions from buttons."""
    try:
        if 'session_id' not in session:
            return jsonify({'error': 'No active session'}), 400
        
        data = request.get_json()
        action = data.get('action')
        params = data.get('params', {})
        session_id = session['session_id']
        
        if not action:
            return jsonify({'error': 'Action is required'}), 400
        
        if not psmiles_processor or not psmiles_processor.available:
            return jsonify({'error': 'PSMILES processor not available'}), 400
        
        # Handle different actions
        if action == 'dimerize':
            star_index = params.get('star', 0)
            return jsonify(_handle_button_dimerization(session_id, star_index))
        
        elif action == 'copolymer_input':
            second_psmiles = params.get('second_psmiles', '').strip()
            pattern_str = params.get('pattern', '[1,1]')
            
            if not second_psmiles:
                return jsonify({'error': 'Second PSMILES string is required'}), 400
            
            # Parse pattern
            try:
                pattern = eval(pattern_str)  # e.g., "[1,1]" -> [1,1]
                if not isinstance(pattern, list) or len(pattern) != 2:
                    raise ValueError("Invalid pattern format")
            except:
                pattern = [1, 1]  # default
            
            return jsonify(_handle_button_copolymerization(session_id, second_psmiles, pattern))
        
        elif action == 'random_copolymer':
            # Generate a random second PSMILES
            random_psmiles = _generate_random_psmiles()
            pattern = [1, 1]  # default pattern
            return jsonify(_handle_button_copolymerization(session_id, random_psmiles, pattern, is_random=True))
        
        elif action == 'addition':
            description = params.get('description', '')
            return jsonify(_handle_button_addition(session_id, description))
        
        elif action == 'fingerprints':
            return jsonify(_handle_button_analysis(session_id, 'fingerprints'))
        
        elif action == 'inchi':
            return jsonify(_handle_button_analysis(session_id, 'inchi'))
        
        elif action == 'complete_analysis':
            return jsonify(_handle_button_analysis(session_id, 'complete'))
        
        else:
            return jsonify({'error': f'Unknown action: {action}'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Action handling error: {str(e)}'}), 500

def _handle_button_dimerization(session_id: str, star_index: int) -> Dict:
    """Handle dimerization button action."""
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session.'
        }
    
    psmiles_index = len(session_psmiles) - 1
    result = psmiles_processor.perform_dimerization(session_id, psmiles_index, star_index)
    
    if not result['success']:
        return {'type': 'error', 'message': result['error']}
    
    response_message = f"""
## 🔗 Dimerization Complete!

**Operation**: {result['operation']}
**Parent PSMILES**: `{result['parent_psmiles']}`
**Dimer PSMILES**: `{result['canonical_psmiles']}`

### 📊 Dimer Structure
The dimer structure has been generated and visualized.
"""
    
    return {
        'type': 'psmiles_dimer',
        'message': response_message,
        'dimer_result': result,
        'svg_content': result.get('svg_content', ''),
        'svg_filename': result.get('svg_filename', ''),
        'interactive_buttons': _generate_interactive_buttons(result['workflow_options']),
        'timestamp': datetime.now().isoformat()
    }

def _handle_button_copolymerization(session_id: str, second_psmiles: str, pattern: List[int], is_random: bool = False) -> Dict:
    """Handle copolymerization button action."""
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session.'
        }
    
    psmiles_index = len(session_psmiles) - 1
    result = psmiles_processor.perform_copolymerization(session_id, psmiles_index, second_psmiles, pattern)
    
    if not result['success']:
        return {'type': 'error', 'message': result['error']}
    
    random_note = " (randomly generated)" if is_random else ""
    
    response_message = f"""
## 🧬 Copolymerization Complete!

**Operation**: {result['operation']}
**First PSMILES**: `{result['parent_psmiles1']}`
**Second PSMILES**: `{result['parent_psmiles2']}`{random_note}
**Copolymer PSMILES**: `{result['canonical_psmiles']}`

### 📊 Copolymer Structure
The alternating copolymer structure has been generated and visualized.
"""
    
    return {
        'type': 'psmiles_copolymer',
        'message': response_message,
        'copolymer_result': result,
        'svg_content': result.get('svg_content', ''),
        'svg_filename': result.get('svg_filename', ''),
        'interactive_buttons': _generate_interactive_buttons(result['workflow_options']),
        'timestamp': datetime.now().isoformat()
    }

def _handle_button_addition(session_id: str, description: str) -> Dict:
    """Handle addition/modification button action using simplified copolymerization approach."""
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session.'
        }
    
    # Get the most recent PSMILES
    base_psmiles = session_psmiles[-1]['original']
    
    # Validate the base PSMILES structure - only check for exactly 2 [*] symbols
    star_count = base_psmiles.count('[*]')
    if star_count != 2:
        return {
            'type': 'error',
            'message': f'Invalid base PSMILES structure: {base_psmiles}. Found {star_count} [*] symbols, but exactly 2 are required.'
        }
    
    # Define functional group fragments as PSMILES units
    functional_group_fragments = {
        'hydroxyl': '[*]C(O)[*]',  # Carbon with -OH attached
        'carboxyl': '[*]C(=O)O[*]',  # Carboxyl group
        'amine': '[*]C(N)[*]',  # Carbon with -NH2 attached
        'amide': '[*]C(=O)N[*]',  # Amide group
        'carbonyl': '[*]C(=O)[*]',  # Carbonyl group
        'ester': '[*]C(=O)OC[*]',  # Ester linkage
        'ether': '[*]COC[*]',  # Ether linkage
        'alkene': '[*]C=C[*]',  # Alkene unit
        'alkyne': '[*]C#C[*]',  # Alkyne unit
        'haloalkane': '[*]C(Cl)[*]',  # Carbon with halogen (using Cl as example)
        'aromatic': None  # Special handling for aromatic rings
    }
    
    # Special handling for aromatic rings - randomly choose approach
    if any(aromatic_keyword in description.lower() for aromatic_keyword in ['aromatic', 'benzene', 'phenyl', 'ring', 'phenylene']):
        # Fixed aromatic options with valid SMILES chemistry
        aromatic_options = [
            '[*]c1ccc([*])cc1',    # Para-phenylene (1,4-substituted benzene)
            '[*]c1cccc([*])c1',    # Meta-phenylene (1,3-substituted benzene) - FIXED
            '[*]Cc1ccccc1C[*]',    # Benzyl groups attached to carbons - FIXED
            '[*]C([*])c1ccccc1'    # Phenyl attached to carbon backbone
        ]
        functional_group_psmiles = random.choice(aromatic_options)
        fragment_description = f"aromatic ring unit ({functional_group_psmiles})"
    else:
        # Find matching functional group - improved matching logic
        functional_group_psmiles = None
        fragment_description = None
        
        # Create a more comprehensive matching dictionary
        matching_keywords = {
            'hydroxyl': ['hydroxyl', 'hydroxy', '-oh', 'alcohol'],
            'carboxyl': ['carboxyl', 'carboxy', '-cooh', 'carboxylic', 'acid'],
            'amine': ['amine', 'amino', '-nh2', 'primary amine'],
            'amide': ['amide', '-conh2', 'amido'],
            'carbonyl': ['carbonyl', 'ketone', 'c=o', 'oxo'],
            'ester': ['ester', '-coor', 'ester linkage'],
            'ether': ['ether', '-ror', 'ether linkage', 'oxygen bridge', 'oxy'],
            'alkene': ['alkene', 'double bond', 'c=c', 'unsaturated'],
            'alkyne': ['alkyne', 'triple bond', 'c≡c', 'acetylene'],
            'haloalkane': ['haloalkane', 'halogen', 'chloro', 'bromo', 'fluoro', 'iodo', '-x']
        }
        
        # Check for matches using the comprehensive keyword list
        description_lower = description.lower()
        for group_name, keywords in matching_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                functional_group_psmiles = functional_group_fragments[group_name]
                fragment_description = f"{group_name.replace('_', ' ')} unit ({functional_group_psmiles})"
                break
        
        if not functional_group_psmiles:
            return {
                'type': 'error',
                'message': f'Could not identify functional group from description: "{description}". Available groups: hydroxyl, carboxyl, amine, amide, carbonyl, ester, ether, alkene, alkyne, haloalkane, aromatic. Description received: "{description_lower}"'
            }
    
    # Randomly choose connection pattern
    connection_patterns = [
        [0, 1],  # First->first, Second->second
        [1, 0],  # First->second, Second->first
        [0, 0],  # Connect both first stars
        [1, 1]   # Connect both second stars
    ]
    chosen_pattern = random.choice(connection_patterns)
    
    # Use existing copolymerization function
    psmiles_index = len(session_psmiles) - 1  # Index of most recent PSMILES
    
    result = psmiles_processor.perform_copolymerization(
        session_id=session_id,
        psmiles1_index=psmiles_index,
        psmiles2_string=functional_group_psmiles,
        connection_pattern=chosen_pattern
    )
    
    if result['success']:
        response_message = f"""
## ➕ Addition Complete!

**Original PSMILES**: `{base_psmiles}`
**Your Request**: {description}
**Added Fragment**: `{functional_group_psmiles}` ({fragment_description})
**Connection Pattern**: {chosen_pattern}
**Result PSMILES**: `{result['canonical_psmiles']}`

### 📊 Modified Structure
The functional group has been successfully added through copolymerization. The structure now contains alternating units of your original polymer and the new functional group.

**Note**: This creates an alternating copolymer pattern rather than a simple substitution, which is often more realistic for polymer chemistry.
"""
        
        return {
            'type': 'psmiles_addition',
            'message': response_message,
            'modification_result': result,
            'svg_content': result.get('svg_content', ''),
            'svg_filename': result.get('svg_filename', ''),
            'interactive_buttons': _generate_interactive_buttons(result['workflow_options']),
            'addition_details': {
                'original_psmiles': base_psmiles,
                'functional_group': functional_group_psmiles,
                'connection_pattern': chosen_pattern,
                'fragment_description': fragment_description
            },
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'type': 'error', 
            'message': f"Addition failed: {result.get('error', 'Unknown error during copolymerization')}"
        }

def _handle_button_analysis(session_id: str, analysis_type: str) -> Dict:
    """Handle analysis button actions."""
    session_psmiles = psmiles_processor.get_session_psmiles(session_id)
    if not session_psmiles:
        return {
            'type': 'error',
            'message': 'No PSMILES found in your session for analysis.'
        }
    
    psmiles_index = len(session_psmiles) - 1  # Use most recent
    
    if analysis_type == 'fingerprints':
        fingerprint_types = ['ci', 'rdkit', 'polyBERT']
        result = psmiles_processor.get_fingerprints(session_id, psmiles_index, fingerprint_types)
        
        if not result['success']:
            return {'type': 'error', 'message': result['error']}
        
        response_message = f"""
## 🔬 Fingerprint Analysis

**PSMILES**: `{result['psmiles']}`

### Generated Fingerprints:
"""
        
        for fp_type, fp_data in result['fingerprints'].items():
            response_message += f"\n**{fp_type.upper()} Fingerprint:**\n"
            if isinstance(fp_data, dict):
                # Mordred fingerprints (showing first 10)
                for key, value in fp_data.items():
                    response_message += f"- {key}: {value}\n"
            elif isinstance(fp_data, list):
                response_message += f"Vector length: {len(fp_data)} (showing first 10): {fp_data[:10]}\n"
            else:
                response_message += f"{fp_data}\n"
        
        return {
            'type': 'psmiles_fingerprints',
            'message': response_message,
            'fingerprint_result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    elif analysis_type == 'inchi':
        result = psmiles_processor.get_inchi_info(session_id, psmiles_index)
        
        if not result['success']:
            return {'type': 'error', 'message': result['error']}
        
        response_message = f"""
## 🔬 InChI Analysis

**PSMILES**: `{result['psmiles']}`

**InChI String**: `{result['inchi']}`

**InChI Key**: `{result['inchi_key']}`

The InChI (International Chemical Identifier) provides a unique text string identifier for the polymer structure.
"""
        
        return {
            'type': 'psmiles_inchi',
            'message': response_message,
            'inchi_result': result,
            'timestamp': datetime.now().isoformat()
        }
    
    elif analysis_type == 'complete':
        # Run both fingerprints and InChI analysis
        fingerprint_result = psmiles_processor.get_fingerprints(session_id, psmiles_index, ['ci', 'rdkit', 'polyBERT'])
        inchi_result = psmiles_processor.get_inchi_info(session_id, psmiles_index)
        
        response_message = f"""
## 🔬 Complete Analysis

**PSMILES**: `{session_psmiles[psmiles_index]['original']}`

### Fingerprint Analysis:
"""
        
        if fingerprint_result['success']:
            for fp_type, fp_data in fingerprint_result['fingerprints'].items():
                response_message += f"\n**{fp_type.upper()}**: "
                if isinstance(fp_data, list):
                    response_message += f"Vector length {len(fp_data)}\n"
                else:
                    response_message += f"{str(fp_data)[:100]}...\n"
        else:
            response_message += f"Fingerprint generation failed: {fingerprint_result['error']}\n"
        
        response_message += "\n### InChI Analysis:\n"
        if inchi_result['success']:
            response_message += f"**InChI**: `{inchi_result['inchi']}`\n"
            response_message += f"**InChI Key**: `{inchi_result['inchi_key']}`\n"
        else:
            response_message += f"InChI generation failed: {inchi_result['error']}\n"
        
        return {
            'type': 'psmiles_complete_analysis',
            'message': response_message,
            'fingerprint_result': fingerprint_result,
            'inchi_result': inchi_result,
            'timestamp': datetime.now().isoformat()
        }
    
    else:
        return {'type': 'error', 'message': f'Unknown analysis type: {analysis_type}'}

def _generate_random_psmiles() -> str:
    """Generate a random PSMILES string for copolymerization."""
    import random
    
    # List of common polymer building blocks
    random_blocks = [
        "[*]CC(=O)[*]",           # Polyethylene oxide-like
        "[*]C(=O)O[*]",           # Polyester-like  
        "[*]NC(=O)[*]",           # Polyamide-like
        "[*]c1ccc(cc1)[*]",       # Aromatic
        "[*]CC(C)[*]",            # Branched alkyl
        "[*]C(=O)N[*]",           # Amide
        "[*]OC(=O)[*]",           # Ester
        "[*]CCOC(=O)[*]",         # Longer ester
        "[*]NC(C)C(=O)[*]",       # Amino acid-like
        "[*]C(=O)CCC(=O)[*]"      # Diketone-like
    ]
    
    return random.choice(random_blocks)

if __name__ == '__main__':
    print("🚀 Starting Insulin AI Web Application...")
    
    # Initialize systems
    if initialize_systems():
        print("🌟 All systems ready!")
        
        # Get port from environment variable (HF Spaces uses 7860)
        import os
        port = int(os.environ.get('PORT', 7860))
        
        print(f"🌐 Starting server on 0.0.0.0:{port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to initialize systems. Please check:")
        print("   1. Ollama is running (ollama serve)")
        print("   2. Required model is available (ollama pull llama3.2)")
        print("   3. Environment variables are set correctly")
