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

from literature_mining_system import MaterialsLiteratureMiner
from chatbot_system import InsulinAIChatbot
from mcp_client import SimplifiedMCPLiteratureMinerSync
from psmiles_generator import PSMILESGenerator
from llamol_integration import llamol_manager, LLAMOL_AVAILABLE

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'insulin-ai-secret-key-2024')
CORS(app)

# Global variables
literature_miner = None
mcp_literature_miner = None
chatbot = None
psmiles_generator = None

def initialize_systems():
    """Initialize the literature mining, chatbot, and PSMILES generation systems."""
    global literature_miner, mcp_literature_miner, chatbot, psmiles_generator
    
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
    """Handle PSMILES generation and validation queries."""
    try:
        if not psmiles_generator:
            return {
                'type': 'error',
                'message': 'PSMILES generation system not available. Please check Ollama server.',
                'suggestion': 'You can still use other modes like Literature Mining or Research Assistant.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Enhanced PSMILES pattern to extract PSMILES strings from text
        psmiles_pattern = r'([A-Za-z0-9\[\]\(\)\=\#\*\-\+\\\/]+)'
        
        # Determine the type of request
        lower_message = message.lower()
        
        # Check for modification requests first (more specific)
        modification_keywords = ['make', 'modify', 'change', 'extend', 'add', 'longer', 'shorter', 'complex']
        is_modification = any(keyword in lower_message for keyword in modification_keywords)
        
        # Extract PSMILES strings from the message
        potential_psmiles = re.findall(psmiles_pattern, message)
        # Filter to likely PSMILES (contain common PSMILES elements)
        psmiles_indicators = ['(', ')', '=', '[', ']', '*', 'C', 'N', 'O']
        valid_psmiles = [p for p in potential_psmiles if len(p) > 2 and any(ind in p for ind in psmiles_indicators)]
        
        if any(keyword in lower_message for keyword in ['validate', 'check', 'verify', 'correct']):
            # This seems like a validation request
            if valid_psmiles:
                # Use the longest match as the PSMILES string
                psmiles_string = max(valid_psmiles, key=len)
                result = psmiles_generator.validate_psmiles(psmiles_string, message)
                
                if result['success']:
                    basic_val = result['basic_validation']
                    detailed_val = result['detailed_validation']
                    
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
**Need help with PSMILES syntax?** Try asking for examples or generation assistance!
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
            else:
                return {
                    'type': 'error', 
                    'message': 'Could not find a PSMILES string to validate in your message. Please include the PSMILES string you want to check.',
                    'suggestion': 'Example: "Please validate this PSMILES: CC"',
                    'timestamp': datetime.now().isoformat()
                }
        
        elif any(keyword in lower_message for keyword in ['example', 'show me', 'list']):
            # This seems like a request for examples
            category = 'all'
            if 'basic' in lower_message:
                category = 'basic'
            elif 'aromatic' in lower_message:
                category = 'aromatic'
            elif 'complex' in lower_message:
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
**Want to generate a custom polymer?** Describe the structure you need!
**Need validation?** Share a PSMILES string and ask me to check it!
"""
            
            return {
                'type': 'psmiles_examples',
                'message': response_message,
                'examples': examples,
                'timestamp': datetime.now().isoformat()
            }
        
        elif is_modification and valid_psmiles:
            # This is a modification request with an existing PSMILES
            base_psmiles = max(valid_psmiles, key=len)  # Use the longest/most complex PSMILES found
            
            # Create enhanced modification prompt
            modification_prompt = f"""
I need to modify this existing PSMILES structure: {base_psmiles}

User's modification request: {message}

Please:
1. Keep the original structure "{base_psmiles}" as the base
2. Apply the requested modifications (add aromatic rings, extend length, etc.)
3. If a target length is specified, try to reach approximately that length
4. Ensure the result is chemically reasonable
5. Use proper PSMILES syntax with connection points [*] where appropriate

Generate the modified PSMILES structure now.
"""
            
            result = psmiles_generator.interactive_generation(modification_prompt)
            
            if result['success']:
                if 'generation' in result and 'validation' in result:
                    # Full interactive generation with validation
                    gen_result = result['generation']
                    val_result = result['validation']
                    
                    psmiles = gen_result['psmiles']
                    explanation = gen_result['explanation']
                    basic_val = val_result['basic_validation']
                    
                    # Check if we achieved the target length if specified
                    length_feedback = ""
                    if any(char.isdigit() for char in message):
                        # Extract numbers from the message (potential target lengths)
                        numbers = re.findall(r'\d+', message)
                        if numbers:
                            target_length = int(numbers[-1])  # Use the last number as target length
                            actual_length = len(psmiles)
                            if abs(actual_length - target_length) > 20:  # If significantly different
                                length_feedback = f"\n**Length Note**: Generated {actual_length} characters (target was ~{target_length}). "
                                if actual_length < target_length:
                                    length_feedback += "Try asking to add more complexity or repeat units for longer structures."
                                else:
                                    length_feedback += "Structure is longer than requested - this often happens with complex aromatics."
                    
                    response_message = f"""
## 🔧 PSMILES Modification Results

**Original PSMILES**: `{base_psmiles}` ({len(base_psmiles)} characters)

**Your Request**: {message}

### Modified PSMILES: `{psmiles}`

### AI Explanation:
{explanation}{length_feedback}

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
**Want to modify further?** Describe additional changes!
**Need different complexity?** Ask for simpler or more complex variations!
"""
                    
                    return {
                        'type': 'psmiles_modification',
                        'message': response_message,
                        'generation_result': result,
                        'base_psmiles': base_psmiles,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # Simple generation result
                    return {
                        'type': 'psmiles_modification',
                        'message': f"**Original**: `{base_psmiles}`\n**Modified PSMILES**: `{result.get('psmiles', 'Not found')}`\n\n{result.get('explanation', '')}",
                        'generation_result': result,
                        'base_psmiles': base_psmiles,
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'type': 'error',
                    'message': f"PSMILES modification failed: {result.get('error', 'Unknown error')}",
                    'suggestion': f"Try simplifying your request or asking for help with modifying: {base_psmiles}",
                    'timestamp': datetime.now().isoformat()
                }
                
        else:
            # This seems like a generation request (no existing PSMILES to modify)
            result = psmiles_generator.interactive_generation(message)
            
            if result['success']:
                if 'generation' in result and 'validation' in result:
                    # Full interactive generation with validation
                    gen_result = result['generation']
                    val_result = result['validation']
                    
                    psmiles = gen_result['psmiles']
                    explanation = gen_result['explanation']
                    basic_val = val_result['basic_validation']
                    
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
**Want to modify this structure?** Describe your changes!
**Need more examples?** Ask for basic, aromatic, or complex examples!
"""
                    
                    return {
                        'type': 'psmiles_generation',
                        'message': response_message,
                        'generation_result': result,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # Simple generation result
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
                
    except Exception as e:
        return {
            'type': 'error',
            'message': f'PSMILES processing error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Start a new chat session."""
    try:
        # Clear session and create new one
        session.clear()
        session['session_id'] = str(uuid.uuid4())
        
        # Clear chatbot history if available
        if chatbot:
            chatbot.clear_history(session['session_id'])
        
        return jsonify({
            'message': 'New chat session started',
            'session_id': session['session_id'],
            'memory_type': chatbot.memory_type if chatbot else 'unknown'
        })
        
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

if __name__ == '__main__':
    print("🚀 Starting Insulin AI Web Application...")
    
    # Initialize systems
    if initialize_systems():
        print("🌟 All systems ready!")
        app.run(debug=True, host='0.0.0.0', port=8003)
    else:
        print("❌ Failed to initialize systems. Please check:")
        print("   1. Ollama is running (ollama serve)")
        print("   2. Required model is available (ollama pull llama3.2)")
        print("   3. Environment variables are set correctly") 
