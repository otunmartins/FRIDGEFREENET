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

from literature_mining_system import MaterialsLiteratureMiner
from chatbot_system import InsulinAIChatbot
from mcp_client import SimplifiedMCPLiteratureMinerSync

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'insulin-ai-secret-key-2024')
CORS(app)

# Global variables
literature_miner = None
mcp_literature_miner = None
chatbot = None

def initialize_systems():
    """Initialize the literature mining and chatbot systems."""
    global literature_miner, mcp_literature_miner, chatbot
    
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
        
        # Initialize chatbot system
        chatbot = InsulinAIChatbot(
            ollama_model=ollama_model,
            ollama_host=ollama_host
        )
        
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
    """Handle chat messages from the frontend."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        chat_type = data.get('type', 'general')  # 'general', 'literature', 'research', 'mcp'
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID for conversation history
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Route message based on type
        if chat_type == 'literature':
            response = handle_literature_query(message, session_id)
        elif chat_type == 'research':
            response = handle_research_query(message, session_id)
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
    """Handle research-focused queries using the specialized chatbot."""
    try:
        if not chatbot:
            return {
                'type': 'error',
                'message': 'Research chatbot not available. Please check Ollama connection.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Use the research chatbot
        response = chatbot.chat(
            message=message,
            session_id=session_id,
            mode='research'
        )
        
        return {
            'type': 'research',
            'message': response['message'],
            'context': response.get('context', ''),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'Research query error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

def handle_general_chat(message: str, session_id: str) -> Dict:
    """Handle general conversation."""
    try:
        if not chatbot:
            return {
                'type': 'error',
                'message': 'Chatbot not available. Please check Ollama connection.',
                'timestamp': datetime.now().isoformat()
            }
        
        # Use the general chatbot
        response = chatbot.chat(
            message=message,
            session_id=session_id,
            mode='general'
        )
        
        return {
            'type': 'general',
            'message': response['message'],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'Chat error: {str(e)}',
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
            'session_id': session['session_id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    try:
        status_info = {
            'literature_miner': literature_miner is not None,
            'chatbot': chatbot is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test Ollama connection
        if chatbot:
            try:
                test_response = chatbot.test_connection()
                status_info['ollama_connection'] = test_response
            except Exception as e:
                status_info['ollama_connection'] = f'Error: {str(e)}'
        else:
            status_info['ollama_connection'] = 'Not initialized'
        
        # Test SemanticScholar literature mining system
        if literature_miner:
            try:
                status_info['literature_mining_status'] = 'Available'
                status_info['literature_mining_features'] = [
                    'Direct Semantic Scholar integration',
                    'Intelligent search strategies',
                    'AI-enhanced material analysis',
                    'Real-time progress updates'
                ]
            except Exception as e:
                status_info['literature_mining_status'] = f'Error: {str(e)}'
        else:
            status_info['literature_mining_status'] = 'Not available'
            status_info['literature_mining_features'] = []
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("❌ Failed to initialize systems. Please check:")
        print("   1. Ollama is running (ollama serve)")
        print("   2. Required model is available (ollama pull llama3.2)")
        print("   3. Environment variables are set correctly") 