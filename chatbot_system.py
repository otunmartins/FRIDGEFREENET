#!/usr/bin/env python3
"""
Insulin AI Chatbot System
A LangChain-based conversational AI for insulin delivery patch research.

This system provides specialized conversation modes:
- General: Project overview and general questions
- Research: Technical research assistance
- Literature: Integration with literature mining

Based on proven patterns from the Medium article on building local AI chatbots.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough


class InsulinAIChatbot:
    """
    Conversational AI system for insulin delivery patch research.
    Uses proven LangChain patterns for reliable chat functionality.
    """
    
    def __init__(self, 
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the Insulin AI Chatbot.
        
        Args:
            ollama_model (str): Ollama model name
            ollama_host (str): Ollama server host
        """
        self.model_name = ollama_model
        self.host = ollama_host
        self.chat_histories = {}  # session_id -> list of messages
        
        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=ollama_model, base_url=ollama_host)
            print(f"✅ Chatbot initialized with {ollama_model}")
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            raise
        
        # Setup prompts for different modes
        self.prompts = self._setup_prompts()
    
    def _setup_prompts(self) -> Dict:
        """Setup prompt templates for different conversation modes using proven patterns."""
        
        # General conversation prompt
        general_system_prompt = """You are an AI assistant for the "AI-Driven Design of Fridge-Free Insulin Delivery Patches" research project. 
Your role is to help users understand the project, answer general questions, and provide guidance on the research framework.

PROJECT OVERVIEW:
You are part of an innovative computational framework for discovering and optimizing novel materials for fridge-free insulin delivery patches. The system leverages:

1. **Large Language Models (LLMs)** for scientific literature mining
2. **Deep generative models** for material design (GNNs, diffusion models)
3. **Molecular dynamics (MD) simulations** with Universal Model for Atoms (UMA) force field
4. **Active learning approach** that iteratively refines material candidates

KEY OBJECTIVES:
- Discover materials that maintain insulin stability without refrigeration
- Evaluate thermal stability, insulin encapsulation efficiency, controlled release kinetics, and biocompatibility
- Accelerate material discovery through AI-driven approaches

CONVERSATION STYLE:
- Be helpful, informative, and enthusiastic about the research
- Use clear, accessible language while maintaining scientific accuracy
- Provide specific details when asked, but keep initial responses concise
- Ask clarifying questions when needed
- Direct users to appropriate tools (literature mining, research mode) when relevant

Previous conversations:
{history}"""

        general_prompt = ChatPromptTemplate.from_messages([
            ("system", general_system_prompt),
            ("human", "{input}"),
        ])
        
        # Research-focused prompt
        research_system_prompt = """You are a specialized research assistant for the "AI-Driven Design of Fridge-Free Insulin Delivery Patches" project.
You provide expert-level technical guidance on materials science, insulin delivery, and computational research methods.

EXPERTISE AREAS:
1. **Insulin Biochemistry & Stability**
   - Protein structure and degradation mechanisms
   - Thermal stability factors
   - Formulation strategies for stabilization

2. **Materials Science**
   - Polymers, hydrogels, and nanomaterials for drug delivery
   - Biocompatibility and safety considerations
   - Transdermal delivery mechanisms

3. **Computational Methods**
   - Molecular dynamics simulations
   - Machine learning for materials discovery
   - Active learning frameworks
   - Force field applications (especially UMA)

4. **Research Methodology**
   - Literature mining strategies
   - Experimental design for material testing
   - Property evaluation protocols

RESPONSE STYLE:
- Provide technically accurate, detailed explanations
- Include relevant scientific context and mechanisms
- Reference appropriate research methodologies
- Suggest experimental approaches when relevant
- Ask probing questions to clarify research objectives
- Connect concepts across disciplines (materials science, biochemistry, computational methods)

Previous conversations:
{history}"""

        research_prompt = ChatPromptTemplate.from_messages([
            ("system", research_system_prompt),
            ("human", "{input}"),
        ])
        
        # Literature discussion prompt
        literature_system_prompt = """You are a literature analysis specialist for the "AI-Driven Design of Fridge-Free Insulin Delivery Patches" project.
Your role is to help interpret literature mining results, suggest search strategies, and provide context for discovered materials.

CAPABILITIES:
1. **Literature Interpretation**
   - Analyze material candidates from research papers
   - Explain reported properties and their relevance
   - Identify knowledge gaps and research opportunities

2. **Search Strategy Development**
   - Suggest effective search terms and queries
   - Recommend focus areas for literature mining
   - Help refine search strategies based on results

3. **Material Evaluation**
   - Compare materials across different studies
   - Assess reported stability data
   - Evaluate biocompatibility and safety profiles

4. **Research Contextualization**
   - Place findings in broader research context
   - Identify emerging trends in the field
   - Suggest follow-up investigations

RESPONSE APPROACH:
- Help users understand literature mining results
- Suggest improvements to search strategies
- Provide context for material properties and applications
- Identify promising research directions
- Ask clarifying questions about specific interests or requirements
- Connect findings to project objectives

Previous conversations:
{history}"""

        literature_prompt = ChatPromptTemplate.from_messages([
            ("system", literature_system_prompt),
            ("human", "{input}"),
        ])
        
        return {
            'general': general_prompt,
            'research': research_prompt,
            'literature': literature_prompt
        }
    
    def chat(self, 
             message: str, 
             session_id: str, 
             mode: str = 'general') -> Dict:
        """
        Send a message to the chatbot and get a response using proven LangChain patterns.
        
        Args:
            message (str): User message
            session_id (str): Session identifier for conversation history
            mode (str): Conversation mode ('general', 'research', 'literature')
            
        Returns:
            Dict: Response with message and metadata
        """
        try:
            if mode not in self.prompts:
                raise ValueError(f"Unknown mode: {mode}. Available: {list(self.prompts.keys())}")
            
            # Get conversation history for this session
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = []
            
            history = self.chat_histories[session_id]
            
            # Format history as string (similar to Medium article pattern)
            history_str = ""
            for msg in history[-10:]:  # Keep last 10 messages for context
                if msg['role'] == 'user':
                    history_str += f"User: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    history_str += f"Assistant: {msg['content']}\n"
            
            # Get the appropriate prompt and apply context using .partial()
            prompt_template = self.prompts[mode]
            qa_prompt_local = prompt_template.partial(history=history_str)
            
            # Create chain using proven pattern from Medium article
            llm_chain = {"input": RunnablePassthrough()} | qa_prompt_local | self.llm
            
            # Invoke the chain
            response = llm_chain.invoke(message)
            
            # Add to conversation history
            self.chat_histories[session_id].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            self.chat_histories[session_id].append({
                'role': 'assistant', 
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'message': response.strip(),
                'mode': mode,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            
            # Still add to history for debugging
            if session_id in self.chat_histories:
                self.chat_histories[session_id].append({
                    'role': 'user',
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                })
                self.chat_histories[session_id].append({
                    'role': 'assistant',
                    'content': error_message,
                    'timestamp': datetime.now().isoformat()
                })
            
            return {
                'message': error_message,
                'mode': mode,
                'session_id': session_id,
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.chat_histories:
            del self.chat_histories[session_id]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        if session_id not in self.chat_histories:
            return []
        
        return self.chat_histories[session_id]
    
    def test_connection(self) -> str:
        """Test the connection to Ollama."""
        try:
            response = self.llm.invoke("Hello, please respond with 'Connection successful'")
            return "Connected successfully"
        except Exception as e:
            return f"Connection failed: {str(e)}"


def test_chatbot():
    """Test function for the chatbot system using Medium article patterns."""
    print("🧪 Testing Insulin AI Chatbot with proven LangChain patterns...")
    
    try:
        # Initialize chatbot
        chatbot = InsulinAIChatbot()
        
        # Test connection first
        connection_status = chatbot.test_connection()
        print(f"🔗 Connection test: {connection_status}")
        
        if "failed" in connection_status.lower():
            print("❌ Cannot proceed with tests - Ollama connection failed")
            return
        
        # Test general mode
        print("\n📝 Testing General Mode...")
        response = chatbot.chat(
            message="What is this insulin AI project about?",
            session_id="test_session",
            mode="general"
        )
        print(f"✅ General mode response: {response['message'][:150]}...")
        
        # Test research mode
        print("\n🔬 Testing Research Mode...")
        response = chatbot.chat(
            message="Explain the key challenges in insulin stability at room temperature",
            session_id="test_session",
            mode="research"
        )
        print(f"✅ Research mode response: {response['message'][:150]}...")
        
        # Test literature mode
        print("\n📚 Testing Literature Mode...")
        response = chatbot.chat(
            message="What search terms should I use to find polymer materials for insulin delivery?",
            session_id="test_session",
            mode="literature"
        )
        print(f"✅ Literature mode response: {response['message'][:150]}...")
        
        print("\n🎉 All chatbot tests passed! Ready for web application integration.")
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print("💡 Make sure the model is available: ollama pull llama3.2")


if __name__ == "__main__":
    test_chatbot() 