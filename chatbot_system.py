#!/usr/bin/env python3
"""
Insulin AI Chatbot System
A LangChain-based conversational AI for insulin delivery patch research.

This system provides specialized conversation modes:
- General: Project overview and general questions
- Research: Technical research assistance
- Literature: Integration with literature mining

Enhanced with persistent conversation memory using LangChain memory components.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pickle

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage


class InsulinAIChatbot:
    """
    Conversational AI system for insulin delivery patch research.
    Enhanced with persistent LangChain memory for conversation context.
    """
    
    def __init__(self, 
                 model_type: str = "ollama",
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434",
                 memory_type: str = "buffer_window",
                 memory_dir: str = "chat_memory"):
        """
        Initialize the Insulin AI Chatbot with persistent memory.
        
        Args:
            model_type (str): Type of model to use (currently only 'ollama' supported)
            ollama_model (str): Ollama model name
            ollama_host (str): Ollama server host
            memory_type (str): Type of memory to use ('buffer', 'summary', 'buffer_window')
            memory_dir (str): Directory to store memory files
        """
        self.model_type = "ollama"  # Only Ollama supported now
        self.memory_type = memory_type
        self.memory_dir = memory_dir
        
        # Create memory directory if it doesn't exist
        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)
        
        # Initialize Ollama LLM
        self.model_name = ollama_model
        self.host = ollama_host
        try:
            self.llm = OllamaLLM(model=ollama_model, base_url=ollama_host)
            print(f"✅ Ollama chatbot initialized with {ollama_model}")
        except Exception as e:
            print(f"❌ Failed to initialize Ollama chatbot: {e}")
            raise
        
        # Initialize memory storage for different sessions
        self.memories = {}  # session_id -> memory object
        self.chat_histories = {}  # Keep for backward compatibility
        
        # Setup prompts for different modes
        self.prompts = self._setup_prompts()
    
    def switch_model(self, 
                    model_type: str, 
                    model_name: Optional[str] = None) -> bool:
        """
        Switch between different Ollama models.
        
        Args:
            model_type (str): Should be 'ollama' 
            model_name (str): Specific Ollama model name
            
        Returns:
            bool: Success status
        """
        try:
            if model_type.lower() != "ollama":
                print(f"❌ Only Ollama models are supported")
                return False
            
            target_model = model_name or "llama3.2"
            host = getattr(self, 'host', 'http://localhost:11434')
            
            new_llm = OllamaLLM(model=target_model, base_url=host)
            self.model_type = "ollama"
            self.model_name = target_model
            self.llm = new_llm
            print(f"✅ Switched to Ollama: {target_model}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to switch model: {e}")
            return False
    
    def _create_memory(self, session_id: str, mode: str) -> BaseMessage:
        """Create appropriate memory type for a session."""
        memory_key = f"{session_id}_{mode}"
        
        if memory_key in self.memories:
            return self.memories[memory_key]
        
        # Load existing memory from file if it exists
        memory_file = os.path.join(self.memory_dir, f"{memory_key}.pkl")
        
        if self.memory_type == "buffer":
            memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
        elif self.memory_type == "summary":
            memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="history",
                return_messages=True
            )
        elif self.memory_type == "buffer_window":
            memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 interactions
                memory_key="history",
                return_messages=True
            )
        else:
            # Default to buffer window
            memory = ConversationBufferWindowMemory(
                k=10,
                memory_key="history", 
                return_messages=True
            )
        
        # Try to load existing memory
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'rb') as f:
                    saved_memory = pickle.load(f)
                    memory.chat_memory = saved_memory.chat_memory
                print(f"📚 Loaded existing memory for {memory_key}")
            except Exception as e:
                print(f"⚠️ Could not load memory file {memory_file}: {e}")
        
        self.memories[memory_key] = memory
        return memory
    
    def _save_memory(self, session_id: str, mode: str):
        """Save memory to file for persistence."""
        memory_key = f"{session_id}_{mode}"
        if memory_key not in self.memories:
            return
        
        memory_file = os.path.join(self.memory_dir, f"{memory_key}.pkl")
        try:
            with open(memory_file, 'wb') as f:
                pickle.dump(self.memories[memory_key], f)
        except Exception as e:
            print(f"⚠️ Could not save memory to {memory_file}: {e}")
    
    def _format_history_for_prompt(self, memory) -> str:
        """Format memory history for use in prompts."""
        try:
            messages = memory.chat_memory.messages
            if not messages:
                return "No previous conversation history."
            
            formatted_history = []
            for message in messages[-20:]:  # Last 20 messages to avoid token limits
                if isinstance(message, HumanMessage):
                    formatted_history.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_history.append(f"Assistant: {message.content}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            print(f"⚠️ Error formatting history: {e}")
            return "No previous conversation history."

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
        Process a chat message with enhanced memory and multi-model support.
        
        Args:
            message (str): User's message
            session_id (str): Session identifier for memory
            mode (str): Conversation mode ('general', 'research', 'literature')
            
        Returns:
            Dict: Response with metadata including model information
        """
        try:
            # Create or get memory for this session and mode
            memory = self._create_memory(session_id, mode)
            
            # Get the appropriate prompt for the mode
            if mode not in self.prompts:
                mode = 'general'  # fallback to general mode
            
            prompt_template = self.prompts[mode]
            
            # Format history for the prompt
            history = self._format_history_for_prompt(memory)
            
            # Generate response using the selected model
            formatted_prompt = prompt_template.format(input=message, history=history)
            response_text = self.llm.invoke(formatted_prompt)
            
            # Add to memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(response_text)
            
            # Save memory to file
            self._save_memory(session_id, mode)
            
            return {
                'success': True,
                'response': response_text,
                'session_id': session_id,
                'mode': mode,
                'model_type': self.model_type,
                'model_name': self.model_name,
                'memory_type': self.memory_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'mode': mode,
                'model_type': self.model_type,
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the currently active model.
        
        Returns:
            Dict: Model information
        """
        info = {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'memory_type': self.memory_type,
            'available_models': {}
        }
        
        # Add Ollama info
        info['available_models']['ollama'] = ['llama3.2', 'llama2', 'codellama', 'mistral', 'mixtral']
        
        return info
    
    def get_chemistry_capabilities(self) -> Dict:
        """
        Get information about chemistry-specific capabilities.
        
        Returns:
            Dict: Chemistry capabilities information
        """
        return {
            'chemistry_model': False,
            'specialized_for': 'General conversation and research assistance'
        }
    
    def clear_history(self, session_id: str, mode: str = None):
        """Clear conversation history for a session."""
        if mode:
            # Clear specific mode memory
            memory_key = f"{session_id}_{mode}"
            if memory_key in self.memories:
                self.memories[memory_key].clear()
                self._save_memory(session_id, mode)
        else:
            # Clear all memories for this session
            modes = ['general', 'research', 'literature']
            for mode in modes:
                memory_key = f"{session_id}_{mode}"
                if memory_key in self.memories:
                    self.memories[memory_key].clear()
                    self._save_memory(session_id, mode)
        
        # Clear backward compatibility history
        if session_id in self.chat_histories:
            del self.chat_histories[session_id]
    
    def get_memory_summary(self, session_id: str, mode: str = 'general') -> Dict:
        """Get a summary of the conversation memory for a session."""
        memory_key = f"{session_id}_{mode}"
        
        if memory_key not in self.memories:
            return {
                'session_id': session_id,
                'mode': mode,
                'message_count': 0,
                'memory_type': self.memory_type,
                'summary': 'No conversation history found.'
            }
        
        memory = self.memories[memory_key]
        messages = memory.chat_memory.messages
        
        return {
            'session_id': session_id,
            'mode': mode,
            'message_count': len(messages),
            'memory_type': self.memory_type,
            'summary': f"Conversation contains {len(messages)} messages using {self.memory_type} memory."
        }
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session (backward compatibility)."""
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
        print(f"✅ General mode response: {response['response'][:150]}...")
        
        # Test research mode
        print("\n🔬 Testing Research Mode...")
        response = chatbot.chat(
            message="Explain the key challenges in insulin stability at room temperature",
            session_id="test_session",
            mode="research"
        )
        print(f"✅ Research mode response: {response['response'][:150]}...")
        
        # Test literature mode
        print("\n📚 Testing Literature Mode...")
        response = chatbot.chat(
            message="What search terms should I use to find polymer materials for insulin delivery?",
            session_id="test_session",
            mode="literature"
        )
        print(f"✅ Literature mode response: {response['response'][:150]}...")
        
        print("\n🎉 All chatbot tests passed! Ready for web application integration.")
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print("💡 Make sure the model is available: ollama pull llama3.2")


if __name__ == "__main__":
    test_chatbot() 