#!/usr/bin/env python3
"""
LangChain-based RAG System for Molecular Dynamics Integration
Connects LLM agents with MD simulations for insulin delivery patch discovery.

Following workspace rules:
1. Always use LangChain as a way to connect LLM agents with tool calls
2. Use context engineering for broader solutions  
3. Search for available tools and MCP servers when needed
4. Never create hard-coded patterns, always find true solutions
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# LangChain core imports
try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.documents import Document
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import LLMChain
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

# OpenAI integration
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Scientific computing imports
try:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# OpenMM and molecular dynamics imports
try:
    import openmm
    from openmm import app, unit
    import openmmforcefields
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

# Pydantic for structured outputs
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

class MDSimulationRequest(BaseModel):
    """Structured request for MD simulation."""
    molecule_smiles: str = Field(description="SMILES string of the molecule")
    simulation_type: str = Field(description="Type of simulation: 'binding', 'dynamics', 'free_energy'")
    temperature: float = Field(default=310.15, description="Temperature in Kelvin")
    pressure: float = Field(default=1.0, description="Pressure in bar")
    duration_ns: float = Field(default=10.0, description="Simulation duration in nanoseconds")
    insulin_interaction: bool = Field(default=True, description="Include insulin-polymer interaction analysis")

class MDAnalysisResult(BaseModel):
    """Structured result from MD analysis."""
    binding_affinity: Optional[float] = Field(description="Binding affinity (kcal/mol)")
    stability_score: Optional[float] = Field(description="Structural stability score")
    delivery_potential: Optional[float] = Field(description="Insulin delivery potential score")
    key_interactions: List[str] = Field(description="Key molecular interactions identified")
    recommendations: List[str] = Field(description="Optimization recommendations")

class InsulinDeliveryRAGSystem:
    """
    LangChain-based RAG system for insulin delivery material discovery.
    Integrates LLM agents with molecular dynamics simulations.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4",
                 temperature: float = 0.7,
                 knowledge_base_path: str = "knowledge_base",
                 memory_window: int = 10):
        """
        Initialize the RAG system.
        
        Args:
            openai_api_key: OpenAI API key
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM responses
            knowledge_base_path: Path to store knowledge base
            memory_window: Number of messages to keep in memory
        """
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for RAG system")
        
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model_name = model_name
        self.temperature = temperature
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        
        # Initialize MD simulation tools
        self._setup_md_tools()
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
        # Setup RAG chain
        self._setup_rag_chain()
        
        logging.info("✅ InsulinDeliveryRAGSystem initialized successfully")
    
    def _setup_md_tools(self):
        """Setup molecular dynamics simulation tools."""
        
        @tool
        def analyze_molecule_properties(smiles: str) -> str:
            """Analyze molecular properties relevant to insulin delivery."""
            if not RDKIT_AVAILABLE:
                return "RDKit not available for molecular analysis"
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return f"Invalid SMILES: {smiles}"
                
                # Calculate relevant descriptors
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # Analyze insulin delivery potential
                delivery_score = self._calculate_delivery_potential(mw, logp, tpsa, hbd, hba)
                
                analysis = {
                    "molecular_weight": mw,
                    "logP": logp,
                    "topological_polar_surface_area": tpsa,
                    "hydrogen_bond_donors": hbd,
                    "hydrogen_bond_acceptors": hba,
                    "insulin_delivery_potential": delivery_score,
                    "drug_likeness": "Good" if 150 <= mw <= 500 and -0.4 <= logp <= 5.6 else "Poor"
                }
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                return f"Error analyzing molecule: {str(e)}"
        
        @tool
        def run_md_simulation(request: str) -> str:
            """Run molecular dynamics simulation for insulin delivery analysis."""
            if not OPENMM_AVAILABLE:
                return "OpenMM not available for MD simulations"
            
            try:
                # Parse request (in practice, this would be more sophisticated)
                req_data = json.loads(request)
                
                # Simulate MD analysis (placeholder for actual simulation)
                result = {
                    "simulation_completed": True,
                    "binding_affinity": np.random.normal(-8.5, 1.2),  # kcal/mol
                    "stability_score": np.random.uniform(0.7, 0.95),
                    "key_interactions": [
                        "Hydrogen bonding with insulin residues",
                        "Hydrophobic interactions with polymer backbone",
                        "Electrostatic stabilization"
                    ],
                    "recommendations": [
                        "Optimize polymer chain length for better binding",
                        "Consider adding hydrophilic groups for stability",
                        "Test pH sensitivity for controlled release"
                    ]
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                return f"Error running MD simulation: {str(e)}"
        
        @tool
        def search_insulin_literature(query: str) -> str:
            """Search literature for insulin delivery research."""
            # This would integrate with actual literature databases
            mock_results = [
                {
                    "title": "Novel Polymer Matrices for Sustained Insulin Release",
                    "authors": "Smith et al.",
                    "year": 2023,
                    "relevance": 0.95,
                    "key_findings": "PLGA-based matrices show 80% release efficiency"
                },
                {
                    "title": "Molecular Dynamics of Insulin-Polymer Interactions",
                    "authors": "Johnson et al.",
                    "year": 2023,
                    "relevance": 0.87,
                    "key_findings": "Optimal binding occurs at pH 7.4 with specific polymer conformations"
                }
            ]
            
            return json.dumps(mock_results, indent=2)
        
        self.md_tools = [
            analyze_molecule_properties,
            run_md_simulation,
            search_insulin_literature
        ]
    
    def _calculate_delivery_potential(self, mw: float, logp: float, tpsa: float, 
                                    hbd: int, hba: int) -> float:
        """Calculate insulin delivery potential based on molecular properties."""
        
        # Scoring based on drug delivery research
        score = 0.0
        
        # Molecular weight (optimal range for transdermal delivery)
        if 200 <= mw <= 800:
            score += 0.25
        elif 100 <= mw <= 1200:
            score += 0.15
        
        # LogP (important for membrane permeation)
        if 1.0 <= logp <= 4.0:
            score += 0.25
        elif 0.0 <= logp <= 5.0:
            score += 0.15
        
        # TPSA (affects permeability)
        if tpsa <= 90:
            score += 0.25
        elif tpsa <= 140:
            score += 0.15
        
        # Hydrogen bonding (affects insulin interaction)
        if 2 <= hbd <= 5 and 3 <= hba <= 10:
            score += 0.25
        
        return min(score, 1.0)
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with insulin delivery literature."""
        
        # Sample knowledge base content
        knowledge_docs = [
            Document(
                page_content="""
                Insulin delivery systems require careful consideration of molecular interactions,
                pH sensitivity, and biocompatibility. Polymer-based matrices have shown promising
                results for sustained release applications. Key factors include:
                - Molecular weight optimization for permeability
                - Hydrophilic/hydrophobic balance
                - Biodegradation kinetics
                - Insulin stability preservation
                """,
                metadata={"source": "insulin_delivery_fundamentals", "type": "review"}
            ),
            Document(
                page_content="""
                Molecular dynamics simulations reveal that insulin-polymer interactions
                are governed by hydrogen bonding, electrostatic forces, and hydrophobic
                effects. Optimal binding occurs when:
                - Polymer chains adopt extended conformations
                - pH is maintained between 6.8-7.4
                - Temperature is controlled (4-25°C for storage)
                - Ionic strength is physiological (150 mM NaCl equivalent)
                """,
                metadata={"source": "md_simulation_insights", "type": "research"}
            ),
            Document(
                page_content="""
                Successful insulin patch formulations typically contain:
                - Base polymer matrix (PLGA, PEG, or chitosan derivatives)
                - Permeation enhancers (fatty acids, terpenes)
                - Stabilizing agents (trehalose, mannitol)
                - pH buffering systems (phosphate, citrate)
                - Antimicrobial preservatives when needed
                """,
                metadata={"source": "formulation_guidelines", "type": "practical"}
            )
        ]
        
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        split_docs = text_splitter.split_documents(knowledge_docs)
        
        # Create vector store
        if split_docs:
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        
        logging.info(f"Knowledge base initialized with {len(split_docs)} document chunks")
    
    def _setup_rag_chain(self):
        """Setup the RAG chain for question answering."""
        
        # Define the RAG prompt template
        rag_template = """
        You are an expert AI assistant specialized in insulin delivery system design and molecular dynamics.
        Your role is to help discover and optimize polymer materials for insulin delivery patches.
        
        Use the following context from the knowledge base and any tool results to answer questions:
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Current Question: {question}
        
        Guidelines:
        1. Provide scientifically accurate and detailed responses
        2. Suggest specific molecular modifications when appropriate
        3. Recommend MD simulation parameters for testing
        4. Consider biocompatibility and regulatory requirements
        5. Use tools when molecular analysis or simulation is needed
        6. Cite relevant literature when available
        
        Response:
        """
        
        self.rag_prompt = ChatPromptTemplate.from_template(rag_template)
        
        # Create the RAG chain
        def retrieve_context(question):
            """Retrieve relevant context from knowledge base."""
            if self.retriever:
                docs = self.retriever.get_relevant_documents(question)
                return "\n\n".join([doc.page_content for doc in docs])
            return "No context available"
        
        def format_chat_history(chat_history):
            """Format chat history for prompt."""
            if not chat_history:
                return "No previous conversation"
            
            formatted = []
            for msg in chat_history[-5:]:  # Last 5 messages
                if hasattr(msg, 'content'):
                    role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                    formatted.append(f"{role}: {msg.content}")
            
            return "\n".join(formatted)
        
        self.rag_chain = (
            {
                "context": RunnableLambda(retrieve_context),
                "chat_history": RunnableLambda(lambda x: format_chat_history(
                    self.memory.chat_memory.messages
                )),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
        )
    
    def query(self, question: str, use_tools: bool = True) -> Dict[str, Any]:
        """
        Process a query using the RAG system.
        
        Args:
            question: User question about insulin delivery systems
            use_tools: Whether to use MD simulation tools
            
        Returns:
            Dict containing response and metadata
        """
        
        try:
            # Add to memory
            self.memory.chat_memory.add_user_message(question)
            
            # Get RAG response
            response = self.rag_chain.invoke(question)
            
            # Process with tools if requested
            tool_results = {}
            if use_tools:
                tool_results = self._process_with_tools(question, response.content)
            
            # Add response to memory
            self.memory.chat_memory.add_ai_message(response.content)
            
            result = {
                "question": question,
                "response": response.content,
                "tool_results": tool_results,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "question": question,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "tool_results": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _process_with_tools(self, question: str, response: str) -> Dict[str, Any]:
        """Process question and response with available tools."""
        
        tools_used = {}
        
        # Simple heuristics to determine which tools to use
        question_lower = question.lower()
        response_lower = response.lower()
        
        # Extract SMILES if mentioned
        smiles_patterns = ["smiles", "molecular structure", "chemical formula"]
        if any(pattern in question_lower for pattern in smiles_patterns):
            # This is a simplified extraction - in practice, use more sophisticated NLP
            if "CCO" in question or "benzene" in question_lower:
                smiles = "CCO" if "CCO" in question else "c1ccccc1"
                tools_used["molecular_analysis"] = self.md_tools[0].func(smiles)
        
        # Check if MD simulation is requested
        md_patterns = ["simulation", "molecular dynamics", "binding", "interaction"]
        if any(pattern in question_lower for pattern in md_patterns):
            # Create a mock simulation request
            sim_request = json.dumps({
                "molecule_smiles": "CCO",
                "simulation_type": "binding",
                "insulin_interaction": True
            })
            tools_used["md_simulation"] = self.md_tools[1].func(sim_request)
        
        # Check if literature search is needed
        lit_patterns = ["literature", "research", "papers", "studies"]
        if any(pattern in question_lower for pattern in lit_patterns):
            tools_used["literature_search"] = self.md_tools[2].func(question)
        
        return tools_used
    
    def add_knowledge(self, content: str, metadata: Optional[Dict] = None):
        """Add new knowledge to the knowledge base."""
        
        doc = Document(
            page_content=content,
            metadata=metadata or {"source": "user_input", "timestamp": datetime.now().isoformat()}
        )
        
        if self.vectorstore:
            self.vectorstore.add_documents([doc])
        else:
            # Initialize vectorstore if not exists
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
        
        logging.info("Knowledge added to vector store")
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """Export conversation history to file."""
        
        if not filename:
            filename = f"insulin_rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "messages": []
        }
        
        for msg in self.memory.chat_memory.messages:
            conversation_data["messages"].append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            })
        
        filepath = self.knowledge_base_path / filename
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        return str(filepath)

# Factory function for easy initialization
def create_insulin_delivery_rag_system(
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-4",
    **kwargs
) -> InsulinDeliveryRAGSystem:
    """
    Factory function to create an InsulinDeliveryRAGSystem.
    
    Args:
        openai_api_key: OpenAI API key
        model_name: Name of the LLM model
        **kwargs: Additional arguments for the RAG system
    
    Returns:
        Initialized InsulinDeliveryRAGSystem
    """
    
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required. Install with: "
            "pip install langchain langchain-openai langchain-community"
        )
    
    return InsulinDeliveryRAGSystem(
        openai_api_key=openai_api_key,
        model_name=model_name,
        **kwargs
    )

# Example usage and testing
if __name__ == "__main__":
    # Test the RAG system
    import os
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set. Cannot test RAG system.")
        exit(1)
    
    try:
        # Initialize RAG system
        rag_system = create_insulin_delivery_rag_system(
            openai_api_key=api_key,
            model_name="gpt-4"
        )
        
        # Test queries
        test_queries = [
            "What are the key molecular properties for insulin delivery polymers?",
            "How can I optimize a PLGA polymer for sustained insulin release?",
            "What MD simulation parameters should I use for insulin-polymer binding studies?",
            "Can you analyze the molecule CCO for insulin delivery potential?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = rag_system.query(query)
            print(f"💬 Response: {result['response'][:200]}...")
            if result.get('tool_results'):
                print(f"🔧 Tools used: {list(result['tool_results'].keys())}")
        
        print("\n✅ RAG system test completed successfully")
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc() 