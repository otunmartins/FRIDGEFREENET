"""
Codebase Understanding Chatbot for Insulin AI Framework
Provides an interactive chatbot to help understand the codebase structure and functionality
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import uuid
from datetime import datetime

# LangChain imports for RAG
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.document_loaders.parsers import LanguageParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"LangChain not available: {e}")

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent.parent.parent

def scan_codebase_files(project_root: Path) -> List[Path]:
    """Scan for relevant codebase files"""
    relevant_extensions = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml'}
    exclude_dirs = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache', 'output'}
    
    files = []
    for file_path in project_root.rglob('*'):
        if (file_path.is_file() and 
            file_path.suffix in relevant_extensions and
            not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs) and
            file_path.stat().st_size < 1024 * 1024):  # Skip files larger than 1MB
            files.append(file_path)
    
    return files[:500]  # Limit to first 500 files for performance

@st.cache_resource
def initialize_codebase_chatbot():
    """Initialize the codebase understanding chatbot with RAG"""
    if not LANGCHAIN_AVAILABLE:
        return None, "LangChain dependencies not available. Please install: pip install langchain langchain-community langchain-openai"
    
    try:
        project_root = get_project_root()
        
        # Load codebase documents
        st.info("🔍 Scanning codebase files...")
        files = scan_codebase_files(project_root)
        
        documents = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():  # Only include non-empty files
                        relative_path = file_path.relative_to(project_root)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(relative_path),
                                "file_type": file_path.suffix,
                                "file_name": file_path.name,
                                "directory": str(relative_path.parent),
                                "size": len(content)
                            }
                        )
                        documents.append(doc)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if not documents:
            return None, "No documents found in codebase"
        
        st.info(f"📚 Loaded {len(documents)} files from codebase")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        st.info(f"✂️ Created {len(splits)} text chunks")
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        # Create custom prompt for codebase understanding
        custom_prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template="""You are an expert software engineer and code analyst helping developers understand the Insulin AI codebase. 
            You have access to the complete codebase including Python files, configuration files, and documentation.

Your role is to:
- Explain code structure, patterns, and architecture
- Help locate specific functionality 
- Suggest best practices and improvements
- Answer questions about implementation details
- Trace data flow and dependencies

Context from codebase:
{context}

Chat History:
{chat_history}

Current Question: {question}

Instructions:
1. Provide clear, technical explanations suitable for developers
2. Reference specific files and line numbers when relevant
3. Include code snippets to illustrate points
4. If you don't know something, say so - don't make up code that doesn't exist
5. Suggest related files or functions that might be useful
6. Be helpful for both newcomers and experienced developers

Answer:"""
        )
        
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Create conversational retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain, None
        
    except Exception as e:
        return None, f"Error initializing chatbot: {str(e)}"

def render_framework_overview():
    """Render the codebase understanding chatbot interface"""
    st.header("🧬 Insulin AI Codebase Assistant")
    st.markdown("""
    Welcome to the **Insulin AI Codebase Understanding Assistant**! This AI-powered chatbot has been trained on the 
    entire codebase and can help you understand the framework architecture, locate specific functionality, 
    and navigate the code structure.
    
    **What you can ask:**
    - "How does the PSMILES generation work?"
    - "Where is the molecular dynamics simulation implemented?"
    - "Show me the main entry points of the application"
    - "How do the UI modules work together?"
    - "What are the key classes and their relationships?"
    - "How is the literature mining integrated?"
    """)
    
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        st.error("🚫 **LangChain Required**: This chatbot requires LangChain to be installed.")
        st.code("pip install langchain langchain-community langchain-openai", language="bash")
        return
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 **OpenAI API Key Required**: Please set your OpenAI API key to use this chatbot.")
        return
    
    # Initialize chatbot
    with st.spinner("🤖 Initializing codebase assistant..."):
        qa_chain, error = initialize_codebase_chatbot()
    
    if error:
        st.error(f"❌ **Initialization Error**: {error}")
        return
    
    if qa_chain is None:
        st.error("❌ **Failed to initialize chatbot**")
        return
    
    st.success("✅ **Codebase Assistant Ready!** Ask me anything about the Insulin AI framework.")
    
    # Initialize chat history
    if "codebase_chat_history" not in st.session_state:
        st.session_state.codebase_chat_history = []
    
    if "codebase_messages" not in st.session_state:
        st.session_state.codebase_messages = [
            {"role": "assistant", "content": "Hello! I'm your Insulin AI codebase assistant. I can help you understand the framework architecture, locate specific functionality, and navigate the code. What would you like to know?"}
        ]
    
    # Display chat history
    for message in st.session_state.codebase_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📁 Source Files Referenced"):
                    for source in message["sources"]:
                        st.code(source, language="text")
    
    # Chat input
    if prompt := st.chat_input("Ask about the codebase..."):
        # Add user message
        st.session_state.codebase_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing codebase..."):
                try:
                    # Get response from chain
                    result = qa_chain({
                        "question": prompt,
                        "chat_history": st.session_state.codebase_chat_history
                    })
                    
                    response = result["answer"]
                    source_docs = result.get("source_documents", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show source files
                    if source_docs:
                        sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]
                        unique_sources = list(set(sources))
                        
                        with st.expander(f"📁 Referenced {len(unique_sources)} source files"):
                            for source in unique_sources:
                                st.code(source, language="text")
                        
                        # Add to assistant message
                        assistant_message = {
                            "role": "assistant", 
                            "content": response,
                            "sources": unique_sources
                        }
                    else:
                        assistant_message = {"role": "assistant", "content": response}
                    
                    st.session_state.codebase_messages.append(assistant_message)
                    
                    # Update chat history for chain
                    st.session_state.codebase_chat_history.append((prompt, response))
                    
                    # Keep only last 10 exchanges to manage memory
                    if len(st.session_state.codebase_chat_history) > 10:
                        st.session_state.codebase_chat_history = st.session_state.codebase_chat_history[-10:]
                
                except Exception as e:
                    error_msg = f"❌ **Error processing question**: {str(e)}"
                    st.error(error_msg)
                    st.session_state.codebase_messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with quick help
    with st.sidebar:
        st.subheader("💡 Quick Help")
        st.markdown("""
        **Example Questions:**
        
        🏗️ **Architecture**
        - "What's the overall structure of the app?"
        - "How are the modules organized?"
        
        🧪 **Functionality**  
        - "How does PSMILES generation work?"
        - "Where is MD simulation implemented?"
        
        🔧 **Implementation**
        - "Show me the main classes"
        - "How do I add a new UI component?"
        
        📊 **Data Flow**
        - "How does data move between components?"
        - "What's the integration workflow?"
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.codebase_chat_history = []
            st.session_state.codebase_messages = [
                {"role": "assistant", "content": "Hello! I'm your Insulin AI codebase assistant. I can help you understand the framework architecture, locate specific functionality, and navigate the code. What would you like to know?"}
            ]
            st.rerun()
        
        # Stats
        if 'qa_chain' in locals():
            st.subheader("📊 Codebase Stats")
            project_root = get_project_root()
            files = scan_codebase_files(project_root)
            st.metric("Files Indexed", len(files))
            st.metric("Chat Exchanges", len(st.session_state.codebase_chat_history)) 