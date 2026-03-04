# Insulin AI: AI-Driven Design of Fridge-Free Insulin Delivery Patches 🧬💊

A **CLI-first** materials discovery platform designed for **OpenCode.ai**, Claude Code, and terminal-based AI coding agents. The system integrates literature mining with PSMILES and CPU-only MD simulation, creating an iterative active learning feedback loop. **The code is the platform** – no web UI required.

## 🎯 Project Overview

This project implements an active learning framework that systematically combines literature knowledge extraction with AI-driven polymer design. The system addresses the critical challenge of maintaining insulin stability without refrigeration while enabling controlled transdermal delivery through computational discovery methods.

### Key Innovation
- **Active Learning Framework**: Iterative cycles combining literature mining with generative material design
- **PSMILES Generation**: Specialized polymer SMILES generator with conversation memory for novel material creation
- **Multi-Modal Interface**: Web-based ChatGPT-like interface with multiple specialized modes
- **MCP Integration**: Model Context Protocol for enhanced literature mining capabilities

## 🚀 Features

### 🔍 **Intelligent Literature Mining**
- **Natural Language Queries**: Ask questions in plain English about materials
- **Smart Interpretation**: LLM understands requests in context of insulin delivery
- **Adaptive Search**: Generates targeted search queries based on interests
- **MCP-Enhanced**: Model Context Protocol integration for sophisticated Semantic Scholar interactions
- **Structured Results**: JSON-formatted material candidates with confidence scoring

### 🧪 **PSMILES Material Generation**
- **Polymer SMILES Generator**: Create chemically valid polymer structures with PSMILES notation
- **Conversation Memory**: LangChain-powered memory system maintains context across interactions
- **Copolymerization**: Systematic exploration of complex polymer architectures
- **Fallback Mechanisms**: Reliable generation for known materials with novel structure capabilities
- **Chemical Validation**: Comprehensive rule sets ensuring chemical validity

### 💬 **Multi-Mode Chat Interface**
- **General Chat**: Project overview and methodology discussions
- **Literature Mining**: Search and analyze scientific literature
- **Research Assistant**: Technical research guidance and PSMILES generation
- **MCP Mode**: Enhanced mining with Model Context Protocol

### 🧠 **Advanced Memory System**
- **Persistent Memory**: Conversations preserved across application restarts
- **Multiple Strategies**: Buffer, Summary, and Buffer Window memory types
- **Session Management**: Separate memory for different conversation modes
- **Automatic Archival**: Research findings automatically saved with metadata

### 🌐 **Modern Web Interface**
- **ChatGPT-inspired Design**: Clean, responsive UI with real-time messaging
- **Mobile-Friendly**: Collapsible sidebar and touch-optimized interface
- **Status Monitoring**: Real-time system health checks
- **Example Prompts**: Context-aware suggestions for each mode

## 🏗️ System Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Literature     │ │  PSMILES        │ │   Web           │
│   Mining        │ │  Generator      │ │  Interface      │
│  + MCP Server   │ │  + Memory       │ │  (Flask)        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌─────────────────┐
                    │  Ollama         │
                    │  LLM Backend    │
                    │  + LangChain    │
                    └─────────────────┘
```

### Core Components

- **`app.py`** (2,296 lines): Main Flask web application with RESTful API
- **`literature_mining_system.py`** (1,432 lines): Intelligent literature mining with Semantic Scholar integration
- **`psmiles_generator.py`** (806 lines): Specialized PSMILES generator with conversation memory
- **`chatbot_system.py`** (516 lines): Multi-mode conversational AI system
- **`mcp_client.py`** (666 lines): Model Context Protocol client for enhanced literature mining
- **`semantic_scholar_server.py`** (503 lines): MCP server for structured academic search

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running (for LLM modes)
- 8GB+ RAM (16GB recommended)
- **CPU-only**: No GPU required; MD simulations run on CPU (OpenMM + PME)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/insulin-ai.git
cd insulin-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your configuration

# Start Ollama and pull models
ollama serve
ollama pull llama3.2

# Run the application
python app.py
```

Navigate to `http://localhost:5000` to access the web interface.

### Environment Configuration

Create a `.env` file with the following variables:

```bash
# Semantic Scholar API (optional, but recommended for higher rate limits)
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here

# Ollama Configuration
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434

# Memory System Configuration
CHATBOT_MEMORY_TYPE=buffer_window  # buffer, summary, buffer_window
CHATBOT_MEMORY_DIR=chat_memory

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

## 📖 Usage Guide

### Web Interface

1. **Start the Application**: `python app.py`
2. **Access Interface**: Navigate to `http://localhost:5000`
3. **Select Mode**: Choose from General, Literature Mining, Research Assistant, or MCP modes
4. **Start Chatting**: Use example prompts or ask your own questions

### Chat Modes

#### 🗨️ General Chat
Project overview and methodology discussions
```
"What is the goal of this insulin AI project?"
"How does the active learning framework work?"
"Explain the PSMILES generation system"
```

#### 📚 Literature Mining
Search and analyze scientific literature
```
"Find polymers for insulin stabilization at room temperature"
"Search for hydrogels used in transdermal drug delivery"
"Analyze recent advances in protein preservation materials"
```

#### 🔬 Research Assistant
Technical guidance and PSMILES generation
```
"Generate PSMILES for a biodegradable insulin delivery polymer"
"Explain insulin degradation mechanisms"
"Create a copolymer with PEG and chitosan"
```

#### 🔗 MCP Mode
Enhanced literature mining with structured analysis
```
"Find biocompatible polymers with thermal stability above 40°C"
"Analyze PLGA-based insulin delivery systems"
"Search for nanoparticle formulations for protein preservation"
```

### Active Learning (CLI – OpenCode / Claude Code Interface)

**Primary interface: CLI**

```bash
# Full feedback loop (literature + MD evaluation)
insulin-ai discover --iterations 2

# Literature-only (no MD)
insulin-ai discover --no-md -n 1

# Literature mining only
insulin-ai mine

# Evaluate PSMILES
insulin-ai evaluate "[*]OCC[*]" "[*]CC[*]"

# Status
insulin-ai status
```

Without installing: `python insulin_ai_cli.py discover` (from project root)

Requires: `openmm`, `openmmforcefields`, `rdkit`. Falls back to RDKit proxy when full MD unavailable.

See **OpenCode_PLATFORM.md** for OpenCode-as-platform workflow.

### Programmatic Usage

#### Literature Mining
```python
from literature_mining_system import MaterialsLiteratureMiner

miner = MaterialsLiteratureMiner()

# Intelligent mining with natural language
results = miner.intelligent_mining(
    "smart polymers that respond to temperature"
)

print(f"Found {len(results['material_candidates'])} materials")
```

#### PSMILES Generation
```python
from psmiles_generator import PSMILESGenerator

generator = PSMILESGenerator(ollama_model='llama3.2')

# Generate polymer structure
result = generator.generate_psmiles("biodegradable polymer for insulin")
print(f"PSMILES: {result['psmiles']}")

# Copolymerization
copolymer = generator.perform_copolymerization(
    "[*]CC[*]", "[*]COC[*]", [1, 1]
)
```

#### Enhanced MCP Mining
```python
from mcp_client import MCPLiteratureMinerSync
from ollama_client import OllamaClient

ollama_client = OllamaClient()
mcp_miner = MCPLiteratureMinerSync(ollama_client=ollama_client)

results = mcp_miner.intelligent_mining_with_mcp(
    user_request="chitosan hydrogels for insulin stabilization",
    max_papers=20,
    recent_only=True
)
```

### Interactive Sessions

#### Literature Mining
```bash
# Interactive literature mining
python interactive_mining.py

# Demo mode
python interactive_mining.py --demo
```

#### Memory System Demo
```bash
# Demonstrate memory persistence and management
python demo_memory_chatbot.py
```

## 🧠 Memory System

### Memory Types
- **Buffer Memory**: Keeps full conversation history (good for short conversations)
- **Summary Memory**: Summarizes old conversations to save tokens (for long conversations)  
- **Buffer Window**: Keeps last N interactions (recommended, balances context and performance)

### Memory Management
```python
from chatbot_system import InsulinAIChatbot

chatbot = InsulinAIChatbot(memory_type="buffer_window")

# Get memory summary
summary = chatbot.get_memory_summary(session_id, mode='research')

# Clear specific mode memory
chatbot.clear_history(session_id, mode='research')
```

### Persistent Storage
Memory automatically saves to disk:
```
chat_memory/
├── user123_general.pkl      # General conversation memory
├── user123_research.pkl     # Research conversation memory  
├── user123_literature.pkl   # Literature conversation memory
└── user456_general.pkl      # Another user's memory
```

## 🔗 API Reference

### REST Endpoints

#### Chat API
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Find biodegradable polymers for insulin patches",
  "type": "literature"  # general, literature, research, mcp
}
```

#### Memory Management
```bash
# Get memory summary
GET /api/memory/summary?session_id=abc123&mode=research

# Clear memory
POST /api/memory/clear
{"session_id": "abc123", "mode": "research"}

# Memory configuration
GET /api/memory/config
```

#### System Status
```bash
GET /api/status
```

Returns:
```json
{
  "literature_miner": true,
  "mcp_literature_miner": true,
  "chatbot": true,
  "mcp_status": "Available",
  "psmiles_generator": true
}
```

### Python API

#### Core Classes
- **`InsulinAIChatbot`**: Multi-mode conversational AI with memory
- **`MaterialsLiteratureMiner`**: Intelligent literature mining system
- **`PSMILESGenerator`**: Polymer SMILES generation with validation
- **`MCPLiteratureMinerSync`**: Enhanced mining with MCP integration

## 📊 Project Milestones

### ✅ Milestone 1: LLM Literature Mining System
- Intelligent mining with natural language queries
- Semantic Scholar API integration
- Structured material database
- Interactive mining interface

### ✅ Milestone 2: PSMILES Generation & Memory System  
- Specialized PSMILES generator with conversation memory
- Copolymerization capabilities
- Persistent memory across sessions
- Web interface with multiple chat modes

### ✅ Milestone 3: CPU-Only MD Simulation Pipeline (Revamp)
- **OpenMM + PME** (Particle Mesh Ewald) for long-range electrostatics
- **CPU-only** execution – no GPU required
- RDKit proxy fallback when GAFF parameterization unavailable (e.g., openff-toolkit)
- Integrated into active learning feedback loop

### 📋 Milestone 4: Complete Active Learning Framework
- Integration of all components
- Dynamic feedback loops between literature mining and generation
- Iterative improvement cycles
- Final material recommendations

## 🔧 Advanced Configuration

### Ollama Models
Supported models with trade-offs:
- **`llama3.2`**: Recommended (3B parameters, good balance)
- **`llama3.2:1b`**: Faster, smaller model
- **`mistral`**: Alternative option
- **`phi3.5`**: Compact alternative

### Performance Tuning
```bash
# Memory optimization
export CHATBOT_MEMORY_TYPE=buffer_window

# Model selection for speed vs quality
export OLLAMA_MODEL=llama3.2:1b  # For faster responses
export OLLAMA_MODEL=llama3.2      # For better quality
```

### Development Mode
```bash
# Enable debug mode
export FLASK_ENV=development

# Verbose logging
export LOG_LEVEL=DEBUG
```

## 🐛 Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Verify model availability
ollama list
```

#### Memory Issues
```bash
# Clear old memory files
rm -rf chat_memory/*

# Use lighter memory strategy
export CHATBOT_MEMORY_TYPE=buffer_window
```

#### MCP Integration Issues
```bash
# Test MCP server directly
python semantic_scholar_server.py

# Check MCP client functionality
python test_mcp_integration.py
```

#### Literature Mining Rate Limits
- Add Semantic Scholar API key to `.env`
- Reduce `max_papers` parameter
- Add delays between requests

### Performance Tips
- **MD simulations**: Run entirely on CPU; install `openmm openmmforcefields` for full MD
- Allocate sufficient system memory (8GB+)
- Consider model size vs response quality trade-offs
- Monitor system resources during heavy usage
- See `REVAMP_PLAN.md` for code-as-platform architecture

## 📁 File Structure

```
insulin-ai/
├── app.py                          # Main Flask web application
├── chatbot_system.py               # Multi-mode chatbot with memory
├── literature_mining_system.py     # Intelligent literature mining
├── psmiles_generator.py            # PSMILES generation system
├── mcp_client.py                   # Model Context Protocol client
├── semantic_scholar_server.py      # MCP server for literature search
├── semantic_scholar_client.py      # Direct Semantic Scholar API client
├── ollama_client.py                # Ollama LLM client
├── requirements.txt                # Python dependencies
├── milestones.txt                  # Project roadmap
├── templates/
│   └── index.html                  # Web interface template
├── static/
│   ├── css/                        # Stylesheets
│   └── js/                         # JavaScript files
├── chat_memory/                    # Persistent conversation memory
├── mining_results/                 # Literature mining results
└── demo_*.py                       # Demonstration scripts
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the AI-Driven Design of Fridge-Free Insulin Delivery Patches research initiative.

## 🙏 Acknowledgments

- **Ollama Team**: For the local LLM infrastructure
- **LangChain**: For conversation memory and LLM integration
- **Semantic Scholar**: For academic literature access
- **Meta FAIR**: For the Universal Model for Atoms (UMA) force field
- **Materials Science Community**: For foundational research

---

**🧬 Accelerating insulin delivery materials discovery through AI-driven research! 💊** 