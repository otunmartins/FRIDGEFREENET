# Insulin AI: Intelligent Material Research System 🧬

A comprehensive AI-powered research platform for insulin delivery patch materials, combining literature mining, intelligent conversation, and advanced chemistry analysis.

## 🚀 Features

- **Literature Mining**: Semantic Scholar integration for intelligent paper discovery and analysis
- **Intelligent Chatbot**: Multi-mode conversational AI for research assistance
- **Memory System**: Persistent conversation memory with multiple strategies
- **PSMILES Generation**: Polymer SMILES generation and validation
- **Enhanced Analysis**: Advanced polymer structure analysis and visualization
- **Multi-Model Support**: Works with Ollama backend

## 🔬 System Architecture

```
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Literature     │ │  Intelligent    │ │   PSMILES       │
│   Mining        │ │   Chatbot       │ │  Generator      │
│  (Semantic      │ │    (Memory-     │ │  (Polymer       │
│   Scholar)      │ │   Enhanced)     │ │   Chemistry)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌─────────────────┐
                    │  Ollama    │ │
                    │  LLM       │ │
                    │  Backend   │ │
                    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM
- Ollama installed and running

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/insulin-ai.git
cd insulin-ai

# Initialize submodules
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies
pip install flask flask-cors langchain-ollama rdkit-pypi

# Set up environment variables
cp env_example.txt .env
# Edit .env with your settings
```

## 🚀 Quick Start

### 1. Start Ollama
```bash
# Make sure Ollama is running
ollama serve

# Pull required models
ollama pull llama3.2
ollama pull mistral
```

### 2. Configure Environment
```bash
# Example .env configuration
SEMANTIC_SCHOLAR_API_KEY=your_key_here
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
CHATBOT_MEMORY_TYPE=buffer_window
CHATBOT_MEMORY_DIR=chat_memory
SECRET_KEY=your_secret_key
```

### 3. Run the Application
```bash
python app.py
```

Navigate to `http://localhost:5000` to access the web interface.

## 💬 Usage

### Basic Chat Interface
The main interface supports multiple conversation modes:

- **General**: Project overview and general questions
- **Research**: Technical research assistance with enhanced prompts
- **Literature**: Automated literature mining and paper analysis
- **PSMILES**: Polymer chemistry and structure generation

#### Example Conversations

```python
# Initialize chatbot system
from chatbot_system import InsulinAIChatbot

chatbot = InsulinAIChatbot(
    model_type="ollama",
    ollama_model="llama3.2"
)

# Research conversation
response = chatbot.chat(
    "What are the key properties for insulin stability in polymer matrices?",
    session_id="research_session",
    mode='research'
)

# Literature mining query
response = chatbot.chat(
    "Find papers about hydrogel-based insulin delivery systems",
    session_id="lit_session", 
    mode='literature'
)

# Chemistry-focused query
response = chatbot.chat(
    "Generate PSMILES for a biocompatible polymer suitable for insulin patches",
    session_id="chem_session",
    mode='research'
)
```

### API Endpoints

#### Model Management
```bash
# Get model information
curl http://localhost:5000/api/models/info

# Switch models
curl -X POST http://localhost:5000/api/models/switch \
-H "Content-Type: application/json" \
-d '{"model_type": "ollama", "model_name": "mistral"}'

# Test model connectivity
curl http://localhost:5000/api/status
```

#### Chat API
```bash
# Send chat message
curl -X POST http://localhost:5000/api/chat \
-H "Content-Type: application/json" \
-d '{
  "message": "What are biocompatible polymers for insulin delivery?",
  "type": "research"
}'
```

#### Memory Management
```bash
# Get memory summary
curl http://localhost:5000/api/memory/summary?session_id=your_session

# Clear conversation history
curl -X POST http://localhost:5000/api/memory/clear \
-d '{"session_id": "your_session", "mode": "research"}'
```

## 🧠 Memory System

The system supports multiple memory strategies:

- **Buffer Memory**: Stores all conversation history
- **Summary Memory**: Maintains running summary of conversations  
- **Buffer Window**: Keeps last N interactions (recommended)

Configure via environment variables:
```bash
CHATBOT_MEMORY_TYPE=buffer_window  # buffer, summary, buffer_window
CHATBOT_MEMORY_DIR=chat_memory
```

## 🧪 PSMILES Generation

Advanced polymer chemistry capabilities:

```python
from psmiles_generator import PSMILESGenerator

generator = PSMILESGenerator(
    model_type='ollama',
    ollama_model='llama3.2'
)

# Generate polymer structure
result = generator.generate_psmiles("biodegradable polymer for insulin")
print(result['psmiles'])  # [*]CC(C(=O)O)[*]

# Validate structure
validation = generator.validate_psmiles("[*]CC(O)[*]")
print(validation['valid'])  # True
```

## ⚙️ Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEMANTIC_SCHOLAR_API_KEY` | API key for paper access | None |
| `OLLAMA_MODEL` | Default Ollama model | llama3.2 |
| `OLLAMA_HOST` | Ollama server URL | http://localhost:11434 |
| `CHATBOT_MEMORY_TYPE` | Memory strategy | buffer_window |
| `CHATBOT_MEMORY_DIR` | Memory storage directory | chat_memory |
| `SECRET_KEY` | Flask session key | random |

## 🔧 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart Ollama if needed
   ollama serve
   ```

2. **Memory Issues**
   ```bash
   # Clear old memory files
   rm -rf chat_memory/*
   
   # Use lighter memory strategy
   export CHATBOT_MEMORY_TYPE=buffer_window
   ```

3. **Literature Mining Issues**
   - Verify Semantic Scholar API key
   - Check network connectivity
   - Review rate limiting

## 📊 System Status

Check system health via:
```bash
curl http://localhost:5000/api/status
```

Returns status for:
- Literature mining system
- Chatbot and memory
- PSMILES generator  
- Model availability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the local LLM infrastructure
- [Semantic Scholar](https://www.semanticscholar.org/) for academic paper access
- [LangChain](https://langchain.com/) for conversation memory and LLM integration
- RDKit community for chemistry informatics tools 