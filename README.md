---
title: FridgeFreeNet - Insulin AI
emoji: 🩹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - healthcare
  - materials-science
  - insulin
  - ai
  - drug-delivery
  - polymer-chemistry
  - literature-mining
short_description: AI-driven discovery of materials for fridge-free insulin delivery patches
---

# 🩹 FridgeFreeNet - Insulin AI

Advanced AI assistant for discovering materials for insulin patches that don't require refrigeration.

## 🚀 Full Flask Application Features

- **📚 Literature Mining**: Real-time paper analysis with material candidate extraction
- **🧬 PSMILES Generation**: AI-powered polymer structure creation with validation
- **🎯 Material Discovery**: Comprehensive candidate ranking and analysis
- **🔧 Interactive Workflow**: SVG visualization, dimerization, copolymerization
- **💬 Enhanced Chat**: Memory and conversation context
- **🌐 API Endpoints**: Complete RESTful API for all functionality

## 🔧 System Requirements

- Ollama server for AI model inference
- Semantic Scholar API key (optional but recommended)
- Full literature mining and PSMILES systems
- Python 3.9+ with all dependencies

## 📡 API Endpoints

- `GET /` - Main web interface
- `POST /api/chat` - Chat with AI system
- `POST /api/psmiles/action` - Interactive PSMILES workflow
- `GET /api/literature-stream/<message>` - Streaming literature mining
- `GET /api/status` - System status check

## 🎯 Usage

The application provides a ChatGPT-like interface for insulin delivery research with advanced AI capabilities for literature mining, material discovery, and polymer structure generation.

## 🚀 Features

### 🤖 Multi-Model AI Support
- **Ollama Integration**: Local LLM support for general conversations and research assistance
- **LlaSMol Integration**: Specialized chemistry language model for molecular analysis and SMILES processing
- **Dynamic Model Switching**: Switch between models based on task requirements

### 🔬 Chemistry-Specialized Capabilities (LlaSMol)
- **Name Conversion**: SMILES ↔ IUPAC ↔ Molecular Formula
- **Property Prediction**: Solubility, toxicity, permeability, stability
- **Molecule Description**: AI-powered molecular captioning and analysis
- **Chemical Synthesis**: Forward and retrosynthesis prediction
- **PSMILES Generation**: Specialized polymer SMILES generation for materials design

### 📚 Literature Mining System
- **Semantic Scholar Integration**: Direct access to 200+ million scientific papers
- **AI-Enhanced Analysis**: Intelligent extraction of material properties and insights
- **Real-time Progress Tracking**: Live updates during literature analysis
- **Material Discovery**: Automated identification of promising material candidates

### 🧠 Advanced Conversation Memory
- **Persistent Memory**: Conversations remembered across sessions
- **Multiple Memory Types**: Buffer, summary, and windowed memory options
- **Context-aware Responses**: Maintains conversation context for better interactions

### 🧪 PSMILES (Polymer SMILES) Generator
- **Specialized Polymer Notation**: Generate and validate polymer SMILES strings
- **Multi-Model Support**: Works with both Ollama and LlaSMol backends
- **Validation System**: Comprehensive rule checking and chemical validation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                    │
├─────────────────────────────────────────────────────────────┤
│  Chat System     │  Literature Mining  │  Model Management  │
│  ┌─────────────┐ │  ┌────────────────┐ │  ┌───────────────┐ │
│  │   Ollama    │ │  │ Semantic       │ │  │   LlaSMol     │ │
│  │   Models    │ │  │ Scholar API    │ │  │   Manager     │ │
│  └─────────────┘ │  └────────────────┘ │  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    LangChain Memory                         │
├─────────────────────────────────────────────────────────────┤
│  PSMILES Generator │  Chemistry Analysis │  Material Discovery │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for LlaSMol)
- Ollama installed and running (for Ollama models)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/insulin-ai.git
cd insulin-ai

# Initialize the LlaSMol submodule
git submodule update --init --recursive
```

### 2. Install Dependencies
```bash
# Install basic dependencies
pip install -r requirements.txt

# For LlaSMol support (requires CUDA for optimal performance)
pip install torch transformers peft accelerate bitsandbytes

# Optional: For enhanced performance
# pip install flash-attn deepspeed
```

### 3. Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration:
# - SEMANTIC_SCHOLAR_API_KEY (optional, for enhanced literature mining)
# - DEFAULT_MODEL_TYPE=ollama  # or 'llamol'
# - LLAMOL_MODEL=osunlp/LlaSMol-Mistral-7B
# - OLLAMA_MODEL=llama3.2
# - OLLAMA_HOST=http://localhost:11434
```

### 4. Initialize Ollama (if using Ollama models)
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2
ollama pull mistral
```

## 🚀 Usage

### Starting the Application
```bash
# Start the Flask web application
python app.py

# The application will be available at http://localhost:5000
```

### Model Selection

#### Using Ollama Models
```python
from chatbot_system import InsulinAIChatbot

chatbot = InsulinAIChatbot(
    model_type="ollama",
    ollama_model="llama3.2"
)
```

#### Using LlaSMol Models
```python
from chatbot_system import InsulinAIChatbot

chatbot = InsulinAIChatbot(
    model_type="llamol",
    llamol_model="osunlp/LlaSMol-Mistral-7B"
)
```

#### Dynamic Model Switching
```python
# Switch to LlaSMol for chemistry tasks
success = chatbot.switch_model("llamol")

# Switch back to Ollama for general conversation
success = chatbot.switch_model("ollama", "llama3.2")
```

### Web Interface Usage

1. **General Chat**: Ask questions about the project, get research guidance
2. **Research Mode**: Technical discussions with enhanced context
3. **Literature Mining**: Search scientific literature for material insights
4. **PSMILES Generation**: Generate polymer SMILES for material design
5. **Model Management**: Switch between different AI models as needed

### API Endpoints

#### Chat API
```bash
# General conversation
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "type": "general"}'

# Chemistry-focused query with LlaSMol
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the SMILES for glucose?", "type": "research", "model_type": "llamol"}'
```

#### Model Management API
```bash
# Get model information
curl http://localhost:5000/api/models/info

# Switch models
curl -X POST http://localhost:5000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_type": "llamol"}'

# Load specific LlaSMol model
curl -X POST http://localhost:5000/api/models/llamol/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "osunlp/LlaSMol-Mistral-7B"}'
```

## 🧪 Testing

### Run the Test Suite
```bash
# Test LlaSMol integration
python test_llamol_integration.py

# Test individual components
python -m pytest tests/
```

### Manual Testing
```bash
# Test chatbot system
python chatbot_system.py

# Test PSMILES generator
python psmiles_generator.py

# Test literature mining
python literature_mining_system.py
```

## 🔧 Configuration

### Environment Variables
- `DEFAULT_MODEL_TYPE`: Default model type ('ollama' or 'llamol')
- `LLAMOL_MODEL`: LlaSMol model name from HuggingFace
- `OLLAMA_MODEL`: Ollama model name
- `OLLAMA_HOST`: Ollama server host
- `CHATBOT_MEMORY_TYPE`: Memory type ('buffer', 'summary', 'buffer_window')
- `SEMANTIC_SCHOLAR_API_KEY`: API key for enhanced literature access

### Model Selection Guidelines
- **Use Ollama** for:
  - General conversation
  - Research discussions
  - Brainstorming sessions
  - Fast responses

- **Use LlaSMol** for:
  - Chemistry questions
  - SMILES analysis
  - Molecular property prediction
  - Chemical synthesis planning

## 📊 Available Models

### Ollama Models
- `llama3.2`: Latest Llama model (recommended)
- `mistral`: Fast and efficient
- `codellama`: Code-focused variant
- `mixtral`: Mixture of experts model

### LlaSMol Models
- `osunlp/LlaSMol-Mistral-7B`: Mistral-based chemistry model (recommended)
- `osunlp/LlaSMol-Llama2-7B`: Llama2-based chemistry model
- `osunlp/LlaSMol-CodeLlama-7B`: Code-focused chemistry model
- `osunlp/LlaSMol-Galactica-6.7B`: Science-focused model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python test_llamol_integration.py`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/new-feature`)
8. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlaSMol Team](https://github.com/OSU-NLP-Group/LLM4Chem) for the chemistry-specialized language model
- [Ollama](https://ollama.ai/) for local LLM infrastructure
- [Semantic Scholar](https://www.semanticscholar.org/) for literature access
- [LangChain](https://langchain.com/) for conversation memory and AI orchestration

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Run the test suite to verify your setup

---

**🎯 Goal**: Accelerate the discovery of materials for fridge-free insulin delivery patches through AI-driven research and analysis. 