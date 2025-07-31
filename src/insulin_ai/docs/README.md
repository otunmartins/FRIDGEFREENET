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

### 🤖 AI Support
- **Ollama Integration**: Local LLM support for conversations and research assistance
- **Chemistry Processing**: Molecular analysis and SMILES processing capabilities
- **Research Enhancement**: AI-powered literature mining and material discovery

### 🔬 Chemistry Capabilities
- **Chemical Structure Processing**: SMILES parsing and validation
- **Property Analysis**: Assessment of material properties for insulin delivery
- **Molecular Understanding**: AI-powered chemical structure analysis
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
- **Ollama Backend**: Uses Ollama models for polymer structure generation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                    │
├─────────────────────────────────────────────────────────────┤
│  Chat System     │  Literature Mining  │  Model Management  │
│  ┌─────────────┐ │  ┌────────────────┐ │  ┌───────────────┐ │
│  │   Ollama    │ │  │ Semantic       │ │  │   Ollama      │ │
│  │   Models    │ │  │ Scholar API    │ │  │   Manager     │ │
│  └─────────────┘ │  └────────────────┘ │  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    LangChain Memory                         │
├─────────────────────────────────────────────────────────────┤
│  PSMILES Generator │  Chemistry Analysis │  Material Discovery │
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Ollama installed and running
- Git for repository cloning

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/insulin-ai.git
cd insulin-ai
```

### 2. Install Dependencies
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration:
# - SEMANTIC_SCHOLAR_API_KEY (optional, for enhanced literature mining)
# - DEFAULT_MODEL_TYPE=ollama
# - OLLAMA_MODEL=llama3.2
# - OLLAMA_HOST=http://localhost:11434
```

## 🚀 Usage

### Starting the Application
```bash
# Start the Flask web application
python app.py

# The application will be available at http://localhost:5000
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

# Chemistry-focused query
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the SMILES for glucose?", "type": "research"}'
```

#### Model Management API
```bash
# Get model information
curl http://localhost:5000/api/models/info

# Switch models
curl -X POST http://localhost:5000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_type": "ollama"}'
```

## 🧪 Testing

### Run the Test Suite
```bash
# Test components
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
- `DEFAULT_MODEL_TYPE`: Default model type ('ollama')
- `OLLAMA_MODEL`: Ollama model name
- `OLLAMA_HOST`: Ollama server host
- `CHATBOT_MEMORY_TYPE`: Memory type ('buffer', 'summary', 'buffer_window')
- `SEMANTIC_SCHOLAR_API_KEY`: API key for enhanced literature access

### Model Selection Guidelines
- **Use Ollama** for:
  - General conversation and research discussions
  - Chemistry questions and SMILES analysis
  - Material discovery and design
  - Fast responses and brainstorming sessions

## �� Available Models

### Ollama Models
- `llama3.2`: Latest Llama model (recommended)
- `mistral`: Fast and efficient
- `codellama`: Code-focused variant
- `mixtral`: Mixture of experts model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest tests/`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/new-feature`)
8. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM infrastructure
- [Semantic Scholar](https://www.semanticscholar.org/) for literature access
- [LangChain](https://langchain.com/) for conversation memory and AI orchestration
- [OpenMM](https://openmm.org/) for molecular dynamics simulations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Run the test suite to verify your setup

---

**🎯 Goal**: Accelerate the discovery of materials for fridge-free insulin delivery patches through AI-driven research and analysis. 