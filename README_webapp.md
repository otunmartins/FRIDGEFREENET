# Insulin AI Web Application

A ChatGPT-like web interface for the AI-Driven Design of Fridge-Free Insulin Delivery Patches project. This application provides an intelligent conversational interface for literature mining, research assistance, and material discovery.

## 🌟 Features

### 💬 Three Chat Modes
- **General Chat**: Project overview and general questions
- **Literature Mining**: Search and analyze scientific literature  
- **Research Assistant**: Technical research guidance and support

### 🎨 Modern Interface
- ChatGPT-inspired design with clean, responsive UI
- Real-time message exchange with markdown support
- Mobile-friendly with collapsible sidebar
- Status indicators for system health
- Example prompts for each mode

### 🔧 Technical Features
- Flask backend with RESTful API
- LangChain integration with Ollama for local LLM
- Session-based conversation history
- Real-time system status monitoring
- Error handling and loading states

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running locally
- Virtual environment (recommended)

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

3. **Pull the required model**:
   ```bash
   ollama pull llama3.2
   ```

4. **Set environment variables** (optional):
   ```bash
   export SEMANTIC_SCHOLAR_API_KEY="your_api_key"  # Optional
   export OLLAMA_MODEL="llama3.2"                  # Default
   export OLLAMA_HOST="http://localhost:11434"     # Default
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## 📖 Usage Guide

### Chat Modes

#### 🗨️ General Chat
Use for:
- Understanding the project goals and methodology
- Learning about the active learning framework
- General questions about insulin delivery research

**Example prompts**:
- "What is the goal of this insulin AI project?"
- "How does the active learning framework work?"
- "What makes this approach innovative?"

#### 📚 Literature Mining
Use for:
- Searching scientific literature for materials
- Finding papers on specific topics
- Analyzing research trends

**Example prompts**:
- "Find polymers for insulin stabilization at room temperature"
- "Search for hydrogels used in transdermal drug delivery"
- "What materials are used for protein preservation without refrigeration?"

#### 🔬 Research Assistant
Use for:
- Technical questions about materials science
- Understanding biological mechanisms
- Research methodology guidance

**Example prompts**:
- "Explain the mechanism of insulin degradation at room temperature"
- "What are the key challenges in transdermal insulin delivery?"
- "How do hydrogels protect proteins from thermal degradation?"

### Interface Features

#### Sidebar
- **New Chat**: Start a fresh conversation
- **Mode Selection**: Switch between chat modes
- **Example Prompts**: Click to use pre-written examples
- **System Status**: Check if all components are online

#### Chat Area
- **Message History**: Scrollable conversation history
- **Markdown Support**: Rich text formatting in responses
- **Real-time Typing**: Auto-resizing input field
- **Loading Indicators**: Visual feedback during processing

#### Keyboard Shortcuts
- **Enter**: Send message
- **Shift+Enter**: New line
- **Ctrl/Cmd+Enter**: Force send
- **Escape**: Close modals/sidebar
- **/**: Focus message input

## 🏗️ Architecture

### Backend Components
```
app.py                 # Main Flask application
chatbot_system.py      # LangChain-based chatbot with multiple modes
literature_mining_system.py  # Existing literature mining system
semantic_scholar_client.py   # API client for paper search
ollama_client.py       # Ollama LLM client
```

### Frontend Components
```
templates/index.html   # Main HTML template
static/css/style.css   # Comprehensive styling
static/js/app.js       # JavaScript functionality
```

### API Endpoints
- `GET /` - Main application page
- `POST /api/chat` - Send messages and receive responses
- `POST /api/new-chat` - Start new conversation
- `GET /api/status` - Check system status
- `GET /api/examples` - Get example prompts

## 🔧 Configuration

### Environment Variables
```bash
# Optional API key for enhanced paper search
SEMANTIC_SCHOLAR_API_KEY="your_api_key"

# Ollama configuration
OLLAMA_MODEL="llama3.2"
OLLAMA_HOST="http://localhost:11434"

# Flask configuration
SECRET_KEY="your_secret_key"
FLASK_ENV="development"  # or "production"
```

### Ollama Models
The application supports various Ollama models:
- `llama3.2` (recommended, 3B parameters)
- `llama3.2:1b` (faster, smaller model)
- `phi3.5` (alternative option)
- Custom models as available

## 🐛 Troubleshooting

### Common Issues

#### "System offline" status
1. Check if Ollama is running: `ollama serve`
2. Verify model is installed: `ollama list`
3. Test model directly: `ollama run llama3.2`

#### Slow responses
1. Consider using a smaller model: `ollama pull llama3.2:1b`
2. Ensure sufficient RAM (8GB+ recommended)
3. Close other applications to free memory

#### Literature mining not working
1. Check internet connection
2. Verify Semantic Scholar API limits
3. Check if rate limiting is occurring

#### Connection errors
1. Verify Ollama is running on correct port (11434)
2. Check firewall settings
3. Try restarting Ollama service

### Performance Tips
- Use GPU acceleration if available
- Allocate sufficient system memory
- Consider model size vs. response quality trade-offs
- Monitor system resources during heavy usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the AI-Driven Design of Fridge-Free Insulin Delivery Patches research initiative.

## 📞 Support

For technical issues or questions:
1. Check the troubleshooting section above
2. Review system logs for error messages
3. Ensure all dependencies are correctly installed
4. Verify Ollama service is properly configured

---

**Happy researching! 🧬💊** 