## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- Conda or pip for package management
- Ollama (for local LLM inference)

### 1. Install Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd insulin-ai

# Create conda environment
conda create -n insulin-ai python=3.9
conda activate insulin-ai

# Install requirements
pip install -r requirements.txt
```

### 2. Setup Ollama
```bash
# Install Ollama (https://ollama.ai)
# Then pull the required model
ollama pull llama3.2
```

### 3. Get Semantic Scholar API Key (Recommended)
**Without an API key, you'll hit severe rate limits (100 requests/5min)**
**With a free API key, you get 1000 requests/5min**

1. Go to [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Set the environment variable:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
SEMANTIC_SCHOLAR_API_KEY=your-api-key-here
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
``` 