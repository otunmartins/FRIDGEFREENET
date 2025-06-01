# AI-Driven Insulin Delivery Materials Literature Mining System

## Overview

This system uses Large Language Models (LLMs) to intelligently mine scientific literature for materials suitable for fridge-free insulin delivery patches. The system can interpret user requests and translate them into targeted literature searches focused on insulin delivery applications.

## Key Features

### 🧠 **Intelligent Mining** (NEW!)
- **Natural language input**: Ask questions in plain English about materials
- **Smart interpretation**: LLM understands your request in context of insulin delivery
- **Adaptive search**: Generates targeted search queries based on your interests
- **Focused extraction**: Customizes material extraction based on your focus area
- **Relevance filtering**: Keeps searches on-topic while allowing creative connections

### 🔍 **Literature Mining**
- Semantic Scholar API integration for academic paper retrieval
- Automated paper deduplication and filtering
- Structured data extraction from scientific abstracts

### 🤖 **LLM Integration** 
- Local OLLAMA integration for privacy and control
- Structured prompt engineering for consistent results
- Fallback parsing for robust data extraction

### 📊 **Material Database**
- JSON-formatted material candidates with comprehensive properties
- Confidence scoring based on literature evidence
- Literature reference tracking

## Installation

### Prerequisites
- Python 3.8+
- OLLAMA installed and running locally
- Internet connection for Semantic Scholar API

### Setup
1. **Install OLLAMA**: Visit [https://ollama.ai](https://ollama.ai) and follow installation instructions

2. **Pull a language model**:
   ```bash
   ollama pull llama3.2
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional):
   ```bash
   cp env_example.txt .env
   # Edit .env with your Semantic Scholar API key (optional but recommended)
   ```

## Usage

### 🎮 **Interactive Mining** (NEW!)

The most user-friendly way to explore the system:

```bash
python interactive_mining.py
```

This starts an interactive session where you can:
- **Type questions in natural language** and see instant results
- **Use commands** like `help`, `examples`, `config`
- **Adjust settings** on the fly (`papers 30`, `save`, `recent`)
- **View previous results** with `last` command
- **Get suggestions** for improving your queries

Example interactive session:
```
🔍 Your question: smart polymers that respond to temperature

🤖 Processing: 'smart polymers that respond to temperature'...
⏳ This may take a few moments...

🎯 RELEVANCE SCORE: 9/10
🧠 INTERPRETATION: Focus on temperature-responsive materials for insulin stabilization
🔬 EXTRACTION FOCUS: Temperature-responsive drug delivery systems

📊 SEARCH RESULTS:
   Papers Analyzed: 18
   Materials Found: 7

🧪 MATERIAL CANDIDATES (showing 3 of 7):
   1. Poly(N-isopropylacrylamide) hydrogel
      Composition: Temperature-responsive polymer with LCST at 32°C
      Confidence: 8/10
```

### Intelligent Mining (Recommended)

The new intelligent mining feature allows you to ask questions in natural language:

```python
from literature_mining_system import MaterialsLiteratureMiner

# Initialize the system
miner = MaterialsLiteratureMiner()

# Ask questions in natural language
results = miner.intelligent_mining("smart polymers that respond to temperature")

# The system will:
# 1. Interpret your request in context of insulin delivery
# 2. Generate targeted search queries
# 3. Find relevant papers
# 4. Extract materials with focus on temperature-responsive properties

print(f"Found {len(results['material_candidates'])} materials")
print(f"Search strategy: {results['search_strategy']['interpretation']}")
```

### Example Intelligent Queries

```python
# Temperature-responsive materials
results = miner.intelligent_mining("smart polymers that respond to temperature")

# Nanotechnology focus
results = miner.intelligent_mining("nanotechnology for drug delivery")

# Sustainability focus
results = miner.intelligent_mining("green chemistry and sustainable materials")

# Related applications (adapts to insulin delivery)
results = miner.intelligent_mining("pain relief patches")

# Manufacturing focus
results = miner.intelligent_mining("3D printing materials for medical devices")

# Even computational approaches
results = miner.intelligent_mining("machine learning for protein folding")
```

### How Intelligent Mining Works

1. **Request Analysis**: LLM analyzes your request for relevance to insulin delivery
2. **Strategy Generation**: Creates targeted search queries and extraction focus
3. **Literature Search**: Searches academic papers using generated queries
4. **Focused Extraction**: Extracts material data with emphasis on your interest area
5. **Results**: Returns materials with context-aware analysis

### Off-Topic Request Handling

The system gracefully handles off-topic requests by trying to find connections:

```python
# This will suggest focusing on "food-safe materials for biomedical applications"
results = miner.intelligent_mining("cooking with organic ingredients")

if 'error' in results:
    print(f"Suggestion: {results['suggestion']}")
```

### Traditional Mining (Legacy)

For standard searches without request interpretation:

```python
# Basic mining with default search queries
results = miner.mine_insulin_delivery_materials(max_papers=30)

# Analyze specific materials
details = miner.get_material_details("chitosan")
```

## Demo Scripts

### 🎮 Interactive Session
```bash
python interactive_mining.py
```
**Best way to explore the system** - Type your own questions and see instant results with an easy-to-use interface.

### 🎮 Quick Demo Mode
```bash
python interactive_mining.py --demo
```
Runs automated sample queries to demonstrate the system capabilities.

### 📊 Intelligent Mining Demo
```bash
python demo_intelligent_mining.py
```
Comprehensive demonstration of how various user requests are translated into targeted searches.

### 📚 Basic Mining Demo  
```bash
python demo_literature_mining.py
```
Shows traditional literature mining functionality without intelligent interpretation.

## Output Format

Both mining methods return structured results:

```json
{
  "search_timestamp": "2024-01-15T10:30:00",
  "user_request": "smart polymers that respond to temperature",
  "search_strategy": {
    "relevant": true,
    "relevance_score": 9,
    "interpretation": "Focus on temperature-responsive materials for insulin stabilization",
    "search_queries": ["thermosensitive polymers insulin delivery", ...],
    "extraction_focus": "Temperature-responsive drug delivery systems"
  },
  "papers_analyzed": 25,
  "material_candidates": [
    {
      "material_name": "Thermoresponsive PNIPAAm hydrogel",
      "material_composition": "Poly(N-isopropylacrylamide) crosslinked network",
      "thermal_stability_temp_range": "15-45°C with phase transition at 32°C",
      "insulin_stability_duration": "72 hours at 37°C",
      "biocompatibility_data": "FDA approved for medical applications",
      "stabilization_mechanism": "Temperature-triggered conformational changes",
      "confidence_score": 8
    }
  ]
}
```

## Configuration

### Environment Variables
- `SEMANTIC_SCHOLAR_API_KEY`: API key for higher rate limits (optional)
- `OLLAMA_HOST`: OLLAMA server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: llama3.2)

### Search Parameters
- `max_papers`: Maximum papers to analyze (default: 50)
- `recent_only`: Focus on recent papers (default: True, 2020+)
- `save_results`: Save results to JSON files (default: True)

## Troubleshooting

### Common Issues

**OLLAMA Connection Error**:
```bash
# Ensure OLLAMA is running
ollama serve

# Verify model is available
ollama list
```

**SSL Warnings on macOS**:
The system includes urllib3 version constraints to handle LibreSSL compatibility.

**Rate Limiting**:
- Use Semantic Scholar API key for higher limits
- Reduce `max_papers` parameter
- Add delays between requests

**No Papers Found**:
- Check internet connection
- Verify search terms are relevant
- Try broader or different queries

### API Limits
- **Semantic Scholar**: 100 requests/5 minutes (public), 1000/5 minutes (with API key)
- **OLLAMA**: No limits (local processing)

## Architecture

### Intelligent Mining Workflow
```
User Request → LLM Strategy Generation → Targeted Search → Focused Extraction → Results
```

1. **Strategy Generation**: LLM interprets request and creates search plan
2. **Query Generation**: Creates 5 targeted search queries
3. **Literature Search**: Searches papers using Semantic Scholar
4. **Focused Extraction**: Extracts materials with custom focus
5. **Results Compilation**: Returns structured data with search context

### Traditional Mining Workflow
```
Default Queries → Literature Search → Standard Extraction → Results
```

### Components
- **MaterialsLiteratureMiner**: Main system class
- **SemanticScholarClient**: Academic paper search
- **OllamaClient**: Local LLM integration
- **Strategy Generation**: Request interpretation
- **Focused Extraction**: Context-aware material analysis

## Future Milestones

This literature mining system is **Milestone 1** of a larger project:

- **Milestone 2**: Generative model integration for novel material design
- **Milestone 3**: Molecular dynamics simulation integration (UMA force fields)
- **Milestone 4**: Complete active learning feedback loop

The intelligent mining feature provides the foundation for iterative improvement based on simulation results in future milestones.

## Project Context

This system supports research into **fridge-free insulin delivery patches** by:

1. **Literature Discovery**: Finding existing materials with relevant properties
2. **Knowledge Extraction**: Structuring material data for analysis
3. **Intelligent Search**: Adapting to user interests while maintaining focus
4. **Database Building**: Creating material candidates for further research

The ultimate goal is discovering materials that can maintain insulin stability at room temperature while enabling effective transdermal delivery.

## Files

- `literature_mining_system.py`: Main system with intelligent mining
- `interactive_mining.py`: **Interactive chat interface** - type your own questions!
- `demo_intelligent_mining.py`: Demonstrates intelligent request handling  
- `demo_literature_mining.py`: Basic mining demonstration
- `semantic_scholar_client.py`: Academic paper search client
- `ollama_client.py`: Local LLM integration
- `iterative_literature_mining.py`: Future active learning features (Milestone 4)

## License

Academic research use only. Commercial applications require additional licensing. 