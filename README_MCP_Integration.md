# MCP (Model Context Protocol) Integration for Insulin AI

## Overview

This project now includes **Model Context Protocol (MCP)** integration to enhance the literature mining capabilities with more sophisticated Semantic Scholar API interactions. The MCP server provides structured, domain-specific tools for insulin delivery materials research.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that enables AI applications to securely access external tools and data sources. In our case, it provides:

- **Structured API Access**: Clean, type-safe interfaces to Semantic Scholar
- **Domain-Specific Tools**: Custom tools optimized for insulin delivery research
- **Enhanced Data Processing**: Advanced filtering and analysis capabilities
- **Better Error Handling**: Robust connection management and fallback strategies

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Flask App     │───▶│   MCP Client     │───▶│   MCP Server        │
│   (Web Interface)│    │   (mcp_client.py)│    │   (semantic_scholar │
└─────────────────┘    └──────────────────┘    │   _server.py)       │
                                                └─────────────────────┘
                                                           │
                                                           ▼
                                                ┌─────────────────────┐
                                                │  Semantic Scholar   │
                                                │  API                │
                                                └─────────────────────┘
```

## Key Components

### 1. **MCP Server** (`semantic_scholar_server.py`)
- **Enhanced Search Tools**: `search_insulin_materials()` with domain-specific filtering
- **Material Analysis**: `analyze_material_properties()` for targeted extraction
- **Research Archival**: `save_research_findings()` for structured data storage
- **Paper Analysis**: Standard Semantic Scholar operations with enhanced error handling

### 2. **MCP Client** (`mcp_client.py`)
- **Async Operations**: Full async support for concurrent operations
- **Sync Wrapper**: `MCPLiteratureMinerSync` for Flask integration
- **Connection Management**: Automatic connection handling and cleanup
- **Error Recovery**: Robust error handling with fallback strategies

### 3. **Flask Integration** (`app.py`)
- **New Endpoint**: `/api/chat` with `type: 'mcp'` for MCP-powered mining
- **Status Monitoring**: Enhanced `/api/status` endpoint with MCP health checks
- **Example Prompts**: MCP-specific examples in `/api/examples`

## Features

### 🚀 **Enhanced Search Strategy**
- Multi-query approach with domain-specific terms
- Automatic material focus detection (polymers, hydrogels, nanoparticles, patches)
- Recent publication filtering (2020+)
- Relevance-based result filtering

### 🧪 **Material-Focused Analysis**
- Keyword-based material extraction
- Property categorization (thermal stability, biocompatibility)
- Confidence scoring
- Source paper tracking

### 🤖 **AI-Enhanced Insights**
- Local LLM integration for deeper analysis
- Research gap identification
- Material suitability assessment
- Contextual reasoning

### 💾 **Structured Data Storage**
- Automatic results archival in `mining_results/`
- Metadata tracking (timestamps, sources, methods)
- JSON format for easy processing
- Research session tracking

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The updated requirements include:
- `fastmcp>=0.1.0` - MCP server framework
- `semanticscholar>=0.4.0` - Enhanced Semantic Scholar client
- `mcp>=1.0.0` - MCP protocol support

### 2. Environment Variables
Add to your `.env` file (optional):
```bash
# Semantic Scholar API (optional, but recommended for higher rate limits)
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here

# Ollama Configuration
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
```

### 3. Test the Integration
```bash
python test_mcp_integration.py
```

This will run comprehensive tests to verify:
- Basic MCP functionality
- Ollama integration
- Async operations
- Error handling

## Usage

### Web Interface

1. **Start the Flask App**:
   ```bash
   python app.py
   ```

2. **Select MCP Mode**: In the web interface, choose "MCP Literature Mining" from the chat type selector

3. **Ask Research Questions**:
   ```
   "Find biocompatible polymers with thermal stability above 40°C"
   "Analyze recent advances in PLGA-based insulin delivery systems"
   "Search for nanoparticle formulations for protein preservation"
   ```

### Programmatic Usage

```python
from mcp_client import MCPLiteratureMinerSync
from ollama_client import OllamaClient

# Initialize
ollama_client = OllamaClient()
mcp_miner = MCPLiteratureMinerSync(ollama_client=ollama_client)

# Run enhanced literature mining
def progress_callback(message, step_type):
    print(f"[{step_type}] {message}")

results = mcp_miner.intelligent_mining_with_mcp(
    user_request="chitosan hydrogels for insulin stabilization",
    max_papers=20,
    recent_only=True,
    progress_callback=progress_callback
)

# Process results
for candidate in results['material_candidates']:
    print(f"Material: {candidate['materials']}")
    print(f"AI Insight: {candidate['ai_insight']}")
```

## API Reference

### MCP Tools

#### `search_insulin_materials(query, num_results, recent_only, material_focus)`
Enhanced search specifically for insulin delivery materials.

**Parameters:**
- `query` (str): Research question or material description
- `num_results` (int): Maximum papers to return (default: 20)
- `recent_only` (bool): Filter for recent papers (default: True)
- `material_focus` (str): Focus area - "polymers", "hydrogels", "nanoparticles", "patches"

#### `analyze_material_properties(papers, extraction_focus)`
Extract material properties from research papers.

**Parameters:**
- `papers` (List[Dict]): List of paper objects
- `extraction_focus` (str): Properties to focus on during extraction

#### `save_research_findings(findings, filename)`
Save research findings with metadata.

**Parameters:**
- `findings` (Dict): Research results
- `filename` (str, optional): Custom filename

### Client Methods

#### `MCPLiteratureMinerSync.intelligent_mining_with_mcp()`
Main method for MCP-enhanced literature mining.

**Returns:**
```python
{
    "success": True,
    "results": {...},
    "material_candidates": [...],
    "metadata": {
        "papers_analyzed": int,
        "material_focus": str,
        "mcp_enhanced": True
    }
}
```

## Benefits Over Standard Literature Mining

| Feature | Standard Mining | MCP-Enhanced Mining |
|---------|----------------|-------------------|
| **Search Strategy** | Single query | Multi-query with domain focus |
| **Material Detection** | Keyword-based | Category-aware with confidence |
| **Result Filtering** | Basic relevance | Advanced domain-specific filters |
| **Data Storage** | Simple files | Structured metadata + archival |
| **Error Handling** | Basic try/catch | Connection management + fallbacks |
| **Performance** | Sequential operations | Async with concurrent processing |
| **AI Integration** | Standard LLM calls | Enhanced with structured context |

## Monitoring & Debugging

### Status Endpoint
```bash
curl http://localhost:5000/api/status
```

Returns:
```json
{
  "literature_miner": true,
  "mcp_literature_miner": true,
  "chatbot": true,
  "mcp_status": "Available",
  "mcp_features": [
    "Enhanced Semantic Scholar integration",
    "Material-focused search strategies",
    "AI-enhanced material analysis",
    "Structured research findings storage"
  ]
}
```

### Log Monitoring
- MCP operations are logged to console with detailed progress
- Error messages include specific failure points
- Connection status is continuously monitored

### Troubleshooting

**MCP Server Issues:**
```bash
# Test direct server functionality
python semantic_scholar_server.py
```

**Dependency Issues:**
```bash
# Reinstall MCP dependencies
pip install --upgrade fastmcp semanticscholar mcp
```

**Connection Problems:**
- Check Semantic Scholar API rate limits
- Verify Ollama server is running
- Test with smaller paper counts first

## Future Enhancements

- **Real-time Streaming**: WebSocket support for live progress updates
- **Advanced Caching**: Redis integration for faster repeated queries
- **Enhanced AI Models**: Integration with specialized scientific LLMs
- **Collaborative Features**: Multi-user research session support
- **Export Formats**: PDF, CSV, and citation format exports

## Contributing

When extending the MCP integration:

1. **Add new tools** in `semantic_scholar_server.py`
2. **Update client** in `mcp_client.py` to handle new tools
3. **Test thoroughly** using `test_mcp_integration.py`
4. **Update documentation** with new features

## License

This MCP integration follows the same license as the main Insulin AI project. 