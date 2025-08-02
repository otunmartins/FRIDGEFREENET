# Installation Guide

## Quick Start

### Option 1: Automated Installation (Recommended)

Run the automated installation script:

```bash
./install_dependencies.sh
```

This script will:
- Check your Python version (3.8+ required)
- Install conda/mamba dependencies (OpenMM, RDKit, PACKMOL, etc.)
- Install pip dependencies from requirements.txt
- Verify all installations
- Set up your environment file

### Option 2: Manual Installation

If you prefer manual installation:

1. **Install conda dependencies:**
```bash
conda install -c conda-forge openmm openmmforcefields pdbfixer rdkit packmol ambertools
```

2. **Install pip dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment:**
```bash
cp .env.example .env  # Edit with your API keys
```

## Requirements Overview

The new consolidated `requirements.txt` includes all dependencies for:

- **Core Framework**: Streamlit, Flask
- **LLM Integration**: LangChain, OpenAI, LangGraph
- **Vector Databases**: ChromaDB, FAISS
- **Scientific Computing**: NumPy, Pandas, SciPy, scikit-learn
- **Molecular Simulation**: OpenMM, OpenMM ForceFields, OpenFF Toolkit, MDTraj
- **Chemistry**: RDKit, PDBFixer, SELFIES
- **Visualization**: Plotly, Matplotlib, py3Dmol
- **APIs**: Semantic Scholar, PubChem
- **Development**: pytest, python-dotenv

## External Dependencies

Some dependencies must be installed via conda (not pip):
- **OpenMM**: Molecular dynamics engine
- **RDKit**: Chemistry toolkit
- **PACKMOL**: Molecular packing tool
- **AmberTools**: Force field tools

## Troubleshooting

### Common Issues

1. **OpenMM not found**: Install via conda, not pip
2. **RDKit import error**: Use conda-forge channel
3. **PACKMOL missing**: Install via conda-forge
4. **Platform errors**: Check OpenMM platform availability

### Testing Installation

Run the verification script:
```python
python -c "
import openmm
import rdkit
import langchain
import streamlit
print('✅ All critical packages imported successfully!')
"
```

### Getting Help

If you encounter issues:
1. Check that you're using Python 3.8+
2. Ensure conda/mamba is installed
3. Try the automated installer first
4. Check the error logs for specific missing packages
5. Open a GitHub issue with error details

## Environment Setup

Create a `.env` file with your API keys:

```bash
# Required for LLM features
OPENAI_API_KEY=your_api_key_here

# Optional features
LANGSMITH_API_KEY=your_langsmith_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

## System Requirements

- **Python**: 3.8+
- **OS**: Linux, macOS, Windows (with conda)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ free space
- **GPU**: Optional (CUDA for faster simulations)

## Next Steps

After installation:
1. Update your `.env` file with API keys
2. Run the application: `streamlit run src/insulin_ai/app.py`
3. Check the documentation for usage examples 