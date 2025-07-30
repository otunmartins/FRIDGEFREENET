# Installation Guide for Insulin-AI

## Overview

Insulin-AI is a comprehensive AI-powered drug delivery system for insulin patch materials discovery. This guide will help you install and set up the package for development or production use.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4+ GB RAM (8+ GB recommended for MD simulations)
- **Storage**: 2+ GB free space

### External Dependencies

#### OpenMM (for Molecular Dynamics)
```bash
# Install via conda (recommended)
conda install -c conda-forge openmm

# Or via pip (less reliable for some platforms)
pip install openmm
```

#### RDKit (for Chemical Processing)
```bash
# Install via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit
```

## Installation Methods

### Method 1: Install from PyPI (Recommended for Users)

Once published, you can install directly from PyPI:

```bash
# Basic installation
pip install insulin-ai

# With all optional dependencies
pip install insulin-ai[all]

# With specific extras
pip install insulin-ai[dev,analysis]
```

### Method 2: Install from Source (For Development)

1. **Clone the repository:**
```bash
git clone https://github.com/insulin-ai/insulin-ai.git
cd insulin-ai
```

2. **Create and activate a virtual environment:**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n insulin-ai python=3.9
conda activate insulin-ai
```

3. **Install dependencies:**
```bash
# Install core dependencies
pip install -e .

# Or install with development dependencies
pip install -e .[dev]

# Or install everything
pip install -e .[all]
```

### Method 3: Development Installation with Conda

```bash
# Create conda environment with major dependencies
conda create -n insulin-ai python=3.9
conda activate insulin-ai

# Install scientific dependencies via conda
conda install -c conda-forge openmm rdkit numpy pandas matplotlib plotly

# Install the package in development mode
pip install -e .[dev]
```

## Dependency Groups

The package has several optional dependency groups:

- **`dev`**: Development tools (pytest, black, mypy, etc.)
- **`docs`**: Documentation building tools (sphinx, etc.)
- **`analysis`**: Enhanced analysis tools (jupyter, mdanalysis, etc.)
- **`gpu`**: GPU acceleration (cupy, pytorch)
- **`all`**: All optional dependencies

## Configuration

### API Keys

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-key"  # Optional
```

Or create a `.env` file in your project directory:
```
OPENAI_API_KEY=your-openai-api-key-here
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key
```

### OpenMM Configuration (Optional)

```bash
# Set default OpenMM platform
export OPENMM_DEFAULT_PLATFORM=CUDA  # or CPU, OpenCL
```

## Testing Installation

### Quick Test
```bash
# Test basic installation
insulin-ai test-installation

# Get package info
insulin-ai info
```

### Comprehensive Test
```bash
# Run test suite (if installed with dev dependencies)
pytest

# Test core functionality
python -c "import insulin_ai; print(insulin_ai.get_package_info())"
```

## Usage

### Command Line Interface

```bash
# Get help
insulin-ai --help

# Launch web interface
insulin-ai web

# Generate PSMILES
insulin-ai generate-psmiles -m "biodegradable polymer for insulin delivery"

# Mine literature
insulin-ai mine-literature -q "insulin delivery polymers" -m 10
```

### Python API

```python
import insulin_ai

# Check what's available
print(insulin_ai.get_package_info())

# Use core components
from insulin_ai import PSMILESGenerator, MaterialsLiteratureMiner

# Initialize systems
generator = PSMILESGenerator(model_type='openai', openai_model='gpt-3.5-turbo')
miner = MaterialsLiteratureMiner(model_type='openai')

# Generate PSMILES
result = generator.generate_truly_diverse_candidates(
    base_request="biodegradable polymer for insulin delivery",
    num_candidates=5
)
```

### Web Application

```bash
# Launch the Streamlit web interface
insulin-ai web --host localhost --port 8501
```

Then open your browser to `http://localhost:8501`

## Troubleshooting

### Common Issues

#### OpenMM Installation Issues
```bash
# Try installing via conda-forge
conda install -c conda-forge openmm

# Verify installation
python -c "import openmm; print(openmm.Platform.getNumPlatforms())"
```

#### RDKit Installation Issues
```bash
# Install via conda (most reliable)
conda install -c conda-forge rdkit

# Verify installation
python -c "from rdkit import Chem; print('RDKit working')"
```

#### Import Errors
```bash
# Check if package is properly installed
pip list | grep insulin-ai

# Reinstall in development mode
pip install -e .
```

#### Permission Errors
```bash
# Use user installation
pip install --user insulin-ai

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install insulin-ai
```

### Getting Help

1. **Check the documentation**: [Documentation URL]
2. **Run diagnostics**: `insulin-ai test-installation`
3. **Check GitHub issues**: [GitHub Issues URL]
4. **Contact support**: [Support Email]

## Development Setup

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/insulin-ai/insulin-ai.git
cd insulin-ai

# Install in development mode with all dependencies
pip install -e .[dev,docs,analysis]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Build documentation
cd docs/
make html
```

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM functionality | Yes | None |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key | No | None |
| `OPENMM_DEFAULT_PLATFORM` | Default OpenMM platform | No | CPU |
| `RDKIT_ERROR_LOGGING` | RDKit logging level | No | ERROR |
| `INSULIN_AI_CONFIG_DIR` | Custom config directory | No | Package default |

## Next Steps

After installation:

1. **Set up API keys** (especially OpenAI)
2. **Run the test installation** to verify everything works
3. **Try the web interface** with `insulin-ai web`
4. **Explore the examples** in the documentation
5. **Check out the tutorials** for specific use cases

For more detailed usage instructions, see the main README.md and documentation. 