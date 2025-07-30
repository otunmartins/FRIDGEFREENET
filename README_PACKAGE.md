# Insulin-AI: AI-Powered Drug Delivery System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Package Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/insulin-ai/insulin-ai)

**Insulin-AI** is a comprehensive AI-powered platform for intelligent material discovery and optimization for insulin delivery patches. It combines state-of-the-art language models, molecular dynamics simulations, and advanced polymer chemistry to accelerate the development of next-generation drug delivery systems.

## 🎯 Features

### Core AI Systems
- **🤖 Advanced Chatbot**: Interactive AI assistant for drug delivery research
- **📚 Literature Mining**: Automated scientific paper analysis and insight extraction
- **🧬 PSMILES Generation**: AI-powered polymer notation generation with chemical validation
- **🔧 PSMILES Processing**: Comprehensive validation and correction of polymer structures

### Integration Capabilities
- **⚗️ Molecular Dynamics**: OpenMM-based simulation integration
- **🔄 Simulation Automation**: Automated pipeline for polymer structure prediction
- **📊 Comprehensive Analysis**: Advanced analysis and visualization tools
- **🛠️ Auto-Correction**: Intelligent PSMILES structure correction

### Web Interface
- **🌐 Streamlit Dashboard**: User-friendly web interface
- **📈 Interactive Visualizations**: Real-time plotting and analysis
- **🔬 Material Evaluation**: Comprehensive material property assessment
- **📖 Active Learning**: Iterative improvement through feedback loops

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when published)
pip install insulin-ai

# Or install from source
git clone https://github.com/insulin-ai/insulin-ai.git
cd insulin-ai
pip install -e .
```

### Basic Usage

```python
import insulin_ai

# Check installation
print(insulin_ai.get_package_info())

# Generate PSMILES for a material
from insulin_ai import PSMILESGenerator

generator = PSMILESGenerator(
    model_type='openai',
    openai_model='gpt-3.5-turbo'
)

result = generator.generate_truly_diverse_candidates(
    base_request="biodegradable polymer for insulin delivery",
    num_candidates=5
)

print(f"Generated PSMILES: {result['best_candidate']}")
```

### Command Line Interface

```bash
# Launch web interface
insulin-ai web

# Generate PSMILES
insulin-ai generate-psmiles -m "biodegradable polymer for insulin delivery"

# Mine literature
insulin-ai mine-literature -q "insulin delivery polymers" -m 10

# Test installation
insulin-ai test-installation
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+ (3.9-3.11 recommended)
- **Memory**: 4+ GB RAM (8+ GB for MD simulations)
- **Storage**: 2+ GB free space

### Required API Keys
- **OpenAI API Key**: Required for LLM functionality
- **Semantic Scholar API Key**: Optional, for enhanced literature mining

### Key Dependencies
- **OpenMM**: Molecular dynamics simulations
- **RDKit**: Chemical informatics
- **LangChain**: LLM integration
- **Streamlit**: Web interface
- **OpenFF Toolkit**: Force field parameterization

## 🏗️ Architecture

```
insulin-ai/
├── core/                   # Core AI systems
│   ├── chatbot_system.py
│   ├── literature_mining_system.py
│   ├── psmiles_generator.py
│   └── psmiles_processor.py
├── integration/            # External system integrations
│   ├── analysis/          # MD simulation integration
│   ├── automation/        # Simulation automation
│   └── corrections/       # Auto-correction systems
├── app/                   # Web application
│   ├── ui/               # UI components
│   └── utils/            # App utilities
├── utils/                 # Utility functions
└── cli.py                # Command-line interface
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-key"
export OPENMM_DEFAULT_PLATFORM="CUDA"  # or CPU, OpenCL
```

### Configuration File

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-openai-api-key-here
SEMANTIC_SCHOLAR_API_KEY=your-semantic-scholar-key
OPENMM_DEFAULT_PLATFORM=CUDA
```

## 📚 Documentation

### Quick Links
- **[Installation Guide](INSTALLATION.md)**: Detailed installation instructions
- **[API Documentation](#)**: Complete API reference
- **[Tutorials](#)**: Step-by-step tutorials
- **[Examples](#)**: Code examples and use cases

### Key Concepts

#### PSMILES (Polymer SMILES)
PSMILES is an extension of SMILES notation specifically designed for polymers. It uses `[*]` markers to indicate connection points between repeating units.

Example: `[*]C(C)C(=O)O[*]` represents a polymer backbone with specific functional groups.

#### Material Discovery Pipeline
1. **Literature Mining**: Extract insights from scientific papers
2. **PSMILES Generation**: AI-powered polymer structure generation
3. **Validation & Correction**: Chemical structure validation
4. **MD Simulation**: Molecular dynamics analysis
5. **Property Evaluation**: Material property assessment
6. **Optimization**: Iterative improvement through active learning

## 🧪 Examples

### Literature Mining

```python
from insulin_ai import MaterialsLiteratureMiner

miner = MaterialsLiteratureMiner(model_type="openai")
results = miner.search_and_analyze_papers(
    query="insulin delivery polymers",
    max_papers=20
)

print(f"Found {len(results['papers'])} relevant papers")
```

### PSMILES Processing

```python
from insulin_ai import PSMILESProcessor

processor = PSMILESProcessor()
result = processor.process_psmiles_workflow_with_autorepair(
    psmiles="[*]C(C)C(=O)O[*]",
    repair_enabled=True
)

print(f"Processed PSMILES: {result['final_psmiles']}")
print(f"Validation status: {result['validation_status']}")
```

### Molecular Dynamics Integration

```python
from insulin_ai.integration.analysis import SimpleMDIntegration

md_integration = SimpleMDIntegration()
simulation_result = md_integration.run_polymer_simulation(
    psmiles="[*]C(C)C(=O)O[*]",
    simulation_time_ns=1.0
)

print(f"Simulation completed: {simulation_result['success']}")
```

## 🚀 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/insulin-ai/insulin-ai.git
cd insulin-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .[dev,docs,analysis]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests

# Run with coverage
pytest --cov=insulin_ai
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](#) for details.

### Areas for Contribution
- **🧬 New polymer chemistry features**
- **⚗️ Additional simulation integrations**
- **📊 Enhanced analysis capabilities**
- **🌐 UI/UX improvements**
- **📚 Documentation and tutorials**
- **🧪 Test coverage expansion**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenMM Team**: For the excellent molecular dynamics framework
- **RDKit Community**: For chemical informatics tools
- **OpenFF Initiative**: For modern force field development
- **LangChain Team**: For LLM integration framework
- **Streamlit Team**: For the web application framework

## 📞 Support

- **📧 Email**: [contact@insulin-ai.org](#)
- **🐛 Issues**: [GitHub Issues](https://github.com/insulin-ai/insulin-ai/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/insulin-ai/insulin-ai/discussions)
- **📖 Documentation**: [Official Docs](#)

## 🗺️ Roadmap

### Version 0.2.0 (Next Release)
- [ ] Enhanced MD simulation capabilities
- [ ] Improved PSMILES validation
- [ ] Additional polymer chemistry features
- [ ] Performance optimizations

### Version 0.3.0
- [ ] Multi-objective optimization
- [ ] Advanced active learning algorithms
- [ ] Cloud deployment options
- [ ] API rate limiting and caching

### Long-term Goals
- [ ] Integration with experimental databases
- [ ] Real-time collaboration features
- [ ] Advanced machine learning models
- [ ] Commercial partnership integrations

---

**Made with ❤️ for the drug delivery research community** 