#!/bin/bash

# ========================================================================
# Insulin AI - Dependency Installation Script
# Installs all required dependencies for the AI-driven material discovery platform
# ========================================================================

set -e  # Exit on any error

echo "🧬 Insulin AI - Installing Dependencies"
echo "======================================"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ Using mamba for conda package management"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✅ Using conda for package management"
else
    echo "❌ Error: conda or mamba not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "🐍 Detected Python version: $python_version"

if [[ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]]; then
    echo "❌ Error: Python 3.8+ required. Current version: $python_version"
    exit 1
fi

echo ""
echo "📦 Step 1: Installing conda dependencies (external packages)..."
echo "============================================================="

# Install conda dependencies that aren't available via pip
$CONDA_CMD install -c conda-forge -y \
    openmm \
    openmmforcefields \
    pdbfixer \
    rdkit \
    packmol \
    ambertools

echo ""
echo "📦 Step 2: Installing pip dependencies..."
echo "========================================"

# Upgrade pip first
python -m pip install --upgrade pip

# Install from requirements.txt
python -m pip install -r requirements.txt

echo ""
echo "🔧 Step 3: Verifying installation..."
echo "==================================="

# Test critical imports
python -c "
import sys
import traceback

packages_to_test = [
    ('OpenMM', 'openmm'),
    ('RDKit', 'rdkit'),
    ('OpenFF Toolkit', 'openff.toolkit'),
    ('OpenMM ForceFields', 'openmmforcefields'),
    ('PDBFixer', 'pdbfixer'),
    ('MDTraj', 'mdtraj'),
    ('LangChain', 'langchain'),
    ('OpenAI', 'openai'),
    ('Streamlit', 'streamlit'),
    ('NumPy', 'numpy'),
    ('Pandas', 'pandas'),
    ('Plotly', 'plotly'),
]

failed_packages = []
for name, module in packages_to_test:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        failed_packages.append(name)

if failed_packages:
    print(f'\n❌ Failed to import: {', '.join(failed_packages)}')
    sys.exit(1)
else:
    print('\n🎉 All critical packages imported successfully!')
"

echo ""
echo "🧪 Step 4: Testing OpenMM installation..."
echo "========================================"

python -c "
import openmm
import openmm.app
import openmm.unit
from openmmforcefields.generators import SystemGenerator
print('✅ OpenMM installation verified')
print(f'OpenMM version: {openmm.__version__}')

# Test platform availability
platforms = []
for i in range(openmm.Platform.getNumPlatforms()):
    platform = openmm.Platform.getPlatform(i)
    platforms.append(platform.getName())
print(f'Available platforms: {', '.join(platforms)}')
"

echo ""
echo "🔧 Step 5: Setting up environment..."
echo "==================================="

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# OpenAI API Key (required for LLM features)
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith (optional - for tracing)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Semantic Scholar API (optional - for literature mining)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here

# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
EOF
    echo "✅ Created .env file - please update with your API keys"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "Next steps:"
echo "1. Update your API keys in the .env file"
echo "2. Run the application with: streamlit run src/insulin_ai/app.py"
echo "3. Or use the CLI: python -m insulin_ai.cli"
echo ""
echo "📚 Documentation: README.md"
echo "🐛 Issues: Please report any problems on GitHub"
echo ""
echo "Happy molecular discovery! 🧬✨" 