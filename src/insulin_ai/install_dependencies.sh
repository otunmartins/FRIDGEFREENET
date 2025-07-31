#!/bin/bash

echo "========================================================================="
echo "🧬 Installing OpenMM and OpenForceFields Dependencies for Ligand Simulations"
echo "========================================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"

# Update conda
echo ""
echo "📦 Updating conda..."
conda update -n base -c defaults conda -y

# Install OpenMM with force field support
echo ""
echo "🧬 Installing OpenMM..."
conda install -c conda-forge openmm -y

# Install OpenMM Force Fields package
echo ""
echo "⚗️ Installing OpenMM Force Fields..."
conda install -c conda-forge openmmforcefields -y

# Install OpenFF toolkit for small molecules
echo ""
echo "🧪 Installing OpenFF Toolkit..."
conda install -c conda-forge openff-toolkit -y

# Install RDKit for chemical informatics
echo ""
echo "🧬 Installing RDKit..."
conda install -c conda-forge rdkit -y

# Install additional molecular tools
echo ""
echo "🔬 Installing additional molecular tools..."
conda install -c conda-forge mdtraj -y
conda install -c conda-forge nglview -y
conda install -c conda-forge parmed -y

# Install data science packages
echo ""
echo "📊 Installing data science packages..."
conda install -c conda-forge numpy pandas matplotlib seaborn jupyter -y

# Install plotting and analysis tools
echo ""
echo "📈 Installing analysis tools..."
conda install -c conda-forge pymol-open-source -y || echo "⚠️ PyMOL installation failed (optional)"

# Verify installations
echo ""
echo "========================================================================="
echo "🔍 Verifying installations..."
echo "========================================================================="

python3 -c "
import sys
packages = [
    ('openmm', 'OpenMM'),
    ('openmmforcefields', 'OpenMM Force Fields'),
    ('openff.toolkit', 'OpenFF Toolkit'),
    ('rdkit', 'RDKit'),
    ('mdtraj', 'MDTraj'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib')
]

failed = []
for package, name in packages:
    try:
        __import__(package)
        print(f'✓ {name} installed successfully')
    except ImportError:
        print(f'❌ {name} installation failed')
        failed.append(name)

if failed:
    print(f'\n⚠️  Some packages failed to install: {failed}')
    sys.exit(1)
else:
    print('\n🎉 All packages installed successfully!')
"

echo ""
echo "========================================================================="
echo "🏁 Installation Complete!"
echo "========================================================================="
echo ""
echo "📝 You can now run OpenMM simulations with ligand-only systems using:"
echo "   • OpenMM - molecular dynamics engine"
echo "   • OpenMM Force Fields - automatic parameterization"
echo "   • OpenFF Toolkit - small molecule force fields"
echo "   • RDKit - chemical informatics"
echo ""
echo "🚀 Try running: python3 polymer_simulation.py"
echo "" 