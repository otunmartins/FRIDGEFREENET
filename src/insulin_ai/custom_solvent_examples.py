#!/usr/bin/env python3
"""
Custom Solvent Examples for OpenMM Simulations
==============================================

This script demonstrates how to use ANY molecule as a "solvent" in OpenMM simulations.
The key insight is that in OpenMM, "solvent" is just any molecule that surrounds your
main molecule of interest.

Examples include:
- Traditional solvents (water, ethanol, DMSO)
- Organic solvents (hexane, toluene, chloroform)
- Ionic liquids
- Polymer matrices
- Drug delivery vehicles
- Biological environments (lipids, sugars)
- Mixed solvent systems
"""

import sys
from polymer_simulation_verbose import run_verbose_simulation, setup_logging

def print_example_header(title: str, description: str):
    """Print formatted example header."""
    print(f"\n{'='*80}")
    print(f"🧪 {title}")
    print(f"{'='*80}")
    print(f"Description: {description}")
    print("-" * 80)

def example_traditional_solvents():
    """Examples of traditional laboratory solvents."""
    logger = setup_logging()
    
    print_example_header(
        "TRADITIONAL LABORATORY SOLVENTS",
        "Common solvents used in chemistry and biochemistry"
    )
    
    examples = [
        {
            'name': 'Water',
            'smiles': 'O',
            'description': 'Standard aqueous environment, most biological systems',
            'count': 200
        },
        {
            'name': 'Ethanol',
            'smiles': 'CCO',
            'description': 'Polar protic solvent, used in pharmaceutical formulations',
            'count': 150
        },
        {
            'name': 'Methanol',
            'smiles': 'CO',
            'description': 'Highly polar solvent, common in analytical chemistry',
            'count': 180
        },
        {
            'name': 'DMSO (Dimethyl sulfoxide)',
            'smiles': 'CS(=O)C',
            'description': 'Aprotic polar solvent, penetrates biological membranes',
            'count': 100
        },
        {
            'name': 'Acetonitrile',
            'smiles': 'CC#N',
            'description': 'Polar aprotic solvent, used in HPLC and synthesis',
            'count': 120
        },
        {
            'name': 'Hexane',
            'smiles': 'CCCCCC',
            'description': 'Nonpolar solvent, models hydrophobic environments',
            'count': 80
        }
    ]
    
    for example in examples:
        print(f"\n🧪 {example['name']} ({example['smiles']})")
        print(f"   💡 {example['description']}")
        print(f"   🔢 Suggested count: {example['count']} molecules")
        
        # Create solvent specification
        solvent_spec = [{
            'smiles': example['smiles'],
            'count': example['count'],
            'name': example['name'].lower()
        }]
        
        print(f"   📝 Usage example:")
        print(f"      custom_solvents = {solvent_spec}")

def example_biological_environments():
    """Examples of biological environment molecules."""
    
    print_example_header(
        "BIOLOGICAL ENVIRONMENT MOLECULES",
        "Molecules that create biological or biomimetic environments"
    )
    
    examples = [
        {
            'name': 'Glucose',
            'smiles': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O',
            'description': 'Blood sugar, creates physiological environment',
            'count': 50
        },
        {
            'name': 'Urea',
            'smiles': 'C(=O)(N)N',
            'description': 'Protein denaturant, models cellular stress conditions',
            'count': 100
        },
        {
            'name': 'Glycerol',
            'smiles': 'C(C(CO)O)O',
            'description': 'Cryoprotectant, viscous biological medium',
            'count': 80
        },
        {
            'name': 'Trehalose',
            'smiles': 'C([C@@H]1[C@@H]([C@H]([C@@H]([C@H](O1)O[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)O)O)O)O)O)O',
            'description': 'Disaccharide, stabilizes proteins and membranes',
            'count': 30
        },
        {
            'name': 'Choline chloride',
            'smiles': 'C[N+](C)(C)CCO',
            'description': 'Component of ionic liquids and deep eutectic solvents',
            'count': 60
        }
    ]
    
    for example in examples:
        print(f"\n🧬 {example['name']}")
        print(f"   💡 {example['description']}")
        print(f"   🔢 Suggested count: {example['count']} molecules")
        print(f"   🧪 SMILES: {example['smiles']}")

def example_drug_delivery_systems():
    """Examples of drug delivery and pharmaceutical vehicles."""
    
    print_example_header(
        "DRUG DELIVERY SYSTEMS",
        "Molecules used in pharmaceutical formulations and drug delivery"
    )
    
    examples = [
        {
            'name': 'PEG (Polyethylene glycol monomer)',
            'smiles': 'CCOCCO',
            'description': 'Biocompatible polymer, used in drug conjugation',
            'count': 100
        },
        {
            'name': 'Propylene glycol',
            'smiles': 'CC(CO)O',
            'description': 'Pharmaceutical excipient, enhances drug solubility',
            'count': 120
        },
        {
            'name': 'Benzyl alcohol',
            'smiles': 'C1=CC=C(C=C1)CO',
            'description': 'Preservative in injectable medications',
            'count': 50
        },
        {
            'name': 'Cremophor (simplified)',
            'smiles': 'CCCCCCCCCCCCOCCOCCOCCOCCOCCOCCOCCOCCOC',
            'description': 'Surfactant for hydrophobic drug solubilization',
            'count': 20
        }
    ]
    
    for example in examples:
        print(f"\n💊 {example['name']}")
        print(f"   💡 {example['description']}")
        print(f"   🔢 Suggested count: {example['count']} molecules")

def example_mixed_solvent_systems():
    """Examples of mixed solvent systems."""
    
    print_example_header(
        "MIXED SOLVENT SYSTEMS",
        "Combinations of multiple solvents for complex environments"
    )
    
    mixed_systems = [
        {
            'name': 'Ethanol-Water Mixture (70:30)',
            'description': 'Common disinfectant concentration, protein denaturation studies',
            'solvents': [
                {'smiles': 'CCO', 'count': 140, 'name': 'ethanol'},
                {'smiles': 'O', 'count': 60, 'name': 'water'}
            ]
        },
        {
            'name': 'DMSO-Water Mixture',
            'description': 'Cryoprotectant solution, pharmaceutical solvent',
            'solvents': [
                {'smiles': 'CS(=O)C', 'count': 50, 'name': 'DMSO'},
                {'smiles': 'O', 'count': 150, 'name': 'water'}
            ]
        },
        {
            'name': 'Oil-Water Emulsion',
            'description': 'Models biological membranes and drug delivery systems',
            'solvents': [
                {'smiles': 'CCCCCCCCCCCCCCCC', 'count': 30, 'name': 'hexadecane'},
                {'smiles': 'O', 'count': 170, 'name': 'water'}
            ]
        },
        {
            'name': 'Deep Eutectic Solvent',
            'description': 'Green chemistry solvent, protein stabilization',
            'solvents': [
                {'smiles': 'C[N+](C)(C)CCO', 'count': 40, 'name': 'choline'},
                {'smiles': 'C(C(CO)O)O', 'count': 80, 'name': 'glycerol'},
                {'smiles': 'O', 'count': 80, 'name': 'water'}
            ]
        }
    ]
    
    for system in mixed_systems:
        print(f"\n🌊 {system['name']}")
        print(f"   💡 {system['description']}")
        print(f"   🧪 Components:")
        for solvent in system['solvents']:
            print(f"      - {solvent['name']}: {solvent['smiles']} (x{solvent['count']})")
        
        print(f"   📝 Usage example:")
        print(f"      custom_solvents = {system['solvents']}")

def example_specialized_environments():
    """Examples of specialized chemical environments."""
    
    print_example_header(
        "SPECIALIZED CHEMICAL ENVIRONMENTS",
        "Unique environments for specific research applications"
    )
    
    examples = [
        {
            'name': 'Ionic Liquid Environment',
            'description': 'Room temperature ionic liquids for green chemistry',
            'solvents': [
                {'smiles': 'CCN1C=C[N+](=C1)C', 'count': 60, 'name': 'EMIm_cation'},
                {'smiles': '[Cl-]', 'count': 60, 'name': 'chloride'}
            ]
        },
        {
            'name': 'Lipid Membrane Mimic',
            'description': 'Simplified model of cell membrane environment',
            'solvents': [
                {'smiles': 'CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC', 'count': 20, 'name': 'DPPC'},
                {'smiles': 'O', 'count': 180, 'name': 'water'}
            ]
        },
        {
            'name': 'Hydrogel Environment',
            'description': 'Biomaterial environment for tissue engineering',
            'solvents': [
                {'smiles': 'CC(=O)NC(CO)C(=O)O', 'count': 40, 'name': 'N_acetyl_serine'},
                {'smiles': 'O', 'count': 160, 'name': 'water'}
            ]
        }
    ]
    
    for example in examples:
        print(f"\n⚗️ {example['name']}")
        print(f"   💡 {example['description']}")
        print(f"   🧪 Components:")
        for solvent in example['solvents']:
            print(f"      - {solvent['name']}: {solvent['smiles']} (x{solvent['count']})")

def run_sample_simulation():
    """Run a quick sample simulation to demonstrate the functionality."""
    
    print_example_header(
        "SAMPLE SIMULATION DEMONSTRATION",
        "Running a quick test with ethanol solvent to show the system works"
    )
    
    logger = setup_logging()
    
    # Define a simple ethanol solvent system
    ethanol_system = [
        {'smiles': 'CCO', 'count': 50, 'name': 'ethanol'}
    ]
    
    print("🚀 Running sample simulation with ethanol solvent...")
    print(f"   📋 Solvent system: {ethanol_system}")
    print("   ⏱️ Duration: 0.01 ns (10 ps) for quick demonstration")
    
    try:
        result = run_verbose_simulation(
            "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb",
            custom_solvents=ethanol_system,
            simulation_time=0.01,  # Very short for demo
            temperature=300.0,
            logger=logger
        )
        
        print("\n✅ Sample simulation completed successfully!")
        print(f"   📁 Files created: {list(result.keys())}")
        print("   💡 You can now use this approach for any solvent system!")
        
    except Exception as e:
        print(f"\n❌ Sample simulation failed: {e}")
        print("   💡 This might be due to missing dependencies or file paths")
        print("   🔧 Check that all required packages are installed")

def main():
    """Main function to display all examples."""
    print("🧪 CUSTOM SOLVENT EXAMPLES FOR OPENMM SIMULATIONS")
    print("=" * 80)
    print("This script shows you how to use ANY molecule as a 'solvent' in OpenMM.")
    print("Key insight: Any molecule can surround your main molecule of interest!")
    print("=" * 80)
    
    # Display all example categories
    example_traditional_solvents()
    example_biological_environments()
    example_drug_delivery_systems()
    example_mixed_solvent_systems()
    example_specialized_environments()
    
    # Ask user if they want to run a demo
    print("\n" + "="*80)
    user_input = input("Would you like to run a quick demonstration simulation? (y/n): ")
    
    if user_input.lower().startswith('y'):
        run_sample_simulation()
    else:
        print("👍 Examples displayed. Use the patterns above to create your own custom solvent systems!")
    
    print("\n🎯 HOW TO USE:")
    print("1. Choose your solvent molecule(s) and their SMILES strings")
    print("2. Specify how many molecules of each type you want")
    print("3. Create a list of dictionaries with 'smiles', 'count', and 'name'")
    print("4. Pass this list to the run_verbose_simulation() function")
    print("5. The script will automatically create the mixed system and run the simulation!")

if __name__ == "__main__":
    main() 