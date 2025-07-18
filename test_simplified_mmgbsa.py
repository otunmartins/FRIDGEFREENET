#!/usr/bin/env python3
"""
Test script for the simplified MM-GBSA calculator using real simulation data
"""

from simplified_mmgbsa_calculator import SimplifiedMMGBSACalculator
import traceback

def test_real_simulation():
    """Test the simplified calculator with real simulation data"""
    
    print("🧪 Testing Simplified MM-GBSA Calculator with Real Data")
    print("=" * 60)
    
    try:
        # Initialize calculator
        calculator = SimplifiedMMGBSACalculator(output_dir="test_simplified_output")
        
        # Test with real simulation data
        frames_file = "integrated_md_simulations/sim_89362bdb/production/frames.pdb"
        simulation_id = "sim_89362bdb"
        
        print(f"📽️ Testing with: {frames_file}")
        print(f"🆔 Simulation ID: {simulation_id}")
        
        # Run the calculation
        results = calculator.calculate_binding_energy(
            frames_pdb_file=frames_file,
            simulation_id=simulation_id
        )
        
        if results['success']:
            print("\n✅ MM-GBSA calculation completed successfully!")
            print(f"🔋 Binding energy: {results['average_binding_energy']:.2f} ± {results['binding_energy_std']:.2f} kcal/mol")
            print(f"🧮 Entropy correction: {results['entropy_correction']:.2f} kcal/mol")
            print(f"🎯 Corrected binding energy: {results['corrected_binding_energy']:.2f} kcal/mol")
            print(f"📊 Frames processed: {results['n_frames']}")
            
            return True
        else:
            print(f"❌ MM-GBSA calculation failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_simulation()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}") 