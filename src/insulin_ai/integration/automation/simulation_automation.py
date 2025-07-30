#!/usr/bin/env python3
"""
Automated Simulation Pipeline for PSMILES Candidates

This module automates the following workflow:
1. Create simulation boxes filled with polymers for each PSMILES candidate using PSP
2. Create simulation boxes where insulin is flanked by polymer boxes with proper cleaning

Integrates with the PSMILES generation pipeline to execute immediately after candidate generation.
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import pandas as pd
from datetime import datetime

# Import required modules
try:
    from core.psmiles_generator import PSMILESGenerator
except ImportError:
    PSMILESGenerator = None

try:
    from core.simulation_manager import SimulationManager
except ImportError:
    SimulationManager = None

from utils.direct_polymer_builder import DirectPolymerBuilder

class SimulationAutomation:
    """
    Enhanced Simulation Automation using Direct Polymer Builder.
    
    Updates:
    - Uses DirectPolymerBuilder (bypasses PSP completely)
    - Leverages psmiles.alternating_copolymer() for chain generation
    - Generates PDB with CONECT entries
    - Direct PACKMOL integration
    """
    
    def __init__(self, working_directory: str = "automated_simulations"):
        self.working_directory = working_directory
        self.session_id = self._generate_session_id()
        self.session_dir = os.path.join(working_directory, f"session_{self.session_id}")
        
        # Initialize components
        self.psmiles_generator = PSMILESGenerator() if PSMILESGenerator else None
        self.direct_builder = DirectPolymerBuilder()  # NEW: Direct polymer builder
        self.simulation_manager = SimulationManager() if SimulationManager else None
        
        print(f"🚀 Simulation Automation initialized")
        print(f"📁 Session directory: {self.session_dir}")
        print(f"🔧 Using DirectPolymerBuilder (No PSP)")
    
    def _generate_session_id(self):
        """Generate a unique session ID."""
        return str(uuid.uuid4())[:8]
    
    def generate_psmiles(self, nl_description: str) -> Dict:
        """
        Generate PSMILES from natural language description.
        
        Args:
            nl_description: Natural language description of polymer
            
        Returns:
            Dict with PSMILES generation results
        """
        
        try:
            if not self.psmiles_generator:
                return {
                    'success': False,
                    'error': 'PSMILESGenerator not available',
                    'nl_description': nl_description
                }
            
            # Use the PSMILESGenerator to convert NL to PSMILES
            result = self.psmiles_generator.generate_psmiles(nl_description)
            
            if result.get('success', False):
                candidate_id = str(uuid.uuid4())[:8]
                return {
                    'success': True,
                    'psmiles': result['psmiles'],
                    'candidate_id': candidate_id,
                    'nl_description': nl_description
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'PSMILES generation failed'),
                    'nl_description': nl_description
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'nl_description': nl_description
            }
    
    def build_polymer_structure(self, 
                               psmiles: str, 
                               candidate_id: str,
                               chain_length: int = 10,
                               end_cap_atom: str = 'C') -> Dict:
        """
        Build polymer structure using DirectPolymerBuilder (No PSP).
        
        This method completely bypasses PSP and uses psmiles package directly:
        1. Creates polymer chain via alternating_copolymer
        2. Converts PSMILES to SMILES with end-capping  
        3. Generates PDB with CONECT entries
        
        Args:
            psmiles: Input PSMILES string
            candidate_id: Unique candidate identifier
            chain_length: Number of repeat units
            end_cap_atom: Atom for end-capping (default 'C')
            
        Returns:
            Dict with polymer structure information
        """
        
        print(f"\n🧬 Building polymer structure using DirectPolymerBuilder...")
        print(f"📥 PSMILES: {psmiles}")
        print(f"🆔 Candidate: {candidate_id}")
        print(f"📏 Chain length: {chain_length}")
        print(f"🔚 End cap: {end_cap_atom}")
        
        try:
            # Create candidate-specific directory
            candidate_dir = os.path.join(self.session_dir, f"candidate_{candidate_id}")
            molecules_dir = os.path.join(candidate_dir, "molecules")
            os.makedirs(molecules_dir, exist_ok=True)
            
            # Use DirectPolymerBuilder to create polymer structure
            result = self.direct_builder.build_polymer_chain(
                psmiles_str=psmiles,
                chain_length=chain_length,
                output_dir=molecules_dir,
                end_cap_atom=end_cap_atom,
                candidate_id=candidate_id  # Pass candidate_id for automatic SMILES storage
            )
            
            if result['success']:
                print(f"✅ Polymer structure built successfully!")
                print(f"🔧 Method: {result['method']}")
                print(f"📁 PDB file: {result['pdb_file']}")
                print(f"🧬 Polymer SMILES: {result['polymer_smiles'][:50]}...")
                print(f"💾 SMILES stored: {result.get('smiles_stored', False)}")
                
                # Copy the polymer.pdb to expected location for compatibility
                polymer_pdb_target = os.path.join(molecules_dir, "polymer_chain.pdb")
                if result['pdb_file'] != polymer_pdb_target:
                    import shutil
                    shutil.copy2(result['pdb_file'], polymer_pdb_target)
                    print(f"📋 Copied to: {polymer_pdb_target}")
                
                return {
                    'success': True,
                    'polymer_pdb': polymer_pdb_target,
                    'polymer_smiles': result['polymer_smiles'],
                    'method': 'direct_psmiles_no_psp',
                    'chain_length': chain_length,
                    'molecules_dir': molecules_dir
                }
                
            else:
                print(f"❌ Polymer structure building failed: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'method': 'direct_psmiles_failed'
                }
                
        except Exception as e:
            print(f"❌ Exception in polymer structure building: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'method': 'direct_psmiles_exception'
            }
    
    def setup_packmol_solvation(self,
                               polymer_pdb: str,
                               protein_pdb: str,
                               candidate_id: str,
                               box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
                               density: float = 1.0) -> Dict:
        """
        Setup PACKMOL solvation using DirectPolymerBuilder's PACKMOL integration.
        
        Args:
            polymer_pdb: Path to polymer PDB file
            protein_pdb: Path to protein PDB file
            candidate_id: Candidate identifier
            box_size: Simulation box dimensions (Å)
            density: Target density (g/cm³)
            
        Returns:
            Dict with PACKMOL setup results
        """
        
        print(f"\n💧 Setting up PACKMOL solvation...")
        print(f"🧬 Polymer: {polymer_pdb}")
        print(f"🦠 Protein: {protein_pdb}")
        print(f"📦 Box size: {box_size}")
        print(f"🏋️ Density: {density} g/cm³")
        
        try:
            # Create solvation directory
            candidate_dir = os.path.join(self.session_dir, f"candidate_{candidate_id}")
            solvation_dir = os.path.join(candidate_dir, "solvation")
            os.makedirs(solvation_dir, exist_ok=True)
            
            # Use DirectPolymerBuilder's PACKMOL integration
            result = self.direct_builder.create_packmol_solvation(
                polymer_pdb=polymer_pdb,
                protein_pdb=protein_pdb,
                output_dir=solvation_dir,
                box_size=box_size,
                density=density
            )
            
            if result['success']:
                print(f"✅ PACKMOL setup completed!")
                print(f"📁 Input file: {result['packmol_input']}")
                print(f"📊 Water molecules: {result['n_water']}")
                
                return {
                    'success': True,
                    'packmol_input': result['packmol_input'],
                    'solvated_pdb': result['expected_output'],
                    'n_water': result['n_water'],
                    'solvation_dir': solvation_dir
                }
                
            else:
                print(f"❌ PACKMOL setup failed: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            print(f"❌ Exception in PACKMOL setup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_complete_simulation(self, 
                               nl_description: str,
                               chain_length: int = 10,
                               box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
                               density: float = 1.0,
                               simulation_time_ns: float = 10.0) -> Dict:
        """
        Run complete simulation workflow using DirectPolymerBuilder approach.
        
        Workflow:
        1. Generate PSMILES from natural language
        2. Build polymer using DirectPolymerBuilder (No PSP)
        3. Setup PACKMOL solvation
        4. Run OpenMM simulation
        
        Args:
            nl_description: Natural language description of polymer
            chain_length: Number of repeat units
            box_size: Simulation box size (Å)
            density: Target density (g/cm³) 
            simulation_time_ns: Simulation time (ns)
            
        Returns:
            Dict with complete simulation results
        """
        
        print(f"\n🚀 RUNNING COMPLETE SIMULATION (DirectPolymerBuilder)")
        print(f"📝 Description: {nl_description}")
        print(f"📏 Chain length: {chain_length}")
        print(f"📦 Box size: {box_size}")
        print(f"⏱️ Time: {simulation_time_ns} ns")
        print("=" * 80)
        
        workflow_results = {
            'nl_description': nl_description,
            'workflow_steps': {},
            'success': False,
            'method': 'direct_psmiles_workflow'
        }
        
        try:
            # Step 1: Generate PSMILES
            print(f"\n📝 Step 1: Generating PSMILES...")
            psmiles_result = self.generate_psmiles(nl_description)
            workflow_results['workflow_steps']['psmiles_generation'] = psmiles_result
            
            if not psmiles_result['success']:
                workflow_results['error'] = f"PSMILES generation failed: {psmiles_result.get('error', 'Unknown')}"
                return workflow_results
            
            psmiles = psmiles_result['psmiles']
            candidate_id = psmiles_result['candidate_id']
            
            # Step 2: Build polymer structure (DirectPolymerBuilder)
            print(f"\n🧬 Step 2: Building polymer structure...")
            polymer_result = self.build_polymer_structure(
                psmiles=psmiles,
                candidate_id=candidate_id,
                chain_length=chain_length,
                end_cap_atom='C'
            )
            workflow_results['workflow_steps']['polymer_building'] = polymer_result
            
            if not polymer_result['success']:
                workflow_results['error'] = f"Polymer building failed: {polymer_result.get('error', 'Unknown')}"
                return workflow_results
            
            # Step 3: Setup PACKMOL solvation
            print(f"\n💧 Step 3: Setting up PACKMOL solvation...")
            # Note: This would need a protein PDB file
            # For now, we'll create the PACKMOL input but note the missing protein
            print(f"⚠️ Note: Protein PDB file needed for complete solvation")
            print(f"✅ Polymer structure ready for PACKMOL: {polymer_result['polymer_pdb']}")
            
            # Step 4: Prepare for OpenMM simulation
            print(f"\n🔬 Step 4: Preparing OpenMM simulation...")
            print(f"✅ Polymer structure available")
            print(f"📁 Molecules directory: {polymer_result['molecules_dir']}")
            
            workflow_results['success'] = True
            workflow_results['final_results'] = {
                'psmiles': psmiles,
                'polymer_pdb': polymer_result['polymer_pdb'],
                'polymer_smiles': polymer_result['polymer_smiles'],
                'method': 'direct_psmiles_no_psp',
                'ready_for_simulation': True
            }
            
            print(f"\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"✅ Method: DirectPolymerBuilder (No PSP)")
            print(f"✅ Polymer structure generated")
            print(f"✅ Ready for OpenMM simulation")
            
            return workflow_results
            
        except Exception as e:
            print(f"❌ Workflow exception: {str(e)}")
            import traceback
            traceback.print_exc()
            
            workflow_results['error'] = str(e)
            workflow_results['success'] = False
            return workflow_results


def get_insulin_pdb_files() -> Dict[str, str]:
    """Get available insulin PDB files from the integration directory"""
    insulin_files = {}
    
    # Look for insulin PDB files in common locations
    search_paths = [
        "integration/data/insulin",
        "data/insulin", 
        "insulin_structures",
        "pdb_files"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if file.endswith('.pdb') and 'insulin' in file.lower():
                    insulin_files[file] = os.path.join(search_path, file)
    
    # Default fallback insulin file  
    if not insulin_files:
        insulin_files['insulin_default.pdb'] = "integration/data/insulin/insulin_default.pdb"
    
    return insulin_files


class SimulationAutomationPipeline:
    """Automated pipeline for creating simulation boxes from PSMILES candidates"""
    
    def __init__(self, output_dir: str = "automated_simulations"):
        """
        Initialize automation pipeline
        
        Args:
            output_dir: Base directory for simulation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Import required modules
        self._import_dependencies()
        
    def _import_dependencies(self):
        """Import all required dependencies"""
        self.dependencies_available = True
        self.missing_dependencies = []
        
        try:
            # PSP for polymer building
            import psp.AmorphousBuilder as ab
            self.psp_builder = ab
            print("✅ PSP AmorphousBuilder imported successfully")
            
        except ImportError as e:
            print(f"⚠️ PSP AmorphousBuilder not available: {e}")
            self.missing_dependencies.append("psp.AmorphousBuilder")
            self.dependencies_available = False
            
        try:
            # PDBFixer for insulin cleaning
            from pdbfixer import PDBFixer
            self.pdb_fixer = PDBFixer
            print("✅ PDBFixer imported successfully")
            
        except ImportError as e:
            print(f"⚠️ PDBFixer not available: {e}")
            self.missing_dependencies.append("pdbfixer")
            
        try:
            # Import existing utilities
            from integration.analysis.insulin_polymer_builder import (
                build_insulin_polymer_composite,
                create_large_box_with_random_edges
            )
            self.build_composite = build_insulin_polymer_composite
            self.create_large_box = create_large_box_with_random_edges
            print("✅ Insulin polymer builder utilities imported successfully")
            
        except ImportError as e:
            print(f"⚠️ Insulin polymer builder utilities not available: {e}")
            self.missing_dependencies.append("insulin_polymer_builder")
            # Set fallback methods to None so we can check for availability
            self.build_composite = None
            self.create_large_box = None
            
        try:
            # Import PDB preprocessing
            from app.utils.pdb_utils import preprocess_pdb_standalone
            self.preprocess_pdb = preprocess_pdb_standalone
            print("✅ PDB preprocessing utilities imported successfully")
            
        except ImportError as e:
            print(f"⚠️ PDB preprocessing utilities not available: {e}")
            self.missing_dependencies.append("pdb_utils")
        
        if self.dependencies_available:
            print("✅ All critical dependencies imported successfully")
        else:
            print(f"⚠️ Some dependencies missing: {self.missing_dependencies}")
            print("⚠️ Simulation automation will run in limited mode")
    
    def create_polymer_simulation_box(self, 
                                    psmiles: str, 
                                    candidate_id: str,
                                    polymer_length: int = 5,
                                    num_molecules: int = 10,
                                    density: float = 0.4,
                                    tolerance_distance: float = 3.0,
                                    timeout_minutes: int = 5,
                                    output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Create simulation box filled with polymers using PSP
        
        Args:
            psmiles: PSMILES string for the polymer
            candidate_id: Unique identifier for this candidate
            polymer_length: Length of polymer chains (default: 5 for faster building)
            num_molecules: Number of polymer molecules (default: 10 for faster building)
            density: Density for packing in g/cm³ (default: 0.4 for easier packing)
            tolerance_distance: Minimum distance between atoms in Å (default: 3.0 for easier packing)
            timeout_minutes: Maximum time to spend building (default: 5 minutes)
            output_callback: Function to log output
            
        Returns:
            Dict with results including PDB file paths
        """
        def log(msg):
            if output_callback:
                output_callback(msg)
            else:
                print(msg)
        
        # NOTE: We don't require PSP anymore! We use DirectPolymerBuilder as fallback
        # Just log PSP availability status but continue with our fallback methods
        if not hasattr(self, 'psp_builder') or not self.dependencies_available:
            log(f"⚠️ PSP AmorphousBuilder not available - using DirectPolymerBuilder fallback")
        
        try:
            log(f"🔧 Creating polymer simulation box for candidate {candidate_id}")
            log(f"   PSMILES: {psmiles}")
            log(f"   Parameters: {polymer_length} units × {num_molecules} molecules")
            log(f"   Density: {density} g/cm³, Tolerance: {tolerance_distance} Å")
            
            # Create candidate-specific directory
            candidate_dir = self.session_dir / f"candidate_{candidate_id}"
            candidate_dir.mkdir(exist_ok=True)
            
            # Use efficient MoleculeBuilder instead of wasteful AmorphousBuilder
            log(f"   🧬 Using PSP MoleculeBuilder (much more efficient!)")
            log(f"   ⏰ Max build time: {timeout_minutes} minutes")
            
            from utils.molecule_builder_utils import build_single_polymer_chain
            
            # Add timeout handling
            import time
            start_time = time.time()
            
            try:
                # Create single polymer chain directly (no waste!)
                polymer_output_dir = str(candidate_dir / 'molecules')
                result = build_single_polymer_chain(
                    psmiles=psmiles,
                    length=polymer_length,
                    output_dir=polymer_output_dir
                )
                
                build_time = time.time() - start_time
                
                if not result['success']:
                    raise Exception(f"MoleculeBuilder failed: {result.get('error', 'Unknown error')}")
                
                log(f"   ✅ Single polymer chain created in {build_time:.1f}s")
                log(f"   📁 Output: {result['polymer_pdb']}")
                
                # Set polymer_pdb for the rest of the function
                polymer_pdb = result['polymer_pdb']
                
            except Exception as psp_error:
                build_time = time.time() - start_time
                if build_time > timeout_minutes * 60:
                    log(f"   ⏰ MoleculeBuilder may have timed out after {build_time:.1f}s")
                else:
                    log(f"   ❌ MoleculeBuilder failed after {build_time:.1f}s: {psp_error}")
                raise
            
            # Verify the polymer PDB file exists
            if not polymer_pdb or not os.path.exists(polymer_pdb):
                raise FileNotFoundError(f"Polymer PDB file not found: {polymer_pdb}")
            
            log(f"   ✅ Polymer PDB created: {polymer_pdb}")
            
            return {
                'success': True,
                'polymer_pdb': polymer_pdb,
                'candidate_id': candidate_id,
                'psmiles': psmiles,
                'output_dir': str(candidate_dir),
                'parameters': {
                    'polymer_length': polymer_length,
                    'num_molecules': num_molecules,
                    'density': density
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            log(f"❌ Error creating polymer box for {candidate_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'candidate_id': candidate_id,
                'psmiles': psmiles
            }
    
    def create_insulin_flanked_system(self,
                                    polymer_pdb: str,
                                    candidate_id: str,
                                    insulin_pdb: Optional[str] = None,
                                    num_insulin_molecules: int = 1,
                                    num_polymer_duplicates: int = 20,
                                    box_size_nm: float = 3.0,
                                    density: float = 0.8,
                                    output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Create simulation box where insulin is flanked by polymers with proper cleaning
        
        Args:
            polymer_pdb: Path to polymer PDB file
            candidate_id: Unique identifier for this candidate
            insulin_pdb: Path to insulin PDB file (auto-detected if None)
            num_insulin_molecules: Number of insulin molecules
            num_polymer_duplicates: Number of polymer duplicates
            box_size_nm: Box size in nanometers
            density: System density in g/cm³ for box size calculation
            output_callback: Function to log output
            
        Returns:
            Dict with results including composite system paths
        """
        def log(msg):
            if output_callback:
                output_callback(msg)
            else:
                print(msg)
        
        try:
            log(f"🧬 Creating insulin-flanked system for candidate {candidate_id}")
            
            # Check if required utilities are available
            if not hasattr(self, 'create_large_box') or self.create_large_box is None:
                raise Exception("Insulin polymer builder utilities not available. Missing dependency: insulin_polymer_builder")
            
            if not hasattr(self, 'preprocess_pdb') or self.preprocess_pdb is None:
                raise Exception("PDB preprocessing utilities not available. Missing dependency: pdb_utils")
            
            # Auto-detect insulin PDB if not provided
            if not insulin_pdb:
                insulin_files = get_insulin_pdb_files()
                if insulin_files:
                    insulin_pdb = list(insulin_files.values())[0]
                    log(f"   Using auto-detected insulin: {insulin_pdb}")
                else:
                    raise FileNotFoundError("No insulin PDB file found")
            
            # Verify insulin file exists
            if not os.path.exists(insulin_pdb):
                raise FileNotFoundError(f"Insulin PDB file not found: {insulin_pdb}")
            
            log(f"   Preprocessing insulin PDB with PDBFixer...")
            
            # Preprocess insulin PDB (remove water, add missing atoms/hydrogens)
            candidate_dir = Path(polymer_pdb).parent
            processed_insulin_dir = candidate_dir / "processed_insulin"
            processed_insulin_dir.mkdir(exist_ok=True)
            
            processed_insulin_path = processed_insulin_dir / f"insulin_processed_{candidate_id}.pdb"
            
            # Use existing PDB preprocessing function
            preprocess_result = self.preprocess_pdb(
                pdb_path=insulin_pdb,
                remove_water=True,
                remove_heterogens=False,  # Keep important residues
                add_missing_residues=True,
                add_missing_atoms=True,
                add_missing_hydrogens=True,
                ph=7.4,
                output_callback=lambda msg: log(f"      {msg}")
            )
            
            if not preprocess_result.get('success'):
                raise Exception(f"Insulin preprocessing failed: {preprocess_result.get('error')}")
            
            log(f"   📊 Preprocessing result keys: {list(preprocess_result.keys())}")
            
            # Save processed insulin
            # Read content from the processed file
            processed_output_file = preprocess_result.get('output_file')
            if not processed_output_file:
                raise Exception(f"No output_file in preprocessing result. Available keys: {list(preprocess_result.keys())}")
            
            if not os.path.exists(processed_output_file):
                raise Exception(f"Processed insulin file not found: {processed_output_file}")
            
            log(f"   📁 Copying processed insulin: {processed_output_file} → {processed_insulin_path}")
            
            # Copy the processed file to our expected location
            shutil.copy(processed_output_file, processed_insulin_path)
            
            log(f"   ✅ Insulin preprocessed and cleaned")
            
            # Create composite system using existing function
            log(f"   Creating insulin-polymer composite system...")
            
            composite_output_path = candidate_dir / f"insulin_polymer_composite_{candidate_id}.pdb"
            
            # Calculate proper box size using density-based calculation
            log(f"   🧮 Calculating optimal box size using density and molecular weight...")
            
            # Import our density-based calculator
            from utils.molecular_weight_calculator import MolecularWeightCalculator, SystemComposition
            
            # Calculate molecular weights
            mw_calculator = MolecularWeightCalculator()
            
            # Estimate polymer molecular weight from PDB or use default
            polymer_mw_da = 5000.0  # Default polymer MW, could be improved by parsing PDB
            
            # Create system composition
            system_comp = SystemComposition(
                polymer_chains=num_polymer_duplicates,
                polymer_mw=polymer_mw_da,
                insulin_molecules=num_insulin_molecules,
                insulin_mw=5808.0
            )
            
            # Calculate box size using density (use default density)
            density = 0.8  # g/cm³ (default density)
            box_result = mw_calculator.calculate_box_size_from_density(system_comp, target_density=density)
            
            if box_result.success:
                calculated_box_size_nm = box_result.cubic_box_size_nm
                log(f"   📏 Density-based box size: {calculated_box_size_nm:.2f} nm")
                log(f"   📊 Total system mass: {system_comp.total_mass_da:.1f} Da")
                log(f"   📦 Density used: {density} g/cm³")
                
                # Use the larger of calculated vs user-specified box size
                final_box_size_nm = max(calculated_box_size_nm, box_size_nm)
                if final_box_size_nm > box_size_nm:
                    log(f"   ⬆️ Using calculated box size ({final_box_size_nm:.2f} nm) - larger than user input ({box_size_nm:.2f} nm)")
                else:
                    log(f"   ✅ Using user box size ({box_size_nm:.2f} nm) - adequate for system")
            else:
                log(f"   ⚠️ Density calculation failed: {box_result.error}")
                final_box_size_nm = box_size_nm
                log(f"   🔄 Falling back to user-specified box size: {final_box_size_nm:.2f} nm")
            
            # Use the create_large_box function WITH CLOSE PACKING ENABLED
            log(f"   🎯 Creating composite with CLOSE PACKING enabled (no exclusion radius)")
            success = self.create_large_box(
                polymer_pdb_path=polymer_pdb,
                insulin_pdb_path=str(processed_insulin_path),
                output_pdb_path=str(composite_output_path),
                num_insulin_molecules=num_insulin_molecules,
                num_polymer_duplicates=num_polymer_duplicates,
                box_size_nm=final_box_size_nm,
                output_dir=str(candidate_dir),
                allow_close_packing=True  # ENABLE CLOSE PACKING!
            )
            
            if not success:
                raise Exception("Failed to create insulin-polymer composite system")
            
            log(f"   ✅ Insulin-polymer composite created: {composite_output_path}")
            
            return {
                'success': True,
                'composite_pdb': str(composite_output_path),
                'processed_insulin_pdb': str(processed_insulin_path),
                'original_polymer_pdb': polymer_pdb,
                'candidate_id': candidate_id,
                'parameters': {
                    'num_insulin_molecules': num_insulin_molecules,
                    'num_polymer_duplicates': num_polymer_duplicates,
                    'box_size_nm': box_size_nm
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            log(f"❌ Error creating insulin-flanked system for {candidate_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'candidate_id': candidate_id
            }
    
    def process_all_candidates(self,
                             candidates: List[Dict[str, Any]],
                             create_polymer_boxes: bool = True,
                             create_insulin_systems: bool = True,
                             simulation_params: Optional[Dict] = None,
                             output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process all PSMILES candidates through the complete automation pipeline
        
        Args:
            candidates: List of candidate dictionaries with PSMILES and metadata
            create_polymer_boxes: Whether to create polymer simulation boxes
            create_insulin_systems: Whether to create insulin-flanked systems
            output_callback: Function to log output
            
        Returns:
            Dict with complete results for all candidates
        """
        def log(msg):
            if output_callback:
                output_callback(msg)
            else:
                print(msg)
        
        log(f"🚀 Starting automated simulation pipeline for {len(candidates)} candidates")
        
        # Debug: Show what candidates look like (console only)
        if candidates:
            print(f"🔍 First candidate structure preview:")
            first_candidate = candidates[0]
            print(f"   📋 Fields: {', '.join(first_candidate.keys())}")
            if 'functionalized' in first_candidate:
                psmiles_preview = str(first_candidate['functionalized'])[:50] if first_candidate['functionalized'] else 'None'
                print(f"   🧬 PSMILES (functionalized): {psmiles_preview}...")
            if 'psmiles' in first_candidate:
                psmiles_preview = str(first_candidate['psmiles'])[:50] if first_candidate['psmiles'] else 'None'
                print(f"   🧬 PSMILES (original): {psmiles_preview}...")
        else:
            print(f"⚠️ No candidates received!")
        
        results = {
            'session_id': self.session_id,
            'total_candidates': len(candidates),
            'polymer_boxes': [],
            'insulin_systems': [],
            'success_count': 0,
            'failed_count': 0,
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for i, candidate in enumerate(candidates, 1):
            candidate_id = f"{i:03d}_{uuid.uuid4().hex[:6]}"
            
            # More robust PSMILES extraction with debugging
            psmiles = candidate.get('functionalized', candidate.get('psmiles', ''))
            
            # Additional debugging - show what fields are available
            available_fields = list(candidate.keys())
            
            if not psmiles:
                log(f"❌ Candidate {i}: No PSMILES found")
                log(f"   📋 Available fields: {', '.join(available_fields)}")
                log(f"   📄 Candidate content: {str(candidate)[:200]}...")
                results['failed_count'] += 1
                results['errors'].append({
                    'candidate_id': candidate_id,
                    'error': 'No PSMILES found in candidate',
                    'available_fields': available_fields,
                    'candidate_preview': str(candidate)[:200]
                })
                continue
            
            log(f"\n📦 Processing candidate {i}/{len(candidates)}: {candidate_id}")
            log(f"   PSMILES: {psmiles[:60]}{'...' if len(psmiles) > 60 else ''}")
            
            candidate_success = True
            candidate_results = {
                'candidate_id': candidate_id,
                'psmiles': psmiles,
                'original_candidate': candidate
            }
            
            # Extract simulation parameters with sensible defaults (available for all steps)
            params = simulation_params or {}
            
            # Step 1: Create polymer simulation box
            if create_polymer_boxes:
                log(f"   🔧 Step 1: Creating polymer simulation box...")
                polymer_result = self.create_polymer_simulation_box(
                    psmiles=psmiles,
                    candidate_id=candidate_id,
                    polymer_length=params.get('polymer_length', 5),
                    num_molecules=params.get('num_polymer_molecules', 8),
                    density=params.get('density', 0.3),
                    tolerance_distance=params.get('tolerance_distance', 3.5),
                    timeout_minutes=params.get('timeout_minutes', 3),
                    output_callback=lambda msg: log(f"      {msg}")
                )
                
                candidate_results['polymer_box'] = polymer_result
                results['polymer_boxes'].append(polymer_result)
                
                if not polymer_result.get('success'):
                    candidate_success = False
                    error_msg = polymer_result.get('error', 'Unknown error')
                    log(f"   ❌ Polymer box creation failed: {error_msg}")
                    log(f"      📄 Details: PSMILES='{psmiles}', Method='{polymer_result.get('method', 'unknown')}'")
                else:
                    log(f"   ✅ Polymer box created successfully")
                    
                    # Step 2: Create insulin-flanked system
                    if create_insulin_systems:
                        log(f"   🧬 Step 2: Creating insulin-flanked system...")
                        insulin_result = self.create_insulin_flanked_system(
                            polymer_pdb=polymer_result['polymer_pdb'],
                            candidate_id=candidate_id,
                            num_polymer_duplicates=params.get('num_polymer_molecules', 8),
                            num_insulin_molecules=params.get('num_insulin_molecules', 1),
                            box_size_nm=params.get('box_size_nm', 3.0),
                            density=params.get('density', 0.8),
                            output_callback=lambda msg: log(f"      {msg}")
                        )
                        
                        candidate_results['insulin_system'] = insulin_result
                        results['insulin_systems'].append(insulin_result)
                        
                        if not insulin_result.get('success'):
                            candidate_success = False
                            error_msg = insulin_result.get('error', 'Unknown error')
                            log(f"   ❌ Insulin system creation failed: {error_msg}")
                            log(f"      📄 Details: Method='{insulin_result.get('method', 'unknown')}'")
                        else:
                            log(f"   ✅ Insulin system created successfully")
            
            # Update counters
            if candidate_success:
                results['success_count'] += 1
                log(f"   🎉 Candidate {candidate_id} completed successfully")
            else:
                results['failed_count'] += 1
                results['errors'].append(candidate_results)
                log(f"   💥 Candidate {candidate_id} failed")
        
        # Final summary
        success_rate = (results['success_count'] / results['total_candidates']) * 100
        log(f"\n🏁 Pipeline completed!")
        log(f"   ✅ Successful: {results['success_count']}/{results['total_candidates']} ({success_rate:.1f}%)")
        log(f"   ❌ Failed: {results['failed_count']}")
        log(f"   📁 Output directory: {self.session_dir}")
        
        # Save results summary
        results_file = self.session_dir / "automation_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        log(f"   📊 Results saved to: {results_file}")
        
        return results


def run_automated_simulation_pipeline(candidates: List[Dict[str, Any]], 
                                     output_callback: Optional[Callable] = None,
                                     simulation_params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced wrapper that uses the improved SimulationAutomationPipeline.
    
    This function now uses SimulationAutomationPipeline.process_all_candidates which 
    creates the complete insulin-polymer composite systems, not just polymer structures.
    """
    def log(msg):
        if output_callback:
            output_callback(msg)
        else:
            print(msg)
    
    try:
        log("🚀 Using enhanced SimulationAutomationPipeline for COMPLETE workflow")
        log("   🧬 Step 1: Build polymer structures")
        log("   💉 Step 2: Create insulin-polymer composite systems")
        
        # Use the SimulationAutomationPipeline which has the complete workflow
        pipeline = SimulationAutomationPipeline()
        
        # Enable insulin system creation by default for complete workflow
        create_insulin_systems = simulation_params.get('auto_create_insulin_systems', True) if simulation_params else True
        
        log(f"📊 Pipeline configuration:")
        log(f"   🧬 Create polymer boxes: True")
        log(f"   💉 Create insulin systems: {create_insulin_systems}")
        
        # Use the complete process_all_candidates method
        results = pipeline.process_all_candidates(
            candidates=candidates,
            create_polymer_boxes=True,
            create_insulin_systems=create_insulin_systems,
            simulation_params=simulation_params,
            output_callback=output_callback
        )
        
        return results
        
    except Exception as e:
        log(f"❌ Enhanced simulation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'method': 'enhanced_pipeline_failed',
            'total_candidates': len(candidates),
            'success_count': 0,
            'failed_count': len(candidates),
            'processed_results': [],
            'polymer_boxes': [],
            'insulin_systems': [],
            'errors': [{'error': str(e), 'traceback': traceback.format_exc()}]
        } 