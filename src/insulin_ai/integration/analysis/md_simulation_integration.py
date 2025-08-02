#!/usr/bin/env python3
"""
MD Simulation Integration for Insulin-AI App
Combines PDBFixer preprocessing, water removal, and OpenMM MD simulation
with real-time output streaming and progress monitoring
"""

import os
import sys
import uuid
import json
import time
import shutil
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import queue
import logging

# OpenMM and PDBFixer imports
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.app import HBonds, Simulation
    from openmm.app import StateDataReporter, DCDReporter, PDBReporter
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False

try:
    from openmmforcefields.generators import SystemGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False

# Import our proper OpenMM simulator
try:
    from .openmm_md_proper import ProperOpenMMSimulator, ProperStateReporter
    PROPER_OPENMM_AVAILABLE = True
except ImportError:
    PROPER_OPENMM_AVAILABLE = False

class MDSimulationIntegration:
    """Integrated MD simulation system for insulin-AI app"""
    
    def __init__(self, output_dir: str = "integrated_md_simulations"):
        """Initialize the MD simulation integration system
        
        Args:
            output_dir: Directory for MD simulation outputs
        """
        
        # Check dependencies
        self.dependencies_available = self._check_dependencies()
        
        if not self.dependencies_available['all_available']:
            missing = [k for k, v in self.dependencies_available.items() if not v and k != 'all_available']
            raise ImportError(f"Missing dependencies: {missing}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenMM simulator
        self.openmm_simulator = ProperOpenMMSimulator(str(self.output_dir))
        
        # Output streaming setup
        self.output_queue = queue.Queue()
        self.simulation_thread = None
        self.simulation_running = False
        
        # Thread-safe message storage for Streamlit app
        self.latest_messages = []
        
        print(f"🚀 MD Simulation Integration initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are available"""
        deps = {
            'openmm': OPENMM_AVAILABLE,
            'pdbfixer': PDBFIXER_AVAILABLE,
            'openmmforcefields': OPENMMFORCEFIELDS_AVAILABLE,
            'proper_openmm': PROPER_OPENMM_AVAILABLE
        }
        deps['all_available'] = all(deps[k] for k in ['openmm', 'pdbfixer', 'openmmforcefields', 'proper_openmm'])
        return deps
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get detailed dependency status for UI display"""
        status = {
            'dependencies': self.dependencies_available,
            'platform_info': {},
            'installation_commands': {
                'openmm': 'conda install -c conda-forge openmm',
                'pdbfixer': 'conda install -c conda-forge pdbfixer',
                'openmmforcefields': 'conda install -c conda-forge openmmforcefields'
            }
        }
        
        if OPENMM_AVAILABLE:
            try:
                # Get platform information
                platforms = []
                for i in range(mm.Platform.getNumPlatforms()):
                    platform = mm.Platform.getPlatform(i)
                    platforms.append({
                        'name': platform.getName(),
                        'speed': platform.getSpeed(),
                        'available': True
                    })
                status['platform_info'] = {
                    'platforms': platforms,
                    'best_platform': self.openmm_simulator.platform.getName() if hasattr(self.openmm_simulator, 'platform') else 'Unknown'
                }
            except Exception as e:
                status['platform_info'] = {'error': str(e)}
        
        return status
    
    def preprocess_pdb_file(self, pdb_path: str, 
                          remove_water: bool = True,
                          remove_heterogens: bool = False,
                          add_missing_residues: bool = True,
                          add_missing_atoms: bool = True,
                          add_missing_hydrogens: bool = True,
                          ph: float = 7.4,
                          output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Preprocess PDB file using PDBFixer for MD simulation
        
        Args:
            pdb_path: Path to input PDB file
            remove_water: Remove water molecules (HOH)
            remove_heterogens: Remove heterogens (except water) - set to False to keep polymers
            add_missing_residues: Add missing residues
            add_missing_atoms: Add missing atoms
            add_missing_hydrogens: Add missing hydrogens
            ph: pH for protonation state (physiological pH = 7.4)
            output_callback: Callback function for output messages
            
        Returns:
            Dict with preprocessing results
        """
        
        def log_output(message: str):
            print(message)  # Always print to console
            if output_callback:
                output_callback(message)  # Also send to app interface
        
        log_output(f"🔧 Starting PDB preprocessing with PDBFixer: {pdb_path}")
        
        try:
            # Create output paths
            pdb_name = Path(pdb_path).stem
            preprocess_dir = self.output_dir / f"preprocessed_{pdb_name}_{uuid.uuid4().hex[:8]}"
            preprocess_dir.mkdir(exist_ok=True)
            
            output_path = preprocess_dir / f"{pdb_name}_pdbfixer_cleaned.pdb"
            
            # Initialize PDBFixer
            log_output("   🔍 Loading PDB file with PDBFixer...")
            fixer = PDBFixer(filename=pdb_path)
            
            # Analyze initial structure
            initial_atoms = len(list(fixer.topology.atoms()))
            initial_residues = len(list(fixer.topology.residues()))
            
            log_output(f"   📊 Initial structure: {initial_atoms} atoms, {initial_residues} residues")
            
            # Find and fix missing residues
            if add_missing_residues:
                log_output("   🔍 Finding missing residues...")
                fixer.findMissingResidues()
                missing_residues = len(fixer.missingResidues)
                if missing_residues > 0:
                    log_output(f"      Found {missing_residues} missing residues")
                    fixer.findNonstandardResidues()
                    fixer.replaceNonstandardResidues()
                    log_output(f"      Replaced non-standard residues")
                else:
                    log_output("      No missing residues found")
            
            # Find and fix missing atoms
            if add_missing_atoms:
                log_output("   🔍 Finding missing atoms...")
                fixer.findMissingAtoms()
                missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
                if missing_atoms > 0:
                    log_output(f"      Found {missing_atoms} missing atoms")
                    fixer.addMissingAtoms()
                    log_output(f"      Added missing atoms")
                else:
                    log_output("      No missing atoms found")
            
            # Remove water molecules if requested
            if remove_water:
                log_output("   💧 Removing water molecules...")
                water_residues = []
                for residue in fixer.topology.residues():
                    if residue.name in ['HOH', 'WAT']:
                        water_residues.append(residue)
                
                if water_residues:
                    # Check if we have polymer (UNL residues) before removing heterogens
                    has_polymer = any(residue.name == 'UNL' for residue in fixer.topology.residues())
                    
                    if has_polymer:
                        log_output("      🧬 Detected polymer (UNL) - removing water selectively...")
                        # Custom water removal that preserves polymer
                        self._remove_water_only(fixer, log_output)
                        log_output(f"      Selectively removed {len(water_residues)} water molecules (preserved UNL)")
                    else:
                        fixer.removeHeterogens(keepWater=False)
                        log_output(f"      Removed {len(water_residues)} water molecules")
                else:
                    log_output("      No water molecules found")
            
            # Note: Check for polymer (UNL) residues and preserve them
            if remove_heterogens:
                # Check if we have polymer (UNL residues) - if so, preserve them
                has_polymer = any(residue.name == 'UNL' for residue in fixer.topology.residues())
                
                if has_polymer:
                    log_output("   🧬 Detected polymer residues (UNL) - preserving them...")
                    log_output("   ⚠️  Skipping heterogen removal to preserve polymer components")
                    # Don't remove heterogens when polymer is present
                else:
                    log_output("   🧪 Removing heterogens (no polymer detected)...")
                    fixer.removeHeterogens(keepWater=not remove_water)
            else:
                log_output("   🧪 Preserving polymer components (not removing heterogens)")
            
            # Add missing hydrogens
            if add_missing_hydrogens:
                log_output("   ➕ Adding missing hydrogens...")
                atoms_before = len(list(fixer.topology.atoms()))
                fixer.addMissingHydrogens(ph)
                atoms_after = len(list(fixer.topology.atoms()))
                hydrogens_added = atoms_after - atoms_before
                log_output(f"      Added {hydrogens_added} hydrogen atoms at pH {ph}")
            
            # Get final structure info
            final_atoms = len(list(fixer.topology.atoms()))
            final_residues = len(list(fixer.topology.residues()))
            
            log_output(f"   ✅ Final structure: {final_atoms} atoms, {final_residues} residues")
            
            # Save processed PDB
            log_output(f"   💾 Saving processed PDB: {output_path}")
            with open(output_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
            # Analyze residue composition
            protein_residues = 0
            unknown_residues = 0
            water_residues = 0
            other_residues = 0
            
            standard_amino_acids = {
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                'THR', 'TRP', 'TYR', 'VAL'
            }
            
            for residue in fixer.topology.residues():
                if residue.name in standard_amino_acids:
                    protein_residues += 1
                elif residue.name in ['HOH', 'WAT']:
                    water_residues += 1
                elif residue.name == 'UNL':
                    unknown_residues += 1
                else:
                    other_residues += 1
            
            log_output(f"   📊 Composition: {protein_residues} protein, {unknown_residues} polymer (UNL), {water_residues} water, {other_residues} other")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'preprocess_dir': str(preprocess_dir),
                'initial_atoms': initial_atoms,
                'final_atoms': final_atoms,
                'atoms_change': final_atoms - initial_atoms,
                'initial_residues': initial_residues,
                'final_residues': final_residues,
                'residue_composition': {
                    'protein_residues': protein_residues,
                    'unknown_residues': unknown_residues,
                    'water_residues': water_residues,
                    'other_residues': other_residues
                },
                'hydrogens_added': final_atoms - initial_atoms if add_missing_hydrogens else 0,
                'processing_completed': True
            }
            
        except Exception as e:
            log_output(f"   ❌ PDB preprocessing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_completed': False
            }
    
    def _remove_water_only(self, fixer, log_output):
        """
        Remove only water molecules (HOH, WAT) while preserving polymer (UNL) and other residues.
        
        Args:
            fixer: PDBFixer instance
            log_output: Logging function
        """
        try:
            # Get the current topology
            topology = fixer.topology
            
            # Find water residues to remove
            water_residues = []
            for residue in topology.residues():
                if residue.name in ['HOH', 'WAT']:
                    water_residues.append(residue)
            
            if not water_residues:
                log_output("      No water molecules found to remove")
                return
            
            # Create a modeller to selectively remove water
            from openmm.app import Modeller
            modeller = Modeller(fixer.topology, fixer.positions)
            
            # Remove water residues
            modeller.delete(water_residues)
            log_output(f"      Removed {len(water_residues)} water molecules")
            
            # Update the fixer with the new topology and positions
            fixer.topology = modeller.topology
            fixer.positions = modeller.positions
            
            # Verify polymer is still there
            unl_count = sum(1 for residue in fixer.topology.residues() if residue.name == 'UNL')
            log_output(f"      ✅ Preserved {unl_count} UNL polymer residues")
            
        except Exception as e:
            log_output(f"      ⚠️ Error in selective water removal: {e}")
            log_output("      Falling back to keeping all residues")
    
    def run_md_simulation_async(self, pdb_file: str,
                              temperature: float = 310.0,
                              equilibration_steps: int = 125000,  # Quick Test: 250 ps
                              production_steps: int = 500000,     # Quick Test: 1 ns (was 2500000 = 5 ns)
                              save_interval: int = 500,
                              output_prefix: str = None,
                              output_callback: Optional[Callable] = None,
                              manual_polymer_dir: str = None) -> str:
        """
        Run MD simulation asynchronously with proper preprocessing
        
        Args:
            pdb_file: Path to input PDB file
            temperature: Simulation temperature in Kelvin
            equilibration_steps: Number of equilibration steps (default: 125000 = 250 ps Quick Test)
            production_steps: Number of production steps (default: 500000 = 1 ns Quick Test)
            save_interval: Steps between saved frames
            output_prefix: Prefix for output files
            output_callback: Callback function for output messages
            manual_polymer_dir: Manual polymer directory for force field parameterization
            
        Returns:
            Simulation ID
        """
        
        # Generate simulation ID
        simulation_id = output_prefix or f"sim_{uuid.uuid4().hex[:8]}"
        
        # Store simulation parameters
        self.current_simulation = {
            'id': simulation_id,
            'pdb_file': pdb_file,
            'temperature': temperature,
            'equilibration_steps': equilibration_steps,
            'production_steps': production_steps,
            'save_interval': save_interval,
            'manual_polymer_dir': manual_polymer_dir,
            'status': 'starting',
            'start_time': time.time()
        }
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(simulation_id, pdb_file, temperature, equilibration_steps, 
                  production_steps, save_interval, output_callback, manual_polymer_dir)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        
        return simulation_id
    
    def _run_simulation_thread(self, simulation_id: str, pdb_file: str,
                             temperature: float, equilibration_steps: int,
                             production_steps: int, save_interval: int,
                             output_callback: Optional[Callable],
                             manual_polymer_dir: str = None):
        """Run the simulation in a separate thread"""
        
        def log_output(message: str):
            print(message)  # Always print to console
        
        try:
            # Step 1: Preprocess PDB file with PDBFixer
            log_output(f"🔧 Step 1: Preprocessing PDB file with PDBFixer")
            
            # Add info about manual polymer selection
            if manual_polymer_dir:
                log_output(f"🎯 Using manual polymer directory: {manual_polymer_dir}")
            
            preprocess_result = self.preprocess_pdb_file(
                pdb_file,
                remove_water=True,
                remove_heterogens=False,  # Keep polymers
                add_missing_residues=True,
                add_missing_atoms=True,
                add_missing_hydrogens=True,
                ph=7.4,
                output_callback=log_output
            )
            
            if not preprocess_result['success']:
                log_output(f"❌ PDB preprocessing failed: {preprocess_result['error']}")
                self.simulation_running = False
                return
            
            # Use preprocessed PDB file
            processed_pdb = preprocess_result['output_path']
            log_output(f"✅ PDB preprocessing completed: {processed_pdb}")
            
            # Step 2: Load processed PDB file to get topology and positions
            log_output(f"📖 Loading processed PDB file for OpenMM simulation")
            
            from openmm.app import PDBFile
            processed_pdb_obj = PDBFile(processed_pdb)
            processed_topology = processed_pdb_obj.topology
            processed_positions = processed_pdb_obj.positions
            
            # Step 3: Run MD simulation with proper OpenMM simulator
            log_output(f"🚀 Step 3: Running MD simulation with OpenMM")
            
            # Update simulation status
            self.current_simulation['status'] = 'running'
            self.current_simulation['processed_pdb'] = processed_pdb
            self.current_simulation['preprocessing_results'] = preprocess_result
            
            # Run simulation using proper OpenMM simulator
            def check_stop_condition():
                return not self.simulation_running

            simulation_results = self.openmm_simulator.run_proper_simulation_with_preprocessing(
                pdb_file=processed_pdb,
                pre_processed_topology=processed_topology,
                pre_processed_positions=processed_positions,
                temperature=temperature,
                equilibration_steps=equilibration_steps,
                production_steps=production_steps,
                save_interval=save_interval,
                output_prefix=simulation_id,
                stop_condition_check=check_stop_condition,
                output_callback=log_output,
                manual_polymer_dir=manual_polymer_dir
            )
            
            # Update simulation status
            if simulation_results['success']:
                if simulation_results.get('user_stopped', False):
                    self.current_simulation['status'] = 'stopped'
                else:
                    self.current_simulation['status'] = 'completed'
            else:
                self.current_simulation['status'] = 'failed'
            
            self.current_simulation['results'] = simulation_results
            self.current_simulation['end_time'] = time.time()
            
            if simulation_results['success']:
                if simulation_results.get('user_stopped', False):
                    log_output(f"🛑 MD simulation stopped by user request!")
                    log_output(f"📊 Performance: {simulation_results['performance']['ns_per_day']:.1f} ns/day")
                    log_output(f"⏱️ Partial time: {simulation_results['timing']['total']:.1f} seconds")
                else:
                    log_output(f"🎉 MD simulation completed successfully!")
                    log_output(f"📊 Performance: {simulation_results['performance']['ns_per_day']:.1f} ns/day")
                    log_output(f"⏱️ Total time: {simulation_results['timing']['total']:.1f} seconds")
                
                # Log key results
                if 'production_stats' in simulation_results and 'temperature' in simulation_results['production_stats']:
                    temp_stats = simulation_results['production_stats']['temperature']
                    log_output(f"🌡️ Final temperature: {temp_stats['mean']:.1f} ± {temp_stats['std']:.1f} K")
                
                if 'energy_analysis' in simulation_results:
                    energy = simulation_results['energy_analysis']
                    log_output(f"🔋 Energy change: {energy['minimization_change']:.1f} kJ/mol")
                
            else:
                log_output(f"❌ MD simulation failed: {simulation_results.get('error', 'Unknown error')}")
                
        except Exception as e:
            log_output(f"❌ Simulation thread failed: {str(e)}")
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
        
        finally:
            self.simulation_running = False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        if hasattr(self, 'current_simulation'):
            return {
                'simulation_running': self.simulation_running,
                'simulation_info': self.current_simulation
            }
        else:
            return {
                'simulation_running': False,
                'simulation_info': None
            }
    
    def stop_simulation(self):
        """Stop the current simulation"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_running = False
            # Update current simulation status if available
            if hasattr(self, 'current_simulation'):
                self.current_simulation['status'] = 'stopping'
            # Note: Thread will check this flag and exit gracefully
            print("🛑 Simulation stop requested")
            return True
        else:
            print("⚠️  No active simulation to stop")
            return False
    
    def get_simulation_files(self, simulation_id: str) -> Dict[str, Any]:
        """Get files for a specific simulation"""
        try:
            sim_dir = self.output_dir / simulation_id
            
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            # Look for common simulation files
            files = {}
            
            # Check for different possible structures
            for subdir in ['production', 'equilibration']:
                subdir_path = sim_dir / subdir
                if subdir_path.exists():
                    for file_pattern in ['*.dcd', '*.pdb', '*.csv', '*.png']:
                        for file_path in subdir_path.glob(file_pattern):
                            file_key = f"{subdir}/{file_path.name}"
                            files[file_key] = str(file_path)
            
            # Check for top-level files
            for file_pattern in ['*.json', '*.png', '*.pdb']:
                for file_path in sim_dir.glob(file_pattern):
                    files[file_path.name] = str(file_path)
            
            return {
                'success': True,
                'files': files,
                'simulation_dir': str(sim_dir)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Analyze simulation results including trajectory data"""
        try:
            sim_dir = self.output_dir / simulation_id
            
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            # Look for simulation report JSON
            report_file = sim_dir / 'simulation_report.json'
            
            if not report_file.exists():
                return {'success': False, 'error': 'Simulation report not found'}
            
            # Load simulation report
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Extract key information for analysis
            analysis_result = {
                'success': True,
                'basic_info': {
                    'simulation_id': simulation_id,
                    'total_atoms': report_data.get('system_info', {}).get('final_atoms', 0),
                    'total_time_minutes': report_data.get('timing', {}).get('total', 0) / 60.0,
                    'performance_ns_per_day': report_data.get('performance', {}).get('ns_per_day', 0),
                    'force_field': report_data.get('system_info', {}).get('force_field', 'Unknown'),
                    'success': report_data.get('success', False)
                },
                'files': report_data.get('files', {}),
                'energy_analysis': report_data.get('energy_analysis', {}),
                'final_stats': report_data.get('production_stats', {}),
                'system_info': report_data.get('system_info', {}),
                'parameters': report_data.get('parameters', {}),
                'timing': report_data.get('timing', {}),
                'performance': report_data.get('performance', {})
            }
            
            return analysis_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_available_simulations(self) -> List[Dict[str, Any]]:
        """Get list of available simulations"""
        simulations = []
        
        try:
            # Look for simulation directories
            for sim_dir in self.output_dir.iterdir():
                if sim_dir.is_dir() and sim_dir.name.startswith('sim_'):
                    
                    # Look for simulation report
                    report_file = sim_dir / 'simulation_report.json'
                    
                    if report_file.exists():
                        try:
                            with open(report_file, 'r') as f:
                                report_data = json.load(f)
                            
                            # Extract simulation info
                            sim_info = {
                                'id': sim_dir.name,
                                'timestamp': report_data.get('timestamp', ''),
                                'input_file': report_data.get('input_file', ''),
                                'total_atoms': report_data.get('system_info', {}).get('final_atoms', 0),
                                'performance': report_data.get('performance', {}).get('ns_per_day', 0),
                                'success': report_data.get('success', False),
                                'force_field': report_data.get('system_info', {}).get('force_field', 'Unknown')
                            }
                            
                            simulations.append(sim_info)
                            
                        except Exception as e:
                            print(f"Error reading simulation report {report_file}: {e}")
                            continue
            
            # Sort by timestamp (newest first)
            simulations.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error getting available simulations: {e}")
        
        return simulations

def get_insulin_polymer_pdb_files(base_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Get list of available insulin-polymer PDB files for MD simulation
    
    Args:
        base_dir: Base directory to search for PDB files
        
    Returns:
        List of PDB file info dictionaries
    """
    
    pdb_files = []
    base_path = Path(base_dir)
    
    # Search patterns for insulin-polymer files
    search_patterns = [
        "**/insulin_embedded*.pdb",
        "**/insulin_polymer*.pdb", 
        "**/packmol*.pdb",
        "**/composite*.pdb",
        "**/*insulin*.pdb"
    ]
    
    for pattern in search_patterns:
        for pdb_path in base_path.glob(pattern):
            try:
                # Get file info
                file_stats = pdb_path.stat()
                size_mb = file_stats.st_size / (1024 * 1024)
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                
                # Try to get atom count
                atom_count = 0
                try:
                    with open(pdb_path, 'r') as f:
                        for line in f:
                            if line.startswith(('ATOM', 'HETATM')):
                                atom_count += 1
                except:
                    atom_count = 0
                
                # Determine file type
                file_type = 'other_insulin'
                if 'insulin_embedded' in pdb_path.name:
                    file_type = 'insulin_embedded'
                elif 'composite' in pdb_path.name:
                    file_type = 'composite'
                elif 'packmol' in pdb_path.name:
                    file_type = 'packmol'
                
                pdb_info = {
                    'name': pdb_path.name,
                    'path': str(pdb_path.absolute()),
                    'size_mb': size_mb,
                    'atom_count': atom_count,
                    'modified': modified_time.isoformat(),
                    'file_type': file_type,
                    'relative_path': str(pdb_path.relative_to(base_path))
                }
                
                pdb_files.append(pdb_info)
                
            except Exception as e:
                print(f"Error processing {pdb_path}: {e}")
                continue
    
    # Remove duplicates and sort
    seen_paths = set()
    unique_files = []
    
    for pdb_info in pdb_files:
        if pdb_info['path'] not in seen_paths:
            unique_files.append(pdb_info)
            seen_paths.add(pdb_info['path'])
    
    # Sort by priority: embedded files first, then by modification time
    unique_files.sort(key=lambda x: (
        x['file_type'] != 'insulin_embedded',  # Embedded files first
        -x['atom_count'],  # Larger files first
        x['modified']  # Newer files first
    ))
    
    return unique_files

# Example usage and testing
if __name__ == "__main__":
    # Test the MD simulation integration
    print("🧪 Testing MD Simulation Integration")
    
    # Check dependencies
    integration = MDSimulationIntegration()
    
    status = integration.get_dependency_status()
    print(f"Dependencies: {status['dependencies']}")
    
    # Find available PDB files
    pdb_files = get_insulin_polymer_pdb_files()
    print(f"Found {len(pdb_files)} insulin-polymer PDB files")
    
    for pdb_file in pdb_files[:3]:  # Show first 3
        print(f"  📁 {pdb_file['name']} ({pdb_file['atom_count']} atoms)") 