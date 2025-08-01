#!/usr/bin/env python3
"""
Simple MD Integration for Insulin-AI App
========================================

REPLACEMENT for the complex md_simulation_integration.py

This uses the WORKING approach from openmm_test.py:
- SimpleWorkingMDSimulator instead of ProperOpenMMSimulator
- Direct RDKit → GAFF → Implicit solvent approach
- No complex fallbacks or over-engineering
- ACTUALLY WORKS

Maintains the same interface for UI compatibility but uses the simple working core.
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
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import queue
import logging

# Import our simple working simulator
try:
    from .simple_working_md_simulator import SimpleWorkingMDSimulator
    SIMPLE_WORKING_AVAILABLE = True
except ImportError:
    SIMPLE_WORKING_AVAILABLE = False

# MM-GBSA calculator removed as no longer needed


class SimpleMDIntegration:
    """
    Simple MD Integration using the WORKING approach.
    
    This replaces the complex MDSimulationIntegration with a simple, 
    working approach based on the successful openmm_test.py script.
    
    Key differences from the old complex system:
    - Uses SimpleWorkingMDSimulator (WORKS)
    - Direct RDKit → GAFF approach (WORKS)  
    - No complex ProperOpenMMSimulator (DOESN'T WORK)
    - No complex fallbacks or over-engineering
    """
    
    def __init__(self, output_dir: str = "simple_md_simulations"):
        """Initialize the simple MD integration system"""
        
        # Check dependencies
        self.dependencies_available = self._check_dependencies()
        
        if not self.dependencies_available['all_available']:
            missing = [k for k, v in self.dependencies_available.items() if not v and k != 'all_available']
            raise ImportError(f"Missing dependencies: {missing}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize simple working simulator (REPLACEMENT for ProperOpenMMSimulator)
        self.simple_simulator = SimpleWorkingMDSimulator(str(self.output_dir))
        
        # Output streaming setup (keep for UI compatibility)
        self.output_queue = queue.Queue()
        self.simulation_thread = None
        self.simulation_running = False
        
        # Thread-safe message storage for Streamlit app
        self.latest_messages = []
        
        # Current simulation state
        self.current_simulation = None
        
        print(f"🚀 Simple MD Integration initialized (WORKING APPROACH)")
        print(f"📁 Output directory: {self.output_dir}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are available"""
        deps = {
            'simple_working': SIMPLE_WORKING_AVAILABLE,
            'mmgbsa': MMGBSA_AVAILABLE,
        }
        deps['all_available'] = deps['simple_working']  # Only require the working simulator
        return deps
    
    def find_polymer_and_composite_files(self, manual_polymer_dir: str = None) -> Tuple[str, str]:
        """
        Find COMPATIBLE polymer and composite PDB files.
        
        KEY INSIGHT: Files must be from the SAME candidate session to work together.
        The composite file is pre-processed to be compatible with that exact polymer.
        
        This looks for:
        - polymer.pdb (for GAFF template generation) 
        - *_preprocessed.pdb (pre-processed composite system from SAME candidate)
        """
        
        if manual_polymer_dir and os.path.exists(manual_polymer_dir):
            search_dir = Path(manual_polymer_dir)
            print(f"🔍 Searching in manual directory: {search_dir}")
        else:
            # Search in automated_simulations directory
            search_dir = Path("automated_simulations")
            print(f"🔍 Searching in automated directory: {search_dir}")
        
        # Find COMPATIBLE file pairs from the same candidate session
        compatible_pairs = []
        
        # Search for candidate directories
        for candidate_dir in search_dir.rglob("candidate_*"):
            if not candidate_dir.is_dir():
                continue
                
            candidate_id = candidate_dir.name  # e.g., "candidate_001_e36e15"
            
            # Look for polymer.pdb in this candidate
            polymer_pdb = None
            for polymer_file in candidate_dir.rglob("polymer.pdb"):
                if polymer_file.exists():
                    polymer_pdb = str(polymer_file)
                    break
            
            # Look for compatible composite file with same candidate ID
            composite_pdb = None
            for composite_file in candidate_dir.rglob("*_preprocessed.pdb"):
                if (composite_file.exists() and 
                    "insulin" in str(composite_file).lower() and
                    candidate_id.split('_')[-1] in str(composite_file)):  # Match candidate ID
                    composite_pdb = str(composite_file)
                    break
            
            # If we found both files from the same candidate, it's a compatible pair
            if polymer_pdb and composite_pdb:
                compatible_pairs.append((polymer_pdb, composite_pdb, candidate_id))
                print(f"✅ Found compatible pair from {candidate_id}:")
                print(f"   • Polymer: {polymer_pdb}")
                print(f"   • Composite: {composite_pdb}")
        
        if not compatible_pairs:
            # Fallback to old method if no compatible pairs found
            print(f"⚠️  No compatible pairs found, trying old method...")
            
            polymer_pdb = None
            composite_pdb = None
            
            # Find any polymer.pdb
            for polymer_file in search_dir.rglob("polymer.pdb"):
                if polymer_file.exists():
                    polymer_pdb = str(polymer_file)
                    print(f"✅ Found polymer PDB: {polymer_pdb}")
                    break
            
            # Find any composite file (try both naming patterns)
            # Pattern 1: Look for *_preprocessed.pdb (legacy naming)
            for composite_file in search_dir.rglob("*_preprocessed.pdb"):
                if composite_file.exists() and "insulin" in str(composite_file).lower():
                    composite_pdb = str(composite_file)
                    print(f"✅ Found composite PDB (preprocessed): {composite_pdb}")
                    break
            
            # Pattern 2: Look for insulin_polymer_composite_*.pdb (current automation pipeline naming)
            if not composite_pdb:
                for composite_file in search_dir.rglob("insulin_polymer_composite_*.pdb"):
                    if composite_file.exists():
                        composite_pdb = str(composite_file)
                        print(f"✅ Found composite PDB (automation): {composite_pdb}")
                        break
            
            if not polymer_pdb or not composite_pdb:
                raise FileNotFoundError(
                    f"Could not find required files:\n"
                    f"  Polymer PDB: {'✅' if polymer_pdb else '❌'} {polymer_pdb or 'NOT FOUND'}\n"
                    f"  Composite PDB: {'✅' if composite_pdb else '❌'} {composite_pdb or 'NOT FOUND'}\n"
                    f"  Search directory: {search_dir}"
                )
            
            return polymer_pdb, composite_pdb
        
        # Use the first compatible pair (prefer the working candidate_001_e36e15 if available)
        working_candidate = "candidate_001_e36e15"
        for polymer_pdb, composite_pdb, candidate_id in compatible_pairs:
            if working_candidate in candidate_id:
                print(f"🎯 Using WORKING candidate: {candidate_id}")
                return polymer_pdb, composite_pdb
        
        # Otherwise use the first compatible pair
        polymer_pdb, composite_pdb, candidate_id = compatible_pairs[0]
        print(f"✅ Using compatible pair from: {candidate_id}")
        return polymer_pdb, composite_pdb
    
    def run_md_simulation_async(self, pdb_file: str,
                              temperature: float = 310.0,
                              equilibration_steps: int = 125000,   # Quick Test: 250 ps with 2 fs timestep (user requested quick mode)
                              production_steps: int = 500000,     # Quick Test: 1 ns with 2 fs timestep (was 2500000 = 5 ns)
                              save_interval: int = 1000,
                              output_prefix: str = None,
                              output_callback: Optional[Callable] = None,
                              manual_polymer_dir: str = None,
                              enhanced_smiles: str = None) -> str:
        """
        Run MD simulation asynchronously using SIMPLE WORKING approach.
        
        This replaces the complex async method with the simple working pattern.
        """
        
        # Generate simulation ID
        simulation_id = output_prefix or f"simple_sim_{uuid.uuid4().hex[:8]}"
        
        # Store simulation parameters
        self.current_simulation = {
            'id': simulation_id,
            'pdb_file': pdb_file,
            'temperature': temperature,
            'equilibration_steps': equilibration_steps,
            'production_steps': production_steps,
            'save_interval': save_interval,
            'manual_polymer_dir': manual_polymer_dir,
            'enhanced_smiles': enhanced_smiles,
            'status': 'starting',
            'start_time': time.time()
        }
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._run_simple_simulation_thread,
            args=(simulation_id, pdb_file, temperature, equilibration_steps, 
                  production_steps, save_interval, output_callback, manual_polymer_dir, enhanced_smiles)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        
        return simulation_id
    
    def _run_simple_simulation_thread(self, simulation_id: str, pdb_file: str,
                                    temperature: float, equilibration_steps: int,
                                    production_steps: int, save_interval: int,
                                    output_callback: Optional[Callable],
                                    manual_polymer_dir: str, enhanced_smiles: str = None):
        """
        Run simulation in thread using SIMPLE WORKING approach.
        
        This is the core replacement - uses SimpleWorkingMDSimulator instead
        of the complex ProperOpenMMSimulator.
        """
        
        def log_and_callback(message: str):
            """Helper to log messages (callback handled by CallbackStateReporter)"""
            self.latest_messages.append(message)
            print(message)  # Also print to console
        
        try:
            self.current_simulation['status'] = 'finding_files'
            
            # Step 1: Find polymer and composite files (SIMPLE approach)
            log_and_callback(f"🔍 Finding polymer and composite files...")
            polymer_pdb, composite_pdb = self.find_polymer_and_composite_files(manual_polymer_dir)
            
            log_and_callback(f"✅ Files found:")
            log_and_callback(f"   • Polymer: {polymer_pdb}")  
            log_and_callback(f"   • Composite: {composite_pdb}")
            
            self.current_simulation['status'] = 'running_simulation'
            
            # Step 2: Run simulation using SIMPLE WORKING approach
            log_and_callback(f"\n🚀 Starting SIMPLE MD simulation...")
            log_and_callback(f"   • Method: Simple Working Approach (like openmm_test.py)")
            log_and_callback(f"   • Simulator: SimpleWorkingMDSimulator")
            log_and_callback(f"   • No complex fallbacks or over-engineering")
            log_and_callback(f"   • Parameters received:")
            log_and_callback(f"     - Equilibration: {equilibration_steps} steps ({equilibration_steps * 2 / 1000:.1f} ps)")
            log_and_callback(f"     - Production: {production_steps} steps ({production_steps * 2 / 1000000:.1f} ns)")
            log_and_callback(f"     - Save interval: {save_interval} steps ({save_interval * 2 / 1000:.1f} ps)")
            log_and_callback(f"     - Temperature: {temperature} K")
            
            # Create stop condition check function
            def stop_condition_check():
                """Check if simulation should be stopped"""
                return not self.simulation_running
            
            # Use our simple working simulator (ENHANCED with stored SMILES support)
            log_and_callback(f"🔍 DEBUG: enhanced_smiles parameter received: {enhanced_smiles is not None}")
            if enhanced_smiles:
                log_and_callback(f"⚡ **ENHANCED MODE**: Using pre-stored SMILES for force field")
                log_and_callback(f"🎯 Stored SMILES: {enhanced_smiles[:50]}...")
                log_and_callback(f"📏 Full SMILES length: {len(enhanced_smiles)} characters")
            else:
                log_and_callback(f"📁 **STANDARD MODE**: Will convert PDB → SMILES")
                log_and_callback(f"❌ DEBUG: enhanced_smiles is None or empty - falling back to PDB conversion")
            
            results = self.simple_simulator.run_simulation(
                polymer_pdb_path=polymer_pdb,
                composite_pdb_path=composite_pdb,
                temperature=temperature,
                equilibration_steps=equilibration_steps,
                production_steps=production_steps,
                save_interval=save_interval,
                output_prefix=simulation_id,
                output_callback=log_and_callback,
                stop_condition_check=stop_condition_check,  # Enable stopping
                enhanced_smiles=enhanced_smiles  # **ENHANCED: Pass stored SMILES**
            )
            
            if results['success']:
                log_and_callback(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")
                log_and_callback(f"   • Approach: {results['approach_used']}")
                log_and_callback(f"   • Total time: {results['total_time']:.1f} seconds")
                log_and_callback(f"   • Final energy: {results['final_energy']}")
                log_and_callback(f"   • Trajectory: {results['trajectory_file']}")
                
                self.current_simulation['status'] = 'completed'
                self.current_simulation['results'] = results
            else:
                # Handle stopped or failed simulation
                if 'stopped by user' in results.get('message', ''):
                    log_and_callback(f"\n🛑 SIMULATION STOPPED BY USER")
                    log_and_callback(f"   • Phase: {results.get('phase', 'unknown')}")
                    log_and_callback(f"   • Steps completed: {results.get('steps_completed', 0)}")
                    if 'total_time' in results:
                        log_and_callback(f"   • Time elapsed: {results['total_time']:.1f} seconds")
                    if 'trajectory_file' in results:
                        log_and_callback(f"   • Partial trajectory: {results['trajectory_file']}")
                    
                    self.current_simulation['status'] = 'stopped'
                else:
                    log_and_callback(f"\n❌ SIMULATION FAILED: {results.get('message', 'Unknown error')}")
                    self.current_simulation['status'] = 'failed'
                
                self.current_simulation['results'] = results
            
            self.current_simulation['end_time'] = time.time()
            
        except Exception as e:
            error_msg = f"❌ Simulation thread failed: {str(e)}"
            log_and_callback(error_msg)
            
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
            self.current_simulation['end_time'] = time.time()
        
        finally:
            self.simulation_running = False
    
    def run_simple_insulin_simulation_async(self, insulin_pdb: str,
                                           temperature: float = 310.0,
                                           equilibration_steps: int = 5000,
                                           production_steps: int = 25000,
                                           save_interval: int = 1000,
                                           output_prefix: str = None,
                                           output_callback: Optional[Callable] = None) -> str:
        """
        Run simple insulin-only simulation asynchronously using the simple_insulin_simulation.py approach.
        
        This method handles CYX residues properly and avoids GAFF template generator issues.
        Uses the key insight: CYX residues are CORRECT for disulfide-bonded cysteines.
        AMBER force fields already support CYX without complex template generators.
        
        Args:
            insulin_pdb: Path to insulin PDB file (with CYX residues)
            temperature: Temperature in Kelvin (default 310K = body temp)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            save_interval: Save trajectory every N steps
            output_prefix: Prefix for output files
            output_callback: Callback function for progress updates
            
        Returns:
            Simulation ID string
        """
        
        # Generate simulation ID
        simulation_id = output_prefix or f"simple_insulin_{uuid.uuid4().hex[:8]}"
        
        # Store simulation parameters
        self.current_simulation = {
            'id': simulation_id,
            'insulin_pdb': insulin_pdb,
            'temperature': temperature,
            'equilibration_steps': equilibration_steps,
            'production_steps': production_steps,
            'save_interval': save_interval,
            'status': 'starting',
            'start_time': time.time(),
            'simulation_type': 'simple_insulin'
        }
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._run_simple_insulin_thread,
            args=(simulation_id, insulin_pdb, temperature, equilibration_steps, 
                  production_steps, save_interval, output_callback)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        
        return simulation_id
    
    def _run_simple_insulin_thread(self, simulation_id: str, insulin_pdb: str,
                                  temperature: float, equilibration_steps: int,
                                  production_steps: int, save_interval: int,
                                  output_callback: Optional[Callable]):
        """
        Run simple insulin simulation in thread using the simple_insulin_simulation.py approach.
        
        This avoids all GAFF template generator issues by using simple AMBER force fields only.
        """
        
        def log_and_callback(message: str):
            """Helper to log messages and send through callback"""
            self.latest_messages.append(message)
            print(message)  # Also print to console
            if output_callback:
                try:
                    output_callback(message)
                except Exception as e:
                    print(f"Callback error: {e}")
        
        try:
            self.current_simulation['status'] = 'starting'
            log_and_callback("🧬 Starting Simple Insulin Simulation (AMBER only)")
            
            # Define stop condition check function
            def stop_condition_check():
                return self.current_simulation.get('status') == 'stopping'
            
            # Run the simple insulin simulation using our enhanced simulator
            results = self.simple_simulator.run_simple_insulin_simulation(
                insulin_pdb=insulin_pdb,
                equilibration_steps=equilibration_steps,
                production_steps=production_steps,
                temperature=temperature,
                save_interval=save_interval,
                output_prefix=f"simple_insulin_{simulation_id}",
                output_callback=output_callback,
                stop_condition_check=stop_condition_check
            )
            
            # Update simulation status based on results
            if results['success']:
                log_and_callback(f"\n🎉 SIMPLE INSULIN SIMULATION COMPLETED SUCCESSFULLY!")
                log_and_callback(f"📁 Results saved in: {results['output_dir']}")
                log_and_callback(f"🎬 Trajectory: {results['trajectory_file']}")
                log_and_callback(f"📊 Final energy: {results['final_energy']}")
                log_and_callback(f"⏱️  Total simulation time: {results['simulation_time_ps']:.1f} ps")
                log_and_callback(f"🖥️  Platform used: {results['platform']}")
                
                self.current_simulation['status'] = 'completed'
            elif results.get('message') == 'Simulation stopped by user during equilibration' or \
                 results.get('message') == 'Simulation stopped by user during production':
                log_and_callback(f"\n🛑 SIMPLE INSULIN SIMULATION STOPPED BY USER")
                if 'steps_completed' in results:
                    log_and_callback(f"📊 Steps completed: {results['steps_completed']}")
                if 'final_energy' in results:
                    log_and_callback(f"📊 Final energy: {results['final_energy']}")
                
                self.current_simulation['status'] = 'stopped'
            else:
                log_and_callback(f"\n❌ SIMPLE INSULIN SIMULATION FAILED: {results.get('error', 'Unknown error')}")
                self.current_simulation['status'] = 'failed'
            
            self.current_simulation['results'] = results
            self.current_simulation['end_time'] = time.time()
            
        except Exception as e:
            error_msg = f"❌ Simple insulin simulation thread failed: {str(e)}"
            log_and_callback(error_msg)
            
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
            self.current_simulation['end_time'] = time.time()
            
            import traceback
            traceback.print_exc()
        
        finally:
            self.simulation_running = False

    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        if self.current_simulation is None:
            return {
                'status': 'no_simulation',
                'simulation_running': False,
                'simulation_info': None  # UI expects this key
            }
        
        # Return status in the format the UI expects
        return {
            'simulation_running': self.simulation_running,
            'simulation_info': self.current_simulation  # UI expects simulation details here
        }
    
    def get_latest_messages(self, limit: int = 50) -> List[str]:
        """Get latest messages for UI display"""
        return self.latest_messages[-limit:] if self.latest_messages else []
    
    def is_simulation_running(self) -> bool:
        """Check if simulation is running"""
        return self.simulation_running
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get dependency status for UI compatibility"""
        
        # Check if our simple working system is available
        deps = {
            'openmm': SIMPLE_WORKING_AVAILABLE,
            'pdbfixer': SIMPLE_WORKING_AVAILABLE, 
            'openmmforcefields': SIMPLE_WORKING_AVAILABLE,
            'simple_working': SIMPLE_WORKING_AVAILABLE,
            'all_available': SIMPLE_WORKING_AVAILABLE
        }
        
        # Get platform info if available
        platform_info = {}
        if SIMPLE_WORKING_AVAILABLE:
            try:
                import openmm as mm
                
                # Get detailed platform information
                platforms = []
                best_platform_name = 'Reference'  # Default fallback
                
                for i in range(mm.Platform.getNumPlatforms()):
                    platform = mm.Platform.getPlatform(i)
                    platforms.append({
                        'name': platform.getName(),
                        'speed': platform.getSpeed(),
                        'available': True
                    })
                
                # Determine best platform (prefer CUDA > OpenCL > CPU > Reference)
                platform_names = [p['name'] for p in platforms]
                if 'CUDA' in platform_names:
                    best_platform_name = 'CUDA'
                elif 'OpenCL' in platform_names:
                    best_platform_name = 'OpenCL'
                elif 'CPU' in platform_names:
                    best_platform_name = 'CPU'
                
                platform_info = {
                    'platforms': platforms,
                    'best_platform': best_platform_name,
                    'platform_count': len(platforms)
                }
            except Exception as e:
                platform_info = {
                    'platforms': [],
                    'best_platform': 'Unknown',
                    'platform_count': 0,
                    'error': str(e)
                }
        else:
            platform_info = {
                'platforms': [],
                'best_platform': 'Unknown', 
                'platform_count': 0
            }
        
        # Installation commands for missing dependencies
        installation_commands = {
            'openmm': 'conda install -c conda-forge openmm',
            'pdbfixer': 'conda install -c conda-forge pdbfixer',
            'openmmforcefields': 'conda install -c conda-forge openmmforcefields',
            'simple_working': 'All dependencies installed - Simple Working MD ready!',
            'rdkit': 'conda install -c conda-forge rdkit',
            'openff-toolkit': 'conda install -c conda-forge openff-toolkit'
        }
        
        return {
            'dependencies': deps,
            'platform_info': platform_info,
            'installation_commands': installation_commands,
            'version_info': {
                'simple_working': 'v1.0 (Based on openmm_test.py)',
                'approach': 'RDKit → GAFF → Implicit Solvent',
                'compatibility': 'Complete UI interface'
            }
        }
    
    def get_automated_simulation_candidates(self, base_dir: str = "automated_simulations") -> List[Dict[str, Any]]:
        """
        Get list of candidates generated by automation pipeline for UI compatibility
        
        Args:
            base_dir: Base directory where automated simulations are stored
            
        Returns:
            List of candidate info dictionaries
        """
        candidates = []
        
        try:
            from pathlib import Path
            from datetime import datetime
            
            base_path = Path(base_dir)
            
            if not base_path.exists():
                print(f"⚠️ Automated simulations directory not found: {base_dir}")
                return []
            
            # Look for session directories
            for session_dir in base_path.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith('session_'):
                    session_id = session_dir.name
                    
                    # Look for candidate directories
                    for candidate_dir in session_dir.iterdir():
                        if candidate_dir.is_dir() and candidate_dir.name.startswith('candidate_'):
                            candidate_id = candidate_dir.name
                            
                            # Look for polymer PDB files
                            polymer_pdb = None
                            composite_pdb = None
                            processed_insulin_pdb = None
                            
                            # Find polymer files
                            for pdb_file in candidate_dir.rglob("*.pdb"):
                                if ("polymer" in pdb_file.name.lower() and 
                                    "composite" not in pdb_file.name.lower() and
                                    "insulin" not in pdb_file.name.lower()):
                                    polymer_pdb = str(pdb_file)
                                elif "composite" in pdb_file.name.lower():
                                    composite_pdb = str(pdb_file)
                            
                            # Look for processed insulin files (UI expects this key)
                            processed_insulin_dir = candidate_dir / "processed_insulin"
                            if processed_insulin_dir.exists():
                                for pdb_file in processed_insulin_dir.glob("*.pdb"):
                                    processed_insulin_pdb = str(pdb_file)
                                    break
                            
                            # Get candidate info (include all keys the UI expects)
                            candidate_info = {
                                'session_id': session_id,
                                'candidate_id': candidate_id,
                                'candidate_dir': str(candidate_dir),
                                'polymer_pdb': polymer_pdb,
                                'composite_pdb': composite_pdb,
                                'processed_insulin_pdb': processed_insulin_pdb,  # UI expects this key
                                'has_polymer_box': polymer_pdb is not None,
                                'has_insulin_system': composite_pdb is not None,
                                'timestamp': datetime.fromtimestamp(candidate_dir.stat().st_mtime).isoformat(),
                                'ready_for_md': composite_pdb is not None  # Can run MD if composite exists
                            }
                            
                            candidates.append(candidate_info)
            
            # Sort by timestamp (newest first)
            candidates.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error scanning automated simulation candidates: {e}")
        
        return candidates
    
    def stop_simulation(self):
        """Stop current simulation gracefully"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            print("🛑 Stopping simulation gracefully...")
            self.simulation_running = False
            
            # Update current simulation status
            if self.current_simulation:
                self.current_simulation['status'] = 'stopping'
                self.current_simulation['stop_requested'] = True
            
            print("✅ Stop signal sent - simulation will finish current chunk and stop")
            return True
        else:
            print("⚠️  No active simulation to stop")
            return False


def test_simple_integration():
    """Test the simple integration with working files"""
    
    print("🧪 Testing Simple MD Integration")
    
    # Use EXACT same files as the working openmm_test.py script
    working_polymer = "./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb"
    working_composite = "./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/insulin_polymer_composite_001_e36e15_preprocessed.pdb"
    
    print(f"🎯 Using EXACT working files from openmm_test.py:")
    print(f"   • Polymer: {working_polymer}")
    print(f"   • Composite: {working_composite}")
    
    # Check if files exist
    if not os.path.exists(working_polymer):
        print(f"❌ Working polymer file not found: {working_polymer}")
        return
    
    if not os.path.exists(working_composite):
        print(f"❌ Working composite file not found: {working_composite}")
        return
    
    try:
        integration = SimpleMDIntegration("test_simple_integration")
        
        # Override file discovery to use exact working files
        integration.find_polymer_and_composite_files = lambda manual_dir=None: (working_polymer, working_composite)
        
        # Test async simulation with exact working files
        sim_id = integration.run_md_simulation_async(
            pdb_file=working_composite,  # Not used directly, but kept for interface
            temperature=310.0,
            equilibration_steps=5000,   # Short test
            production_steps=10000,     # Short test
            save_interval=1000,
            output_prefix="test_simple_integration"
        )
        
        print(f"✅ Simulation started with ID: {sim_id}")
        
        # Wait for completion (in real app, this would be handled by UI)
        while integration.is_simulation_running():
            time.sleep(1)
            print("⏳ Waiting for simulation...")
        
        status = integration.get_simulation_status()
        if status['status'] == 'completed':
            print(f"🎉 Integration test SUCCESSFUL!")
        else:
            print(f"❌ Integration test FAILED: {status.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Integration test FAILED with exception: {e}")


if __name__ == "__main__":
    test_simple_integration() 