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

# Import our simple working simulator (REPLACEMENT for complex system)
try:
    from .simple_working_md_simulator import SimpleWorkingMDSimulator
    SIMPLE_WORKING_AVAILABLE = True
except ImportError:
    SIMPLE_WORKING_AVAILABLE = False

# Keep old import for backward compatibility (but prefer simple approach)
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
            output_dir: Base directory for storing simulation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check dependencies
        self.dependencies = self._check_dependencies()
        
        if not self.dependencies['simulation_available']:
            missing = [k for k, v in self.dependencies.items() if not v]
            print(f"⚠️ Missing dependencies: {missing}")
            raise ImportError(f"Missing critical dependencies for MD simulation: {missing}")
        
        # Use simple working simulator if available (preferred), otherwise fallback
        if SIMPLE_WORKING_AVAILABLE:
            self.openmm_simulator = SimpleWorkingMDSimulator(str(self.output_dir))
            print(f"✅ Using simple working MD simulator")
        else:
            # Fallback to complex system if simple not available
            self.openmm_simulator = ProperOpenMMSimulator(str(self.output_dir))
            print(f"⚠️  Using complex fallback system")
        
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
            'simple_working': SIMPLE_WORKING_AVAILABLE,
            'proper_openmm': PROPER_OPENMM_AVAILABLE
        }
        # Prefer simple working simulator, fallback to complex
        deps['simulation_available'] = (
            deps['openmm'] and 
            deps['pdbfixer'] and 
            (deps['simple_working'] or deps['proper_openmm'])
        )
        return deps
    
    def get_dependencies_status(self) -> Dict[str, Any]:
        """Get detailed dependency status for UI display"""
        return {
            'dependencies': self.dependencies,
            'capabilities': {
                'md_simulation': self.dependencies['simulation_available'],
                'structure_preparation': self.dependencies['pdbfixer'],
                'simple_md': self.dependencies['simple_working'],
                'advanced_md': self.dependencies['proper_openmm']
            }
        }
    
    def start_simulation_async(self, 
                              pdb_content: str, 
                              simulation_params: Optional[Dict] = None,
                              simulation_id: Optional[str] = None,
                              output_callback: Optional[Callable] = None,
                              enable_binding_energy: bool = False) -> str:
        """
        Start MD simulation asynchronously with real-time output
        
        Args:
            pdb_content: PDB file content as string
            simulation_params: Simulation parameters (timestep, time, temperature, etc.)
            simulation_id: Optional custom simulation ID
            output_callback: Optional callback for real-time output
            enable_binding_energy: Whether to calculate binding energies (not MM-GBSA)
            
        Returns:
            simulation_id for tracking
        """
        
        # Generate simulation ID if not provided
        if simulation_id is None:
            simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        
        # Store simulation parameters
        if simulation_params is None:
            simulation_params = {
                'timestep_fs': 2.0,
                'equilibration_steps': 10000,
                'production_steps': 100000,
                'temperature_k': 310.0,
                'pressure_bar': 1.0,
                'output_frequency': 1000
            }
        
        # Store current simulation info
        self.current_simulation = {
            'id': simulation_id,
            'status': 'starting',
            'start_time': time.time(),
            'params': simulation_params,
            'enable_binding_energy': enable_binding_energy,
            'results': None,
            'error': None
        }
        
        # Clear previous messages
        self.latest_messages = []
        
        # Start simulation in separate thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(pdb_content, simulation_params, simulation_id, output_callback, enable_binding_energy)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        return simulation_id
    
    def _run_simulation_thread(self, pdb_content: str, simulation_params: Dict,
                             simulation_id: str, output_callback: Optional[Callable],
                             enable_binding_energy: bool = False):
        """Run the simulation in a separate thread"""
        
        def log_output(message: str):
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            
            # Store message for later retrieval
            self.latest_messages.append(formatted_message)
            
            # Keep only last 1000 messages to prevent memory issues
            if len(self.latest_messages) > 1000:
                self.latest_messages = self.latest_messages[-1000:]
            
            print(formatted_message)  # Always print to console
            if output_callback:
                output_callback(formatted_message)  # Also send to app interface
        
        try:
            log_output(f"🚀 Starting MD Simulation: {simulation_id}")
            log_output("=" * 80)
            
            # Update status
            self.current_simulation['status'] = 'preprocessing'
            
            # Step 1: Create temporary working directory
            sim_dir = self.output_dir / simulation_id
            sim_dir.mkdir(exist_ok=True)
            
            log_output(f"📁 Created simulation directory: {sim_dir}")
            
            # Step 2: Save input PDB and preprocess
            input_pdb = sim_dir / "input.pdb"
            with open(input_pdb, 'w') as f:
                f.write(pdb_content)
            
            log_output(f"💾 Saved input PDB file: {input_pdb}")
            
            # Step 3: Run MD simulation
            log_output(f"\n🔬 Step 3: Starting MD simulation")
            self.current_simulation['status'] = 'running_md'
            
            # Pass enable_binding_energy but this doesn't refer to MM-GBSA anymore
            simulation_results = self.openmm_simulator.run_simulation(
                str(input_pdb),
                simulation_params=simulation_params,
                simulation_id=simulation_id,
                output_callback=log_output,
                enable_binding_energy=enable_binding_energy
            )
            
            # Store results
            self.current_simulation['results'] = simulation_results
            
            if simulation_results.get('success', False):
                log_output(f"✅ MD simulation completed successfully!")
                
                # Log key results
                if 'final_energy' in simulation_results:
                    energy = simulation_results['final_energy']
                    log_output(f"🔋 Final potential energy: {energy['potential_energy']:.1f} kJ/mol")
                    log_output(f"🔋 Final kinetic energy: {energy['kinetic_energy']:.1f} kJ/mol")
                    log_output(f"🔋 Energy change: {energy['minimization_change']:.1f} kJ/mol")
                
            else:
                log_output(f"❌ MD simulation failed: {simulation_results.get('error', 'Unknown error')}")
                
            # Set final status
            if simulation_results.get('success', False):
                self.current_simulation['status'] = 'completed'
            else:
                self.current_simulation['status'] = 'failed'
            
            self.current_simulation['end_time'] = time.time()
            
            # Generate summary report
            total_time = time.time() - self.current_simulation['start_time']
            log_output(f"\n📊 Simulation Summary:")
            log_output(f"   Simulation ID: {simulation_id}")
            log_output(f"   Status: {self.current_simulation['status']}")
            log_output(f"   Total time: {total_time:.1f} seconds")
            log_output(f"   Results directory: {sim_dir}")
            
            # Save simulation summary
            summary = {
                'simulation_id': simulation_id,
                'status': self.current_simulation['status'],
                'start_time': self.current_simulation['start_time'],
                'end_time': self.current_simulation.get('end_time'),
                'total_time': total_time,
                'parameters': simulation_params,
                'results': simulation_results,
                'simulation_directory': str(sim_dir)
            }
            
            summary_file = sim_dir / f"simulation_summary_{simulation_id}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            log_output(f"💾 Simulation summary saved: {summary_file}")
            
        except Exception as e:
            error_msg = f"❌ MD simulation failed with error: {str(e)}"
            log_output(error_msg)
            
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
            self.current_simulation['end_time'] = time.time()
            
            # Log full traceback for debugging
            import traceback
            log_output(f"Full error traceback: {traceback.format_exc()}")
        
        finally:
            self.simulation_running = False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        if hasattr(self, 'current_simulation') and self.current_simulation:
            return {
                'simulation_running': self.simulation_running,
                'simulation_info': self.current_simulation,
                'latest_messages': self.latest_messages[-10:] if self.latest_messages else []
            }
        else:
            return {
                'simulation_running': False,
                'simulation_info': None,
                'latest_messages': []
            }
    
    def stop_simulation(self):
        """Stop the current simulation"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_running = False
            # Update current simulation status if available
            if hasattr(self, 'current_simulation') and self.current_simulation:
                self.current_simulation['status'] = 'stopping'
            print("🛑 MD simulation stop requested")
            return True
        else:
            print("⚠️ No active MD simulation to stop")
            return False
    
    def analyze_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Analyze simulation results including trajectory data"""
        try:
            # Find simulation directory
            sim_dir = self.output_dir / simulation_id
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            # Look for simulation report
            report_file = sim_dir / "simulation_report.json"
            if not report_file.exists():
                return {'success': False, 'error': f'Simulation report not found: {report_file}'}
            
            # Load simulation data
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Create analysis result
            analysis_result = {
                'success': True,
                'simulation_id': simulation_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'basic_info': {
                    'simulation_success': report_data.get('success', False),
                    'simulation_time_ps': report_data.get('production_stats', {}).get('simulation_time', 0),
                    'total_steps': report_data.get('production_stats', {}).get('total_steps', 0),
                    'final_energy': report_data.get('production_stats', {}).get('final_energy', 'N/A'),
                    'final_temperature': report_data.get('production_stats', {}).get('final_temperature', 'N/A'),
                    'performance_ns_day': report_data.get('performance', {}).get('ns_per_day', 'N/A'),
                    'final_atoms': report_data.get('system_info', {}).get('final_atoms', 'N/A')
                },
                'trajectory_analysis': report_data.get('trajectory_analysis', {}),
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
        """Get list of available completed simulations"""
        simulations = []
        
        try:
            for sim_dir in self.output_dir.iterdir():
                if sim_dir.is_dir():
                    # Look for simulation report
                    report_file = sim_dir / "simulation_report.json"
                    summary_file = sim_dir / f"simulation_summary_{sim_dir.name}.json"
                    
                    sim_info = {
                        'id': sim_dir.name,
                        'path': str(sim_dir),
                        'has_report': report_file.exists(),
                        'has_summary': summary_file.exists()
                    }
                    
                    # Get basic info from report if available
                    if report_file.exists():
                        try:
                            with open(report_file, 'r') as f:
                                report_data = json.load(f)
                            
                            sim_info.update({
                                'timestamp': report_data.get('timestamp', ''),
                                'success': report_data.get('success', False),
                                'simulation_time_ps': report_data.get('production_stats', {}).get('simulation_time', 0),
                                'performance': report_data.get('performance', {}).get('ns_per_day', 0),
                                'total_atoms': report_data.get('system_info', {}).get('final_atoms', 0)
                            })
                            
                        except Exception as e:
                            sim_info['report_error'] = str(e)
                    
                    simulations.append(sim_info)
            
            # Sort by timestamp (newest first)
            simulations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            print(f"Error scanning simulations: {e}")
        
        return simulations
    
    def get_simulation_files(self, simulation_id: str) -> Dict[str, Any]:
        """Get list of output files for a simulation"""
        try:
            sim_dir = self.output_dir / simulation_id
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            files = {}
            for file_path in sim_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(sim_dir)
                    files[str(rel_path)] = str(file_path)
            
            return {
                'success': True,
                'simulation_id': simulation_id,
                'files': files,
                'simulation_dir': str(sim_dir)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

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
    
    status = integration.get_dependencies_status()
    print(f"Dependencies: {status['dependencies']}")
    
    # Find available PDB files
    pdb_files = get_insulin_polymer_pdb_files()
    print(f"Found {len(pdb_files)} insulin-polymer PDB files")
    
    for pdb_file in pdb_files[:3]:  # Show first 3
        print(f"  📁 {pdb_file['name']} ({pdb_file['atom_count']} atoms)") 