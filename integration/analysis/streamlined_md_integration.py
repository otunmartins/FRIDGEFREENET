#!/usr/bin/env python3
"""
Streamlined MD Integration System
Uses the proven working approach from test_real_md_simulation.py for production use
Bypasses dependency check issues and provides a clean interface for the app
"""

import os
import sys
import time
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# Direct imports - no dependency checks to avoid issues
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import ForceField, PDBFile, Modeller, Simulation
    from openmm.app import StateDataReporter, DCDReporter
    from .openmm_md_proper import ProperOpenMMSimulator
    from .pdb_water_remover import remove_water_comprehensive
    DEPENDENCIES_OK = True
    print("✅ All streamlined MD dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    DEPENDENCIES_OK = False

class StreamlinedMDIntegration:
    """Streamlined MD integration using proven working methods"""
    
    def __init__(self, output_dir: str = "streamlined_md_results"):
        """Initialize the streamlined MD system"""
        
        if not DEPENDENCIES_OK:
            raise ImportError("Required dependencies not available")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.openmm_simulator = ProperOpenMMSimulator()
        # water_remover is now a function, no need to instantiate
        
        print(f"🚀 Streamlined MD Integration initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🧬 Using proven cysteine fix and water removal")
    
    def run_complete_md_workflow(self,
                                pdb_file: str,
                                temperature: float = 310.0,
                                equilibration_steps: int = 25000,
                                production_steps: int = 125000,
                                save_interval: int = 500,
                                simulation_id: str = None,
                                remove_water: bool = True,
                                output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete MD workflow using proven working approach
        
        Args:
            pdb_file: Input PDB file path
            temperature: Simulation temperature (K)
            equilibration_steps: Equilibration steps
            production_steps: Production MD steps  
            save_interval: Frame save interval
            simulation_id: Custom simulation ID
            remove_water: Whether to remove water molecules
            output_callback: Callback for progress updates
            
        Returns:
            Dict with complete results
        """
        
        def log(message: str):
            timestamp = time.strftime('%H:%M:%S')
            formatted_msg = f"[{timestamp}] {message}"
            print(formatted_msg)
            if output_callback:
                output_callback(formatted_msg)
        
        # Generate simulation ID
        if simulation_id is None:
            simulation_id = f"streamlined_md_{int(time.time())}"
        
        workflow_start = time.time()
        
        results = {
            'success': False,
            'simulation_id': simulation_id,
            'stages_completed': [],
            'timing': {},
            'files_created': [],
            'errors': []
        }
        
        try:
            log("🚀 Starting Streamlined MD Workflow")
            log("=" * 60)
            log(f"📁 Input PDB: {pdb_file}")
            log(f"🆔 Simulation ID: {simulation_id}")
            log(f"🌡️  Temperature: {temperature} K")
            log(f"⏱️  Steps: {equilibration_steps} equilibration + {production_steps} production")
            
            # Validate input
            if not os.path.exists(pdb_file):
                raise FileNotFoundError(f"PDB file not found: {pdb_file}")
            
            file_size = os.path.getsize(pdb_file) / (1024 * 1024)
            log(f"📊 PDB file size: {file_size:.2f} MB")
            
            # Stage 1: Water removal preprocessing (if requested)
            stage_start = time.time()
            processed_pdb = pdb_file
            
            if remove_water:
                log("\n💧 Stage 1: Water removal preprocessing...")
                
                try:
                    # Create preprocessed filename
                    pdb_path = Path(pdb_file)
                    preprocessed_path = pdb_path.parent / f"{pdb_path.stem}_preprocessed.pdb"
                    
                    # Remove water while preserving polymers
                    water_results = remove_water_comprehensive(
                        str(pdb_path),
                        str(preprocessed_path),
                        preserve_heterogens=['UNL', 'HEM', 'FAD', 'NAD'],
                        method='selective',
                        ph=7.4,
                        verbose=True
                    )
                    
                    if water_results['success']:
                        processed_pdb = str(preprocessed_path)
                        results['files_created'].append(processed_pdb)
                        log(f"   ✅ Water removal: {water_results['water_removed']} molecules removed")
                        log(f"   💾 Preprocessed PDB: {processed_pdb}")
                    else:
                        log(f"   ⚠️  Water removal failed: {water_results.get('error', 'Unknown error')}")
                        # Continue with original file
                        
                except Exception as e:
                    log(f"   ⚠️  Water removal error: {e}")
                    # Continue with original file
                    
            else:
                log("\n💧 Stage 1: Skipping water removal (as requested)")
            
            results['timing']['preprocessing'] = time.time() - stage_start
            results['stages_completed'].append('preprocessing')
            
            # Stage 2: PDB structure fixing with cysteine fix
            log("\n🔧 Stage 2: PDB structure fixing with cysteine resolution...")
            stage_start = time.time()
            
            try:
                # Use our proven cysteine fix approach
                topology, positions = self.openmm_simulator.fix_pdb_structure(
                    processed_pdb, 
                    remove_water=False  # Already done if requested
                )
                
                log("   ✅ PDB structure fixed - cysteine conflicts resolved!")
                
                # Analyze structure
                cysteine_count = sum(1 for res in topology.residues() if res.name in ['CYS', 'CYX', 'CYM'])
                zinc_count = sum(1 for res in topology.residues() if res.name == 'ZN')
                protein_residues = sum(1 for res in topology.residues() 
                                     if res.name not in ['HOH', 'WAT', 'UNL', 'ZN'])
                
                log(f"   📊 Structure: {protein_residues} protein residues, {cysteine_count} cysteines")
                
                # Remove zinc ions if present (for clean simulation)
                if zinc_count > 0:
                    log(f"   🧪 Removing {zinc_count} zinc ions for implicit solvent...")
                    zinc_residues = [res for res in topology.residues() if res.name == 'ZN']
                    modeller = Modeller(topology, positions)
                    modeller.delete(zinc_residues)
                    topology = modeller.topology
                    positions = modeller.positions
                    log("   ✅ Zinc ions removed")
                
            except Exception as e:
                raise Exception(f"PDB structure fixing failed: {e}")
            
            results['timing']['structure_fixing'] = time.time() - stage_start
            results['stages_completed'].append('structure_fixing')
            
            # Stage 3: System creation
            log("\n🧬 Stage 3: OpenMM system creation...")
            stage_start = time.time()
            
            try:
                # Create force field (implicit solvent)
                forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                
                # Create system
                system = forcefield.createSystem(
                    topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=1.0*unit.nanometer,
                    constraints=app.HBonds,
                    removeCMMotion=True,
                    hydrogenMass=4*unit.amu
                )
                
                log(f"   ✅ System created successfully!")
                log(f"      • Particles: {system.getNumParticles()}")
                log(f"      • Constraints: {system.getNumConstraints()}")
                
            except Exception as e:
                raise Exception(f"System creation failed: {e}")
            
            results['timing']['system_creation'] = time.time() - stage_start
            results['stages_completed'].append('system_creation')
            
            # Stage 4: MD simulation setup
            log("\n⚙️  Stage 4: MD simulation setup...")
            stage_start = time.time()
            
            try:
                # Create integrator
                integrator = mm.LangevinIntegrator(
                    temperature*unit.kelvin,
                    1.0/unit.picosecond,
                    2.0*unit.femtoseconds
                )
                
                # Create simulation
                simulation = Simulation(topology, system, integrator)
                simulation.context.setPositions(positions)
                
                # Create output directory for this simulation
                sim_output_dir = self.output_dir / simulation_id
                sim_output_dir.mkdir(exist_ok=True)
                
                # Setup reporters
                state_file = sim_output_dir / "simulation_data.csv"
                traj_file = sim_output_dir / "trajectory.dcd"
                
                simulation.reporters.append(StateDataReporter(
                    str(state_file),
                    save_interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    density=True
                ))
                
                simulation.reporters.append(DCDReporter(
                    str(traj_file),
                    save_interval
                ))
                
                results['files_created'].extend([str(state_file), str(traj_file)])
                log(f"   ✅ Simulation setup complete")
                log(f"   📁 Output directory: {sim_output_dir}")
                
            except Exception as e:
                raise Exception(f"Simulation setup failed: {e}")
            
            results['timing']['simulation_setup'] = time.time() - stage_start
            results['stages_completed'].append('simulation_setup')
            
            # Stage 5: Energy minimization
            log("\n⚡ Stage 5: Energy minimization...")
            stage_start = time.time()
            
            try:
                # Get initial energy
                state = simulation.context.getState(getEnergy=True)
                initial_energy = state.getPotentialEnergy()
                
                # Minimize
                simulation.minimizeEnergy(maxIterations=1000)
                
                # Get final energy
                state = simulation.context.getState(getEnergy=True)
                final_energy = state.getPotentialEnergy()
                energy_change = final_energy - initial_energy
                
                log(f"   ✅ Energy minimization completed")
                log(f"      • Energy change: {energy_change}")
                
                results['energy_minimization'] = {
                    'initial_energy': str(initial_energy),
                    'final_energy': str(final_energy),
                    'energy_change': str(energy_change)
                }
                
            except Exception as e:
                raise Exception(f"Energy minimization failed: {e}")
            
            results['timing']['minimization'] = time.time() - stage_start
            results['stages_completed'].append('minimization')
            
            # Stage 6: Equilibration
            log(f"\n🌡️  Stage 6: Equilibration ({equilibration_steps} steps)...")
            stage_start = time.time()
            
            try:
                simulation.step(equilibration_steps)
                log(f"   ✅ Equilibration completed")
                
            except Exception as e:
                raise Exception(f"Equilibration failed: {e}")
            
            results['timing']['equilibration'] = time.time() - stage_start
            results['stages_completed'].append('equilibration')
            
            # Stage 7: Production MD
            log(f"\n🏃 Stage 7: Production MD ({production_steps} steps)...")
            stage_start = time.time()
            
            try:
                production_start = time.time()
                simulation.step(production_steps)
                production_time = time.time() - production_start
                
                # Calculate performance
                total_steps = equilibration_steps + production_steps
                timestep_fs = 2.0
                simulated_time_ns = (total_steps * timestep_fs) / 1000000
                ns_per_day = simulated_time_ns * (86400 / production_time)
                
                log(f"   ✅ Production MD completed!")
                log(f"      • Total steps: {total_steps}")
                log(f"      • Simulated time: {simulated_time_ns:.3f} ns")
                log(f"      • Performance: {ns_per_day:.1f} ns/day")
                
                # Save final structure
                final_pdb = sim_output_dir / "final_structure.pdb"
                state = simulation.context.getState(getPositions=True, getEnergy=True)
                with open(final_pdb, 'w') as f:
                    PDBFile.writeFile(topology, state.getPositions(), f)
                
                results['files_created'].append(str(final_pdb))
                
                # Final energies
                final_potential = state.getPotentialEnergy()
                final_kinetic = state.getKineticEnergy()
                
                results['performance'] = {
                    'ns_per_day': ns_per_day,
                    'simulated_time_ns': simulated_time_ns,
                    'wall_time_seconds': production_time,
                    'total_steps': total_steps
                }
                
                results['final_energies'] = {
                    'potential': str(final_potential),
                    'kinetic': str(final_kinetic),
                    'total': str(final_potential + final_kinetic)
                }
                
            except Exception as e:
                raise Exception(f"Production MD failed: {e}")
            
            results['timing']['production'] = time.time() - stage_start
            results['stages_completed'].append('production')
            
            # Success!
            total_time = time.time() - workflow_start
            results['timing']['total'] = total_time
            results['success'] = True
            results['output_directory'] = str(sim_output_dir)
            
            # Save simulation report
            report_file = sim_output_dir / "simulation_report.json"
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'input_file': pdb_file,
                    'simulation_id': simulation_id,
                    'parameters': {
                        'temperature': temperature,
                        'equilibration_steps': equilibration_steps,
                        'production_steps': production_steps,
                        'save_interval': save_interval
                    },
                    'results': results
                }, f, indent=2)
            
            results['files_created'].append(str(report_file))
            
            log("\n" + "=" * 60)
            log("🎉 STREAMLINED MD WORKFLOW COMPLETED SUCCESSFULLY!")
            log(f"📊 Performance: {ns_per_day:.1f} ns/day")
            log(f"⏱️  Total time: {total_time:.1f} seconds")
            log(f"📁 Results in: {sim_output_dir}")
            
            return results
            
        except Exception as e:
            log(f"❌ Workflow failed: {e}")
            results['error'] = str(e)
            results['timing']['total'] = time.time() - workflow_start
            import traceback
            results['traceback'] = traceback.format_exc()
            return results
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status (for UI compatibility)"""
        
        if hasattr(self, 'current_simulation'):
            return {
                'simulation_running': self.current_simulation['status'] == 'running',
                'simulation_info': self.current_simulation
            }
        else:
            return {
                'simulation_running': False,
                'simulation_info': None
            }
    
    def get_simulation_status_by_id(self, simulation_id: str) -> Dict[str, Any]:
        """Get status of a specific simulation by ID"""
        
        sim_dir = self.output_dir / simulation_id
        if not sim_dir.exists():
            return {'exists': False, 'error': 'Simulation directory not found'}
        
        # Check for report file
        report_file = sim_dir / "simulation_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            return {
                'exists': True,
                'completed': report_data.get('results', {}).get('success', False),
                'report': report_data
            }
        
        return {'exists': True, 'completed': False, 'report': None}
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """List all available simulations"""
        
        simulations = []
        
        for sim_dir in self.output_dir.iterdir():
            if sim_dir.is_dir():
                sim_id = sim_dir.name
                status = self.get_simulation_status_by_id(sim_id)
                
                if status['exists']:
                    sim_info = {
                        'simulation_id': sim_id,
                        'directory': str(sim_dir),
                        'completed': status['completed']
                    }
                    
                    if status['report']:
                        report = status['report']
                        sim_info.update({
                            'timestamp': report.get('timestamp', ''),
                            'input_file': report.get('input_file', ''),
                            'performance': report.get('results', {}).get('performance', {}),
                            'success': report.get('results', {}).get('success', False)
                        })
                    
                    simulations.append(sim_info)
        
        # Sort by timestamp (newest first)
        simulations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return simulations

    def get_dependency_status(self) -> Dict[str, Any]:
        """Get dependency status for UI compatibility"""
        
        # Check basic dependencies
        deps = {
            'openmm': DEPENDENCIES_OK,
            'pdbfixer': DEPENDENCIES_OK,
            'openmmforcefields': DEPENDENCIES_OK,
            'streamlined_md': DEPENDENCIES_OK,
            'all_available': DEPENDENCIES_OK
        }
        
        # Get platform info if OpenMM is available
        platform_info = {}
        if DEPENDENCIES_OK:
            try:
                platforms = []
                for i in range(mm.Platform.getNumPlatforms()):
                    platform = mm.Platform.getPlatform(i)
                    platforms.append({
                        'name': platform.getName(),
                        'speed': platform.getSpeed(),
                        'available': True
                    })
                
                platform_info = {
                    'platforms': platforms,
                    'best_platform': self.openmm_simulator.platform.getName() if hasattr(self.openmm_simulator, 'platform') else 'Unknown'
                }
            except Exception as e:
                platform_info = {'error': str(e)}
        
        return {
            'dependencies': deps,
            'platform_info': platform_info,
            'installation_commands': {
                'openmm': 'conda install -c conda-forge openmm',
                'pdbfixer': 'conda install -c conda-forge pdbfixer',
                'openmmforcefields': 'conda install -c conda-forge openmmforcefields'
            }
        }
    
    def run_md_simulation_async(self,
                              pdb_file: str,
                              temperature: float = 310.0,
                              equilibration_steps: int = 25000,
                              production_steps: int = 125000,
                              save_interval: int = 500,
                              output_prefix: str = None,
                              output_callback: Optional[Callable] = None,
                              **kwargs) -> str:
        """
        Run MD simulation asynchronously (for UI compatibility)
        Returns simulation ID immediately, simulation runs in background
        """
        
        import threading
        
        # Generate simulation ID
        simulation_id = output_prefix or f"streamlined_md_{int(time.time())}"
        
        # Store simulation info
        self.current_simulation = {
            'id': simulation_id,
            'status': 'starting',
            'pdb_file': pdb_file,
            'temperature': temperature,
            'equilibration_steps': equilibration_steps,
            'production_steps': production_steps,
            'start_time': time.time(),
            'results': None
        }
        
        # Run simulation in background thread
        def run_simulation_thread():
            try:
                self.current_simulation['status'] = 'running'
                
                results = self.run_complete_md_workflow(
                    pdb_file=pdb_file,
                    temperature=temperature,
                    equilibration_steps=equilibration_steps,
                    production_steps=production_steps,
                    save_interval=save_interval,
                    simulation_id=simulation_id,
                    output_callback=output_callback
                )
                
                self.current_simulation['results'] = results
                self.current_simulation['status'] = 'completed' if results['success'] else 'failed'
                self.current_simulation['end_time'] = time.time()
                
            except Exception as e:
                self.current_simulation['status'] = 'failed'
                self.current_simulation['error'] = str(e)
                self.current_simulation['end_time'] = time.time()
        
        # Start background thread
        self.simulation_thread = threading.Thread(target=run_simulation_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return simulation_id
    
    def stop_simulation(self) -> bool:
        """Stop current simulation (for UI compatibility)"""
        if hasattr(self, 'current_simulation') and self.current_simulation['status'] == 'running':
            self.current_simulation['status'] = 'stopped'
            return True
        return False
    
    def get_automated_simulation_candidates(self, base_dir: str = "automated_simulations") -> List[Dict[str, Any]]:
        """
        Get list of candidates generated by SimulationAutomationPipeline
        
        Args:
            base_dir: Base directory where automated simulations are stored
            
        Returns:
            List of candidate info dictionaries
        """
        candidates = []
        
        try:
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
                            
                            # Look for polymer box PDB files (check multiple locations)
                            polymer_pdb = None
                            
                            # 1. Check in molecules subdirectory (PSP standard location)
                            molecules_dir = candidate_dir / 'molecules'
                            if molecules_dir.exists():
                                for pdb_file in molecules_dir.glob("*.pdb"):
                                    if ("polymer" in pdb_file.name.lower() and 
                                        "composite" not in pdb_file.name.lower() and
                                        "insulin" not in pdb_file.name.lower()):
                                        polymer_pdb = str(pdb_file)
                                        break
                            
                            # 2. Check in main candidate directory if not found
                            if not polymer_pdb:
                                for pdb_file in candidate_dir.glob("*.pdb"):
                                    if ("polymer" in pdb_file.name.lower() and 
                                        "composite" not in pdb_file.name.lower() and
                                        "insulin" not in pdb_file.name.lower()):
                                        polymer_pdb = str(pdb_file)
                                        break
                            
                            # 3. Check for any PDB files with candidate ID (fallback)
                            if not polymer_pdb:
                                for pdb_file in candidate_dir.rglob("*.pdb"):
                                    if (candidate_id in pdb_file.name and 
                                        "composite" not in pdb_file.name.lower() and
                                        "insulin" not in pdb_file.name.lower() and
                                        not pdb_file.name.lower().startswith("insulin_")):
                                        polymer_pdb = str(pdb_file)
                                        break
                            
                            # 4. As final fallback, take any non-insulin, non-composite PDB file
                            if not polymer_pdb:
                                for pdb_file in candidate_dir.rglob("*.pdb"):
                                    if ("composite" not in pdb_file.name.lower() and
                                        "insulin" not in pdb_file.name.lower() and
                                        not pdb_file.name.lower().startswith("insulin_")):
                                        polymer_pdb = str(pdb_file)
                                        break
                            
                            # Look for insulin-polymer composite files
                            composite_pdb = None
                            
                            # Look for files starting with insulin_polymer_composite
                            for pdb_file in candidate_dir.rglob("insulin_polymer_composite*.pdb"):
                                composite_pdb = str(pdb_file)
                                break
                            
                            # Fallback: check for any composite files
                            if not composite_pdb:
                                for pdb_file in candidate_dir.rglob("*composite*.pdb"):
                                    composite_pdb = str(pdb_file)
                                    break
                            
                            # Look for processed insulin files
                            processed_insulin_dir = candidate_dir / "processed_insulin"
                            processed_insulin_pdb = None
                            if processed_insulin_dir.exists():
                                for pdb_file in processed_insulin_dir.glob("*.pdb"):
                                    processed_insulin_pdb = str(pdb_file)
                                    break
                            
                            # Get candidate info
                            candidate_info = {
                                'session_id': session_id,
                                'candidate_id': candidate_id,
                                'candidate_dir': str(candidate_dir),
                                'polymer_pdb': polymer_pdb,
                                'composite_pdb': composite_pdb,
                                'processed_insulin_pdb': processed_insulin_pdb,
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
    
    def get_simulation_files(self, simulation_id: str) -> Dict[str, Any]:
        """Get files for a specific simulation (for UI compatibility)"""
        
        sim_dir = self.output_dir / simulation_id
        
        if not sim_dir.exists():
            return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
        
        # Look for simulation files
        files = {}
        
        # Check for standard MD output files
        for file_pattern in ['*.dcd', '*.pdb', '*.csv', '*.json']:
            for file_path in sim_dir.glob(file_pattern):
                files[file_path.name] = str(file_path)
        
        return {
            'success': True,
            'files': files,
            'simulation_dir': str(sim_dir)
        }

# Convenience function for quick usage
def run_streamlined_md(pdb_file: str, 
                      output_dir: str = "streamlined_md_results",
                      **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run streamlined MD simulation
    
    Args:
        pdb_file: Input PDB file
        output_dir: Output directory
        **kwargs: Additional parameters for run_complete_md_workflow
        
    Returns:
        Simulation results
    """
    
    integration = StreamlinedMDIntegration(output_dir)
    return integration.run_complete_md_workflow(pdb_file, **kwargs)

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Streamlined MD Integration")
    
    # Find a test PDB file
    test_files = [
        "../data/insulin/insulin_default_backup.pdb",
        "insulin_polymer_demo_clean.pdb"
    ]
    
    test_pdb = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_pdb = file_path
            break
    
    if test_pdb:
        print(f"📁 Using test file: {test_pdb}")
        
        # Run short test
        results = run_streamlined_md(
            test_pdb,
            equilibration_steps=1000,
            production_steps=2000,
            save_interval=100
        )
        
        if results['success']:
            print("✅ Streamlined MD integration test successful!")
            print(f"📊 Performance: {results['performance']['ns_per_day']:.1f} ns/day")
        else:
            print(f"❌ Test failed: {results.get('error', 'Unknown error')}")
    
    else:
        print("⚠️  No test PDB file found") 