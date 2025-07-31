#!/usr/bin/env python3
"""
Enhanced MD Simulation Integration with Stored SMILES

This module provides MD simulation capabilities that use pre-stored SMILES strings
from the PSMILES generation workflow instead of reconstructing them from PDB files.
This eliminates redundant work and improves reliability.
"""

import streamlit as st
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import os
import time
import traceback
from datetime import datetime

# Import DirectPolymerBuilder for SMILES storage
try:
    from utils.direct_polymer_builder import DirectPolymerBuilder
    POLYMER_BUILDER_AVAILABLE = True
except ImportError:
    POLYMER_BUILDER_AVAILABLE = False

# Import MD simulation components
try:
    from integration.analysis.simple_md_integration import SimpleMDIntegration
    MD_INTEGRATION_AVAILABLE = True
except ImportError:
    MD_INTEGRATION_AVAILABLE = False

# Import OpenMM and force field components  
try:
    import openmm as mm
    from openmm import app
    from openmm.app import ForceField
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


class EnhancedMDWithStoredSMILES:
    """
    Enhanced MD simulation system that uses pre-stored SMILES for efficient force field generation.
    Eliminates the need to reconstruct SMILES from PDB files.
    """
    
    def __init__(self, output_dir: str = "enhanced_md_simulations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check availability of required components
        self.smiles_storage_available = POLYMER_BUILDER_AVAILABLE
        self.md_integration_available = MD_INTEGRATION_AVAILABLE  
        self.openmm_available = OPENMM_AVAILABLE
        
        # Initialize base MD integration if available
        self.md_integration = SimpleMDIntegration() if MD_INTEGRATION_AVAILABLE else None
        
    def get_simulation_readiness_status(self) -> Dict[str, Any]:
        """Check readiness for MD simulation with stored SMILES approach"""
        return {
            'overall_ready': all([
                self.smiles_storage_available,
                self.md_integration_available, 
                self.openmm_available
            ]),
            'components': {
                'smiles_storage': self.smiles_storage_available,
                'md_integration': self.md_integration_available,
                'openmm': self.openmm_available
            },
            'missing_components': [
                comp for comp, available in {
                    'SMILES Storage System': self.smiles_storage_available,
                    'MD Integration': self.md_integration_available,
                    'OpenMM': self.openmm_available
                }.items() if not available
            ]
        }
    
    def setup_force_field_with_stored_smiles(
        self, 
        psmiles: str, 
        force_field_type: str = 'smirnoff',
        fallback_to_reconstruction: bool = True
    ) -> Tuple[Optional[ForceField], Dict[str, Any]]:
        """
        Setup force field using pre-stored SMILES instead of PDB reconstruction.
        
        Args:
            psmiles: PSMILES string to find stored SMILES for
            force_field_type: Type of force field ('smirnoff' or 'gaff')
            fallback_to_reconstruction: Whether to fallback to PDB reconstruction if SMILES not found
            
        Returns:
            Tuple of (ForceField object, status dict)
        """
        status = {
            'success': False,
            'method_used': None,
            'smiles_source': None,
            'error': None,
            'force_field_type': force_field_type
        }
        
        if not self.openmm_available:
            status['error'] = "OpenMM not available"
            return None, status
            
        try:
            print(f"🚀 Setting up force field for PSMILES: {psmiles}")
            
            # **STEP 1: Get SMILES from automation_results.json storage (primary method)**
            smiles = None
            if self.smiles_storage_available:
                # **ENHANCED: Look for stored polymer_smiles in automation_results.json files**
                print(f"🔍 DEBUG: Searching for PSMILES: {psmiles}")
                smiles = self._find_stored_polymer_smiles_for_psmiles(psmiles)
                if smiles:
                    print(f"✅ DEBUG: Found stored SMILES (length: {len(smiles)})")
                    print(f"🎯 DEBUG: SMILES: {smiles[:100]}...")
                else:
                    print(f"❌ DEBUG: No stored SMILES found for this PSMILES")
                if smiles:
                    status['smiles_source'] = 'stored_polymer_chain'
                    print(f"✅ Using stored polymer chain SMILES: {smiles[:50]}...")
                else:
                    print(f"⚠️ No stored polymer SMILES found for PSMILES: {psmiles}")
            
            # **STEP 2: Fallback to PDB reconstruction if needed and allowed**
            if not smiles and fallback_to_reconstruction and self.md_integration_available:
                print(f"🔄 Falling back to PDB reconstruction method")
                # Here you could call the old reconstruction method if needed
                # For now, we'll skip this to enforce the new workflow
                status['error'] = "No stored SMILES available and fallback disabled to encourage new workflow"
                return None, status
            
            if not smiles:
                status['error'] = "No SMILES available for force field generation"
                return None, status
            
            # **STEP 3: Create molecule from SMILES**
            print(f"🧬 Creating molecule from SMILES: {smiles}")
            try:
                molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
                print(f"✅ Created molecule: {molecule.n_atoms} atoms, {molecule.n_bonds} bonds")
            except Exception as e:
                status['error'] = f"Failed to create molecule from SMILES: {e}"
                return None, status
            
            # **STEP 4: Create force field with template generator**
            print(f"⚙️ Setting up {force_field_type.upper()} force field with template generator")
            
            try:
                if force_field_type.lower() == 'gaff':
                    template_generator = GAFFTemplateGenerator(
                        molecules=molecule, 
                        forcefield='gaff-2.11'
                    )
                    status['method_used'] = 'gaff_template_generator'
                else:  # Default to SMIRNOFF
                    template_generator = SMIRNOFFTemplateGenerator(
                        molecules=molecule, 
                        forcefield='openff-2.1.0'
                    )
                    status['method_used'] = 'smirnoff_template_generator'
                
                # Create base force field
                forcefield = ForceField(
                    'amber/protein.ff14SB.xml',  # For any proteins in system
                    'amber/tip3p_standard.xml'   # Water model
                )
                
                # Register template generator for small molecules
                forcefield.registerTemplateGenerator(template_generator.generator)
                
                print(f"✅ Force field setup complete with {force_field_type.upper()} template generator")
                status['success'] = True
                
                return forcefield, status
                
            except Exception as e:
                status['error'] = f"Failed to setup force field with template generator: {e}"
                return None, status
                
        except Exception as e:
            status['error'] = f"Unexpected error in force field setup: {e}"
            traceback.print_exc()
            return None, status
    
    def run_simulation_with_stored_smiles(
        self,
        psmiles: str,
        pdb_file: Optional[str] = None,
        simulation_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run MD simulation using stored SMILES for force field generation.
        
        Args:
            psmiles: PSMILES string to simulate
            pdb_file: Optional PDB file path (for structure, not SMILES extraction)
            simulation_params: Simulation parameters
            
        Returns:
            Simulation results dict
        """
        results = {
            'success': False,
            'psmiles': psmiles,
            'approach': 'stored_smiles_workflow',
            'error': None,
            'smiles_used': None,
            'force_field_status': None,
            'simulation_output': None
        }
        
        # Check readiness
        readiness = self.get_simulation_readiness_status()
        if not readiness['overall_ready']:
            results['error'] = f"System not ready: missing {readiness['missing_components']}"
            return results
        
        try:
            print(f"🚀 Starting MD simulation with stored SMILES workflow")
            print(f"   PSMILES: {psmiles}")
            print(f"   PDB file: {pdb_file or 'None provided'}")
            
            # **STEP 1: Setup force field using stored SMILES**
            force_field_type = simulation_params.get('force_field_type', 'smirnoff') if simulation_params else 'smirnoff'
            forcefield, ff_status = self.setup_force_field_with_stored_smiles(
                psmiles, 
                force_field_type=force_field_type
            )
            
            results['force_field_status'] = ff_status
            results['smiles_used'] = DirectPolymerBuilder().get_polymer_smiles_for_md(psmiles) if self.smiles_storage_available else None
            
            if not ff_status['success']:
                results['error'] = f"Force field setup failed: {ff_status['error']}"
                return results
            
            print(f"✅ Force field ready using {ff_status['method_used']} with {ff_status['smiles_source']} SMILES")
            
            # **STEP 2: Run actual MD simulation using base MD integration with enhanced force field**
            if self.md_integration_available and forcefield:
                print(f"🚀 Running actual MD simulation with enhanced force field setup...")
                
                # Convert enhanced simulation parameters to standard MD parameters
                temperature = simulation_params.get('temperature', 310.0)
                steps = simulation_params.get('steps', 5000)
                
                # Calculate equilibration and production steps
                equilibration_steps = max(1000, steps // 10)  # 10% for equilibration
                production_steps = steps - equilibration_steps
                save_interval = max(100, steps // 50)  # Save every 2% of simulation
                
                print(f"🔧 MD Parameters:")
                print(f"   • Temperature: {temperature} K")
                print(f"   • Equilibration: {equilibration_steps} steps")
                print(f"   • Production: {production_steps} steps") 
                print(f"   • Save interval: {save_interval} steps")
                
                try:
                    # **STEP 2A: Store enhanced force field and SMILES for use by MD integration**
                    # Store this in the MD integration for it to pick up during simulation
                    if hasattr(self.md_integration, 'set_enhanced_force_field'):
                        self.md_integration.set_enhanced_force_field(forcefield, results['smiles_used'], psmiles)
                        print(f"🎯 Enhanced force field stored in MD integration")
                    else:
                        print(f"⚠️ MD integration doesn't support enhanced force field - will use regular approach")
                    
                    # **STEP 2B: Actually run the real MD simulation with insulin composite system**
                    # For now, we need a PDB file for the MD simulation. The enhanced approach should
                    # create a PDB from the SMILES or use an existing insulin+polymer structure
                    
                    if not pdb_file:
                        # Generate a temporary PDB file from the SMILES for simulation
                        print(f"🧬 Generating temporary PDB structure from stored SMILES...")
                        temp_pdb = self._create_temp_pdb_from_smiles(results['smiles_used'], psmiles)
                        pdb_file = temp_pdb
                        print(f"✅ Temporary PDB created: {pdb_file}")
                    
                    # **CRITICAL: Actually run the real MD simulation with enhanced SMILES**
                    print(f"🚀 DEBUG: About to call MD integration with:")
                    print(f"   • enhanced_smiles: {results['smiles_used'][:100] if results['smiles_used'] else 'None'}...")
                    print(f"   • pdb_file: {pdb_file}")
                    print(f"   • smiles_source: {ff_status.get('smiles_source', 'unknown')}")
                    
                    simulation_id = self.md_integration.run_md_simulation_async(
                        pdb_file=pdb_file,
                        temperature=temperature,
                        equilibration_steps=equilibration_steps,
                        production_steps=production_steps,
                        save_interval=save_interval,
                        output_callback=lambda msg: print(f"📋 MD: {msg}"),
                        enhanced_smiles=results['smiles_used']  # **ENHANCED: Pass stored SMILES**
                    )
                    
                    print(f"🎯 Real MD simulation started with ID: {simulation_id}")
                    print(f"⚡ Using enhanced stored SMILES workflow: {ff_status['smiles_source']} SMILES")
                    
                    # Return immediately with running status - the simulation will continue async
                    simulation_output = {
                        'status': 'running', 
                        'method': 'enhanced_md_with_stored_smiles',
                        'simulation_id': simulation_id,
                        'force_field_type': force_field_type,
                        'smiles_source': ff_status['smiles_source'],
                        'steps_requested': steps,
                        'enhanced_workflow': True,
                        'note': 'Simulation running asynchronously - check MD Simulation tab for progress'
                    }
                    
                    results['simulation_output'] = simulation_output
                    results['success'] = True
                    results['simulation_id'] = simulation_id
                    
                    print(f"✅ Real MD simulation launched successfully using enhanced stored SMILES workflow")
                    print(f"📋 Track progress in the MD Simulation tab")
                    
                except Exception as md_error:
                    print(f"❌ MD simulation failed: {md_error}")
                    results['error'] = f"MD simulation execution failed: {md_error}"
                    results['success'] = False
                
            else:
                results['error'] = "MD integration not available for simulation execution"
                
        except Exception as e:
            results['error'] = f"Simulation failed: {e}"
            traceback.print_exc()
        
        return results
    
    def get_available_candidates_for_simulation(self) -> List[Dict[str, Any]]:
        """Get PSMILES candidates that are ready for MD simulation (have stored SMILES)"""
        ready_candidates = []
        
        # **PART 1: Get candidates from session state (live candidates)**
        session_candidates = self._get_session_state_candidates()
        ready_candidates.extend(session_candidates)
        
        # **PART 2: Get candidates from automated_simulations folder (persisted candidates)**
        disk_candidates = self._get_disk_candidates()
        ready_candidates.extend(disk_candidates)
        
        # Remove duplicates (prefer session state candidates over disk candidates)
        seen_ids = set()
        unique_candidates = []
        for candidate in ready_candidates:
            if candidate['id'] not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate['id'])
        
        return unique_candidates
    
    def _get_session_state_candidates(self) -> List[Dict[str, Any]]:
        """Get candidates from current session state"""
        candidates = []
        
        if 'psmiles_candidates' not in st.session_state:
            return candidates
        
        for i, candidate in enumerate(st.session_state.psmiles_candidates):
            psmiles = candidate.get('psmiles')
            has_smiles = bool(candidate.get('smiles'))
            conversion_success = candidate.get('smiles_conversion_success', False)
            
            if psmiles and has_smiles and conversion_success:
                candidates.append({
                    'index': i,
                    'id': candidate.get('id', f'session_candidate_{i}'),
                    'psmiles': psmiles,
                    'smiles': candidate['smiles'],
                    'conversion_method': candidate.get('smiles_conversion_method'),
                    'request': candidate.get('request', 'Unknown request'),
                    'timestamp': candidate.get('timestamp', 'Unknown'),
                    'ready_for_md': True,
                    'source': 'session_state'
                })
            else:
                candidates.append({
                    'index': i,
                    'id': candidate.get('id', f'session_candidate_{i}'),
                    'psmiles': psmiles or 'No PSMILES',
                    'smiles': candidate.get('smiles', 'No SMILES'),
                    'request': candidate.get('request', 'Unknown request'),
                    'timestamp': candidate.get('timestamp', 'Unknown'),
                    'ready_for_md': False,
                    'issue': 'Missing SMILES data' if not has_smiles else 'SMILES conversion failed',
                    'source': 'session_state'
                })
        
        return candidates
    
    def _get_disk_candidates(self) -> List[Dict[str, Any]]:
        """Get candidates from automated_simulations folder on disk"""
        candidates = []
        
        if not self.md_integration_available:
            return candidates
        
        try:
            # Use MD integration to scan automated_simulations folder
            disk_candidates = self.md_integration.get_automated_simulation_candidates()
            
            print(f"🔍 Found {len(disk_candidates)} candidates in automated_simulations folder")
            
            for disk_candidate in disk_candidates:
                candidate_id = disk_candidate['candidate_id']
                candidate_dir = Path(disk_candidate['candidate_dir'])
                
                # Try to load PSMILES and SMILES data from candidate directory
                psmiles, smiles, metadata = self._load_candidate_smiles_data(candidate_dir)
                
                # **ENHANCED: Check if stored polymer chain SMILES were used**
                # The metadata tells us if the SMILES came from stored polymer_smiles or conversion
                smiles_source = 'converted_from_psmiles'  # Default
                if metadata.get('conversion_method') == 'stored_polymer_smiles':
                    smiles_source = 'stored_polymer_chain'
                    print(f"✅ Using stored polymer chain SMILES from automation_results.json")
                elif not smiles:
                    # Try to convert PSMILES if no stored polymer smiles available
                    print(f"🔄 No stored polymer SMILES - trying PSMILES conversion...")
                
                if psmiles and smiles:
                    candidates.append({
                        'id': f"disk_{candidate_id}",
                        'psmiles': psmiles,
                        'smiles': smiles,
                        'smiles_source': smiles_source,  # Track where SMILES came from
                        'conversion_method': metadata.get('conversion_method', 'unknown'),
                        'request': metadata.get('request', f'Disk candidate {candidate_id}'),
                        'timestamp': disk_candidate['timestamp'],
                        'ready_for_md': True,
                        'source': 'automated_simulations_disk',
                        'candidate_dir': str(candidate_dir),
                        'polymer_pdb': disk_candidate.get('polymer_pdb'),
                        'composite_pdb': disk_candidate.get('composite_pdb')
                    })
                    print(f"✅ Loaded SMILES data for disk candidate: {candidate_id} (source: {smiles_source})")
                else:
                    candidates.append({
                        'id': f"disk_{candidate_id}",
                        'psmiles': 'No PSMILES data',
                        'smiles': 'No SMILES data',
                        'request': f'Disk candidate {candidate_id}',
                        'timestamp': disk_candidate['timestamp'],
                        'ready_for_md': False,
                        'issue': 'No PSMILES/SMILES data found on disk',
                        'source': 'automated_simulations_disk',
                        'candidate_dir': str(candidate_dir),
                        'polymer_pdb': disk_candidate.get('polymer_pdb'),
                        'composite_pdb': disk_candidate.get('composite_pdb')
                    })
                    print(f"⚠️ No SMILES data found for disk candidate: {candidate_id}")
                    
        except Exception as e:
            print(f"❌ Error scanning automated_simulations folder: {e}")
        
        return candidates
    
    def _load_candidate_smiles_data(self, candidate_dir: Path) -> tuple:
        """Try to load PSMILES and SMILES data from candidate directory"""
        psmiles = None
        smiles = None
        metadata = {}
        
        # Extract candidate ID from path (e.g., "candidate_001_6d4d0a" -> "001_6d4d0a")
        candidate_id = candidate_dir.name.replace('candidate_', '')
        session_dir = candidate_dir.parent
        
        # **PRIMARY SOURCE: Check automation_results.json in session directory**
        automation_results_file = session_dir / "automation_results.json"
        if automation_results_file.exists():
            try:
                import json
                with open(automation_results_file, 'r') as f:
                    data = json.load(f)
                
                # Look in polymer_boxes section
                for polymer_box in data.get('polymer_boxes', []):
                    if polymer_box.get('candidate_id') == candidate_id:
                        psmiles = polymer_box.get('psmiles')
                        
                        # **ENHANCED: First check for stored polymer chain SMILES**
                        stored_polymer_smiles = polymer_box.get('polymer_smiles')
                        if stored_polymer_smiles:
                            smiles = stored_polymer_smiles
                            metadata = {
                                'request': f"Automation candidate {candidate_id}",
                                'conversion_method': 'stored_polymer_smiles',
                                'source_file': 'automation_results.json',
                                'automation_data': polymer_box
                            }
                            print(f"📄 Loaded PSMILES from automation_results.json: {psmiles}")
                            print(f"✅ Using stored polymer chain SMILES: {smiles[:50]}...")
                        else:
                            # Fallback: Try to convert PSMILES to SMILES
                            print(f"🔄 No stored polymer SMILES - trying PSMILES conversion...")
                            smiles = self._try_convert_psmiles_to_smiles(psmiles)
                            metadata = {
                                'request': f"Automation candidate {candidate_id}",
                                'conversion_method': 'psmiles_to_smiles_conversion',
                                'source_file': 'automation_results.json',
                                'automation_data': polymer_box
                            }
                            print(f"📄 Loaded PSMILES from automation_results.json: {psmiles}")
                            if smiles:
                                print(f"✅ Converted to SMILES: {smiles[:50]}...")
                        break
                        
                # Also check insulin_systems section
                if not psmiles:
                    for insulin_system in data.get('insulin_systems', []):
                        system_candidate_id = insulin_system.get('candidate_id', '')
                        if candidate_id in system_candidate_id:
                            # Try to get PSMILES from associated polymer_box
                            for polymer_box in data.get('polymer_boxes', []):
                                if polymer_box.get('candidate_id') == candidate_id:
                                    psmiles = polymer_box.get('psmiles')
                                    
                                    # **ENHANCED: First check for stored polymer chain SMILES**
                                    stored_polymer_smiles = polymer_box.get('polymer_smiles')
                                    if stored_polymer_smiles:
                                        smiles = stored_polymer_smiles
                                        metadata = {
                                            'request': f"Insulin system candidate {candidate_id}",
                                            'conversion_method': 'stored_polymer_smiles',
                                            'source_file': 'automation_results.json',
                                            'automation_data': polymer_box
                                        }
                                        print(f"📄 Loaded PSMILES from insulin system: {psmiles}")
                                        print(f"✅ Using stored polymer chain SMILES: {smiles[:50]}...")
                                    else:
                                        # Fallback: Try to convert PSMILES to SMILES
                                        smiles = self._try_convert_psmiles_to_smiles(psmiles)
                                        metadata = {
                                            'request': f"Insulin system candidate {candidate_id}",
                                            'conversion_method': 'psmiles_to_smiles_conversion',
                                            'source_file': 'automation_results.json',
                                            'automation_data': polymer_box
                                        }
                                        print(f"📄 Loaded PSMILES from insulin system: {psmiles}")
                                        if smiles:
                                            print(f"✅ Converted to SMILES: {smiles[:50]}...")
                                    break
                        
            except Exception as e:
                print(f"❌ Error reading automation_results.json: {e}")
        
        # **FALLBACK: Look for other metadata files in candidate directory**
        if not psmiles:
            possible_files = [
                candidate_dir / "metadata.json",
                candidate_dir / "candidate_info.json", 
                candidate_dir / "psmiles_data.json",
                candidate_dir / "smiles_data.json"
            ]
            
            for metadata_file in possible_files:
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                            
                        psmiles = data.get('psmiles') or data.get('PSMILES')
                        smiles = data.get('smiles') or data.get('SMILES')
                        metadata = data
                        
                        if psmiles:
                            print(f"📄 Loaded PSMILES from fallback file: {metadata_file.name}")
                            if not smiles:
                                smiles = self._try_convert_psmiles_to_smiles(psmiles)
                            break
                            
                    except Exception as e:
                        print(f"❌ Error reading {metadata_file}: {e}")
        
        if not psmiles:
            print(f"⚠️ No PSMILES data found for candidate {candidate_id} in {candidate_dir}")
        
        return psmiles, smiles, metadata
    
    def _try_convert_psmiles_to_smiles(self, psmiles: str) -> str:
        """Try to convert PSMILES to SMILES using available conversion methods"""
        if not psmiles:
            return None
            
        try:
            # Use the DirectPolymerBuilder if available
            if POLYMER_BUILDER_AVAILABLE:
                from utils.direct_polymer_builder import DirectPolymerBuilder
                builder = DirectPolymerBuilder()
                smiles = builder.get_polymer_smiles_for_md(psmiles)
                if smiles:
                    print(f"✅ Converted PSMILES to SMILES using DirectPolymerBuilder")
                    return smiles
            
            # Try other conversion methods here if available
            # For now, return None if conversion fails
            print(f"⚠️ Could not convert PSMILES to SMILES: {psmiles[:50]}...")
            return None
            
        except Exception as e:
            print(f"❌ Error converting PSMILES to SMILES: {e}")
            return None
    
    def _find_stored_polymer_smiles_for_psmiles(self, target_psmiles: str) -> Optional[str]:
        """
        Search all automation_results.json files for stored polymer_smiles matching the target PSMILES.
        
        Args:
            target_psmiles: The PSMILES string to find stored polymer SMILES for
            
        Returns:
            The stored polymer chain SMILES string, or None if not found
        """
        try:
            import json
            from pathlib import Path
            
            # Find all automation_results.json files
            automated_simulations_dir = Path("automated_simulations")
            if not automated_simulations_dir.exists():
                return None
            
            automation_files = list(automated_simulations_dir.glob("*/automation_results.json"))
            
            for automation_file in automation_files:
                try:
                    with open(automation_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check polymer_boxes for matching PSMILES
                    for polymer_box in data.get('polymer_boxes', []):
                        if polymer_box.get('psmiles') == target_psmiles:
                            stored_polymer_smiles = polymer_box.get('polymer_smiles')
                            if stored_polymer_smiles:
                                print(f"💎 Found stored polymer SMILES in {automation_file.name}")
                                return stored_polymer_smiles
                                
                except Exception as e:
                    print(f"⚠️ Error reading {automation_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"❌ Error searching for stored polymer SMILES: {e}")
            return None
    
    def _create_temp_pdb_from_smiles(self, smiles: str, psmiles: str) -> str:
        """Create a temporary PDB file from SMILES for MD simulation"""
        try:
            from openff.toolkit import Molecule
            import tempfile
            import os
            
            print(f"🧬 Creating 3D structure from SMILES: {smiles[:50]}...")
            
            # Create molecule from SMILES
            molecule = Molecule.from_smiles(smiles)
            print(f"✅ Molecule created: {molecule.n_atoms} atoms")
            
            # Generate 3D conformer
            molecule.generate_conformers(n_conformers=1)
            print(f"✅ 3D conformer generated")
            
            # Create temporary PDB file
            temp_dir = self.output_dir / "temp_structures"
            temp_dir.mkdir(exist_ok=True)
            
            pdb_file = temp_dir / f"enhanced_md_{int(time.time())}.pdb"
            
            # Convert to PDB (simplified - in practice would need proper insulin embedding)
            topology = molecule.to_topology()
            with open(pdb_file, 'w') as f:
                # Write basic PDB header
                f.write("HEADER    ENHANCED MD STRUCTURE FROM STORED SMILES\n")
                f.write(f"REMARK   ORIGINAL PSMILES: {psmiles}\n")
                f.write(f"REMARK   SMILES: {smiles}\n")
                
                # Write atoms (simplified format)
                for i, atom in enumerate(molecule.atoms):
                    positions = molecule.conformers[0][i]
                    f.write(f"ATOM  {i+1:5d}  {atom.element.symbol:>2s}   UNK A   1    {positions[0]:8.3f}{positions[1]:8.3f}{positions[2]:8.3f}  1.00 20.00           {atom.element.symbol}\n")
                f.write("END\n")
            
            print(f"✅ Temporary PDB created: {pdb_file}")
            return str(pdb_file)
            
        except Exception as e:
            print(f"❌ Failed to create temporary PDB from SMILES: {e}")
            # Fallback - return None and let regular MD handle it
            return None


def create_enhanced_md_simulator():
    """Factory function to create EnhancedMDWithStoredSMILES instance"""
    return EnhancedMDWithStoredSMILES()


def check_enhanced_md_availability() -> Dict[str, Any]:
    """Check if enhanced MD with stored SMILES is available"""
    simulator = EnhancedMDWithStoredSMILES()
    return simulator.get_simulation_readiness_status() 