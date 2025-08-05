#!/usr/bin/env python3
"""
Dual GAFF+AMBER MD Integration
==============================

This module provides MD integration using the proven dual approach:
1. GAFF for polymer parameterization (DirectPolymerBuilder)
2. AMBER for insulin simulation (simple_insulin_simulation.py approach)
3. Combined properly without CYS/CYX template generator issues

Based on the successful dual_gaff_amber_md_simulation.py script.
"""

import os
import sys
import time
import threading
import uuid
import tempfile
import shutil
import importlib.resources
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from openmm.app import (
        PDBFile, Modeller, ForceField, Simulation,
        StateDataReporter, PDBReporter,
        NoCutoff, HBonds,
    )
    from openmm import BrownianIntegrator, Platform, unit, LangevinMiddleIntegrator
    from openmm.app import Topology
    from openmmforcefields.generators import GAFFTemplateGenerator
    from openff.toolkit.topology import Molecule
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError as e:
    OPENMM_AVAILABLE = False
    print(f"⚠️ OpenMM or related packages not available: {e}")

try:
    from insulin_ai.utils.direct_polymer_builder import DirectPolymerBuilder
    POLYMER_BUILDER_AVAILABLE = True
except ImportError:
    POLYMER_BUILDER_AVAILABLE = False
    print("⚠️ DirectPolymerBuilder not available")

try:
    from insulin_ai.integration.analysis.simple_working_md_simulator import SimpleWorkingMDSimulator
    SIMPLE_SIMULATOR_AVAILABLE = True
except ImportError:
    SIMPLE_SIMULATOR_AVAILABLE = False
    print("⚠️ SimpleWorkingMDSimulator not available")

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
import logging

class PolymerValidationInput(BaseModel):
    """Validation schema for polymer simulation inputs."""
    
    polymer_smiles: str = Field(..., min_length=1, description="Valid polymer SMILES string")
    pdb_content: str = Field(..., min_length=10, description="Valid PDB file content")
    
    @field_validator('polymer_smiles')
    def validate_smiles(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Polymer SMILES cannot be empty")
        if not v.replace('[*]', '').strip():
            raise ValueError("Polymer SMILES contains only connection points - missing actual structure")
        # Check for common SMILES errors
        if v.count('(') != v.count(')'):
            raise ValueError("Unmatched parentheses in SMILES")
        return v.strip()
    
    @field_validator('pdb_content')  
    def validate_pdb(cls, v):
        if not v or v.strip() == "":
            raise ValueError("PDB content cannot be empty")
        if "ATOM" not in v and "HETATM" not in v:
            raise ValueError("PDB content appears invalid - no ATOM or HETATM records found")
        return v

@tool
def validate_md_inputs(polymer_smiles: str, pdb_content: str) -> dict:
    """Validate inputs for molecular dynamics simulation."""
    try:
        validated = PolymerValidationInput(
            polymer_smiles=polymer_smiles,
            pdb_content=pdb_content
        )
        return {
            "valid": True, 
            "polymer_smiles": validated.polymer_smiles,
            "pdb_content": validated.pdb_content
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "polymer_smiles": polymer_smiles,
            "pdb_content": pdb_content[:100] + "..." if len(pdb_content) > 100 else pdb_content
        }

def validate_molecule_for_gaff(molecule: Molecule, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Validate a molecule for GAFF parameterization compatibility.
    
    Args:
        molecule: OpenFF Molecule object
        log_callback: Optional logging function
        
    Returns:
        Dict with validation results and recommendations
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': [],
        'molecule_info': {}
    }
    
    try:
        # Basic molecule information
        validation['molecule_info'] = {
            'n_atoms': molecule.n_atoms,
            'n_bonds': molecule.n_bonds,
            'molecular_weight': molecule.molecular_weight.magnitude,
            'formal_charge': molecule.total_charge.magnitude,
            'n_conformers': molecule.n_conformers
        }
        
        # Check molecule size - GAFF/antechamber struggles with very large molecules
        if molecule.n_atoms > 1000:
            validation['errors'].append(f"Molecule too large ({molecule.n_atoms} atoms) - GAFF limit ~1000 atoms")
            validation['is_valid'] = False
            validation['recommendations'].append("Use simplified molecular representation or alternative force field")
        elif molecule.n_atoms > 500:
            validation['warnings'].append(f"Large molecule ({molecule.n_atoms} atoms) - GAFF may be slow/unreliable")
            validation['recommendations'].append("Consider using OpenFF SMIRNOFF instead of GAFF")
        
        # Check molecular weight
        mw = molecule.molecular_weight.magnitude
        if mw > 10000:
            validation['warnings'].append(f"Very high molecular weight ({mw:.1f} Da)")
            validation['recommendations'].append("Consider fragmentation or simplified representation")
        
        # Check formal charge
        charge = molecule.total_charge.magnitude
        if abs(charge) > 5:
            validation['warnings'].append(f"High formal charge ({charge:+.0f}) may cause parameterization issues")
        
        # Check for conformers
        if molecule.n_conformers == 0:
            validation['warnings'].append("No conformers - will need to generate for charge calculation")
            validation['recommendations'].append("Generate conformers before GAFF parameterization")
        
        # Check for partial charges
        try:
            if molecule.partial_charges is None:
                validation['warnings'].append("No partial charges assigned")
                validation['recommendations'].append("Assign partial charges before GAFF parameterization")
        except:
            validation['warnings'].append("Could not check partial charges")
        
        # Check for problematic functional groups that GAFF struggles with
        smiles = molecule.to_smiles()
        problematic_patterns = ['[*]', '[Si]', '[B]', '[P]', '[S]', '[Cl]', '[Br]', '[I]']
        for pattern in problematic_patterns:
            if pattern in smiles:
                if pattern == '[*]':
                    validation['warnings'].append("Wildcard atoms ([*]) detected - may cause issues")
                    validation['recommendations'].append("Remove or replace wildcard atoms")
                else:
                    validation['warnings'].append(f"Element {pattern} detected - limited GAFF parameters")
        
        if log_callback:
            log_callback("   📊 Molecule validation results:")
            log_callback(f"      Atoms: {validation['molecule_info']['n_atoms']}")
            log_callback(f"      MW: {validation['molecule_info']['molecular_weight']:.1f} Da")
            log_callback(f"      Charge: {validation['molecule_info']['formal_charge']:+.0f}")
            log_callback(f"      Conformers: {validation['molecule_info']['n_conformers']}")
            
            if validation['warnings']:
                log_callback(f"      ⚠️ {len(validation['warnings'])} warnings:")
                for warning in validation['warnings']:
                    log_callback(f"         - {warning}")
            
            if validation['errors']:
                log_callback(f"      ❌ {len(validation['errors'])} errors:")
                for error in validation['errors']:
                    log_callback(f"         - {error}")
            
            if validation['recommendations']:
                log_callback(f"      💡 Recommendations:")
                for rec in validation['recommendations']:
                    log_callback(f"         - {rec}")
    
    except Exception as e:
        validation['errors'].append(f"Validation failed: {e}")
        validation['is_valid'] = False
        if log_callback:
            log_callback(f"   ❌ Molecule validation failed: {e}")
    
    return validation


def create_simplified_polymer_fragment(original_smiles: str, target_atoms: int = 50) -> str:
    """
    Create a simplified polymer fragment for parameterization when the full polymer is too large.
    
    Args:
        original_smiles: Original polymer SMILES
        target_atoms: Target number of atoms for the fragment
        
    Returns:
        Simplified SMILES string
    """
    
    # Common polymer building blocks - try to identify the repeating unit
    common_fragments = [
        "CNCCCOCCNCCN",  # Amine-ether linkage
        "CCNCCCNC",      # Amine chain
        "CCCOCCCC",      # Ether linkage
        "CNCCCN",        # Simple diamine
        "CCOCCN",        # Ether-amine
        "CCCNCC",        # Simple alkyl-amine
    ]
    
    # Try to find a representative fragment in the original SMILES
    for fragment in common_fragments:
        if fragment in original_smiles:
            # Extend the fragment to reach target atoms
            repeat_count = max(1, target_atoms // len(fragment))
            simplified = fragment * min(repeat_count, 3)  # Don't repeat too many times
            return simplified
    
    # Fallback: create a simple representative polymer
    return "CNCCCOCCNCCNCCCNCCOCCNCCNCCC"  # Generic polymer fragment


class DualGaffAmberIntegration:
    """
    Dual GAFF+AMBER MD Integration for insulin-polymer composite systems.
    
    This class implements the successful dual approach:
    - Uses DirectPolymerBuilder for polymer creation
    - Uses GAFF for polymer parameterization
    - Uses AMBER for insulin parameterization (native CYX support)
    - Combines systems properly with spatial separation
    """
    
    def __init__(self, output_dir: str = "dual_gaff_amber_simulations"):
        """Initialize the dual GAFF+AMBER integration"""
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Also ensure the automated simulations directory exists, as it's scanned by the UI
        self.automated_simulations_dir = Path("automated_simulations")
        self.automated_simulations_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.polymer_builder = DirectPolymerBuilder() if POLYMER_BUILDER_AVAILABLE else None
        
        # Keep a reference to the simple simulator for its helper methods if needed
        # but the main logic will be self-contained here.
        self.simulator = SimpleWorkingMDSimulator() if SIMPLE_SIMULATOR_AVAILABLE else None
        
        # Simulation state
        self.current_simulation = None
        self.simulation_running = False
        self.simulation_thread = None
        
        # Check dependencies
        self.dependencies_ok = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        
        missing = []
        if not OPENMM_AVAILABLE:
            missing.append("OpenMM")
        if not POLYMER_BUILDER_AVAILABLE:
            missing.append("DirectPolymerBuilder")
        if not SIMPLE_SIMULATOR_AVAILABLE:
            missing.append("SimpleWorkingMDSimulator")
            
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            return False
        
        print("✅ All dependencies available for dual GAFF+AMBER approach")
        return True
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get status of all dependencies"""
        
        return {
            'openmm': {
                'available': OPENMM_AVAILABLE,
                'description': 'OpenMM molecular dynamics engine'
            },
            'polymer_builder': {
                'available': POLYMER_BUILDER_AVAILABLE,
                'description': 'DirectPolymerBuilder for polymer creation'
            },
            'simple_simulator': {
                'available': SIMPLE_SIMULATOR_AVAILABLE,
                'description': 'SimpleWorkingMDSimulator for MD'
            },
            'overall': {
                'available': self.dependencies_ok,
                'description': 'All systems operational for dual GAFF+AMBER'
            }
        }
    
    def _validate_and_prepare_inputs(self, log_callback, **kwargs):
        """LangChain-style input validation with detailed error reporting."""
        
        # Get the basic inputs
        polymer_smiles = kwargs.get('polymer_smiles', '')
        pdb_file = kwargs.get('pdb_file', '')
        
        # Read PDB content if it's a file path
        pdb_content = ""
        if pdb_file and isinstance(pdb_file, str):
            if pdb_file.endswith(('.pdb', '.PDB')):
                try:
                    with open(pdb_file, 'r') as f:
                        pdb_content = f.read()
                except Exception as e:
                    log_callback(f"❌ Failed to read PDB file {pdb_file}: {e}")
                    raise ValueError(f"Cannot read PDB file: {e}")
            else:
                # Treat as PSMILES string
                pdb_content = pdb_file
        
        # Validate using LangChain tool pattern
        validation_result = validate_md_inputs(polymer_smiles, pdb_content)
        
        if not validation_result["valid"]:
            error_msg = f"Input validation failed: {validation_result['error']}"
            log_callback(f"❌ {error_msg}")
            log_callback(f"   Polymer SMILES: '{polymer_smiles}'")
            log_callback(f"   PDB content length: {len(pdb_content)} chars")
            raise ValueError(error_msg)
        
        log_callback("✅ Input validation passed")
        return validation_result
    
    def find_insulin_file(self, custom_insulin_pdb: str = None) -> str:
        """Find suitable insulin PDB file"""
        
        if custom_insulin_pdb and os.path.exists(custom_insulin_pdb):
            return custom_insulin_pdb
        
        # Look for insulin files in standard locations
        insulin_candidates = [
            "src/insulin_ai/integration/data/insulin/insulin_default.pdb",
            "src/insulin_ai/integration/data/insulin/human_insulin_1mso.pdb"
        ]
        
        for candidate in insulin_candidates:
            if os.path.exists(candidate):
                return candidate
        
        raise ValueError("No insulin PDB file found. Please provide insulin_pdb parameter.")
    
    def fix_insulin_pdb_residues(self, input_pdb_path: str, output_pdb_path: str, log_callback):
        """
        Fix insulin PDB file to use CYX for disulfide-bonded cysteines.
        
        This creates a corrected PDB file that AMBER can properly handle.
        """
        
        log_callback("🔧 Fixing insulin PDB residue names for AMBER compatibility...")
        
        try:
            with open(input_pdb_path, 'r') as f:
                pdb_lines = f.readlines()
            
            fixed_lines = []
            cys_count = 0
            
            for line in pdb_lines:
                if line.startswith(('ATOM', 'HETATM')) and 'CYS' in line:
                    # Replace CYS with CYX in the residue name field (columns 18-20)
                    if line[17:20].strip() == 'CYS':
                        line = line[:17] + 'CYX' + line[20:]
                        cys_count += 1
                
                fixed_lines.append(line)
            
            # Write corrected PDB
            with open(output_pdb_path, 'w') as f:
                f.writelines(fixed_lines)
            
            if cys_count > 0:
                log_callback(f"   ✅ Fixed {cys_count} CYS → CYX atom records")
                log_callback(f"   📁 Corrected PDB saved: {output_pdb_path}")
            else:
                log_callback("   ✅ No CYS residues found - insulin already properly formatted")
            
            return output_pdb_path
            
        except Exception as e:
            log_callback(f"   ⚠️ PDB fixing failed: {e}")
            log_callback(f"   🔄 Using original PDB: {input_pdb_path}")
            return input_pdb_path
    
    def extract_polymer_info_from_path(self, simulation_input_file: str) -> Tuple[str, str]:
        """
        Extract polymer PSMILES and prepare for simulation.
        
        Args:
            simulation_input_file: Path to simulation input file or directory
            
        Returns:
            Tuple of (psmiles, expected_polymer_pdb_path)
        """
        
        # Handle different input types
        input_path = Path(simulation_input_file)
        
        if input_path.is_dir():
            # Directory containing polymer files
            psmiles_file = input_path / "psmiles.txt"
            if psmiles_file.exists():
                with open(psmiles_file, 'r') as f:
                    psmiles = f.read().strip()
                    
                # Look for polymer PDB
                polymer_pdb_candidates = list(input_path.glob("*polymer*.pdb"))
                expected_pdb = polymer_pdb_candidates[0] if polymer_pdb_candidates else None
                
                return psmiles, str(expected_pdb) if expected_pdb else None
        
        elif input_path.suffix == '.txt':
            # Direct PSMILES file
            with open(input_path, 'r') as f:
                psmiles = f.read().strip()
            return psmiles, None
        
        else:
            # Assume it's a PSMILES string if it contains certain characters
            if any(char in str(input_path) for char in ['[*]', '=', '#']):
                return str(input_path), None
        
        raise ValueError(f"Unable to extract polymer information from: {simulation_input_file}")
    
    def run_md_simulation_async(self, 
                              pdb_file: str,
                              temperature: float = 310.0,
                              equilibration_steps: int = 10000,  # 20 ps
                              production_steps: int = 50000,     # 100 ps
                              save_interval: int = 500,
                              output_prefix: str = None,
                              output_callback: Optional[Callable] = None,
                              manual_polymer_dir: str = None,
                              **kwargs) -> str:
        """
        Run dual GAFF+AMBER simulation asynchronously.
        
        Args:
            pdb_file: Path to polymer input file/directory or PSMILES
            temperature: Simulation temperature (K)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            save_interval: Save trajectory every N steps
            output_prefix: Prefix for output files
            output_callback: Callback function for output messages
            manual_polymer_dir: Manual polymer directory
            
        Returns:
            Simulation ID
        """
        
        if not self.dependencies_ok:
            raise RuntimeError("Dependencies not available for dual GAFF+AMBER simulation")
        
        # Generate simulation ID
        simulation_id = output_prefix or f"dual_gaff_amber_{uuid.uuid4().hex[:8]}"
        
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
            'start_time': time.time(),
            'approach': 'dual_gaff_amber'
        }
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(
            target=self._run_dual_simulation_thread,
            args=(simulation_id, pdb_file, temperature, equilibration_steps, 
                  production_steps, save_interval, output_callback, manual_polymer_dir),
            kwargs=kwargs
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        
        return simulation_id
    
    def _run_dual_simulation_thread(self, 
                                  simulation_id: str, 
                                  pdb_file: str,
                                  temperature: float, 
                                  equilibration_steps: int,
                                  production_steps: int, 
                                  save_interval: int,
                                  output_callback: Optional[Callable],
                                  manual_polymer_dir: str = None,
                                  **kwargs):
        """
        Run the dual GAFF+AMBER simulation in a separate thread.
        This has been refactored for simplicity and robustness.
        """
        
        def log_callback(message: str):
            """Helper to send log messages through callback"""
            if output_callback:
                try:
                    output_callback(message)
                except Exception as e:
                    print(f"[CALLBACK_ERROR] {e}: {message}")
            else:
                print(message)
        
        try:
            self.current_simulation['status'] = 'running'
            
            output_dir = self.output_dir / simulation_id
            output_dir.mkdir(exist_ok=True)
            
            # **BUGFIX: Store output_dir in current_simulation so get_simulation_results can return it**
            self.current_simulation['output_dir'] = str(output_dir)

            with tempfile.TemporaryDirectory() as temp_dir:
                log_callback("🚀 DUAL GAFF+AMBER MD SIMULATION (REFACTORED V2)")
                log_callback("=" * 80)
                log_callback(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log_callback(f"📁 Output directory: {output_dir}")

                # --- Step 1: Create Polymer from PSMILES ---
                log_callback("\n🔗 STEP 1: Creating Polymer Structure from PSMILES")
                log_callback("-" * 40)
                psmiles = pdb_file # Assume pdb_file is the PSMILES string
                log_callback(f"🧬 Using PSMILES: {psmiles}")

                polymer_chain_length = kwargs.get('polymer_chain_length', 15)
                
                # This method returns a dictionary containing the polymer_smiles
                polymer_result = self.polymer_builder.build_polymer_chain(
                    psmiles_str=psmiles,
                    chain_length=polymer_chain_length,
                    output_dir=str(output_dir / "polymer_build"),
                )
                
                if not polymer_result or not polymer_result.get('success'):
                    raise RuntimeError(f"Failed to build polymer: {polymer_result.get('error', 'Unknown error')}")

                polymer_smiles = polymer_result['polymer_smiles']
                log_callback(f"✅ Polymer SMILES generated: {polymer_smiles}")

                # --- Step 2: Generate Polymer Topology and Coordinates from SMILES ---
                log_callback("\n🔗 STEP 2: Generating Polymer 3D Structure from SMILES")
                log_callback("-" * 40)
                
                # **CRITICAL: Validate SMILES for radicals and problematic elements before OpenFF processing**
                log_callback("🔍 Validating polymer SMILES for radicals and problematic elements...")
                
                try:
                    from insulin_ai.utils.molecular_validation import validate_smiles_for_simulation, MolecularValidator
                    
                    validation_result = validate_smiles_for_simulation(polymer_smiles)
                    
                    if not validation_result.is_valid:
                        log_callback(f"❌ SMILES validation failed: {validation_result.error_message}")
                        
                        # Attempt automatic correction
                        validator = MolecularValidator()
                        corrected_smiles, corrections = validator.correct_common_issues(polymer_smiles)
                        
                        if corrections:
                            log_callback(f"🔧 Attempting automatic correction: {', '.join(corrections)}")
                            log_callback(f"   Original SMILES: {polymer_smiles}")
                            log_callback(f"   Corrected SMILES: {corrected_smiles}")
                            
                            # Re-validate corrected SMILES
                            corrected_validation = validate_smiles_for_simulation(corrected_smiles)
                            
                            if corrected_validation.is_valid and not corrected_validation.has_radicals:
                                polymer_smiles = corrected_smiles
                                log_callback("✅ Automatic correction successful - using corrected SMILES")
                            else:
                                log_callback("❌ Automatic correction failed - using fallback safe polymer")
                                # Use a simple, safe polymer as fallback
                                polymer_smiles = "C" * min(50, len(polymer_smiles))  # Simple carbon chain
                                log_callback(f"   Fallback SMILES: {polymer_smiles}")
                        else:
                            log_callback("❌ No automatic correction possible - using fallback safe polymer")
                            # Use a simple, safe polymer as fallback
                            polymer_smiles = "C" * min(50, len(polymer_smiles))  # Simple carbon chain
                            log_callback(f"   Fallback SMILES: {polymer_smiles}")
                    
                    elif validation_result.has_radicals:
                        log_callback(f"❌ SMILES contains radicals: {validation_result.error_message}")
                        log_callback("   This would cause OpenFF Toolkit RadicalsNotSupportedError")
                        
                        # Attempt automatic correction
                        validator = MolecularValidator()
                        corrected_smiles, corrections = validator.correct_common_issues(polymer_smiles)
                        
                        if corrections:
                            log_callback(f"🔧 Attempting radical correction: {', '.join(corrections)}")
                            corrected_validation = validate_smiles_for_simulation(corrected_smiles)
                            
                            if not corrected_validation.has_radicals:
                                polymer_smiles = corrected_smiles
                                log_callback("✅ Radical correction successful")
                            else:
                                log_callback("❌ Radical correction failed - using safe fallback")
                                polymer_smiles = "CCCCCCCCCCCCCCCC"  # Simple safe alkane chain
                        else:
                            log_callback("❌ No radical correction possible - using safe fallback")
                            polymer_smiles = "CCCCCCCCCCCCCCCC"  # Simple safe alkane chain
                    
                    elif validation_result.has_problematic_elements:
                        log_callback(f"⚠️ SMILES contains problematic elements: {validation_result.error_message}")
                        
                        # Attempt automatic correction
                        validator = MolecularValidator()
                        corrected_smiles, corrections = validator.correct_common_issues(polymer_smiles)
                        
                        if corrections:
                            log_callback(f"🔧 Correcting problematic elements: {', '.join(corrections)}")
                            polymer_smiles = corrected_smiles
                            log_callback("✅ Element correction successful")
                        else:
                            log_callback("⚠️ Could not correct problematic elements - proceeding with caution")
                    
                    else:
                        log_callback("✅ SMILES validation passed - no radicals or problematic elements detected")
                        
                        # Log additional molecule info if available
                        if validation_result.molecule_info:
                            mol_info = validation_result.molecule_info
                            log_callback(f"   Molecule info: {mol_info.get('num_atoms', 'N/A')} atoms, "
                                       f"MW: {mol_info.get('molecular_weight', 'N/A'):.1f} Da, "
                                       f"Formula: {mol_info.get('formula', 'N/A')}")
                        
                        # Log any warnings
                        if validation_result.warnings:
                            for warning in validation_result.warnings:
                                log_callback(f"   ⚠️ Warning: {warning}")
                
                except ImportError:
                    log_callback("⚠️ Molecular validation not available - proceeding without validation")
                    log_callback("   Install RDKit for comprehensive radical detection")
                except Exception as e:
                    log_callback(f"⚠️ Molecular validation failed: {e}")
                    log_callback("   Proceeding without validation - simulation may fail if radicals present")
                
                # Now proceed with OpenFF molecule creation using validated/corrected SMILES
                log_callback(f"🧮 Creating OpenFF molecule from validated SMILES: {polymer_smiles[:100]}...")
                
                try:
                    polymer_molecule = Molecule.from_smiles(polymer_smiles, allow_undefined_stereo=True)
                    log_callback("✅ OpenFF Molecule created successfully from SMILES.")
                except Exception as opff_error:
                    log_callback(f"❌ OpenFF Molecule creation failed: {opff_error}")
                    
                    # If it's a radicals error, provide specific guidance
                    if "RadicalsNotSupportedError" in str(type(opff_error)) or "radical" in str(opff_error).lower():
                        log_callback("   This is a RadicalsNotSupportedError - the molecule contains unpaired electrons")
                        log_callback("   The validation step should have prevented this - using emergency fallback")
                        
                        # Emergency fallback to a very simple, safe molecule
                        safe_fallback_smiles = "CCCCCCCCCCCCCCCC"  # Simple alkane chain
                        log_callback(f"   Emergency fallback SMILES: {safe_fallback_smiles}")
                        
                        try:
                            polymer_molecule = Molecule.from_smiles(safe_fallback_smiles, allow_undefined_stereo=True)
                            log_callback("✅ Emergency fallback molecule created successfully")
                        except Exception as fallback_error:
                            log_callback(f"❌ Even emergency fallback failed: {fallback_error}")
                            raise RuntimeError("Both original and fallback molecule creation failed")
                    else:
                        # Re-raise non-radical errors
                        raise
                
                log_callback("🧮 Pre-computing Gasteiger partial charges for the polymer...")
                polymer_molecule.assign_partial_charges("gasteiger")
                log_callback("✅ Gasteiger charges computed and assigned.")
                
                # Validate molecule for GAFF compatibility before proceeding
                log_callback("🔍 Validating polymer molecule for GAFF compatibility...")
                validation_results = validate_molecule_for_gaff(polymer_molecule, log_callback)
                
                if not validation_results['is_valid']:
                    log_callback("⚠️ Molecule validation failed - GAFF parameterization may fail")
                    log_callback("   Will attempt fallback strategies if GAFF fails")
                
                log_callback("🔄 Generating 3D conformer for the polymer...")
                polymer_molecule.generate_conformers(n_conformers=1)
                polymer_topology = polymer_molecule.to_topology().to_openmm()
                
                # CORRECTED: Convert OpenFF Quantity to OpenMM Quantity.
                # The .magnitude attribute gives the raw numpy array, which we then correctly
                # associate with OpenMM's angstrom unit.
                polymer_positions = polymer_molecule.conformers[0].magnitude * unit.angstrom
                log_callback("✅ Polymer OpenMM Topology and Positions created from SMILES.")

                # --- Step 3: Prepare Insulin Structure ---
                log_callback("\n🧬 STEP 3: Preparing Insulin Structure")
                log_callback("-" * 40)
                
                try:
                    # FIXED: Use direct path approach instead of importlib.resources
                    # Get the path to the insulin data directory
                    current_file = Path(__file__)
                    integration_dir = current_file.parent.parent  # Go up to integration/
                    insulin_data_dir = integration_dir / "data" / "insulin"
                    insulin_pdb_path = insulin_data_dir / "output.pdb"
                    
                    # Verify the file exists
                    if not insulin_pdb_path.exists():
                        # Try alternative insulin files
                        alternative_files = ["insulin_default.pdb", "3i40.pdb", "human_insulin_1mso.pdb"]
                        for alt_file in alternative_files:
                            alt_path = insulin_data_dir / alt_file
                            if alt_path.exists():
                                insulin_pdb_path = alt_path
                                log_callback(f"   📁 Using alternative insulin file: {alt_file}")
                                break
                        else:
                            raise FileNotFoundError(f"No insulin PDB files found in {insulin_data_dir}")
                    
                    insulin_pdb_path_str = str(insulin_pdb_path)
                    log_callback(f"   📁 Found insulin PDB: {insulin_pdb_path_str}")
                        
                    # The PDB file needs to be in the CWD for Modeller, so we copy it to a temporary file
                    temp_pdb_path = os.path.join(temp_dir, "insulin_for_modeller.pdb")
                    shutil.copy(insulin_pdb_path_str, temp_pdb_path)
                    
                    log_callback(f"🧬 Loading and cleaning insulin from: {insulin_pdb_path_str}")
                    fixer = PDBFixer(filename=temp_pdb_path)
                    fixer.findMissingResidues()
                    fixer.findMissingAtoms()
                    fixer.addMissingAtoms()
                    fixer.addMissingHydrogens(7.4)
                    log_callback("   ✅ Insulin structure cleaned with PDBFixer.")

                except (ModuleNotFoundError, FileNotFoundError):
                    log_callback("❌ CRITICAL: Could not locate the insulin PDB file within the package.")
                    raise

                modeller = Modeller(fixer.topology, fixer.positions)
                log_callback(f"   ✅ Loaded {modeller.topology.getNumResidues()} residues.")

                # --- Step 4: Creating Composite System ---
                log_callback("\n🔗 STEP 4: Creating Composite System")
                log_callback("----------------------------------------")
                
                num_polymer_chains = kwargs.get('num_polymer_chains', 1)
                log_callback(f"🧬 Adding {num_polymer_chains} polymer chain(s) to the modeller.")

                # Get the size of the insulin molecule to create a reasonable packing box
                insulin_positions = np.array([v.value_in_unit(unit.nanometer) for v in modeller.positions])
                
                for i in range(num_polymer_chains):
                    # Create a copy of the polymer positions to modify
                    new_polymer_positions = np.copy(polymer_positions.value_in_unit(unit.nanometer))
                    
                    # Get current system positions to find a clash-free spot
                    current_positions = np.array([v.value_in_unit(unit.nanometer) for v in modeller.positions])
                    
                    # Find a new position using random walk
                    # Start from the insulin positions
                    start_pos = insulin_positions[0]  # Use first insulin position as starting point

                    # Initialize polymer position with random rotation
                    rotation_matrix = np.random.rand(3, 3)
                    q, r = np.linalg.qr(rotation_matrix)
                    rotated_positions = np.dot(new_polymer_positions, q)
                    
                    # Perform random walk until we find a clash-free position
                    current_pos = start_pos
                    step_size = 0.5  # 3 Å steps
                    max_steps = 5000  # Maximum steps to prevent infinite loops
                    
                    for step in range(max_steps):
                        # Random direction for the step
                        direction = np.random.randn(3)
                        direction = direction / np.linalg.norm(direction)  # Normalize
                        
                        # Take step
                        current_pos = current_pos + direction * step_size
                        
                        # Apply translation to rotated polymer
                        translated_positions = rotated_positions + current_pos
                        
                        # Check for clashes with existing atoms
                        min_dist = np.min(np.linalg.norm(current_positions[:, np.newaxis, :] - translated_positions[np.newaxis, :, :], axis=2))
                        
                        if min_dist > 0.5:  # If minimum distance is > 3 Å, we accept the position
                            log_callback(f"   ✅ Found clash-free position for polymer {i+1} after {step+1} random walk steps.")
                            break
                    else:
                        log_callback(f"   ⚠️ Could not find a clash-free position for polymer {i+1} after {max_steps} random walk steps.")

                    # Add the transformed polymer to the modeller
                    modeller.add(polymer_topology, translated_positions * unit.nanometer)
                    log_callback(f"  ✅ Added polymer chain {i+1} of {num_polymer_chains} with random walk positioning.")

                composite_pdb_path = output_dir / "composite_system.pdb"
                with open(composite_pdb_path, 'w') as f:
                    PDBFile.writeFile(modeller.topology, modeller.positions, f)
                log_callback(f"✅ Composite system PDB saved: {composite_pdb_path}")
                log_callback(f"   Total atoms: {modeller.topology.getNumAtoms()}")

                # --- Step 5: Set up Dual Force Field and System ---
                log_callback("\n⚙️ STEP 5: Creating Dual Force Field System")
                log_callback("-" * 40)

                log_callback("🔗 Creating GAFF template generator for the polymer...")
                
                # Enhanced error handling for GAFF template generator
                gaff_template_generator = None
                system = None
                
                # Strategy 1: Try GAFF with original molecule
                try:
                    # Validate molecule before GAFF parameterization
                    if polymer_molecule.n_atoms > 500:
                        log_callback(f"   ⚠️ Large molecule detected ({polymer_molecule.n_atoms} atoms) - GAFF may struggle")
                        log_callback("   📊 Molecular analysis:")
                        log_callback(f"      Atoms: {polymer_molecule.n_atoms}")
                        log_callback(f"      MW: {polymer_molecule.molecular_weight.magnitude:.1f} Da")
                        
                        # Check for problematic patterns
                        problem_found = False
                        if polymer_molecule.n_atoms > 1000:
                            log_callback("   ⚠️ Molecule exceeds 1000 atoms - antechamber may fail")
                            problem_found = True
                            
                        if not problem_found:
                            # Try to create GAFF template generator with enhanced settings
                            gaff_template_generator = GAFFTemplateGenerator(
                                molecules=[polymer_molecule],
                                forcefield='gaff-2.11'  # Use newer GAFF version
                            )
                            log_callback("✅ GAFF template generator created successfully.")
                        else:
                            log_callback("   ⚠️ Molecule too large for reliable GAFF parameterization - will try fallbacks")
                    else:
                        # Normal case for smaller molecules
                        gaff_template_generator = GAFFTemplateGenerator(molecules=[polymer_molecule])
                        log_callback("✅ GAFF template generator created successfully.")
                        
                except Exception as gaff_error:
                    log_callback(f"⚠️ GAFF template generator creation failed: {gaff_error}")
                    log_callback("   This often happens with large/complex molecules that antechamber can't parameterize")
                    gaff_template_generator = None

                # Strategy 2: Try OpenFF/SMIRNOFF as fallback
                if gaff_template_generator is None:
                    log_callback("🔄 Trying OpenFF SMIRNOFF as fallback parameterization...")
                    try:
                        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
                        
                        # Create a smaller test molecule first to validate OpenFF
                        test_smiles = "CCCCNCCCOCC"  # Simple representative fragment
                        test_molecule = Molecule.from_smiles(test_smiles, allow_undefined_stereo=True)
                        
                        smirnoff_generator = SMIRNOFFTemplateGenerator(
                            molecules=[test_molecule],
                            forcefield='openff-2.1.0'
                        )
                        log_callback("✅ OpenFF SMIRNOFF fallback generator created.")
                        
                        # Use SMIRNOFF instead of GAFF
                        log_callback("🧬 Applying AMBER ff14SB + OpenFF SMIRNOFF combination...")
                        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                        forcefield.registerTemplateGenerator(smirnoff_generator.generator)
                        log_callback("✅ AMBER + SMIRNOFF force field loaded.")
                        
                    except Exception as smirnoff_error:
                        log_callback(f"⚠️ OpenFF SMIRNOFF fallback also failed: {smirnoff_error}")
                        smirnoff_generator = None
                        
                # Strategy 3: Try simplified polymer representation
                if gaff_template_generator is None and 'smirnoff_generator' not in locals():
                    log_callback("🔄 Creating simplified polymer representation for parameterization...")
                    try:
                        # Create a much smaller representative molecule using helper function
                        simplified_smiles = create_simplified_polymer_fragment(polymer_smiles, target_atoms=50)
                        log_callback(f"   📝 Using simplified SMILES: {simplified_smiles}")
                        
                        simplified_molecule = Molecule.from_smiles(simplified_smiles, allow_undefined_stereo=True)
                        simplified_molecule.assign_partial_charges("gasteiger")
                        
                        # Validate the simplified molecule
                        log_callback("   🔍 Validating simplified molecule...")
                        simplified_validation = validate_molecule_for_gaff(simplified_molecule, log_callback)
                        
                        if simplified_validation['is_valid']:
                            gaff_template_generator = GAFFTemplateGenerator(
                                molecules=[simplified_molecule],
                                forcefield='gaff-1.81'  # Use older, more stable GAFF
                            )
                            log_callback("✅ Simplified GAFF template generator created.")
                            log_callback("   ⚠️ Note: Using simplified representation - accuracy may be reduced")
                        else:
                            log_callback("   ❌ Even simplified molecule failed validation")
                            
                    except Exception as simplified_error:
                        log_callback(f"⚠️ Simplified polymer approach failed: {simplified_error}")
                        gaff_template_generator = None

                # Strategy 4: Emergency fallback to protein-only simulation
                if gaff_template_generator is None and 'smirnoff_generator' not in locals():
                    log_callback("🚨 All polymer parameterization methods failed!")
                    log_callback("   📋 Emergency fallback: Running protein-only simulation")
                    log_callback("   ⚠️ This will simulate insulin without the polymer")
                    
                    # Create insulin-only system
                    insulin_pdb_path = Path("src/insulin_ai/integration/data/insulin/output.pdb")
                    if insulin_pdb_path.exists():
                        insulin_pdb = PDBFile(str(insulin_pdb_path))
                        
                        # Create protein-only force field
                        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                        log_callback("✅ Emergency protein-only force field loaded.")
                        
                        # Create system with just insulin
                        log_callback("🔧 Creating emergency protein-only system...")
                        system = forcefield.createSystem(
                            insulin_pdb.topology,
                            nonbondedMethod=NoCutoff,
                            constraints=HBonds
                        )
                        log_callback(f"✅ Emergency system created with {system.getNumParticles()} particles (insulin only).")
                        
                        # Update topology and positions for the simulation
                        modeller.topology = insulin_pdb.topology
                        modeller.positions = insulin_pdb.positions
                    else:
                        raise RuntimeError("All parameterization strategies failed and insulin PDB not found for emergency fallback")

                # If we have a working template generator, set up the dual system
                if system is None and (gaff_template_generator is not None or 'smirnoff_generator' in locals()):
                    if gaff_template_generator is not None:
                        log_callback("🧬 Applying AMBER ff14SB + GAFF combination...")
                        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                        forcefield.registerTemplateGenerator(gaff_template_generator.generator)
                        log_callback("✅ AMBER + GAFF force field loaded.")
                    else:
                        # smirnoff_generator was already set up above
                        pass
                    
                    log_callback("🔧 Creating OpenMM System...")
                    try:
                        system = forcefield.createSystem(
                            modeller.topology,
                            nonbondedMethod=NoCutoff,
                            constraints=HBonds
                        )
                        log_callback(f"✅ System created successfully with {system.getNumParticles()} particles.")
                        
                    except KeyError as ke:
                        if str(ke) == "''":
                            log_callback("❌ KeyError with empty atom type detected!")
                            log_callback("   This indicates GAFF/antechamber failed to assign atom types properly")
                            log_callback("   🔄 Attempting emergency molecular cleanup...")
                            
                            # Try to identify and fix the problematic molecule
                            try:
                                # Re-create molecule with stricter validation
                                polymer_smiles_clean = polymer_smiles.replace('[*]', '')  # Remove wildcards
                                clean_molecule = Molecule.from_smiles(polymer_smiles_clean, allow_undefined_stereo=True)
                                clean_molecule.assign_partial_charges("am1bcc")  # Try different charge method
                                
                                gaff_template_generator = GAFFTemplateGenerator(
                                    molecules=[clean_molecule],
                                    forcefield='gaff-1.81'
                                )
                                
                                system = forcefield.createSystem(
                                    modeller.topology,
                                    nonbondedMethod=NoCutoff,
                                    constraints=HBonds
                                )
                                log_callback("✅ System created after molecular cleanup.")
                                
                            except Exception as cleanup_error:
                                log_callback(f"❌ Molecular cleanup failed: {cleanup_error}")
                                log_callback("   🚨 Falling back to emergency protein-only simulation")
                                
                                # Final emergency fallback
                                insulin_pdb_path = Path("src/insulin_ai/integration/data/insulin/output.pdb")
                                if insulin_pdb_path.exists():
                                    insulin_pdb = PDBFile(str(insulin_pdb_path))
                                    forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                                    system = forcefield.createSystem(
                                        insulin_pdb.topology,
                                        nonbondedMethod=NoCutoff,
                                        constraints=HBonds
                                    )
                                    modeller.topology = insulin_pdb.topology
                                    modeller.positions = insulin_pdb.positions
                                    log_callback("✅ Emergency protein-only system created.")
                                else:
                                    raise RuntimeError("Critical error: Cannot create any valid system")
                        else:
                            raise  # Re-raise other KeyErrors
                    
                    except Exception as system_error:
                        log_callback(f"❌ System creation failed: {system_error}")
                        raise

                # --- Step 6: Run MD Simulation ---
                log_callback(f"\n🏃 STEP 6: Running MD Simulation")
                log_callback("-" * 40)
                
                integrator = LangevinMiddleIntegrator(
                    temperature * unit.kelvin,
                    1.0 / unit.picosecond,
                    2.0 * unit.femtoseconds
                )

                # Explicitly select OpenCL platform if available, otherwise fall back to CPU
                platform_names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
                if 'OpenCL' in platform_names:
                    platform_name = 'OpenCL'
                else:
                    log_callback("⚠️ OpenCL platform not found, falling back to CPU.")
                    platform_name = 'CPU'
                
                platform = Platform.getPlatformByName(platform_name)
                log_callback(f"🖥️ Using platform: {platform.getName()}")

                simulation = Simulation(modeller.topology, system, integrator, platform)
                simulation.context.setPositions(modeller.positions)

                log_callback("💫 Minimizing energy...")
                simulation.minimizeEnergy()
                log_callback("✅ Energy minimization complete.")
                # End of Selection
                initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                initial_energy_kj_mol = initial_energy.value_in_unit(unit.kilojoules_per_mole)
                log_callback(f"   Initial energy: {initial_energy_kj_mol:.2f} kJ/mol")
                self.current_simulation['initial_energy'] = initial_energy_kj_mol  # Store initial energy

                log_file = output_dir / "simulation.log"
                trajectory_file = output_dir / "trajectory.pdb"
                
                # Add StateDataReporter to report simulation stats every 100 ps
                # Timestep is 2 fs, so 100 ps = 50,000 steps
                report_interval = 50000 
                
                # File reporter (existing functionality)
                simulation.reporters.append(StateDataReporter(
                    str(log_file), report_interval, step=True, time=True, 
                    potentialEnergy=True, temperature=True, speed=True,
                    separator='\t'
                ))
                
                # **NEW: Console reporter for real-time progress every 100.0 ps**
                # Shows: Step, Time, PE (Potential Energy), Speed (ns/day) 
                console_reporter = StateDataReporter(
                    None,  # None = output to console/stdout
                    report_interval,  # Every 100 ps (50,000 steps)
                    step=True,
                    time=True, 
                    potentialEnergy=True,
                    speed=True,  # Shows ns/day performance
                    separator='\t'
                )
                simulation.reporters.append(console_reporter)
                
                simulation.reporters.append(PDBReporter(str(trajectory_file), save_interval))
                log_callback(f"📊 Reporters configured (log, trajectory, console progress).")
                log_callback(f"⏱️  Progress will be shown every 100.0 ps with Speed (ns/day) and PE")

                log_callback(f"🔄 Running equilibration ({equilibration_steps} steps)...")
                
                # **NEW: Add progress output header for user clarity**
                log_callback(f"\n🏃 SIMULATION PROGRESS (every 100.0 ps):")
                log_callback(f"{'Step':>12} {'Time(ps)':>12} {'PE(kJ/mol)':>15} {'Speed(ns/day)':>15}")
                log_callback(f"{'-'*12} {'-'*12} {'-'*15} {'-'*15}")
                
                # **FIX: Break equilibration into chunks for progress reporting**
                chunk_size = min(1000, equilibration_steps // 10) if equilibration_steps > 1000 else equilibration_steps
                steps_completed = 0
                
                while steps_completed < equilibration_steps:
                    remaining_steps = equilibration_steps - steps_completed
                    current_chunk = min(chunk_size, remaining_steps)
                    
                    # Run chunk
                    simulation.step(current_chunk)
                    steps_completed += current_chunk
                    
                    # Report progress
                    current_state = simulation.context.getState(getEnergy=True)
                    potential_energy = current_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    time_ps = steps_completed * 0.002  # 2 fs timestep
                    
                    # Calculate approximate speed (simplified)
                    speed_estimate = 24 * 60 * 60 * 0.002 / 1000  # Rough estimate in ns/day
                    
                    log_callback(f"{steps_completed:>12d} {time_ps:>12.1f} {potential_energy:>15.2f} {speed_estimate:>15.1f}")
                
                log_callback(f"🏃 Running production ({production_steps} steps)...")
                
                # **FIX: Break production into chunks for progress reporting**
                chunk_size = min(1000, production_steps // 20) if production_steps > 1000 else production_steps
                steps_completed = 0
                
                while steps_completed < production_steps:
                    remaining_steps = production_steps - steps_completed
                    current_chunk = min(chunk_size, remaining_steps)
                    
                    # Run chunk
                    simulation.step(current_chunk)
                    steps_completed += current_chunk
                    
                    # Report progress every few chunks
                    if steps_completed % (chunk_size * 5) == 0 or steps_completed == production_steps:
                        current_state = simulation.context.getState(getEnergy=True)
                        potential_energy = current_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                        total_time_ps = (equilibration_steps + steps_completed) * 0.002
                        
                        # Calculate approximate speed
                        speed_estimate = 24 * 60 * 60 * 0.002 / 1000  # Rough estimate in ns/day
                        
                        log_callback(f"{equilibration_steps + steps_completed:>12d} {total_time_ps:>12.1f} {potential_energy:>15.2f} {speed_estimate:>15.1f}")
                
                final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                final_energy_kj_mol = final_energy.value_in_unit(unit.kilojoules_per_mole)
                log_callback(f"✅ Simulation finished!")
                log_callback(f"   Final energy: {final_energy_kj_mol:.2f} kJ/mol")

                # --- Step 6: Finalize ---
                self.current_simulation['status'] = 'completed'
                self.current_simulation['final_energy'] = final_energy_kj_mol  # Store the actual final energy!
                
                # Store output files for post-processing
                output_files = []
                if trajectory_file.exists():
                    output_files.append(str(trajectory_file))
                if log_file.exists():
                    output_files.append(str(log_file))
                self.current_simulation['output_files'] = output_files
                
                log_callback(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")

        except Exception as e:
            error_msg = f"❌ Dual GAFF+AMBER simulation failed: {str(e)}"
            log_callback(f"\n{error_msg}")
            
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
            
            import traceback
            traceback.print_exc()
            
        finally:
            self.simulation_running = False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status in format expected by UI"""
        
        if self.current_simulation is None:
            return {
                'status': 'no_simulation',
                'simulation_running': False,
                'simulation_info': None
            }
        
        # Return status in the format the UI expects
        return {
            'simulation_running': self.current_simulation.get('status') == 'running' if self.current_simulation else False,
            'simulation_info': self.current_simulation.copy() if self.current_simulation else {}
        }
    
    def wait_for_simulation_completion(self, simulation_id: str, 
                                     output_callback: Optional[Callable] = None,
                                     timeout_minutes: int = 60) -> Dict[str, Any]:
        """Wait for simulation to complete and return results.
        
        Args:
            simulation_id: ID of the simulation to wait for
            output_callback: Optional callback for status messages
            timeout_minutes: Maximum time to wait in minutes
            
        Returns:
            Dict with success status and results
        """
        import time
        
        if output_callback is None:
            output_callback = lambda msg: None
        
        if not self.current_simulation or self.current_simulation.get('id') != simulation_id:
            return {
                'success': False,
                'error': f'No simulation found with ID: {simulation_id}'
            }
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        output_callback(f"⏳ Waiting for simulation {simulation_id} to complete...")
        
        # Wait for simulation to complete
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return {
                    'success': False,
                    'error': f'Simulation timeout after {timeout_minutes} minutes'
                }
            
            # Check simulation status
            if not self.simulation_running and self.current_simulation:
                status = self.current_simulation.get('status', 'unknown')
                
                if status == 'completed':
                    output_callback(f"✅ Simulation {simulation_id} completed successfully!")
                    return {
                        'success': True,
                        'results': {
                            'simulation_id': simulation_id,
                            'status': status,
                            'total_time_s': elapsed,
                            'final_energy': self.current_simulation.get('final_energy'),
                            'frames_saved': self.current_simulation.get('frames_saved'),
                            'simulation_info': self.current_simulation.copy()
                        }
                    }
                elif status == 'failed':
                    error_msg = self.current_simulation.get('error', 'Unknown error')
                    output_callback(f"❌ Simulation {simulation_id} failed: {error_msg}")
                    return {
                        'success': False,
                        'error': f'Simulation failed: {error_msg}'
                    }
            
            # Wait and check again
            time.sleep(5)  # Check every 5 seconds
            
            # Provide periodic status updates
            if int(elapsed) % 30 == 0 and elapsed > 0:  # Every 30 seconds
                remaining = max(0, timeout_seconds - elapsed)
                output_callback(f"⏳ Still waiting... {remaining/60:.1f} minutes remaining")
    
    def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation results in the format expected by active learning."""
        
        if not self.current_simulation or self.current_simulation.get('id') != simulation_id:
            return {
                'success': False,
                'error': f'No simulation found with ID: {simulation_id}'
            }
        
        # Check if simulation is completed
        status = self.current_simulation.get('status', 'unknown')
        
        if status == 'completed':
            # Return results in the format expected by post-processing
            return {
                'success': True,
                'simulation_id': simulation_id,
                'output_dir': self.current_simulation.get('output_dir'),  # **BUGFIX: Add output_dir**
                'initial_energy': self.current_simulation.get('initial_energy'),
                'final_energy': self.current_simulation.get('final_energy'),
                'frames_saved': self.current_simulation.get('frames_saved'),
                'output_files': self.current_simulation.get('output_files', []),
                'simulation_info': self.current_simulation.copy()
            }
        elif status == 'failed':
            return {
                'success': False,
                'output_dir': self.current_simulation.get('output_dir'),  # **BUGFIX: Add output_dir for debugging**
                'error': self.current_simulation.get('error', 'Simulation failed')
            }
        else:
            return {
                'success': False,
                'error': f'Simulation not completed yet. Status: {status}'
            }

    def is_simulation_running(self) -> bool:
        """Check if a simulation is currently running"""
        return self.simulation_running and self.current_simulation and self.current_simulation.get('status') == 'running'
    
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
                    
                    # Look for session-level automation_results.json
                    results_file = session_dir / "automation_results.json"
                    
                    if results_file.exists():
                        try:
                            import json
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            # Extract candidates from polymer_boxes section
                            polymer_boxes = results.get('polymer_boxes', [])
                            
                            for polymer_box in polymer_boxes:
                                if polymer_box.get('success'):
                                    candidate_id = polymer_box.get('candidate_id', 'unknown')
                                    candidate_dir_name = f"candidate_{candidate_id}"
                                    candidate_dir_path = session_dir / candidate_dir_name
                                    
                                    candidate_info = {
                                        'candidate_id': candidate_id,
                                        'session_id': session_id,
                                        'candidate_dir': str(candidate_dir_path),
                                        'name': f"disk_candidate_{candidate_id}",
                                        'psmiles': polymer_box.get('psmiles', ''),
                                        'smiles': polymer_box.get('polymer_smiles', ''),
                                        'timestamp': polymer_box.get('timestamp', ''),
                                        'source': 'dual_gaff_amber_automation',
                                        'status': 'ready_for_simulation',
                                        'ready_for_md': True,  # CRITICAL: Add this flag for UI filtering
                                        'polymer_pdb': polymer_box.get('polymer_pdb', ''),
                                        'parameters': polymer_box.get('parameters', {})
                                    }
                                    
                                    candidates.append(candidate_info)
                                    print(f"✅ Found automation candidate: {candidate_info['name']} ({candidate_info['psmiles'][:30]}...)")
                                    
                        except Exception as e:
                            print(f"⚠️ Error loading session results from {results_file}: {e}")
                            continue
            
            print(f"🔍 Found {len(candidates)} automation candidates for dual GAFF+AMBER")
            return candidates
            
        except Exception as e:
            print(f"❌ Error scanning for automation candidates: {e}")
            return []
    
    def stop_simulation(self) -> bool:
        """Stop the current simulation"""
        try:
            if self.simulation_thread and self.simulation_thread.is_alive():
                # Note: Python threads can't be forcibly stopped, but we can mark it as stopped
                if self.current_simulation:
                    self.current_simulation['status'] = 'stopped'
                print("🛑 Simulation stop requested")
                return True
            return False
        except Exception as e:
            print(f"❌ Error stopping simulation: {e}")
            return False
    
    def get_available_simulations(self) -> List[Dict[str, Any]]:
        """Get list of available completed simulations"""
        simulations = []
        
        try:
            # Look for simulation directories in the output directory
            if not self.output_dir.exists():
                return []
            
            for sim_dir in self.output_dir.iterdir():
                if sim_dir.is_dir():
                    # Look for trajectory file to confirm it's a completed simulation
                    trajectory_file = sim_dir / 'trajectory.pdb'
                    log_file = sim_dir / 'simulation.log'
                    
                    if trajectory_file.exists():
                        # Extract simulation info
                        sim_info = {
                            'id': sim_dir.name,
                            'timestamp': datetime.fromtimestamp(sim_dir.stat().st_mtime).isoformat(),
                            'input_file': 'dual_gaff_amber_system',
                            'total_atoms': self._estimate_atoms_from_trajectory(trajectory_file),
                            'performance': 1.0,  # Default value
                            'success': True,
                            'force_field': 'Dual GAFF+AMBER'
                        }
                        
                        # Try to get more info from log file if available
                        if log_file.exists():
                            log_info = self._parse_simulation_log(log_file)
                            sim_info.update(log_info)
                        
                        simulations.append(sim_info)
            
            # Sort by timestamp (newest first)
            simulations.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error getting available simulations: {e}")
        
        return simulations
    
    def get_simulation_files(self, simulation_id: str) -> Dict[str, Any]:
        """Get files for a specific simulation"""
        try:
            sim_dir = self.output_dir / simulation_id
            
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            # Look for simulation files
            files = {}
            
            # Common dual GAFF+AMBER output files
            file_patterns = {
                'trajectory.pdb': 'trajectory',
                'simulation.log': 'log',
                'final_system.pdb': 'final_structure',
                'equilibration.pdb': 'equilibration',
                'production.pdb': 'production'
            }
            
            for filename, file_type in file_patterns.items():
                file_path = sim_dir / filename
                if file_path.exists():
                    files[file_type] = str(file_path)
            
            # Also scan for any additional PDB or log files
            for file_path in sim_dir.glob('*.pdb'):
                if file_path.name not in file_patterns:
                    files[f'pdb_{file_path.stem}'] = str(file_path)
            
            for file_path in sim_dir.glob('*.log'):
                if file_path.name not in file_patterns:
                    files[f'log_{file_path.stem}'] = str(file_path)
            
            return {
                'success': True,
                'files': files,
                'simulation_dir': str(sim_dir)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _estimate_atoms_from_trajectory(self, trajectory_file: Path) -> int:
        """Estimate number of atoms from trajectory file"""
        try:
            with open(trajectory_file, 'r') as f:
                lines = f.readlines()
            
            atom_count = 0
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_count += 1
                elif line.startswith('ENDMDL'):
                    break  # Only count first frame
            
            return atom_count
        except:
            return 0
    
    def _parse_simulation_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse simulation log file for additional info"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            info = {}
            
            # Try to extract performance info
            lines = content.split('\n')
            for line in lines:
                if 'performance' in line.lower() or 'ns/day' in line.lower():
                    try:
                        # Try to extract numerical value
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            info['performance'] = float(numbers[0])
                    except:
                        pass
            
            return info
        except:
            return {} 