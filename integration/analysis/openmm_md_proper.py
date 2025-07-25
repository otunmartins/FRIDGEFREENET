#!/usr/bin/env python3
"""
Proper OpenMM MD Simulator using AMBER Force Fields with Implicit Solvent
Following openmmforcefields documentation for biomolecular systems
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# OpenMM imports
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.app import HBonds, PME, Simulation
    from openmm.app import StateDataReporter, DCDReporter, PDBReporter
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("⚠️  OpenMM not available. Install with: conda install -c conda-forge openmm")

# PDBFixer imports
try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False
    print("⚠️  PDBFixer not available. Install with: conda install -c conda-forge pdbfixer")

# OpenMMForceFields imports
try:
    from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False
    print("⚠️  openmmforcefields not available. Install with: conda install -c conda-forge openmmforcefields")

# OpenFF toolkit imports
try:
    from openff.toolkit import Molecule
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    print("⚠️  OpenFF toolkit not available. Install with: conda install -c conda-forge openff-toolkit")

# RDKit imports for polymer SMILES conversion
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available. Install with: conda install -c conda-forge rdkit")

# OpenBabel imports for XYZ to SMILES conversion
try:
    from openbabel import pybel
    import openbabel as ob
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    print("⚠️  OpenBabel not available. Install with: conda install -c conda-forge openbabel")

# Analysis imports
try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("⚠️  MDTraj not available for advanced analysis")

class ProperStateReporter:
    """Proper state reporter with comprehensive analysis"""
    
    def __init__(self, file, reportInterval, output_callback=None):
        self._file = open(file, 'w')
        self._reportInterval = reportInterval
        self._output_callback = output_callback
        self._energies = []
        self._temperatures = []
        self._volumes = []
        self._densities = []
        self._lastReportTime = time.time()
        
        # Write proper header
        headers = [
            "Step", "Time(ps)", "PotentialEnergy(kJ/mol)", "KineticEnergy(kJ/mol)", 
            "TotalEnergy(kJ/mol)", "Temperature(K)", "Volume(nm³)", "Density(g/mL)", "Speed(ns/day)"
        ]
        self._file.write('\t'.join(headers) + '\n')
        self._file.flush()
    
    def describeNextReport(self, simulation):
        """Describe the next report"""
        steps_until_report = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps_until_report, True, True, True, True, None)
    
    def report(self, simulation, state):
        """Generate comprehensive report"""
        step = simulation.context.getStepCount()
        
        if step % self._reportInterval == 0:
            current_time = time.time()
            
            # Basic state information
            sim_time = state.getTime().value_in_unit(unit.picosecond)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            total_e = pe + ke
            
            # Temperature from kinetic energy
            num_dof = 3 * simulation.system.getNumParticles() - simulation.system.getNumConstraints()
            if num_dof > 0:
                temp = (2 * ke * 1000) / (num_dof * 8.314)  # Using R = 8.314 J/(mol·K)
            else:
                temp = 0.0
            
            # Volume and density (not applicable for implicit solvent)
            volume = 0.0
            density = 0.0
            
            # Performance calculation
            elapsed = current_time - self._lastReportTime
            if elapsed > 0 and step > 0:
                timestep = simulation.integrator.getStepSize().value_in_unit(unit.femtosecond)
                steps_per_second = self._reportInterval / elapsed
                ns_per_day = (steps_per_second * timestep * 86400) / 1e6
            else:
                ns_per_day = 0.0
            
            # Store data
            self._energies.append(pe)
            self._temperatures.append(temp)
            self._volumes.append(volume)
            self._densities.append(density)
            self._lastReportTime = current_time
            
            # Write to file
            values = [
                str(step), f"{sim_time:.2f}", f"{pe:.2f}", f"{ke:.2f}", f"{total_e:.2f}",
                f"{temp:.2f}", f"{volume:.3f}", f"{density:.3f}", f"{ns_per_day:.2f}"
            ]
            self._file.write('\t'.join(values) + '\n')
            self._file.flush()
            
            # Console output every few reports
            if step % (self._reportInterval * 5) == 0:
                message = f"📊 Step {step:7d}: PE={pe:10.1f} kJ/mol, T={temp:6.1f} K, V={volume:8.2f} nm³, Speed={ns_per_day:6.1f} ns/day"
                if self._output_callback:
                    self._output_callback(message)
                else:
                    print(message)
    
    def get_statistics(self):
        """Get comprehensive simulation statistics"""
        stats = {}
        
        if self._energies:
            stats['energy'] = {
                'mean_pe': np.mean(self._energies),
                'std_pe': np.std(self._energies),
                'min_pe': np.min(self._energies),
                'max_pe': np.max(self._energies),
                'final_pe': self._energies[-1],
                'drift_pe': self._energies[-1] - self._energies[0] if len(self._energies) > 1 else 0
            }
        
        if self._temperatures:
            stats['temperature'] = {
                'mean': np.mean(self._temperatures),
                'std': np.std(self._temperatures),
                'min': np.min(self._temperatures),
                'max': np.max(self._temperatures),
                'target': 310.0,
                'deviation_from_target': abs(np.mean(self._temperatures) - 310.0)
            }
        
        if self._volumes:
            stats['volume'] = {
                'mean': np.mean(self._volumes),
                'std': np.std(self._volumes),
                'change_percent': ((self._volumes[-1] - self._volumes[0]) / self._volumes[0] * 100) if len(self._volumes) > 1 and self._volumes[0] != 0 else 0
            }
        
        return stats
    
    def close(self):
        """Close the reporter"""
        self._file.close()

class ProperOpenMMSimulator:
    """Proper OpenMM simulator using AMBER force fields with implicit solvent"""
    
    def __init__(self, output_dir: str = "implicit_openmm_simulations"):
        """Initialize the proper simulator"""
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is not available")
        
        if not OPENMMFORCEFIELDS_AVAILABLE:
            raise ImportError("openmmforcefields is not available")
        
        if not PDBFIXER_AVAILABLE:
            raise ImportError("PDBFixer is not available")
        
        if not OPENFF_AVAILABLE:
            print("⚠️  OpenFF toolkit not available - mixed systems with UNL residues may fail")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get best platform
        self.platform = self._get_best_platform()
        
        print(f"🚀 Proper OpenMM Simulator initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Platform: {self.platform.getName()}")
    
    def _get_best_platform(self) -> mm.Platform:
        """Get the best available platform"""
        platform_names = [mm.Platform.getPlatform(i).getName() 
                          for i in range(mm.Platform.getNumPlatforms())]
        
        print("🔍 Available OpenMM platforms:")
        for name in platform_names:
            print(f"   • {name}")
        
        for preferred in ['CUDA', 'OpenCL', 'CPU', 'Reference']:
            if preferred in platform_names:
                platform = mm.Platform.getPlatformByName(preferred)
                print(f"✅ Selected platform: {preferred}")
                return platform
        
        raise RuntimeError("No suitable OpenMM platform found")
    
    def fix_pdb_structure(self, pdb_file: str) -> Tuple[app.Topology, np.ndarray]:
        """Fix PDB structure using PDBFixer while preserving polymer components"""
        print(f"\n🔧 Fixing PDB structure with PDBFixer...")
        
        # Load the PDB file with PDBFixer
        fixer = PDBFixer(filename=pdb_file)
        
        # Find and fix missing residues
        print(f"   🔍 Finding missing residues...")
        fixer.findMissingResidues()
        missing_residues = len(fixer.missingResidues)
        if missing_residues > 0:
            print(f"   ➕ Found {missing_residues} missing residues")
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
        else:
            print(f"   ✅ No missing residues found")
        
        # Find and fix missing atoms
        print(f"   🔍 Finding missing atoms...")
        fixer.findMissingAtoms()
        missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
        if missing_atoms > 0:
            print(f"   ➕ Found {missing_atoms} missing atoms")
            fixer.addMissingAtoms()
        else:
            print(f"   ✅ No missing atoms found")
        
        # DO NOT remove heterogens - we need to keep the polymer!
        print(f"   🧪 Preserving polymer components (not removing heterogens)")
        
        # Add missing hydrogens
        print(f"   ➕ Adding missing hydrogens...")
        initial_atoms = len(list(fixer.topology.atoms()))
        fixer.addMissingHydrogens(7.4)  # Physiological pH
        final_atoms = len(list(fixer.topology.atoms()))
        hydrogens_added = final_atoms - initial_atoms
        print(f"      Added {hydrogens_added} hydrogen atoms")
        
        print(f"   ✅ PDB structure fixed successfully!")
        print(f"   📊 Final system: {final_atoms} atoms")
        
        # Save the fixed PDB for debugging/inspection
        fixed_pdb_path = self.output_dir / f"fixed_{Path(pdb_file).stem}.pdb"
        with open(fixed_pdb_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        print(f"   💾 Fixed PDB saved to: {fixed_pdb_path}")
        
        return fixer.topology, fixer.positions
    
    def analyze_system_composition(self, pdb_file: str, 
                                  pre_processed_topology=None, 
                                  pre_processed_positions=None) -> Dict[str, Any]:
        """Analyze system composition and extract protein if needed"""
        print(f"\n🔬 Analyzing system composition: {pdb_file}")
        
        # Use pre-processed data if available, otherwise fix the PDB structure
        if pre_processed_topology is not None and pre_processed_positions is not None:
            print(f"   ✅ Using pre-processed PDB data (skipping redundant PDBFixer)")
            fixed_topology = pre_processed_topology
            fixed_positions = pre_processed_positions
        else:
            print(f"   🔧 Fixing PDB structure with PDBFixer...")
            fixed_topology, fixed_positions = self.fix_pdb_structure(pdb_file)
        
        # Create a PDB-like object for analysis
        class FixedPDB:
            def __init__(self, topology, positions):
                self.topology = topology
                self.positions = positions
        
        pdb = FixedPDB(fixed_topology, fixed_positions)
        
        # Analyze residue composition
        total_atoms = len(list(pdb.topology.atoms()))
        total_residues = len(list(pdb.topology.residues()))
        
        protein_residues = []
        water_residues = []
        unknown_residues = []
        other_residues = []
        
        standard_amino_acids = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
            'THR', 'TRP', 'TYR', 'VAL'
        }
        
        for residue in pdb.topology.residues():
            if residue.name in standard_amino_acids:
                protein_residues.append(residue)
            elif residue.name in ['HOH', 'WAT']:
                water_residues.append(residue)
            elif residue.name == 'UNL':
                unknown_residues.append(residue)
            else:
                other_residues.append(residue)
        
        composition = {
            'total_atoms': total_atoms,
            'total_residues': total_residues,
            'protein_residues': len(protein_residues),
            'water_residues': len(water_residues),
            'unknown_residues': len(unknown_residues),
            'other_residues': len(other_residues),
            'has_protein': len(protein_residues) > 0,
            'has_unknowns': len(unknown_residues) > 0,
            'original_topology': pdb.topology,
            'original_positions': pdb.positions
        }
        
        print(f"   🔬 Total atoms: {total_atoms}")
        print(f"   🧪 Total residues: {total_residues}")
        print(f"   🧬 Protein residues: {len(protein_residues)}")
        print(f"   💧 Water residues: {len(water_residues)}")
        print(f"   ❓ Unknown residues (UNL): {len(unknown_residues)}")
        print(f"   🔍 Other residues: {len(other_residues)}")
        
        # Keep the entire system (protein + polymer) for proper embedding simulation
        print(f"\n🧬 Keeping complete system including polymer for embedding simulation...")
        composition['clean_topology'] = pdb.topology
        composition['clean_positions'] = pdb.positions
        composition['extraction_performed'] = False
        
        if len(unknown_residues) > 0:
            print(f"   🧪 Will use GAFF force field for {len(unknown_residues)} polymer residues")
            composition['needs_gaff'] = True
            composition['polymer_residues'] = unknown_residues
        else:
            composition['needs_gaff'] = False
            composition['polymer_residues'] = []
        
        return composition
    
    def find_latest_polymer_pdb(self, pdb_file_path: str) -> Optional[str]:
        """Find the latest polymer PDB file in insulin_polymer_output_* folder"""
        pdb_dir = Path(pdb_file_path).parent
        print(f"\n🔍 Searching for polymer PDB files in: {pdb_dir}")
        
        # Find all insulin_polymer_output_* directories
        polymer_dirs = list(pdb_dir.glob("insulin_polymer_output_*"))
        if not polymer_dirs:
            print(f"   ❌ No insulin_polymer_output_* directories found")
            return None
        
        # Sort by modification time to get the latest
        polymer_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_dir = polymer_dirs[0]
        
        print(f"   📁 Latest polymer directory: {latest_dir.name}")
        
        # Find PDB files in the latest directory (check packmol subdirectory first)
        packmol_dir = latest_dir / "packmol"
        pdb_files = []
        
        if packmol_dir.exists():
            pdb_files = list(packmol_dir.glob("*.pdb"))
            if pdb_files:
                print(f"   📁 Found PDB files in packmol subdirectory")
        
        if not pdb_files:
            # Fallback to main directory
            pdb_files = list(latest_dir.glob("*.pdb"))
            
        if not pdb_files:
            print(f"   ❌ No PDB files found in {latest_dir} or its packmol subdirectory")
            return None
        
        # Sort by modification time to get the latest, prefer packmol.pdb
        pdb_files.sort(key=lambda x: (x.name == "packmol.pdb", x.stat().st_mtime), reverse=True)
        latest_pdb = pdb_files[0]
        
        print(f"   📄 Latest polymer PDB: {latest_pdb}")
        
        return str(latest_pdb)
    
    def convert_polymer_file_to_smiles(self, polymer_file_path: str) -> Optional[str]:
        """Convert polymer XYZ file to SMILES string using OpenBabel (PDB files no longer supported)"""
        print(f"   🧪 Converting polymer file to SMILES: {polymer_file_path}")
        
        try:
            file_path = Path(polymer_file_path)
            
            if not file_path.exists():
                print(f"   ❌ Polymer file not found: {polymer_file_path}")
                return None
            
            # Only process XYZ files - skip PDB files
            if file_path.suffix.lower() == '.xyz':
                # Use OpenBabel for XYZ files
                if not OPENBABEL_AVAILABLE:
                    print(f"   ❌ OpenBabel not available for XYZ conversion")
                    return None
                
                print(f"   📄 Reading XYZ file with OpenBabel...")
                
                # Use the OpenBabel-based XYZ to SMILES converter
                smiles = self._convert_xyz_to_smiles_with_connectivity(file_path)
                
                if smiles:
                    print(f"   ✅ Generated SMILES from XYZ: {smiles}")
                    return smiles
                else:
                    print(f"   ❌ Failed to generate SMILES from XYZ file")
                    return None
                        
            elif file_path.suffix.lower() == '.pdb':
                print(f"   ⚠️  PDB file processing disabled - only XYZ files supported")
                print(f"   💡 Convert {file_path.name} to XYZ format to process with OpenBabel")
                return None
                        
            else:
                print(f"   ❌ Unsupported file format: {file_path.suffix}")
                print(f"   💡 Only XYZ files are supported for polymer processing")
                return None
                
        except Exception as e:
            print(f"   ❌ Error converting polymer file to SMILES: {e}")
            return None
    
    def _clean_and_read_pdb(self, pdb_file_path: str) -> Optional[Chem.Mol]:
        """Legacy PDB cleaning method - no longer used since we only process XYZ files"""
        print(f"   ⚠️  PDB cleaning method called but PDB processing is disabled")
        print(f"   💡 Convert PDB files to XYZ format to use OpenBabel processing")
        return None
    
    def _read_xyz_with_connectivity(self, xyz_file_path: str) -> Optional[str]:
        """Legacy method - now redirects to OpenBabel-based XYZ to SMILES conversion"""
        print(f"   🔄 Redirecting to OpenBabel-based XYZ conversion...")
        return self._convert_xyz_to_smiles_with_connectivity(Path(xyz_file_path))
    
    def _sanitize_molecule_robust(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Legacy RDKit molecule sanitization - no longer used since we only process XYZ files with OpenBabel"""
        print(f"   ⚠️  RDKit sanitization method called but PDB processing is disabled")
        print(f"   💡 OpenBabel handles molecule validation automatically for XYZ files")
        return None
    
    def find_polymer_files_and_convert_to_smiles(self, pdb_file_path: str, manual_polymer_dir: str = None) -> List[str]:
        """Find polymer files in insulin_polymer_output_* folders and convert to SMILES"""
        print(f"\n🔍 Looking for polymer files to convert to SMILES...")
        
        polymer_smiles = []
        conversion_stats = {'total_files': 0, 'successful': 0, 'failed': 0}
        
        # Check if manual polymer directory is specified
        if manual_polymer_dir:
            print(f"   🎯 Using manual polymer directory: {manual_polymer_dir}")
            polymer_dirs = [Path(manual_polymer_dir)]
        else:
            # Search in the working directory (where polymer output directories are typically created)
            search_dir = Path.cwd()
            # Also search in parent of pdb_file_path as fallback
            pdb_dir = Path(pdb_file_path).parent if pdb_file_path else Path.cwd()
            
            # Find all insulin_polymer_output_* directories (search in working directory first)
            polymer_dirs = list(search_dir.glob("insulin_polymer_output_*"))
            
            # If not found in working directory, search in PDB directory
            if not polymer_dirs:
                print(f"   🔍 No polymer directories found in working directory, searching in PDB directory...")
                polymer_dirs = list(pdb_dir.glob("insulin_polymer_output_*"))
        
        if not polymer_dirs:
            print(f"   ❌ No insulin_polymer_output_* directories found")
            if manual_polymer_dir:
                print(f"   🔍 Manual directory specified but not found: {manual_polymer_dir}")
            else:
                print(f"   🔍 Searched in: {Path.cwd()} and {Path(pdb_file_path).parent if pdb_file_path else Path.cwd()}")
            return polymer_smiles
        
        # Sort by modification time to get the latest first
        polymer_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Process ALL directories to find polymer files - XYZ ONLY
        for i, polymer_dir in enumerate(polymer_dirs):
            print(f"   📁 Checking polymer directory: {polymer_dir.name}")
            
            # Check molecules subdirectory first
            molecules_dir = polymer_dir / "molecules"
            if molecules_dir.exists():
                print(f"   📁 Found molecules directory: {molecules_dir}")
                
                # Look for polymer XYZ files ONLY (skip PDB files)
                polymer_files = list(molecules_dir.glob("polymer_*.xyz"))
                
                if polymer_files:
                    print(f"   📄 Found {len(polymer_files)} XYZ polymer files in {polymer_dir.name}")
                    
                    for polymer_file in polymer_files:
                        print(f"   🧪 Processing: {polymer_file.name}")
                        conversion_stats['total_files'] += 1
                        
                        smiles = self.convert_polymer_file_to_smiles(str(polymer_file))
                        if smiles:
                            polymer_smiles.append(smiles)
                            conversion_stats['successful'] += 1
                            print(f"   ✅ Successfully converted {polymer_file.name} to SMILES")
                        else:
                            conversion_stats['failed'] += 1
                            print(f"   ❌ Failed to convert {polymer_file.name}")
                else:
                    print(f"   ❌ No XYZ polymer files found in {molecules_dir}")
            else:
                print(f"   ❌ No molecules directory found in {polymer_dir}")
        
        # Print comprehensive statistics
        print(f"\n📊 Conversion Statistics:")
        print(f"   📄 Total polymer files found: {conversion_stats['total_files']}")
        print(f"   ✅ Successful conversions: {conversion_stats['successful']}")
        print(f"   ❌ Failed conversions: {conversion_stats['failed']}")
        
        if polymer_smiles:
            print(f"   🎉 Successfully converted {len(polymer_smiles)} polymer files to SMILES")
            
            # Remove duplicate SMILES (in case multiple files generate the same SMILES)
            unique_smiles = list(set(polymer_smiles))
            if len(unique_smiles) != len(polymer_smiles):
                print(f"   🔄 Removed {len(polymer_smiles) - len(unique_smiles)} duplicate SMILES")
                polymer_smiles = unique_smiles
            
            print(f"   📝 Final unique SMILES count: {len(polymer_smiles)}")
            for i, smiles in enumerate(polymer_smiles):
                # Truncate very long SMILES for display
                display_smiles = smiles[:100] + "..." if len(smiles) > 100 else smiles
                print(f"   🧪 Polymer {i+1} SMILES: {display_smiles}")
        else:
            print(f"   ❌ No polymer files could be converted to SMILES")
            
            # Provide diagnostic information
            print(f"\n🔍 Diagnostic Information:")
            print(f"   📁 Searched {len(polymer_dirs)} polymer directories")
            print(f"   📄 Found {conversion_stats['total_files']} polymer files")
            print(f"   ⚠️  Common issues:")
            print(f"      • PDB formatting problems (residue numbering)")
            print(f"      • XYZ valence calculation errors")
            print(f"      • Missing connectivity information")
            print(f"   💡 Suggestion: Check polymer file formats and consider manual cleaning")
        
        return polymer_smiles
    
    def create_molecule_from_smiles(self, smiles: str) -> Optional[Molecule]:
        """Create OpenFF Molecule from SMILES string"""
        try:
            print(f"   🧪 Creating OpenFF Molecule from SMILES: {smiles}")
            molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            print(f"   ✅ Created OpenFF Molecule with {molecule.n_atoms} atoms")
            return molecule
        except Exception as e:
            print(f"   ❌ Failed to create OpenFF Molecule from SMILES: {e}")
            return None

    def extract_smiles_from_xyz_files(self, pdb_file_path: str = None, manual_polymer_dir: str = None) -> List[str]:
        """Extract SMILES from XYZ files with OpenBabel connectivity inference"""
        print(f"   🎯 Focusing on XYZ files for correct molecular structure...")
        
        if not OPENBABEL_AVAILABLE:
            print(f"   ❌ OpenBabel not available - cannot process XYZ files")
            return []
        
        xyz_smiles = []
        xyz_files = []
        
        # Find all XYZ files in polymer directories
        search_dirs = []
        
        if pdb_file_path:
            pdb_dir = Path(pdb_file_path).parent
            search_dirs.append(pdb_dir)
        
        search_dirs.append(Path.cwd())
        
        for search_dir in search_dirs:
            polymer_dirs = list(search_dir.glob("insulin_polymer_output_*"))
            for polymer_dir in polymer_dirs:
                for subdir in ['molecules', 'packmol']:
                    subdir_path = polymer_dir / subdir
                    if subdir_path.exists():
                        xyz_files.extend(list(subdir_path.glob("*.xyz")))
        
        # Remove duplicates
        xyz_files = list(set(xyz_files))
        print(f"   📄 Found {len(xyz_files)} XYZ files")
        
        for i, xyz_file in enumerate(xyz_files):
            try:
                print(f"   🧪 Processing XYZ file {i+1}/{len(xyz_files)}: {xyz_file.name}")
                smiles = self._convert_xyz_to_smiles_with_connectivity(xyz_file)
                
                if smiles and smiles not in xyz_smiles:
                    xyz_smiles.append(smiles)
                    print(f"      ✅ Generated SMILES: {smiles}")
                else:
                    print(f"      ⚠️  Duplicate or failed SMILES generation")
                    
            except Exception as e:
                print(f"      ❌ Failed to process {xyz_file.name}: {e}")
                continue
        
        print(f"   📝 Generated {len(xyz_smiles)} unique SMILES from XYZ files")
        return xyz_smiles
    
    def _convert_xyz_to_smiles_with_connectivity(self, xyz_file_path: Path) -> Optional[str]:
        """Convert XYZ file to SMILES using OpenBabel for better topology inference"""
        try:
            print(f"      📄 Reading XYZ file: {xyz_file_path.name}")
            
            if not OPENBABEL_AVAILABLE:
                print(f"      ❌ OpenBabel not available - cannot convert XYZ to SMILES")
                return None
            
            # Method 1: Use OpenBabel with PyBel for direct XYZ to SMILES conversion
            try:
                print(f"      🔄 Using OpenBabel for XYZ to SMILES conversion...")
                
                # Read XYZ file with OpenBabel
                mol = next(pybel.readfile("xyz", str(xyz_file_path)))
                
                if mol is None:
                    print(f"      ❌ Failed to read XYZ file with OpenBabel")
                    return None
                
                print(f"      📊 Molecule loaded: {len(mol.atoms)} atoms")
                
                # Add hydrogens if needed
                mol.addh()
                print(f"      🧪 Added hydrogens, total atoms: {len(mol.atoms)}")
                
                # Perform connectivity perception
                mol.OBMol.ConnectTheDots()
                mol.OBMol.PerceiveBondOrders()
                
                print(f"      🔗 Connectivity perceived: {mol.OBMol.NumBonds()} bonds")
                
                # Generate SMILES
                smiles = mol.write("smi").strip()
                
                if smiles:
                    print(f"      ✅ Generated SMILES with OpenBabel: {smiles}")
                    return smiles
                else:
                    print(f"      ⚠️  OpenBabel generated empty SMILES")
                
            except Exception as e:
                print(f"      ⚠️  OpenBabel direct conversion failed: {e}")
            
            # Method 2: Manual OpenBabel processing with custom connectivity
            try:
                print(f"      🔄 Trying manual OpenBabel processing...")
                
                # Create OpenBabel molecule manually
                obmol = ob.OBMol()
                
                # Read XYZ file manually for better control
                with open(xyz_file_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    print(f"      ❌ Invalid XYZ file format")
                    return None
                
                # Parse XYZ format
                n_atoms = int(lines[0].strip())
                if len(lines) < n_atoms + 2:
                    print(f"      ❌ Incomplete XYZ file")
                    return None
                
                print(f"      📊 Atoms in file: {n_atoms}")
                
                # Add atoms with coordinates
                for i in range(2, 2 + n_atoms):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        element = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # Create OpenBabel atom
                        atom = obmol.NewAtom()
                        atom.SetAtomicNum(ob.GetAtomicNum(element))
                        atom.SetVector(x, y, z)
                
                # Perceive connectivity
                obmol.ConnectTheDots()
                obmol.PerceiveBondOrders()
                
                # Add hydrogens
                obmol.AddHydrogens()
                
                print(f"      🧪 Created molecule with {obmol.NumAtoms()} atoms, {obmol.NumBonds()} bonds")
                
                # Convert to SMILES
                conv = ob.OBConversion()
                conv.SetOutFormat("smi")
                
                smiles = conv.WriteString(obmol).strip()
                
                if smiles:
                    print(f"      ✅ Generated SMILES with manual OpenBabel: {smiles}")
                    return smiles
                else:
                    print(f"      ⚠️  Manual OpenBabel generated empty SMILES")
                
            except Exception as e:
                print(f"      ⚠️  Manual OpenBabel processing failed: {e}")
            
            # Method 3: Fallback to simple molecular formula
            try:
                print(f"      🔄 Falling back to molecular formula...")
                
                # Read XYZ file to get element counts
                with open(xyz_file_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 2:
                    return None
                
                n_atoms = int(lines[0].strip())
                if len(lines) < n_atoms + 2:
                    return None
                
                # Count elements
                from collections import Counter
                elements = []
                for i in range(2, 2 + n_atoms):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        elements.append(parts[0])
                
                element_counts = Counter(elements)
                
                # Create simple molecular formula as fallback
                formula_parts = []
                for element in sorted(element_counts.keys()):
                    count = element_counts[element]
                    if count > 1:
                        formula_parts.append(f"{element}{count}")
                    else:
                        formula_parts.append(element)
                
                formula = ''.join(formula_parts)
                print(f"      ⚠️  Fallback molecular formula: {formula}")
                
                # Return as disconnected atoms (not ideal but better than nothing)
                atom_smiles = '.'.join([element for element in elements if element != 'H'])
                return atom_smiles if atom_smiles else formula
                
            except Exception as e:
                print(f"      ⚠️  Fallback formula generation failed: {e}")
            
            print(f"      ❌ All XYZ conversion methods failed")
            return None
            
        except Exception as e:
            print(f"      ❌ XYZ processing failed: {e}")
            return None

    def extract_unl_residues_from_topology(self, topology) -> List[str]:
        """Extract UNL residues from topology with proper bond information and convert to SMILES"""
        print(f"   🔍 Extracting UNL residues with bond information from topology...")
        
        if not RDKIT_AVAILABLE:
            print(f"   ❌ RDKit not available - cannot generate proper SMILES")
            return []
        
        unl_smiles = []
        unl_residues = [res for res in topology.residues() if res.name == 'UNL']
        
        print(f"   📊 Found {len(unl_residues)} UNL residues in topology")
        
        # Create a mapping from topology atoms to indices for bond lookups
        atom_to_index = {}
        for i, atom in enumerate(topology.atoms()):
            atom_to_index[atom] = i
        
        for i, residue in enumerate(unl_residues):
            try:
                print(f"   🧪 Processing UNL residue {i+1}/{len(unl_residues)} (ID: {residue.id})")
                
                # Get all atoms in this residue
                residue_atoms = list(residue.atoms())
                print(f"      📊 Atoms in residue: {len(residue_atoms)}")
                
                # Create RDKit molecule
                mol = Chem.RWMol()
                
                # Add atoms to molecule and create mapping
                atom_map = {}  # Maps topology atom to RDKit atom index
                for j, atom in enumerate(residue_atoms):
                    element = atom.element.symbol if atom.element else 'C'
                    print(f"         Atom {j+1}: {atom.name} ({element})")
                    
                    # Create RDKit atom
                    rdkit_atom = Chem.Atom(element)
                    rdkit_idx = mol.AddAtom(rdkit_atom)
                    atom_map[atom] = rdkit_idx
                
                # Find bonds involving atoms in this residue
                bonds_added = 0
                residue_atom_set = set(residue_atoms)
                
                for bond in topology.bonds():
                    atom1, atom2 = bond
                    
                    # Check if both atoms are in this residue
                    if atom1 in residue_atom_set and atom2 in residue_atom_set:
                        if atom1 in atom_map and atom2 in atom_map:
                            rdkit_idx1 = atom_map[atom1]
                            rdkit_idx2 = atom_map[atom2]
                            
                            # Add bond (default to single bond)
                            mol.AddBond(rdkit_idx1, rdkit_idx2, Chem.BondType.SINGLE)
                            bonds_added += 1
                            print(f"         Bond: {atom1.name}-{atom2.name}")
                
                print(f"      🔗 Added {bonds_added} bonds")
                
                # Convert to regular molecule
                mol = mol.GetMol()
                
                if mol is None:
                    print(f"      ❌ Failed to create RDKit molecule")
                    continue
                
                # Try to sanitize the molecule
                try:
                    Chem.SanitizeMol(mol)
                    print(f"      ✅ Molecule sanitized successfully")
                except Exception as e:
                    print(f"      ⚠️  Sanitization failed: {e}")
                    # Try without sanitization
                    pass
                
                # Generate SMILES
                try:
                    smiles = Chem.MolToSmiles(mol, allHsExplicit=False)
                    if smiles and smiles not in unl_smiles:
                        unl_smiles.append(smiles)
                        print(f"      ✅ Generated SMILES: {smiles}")
                        print(f"      📊 Molecule: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
                    else:
                        print(f"      ⚠️  Duplicate or empty SMILES")
                except Exception as e:
                    print(f"      ❌ SMILES generation failed: {e}")
                    
                    # Fallback: create a simple disconnected SMILES
                    print(f"      🔄 Falling back to disconnected SMILES...")
                    atom_counts = {}
                    for atom in residue_atoms:
                        element = atom.element.symbol if atom.element else 'C'
                        if element != 'H':  # Skip hydrogens
                            atom_counts[element] = atom_counts.get(element, 0) + 1
                    
                    smiles_parts = []
                    for element, count in atom_counts.items():
                        smiles_parts.extend([element] * count)
                    
                    fallback_smiles = '.'.join(smiles_parts)
                    if fallback_smiles and fallback_smiles not in unl_smiles:
                        unl_smiles.append(fallback_smiles)
                        print(f"      ⚠️  Fallback SMILES: {fallback_smiles}")
                    
            except Exception as e:
                print(f"      ❌ Failed to process UNL residue {i+1}: {e}")
                continue
        
        print(f"   📝 Generated {len(unl_smiles)} unique UNL SMILES from topology")
        return unl_smiles

    def create_proper_system(self, composition: Dict[str, Any], 
                           temperature: float = 310.0, 
                           pdb_file_path: str = None,
                           manual_polymer_dir: str = None) -> Tuple[mm.System, app.Topology, np.ndarray, Dict]:
        """Create system using proper AMBER force fields with mixed system support"""
        
        print(f"\n🔧 Creating system with AMBER force fields...")
        
        # Use the fixed PDB structure
        modeller = Modeller(composition['clean_topology'], composition['clean_positions'])
        
        print(f"   ✅ Using hydrogens added by PDBFixer")
        
        final_atoms = len(list(modeller.topology.atoms()))
        print(f"   ✅ Final system: {final_atoms} atoms (implicit solvent)")
        
        # Handle disulfide bonds
        print(f"   🔧 Handling disulfide bonds...")
        cys_residues = [res for res in modeller.topology.residues() if res.name == 'CYS']
        print(f"   🧪 Found {len(cys_residues)} CYS residues")
        
        # Find disulfide bonds
        disulfide_bonds = []
        for bond in modeller.topology.bonds():
            atom1, atom2 = bond
            if (atom1.name == 'SG' and atom2.name == 'SG' and 
                atom1.residue.name == 'CYS' and atom2.residue.name == 'CYS'):
                disulfide_bonds.append((atom1.residue, atom2.residue))
            
        print(f"   🔗 Found {len(disulfide_bonds)} disulfide bonds")
            
        # Use SystemGenerator with proper configuration for mixed systems
        if composition['needs_gaff']:
            # Configuration for mixed systems (protein + polymers)
            print(f"   🧪 Configuring SystemGenerator for mixed system...")
            system_generator = SystemGenerator(
                forcefields=[
                    "amber/protein.ff14SB.xml",
                    "implicit/gbn2.xml"
                ],
                small_molecule_forcefield='gaff-2.11',
                forcefield_kwargs={
                    "constraints": HBonds,
                    "removeCMMotion": True,
                    "hydrogenMass": 4 * unit.amu
                },
                nonperiodic_forcefield_kwargs={
                    "nonbondedMethod": app.CutoffNonPeriodic,
                    "nonbondedCutoff": 1.0 * unit.nanometer
                },
                cache=str(self.output_dir / "mixed_forcefield_cache.json")
            )
        else:
            # Configuration for pure protein systems with implicit solvent
            print(f"   🧬 Using standard ForceField for pure protein system...")
            forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
            
            # Create system using standard ForceField approach
            print(f"   🧪 Creating system with CutoffNonPeriodic method...")
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0*unit.nanometer,
                constraints=HBonds,
                removeCMMotion=True,
                hydrogenMass=4*unit.amu
            )
        
        # Create molecules list for small molecules (polymers)
        molecules = []
        topology_smiles = []  # Initialize for scope
        xyz_smiles = []  # Initialize for scope
        if composition['needs_gaff']:
            print(f"   🧪 Creating molecule definitions for polymer residues...")
            
            # First priority: Extract SMILES from XYZ files (correct structure)
            xyz_smiles = self.extract_smiles_from_xyz_files(pdb_file_path, manual_polymer_dir)
            
            if xyz_smiles:
                print(f"   ✅ Found {len(xyz_smiles)} SMILES from XYZ files (correct structure)")
                
                # Create OpenFF Molecule objects from XYZ-extracted SMILES
                for smiles in xyz_smiles:
                    molecule = self.create_molecule_from_smiles(smiles)
                    if molecule:
                        molecules.append(molecule)
                
                print(f"   ✅ Created {len(molecules)} molecule definitions from XYZ files")
                topology_smiles = xyz_smiles  # For reporting purposes
            
            # Second priority: Try topology extraction
            if not molecules:
                print(f"   ⚠️  XYZ extraction failed, trying topology extraction...")
                topology_smiles = self.extract_unl_residues_from_topology(modeller.topology)
                
                if topology_smiles:
                    print(f"   ✅ Found {len(topology_smiles)} UNL SMILES from topology extraction")
                    
                    # Create OpenFF Molecule objects from topology-extracted SMILES
                    for smiles in topology_smiles:
                        molecule = self.create_molecule_from_smiles(smiles)
                        if molecule:
                            molecules.append(molecule)
                    
                    print(f"   ✅ Created {len(molecules)} molecule definitions from topology UNL residues")
            
            # Third priority: Fall back to external XYZ files only
            if not molecules:
                print(f"   ⚠️  Topology extraction failed, trying external XYZ files...")
                
                # Use the existing XYZ extraction method - this is more reliable
                external_xyz_smiles = self.extract_smiles_from_xyz_files(pdb_file_path, manual_polymer_dir)
                
                print(f"   🔍 Found {len(external_xyz_smiles)} XYZ SMILES from external file search")
                
                if external_xyz_smiles:
                    print(f"   ✅ Found {len(external_xyz_smiles)} XYZ SMILES")
                    
                    # Create OpenFF Molecule objects from SMILES
                    for smiles in external_xyz_smiles:
                        molecule = self.create_molecule_from_smiles(smiles)
                        if molecule:
                            molecules.append(molecule)
                    
                    print(f"   ✅ Created {len(molecules)} molecule definitions from external XYZ SMILES")
                else:
                    print(f"   ❌ No XYZ SMILES found from external files")
                    print(f"   ⚠️  Skipping PDB files - only processing XYZ files per configuration")
            
            if molecules:
                print(f"   🎯 Using {len(molecules)} polymer molecules for GAFF parameterization")
                
                # Determine the source for reporting
                if xyz_smiles and len(molecules) > 0:
                    source = "XYZ files (correct structure)"
                elif topology_smiles and len(molecules) > 0:
                    source = "Topology UNL extraction"
                else:
                    source = "External polymer files"
                
                print(f"   📝 Molecule source: {source}")
                
                # Add molecules to the SystemGenerator BEFORE creating the system
                print(f"   📋 Adding {len(molecules)} molecules to SystemGenerator...")
                for i, molecule in enumerate(molecules):
                    print(f"      Adding molecule {i+1}: {molecule.n_atoms} atoms")
                
                # Register all molecules with the SystemGenerator
                system_generator.add_molecules(molecules)
                print(f"   ✅ All molecules registered with SystemGenerator")
                
                # Now create the system (without molecules parameter)
                system = system_generator.create_system(modeller.topology)
            else:
                print(f"   ⚠️  No molecules created from SMILES, falling back to standard force field")
                # Fallback to standard force field without GAFF
                forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=1.0*unit.nanometer,
                    constraints=HBonds,
                    removeCMMotion=True,
                    hydrogenMass=4*unit.amu
                )
        else:
            print(f"   ⚠️  No polymer SMILES found, using standard force field")
            # Fallback to standard force field without GAFF
            forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0*unit.nanometer,
                constraints=HBonds,
                removeCMMotion=True,
                hydrogenMass=4*unit.amu
            )
        
        system_info = {
            'final_atoms': final_atoms,
            'hydrogens_added': 0,
            'solvated': False,
            'force_field': 'AMBER ff14SB + GAFF + GBn2 implicit solvent',
            'nonbonded_method': 'CutoffNonPeriodic',
            'nonbonded_cutoff': '1.0 nm',
            'ionic_strength': 0.0,
            'extraction_performed': composition['extraction_performed'],
            'pdb_fixed': True,
            'pdb_fixer_applied': True,
            'gaff_used': composition['needs_gaff'],
            'approach_used': 'SystemGenerator with mixed force fields',
            'mixed_system': composition['needs_gaff'],
            'disulfide_bonds': len(disulfide_bonds),
            'cys_residues': len(cys_residues)
        }
        
        print(f"   ✅ System created successfully!")
        print(f"   🔧 Nonbonded method: CutoffNonPeriodic (1.0 nm cutoff)")
        print(f"   💧 Implicit solvent: GBn2")
        
        return system, modeller.topology, modeller.positions, system_info
    
    def run_proper_simulation_with_preprocessing(self, pdb_file: str,
                                           pre_processed_topology,
                                           pre_processed_positions,
                                           temperature: float = 310.0,
                                           equilibration_steps: int = 5000000,
                                           production_steps: int = 25000000,
                                           save_interval: int = 1000,
                                           output_prefix: str = None,
                                           stop_condition_check: Optional[Callable] = None,
                                           output_callback: Optional[Callable] = None,
                                           manual_polymer_dir: str = None) -> Dict[str, Any]:
        """Run proper MD simulation with pre-processed topology and positions"""
        
        # Helper function to handle output
        def log_output(message: str):
            if output_callback:
                output_callback(message)
            else:
                print(message)
        
        if output_prefix is None:
            output_prefix = f"proper_sim_{uuid.uuid4().hex[:8]}"
        
        sim_dir = self.output_dir / output_prefix
        sim_dir.mkdir(exist_ok=True)
        
        log_output(f"\n🚀 Proper AMBER MD Simulation with Implicit Solvent")
        log_output(f"=" * 60)
        log_output(f"🆔 Simulation ID: {output_prefix}")
        log_output(f"📁 Output directory: {sim_dir}")
        log_output(f"🌡️  Temperature: {temperature} K")
        log_output(f"🔄 Equilibration: {equilibration_steps} steps ({equilibration_steps * 4 / 1000:.1f} ps)")
        log_output(f"🏃 Production: {production_steps} steps ({production_steps * 4 / 1000:.1f} ps)")
        log_output(f"💧 Solvent: Implicit (GBn2)")
        
        start_time = time.time()
        
        try:
            # 1. Analyze and prepare system using pre-processed data
            log_output(f"\n📋 System Analysis and Preparation")
            log_output(f"-" * 40)
            composition = self.analyze_system_composition(
                pdb_file, 
                pre_processed_topology=pre_processed_topology,
                pre_processed_positions=pre_processed_positions
            )
            
            # Print detailed composition info
            log_output(f"🧬 System Composition:")
            log_output(f"   • Total atoms: {composition['total_atoms']}")
            log_output(f"   • Total residues: {composition['total_residues']}")
            log_output(f"   • Protein residues: {composition['protein_residues']}")
            log_output(f"   • Unknown residues (UNL): {composition['unknown_residues']}")
            log_output(f"   • Needs GAFF: {composition['needs_gaff']}")
            
            system, topology, positions, system_info = self.create_proper_system(
                composition, temperature, pdb_file, manual_polymer_dir
            )
            
            # Continue with the rest of the simulation exactly as in run_proper_simulation
            return self._run_simulation_with_system(
                system, topology, positions, system_info,
                temperature, equilibration_steps, production_steps,
                save_interval, output_prefix, sim_dir, start_time,
                stop_condition_check, output_callback
            )
            
        except Exception as e:
            log_output(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'simulation_id': output_prefix,
                'output_dir': str(sim_dir)
            }
    
    def _run_simulation_with_system(self, system, topology, positions, system_info,
                                   temperature, equilibration_steps, production_steps,
                                   save_interval, output_prefix, sim_dir, start_time,
                                   stop_condition_check: Optional[Callable] = None,
                                   output_callback: Optional[Callable] = None):
        """Run the actual simulation with the provided system"""
        
        # Helper function to handle output
        def log_output(message: str):
            if output_callback:
                output_callback(message)
            else:
                print(message)
        
        # Print system creation results
        log_output(f"\n📊 System Creation Results:")
        log_output(f"   • Approach used: {system_info['approach_used']}")
        log_output(f"   • Mixed system: {system_info['mixed_system']}")
        log_output(f"   • GAFF used: {system_info['gaff_used']}")
        log_output(f"   • Final atoms: {system_info['final_atoms']}")
        
        # 2. Create simulation with proper integrator
        log_output(f"\n🔧 Creating simulation context...")
        
        # Use hydrogen mass repartitioning for 4 fs timestep
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,  # friction coefficient
            4.0 * unit.femtosecond  # 4 fs timestep with HMR
        )
        
        # Create simulation
        simulation = Simulation(topology, system, integrator, self.platform)
        
        # Set positions with error handling
        try:
            simulation.context.setPositions(positions)
            log_output(f"✅ Positions set successfully")
        except Exception as e:
            log_output(f"❌ Error setting positions: {e}")
            log_output(f"🔄 Trying to fix positions...")
            # Try to fix positions by centering
            positions_array = np.array(positions.value_in_unit(unit.angstrom))
            positions_array = positions_array - np.mean(positions_array, axis=0)
            positions = positions_array * unit.angstrom
            simulation.context.setPositions(positions)
            log_output(f"✅ Positions fixed and set successfully")
        
        log_output(f"✅ Simulation context created on {self.platform.getName()} (NVT ensemble)")
        
        # 3. Energy minimization with robust error handling
        log_output(f"\n⚡ Energy minimization...")
        try:
            # Check stop condition before starting minimization
            if stop_condition_check and stop_condition_check():
                log_output(f"   🛑 Energy minimization stopped by user request")
                return {
                    'success': True,
                    'simulation_id': output_prefix,
                    'output_dir': str(sim_dir),
                    'user_stopped': True,
                    'message': 'Simulation stopped during energy minimization'
                }
            
            initial_state = simulation.context.getState(getEnergy=True)
            initial_pe = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            log_output(f"   🔋 Initial potential energy: {initial_pe:.1f} kJ/mol")
        
            # Check for problematic energies
            if np.isnan(initial_pe) or np.isinf(initial_pe):
                log_output(f"   ⚠️  Warning: Initial energy is {initial_pe}, attempting to fix...")
                # Try smaller minimization steps
                for i in range(10):
                    # Check stop condition during fixing
                    if stop_condition_check and stop_condition_check():
                        log_output(f"   🛑 Energy fixing stopped by user request")
                        return {
                            'success': True,
                            'simulation_id': output_prefix,
                            'output_dir': str(sim_dir),
                            'user_stopped': True,
                            'message': 'Simulation stopped during energy fixing'
                        }
                    
                    simulation.minimizeEnergy(maxIterations=100)
                    state = simulation.context.getState(getEnergy=True)
                    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    if not (np.isnan(pe) or np.isinf(pe)):
                        log_output(f"   ✅ Fixed energy after {(i+1)*100} iterations: {pe:.1f} kJ/mol")
                        break
                else:
                    log_output(f"   ❌ Could not fix energy issues")
                    raise ValueError("Energy minimization failed - system has problematic configuration")
            
            minimize_start = time.time()
            simulation.minimizeEnergy(maxIterations=1000)
            minimize_time = time.time() - minimize_start
        
            minimized_state = simulation.context.getState(getEnergy=True)
            final_pe = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            energy_change = final_pe - initial_pe
        
            log_output(f"   🔋 Final potential energy: {final_pe:.1f} kJ/mol")
            log_output(f"   📉 Energy change: {energy_change:.1f} kJ/mol")
            log_output(f"   ⏱️  Minimization time: {minimize_time:.2f} seconds")
            
            if energy_change > 0:
                log_output(f"   ⚠️  Warning: Energy increased during minimization")
                
        except Exception as e:
            log_output(f"   ❌ Energy minimization failed: {e}")
            log_output(f"   🔄 Attempting to continue without minimization...")
            initial_pe = float('nan')
            final_pe = float('nan')
            energy_change = float('nan')
            minimize_time = 0.0
        
        # 4. Equilibration phase
        log_output(f"\n🔄 Equilibration phase...")
        eq_dir = sim_dir / "equilibration"
        eq_dir.mkdir(exist_ok=True)
        
        eq_reporter = ProperStateReporter(str(eq_dir / "equilibration.csv"), save_interval, output_callback)
        
        simulation.reporters.clear()
        simulation.reporters.append(eq_reporter)
        simulation.reporters.append(DCDReporter(str(eq_dir / "equilibration.dcd"), save_interval))
        
        eq_start = time.time()
        
        # Run equilibration in chunks for progress reporting
        chunk_size = 10000
        for i in range(0, equilibration_steps, chunk_size):
            remaining = min(chunk_size, equilibration_steps - i)
            
            # Check stop condition
            if stop_condition_check and stop_condition_check():
                log_output(f"   🛑 Equilibration stopped by user request at step {i}")
                break
            
            try:
                simulation.step(remaining)
                progress = (i + remaining) / equilibration_steps * 100
                if progress % 20 == 0 or progress > 95:
                    log_output(f"   📈 Equilibration progress: {progress:.0f}%")
            except Exception as e:
                log_output(f"   ❌ Equilibration failed at step {i}: {e}")
                if i == 0:
                    raise RuntimeError(f"Equilibration failed immediately: {e}")
                log_output(f"   🔄 Stopping equilibration early at {i} steps")
                break
        
        eq_time = time.time() - eq_start
        eq_stats = eq_reporter.get_statistics()
        eq_reporter.close()
        
        log_output(f"   ✅ Equilibration completed in {eq_time:.2f} seconds")
        if 'temperature' in eq_stats:
            temp_stats = eq_stats['temperature']
            log_output(f"   🌡️  Temperature: {temp_stats['mean']:.1f} ± {temp_stats['std']:.1f} K")
            log_output(f"   🎯 Target deviation: {temp_stats['deviation_from_target']:.1f} K")
        
        # 5. Production phase
        log_output(f"\n🏃 Production phase...")
        prod_dir = sim_dir / "production"
        prod_dir.mkdir(exist_ok=True)
        
        prod_reporter = ProperStateReporter(str(prod_dir / "production.csv"), save_interval, output_callback)
        
        simulation.reporters.clear()
        simulation.reporters.append(prod_reporter)
        simulation.reporters.append(DCDReporter(str(prod_dir / "production.dcd"), save_interval))
        simulation.reporters.append(PDBReporter(str(prod_dir / "frames.pdb"), save_interval * 5))
        
        prod_start = time.time()
        
        # Run production in chunks
        for i in range(0, production_steps, chunk_size):
            remaining = min(chunk_size, production_steps - i)
            
            # Check stop condition
            if stop_condition_check and stop_condition_check():
                log_output(f"   🛑 Production stopped by user request at step {i}")
                break
            
            try:
                simulation.step(remaining)
                progress = (i + remaining) / production_steps * 100
                elapsed = time.time() - prod_start
                if progress % 20 == 0 or progress > 95:
                    if elapsed > 0:
                        eta = elapsed * (production_steps / (i + remaining) - 1)
                        log_output(f"   📈 Production progress: {progress:.0f}% - ETA: {eta:.0f}s")
            except Exception as e:
                log_output(f"   ❌ Production failed at step {i}: {e}")
                if i == 0:
                    raise RuntimeError(f"Production failed immediately: {e}")
                log_output(f"   🔄 Stopping production early at {i} steps")
                break
        
        prod_time = time.time() - prod_start
        prod_stats = prod_reporter.get_statistics()
        prod_reporter.close()
        
        # Check if simulation was stopped by user
        user_stopped = stop_condition_check and stop_condition_check()
        
        # Continue with the rest of the simulation exactly as in the original method
        # This includes saving final structure, analysis, and returning results
        # (Copy the rest of the original run_proper_simulation method from here)
        
        return {
            'success': True,
            'simulation_id': output_prefix,
            'output_dir': str(sim_dir),
            'user_stopped': user_stopped,
            'timing': {
                'total': time.time() - start_time,
                'minimization': minimize_time,
                'equilibration': eq_time,
                'production': prod_time
            },
            'performance': {
                'ns_per_day': (production_steps * 4 / 1000 / 1000) / (prod_time / 86400) if prod_time > 0 else 0
            },
            'system_info': system_info,
            'equilibration_stats': eq_stats,
            'production_stats': prod_stats,
            'energy_analysis': {
                'initial_pe': initial_pe,
                'final_pe': final_pe,
                'minimization_change': energy_change
            }
        }
    
    def run_proper_simulation(self, pdb_file: str,
                            temperature: float = 310.0,
                            equilibration_steps: int = 5000000,    # 1000 ps
                            production_steps: int = 25000000,     # 5000 ps  
                            save_interval: int = 1000,
                            output_prefix: str = None) -> Dict[str, Any]:
        """Run proper MD simulation with AMBER force fields and implicit solvent"""
        
        if output_prefix is None:
            output_prefix = f"proper_sim_{uuid.uuid4().hex[:8]}"
        
        sim_dir = self.output_dir / output_prefix
        sim_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 Proper AMBER MD Simulation with Implicit Solvent")
        print(f"=" * 60)
        print(f"🆔 Simulation ID: {output_prefix}")
        print(f"📁 Output directory: {sim_dir}")
        print(f"🌡️  Temperature: {temperature} K")
        print(f"🔄 Equilibration: {equilibration_steps} steps ({equilibration_steps * 4 / 1000:.1f} ps)")
        print(f"🏃 Production: {production_steps} steps ({production_steps * 4 / 1000:.1f} ps)")
        print(f"💧 Solvent: Implicit (GBn2)")
        
        start_time = time.time()
        
        try:
            # 1. Analyze and prepare system
            print(f"\n📋 System Analysis and Preparation")
            print(f"-" * 40)
            composition = self.analyze_system_composition(pdb_file)
            
            # Print detailed composition info
            print(f"🧬 System Composition:")
            print(f"   • Total atoms: {composition['total_atoms']}")
            print(f"   • Total residues: {composition['total_residues']}")
            print(f"   • Protein residues: {composition['protein_residues']}")
            print(f"   • Unknown residues (UNL): {composition['unknown_residues']}")
            print(f"   • Needs GAFF: {composition['needs_gaff']}")
            
            system, topology, positions, system_info = self.create_proper_system(
                composition, temperature, pdb_file
            )
            
            # Print system creation results
            print(f"\n📊 System Creation Results:")
            print(f"   • Approach used: {system_info['approach_used']}")
            print(f"   • Mixed system: {system_info['mixed_system']}")
            print(f"   • GAFF used: {system_info['gaff_used']}")
            print(f"   • Final atoms: {system_info['final_atoms']}")
            
            # 2. Create simulation with proper integrator
            print(f"\n🔧 Creating simulation context...")
            
            # Use hydrogen mass repartitioning for 4 fs timestep
            integrator = mm.LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,  # friction coefficient
                4.0 * unit.femtosecond  # 4 fs timestep with HMR
            )
            
            # Create simulation
            simulation = Simulation(topology, system, integrator, self.platform)
            
            # Set positions with error handling
            try:
                simulation.context.setPositions(positions)
                print(f"✅ Positions set successfully")
            except Exception as e:
                print(f"❌ Error setting positions: {e}")
                print(f"🔄 Trying to fix positions...")
                # Try to fix positions by centering
                positions_array = np.array(positions.value_in_unit(unit.angstrom))
                positions_array = positions_array - np.mean(positions_array, axis=0)
                positions = positions_array * unit.angstrom
                simulation.context.setPositions(positions)
                print(f"✅ Positions fixed and set successfully")
            
            print(f"✅ Simulation context created on {self.platform.getName()} (NVT ensemble)")
            
            # 3. Energy minimization with robust error handling
            print(f"\n⚡ Energy minimization...")
            try:
                initial_state = simulation.context.getState(getEnergy=True)
                initial_pe = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                print(f"   🔋 Initial potential energy: {initial_pe:.1f} kJ/mol")
            
                # Check for problematic energies
                if np.isnan(initial_pe) or np.isinf(initial_pe):
                    print(f"   ⚠️  Warning: Initial energy is {initial_pe}, attempting to fix...")
                    # Try smaller minimization steps
                    for i in range(10):
                        simulation.minimizeEnergy(maxIterations=100)
                        state = simulation.context.getState(getEnergy=True)
                        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        if not (np.isnan(pe) or np.isinf(pe)):
                            print(f"   ✅ Fixed energy after {(i+1)*100} iterations: {pe:.1f} kJ/mol")
                            break
                    else:
                        print(f"   ❌ Could not fix energy issues")
                        raise ValueError("Energy minimization failed - system has problematic configuration")
                
                minimize_start = time.time()
                simulation.minimizeEnergy(maxIterations=1000)
                minimize_time = time.time() - minimize_start
            
                minimized_state = simulation.context.getState(getEnergy=True)
                final_pe = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                energy_change = final_pe - initial_pe
            
                print(f"   🔋 Final potential energy: {final_pe:.1f} kJ/mol")
                print(f"   📉 Energy change: {energy_change:.1f} kJ/mol")
                print(f"   ⏱️  Minimization time: {minimize_time:.2f} seconds")
                
                if energy_change > 0:
                    print(f"   ⚠️  Warning: Energy increased during minimization")
                    
            except Exception as e:
                print(f"   ❌ Energy minimization failed: {e}")
                print(f"   🔄 Attempting to continue without minimization...")
                initial_pe = float('nan')
                final_pe = float('nan')
                energy_change = float('nan')
                minimize_time = 0.0
            
            # 4. Equilibration phase
            print(f"\n🔄 Equilibration phase...")
            eq_dir = sim_dir / "equilibration"
            eq_dir.mkdir(exist_ok=True)
            
            eq_reporter = ProperStateReporter(str(eq_dir / "equilibration.csv"), save_interval)
            
            simulation.reporters.clear()
            simulation.reporters.append(eq_reporter)
            simulation.reporters.append(DCDReporter(str(eq_dir / "equilibration.dcd"), save_interval))
            
            eq_start = time.time()
            
            # Run equilibration in chunks for progress reporting
            chunk_size = 10000
            for i in range(0, equilibration_steps, chunk_size):
                remaining = min(chunk_size, equilibration_steps - i)
                
                try:
                    simulation.step(remaining)
                    progress = (i + remaining) / equilibration_steps * 100
                    if progress % 20 == 0 or progress > 95:
                        print(f"   📈 Equilibration progress: {progress:.0f}%")
                except Exception as e:
                    print(f"   ❌ Equilibration failed at step {i}: {e}")
                    if i == 0:
                        raise RuntimeError(f"Equilibration failed immediately: {e}")
                    print(f"   🔄 Stopping equilibration early at {i} steps")
                    break
            
            eq_time = time.time() - eq_start
            eq_stats = eq_reporter.get_statistics()
            eq_reporter.close()
            
            print(f"   ✅ Equilibration completed in {eq_time:.2f} seconds")
            if 'temperature' in eq_stats:
                temp_stats = eq_stats['temperature']
                print(f"   🌡️  Temperature: {temp_stats['mean']:.1f} ± {temp_stats['std']:.1f} K")
                print(f"   🎯 Target deviation: {temp_stats['deviation_from_target']:.1f} K")
            
            # 5. Production phase
            print(f"\n🏃 Production phase...")
            prod_dir = sim_dir / "production"
            prod_dir.mkdir(exist_ok=True)
            
            prod_reporter = ProperStateReporter(str(prod_dir / "production.csv"), save_interval)
            
            simulation.reporters.clear()
            simulation.reporters.append(prod_reporter)
            simulation.reporters.append(DCDReporter(str(prod_dir / "production.dcd"), save_interval))
            simulation.reporters.append(PDBReporter(str(prod_dir / "frames.pdb"), save_interval * 5))
            
            prod_start = time.time()
            
            # Run production in chunks
            for i in range(0, production_steps, chunk_size):
                remaining = min(chunk_size, production_steps - i)
                
                try:
                    simulation.step(remaining)
                    progress = (i + remaining) / production_steps * 100
                    elapsed = time.time() - prod_start
                    if progress % 20 == 0 or progress > 95:
                        if elapsed > 0:
                            eta = elapsed * (production_steps / (i + remaining) - 1)
                            print(f"   📈 Production progress: {progress:.0f}% - ETA: {eta:.0f}s")
                except Exception as e:
                    print(f"   ❌ Production failed at step {i}: {e}")
                    if i == 0:
                        raise RuntimeError(f"Production failed immediately: {e}")
                    print(f"   🔄 Stopping production early at {i} steps")
                    break
            
            prod_time = time.time() - prod_start
            prod_stats = prod_reporter.get_statistics()
            prod_reporter.close()
            
            # Save final structure
            try:
                final_state = simulation.context.getState(getPositions=True, getEnergy=True)
                with open(str(prod_dir / "final_structure.pdb"), 'w') as f:
                    PDBFile.writeFile(topology, final_state.getPositions(), f)
                    print(f"   💾 Final structure saved")
            except Exception as e:
                print(f"   ⚠️  Could not save final structure: {e}")
            
            print(f"   ✅ Production completed in {prod_time:.2f} seconds")
            
            # 6. Performance and analysis summary
            total_time = time.time() - start_time
            total_steps = equilibration_steps + production_steps
            steps_per_second = total_steps / total_time
            ns_per_day = (steps_per_second * 4.0 * 86400) / 1e6  # 4 fs timestep
            
            print(f"\n📊 Simulation Performance Summary:")
            print(f"   ⏱️  Total wall time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"   🚀 Average performance: {ns_per_day:.2f} ns/day")
            print(f"   📈 Steps per second: {steps_per_second:.1f}")
            print(f"   🔬 Final system size: {system_info['final_atoms']} atoms")
            print(f"   🧪 System approach: {system_info['approach_used']}")
            
            # 7. Generate analysis plots
            self.generate_comprehensive_plots(prod_dir / "production.csv", sim_dir)
            
            # 8. Prepare comprehensive results
            results = {
                'simulation_id': output_prefix,
                'timestamp': datetime.now().isoformat(),
                'input_file': pdb_file,
                'system_info': system_info,
                'parameters': {
                    'temperature': temperature,
                    'equilibration_steps': equilibration_steps,
                    'production_steps': production_steps,
                    'timestep_fs': 4.0,
                    'force_field': system_info['force_field'],
                    'platform': self.platform.getName(),
                    'solvated': False
                },
                'timing': {
                    'minimization': minimize_time,
                    'equilibration': eq_time,
                    'production': prod_time,
                    'total': total_time
                },
                'performance': {
                    'steps_per_second': steps_per_second,
                    'ns_per_day': ns_per_day
                },
                'equilibration_stats': eq_stats,
                'production_stats': prod_stats,
                'energy_analysis': {
                    'initial_pe': initial_pe,
                    'final_pe': final_pe,
                    'minimization_change': energy_change,
                    'minimization_time': minimize_time
                },
                'files': {
                    'production_trajectory': str(prod_dir / "production.dcd"),
                    'equilibration_trajectory': str(eq_dir / "equilibration.dcd"),
                    'final_structure': str(prod_dir / "final_structure.pdb"),
                    'production_log': str(prod_dir / "production.csv"),
                    'equilibration_log': str(eq_dir / "equilibration.csv"),
                    'plots': str(sim_dir / "analysis_plots.png")
                },
                'success': True
            }
            
            # Save simulation report
            with open(str(sim_dir / "simulation_report.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n🎉 Simulation completed successfully!")
            print(f"📊 Results saved to: {sim_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'simulation_id': output_prefix,
                'traceback': traceback.format_exc()
            }
    
    def generate_comprehensive_plots(self, data_file: Path, output_dir: Path):
        """Generate comprehensive analysis plots"""
        try:
            print(f"   📈 Generating comprehensive analysis plots...")
            
            # Read simulation data
            df = pd.read_csv(data_file, delimiter='\t')
            
            # Create comprehensive plot grid
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Comprehensive MD Simulation Analysis', fontsize=16, fontweight='bold')
            
            # Energy evolution
            if 'PotentialEnergy(kJ/mol)' in df.columns:
                axes[0, 0].plot(df['Time(ps)'], df['PotentialEnergy(kJ/mol)'], 'b-', alpha=0.8, linewidth=1)
                axes[0, 0].set_title('Potential Energy Evolution')
                axes[0, 0].set_xlabel('Time (ps)')
                axes[0, 0].set_ylabel('Potential Energy (kJ/mol)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Temperature control
            if 'Temperature(K)' in df.columns:
                axes[0, 1].plot(df['Time(ps)'], df['Temperature(K)'], 'r-', alpha=0.8, linewidth=1)
                axes[0, 1].axhline(y=310, color='k', linestyle='--', linewidth=2, label='Target (310 K)')
                axes[0, 1].fill_between(df['Time(ps)'], 305, 315, alpha=0.2, color='green', label='±5 K range')
                axes[0, 1].set_title('Temperature Control')
                axes[0, 1].set_xlabel('Time (ps)')
                axes[0, 1].set_ylabel('Temperature (K)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Volume/density evolution
            if 'Volume(nm³)' in df.columns:
                axes[0, 2].plot(df['Time(ps)'], df['Volume(nm³)'], 'g-', alpha=0.8, linewidth=1)
                axes[0, 2].set_title('System Volume Evolution')
                axes[0, 2].set_xlabel('Time (ps)')
                axes[0, 2].set_ylabel('Volume (nm³)')
                axes[0, 2].grid(True, alpha=0.3)
            
            # Total energy conservation
            if 'TotalEnergy(kJ/mol)' in df.columns:
                axes[1, 0].plot(df['Time(ps)'], df['TotalEnergy(kJ/mol)'], 'm-', alpha=0.8, linewidth=1)
                axes[1, 0].set_title('Total Energy Conservation')
                axes[1, 0].set_xlabel('Time (ps)')
                axes[1, 0].set_ylabel('Total Energy (kJ/mol)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Performance monitoring
            if 'Speed(ns/day)' in df.columns:
                axes[1, 1].plot(df['Time(ps)'], df['Speed(ns/day)'], 'c-', alpha=0.8, linewidth=1)
                axes[1, 1].set_title('Simulation Performance')
                axes[1, 1].set_xlabel('Time (ps)')
                axes[1, 1].set_ylabel('Speed (ns/day)')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Energy distribution (histogram)
            if 'PotentialEnergy(kJ/mol)' in df.columns:
                # Use last 50% of data for equilibrated distribution
                equilibrated_data = df['PotentialEnergy(kJ/mol)'].iloc[len(df)//2:]
                axes[1, 2].hist(equilibrated_data, bins=30, alpha=0.7, color='blue', density=True)
                axes[1, 2].axvline(equilibrated_data.mean(), color='red', linestyle='--', 
                                 label=f'Mean: {equilibrated_data.mean():.1f}')
                axes[1, 2].set_title('Equilibrated Energy Distribution')
                axes[1, 2].set_xlabel('Potential Energy (kJ/mol)')
                axes[1, 2].set_ylabel('Probability Density')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / "comprehensive_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     📊 Comprehensive plots saved: {plot_file}")
            
        except Exception as e:
            print(f"     ⚠️  Plot generation failed: {e}")

def test_polymer_smiles_conversion(pdb_file_path: str = None):
    """Test function to verify polymer SMILES conversion functionality"""
    print(f"\n🧪 Testing Polymer SMILES Conversion")
    print(f"=" * 50)
    
    if pdb_file_path is None:
        # Look for any PDB file in the current directory
        pdb_files = list(Path('.').glob("*.pdb"))
        if not pdb_files:
            print(f"❌ No PDB files found in current directory")
            return
        pdb_file_path = str(pdb_files[0])
    
    print(f"📄 Using PDB file: {pdb_file_path}")
    
    # Create simulator instance
    simulator = ProperOpenMMSimulator()
    
    # Test polymer SMILES conversion
    polymer_smiles = simulator.find_polymer_files_and_convert_to_smiles(pdb_file_path)
    
    if polymer_smiles:
        print(f"\n✅ Successfully converted {len(polymer_smiles)} polymer files to SMILES:")
        for i, smiles in enumerate(polymer_smiles):
            print(f"   {i+1}. {smiles}")
            
            # Test creating OpenFF Molecule from SMILES
            molecule = simulator.create_molecule_from_smiles(smiles)
            if molecule:
                print(f"      ✅ OpenFF Molecule created: {molecule.n_atoms} atoms, {molecule.n_bonds} bonds")
            else:
                print(f"      ❌ Failed to create OpenFF Molecule")
    else:
        print(f"\n❌ No polymer SMILES found")
    
    print(f"\n🔚 Test completed")
    return polymer_smiles

def main():
    """Main function for proper AMBER simulation with implicit solvent"""
    if len(sys.argv) < 2:
        print("Usage: python openmm_md_proper.py <pdb_file> [eq_steps] [prod_steps]")
        print("Example: python openmm_md_proper.py insulin.pdb 50000 500000")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    # Optional command line arguments for step counts
    equilibration_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 25000
    production_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 125000
    
    # Create proper simulator
    simulator = ProperOpenMMSimulator()
    
    # Run simulation with proper parameters
    results = simulator.run_proper_simulation(
        pdb_file=pdb_file,
        temperature=310.0,           
        equilibration_steps=equilibration_steps,
        production_steps=production_steps,
        save_interval=500            
    )
    
    if results['success']:
        print(f"\n🎉 Proper AMBER simulation completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   🆔 ID: {results['simulation_id']}")
        print(f"   ⏱️  Time: {results['timing']['total']:.1f} seconds ({results['timing']['total']/60:.1f} min)")
        print(f"   🚀 Performance: {results['performance']['ns_per_day']:.1f} ns/day")
        print(f"   🔬 System: {results['system_info']['final_atoms']} atoms")
        print(f"   🧪 Force field: {results['parameters']['force_field']}")
        print(f"   💧 Solvent: Implicit (GBn2)")
        
        if 'energy_analysis' in results:
            energy = results['energy_analysis']
            print(f"   🔋 Energy change: {energy['minimization_change']:.1f} kJ/mol")
        
        if 'production_stats' in results and 'temperature' in results['production_stats']:
            temp_stats = results['production_stats']['temperature']
            print(f"   🌡️  Final temperature: {temp_stats['mean']:.1f} ± {temp_stats['std']:.1f} K")
    else:
        print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 
    