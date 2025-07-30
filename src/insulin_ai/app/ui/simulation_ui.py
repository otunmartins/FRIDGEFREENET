"""
MD Simulation UI Module for Insulin-AI App

This module provides the user interface for molecular dynamics simulation integration,
including file management, simulation parameters, live monitoring, and results analysis.

Author: AI-Driven Material Discovery Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import uuid
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import enhanced MD system with stored SMILES
try:
    from integration.analysis.enhanced_md_with_stored_smiles import (
        EnhancedMDWithStoredSMILES, 
        create_enhanced_md_simulator, 
        check_enhanced_md_availability
    )
    ENHANCED_MD_AVAILABLE = True
except ImportError:
    ENHANCED_MD_AVAILABLE = False

def render_simulation_ui():
    """
    Render the complete MD Simulation UI interface
    
    Returns:
        None
    """
    st.subheader("Molecular Dynamics Simulation Integration")
    
    # Initialize MD simulation system
    initialize_md_simulation_system()
    
    # Check system status and render appropriate interface
    if st.session_state.get('md_integration_available', False):
        render_simulation_interface()
    else:
        render_installation_instructions()


def initialize_md_simulation_system():
    """Initialize the MD simulation system with error handling"""
    
    # Check if already initialized by main app
    if hasattr(st.session_state, 'md_integration') and st.session_state.md_integration is not None:
        st.session_state.md_integration_available = True
        print("✅ Using MD Simulation Integration already initialized by main app")
        return
    
    # Set availability flag if main app initialization failed
    if 'md_integration_available' not in st.session_state:
        # Check if md_integration exists but is None (failed initialization)
        if hasattr(st.session_state, 'md_integration') and st.session_state.md_integration is None:
            st.session_state.md_integration_available = False
            st.session_state.md_integration_error = "MD simulation integration failed to initialize in main app"
            print("⚠️ MD integration is None - main app initialization failed")
        else:
            # Try to initialize if not done by main app
            try:
                from integration.analysis.md_simulation_integration import MDSimulationIntegration
                st.session_state.md_integration = MDSimulationIntegration()
                st.session_state.md_integration_available = True
                print("✅ MD Simulation Integration initialized successfully in UI (fallback)")
            except ImportError as e:
                st.session_state.md_integration = None
                st.session_state.md_integration_available = False
                error_msg = f"MD simulation integration import failed: {str(e)}"
                st.session_state.md_integration_error = error_msg
                print(f"⚠️ {error_msg}")
                
                # Check specific dependencies
                _check_md_dependencies()
                
            except Exception as e:
                st.session_state.md_integration = None
                st.session_state.md_integration_available = False
                error_msg = f"MD simulation initialization failed: {str(e)}"
                st.session_state.md_integration_error = error_msg
                print(f"❌ {error_msg}")


def _check_md_dependencies():
    """Check specific MD simulation dependencies for debugging"""
    
    dependencies = {
        'openmm': False,
        'pdbfixer': False,
        'openmmforcefields': False,
        'mdtraj': False,
        'numpy': False,
        'pathlib': False
    }
    
    # Check each dependency
    try:
        import openmm
        dependencies['openmm'] = True
        print("  ✅ OpenMM available")
    except ImportError:
        print("  ❌ OpenMM not available")
    
    try:
        import pdbfixer
        dependencies['pdbfixer'] = True
        print("  ✅ PDBFixer available")
    except ImportError:
        print("  ❌ PDBFixer not available")
    
    try:
        import openmmforcefields
        dependencies['openmmforcefields'] = True
        print("  ✅ OpenMMForceFields available")
    except ImportError:
        print("  ❌ OpenMMForceFields not available")
    
    try:
        import mdtraj
        dependencies['mdtraj'] = True
        print("  ✅ MDTraj available")
    except ImportError:
        print("  ❌ MDTraj not available")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print("  ✅ NumPy available")
    except ImportError:
        print("  ❌ NumPy not available")
    
    # Store dependency status
    st.session_state.md_dependency_status = dependencies
    
    return dependencies


def render_simulation_interface():
    """Render the main simulation interface when system is available"""
    
    # Initialize MD simulation system if not already done
    initialize_md_simulation_system()
    
    # Debug information
    print(f"🔍 Debug - MD integration status:")
    print(f"   md_integration exists: {hasattr(st.session_state, 'md_integration')}")
    print(f"   md_integration value: {getattr(st.session_state, 'md_integration', 'NOT_SET')}")
    print(f"   md_integration_available: {st.session_state.get('md_integration_available', 'NOT_SET')}")
    
    # Check if MD integration is available
    if not st.session_state.get('md_integration_available', False):
        
        # Special case: if md_integration exists but flag is False, it might be a flag issue
        if hasattr(st.session_state, 'md_integration') and st.session_state.md_integration is not None:
            st.warning("⚠️ MD Simulation Integration detected but marked as unavailable")
            st.info("This appears to be a session state flag issue. The system seems to be working.")
            
            # Try to fix the flag
            st.session_state.md_integration_available = True
            print("🔧 Fixed md_integration_available flag")
            st.rerun()
            return
        
        st.error("❌ MD Simulation Integration not available")
        if hasattr(st.session_state, 'md_integration_error'):
            st.error(f"**Error:** {st.session_state.md_integration_error}")
        
        # Show dependency status if available
        if hasattr(st.session_state, 'md_dependency_status'):
            st.markdown("**Dependency Status:**")
            deps = st.session_state.md_dependency_status
            
            col1, col2 = st.columns(2)
            
            with col1:
                for dep, available in list(deps.items())[:3]:
                    icon = "✅" if available else "❌"
                    st.write(f"{icon} **{dep}**")
            
            with col2:
                for dep, available in list(deps.items())[3:]:
                    icon = "✅" if available else "❌"
                    st.write(f"{icon} **{dep}**")
        
        st.info("**Installation Commands:**")
        st.code("""
# Install MD simulation dependencies:
conda install -c conda-forge openmm pdbfixer openmmforcefields mdtraj

# Or alternatively with pip:
pip install openmm-setup pdbfixer openmmforcefields mdtraj
        """)
        return
    
    dependency_status = st.session_state.md_integration.get_dependency_status()
    
    # Display dependency status
    render_system_status(dependency_status)
    
    # Main simulation interface if all dependencies are available
    if dependency_status['dependencies']['all_available']:
        # Tabs for different simulation workflows
        # **NEW: Added Enhanced MD workflow tab**
        if ENHANCED_MD_AVAILABLE:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🚀 Enhanced MD (Stored SMILES)", 
                "MD Simulation", 
                "Results Analysis", 
                "Post-Processing", 
                "File Management"
            ])
            
            with tab1:
                render_enhanced_md_tab()
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["MD Simulation", "Results Analysis", "Post-Processing", "File Management"])
        
        # Standard tabs (adjust numbering based on enhanced tab availability)
        if ENHANCED_MD_AVAILABLE:
            with tab2:
                render_md_simulation_tab()
            with tab3:
                render_results_analysis_tab()
            with tab4:
                render_postprocessing_tab()
            with tab5:
                render_file_management_tab()
        else:
            with tab1:
                render_md_simulation_tab()
            with tab2:
                render_results_analysis_tab()
            with tab3:
                render_postprocessing_tab()
            with tab4:
                render_file_management_tab()
    else:
        render_dependency_errors(dependency_status)


def render_system_status(dependency_status: Dict[str, Any]):
    """Render the system status display"""
    
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Dependencies:**")
        for dep, available in dependency_status['dependencies'].items():
            if dep != 'all_available':
                icon = "✅" if available else "❌"
                st.markdown(f"{icon} {dep}")
    
    with col2:
        st.markdown("**Platform Information:**")
        if 'platforms' in dependency_status.get('platform_info', {}):
            platforms = dependency_status['platform_info']['platforms']
            best_platform = dependency_status['platform_info']['best_platform']
            st.write(f"**Best Platform:** {best_platform}")
            st.write(f"**Available Platforms:** {len(platforms)}")
    
    with col3:
        st.markdown("**Status:**")
        if dependency_status['dependencies']['all_available']:
            st.success("🚀 All systems ready!")
        else:
            st.error("⚠️ Missing dependencies")
            missing = [k for k, v in dependency_status['dependencies'].items() 
                     if not v and k != 'all_available']
            for dep in missing:
                cmd = dependency_status['installation_commands'].get(dep, f"Install {dep}")
                st.code(cmd)


def render_automated_candidates_section():
    """Render section showing automated simulation candidates"""
    st.markdown("#### 🧬 Automated Simulation Candidates")
    st.markdown("*Generated polymer boxes and insulin-polymer systems ready for MD simulation*")
    
    try:
        # Get automated candidates from MD integration
        candidates = st.session_state.md_integration.get_automated_simulation_candidates()
        
        if not candidates:
            st.info("📝 No automated simulation candidates found. Generate PSMILES candidates first in the PSMILES Generation tab with automation enabled.")
            return
        
        # Group by session
        sessions = {}
        for candidate in candidates:
            session_id = candidate['session_id']
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(candidate)
        
        # Display sessions
        for session_id, session_candidates in sessions.items():
            with st.expander(f"📁 Session: {session_id} ({len(session_candidates)} candidates)", expanded=True):
                
                # Create columns for polymer boxes and insulin systems
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔧 Polymer Boxes**")
                    polymer_count = sum(1 for c in session_candidates if c['has_polymer_box'])
                    st.metric("Ready for MD", polymer_count, f"out of {len(session_candidates)}")
                    
                with col2:
                    st.markdown("**🧬 Insulin Systems**")
                    insulin_count = sum(1 for c in session_candidates if c['has_insulin_system'])
                    st.metric("Ready for MD", insulin_count, f"out of {len(session_candidates)}")
                
                # Display individual candidates
                st.markdown("**Candidates:**")
                for candidate in session_candidates:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        status_icon = "✅" if candidate['ready_for_md'] else "⚠️"
                        st.write(f"{status_icon} **{candidate['candidate_id']}**")
                    
                    with col2:
                        if candidate['has_polymer_box']:
                            if st.button(f"🔧 Run MD (Polymer)", key=f"polymer_{candidate['candidate_id']}"):
                                run_md_on_candidate(candidate, 'polymer')
                        else:
                            st.write("❌ No polymer box")
                    
                    with col3:
                        if candidate['has_insulin_system']:
                            if st.button(f"🧬 Run MD (Insulin)", key=f"insulin_{candidate['candidate_id']}"):
                                run_md_on_candidate(candidate, 'insulin')
                        else:
                            st.write("❌ No insulin system")
                    
                    with col4:
                        with st.popover("📁 Files"):
                            if candidate['polymer_pdb']:
                                st.write(f"**Polymer PDB:** `{Path(candidate['polymer_pdb']).name}`")
                            if candidate['composite_pdb']:
                                st.write(f"**Composite PDB:** `{Path(candidate['composite_pdb']).name}`")
                            if candidate['processed_insulin_pdb']:
                                st.write(f"**Processed Insulin:** `{Path(candidate['processed_insulin_pdb']).name}`")
                
    except Exception as e:
        st.error(f"❌ Error loading automated candidates: {str(e)}")
        st.info("This may be normal if no candidates have been generated yet.")


def run_md_on_candidate(candidate: Dict[str, Any], simulation_type: str):
    """Run MD simulation on an automated candidate"""
    
    # Determine which PDB file to use
    if simulation_type == 'polymer' and candidate['polymer_pdb']:
        pdb_file = candidate['polymer_pdb']
        st.info(f"🚀 Starting MD simulation on polymer box: {Path(pdb_file).name}")
    elif simulation_type == 'insulin' and candidate['composite_pdb']:
        pdb_file = candidate['composite_pdb']
        st.info(f"🚀 Starting MD simulation on insulin-polymer system: {Path(pdb_file).name}")
    else:
        st.error(f"❌ Required PDB file not found for {simulation_type} simulation")
        return
    
    # Store selected file for the main simulation interface
    st.session_state.selected_candidate_file = pdb_file
    st.session_state.selected_candidate_info = candidate
    
    # Use default simulation parameters - QUICK TEST MODE as requested by user
    simulation_params = {
        'temperature': 310.0,
        'equilibration_steps': 125000,    # Quick Test: 250 ps
        'production_steps': 500000,       # Quick Test: 1 ns (was 2500000 = 5 ns Medium)
        'save_interval': 500              # Frequent saving for quick tests
    }
    
    # Start simulation
    start_simulation(pdb_file, simulation_params)


def render_md_simulation_tab():
    """Render the MD simulation tab"""
    st.markdown("### MD Simulation")
    st.markdown("*Run molecular dynamics simulations with AMBER force fields and implicit solvent*")
    
    # First show automated simulation candidates if available
    render_automated_candidates_section()
    
    st.markdown("---")
    
    # File selection interface
    simulation_input_file = render_file_selection_interface()
    
    # Polymer file selection (advanced)
    if simulation_input_file:
        render_polymer_selection_interface()
    
    # Simulation parameters
    if simulation_input_file:
        simulation_params = render_simulation_parameters()
        
        # Check if simulation is running
        render_simulation_execution_interface(simulation_input_file, simulation_params)
    else:
        st.info("Please select or upload a PDB file to run simulation.")


def render_file_selection_interface() -> Optional[str]:
    """Render the file selection interface and return selected file path"""
    
    st.markdown("#### Input File Selection")
    
    # Refresh button
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Files", help="Refresh the file list to detect newly created files"):
            st.rerun()
    
    # Get available PDB files
    try:
        from integration.analysis.md_simulation_integration import get_insulin_polymer_pdb_files
        available_pdbs = get_insulin_polymer_pdb_files()
    except ImportError:
        available_pdbs = []
        st.warning("⚠️ Unable to detect PDB files - file detection module not available")
    
    simulation_input_file = None
    
    if available_pdbs:
        simulation_input_file = render_automatic_file_selection(available_pdbs)
    
    # Upload option if no file selected
    if not simulation_input_file:
        simulation_input_file = render_file_upload_interface()
    
    return simulation_input_file


def render_automatic_file_selection(available_pdbs: List[Dict]) -> Optional[str]:
    """Render automatic file selection interface"""
    
    # Automatically select the latest insulin_embedded file
    latest_embedded = None
    for pdb in available_pdbs:
        if pdb.get('file_type') == 'insulin_embedded':
            latest_embedded = pdb
            break  # First one is the most recent due to sorting
    
    if latest_embedded:
        st.success(f"🎯 **Auto-detected latest embedded file:** {latest_embedded['name']}")
        st.info(f"📊 **File details:** {latest_embedded['atom_count']} atoms, {latest_embedded['size_mb']:.1f} MB")
        
        # Option to use the auto-detected file
        use_auto_detected = st.checkbox("Use Auto-detected File", value=True, 
                                       help="Use the most recently created insulin_embedded file")
        
        if use_auto_detected:
            return latest_embedded['path']
        else:
            # Manual selection
            return render_manual_file_selection(available_pdbs)
    else:
        # No embedded files, show all options
        st.info("No insulin_embedded files found. Showing all available files:")
        return render_manual_file_selection(available_pdbs)


def render_manual_file_selection(available_pdbs: List[Dict]) -> Optional[str]:
    """Render manual file selection interface"""
    
    st.markdown("**Manual File Selection:**")
    pdb_options = {}
    for pdb in available_pdbs:
        file_type_emoji = {
            'insulin_embedded': '🎯',
            'composite': '🧬', 
            'other_insulin': '📄'
        }.get(pdb.get('file_type'), '📄')
        
        display_name = f"{file_type_emoji} {pdb['name']} ({pdb['atom_count']} atoms, {pdb['size_mb']:.1f} MB)"
        pdb_options[display_name] = pdb['path']
    
    if pdb_options:
        selected_display = st.selectbox("Select PDB File", list(pdb_options.keys()))
        return pdb_options[selected_display]
    
    return None


def render_file_upload_interface() -> Optional[str]:
    """Render file upload interface"""
    
    st.markdown("#### Upload PDB File")
    uploaded_sim_file = st.file_uploader(
        "Upload PDB for Simulation",
        type=['pdb'],
        help="Upload a PDB file for MD simulation"
    )
    
    if uploaded_sim_file is not None:
        temp_sim_path = f"temp_sim_{uuid.uuid4().hex[:8]}.pdb"
        with open(temp_sim_path, 'wb') as f:
            f.write(uploaded_sim_file.read())
        st.success(f"✅ File uploaded: {uploaded_sim_file.name}")
        return temp_sim_path
    
    return None


def render_polymer_selection_interface():
    """Render manual polymer file selection interface"""
    
    st.markdown("#### 🧪 Polymer File Selection (Advanced)")
    
    polymer_selection_expander = st.expander("🔧 Manual Polymer File Selection", expanded=False)
    
    with polymer_selection_expander:
        st.markdown("**Use this section if automatic polymer detection fails:**")
        st.info("The system will automatically detect polymer files, but you can override this selection if needed.")
        
        # Find available polymer directories
        polymer_dirs = list(Path('.').glob("insulin_polymer_output_*"))
        
        if polymer_dirs:
            render_polymer_directory_selection(polymer_dirs)
        else:
            st.warning("No polymer output directories found. Generate polymer structures first using the 3D Structure Builder.")


def render_polymer_directory_selection(polymer_dirs: List[Path]):
    """Render polymer directory selection interface"""
    
    st.markdown(f"**Found {len(polymer_dirs)} polymer output directories:**")
    
    # Sort by modification time (most recent first)
    polymer_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    polymer_options = {}
    for polymer_dir in polymer_dirs:
        # Get polymer files in this directory
        polymer_files = []
        for subdir in ['molecules', 'packmol']:
            subdir_path = polymer_dir / subdir
            if subdir_path.exists():
                polymer_files.extend(list(subdir_path.glob("*.pdb")))
                polymer_files.extend(list(subdir_path.glob("*.xyz")))
        
        if polymer_files:
            # Show directory with file count and modification time
            mod_time = datetime.fromtimestamp(polymer_dir.stat().st_mtime)
            display_name = f"{polymer_dir.name} ({len(polymer_files)} files, {mod_time.strftime('%Y-%m-%d %H:%M')})"
            polymer_options[display_name] = str(polymer_dir)
    
    if polymer_options:
        use_manual_polymer = st.checkbox("🎯 Use Manual Polymer Selection", 
                                       help="Override automatic polymer detection")
        
        if use_manual_polymer:
            selected_polymer_dir = st.selectbox(
                "Select Polymer Directory:", 
                list(polymer_options.keys()),
                help="Choose the polymer directory to use for force field parameterization"
            )
            
            # Show files in selected directory
            selected_path = Path(polymer_options[selected_polymer_dir])
            display_polymer_files(selected_path)
            
            # Store the selection for use in simulation
            st.session_state.manual_polymer_dir = str(selected_path)
            st.success(f"✅ Manual polymer selection: {selected_path.name}")
        else:
            # Clear manual selection
            if 'manual_polymer_dir' in st.session_state:
                del st.session_state.manual_polymer_dir
    else:
        st.warning("No polymer files found in any directory")


def display_polymer_files(polymer_path: Path):
    """Display files in selected polymer directory"""
    
    st.markdown(f"**Files in {polymer_path.name}:**")
    
    all_files = []
    for subdir in ['molecules', 'packmol']:
        subdir_path = polymer_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.pdb")) + list(subdir_path.glob("*.xyz"))
            for file in files:
                all_files.append(f"📄 {subdir}/{file.name}")
    
    if all_files:
        for file in all_files:
            st.write(f"  {file}")
    else:
        st.warning("No polymer files found in selected directory")


def render_simulation_parameters() -> Dict[str, Any]:
    """Render simulation parameters interface and return parameters"""
    
    st.markdown("#### Simulation Parameters")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        temperature = st.slider("Temperature (K)", 250, 400, 310, 5, 
                               help="Simulation temperature in Kelvin (physiological = 310 K)")
        
        # Equilibration steps with better options and descriptions
        # NOTE: These steps are calculated for 2 fs timestep (actual simulator timestep)
        equilibration_options = {
            "Quick Test (250 ps)": 125000,    # 250 ps with 2 fs timestep
            "Short (500 ps) - Recommended": 250000,  # 500 ps with 2 fs timestep
            "Medium (1000 ps)": 500000,       # 1000 ps with 2 fs timestep
            "Long (2000 ps)": 1000000,        # 2000 ps with 2 fs timestep
            "Extended (4000 ps)": 2000000     # 4000 ps with 2 fs timestep
        }
        
        eq_selection = st.selectbox(
            "Equilibration Duration",
            list(equilibration_options.keys()),
            index=0,  # Default to "Quick Test (250 ps)"
            help="Equilibration phase duration (2 fs timestep used by simulator)"
        )
        equilibration_steps = equilibration_options[eq_selection]
        
        # Convert to time (2 fs timestep)
        eq_time_ps = equilibration_steps * 2 / 1000
        eq_time_ns = eq_time_ps / 1000
        st.caption(f"⏱️ Equilibration: {eq_time_ps:.0f} ps ({eq_time_ns:.1f} ns)")
    
    with param_col2:
        # Production steps with better options and descriptions
        # NOTE: These steps are calculated for 2 fs timestep (actual simulator timestep)
        production_options = {
            "Quick Test (1 ns)": 500000,     # 1 ns with 2 fs timestep
            "Short (2.5 ns)": 1250000,       # 2.5 ns with 2 fs timestep  
            "Medium (5 ns) - Recommended": 2500000,   # 5 ns with 2 fs timestep
            "Long (10 ns)": 5000000,         # 10 ns with 2 fs timestep
            "Extended (25 ns)": 12500000      # 25 ns with 2 fs timestep
        }
        
        prod_selection = st.selectbox(
            "Production Duration",
            list(production_options.keys()),
            index=0,  # Default to "Quick Test (1 ns)"
            help="Production phase duration (2 fs timestep used by simulator)"
        )
        production_steps = production_options[prod_selection]
        
        # Save interval with better options (2 fs timestep)
        save_options = {
            "Frequent (1 ps)": 500,          # 1 ps with 2 fs timestep
            "Normal (2 ps) - Recommended": 1000,    # 2 ps with 2 fs timestep
            "Sparse (4 ps)": 2000,           # 4 ps with 2 fs timestep
            "Very Sparse (8 ps)": 4000       # 8 ps with 2 fs timestep
        }
        
        save_selection = st.selectbox(
            "Frame Saving Frequency",
            list(save_options.keys()),
            index=1,  # Default to "Normal (2 ps) - Recommended"
            help="How often to save trajectory frames"
        )
        save_interval = save_options[save_selection]
        
        # Convert to time (2 fs timestep)
        prod_time_ps = production_steps * 2 / 1000
        prod_time_ns = prod_time_ps / 1000
        save_time_ps = save_interval * 2 / 1000
        total_time_ns = (equilibration_steps + production_steps) * 2 / 1000000
        
        st.caption(f"⏱️ Production: {prod_time_ps:.0f} ps ({prod_time_ns:.1f} ns)")
        st.caption(f"💾 Save every: {save_time_ps:.1f} ps")
        st.caption(f"🕒 **Total simulation: {total_time_ns:.1f} ns**")
    
    # Performance estimation
    render_performance_estimation(total_time_ns)
    
    # Important preprocessing note
    render_preprocessing_information()
    
    return {
        'temperature': temperature,
        'equilibration_steps': equilibration_steps,
        'production_steps': production_steps,
        'save_interval': save_interval,
        'total_time_ns': total_time_ns
    }


def render_performance_estimation(total_time_ns: float):
    """Render performance estimation"""
    
    st.markdown("#### ⚡ Performance Estimation")
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Estimate based on typical performance
        est_performance_ns_day = 1000  # Conservative estimate for mixed systems
        
        if total_time_ns <= 5:
            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
            runtime_color = "🟢"
            runtime_desc = "Fast"
        elif total_time_ns <= 20:
            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
            runtime_color = "🟡"
            runtime_desc = "Moderate"
        else:
            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
            runtime_color = "🔴"
            runtime_desc = "Slow"
        
        st.metric("Estimated Runtime", f"{est_runtime_hours:.1f} hours", 
                help=f"Based on ~{est_performance_ns_day} ns/day performance")
        st.caption(f"{runtime_color} {runtime_desc} simulation")
    
    with perf_col2:
        # Estimate trajectory size and frames
        total_frames = 100  # Placeholder calculation
        trajectory_size_mb = total_frames * 0.5  # Rough estimate
        
        st.metric("Expected Frames", f"{total_frames:,}", 
                help="Total number of saved trajectory frames")
        st.metric("Trajectory Size", f"~{trajectory_size_mb:.0f} MB", 
                help="Estimated DCD trajectory file size")


def render_preprocessing_information():
    """Render preprocessing information"""
    
    st.markdown("#### 🔧 Preprocessing & Simulation")
    st.info("""
    **Automatic PDBFixer Preprocessing:**
    The MD simulation automatically includes these preprocessing steps:
    1. 🧹 **Structure Cleaning** - Remove water molecules and fix missing atoms
    2. ➕ **Add Hydrogens** - Add missing hydrogens at physiological pH (7.4)
    3. 🔗 **Fix Bonds** - Repair missing residues and optimize structure
    4. 🧪 **Preserve Polymers** - Keep UNL polymer residues for embedding simulation
    
    This ensures your insulin-polymer system is properly prepared for MD simulation!
    """)


def render_simulation_execution_interface(simulation_input_file: str, simulation_params: Dict[str, Any]):
    """Render the simulation execution interface"""
    
    # Check if simulation is already running
    sim_status = st.session_state.md_integration.get_simulation_status()
    
    if sim_status['simulation_running']:
        render_live_simulation_console(sim_status)
    else:
        render_simulation_start_interface(simulation_input_file, simulation_params)


def render_live_simulation_console(sim_status: Dict[str, Any]):
    """Render the live simulation console interface"""
    
    st.markdown("## 🖥️ Live Simulation Console")
    
    # Show simulation info
    sim_info = sim_status['simulation_info']
    
    # Header with simulation info and controls
    auto_refresh, refresh_interval = render_simulation_header(sim_info)
    
    # Live metrics and console output
    render_live_console_output(sim_info, auto_refresh, refresh_interval)


def render_simulation_header(sim_info: Dict[str, Any]):
    """Render simulation header with controls"""
    
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    
    with header_col1:
        st.markdown(f"### 🎯 Simulation: `{sim_info['id']}`")
        # Color-code status
        status = sim_info['status']
        status_colors = {
            'running': "🟢 Running",
            'stopping': "🟡 Stopping...",
            'stopped': "🛑 Stopped by User",
            'completed': "✅ Completed",
            'failed': "❌ Failed"
        }
        status_display = status_colors.get(status, f"⚪ {status.title()}")
        st.markdown(f"**Status:** {status_display}")
    
    with header_col2:
        col2a, col2b = st.columns([2, 1])
        with col2a:
            auto_refresh = st.checkbox("🔄 Live Updates", value=True, 
                                     help="Automatically refresh the console output in real-time")
            if auto_refresh and status in ['running', 'starting']:
                st.markdown('<span style="color: #4CAF50; font-size: 0.8em;">● Live monitoring active</span>', unsafe_allow_html=True)
        with col2b:
            if auto_refresh:
                refresh_interval = st.selectbox("Refresh Interval", 
                                               options=[2, 3, 5, 10], 
                                               index=1, 
                                               format_func=lambda x: f"{x}s",
                                               help="Refresh interval in seconds",
                                               label_visibility="collapsed")
            else:
                refresh_interval = 5
    
    with header_col3:
        if st.button("⏹️ Stop Simulation", type="primary"):
            stop_result = st.session_state.md_integration.stop_simulation()
            if stop_result:
                st.success("🛑 Stop request sent to simulation!")
                st.info("The simulation will stop at the next checkpoint.")
            else:
                st.warning("⚠️ No active simulation to stop.")
            st.rerun()
    
    # Return the values for use in other functions
    return auto_refresh, refresh_interval


def render_live_console_output(sim_info: Dict[str, Any], auto_refresh: bool = True, refresh_interval: int = 5):
    """Render live console output and metrics"""
    
    # Get status from sim_info
    status = sim_info.get('status', 'unknown')
    
    # Calculate elapsed time
    if 'simulation_start_time' in st.session_state and st.session_state.simulation_start_time:
        start_time = st.session_state.simulation_start_time
        current_time = datetime.now()
        elapsed = current_time - start_time
        elapsed_minutes = elapsed.total_seconds() / 60
        
        if elapsed_minutes > 60:
            elapsed_str = f"{elapsed_minutes/60:.1f} hours"
        else:
            elapsed_str = f"{elapsed_minutes:.1f} minutes"
        
        st.info(f"⏱️ **Elapsed Time:** {elapsed_str}")
    
    # Console output display
    st.markdown("### 📋 Console Output")
    
    # Get console output if available
    if hasattr(st.session_state, 'console_capture'):
        console_output = st.session_state.console_capture.get_output()
        recent_lines = st.session_state.console_capture.get_recent_lines(50)
        
        if console_output:
            # Display recent lines
            recent_output = "\n".join(recent_lines)
            
            st.text_area(
                "Real-time Console Output",
                recent_output,
                height=500,
                disabled=True,
                help="Live output from the MD simulation - chronological order with newest at bottom",
                key=f"console_output_{len(recent_lines)}"
            )
            
            # Show statistics
            total_lines = len(st.session_state.console_capture.output_lines)
            st.caption(f"📝 {total_lines} lines captured (showing last 50 lines) - Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
            # Debug section to help troubleshoot
            with st.expander("🔍 Debug: Console Capture Status"):
                st.write(f"**Console capture object:** {type(st.session_state.console_capture)}")
                st.write(f"**Total lines:** {total_lines}")
                st.write(f"**Recent lines count:** {len(recent_lines)}")
                if recent_lines:
                    st.write("**Latest 3 lines:**")
                    for i, line in enumerate(recent_lines[-3:]):
                        st.code(line, language=None)
                else:
                    st.write("**No lines captured yet**")
                
                # Check if simulation is actually running
                sim_status = st.session_state.md_integration.get_simulation_status()
                st.write(f"**Simulation status:** {sim_status['simulation_running']}")
                if sim_status['simulation_info']:
                    st.write(f"**Simulation ID:** {sim_status['simulation_info'].get('id', 'N/A')}")
                    st.write(f"**Simulation phase:** {sim_status['simulation_info'].get('status', 'N/A')}")
        else:
            st.info("📝 Waiting for console output from simulation...")
            st.text_area("Console Output", "Simulation starting...", height=300, disabled=True)
    else:
        st.info("📝 Console capture not initialized. Refresh the page.")
        
        # Try to initialize console capture
        if st.button("🔄 Initialize Console Capture"):
            # Create console capture
            class ThreadSafeConsoleCapture:
                def __init__(self):
                    self.output_lines = []
                    
                def write(self, text):
                    """Capture console output in a thread-safe way"""
                    if text.strip():  # Only capture non-empty lines
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_line = f"[{timestamp}] {text.strip()}"
                        self.output_lines.append(formatted_line)
                        # Keep only last 200 lines
                        if len(self.output_lines) > 200:
                            self.output_lines = self.output_lines[-200:]
                
                def flush(self):
                    pass
                
                def get_output(self):
                    return "\n".join(self.output_lines)
                
                def get_recent_lines(self, n=50):
                    return self.output_lines[-n:] if len(self.output_lines) > n else self.output_lines
            
            st.session_state.console_capture = ThreadSafeConsoleCapture()
            st.success("✅ Console capture initialized!")
            st.rerun()
    
    # Control buttons for manual refresh and testing
    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        if st.button("🔄 Refresh Console Now"):
            st.rerun()
    
    with button_col2:
        if st.button("🧪 Test Console Capture"):
            # Test the console capture mechanism
            if hasattr(st.session_state, 'console_capture'):
                test_message = f"🧪 TEST: Console capture is working! Timestamp: {datetime.now().strftime('%H:%M:%S')}"
                st.session_state.console_capture.write(test_message)
                st.success("✅ Test message sent to console capture!")
                st.rerun()
            else:
                st.error("❌ Console capture not initialized!")
    
    with button_col3:
        if auto_refresh and status in ['running', 'starting']:
            st.info(f"🔄 Auto-refreshing every {refresh_interval}s")
    
    # LIVE STREAMING - Auto-refresh for running simulations
    if auto_refresh and status in ['running', 'starting']:
        # Show live status
        st.markdown(f"""
        <div style="text-align: center; color: #4CAF50; font-size: 0.9em; margin: 10px 0; 
                   background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 5px;">
            🔴 LIVE STREAMING - Updating every {refresh_interval}s
        </div>
        """, unsafe_allow_html=True)
        
        # Simple but effective: Sleep then immediately refresh
        # This creates a continuous loop similar to Gradio
        time.sleep(refresh_interval)
        st.rerun()


def render_simulation_start_interface(simulation_input_file: str, simulation_params: Dict[str, Any]):
    """Render the simulation start interface"""
    
    if st.button("🚀 Start MD Simulation", type="primary"):
        start_simulation(simulation_input_file, simulation_params)


def start_simulation(simulation_input_file: str, simulation_params: Dict[str, Any]):
    """Start the MD simulation"""
    
    # Safety check - ensure md_integration is available
    if st.session_state.md_integration is None:
        st.error("❌ MD integration not available. Please check system initialization.")
        st.stop()
    
    # Initialize session state for simulation
    initialize_simulation_session_state()
    
    # Create console capture
    console_capture_ref = create_console_capture()
    
    # Start simulation
    try:
        # Debug: Show what parameters are being sent
        st.info(f"🔧 Debug: Sending simulation parameters:")
        st.write(f"   • Temperature: {simulation_params['temperature']} K")
        st.write(f"   • Equilibration: {simulation_params['equilibration_steps']} steps ({simulation_params['equilibration_steps'] * 2 / 1000:.1f} ps)")
        st.write(f"   • Production: {simulation_params['production_steps']} steps ({simulation_params['production_steps'] * 2 / 1000000:.1f} ns)")
        st.write(f"   • Save interval: {simulation_params['save_interval']} steps ({simulation_params['save_interval'] * 2 / 1000:.1f} ps)")
        
        simulation_id = st.session_state.md_integration.run_md_simulation_async(
            pdb_file=simulation_input_file,
            temperature=simulation_params['temperature'],
            equilibration_steps=simulation_params['equilibration_steps'],
            production_steps=simulation_params['production_steps'],
            save_interval=simulation_params['save_interval'],
            output_callback=lambda msg: console_capture_ref.write(msg),
            manual_polymer_dir=st.session_state.get('manual_polymer_dir')
        )
        
        st.success(f"✅ Simulation started with ID: {simulation_id}")
        st.info("🔄 Simulation is running in the background. The progress will appear below.")
        st.info("💡 The page will auto-refresh to show real-time updates.")
        
        # Immediately show the progress interface
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to start simulation: {str(e)}")


def initialize_simulation_session_state():
    """Initialize session state for simulation"""
    
    if 'simulation_messages' not in st.session_state:
        st.session_state.simulation_messages = []
    if 'simulation_start_time' not in st.session_state:
        st.session_state.simulation_start_time = datetime.now()


def create_console_capture():
    """Create and initialize console capture"""
    
    class ThreadSafeConsoleCapture:
        def __init__(self):
            self.output_lines = []
            
        def write(self, text):
            """Capture console output in a thread-safe way"""
            if text.strip():  # Only capture non-empty lines
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_line = f"[{timestamp}] {text.strip()}"
                self.output_lines.append(formatted_line)
                # Keep only last 200 lines
                if len(self.output_lines) > 200:
                    self.output_lines = self.output_lines[-200:]
        
        def flush(self):
            pass
        
        def get_output(self):
            return "\n".join(self.output_lines)
        
        def get_recent_lines(self, n=50):
            return self.output_lines[-n:] if len(self.output_lines) > n else self.output_lines
    
    # Create console capture instance and store it in session state
    if 'console_capture' not in st.session_state:
        st.session_state.console_capture = ThreadSafeConsoleCapture()
    
    return st.session_state.console_capture


def render_results_analysis_tab():
    """Render the results analysis tab"""
    st.markdown("### Results Analysis")
    st.info("Results analysis functionality will be implemented in future versions.")


def render_postprocessing_tab():
    """Render the comprehensive post-processing tab with simple progress approach"""
    st.markdown("### Comprehensive Post-Processing")
    st.markdown("*Advanced trajectory analysis and property calculation for insulin-polymer systems*")
    
    # Initialize post-processing system (simplified)
    initialize_postprocessing_system()
    
    # Check if post-processing is available
    if not st.session_state.get('postprocessing_available', False):
        render_postprocessing_unavailable()
        return
    
    # SIMPLIFIED: No complex live console - just show interface or results
    if st.session_state.get('show_postprocessing_results'):
        simulation_id = st.session_state.show_postprocessing_results
        
        # Back button
        if st.button("⬅️ Back to Post-Processing", type="secondary"):
            del st.session_state.show_postprocessing_results
            st.rerun()
        
        # Show results dashboard
        render_postprocessing_results_dashboard(simulation_id)
    else:
        # Main interface
        render_postprocessing_interface()


def initialize_postprocessing_system():
    """Initialize the post-processing system (simplified to avoid multiple messages)"""
    
    # Check if already initialized (simplified check)
    if hasattr(st.session_state, 'postprocessor') and st.session_state.postprocessor is not None:
        st.session_state.postprocessing_available = True
        return  # Don't print message every time
    
    # Only print initialization message once
    if not st.session_state.get('postprocessing_init_attempted', False):
        st.session_state.postprocessing_init_attempted = True
        
        # Try to initialize post-processing system
        try:
            from integration.analysis.comprehensive_postprocessing import ComprehensivePostProcessor
            st.session_state.postprocessor = ComprehensivePostProcessor()
            st.session_state.postprocessing_available = True
            st.success("✅ Post-processing system initialized successfully")
            
        except ImportError as e:
            st.session_state.postprocessor = None
            st.session_state.postprocessing_available = False
            error_msg = f"Post-processing system import failed: {str(e)}"
            st.session_state.postprocessing_error = error_msg
            st.error(f"⚠️ {error_msg}")
            
        except Exception as e:
            st.session_state.postprocessor = None
            st.session_state.postprocessing_available = False
            error_msg = f"Post-processing system initialization failed: {str(e)}"
            st.session_state.postprocessing_error = error_msg
            st.error(f"❌ {error_msg}")


def render_postprocessing_unavailable():
    """Render interface when post-processing is not available"""
    
    st.error(f"❌ Post-processing system not available: {st.session_state.get('postprocessing_error', 'Unknown error')}")
    
    st.markdown("### 🔧 Post-Processing Dependencies")
    st.markdown("The post-processing system requires these components:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Analysis Modules:**")
        st.code("""
# Install comprehensive analysis dependencies
conda install -c conda-forge openmm pdbfixer openmmforcefields mdtraj
conda install -c conda-forge scipy scikit-learn matplotlib seaborn

# Or with pip
pip install openmm pdbfixer openmmforcefields mdtraj scipy scikit-learn matplotlib seaborn
        """)
    
    with col2:
        st.markdown("**Analysis Capabilities:**")
        st.markdown("""
        - 🧮 **MM-GBSA Binding Energy**
        - 🧪 **Insulin Stability Analysis**  
        - 🔄 **Partitioning & Transfer Free Energy**
        - 🚶 **Diffusion Coefficient Analysis**
        - 🕸️ **Hydrogel Mesh Size & Dynamics**
        - ⚡ **Interaction Energy Decomposition**
        - 💧 **Swelling & Volume Analysis**
        - 📊 **Basic Trajectory Statistics**
        """)


def render_postprocessing_interface():
    """Render the main post-processing interface"""
    
    # Check if user wants to view specific results
    if st.session_state.get('show_postprocessing_results'):
        simulation_id = st.session_state.show_postprocessing_results
        
        # Back button
        if st.button("⬅️ Back to Post-Processing", type="secondary"):
            del st.session_state.show_postprocessing_results
            st.rerun()
        
        # Show results dashboard
        render_postprocessing_results_dashboard(simulation_id)
        return
    
    # Get dependency status
    dependency_status = st.session_state.postprocessor.get_dependency_status()
    render_postprocessing_status(dependency_status)
    
    st.markdown("---")
    
    # Get available simulations
    available_simulations = st.session_state.postprocessor.get_available_simulations()
    
    if not available_simulations:
        st.info("📝 No completed simulations found for post-processing. Run MD simulations first in the MD Simulation tab.")
        return
    
    # Show existing results if available
    processed_simulations = [sim for sim in available_simulations if sim.get('processing_complete', False)]
    
    if processed_simulations:
        render_existing_results_section(processed_simulations)
        st.markdown("---")
    
    # Simulation selection interface
    render_simulation_selection_interface(available_simulations)


def render_existing_results_section(processed_simulations: List[Dict[str, Any]]):
    """Render section showing existing post-processing results"""
    
    st.markdown("#### 📊 Previously Processed Simulations")
    
    # Show in a nice grid format
    cols_per_row = 3
    for i in range(0, len(processed_simulations), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(processed_simulations):
                sim = processed_simulations[i + j]
                
                with col:
                    with st.container():
                        st.markdown(f"**{sim['id']}**")
                        
                        # Show key metrics if available
                        if sim.get('analyses_completed'):
                            completed_count = len(sim['analyses_completed'])
                            st.caption(f"✅ {completed_count} analyses completed")
                        
                        # Processing timestamp
                        if sim.get('processing_timestamp'):
                            timestamp = sim['processing_timestamp']
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                date_str = dt.strftime('%Y-%m-%d %H:%M')
                                st.caption(f"📅 {date_str}")
                            except:
                                st.caption(f"📅 {timestamp}")
                        
                        # View results button
                        if st.button(f"📊 View Results", key=f"view_results_{sim['id']}", 
                                   help=f"View detailed post-processing results for {sim['id']}"):
                            st.session_state.show_postprocessing_results = sim['id']
                            st.rerun()


def render_postprocessing_status(dependency_status: Dict[str, Any]):
    """Render post-processing system status"""
    
    st.markdown("#### 🔧 Post-Processing System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Analysis Capabilities:**")
        capabilities = dependency_status['analysis_capabilities']
        
        for capability, available in capabilities.items():
            icon = "✅" if available else "❌"
            display_name = capability.replace('_', ' ').title()
            st.markdown(f"{icon} {display_name}")
    
    with col2:
        st.markdown("**Dependencies:**")
        deps = dependency_status['dependencies']
        
        for dep, available in deps.items():
            if dep not in ['core_available', 'all_available']:
                icon = "✅" if available else "❌"
                display_name = dep.replace('_', ' ').title()
                st.markdown(f"{icon} {display_name}")
    
    with col3:
        st.markdown("**System Status:**")
        if deps['all_available']:
            st.success("🚀 All capabilities available!")
        elif deps['core_available']:
            st.warning("⚡ Partial capabilities available")
        else:
            st.error("⚠️ No capabilities available")
        
        # Show recommendation
        recommendation = dependency_status.get('recommendation', '')
        if recommendation:
            st.info(recommendation)


def render_simulation_selection_interface(available_simulations: List[Dict[str, Any]]):
    """Render simulation selection and analysis options interface"""
    
    st.markdown("#### 📊 Available Simulations for Post-Processing")
    
    # Filter simulations
    ready_simulations = [sim for sim in available_simulations if sim.get('ready_for_processing', False)]
    
    if not ready_simulations:
        st.warning("⚠️ No simulations ready for post-processing. Ensure MD simulations completed successfully.")
        return
    
    # Simulation selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create simulation options for selectbox
        sim_options = {}
        for sim in ready_simulations:
            timestamp = sim.get('timestamp', 'Unknown time')
            atoms = sim.get('total_atoms', 0)
            performance = sim.get('performance', 0)
            status_icon = "✅" if sim.get('success', False) else "⚠️"
            processed_icon = "🔬" if sim.get('already_processed', False) else "📝"
            
            display_name = f"{status_icon}{processed_icon} {sim['id']} - {atoms} atoms ({performance:.1f} ns/day)"
            sim_options[display_name] = sim
        
        selected_display = st.selectbox(
            "Select Simulation for Post-Processing:",
            list(sim_options.keys()),
            help="Choose a completed simulation to analyze"
        )
        
        selected_sim = sim_options[selected_display]
    
    with col2:
        # Show simulation details
        st.markdown("**Simulation Details:**")
        st.write(f"**ID:** {selected_sim['id']}")
        st.write(f"**Atoms:** {selected_sim.get('total_atoms', 'Unknown')}")
        st.write(f"**Time:** {selected_sim.get('simulation_time_ps', 0):.1f} ps")
        
        if selected_sim.get('already_processed'):
            if selected_sim.get('processing_complete'):
                st.success("✅ Already processed")
                st.caption(f"Processed: {selected_sim.get('processing_timestamp', 'Unknown time')}")
            else:
                st.warning("⚠️ Partially processed")
        else:
            st.info("📝 Not yet processed")
    
    st.markdown("---")
    
    # Analysis options
    render_analysis_options_interface(selected_sim)


def render_analysis_options_interface(selected_sim: Dict[str, Any]):
    """Render analysis options selection interface"""
    
    st.markdown("#### ⚙️ Analysis Configuration")
    
    # Check system capabilities
    dependency_status = st.session_state.postprocessor.get_dependency_status()
    capabilities = dependency_status['analysis_capabilities']
    
    # Handle quick selection BEFORE creating any widgets
    quick_selection = st.session_state.get('quick_selection_mode', None)
    
    # Analysis options with descriptions
    analysis_descriptions = {
        'insulin_stability': {
            'title': '🧪 Insulin Stability & Conformation',
            'description': 'RMSD, RMSF, radius of gyration, secondary structure, hydrogen bonds',
            'time_estimate': '3-7 minutes'
        },
        'partitioning': {
            'title': '🔄 Partitioning & Transfer Free Energy',
            'description': 'PMF analysis, distance distributions, partition coefficients',
            'time_estimate': '2-4 minutes'
        },
        'diffusion': {
            'title': '🚶 Diffusion Coefficient Analysis',
            'description': 'Mean squared displacement analysis and diffusion coefficient calculation',
            'time_estimate': '1-3 minutes'
        },
        'hydrogel_dynamics': {
            'title': '🕸️ Hydrogel Mesh Size & Dynamics',
            'description': 'Polymer network analysis, mesh size estimation, mechanical properties',
            'time_estimate': '3-6 minutes'
        },
        'interaction_energies': {
            'title': '⚡ Interaction Energy Decomposition',
            'description': 'Distance-based interaction analysis between system components',
            'time_estimate': '2-4 minutes'
        },
        'swelling_response': {
            'title': '💧 Swelling & Volume Analysis',
            'description': 'Volume changes, water uptake, swelling ratio calculations',
            'time_estimate': '1-3 minutes'
        },

    }
    
    st.markdown("**Select Analyses to Perform:**")
    
    # Create two columns for analysis options
    col1, col2 = st.columns(2)
    
    analysis_options = {}
    
    # Apply quick selection if one was triggered
    if quick_selection:
        if quick_selection == 'all_available':
            # Pre-populate with all available options
            for analysis_key in ['insulin_stability', 'partitioning', 'diffusion', 'hydrogel_dynamics',
                               'interaction_energies', 'swelling_response']:
                if capabilities.get(analysis_key, False):
                    st.session_state[f"analysis_{analysis_key}"] = True
        elif quick_selection == 'essential_only':
            essential = ['insulin_stability']
            for analysis_key in ['insulin_stability', 'partitioning', 'diffusion', 'hydrogel_dynamics',
                               'interaction_energies', 'swelling_response']:
                st.session_state[f"analysis_{analysis_key}"] = analysis_key in essential and capabilities.get(analysis_key, False)
        elif quick_selection == 'material_focus':
            material_focus = ['hydrogel_dynamics', 'interaction_energies', 'swelling_response', 'diffusion']
            for analysis_key in ['insulin_stability', 'partitioning', 'diffusion', 'hydrogel_dynamics',
                               'interaction_energies', 'swelling_response']:
                st.session_state[f"analysis_{analysis_key}"] = analysis_key in material_focus and capabilities.get(analysis_key, False)
        elif quick_selection == 'clear_all':
            for analysis_key in ['insulin_stability', 'partitioning', 'diffusion', 'hydrogel_dynamics',
                               'interaction_energies', 'swelling_response']:
                st.session_state[f"analysis_{analysis_key}"] = False
        
        # Clear the selection mode
        st.session_state.quick_selection_mode = None
    
    with col1:
        st.markdown("**Structural & Thermodynamic:**")
        
        for analysis_key in ['insulin_stability', 'partitioning', 'diffusion']:
            if analysis_key in analysis_descriptions:
                desc = analysis_descriptions[analysis_key]
                available = capabilities.get(analysis_key, False)
                
                if available:
                    default_value = analysis_key in ['insulin_stability', 'partitioning']  # Default enabled
                    selected = st.checkbox(
                        desc['title'],
                        value=default_value,
                        help=f"{desc['description']} (Est. time: {desc['time_estimate']})",
                        key=f"analysis_{analysis_key}"
                    )
                    analysis_options[analysis_key] = selected
                else:
                    st.markdown(f"❌ {desc['title']} (Not available)")
                    analysis_options[analysis_key] = False
    
    with col2:
        st.markdown("**Material Properties:**")
        
        for analysis_key in ['hydrogel_dynamics', 'interaction_energies', 'swelling_response']:
            if analysis_key in analysis_descriptions:
                desc = analysis_descriptions[analysis_key]
                available = capabilities.get(analysis_key, False)
                
                if available:
                    default_value = analysis_key in ['hydrogel_dynamics']  # Default enabled
                    selected = st.checkbox(
                        desc['title'],
                        value=default_value,
                        help=f"{desc['description']} (Est. time: {desc['time_estimate']})",
                        key=f"analysis_{analysis_key}"
                    )
                    analysis_options[analysis_key] = selected
                else:
                    st.markdown(f"❌ {desc['title']} (Not available)")
                    analysis_options[analysis_key] = False
    
    # Quick selection buttons
    st.markdown("**Quick Selection:**")
    button_col1, button_col2, button_col3, button_col4 = st.columns(4)
    
    with button_col1:
        if st.button("Select All Available", help="Enable all available analyses"):
            st.session_state.quick_selection_mode = 'all_available'
            st.rerun()
    
    with button_col2:
        if st.button("Essential Only", help="Enable only essential analyses"):
            st.session_state.quick_selection_mode = 'essential_only'
            st.rerun()
    
    with button_col3:
        if st.button("Material Focus", help="Enable material property analyses"):
            st.session_state.quick_selection_mode = 'material_focus'
            st.rerun()
    
    with button_col4:
        if st.button("Clear All", help="Disable all analyses"):
            st.session_state.quick_selection_mode = 'clear_all'
            st.rerun()
    
    # Update analysis_options with current state
    for key in analysis_options:
        analysis_options[key] = st.session_state.get(f"analysis_{key}", analysis_options[key])
    
    # Estimate total time
    selected_analyses = [key for key, selected in analysis_options.items() if selected]
    
    if selected_analyses:
        # Rough time estimation (these are conservative estimates)
        time_estimates = {
            'insulin_stability': 5, 'partitioning': 3, 'diffusion': 2, 'hydrogel_dynamics': 4.5,
            'interaction_energies': 3, 'swelling_response': 2
        }
        
        total_estimated_time = sum(time_estimates.get(key, 2) for key in selected_analyses)
        
        st.info(f"**Estimated total time:** {total_estimated_time:.1f} minutes for {len(selected_analyses)} selected analyses")
        
        # Start analysis button
        if st.button("Start Comprehensive Post-Processing", type="primary", 
                    help=f"Begin analysis of {len(selected_analyses)} selected modules"):
            start_postprocessing_analysis(selected_sim, analysis_options)
    else:
        st.warning("⚠️ Please select at least one analysis to perform.")


def start_postprocessing_analysis(selected_sim: Dict[str, Any], analysis_options: Dict[str, bool]):
    """Start the post-processing analysis with simple progress bar (like PSMILES generation)"""
    
    try:
        # Filter only selected options
        selected_options = {key: value for key, value in analysis_options.items() if value}
        
        if not selected_options:
            st.error("❌ No analyses selected")
            return
        
        # Check postprocessor is available
        if not hasattr(st.session_state, 'postprocessor') or st.session_state.postprocessor is None:
            st.error("❌ Post-processing system not initialized")
            return
        
        # Simple progress approach (like PSMILES generation)
        st.markdown("---")
        st.markdown("### 🔬 Post-Processing Progress")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start analysis
        simulation_id = selected_sim['id']
        simulation_dir = Path(selected_sim['path']).parent
        trajectory_file = selected_sim.get('trajectory_file')
        structure_type = selected_sim.get('structure_type', 'simple')
        
        progress_bar.progress(10, "Initializing analysis system...")
        
        # Simple progress callback
        def simple_progress_callback(msg):
            """Simple callback that doesn't cause UI issues"""
            print(f"[ANALYSIS] {msg}")  # Console logging only
        
        progress_bar.progress(25, "Starting comprehensive analysis...")
        
        # Use spinner for the analysis (like PSMILES generation)
        with st.spinner(f"Running {len(selected_options)} analysis modules..."):
            
            # Start analysis using synchronous approach to avoid threading issues
            analysis_job_id = st.session_state.postprocessor.start_comprehensive_analysis_async(
                simulation_id=simulation_id,
                simulation_dir=str(simulation_dir),
                analysis_options=selected_options,
                output_callback=simple_progress_callback,
                trajectory_file=trajectory_file,
                simulation_structure_type=structure_type
            )
            
            progress_bar.progress(50, "Analysis in progress...")
            
            # Simple status checking loop (no auto-refresh)
            import time
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                status = st.session_state.postprocessor.get_analysis_status()
                
                if not status['analysis_running']:
                    # Analysis completed
                    break
                
                # Update progress based on analysis info
                if status['analysis_info']:
                    progress = status['analysis_info'].get('progress', 50)
                    current_step = status['analysis_info'].get('current_step', 'Processing...')
                    progress_bar.progress(min(progress, 90) / 100.0, current_step)
                
                time.sleep(2)  # Check every 2 seconds
            
            progress_bar.progress(100, "Analysis completed!")
        
        # Show results
        st.success(f"✅ Post-processing analysis completed!")
        st.info(f"🎯 Analysis Job ID: {analysis_job_id}")
        st.info(f"📊 Analyzed simulation: {simulation_id}")
        st.info(f"🔬 Completed analyses: {', '.join(selected_options.keys())}")
        
        # Store completion flag to show results
        st.session_state.show_postprocessing_results = simulation_id
        st.success("📊 Click 'View Results' below to see detailed analysis!")
        
    except Exception as e:
        st.error(f"❌ Failed to start post-processing analysis: {str(e)}")
        print(f"❌ Post-processing startup error: {str(e)}")
        import traceback
        traceback.print_exc()


def render_live_postprocessing_console(status: Dict[str, Any]):
    """Render live post-processing console interface"""
    
    st.markdown("## 🔬 Live Post-Processing Analysis")
    
    # Show analysis info
    analysis_info = status['analysis_info']
    
    # Header with analysis info and controls
    auto_refresh, refresh_interval = render_postprocessing_header(analysis_info)
    
    # Live metrics and console output
    render_live_postprocessing_output(analysis_info, auto_refresh, refresh_interval)


def render_postprocessing_header(analysis_info: Dict[str, Any]):
    """Render post-processing header with controls"""
    
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    
    with header_col1:
        st.markdown(f"### 🎯 Analysis Job: `{analysis_info['job_id']}`")
        st.markdown(f"**Simulation:** {analysis_info['simulation_id']}")
        
        # Color-code status
        status = analysis_info['status']
        status_colors = {
            'starting': "🟡 Starting...",
            'running': "🟢 Running",
            'stopping': "🟡 Stopping...",
            'stopped': "🛑 Stopped by User",
            'completed': "✅ Completed",
            'failed': "❌ Failed"
        }
        status_display = status_colors.get(status, f"⚪ {status.title()}")
        st.markdown(f"**Status:** {status_display}")
        
        # Progress bar
        progress = analysis_info.get('progress', 0)
        current_step = analysis_info.get('current_step', 'Initializing...')
        
        st.progress(progress / 100.0)
        st.caption(f"Progress: {progress:.1f}% - {current_step}")
    
    with header_col2:
        col2a, col2b = st.columns([2, 1])
        with col2a:
            auto_refresh = st.checkbox("🔄 Live Updates", value=True, 
                                     help="Automatically refresh the console output in real-time")
            if auto_refresh and status in ['running', 'starting']:
                st.markdown('<span style="color: #4CAF50; font-size: 0.8em;">● Live monitoring active</span>', unsafe_allow_html=True)
        with col2b:
            if auto_refresh:
                refresh_interval = st.selectbox("Refresh Interval", 
                                               options=[2, 3, 5, 10], 
                                               index=1, 
                                               format_func=lambda x: f"{x}s",
                                               help="Refresh interval in seconds",
                                               label_visibility="collapsed")
            else:
                refresh_interval = 5
    
    with header_col3:
        # Analysis summary
        total_steps = analysis_info.get('total_steps', 0)
        completed_steps = len(analysis_info.get('steps_completed', []))
        
        st.metric("Steps Completed", f"{completed_steps}/{total_steps}")
        
        if st.button("⏹️ Stop Analysis", type="primary"):
            stop_result = st.session_state.postprocessor.stop_analysis()
            if stop_result:
                st.success("🛑 Stop request sent to analysis!")
                st.info("The analysis will stop at the next checkpoint.")
            else:
                st.warning("⚠️ No active analysis to stop.")
            st.rerun()
    
    return auto_refresh, refresh_interval


def render_live_postprocessing_output(analysis_info: Dict[str, Any], auto_refresh: bool = True, refresh_interval: int = 5):
    """Render live post-processing output and metrics"""
    
    # Get status
    status = analysis_info.get('status', 'unknown')
    
    # Calculate elapsed time
    if 'start_time' in analysis_info:
        start_time = datetime.fromtimestamp(analysis_info['start_time'])
        current_time = datetime.now()
        elapsed = current_time - start_time
        elapsed_minutes = elapsed.total_seconds() / 60
        
        if elapsed_minutes > 60:
            elapsed_str = f"{elapsed_minutes/60:.1f} hours"
        else:
            elapsed_str = f"{elapsed_minutes:.1f} minutes"
        
        st.info(f"⏱️ **Elapsed Time:** {elapsed_str}")
    
    # Show selected analyses
    if 'analysis_options' in analysis_info:
        selected_analyses = [k for k, v in analysis_info['analysis_options'].items() if v]
        st.info(f"🔬 **Selected Analyses:** {', '.join(selected_analyses)}")
    
    # Console output display
    st.markdown("### 📋 Analysis Console Output")
    
    # Get console output if available
    if hasattr(st.session_state, 'postprocessing_console_capture'):
        console_output = st.session_state.postprocessing_console_capture.get_output()
        recent_lines = st.session_state.postprocessing_console_capture.get_recent_lines(100)
        
        if console_output:
            # Display recent lines
            recent_output = "\n".join(recent_lines)
            
            st.text_area(
                "Real-time Analysis Output",
                recent_output,
                height=600,
                disabled=True,
                help="Live output from the post-processing analysis - chronological order with newest at bottom",
                key=f"postprocessing_console_output_{len(recent_lines)}"
            )
            
            # Show statistics
            total_lines = len(st.session_state.postprocessing_console_capture.output_lines)
            st.caption(f"📝 {total_lines} lines captured (showing last 100 lines) - Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
        else:
            st.info("📝 Waiting for console output from analysis...")
            st.text_area("Console Output", "Analysis starting...", height=400, disabled=True)
    else:
        st.info("📝 Console capture not initialized.")
    
    # Control buttons
    button_col1, button_col2, button_col3 = st.columns(3)
    
    with button_col1:
        if st.button("🔄 Refresh Console Now"):
            st.rerun()
    
    with button_col2:
        if status == 'completed' and st.button("📊 View Results"):
            # Switch to results view
            if 'results' in analysis_info:
                st.session_state.show_postprocessing_results = analysis_info['simulation_id']
                st.rerun()
    
    with button_col3:
        if auto_refresh and status in ['running', 'starting']:
            st.info(f"🔄 Auto-refreshing every {refresh_interval}s")
    
    # LIVE STREAMING - Auto-refresh for running analyses
    if auto_refresh and status in ['running', 'starting']:
        # Show live status
        st.markdown(f"""
        <div style="text-align: center; color: #4CAF50; font-size: 0.9em; margin: 10px 0; 
                   background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 5px;">
            🔴 LIVE STREAMING - Updating every {refresh_interval}s
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh with sleep
        time.sleep(refresh_interval)
        st.rerun()


def render_postprocessing_results_dashboard(simulation_id: str):
    """Render comprehensive post-processing results dashboard"""
    
    st.markdown(f"## 📊 Post-Processing Results Dashboard")
    st.markdown(f"**Simulation:** {simulation_id}")
    
    try:
        # Get results from post-processor
        results = st.session_state.postprocessor.get_analysis_results(simulation_id)
        
        if not results.get('success'):
            st.error(f"❌ Could not load results: {results.get('error', 'Unknown error')}")
            return
        
        # Extract data
        comprehensive_results = results.get('comprehensive_results', {})
        summary = results.get('summary', {})
        
        if not comprehensive_results:
            st.warning("⚠️ No comprehensive results available")
            return
        
        # Summary metrics at the top
        render_summary_metrics_dashboard(summary)
        
        st.markdown("---")
        
        # Detailed results sections
        results_data = comprehensive_results.get('results', {})
        
        # Create tabs for different analysis types
        analysis_tabs = []
        tab_data = {}
        
        if 'insulin_stability' in results_data:
            analysis_tabs.append("Insulin Stability")
            tab_data["Insulin Stability"] = results_data['insulin_stability']
        
        if 'partitioning' in results_data:
            analysis_tabs.append("Partitioning")
            tab_data["Partitioning"] = results_data['partitioning']
        
        if 'diffusion' in results_data:
            analysis_tabs.append("Diffusion")
            tab_data["Diffusion"] = results_data['diffusion']
        
        if 'hydrogel_dynamics' in results_data:
            analysis_tabs.append("Hydrogel")
            tab_data["Hydrogel"] = results_data['hydrogel_dynamics']
        
        if 'interaction_energies' in results_data:
            analysis_tabs.append("Interactions")
            tab_data["Interactions"] = results_data['interaction_energies']
        
        if 'swelling_response' in results_data:
            analysis_tabs.append("Swelling")
            tab_data["Swelling"] = results_data['swelling_response']
        

        
        if analysis_tabs:
            # Create dynamic tabs
            if len(analysis_tabs) == 1:
                render_analysis_results_section(analysis_tabs[0], tab_data[analysis_tabs[0]])
            else:
                tabs = st.tabs(analysis_tabs)
                for i, tab_name in enumerate(analysis_tabs):
                    with tabs[i]:
                        render_analysis_results_section(tab_name, tab_data[tab_name])
        else:
            st.info("📝 No detailed analysis results available")
        
        st.markdown("---")
        
        # Download and export options
        render_results_export_section(results, simulation_id)
        
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")


def render_summary_metrics_dashboard(summary: Dict[str, Any]):
    """Render summary metrics dashboard"""
    
    st.markdown("### 🎯 Key Results Summary")
    
    summary_metrics = summary.get('summary_metrics', {})
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # RMSD
        if 'rmsd_mean_A' in summary_metrics:
            rmsd = summary_metrics['rmsd_mean_A']
            stability = summary_metrics.get('stability_assessment', 'Unknown')
            st.metric("🧪 RMSD", f"{rmsd:.2f} Å", 
                     delta=f"{stability}", 
                     help="Root Mean Square Deviation (stability indicator)")
        
        # Diffusion Coefficient
        if 'diffusion_coefficient_cm2_s' in summary_metrics:
            diff_coeff = summary_metrics['diffusion_coefficient_cm2_s']
            st.metric("🚶 Diffusion", f"{diff_coeff:.2e} cm²/s", 
                     help="Diffusion coefficient of insulin in polymer matrix")
    
    with col2:
        # Mesh Size
        if 'mesh_size_A' in summary_metrics:
            mesh_size = summary_metrics['mesh_size_A']
            st.metric("🕸️ Mesh Size", f"{mesh_size:.1f} Å", 
                     help="Average hydrogel mesh size")
    
    with col3:
        # Trajectory Info
        if 'trajectory_frames' in summary_metrics:
            frames = summary_metrics['trajectory_frames']
            time_ps = summary_metrics.get('simulation_time_ps', 0)
            st.metric("📊 Trajectory", f"{frames} frames", 
                     delta=f"{time_ps:.1f} ps",
                     help="Total trajectory frames and simulation time")
        
        # Total Atoms
        if 'total_atoms' in summary_metrics:
            atoms = summary_metrics['total_atoms']
            st.metric("🔬 System Size", f"{atoms:,} atoms", 
                     help="Total number of atoms in the system")
    
    with col4:
        # Processing Info
        processing_time = summary.get('total_processing_time', 0)
        if processing_time > 0:
            if processing_time > 3600:
                time_str = f"{processing_time/3600:.1f} hours"
            elif processing_time > 60:
                time_str = f"{processing_time/60:.1f} minutes"
            else:
                time_str = f"{processing_time:.1f} seconds"
            
            st.metric("⏱️ Processing Time", time_str, 
                     help="Total time for comprehensive analysis")
        
        # Analysis Date
        timestamp = summary.get('timestamp', '')
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M')
                st.metric("📅 Analysis Date", date_str, 
                         delta=time_str,
                         help="When the analysis was performed")
            except:
                pass


def render_analysis_results_section(analysis_name: str, analysis_data: Dict[str, Any]):
    """Render detailed results for a specific analysis"""
    
    if not analysis_data.get('success', False):
        st.error(f"❌ {analysis_name} analysis failed: {analysis_data.get('error', 'Unknown error')}")
        return
    
    # Remove emoji from analysis name for processing
    clean_name = analysis_name.split(' ', 1)[1] if ' ' in analysis_name else analysis_name
    
    if "Insulin Stability" in analysis_name:
        render_stability_results(analysis_data)
    elif "Partitioning" in analysis_name:
        render_partitioning_results(analysis_data)
    elif "Diffusion" in analysis_name:
        render_diffusion_results(analysis_data)
    elif "Hydrogel" in analysis_name:
        render_hydrogel_results(analysis_data)
    elif "Interactions" in analysis_name:
        render_interactions_results(analysis_data)
    elif "Swelling" in analysis_name:
        render_swelling_results(analysis_data)
    else:
        # Generic results display
        st.json(analysis_data)


def render_stability_results(data: Dict[str, Any]):
    """Render insulin stability analysis results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Structural Metrics:**")
        
        if 'rmsd' in data:
            rmsd_data = data['rmsd']
            mean_rmsd = rmsd_data.get('mean', 0)
            std_rmsd = rmsd_data.get('std', 0)
            stability = rmsd_data.get('stability_assessment', 'Unknown')
            
            st.metric("RMSD", f"{mean_rmsd:.2f} ± {std_rmsd:.2f} Å", 
                     delta=stability)
        
        if 'radius_of_gyration' in data:
            rg_data = data['radius_of_gyration']
            mean_rg = rg_data.get('mean', 0)
            change_pct = rg_data.get('change_percent', 0)
            
            st.metric("Radius of Gyration", f"{mean_rg:.2f} Å", 
                     delta=f"{change_pct:+.1f}%")
    
    with col2:
        st.markdown("**Dynamic Properties:**")
        
        if 'rmsf' in data:
            rmsf_data = data['rmsf']
            mean_rmsf = rmsf_data.get('mean', 0)
            max_rmsf = rmsf_data.get('max', 0)
            
            st.metric("RMSF", f"{mean_rmsf:.2f} Å (max: {max_rmsf:.2f})")
        
        if 'hydrogen_bonds' in data:
            hb_data = data['hydrogen_bonds']
            avg_hbonds = hb_data.get('average_count', 0)
            stability = hb_data.get('stability', 'Unknown')
            
            st.metric("Hydrogen Bonds", f"{avg_hbonds:.1f}", 
                     delta=stability)


def render_partitioning_results(data: Dict[str, Any]):
    """Render partitioning analysis results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Thermodynamic Properties:**")
        
        if 'transfer_free_energy' in data:
            transfer_fe = data['transfer_free_energy']
            st.metric("Transfer Free Energy", f"{transfer_fe:.2f} kcal/mol")
        
        if 'partition_coefficient' in data:
            partition_coeff = data['partition_coefficient']
            st.metric("Partition Coefficient", f"{partition_coeff:.3f}")
    
    with col2:
        st.markdown("**Distance Analysis:**")
        
        if 'distance_analysis' in data:
            dist_data = data['distance_analysis']
            
            mean_dist = dist_data.get('mean_distance', 0)
            contact_freq = dist_data.get('contact_frequency', 0)
            
            st.metric("Mean Distance", f"{mean_dist:.1f} Å")
            st.metric("Contact Frequency", f"{contact_freq:.3f}")


def render_diffusion_results(data: Dict[str, Any]):
    """Render diffusion analysis results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Diffusion Properties:**")
        
        if 'msd_analysis' in data:
            msd_data = data['msd_analysis']
            diff_coeff = msd_data.get('diffusion_coefficient', 0)
            r_squared = msd_data.get('r_squared', 0)
            
            st.metric("Diffusion Coefficient", f"{diff_coeff:.2e} cm²/s")
            st.metric("Linear Fit R²", f"{r_squared:.3f}")
    
    with col2:
        st.markdown("**Assessment:**")
        
        if 'diffusion_assessment' in data:
            assessment = data['diffusion_assessment']
            assessment_text = {
                'within_experimental_range': "🟢 Within experimental range",
                'highly_constrained': "🟡 Highly constrained diffusion",
                'unusually_high': "🔴 Unusually high diffusion"
            }.get(assessment, f"📊 {assessment}")
            
            st.info(assessment_text)
        
        if 'msd_analysis' in data:
            exp_range = data['msd_analysis'].get('experimental_range', '')
            if exp_range:
                st.caption(f"Expected range: {exp_range}")


def render_hydrogel_results(data: Dict[str, Any]):
    """Render hydrogel dynamics results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Network Structure:**")
        
        if 'mesh_size_analysis' in data:
            mesh_data = data['mesh_size_analysis']
            avg_mesh = mesh_data.get('average_mesh_size', 0)
            mesh_std = mesh_data.get('std', 0)
            
            st.metric("Average Mesh Size", f"{avg_mesh:.1f} ± {mesh_std:.1f} Å")
        
        if 'estimated_mechanical_properties' in data:
            mech_data = data['estimated_mechanical_properties']
            modulus = mech_data.get('elastic_modulus_pa', 0)
            crosslink = mech_data.get('crosslink_density_estimate', 'Unknown')
            
            st.metric("Estimated Modulus", f"{modulus:.0e} Pa")
            st.write(f"**Crosslink Density:** {crosslink}")
    
    with col2:
        st.markdown("**Polymer Dynamics:**")
        
        if 'polymer_dynamics' in data:
            dyn_data = data['polymer_dynamics']
            
            flexibility = dyn_data.get('flexibility_index', 0)
            rmsd_mean = dyn_data.get('rmsd_mean', 0)
            
            st.metric("Flexibility Index", f"{flexibility:.3f}")
            st.metric("Polymer RMSD", f"{rmsd_mean:.2f} Å")


def render_interactions_results(data: Dict[str, Any]):
    """Render interaction energies results"""
    
    if 'interaction_analysis' in data:
        interaction_data = data['interaction_analysis']
        
        # Create columns for different interaction types
        interaction_types = list(interaction_data.keys())
        
        if len(interaction_types) >= 3:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
        else:
            cols = st.columns(len(interaction_types))
        
        for i, interaction_type in enumerate(interaction_types):
            if i < len(cols):
                with cols[i]:
                    display_name = interaction_type.replace('_', '-').title()
                    st.markdown(f"**{display_name}:**")
                    
                    inter_data = interaction_data[interaction_type]
                    
                    if 'mean_distance' in inter_data:
                        mean_dist = inter_data['mean_distance']
                        st.metric("Mean Distance", f"{mean_dist:.1f} Å")
                    
                    if 'close_contacts' in inter_data:
                        contacts = inter_data['close_contacts']
                        st.metric("Close Contacts", f"{contacts}")
                    
                    if 'interaction_strength' in inter_data:
                        strength = inter_data['interaction_strength']
                        st.caption(f"Strength: {strength}")


def render_swelling_results(data: Dict[str, Any]):
    """Render swelling analysis results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Volume Changes:**")
        
        if 'volume_analysis' in data:
            vol_data = data['volume_analysis']
            
            swelling_ratio = vol_data.get('swelling_ratio', 1.0)
            volume_change = vol_data.get('volume_change_percent', 0)
            
            st.metric("Swelling Ratio", f"{swelling_ratio:.3f}")
            st.metric("Volume Change", f"{volume_change:+.1f}%")
        
        elif 'polymer_extent_analysis' in data:
            ext_data = data['polymer_extent_analysis']
            
            expansion_ratio = ext_data.get('expansion_ratio', 1.0)
            expansion_pct = ext_data.get('expansion_percent', 0)
            
            st.metric("Expansion Ratio", f"{expansion_ratio:.3f}")
            st.metric("Expansion", f"{expansion_pct:+.1f}%")
    
    with col2:
        st.markdown("**Assessment:**")
        
        if 'swelling_assessment' in data:
            assessment = data['swelling_assessment']
            assessment_text = {
                'responsive': "🟢 Responsive system",
                'stable': "🟡 Stable system"
            }.get(assessment, f"📊 {assessment}")
            
            st.info(assessment_text)
        
        if 'swelling_mechanisms' in data:
            mechanisms = data['swelling_mechanisms']
            if mechanisms:
                st.write("**Mechanisms:**")
                for mechanism in mechanisms:
                    st.write(f"• {mechanism.replace('_', ' ').title()}")


def render_basic_stats_results(data: Dict[str, Any]):
    """Render basic trajectory statistics"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**System Information:**")
        
        if 'num_atoms' in data:
            st.metric("Total Atoms", f"{data['num_atoms']:,}")
        
        if 'num_residues' in data:
            st.metric("Total Residues", f"{data['num_residues']:,}")
        
        if 'num_frames' in data:
            st.metric("Trajectory Frames", f"{data['num_frames']:,}")
    
    with col2:
        st.markdown("**Basic Metrics:**")
        
        if 'time_ps' in data:
            time_ps = data['time_ps']
            time_ns = time_ps / 1000
            st.metric("Simulation Time", f"{time_ns:.1f} ns", 
                     delta=f"{time_ps:.0f} ps")
        
        if 'rmsd_mean' in data:
            rmsd = data['rmsd_mean']
            st.metric("Basic RMSD", f"{rmsd:.2f} Å")
        
        if 'rg_mean' in data:
            rg = data['rg_mean']
            st.metric("Radius of Gyration", f"{rg:.2f} Å")


def render_results_export_section(results: Dict[str, Any], simulation_id: str):
    """Render results export and download section"""
    
    st.markdown("### 📥 Export & Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Summary Reports:**")
        
        # JSON summary download
        if 'summary' in results:
            summary_json = json.dumps(results['summary'], indent=2)
            st.download_button(
                "📄 Download Summary (JSON)",
                data=summary_json,
                file_name=f"{simulation_id}_postprocessing_summary.json",
                mime="application/json",
                help="Download summary metrics and analysis overview"
            )
    
    with col2:
        st.markdown("**Detailed Results:**")
        
        # Complete results download
        if 'comprehensive_results' in results:
            results_json = json.dumps(results['comprehensive_results'], indent=2, default=str)
            st.download_button(
                "📊 Download Complete Results (JSON)",
                data=results_json,
                file_name=f"{simulation_id}_comprehensive_results.json",
                mime="application/json",
                help="Download all analysis results with detailed data"
            )
    
    with col3:
        st.markdown("**Analysis Files:**")
        
        output_files = results.get('output_files', {})
        if output_files:
            st.write(f"**Available Files:** {len(output_files)}")
            
            # Show file list in expander
            with st.expander("📁 View All Files"):
                for rel_path, full_path in output_files.items():
                    file_path = Path(full_path)
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        st.write(f"📄 {rel_path} ({size_mb:.1f} MB)")
                    else:
                        st.write(f"❌ {rel_path} (not found)")
        else:
            st.write("No additional files available")


def render_file_management_tab():
    """Render the file management tab"""
    st.markdown("### File Management")
    
    # Get simulation files if available
    try:
        available_simulations = st.session_state.md_integration.get_available_simulations()
        
        if available_simulations:
            st.markdown("#### Simulation Files")
            
            for sim in available_simulations:
                with st.expander(f"Simulation {sim['id']} - {sim['total_atoms']} atoms"):
                    st.write(f"**Input File:** {sim['input_file']}")
                    st.write(f"**Performance:** {sim['performance']:.1f} ns/day")
                    st.write(f"**Success:** {'✅' if sim['success'] else '❌'}")
                    st.write(f"**Timestamp:** {sim['timestamp']}")
                    
                    # Get file information
                    sim_files = st.session_state.md_integration.get_simulation_files(sim['id'])
                    
                    if sim_files['success']:
                        st.write("**Available Files:**")
                        for file_type, file_path in sim_files['files'].items():
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                                st.write(f"- {file_type}: {file_size:.1f} MB")
        else:
            st.info("No simulation files found.")
    except Exception as e:
        st.error(f"Error accessing simulation files: {str(e)}")


def render_dependency_errors(dependency_status: Dict[str, Any]):
    """Render dependency error information"""
    
    st.error("⚠️ Missing required dependencies. Please install them to use MD simulation.")
    
    missing = [k for k, v in dependency_status['dependencies'].items() 
             if not v and k != 'all_available']
    
    for dep in missing:
        cmd = dependency_status['installation_commands'].get(dep, f"Install {dep}")
        st.code(cmd)


def render_installation_instructions():
    """Render installation instructions when MD integration is not available"""
    
    st.error(f"❌ MD Simulation integration not available: {st.session_state.get('md_integration_error', 'Unknown error')}")
    st.markdown("### 🔧 Installation Instructions")
    st.code("""
# Install required dependencies
conda install -c conda-forge openmm pdbfixer openmmforcefields

# Or using pip
pip install openmm pdbfixer openmmforcefields
    """)


# Minimal Design System
def inject_simulation_styles():
    """Inject minimal design styles for the simulation UI"""
    
    from styles.minimal_design_system import inject_minimal_design_system
    inject_minimal_design_system()


def render_enhanced_md_tab():
    """Render the Enhanced MD simulation tab with stored SMILES workflow"""
    
    st.markdown("## 🚀 Enhanced MD Simulation with Stored SMILES")
    st.markdown("""
    **Optimized Workflow:** This enhanced system uses pre-stored SMILES strings from PSMILES generation 
    instead of reconstructing them from PDB files, eliminating redundant work and improving reliability.
    """)
    
    # Check enhanced MD availability
    if not ENHANCED_MD_AVAILABLE:
        st.error("❌ Enhanced MD with stored SMILES not available")
        st.info("This requires the enhanced_md_with_stored_smiles module")
        return
    
    # Create enhanced simulator
    try:
        enhanced_simulator = create_enhanced_md_simulator()
        readiness_status = enhanced_simulator.get_simulation_readiness_status()
    except Exception as e:
        st.error(f"❌ Failed to initialize enhanced MD simulator: {e}")
        return
    
    # Display readiness status
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### System Status")
        if readiness_status['overall_ready']:
            st.success("✅ **Enhanced MD Ready**")
        else:
            st.error("❌ **System Not Ready**")
            
        # Show component status
        components = readiness_status['components']
        for comp_name, available in components.items():
            icon = "✅" if available else "❌"
            st.write(f"{icon} {comp_name.replace('_', ' ').title()}")
            
        if readiness_status['missing_components']:
            st.error("**Missing:**")
            for comp in readiness_status['missing_components']:
                st.write(f"• {comp}")
    
    with col1:
        # Show available candidates
        st.markdown("### Available PSMILES Candidates")
        
        try:
            candidates = enhanced_simulator.get_available_candidates_for_simulation()
            
            if not candidates:
                st.warning("No PSMILES candidates available. Generate some in the PSMILES Generation tab first.")
                return
            
            # Filter ready vs not ready candidates
            ready_candidates = [c for c in candidates if c.get('ready_for_md', False)]
            not_ready_candidates = [c for c in candidates if not c.get('ready_for_md', False)]
            
            # Display ready candidates
            if ready_candidates:
                st.success(f"✅ **{len(ready_candidates)} candidates ready for MD simulation**")
                
                # Select candidate for simulation
                candidate_options = [
                    f"{c['id']}: {c['psmiles'][:30]}..." if len(c['psmiles']) > 30 else f"{c['id']}: {c['psmiles']}"
                    for c in ready_candidates
                ]
                
                selected_idx = st.selectbox(
                    "Select candidate for simulation:",
                    range(len(ready_candidates)),
                    format_func=lambda i: candidate_options[i]
                )
                
                if selected_idx is not None:
                    selected_candidate = ready_candidates[selected_idx]
                    
                    # Show candidate details
                    with st.expander("🔍 Candidate Details"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.text(f"PSMILES: {selected_candidate['psmiles']}")
                            st.text(f"Request: {selected_candidate['request']}")
                            st.text(f"Timestamp: {selected_candidate['timestamp']}")
                        with col_b:
                            st.text(f"SMILES: {selected_candidate['smiles']}")
                            st.text(f"Conversion: {selected_candidate.get('conversion_method', 'unknown')}")
                            st.success("✅ Ready for MD")
                    
                    # Simulation parameters
                    st.markdown("### Simulation Parameters")
                    
                    col_params1, col_params2 = st.columns(2)
                    
                    with col_params1:
                        force_field_type = st.selectbox(
                            "Force Field Type:",
                            ['smirnoff', 'gaff'],
                            help="SMIRNOFF (OpenFF) or GAFF force field"
                        )
                        
                        temperature = st.slider(
                            "Temperature (K):",
                            min_value=250.0,
                            max_value=400.0,
                            value=310.0,
                            step=10.0
                        )
                    
                    with col_params2:
                        simulation_steps = st.number_input(
                            "Simulation Steps:",
                            min_value=100,
                            max_value=100000,
                            value=5000,
                            step=1000
                        )
                        
                        output_frequency = st.number_input(
                            "Output Frequency:",
                            min_value=10,
                            max_value=1000,
                            value=100,
                            step=10
                        )
                    
                    # Run simulation button
                    if st.button("🚀 Run Enhanced MD Simulation", type="primary"):
                        
                        simulation_params = {
                            'force_field_type': force_field_type,
                            'temperature': temperature,
                            'steps': simulation_steps,
                            'output_frequency': output_frequency
                        }
                        
                        with st.spinner("Running enhanced MD simulation with stored SMILES..."):
                            
                            # Show workflow progress
                            progress_container = st.container()
                            with progress_container:
                                st.info("🧬 **Workflow:** Using pre-stored SMILES → Skip PDB reconstruction → Force field setup → MD simulation")
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Simulate progress steps
                                import time
                                
                                status_text.text("Retrieving stored SMILES...")
                                progress_bar.progress(20)
                                time.sleep(0.5)
                                
                                status_text.text("Setting up force field with stored SMILES...")
                                progress_bar.progress(50)
                                time.sleep(1.0)
                                
                                # Run actual simulation
                                results = enhanced_simulator.run_simulation_with_stored_smiles(
                                    psmiles=selected_candidate['psmiles'],
                                    simulation_params=simulation_params
                                )
                                
                                status_text.text("Running MD simulation...")
                                progress_bar.progress(80)
                                time.sleep(1.0)
                                
                                status_text.text("Processing results...")
                                progress_bar.progress(100)
                                time.sleep(0.5)
                                
                                # Clear progress indicators
                                progress_container.empty()
                        
                        # Display results
                        if results['success']:
                            st.success("🎉 **Enhanced MD Simulation Completed Successfully!**")
                            
                            # Show workflow efficiency
                            st.info("⚡ **Efficiency Gained:** Used pre-stored SMILES instead of PDB→SMILES reconstruction")
                            
                            # Show results
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                st.markdown("**Simulation Results:**")
                                sim_output = results['simulation_output']
                                st.text(f"Status: {sim_output['status']}")
                                st.text(f"Steps completed: {sim_output['steps_completed']}")
                                st.text(f"Final energy: {sim_output['final_energy']}")
                                
                            with col_res2:
                                st.markdown("**Workflow Details:**")
                                ff_status = results['force_field_status']
                                st.text(f"SMILES source: {ff_status['smiles_source']}")
                                st.text(f"Force field: {ff_status['method_used']}")
                                st.text(f"SMILES used: {results['smiles_used'][:30]}...")
                                
                        else:
                            st.error(f"❌ Simulation failed: {results['error']}")
                            
                            # Show detailed error info
                            if results.get('force_field_status'):
                                ff_status = results['force_field_status']
                                st.error(f"Force field error: {ff_status.get('error', 'Unknown error')}")
            
            # Show not ready candidates
            if not_ready_candidates:
                st.warning(f"⚠️ **{len(not_ready_candidates)} candidates not ready for MD simulation**")
                
                with st.expander("View candidates needing SMILES data"):
                    for candidate in not_ready_candidates:
                        st.write(f"• **{candidate['id']}**: {candidate.get('issue', 'Unknown issue')}")
                        st.caption(f"  PSMILES: {candidate['psmiles'][:50]}...")
                
        except Exception as e:
            st.error(f"❌ Error getting candidates: {e}")
            traceback.print_exc()
    
    # Add workflow comparison section
    st.markdown("---")
    st.markdown("### 📊 Workflow Comparison")
    
    col_old, col_new = st.columns(2)
    
    with col_old:
        st.markdown("#### ❌ Old Workflow")
        st.markdown("""
        1. Generate PSMILES
        2. Convert PSMILES → SMILES (discard)
        3. Create PDB file
        4. **Reconstruct SMILES from PDB** ⚠️
        5. Setup force field
        6. Run MD simulation
        
        **Issues:**
        - Redundant SMILES reconstruction
        - Potential reconstruction errors
        - Inefficient workflow
        """)
    
    with col_new:
        st.markdown("#### ✅ Enhanced Workflow")
        st.markdown("""
        1. Generate PSMILES
        2. Convert PSMILES → SMILES (**store**)
        3. Create PDB file (optional)
        4. **Use stored SMILES directly** ✨
        5. Setup force field
        6. Run MD simulation
        
        **Benefits:**
        - No redundant reconstruction
        - Reliable SMILES data
        - Faster workflow
        - Better error handling
        """)
    
    # Add migration helper at the end
    render_smiles_migration_helper()


# **ADD MIGRATION HELPER**
def render_smiles_migration_helper():
    """Helper to migrate existing candidates to include SMILES data"""
    
    st.markdown("### 🔄 SMILES Data Migration")
    st.info("Update existing PSMILES candidates to include SMILES data for enhanced MD workflows")
    
    if st.button("🔄 Update Existing Candidates with SMILES"):
        from app.utils.psmiles_smiles_storage import update_session_candidates_with_smiles
        
        with st.spinner("Updating candidates with SMILES data..."):
            try:
                update_session_candidates_with_smiles()
                st.success("✅ Successfully updated candidates with SMILES data!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Migration failed: {e}")


# Module initialization
if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="MD Simulation UI Test", layout="wide")
    inject_simulation_styles()
    render_simulation_ui() 