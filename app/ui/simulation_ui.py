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
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

def render_simulation_ui():
    """
    Render the complete MD Simulation UI interface
    
    Returns:
        None
    """
    st.subheader("🔬 Molecular Dynamics Simulation Integration")
    
    # Initialize MD simulation system
    initialize_md_simulation_system()
    
    # Check system status and render appropriate interface
    if st.session_state.get('md_integration_available', False):
        render_simulation_interface()
    else:
        render_installation_instructions()


def initialize_md_simulation_system():
    """Initialize the MD simulation system with error handling"""
    
    if 'md_integration' not in st.session_state:
        try:
            from integration.analysis.md_simulation_integration import MDSimulationIntegration
            st.session_state.md_integration = MDSimulationIntegration()
            st.session_state.md_integration_available = True
        except ImportError:
            st.session_state.md_integration = None
            st.session_state.md_integration_available = False
            st.session_state.md_integration_error = "MD simulation integration not available - missing dependencies"
        except Exception as e:
            st.session_state.md_integration = None
            st.session_state.md_integration_available = False
            st.session_state.md_integration_error = f"MD simulation initialization failed: {str(e)}"


def render_simulation_interface():
    """Render the main simulation interface when system is available"""
    
    dependency_status = st.session_state.md_integration.get_dependency_status()
    
    # Display dependency status
    render_system_status(dependency_status)
    
    # Main simulation interface if all dependencies are available
    if dependency_status['dependencies']['all_available']:
        # Tabs for different simulation workflows
        tab1, tab2, tab3 = st.tabs(["🚀 MD Simulation", "📊 Results Analysis", "📁 File Management"])
        
        with tab1:
            render_md_simulation_tab()
        
        with tab2:
            render_results_analysis_tab()
        
        with tab3:
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


def render_md_simulation_tab():
    """Render the MD simulation tab"""
    st.markdown("### 🚀 MD Simulation")
    st.markdown("*Run molecular dynamics simulations with AMBER force fields and implicit solvent*")
    
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
        equilibration_options = {
            "Quick Test (125 ps)": 31250,
            "Short (500 ps) - Recommended": 125000,
            "Medium (1000 ps)": 250000,
            "Long (2000 ps)": 500000,
            "Extended (4000 ps)": 1000000
        }
        
        eq_selection = st.selectbox(
            "Equilibration Duration",
            list(equilibration_options.keys()),
            index=1,  # Default to "Short (500 ps) - Recommended"
            help="Equilibration phase duration (4 fs timestep with hydrogen mass repartitioning)"
        )
        equilibration_steps = equilibration_options[eq_selection]
        
        # Convert to time
        eq_time_ps = equilibration_steps * 4 / 1000
        eq_time_ns = eq_time_ps / 1000
        st.caption(f"⏱️ Equilibration: {eq_time_ps:.0f} ps ({eq_time_ns:.1f} ns)")
    
    with param_col2:
        # Production steps with better options and descriptions
        production_options = {
            "Quick Test (1 ns)": 250000,
            "Short (5 ns)": 1250000,
            "Medium (10 ns) - Recommended": 2500000,
            "Long (20 ns)": 5000000,
            "Extended (50 ns)": 12500000
        }
        
        prod_selection = st.selectbox(
            "Production Duration",
            list(production_options.keys()),
            index=2,  # Default to "Medium (10 ns) - Recommended"
            help="Production phase duration (4 fs timestep with hydrogen mass repartitioning)"
        )
        production_steps = production_options[prod_selection]
        
        # Save interval with better options
        save_options = {
            "Frequent (1 ps)": 250,
            "Normal (2 ps) - Recommended": 500,
            "Sparse (4 ps)": 1000,
            "Very Sparse (8 ps)": 2000
        }
        
        save_selection = st.selectbox(
            "Frame Saving Frequency",
            list(save_options.keys()),
            index=1,  # Default to "Normal (2 ps) - Recommended"
            help="How often to save trajectory frames"
        )
        save_interval = save_options[save_selection]
        
        # Convert to time
        prod_time_ps = production_steps * 4 / 1000
        prod_time_ns = prod_time_ps / 1000
        save_time_ps = save_interval * 4 / 1000
        total_time_ns = (equilibration_steps + production_steps) * 4 / 1000000
        
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
    render_simulation_header(sim_info)
    
    # Live metrics and console output
    render_live_console_output(sim_info)


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
        auto_refresh = st.checkbox("🔄 Live Updates", value=True, 
                                 help="Automatically refresh the console output in real-time")
        if auto_refresh and status in ['running', 'starting']:
            st.markdown('<span style="color: #4CAF50; font-size: 0.8em;">● Live monitoring active</span>', unsafe_allow_html=True)
    
    with header_col3:
        if st.button("⏹️ Stop Simulation", type="primary"):
            stop_result = st.session_state.md_integration.stop_simulation()
            if stop_result:
                st.success("🛑 Stop request sent to simulation!")
                st.info("The simulation will stop at the next checkpoint.")
            else:
                st.warning("⚠️ No active simulation to stop.")
            st.rerun()


def render_live_console_output(sim_info: Dict[str, Any]):
    """Render live console output and metrics"""
    
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
        else:
            st.info("📝 Waiting for console output from simulation...")
            st.text_area("Console Output", "Simulation starting...", height=300, disabled=True)
    else:
        st.info("📝 Console capture not initialized. Refresh the page.")


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
    st.markdown("### 📊 Results Analysis")
    st.info("Results analysis functionality will be implemented in future versions.")


def render_file_management_tab():
    """Render the file management tab"""
    st.markdown("### 📁 File Management")
    
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


# CSS Styles for the module
def inject_simulation_styles():
    """Inject CSS styles for the simulation UI"""
    
    st.markdown("""
    <style>
    .simulation-dashboard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .simulation-status {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .console-output {
        background: #1e1e1e;
        color: #ffffff;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 5px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .performance-metric {
        text-align: center;
        padding: 0.5rem;
        margin: 0.25rem;
        border-radius: 5px;
        background: #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)


# Module initialization
if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="MD Simulation UI Test", layout="wide")
    inject_simulation_styles()
    render_simulation_ui() 