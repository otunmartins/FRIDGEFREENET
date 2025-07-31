#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDB Visualizer Component for Insulin-AI App
Provides 3D molecular visualization using 3Dmol.js for trajectory analysis

Features:
- Frame-by-frame trajectory navigation
- Protein and ligand representation styles
- Transparency controls for insulin and polymer
- Interactive 3D visualization with zoom/rotate
- Multiple color schemes and representation modes
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import base64
import re


def create_3dmol_viewer(pdb_content: str, viewer_id: str = "pdb_viewer", 
                       height: int = 600, insulin_transparent: bool = False,
                       polymer_transparent: bool = False, 
                       insulin_style: str = "cartoon", 
                       polymer_style: str = "stick") -> str:
    """
    Create a 3Dmol.js viewer for PDB content
    
    Args:
        pdb_content: PDB file content as string
        viewer_id: Unique ID for the viewer element
        height: Height of the viewer in pixels
        insulin_transparent: Whether to make insulin transparent
        polymer_transparent: Whether to make polymer transparent
        insulin_style: Visualization style for insulin ("cartoon", "stick", "sphere")
        polymer_style: Visualization style for polymer ("stick", "sphere", "line")
    
    Returns:
        HTML string for the 3Dmol.js viewer
    """
    
    # Escape the PDB content for JavaScript
    pdb_escaped = json.dumps(pdb_content)
    
    # Define transparency values
    insulin_alpha = 0.3 if insulin_transparent else 1.0
    polymer_alpha = 0.3 if polymer_transparent else 1.0
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <style>
            #{viewer_id} {{
                height: {height}px;
                width: 100%;
                position: relative;
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
            .viewer-controls {{
                margin: 10px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
                font-family: sans-serif;
                font-size: 14px;
            }}
            .control-group {{
                display: inline-block;
                margin-right: 20px;
                vertical-align: top;
            }}
            .control-label {{
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
                display: block;
            }}
            button {{
                background: #007bff;
                color: white;
                border: none;
                padding: 5px 10px;
                margin: 2px;
                border-radius: 3px;
                cursor: pointer;
            }}
            button:hover {{
                background: #0056b3;
            }}
            button.active {{
                background: #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-controls">
            <div class="control-group">
                <span class="control-label">🧪 Insulin Style:</span>
                <button onclick="setInsulinStyle('cartoon')" id="insulin-cartoon" class="{'active' if insulin_style == 'cartoon' else ''}">Cartoon</button>
                <button onclick="setInsulinStyle('stick')" id="insulin-stick" class="{'active' if insulin_style == 'stick' else ''}">Stick</button>
                <button onclick="setInsulinStyle('sphere')" id="insulin-sphere" class="{'active' if insulin_style == 'sphere' else ''}">Sphere</button>
            </div>
            
            <div class="control-group">
                <span class="control-label">🧬 Polymer Style:</span>
                <button onclick="setPolymerStyle('stick')" id="polymer-stick" class="{'active' if polymer_style == 'stick' else ''}">Stick</button>
                <button onclick="setPolymerStyle('sphere')" id="polymer-sphere" class="{'active' if polymer_style == 'sphere' else ''}">Sphere</button>
                <button onclick="setPolymerStyle('line')" id="polymer-line" class="{'active' if polymer_style == 'line' else ''}">Line</button>
            </div>
            
            <div class="control-group">
                <span class="control-label">👻 Transparency:</span>
                <button onclick="toggleInsulinTransparency()" id="insulin-transparency">Insulin: {'ON' if insulin_transparent else 'OFF'}</button>
                <button onclick="togglePolymerTransparency()" id="polymer-transparency">Polymer: {'ON' if polymer_transparent else 'OFF'}</button>
            </div>
            
            <div class="control-group">
                <span class="control-label">🎨 View:</span>
                <button onclick="centerOnInsulin()" id="center-insulin">Center on Insulin</button>
                <button onclick="toggleInsulinTracking()" id="insulin-tracking" class="active">Auto-Track: ON</button>
                <button onclick="centerView()">Center All</button>
                <button onclick="resetView()">Reset</button>
                <button onclick="toggleFullscreen()">Fullscreen</button>
            </div>
        </div>
        
        <div id="{viewer_id}"></div>
        
        <script>
            let viewer;
            let currentInsulinStyle = '{insulin_style}';
            let currentPolymerStyle = '{polymer_style}';
            let insulinTransparent = {str(insulin_transparent).lower()};
            let polymerTransparent = {str(polymer_transparent).lower()};
            let insulinTrackingEnabled = true; // Auto-center on insulin by default
            
            // Initialize 3Dmol viewer
            function initViewer() {{
                viewer = $3Dmol.createViewer(document.getElementById('{viewer_id}'), {{
                    backgroundColor: 'white',
                    antialias: true
                }});
                
                // Load PDB content
                viewer.addModel({pdb_escaped}, "pdb");
                
                // Apply initial styles
                applyStyles();
                
                // Wait for model to be fully loaded before initial centering
                setTimeout(() => {{
                    if (insulinTrackingEnabled) {{
                        centerOnInsulin();
                    }} else {{
                        viewer.zoomTo();
                        viewer.render();
                    }}
                }}, 300);
            }}
            
            function applyStyles() {{
                viewer.removeAllSurfaces();
                viewer.setStyle({{}}, {{}}); // Clear all styles
                
                // Style insulin (protein residues)
                const insulinSelector = {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                "THR", "TRP", "TYR", "VAL"]}};
                
                let insulinStyleObj = {{}};
                if (currentInsulinStyle === 'cartoon') {{
                    insulinStyleObj = {{
                        cartoon: {{
                            color: 'spectrum',
                            opacity: insulinTransparent ? 0.3 : 1.0
                        }}
                    }};
                }} else if (currentInsulinStyle === 'stick') {{
                    insulinStyleObj = {{
                        stick: {{
                            radius: 0.3,
                            colorscheme: 'greenCarbon',
                            opacity: insulinTransparent ? 0.3 : 1.0
                        }}
                    }};
                }} else if (currentInsulinStyle === 'sphere') {{
                    insulinStyleObj = {{
                        sphere: {{
                            radius: 1.0,
                            colorscheme: 'greenCarbon',
                            opacity: insulinTransparent ? 0.3 : 1.0
                        }}
                    }};
                }}
                
                viewer.setStyle(insulinSelector, insulinStyleObj);
                
                // Style polymer (non-protein residues)
                const polymerSelector = {{not: {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                       "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                       "THR", "TRP", "TYR", "VAL", "HOH", "WAT"]}}}};
                
                let polymerStyleObj = {{}};
                if (currentPolymerStyle === 'stick') {{
                    polymerStyleObj = {{
                        stick: {{
                            radius: 0.4,
                            colorscheme: 'blueCarbon',
                            opacity: polymerTransparent ? 0.3 : 1.0
                        }}
                    }};
                }} else if (currentPolymerStyle === 'sphere') {{
                    polymerStyleObj = {{
                        sphere: {{
                            radius: 1.5,
                            colorscheme: 'blueCarbon',
                            opacity: polymerTransparent ? 0.3 : 1.0
                        }}
                    }};
                }} else if (currentPolymerStyle === 'line') {{
                    polymerStyleObj = {{
                        line: {{
                            linewidth: 3,
                            colorscheme: 'blueCarbon',
                            opacity: polymerTransparent ? 0.3 : 1.0
                        }}
                    }};
                }}
                
                viewer.setStyle(polymerSelector, polymerStyleObj);
                
                // Hide water molecules
                viewer.setStyle({{resn: ["HOH", "WAT"]}}, {{}});
                
                // Auto-center on insulin if tracking is enabled
                if (insulinTrackingEnabled) {{
                    // Use setTimeout to ensure styles are applied before centering
                    setTimeout(() => {{
                        centerOnInsulinSilent();
                    }}, 250);
                }} else {{
                    viewer.render();
                }}
            }}
            
            function setInsulinStyle(style) {{
                currentInsulinStyle = style;
                
                // Update button states
                document.querySelectorAll('[id^="insulin-"]').forEach(btn => btn.classList.remove('active'));
                document.getElementById('insulin-' + style).classList.add('active');
                
                applyStyles();
            }}
            
            function setPolymerStyle(style) {{
                currentPolymerStyle = style;
                
                // Update button states
                document.querySelectorAll('[id^="polymer-"]').forEach(btn => btn.classList.remove('active'));
                document.getElementById('polymer-' + style).classList.add('active');
                
                applyStyles();
            }}
            
            function toggleInsulinTransparency() {{
                insulinTransparent = !insulinTransparent;
                document.getElementById('insulin-transparency').textContent = 
                    'Insulin: ' + (insulinTransparent ? 'ON' : 'OFF');
                applyStyles();
            }}
            
            function togglePolymerTransparency() {{
                polymerTransparent = !polymerTransparent;
                document.getElementById('polymer-transparency').textContent = 
                    'Polymer: ' + (polymerTransparent ? 'ON' : 'OFF');
                applyStyles();
            }}
            
            function centerView() {{
                viewer.zoomTo();
                viewer.render();
            }}
            
            function resetView() {{
                viewer.clear();
                viewer.addModel({pdb_escaped}, "pdb");
                applyStyles();
                
                // Respect insulin tracking setting when resetting
                if (insulinTrackingEnabled) {{
                    centerOnInsulin();
                }} else {{
                    viewer.zoomTo();
                }}
            }}
            
            function toggleFullscreen() {{
                const element = document.getElementById('{viewer_id}');
                if (document.fullscreenElement) {{
                    document.exitFullscreen();
                }} else {{
                    element.requestFullscreen();
                }}
            }}
            
            function calculateInsulinCenter() {{
                // Get all insulin atoms
                const insulinSelector = {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                "THR", "TRP", "TYR", "VAL"]}};
                
                const model = viewer.getModel();
                if (!model) {{
                    console.log('No model available for insulin centering');
                    return null;
                }}
                
                const atoms = model.selectedAtoms(insulinSelector);
                if (atoms.length === 0) {{
                    console.log('No insulin atoms found for centering');
                    return null;
                }}
                
                console.log(`Found ${{atoms.length}} insulin atoms for centering`);
                
                // Calculate center of mass (geometric center for simplicity)
                let x = 0, y = 0, z = 0;
                for (let atom of atoms) {{
                    x += atom.x;
                    y += atom.y;
                    z += atom.z;
                }}
                
                const center = {{
                    x: x / atoms.length,
                    y: y / atoms.length,
                    z: z / atoms.length
                }};
                
                console.log('Calculated insulin center:', center);
                return center;
            }}
            
            function centerOnInsulin() {{
                console.log('Attempting to center on insulin...');
                const center = calculateInsulinCenter();
                if (center) {{
                    console.log('Setting viewer center to insulin position:', center);
                    // Set the center point for the viewer
                    viewer.setCenter(center);
                    // Zoom to insulin with a reasonable distance
                    const insulinSelector = {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                   "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                   "THR", "TRP", "TYR", "VAL"]}};
                    viewer.zoomTo(insulinSelector, 800);
                    viewer.render();
                    console.log('Successfully centered on insulin');
                }} else {{
                    console.warn('Could not find insulin atoms to center on - using fallback');
                    viewer.zoomTo(); // Fallback to normal centering
                    viewer.render();
                }}
            }}
            
            function centerOnInsulinSilent() {{
                // Same as centerOnInsulin but with minimal logging (for auto-tracking)
                const center = calculateInsulinCenter();
                if (center) {{
                    viewer.setCenter(center);
                    const insulinSelector = {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                   "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                   "THR", "TRP", "TYR", "VAL"]}};
                    viewer.zoomTo(insulinSelector, 800);
                    viewer.render();
                }} else {{
                    // Fallback: just render without changing view
                    viewer.render();
                }}
            }}
            
            function toggleInsulinTracking() {{
                insulinTrackingEnabled = !insulinTrackingEnabled;
                const button = document.getElementById('insulin-tracking');
                button.textContent = 'Auto-Track: ' + (insulinTrackingEnabled ? 'ON' : 'OFF');
                button.classList.toggle('active', insulinTrackingEnabled);
                
                console.log('Insulin tracking toggled:', insulinTrackingEnabled);
                
                if (insulinTrackingEnabled) {{
                    // Delay slightly to ensure the toggle is complete
                    setTimeout(() => {{
                        centerOnInsulin();
                    }}, 100);
                }}
            }}
            
            // Debug function to check insulin centering status
            function debugInsulinCentering() {{
                console.log('=== Insulin Centering Debug ===');
                console.log('Tracking enabled:', insulinTrackingEnabled);
                console.log('Viewer available:', !!viewer);
                
                if (viewer) {{
                    const model = viewer.getModel();
                    console.log('Model available:', !!model);
                    
                    if (model) {{
                        const insulinSelector = {{resn: ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
                                                        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                                                        "THR", "TRP", "TYR", "VAL"]}};
                        const atoms = model.selectedAtoms(insulinSelector);
                        console.log('Insulin atoms found:', atoms.length);
                        
                        if (atoms.length > 0) {{
                            console.log('First insulin atom:', atoms[0]);
                            const center = calculateInsulinCenter();
                            console.log('Calculated center:', center);
                        }}
                    }}
                }}
                console.log('=== End Debug ===');
            }}
            
            // Make debug function available globally for console access
            window.debugInsulinCentering = debugInsulinCentering;
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', initViewer);
        </script>
    </body>
    </html>
    """
    
    return html_template


def render_pdb_visualizer(trajectory_file_path: str, simulation_id: str):
    """
    Render the PDB trajectory visualizer interface
    
    Args:
        trajectory_file_path: Path to the trajectory PDB file
        simulation_id: ID of the simulation for unique viewer identification
    """
    
    st.markdown("### 🧬 3D Molecular Visualization")
    
    try:
        trajectory_path = Path(trajectory_file_path)
        
        if not trajectory_path.exists():
            st.error(f"❌ Trajectory file not found: {trajectory_file_path}")
            return
        
        # Read and parse trajectory file
        frames_data = parse_trajectory_frames(trajectory_path)
        
        if not frames_data:
            st.error("❌ Could not parse trajectory frames")
            st.markdown("**Possible causes:**")
            st.markdown("- Empty or corrupted trajectory file")
            st.markdown("- Unsupported PDB format")
            st.markdown("- File does not contain valid ATOM/HETATM records")
            return
        
        if len(frames_data) == 0:
            st.error("❌ No valid frames found in trajectory")
            return
        
        st.success(f"✅ Loaded trajectory with {len(frames_data)} frames")
        
        # Frame selection controls
        current_frame = 0  # Default to first frame
        
        if len(frames_data) > 1:
            # Multiple frames - show slider and navigation controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                current_frame = st.slider(
                    "🎬 Select Frame",
                    min_value=0,
                    max_value=len(frames_data) - 1,
                    value=0,
                    help=f"Navigate through {len(frames_data)} trajectory frames"
                )
            
            with col2:
                if st.button("⏪ First Frame"):
                    current_frame = 0
                    st.rerun()
            
            with col3:
                if st.button("⏩ Last Frame"):
                    current_frame = len(frames_data) - 1
                    st.rerun()
        else:
            # Single frame - show info message instead of controls
            st.info(f"📝 Single frame trajectory - showing frame 1 of 1")
            current_frame = 0
        
        # Visualization options
        st.markdown("#### 🎨 Visualization Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            insulin_style = st.selectbox(
                "🧪 Insulin Style",
                ["cartoon", "stick", "sphere"],
                index=0,
                help="Visualization style for insulin protein"
            )
        
        with col2:
            polymer_style = st.selectbox(
                "🧬 Polymer Style", 
                ["stick", "sphere", "line"],
                index=0,
                help="Visualization style for polymer molecules"
            )
        
        with col3:
            insulin_transparent = st.checkbox(
                "👻 Insulin Transparent",
                value=False,
                help="Make insulin partially transparent"
            )
        
        with col4:
            polymer_transparent = st.checkbox(
                "👻 Polymer Transparent",
                value=False,
                help="Make polymer partially transparent"
            )
        
        # Display frame information
        # Ensure current_frame is within valid range
        current_frame = max(0, min(current_frame, len(frames_data) - 1))
        frame_info = frames_data[current_frame]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Frame", f"{current_frame + 1}/{len(frames_data)}")
        
        with col2:
            if 'time' in frame_info:
                st.metric("⏱️ Time", f"{frame_info['time']:.1f} ps")
        
        with col3:
            if 'atoms' in frame_info:
                st.metric("🔬 Atoms", f"{frame_info['atoms']:,}")
        
        # Generate and display 3D viewer
        viewer_id = f"pdb_viewer_{simulation_id}_{current_frame}"
        
        viewer_html = create_3dmol_viewer(
            pdb_content=frame_info['pdb_content'],
            viewer_id=viewer_id,
            height=600,
            insulin_transparent=insulin_transparent,
            polymer_transparent=polymer_transparent,
            insulin_style=insulin_style,
            polymer_style=polymer_style
        )
        
        # Display the viewer
        components.html(viewer_html, height=750, scrolling=False)
        
        # Additional information
        with st.expander("📊 Frame Details", expanded=False):
            st.markdown(f"**Frame:** {current_frame + 1}")
            st.markdown(f"**Trajectory File:** `{trajectory_path.name}`")
            
            if 'time' in frame_info:
                st.markdown(f"**Simulation Time:** {frame_info['time']:.1f} ps")
            
            if 'atoms' in frame_info:
                st.markdown(f"**Total Atoms:** {frame_info['atoms']:,}")
            
            st.markdown(f"**File Size:** {trajectory_path.stat().st_size / (1024*1024):.1f} MB")
    
    except Exception as e:
        st.error(f"❌ Error loading trajectory visualization: {str(e)}")
        st.exception(e)


def parse_trajectory_frames(trajectory_path: Path) -> List[Dict[str, Any]]:
    """
    Parse trajectory PDB file into individual frames
    
    Args:
        trajectory_path: Path to trajectory PDB file
        
    Returns:
        List of frame dictionaries with PDB content and metadata
    """
    
    frames = []
    current_frame_lines = []
    frame_number = 0
    
    try:
        with open(trajectory_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Start of new frame
            if line.startswith('MODEL'):
                if current_frame_lines:
                    # Process previous frame
                    frame_data = process_frame_lines(current_frame_lines, frame_number)
                    if frame_data:
                        frames.append(frame_data)
                    frame_number += 1
                
                current_frame_lines = [line]
            
            # End of frame
            elif line.startswith('ENDMDL'):
                current_frame_lines.append(line)
                frame_data = process_frame_lines(current_frame_lines, frame_number)
                if frame_data:
                    frames.append(frame_data)
                frame_number += 1
                current_frame_lines = []
            
            # Frame content
            else:
                current_frame_lines.append(line)
        
        # Handle case where trajectory doesn't use MODEL/ENDMDL
        if current_frame_lines and not frames:
            frame_data = process_frame_lines(current_frame_lines, 0)
            if frame_data:
                frames.append(frame_data)
    
    except Exception as e:
        st.error(f"Error parsing trajectory: {e}")
        return []
    
    return frames


def process_frame_lines(frame_lines: List[str], frame_number: int) -> Optional[Dict[str, Any]]:
    """
    Process lines for a single frame and extract metadata
    
    Args:
        frame_lines: List of lines for the frame
        frame_number: Frame number
        
    Returns:
        Frame data dictionary
    """
    
    try:
        # Count atoms
        atom_count = sum(1 for line in frame_lines if line.startswith(('ATOM', 'HETATM')))
        
        # Skip frames with no atoms (empty frames)
        if atom_count == 0:
            print(f"Warning: Frame {frame_number} has no atoms, skipping")
            return None
        
        # Extract time from REMARK if available
        time_ps = None
        for line in frame_lines:
            if line.startswith('REMARK') and 'time' in line.lower():
                # Try to extract time value
                time_match = re.search(r'(\d+\.?\d*)\s*ps', line, re.IGNORECASE)
                if time_match:
                    time_ps = float(time_match.group(1))
                    break
        
        # If no time found, estimate based on frame number (assume 1ps intervals)
        if time_ps is None:
            time_ps = frame_number * 1.0
        
        return {
            'frame_number': frame_number,
            'pdb_content': ''.join(frame_lines),
            'atoms': atom_count,
            'time': time_ps
        }
    
    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        return None


def render_trajectory_info(trajectory_path: Path) -> Dict[str, Any]:
    """
    Get basic information about a trajectory file
    
    Args:
        trajectory_path: Path to trajectory file
        
    Returns:
        Dictionary with trajectory information
    """
    
    try:
        frames_data = parse_trajectory_frames(trajectory_path)
        
        if not frames_data:
            return {'success': False, 'error': 'Could not parse trajectory'}
        
        total_atoms = frames_data[0].get('atoms', 0) if frames_data else 0
        total_frames = len(frames_data)
        simulation_time = frames_data[-1].get('time', 0) if frames_data else 0
        file_size_mb = trajectory_path.stat().st_size / (1024 * 1024)
        
        return {
            'success': True,
            'total_frames': total_frames,
            'total_atoms': total_atoms,
            'simulation_time_ps': simulation_time,
            'file_size_mb': file_size_mb,
            'file_path': str(trajectory_path)
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


# Testing function
def test_pdb_visualizer():
    """Test the PDB visualizer component"""
    
    # Create a simple test PDB content
    test_pdb = """HEADER    TEST STRUCTURE
MODEL        1
ATOM      1  N   ALA A   1      20.154  16.967  25.015  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  24.618  1.00 10.00           C  
ATOM      3  C   ALA A   1      17.687  16.849  24.447  1.00 10.00           C  
ATOM      4  O   ALA A   1      17.705  18.064  24.263  1.00 10.00           O  
ENDMDL
"""
    
    try:
        html = create_3dmol_viewer(test_pdb, "test_viewer")
        assert "3dmol.org" in html
        assert "test_viewer" in html
        print("✅ PDB visualizer test passed")
        return True
    except Exception as e:
        print(f"❌ PDB visualizer test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_pdb_visualizer()
    print(f"Test result: {'PASSED' if success else 'FAILED'}") 