#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Post-Processing Integration for Insulin-AI App
Combines all trajectory analysis capabilities into a single, user-friendly interface

This module integrates:
1. 🧪 Insulin Stability & Conformation Analysis  
2. 🔄 Partitioning & Transfer Free Energy Analysis
3. 🚶 Diffusion Coefficient Analysis
4. 🕸️ Hydrogel Mesh Size & Dynamics Analysis
5. ⚡ Interaction Energy Decomposition
6. 💧 Swelling & Poroelastic Response Analysis
7. 📊 Basic Trajectory Statistics
8. 📚 AI-Powered Literature Analysis

Provides real-time progress tracking and user-friendly results presentation.
"""

import os
import json
import time
import uuid
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

# Import all analysis modules
try:
    from .insulin_comprehensive_analyzer import InsulinComprehensiveAnalyzer
    COMPREHENSIVE_ANALYZER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYZER_AVAILABLE = False
    logging.warning("Comprehensive analyzer not available")

try:
    from .openmm_md_simulator import OpenMMInsulinSimulator
    BASIC_ANALYZER_AVAILABLE = True
except ImportError:
    BASIC_ANALYZER_AVAILABLE = False
    logging.warning("Basic analyzer not available")

try:
    from ..rag_literature_mining import RAGLiteratureMiningSystem
    RAG_LITERATURE_AVAILABLE = True
except ImportError:
    RAG_LITERATURE_AVAILABLE = False
    logging.warning("RAG Literature Mining system not available")

class ComprehensivePostProcessor:
    """
    Comprehensive post-processing system for insulin-AI MD simulations
    
    Integrates all analysis capabilities with progress tracking and user-friendly reporting
    """
    
    def __init__(self, output_dir: str = "postprocessing_results", 
                 openai_model: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize the comprehensive post-processor
        
        Args:
            output_dir: Directory for storing analysis results
            openai_model: OpenAI model to use for analysis
            temperature: Temperature setting for the model
        """
        
        # Check dependencies
        self.dependencies_available = self._check_dependencies()
        
        if not self.dependencies_available['core_available']:
            missing = [k for k, v in self.dependencies_available.items() 
                      if not v and k not in ['core_available', 'all_available']]
            raise ImportError(f"Missing critical dependencies: {missing}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store model configuration
        self.openai_model = openai_model
        self.temperature = temperature
        
        # Initialize analyzers
        if COMPREHENSIVE_ANALYZER_AVAILABLE:
            comp_output = self.output_dir / "comprehensive_analysis"
            self.comprehensive_analyzer = InsulinComprehensiveAnalyzer(str(comp_output))
        else:
            self.comprehensive_analyzer = None
        
        if BASIC_ANALYZER_AVAILABLE:
            basic_output = self.output_dir / "basic_analysis"
            basic_output.mkdir(exist_ok=True)
            self.basic_analyzer = OpenMMInsulinSimulator(str(basic_output))
        else:
            self.basic_analyzer = None
        
        if RAG_LITERATURE_AVAILABLE:
            rag_output = self.output_dir / "literature_analysis"
            self.rag_system = RAGLiteratureMiningSystem(str(rag_output), openai_model=self.openai_model, temperature=self.temperature)
        else:
            self.rag_system = None
        
        # Progress tracking
        self.processing_thread = None
        self.processing_running = False
        self.current_analysis = None
        
        print(f"🔬 Comprehensive Post-Processor initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🧪 Comprehensive Analysis: {'✅ Available' if self.comprehensive_analyzer else '❌ Not Available'}")
        print(f"📊 Basic Analysis: {'✅ Available' if self.basic_analyzer else '❌ Not Available'}")
        print(f"📚 RAG Literature Mining: {'✅ Available' if self.rag_system else '❌ Not Available'}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check availability of analysis dependencies"""
        deps = {
            'comprehensive_analyzer': COMPREHENSIVE_ANALYZER_AVAILABLE,
            'basic_analyzer': BASIC_ANALYZER_AVAILABLE,
            'rag_literature_mining': RAG_LITERATURE_AVAILABLE
        }
        
        # At least one analyzer must be available
        deps['core_available'] = any(deps.values())
        deps['all_available'] = all(deps.values())
        
        return deps
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get detailed dependency status for UI display"""
        return {
            'dependencies': self.dependencies_available,
            'analysis_capabilities': {
                'insulin_stability': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'partitioning': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'diffusion': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'hydrogel_dynamics': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'interaction_energies': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'swelling_response': COMPREHENSIVE_ANALYZER_AVAILABLE,
                'basic_trajectory_stats': BASIC_ANALYZER_AVAILABLE,
                'literature_analysis': RAG_LITERATURE_AVAILABLE,
                'material_recommendations': RAG_LITERATURE_AVAILABLE
            },
            'recommendation': self._get_capability_recommendation()
        }
    
    def _get_capability_recommendation(self) -> str:
        """Get recommendation based on available capabilities"""
        if self.dependencies_available['all_available']:
            return "🎉 All analysis capabilities available! Full comprehensive post-processing enabled including AI-powered literature insights."
        
        available_systems = []
        if self.dependencies_available['comprehensive_analyzer']:
            available_systems.append("🔬 Comprehensive analysis")
        if self.dependencies_available['rag_literature_mining']:
            available_systems.append("📚 AI literature mining")
        if self.dependencies_available['basic_analyzer']:
            available_systems.append("📊 Basic trajectory analysis")
        
        if available_systems:
            systems_text = ", ".join(available_systems)
            return f"✅ Available: {systems_text}. Consider installing all modules for maximum capability."
        else:
            return "❌ No analysis capabilities available. Please check dependencies."
    
    def get_available_simulations(self, simulation_output_dir: str = "integrated_md_simulations") -> List[Dict[str, Any]]:
        """Get list of available simulations for post-processing"""
        simulations = []
        
        # Check multiple simulation directories
        simulation_dirs = [
            simulation_output_dir,  # Default integrated simulations
            "simple_md_simulations",  # Simple MD simulations
            "dual_gaff_amber_simulations", # Dual GAFF+AMBER simulations
        ]
        
        for dir_path in simulation_dirs:
            simulations.extend(self._scan_simulation_directory(dir_path))
        
        return simulations
    
    def _scan_simulation_directory(self, simulation_output_dir: str) -> List[Dict[str, Any]]:
        """Scan a specific directory for completed simulations"""
        simulations = []
        
        try:
            base_dir = Path(simulation_output_dir)
            if not base_dir.exists():
                return []
            
            print(f"🔍 Scanning simulation directory: {base_dir}")
            
            # Look for simulation directories
            for sim_dir in base_dir.iterdir():
                if sim_dir.is_dir():
                    # Skip special directories
                    if sim_dir.name in ['__pycache__', '.git']:
                        continue
                    
                    print(f"   📁 Found simulation directory: {sim_dir.name}")
                    
                    # Check for different simulation structures
                    sim_info = self._analyze_simulation_directory(sim_dir)
                    
                    if sim_info:
                        # Finalize with post-processing status
                        sim_info = self._finalize_simulation_info(sim_info)
                        simulations.append(sim_info)
                        print(f"   ✅ Valid simulation: {sim_info['id']}")
                    else:
                        print(f"   ❌ Invalid simulation structure: {sim_dir.name}")
        
        except Exception as e:
            print(f"❌ Error scanning {simulation_output_dir}: {str(e)}")
        
        # Sort by timestamp (newest first) using file_timestamp if no timestamp available
        simulations.sort(key=lambda x: x.get('timestamp') or x.get('file_timestamp', 0), reverse=True)
        
        return simulations
    
    def _analyze_simulation_directory(self, sim_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a simulation directory to extract simulation info"""
        
        # Method 1: Check for dual GAFF+AMBER simulation structure
        if "dual_gaff_amber" in str(sim_dir.parent) or sim_dir.name.startswith("enhanced_dual"):
            trajectory_file = sim_dir / "trajectory.pdb"
            log_file = sim_dir / "simulation.log"

            if trajectory_file.exists():
                sim_info = {
                    'id': sim_dir.name,
                    'path': str(sim_dir),
                    'has_trajectory': True,
                    'has_report': log_file.exists(),
                    'trajectory_file': str(trajectory_file),
                    'ready_for_processing': True,
                    'structure_type': 'dual_gaff_amber'
                }
                
                # Get basic trajectory info
                traj_info = self._estimate_trajectory_info(trajectory_file)
                sim_info.update(traj_info)

                if log_file.exists():
                    log_info = self._parse_simple_log_file(log_file)
                    sim_info.update(log_info)

                sim_info['success'] = True
                return sim_info

        # Method 2: Check for integrated MD simulation structure (sim_*/production/frames.pdb)
        if sim_dir.name.startswith('sim_'):
            production_dir = sim_dir / "production"
            frames_file = production_dir / "frames.pdb"
            report_file = sim_dir / "simulation_report.json"
            
            if frames_file.exists():
                sim_info = {
                    'id': sim_dir.name,
                    'path': str(sim_dir),
                    'has_trajectory': True,
                    'has_report': report_file.exists(),
                    'trajectory_file': str(frames_file),
                    'ready_for_processing': True,
                    'structure_type': 'integrated'
                }
                
                # Get basic info from report if available
                if report_file.exists():
                    try:
                        with open(report_file, 'r') as f:
                            report_data = json.load(f)
                        
                        sim_info.update({
                            'timestamp': report_data.get('timestamp', ''),
                            'total_atoms': report_data.get('system_info', {}).get('final_atoms', 0),
                            'performance': report_data.get('performance', {}).get('ns_per_day', 0),
                            'simulation_time_ps': report_data.get('production_stats', {}).get('simulation_time', 0),
                            'success': report_data.get('success', False)
                        })
                        
                    except Exception as e:
                        sim_info['report_error'] = str(e)
                        sim_info['success'] = True  # Assume success if trajectory exists
                
                # Estimate info from trajectory if no report
                if not sim_info.get('success') and not sim_info.get('total_atoms'):
                    sim_info.update(self._estimate_trajectory_info(frames_file))
                
                return sim_info
        
        # Method 2: Check for simple MD simulation structure (*_trajectory.pdb)
        trajectory_files = list(sim_dir.glob("*_trajectory.pdb"))
        log_files = list(sim_dir.glob("*_log.txt"))
        
        # Add check for trajectory.pdb
        if not trajectory_files:
            if (sim_dir / "trajectory.pdb").exists():
                trajectory_files = [sim_dir / "trajectory.pdb"]

        if trajectory_files:
            trajectory_file = trajectory_files[0]  # Use first trajectory file found
            log_file = log_files[0] if log_files else None
            
            sim_info = {
                'id': sim_dir.name,
                'path': str(sim_dir),
                'has_trajectory': True,
                'has_report': log_file is not None,
                'trajectory_file': str(trajectory_file),
                'ready_for_processing': True,
                'structure_type': 'simple'
            }
            
            # Get info from log file if available
            if log_file:
                try:
                    log_info = self._parse_simple_log_file(log_file)
                    sim_info.update(log_info)
                except Exception as e:
                    sim_info['log_error'] = str(e)
            
            # Get basic trajectory info
            traj_info = self._estimate_trajectory_info(trajectory_file)
            sim_info.update(traj_info)
            
            # Set default success if trajectory exists
            if 'success' not in sim_info:
                sim_info['success'] = True
            
            return sim_info
        
        # Method 3: Check for other trajectory formats
        other_trajectory_files = (
            list(sim_dir.glob("*.dcd")) + 
            list(sim_dir.glob("trajectory.pdb")) +
            list(sim_dir.glob("output.pdb"))
        )
        
        if other_trajectory_files:
            trajectory_file = other_trajectory_files[0]
            
            sim_info = {
                'id': sim_dir.name,
                'path': str(sim_dir),
                'has_trajectory': True,
                'has_report': False,
                'trajectory_file': str(trajectory_file),
                'ready_for_processing': True,
                'structure_type': 'generic'
            }
            
            # Get basic trajectory info
            traj_info = self._estimate_trajectory_info(trajectory_file)
            sim_info.update(traj_info)
            
            sim_info['success'] = True
            return sim_info
        
        return None
    
    def _parse_simple_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Parse simple MD simulation log file"""
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header line
            data_lines = [line for line in lines[1:] if line.strip()]
            
            if not data_lines:
                return {'success': False}
            
            # Parse last line for final values
            last_line = data_lines[-1].strip()
            parts = last_line.split('\t')
            
            if len(parts) >= 6:
                step = int(float(parts[0]))
                time_ps = float(parts[1])
                temperature = float(parts[5])
                
                return {
                    'simulation_time_ps': time_ps,
                    'final_step': step,
                    'final_temperature': temperature,
                    'success': True,
                    'timestamp': log_file.stat().st_mtime
                }
        
        except Exception as e:
            print(f"Warning: Could not parse log file {log_file}: {e}")
        
        return {'success': False}
    
    def _estimate_trajectory_info(self, trajectory_file: Path) -> Dict[str, Any]:
        """Estimate basic info from trajectory file"""
        
        try:
            # File size and modification time
            file_stat = trajectory_file.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            
            # Rough estimates based on file size
            estimated_atoms = int(file_size_mb * 100)  # Very rough estimate
            estimated_frames = max(1, int(file_size_mb / 10))  # Rough estimate
            
            return {
                'total_atoms': estimated_atoms,
                'estimated_frames': estimated_frames,
                'trajectory_size_mb': file_size_mb,
                'file_timestamp': file_stat.st_mtime,
                'performance': 1.0  # Default placeholder
            }
        
        except Exception:
            return {
                'total_atoms': 0,
                'estimated_frames': 0,
                'trajectory_size_mb': 0,
                'performance': 1.0
            }
    
    def _finalize_simulation_info(self, sim_info: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize simulation info with post-processing status"""
        
        # Check if post-processing directory exists
        postproc_dir = self.output_dir / sim_info['id']
        sim_info['already_processed'] = False
        sim_info['processing_complete'] = False
        
        if postproc_dir.exists():
            # Check for completed analysis report
            comp_report = postproc_dir / "comprehensive_analysis_report.json"
            
            if comp_report.exists():
                try:
                    with open(comp_report, 'r') as f:
                        proc_data = json.load(f)
                    
                    # Check if analysis actually succeeded (FIXED: Add type checking)
                    results = proc_data.get('results', {})
                    
                    # Ensure results is a dictionary
                    if not isinstance(results, dict):
                        print(f"⚠️  Analysis results is not a dictionary for {sim_info['id']}: {type(results)}")
                        results = {}
                    
                    has_successful_analyses = False
                    analysis_count = 0
                    successful_analyses = []
                    
                    for analysis_name, analysis_result in results.items():
                        # FIXED: Check if analysis_result is a dictionary before calling .get()
                        if isinstance(analysis_result, dict):
                            analysis_count += 1
                            if analysis_result.get('success', False):
                                has_successful_analyses = True
                                successful_analyses.append(analysis_name)
                        else:
                            # FIXED: More informative logging and better handling
                            print(f"⚠️  Found non-dict field '{analysis_name}' in results: {type(analysis_result)}")
                            print(f"     This suggests a data structure issue - field should be in metadata, not results")
                            # Don't count these as failed analyses, they're likely metadata
                            continue
                    
                    # Only mark as processed if there are successful analyses
                    if has_successful_analyses:
                        sim_info['already_processed'] = True
                        sim_info['processing_complete'] = True
                        sim_info['analyses_completed'] = successful_analyses
                        sim_info['processing_timestamp'] = proc_data.get('timestamp', 'Unknown')
                        print(f"✅ Found completed post-processing for {sim_info['id']} with {len(successful_analyses)} successful analyses")
                    elif analysis_count > 0:
                        # Had analysis attempts but none succeeded
                        sim_info['already_processed'] = True
                        sim_info['processing_complete'] = False
                        print(f"⚠️  Found failed post-processing for {sim_info['id']} - marking as available for retry")
                    else:
                        # No valid analysis results found
                        print(f"⚠️  No valid analysis results found for {sim_info['id']} - marking as available for retry")
                        
                except Exception as e:
                    print(f"⚠️  Error reading analysis report for {sim_info['id']}: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    # Add more detailed error information for debugging
                    import traceback
                    print(f"   Full traceback: {traceback.format_exc()}")
                    sim_info['already_processed'] = False
                    sim_info['processing_complete'] = False
        
        return sim_info
    
    def start_comprehensive_analysis_async(self, simulation_id: str, 
                                          simulation_dir: str,
                                          analysis_options: Optional[Dict[str, bool]] = None,
                                          output_callback: Optional[Callable] = None,
                                          trajectory_file: Optional[str] = None,
                                          simulation_structure_type: str = 'integrated') -> str:
        """
        Start comprehensive post-processing analysis asynchronously
        
        Args:
            simulation_id: ID of the simulation to analyze
            simulation_dir: Directory containing simulation results
            analysis_options: Dict specifying which analyses to run
            output_callback: Callback for progress updates
            
        Returns:
            Analysis job ID
        """
        
        # Generate analysis job ID
        analysis_job_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        # Default analysis options (run all available analyses)
        if analysis_options is None:
            analysis_options = {
                'insulin_stability': self.comprehensive_analyzer is not None,
                'partitioning': self.comprehensive_analyzer is not None,
                'diffusion': self.comprehensive_analyzer is not None,
                'hydrogel_dynamics': self.comprehensive_analyzer is not None,
                'interaction_energies': self.comprehensive_analyzer is not None,
                'swelling_response': self.comprehensive_analyzer is not None,
                'basic_trajectory_stats': True,  # Basic stats always available
                'literature_analysis': self.rag_system is not None,
                'material_recommendations': self.rag_system is not None
            }
        
        # Store analysis parameters
        self.current_analysis = {
            'job_id': analysis_job_id,
            'simulation_id': simulation_id,
            'simulation_dir': simulation_dir,
            'analysis_options': analysis_options,
            'status': 'starting',
            'start_time': time.time(),
            'progress': 0,
            'current_step': 'Initializing...',
            'steps_completed': [],
            'total_steps': sum(1 for v in analysis_options.values() if v)
        }
        
        # Start analysis thread
        self.processing_thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(analysis_job_id, simulation_id, simulation_dir, analysis_options, output_callback, trajectory_file, simulation_structure_type)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.processing_running = True
        
        return analysis_job_id
    
    def _run_analysis_thread(self, analysis_job_id: str, simulation_id: str,
                           simulation_dir: str, analysis_options: Dict[str, bool],
                           output_callback: Optional[Callable],
                           trajectory_file: Optional[str] = None,
                           simulation_structure_type: str = 'integrated'):
        """Run the comprehensive analysis in a separate thread"""
        
        def log_output(message: str):
            print(message)  # Always print to console
            if output_callback:
                output_callback(message)  # Also send to app interface
        
        try:
            log_output(f"🔬 Starting Comprehensive Post-Processing Analysis")
            log_output(f"🎯 Analysis Job: {analysis_job_id}")
            log_output(f"📊 Simulation: {simulation_id}")
            log_output("=" * 80)
            
            # Update status
            self.current_analysis['status'] = 'running'
            self.current_analysis['current_step'] = 'Setting up analysis environment...'
            
            # Create analysis output directory
            analysis_output_dir = self.output_dir / simulation_id
            analysis_output_dir.mkdir(exist_ok=True)
            
            # Initialize results container
            comprehensive_results = {
                'analysis_job_id': analysis_job_id,
                'simulation_id': simulation_id,
                'timestamp': datetime.now().isoformat(),
                'analysis_options': analysis_options,
                'results': {},
                'summary_metrics': {},
                'processing_time': {}
            }
            
            step_count = 0
            total_steps = self.current_analysis['total_steps']
            
            # 1. Comprehensive Analysis (if available)
            if self.comprehensive_analyzer and any(analysis_options.get(k, False) for k in 
                ['insulin_stability', 'partitioning', 'diffusion', 'hydrogel_dynamics', 
                 'interaction_energies', 'swelling_response']):
                
                step_count += 1
                log_output(f"\n🧪 Step {step_count}/{total_steps}: Comprehensive Property Analysis")
                self.current_analysis['current_step'] = f"Comprehensive Analysis ({step_count}/{total_steps})"
                self.current_analysis['progress'] = (step_count - 1) / total_steps * 100
                
                start_time = time.time()
                try:
                    # Filter analysis options for comprehensive analyzer
                    comp_options = {
                        'insulin_stability': analysis_options.get('insulin_stability', False),
                        'partitioning': analysis_options.get('partitioning', False),
                        'diffusion': analysis_options.get('diffusion', False),
                        'hydrogel_dynamics': analysis_options.get('hydrogel_dynamics', False),
                        'interaction_energies': analysis_options.get('interaction_energies', False),
                        'swelling_response': analysis_options.get('swelling_response', False)
                    }
                    
                    # Use different approach based on simulation structure
                    if simulation_structure_type == 'simple' and trajectory_file:
                        log_output(f"🔬 Using simple simulation structure with trajectory: {trajectory_file}")
                        comp_results = self.comprehensive_analyzer.analyze_trajectory_file(
                            trajectory_file, simulation_id, comp_options, log_output
                        )
                    elif simulation_structure_type == 'dual_gaff_amber' and trajectory_file:
                        log_output(f"🔬 Using dual GAFF+AMBER simulation structure from: {simulation_dir}")
                        comp_results = self.comprehensive_analyzer.analyze_complete_system(
                            simulation_dir, simulation_id, comp_options, log_output
                        )
                    else:
                        log_output(f"🔬 Using integrated simulation structure from: {simulation_dir}")
                        comp_results = self.comprehensive_analyzer.analyze_complete_system(
                            simulation_dir, simulation_id, comp_options, log_output
                        )
                    
                    # Fix the problematic merge on line 577 by filtering metadata fields
                    # Merge comprehensive results - FIXED: Filter out metadata fields
                    if comp_results.get('success'):
                        # Filter out metadata fields that should not be in results section
                        metadata_fields = {'simulation_id', 'timestamp', 'analysis_options', 'trajectory_info', 'success', 'processing_time', 'analysis_completed'}
                        
                        # Only add actual analysis results (dictionaries with success field)
                        for key, value in comp_results.items():
                            if key not in metadata_fields:
                                # Ensure it's an analysis result (dictionary with success field)
                                if isinstance(value, dict) and ('success' in value or 'error' in value):
                                    comprehensive_results['results'][key] = value
                                else:
                                    # Log unexpected structure but don't break
                                    log_output(f"⚠️ Skipping non-analysis field '{key}': {type(value)}")
                        
                        # Extract key summary metrics
                        if 'insulin_stability' in comp_results:
                            stability = comp_results['insulin_stability']
                            if stability.get('success') and 'rmsd' in stability:
                                comprehensive_results['summary_metrics']['rmsd_mean_A'] = stability['rmsd']['mean']
                                comprehensive_results['summary_metrics']['stability_assessment'] = stability['rmsd']['stability_assessment']
                        
                        if 'diffusion' in comp_results:
                            diffusion = comp_results['diffusion']
                            if diffusion.get('success') and 'msd_analysis' in diffusion:
                                comprehensive_results['summary_metrics']['diffusion_coefficient_cm2_s'] = diffusion['msd_analysis']['diffusion_coefficient']
                        
                        if 'hydrogel_dynamics' in comp_results:
                            hydrogel = comp_results['hydrogel_dynamics']
                            if hydrogel.get('success') and 'mesh_size_analysis' in hydrogel:
                                comprehensive_results['summary_metrics']['mesh_size_A'] = hydrogel['mesh_size_analysis']['average_mesh_size']
                        
                        log_output(f"✅ Comprehensive analysis completed successfully")
                    else:
                        log_output(f"⚠️ Comprehensive analysis failed: {comp_results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    log_output(f"❌ Comprehensive analysis failed: {str(e)}")
                    comprehensive_results['results']['comprehensive_error'] = str(e)
                
                comprehensive_results['processing_time']['comprehensive_analysis'] = time.time() - start_time
                self.current_analysis['steps_completed'].extend(['comprehensive_analysis'])
            
            # 2. Basic Trajectory Analysis (if available and requested)
            if analysis_options.get('basic_trajectory_stats', False) and self.basic_analyzer:
                step_count += 1
                log_output(f"\n📊 Step {step_count}/{total_steps}: Basic Trajectory Statistics")
                self.current_analysis['current_step'] = f"Basic Analysis ({step_count}/{total_steps})"
                self.current_analysis['progress'] = (step_count - 1) / total_steps * 100
                
                start_time = time.time()
                try:
                    # Find trajectory files - handle both integrated and simple MD structures
                    sim_path = Path(simulation_dir) / simulation_id
                    
                    # Try different possible locations for trajectory files
                    possible_frames_files = [
                        sim_path / "production" / "frames.pdb",  # Integrated MD structure
                        sim_path / "frames.pdb",                # Simple MD structure (old naming)
                        sim_path / f"{simulation_id}_trajectory.pdb",  # Simple MD structure (actual naming)
                        Path(simulation_dir) / "frames.pdb"     # Direct path
                    ]
                    
                    frames_file = None
                    for possible_file in possible_frames_files:
                        if possible_file.exists():
                            frames_file = possible_file
                            log_output(f"   Found trajectory file: {frames_file}")
                            break
                    
                    if frames_file and frames_file.exists():
                        # Use the first frame as topology
                        basic_results = self.basic_analyzer.analyze_trajectory(
                            str(frames_file), str(frames_file)
                        )
                        comprehensive_results['results']['basic_trajectory'] = basic_results
                        
                        if basic_results.get('analysis_available'):
                            comprehensive_results['summary_metrics']['trajectory_frames'] = basic_results['num_frames']
                            comprehensive_results['summary_metrics']['total_atoms'] = basic_results['num_atoms']
                            comprehensive_results['summary_metrics']['simulation_time_ps'] = basic_results['time_ps']
                            
                            if 'rmsd_mean' in basic_results:
                                comprehensive_results['summary_metrics']['basic_rmsd_A'] = basic_results['rmsd_mean']
                            if 'rg_mean' in basic_results:
                                comprehensive_results['summary_metrics']['radius_gyration_A'] = basic_results['rg_mean']
                        
                        log_output(f"Basic trajectory analysis completed")
                    else:
                        log_output(f"WARNING: Trajectory file not found: {frames_file}")
                        comprehensive_results['results']['basic_trajectory'] = {'success': False, 'error': 'Trajectory file not found'}
                        
                except Exception as e:
                    log_output(f"❌ Basic analysis failed: {str(e)}")
                    comprehensive_results['results']['basic_trajectory'] = {'success': False, 'error': str(e)}
                
                comprehensive_results['processing_time']['basic_analysis'] = time.time() - start_time
                self.current_analysis['steps_completed'].append('basic_trajectory_stats')
            
            # 3. Literature Analysis (if available and requested)
            if (analysis_options.get('literature_analysis', False) or 
                analysis_options.get('material_recommendations', False)) and self.rag_system:
                
                step_count += 1
                log_output(f"\n📚 Step {step_count}/{total_steps}: AI-Powered Literature Analysis")
                self.current_analysis['current_step'] = f"Literature Analysis ({step_count}/{total_steps})"
                self.current_analysis['progress'] = (step_count - 1) / total_steps * 100
                
                start_time = time.time()
                try:
                    # Generate research questions based on simulation results
                    research_questions = []
                    
                    # Add questions based on available simulation results
                    if 'insulin_stability' in comprehensive_results['results']:
                        research_questions.append(
                            "What are the latest developments in materials that enhance insulin stability in hydrogel delivery systems?"
                        )
                    
                    if 'hydrogel_dynamics' in comprehensive_results['results']:
                        research_questions.append(
                            "What are the optimal hydrogel mesh properties for sustained insulin release in subcutaneous applications?"
                        )
                    
                    # Default research question
                    if not research_questions:
                        research_questions.append(
                            "What are the most promising biocompatible materials for insulin delivery systems with controlled release properties?"
                        )
                    
                    # Perform literature analysis with progress tracking
                    literature_results = {}
                    
                    for i, question in enumerate(research_questions):
                        log_output(f"   🔍 Question {i+1}/{len(research_questions)}: {question[:60]}...")
                        progress_offset = (i / len(research_questions)) * 15  # 15% for literature analysis
                        self.current_analysis['progress'] = ((step_count - 1) / total_steps * 100) + progress_offset
                        
                        try:
                            # Use async version with timeout to prevent hanging
                            import asyncio
                            
                            # Create a timeout wrapper
                            async def analyze_with_timeout():
                                try:
                                    # Create progress callback for this specific question
                                    def question_progress(message: str):
                                        log_output(f"      {message}")
                                    
                                    return await asyncio.wait_for(
                                        self.rag_system.analyze_literature_async(
                                            question, 
                                            progress_callback=question_progress
                                        ),
                                        timeout=300.0  # 5 minute timeout per question
                                    )
                                except asyncio.TimeoutError:
                                    log_output(f"   ⚠️ Question {i+1} timed out after 5 minutes")
                                    return {'error': 'Analysis timed out', 'success': False}
                            
                            # Run with proper event loop handling
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # If we're already in an event loop, create a task
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(
                                            lambda: asyncio.run(analyze_with_timeout())
                                        )
                                        result = future.result(timeout=310)  # Slightly longer than inner timeout
                                else:
                                    result = asyncio.run(analyze_with_timeout())
                            except RuntimeError:
                                # Fallback to thread-based execution
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(analyze_with_timeout())
                                    )
                                    result = future.result(timeout=310)
                            
                            literature_results[f'research_query_{i+1}'] = {
                                'question': question,
                                'analysis': result,
                                'success': result.get('success', False)
                            }
                            
                            if result.get('success', False):
                                log_output(f"   ✅ Question {i+1} completed successfully")
                            else:
                                log_output(f"   ⚠️ Question {i+1} completed with issues")
                        
                        except Exception as e:
                            log_output(f"   ❌ Question {i+1} failed: {str(e)}")
                            literature_results[f'research_query_{i+1}'] = {
                                'question': question,
                                'analysis': {'error': str(e)},
                                'success': False
                            }
                    
                    # Get material recommendations if requested
                    if analysis_options.get('material_recommendations', False):
                        log_output(f"   🧬 Generating material recommendations...")
                        self.current_analysis['progress'] = (step_count / total_steps * 100) - 3
                        
                        try:
                            # Use the async version for recommendations too
                            async def get_recommendations_with_timeout():
                                try:
                                    return await asyncio.wait_for(
                                        self.rag_system.get_material_recommendations_async(
                                            application="insulin delivery systems"
                                        ),
                                        timeout=120.0  # 2 minute timeout for recommendations
                                    )
                                except asyncio.TimeoutError:
                                    log_output(f"   ⚠️ Material recommendations timed out")
                                    return {'error': 'Recommendations timed out', 'success': False}
                            
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    import concurrent.futures
                                    with concurrent.futures.ThreadPoolExecutor() as executor:
                                        future = executor.submit(
                                            lambda: asyncio.run(get_recommendations_with_timeout())
                                        )
                                        recommendations = future.result(timeout=130)
                                else:
                                    recommendations = asyncio.run(get_recommendations_with_timeout())
                            except RuntimeError:
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(
                                        lambda: asyncio.run(get_recommendations_with_timeout())
                                    )
                                    recommendations = future.result(timeout=130)
                            
                            literature_results['material_recommendations'] = {
                                'recommendations': recommendations,
                                'success': True
                            }
                            log_output(f"   ✅ Material recommendations completed")
                            
                        except Exception as e:
                            log_output(f"   ❌ Material recommendations failed: {str(e)}")
                            literature_results['material_recommendations'] = {
                                'recommendations': [],
                                'success': False,
                                'error': str(e)
                            }

                    # Add literature results to comprehensive results
                    comprehensive_results['results']['literature_analysis'] = literature_results
                    
                    # Extract key literature insights for summary
                    if literature_results:
                        comprehensive_results['summary_metrics']['literature_queries_performed'] = len([
                            k for k in literature_results.keys() if k.startswith('research_query')
                        ])
                        
                        successful_queries = len([
                            r for r in literature_results.values() 
                            if isinstance(r, dict) and r.get('success', False)
                        ])
                        comprehensive_results['summary_metrics']['literature_insights_generated'] = successful_queries
                    
                    log_output(f"✅ Literature analysis completed successfully")
                    
                except Exception as e:
                    log_output(f"❌ Literature analysis failed: {str(e)}")
                    comprehensive_results['results']['literature_analysis'] = {'success': False, 'error': str(e)}
                
                comprehensive_results['processing_time']['literature_analysis'] = time.time() - start_time
                self.current_analysis['steps_completed'].extend(['literature_analysis'])
            
            # 4. Generate Comprehensive Report
            log_output(f"\n📊 Generating comprehensive post-processing report...")
            self.current_analysis['current_step'] = 'Generating final report...'
            self.current_analysis['progress'] = 95
            
            try:
                self._generate_postprocessing_report(comprehensive_results, analysis_output_dir, log_output)
                comprehensive_results['report_generated'] = True
            except Exception as e:
                log_output(f"⚠️ Report generation failed: {str(e)}")
                comprehensive_results['report_generated'] = False
                comprehensive_results['report_error'] = str(e)
            
            # Update final status
            self.current_analysis['status'] = 'completed'
            self.current_analysis['progress'] = 100
            self.current_analysis['current_step'] = 'Analysis completed!'
            self.current_analysis['end_time'] = time.time()
            self.current_analysis['results'] = comprehensive_results
            
            # Calculate total processing time
            total_time = time.time() - self.current_analysis['start_time']
            comprehensive_results['total_processing_time'] = total_time
            
            log_output(f"\n✅ Comprehensive Post-Processing Analysis Completed!")
            log_output(f"⏱️ Total processing time: {total_time:.1f} seconds")
            log_output(f"📁 Results saved to: {analysis_output_dir}")
            log_output(f"🎯 Analysis job: {analysis_job_id}")
            
            # Save final results
            results_file = analysis_output_dir / f"postprocessing_results_{analysis_job_id}.json"
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
        except Exception as e:
            log_output(f"❌ Post-processing analysis failed: {str(e)}")
            logging.error(traceback.format_exc())
            
            self.current_analysis['status'] = 'failed'
            self.current_analysis['error'] = str(e)
            self.current_analysis['end_time'] = time.time()
        
        finally:
            self.processing_running = False
    
    def _generate_postprocessing_report(self, comprehensive_results: Dict[str, Any],
                                      analysis_output_dir: Path, log_output: Callable):
        """Generate comprehensive post-processing report"""
        
        try:
            log_output("   📊 Creating summary report...")
            
            # Create summary report
            summary_report = {
                'analysis_job_id': comprehensive_results['analysis_job_id'],
                'simulation_id': comprehensive_results['simulation_id'],
                'timestamp': comprehensive_results['timestamp'],
                'summary_metrics': comprehensive_results['summary_metrics'],
                'processing_time': comprehensive_results['processing_time'],
                'total_processing_time': comprehensive_results.get('total_processing_time', 0),
                'analyses_performed': list(comprehensive_results['analysis_options'].keys()),
                'success_status': {}
            }
            
            # Check success status of each analysis
            results = comprehensive_results['results']
            for analysis_name, analysis_data in results.items():
                if isinstance(analysis_data, dict):
                    summary_report['success_status'][analysis_name] = analysis_data.get('success', False)
            
            # Save comprehensive report (same format as comprehensive analyzer)
            report_file = analysis_output_dir / "comprehensive_analysis_report.json"
            with open(report_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = analysis_output_dir / "postprocessing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            # Create human-readable summary
            summary_text_file = analysis_output_dir / "postprocessing_summary.txt"
            with open(summary_text_file, 'w') as f:
                f.write("COMPREHENSIVE POST-PROCESSING ANALYSIS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Analysis Job ID: {comprehensive_results['analysis_job_id']}\n")
                f.write(f"Simulation ID: {comprehensive_results['simulation_id']}\n")
                f.write(f"Analysis Date: {comprehensive_results['timestamp']}\n")
                f.write(f"Total Processing Time: {comprehensive_results.get('total_processing_time', 0):.1f} seconds\n\n")
                
                # Key results
                metrics = comprehensive_results['summary_metrics']
                
                f.write("KEY RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                if 'rmsd_mean_A' in metrics:
                    f.write(f"🧪 INSULIN STABILITY: {metrics.get('stability_assessment', 'Unknown')} ")
                    f.write(f"(RMSD: {metrics['rmsd_mean_A']:.2f} Å)\n")
                
                if 'diffusion_coefficient_cm2_s' in metrics:
                    f.write(f"🚶 DIFFUSION COEFFICIENT: {metrics['diffusion_coefficient_cm2_s']:.2e} cm²/s\n")
                
                if 'mesh_size_A' in metrics:
                    f.write(f"🕸️ HYDROGEL MESH SIZE: {metrics['mesh_size_A']:.1f} Å\n")
                
                if 'trajectory_frames' in metrics:
                    f.write(f"📊 TRAJECTORY: {metrics['trajectory_frames']} frames, ")
                    f.write(f"{metrics.get('total_atoms', 0)} atoms, ")
                    f.write(f"{metrics.get('simulation_time_ps', 0):.1f} ps\n")
                
                if 'literature_queries_performed' in metrics:
                    f.write(f"📚 LITERATURE ANALYSIS: {metrics['literature_queries_performed']} research queries performed, ")
                    f.write(f"{metrics.get('literature_insights_generated', 0)} insights generated\n")
                
                f.write(f"\n📁 Detailed results available in: {analysis_output_dir}\n")
            
            log_output(f"✅ Post-processing report generated successfully")
            
        except Exception as e:
            log_output(f"⚠️ Report generation failed: {e}")
            raise
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status in the format expected by the active learning orchestrator"""
        
        # Initialize the expected structure with completed_jobs and failed_jobs
        status = {
            'analysis_running': self.processing_running,
            'completed_jobs': [],
            'failed_jobs': [],
            'running_jobs': [],
            'analysis_info': None
        }
        
        if hasattr(self, 'current_analysis') and self.current_analysis:
            job_id = self.current_analysis.get('job_id', '')
            current_status = self.current_analysis.get('status', 'unknown')
            
            # Categorize the job based on its status
            if current_status == 'completed':
                status['completed_jobs'].append(job_id)
            elif current_status == 'failed':
                status['failed_jobs'].append(job_id)
            elif current_status in ['running', 'starting']:
                status['running_jobs'].append(job_id)
                
            status['analysis_info'] = self.current_analysis
        
        return status
    
    def stop_analysis(self):
        """Stop the current analysis"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_running = False
            # Update current analysis status if available
            if hasattr(self, 'current_analysis') and self.current_analysis:
                self.current_analysis['status'] = 'stopping'
            print("🛑 Post-processing analysis stop requested")
            return True
        else:
            print("⚠️ No active post-processing analysis to stop")
            return False
    
    def get_analysis_results(self, simulation_id: str) -> Dict[str, Any]:
        """Get post-processing results for a specific simulation"""
        try:
            analysis_dir = self.output_dir / simulation_id
            
            if not analysis_dir.exists():
                return {'success': False, 'error': f'No post-processing results found for {simulation_id}'}
            
            # Look for comprehensive report
            report_file = analysis_dir / "comprehensive_analysis_report.json"
            summary_file = analysis_dir / "postprocessing_summary.json"
            
            results = {'success': True, 'simulation_id': simulation_id}
            
            if report_file.exists():
                with open(report_file, 'r') as f:
                    results['comprehensive_results'] = json.load(f)
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    results['summary'] = json.load(f)
            
            # Get available output files
            output_files = {}
            for file_path in analysis_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(analysis_dir)
                    output_files[str(rel_path)] = str(file_path)
            
            results['output_files'] = output_files
            results['analysis_dir'] = str(analysis_dir)
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Testing function
def test_comprehensive_postprocessor():
    """Test the comprehensive post-processor"""
    try:
        processor = ComprehensivePostProcessor()
        print("✅ Comprehensive post-processor test passed")
        return True
    except Exception as e:
        print(f"❌ Comprehensive post-processor test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_comprehensive_postprocessor()
    print(f"Test result: {'PASSED' if success else 'FAILED'}") 