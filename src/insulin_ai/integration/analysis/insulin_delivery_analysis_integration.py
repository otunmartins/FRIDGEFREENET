#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Insulin Delivery Analysis Integration
Demonstrates how to use the comprehensive analyzer with the existing MD simulation workflow

This module integrates:
1. Existing MD simulation system (from md_simulation_integration.py)
2. Comprehensive analyzer (from insulin_comprehensive_analyzer.py)
3. Unified workflow for complete insulin delivery analysis

Usage:
    analyzer = InsulinDeliveryAnalysisIntegration()
    results = analyzer.run_complete_analysis(pdb_file, analysis_options)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

# Import existing systems
try:
    from md_simulation_integration import MDSimulationIntegration
    MD_INTEGRATION_AVAILABLE = True
except ImportError:
    MD_INTEGRATION_AVAILABLE = False

try:
    from insulin_comprehensive_analyzer import InsulinComprehensiveAnalyzer
    COMPREHENSIVE_ANALYZER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYZER_AVAILABLE = False

class InsulinDeliveryAnalysisIntegration:
    """
    Complete insulin delivery analysis system
    Integrates MD simulation with comprehensive property analysis
    """
    
    def __init__(self, output_dir: str = "complete_insulin_analysis"):
        """Initialize the complete analysis system"""
        
        if not MD_INTEGRATION_AVAILABLE:
            raise ImportError("MD simulation integration not available")
        
        if not COMPREHENSIVE_ANALYZER_AVAILABLE:
            raise ImportError("Comprehensive analyzer not available")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MD simulation system
        md_output_dir = self.output_dir / "md_simulations"
        self.md_integration = MDSimulationIntegration(str(md_output_dir), enable_mmgbsa=False)
        
        # Initialize comprehensive analyzer
        analysis_output_dir = self.output_dir / "comprehensive_analysis"
        self.comprehensive_analyzer = InsulinComprehensiveAnalyzer(str(analysis_output_dir))
        
        print(f"🔬 Complete Insulin Delivery Analysis System initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🧬 MD simulation system: Ready")
        print(f"📊 Comprehensive analyzer: Ready")
        print(f"🎯 Analysis categories: All 7 property types available")
    
    def run_complete_analysis(self, 
                            pdb_file: str,
                            simulation_options: Optional[Dict[str, Any]] = None,
                            analysis_options: Optional[Dict[str, bool]] = None,
                            output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete insulin delivery analysis workflow
        
        Args:
            pdb_file: Input PDB file (insulin + polymer system)
            simulation_options: MD simulation parameters
            analysis_options: Which analyses to run
            output_callback: Progress callback function
            
        Returns:
            Complete analysis results
        """
        def log_output(message: str):
            print(message)  # Always print to console
            if output_callback:
                output_callback(message)  # Also send to app interface
        
        log_output(f"\n🚀 Starting Complete Insulin Delivery Analysis")
        log_output(f"📄 Input PDB: {pdb_file}")
        log_output("=" * 80)
        
        # Default simulation options - Quick Test mode for user convenience
        if simulation_options is None:
            simulation_options = {
                'temperature': 310.0,          # Physiological temperature
                'equilibration_steps': 125000, # Quick Test: 250 ps equilibration
                'production_steps': 500000,    # Quick Test: 1 ns production (was 2500000 = 5 ns)
                'save_interval': 500           # Save every 1 ps for quick tests
            }
        
        # Default analysis options (run all analyses)
        if analysis_options is None:
            analysis_options = {
                'binding_energy': True,        # MM-GBSA binding free energy
                'insulin_stability': True,     # RMSD, RMSF, secondary structure
                'partitioning': True,         # PMF, partition coefficient
                'diffusion': True,            # MSD, diffusion coefficient
                'hydrogel_dynamics': True,    # Mesh size, polymer dynamics
                'interaction_energies': True, # Energy decomposition
                'swelling_response': True,    # Volume changes, water uptake
                'stimuli_response': False     # Advanced stimuli analysis
            }
        
        complete_results = {
            'input_file': pdb_file,
            'timestamp': datetime.now().isoformat(),
            'simulation_options': simulation_options,
            'analysis_options': analysis_options,
            'workflow_stages': {}
        }
        
        try:
            start_time = time.time()
            
            # Stage 1: Run MD Simulation
            log_output(f"\n🧬 Stage 1: Running MD Simulation")
            log_output("-" * 40)
            
            simulation_id = self.md_integration.run_md_simulation_async(
                pdb_file=pdb_file,
                temperature=simulation_options['temperature'],
                equilibration_steps=simulation_options['equilibration_steps'],
                production_steps=simulation_options['production_steps'],
                save_interval=simulation_options['save_interval'],
                output_callback=log_output
            )
            
            # Wait for simulation to complete
            log_output("⏳ Waiting for MD simulation to complete...")
            simulation_status = self.md_integration.wait_for_simulation_completion(
                simulation_id, output_callback=log_output
            )
            
            if not simulation_status.get('success'):
                raise RuntimeError(f"MD simulation failed: {simulation_status.get('error')}")
            
            simulation_results = simulation_status['results']
            complete_results['workflow_stages']['md_simulation'] = {
                'success': True,
                'simulation_id': simulation_id,
                'simulation_time': simulation_results.get('total_time_s'),
                'frames_generated': simulation_results.get('frames_saved'),
                'final_energy': simulation_results.get('final_energy')
            }
            
            log_output(f"✅ MD simulation completed successfully")
            log_output(f"🎯 Simulation ID: {simulation_id}")
            
            # Stage 2: Comprehensive Analysis
            log_output(f"\n📊 Stage 2: Comprehensive Property Analysis")
            log_output("-" * 40)
            
            # Run comprehensive analysis on simulation results
            analysis_results = self.comprehensive_analyzer.analyze_complete_system(
                simulation_dir=str(self.md_integration.output_dir),
                simulation_id=simulation_id,
                analysis_options=analysis_options,
                output_callback=log_output
            )
            
            if not analysis_results.get('success'):
                raise RuntimeError(f"Comprehensive analysis failed: {analysis_results.get('error')}")
            
            complete_results['workflow_stages']['comprehensive_analysis'] = {
                'success': True,
                'analysis_results': analysis_results
            }
            
            log_output(f"✅ Comprehensive analysis completed successfully")
            
            # Stage 3: Generate Integrated Report
            log_output(f"\n📋 Stage 3: Generating Integrated Report")
            log_output("-" * 40)
            
            integrated_report = self._generate_integrated_report(
                complete_results, simulation_id, log_output
            )
            
            complete_results['workflow_stages']['integrated_report'] = {
                'success': True,
                'report_file': integrated_report['report_file'],
                'summary_file': integrated_report['summary_file']
            }
            
            # Final results
            total_time = time.time() - start_time
            complete_results['success'] = True
            complete_results['total_analysis_time_s'] = total_time
            complete_results['simulation_id'] = simulation_id
            
            log_output(f"\n🎉 Complete Analysis Finished Successfully!")
            log_output(f"⏱️  Total time: {total_time:.1f} seconds")
            log_output(f"📁 Results saved to: {self.output_dir}")
            log_output(f"🆔 Simulation ID: {simulation_id}")
            
            # Print key findings
            self._print_key_findings(complete_results, log_output)
            
            return complete_results
            
        except Exception as e:
            log_output(f"❌ Complete analysis failed: {str(e)}")
            complete_results['success'] = False
            complete_results['error'] = str(e)
            return complete_results
    
    def run_analysis_on_existing_simulation(self,
                                          simulation_dir: str,
                                          simulation_id: str,
                                          analysis_options: Optional[Dict[str, bool]] = None,
                                          output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run comprehensive analysis on existing MD simulation results
        
        Args:
            simulation_dir: Directory containing existing simulation results
            simulation_id: ID of the simulation to analyze
            analysis_options: Which analyses to run
            output_callback: Progress callback function
            
        Returns:
            Analysis results
        """
        def log_output(message: str):
            print(message)  # Always print to console
            if output_callback:
                output_callback(message)  # Also send to app interface
        
        log_output(f"\n📊 Running Analysis on Existing Simulation")
        log_output(f"🎯 Simulation ID: {simulation_id}")
        log_output(f"📁 Simulation dir: {simulation_dir}")
        log_output("=" * 60)
        
        # Default analysis options
        if analysis_options is None:
            analysis_options = {
                'binding_energy': True,
                'insulin_stability': True,
                'partitioning': True,
                'diffusion': True,
                'hydrogel_dynamics': True,
                'interaction_energies': True,
                'swelling_response': True,
                'stimuli_response': False
            }
        
        try:
            # Run comprehensive analysis
            analysis_results = self.comprehensive_analyzer.analyze_complete_system(
                simulation_dir=simulation_dir,
                simulation_id=simulation_id,
                analysis_options=analysis_options,
                output_callback=log_output
            )
            
            if analysis_results.get('success'):
                log_output(f"✅ Analysis completed successfully")
                self._print_key_findings({'workflow_stages': {'comprehensive_analysis': {'analysis_results': analysis_results}}}, log_output)
            else:
                log_output(f"❌ Analysis failed: {analysis_results.get('error')}")
            
            return analysis_results
            
        except Exception as e:
            log_output(f"❌ Analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_integrated_report(self, complete_results: Dict[str, Any], 
                                  simulation_id: str, log_output: Callable) -> Dict[str, str]:
        """Generate integrated report combining all analysis results"""
        
        log_output("   📊 Creating integrated analysis report...")
        
        # Create integrated report directory
        report_dir = self.output_dir / f"integrated_report_{simulation_id}"
        report_dir.mkdir(exist_ok=True)
        
        # Extract key results from all stages
        simulation_stage = complete_results['workflow_stages'].get('md_simulation', {})
        analysis_stage = complete_results['workflow_stages'].get('comprehensive_analysis', {})
        analysis_results = analysis_stage.get('analysis_results', {})
        
        # Create integrated report
        integrated_report = {
            'report_title': 'Complete Insulin Delivery System Analysis',
            'simulation_id': simulation_id,
            'timestamp': complete_results['timestamp'],
            'input_file': complete_results['input_file'],
            'total_analysis_time_s': complete_results.get('total_analysis_time_s'),
            
            'workflow_summary': {
                'md_simulation': {
                    'success': simulation_stage.get('success'),
                    'simulation_time_s': simulation_stage.get('simulation_time'),
                    'frames_generated': simulation_stage.get('frames_generated')
                },
                'comprehensive_analysis': {
                    'success': analysis_stage.get('success'),
                    'analyses_completed': len([k for k, v in analysis_results.items() 
                                             if isinstance(v, dict) and v.get('success')])
                }
            },
            
            'key_findings': {}
        }
        
        # Extract key findings from comprehensive analysis
        if analysis_results.get('success'):
            # Binding energy
            if 'binding_energy' in analysis_results:
                be_data = analysis_results['binding_energy']
                integrated_report['key_findings']['binding_energy'] = {
                    'value_kcal_mol': be_data.get('corrected_binding_energy'),
                    'stability': 'favorable' if be_data.get('corrected_binding_energy', 0) < 0 else 'unfavorable',
                    'method': be_data.get('method')
                }
            
            # Insulin stability
            if 'insulin_stability' in analysis_results:
                is_data = analysis_results['insulin_stability']
                integrated_report['key_findings']['insulin_stability'] = {
                    'rmsd_mean_A': is_data.get('rmsd', {}).get('mean'),
                    'assessment': is_data.get('rmsd', {}).get('stability_assessment'),
                    'structural_integrity': 'maintained' if is_data.get('rmsd', {}).get('mean', 999) < 3.0 else 'compromised'
                }
            
            # Diffusion
            if 'diffusion' in analysis_results:
                diff_data = analysis_results['diffusion']
                integrated_report['key_findings']['diffusion'] = {
                    'coefficient_cm2_s': diff_data.get('msd_analysis', {}).get('diffusion_coefficient'),
                    'assessment': diff_data.get('diffusion_assessment'),
                    'delivery_feasibility': 'excellent' if diff_data.get('diffusion_assessment') == 'within_experimental_range' else 'needs_optimization'
                }
            
            # Hydrogel properties
            if 'hydrogel_dynamics' in analysis_results:
                hg_data = analysis_results['hydrogel_dynamics']
                integrated_report['key_findings']['hydrogel_properties'] = {
                    'mesh_size_A': hg_data.get('mesh_size_analysis', {}).get('average_mesh_size'),
                    'polymer_flexibility': hg_data.get('polymer_dynamics', {}).get('flexibility_index'),
                    'network_quality': 'good' if hg_data.get('mesh_size_analysis', {}).get('average_mesh_size', 0) > 10 else 'tight'
                }
            
            # Overall assessment
            integrated_report['overall_assessment'] = self._assess_delivery_potential(integrated_report['key_findings'])
        
        # Save integrated report
        report_file = report_dir / "integrated_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(integrated_report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_file = report_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPLETE INSULIN DELIVERY SYSTEM ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Simulation ID: {simulation_id}\n")
            f.write(f"Analysis Date: {complete_results['timestamp']}\n")
            f.write(f"Input File: {complete_results['input_file']}\n\n")
            
            if 'key_findings' in integrated_report:
                findings = integrated_report['key_findings']
                
                f.write("KEY FINDINGS:\n")
                f.write("-" * 40 + "\n")
                
                if 'binding_energy' in findings:
                    be = findings['binding_energy']
                    binding_energy = be.get('value_kcal_mol')
                    if binding_energy is not None:
                        f.write(f"🧮 Binding Energy: {binding_energy:.2f} kcal/mol ({be.get('stability', 'unknown')})\n")
                    else:
                        f.write(f"🧮 Binding Energy: N/A ({be.get('stability', 'unknown')})\n")
                
                if 'insulin_stability' in findings:
                    ins = findings['insulin_stability']
                    rmsd_mean = ins.get('rmsd_mean_A')
                    if rmsd_mean is not None:
                        f.write(f"🧪 Insulin Stability: {ins.get('assessment', 'unknown')} (RMSD: {rmsd_mean:.2f} Å)\n")
                    else:
                        f.write(f"🧪 Insulin Stability: {ins.get('assessment', 'unknown')} (RMSD: N/A)\n")
                
                if 'diffusion' in findings:
                    diff = findings['diffusion']
                    diff_coef = diff.get('coefficient_cm2_s')
                    if diff_coef is not None:
                        f.write(f"🚶 Diffusion: {diff_coef:.2e} cm²/s ({diff.get('assessment', 'unknown')})\n")
                    else:
                        f.write(f"🚶 Diffusion: N/A ({diff.get('assessment', 'unknown')})\n")
                
                if 'hydrogel_properties' in findings:
                    hg = findings['hydrogel_properties']
                    mesh_size = hg.get('mesh_size_A')
                    if mesh_size is not None:
                        f.write(f"🕸️ Mesh Size: {mesh_size:.1f} Å ({hg.get('network_quality', 'unknown')} network)\n")
                    else:
                        f.write(f"🕸️ Mesh Size: N/A ({hg.get('network_quality', 'unknown')} network)\n")
                
                f.write(f"\n🎯 OVERALL ASSESSMENT: {integrated_report.get('overall_assessment', 'Incomplete analysis')}\n")
            
            f.write(f"\n📁 Detailed results available in: {report_dir}\n")
        
        log_output(f"✅ Integrated report saved: {report_file}")
        log_output(f"📋 Summary saved: {summary_file}")
        
        return {
            'report_file': str(report_file),
            'summary_file': str(summary_file)
        }
    
    def _assess_delivery_potential(self, key_findings: Dict[str, Any]) -> str:
        """Assess overall insulin delivery potential based on key findings"""
        
        scores = []
        
        # Binding energy assessment
        if 'binding_energy' in key_findings:
            binding_energy = key_findings['binding_energy'].get('value_kcal_mol', 0)
            if binding_energy < -5:  # Strong binding
                scores.append(2)
            elif binding_energy < 0:  # Weak binding
                scores.append(1)
            else:  # No binding or repulsive
                scores.append(0)
        
        # Stability assessment
        if 'insulin_stability' in key_findings:
            assessment = key_findings['insulin_stability'].get('assessment', 'unknown')
            if assessment == 'stable':
                scores.append(2)
            elif assessment == 'unstable':
                scores.append(0)
            else:
                scores.append(1)
        
        # Diffusion assessment
        if 'diffusion' in key_findings:
            assessment = key_findings['diffusion'].get('assessment', 'unknown')
            if assessment == 'within_experimental_range':
                scores.append(2)
            elif assessment == 'highly_constrained':
                scores.append(1)
            else:
                scores.append(0)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 1.5:
                return "EXCELLENT - High potential for effective insulin delivery"
            elif avg_score >= 1.0:
                return "GOOD - Promising system with some optimization potential"
            elif avg_score >= 0.5:
                return "MODERATE - Significant optimization needed"
            else:
                return "POOR - Major redesign recommended"
        else:
            return "INCOMPLETE - Insufficient data for assessment"
    
    def _print_key_findings(self, complete_results: Dict[str, Any], log_output: Callable):
        """Print key findings to console"""
        
        analysis_results = complete_results.get('workflow_stages', {}).get('comprehensive_analysis', {}).get('analysis_results', {})
        
        if not analysis_results.get('success'):
            return
        
        log_output(f"\n🔍 KEY FINDINGS:")
        log_output("=" * 40)
        
        # Binding energy
        if 'binding_energy' in analysis_results:
            be_data = analysis_results['binding_energy']
            energy = be_data.get('corrected_binding_energy', 'N/A')
            log_output(f"🧮 Binding Energy: {energy:.2f} kcal/mol")
        
        # Insulin stability  
        if 'insulin_stability' in analysis_results:
            is_data = analysis_results['insulin_stability']
            rmsd = is_data.get('rmsd', {}).get('mean', 'N/A')
            assessment = is_data.get('rmsd', {}).get('stability_assessment', 'unknown')
            log_output(f"🧪 Insulin Stability: {assessment} (RMSD: {rmsd:.2f} Å)")
        
        # Diffusion
        if 'diffusion' in analysis_results:
            diff_data = analysis_results['diffusion']
            coeff = diff_data.get('msd_analysis', {}).get('diffusion_coefficient', 'N/A')
            assessment = diff_data.get('diffusion_assessment', 'unknown')
            log_output(f"🚶 Diffusion: {coeff:.2e} cm²/s ({assessment})")
        
        # Hydrogel mesh size
        if 'hydrogel_dynamics' in analysis_results:
            hg_data = analysis_results['hydrogel_dynamics']
            mesh_size = hg_data.get('mesh_size_analysis', {}).get('average_mesh_size', 'N/A')
            log_output(f"🕸️ Mesh Size: {mesh_size:.1f} Å")
        
        # Swelling
        if 'swelling_response' in analysis_results:
            sw_data = analysis_results['swelling_response']
            ratio = sw_data.get('swelling_ratio', 'N/A')
            assessment = sw_data.get('swelling_assessment', 'unknown')
            log_output(f"💧 Swelling: {ratio:.2f} ({assessment})")

def test_complete_integration():
    """Test the complete integration system"""
    try:
        integration = InsulinDeliveryAnalysisIntegration()
        print("✅ Complete integration system test passed")
        return True
    except Exception as e:
        print(f"❌ Complete integration system test failed: {e}")
        return False

if __name__ == "__main__":
    if not MD_INTEGRATION_AVAILABLE:
        print("❌ MD simulation integration not available")
    elif not COMPREHENSIVE_ANALYZER_AVAILABLE:
        print("❌ Comprehensive analyzer not available")
    else:
        success = test_complete_integration()
        print(f"Test result: {'PASSED' if success else 'FAILED'}") 