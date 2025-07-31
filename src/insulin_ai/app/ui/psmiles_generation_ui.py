"""
Enhanced PSMILES Generation UI Module for Insulin-AI Application

This module provides comprehensive PSMILES generation interface including detailed
progress tracking, advanced functionalization, and enhanced visualization to match
the original app's sophisticated functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Use absolute imports with fallbacks
try:
    from app.utils.session_utils import safe_get_session_object
    from app.services.psmiles_service import process_psmiles_workflow_with_autorepair
    from app.utils.psmiles_smiles_storage import update_session_candidates_with_smiles
    print("✅ PSMILES Generation UI: All imports successful")
except ImportError as e:
    print(f"❌ PSMILES Generation UI: Import error: {e}")
    # Fallback imports for standalone operation
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from app.utils.session_utils import safe_get_session_object
        from app.services.psmiles_service import process_psmiles_workflow_with_autorepair  
        from app.utils.psmiles_smiles_storage import update_session_candidates_with_smiles
        print("✅ PSMILES Generation UI: Fallback imports successful")
    except ImportError as e2:
        print(f"❌ PSMILES Generation UI: Fallback imports failed: {e2}")
        # Define minimal stubs
        def safe_get_session_object(key):
            return getattr(st.session_state, key, None)
        def process_psmiles_workflow_with_autorepair(*args, **kwargs):
            return None
        def update_session_candidates_with_smiles():
            pass


# Define validation functions locally to avoid circular imports
def validate_psmiles_processor(processor) -> bool:
    """Validate that PSMILESProcessor has required methods"""
    if not processor:
        return False
    
    required_methods = [
        'process_psmiles_workflow_with_autorepair',
        'add_random_functional_groups',
        'get_session_psmiles'
    ]
    
    for method in required_methods:
        if not hasattr(processor, method):
            return False
    
    return True

def force_refresh_psmiles_processor():
    """Force refresh PSMILESProcessor with latest functionality"""
    try:
        # Clear cached processor
        if 'psmiles_processor' in st.session_state:
            del st.session_state.psmiles_processor
        
        # Reinitialize
        from insulin_ai import PSMILESProcessor
        processor = PSMILESProcessor()
        st.session_state.psmiles_processor = processor
        
        # Validate
        if validate_psmiles_processor(processor):
            return True, "✅ PSMILESProcessor refreshed with auto-repair functionality!"
        else:
            return False, "❌ PSMILESProcessor still missing required methods"
            
    except Exception as e:
        return False, f"❌ Failed to refresh PSMILESProcessor: {str(e)}"

# Import auto-correction functionality
try:
    from integration.corrections.instant_psmiles_corrector import apply_instant_corrections_ui
    AUTOCORRECTOR_AVAILABLE = True
except ImportError:
    AUTOCORRECTOR_AVAILABLE = False


def check_pipeline_health() -> Tuple[bool, str]:
    """Check if the PSMILES generation pipeline is working correctly"""
    if hasattr(st.session_state, 'psmiles_generator') and st.session_state.psmiles_generator:
        psmiles_gen = st.session_state.psmiles_generator
        has_nl_pipeline = hasattr(psmiles_gen, 'nl_to_psmiles') and psmiles_gen.nl_to_psmiles is not None
        
        if not has_nl_pipeline:
            return False, "🚨 **BROKEN PIPELINE DETECTED** - Using direct PSMILES generation (problematic)"
        else:
            return True, "✅ **WORKING PIPELINE ACTIVE** - Using Natural Language → SMILES → PSMILES"
    
    return False, "❌ PSMILES generator not initialized"


def render_comprehensive_pipeline_health_check():
    """Render simplified pipeline health check and fix options"""
    healthy, message = check_pipeline_health()
    
    if healthy:
        st.success(message)
        st.info("🔧 Pipeline: Natural Language → SMILES (with repair) → PSMILES conversion")
    else:
        st.error(message)
        if "BROKEN PIPELINE" in message:
            st.error("❌ The app is not using the working Natural Language → SMILES → PSMILES pipeline")
            st.warning("🔄 **SOLUTION**: Click 'Force Update Generator' button below, then restart the Streamlit app")
            
            col_fix1, col_fix2 = st.columns(2) 
            with col_fix1:
                if st.button("🔄 Quick Fix - Update Generator", type="primary"):
                    # Clear generator and force reinit
                    if 'psmiles_generator' in st.session_state:
                        del st.session_state.psmiles_generator
                    st.cache_resource.clear()
                    st.success("✅ Generator cleared - refresh the page!")
                    st.rerun()
            with col_fix2:
                st.info("💡 After clicking, **restart Streamlit** with Ctrl+C then `streamlit run app.py`")
        
        # Additional PSMILESProcessor validation
        psmiles_processor = safe_get_session_object('psmiles_processor')
        if not psmiles_processor:
            st.error("❌ PSMILES Processor not available. Please restart the application.")
        elif not validate_psmiles_processor(psmiles_processor):
            st.warning("🔧 PSMILESProcessor missing auto-repair functionality. Refreshing...")
            if st.button("🔧 Fix PSMILESProcessor", type="secondary"):
                success, fix_message = force_refresh_psmiles_processor()
                if success:
                    st.success("✅ PSMILESProcessor refreshed with auto-repair!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to refresh PSMILESProcessor: {fix_message}")
                    st.error("🔄 **Please use the 'Fix PSMILESProcessor' button in the sidebar and try again.**")


def render_advanced_generation_options():
    """Render simplified material generation options and parameters"""
    st.markdown("#### 🤖 Automated Pipeline")
    st.markdown("_Generate candidates → Functionalize → Create simulation boxes → Visualize structures_")
    
    # Input area (full width)
    material_request = st.text_area(
        "Describe your material:",
        placeholder="biocompatible polymer for insulin delivery with amide groups and sulfur functionality...",
        height=100,
        help="Describe the polymer properties and functional groups you want"
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        num_candidates = st.slider("Number of candidates:", 3, 10, 5)
        auto_functionalize = st.checkbox("Multi-step functionalization", value=True)
    with col2:
        st.info("🌡️ Using diverse generation (T=0.6-1.0) for variety")
        
        # Cache clearing option  
        if st.button("🔄 Force Update Generator", help="Clear cache and reinitialize with latest features"):
            # Clear all related session state
            keys_to_clear = ['psmiles_generator', 'literature_miner', 'chatbot', 'systems_initialized']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.cache_resource.clear()
            
            # Force reinitialize systems to pick up latest code
            st.session_state.systems_initialized = False
            
            st.success("🎉 Generator cache cleared! Page will refresh with latest features.")
            st.info("🔄 Systems will reinitialize with the working SMILES→PSMILES pipeline on next refresh.")
            st.rerun()
    
    # Set default values for the simulation parameters that are no longer part of this UI
    auto_create_polymer_boxes = False
    auto_create_insulin_systems = False
    polymer_length = 5
    num_polymer_molecules = 8
    density = 0.3
    tolerance_distance = 3.5
    timeout_minutes = 15
    num_insulin_molecules = 1
    box_size_nm = 3.0
    
    return (material_request, num_candidates, auto_functionalize, 
            auto_create_polymer_boxes, auto_create_insulin_systems,
            polymer_length, num_polymer_molecules, density, 
            tolerance_distance, timeout_minutes,
            num_insulin_molecules, box_size_nm)


def render_hybrid_approach_info():
    """Render comprehensive information about the hybrid generation approach"""
    with st.expander("🔬 Hybrid Generation Approach", expanded=False):
        st.markdown("""
        **Our system uses a sophisticated 3-tier approach with automatic repair:**
        
        1. **🎯 Direct PSMILES Generation** - Creative, diverse structures from LLM
        2. **✅ Chemical Validation + Repair** - Multi-layer SMILES repair system  
        3. **🧪 Simple SMILES→PSMILES** - Generate SMILES, validate, then `[*]SMILES[*]`
        
        **Auto-Repair System:**
        - **🔧 Basic Cleaning**: Fixes brackets, parentheses, ring closures
        - **🧬 SELFIES Repair**: Uses SELFIES format for autocorrection
        - **🛡️ Smart Fallback**: Pattern-based molecular recognition
        
        **Benefits:**
        - **Diversity**: Direct generation creates unique polymer structures
        - **Auto-Fix**: Repairs invalid SMILES (like `B(OH)` → `B(O)(O)`)  
        - **Robustness**: Multiple repair layers ensure valid chemistry
        
        Look for these badges in results:
        - 🎯✅ **Direct + Validated**: Best of both worlds!
        - 🧪✅ **SMILES→PSMILES + Validated**: Simple, robust conversion
        - 🔧 **Cleaned**: Basic syntax repair applied
        - 🧬 **SELFIES**: Advanced autocorrection used
        """)


def execute_automated_pipeline_with_progress(material_request: str, num_candidates: int, auto_functionalize: bool, simulation_params: Optional[Dict] = None) -> Optional[Dict]:
    """Execute the automated generation pipeline with simplified progress tracking"""
    
    st.markdown("---")
    st.markdown("### 🔄 Generation Progress")
    
    progress_bar = st.progress(0)
    
    # Step 1: Generation
    progress_bar.progress(25, "Generating diverse candidates...")
    
    # Validate and get psmiles_generator
    psmiles_generator = safe_get_session_object('psmiles_generator')
    if not psmiles_generator:
        st.error("❌ PSMILES Generator not available. Please restart the application.")
        return None
    
    # Check if generator has the new method, reinitialize if not
    if not hasattr(psmiles_generator, 'generate_diverse_candidates'):
        st.info("🔄 Updating PSMILES Generator...")
        # Get current OpenAI settings from session state
        openai_model = st.session_state.get('openai_model', 'gpt-4o')
        temperature = st.session_state.get('temperature', 0.7)
        
        try:
            from insulin_ai import PSMILESGenerator
            # Reinitialize with new features using OpenAI
            psmiles_generator = PSMILESGenerator(
                model_type='openai',
                openai_model=openai_model,
                temperature=temperature
            )
            st.session_state.psmiles_generator = psmiles_generator
            st.success("✅ Generator updated!")
        except Exception as e:
            st.error(f"❌ Failed to update generator: {e}")
            return None
    
    # Generate candidates
    try:
        if hasattr(psmiles_generator, 'nl_to_psmiles') and psmiles_generator.nl_to_psmiles:
            st.success("✅ Using working pipeline: Natural Language → SMILES → PSMILES")
        
        # Generate with enhanced diversity
        diverse_results = psmiles_generator.generate_diverse_candidates(
            base_request=material_request,
            num_candidates=num_candidates * 2,  # Generate more to ensure diversity
            temperature_range=(0.6, 1.0)
        )
        
        # Convert to the format expected by the pipeline
        generated_candidates = []
        
        if diverse_results.get('success') and diverse_results.get('candidates'):
            candidates_list = diverse_results['candidates']
            for result in candidates_list:
                generated_candidates.append({
                    'psmiles': result['psmiles'],
                    'prompt': result.get('diversity_prompt', material_request),
                    'method': result.get('generation_method', 'working_pipeline_diverse'),
                    'explanation': result.get('explanation', 'Diverse generated structure'),
                    'generation_temperature': result.get('temperature_used', 0.8),
                    'generation_attempt': result.get('attempt_number', 1)
                })
        else:
            raise Exception(f"Diverse generation failed: {diverse_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.warning(f"⚠️ Diverse generation failed: {e}")
        st.info("🔄 Using fallback generation...")
        
        # Fallback to standard generation
        generated_candidates = []
        diversity_prompts = [
            material_request,
            f"biocompatible polymer incorporating {material_request}",
            f"linear polymer backbone with {material_request} functional groups",
            f"branched copolymer design featuring {material_request}",
        ]
        
        for i in range(num_candidates):
            prompt = diversity_prompts[i % len(diversity_prompts)]
            result = psmiles_generator.generate_psmiles(prompt)
            
            if result.get('success'):
                generated_candidates.append({
                    'psmiles': result['psmiles'],
                    'prompt': prompt,
                    'method': 'fallback_generation',
                    'explanation': result.get('explanation', 'Fallback generated structure'),
                    'generation_temperature': 0.8,
                    'generation_attempt': i + 1
                })
    
    # Step 2: Selection
    progress_bar.progress(50, "Selecting best candidates...")
    unique_psmiles = set(candidate['psmiles'] for candidate in generated_candidates)
    avg_temperature = np.mean([candidate.get('generation_temperature', 0.8) for candidate in generated_candidates])
    
    st.info(f"🎯 Generated {len(unique_psmiles)} unique structures (avg T={avg_temperature:.2f})")
    selected_candidates = generated_candidates[:min(5, len(generated_candidates))]
    
    # Step 3: Functionalization
    progress_bar.progress(75, "Applying functionalization...")
    
    functionalized_candidates = []
    if auto_functionalize:
        psmiles_processor = safe_get_session_object('psmiles_processor')
        if not psmiles_processor:
            st.error("❌ PSMILES Processor not available.")
            return None
        
        # Process each candidate with functionalization (simplified progress)
        for i, candidate in enumerate(selected_candidates):
            try:
                original = candidate['psmiles']
                
                # Process with auto-repair
                workflow_result = psmiles_processor.process_psmiles_workflow_with_autorepair(
                    original, st.session_state.session_id, "automated_functionalization"
                )
                
                if workflow_result['success']:
                    session_psmiles = psmiles_processor.get_session_psmiles(st.session_state.session_id)
                    if session_psmiles:
                        psmiles_index = len(session_psmiles) - 1
                        
                        # Apply functionalization
                        first_result = psmiles_processor.add_random_functional_groups(
                            session_id=st.session_state.session_id,
                            psmiles_index=psmiles_index,
                            num_groups=2,
                            random_seed=42 + hash(original) % 1000
                        )
                        
                        if first_result['success']:
                            final_functionalized = first_result['canonical_psmiles']
                            first_groups = [group['name'] for group in first_result['applied_groups']]
                            
                            functionalized_candidates.append({
                                'original': original,
                                'functionalized': final_functionalized,
                                'modification': f"Applied {len(first_groups)} groups: {', '.join(first_groups)}",
                                'modifications_count': len(first_groups),
                                'first_round_groups': first_groups,
                                'prompt': candidate['prompt'],
                                'method': candidate['method'],
                                'functionalization_method': 'automated_psmiles_library',
                                'is_valid': True,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            # Add unfunctionalized
                            functionalized_candidates.append({
                                'original': original,
                                'functionalized': original,
                                'modification': 'Functionalization failed',
                                'modifications_count': 0,
                                'prompt': candidate['prompt'],
                                'method': candidate['method'],
                                'functionalization_method': 'failed',
                                'is_valid': True,
                                'timestamp': datetime.now().isoformat()
                            })
                            
            except Exception as e:
                st.warning(f"⚠️ Error processing candidate: {e}")
                continue
    else:
        # No functionalization - just format the candidates
        for candidate in selected_candidates:
            functionalized_candidates.append({
                'original': candidate['psmiles'],
                'functionalized': candidate['psmiles'],
                'modification': 'Auto-functionalization disabled',
                'modifications_count': 0,
                'prompt': candidate['prompt'],
                'method': candidate['method'],
                'functionalization_method': 'disabled',
                'is_valid': True,
                'timestamp': datetime.now().isoformat()
            })
    
    # Step 4: Simulation Automation (if enabled)
    simulation_results = None
    
    if simulation_params and (simulation_params.get('auto_create_polymer_boxes') or simulation_params.get('auto_create_insulin_systems')):
        st.info("🧪 Starting automated simulation box creation...")
        progress_bar.progress(80, "Creating simulation boxes...")
        
        try:
            # Import the automation pipeline
            from integration.automation.simulation_automation import run_automated_simulation_pipeline
            
            # Create UI callback for logging
            simulation_log_container = st.empty()
            simulation_logs = []
            
            def simulation_callback(msg):
                simulation_logs.append(msg)
                with simulation_log_container.container():
                    # Show more log lines to capture error details
                    recent_logs = simulation_logs[-10:]  # Show last 10 log lines
                    st.text("\n".join(recent_logs))
            
            st.write("**Simulation Progress:**")
            
            # Run the automation pipeline with parameters
            simulation_results = run_automated_simulation_pipeline(
                candidates=functionalized_candidates,
                simulation_params=simulation_params,
                output_callback=simulation_callback
            )
            
            if simulation_results and simulation_results.get('success_count', 0) > 0:
                st.success(f"✅ Created simulation boxes for {simulation_results.get('success_count', 0)}/{simulation_results.get('total_candidates', 0)} candidates")
                
                # Display simulation summary
                with st.expander("📊 Simulation Results Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Polymer Boxes Created", len([r for r in simulation_results.get('polymer_boxes', []) if r.get('success')]))
                    with col2:
                        st.metric("Insulin Systems Created", len([r for r in simulation_results.get('insulin_systems', []) if r.get('success')]))
                    with col3:
                        st.metric("Success Rate", f"{(simulation_results.get('success_count', 0)/max(1, simulation_results.get('total_candidates', 1))*100):.1f}%")
                    
                    if simulation_results.get('session_id'):
                        st.info(f"📁 Results saved in session: {simulation_results.get('session_id', 'unknown')}")
            else:
                st.warning("⚠️ Simulation automation completed with limited success")
                
                # Show detailed error information for failed candidates
                if simulation_results and simulation_results.get('errors'):
                    with st.expander("🔍 Error Details", expanded=False):
                        for error_info in simulation_results.get('errors', []):
                            candidate_id = error_info.get('candidate_id', 'unknown')
                            psmiles = error_info.get('psmiles', 'N/A')
                            
                            if error_info.get('polymer_box', {}).get('error'):
                                st.error(f"**Candidate {candidate_id}**: {error_info['polymer_box']['error']}")
                                st.code(f"PSMILES: {psmiles}", language="text")
                            
                            if error_info.get('insulin_system', {}).get('error'):
                                st.error(f"**Candidate {candidate_id} (Insulin System)**: {error_info['insulin_system']['error']}")
                
        except ImportError:
            st.warning("⚠️ Simulation automation not available - missing dependencies")
        except Exception as e:
            st.error(f"❌ Simulation automation failed: {e}")
            
            # Show full error details in an expander
            with st.expander("🔍 Full Error Details", expanded=False):
                import traceback
                st.code(traceback.format_exc(), language="python")
    
    # Step 5: Final Completion
    progress_bar.progress(100, "Pipeline completed!")
    st.success("🎉 Generation and automation pipeline completed successfully!")
    
    # **NEW: Enhance candidates with SMILES data for MD simulation readiness**
    from app.utils.psmiles_smiles_storage import PSMILESWithSMILESStorage
    
    if functionalized_candidates:
        # Initialize SMILES storage utility
        storage = PSMILESWithSMILESStorage()
        
        # Enhance each candidate with SMILES data
        progress_bar.progress(90, "Enhancing candidates with SMILES data for MD simulation...")
        enhanced_candidates = []
        
        for i, candidate in enumerate(functionalized_candidates):
            # Enhance with SMILES
            enhanced_candidate = storage.enhance_candidate_with_smiles(candidate)
            enhanced_candidate['request'] = material_request
            # Add simulation results to candidates if available
            enhanced_candidate['simulation_results'] = simulation_results or {}
            enhanced_candidates.append(enhanced_candidate)
            
            # Show progress for SMILES enhancement
            if enhanced_candidate.get('smiles_conversion_success'):
                st.info(f"✅ Candidate {i+1}: Enhanced with SMILES - Ready for MD simulation")
            else:
                st.warning(f"⚠️ Candidate {i+1}: SMILES conversion failed - {enhanced_candidate.get('smiles_conversion_error', 'Unknown error')}")
        
        # Store enhanced candidates in session state
        for enhanced_candidate in enhanced_candidates:
            st.session_state.psmiles_candidates.append(enhanced_candidate)
            
        progress_bar.progress(100, "Pipeline completed with SMILES enhancement!")
        
        # Return enhanced candidates
        functionalized_candidates = enhanced_candidates
    
    # Add to session state
    # if functionalized_candidates:
    #     for candidate in functionalized_candidates:
    #         candidate['request'] = material_request
    #         # Add simulation results to candidates if available
    #         candidate['simulation_results'] = simulation_results or {}
    #         st.session_state.psmiles_candidates.append(candidate)
    
    return {
        'candidates': functionalized_candidates,
        'simulation_results': simulation_results or {},
        'workflow_summary': {
            'total_generated': len(generated_candidates),
            'unique_structures': len(unique_psmiles),
            'average_temperature': avg_temperature,
            'functionalized_count': len([c for c in functionalized_candidates if c['functionalization_method'] != 'disabled']),
            'success_rate': len(functionalized_candidates) / max(1, len(selected_candidates)),
            'simulation_enabled': bool(simulation_params and (simulation_params.get('auto_create_polymer_boxes') or simulation_params.get('auto_create_insulin_systems'))),
            'simulation_success_rate': (simulation_results.get('success_count', 0)/max(1, simulation_results.get('total_candidates', 1))) if simulation_results else 0.0
        }
    }


def render_enhanced_candidate_details(candidate: Dict[str, Any], index: int):
    """Render detailed information for a PSMILES candidate with SVG visualization and SMILES data"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # **PSMILES Structure Display**
        structure_title = candidate.get('functionalized', candidate.get('psmiles', 'Unknown')).replace('[]', '[*]')
        
        st.markdown(f"""
        <div class="psmiles-display">
            <strong>PSMILES:</strong> <code>{structure_title}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # **NEW: SMILES Display (if available from enhanced storage)**
        if candidate.get('smiles'):
            smiles_status = "✅" if candidate.get('smiles_conversion_success', False) else "⚠️"
            conversion_method = candidate.get('smiles_conversion_method', 'unknown')
            
            st.markdown(f"""
            <div class="psmiles-display" style="border-color: #17a2b8; background: #e9f7ff;">
                <strong>SMILES {smiles_status}:</strong> <code>{candidate['smiles']}</code>
                <br><small><em>Converted via: {conversion_method}</em></small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show conversion metadata if available
            if candidate.get('conversion_metadata'):
                metadata = candidate['conversion_metadata']
                with st.expander("🔍 SMILES Conversion Details"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Atom Count", metadata.get('atom_count', 'unknown'))
                        st.text(f"Method: {conversion_method}")
                    with col_b:
                        successful_methods = metadata.get('successful_methods', [])
                        st.text(f"Methods tried: {len(metadata.get('all_methods_tried', []))}")
                        st.text(f"Successful: {len(successful_methods)}")
                        
                    st.caption(f"Description: {metadata.get('conversion_description', 'N/A')}")
        elif candidate.get('smiles_conversion_error'):
            st.markdown(f"""
            <div class="psmiles-display" style="border-color: #dc3545; background: #ffe6e6;">
                <strong>SMILES ❌:</strong> <em>Conversion failed</em>
                <br><small><em>Error: {candidate['smiles_conversion_error']}</em></small>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show notice about missing SMILES data (for older candidates)
            st.info("ℹ️ **SMILES data not available** - Generated before enhanced storage system")
        
        # **CRITICAL: SVG Visualization - the most important part**
        psmiles_to_visualize = candidate.get('functionalized', candidate.get('psmiles'))
        if psmiles_to_visualize:
            # Get the PSMILES processor to generate visualization
            psmiles_processor = safe_get_session_object('psmiles_processor')
            if psmiles_processor:
                try:
                    # Process the PSMILES through workflow to get SVG visualization
                    workflow_result = psmiles_processor.process_psmiles_workflow_with_autorepair(
                        psmiles_to_visualize, st.session_state.session_id, f"visualization_{index}"
                    )
                    
                    # Extract SVG content from workflow result
                    svg_content = None
                    if workflow_result.get('success') and workflow_result.get('svg_content'):
                        svg_content = workflow_result['svg_content']
                    
                    # Display SVG visualization
                    if svg_content:
                        st.markdown("### 🧪 Structure Visualization")
                        
                        # Clean SVG content for Streamlit compatibility
                        if svg_content.startswith('<?xml'):
                            svg_start = svg_content.find('<svg')
                            if svg_start > 0:
                                svg_content = svg_content[svg_start:]
                        
                        # Display SVG with clean styling
                        import streamlit.components.v1 as components
                        components.html(f"""
                        <div class="visualization-container">
                            {svg_content}
                        </div>
                        """, height=300)
                    else:
                        st.markdown("### 📝 Structure")
                        st.code(psmiles_to_visualize)
                        st.caption("(SVG visualization not available)")
                        
                except Exception as e:
                    st.markdown("### 📝 Structure")
                    st.code(psmiles_to_visualize)
                    st.caption(f"(Visualization error: {e})")
            else:
                st.markdown("### 📝 Structure")
                st.code(psmiles_to_visualize)
                st.caption("(PSMILES processor not available)")
        
        # Validation status
        if candidate.get('is_valid', True):
            st.success("✅ Chemically Valid")
        else:
            st.error("❌ Validation Failed")
        
        # Processing mode status (if fallback was used)
        if candidate.get('processing_mode') == 'fallback':
            st.warning("🔧 Processed with fallback mode (psmiles library limitations)")
        
        # **NEW: MD Simulation Readiness Indicator**
        if candidate.get('smiles_conversion_success'):
            st.success("🚀 **Ready for MD Simulation** - SMILES available for force field generation")
        elif candidate.get('smiles'):
            st.warning("⚠️ **MD Simulation may have issues** - SMILES conversion had errors")
        else:
            st.error("❌ **Not ready for MD Simulation** - No SMILES data available")
        
        # **Simplified functionalization display**
        if candidate.get('functionalization_method') == 'automated_psmiles_library':
            st.markdown("#### 🔄 Functionalization Progression")
            
            # Show progression more simply
            st.markdown("**1. Original:** `{}`".format(candidate.get('original', 'Unknown')))
            st.markdown("**2. Functionalized:** `{}`".format(candidate.get('functionalized', 'Unknown')))
            
            # Summary of modifications
            total_groups = len(candidate.get('first_round_groups', []))
            if total_groups > 0:
                st.info(f"🎯 **Total functional groups added:** {total_groups}")
                if candidate.get('first_round_groups'):
                    st.text(f"Round 1: {', '.join(candidate['first_round_groups'])}")
                
        elif candidate.get('functionalization_method') == 'disabled':
            st.info("❌ **Functionalization Disabled**")
        
        # Show modification details
        if candidate.get('modification'):
            st.markdown("**Modification Details:**")
            st.text(candidate['modification'])
    
    with col2:
        # **Simplified candidate information**
        st.markdown("### 📊 Candidate Info")
        
        # Generation method
        method = candidate.get('method', 'unknown')
        if 'working_pipeline' in method:
            st.success(f"🎯 Working Pipeline")
        else:
            st.info(f"🔧 {method}")
        
        # Temperature used
        temp = candidate.get('generation_temperature', 'N/A')
        if isinstance(temp, (int, float)):
            st.metric("Temperature", f"{temp:.2f}")
        
        # Functionalization status
        func_method = candidate.get('functionalization_method', 'unknown')
        if func_method == 'automated_psmiles_library':
            st.success("🏗️ Functionalized")
        elif func_method == 'disabled':
            st.info("❌ Not Functionalized")
        
        # Timestamp
        if candidate.get('timestamp'):
            try:
                timestamp = datetime.fromisoformat(candidate['timestamp'])
                st.caption(f"Generated: {timestamp.strftime('%H:%M:%S')}")
            except:
                st.caption("Recently generated")


def render_enhanced_generation_results(results: Dict[str, Any]):
    """Display simplified results of PSMILES generation with focus on visualization"""
    st.markdown("### 🎯 Generation Results")
    
    if results.get('candidates'):
        st.success(f"✅ Generated {len(results['candidates'])} candidates successfully!")
        
        # **Enhanced metrics showing all vs selected candidates**
        total_generated = results.get('total_generated', len(results.get('candidates', [])))
        selected_count = len(results.get('candidates', []))
        all_candidates_count = len(results.get('all_candidates', []))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_generated}</h3>
                <p>Total Generated</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{all_candidates_count}</h3>
                <p>Including Variants</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{selected_count}</h3>
                <p>Selected Best</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            # Calculate success rate from workflow summary or estimate
            if results.get('workflow_summary'):
                success_rate = results['workflow_summary'].get("success_rate", 0) * 100
            else:
                # Estimate success rate as valid candidates / total attempts
                success_rate = (selected_count / max(1, total_generated)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # **Display ALL candidates with visualization**
        # Check if we have both selected and all candidates
        all_candidates = results.get('all_candidates', results.get('candidates', []))
        selected_candidates = results.get('candidates', [])
        
        if len(all_candidates) > len(selected_candidates):
            st.info(f"🎯 Showing ALL {len(all_candidates)} generated candidates (including variants) instead of just {len(selected_candidates)} selected ones")
            candidates_to_show = all_candidates
        else:
            candidates_to_show = selected_candidates
        
        for i, candidate in enumerate(candidates_to_show, 1):
            psmiles_preview = candidate.get('functionalized', candidate.get('psmiles', 'Unknown'))[:50]
            
            # Add indicator if this is a selected candidate vs. variant
            is_selected = candidate in selected_candidates if len(all_candidates) > len(selected_candidates) else True
            status_icon = "⭐" if is_selected else "🧪"
            status_text = "Selected" if is_selected else "Variant"
            
            with st.expander(f"{status_icon} Candidate {i} ({status_text}): {psmiles_preview}{'...' if len(psmiles_preview) >= 50 else ''}", expanded=i == 1):
                render_enhanced_candidate_details(candidate, i)
    
    # Enhanced workflow summary with simulation details
    if results.get('workflow_summary'):
        with st.expander("📊 Workflow Details", expanded=False):
            summary = results['workflow_summary']
            
            # Generation metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🧪 Generation Metrics:**")
                st.write(f"• Total Generated: {summary.get('total_generated', 0)}")
                st.write(f"• Unique Structures: {summary.get('unique_structures', 0)}")
                st.write(f"• Average Temperature: {summary.get('average_temperature', 0):.2f}")
                st.write(f"• Functionalized Count: {summary.get('functionalized_count', 0)}")
                st.write(f"• Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
            
            with col2:
                # Simulation metrics (if available)
                if summary.get('simulation_enabled'):
                    st.markdown("**🔬 Simulation Metrics:**")
                    sim_results = results.get('simulation_results') or {}
                    st.write(f"• Polymer Boxes: {len([r for r in sim_results.get('polymer_boxes', []) if r.get('success')])}")
                    st.write(f"• Insulin Systems: {len([r for r in sim_results.get('insulin_systems', []) if r.get('success')])}")
                    st.write(f"• Simulation Success Rate: {summary.get('simulation_success_rate', 0)*100:.1f}%")
                    if sim_results.get('session_id'):
                        st.write(f"• Session ID: `{sim_results['session_id']}`")
                else:
                    st.markdown("**🚫 Simulation:**")
                    st.write("Automation disabled")
    
    # **NEW: Simulation Results Section**
    if results.get('simulation_results'):
        with st.expander("🔬 Simulation Box Results", expanded=True):
            sim_results = results.get('simulation_results') or {}
            
            # Summary metrics
            st.markdown("### Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", sim_results.get('total_candidates', 0))
            with col2:
                st.metric("Successful", sim_results.get('success_count', 0), 
                         delta=f"{(sim_results.get('success_count', 0)/max(1, sim_results.get('total_candidates', 1))*100):.1f}%")
            with col3:
                st.metric("Failed", sim_results.get('failed_count', 0))
            with col4:
                if sim_results.get('session_id'):
                    st.info(f"📁 Session: `{sim_results['session_id']}`")
            
            # Detailed results tabs
            if sim_results.get('polymer_boxes') or sim_results.get('insulin_systems'):
                sim_tab1, sim_tab2 = st.tabs(["🔧 Polymer Boxes", "🧬 Insulin Systems"])
                
                with sim_tab1:
                    polymer_results = sim_results.get('polymer_boxes', [])
                    if polymer_results:
                        for i, result in enumerate(polymer_results):
                            success_icon = "✅" if result.get('success') else "❌"
                            with st.expander(f"{success_icon} Candidate {i+1}: {result.get('candidate_id', 'Unknown')}", 
                                           expanded=(i < 2)):  # Expand first 2
                                if result.get('success'):
                                    st.write(f"**PDB File:** `{result.get('polymer_pdb', 'N/A')}`")
                                    st.write(f"**PSMILES:** `{result.get('psmiles', 'N/A')[:60]}...`")
                                    
                                    params = result.get('parameters', {})
                                    st.write(f"**Parameters:** Length={params.get('polymer_length', 'N/A')}, "
                                           f"Molecules={params.get('num_molecules', 'N/A')}, "
                                           f"Density={params.get('density', 'N/A')}")
                                else:
                                    st.error(f"Failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.info("No polymer box results available")
                
                with sim_tab2:
                    insulin_results = sim_results.get('insulin_systems', [])
                    if insulin_results:
                        for i, result in enumerate(insulin_results):
                            success_icon = "✅" if result.get('success') else "❌"
                            with st.expander(f"{success_icon} Candidate {i+1}: {result.get('candidate_id', 'Unknown')}", 
                                           expanded=(i < 2)):  # Expand first 2
                                if result.get('success'):
                                    st.write(f"**Composite PDB:** `{result.get('composite_pdb', 'N/A')}`")
                                    st.write(f"**Processed Insulin:** `{result.get('processed_insulin_pdb', 'N/A')}`")
                                    st.write(f"**Original Polymer:** `{result.get('original_polymer_pdb', 'N/A')}`")
                                    
                                    params = result.get('parameters', {})
                                    st.write(f"**Parameters:** Insulin molecules={params.get('num_insulin_molecules', 'N/A')}, "
                                           f"Polymer duplicates={params.get('num_polymer_duplicates', 'N/A')}, "
                                           f"Box size={params.get('box_size_nm', 'N/A')} nm")
                                else:
                                    st.error(f"Failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.info("No insulin system results available")


def render_material_generation_tab():
    """Render the simplified material generation tab with focus on visualization"""
    st.markdown("### AI-Powered Polymer Structure Generation")
    
    generation_mode = st.radio(
        "Generation Mode:",
        ["Automated Pipeline"],
        horizontal=True
    )
    
    # **Simplified pipeline health check**
    render_comprehensive_pipeline_health_check()
    
    if generation_mode == "Automated Pipeline":
        # **Enhanced generation options with simulation automation**
        (material_request, num_candidates, auto_functionalize, 
         auto_create_polymer_boxes, auto_create_insulin_systems,
         polymer_length, num_polymer_molecules, density, 
         tolerance_distance, timeout_minutes,
         num_insulin_molecules, box_size_nm) = render_advanced_generation_options()
        
        # **Simplified hybrid approach info**
        with st.expander("🔬 Generation Approach", expanded=False):
            st.markdown("""
            **Our system uses a 3-tier approach with automatic repair:**
            
            1. **🎯 Direct PSMILES Generation** - Creative structures from LLM
            2. **✅ Chemical Validation + Repair** - Multi-layer SMILES repair  
            3. **🧪 SMILES→PSMILES Conversion** - Generate SMILES, then `[*]SMILES[*]`
            
            **Auto-Repair Features:**
            - 🔧 Basic syntax fixes
            - 🧬 SELFIES format autocorrection  
            - 🛡️ Pattern-based molecular recognition
            """)
        
        # **Generation button and workflow**
        if st.button("🚀 Run Automated Pipeline", type="primary", disabled=not material_request):
            if not st.session_state.systems_initialized:
                st.error("⚠️ AI systems not initialized. Please restart the application.")
                return
            
            if not material_request.strip():
                st.warning("⚠️ Please provide a material description")
                return
            
            # **Start generation workflow with progress tracking**
            simulation_params = {
                'auto_create_polymer_boxes': auto_create_polymer_boxes,
                'auto_create_insulin_systems': auto_create_insulin_systems,
                'polymer_length': polymer_length,
                'num_polymer_molecules': num_polymer_molecules,
                'density': density,
                'tolerance_distance': tolerance_distance,
                'timeout_minutes': timeout_minutes,
                'num_insulin_molecules': num_insulin_molecules,
                'box_size_nm': box_size_nm
            }
            
            with st.spinner("Running automated PSMILES generation and simulation setup..."):
                results = execute_automated_pipeline_with_progress(
                    material_request, num_candidates, auto_functionalize, simulation_params)
                
                if results:
                    render_enhanced_generation_results(results)
                else:
                    st.error("❌ Generation failed. Please try again.")


def render_insulin_embedding_tab():
    """Render insulin embedding tab (placeholder for now)"""
    st.markdown("### 🧬 Insulin Embedding in Polymer Matrix")
    st.info("🚧 This feature will be enhanced to match the comprehensive embedding workflow from the original app.")
    
    # Basic placeholder implementation
    st.markdown("#### Embedding Options")
    
    polymer_source = st.radio(
        "Polymer Source:",
        ["Use Generated Structure", "Upload PDB File"],
        help="Choose the polymer structure for insulin embedding"
    )
    
    if polymer_source == "Upload PDB File":
        uploaded_file = st.file_uploader("Upload Polymer PDB File", type=['pdb'])
        if uploaded_file:
            st.success("✅ PDB file uploaded successfully!")
    
    # Embedding parameters
    col1, col2 = st.columns(2)
    with col1:
        num_insulin_molecules = st.number_input("Number of insulin molecules:", min_value=1, max_value=20, value=5)
        buffer_distance = st.number_input("Buffer distance (Å):", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    
    with col2:
        max_atoms = st.number_input("Maximum atoms:", min_value=1000, max_value=100000, value=50000, step=1000)
        manual_box_size = st.number_input("Box size (nm):", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    
    if st.button("🧬 Embed Insulin in Polymer", type="primary"):
        st.info("🚧 Embedding functionality will be implemented to match the original app's comprehensive workflow.")


# **NEW: Add SMILES enhancement for existing candidates**
def enhance_existing_candidates_with_smiles():
    """Enhance existing candidates in session state with SMILES data if missing"""
    if 'psmiles_candidates' in st.session_state and st.session_state.psmiles_candidates:
        # Check if any candidates are missing SMILES data
        candidates_needing_enhancement = [
            c for c in st.session_state.psmiles_candidates 
            if not c.get('smiles_conversion_success') and not c.get('smiles')
        ]
        
        if candidates_needing_enhancement:
            st.info(f"🔄 Enhancing {len(candidates_needing_enhancement)} existing candidates with SMILES data...")
            update_session_candidates_with_smiles()
            st.success("✅ Existing candidates enhanced with SMILES data!")


def render_psmiles_generation():
    """
    Render the complete enhanced PSMILES generation page
    
    This includes material generation and insulin embedding workflows with
    comprehensive functionality matching the original app.
    """
    st.subheader("🧪 PSMILES Generation with Interactive Workflow")
    
    if not st.session_state.systems_initialized:
        st.error("⚠️ AI systems not initialized. Please restart the application.")
        st.stop()
    
    # Initialize workflow state
    if 'psmiles_workflow_active' not in st.session_state:
        st.session_state.psmiles_workflow_active = False
    if 'current_psmiles' not in st.session_state:
        st.session_state.current_psmiles = None
    if 'svg_content' not in st.session_state:
        st.session_state.svg_content = None
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # Enhance existing candidates with SMILES data
    enhance_existing_candidates_with_smiles()

    # **ENHANCED**: Main tabs with enhanced functionality
    tab1, tab2 = st.tabs(["Material Generation", "Insulin Embedding"])
    
    with tab1:
        render_material_generation_tab()
    
    with tab2:
        render_insulin_embedding_tab()
    
    # **ENHANCED**: Instant corrections section (if available)
    if AUTOCORRECTOR_AVAILABLE and st.session_state.psmiles_candidates:
        st.markdown("---")
        st.subheader("🔧 Instant PSMILES Corrections")
        
        # Get failed PSMILES for correction
        failed_candidates = [
            candidate for candidate in st.session_state.psmiles_candidates 
            if not candidate.get('is_valid', True)
        ]
        
        if failed_candidates:
            st.warning(f"Found {len(failed_candidates)} invalid PSMILES that can be corrected")
            
            if st.button("🛠️ Apply Instant Corrections"):
                processor = safe_get_session_object('psmiles_processor')
                corrected = apply_instant_corrections_ui([c.get('psmiles') for c in failed_candidates], processor)
                
                if corrected:
                    st.success(f"✅ Applied corrections to {len(corrected)} PSMILES")
        else:
            st.info("✅ All generated PSMILES are valid - no corrections needed") 