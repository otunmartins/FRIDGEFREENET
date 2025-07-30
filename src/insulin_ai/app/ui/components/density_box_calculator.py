"""
Density-Based Box Size Calculator UI Component

This component provides an enhanced interface for calculating simulation box sizes
based on molecular weight, system composition, and target density.

Author: AI-Driven Material Discovery Team
"""

import streamlit as st
from typing import Dict, Any, Tuple, Optional
from utils.molecular_weight_calculator import (
    MolecularWeightCalculator, 
    SystemComposition, 
    BoxSizeResult
)


class DensityBoxCalculatorUI:
    """UI component for density-based box size calculation"""
    
    def __init__(self):
        """Initialize the calculator UI"""
        self.calculator = MolecularWeightCalculator()
        
        # Initialize session state keys
        if 'box_calculation_mode' not in st.session_state:
            st.session_state.box_calculation_mode = 'density_based'
        if 'calculated_box_size' not in st.session_state:
            st.session_state.calculated_box_size = None
    
    def render_enhanced_simulation_parameters(self, 
                                           auto_create_polymer_boxes: bool = False,
                                           auto_create_insulin_systems: bool = False,
                                           default_polymer_length: int = 5,
                                           default_num_molecules: int = 8,
                                           default_density: float = 0.3,
                                           default_tolerance: float = 3.5,
                                           default_timeout: int = 15,
                                           default_insulin_molecules: int = 1,
                                           default_box_size: float = 3.0) -> Dict[str, Any]:
        """
        Render enhanced simulation parameters with density-based box calculation
        
        Returns:
            Dictionary containing all simulation parameters
        """
        
        if not (auto_create_polymer_boxes or auto_create_insulin_systems):
            # Return defaults if automation is disabled
            return self._get_default_parameters(
                default_polymer_length, default_num_molecules, default_density,
                default_tolerance, default_timeout, default_insulin_molecules, default_box_size
            )
        
        st.info("💡 **Enhanced Parameters**: Choose between manual density or automatic box calculation")
        
        # Box calculation mode selection
        calculation_mode = st.radio(
            "📐 Box Size Calculation Method:",
            options=['density_based', 'manual_override'],
            format_func=lambda x: {
                'density_based': '🧮 Density-Based Calculation (Recommended)',
                'manual_override': '✋ Manual Override'
            }[x],
            help="Choose how to determine simulation box size",
            horizontal=True,
            key='box_calculation_mode'
        )
        
        # Render appropriate interface based on mode
        if calculation_mode == 'density_based':
            return self._render_density_based_interface(
                auto_create_insulin_systems, default_polymer_length, 
                default_num_molecules, default_tolerance, default_timeout, 
                default_insulin_molecules
            )
        else:
            return self._render_manual_override_interface(
                auto_create_insulin_systems, default_polymer_length,
                default_num_molecules, default_density, default_tolerance, 
                default_timeout, default_insulin_molecules, default_box_size
            )
    
    def _render_density_based_interface(self, 
                                      auto_create_insulin_systems: bool,
                                      default_polymer_length: int,
                                      default_num_molecules: int,
                                      default_tolerance: float,
                                      default_timeout: int,
                                      default_insulin_molecules: int) -> Dict[str, Any]:
        """Render density-based calculation interface"""
        
        st.markdown("#### 🧮 Density-Based Box Size Calculation")
        
        # System composition inputs
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("**System Composition:**")
            
            polymer_length = st.slider(
                "Polymer chain length:", 3, 50, default_polymer_length,
                help="Number of monomers per polymer chain"
            )
            
            num_polymer_molecules = st.slider(
                "Number of polymer molecules:", 5, 100, default_num_molecules,
                help="Total polymer chains in the system"
            )
            
            if auto_create_insulin_systems:
                num_insulin_molecules = st.slider(
                    "Number of insulin molecules:", 1, 10, default_insulin_molecules
                )
            else:
                num_insulin_molecules = default_insulin_molecules
        
        with comp_col2:
            st.markdown("**Material Properties:**")
            
            # Material type selection for density recommendations
            material_type = st.selectbox(
                "Material Type:",
                options=['hydrogel', 'polymer_solution', 'dense_polymer', 'aqueous_system'],
                format_func=lambda x: {
                    'hydrogel': '🌊 Hydrogel (1.0-1.3 g/cm³)',
                    'polymer_solution': '💧 Polymer Solution (0.9-1.2 g/cm³)',
                    'dense_polymer': '🧱 Dense Polymer (1.1-1.5 g/cm³)',
                    'aqueous_system': '💦 Aqueous System (0.99-1.05 g/cm³)'
                }[x],
                help="Select material type for density recommendations"
            )
            
            # Get density recommendations
            density_recs = self.calculator.get_density_recommendations(material_type)
            min_density, max_density = density_recs['recommended_range']
            typical_density = density_recs['typical_value']
            
            target_density = st.slider(
                "Target Density (g/cm³):",
                min_value=min_density - 0.2,
                max_value=max_density + 0.2,
                value=typical_density,
                step=0.05,
                help=f"Typical range for {material_type}: {min_density}-{max_density} g/cm³"
            )
            
            st.caption(f"📊 {density_recs['description']}")
        
        # Molecular weight estimation section
        st.markdown("#### 🧬 Molecular Weight Estimation")
        
        mw_col1, mw_col2 = st.columns(2)
        
        with mw_col1:
            # Polymer MW estimation options
            mw_estimation_method = st.radio(
                "Polymer MW Estimation:",
                options=['auto_detect', 'manual_input'],
                format_func=lambda x: {
                    'auto_detect': '🤖 Auto-detect from SMILES',
                    'manual_input': '✍️ Manual Input'
                }[x],
                help="How to estimate polymer molecular weight"
            )
            
            if mw_estimation_method == 'auto_detect':
                # Try to get SMILES from session state or user input
                polymer_smiles = st.text_input(
                    "Polymer SMILES (optional):",
                    placeholder="Enter SMILES to calculate exact MW",
                    help="Leave empty to use estimated values"
                )
                
                if polymer_smiles:
                    mw_result = self.calculator.calculate_mw_from_smiles(polymer_smiles)
                    if mw_result['success']:
                        monomer_mw = mw_result['molecular_weight']
                        st.success(f"✅ Detected MW: {monomer_mw:.1f} Da/monomer ({mw_result['formula']})")
                    else:
                        st.warning(f"⚠️ MW detection failed: {mw_result.get('error', 'Unknown error')}")
                        monomer_mw = st.number_input("Monomer MW (Da):", value=100.0, min_value=50.0, max_value=1000.0)
                else:
                    monomer_mw = st.number_input("Estimated Monomer MW (Da):", value=100.0, min_value=50.0, max_value=1000.0)
            else:
                monomer_mw = st.number_input("Monomer MW (Da):", value=100.0, min_value=50.0, max_value=1000.0)
            
            # Calculate total polymer MW
            polymer_mw = monomer_mw * polymer_length
            st.metric("Total Polymer MW", f"{polymer_mw:.0f} Da")
        
        with mw_col2:
            # System composition summary
            st.markdown("**System Summary:**")
            
            composition = SystemComposition(
                polymer_chains=num_polymer_molecules,
                polymer_mw=polymer_mw,
                insulin_molecules=num_insulin_molecules,
                insulin_mw=5808.0,  # Standard insulin MW
                additional_mass=0.0
            )
            
            st.metric("Total System Mass", f"{composition.total_mass_da:.0f} Da")
            st.metric("Mass in Grams", f"{composition.total_mass_grams:.2e} g")
        
        # Calculate box size
        st.markdown("#### 📐 Box Size Calculation")
        
        box_result = self.calculator.calculate_box_size_from_density(
            composition=composition,
            target_density=target_density,
            material_type=material_type
        )
        
        if box_result.success:
            # Display results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric("Calculated Box Size", f"{box_result.cubic_box_size_nm:.2f} nm")
                st.metric("System Volume", f"{box_result.volume_nm3:.1f} nm³")
            
            with result_col2:
                st.metric("Verified Density", f"{box_result.calculated_density:.3f} g/cm³")
                
                # Show density comparison
                if abs(box_result.calculated_density - target_density) < 0.001:
                    st.success("✅ Density matches target")
                else:
                    st.info(f"ℹ️ Slight difference due to rounding")
            
            with result_col3:
                # Validation indicators
                if box_result.warnings:
                    for warning in box_result.warnings:
                        st.warning(f"⚠️ {warning}")
                else:
                    st.success("✅ All validations passed")
            
            # Recommendations
            if box_result.recommendations:
                with st.expander("💡 Recommendations", expanded=False):
                    for rec in box_result.recommendations:
                        st.info(f"• {rec}")
            
            # Store calculated box size for use
            st.session_state.calculated_box_size = box_result.cubic_box_size_nm
            calculated_box_size = box_result.cubic_box_size_nm
            
        else:
            st.error("❌ Box size calculation failed")
            for warning in box_result.warnings:
                st.error(f"• {warning}")
            calculated_box_size = 3.0  # Fallback
        
        # Additional parameters
        st.markdown("#### ⚙️ Additional Parameters")
        
        add_col1, add_col2 = st.columns(2)
        
        with add_col1:
            tolerance_distance = st.slider(
                "Tolerance distance (Å):", 2.0, 5.0, default_tolerance, step=0.1,
                help="Minimum distance between molecules during packing"
            )
        
        with add_col2:
            timeout_minutes = st.slider(
                "Timeout (minutes):", 1, 60, default_timeout,
                help="Maximum time to spend building each box (up to 1 hour)"
            )
        
        return {
            'polymer_length': polymer_length,
            'num_polymer_molecules': num_polymer_molecules,
            'density': target_density,
            'tolerance_distance': tolerance_distance,
            'timeout_minutes': timeout_minutes,
            'num_insulin_molecules': num_insulin_molecules,
            'box_size_nm': calculated_box_size,
            'calculation_method': 'density_based',
            'molecular_weight': polymer_mw,
            'system_composition': composition.to_dict() if hasattr(composition, 'to_dict') else None,
            'box_calculation_result': box_result.to_dict()
        }
    
    def _render_manual_override_interface(self,
                                        auto_create_insulin_systems: bool,
                                        default_polymer_length: int,
                                        default_num_molecules: int,
                                        default_density: float,
                                        default_tolerance: float,
                                        default_timeout: int,
                                        default_insulin_molecules: int,
                                        default_box_size: float) -> Dict[str, Any]:
        """Render manual override interface (original behavior)"""
        
        st.markdown("#### ✋ Manual Parameter Override")
        st.info("Using manual parameter specification (original behavior)")
        
        # Standard parameters
        col1, col2 = st.columns(2)
        
        with col1:
            polymer_length = st.slider(
                "Polymer chain length:", 3, 50, default_polymer_length,
                help="Shorter chains pack faster"
            )
            
            num_polymer_molecules = st.slider(
                "Number of polymer molecules:", 5, 100, default_num_molecules,
                help="Fewer molecules pack faster"
            )
            
            density = st.slider(
                "Packing density (g/cm³):", 0.2, 1.5, default_density, step=0.1,
                help="Lower density packs faster"
            )
        
        with col2:
            tolerance_distance = st.slider(
                "Tolerance distance (Å):", 2.0, 5.0, default_tolerance, step=0.1,
                help="Higher tolerance packs faster"
            )
            
            timeout_minutes = st.slider(
                "Timeout (minutes):", 1, 60, default_timeout,
                help="Maximum time to spend building each box (up to 1 hour)"
            )
            
            if auto_create_insulin_systems:
                num_insulin_molecules = st.slider(
                    "Number of insulin molecules:", 1, 10, default_insulin_molecules
                )
                box_size_nm = st.slider(
                    "Box size (nm):", 1.0, 10.0, default_box_size, step=0.5
                )
            else:
                num_insulin_molecules = default_insulin_molecules
                box_size_nm = default_box_size
        
        return {
            'polymer_length': polymer_length,
            'num_polymer_molecules': num_polymer_molecules,
            'density': density,
            'tolerance_distance': tolerance_distance,
            'timeout_minutes': timeout_minutes,
            'num_insulin_molecules': num_insulin_molecules,
            'box_size_nm': box_size_nm,
            'calculation_method': 'manual_override'
        }
    
    def _get_default_parameters(self,
                              default_polymer_length: int,
                              default_num_molecules: int,
                              default_density: float,
                              default_tolerance: float,
                              default_timeout: int,
                              default_insulin_molecules: int,
                              default_box_size: float) -> Dict[str, Any]:
        """Return default parameters when automation is disabled"""
        
        return {
            'polymer_length': default_polymer_length,
            'num_polymer_molecules': default_num_molecules,
            'density': default_density,
            'tolerance_distance': default_tolerance,
            'timeout_minutes': default_timeout,
            'num_insulin_molecules': default_insulin_molecules,
            'box_size_nm': default_box_size,
            'calculation_method': 'disabled'
        }


# Helper functions to add to existing PSMILES generation UI
def extend_simulation_composition_to_dict(composition):
    """Add to_dict method to SystemComposition if it doesn't exist"""
    if hasattr(composition, 'to_dict'):
        return composition.to_dict()
    
    # Fallback manual conversion
    return {
        'polymer_chains': composition.polymer_chains,
        'polymer_mw': composition.polymer_mw,
        'insulin_molecules': composition.insulin_molecules,
        'insulin_mw': composition.insulin_mw,
        'additional_mass': composition.additional_mass,
        'total_mass_da': composition.total_mass_da,
        'total_mass_grams': composition.total_mass_grams
    }


def test_density_box_calculator_ui():
    """Test function for the density box calculator UI component"""
    
    st.set_page_config(page_title="Density Box Calculator Test", layout="wide")
    
    st.title("🧪 Density-Based Box Calculator UI Test")
    
    # Initialize the UI component
    calculator_ui = DensityBoxCalculatorUI()
    
    # Test the interface
    st.markdown("## Test Interface")
    
    parameters = calculator_ui.render_enhanced_simulation_parameters(
        auto_create_polymer_boxes=True,
        auto_create_insulin_systems=True,
        default_polymer_length=5,
        default_num_molecules=8,
        default_density=0.3,
        default_tolerance=3.5,
        default_timeout=15,
        default_insulin_molecules=1,
        default_box_size=3.0
    )
    
    # Display results
    st.markdown("## Results")
    st.json(parameters)
    
    st.success("✅ Density-based box calculator UI test completed!")


if __name__ == "__main__":
    test_density_box_calculator_ui() 