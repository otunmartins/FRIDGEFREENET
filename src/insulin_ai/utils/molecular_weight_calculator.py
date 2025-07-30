"""
Molecular Weight Calculator and Density-Based Box Size Utility

This module provides functionality to:
1. Calculate molecular weights from SMILES strings
2. Compute simulation box sizes based on density and system composition
3. Validate density ranges and provide recommendations

Author: AI-Driven Material Discovery Team
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class SystemComposition:
    """Data class to hold system composition information"""
    polymer_chains: int
    polymer_mw: float  # Da (Daltons)
    insulin_molecules: int
    insulin_mw: float = 5808.0  # Da (human insulin)
    additional_mass: float = 0.0  # Da (water, ions, etc.)
    
    @property
    def total_mass_da(self) -> float:
        """Calculate total system mass in Daltons"""
        return (self.polymer_chains * self.polymer_mw + 
                self.insulin_molecules * self.insulin_mw + 
                self.additional_mass)
    
    @property
    def total_mass_grams(self) -> float:
        """Calculate total system mass in grams"""
        # 1 Da = 1.66054e-24 grams
        return self.total_mass_da * 1.66054e-24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'polymer_chains': self.polymer_chains,
            'polymer_mw': self.polymer_mw,
            'insulin_molecules': self.insulin_molecules,
            'insulin_mw': self.insulin_mw,
            'additional_mass': self.additional_mass,
            'total_mass_da': self.total_mass_da,
            'total_mass_grams': self.total_mass_grams
        }


@dataclass
class BoxSizeResult:
    """Result of box size calculation"""
    success: bool
    cubic_box_size_nm: float
    volume_nm3: float
    calculated_density: float
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'success': self.success,
            'cubic_box_size_nm': self.cubic_box_size_nm,
            'volume_nm3': self.volume_nm3,
            'calculated_density': self.calculated_density,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class MolecularWeightCalculator:
    """Calculator for molecular weights and box sizes"""
    
    # Standard density ranges for different materials (g/cm³)
    DENSITY_RANGES = {
        'hydrogel': (1.0, 1.3),
        'protein': (1.2, 1.4),
        'polymer_solution': (0.9, 1.2),
        'dense_polymer': (1.1, 1.5),
        'aqueous_system': (0.99, 1.05)
    }
    
    # Minimum box size for MD simulations (nm)
    MIN_BOX_SIZE_NM = 3.0  # At least 2x typical cutoff (1.2-1.4 nm)
    
    def __init__(self):
        """Initialize the calculator"""
        self.rdkit_available = self._check_rdkit_availability()
    
    def _check_rdkit_availability(self) -> bool:
        """Check if RDKit is available for SMILES parsing"""
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            return True
        except ImportError:
            print("⚠️ RDKit not available - molecular weight calculation from SMILES will be limited")
            return False
    
    def calculate_mw_from_smiles(self, smiles: str) -> Dict[str, Any]:
        """
        Calculate molecular weight from SMILES string
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary with molecular weight information
        """
        result = {
            'success': False,
            'molecular_weight': 0.0,
            'formula': '',
            'error': None,
            'method': 'unknown'
        }
        
        if not self.rdkit_available:
            result['error'] = "RDKit not available for SMILES parsing"
            result['molecular_weight'] = self._estimate_mw_from_smiles_simple(smiles)
            result['method'] = 'simple_estimation'
            result['success'] = result['molecular_weight'] > 0
            return result
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['error'] = f"Invalid SMILES string: {smiles}"
                return result
            
            # Calculate molecular weight
            mw = Descriptors.MolWt(mol)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            
            result.update({
                'success': True,
                'molecular_weight': mw,
                'formula': formula,
                'method': 'rdkit_exact'
            })
            
        except Exception as e:
            result['error'] = f"RDKit calculation failed: {str(e)}"
            # Fallback to simple estimation
            result['molecular_weight'] = self._estimate_mw_from_smiles_simple(smiles)
            result['method'] = 'simple_estimation_fallback'
            result['success'] = result['molecular_weight'] > 0
        
        return result
    
    def _estimate_mw_from_smiles_simple(self, smiles: str) -> float:
        """
        Simple molecular weight estimation without RDKit
        Based on counting common atoms in SMILES notation
        """
        if not smiles:
            return 0.0
        
        # Atomic weights (Da)
        atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.06,
            'H': 1.008, 'P': 30.97, 'F': 19.00, 'Cl': 35.45,
            'Br': 79.90, 'I': 126.90
        }
        
        # Simple counting approach
        total_mw = 0.0
        
        # Count explicit atoms (very basic - doesn't handle complex SMILES)
        for atom, weight in atomic_weights.items():
            if atom in smiles:
                # Very rough count - this is a fallback method
                count = smiles.count(atom)
                total_mw += count * weight
        
        # Add some hydrogen estimation (very rough)
        carbon_count = smiles.count('C')
        estimated_hydrogens = carbon_count * 2  # Rough estimate
        total_mw += estimated_hydrogens * atomic_weights['H']
        
        return max(total_mw, 100.0)  # Minimum reasonable MW
    
    def calculate_box_size_from_density(self, 
                                      composition: SystemComposition,
                                      target_density: float,
                                      material_type: str = 'hydrogel') -> BoxSizeResult:
        """
        Calculate cubic box size from density and system composition
        
        Args:
            composition: System composition information
            target_density: Target density in g/cm³
            material_type: Type of material for validation
            
        Returns:
            BoxSizeResult with calculation results
        """
        warnings = []
        recommendations = []
        
        # Validate inputs
        if target_density <= 0:
            return BoxSizeResult(
                success=False,
                cubic_box_size_nm=0.0,
                volume_nm3=0.0,
                calculated_density=0.0,
                warnings=["Invalid density: must be positive"],
                recommendations=["Please provide a positive density value"]
            )
        
        if composition.total_mass_da <= 0:
            return BoxSizeResult(
                success=False,
                cubic_box_size_nm=0.0,
                volume_nm3=0.0,
                calculated_density=0.0,
                warnings=["Invalid system composition: total mass is zero"],
                recommendations=["Check polymer and insulin counts and molecular weights"]
            )
        
        # Validate density range
        if material_type in self.DENSITY_RANGES:
            min_density, max_density = self.DENSITY_RANGES[material_type]
            if target_density < min_density:
                warnings.append(f"Density {target_density:.3f} g/cm³ is below typical range for {material_type} ({min_density}-{max_density} g/cm³)")
            elif target_density > max_density:
                warnings.append(f"Density {target_density:.3f} g/cm³ is above typical range for {material_type} ({min_density}-{max_density} g/cm³)")
        
        # Calculate volume
        # Volume = Mass / Density
        mass_grams = composition.total_mass_grams
        volume_cm3 = mass_grams / target_density
        
        # Convert to nm³ (1 cm³ = 1e21 nm³)
        volume_nm3 = volume_cm3 * 1e21
        
        # Calculate cubic box size
        cubic_box_size_nm = volume_nm3 ** (1/3)
        
        # Validate box size
        if cubic_box_size_nm < self.MIN_BOX_SIZE_NM:
            warnings.append(f"Calculated box size ({cubic_box_size_nm:.2f} nm) is smaller than recommended minimum ({self.MIN_BOX_SIZE_NM} nm)")
            recommendations.append("Consider increasing system size or decreasing density")
        
        # Calculate actual density for verification
        calculated_density = mass_grams / volume_cm3
        
        # Generate recommendations
        if not warnings:
            recommendations.append("Box size calculation looks reasonable")
        
        recommendations.append(f"System contains {composition.polymer_chains} polymer chains and {composition.insulin_molecules} insulin molecules")
        recommendations.append(f"Total system mass: {composition.total_mass_da:.0f} Da ({mass_grams:.2e} g)")
        
        return BoxSizeResult(
            success=True,
            cubic_box_size_nm=cubic_box_size_nm,
            volume_nm3=volume_nm3,
            calculated_density=calculated_density,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def get_density_recommendations(self, material_type: str) -> Dict[str, Any]:
        """Get density recommendations for a material type"""
        
        recommendations = {
            'material_type': material_type,
            'recommended_range': self.DENSITY_RANGES.get(material_type, (1.0, 1.3)),
            'typical_value': 1.1,
            'description': ''
        }
        
        descriptions = {
            'hydrogel': 'Hydrogels typically have densities close to water (1.0 g/cm³) but slightly higher due to polymer content',
            'protein': 'Proteins typically have densities of 1.2-1.4 g/cm³',
            'polymer_solution': 'Polymer solutions vary widely but are often close to water density',
            'dense_polymer': 'Dense polymer materials can have higher densities',
            'aqueous_system': 'Aqueous systems are close to water density (1.0 g/cm³)'
        }
        
        recommendations['description'] = descriptions.get(material_type, 'Density range depends on material composition')
        
        min_density, max_density = recommendations['recommended_range']
        recommendations['typical_value'] = (min_density + max_density) / 2
        
        return recommendations


def test_molecular_weight_calculator():
    """Test function for the molecular weight calculator"""
    
    print("🧪 Testing Molecular Weight Calculator...")
    
    calculator = MolecularWeightCalculator()
    
    # Test 1: Simple SMILES molecular weight
    print("\n1. Testing SMILES molecular weight calculation:")
    test_smiles = "CCO"  # Ethanol
    mw_result = calculator.calculate_mw_from_smiles(test_smiles)
    print(f"   SMILES: {test_smiles}")
    print(f"   Result: {mw_result}")
    
    # Test 2: Box size calculation
    print("\n2. Testing box size calculation:")
    composition = SystemComposition(
        polymer_chains=10,
        polymer_mw=1000.0,  # 1 kDa polymer
        insulin_molecules=5,
        insulin_mw=5808.0
    )
    
    target_density = 1.1  # g/cm³
    
    box_result = calculator.calculate_box_size_from_density(
        composition=composition,
        target_density=target_density,
        material_type='hydrogel'
    )
    
    print(f"   System: {composition.polymer_chains} polymers ({composition.polymer_mw} Da) + {composition.insulin_molecules} insulin")
    print(f"   Target density: {target_density} g/cm³")
    print(f"   Result: {box_result.to_dict()}")
    
    # Test 3: Density recommendations
    print("\n3. Testing density recommendations:")
    recommendations = calculator.get_density_recommendations('hydrogel')
    print(f"   Hydrogel recommendations: {recommendations}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_molecular_weight_calculator() 