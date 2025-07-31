# Core infrastructure confirms rules are active
"""
Property Scoring System for Drug Delivery Materials

This module converts MD simulation-computed properties into meaningful scores for drug delivery
applications, specifically focusing on biocompatibility, degradation rate, and mechanical strength.

Based on literature research on MD simulations for:
- Biocompatibility prediction through molecular interactions and surface properties
- Degradation rate estimation from polymer chain scission and molecular mobility  
- Mechanical strength from elastic moduli and intermolecular forces
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MDSimulationProperties:
    """Raw properties computed from MD simulation"""
    
    # Mechanical properties (GPa)
    youngs_modulus_x: float
    youngs_modulus_y: float  
    youngs_modulus_z: float
    bulk_modulus: float
    shear_modulus: float
    
    # Thermodynamic properties (kJ/mol)
    cohesive_energy: float
    mixing_energy: float
    vaporization_enthalpy: float
    
    # Dynamic properties
    glass_transition_temp: float  # K
    density: float  # g/cm³
    diffusion_coefficient_water: float  # cm²/s
    diffusion_coefficient_drug: float  # cm²/s
    
    # Structural properties
    hydrogen_bond_count: float
    hydrogen_bond_lifetime: float  # ps
    surface_area: float  # Ų
    free_volume_fraction: float
    
    # Degradation-related properties
    ester_bond_count: float
    ester_bond_strength: float  # kJ/mol
    water_accessibility: float  # fraction
    chain_scission_rate: float  # 1/ns
    
    # Molecular mobility
    rmsf_polymer: float  # Å
    rmsf_drug: float  # Å
    rotational_correlation_time: float  # ps


@dataclass 
class TargetPropertyScores:
    """Target property scores for drug delivery (0-1 scale)"""
    biocompatibility: float
    degradation_rate: float
    mechanical_strength: float
    overall_score: float


class PropertyScoring:
    """
    Converts MD simulation properties to drug delivery performance scores
    
    Based on literature correlations between:
    - Biocompatibility and molecular interactions, surface properties, Tg
    - Degradation rate and polymer chain scission, water diffusion, bond stability
    - Mechanical strength and elastic moduli, intermolecular forces
    """
    
    def __init__(self):
        """Initialize scoring parameters based on literature data"""
        self.setup_scoring_parameters()
        logger.info("PropertyScoring initialized with literature-based parameters")
    
    def setup_scoring_parameters(self):
        """Setup scoring parameters based on literature research"""
        
        # Biocompatibility scoring weights (based on polymer-protein interaction studies)
        self.biocomp_weights = {
            'glass_transition': 0.25,  # Tg close to body temp (37°C) is optimal
            'hydrogen_bonding': 0.20,  # Moderate H-bonding for stability without toxicity
            'surface_energy': 0.15,    # Low surface energy reduces inflammatory response
            'density': 0.15,           # Optimal density for tissue compatibility
            'water_interaction': 0.15, # Good water compatibility
            'free_volume': 0.10        # Porosity for drug release
        }
        
        # Degradation rate scoring weights (based on biodegradable polymer studies)
        self.degrd_weights = {
            'chain_scission': 0.30,    # Primary degradation mechanism
            'water_diffusion': 0.25,   # Water access drives hydrolysis
            'ester_stability': 0.20,   # Bond strength determines degradation speed
            'mobility': 0.15,          # Molecular mobility affects reaction rates
            'crystallinity': 0.10      # Amorphous regions degrade faster
        }
        
        # Mechanical strength scoring weights (based on polymer mechanics studies)
        self.mech_weights = {
            'elastic_moduli': 0.35,    # Primary mechanical indicator
            'cohesive_energy': 0.25,   # Intermolecular forces
            'density': 0.20,           # Packing efficiency
            'crystallinity': 0.20      # Crystal content increases strength
        }
        
        # Reference values for normalization (from literature)
        self.reference_values = {
            # Biocompatibility references (based on FDA-approved polymers)
            'optimal_tg': 310.0,       # 37°C in Kelvin
            'safe_hbond_density': 2.5,  # H-bonds per nm³
            'biocompat_density_range': (1.0, 1.4),  # g/cm³
            
            # Degradation references (based on PLA/PLGA studies)
            'target_degrd_time': 30,   # days for insulin delivery
            'water_diffusion_fast': 1e-8,  # cm²/s for fast degradation
            'ester_bond_weak': 300,    # kJ/mol for biodegradable bonds
            
            # Mechanical references (based on biomedical applications)
            'min_youngs_modulus': 0.1, # GPa minimum strength
            'max_youngs_modulus': 5.0, # GPa maximum before brittleness
            'min_cohesive_energy': 50, # kJ/mol
        }
    
    def calculate_biocompatibility_score(self, props: MDSimulationProperties) -> float:
        """
        Calculate biocompatibility score based on molecular interactions and surface properties
        
        Args:
            props: MD simulation properties
            
        Returns:
            Biocompatibility score (0-1, higher is better)
        """
        
        # Glass transition score (optimal around body temperature)
        tg_diff = abs(props.glass_transition_temp - self.reference_values['optimal_tg'])
        tg_score = max(0, 1 - tg_diff / 50)  # Penalize deviation from 37°C
        
        # Hydrogen bonding score (moderate bonding is optimal)
        hb_density = props.hydrogen_bond_count / props.surface_area if props.surface_area > 0 else 0
        target_hb = self.reference_values['safe_hbond_density']
        hb_score = 1 - abs(hb_density - target_hb) / target_hb
        hb_score = max(0, min(1, hb_score))
        
        # Density score (within biocompatible range)
        dens_min, dens_max = self.reference_values['biocompat_density_range']
        if dens_min <= props.density <= dens_max:
            dens_score = 1.0
        else:
            dens_score = max(0, 1 - abs(props.density - (dens_min + dens_max)/2) / dens_max)
        
        # Water interaction score (good water compatibility)
        water_score = min(1, props.diffusion_coefficient_water / 1e-9)  # Normalize to reasonable range
        
        # Surface accessibility score
        surface_score = min(1, props.water_accessibility)
        
        # Free volume score (porosity for drug release)
        free_vol_score = min(1, props.free_volume_fraction / 0.3)  # 30% is high porosity
        
        # Weighted combination
        biocomp_score = (
            self.biocomp_weights['glass_transition'] * tg_score +
            self.biocomp_weights['hydrogen_bonding'] * hb_score +
            self.biocomp_weights['surface_energy'] * dens_score +
            self.biocomp_weights['density'] * dens_score +
            self.biocomp_weights['water_interaction'] * water_score +
            self.biocomp_weights['free_volume'] * free_vol_score
        )
        
        return min(1.0, max(0.0, biocomp_score))
    
    def calculate_degradation_rate_score(self, props: MDSimulationProperties, target_time_days: float = 30) -> float:
        """
        Calculate degradation rate score based on polymer chain scission and molecular mobility
        
        Args:
            props: MD simulation properties
            target_time_days: Target degradation time in days
            
        Returns:
            Degradation rate score (0-1, higher means closer to target rate)
        """
        
        # Chain scission rate score (primary degradation mechanism)
        scission_score = min(1, props.chain_scission_rate / 0.1)  # Normalize to 0.1 ns⁻¹
        
        # Water diffusion score (water access drives hydrolysis)
        target_water_diff = self.reference_values['water_diffusion_fast']
        water_diff_score = min(1, props.diffusion_coefficient_water / target_water_diff)
        
        # Ester bond stability score (weaker bonds degrade faster)
        weak_bond_energy = self.reference_values['ester_bond_weak']
        if props.ester_bond_strength > 0:
            bond_score = max(0, 1 - (props.ester_bond_strength - weak_bond_energy) / weak_bond_energy)
        else:
            bond_score = 0
        
        # Molecular mobility score (higher mobility increases reaction rates)
        mobility_score = min(1, (props.rmsf_polymer + props.rmsf_drug) / 2.0)  # Normalize to 2 Å
        
        # Crystallinity score (amorphous regions degrade faster)
        # Higher free volume indicates more amorphous character
        crystal_score = props.free_volume_fraction
        
        # Weighted combination
        degrd_score = (
            self.degrd_weights['chain_scission'] * scission_score +
            self.degrd_weights['water_diffusion'] * water_diff_score +
            self.degrd_weights['ester_stability'] * bond_score +
            self.degrd_weights['mobility'] * mobility_score +
            self.degrd_weights['crystallinity'] * crystal_score
        )
        
        return min(1.0, max(0.0, degrd_score))
    
    def calculate_mechanical_strength_score(self, props: MDSimulationProperties) -> float:
        """
        Calculate mechanical strength score based on elastic moduli and intermolecular forces
        
        Args:
            props: MD simulation properties
            
        Returns:
            Mechanical strength score (0-1, higher is better)
        """
        
        # Elastic moduli score (primary mechanical indicator)
        avg_youngs = (props.youngs_modulus_x + props.youngs_modulus_y + props.youngs_modulus_z) / 3
        min_e = self.reference_values['min_youngs_modulus']
        max_e = self.reference_values['max_youngs_modulus']
        
        if avg_youngs < min_e:
            moduli_score = avg_youngs / min_e  # Linear increase to minimum
        elif avg_youngs > max_e:
            moduli_score = max(0.5, 1 - (avg_youngs - max_e) / max_e)  # Penalize brittleness
        else:
            moduli_score = 1.0  # In optimal range
        
        # Bulk and shear moduli contribution
        bulk_score = min(1, props.bulk_modulus / 3.0)  # Normalize to 3 GPa
        shear_score = min(1, props.shear_modulus / 1.5)  # Normalize to 1.5 GPa
        
        # Cohesive energy score (intermolecular forces)
        min_cohesive = self.reference_values['min_cohesive_energy']
        cohesive_score = min(1, props.cohesive_energy / (min_cohesive * 3))
        
        # Density score (packing efficiency)
        density_score = min(1, props.density / 1.5)  # Normalize to 1.5 g/cm³
        
        # Weighted combination
        mech_score = (
            self.mech_weights['elastic_moduli'] * (moduli_score + 0.3*bulk_score + 0.3*shear_score) / 1.6 +
            self.mech_weights['cohesive_energy'] * cohesive_score +
            self.mech_weights['density'] * density_score +
            self.mech_weights['crystallinity'] * (1 - props.free_volume_fraction)  # Higher density = lower free volume
        )
        
        return min(1.0, max(0.0, mech_score))
    
    def calculate_overall_score(self, biocomp: float, degrd: float, mech: float, 
                              weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall material performance score
        
        Args:
            biocomp: Biocompatibility score
            degrd: Degradation rate score  
            mech: Mechanical strength score
            weights: Custom weights for properties
            
        Returns:
            Overall score (0-1)
        """
        if weights is None:
            weights = {'biocompatibility': 0.4, 'degradation_rate': 0.35, 'mechanical_strength': 0.25}
        
        overall = (
            weights['biocompatibility'] * biocomp +
            weights['degradation_rate'] * degrd +
            weights['mechanical_strength'] * mech
        )
        
        return min(1.0, max(0.0, overall))
    
    def score_material_properties(self, props: MDSimulationProperties, 
                                target_weights: Optional[Dict[str, float]] = None) -> TargetPropertyScores:
        """
        Convert MD simulation properties to target property scores
        
        Args:
            props: MD simulation properties
            target_weights: Custom weights for overall score calculation
            
        Returns:
            Target property scores
        """
        
        biocomp_score = self.calculate_biocompatibility_score(props)
        degrd_score = self.calculate_degradation_rate_score(props)
        mech_score = self.calculate_mechanical_strength_score(props)
        overall_score = self.calculate_overall_score(biocomp_score, degrd_score, mech_score, target_weights)
        
        scores = TargetPropertyScores(
            biocompatibility=biocomp_score,
            degradation_rate=degrd_score,
            mechanical_strength=mech_score,
            overall_score=overall_score
        )
        
        logger.info(f"Material scoring complete: Biocomp={biocomp_score:.3f}, "
                   f"Degrd={degrd_score:.3f}, Mech={mech_score:.3f}, Overall={overall_score:.3f}")
        
        return scores
    
    def generate_mock_md_properties(self, material_type: str = "random") -> MDSimulationProperties:
        """
        Generate mock MD properties for testing purposes
        
        Args:
            material_type: Type of material to simulate ("pla", "plga", "peg", "random")
            
        Returns:
            Mock MD simulation properties
        """
        
        np.random.seed(42)  # For reproducible results
        
        if material_type.lower() == "pla":
            # Properties typical of polylactic acid
            props = MDSimulationProperties(
                youngs_modulus_x=3.5 + np.random.normal(0, 0.5),
                youngs_modulus_y=3.2 + np.random.normal(0, 0.4),
                youngs_modulus_z=3.8 + np.random.normal(0, 0.6),
                bulk_modulus=2.1 + np.random.normal(0, 0.3),
                shear_modulus=1.3 + np.random.normal(0, 0.2),
                cohesive_energy=85 + np.random.normal(0, 10),
                mixing_energy=-15 + np.random.normal(0, 5),
                vaporization_enthalpy=95 + np.random.normal(0, 8),
                glass_transition_temp=330 + np.random.normal(0, 15),
                density=1.25 + np.random.normal(0, 0.05),
                diffusion_coefficient_water=2.5e-9 + np.random.normal(0, 5e-10),
                diffusion_coefficient_drug=1.2e-10 + np.random.normal(0, 2e-11),
                hydrogen_bond_count=45 + np.random.normal(0, 8),
                hydrogen_bond_lifetime=15.2 + np.random.normal(0, 3),
                surface_area=850 + np.random.normal(0, 100),
                free_volume_fraction=0.12 + np.random.normal(0, 0.02),
                ester_bond_count=180 + np.random.normal(0, 20),
                ester_bond_strength=320 + np.random.normal(0, 25),
                water_accessibility=0.65 + np.random.normal(0, 0.1),
                chain_scission_rate=0.08 + np.random.normal(0, 0.02),
                rmsf_polymer=1.2 + np.random.normal(0, 0.2),
                rmsf_drug=1.8 + np.random.normal(0, 0.3),
                rotational_correlation_time=25 + np.random.normal(0, 5)
            )
        
        elif material_type.lower() == "plga":
            # Properties typical of PLGA (faster degrading)
            props = MDSimulationProperties(
                youngs_modulus_x=2.8 + np.random.normal(0, 0.4),
                youngs_modulus_y=2.5 + np.random.normal(0, 0.3),
                youngs_modulus_z=3.1 + np.random.normal(0, 0.5),
                bulk_modulus=1.8 + np.random.normal(0, 0.2),
                shear_modulus=1.1 + np.random.normal(0, 0.15),
                cohesive_energy=75 + np.random.normal(0, 8),
                mixing_energy=-20 + np.random.normal(0, 6),
                vaporization_enthalpy=82 + np.random.normal(0, 7),
                glass_transition_temp=315 + np.random.normal(0, 12),
                density=1.35 + np.random.normal(0, 0.04),
                diffusion_coefficient_water=4.2e-9 + np.random.normal(0, 8e-10),
                diffusion_coefficient_drug=2.1e-10 + np.random.normal(0, 4e-11),
                hydrogen_bond_count=38 + np.random.normal(0, 6),
                hydrogen_bond_lifetime=12.8 + np.random.normal(0, 2.5),
                surface_area=920 + np.random.normal(0, 110),
                free_volume_fraction=0.18 + np.random.normal(0, 0.03),
                ester_bond_count=220 + np.random.normal(0, 25),
                ester_bond_strength=285 + np.random.normal(0, 20),
                water_accessibility=0.78 + np.random.normal(0, 0.08),
                chain_scission_rate=0.12 + np.random.normal(0, 0.025),
                rmsf_polymer=1.6 + np.random.normal(0, 0.25),
                rmsf_drug=2.3 + np.random.normal(0, 0.4),
                rotational_correlation_time=18 + np.random.normal(0, 4)
            )
        
        else:  # Random material
            props = MDSimulationProperties(
                youngs_modulus_x=np.random.uniform(0.5, 5.0),
                youngs_modulus_y=np.random.uniform(0.5, 5.0),
                youngs_modulus_z=np.random.uniform(0.5, 5.0),
                bulk_modulus=np.random.uniform(0.3, 3.0),
                shear_modulus=np.random.uniform(0.2, 2.0),
                cohesive_energy=np.random.uniform(30, 120),
                mixing_energy=np.random.uniform(-30, 10),
                vaporization_enthalpy=np.random.uniform(40, 150),
                glass_transition_temp=np.random.uniform(280, 380),
                density=np.random.uniform(0.8, 1.6),
                diffusion_coefficient_water=np.random.uniform(1e-10, 1e-8),
                diffusion_coefficient_drug=np.random.uniform(1e-12, 1e-9),
                hydrogen_bond_count=np.random.uniform(10, 80),
                hydrogen_bond_lifetime=np.random.uniform(5, 30),
                surface_area=np.random.uniform(500, 1500),
                free_volume_fraction=np.random.uniform(0.05, 0.3),
                ester_bond_count=np.random.uniform(50, 300),
                ester_bond_strength=np.random.uniform(200, 400),
                water_accessibility=np.random.uniform(0.3, 0.9),
                chain_scission_rate=np.random.uniform(0.01, 0.2),
                rmsf_polymer=np.random.uniform(0.5, 2.5),
                rmsf_drug=np.random.uniform(0.8, 3.0),
                rotational_correlation_time=np.random.uniform(10, 40)
            )
        
        return props
    
    def get_property_explanations(self) -> Dict[str, str]:
        """Get explanations for how each property contributes to scores"""
        
        return {
            "biocompatibility": """
Biocompatibility Score Factors:
• Glass Transition Temperature: Optimal around 37°C (body temperature)
• Hydrogen Bonding: Moderate bonding provides stability without toxicity
• Density: Within biocompatible range (1.0-1.4 g/cm³)
• Water Interaction: Good water compatibility reduces inflammatory response
• Surface Properties: Low surface energy and appropriate porosity
            """,
            
            "degradation_rate": """
Degradation Rate Score Factors:
• Chain Scission Rate: Primary mechanism for polymer breakdown
• Water Diffusion: Higher water access accelerates hydrolysis
• Ester Bond Strength: Weaker bonds degrade faster (ideal ~300 kJ/mol)
• Molecular Mobility: Higher mobility increases reaction rates
• Crystallinity: Amorphous regions degrade faster than crystalline
            """,
            
            "mechanical_strength": """
Mechanical Strength Score Factors:
• Elastic Moduli: Young's, bulk, and shear moduli (optimal 0.1-5.0 GPa)
• Cohesive Energy: Intermolecular forces holding material together
• Density: Higher density usually correlates with strength
• Crystallinity: Crystalline regions provide mechanical reinforcement
• Balance: Avoid brittleness while maintaining adequate strength
            """
        } 