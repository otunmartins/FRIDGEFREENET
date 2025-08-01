#!/usr/bin/env python3
"""
Enhanced Chemistry Description Generator using Domain-Knowledge Embedded Prompt Engineering.

Based on research from:
- "Integrating Chemistry Knowledge in Large Language Models via Prompt Engineering" (Liu et al., 2024)
- "VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design" (2025)
- "A Review of Large Language Models and Autonomous Agents in Chemistry" (2024)

This module implements advanced prompt engineering strategies specifically designed to generate
technical, detailed chemistry descriptions instead of generic biocompatibility statements.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# VALID-Mol constraint validation imports
try:
    from .valid_mol_framework import (
        ChemicalConstraint, ConstraintType, VALIDMolValidator, 
        ChemicalKnowledgeBase, MolecularWeight, LogP, TPSA
    )
    VALID_MOL_AVAILABLE = True
except ImportError:
    VALID_MOL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChemistryExpertise(Enum):
    """Different chemistry expertise domains for prompt engineering."""
    POLYMER_CHEMIST = "polymer_chemist"
    MEDICINAL_CHEMIST = "medicinal_chemist"
    MATERIALS_SCIENTIST = "materials_scientist"
    BIOCHEMIST = "biochemist"
    DRUG_DELIVERY_EXPERT = "drug_delivery_expert"


@dataclass
class ChemicalAnalysis:
    """Structured chemical analysis results."""
    molecular_weight: float
    functional_groups: List[str]
    backbone_type: str
    branching_pattern: str
    hydrophilicity: str
    crosslinking_potential: str
    degradation_mechanism: str
    biointeraction_sites: List[str]
    technical_properties: Dict[str, Any]


class EnhancedChemistryDescriber:
    """
    Advanced chemistry description generator using domain-knowledge embedded prompting.
    
    Implements multi-expert approach with technical chemistry knowledge integration
    to avoid generic 'biocompatible polymer' descriptions.
    """
    
    def __init__(self, llm=None, model_type="openai", openai_model="gpt-4o", temperature=0.1):
        """
        Initialize Enhanced Chemistry Describer with insulin AI app LLM integration.
        
        Args:
            llm: Pre-initialized LLM instance (preferred)
            model_type: Type of model to use if llm not provided ("openai", "ollama")
            openai_model: OpenAI model name if creating new LLM
            temperature: Temperature for LLM if creating new LLM
        """
        self.model_type = model_type
        self.openai_model = openai_model
        self.temperature = temperature
        
        # Use provided LLM or initialize new one
        if llm is not None:
            self.llm = llm
            logger.info("🧬 Enhanced Chemistry Describer: Using provided LLM instance")
        else:
            self.llm = self._initialize_llm(model_type, openai_model, temperature)
            
        self.chemistry_knowledge_base = self._initialize_chemistry_kb()
        self.expert_prompts = self._create_expert_prompts()
        
        # Track if LLM is available for fallback logic
        self.llm_available = self.llm is not None
        
        # Initialize VALID-Mol validator if available
        self.validator = self._initialize_validator()
        
    def _initialize_llm(self, model_type: str, model_name: str, temperature: float):
        """Initialize LLM following insulin AI app patterns."""
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available for Enhanced Chemistry Describer")
            return None
            
        try:
            if model_type == "openai":
                import os
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("OpenAI API key not found for Enhanced Chemistry Describer")
                    return None
                    
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model_name, temperature=temperature)
                logger.info(f"🧬 Enhanced Chemistry Describer: Initialized OpenAI {model_name} (temp: {temperature})")
                return llm
                
            elif model_type == "ollama":
                try:
                    from langchain_community.llms import Ollama
                    llm = Ollama(model=model_name, temperature=temperature)
                    logger.info(f"🧬 Enhanced Chemistry Describer: Initialized Ollama {model_name} (temp: {temperature})")
                    return llm
                except ImportError:
                    logger.warning("Ollama not available for Enhanced Chemistry Describer")
                    return None
                    
            else:
                logger.warning(f"Unknown model type '{model_type}' for Enhanced Chemistry Describer")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM for Enhanced Chemistry Describer: {e}")
            return None
    
    def _get_default_llm(self):
        """Get default LLM if none provided (legacy method)."""
        return self._initialize_llm("openai", "gpt-4o", 0.1)
    
    def _initialize_validator(self):
        """Initialize VALID-Mol validator for chemical accuracy checking."""
        if VALID_MOL_AVAILABLE:
            try:
                knowledge_base = ChemicalKnowledgeBase()
                validator = VALIDMolValidator(knowledge_base)
                logger.info("🧬 VALID-Mol validator initialized for Enhanced Chemistry Describer")
                return validator
            except Exception as e:
                logger.warning(f"Failed to initialize VALID-Mol validator: {e}")
        else:
            logger.info("VALID-Mol framework not available - using basic validation")
        return None
    
    def validate_chemical_description(self, chemical_input: str, description: str) -> Dict[str, Any]:
        """
        Validate the chemical accuracy of a generated description.
        
        Args:
            chemical_input: Original chemical input (SMILES or description)
            description: Generated technical description
            
        Returns:
            Dict containing validation results and recommendations
        """
        validation_result = {
            'is_valid': True,
            'validation_score': 1.0,
            'issues': [],
            'recommendations': [],
            'constraints_passed': [],
            'constraints_failed': []
        }
        
        try:
            # Extract potential SMILES from input if available
            smiles = self._extract_smiles_from_input(chemical_input)
            
            if smiles and RDKIT_AVAILABLE:
                # Validate using VALID-Mol constraints if available
                if self.validator:
                    validation_result.update(self._validate_with_valid_mol(smiles, description))
                else:
                    # Basic RDKit validation
                    validation_result.update(self._validate_with_rdkit(smiles, description))
            
            # Validate description content for technical accuracy
            content_validation = self._validate_description_content(description)
            validation_result.update(content_validation)
            
        except Exception as e:
            logger.error(f"Chemical description validation failed: {e}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _extract_smiles_from_input(self, chemical_input: str) -> Optional[str]:
        """Extract SMILES string from input if present."""
        if self._is_smiles(chemical_input):
            return chemical_input
        
        # Try to extract SMILES from context or description
        # This is a simple heuristic - could be enhanced
        return None
    
    def _validate_with_valid_mol(self, smiles: str, description: str) -> Dict[str, Any]:
        """Validate using VALID-Mol constraint framework."""
        result = {
            'constraints_passed': [],
            'constraints_failed': [],
            'validation_score': 1.0
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['constraints_failed'].append("Invalid SMILES structure")
                result['validation_score'] = 0.0
                return result
            
            # Define drug-like constraints for insulin delivery
            constraints = [
                ChemicalConstraint(
                    name="molecular_weight",
                    constraint_type=ConstraintType.RANGE,
                    value=(100, 1000),  # Reasonable for drug delivery polymers
                    description="Molecular weight suitable for drug delivery"
                ),
                ChemicalConstraint(
                    name="logp",
                    constraint_type=ConstraintType.RANGE,
                    value=(-2, 5),  # Hydrophilic to moderately lipophilic
                    description="LogP suitable for biological systems"
                ),
                ChemicalConstraint(
                    name="hbd_count",
                    constraint_type=ConstraintType.RANGE,
                    value=(0, 10),  # Reasonable hydrogen bond donors
                    description="Hydrogen bond donor count"
                ),
                ChemicalConstraint(
                    name="hba_count", 
                    constraint_type=ConstraintType.RANGE,
                    value=(0, 15),  # Reasonable hydrogen bond acceptors
                    description="Hydrogen bond acceptor count"
                )
            ]
            
            # Validate constraints
            constraint_results = self.validator.validate_constraints(mol, constraints)
            
            passed_count = sum(1 for passed in constraint_results.values() if passed)
            total_count = len(constraints)
            result['validation_score'] = passed_count / total_count if total_count > 0 else 1.0
            
            for constraint, passed in constraint_results.items():
                if passed:
                    result['constraints_passed'].append(constraint)
                else:
                    result['constraints_failed'].append(constraint)
            
        except Exception as e:
            logger.error(f"VALID-Mol validation failed: {e}")
            result['constraints_failed'].append(f"VALID-Mol validation error: {str(e)}")
            result['validation_score'] = 0.5
        
        return result
    
    def _validate_with_rdkit(self, smiles: str, description: str) -> Dict[str, Any]:
        """Basic validation using RDKit descriptors."""
        result = {
            'constraints_passed': [],
            'constraints_failed': [],
            'validation_score': 1.0
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['constraints_failed'].append("Invalid SMILES structure")
                result['validation_score'] = 0.0
                return result
            
            # Calculate basic descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            
            # Basic drug-like rules
            checks = [
                ("molecular_weight", 100 <= mw <= 1000),
                ("logp", -2 <= logp <= 5),
                ("hbd_count", hbd <= 10),
                ("hba_count", hba <= 15)
            ]
            
            passed_count = 0
            for check_name, passed in checks:
                if passed:
                    result['constraints_passed'].append(check_name)
                    passed_count += 1
                else:
                    result['constraints_failed'].append(check_name)
            
            result['validation_score'] = passed_count / len(checks)
            
        except Exception as e:
            logger.error(f"RDKit validation failed: {e}")
            result['constraints_failed'].append(f"RDKit validation error: {str(e)}")
            result['validation_score'] = 0.5
        
        return result
    
    def _validate_description_content(self, description: str) -> Dict[str, Any]:
        """Validate the technical content of the description."""
        result = {
            'content_issues': [],
            'content_recommendations': [],
            'technical_score': 1.0
        }
        
        # Check for generic terms that should be avoided
        generic_terms = ['biocompatible', 'suitable', 'appropriate']
        generic_count = sum(1 for term in generic_terms if term.lower() in description.lower())
        
        # Check for technical terms that indicate quality
        technical_terms = [
            'amino', 'hydroxyl', 'carboxyl', 'polymer', 'crosslinking',
            'molecular', 'functional', 'chemical', 'backbone', 'mechanism'
        ]
        technical_count = sum(1 for term in technical_terms if term.lower() in description.lower())
        
        # Calculate technical quality score
        if len(description) > 0:
            generic_ratio = generic_count / len(description.split()) 
            technical_ratio = technical_count / len(description.split())
            
            # Prefer high technical content, low generic content
            result['technical_score'] = max(0.0, min(1.0, technical_ratio - generic_ratio))
        
        # Add recommendations
        if generic_count > 2:
            result['content_issues'].append("Contains too many generic terms")
            result['content_recommendations'].append("Use more specific chemical terminology")
        
        if technical_count < 3:
            result['content_issues'].append("Lacks sufficient technical detail")
            result['content_recommendations'].append("Include more specific chemical mechanisms and structures")
        
        if len(description) < 50:
            result['content_issues'].append("Description too brief")
            result['content_recommendations'].append("Provide more detailed technical explanation")
        
        return result
    
    def _initialize_chemistry_kb(self) -> Dict[str, Any]:
        """Initialize comprehensive chemistry knowledge base for insulin delivery applications."""
        return {
            'functional_groups': {
                'hydroxyl': {
                    'smiles_pattern': r'O[H]?(?![=])',
                    'description': 'hydroxyl groups (-OH)',
                    'properties': ['hydrophilic', 'hydrogen bonding', 'polar'],
                    'bioactivity': 'facilitates protein adsorption and cell adhesion through hydrogen bonding',
                    'insulin_relevance': 'promotes insulin stabilization via hydrogen bonding with protein backbone'
                },
                'amino': {
                    'smiles_pattern': r'N[H]?[H]?(?![=])',
                    'description': 'primary/secondary amino groups (-NH2, -NH-)',
                    'properties': ['basic', 'hydrogen bonding', 'cationic at physiological pH'],
                    'bioactivity': 'enables electrostatic interaction with negatively charged biomolecules',
                    'insulin_relevance': 'provides cationic sites for insulin complexation and mucoadhesion'
                },
                'carboxyl': {
                    'smiles_pattern': r'C\(=O\)O[H]?',
                    'description': 'carboxylic acid groups (-COOH)',
                    'properties': ['acidic', 'anionic at physiological pH', 'chelating'],
                    'bioactivity': 'forms ionic complexes with insulin and divalent cations',
                    'insulin_relevance': 'enables pH-responsive insulin release and zinc chelation'
                },
                'ester': {
                    'smiles_pattern': r'C\(=O\)O(?!H)',
                    'description': 'ester linkages (-COO-)',
                    'properties': ['hydrolyzable', 'biodegradable', 'pH-sensitive'],
                    'bioactivity': 'provides controlled degradation via enzymatic hydrolysis',
                    'insulin_relevance': 'controls insulin release kinetics through hydrolytic degradation'
                },
                'ether': {
                    'smiles_pattern': r'O(?![=H])',
                    'description': 'ether bonds (-O-)',
                    'properties': ['flexible', 'hydrophilic', 'biocompatible'],
                    'bioactivity': 'enhances chain flexibility and protein resistance',
                    'insulin_relevance': 'provides stealth properties and reduces protein adsorption'
                },
                'amide': {
                    'smiles_pattern': r'C\(=O\)N',
                    'description': 'amide bonds (-CONH-)',
                    'properties': ['strong hydrogen bonding', 'stable', 'rigid'],
                    'bioactivity': 'mimics peptide bonds for protein-like interactions',
                    'insulin_relevance': 'enhances insulin binding through peptide-like recognition'
                },
                'sulfide': {
                    'smiles_pattern': r'S(?![=])',
                    'description': 'sulfide groups (-S-)',
                    'properties': ['oxidation-sensitive', 'redox-responsive', 'crosslinkable'],
                    'bioactivity': 'provides redox-responsive behavior in biological environments',
                    'insulin_relevance': 'enables glutathione-responsive insulin release'
                },
                'phosphate': {
                    'smiles_pattern': r'P\(=O\)\(O\)',
                    'description': 'phosphate groups (-PO4)',
                    'properties': ['polyanionic', 'calcium-binding', 'pH-buffering'],
                    'bioactivity': 'provides strong electrostatic interactions and calcium chelation',
                    'insulin_relevance': 'mimics natural phospholipid interactions for cellular uptake'
                }
            },
            'polymer_backbones': {
                'chitosan': {
                    'structure': 'β(1→4)-linked glucosamine/N-acetylglucosamine',
                    'properties': ['cationic', 'biodegradable', 'mucoadhesive', 'pH-responsive'],
                    'molecular_features': 'deacetylated chitin with primary amino groups',
                    'insulin_delivery': 'excellent mucoadhesion and pH-responsive swelling for oral insulin delivery',
                    'molecular_weight_range': '10-1000 kDa',
                    'degree_substitution': 'deacetylation degree 70-95%'
                },
                'alginate': {
                    'structure': 'alternating guluronic and mannuronic acid residues',
                    'properties': ['anionic', 'gel-forming', 'Ca2+-crosslinkable', 'biocompatible'],
                    'molecular_features': 'β(1→4) linked uronic acids with varying G/M ratio',
                    'insulin_delivery': 'calcium-induced gelation for controlled insulin encapsulation',
                    'molecular_weight_range': '50-500 kDa',
                    'gel_strength': 'depends on guluronic acid content and molecular weight'
                },
                'hyaluronic_acid': {
                    'structure': 'β(1→4)-linked glucuronic acid and N-acetylglucosamine',
                    'properties': ['anionic', 'high water retention', 'CD44 targeting', 'viscoelastic'],
                    'molecular_features': 'alternating disaccharide repeat with carboxyl groups',
                    'insulin_delivery': 'CD44 receptor targeting for cellular insulin uptake enhancement',
                    'molecular_weight_range': '100-2000 kDa',
                    'targeting': 'CD44 receptor-mediated endocytosis'
                },
                'peg': {
                    'structure': 'linear polyethylene glycol chain',
                    'properties': ['hydrophilic', 'protein-resistant', 'non-immunogenic', 'flexible'],
                    'molecular_features': 'linear chain of ethylene oxide units (-CH2CH2O-)',
                    'insulin_delivery': 'stealth coating for prolonged circulation and reduced immunogenicity',
                    'molecular_weight_range': '0.4-40 kDa',
                    'stealth_properties': 'reduces protein adsorption and immune recognition'
                },
                'plga': {
                    'structure': 'poly(lactic-co-glycolic acid) copolymer',
                    'properties': ['biodegradable', 'biocompatible', 'tunable degradation', 'hydrophobic'],
                    'molecular_features': 'ester-linked lactic and glycolic acid units',
                    'insulin_delivery': 'sustained insulin release through bulk erosion mechanism',
                    'molecular_weight_range': '10-200 kDa',
                    'degradation_time': '2 weeks to 12 months depending on LA:GA ratio'
                },
                'pectin': {
                    'structure': 'galacturonic acid backbone with rhamnose insertions',
                    'properties': ['pH-responsive', 'calcium-sensitive', 'mucoadhesive', 'biodegradable'],
                    'molecular_features': 'α(1→4)-linked galacturonic acid with methyl esterification',
                    'insulin_delivery': 'colon-specific insulin delivery via bacterial enzyme degradation',
                    'molecular_weight_range': '50-150 kDa',
                    'methoxylation': 'degree of esterification affects gel properties'
                },
                'cellulose_derivatives': {
                    'structure': 'modified cellulose with hydroxypropyl or carboxymethyl groups',
                    'properties': ['pH-responsive', 'thermogelling', 'mucoadhesive', 'non-ionic'],
                    'molecular_features': 'β(1→4)-linked glucose with pendant functional groups',
                    'insulin_delivery': 'thermoreversible gelation for in situ depot formation',
                    'molecular_weight_range': '80-120 kDa',
                    'substitution': 'degree of substitution affects gelation temperature'
                },
                'dextran': {
                    'structure': 'α(1→6)-linked glucose with α(1→3) branches',
                    'properties': ['biocompatible', 'low immunogenicity', 'tunable molecular weight', 'biodegradable'],
                    'molecular_features': 'branched glucose polymer with hydroxyl functionalization sites',
                    'insulin_delivery': 'size-dependent renal clearance for controlled pharmacokinetics',
                    'molecular_weight_range': '6-2000 kDa',
                    'clearance': 'molecular weight determines renal vs hepatic clearance'
                }
            },
            'crosslinking_mechanisms': {
                'ionic': {
                    'description': 'electrostatic interactions between oppositely charged groups',
                    'examples': ['calcium-alginate', 'chitosan-TPP', 'pectin-calcium'],
                    'reversibility': 'reversible through ion exchange',
                    'insulin_application': 'rapid gelation for insulin encapsulation',
                    'strength': 'moderate, pH and ionic strength dependent'
                },
                'covalent': {
                    'description': 'chemical bonds formed through crosslinking agents',
                    'examples': ['glutaraldehyde', 'EDC/NHS', 'genipin', 'transglutaminase'],
                    'reversibility': 'irreversible under physiological conditions',
                    'insulin_application': 'stable hydrogels for long-term insulin delivery',
                    'strength': 'high, permanent crosslinks'
                },
                'physical': {
                    'description': 'hydrogen bonding, van der Waals, and hydrophobic interactions',
                    'examples': ['PEG-PLA thermogels', 'cellulose derivatives', 'Pluronic gels'],
                    'reversibility': 'reversible through temperature or pH changes',
                    'insulin_application': 'injectable gels for minimally invasive delivery',
                    'strength': 'moderate, temperature dependent'
                },
                'enzymatic': {
                    'description': 'transglutaminase-mediated lysine-glutamine crosslinks',
                    'examples': ['protein-polysaccharide conjugates', 'gelatin crosslinking'],
                    'reversibility': 'biodegradable through protease activity',
                    'insulin_application': 'biomimetic crosslinking for natural degradation',
                    'strength': 'high, but enzymatically degradable'
                },
                'photo': {
                    'description': 'UV or visible light-induced radical polymerization',
                    'examples': ['methacrylated polymers', 'thiol-ene systems', 'azide-alkyne'],
                    'reversibility': 'irreversible once crosslinked',
                    'insulin_application': 'spatial control of gel formation for targeted delivery',
                    'strength': 'tunable through light exposure time and intensity'
                }
            },
            'release_mechanisms': {
                'diffusion': {
                    'description': 'Fickian diffusion through polymer matrix',
                    'kinetics': 'square root of time dependence',
                    'controlling_factors': ['polymer mesh size', 'drug molecular weight', 'tortuosity'],
                    'insulin_suitability': 'suitable for small insulin formulations',
                    'mathematical_model': 'Higuchi model for matrix tablets'
                },
                'swelling': {
                    'description': 'osmotic pressure-driven matrix expansion',
                    'kinetics': 'zero-order to first-order depending on geometry',
                    'controlling_factors': ['crosslink density', 'osmotic pressure', 'polymer hydrophilicity'],
                    'insulin_suitability': 'excellent for pH-responsive insulin release',
                    'mathematical_model': 'Korsmeyer-Peppas model'
                },
                'erosion': {
                    'description': 'bulk or surface degradation of polymer matrix',
                    'kinetics': 'zero-order for surface erosion, first-order for bulk erosion',
                    'controlling_factors': ['polymer composition', 'molecular weight', 'crystallinity'],
                    'insulin_suitability': 'provides sustained insulin release over weeks',
                    'mathematical_model': 'zero-order or first-order kinetics'
                },
                'enzymatic': {
                    'description': 'site-specific enzymatic cleavage of polymer backbone',
                    'kinetics': 'Michaelis-Menten enzyme kinetics',
                    'controlling_factors': ['enzyme concentration', 'substrate affinity', 'accessibility'],
                    'insulin_suitability': 'colon-specific insulin delivery via bacterial enzymes',
                    'mathematical_model': 'enzyme kinetics coupled with diffusion'
                },
                'osmotic': {
                    'description': 'osmotic pressure-driven drug release through semipermeable membrane',
                    'kinetics': 'zero-order release independent of pH and agitation',
                    'controlling_factors': ['osmotic pressure', 'membrane permeability', 'orifice size'],
                    'insulin_suitability': 'constant insulin release for basal therapy',
                    'mathematical_model': 'osmotic pressure equation'
                }
            },
            'insulin_specific_considerations': {
                'stability_challenges': {
                    'aggregation': 'insulin fibrillation at interfaces and elevated temperatures',
                    'deamidation': 'asparagine and glutamine deamidation at physiological pH',
                    'oxidation': 'methionine oxidation leading to reduced bioactivity',
                    'proteolysis': 'enzymatic degradation in GI tract and tissues'
                },
                'formulation_strategies': {
                    'stabilizers': ['trehalose', 'sucrose', 'mannitol', 'zinc', 'phenol'],
                    'penetration_enhancers': ['EDTA', 'sodium caprate', 'oleic acid'],
                    'enzyme_inhibitors': ['aprotinin', 'bestatin', 'boroleucine'],
                    'ph_modifiers': ['citrate buffer', 'phosphate buffer', 'tris buffer']
                },
                'delivery_routes': {
                    'subcutaneous': 'standard route with rapid absorption',
                    'oral': 'challenging due to GI degradation and poor permeability',
                    'pulmonary': 'rapid absorption but technical delivery challenges',
                    'transdermal': 'steady delivery but requires permeation enhancement',
                    'nasal': 'direct CNS access but limited dose capacity'
                },
                'pharmacokinetic_targets': {
                    'basal_insulin': 'constant low-level insulin for 12-24 hours',
                    'prandial_insulin': 'rapid insulin peak within 15-30 minutes',
                    'combination_therapy': 'dual release profile with immediate and sustained phases'
                }
            }
        }
    
    def _create_expert_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create expert-specific prompts for different chemistry domains."""
        
        base_context = """You are a world-class expert in chemistry with deep knowledge of molecular structure, 
chemical properties, and biomedical applications. Your role is to provide highly technical, specific 
chemical descriptions that demonstrate expert-level understanding.

CRITICAL REQUIREMENTS:
1. Use precise chemical terminology and nomenclature
2. Describe specific molecular interactions and mechanisms
3. Explain structure-property relationships
4. Avoid generic terms like "biocompatible" without technical explanation
5. Reference specific chemical bonds, functional groups, and molecular features
6. Discuss quantitative properties when relevant (molecular weight, degree of substitution, etc.)

DOMAIN KNOWLEDGE INTEGRATION:
- Always explain WHY certain chemical features provide specific properties
- Connect molecular structure to biological/physical behavior
- Use appropriate units and chemical notation
- Reference established chemical principles and mechanisms"""

        prompts = {}
        
        # Polymer Chemistry Expert
        prompts[ChemistryExpertise.POLYMER_CHEMIST] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_context}

EXPERT SPECIALIZATION: Polymer Chemistry and Materials Science
You understand polymer synthesis, chain structure, crosslinking mechanisms, and structure-property relationships.

TECHNICAL FOCUS AREAS:
- Polymer backbone chemistry and repeat unit structure
- Crosslinking density and network topology
- Molecular weight distribution and polydispersity
- Chain conformation and entanglement
- Degradation mechanisms (hydrolytic, enzymatic, oxidative)
- Mechanical properties from molecular architecture

REQUIRED TECHNICAL ELEMENTS:
- Describe specific polymer backbone chemistry
- Explain crosslinking mechanism and density
- Discuss molecular weight characteristics
- Detail degradation pathways and kinetics
- Connect molecular structure to bulk properties"""),
            
            ("human", """Analyze this chemical structure and provide a technical polymer chemistry description: {chemical_input}

Focus on:
1. Detailed backbone chemistry and repeat unit structure
2. Specific crosslinking mechanisms and network architecture  
3. Molecular weight characteristics and distribution
4. Degradation pathways and mechanisms
5. Structure-property relationships for the intended application

Provide a technical description that demonstrates expert polymer chemistry knowledge.""")
        ])
        
        # Drug Delivery Expert
        prompts[ChemistryExpertise.DRUG_DELIVERY_EXPERT] = ChatPromptTemplate.from_messages([
            ("system", f"""{base_context}

EXPERT SPECIALIZATION: Drug Delivery Systems and Pharmaceutical Materials
You understand drug-polymer interactions, release mechanisms, and pharmacokinetics.

TECHNICAL FOCUS AREAS:
- Drug-polymer interaction mechanisms (ionic, hydrophobic, hydrogen bonding)
- Release kinetics and mathematical models (Higuchi, Korsmeyer-Peppas)
- Bioadhesion mechanisms and mucoadhesive properties
- Barrier properties and diffusion coefficients
- Bioavailability enhancement strategies
- Targeting mechanisms and site-specific delivery

REQUIRED TECHNICAL ELEMENTS:
- Describe specific drug-polymer interaction mechanisms
- Explain release kinetics and controlling factors
- Detail bioadhesion mechanisms at molecular level
- Discuss permeation enhancement strategies
- Connect molecular features to pharmacokinetic parameters"""),
            
            ("human", """Analyze this chemical structure for drug delivery applications: {chemical_input}

Focus on:
1. Specific drug-polymer interaction mechanisms
2. Release kinetics and controlling molecular factors
3. Bioadhesion mechanisms and tissue interaction
4. Permeation enhancement properties
5. Targeting capabilities and selectivity mechanisms

Provide a technical description emphasizing drug delivery expertise.""")
        ])
        
        return prompts
    
    def analyze_chemical_structure(self, smiles: str, additional_info: Dict[str, Any] = None) -> ChemicalAnalysis:
        """Perform detailed chemical analysis of the structure."""
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available for chemical analysis")
            return self._create_fallback_analysis(smiles, additional_info)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_fallback_analysis(smiles, additional_info)
            
            # Calculate molecular properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            
            # Identify functional groups
            functional_groups = self._identify_functional_groups(smiles)
            
            # Determine backbone type
            backbone_type = self._determine_backbone_type(smiles, additional_info)
            
            # Assess molecular features
            branching_pattern = self._assess_branching(mol)
            hydrophilicity = self._assess_hydrophilicity(logp, hbd, hba)
            crosslinking_potential = self._assess_crosslinking_potential(functional_groups)
            degradation_mechanism = self._predict_degradation_mechanism(functional_groups)
            biointeraction_sites = self._identify_biointeraction_sites(functional_groups)
            
            technical_properties = {
                'molecular_weight': mw,
                'logP': logp,
                'hydrogen_bond_donors': hbd,
                'hydrogen_bond_acceptors': hba,
                'estimated_pka': self._estimate_pka(functional_groups),
                'charge_density': self._calculate_charge_density(functional_groups, mw)
            }
            
            return ChemicalAnalysis(
                molecular_weight=mw,
                functional_groups=functional_groups,
                backbone_type=backbone_type,
                branching_pattern=branching_pattern,
                hydrophilicity=hydrophilicity,
                crosslinking_potential=crosslinking_potential,
                degradation_mechanism=degradation_mechanism,
                biointeraction_sites=biointeraction_sites,
                technical_properties=technical_properties
            )
            
        except Exception as e:
            logger.error(f"Chemical analysis failed: {e}")
            return self._create_fallback_analysis(smiles, additional_info)
    
    def _identify_functional_groups(self, smiles: str) -> List[str]:
        """Identify functional groups in the SMILES structure."""
        groups = []
        
        for group_name, group_data in self.chemistry_knowledge_base['functional_groups'].items():
            pattern = group_data['smiles_pattern']
            if re.search(pattern, smiles):
                groups.append(group_name)
        
        return groups
    
    def _determine_backbone_type(self, smiles: str, additional_info: Dict[str, Any] = None) -> str:
        """Determine the polymer backbone type."""
        
        # Check for known polymer patterns
        for backbone_name, backbone_data in self.chemistry_knowledge_base['polymer_backbones'].items():
            if additional_info and backbone_name.lower() in str(additional_info).lower():
                return backbone_name
        
        # Analyze SMILES for backbone characteristics
        if 'O' in smiles and 'C' in smiles:
            if smiles.count('O') > smiles.count('C') * 0.3:
                return 'polyether-based'
        
        if 'N' in smiles:
            return 'amino-containing backbone'
        
        return 'carbon-based backbone'
    
    def _assess_branching(self, mol) -> str:
        """Assess the branching pattern of the molecule."""
        # Simplified branching assessment
        if mol.GetNumAtoms() < 10:
            return 'linear or lightly branched'
        else:
            return 'potentially branched architecture'
    
    def _assess_hydrophilicity(self, logp: float, hbd: int, hba: int) -> str:
        """Assess hydrophilicity based on molecular descriptors."""
        if logp < 0 and (hbd + hba) > 5:
            return 'highly hydrophilic with extensive hydrogen bonding capacity'
        elif logp < 2 and (hbd + hba) > 2:
            return 'moderately hydrophilic with balanced solubility'
        else:
            return 'hydrophobic with limited water solubility'
    
    def _assess_crosslinking_potential(self, functional_groups: List[str]) -> str:
        """Assess crosslinking potential based on functional groups."""
        if 'amino' in functional_groups and 'carboxyl' in functional_groups:
            return 'high crosslinking potential via ionic and covalent mechanisms'
        elif 'hydroxyl' in functional_groups:
            return 'moderate crosslinking potential via hydrogen bonding'
        else:
            return 'limited crosslinking potential'
    
    def _predict_degradation_mechanism(self, functional_groups: List[str]) -> str:
        """Predict the primary degradation mechanism."""
        if 'ester' in functional_groups:
            return 'hydrolytic degradation via ester bond cleavage'
        elif 'amide' in functional_groups:
            return 'enzymatic degradation via peptidase activity'
        elif 'ether' in functional_groups:
            return 'oxidative degradation with slow kinetics'
        else:
            return 'minimal degradation under physiological conditions'
    
    def _identify_biointeraction_sites(self, functional_groups: List[str]) -> List[str]:
        """Identify sites for biological interactions."""
        sites = []
        
        if 'amino' in functional_groups:
            sites.append('cationic sites for anionic biomolecule binding')
        if 'carboxyl' in functional_groups:
            sites.append('anionic sites for cationic protein domains')
        if 'hydroxyl' in functional_groups:
            sites.append('hydrogen bonding sites for protein adsorption')
        
        return sites or ['limited specific biointeraction sites']
    
    def _estimate_pka(self, functional_groups: List[str]) -> Optional[float]:
        """Estimate pKa values for ionizable groups."""
        if 'carboxyl' in functional_groups:
            return 4.2  # Typical carboxylic acid pKa
        elif 'amino' in functional_groups:
            return 9.5  # Typical amino group pKa
        return None
    
    def _calculate_charge_density(self, functional_groups: List[str], mw: float) -> float:
        """Calculate approximate charge density."""
        ionizable_groups = sum(1 for group in functional_groups if group in ['amino', 'carboxyl'])
        return (ionizable_groups / mw) * 1000 if mw > 0 else 0  # charges per kDa
    
    def _create_fallback_analysis(self, smiles: str, additional_info: Dict[str, Any] = None) -> ChemicalAnalysis:
        """Create fallback analysis when RDKit is not available."""
        return ChemicalAnalysis(
            molecular_weight=500.0,  # Estimated
            functional_groups=self._identify_functional_groups(smiles),
            backbone_type=self._determine_backbone_type(smiles, additional_info),
            branching_pattern='unknown branching pattern',
            hydrophilicity='moderate hydrophilicity',
            crosslinking_potential='moderate crosslinking potential',
            degradation_mechanism='hydrolytic degradation',
            biointeraction_sites=['general biointeraction sites'],
            technical_properties={'estimated': True}
        )
    
    def generate_expert_description(self, 
                                  chemical_input: str, 
                                  expertise: ChemistryExpertise = ChemistryExpertise.DRUG_DELIVERY_EXPERT,
                                  context: Dict[str, Any] = None,
                                  validate_output: bool = True,
                                  max_iterations: int = 2) -> str:
        """
        Generate expert-level technical chemistry description using domain-knowledge embedded prompting.
        
        Args:
            chemical_input: SMILES string or chemical description
            expertise: Type of chemistry expertise to apply
            context: Additional context about the application
            validate_output: Whether to validate and potentially improve the output
            max_iterations: Maximum iterations for validation-based improvement
            
        Returns:
            Technical chemistry description with expert-level detail
        """
        
        if not self.llm or not LANGCHAIN_AVAILABLE:
            return self._generate_rule_based_description(chemical_input, context)
        
        try:
            best_description = None
            best_score = 0.0
            
            for iteration in range(max_iterations):
                # Analyze chemical structure if SMILES provided
                analysis = None
                if self._is_smiles(chemical_input):
                    analysis = self.analyze_chemical_structure(chemical_input, context)
                
                # Prepare enhanced input with analysis
                enhanced_input = self._prepare_enhanced_input(chemical_input, analysis, context)
                
                # Get expert prompt
                prompt_template = self.expert_prompts.get(expertise)
                if not prompt_template:
                    expertise = ChemistryExpertise.DRUG_DELIVERY_EXPERT
                    prompt_template = self.expert_prompts[expertise]
                
                # Generate description using expert prompting
                messages = prompt_template.format_messages(chemical_input=enhanced_input)
                response = self.llm.invoke(messages)
                
                description = self._post_process_description(response.content, analysis)
                
                # Validate if requested
                if validate_output:
                    validation = self.validate_chemical_description(chemical_input, description)
                    
                    # Calculate combined score (validation + technical quality)
                    validation_score = validation.get('validation_score', 1.0)
                    technical_score = validation.get('technical_score', 1.0)
                    combined_score = (validation_score + technical_score) / 2
                    
                    # Keep best description
                    if combined_score > best_score:
                        best_description = description
                        best_score = combined_score
                    
                    # If we have a high-quality description, stop iterating
                    if combined_score >= 0.8:
                        logger.info(f"🧬 High-quality description achieved (score: {combined_score:.2f})")
                        break
                    
                    # Use validation feedback to improve next iteration
                    if iteration < max_iterations - 1:
                        enhanced_input = self._enhance_input_with_feedback(
                            enhanced_input, validation
                        )
                        
                else:
                    return description
            
            return best_description or self._generate_rule_based_description(chemical_input, context)
            
        except Exception as e:
            logger.error(f"Expert description generation failed: {e}")
            return self._generate_rule_based_description(chemical_input, context)
    
    def _enhance_input_with_feedback(self, original_input: str, validation: Dict[str, Any]) -> str:
        """Enhance input with validation feedback for iterative improvement."""
        
        enhanced_input = original_input
        
        # Add specific improvement guidance based on validation results
        if validation.get('content_issues'):
            enhanced_input += "\n\nIMPROVEMENT REQUIREMENTS:\n"
            
            for issue in validation['content_issues']:
                if "generic terms" in issue:
                    enhanced_input += "- Avoid generic terms like 'biocompatible' and 'suitable'\n"
                    enhanced_input += "- Use specific chemical terminology and mechanisms\n"
                elif "technical detail" in issue:
                    enhanced_input += "- Include more specific molecular structures and functional groups\n"
                    enhanced_input += "- Describe detailed chemical mechanisms and interactions\n"
                elif "too brief" in issue:
                    enhanced_input += "- Provide more comprehensive technical explanation\n"
                    enhanced_input += "- Include quantitative aspects where relevant\n"
            
        if validation.get('content_recommendations'):
            enhanced_input += "\nSPECIFIC RECOMMENDATIONS:\n"
            for rec in validation['content_recommendations']:
                enhanced_input += f"- {rec}\n"
        
        return enhanced_input
    
    def _is_smiles(self, input_str: str) -> bool:
        """Check if input string is a SMILES."""
        # Simple heuristic: contains typical SMILES characters and no spaces
        smiles_chars = set('CNOPSFClBrI[]()=#+@/-\\1234567890')
        return all(c in smiles_chars for c in input_str.replace(' ', '')) and len(input_str) > 3
    
    def _prepare_enhanced_input(self, chemical_input: str, analysis: ChemicalAnalysis = None, context: Dict[str, Any] = None) -> str:
        """Prepare enhanced input with chemical analysis and context."""
        
        enhanced = f"Chemical Structure: {chemical_input}\n\n"
        
        if analysis:
            enhanced += f"Molecular Analysis:\n"
            enhanced += f"- Molecular Weight: {analysis.molecular_weight:.1f} Da\n"
            enhanced += f"- Functional Groups: {', '.join(analysis.functional_groups)}\n"
            enhanced += f"- Backbone Type: {analysis.backbone_type}\n"
            enhanced += f"- Hydrophilicity: {analysis.hydrophilicity}\n"
            enhanced += f"- Crosslinking Potential: {analysis.crosslinking_potential}\n"
            enhanced += f"- Degradation Mechanism: {analysis.degradation_mechanism}\n"
            enhanced += f"- Biointeraction Sites: {', '.join(analysis.biointeraction_sites)}\n\n"
        
        if context:
            enhanced += f"Application Context: {context}\n\n"
        
        return enhanced
    
    def _post_process_description(self, description: str, analysis: ChemicalAnalysis = None) -> str:
        """Post-process the generated description for quality and technical accuracy."""
        
        # Remove generic terms if they appear without technical context
        generic_terms = ['biocompatible', 'suitable for', 'appropriate for']
        
        for term in generic_terms:
            # Only remove if not followed by technical explanation
            pattern = rf'{term}(?!\s+(?:due to|through|via|because of))'
            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
        
        # Ensure technical content is preserved
        if analysis and len(description) < 200:
            # Add more technical detail if description is too short
            technical_addition = f"\n\nTechnical Characteristics: This structure contains {', '.join(analysis.functional_groups)} functional groups with {analysis.crosslinking_potential.lower()}. The {analysis.degradation_mechanism} provides controlled release kinetics."
            description += technical_addition
        
        return description.strip()
    
    def _generate_rule_based_description(self, chemical_input: str, context: Dict[str, Any] = None) -> str:
        """Generate rule-based description when LLM is not available, using expanded knowledge base."""
        
        # Analyze functional groups with enhanced knowledge
        functional_groups = self._identify_functional_groups(chemical_input)
        
        # Determine polymer backbone with insulin-specific information
        backbone_info = self._get_backbone_information(chemical_input, context)
        
        # Get insulin-specific considerations
        insulin_considerations = self._get_insulin_specific_features(functional_groups, backbone_info, context)
        
        if not functional_groups and not backbone_info:
            return "Complex molecular architecture designed for controlled insulin delivery with specialized structure-property relationships optimized for biomedical applications."
        
        # Build comprehensive technical description
        description_parts = []
        
        # Start with backbone description
        if backbone_info:
            backbone_desc = backbone_info.get('structure', 'specialized polymer backbone')
            molecular_features = backbone_info.get('molecular_features', '')
            if molecular_features:
                description_parts.append(f"{backbone_desc} featuring {molecular_features}")
            else:
                description_parts.append(backbone_desc)
        
        # Add functional group details with insulin relevance
        if functional_groups:
            fg_descriptions = []
            for fg in functional_groups[:3]:  # Limit to top 3 for readability
                fg_data = self.chemistry_knowledge_base['functional_groups'].get(fg, {})
                fg_desc = fg_data.get('description', fg)
                insulin_relevance = fg_data.get('insulin_relevance', '')
                if insulin_relevance:
                    fg_descriptions.append(f"{fg_desc} that {insulin_relevance}")
                else:
                    fg_descriptions.append(fg_desc)
            
            if len(fg_descriptions) == 1:
                description_parts.append(f"incorporating {fg_descriptions[0]}")
            elif len(fg_descriptions) == 2:
                description_parts.append(f"incorporating {fg_descriptions[0]} and {fg_descriptions[1]}")
            else:
                description_parts.append(f"incorporating {', '.join(fg_descriptions[:-1])}, and {fg_descriptions[-1]}")
        
        # Add crosslinking and release mechanisms
        crosslinking_info = self._infer_crosslinking_mechanism(functional_groups, context)
        if crosslinking_info:
            description_parts.append(f"utilizing {crosslinking_info}")
        
        # Add insulin delivery specifics
        if insulin_considerations:
            description_parts.append(f"engineered for {insulin_considerations}")
        
        # Combine all parts
        if len(description_parts) == 1:
            full_description = description_parts[0]
        else:
            # Connect parts with appropriate conjunctions
            full_description = description_parts[0]
            for i, part in enumerate(description_parts[1:], 1):
                if i == len(description_parts) - 1:
                    full_description += f", and {part}"
                else:
                    full_description += f", {part}"
        
        # Ensure proper ending
        if not full_description.endswith('.'):
            full_description += '.'
        
        return full_description
    
    def _get_backbone_information(self, chemical_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get detailed backbone information from the knowledge base."""
        
        input_lower = chemical_input.lower()
        
        # Check context first
        if context:
            for key, value in context.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            item_lower = item.lower()
                            for backbone_name, backbone_data in self.chemistry_knowledge_base['polymer_backbones'].items():
                                if backbone_name in item_lower:
                                    return backbone_data
        
        # Check direct matches in input
        for backbone_name, backbone_data in self.chemistry_knowledge_base['polymer_backbones'].items():
            if backbone_name in input_lower or backbone_name.replace('_', ' ') in input_lower:
                return backbone_data
        
        return {}
    
    def _infer_crosslinking_mechanism(self, functional_groups: List[str], context: Dict[str, Any] = None) -> str:
        """Infer crosslinking mechanism from functional groups and context."""
        
        mechanisms = []
        
        # Ionic crosslinking
        if any(fg in functional_groups for fg in ['amino', 'carboxyl']):
            mech_data = self.chemistry_knowledge_base['crosslinking_mechanisms']['ionic']
            mechanisms.append(f"{mech_data['description']} for {mech_data['insulin_application']}")
        
        # Covalent crosslinking
        if any(fg in functional_groups for fg in ['ester', 'amide']):
            mech_data = self.chemistry_knowledge_base['crosslinking_mechanisms']['covalent']
            mechanisms.append(f"{mech_data['description']}")
        
        # Physical crosslinking
        if any(fg in functional_groups for fg in ['hydroxyl', 'ether']):
            mech_data = self.chemistry_knowledge_base['crosslinking_mechanisms']['physical']
            mechanisms.append(f"{mech_data['description']}")
        
        if mechanisms:
            if len(mechanisms) == 1:
                return mechanisms[0]
            else:
                return f"{mechanisms[0]} combined with {mechanisms[1]}"
        
        return "controlled crosslinking architecture"
    
    def _get_insulin_specific_features(self, functional_groups: List[str], backbone_info: Dict[str, Any], 
                                     context: Dict[str, Any] = None) -> str:
        """Get insulin-specific delivery features and applications."""
        
        features = []
        
        # Get insulin delivery information from backbone
        if backbone_info.get('insulin_delivery'):
            features.append(backbone_info['insulin_delivery'])
        
        # Add route-specific information
        if context:
            application_details = context.get('application_details', [])
            for detail in application_details:
                detail_lower = detail.lower()
                if 'oral' in detail_lower:
                    features.append("gastric acid resistance and intestinal mucoadhesion")
                elif 'transdermal' in detail_lower or 'patch' in detail_lower:
                    features.append("sustained percutaneous insulin delivery with enhanced skin permeation")
                elif 'injectable' in detail_lower:
                    features.append("in situ gel formation for depot insulin release")
                elif 'nasal' in detail_lower:
                    features.append("rapid nasal absorption with minimized enzymatic degradation")
        
        # Add mechanism-specific features
        if any(fg in functional_groups for fg in ['amino']):
            features.append("pH-responsive insulin release triggered by physiological pH changes")
        
        if any(fg in functional_groups for fg in ['carboxyl']):
            features.append("zinc chelation for insulin stabilization and controlled release")
        
        # Default insulin delivery feature
        if not features:
            features.append("controlled insulin delivery with optimized pharmacokinetic profile")
        
        # Combine features
        if len(features) == 1:
            return features[0]
        elif len(features) == 2:
            return f"{features[0]} and {features[1]}"
        else:
            return f"{', '.join(features[:-1])}, and {features[-1]}"


# Factory function for easy usage
def create_chemistry_describer(llm=None, model_type="openai", openai_model="gpt-4o", temperature=0.1) -> EnhancedChemistryDescriber:
    """
    Create an enhanced chemistry describer instance with flexible LLM configuration.
    
    Args:
        llm: Pre-initialized LLM instance (preferred for integration)
        model_type: Model type if creating new LLM ("openai", "ollama") 
        openai_model: OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo")
        temperature: Temperature for text generation (0.0-1.0)
        
    Returns:
        EnhancedChemistryDescriber instance
    """
    return EnhancedChemistryDescriber(
        llm=llm, 
        model_type=model_type, 
        openai_model=openai_model, 
        temperature=temperature
    )


# Convenience functions
def describe_polymer_chemistry(chemical_input: str, llm=None, context: Dict[str, Any] = None, 
                              model_type="openai", openai_model="gpt-4o", temperature=0.1) -> str:
    """
    Generate polymer chemistry expert description.
    
    Args:
        chemical_input: SMILES string or chemical description
        llm: Pre-initialized LLM instance
        context: Additional context about the chemistry
        model_type: Model type for new LLM initialization
        openai_model: OpenAI model name
        temperature: Temperature for generation
        
    Returns:
        Expert polymer chemistry description
    """
    describer = create_chemistry_describer(llm, model_type, openai_model, temperature)
    return describer.generate_expert_description(
        chemical_input, 
        ChemistryExpertise.POLYMER_CHEMIST,
        context
    )


def describe_drug_delivery_chemistry(chemical_input: str, llm=None, context: Dict[str, Any] = None,
                                    model_type="openai", openai_model="gpt-4o", temperature=0.1) -> str:
    """
    Generate drug delivery expert description.
    
    Args:
        chemical_input: SMILES string or chemical description
        llm: Pre-initialized LLM instance
        context: Additional context about the application
        model_type: Model type for new LLM initialization
        openai_model: OpenAI model name
        temperature: Temperature for generation
        
    Returns:
        Expert drug delivery chemistry description
    """
    describer = create_chemistry_describer(llm, model_type, openai_model, temperature)
    return describer.generate_expert_description(
        chemical_input,
        ChemistryExpertise.DRUG_DELIVERY_EXPERT, 
        context
    ) 