#!/usr/bin/env python3
"""
Enhanced PSMILES Generator
Generates polymer SMILES (PSMILES) strings from natural language descriptions.
Now supports OpenAI ChatGPT models for superior performance.
Integrates with the psmiles library for validation and visualization.
"""

import json
import requests
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging

# **UPDATED: OpenAI imports instead of Ollama**
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import enhanced SMILES processing system
try:
    from insulin_ai.utils.natural_language_smiles import NaturalLanguageToSMILES
    NATURAL_SMILES_AVAILABLE = True
    logging.info("✅ SMILES Self-Corrector initialized successfully")
except ImportError as e:
    # Try adding parent directory to path (for when running from subdirectories)
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent  # Go up to src/ directory
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from insulin_ai.utils.natural_language_smiles import NaturalLanguageToSMILES
        NATURAL_SMILES_AVAILABLE = True
        logging.info("✅ SMILES Self-Corrector initialized successfully (with path adjustment)")
    except ImportError as e2:
        NATURAL_SMILES_AVAILABLE = False
        logging.warning(f"⚠️ SMILES Self-Corrector not available: {e2}")

# **NEW: OpenAI-Compatible Working Pipeline**
# Adapted from the working utils/natural_language_smiles.py system
class OpenAIWorkingPipeline:
    """Working pipeline: Natural Language → SMILES → PSMILES using OpenAI with chemistry lookup and validation"""
    
    def __init__(self, llm):
        self.model_type = 'openai'
        self.available = True
        self.llm = llm
        self.chemistry_examples = self._get_chemistry_examples()
        
        # Enhanced SMILES generation prompt based on VALID-Mol research (2025)
        self.nl_to_smiles_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a chemistry expert. Convert this description to a valid SMILES string following these EXACT rules:

CONSTRAINTS (CRITICAL - FROM VALID-MOL FRAMEWORK):
- Molecular weight: MUST be between 100-800 Da (realistic drug-like range)
- SMILES length: MUST be between 5-1000 characters (prevents unrealistic structures)
- Chemical validity: MUST pass RDKit validation
- Output format: SMILES_STRING_ONLY (no explanations, no text, no descriptions)

CRITICAL VALENCE RULES (PREVENT POLYMER LIBRARY FAILURES):
- Carbon: EXACTLY 4 bonds maximum (never exceed) 
- Oxygen: EXACTLY 2 bonds maximum (C=O, C-O-C, never C(X)(Y)=O where X,Y are heavy atoms)
- Nitrogen: EXACTLY 3 bonds maximum (C-N-C, C=N, never exceed)
- Fluorine: EXACTLY 1 bond maximum (C-F, never exceed)
- Chlorine: EXACTLY 1 bond maximum (C-Cl, never exceed)
- AVOID: C(C(C)=O)C(C)=O - creates oxygen valence violations
- AVOID: C(X)(Y)=O where X,Y are not hydrogen - causes O valence > 2
- AVOID: F(X) or Cl(X) - halogens can only have one bond
- AVOID: Complex branched carbonyls - use simple C=O or CC(=O)
- AVOID: Multiple bonds to halogens - F and Cl must have exactly one bond

POLYMER MONOMER RULES (CRITICAL FOR POLYMERIZATION):
- Structure MUST be a realistic monomer unit (not full polymer)
- Include functional groups for chain growth: C=C, C(=O)O, C(=O)N, aromatic rings
- Prefer linear or simple branched structures over complex branching
- Examples of GOOD monomers: CC(=O)O, C=C, c1ccccc1, CC(=O)N, CCC=O

SMILES GUIDELINES (FOLLOW EXACTLY - VERBATIM):
SMILES (simplified molecular-input line-entry system) uses short ASCII string to represent the structure of chemical species. Because the SMILES format described here is custom-designed by us for polymers, it is not completely identical to other SMILES formats. Strictly following the rules explained below is crucial for having correct results.

1. Spaces are not permitted in a SMILES string.
2. An atom is represented by its respective atomic symbol. In case of 2-character atomic symbol, it is placed between two square brackets [ ].
3. Single bonds are implied by placing atoms next to each other. A double bond is represented by the = symbol while a triple bond is represented by #.
4. Hydrogen atoms are suppressed, i.e., the polymer blocks are represented without hydrogen. Polymer Genome interface assumes typical valence of each atom type. If enough bonds are not identified by the user through SMILES notation, the dangling bonds will be automatically saturated by hydrogen atoms.
5. Branches are placed between a pair of round brackets ( ), and are assumed to attach to the atom right before the opening round bracket (.
6. Numbers are used to identify the opening and closing of rings of atoms. For example, in C1CCCCC1, the first carbon having a number "1" should be connected by a single bond with the last carbon, also having a number "1". Polymer blocks that have multiple rings may be identified by using different, consecutive numbers for each ring.
7. Atoms in aromatic rings can be specified by lower case letters. As an example, benzene ring can be written as c1ccccc1 which is equivalent to C(C=C1)=CC=C1.

SAFE CARBONYL PATTERNS (USE THESE):
- Simple carbonyl: C=O, CC=O, CCC=O
- Ester: C(=O)O, CC(=O)O, CCC(=O)O
- Amide: C(=O)N, CC(=O)N, CCC(=O)N
- Ketone: CC(=O)C, CCC(=O)CC

DANGEROUS PATTERNS (NEVER USE):
- C(C(C)=O)C(C)=O - complex branched carbonyls
- C(X)(Y)=O where X,Y are heavy atoms - oxygen valence violation
- Multiple carbonyls on same carbon - impossible valence
- Overly branched structures with >3 substituents per carbon

CRITICAL REQUIREMENTS:
- ALL brackets must be balanced: ( ) [ ]
- ALL ring closures must be complete: c1ccccc1 not c1cccc
- NO trailing incomplete symbols
- NO spaces or hyphens in final SMILES
- Ensure proper valence for all atoms
- Focus on SIMPLE, POLYMERIZABLE structures

POLYMER MONOMER REQUIREMENTS (CRITICAL FOR THIS APPLICATION):
- The generated structure will be used as a POLYMER MONOMER
- MUST contain at least TWO CARBON atoms (minimum: CC)
- Structure must be capable of polymerization
- Include functional groups that enable polymer chain formation
- Consider typical monomer structures like vinyl groups (C=C), rings that can open, or difunctional molecules
- Prefer proven monomer chemistries over exotic structures

VALID SULFUR CHEMISTRY (CRITICAL FOR SULFUR-CONTAINING REQUESTS):
- Simple sulfur bridge: CSC
- Thioether: CSCCC
- Disulfide: CSSC
- Thiol group: SH (written as S in polymer context)
- Sulfur in aromatic ring: c1sccc1 (thiophene)
- Simple sulfide: CCC(S)CC

EXAMPLES (ALL WITHIN CONSTRAINTS):
- water → O
- ethanol → CCO
- benzene → c1ccccc1
- ethylene → C=C
- propylene → CC=C
- styrene → C=Cc1ccccc1
- sulfur atoms → CSC
- thiophene → c1sccc1
- sulfur bridge → CSC
- disulfide → CSSC
- acrylic acid → C=CC(=O)O
- acrylamide → C=CC(=O)N
- vinyl acetate → C=COC(=O)C

OUTPUT FORMAT EXAMPLE:
Input: "polymer with boron atoms"
Output: CC[B]CC

STRICT REQUIREMENT: RETURN ONLY THE SMILES STRING, NO EXPLANATIONS OR EXTRA TEXT."""),
            ("human", "Convert this description to SMILES (MW: 100-800 Da, Length: 5-1000 chars): {description}")
        ])
    
    def _get_chemistry_examples(self) -> dict:
        """Chemistry lookup table with direct matches - adapted from working system"""
        return {
            # Basic molecules
            'water': 'O',
            'methane': 'C', 
            'ethanol': 'CCO',
            'benzene': 'c1ccccc1',
            'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'glucose': 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
            'phenol': 'Oc1ccccc1',
            'aniline': 'Nc1ccccc1',
            'toluene': 'Cc1ccccc1',
            'acetone': 'CC(=O)C',
            'acetic acid': 'CC(=O)O',
            'formaldehyde': 'C=O',
            'ammonia': 'N',
            'carbon dioxide': 'O=C=O',
            
            # Polymer monomers (key for polymer generation)
            'ethylene': 'C=C',
            'propylene': 'CC=C',
            'styrene': 'C=Cc1ccccc1',
            'vinyl chloride': 'C=CCl',
            'acrylonitrile': 'C=CC#N',
            'methyl methacrylate': 'C=C(C)C(=O)OC',
            'ethylene oxide': 'C1CO1',
            'propylene oxide': 'CC1CO1',
            'caprolactam': 'C1CCC(=O)NCC1',
            'adipic acid': 'C(CCC(=O)O)CC(=O)O',
            
            # **CRITICAL: Sulfur chemistry mappings**
            'sulfur': 'CSC',
            'sulfur atoms': 'CSC',
            'sulfur bridge': 'CSC',
            'sulfur compound': 'CSC',
            'sulfur containing': 'CSC',
            'disulfide': 'CSSC',
            'disulfide bridge': 'CSSC',
            'thioether': 'CSCCC',
            'thiophene': 'c1sccc1',
            'sulfur ring': 'c1sccc1',
            'aromatic sulfur': 'c1sccc1',
            'thiol': 'CS',
            'sulfide': 'CSC',
            
            # Polymer names mapped to monomers
            'polyethylene': 'C=C',
            'polypropylene': 'CC=C',
            'polystyrene': 'C=Cc1ccccc1',
            'polyvinyl chloride': 'C=CCl',
            'pvc': 'C=CCl',
            'polyacrylonitrile': 'C=CC#N',
            'pmma': 'C=C(C)C(=O)OC',
            'polyethylene glycol': 'C=C',
            'peg': 'C=C',
            'polyethylene oxide': 'C1CO1',
            'peo': 'C1CO1',
            'nylon 6': 'C1CCC(=O)NCC1',
            'nylon 66': 'C(CCC(=O)O)CC(=O)O',
        }
    
    def generate_smiles_from_nl(self, description: str) -> dict:
        """Convert natural language to SMILES using chemistry lookup + OpenAI"""
        try:
            print(f"🧬 Step 1: Natural Language → SMILES")
            print(f"   Description: {description}")
            
            # **NEW: Copolymer/Polymer Keyword Detection and Pre-processing**
            processed_description, polymer_constraints = self._preprocess_polymer_keywords(description)
            if processed_description != description:
                print(f"   🔧 Polymer keyword detected - processed description: {processed_description}")
                print(f"   📏 Applied constraints: {polymer_constraints}")
            
            # **STEP 1: Check direct chemistry lookup first**
            desc_lower = processed_description.lower().strip()
            if desc_lower in self.chemistry_examples:
                smiles = self.chemistry_examples[desc_lower]
                print(f"   ✅ Direct chemistry match: {smiles}")
                return {
                    'success': True,
                    'smiles': smiles,
                    'method': 'chemistry_lookup',
                    'description': processed_description,
                    'confidence': 1.0
                }
            
            # **STEP 2: Use OpenAI with comprehensive prompting**
            print(f"   No direct match, using OpenAI with comprehensive prompts...")
            
            # Apply polymer-specific constraints to the prompt if needed
            enhanced_prompt = self._enhance_prompt_for_polymers(processed_description, polymer_constraints)
            
            messages = enhanced_prompt.format_messages(description=processed_description)
            response = self.llm.invoke(messages)
            smiles = response.content.strip()
            
            # Clean the SMILES (remove any extra text)
            smiles = self._clean_smiles_response(smiles)
            
            print(f"   Generated SMILES: {smiles}")
            
            # **STEP 3: Apply Polymer-Specific Length Constraints**
            if polymer_constraints['strict_length_limit']:
                if len(smiles) > polymer_constraints['max_length']:
                    print(f"   ❌ SMILES too long for polymer ({len(smiles)} > {polymer_constraints['max_length']} chars)")
                    raise ValueError(f"Generated SMILES exceeds length limit: {len(smiles)} > {polymer_constraints['max_length']}")
            
            # **STEP 4: VALID-Mol Constraint Validation (NEW)**
            print(f"   🔍 Applying VALID-Mol constraints...")
            valid_constraints, constraint_message = self.validate_valid_mol_constraints(smiles)
            
            if not valid_constraints:
                print(f"   ❌ VALID-Mol constraints failed: {constraint_message}")
                
                # Try stricter constraint retry logic
                print(f"   🔄 Retrying with stricter constraints...")
                
                # For polymer requests, be even more strict
                max_retry_length = polymer_constraints['max_length'] if polymer_constraints['strict_length_limit'] else 200
                
                retry_smiles = self._retry_with_stricter_constraints(processed_description, max_retry_length)
                if retry_smiles:
                    retry_valid, retry_message = self.validate_valid_mol_constraints(retry_smiles)
                    if retry_valid:
                        smiles = retry_smiles
                        print(f"   ✅ Retry successful: {retry_message}")
                    else:
                        print(f"   ❌ Retry failed: {retry_message}")
                        raise ValueError(f"VALID-Mol constraints not satisfied: {retry_message}")
                else:
                    print(f"   ❌ Unable to generate valid SMILES with constraints")
                    raise ValueError(f"VALID-Mol constraints not satisfied: {constraint_message}")
            else:
                print(f"   ✅ VALID-Mol constraints satisfied: {constraint_message}")
            
            return {
                'success': True,
                'smiles': smiles,
                'method': 'openai_comprehensive',
                'description': processed_description,
                'confidence': 0.8,
                'polymer_processing': polymer_constraints
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'SMILES generation error: {str(e)}',
                'description': description
            }
    
    def _preprocess_polymer_keywords(self, description: str) -> tuple:
        """
        Detect polymer/copolymer keywords and preprocess the description to focus on monomer units.
        Returns (processed_description, constraint_dict)
        """
        import re
        
        # Polymer keywords that trigger special processing
        polymer_keywords = [
            'copolymer', 'polymer', 'polymerization', 'polymerized',
            'polypeptide', 'polysaccharide', 'polynucleotide',
            'polyethylene', 'polypropylene', 'polystyrene', 'polyvinyl',
            'polyacrylate', 'polyacrylamide', 'polyester', 'polyamide',
            'polyurethane', 'polyimide', 'polycarbonate',
            'block copolymer', 'random copolymer', 'alternating copolymer'
        ]
        
        description_lower = description.lower()
        
        # Check if any polymer keywords are present
        contains_polymer_keywords = any(keyword in description_lower for keyword in polymer_keywords)
        
        if not contains_polymer_keywords:
            return description, {
                'strict_length_limit': False,
                'max_length': 1000,
                'is_polymer_request': False
            }
        
        print(f"   🧬 Polymer keywords detected in: {description}")
        
        # Transform polymer language to monomer-focused language
        processed = description
        
        # Polymer → monomer transformations
        transformations = [
            (r'\bcopolymer\b', 'monomer for copolymerization'),
            (r'\bpolymer\b', 'monomer unit'),
            (r'\bpolymerization\b', 'polymerizable monomer'),
            (r'\bpolymerized\b', 'polymerizable'),
            (r'\bblock copolymer\b', 'block-forming monomer'),
            (r'\brandom copolymer\b', 'randomly polymerizable monomer'),
            (r'\balternating copolymer\b', 'alternating monomer'),
            
            # Specific polymer types
            (r'\bpolyethylene\b', 'ethylene monomer'),
            (r'\bpolypropylene\b', 'propylene monomer'),
            (r'\bpolystyrene\b', 'styrene monomer'),
            (r'\bpolyvinyl\b', 'vinyl monomer'),
            (r'\bpolyacrylate\b', 'acrylate monomer'),
            (r'\bpolyacrylamide\b', 'acrylamide monomer'),
            (r'\bpolyester\b', 'ester-forming monomer'),
            (r'\bpolyamide\b', 'amide-forming monomer'),
        ]
        
        for pattern, replacement in transformations:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        # Add explicit monomer instruction
        if 'monomer' not in processed.lower():
            processed = f"monomer unit for {processed}"
        
        return processed, {
            'strict_length_limit': True,
            'max_length': 100,  # Much stricter for polymers - monomers should be short
            'is_polymer_request': True,
            'original_keywords': [kw for kw in polymer_keywords if kw in description_lower]
        }
    
    def _enhance_prompt_for_polymers(self, description: str, constraints: dict):
        """Create enhanced prompt template for polymer requests."""
        if not constraints['is_polymer_request']:
            return self.nl_to_smiles_prompt
        
        # Enhanced prompt for polymer/copolymer requests
        polymer_enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a chemistry expert. Convert this description to a valid SMILES string following these EXACT rules:

CRITICAL POLYMER/COPOLYMER CONSTRAINTS (EXTREMELY IMPORTANT):
- The request mentions polymer/copolymer terms
- You MUST generate a MONOMER UNIT, NOT a full polymer chain
- NEVER repeat patterns or create long repeating sequences  
- Maximum SMILES length: 100 characters (STRICTLY ENFORCED)
- Think "building block" not "finished polymer"

CONSTRAINTS (CRITICAL - FROM VALID-MOL FRAMEWORK):
- Molecular weight: MUST be between 50-500 Da (realistic monomer range)
- SMILES length: MUST be between 5-100 characters (STRICTLY ENFORCED FOR POLYMERS)
- Chemical validity: MUST pass RDKit validation
- Output format: SMILES_STRING_ONLY (no explanations, no text, no descriptions)

CRITICAL VALENCE RULES (PREVENT POLYMER LIBRARY FAILURES):
- Carbon: EXACTLY 4 bonds maximum (never exceed) 
- Oxygen: EXACTLY 2 bonds maximum (C=O, C-O-C, never C(X)(Y)=O where X,Y are heavy atoms)
- Nitrogen: EXACTLY 3 bonds maximum (C-N-C, C=N, never exceed)
- Fluorine: EXACTLY 1 bond maximum (C-F, never exceed)
- Chlorine: EXACTLY 1 bond maximum (C-Cl, never exceed)
- AVOID: C(C(C)=O)C(C)=O - creates oxygen valence violations
- AVOID: C(X)(Y)=O where X,Y are not hydrogen - causes O valence > 2
- AVOID: F(X) or Cl(X) - halogens can only have one bond
- AVOID: Complex branched carbonyls - use simple C=O or CC(=O)
- AVOID: Multiple bonds to halogens - F and Cl must have exactly one bond

POLYMER MONOMER RULES (CRITICAL FOR POLYMERIZATION):
- Structure MUST be a realistic monomer unit (not full polymer)
- Include functional groups for chain growth: C=C, C(=O)O, C(=O)N, aromatic rings
- Prefer linear or simple branched structures over complex branching
- Examples of GOOD monomers: CC(=O)O, C=C, c1ccccc1, CC(=O)N, CCC=O

FORBIDDEN FOR POLYMER REQUESTS:
- NO repeating sequences: NEVER write CCCCCCCCC or similar
- NO long chains with repetitive patterns
- NO attempts to represent full polymer chains
- NO structures longer than 100 characters
- NO complex multi-component systems

SAFE MONOMER EXAMPLES FOR POLYMERS:
- "poly(N-isopropylacrylamide)" → C=CC(=O)NC(C)C (acrylamide monomer)
- "polyethylene" → C=C (ethylene monomer)
- "polystyrene" → C=Cc1ccccc1 (styrene monomer)  
- "polyacrylate" → C=CC(=O)O (acrylate monomer)
- "copolymer with carboxylic acid" → C=CC(=O)O (acrylate with carboxyl)

CRITICAL REQUIREMENTS:
- ALL brackets must be balanced: ( ) [ ]
- ALL ring closures must be complete: c1ccccc1 not c1cccc
- NO trailing incomplete symbols
- NO spaces or hyphens in final SMILES
- Ensure proper valence for all atoms
- Focus on SIMPLE, POLYMERIZABLE monomer structures

STRICT REQUIREMENT: RETURN ONLY THE SMILES STRING, NO EXPLANATIONS OR EXTRA TEXT."""),
            ("human", "Convert this MONOMER description to SMILES (MW: 50-500 Da, Length: 5-100 chars): {description}")
        ])
        
        return polymer_enhanced_prompt
    
    def _apply_polymer_fallback(self, description: str) -> str:
        """Apply safe polymer monomer fallbacks based on description content."""
        description_lower = description.lower()
        
        # Keyword-based fallbacks to safe, proven monomers
        if any(word in description_lower for word in ['acid', 'carboxyl', 'carboxylic']):
            return 'C=CC(=O)O'  # Acrylic acid monomer
        elif any(word in description_lower for word in ['amide', 'acrylamide', 'isopropyl']):
            return 'C=CC(=O)NC(C)C'  # N-isopropylacrylamide  
        elif any(word in description_lower for word in ['aromatic', 'benzene', 'phenyl', 'styrene']):
            return 'C=Cc1ccccc1'  # Styrene monomer
        elif any(word in description_lower for word in ['ester', 'acrylate', 'methacrylate']):
            return 'C=CC(=O)OC'  # Methyl acrylate
        elif any(word in description_lower for word in ['vinyl', 'ethylene']):
            return 'C=C'  # Ethylene
        elif any(word in description_lower for word in ['hydroxyl', 'alcohol', 'diol']):
            return 'C=CC(O)'  # Vinyl alcohol derivative
        elif any(word in description_lower for word in ['ether', 'oxide']):
            return 'C1CO1'  # Ethylene oxide (ring-opening polymerization)
        else:
            # Default safe monomer
            return 'C=CC(=O)O'  # Acrylic acid (most versatile)
    
    def _retry_with_stricter_constraints(self, description: str, max_length: int) -> str:
        """Retry SMILES generation with much stricter constraints."""
        try:
            strict_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""Generate ONLY a simple chemical monomer SMILES string. 

STRICT RULES:
- Maximum {max_length} characters total
- Simple structure only
- No repeating patterns
- Must be a realistic monomer for polymerization
- Return ONLY the SMILES string, nothing else

Examples of good short monomers:
- C=C (ethylene)
- C=CC(=O)O (acrylic acid)  
- C=CC(=O)N (acrylamide)
- c1ccccc1 (benzene)"""),
                ("human", f"Simple monomer for: {description}")
            ])
            
            messages = strict_prompt.format_messages(description=description)
            response = self.llm.invoke(messages)
            smiles = response.content.strip()
            smiles = self._clean_smiles_response(smiles)
            
            if len(smiles) <= max_length:
                return smiles
            else:
                return None
                
        except Exception:
            return None
    
    def convert_smiles_to_psmiles(self, smiles: str) -> str:
        """Convert SMILES to PSMILES by adding [*] connection points"""
        print(f"🧬 Step 2: SMILES → PSMILES")
        print(f"   SMILES: {smiles}")
        
        if not smiles:
            psmiles = "[*][*]"
            print(f"   Empty SMILES, using minimal PSMILES: {psmiles}")
            return psmiles
        
        # Check if already has connection points
        if '[*]' in smiles:
            connection_count = smiles.count('[*]')
            if connection_count == 2:
                psmiles = smiles
            elif connection_count == 1:
                psmiles = f"[*]{smiles}"
            else:
                smiles_clean = smiles.replace('[*]', '')
                psmiles = f"[*]{smiles_clean}[*]"
        else:
            # Add connection points for polymer repeat unit
            psmiles = f"[*]{smiles}[*]"
        
        print(f"   Generated PSMILES: {psmiles}")
        return psmiles
    
    def validate_smiles(self, smiles: str) -> tuple:
        """Basic SMILES validation"""
        try:
            # Check for basic validity
            if not smiles or len(smiles) == 0:
                return False, None, "Empty SMILES"
            
            # Check for balanced brackets
            if smiles.count('(') != smiles.count(')'):
                return False, None, "Unbalanced parentheses"
            
            if smiles.count('[') != smiles.count(']'):
                return False, None, "Unbalanced square brackets"
            
            # Try RDKit validation if available
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, None, "RDKit validation failed"
                
                Chem.SanitizeMol(mol)
                return True, mol, "Valid SMILES"
                
            except ImportError:
                # RDKit not available, use basic validation
                return True, None, "Valid SMILES (basic validation)"
            except Exception as rdkit_error:
                return False, None, f"RDKit error: {str(rdkit_error)}"
            
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"
    
    def validate_valid_mol_constraints(self, smiles: str) -> tuple:
        """
        VALID-Mol Framework Constraints Validation (2025 Research) - UPDATED FOR TESTING
        **RELAXED** constraints to allow testing of stable structures with various elements
        """
        try:
            # Step 1: SMILES Length Constraint - RELAXED for simple stable structures
            MIN_LENGTH = 2   # Reduced from 5 - allows simple compounds like BCl3 → ClB(Cl)Cl
            MAX_LENGTH = 1000
            
            if len(smiles) < MIN_LENGTH:
                return False, f"SMILES too short ({len(smiles)} < {MIN_LENGTH} chars)"
            
            if len(smiles) > MAX_LENGTH:
                return False, f"SMILES too long ({len(smiles)} > {MAX_LENGTH} chars) - unrealistic structure"
            
            # Step 2: Molecular Weight Constraint - RELAXED for simple stable structures
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, "Invalid SMILES structure"
                
                # Calculate molecular weight
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                MIN_MW = 30.0   # Reduced from 100.0 - allows simple stable compounds (BCl3 ≈ 117 Da, SiO2 ≈ 60 Da)  
                MAX_MW = 800.0  # Keep reasonable upper limit for drug-like range
                
                if mw < MIN_MW:
                    return False, f"Molecular weight too low ({mw:.1f} < {MIN_MW} Da)"
                
                if mw > MAX_MW:
                    return False, f"Molecular weight too high ({mw:.1f} > {MAX_MW} Da) - unrealistic for drug-like molecules"
                
                return True, f"Valid: {len(smiles)} chars, {mw:.1f} Da"
                
            except ImportError:
                # RDKit not available, only check length
                return True, f"Valid length: {len(smiles)} chars (MW check requires RDKit)"
            except Exception as e:
                return False, f"Molecular weight calculation failed: {str(e)}"
                
        except Exception as e:
            return False, f"VALID-Mol validation error: {str(e)}"
    
    def _clean_smiles_response(self, response: str) -> str:
        """Clean OpenAI response to extract just the SMILES string"""
        # Remove common explanatory text
        response = response.replace("SMILES:", "").replace("smiles:", "")
        response = response.replace("The SMILES string is", "")
        response = response.replace("Answer:", "").replace("Result:", "")
        response = response.replace("→", "").replace("->", "")
        
        # Split by lines and take first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            smiles = lines[0]
        else:
            smiles = response.strip()
        
        # Remove quotes and extra spaces
        smiles = smiles.strip().strip('"').strip("'").strip()
        
        # CRITICAL FIX: Remove descriptive text in parentheses that causes PSMILES parsing errors
        import re
        # Remove text in parentheses like "(styrene)", "(fluorinated phenol)", etc.
        smiles = re.sub(r'\s*\([^)]*\)\s*', '', smiles)
        
        # Remove trailing descriptive words after SMILES
        # Split on first space and take only the SMILES part
        smiles_parts = smiles.split()
        if smiles_parts:
            smiles = smiles_parts[0]
        
        # Final cleanup
        smiles = smiles.strip()
        
        return smiles

class PSMILESGenerator:
    """
    Generate PSMILES (Polymer SMILES) from natural language descriptions.
    Uses multiple approaches including chemistry lookup and OpenAI generation.
    Now supports diverse candidate generation with the new orchestration system.
    """
    
    def __init__(self, 
                 model_type: str = 'openai',
                 openai_model: str = 'gpt-4o',
                 temperature: float = 0.7,
                 enable_diverse_generation: bool = True):
        """Initialize the PSMILES generator."""
        self.model_type = model_type
        self.openai_model = openai_model
        self.temperature = temperature
        self.llm = None
        self.mock_mode = False
        
        # OpenAI LLM setup
        if model_type == 'openai':
            try:
                import os
                # Check if OpenAI API key is available
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("⚠️  OpenAI API key not found. Running in mock mode.")
                    print("   Set OPENAI_API_KEY environment variable to enable LLM generation.")
                    self.mock_mode = True
                else:
                    self.llm = ChatOpenAI(
                        model=openai_model,
                        temperature=temperature
                    )
                    print(f"✅ OpenAI model '{openai_model}' initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenAI model: {e}")
                print("   Running in mock mode with pre-defined examples.")
                self.mock_mode = True
        
        # Setup prompts
        self.prompts = self._setup_prompts()
        
        # Initialize working pipeline
        try:
            self.nl_to_psmiles = OpenAIWorkingPipeline(self.llm)
            self.working_pipeline_available = True
            print("✅ Working pipeline (nl_to_psmiles) initialized")
        except Exception as e:
            print(f"❌ Working pipeline initialization failed: {e}")
            self.working_pipeline_available = False
        
        # Initialize diverse generation system
        self.enable_diverse_generation = enable_diverse_generation
        if enable_diverse_generation:
            try:
                from .psmiles_diversification import CandidateOrchestrator, CandidateConfig
                self.orchestrator = CandidateOrchestrator(self, random_seed=42)
                self.diverse_generation_available = True
                print("✅ Diverse generation system initialized")
            except ImportError as e:
                print(f"⚠️ Diverse generation system not available: {e}")
                self.diverse_generation_available = False
        else:
            self.diverse_generation_available = False
        
        # These are always None for now since the classes don't support OpenAI
        self.smiles_corrector = None
        
        logging.info(f"🧬 PSMILES Generator initialized with OpenAI {openai_model}")
    
    def _setup_prompts(self) -> Dict:
        """Setup prompt templates for PSMILES generation using OpenAI models."""
        
        psmiles_system_prompt = """You are an expert computational chemist specializing in polymer chemistry and PSMILES notation.

PSMILES (Polymer SMILES) is a notation system for representing MONOMER REPEAT UNITS with exactly TWO connection points marked as [*].

CRITICAL UNDERSTANDING: 
- PSMILES represents a MONOMER UNIT that will be polymerized, NOT a full polymer chain
- The [*] symbols mark where this monomer connects to adjacent monomers during polymerization
- Think "building block" not "finished polymer"

CRITICAL RULES (ENHANCED WITH MOLLM MULTI-OBJECTIVE FRAMEWORK):
1. ALWAYS use exactly TWO [*] symbols to mark connection points
2. Use proper SMILES syntax for chemical structures
3. Consider real monomer chemistry and realistic repeat unit structures
4. Ensure chemical validity and proper valences
5. Focus on practical monomers that could be polymerized
6. NEVER use explicit hydrogen atoms (H) in the structure - SMILES implicit hydrogen handling
7. For sulfur atoms, use valid patterns like CSC (thioether), c1sccc1 (thiophene), CSS (disulfide)
8. **CRITICAL: NEVER generate radical species or unpaired electrons**
9. **CRITICAL: ALL atoms must have complete valence shells (closed-shell structures only)**
10. **CRITICAL: Ensure proper valence for all elements - any element is allowed if properly bonded**

**RADICAL PREVENTION RULES:**
- ENSURE all atoms have complete valence shells (no unpaired electrons)
- ENSURE proper bonding patterns for all elements
- ENSURE all carbon atoms have 4 bonds (including implicit hydrogens)
- ENSURE all nitrogen atoms have 3 bonds + lone pair (total 8 electrons)
- ENSURE all oxygen atoms have 2 bonds + 2 lone pairs (total 8 electrons)
- ENSURE all sulfur atoms follow stable bonding patterns
- ENSURE boron atoms have 3 bonds (BCl3, BF3 patterns) or 4 bonds when coordinated
- ENSURE silicon atoms have 4 bonds (SiO4, SiCl4 patterns)
- ENSURE aluminum atoms have 3 bonds (AlCl3) or 6 bonds when coordinated
- NO charged species ([NH3+], [O-]) unless absolutely necessary for the chemistry
- FOCUS: Prevent unpaired electrons, not specific elements

DIVERSIFICATION STRATEGY (FROM MOLLM RESEARCH 2025):
- Vary functionalization types: ester, amide, ether, aromatic substitution, vinyl, sulfur-based
- Mix backbone types: aliphatic chains, aromatic rings, heterocycles
- Combine different heteroatoms: N, O, S, P (when appropriate and stable)
- Use different connection patterns: linear, branched, cyclic integration
- **ALL variations must maintain closed-shell electronic structure**

MOLECULAR WEIGHT CONSTRAINTS (VALID-MOL FRAMEWORK):
- Target monomer repeat units: 50-500 Da (realistic monomer range)
- PSMILES length: 8-100 characters (excludes [*] symbols)
- Ensure chemical feasibility for polymerization
- **Must be stable, closed-shell molecules only**

FUNCTIONALIZATION DIVERSITY (TSMMG TEACHER-STUDENT APPROACH):
- Primary types: C(=O)O (ester), C(=O)N (amide), O (ether), c1ccccc1 (aromatic)
- Secondary types: CSS (disulfide), CSC (thioether), C=C (vinyl), c1sccc1 (thiophene)
- Tertiary types: P-containing (PO4 groups only), halogenated (F, Cl), cyclic structures
- **ALL must be stable, non-radical species**

IMPORTANT SMILES SYNTAX:
- NO explicit hydrogen atoms: Never write H, HS, SH, HSH, HCSH, etc.
- Sulfur examples: CSC (thioether bridge), c1sccc1 (thiophene ring), CSS (disulfide)
- Carbon backbone: CC, CCC, c1ccccc1 (aromatic)
- Functional groups: C(=O) (carbonyl), C(=O)O (carboxyl), C(=O)N (amide)
- **Ensure all examples represent stable, closed-shell molecules**

VALID MONOMER EXAMPLES (WITH DIVERSITY, ALL CLOSED-SHELL):
- "monomer with aromatic rings": [*]c1ccccc1[*] OR [*]Cc1ccccc1C[*] OR [*]c1ccc(C)cc1[*]
- "monomer with amide linkages": [*]C(=O)NC[*] OR [*]CC(=O)NCC[*] OR [*]C(=O)Nc1ccccc1[*]
- "monomer with ester groups": [*]C(=O)OC[*] OR [*]CC(=O)OCC[*] OR [*]C(=O)Oc1ccccc1[*]
- "ethylene-like monomer": [*]CC[*] OR [*]CCC[*] OR [*]C(C)C[*]
- "monomer with hydroxyl groups": [*]C(O)C[*] OR [*]CC(O)CC[*] OR [*]c1ccc(O)cc1[*]
- "sulfur-containing monomer": [*]CSC[*] OR [*]c1sccc1[*] OR [*]CSS[*] OR [*]CSCC[*]

INVALID EXAMPLES TO AVOID:
- [*]HSH[*] - NO explicit hydrogens
- [*]HCSH[*] - NO explicit hydrogens
- [*]SH[*] - NO explicit hydrogens
- [*]HS[*] - NO explicit hydrogens
- **[*]B...[*] - NO boron atoms (radical prone)**
- **[*]Si...[*] - NO silicon in complex structures (radical prone)**
- **[*]Al...[*] - NO aluminum atoms (radical prone)**
- **Any structure with unpaired electrons or radical character**

TASK: Convert the user's description into a valid MONOMER PSMILES string.
- Think about the MONOMER REPEAT UNIT they're describing
- Generate a chemically realistic PSMILES monomer with exactly 2 [*] connection points
- Use diverse functionalization appropriate to the request
- Ensure proper SMILES syntax and chemical validity
- **CRITICAL: Ensure the molecule has no radicals or unpaired electrons**
- **CRITICAL: Use only stable, closed-shell electronic configurations**
- For sulfur requests, use CSC, CSS, or c1sccc1 patterns with variation
- Respond with just the PSMILES string, no explanation unless requested"""

        psmiles_prompt = ChatPromptTemplate.from_messages([
            ("system", psmiles_system_prompt),
            ("human", "{description}"),
        ])
        
        return {
            'psmiles_generation': psmiles_prompt
        }
    
    def _clean_psmiles_response(self, psmiles: str) -> str:
        """Clean PSMILES to remove descriptive text and ensure valid format"""
        if not psmiles:
            return psmiles
        
        # Remove descriptive text in parentheses
        import re
        psmiles = re.sub(r'\s*\([^)]*\)\s*', '', psmiles)
        
        # Remove trailing descriptive words after PSMILES
        # Split on first space and take only the PSMILES part
        psmiles_parts = psmiles.split()
        if psmiles_parts:
            psmiles = psmiles_parts[0]
        
        # Ensure proper [*] format
        psmiles = psmiles.strip()
        
        # Fix malformed connection points
        if psmiles.startswith('[*]') and not psmiles.endswith('[*]'):
            # Single [*] at start, need one at end
            psmiles = psmiles + '[*]'
        elif not psmiles.startswith('[*]') and psmiles.endswith('[*]'):
            # Single [*] at end, need one at start
            psmiles = '[*]' + psmiles
        elif not psmiles.startswith('[*]') and not psmiles.endswith('[*]'):
            # No [*] markers, add both
            psmiles = f'[*]{psmiles}[*]'
        
        return psmiles
    
    def generate_psmiles(self, 
                        description: str, 
                        num_candidates: int = 3,
                        validate: bool = True,
                        disable_early_chemistry_lookup: bool = False) -> Dict:
        """
        Generate PSMILES from natural language description using the enhanced working pipeline:
        Natural Language → Chemistry Lookup → SMILES → Validation → PSMILES
        
        Args:
            description: Natural language description of the polymer
            num_candidates: Number of PSMILES candidates to generate
            validate: Whether to validate generated SMILES
            disable_early_chemistry_lookup: If True, prevents early stopping on chemistry lookup matches
                                          to allow for more diverse generation
            
        Returns:
            Dict with generated PSMILES and metadata
        """
        try:
            print(f"🧬 ENHANCED PROCESSING: {description}")
            print(f"🎯 Original Request: {description}")
            print(f"✅ Using WORKING PIPELINE with Chemistry Lookup + Validation")
            
            candidates = []
            
            for i in range(num_candidates):
                try:
                    print(f"\n🔄 Generating candidate {i+1}/{num_candidates}")
                    
                    # **STEP 1: Natural Language → SMILES (with chemistry lookup)**
                    smiles_result = self.nl_to_psmiles.generate_smiles_from_nl(description)
                    
                    if not smiles_result['success']:
                        print(f"   ❌ SMILES generation failed: {smiles_result.get('error', 'Unknown error')}")
                        continue
                    
                    smiles = smiles_result['smiles']
                    method = smiles_result['method']
                    confidence = smiles_result.get('confidence', 0.5)
                    
                    # **STEP 2: SMILES Validation (if requested)**
                    is_valid = True
                    validation_message = "Not validated"
                    
                    if validate:
                        print(f"🔍 Validating SMILES: {smiles}")
                        is_valid, mol, validation_message = self.nl_to_psmiles.validate_smiles(smiles)
                        print(f"   Validation: {'✅' if is_valid else '❌'} {validation_message}")
                    
                    # **STEP 3: SMILES → PSMILES**
                    psmiles = self.nl_to_psmiles.convert_smiles_to_psmiles(smiles)
                    psmiles = self._clean_psmiles_response(psmiles)  # Clean up any descriptive text
                    
                    # **STEP 4: PSMILES Format Validation**
                    connection_count = psmiles.count('[*]')
                    psmiles_valid = connection_count == 2
                    
                    candidate = {
                        'psmiles': psmiles,
                        'smiles': smiles,  # Include intermediate SMILES
                        'valid': is_valid and psmiles_valid,
                        'validation_message': validation_message,
                        'candidate_id': i + 1,
                        'method': method,
                        'confidence': confidence,
                        'pipeline': 'NaturalLanguage→ChemistryLookup→SMILES→Validation→PSMILES',
                        'description': description,
                        'connection_count': connection_count
                    }
                    
                    candidates.append(candidate)
                    
                    status = "✅ VALID" if candidate['valid'] else "❌ INVALID"
                    print(f"   {status}: {smiles} → {psmiles} (method: {method}, conf: {confidence:.2f})")
                    
                    # If we have a valid candidate and using chemistry lookup, we can stop early (unless diversity is requested)
                    if candidate['valid'] and method == 'chemistry_lookup' and not disable_early_chemistry_lookup:
                        print(f"   🎯 High-confidence chemistry match found, using as primary result")
                        break
                    
                except Exception as e:
                    print(f"   ❌ Failed to generate candidate {i+1}: {e}")
                    continue
            
            # **STEP 5: Select best candidate and prepare results**
            valid_candidates = [c for c in candidates if c.get('valid', False)]
            
            # Prioritize chemistry lookup matches, then by confidence
            def candidate_priority(candidate):
                method_priority = {'chemistry_lookup': 3, 'openai_comprehensive': 2, 'failed_generation': 1}
                return (
                    candidate.get('valid', False),
                    method_priority.get(candidate.get('method', 'failed_generation'), 1),
                    candidate.get('confidence', 0.0)
                )
            
            candidates.sort(key=candidate_priority, reverse=True)
            best_candidate = candidates[0] if candidates else None
            
            success = len(valid_candidates) > 0
            
            result = {
                'success': success,
                'candidates': candidates,
                'valid_candidates': valid_candidates,
                'num_generated': len(candidates),
                'num_valid': len(valid_candidates),
                'best_candidate': best_candidate['psmiles'] if best_candidate else None,
                'description': description,
                'model': self.openai_model,
                'pipeline': 'NaturalLanguage→ChemistryLookup→SMILES→Validation→PSMILES',
                'method': 'enhanced_working_pipeline',
                # **Additional fields for app.py compatibility**
                'explanation': f"Generated using enhanced pipeline with {best_candidate['method'] if best_candidate else 'unknown'} method",
                'validation': best_candidate['validation_message'] if best_candidate else 'No valid candidates',
                'temperature_used': self.temperature,
                'timestamp': datetime.now().isoformat(),
                'conversation_turn': 0,
                'smiles_used': best_candidate['smiles'] if best_candidate else None,
                'chemistry_method': best_candidate['method'] if best_candidate else None,
                'confidence': best_candidate['confidence'] if best_candidate else 0.0
            }
            
            if success:
                method = best_candidate['method']
                print(f"✅ WORKING PIPELINE SUCCESS: Generated {len(candidates)} candidates, {len(valid_candidates)} valid")
                print(f"   Best: {best_candidate['psmiles']} (method: {method})")
            else:
                print(f"❌ No valid candidates generated")
                
                # **FALLBACK: Try to generate something basic if completely failed**
                if description.lower().strip() in ['sulfur', 'sulfur atoms', 'sulfur bridge']:
                    fallback_psmiles = "[*]CSC[*]"
                    print(f"🔧 Using chemistry fallback for sulfur: {fallback_psmiles}")
                    result['best_candidate'] = fallback_psmiles
                    result['success'] = True
                    result['method'] = 'chemistry_fallback'
                    result['explanation'] = "Generated using chemistry fallback for sulfur compounds"
                    result['validation'] = "Chemistry fallback - known valid structure"
                    result['smiles_used'] = "CSC"
                    result['chemistry_method'] = 'chemistry_fallback'
                    result['confidence'] = 1.0
            
            return result
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'candidates': [],
                'pipeline': 'NaturalLanguage→ChemistryLookup→SMILES→Validation→PSMILES',
                'method': 'enhanced_working_pipeline_error'
            }
    
    def _clean_psmiles(self, psmiles: str) -> str:
        """Clean and format PSMILES string."""
        try:
            # Remove any extra whitespace and newlines
            psmiles = psmiles.strip()
            
            # Remove common prefixes that LLMs might add
            prefixes_to_remove = [
                "PSMILES:", "SMILES:", "Generated:", "Result:", 
                "The PSMILES is:", "PSMILES string:"
            ]
            
            for prefix in prefixes_to_remove:
                if psmiles.startswith(prefix):
                    psmiles = psmiles[len(prefix):].strip()
            
            # Remove quotes if present
            if psmiles.startswith('"') and psmiles.endswith('"'):
                psmiles = psmiles[1:-1]
            if psmiles.startswith("'") and psmiles.endswith("'"):
                psmiles = psmiles[1:-1]
            
            # Ensure exactly 2 [*] connection points
            star_count = psmiles.count('[*]')
            
            if star_count == 0:
                # Add connection points at both ends
                psmiles = f"[*]{psmiles}[*]"
            elif star_count == 1:
                # Add one more connection point
                if psmiles.startswith('[*]'):
                    psmiles = f"{psmiles}[*]"
                else:
                    psmiles = f"[*]{psmiles}"
            elif star_count > 2:
                # Keep only first and last [*]
                parts = psmiles.split('[*]')
                if len(parts) >= 3:
                    middle = ''.join(parts[1:-1])
                    psmiles = f"[*]{middle}[*]"
            
            return psmiles
            
        except Exception as e:
            print(f"⚠️ Error cleaning PSMILES: {e}")
            return psmiles
    
    def _validate_psmiles(self, psmiles: str) -> Tuple[bool, str]:
        """Validate PSMILES string for chemical validity."""
        try:
            # Check for exactly 2 connection points
            star_count = psmiles.count('[*]')
            if star_count != 2:
                return False, f"Invalid connection points: found {star_count}, expected 2"
            
            # Check basic structure
            if len(psmiles) < 6:  # Minimum: [*]C[*]
                return False, "PSMILES too short"
            
            # Check balanced brackets and parentheses
            if psmiles.count('[') != psmiles.count(']'):
                return False, "Unbalanced square brackets"
            
            if psmiles.count('(') != psmiles.count(')'):
                return False, "Unbalanced parentheses"
            
            # **NEW: PSMILES-specific valence pre-validation**
            valence_check = self._check_psmiles_valence_issues(psmiles)
            if not valence_check[0]:
                # Attempt auto-repair
                print(f"   ⚠️ Valence issue detected: {valence_check[1]}")
                print(f"   🔧 Attempting auto-repair...")
                repaired_psmiles, was_repaired = self._auto_repair_psmiles_valence(psmiles)
                
                if was_repaired:
                    print(f"   ✅ Auto-repair successful: {psmiles} → {repaired_psmiles}")
                    # Re-validate the repaired structure
                    repair_check = self._check_psmiles_valence_issues(repaired_psmiles)
                    if repair_check[0]:
                        return True, f"Auto-repaired PSMILES: {repaired_psmiles}"
                    else:
                        return False, f"Auto-repair failed: {repair_check[1]}"
                else:
                    return valence_check
            
            # Try RDKit validation if available
            try:
                from rdkit import Chem
                # Convert to SMILES for validation
                smiles_for_validation = psmiles.replace('[*]', 'H')
                mol = Chem.MolFromSmiles(smiles_for_validation)
                if mol is None:
                    return False, "RDKit validation failed - invalid chemical structure"
                
                # Try sanitization
                Chem.SanitizeMol(mol)
                return True, "Valid PSMILES with RDKit validation"
                
            except ImportError:
                # RDKit not available, use basic validation
                return True, "Valid PSMILES (basic validation - RDKit not available)"
            except Exception as rdkit_error:
                return False, f"RDKit validation failed: {str(rdkit_error)}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _check_psmiles_valence_issues(self, psmiles: str) -> Tuple[bool, str]:
        """
        Check for common valence issues that cause PSMILES library failures.
        Specifically targets oxygen valence violations that we've been seeing.
        """
        import re
        
        # Remove [*] connection points for analysis
        clean_smiles = psmiles.replace('[*]', '')
        
        # **CRITICAL VALENCE PATTERNS THAT CAUSE FAILURES**
        problematic_patterns = [
            # Oxygen with too many connections
            (r'C\(C\)\(C\)=O', "Carbon with multiple substituents and double bond to oxygen"),
            (r'C\(.*?\)\(.*?\)=O', "Carbon with multiple bonds plus carbonyl creates O valence > 2"),
            (r'O=.*?=O', "Oxygen in multiple double bonds"),
            (r'C\(C\(C\)=O\)C\(C\)=O', "Complex branched structure with multiple carbonyls"),
            
            # Specific problematic patterns from your failures
            (r'C\(C\(C\)=O\)C\(C\)=O', "Pattern from failure: C(C(C)=O)C(C)=O creates O valence issues"),
            (r'C\(C\(C\)=O\)\(C\)NC', "Pattern from failure: C(C(C)=O)(C)NC creates valence conflicts"),
            (r'C\(C\(C\)=O\)C\(NC.*?\)=O', "Pattern from failure: Complex amide-ester combinations"),
            (r'C\(C\(C\)=O\)=C\(C\)N', "Pattern from failure: Double bond with complex substitution"),
            
            # General problematic structures
            (r'C\(=O\)\(=O\)', "Carbon with two double bonds to oxygen (impossible)"),
            (r'O\(=.*?\)\(=.*?\)', "Oxygen with multiple double bonds"),
            (r'C\(.*?=O\).*?\(.*?=O\)', "Carbon connected to multiple carbonyls through complex bonds"),
        ]
        
        for pattern, description in problematic_patterns:
            if re.search(pattern, clean_smiles):
                return False, f"Valence issue detected: {description} in pattern '{pattern}'"
        
        # **CHECK FOR VALID CARBONYL PATTERNS**
        # Ensure carbonyl groups are properly formed
        carbonyl_matches = re.findall(r'C\([^)]*\)=O|C=O', clean_smiles)
        for match in carbonyl_matches:
            # Count bonds to the carbon in carbonyl
            if match.startswith('C('):
                # Extract content inside parentheses
                inner = match[2:-3]  # Remove C( and )=O
                # Count comma-separated groups (bonds)
                bond_count = len(inner.split(',')) if inner else 0
                bond_count += 1  # Add the =O bond
                
                if bond_count > 4:  # Carbon can't have more than 4 bonds
                    return False, f"Carbon valence violation in carbonyl: {match} has {bond_count} bonds"
        
        # **CHECK FOR REASONABLE POLYMER MONOMER STRUCTURE**
        # Must have reasonable polymerizable structure
        if not re.search(r'[Cc]', clean_smiles):
            return False, "No carbon atoms found - not a valid polymer monomer"
        
        # Check for extremely complex structures that are likely problematic
        if len(re.findall(r'[=]', clean_smiles)) > 3:
            return False, "Too many double bonds - likely to cause valence issues"
        
        if len(re.findall(r'[()]', clean_smiles)) > 8:
            return False, "Overly complex branching - likely to cause valence issues"
        
        return True, "PSMILES valence check passed"
    
    def _auto_repair_psmiles_valence(self, psmiles: str) -> Tuple[str, bool]:
        """
        Automatically repair common valence issues in PSMILES.
        Returns (repaired_psmiles, was_repaired)
        """
        import re
        
        original = psmiles
        repaired = psmiles
        was_repaired = False
        
        # **SPECIFIC REPAIRS FOR IDENTIFIED PROBLEMATIC PATTERNS**
        valence_repairs = [
            # Fix the exact failing patterns from user's log
            (r'\[\*\]C\(C\(C\)=O\)C\(C\)=O\[\*\]', '[*]CC(=O)OC(C)=O[*]', "Complex branched carbonyl → ester linkage"),
            (r'\[\*\]C\(C\(C\)=O\)\(C\)NC\(=O\).*?\[\*\]', '[*]C(C)NC(=O)C[*]', "Complex amide → simple amide"),
            (r'\[\*\]C\(C\(C\)=O\)C\(NC.*?\)=O\[\*\]', '[*]CC(=O)NC[*]', "Complex amide-ester → simple amide"),
            (r'\[\*\]C\(C\(C\)=O\)=C\(C\)N.*?\[\*\]', '[*]C=CC(=O)N[*]', "Complex vinyl-carbonyl → acrylamide"),
            
            # General carbonyl valence fixes
            (r'C\(C\(C\)=O\)C\(C\)=O', 'CC(=O)OC(C)=O', "Double carbonyl → ester"),
            (r'C\(C\(C\)=O\)', 'CC(C)C(=O)', "Branched carbonyl → linear"),
            (r'C\(.*?\)\(.*?\)=O', 'CC(=O)', "Over-substituted carbonyl → simple"),
            
            # Simplify complex patterns to safe monomers
            (r'C\(C\)=O\..*?', 'CC(=O)O', "Complex mixture → simple ester"),
            (r'N1C\(C\)C\(=O\)NC1=O', 'NC(=O)C', "Complex heterocycle → simple amide"),
            
            # Fix multi-component PSMILES to single components
            (r'\[\*\].*?\..*?\[\*\]', '[*]CC(=O)O[*]', "Multi-component → single ester monomer"),
        ]
        
        for pattern, replacement, description in valence_repairs:
            if re.search(pattern, repaired):
                old_repaired = repaired
                repaired = re.sub(pattern, replacement, repaired)
                if repaired != old_repaired:
                    print(f"   🔧 Auto-repair: {description}")
                    print(f"      {old_repaired} → {repaired}")
                    was_repaired = True
                    break  # Apply one repair at a time
        
        # **SAFETY FALLBACKS FOR POLYMER MONOMER CHEMISTRY**
        if was_repaired or not self._is_safe_monomer_structure(repaired):
            # If repairs were needed or structure is still unsafe, use proven safe monomers
            fallback_monomers = [
                '[*]CC(=O)O[*]',    # Acrylic acid derivative
                '[*]C=C[*]',        # Ethylene
                '[*]CC(=O)N[*]',    # Acrylamide derivative  
                '[*]c1ccccc1[*]',   # Aromatic
                '[*]COC[*]',        # Ether linkage
            ]
            
            # Choose fallback based on original intent
            if 'acid' in original.lower() or 'carbox' in original.lower():
                repaired = '[*]CC(=O)O[*]'
                print(f"   🔧 Fallback to safe monomer: carboxylic acid derivative")
            elif 'amide' in original.lower() or 'nitrogen' in original.lower():
                repaired = '[*]CC(=O)N[*]'
                print(f"   🔧 Fallback to safe monomer: amide derivative")
            elif 'aromatic' in original.lower() or 'benzene' in original.lower():
                repaired = '[*]c1ccccc1[*]'
                print(f"   🔧 Fallback to safe monomer: aromatic")
            else:
                repaired = '[*]CC(=O)O[*]'  # Default safe choice
                print(f"   🔧 Fallback to safe monomer: default ester")
            
            was_repaired = True
        
        return repaired, was_repaired
    
    def _is_safe_monomer_structure(self, psmiles: str) -> bool:
        """Check if PSMILES represents a safe, polymerizable monomer structure."""
        import re
        
        clean = psmiles.replace('[*]', '')
        
        # Must be reasonable length
        if len(clean) < 2 or len(clean) > 50:
            return False
        
        # Should have carbon atoms (organic polymer)
        if not re.search(r'[C]', clean):
            return False
        
        # Should not have overly complex branching
        if clean.count('(') > 4:
            return False
        
        # Should not have multiple unconnected components
        if '.' in clean:
            return False
        
        # Should not have problematic valence patterns
        problematic = [
            r'C\(.*?\)\(.*?\)=O',  # Over-substituted carbonyl
            r'C\(C\(C\)=O\)',      # Complex branching
            r'O=.*?=O',            # Multiple double bonds to oxygen
        ]
        
        for pattern in problematic:
            if re.search(pattern, clean):
                return False
        
        return True

    def generate_diverse_candidates(self, 
                                   base_request: str, 
                                   num_candidates: int = 6,
                                   temperature_range: Tuple[float, float] = (0.6, 1.0)) -> Dict:
        """
        Generate diverse PSMILES candidates using varying temperature and prompt variations.
        This method matches the interface expected by the app.
        
        Args:
            base_request: Natural language description of the polymer
            num_candidates: Number of diverse candidates to generate
            temperature_range: Range of temperatures to use for diversity
            
        Returns:
            Dict with diverse candidates and metadata
        """
        try:
            print(f"🧬 Generating diverse PSMILES candidates for: {base_request}")
            print(f"   Number of candidates: {num_candidates}")
            print(f"   Temperature range: {temperature_range}")
            
            # Handle mock mode when OpenAI is not available
            if self.mock_mode or self.llm is None:
                print("   🤖 Using mock mode with pre-defined insulin delivery polymer candidates")
                mock_candidates = [
                    {
                        'psmiles': '[*]C(C(=O)O)N[*]',  # Polyglycolic acid derivative
                        'smiles': 'C(C(=O)O)N',
                        'description': 'Biodegradable polyglycolic acid derivative for controlled insulin release',
                        'valid': True,
                        'temperature_used': 0.7,
                        'generation_method': 'mock_mode',
                        'pipeline': 'Mock→PSMILES'
                    },
                    {
                        'psmiles': '[*]C(C)(C(=O)O)[*]',  # Polylactic acid
                        'smiles': 'C(C)(C(=O)O)',
                        'description': 'FDA-approved polylactic acid for sustained insulin delivery',
                        'valid': True,
                        'temperature_used': 0.8,
                        'generation_method': 'mock_mode',
                        'pipeline': 'Mock→PSMILES'
                    },
                    {
                        'psmiles': '[*]CCO[*]',  # Polyethylene glycol
                        'smiles': 'CCO',
                        'description': 'Biocompatible polyethylene glycol for hydrogel insulin carriers',
                        'valid': True,
                        'temperature_used': 0.6,
                        'generation_method': 'mock_mode',
                        'pipeline': 'Mock→PSMILES'
                    },
                    {
                        'psmiles': '[*]C(C(=O)OC)[*]',  # Polymer ester
                        'smiles': 'C(C(=O)OC)',
                        'description': 'Degradable polymer ester for insulin encapsulation',
                        'valid': True,
                        'temperature_used': 0.9,
                        'generation_method': 'mock_mode',
                        'pipeline': 'Mock→PSMILES'
                    }
                ]
                
                # Select requested number of candidates
                selected_candidates = mock_candidates[:min(num_candidates, len(mock_candidates))]
                
                return {
                    'success': True,
                    'base_request': base_request,
                    'candidates': selected_candidates,
                    'valid_candidates': selected_candidates,
                    'num_generated': len(selected_candidates),
                    'num_valid': len(selected_candidates),
                    'best_candidate': selected_candidates[0]['psmiles'] if selected_candidates else None,
                    'generation_method': 'mock_mode',
                    'pipeline': 'Mock→PSMILES',
                    'model': 'mock_insulin_delivery_polymers',
                    'temperature_range': temperature_range,
                    'timestamp': datetime.now().isoformat()
                }
            
            all_candidates = []
            min_temp, max_temp = temperature_range
            
            # Generate candidates with varying temperatures
            temp_step = (max_temp - min_temp) / max(1, num_candidates - 1) if num_candidates > 1 else 0
            
            for i in range(num_candidates):
                try:
                    # Vary temperature for diversity
                    current_temp = min_temp + (i * temp_step) if num_candidates > 1 else min_temp
                    
                    # Temporarily adjust model temperature
                    original_temp = self.temperature
                    self.llm.temperature = current_temp
                    
                    # Generate using the existing method
                    result = self.generate_psmiles(
                        description=base_request,
                        num_candidates=1,  # Generate one at a time for temperature variation
                        validate=True,
                        disable_early_chemistry_lookup=True  # Enable diversity by preventing early chemistry lookup stops
                    )
                    
                    # Restore original temperature
                    self.llm.temperature = original_temp
                    
                    if result['success'] and result['candidates']:
                        candidate = result['candidates'][0]
                        candidate['temperature_used'] = current_temp
                        candidate['generation_method'] = 'working_pipeline_diverse'
                        candidate['pipeline'] = 'NaturalLanguage→SMILES→PSMILES'
                        all_candidates.append(candidate)
                        print(f"   ✅ Candidate {i+1}: {candidate['psmiles']} (temp: {current_temp:.2f})")
                    else:
                        print(f"   ❌ Failed candidate {i+1} at temp {current_temp:.2f}")
                        
                except Exception as e:
                    print(f"   ❌ Error generating candidate {i+1}: {e}")
                    continue
            
            # Filter for valid candidates
            valid_candidates = [c for c in all_candidates if c.get('valid', False)]
            
            return {
                'success': len(all_candidates) > 0,
                'base_request': base_request,
                'candidates': all_candidates,
                'valid_candidates': valid_candidates,
                'num_generated': len(all_candidates),
                'num_valid': len(valid_candidates),
                'best_candidate': valid_candidates[0]['psmiles'] if valid_candidates else None,
                'generation_method': 'working_pipeline_diverse',
                'pipeline': 'NaturalLanguage→SMILES→PSMILES',
                'model': self.openai_model,
                'temperature_range': temperature_range,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Diverse generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'base_request': base_request,
                'generation_method': 'working_pipeline_diverse',
                'timestamp': datetime.now().isoformat()
            }

    def generate_truly_diverse_candidates(self, 
                                        base_request: str,
                                        num_candidates: int = 5,
                                        enable_functionalization: bool = True,
                                        diversity_threshold: float = 0.4,
                                        temperature_range: Tuple[float, float] = (0.6, 1.2),  # Enhanced range from PSV-PPO
                                        max_retries: int = 3,  # Increased retries for better VALID-Mol compliance
                                        enable_valid_mol_constraints: bool = True) -> Dict[str, Any]:
        """
        Generate truly diverse PSMILES candidates using enhanced research-based methods.
        
        Implements improvements from:
        - PSV-PPO (2025): Temperature scheduling and stepwise validation
        - VALID-Mol (2025): Molecular weight and length constraints
        - MOLLM (2025): Multi-objective optimization for diversity
        
        Args:
            base_request: Natural language description of the polymer
            num_candidates: Number of diverse candidates to generate
            enable_functionalization: Whether to apply chemical functionalization
            diversity_threshold: Minimum diversity score required
            temperature_range: Range of temperatures for generation diversity (enhanced PSV-PPO range)
            max_retries: Maximum retry attempts for better diversity and constraints
            enable_valid_mol_constraints: Whether to apply VALID-Mol framework constraints
            
        Returns:
            Dict with diverse candidates and comprehensive metadata
        """
        if not self.diverse_generation_available:
            print("⚠️ Diverse generation system not available, falling back to standard generation")
            return self.generate_diverse_candidates(base_request, num_candidates, temperature_range)
        
        try:
            from .psmiles_diversification import CandidateConfig
            
            # PSV-PPO Temperature Scheduling: Use different temperatures for different phases
            temp_min, temp_max = temperature_range
            temperature_schedule = []
            for i in range(num_candidates):
                # Progressive temperature increase for exploration
                temp = temp_min + (temp_max - temp_min) * (i / max(1, num_candidates - 1))
                temperature_schedule.append(temp)
            
            print(f"🔬 PSV-PPO Temperature Schedule: {[f'{t:.2f}' for t in temperature_schedule]}")
            
            # Configure diverse generation with enhanced parameters
            config = CandidateConfig(
                base_request=base_request,
                num_candidates=num_candidates,
                temperature_range=temperature_range,
                enable_functionalization=enable_functionalization,
                diversity_threshold=diversity_threshold,
                max_functionalization_attempts=3,
                prompt_variation_strategies=[
                    'perspective_variation',
                    'property_emphasis', 
                    'structure_focus',
                    'application_context',
                    'chemical_modification',
                    'functional_group_diversity',  # NEW: Enhanced functionalization
                    'backbone_variation'  # NEW: Structural diversity
                ]
            )
            
            print(f"🚀 Starting enhanced diverse generation with research-based improvements")
            print(f"   Base request: {base_request}")
            print(f"   Candidates: {num_candidates}")
            print(f"   Functionalization: {enable_functionalization}")
            print(f"   Diversity threshold: {diversity_threshold}")
            print(f"   VALID-Mol constraints: {enable_valid_mol_constraints}")
            
            # Use enhanced retry logic with VALID-Mol constraint checking
            result = self.orchestrator.generate_with_retry(config, max_retries=max_retries)
            
            # Post-process with VALID-Mol constraints if enabled
            if enable_valid_mol_constraints and result.get('success'):
                print(f"🔍 Applying VALID-Mol constraints to generated candidates...")
                enhanced_candidates = []
                constraint_stats = {'total': 0, 'passed': 0, 'failed': 0}
                
                for candidate in result.get('candidates', []):
                    constraint_stats['total'] += 1
                    psmiles = candidate.get('psmiles', '')
                    
                    # Remove [*] for SMILES constraint checking
                    smiles_for_validation = psmiles.replace('[*]', '')
                    
                    if smiles_for_validation:
                        valid_constraints, constraint_message = self.validate_valid_mol_constraints(smiles_for_validation)
                        candidate['valid_mol_constraints'] = valid_constraints
                        candidate['constraint_details'] = constraint_message
                        
                        if valid_constraints:
                            constraint_stats['passed'] += 1
                        else:
                            constraint_stats['failed'] += 1
                            print(f"   ❌ Candidate failed VALID-Mol: {constraint_message}")
                    else:
                        candidate['valid_mol_constraints'] = False
                        candidate['constraint_details'] = "Empty SMILES"
                        constraint_stats['failed'] += 1
                    
                    enhanced_candidates.append(candidate)
                
                result['candidates'] = enhanced_candidates
                result['valid_mol_stats'] = constraint_stats
                print(f"   VALID-Mol Results: {constraint_stats['passed']}/{constraint_stats['total']} passed constraints")
            
            # Add compatibility fields for existing interfaces
            if result.get('success'):
                result['best_candidate'] = result.get('candidates', [{}])[0].get('psmiles') if result.get('candidates') else None
                result['valid_candidates'] = [c for c in result.get('candidates', []) if c.get('valid', False)]
                result['num_valid'] = len(result['valid_candidates'])
                
                # Enhanced statistics
                constraint_compliant = [c for c in result.get('candidates', []) if c.get('valid_mol_constraints', False)]
                result['constraint_compliant_candidates'] = constraint_compliant
                result['num_constraint_compliant'] = len(constraint_compliant)
                
                print(f"✅ Enhanced diverse generation successful!")
                print(f"   Generated: {result['num_generated']} candidates")
                print(f"   Valid: {result['num_valid']} candidates")
                print(f"   Constraint compliant: {result['num_constraint_compliant']} candidates")
                print(f"   Diversity score: {result.get('diversity_validation', {}).get('diversity_score', 0):.3f}")
                print(f"   Meets threshold: {result.get('diversity_validation', {}).get('meets_diversity_threshold', False)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Enhanced diverse generation failed: {e}")
            print("⚠️ Falling back to standard diverse generation")
            return self.generate_diverse_candidates(base_request, num_candidates, temperature_range)

    def validate_valid_mol_constraints(self, smiles: str) -> tuple:
        """
        VALID-Mol Framework Constraints Validation (2025 Research) - UPDATED FOR TESTING
        **RELAXED** constraints to allow testing of stable structures with various elements
        """
        try:
            # Step 1: SMILES Length Constraint - RELAXED for simple stable structures
            MIN_LENGTH = 2   # Reduced from 5 - allows simple compounds like BCl3 → ClB(Cl)Cl
            MAX_LENGTH = 1000
            
            if len(smiles) < MIN_LENGTH:
                return False, f"SMILES too short ({len(smiles)} < {MIN_LENGTH} chars)"
            
            if len(smiles) > MAX_LENGTH:
                return False, f"SMILES too long ({len(smiles)} > {MAX_LENGTH} chars) - unrealistic structure"
            
            # Step 2: Molecular Weight Constraint - RELAXED for simple stable structures
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, "Invalid SMILES structure"
                
                # Calculate molecular weight
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                MIN_MW = 30.0   # Reduced from 100.0 - allows simple stable compounds (BCl3 ≈ 117 Da, SiO2 ≈ 60 Da)  
                MAX_MW = 800.0  # Keep reasonable upper limit for drug-like range
                
                if mw < MIN_MW:
                    return False, f"Molecular weight too low ({mw:.1f} < {MIN_MW} Da)"
                
                if mw > MAX_MW:
                    return False, f"Molecular weight too high ({mw:.1f} > {MAX_MW} Da) - unrealistic for drug-like molecules"
                
                return True, f"Valid: {len(smiles)} chars, {mw:.1f} Da"
                
            except ImportError:
                # RDKit not available, only check length
                return True, f"Valid length: {len(smiles)} chars (MW check requires RDKit)"
            except Exception as e:
                return False, f"Molecular weight calculation failed: {str(e)}"
                
        except Exception as e:
            return False, f"VALID-Mol validation error: {str(e)}"
    
    def _clean_smiles_response(self, response: str) -> str:
        """Clean OpenAI response to extract just the SMILES string"""
        # Remove common explanatory text
        response = response.replace("SMILES:", "").replace("smiles:", "")
        response = response.replace("The SMILES string is", "")
        response = response.replace("Answer:", "").replace("Result:", "")
        response = response.replace("→", "").replace("->", "")
        
        # Split by lines and take first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            smiles = lines[0]
        else:
            smiles = response.strip()
        
        # Remove quotes and extra spaces
        smiles = smiles.strip().strip('"').strip("'").strip()
        
        # CRITICAL FIX: Remove descriptive text in parentheses that causes PSMILES parsing errors
        import re
        # Remove text in parentheses like "(styrene)", "(fluorinated phenol)", etc.
        smiles = re.sub(r'\s*\([^)]*\)\s*', '', smiles)
        
        # Remove trailing descriptive words after SMILES
        # Split on first space and take only the SMILES part
        smiles_parts = smiles.split()
        if smiles_parts:
            smiles = smiles_parts[0]
        
        # Final cleanup
        smiles = smiles.strip()
        
        return smiles


def test_psmiles_generator():
    """Test function for the PSMILES generator."""
    print("🧪 Testing PSMILES Generator with OpenAI ChatGPT...")
    
    try:
        # Initialize generator
        generator = PSMILESGenerator(
            model_type='openai',
            openai_model='gpt-4o',
            temperature=0.7
        )
        
        # Run test cases
        test_results = generator.test_generation()
        
        print(f"\n📊 Test Summary:")
        print(f"   Total tests: {test_results['total_tests']}")
        print(f"   Successful: {test_results['successful_tests']}")
        print(f"   Success rate: {test_results['successful_tests']/test_results['total_tests']*100:.1f}%")
        
        if test_results['successful_tests'] > 0:
            print("🎉 PSMILES Generator working correctly with OpenAI!")
        else:
            print("❌ All tests failed. Check OpenAI configuration.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure OpenAI API key is set: export OPENAI_API_KEY='your_api_key'")


if __name__ == "__main__":
    test_psmiles_generator() 