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

SMILES GUIDELINES (FOLLOW EXACTLY - VERBATIM):
SMILES (simplified molecular-input line-entry system) uses short ASCII string to represent the structure of chemical species. Because the SMILES format described here is custom-designed by us for polymers, it is not completely identical to other SMILES formats. Strictly following the rules explained below is crucial for having correct results.

1. Spaces are not permitted in a SMILES string.
2. An atom is represented by its respective atomic symbol. In case of 2-character atomic symbol, it is placed between two square brackets [ ].
3. Single bonds are implied by placing atoms next to each other. A double bond is represented by the = symbol while a triple bond is represented by #.
4. Hydrogen atoms are suppressed, i.e., the polymer blocks are represented without hydrogen. Polymer Genome interface assumes typical valence of each atom type. If enough bonds are not identified by the user through SMILES notation, the dangling bonds will be automatically saturated by hydrogen atoms.
5. Branches are placed between a pair of round brackets ( ), and are assumed to attach to the atom right before the opening round bracket (.
6. Numbers are used to identify the opening and closing of rings of atoms. For example, in C1CCCCC1, the first carbon having a number "1" should be connected by a single bond with the last carbon, also having a number "1". Polymer blocks that have multiple rings may be identified by using different, consecutive numbers for each ring.
7. Atoms in aromatic rings can be specified by lower case letters. As an example, benzene ring can be written as c1ccccc1 which is equivalent to C(C=C1)=CC=C1.

CRITICAL REQUIREMENTS:
- ALL brackets must be balanced: ( ) [ ]
- ALL ring closures must be complete: c1ccccc1 not c1cccc
- NO trailing incomplete symbols
- NO spaces or hyphens in final SMILES
- Ensure proper valence for all atoms

VALENCE RULES (CRITICAL - MUST FOLLOW):
- Carbon: maximum 4 bonds (C, CC, C(C)(C)(C)C)
- Oxygen: maximum 2 bonds (O, CO, C(=O))  
- Nitrogen: maximum 3 bonds (N, CN, C(=O)N)
- Sulfur: maximum 2 bonds in simple form (S, CS, CSC, CSS)
- Fluorine: EXACTLY 1 bond only (CF, CCF, never S(F) or C(F)(F)F)
- NEVER write: S(F), C(F)(F)F, S(F)(F), or any multi-fluorinated atoms
- VALID fluorinated: CF, CCF, CFC, c1ccc(F)cc1
- INVALID fluorinated: S(F), C(F)(F)F, S(F)(F)F, CC(F)(F)S

POLYMER MONOMER REQUIREMENTS (CRITICAL FOR THIS APPLICATION):
- The generated structure will be used as a POLYMER MONOMER
- MUST contain at least TWO CARBON atoms (minimum: CC)
- Structure must be capable of polymerization
- Include functional groups that enable polymer chain formation
- Consider typical monomer structures like vinyl groups (C=C), rings that can open, or difunctional molecules

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
            
            # **STEP 1: Check direct chemistry lookup first**
            desc_lower = description.lower().strip()
            if desc_lower in self.chemistry_examples:
                smiles = self.chemistry_examples[desc_lower]
                print(f"   ✅ Direct chemistry match: {smiles}")
                return {
                    'success': True,
                    'smiles': smiles,
                    'method': 'chemistry_lookup',
                    'description': description,
                    'confidence': 1.0
                }
            
            # **STEP 2: Use OpenAI with comprehensive prompting**
            print(f"   No direct match, using OpenAI with comprehensive prompts...")
            messages = self.nl_to_smiles_prompt.format_messages(description=description)
            response = self.llm.invoke(messages)
            smiles = response.content.strip()
            
            # Clean the SMILES (remove any extra text)
            smiles = self._clean_smiles_response(smiles)
            
            print(f"   Generated SMILES: {smiles}")
            
            # **STEP 3: VALID-Mol Constraint Validation (NEW)**
            print(f"   🔍 Applying VALID-Mol constraints...")
            valid_constraints, constraint_message = self.validate_valid_mol_constraints(smiles)
            
            if not valid_constraints:
                print(f"   ❌ VALID-Mol constraints failed: {constraint_message}")
                # Retry with more constrained prompt
                print(f"   🔄 Retrying with stricter constraints...")
                retry_prompt = f"Generate a REALISTIC small molecule for: {description}. CRITICAL: Must be 5-1000 characters, 100-800 Da molecular weight. Output ONLY the SMILES string."
                retry_messages = [("system", "You are a chemistry expert. Generate ONLY a valid, realistic SMILES string within molecular weight 100-800 Da and 5-1000 characters."), ("human", retry_prompt)]
                
                retry_response = self.llm.invoke(retry_messages)
                retry_smiles = self._clean_smiles_response(retry_response.content.strip())
                
                # Validate retry
                retry_valid, retry_message = self.validate_valid_mol_constraints(retry_smiles)
                if retry_valid:
                    print(f"   ✅ Retry successful: {retry_message}")
                    smiles = retry_smiles
                else:
                    print(f"   ❌ Retry also failed: {retry_message}")
                    # Continue with original SMILES but mark as constraint-violating
                    
            else:
                print(f"   ✅ VALID-Mol constraints satisfied: {constraint_message}")
            
            return {
                'success': True,
                'smiles': smiles,
                'method': 'openai_comprehensive',
                'description': description,
                'confidence': 0.8,
                'constraints_satisfied': valid_constraints,
                'constraint_details': constraint_message
            }
            
        except Exception as e:
            print(f"   ❌ SMILES generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'failed_generation'
            }
    
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
        VALID-Mol Framework Constraints Validation (2025 Research)
        Implements molecular weight and length constraints that improved validity from 3% to 83%
        """
        try:
            # Step 1: SMILES Length Constraint (5-1000 characters)
            MIN_LENGTH = 5
            MAX_LENGTH = 1000
            
            if len(smiles) < MIN_LENGTH:
                return False, f"SMILES too short ({len(smiles)} < {MIN_LENGTH} chars)"
            
            if len(smiles) > MAX_LENGTH:
                return False, f"SMILES too long ({len(smiles)} > {MAX_LENGTH} chars) - unrealistic structure"
            
            # Step 2: Molecular Weight Constraint (100-800 Da)
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False, "Invalid SMILES structure"
                
                # Calculate molecular weight
                mw = rdMolDescriptors.CalcExactMolWt(mol)
                MIN_MW = 100.0  # Da
                MAX_MW = 800.0  # Da (drug-like range)
                
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

PSMILES (Polymer SMILES) is a notation system for representing polymers with exactly TWO connection points marked as [*].

CRITICAL RULES (ENHANCED WITH MOLLM MULTI-OBJECTIVE FRAMEWORK):
1. ALWAYS use exactly TWO [*] symbols to mark connection points
2. Use proper SMILES syntax for chemical structures
3. Consider real polymer chemistry and realistic structures
4. Ensure chemical validity and proper valences
5. Focus on practical polymers that could be synthesized
6. NEVER use explicit hydrogen atoms (H) in the structure - SMILES implicit hydrogen handling
7. For sulfur atoms, use valid patterns like CSC (thioether), c1sccc1 (thiophene), CSS (disulfide)

DIVERSIFICATION STRATEGY (FROM MOLLM RESEARCH 2025):
- Vary functionalization types: ester, amide, ether, aromatic substitution, vinyl, sulfur-based
- Mix backbone types: aliphatic chains, aromatic rings, heterocycles
- Combine different heteroatoms: N, O, S, P (when appropriate)
- Use different connection patterns: linear, branched, cyclic integration

MOLECULAR WEIGHT CONSTRAINTS (VALID-MOL FRAMEWORK):
- Target polymer repeat units: 50-500 Da (realistic monomer range)
- PSMILES length: 8-100 characters (excludes [*] symbols)
- Ensure chemical feasibility for polymerization

FUNCTIONALIZATION DIVERSITY (TSMMG TEACHER-STUDENT APPROACH):
- Primary types: C(=O)O (ester), C(=O)N (amide), O (ether), c1ccccc1 (aromatic)
- Secondary types: CSS (disulfide), CSC (thioether), C=C (vinyl), c1sccc1 (thiophene)
- Tertiary types: P-containing (when specified), halogenated (F, Cl), cyclic structures

IMPORTANT SMILES SYNTAX:
- NO explicit hydrogen atoms: Never write H, HS, SH, HSH, HCSH, etc.
- Sulfur examples: CSC (thioether bridge), c1sccc1 (thiophene ring), CSS (disulfide)
- Carbon backbone: CC, CCC, c1ccccc1 (aromatic)
- Functional groups: C(=O) (carbonyl), C(=O)O (carboxyl), C(=O)N (amide)

VALID EXAMPLES (WITH DIVERSITY):
- "polymer with aromatic rings": [*]c1ccccc1[*] OR [*]Cc1ccccc1C[*] OR [*]c1ccc(C)cc1[*]
- "polymer with amide linkages": [*]C(=O)NC[*] OR [*]CC(=O)NCC[*] OR [*]C(=O)Nc1ccccc1[*]
- "polymer with ester groups": [*]C(=O)OC[*] OR [*]CC(=O)OCC[*] OR [*]C(=O)Oc1ccccc1[*]
- "polyethylene-like": [*]CC[*] OR [*]CCC[*] OR [*]C(C)C[*]
- "polymer with hydroxyl groups": [*]C(O)C[*] OR [*]CC(O)CC[*] OR [*]c1ccc(O)cc1[*]
- "sulfur-containing polymer": [*]CSC[*] OR [*]c1sccc1[*] OR [*]CSS[*] OR [*]CSCC[*]

INVALID EXAMPLES TO AVOID:
- [*]HSH[*] - NO explicit hydrogens
- [*]HCSH[*] - NO explicit hydrogens
- [*]SH[*] - NO explicit hydrogens
- [*]HS[*] - NO explicit hydrogens

TASK: Convert the user's description into a valid PSMILES string.
- Think about the polymer structure they're describing
- Generate a chemically realistic PSMILES with exactly 2 [*] connection points
- Use diverse functionalization appropriate to the request
- Ensure proper SMILES syntax and chemical validity
- For sulfur requests, use CSC, CSS, or c1sccc1 patterns with variation
- Respond with just the PSMILES string, no explanation unless requested"""

        psmiles_prompt = ChatPromptTemplate.from_messages([
            ("system", psmiles_system_prompt),
            ("human", "{description}"),
        ])
        
        return {
            'psmiles_generation': psmiles_prompt
        }
    
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
        VALID-Mol Framework Constraints Validation (2025 Research) - Copy for PSMILESGenerator
        Implements molecular weight and length constraints that improved validity from 3% to 83%
        """
        return self.nl_to_psmiles.validate_valid_mol_constraints(smiles)


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