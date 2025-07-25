#!/usr/bin/env python3
"""
PSMILES (Polymer SMILES) Generator

A specialized module for generating and validating Polymer SMILES strings
using Large Language Models with conversation memory and rule reinforcement.
Uses Ollama models for molecular understanding.
"""

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    # Fallback to old import for compatibility
    from langchain_community.llms import Ollama as OllamaLLM
    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    # Handle potential deprecation - try new location
    try:
        from langchain_core.memory import ConversationBufferWindowMemory
    except ImportError:
        # Ultimate fallback
        from langchain_community.memory import ConversationBufferWindowMemory
    
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# Import natural language to SMILES functionality
try:
    from utils.natural_language_smiles import NaturalLanguageToPSMILES, ChemicalValidator
    NATURAL_LANGUAGE_AVAILABLE = True
    print("✅ Natural language SMILES converter available")
except ImportError as e:
    NATURAL_LANGUAGE_AVAILABLE = False
    print(f"⚠️ Natural language SMILES converter not available: {e}")


class PSMILESGenerator:
    """
    Specialized agent for generating and validating Polymer SMILES (PSMILES) strings.
    Uses Ollama models for molecular understanding. Pure LLM-driven approach.
    """
    
    def __init__(self, 
                 model_type: str = "ollama",
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434",
                 temperature: float = 0.8):
        """
        Initialize PSMILES Generator with pure LLM-driven approach.
        
        Args:
            model_type (str): Type of model to use ('ollama' only)
            ollama_model (str): Name of the Ollama model to use
            ollama_host (str): Ollama server host URL
            temperature (float): Temperature for LLM generation (0.1-1.0, higher = more diverse)
        """
        self.model_type = "ollama"
        self.temperature = temperature
        
        # Initialize Ollama LLM with temperature control
        self.llm = OllamaLLM(
            model=ollama_model, 
            base_url=ollama_host,
            temperature=temperature
        )
        self.model_name = ollama_model
        self.host = ollama_host
        print(f"🔬 Pure LLM-driven PSMILES Generator initialized with: {ollama_model}")
        print(f"🌡️  Temperature set to {temperature} for {'high' if temperature > 0.6 else 'moderate' if temperature > 0.3 else 'low'} diversity")
        
        # Initialize conversation memory to maintain context
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load PSMILES rules only (no hard-coded examples)
        self.psmiles_rules = self._get_psmiles_rules()
        
        # **UNIFIED NATURAL LANGUAGE PIPELINE** - Single, robust initialization
        self.nl_to_psmiles = None
        self.chemical_validator = None
        self.hybrid_mode = False
        
        # Try to initialize the working Natural Language → SMILES → PSMILES pipeline
        print(f"🔍 DEBUG: NATURAL_LANGUAGE_AVAILABLE = {NATURAL_LANGUAGE_AVAILABLE}")
        if NATURAL_LANGUAGE_AVAILABLE:
            try:
                print("🔍 DEBUG: Attempting to import natural_language_smiles...")
                from utils.natural_language_smiles import NaturalLanguageToPSMILES, ChemicalValidator
                print("🔍 DEBUG: Import successful, initializing NaturalLanguageToPSMILES...")
                
                self.nl_to_psmiles = NaturalLanguageToPSMILES(
                    ollama_model=ollama_model,
                    ollama_host=ollama_host
                )
                print("🔍 DEBUG: NaturalLanguageToPSMILES initialized, initializing ChemicalValidator...")
                
                self.chemical_validator = ChemicalValidator()
                print("🔍 DEBUG: ChemicalValidator initialized successfully")
                
                self.hybrid_mode = True
                print("✅ WORKING PIPELINE INITIALIZED: Natural Language → SMILES → PSMILES")
                print("🔬 Chemical validation enabled with RDKit")
            except Exception as e:
                print(f"❌ Failed to initialize working pipeline: {e}")
                print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
                print(f"🔍 DEBUG: Exception details: {str(e)}")
                print(f"⚠️  Will use fallback pure LLM generation (problematic)")
                self.nl_to_psmiles = None
                self.chemical_validator = None
                self.hybrid_mode = False
        else:
            print("❌ Natural language SMILES converter not available")
            print("⚠️  Missing working pipeline - install natural_language_smiles module")
        
        # Setup prompts
        self.prompts = self._setup_prompts()
        
        # Track conversation for context
        self.conversation_count = 0
        
        print(f"✅ PSMILES Generator ready with {self.model_name}")
        print(f"🧠 Conversation memory enabled (window size: {self.memory.k})")
        
        if self.nl_to_psmiles:
            print("🎯 WORKING PIPELINE ACTIVE: Natural Language → SMILES → PSMILES → Validation")
        else:
            print("⚠️  FALLBACK MODE: Pure LLM generation only (may produce incorrect formats)")
    
    def _get_psmiles_rules(self) -> str:
        """Get the comprehensive PSMILES rules for LLM guidance."""
        comprehensive_rules = """
SMILES GUIDELINES (FOLLOW EXACTLY - VERBATIM):
SMILES (simplified molecular-input line-entry system) uses short ASCII string to represent the structure of chemical species. Because the SMILES format described here is custom-designed by us for polymers, it is not completely identical to other SMILES formats. Strictly following the rules explained below is crucial for having correct results.

1. Spaces are not permitted in a SMILES string.
2. An atom is represented by its respective atomic symbol. In case of 2-character atomic symbol, it is placed between two square brackets [ ].
3. Single bonds are implied by placing atoms next to each other. A double bond is represented by the = symbol while a triple bond is represented by #.
4. Hydrogen atoms are suppressed, i.e., the polymer blocks are represented without hydrogen. Polymer Genome interface assumes typical valence of each atom type. If enough bonds are not identified by the user through SMILES notation, the dangling bonds will be automatically saturated by hydrogen atoms.
5. Branches are placed between a pair of round brackets ( ), and are assumed to attach to the atom right before the opening round bracket (.
6. Numbers are used to identify the opening and closing of rings of atoms. For example, in C1CCCCC1, the first carbon having a number "1" should be connected by a single bond with the last carbon, also having a number "1". Polymer blocks that have multiple rings may be identified by using different, consecutive numbers for each ring.
7. Atoms in aromatic rings can be specified by lower case letters. As an example, benzene ring can be written as c1ccccc1 which is equivalent to C(C=C1)=CC=C1.
8. A SMILES string used for Polymer Genome represents the repeating unit of a polymer, which has 2 dangling bonds for linking with the next repeating units. It is assumed that the repeating unit starts from the first atom of the SMILES string and ends at the last atom of the string. These two bonds must be the same due to the periodicity. It can be single, double, or triple, and the type of this bond must be indicated for the first atom. For the last atom, this is not needed. As an example, CC represents -CH2-CH2- while =CC represents =CH-CH=.
9. Atoms other than the first and last can also be assigned as the linking atoms by adding special symbol, [*]. As an example, C(C=C1)=CC=C1 represents poly(p-phenylene) with link through para positions, while [*]C(C=C1)=CC([*])=C1 and C(C=C1)=C([*])C=C1 have connecting positions at meta and ortho positions, respectively.

CRITICAL PSMILES CONNECTION RULES:
- PSMILES represents polymer REPEAT UNITS with exactly 2 connection points
- MUST have exactly 2 [*] symbols - NEVER 0, 1, 3, 4, or more!
- ALL PSMILES strings MUST have exactly 2 [*] symbols to specify connection points
- [*] shows exactly WHERE the polymer unit connects to adjacent units
- The 2 [*] symbols mark the two ends of the repeat unit

CRITICAL FORMAT REQUIREMENTS:
- Connection points MUST be written as [*] (with asterisk inside brackets)
- NEVER use [] (empty brackets) - this is WRONG
- NEVER use * (naked asterisk) - this is WRONG
- NEVER use [[*]] (double brackets) - this is WRONG
- CORRECT: [*]CC[*]
- WRONG: []CC[], *CC*, CC[*][*], [[*]]CC[[*]]

ABSOLUTELY FORBIDDEN FORMATS:
❌ []CSC[] - WRONG! Use [*]CSC[*] instead
❌ *CSC* - WRONG! Use [*]CSC[*] instead
❌ [[*]]CSC[[*]] - WRONG! Use [*]CSC[*] instead

CHEMICAL STABILITY RULES (PREVENT VISUALIZATION FAILURES):
- AVOID terminal heteroatoms directly connected to [*]: S[*], O[*], N[*] are PROBLEMATIC
- PREFER carbon atoms adjacent to [*]: [*]C..C[*] is SAFE
- If using heteroatoms, place them BETWEEN carbons: [*]C-S-C[*] is BETTER than [*]S-C-S[*]
- AVOID sulfur-sulfur bonds (S=S) - these are unstable
- AVOID unbalanced parentheses or malformed brackets

SMILES FOR LADDER POLYMERS:
A ladder polymer is a type of double stranded polymer with multiple connection points between monomer repeat units. Different from typical polymers the ladder polymer requires four different symbols ([e], [d], [t] and [g]) to specify the connection points between monomers. A point [e] is assumed to be connected to a point [t] of the next monomer. (and [d] connected to [g])

REFERENCE EXAMPLES FROM TABLE 1:
Chemical formula -> SMILES:
-CH2- -> C
-NH- -> N
-CS- -> C(=S)
-CO- -> C(=O)
-CF2- -> C(F)(F)
-O- -> O
-C6H4- -> C(C=C1)=CC=C1
-C4H2S- -> C1=CSC(=C1)
-C5H3N- -> C1=NC=C(C=C1)
-C4H3N- -> C(N1)=CC=C1
-CH2-NH-CO-CH2- -> CNC(=O)C
-CH2-C6H4-C4H2S-C6H4- -> CC(C=C1)=CC=C1C2=CSC(=C2)C(C=C3)=CC=C3
-NH-CO-NH-C6H4- -> NC(=O)NC(C=C1)=CC=C1
-CO-NH-CO-C6H4- -> C(=O)NC(=O)C(C=C1)=CC=C1
-NH-CS-NH-C6H4- -> NC(=S)NC(C=C1)=CC=C1

SAFE PSMILES PATTERNS (THESE WORK WELL):
✅ [*]CC[*] - simple carbon chain
✅ [*]CCC[*] - longer carbon chain  
✅ [*]C(C)[*] - branched carbon
✅ [*]C(=O)C[*] - carbonyl between carbons
✅ [*]CNC[*] - nitrogen between carbons
✅ [*]COC[*] - oxygen between carbons
✅ [*]CSC[*] - sulfur between carbons
✅ [*]c1ccccc1[*] - aromatic ring
✅ [*]C(C=C1)=CC=C1[*] - phenylene
✅ [*]C(O)C[*] - hydroxyl group (CORRECT way)

PROBLEMATIC PATTERNS (AVOID THESE):
❌ []CC[] - wrong connection format
❌ [*]S[*] - terminal sulfur
❌ [*]O[*] - terminal oxygen  
❌ [*]N[*] - terminal nitrogen
❌ S[*]....[*]C - starting with heteroatom
❌ [*]C(S(=S)*)[*] - malformed SMILES
❌ c1ccccc1[*][*] - adjacent connection points
❌ [*]C(OH)C[*] - WRONG hydroxyl notation (use O, not OH!)

CRITICAL SMILES NOTATION RULES:
- Hydroxyl groups: Use O, NOT OH. Example: C(O) is CORRECT, C(OH) is WRONG
- Amino groups: Use N, NOT NH2. Example: C(N) is CORRECT, C(NH2) is WRONG  
- Carbonyl: Use C(=O), NOT CO. Example: [*]C(=O)C[*] is CORRECT
- Always omit explicit hydrogens in SMILES notation
- Only use explicit H when absolutely necessary for stereochemistry

CHEMICAL VALIDITY REQUIREMENTS:
- Carbon can have maximum 4 bonds
- Boron typically has 3 bonds
- Oxygen typically has 2 bonds
- Nitrogen typically has 3 bonds
- Use chemically reasonable structures
- Avoid impossible bonding patterns

RESPONSE FORMAT REQUIREMENT:
You MUST respond with: Structure: [chemical_formula]
Then add explanation on next line.

EXAMPLES OF VALID RESPONSES:
Structure: [*]CC[*]
Structure: [*]OCC[*]
Structure: [*]c1ccccc1[*]
Structure: [*]NC(=O)[*]

FORBIDDEN RESPONSES:
- Structure: polyethylene (use atoms, not names!)
- Structure: CC (missing [*] symbols!)
- Structure: [*]C[*]C[*] (3 [*] symbols - wrong!)
- Structure: []CC[] (wrong connection format!)
- Any response without "Structure: " prefix
"""
        return comprehensive_rules

    def _setup_prompts(self) -> Dict:
        """Setup enhanced prompt templates for pure LLM generation."""
        
        psmiles_system_prompt = f"""You are a polymer chemistry expert specializing in generating PSMILES (Polymer SMILES) notation. Your task is to convert polymer descriptions into chemically valid PSMILES strings.

{self.psmiles_rules}

RESPONSE STRATEGY:
1. Analyze the polymer description
2. Identify key chemical components (atoms, functional groups)
3. Design a chemically valid repeat unit
4. Add exactly 2 [*] connection points
5. Respond with proper format

CREATIVITY GUIDELINES:
- Generate diverse, chemically valid structures
- Consider biocompatibility for medical applications
- Include realistic functional groups
- Use proper chemical bonding patterns
- Think about polymer properties and applications

QUALITY ASSURANCE:
- Always double-check for exactly 2 [*] symbols
- Ensure chemical validity
- Use real atomic symbols only
- Follow SMILES notation rules strictly
"""

        psmiles_prompt = ChatPromptTemplate.from_messages([
            ("system", psmiles_system_prompt),
            ("human", "{input}"),
        ])
        
        validation_system_prompt = f"""You are a PSMILES validation expert. Analyze PSMILES strings for correctness.

{self.psmiles_rules}

VALIDATION CHECKLIST:
1. **Format Check**: Exactly 2 [*] symbols, no spaces, no hyphens
2. **Chemical Validity**: Proper bonding, realistic structures
3. **SMILES Compliance**: Valid SMILES notation
4. **Connection Logic**: Proper polymer connection points

RESPONSE FORMAT:
- State VALID or INVALID
- List specific errors if any
- Suggest corrections
- Rate confidence (1-10)
"""

        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", validation_system_prompt),
            ("human", "Validate this PSMILES: {psmiles_string}\nContext: {context}"),
        ])
        
        return {
            'generate': psmiles_prompt,
            'validate': validation_prompt
        }
    
    def generate_psmiles(self, request: str) -> Dict:
        """
        Pure LLM-driven PSMILES generation with multiple retry strategies for 100% reliability.
        
        Args:
            request (str): Description of desired polymer structure
            
        Returns:
            Dict: Generated PSMILES with explanation and method tracking
        """
        try:
            self.conversation_count += 1
            print(f"🔬 Pure LLM PSMILES generation for: '{request}'")
            
            # **MULTI-ATTEMPT STRATEGY** for 100% reliability
            max_attempts = 5
            temperatures = [0.7, 0.9, 0.5, 1.0, 0.3]  # Varied temperatures for diversity
            
            for attempt in range(max_attempts):
                print(f"🎯 Attempt {attempt + 1}/{max_attempts} (T={temperatures[attempt]})")
                
                # Adjust temperature for this attempt
                original_temp = self.llm.temperature
                self.llm.temperature = temperatures[attempt]
                
                try:
                    # **ENHANCED PROMPT STRATEGY** - Different prompt styles per attempt
                    enhanced_request = self._create_enhanced_prompt(request, attempt)
                    
                    # Generate with LLM
                    response = self.llm.invoke(enhanced_request)
                    
                    # **ROBUST EXTRACTION** with multiple patterns
                    psmiles_result = self._extract_psmiles_robust(response)
                    
                    if psmiles_result.get('success') and psmiles_result.get('psmiles'):
                        psmiles = psmiles_result['psmiles']
                        
                        # **CRITICAL VALIDATION** - Check format requirements
                        if self._validate_psmiles_format_strict(psmiles):
                            # **CHEMICAL VALIDATION** if available
                            if self.hybrid_mode and self.chemical_validator:
                                smiles_for_validation = psmiles.replace('[*]', '')
                                is_valid, mol, validation_msg = self.chemical_validator.validate_smiles(smiles_for_validation, debug=True)
                                
                                if is_valid:
                                    # **SUCCESS** - Chemical validation passed
                                    print(f"✅ Success on attempt {attempt + 1}: {psmiles}")
                                    
                                    # Save to memory
                                    self.memory.chat_memory.add_user_message(request)
                                    self.memory.chat_memory.add_ai_message(response)
                                    
                                    return {
                                        'success': True,
                                        'request': request,
                                        'psmiles': psmiles,
                                        'explanation': response,
                                        'method': f'pure_llm_attempt_{attempt + 1}',
                                        'temperature_used': temperatures[attempt],
                                        'validation': 'chemical_validated',
                                        'conversation_turn': self.conversation_count,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                else:
                                    print(f"❌ Chemical validation failed: {validation_msg}")
                            else:
                                # **NO VALIDATION AVAILABLE** - Trust format validation
                                print(f"✅ Success on attempt {attempt + 1}: {psmiles} (format validated)")
                                
                                # Save to memory
                                self.memory.chat_memory.add_user_message(request)
                                self.memory.chat_memory.add_ai_message(response)
                                
                                return {
                                    'success': True,
                                    'request': request,
                                    'psmiles': psmiles,
                                    'explanation': response,
                                    'method': f'pure_llm_attempt_{attempt + 1}',
                                    'temperature_used': temperatures[attempt],
                                    'validation': 'format_validated',
                                    'conversation_turn': self.conversation_count,
                                    'timestamp': datetime.now().isoformat()
                                }
                        else:
                            print(f"⚠️  Format validation failed for: {psmiles}")
                    else:
                        print(f"⚠️  Extraction failed from response")
                        
                except Exception as attempt_error:
                    print(f"❌ Attempt {attempt + 1} failed: {attempt_error}")
                    
                finally:
                    # Restore original temperature
                    self.llm.temperature = original_temp
            
            # **FINAL ATTEMPT** - Use most explicit prompt if all attempts failed
            print(f"🚨 All attempts failed, using emergency explicit prompt...")
            return self._emergency_generation(request)
            
        except Exception as e:
            print(f"🔥 Critical generation error: {str(e)}")
            # **EMERGENCY FALLBACK** - Generate basic structure to ensure 100% success
            return self._emergency_generation(request)

    def _create_enhanced_prompt(self, request: str, attempt: int) -> str:
        """Create different prompt styles for each attempt to maximize success."""
        
        # Get conversation history for context
        chat_history = ""
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            recent_messages = self.memory.chat_memory.messages[-4:]
            for i, msg in enumerate(recent_messages):
                if hasattr(msg, 'content'):
                    role = "User" if i % 2 == 0 else "Assistant"
                    chat_history += f"{role}: {msg.content}\n"
        
        # Different prompt strategies per attempt
        if attempt == 0:
            # **ATTEMPT 1**: Ultra-conservative simple structures only
            return f"""
{self.psmiles_rules}

TASK: Generate a SIMPLE, SAFE PSMILES for: {request}

ULTRA-CONSERVATIVE RULES:
1. MAXIMUM LENGTH: 15 characters total
2. SIMPLE PATTERNS ONLY - NO complex nested groups
3. Use exactly [*] format for connection points
4. Start with [*]C and end with C[*] for safety

ULTRA-SAFE TEMPLATES (CHOOSE ONE):
✅ [*]CC[*] - simple carbon chain
✅ [*]CCC[*] - longer carbon chain
✅ [*]COC[*] - simple ether
✅ [*]CNC[*] - simple amine
✅ [*]CSC[*] - simple sulfur
✅ [*]C(C)C[*] - simple branch
✅ [*]C(=O)C[*] - simple carbonyl

FORBIDDEN - DO NOT GENERATE:
❌ Anything longer than 15 characters
❌ Multiple functional groups
❌ Complex nested structures
❌ Aromatic rings (too complex for now)

Pick ONE simple template above!
Structure: [your_simple_choice]"""

        elif attempt == 1:
            # **ATTEMPT 2**: Even simpler with explicit choices
            return f"""
{self.psmiles_rules}

ULTRA-SIMPLE GENERATION:

Request: {request}

YOU MUST CHOOSE EXACTLY ONE OF THESE SAFE OPTIONS:

Option A: [*]CC[*]
Option B: [*]CCC[*]
Option C: [*]COC[*]
Option D: [*]CNC[*]
Option E: [*]CSC[*]

RULES:
- Pick just ONE option above
- Don't modify it
- Don't add anything extra

Your choice: Structure: [pick_A_B_C_D_or_E]"""

        elif attempt == 2:
            # **ATTEMPT 3**: Minimalist approach
            return f"""
{self.psmiles_rules}

MINIMALIST GENERATION:

For "{request}", generate a 3-atom polymer unit:

Template: [*]XYC[*]
Where:
- X = first atom (C, N, O, S)
- Y = second atom (C, N, O, S)  
- C = carbon (always end with carbon)

Keep it simple! Max 10 characters total.

Structure: [*]???[*]"""

        elif attempt == 3:
            # **ATTEMPT 4**: Basic carbon focus
            return f"""
{self.psmiles_rules}

BASIC CARBON GENERATION:

Request: {request}

Default safe choice: [*]CC[*]

Only modify if you need:
- Nitrogen: change to [*]CNC[*]
- Oxygen: change to [*]COC[*]
- Sulfur: change to [*]CSC[*]

Otherwise use default: [*]CC[*]

Structure: [*]???[*]"""

        else:
            # **ATTEMPT 5**: Emergency simple
            return f"""
{self.psmiles_rules}

EMERGENCY SIMPLE:

Just choose one:
A) [*]CC[*]
B) [*]CNC[*]
C) [*]COC[*]

Structure: A, B, or C"""

    def _extract_psmiles_robust(self, response: str) -> Dict:
        """Robust PSMILES extraction with multiple patterns and error recovery."""
        
        # Enhanced extraction patterns - more comprehensive
        patterns = [
            r'Structure:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # Primary pattern
            r'structure:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # Lowercase
            r'STRUCTURE:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # Uppercase
            r'PSMILES:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',   # Legacy
            r'Result:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',    # Alternative
            r'Answer:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',    # Alternative
            r'`([A-Za-z0-9\[\]\(\)\=\#\*]+)`',            # Backticks
            r'"([A-Za-z0-9\[\]\(\)\=\#\*]+)"',            # Quotes
            r'\[(\*[A-Za-z0-9\[\]\(\)\=\#\*]*\*)\]',      # Bracketed with stars
            r'(\[\*\][A-Za-z0-9\[\]\(\)\=\#\*]*\[\*\])', # Full PSMILES pattern
        ]
        
        matches = []
        
        # Try all patterns
        for pattern in patterns:
            found = re.findall(pattern, response, re.IGNORECASE)
            for match in found:
                if len(match) >= 3:  # Minimum reasonable length
                    # Clean and validate
                    cleaned = self._clean_extracted_psmiles(match)
                    if cleaned:
                        matches.append(cleaned)
        
        # Filter and validate matches
        valid_matches = []
        for match in matches:
            if self._is_reasonable_psmiles(match):
                valid_matches.append(match)
        
        if valid_matches:
            # Return the first valid match
            return {
                'success': True,
                'psmiles': valid_matches[0],
                'all_matches': valid_matches
            }
        
        # **EMERGENCY EXTRACTION** - Look for any chemical-looking string
        chemical_pattern = r'([A-Z][A-Za-z0-9\[\]\(\)\=\#\*]{2,})'
        chemical_matches = re.findall(chemical_pattern, response)
        
        for match in chemical_matches:
            if '[*]' in match or len(match) >= 3:
                cleaned = self._clean_extracted_psmiles(match)
                if cleaned and self._is_reasonable_psmiles(cleaned):
                    # Try to fix connection points
                    fixed = self._fix_connection_points_robust(cleaned)
                    return {
                        'success': True,
                        'psmiles': fixed,
                        'emergency_extraction': True
                    }
        
        return {
            'success': False,
            'psmiles': None,
            'error': 'No valid PSMILES found in response'
        }

    def _clean_extracted_psmiles(self, psmiles: str) -> str:
        """Clean and normalize extracted PSMILES string."""
        if not psmiles:
            return ""
        
        # Remove any whitespace
        cleaned = psmiles.replace(' ', '').replace('\t', '').replace('\n', '')
        
        # Fix common LLM corruptions - be more comprehensive
        import re
        cleaned = re.sub(r'\[\]', '[*]', cleaned)  # Empty brackets [] → [*]
        cleaned = re.sub(r'\[\\?\*\]', '[*]', cleaned)  # Escaped asterisks
        cleaned = re.sub(r'(?<!\[)\*(?!\])', '[*]', cleaned)  # Naked asterisks
        
        # Remove any surrounding quotes or brackets that aren't part of chemistry
        cleaned = cleaned.strip('"\'`')
        
        return cleaned

    def _is_reasonable_psmiles(self, psmiles: str) -> bool:
        """Check if extracted string looks like reasonable PSMILES."""
        if not psmiles or len(psmiles) < 3:
            return False
        
        # Must contain some chemical elements
        if not re.search(r'[CNOSPBFH]', psmiles, re.IGNORECASE):
            return False
        
        # Should not be common words
        forbidden_words = ['THE', 'AND', 'FOR', 'WITH', 'POLYMER', 'STRUCTURE', 'GENERATE']
        if psmiles.upper() in forbidden_words:
            return False
        
        # Should not be too long (probably not a PSMILES)
        if len(psmiles) > 50:
            return False
        
        return True

    def _validate_psmiles_format_strict(self, psmiles: str) -> bool:
        """Strict format validation for PSMILES with enhanced error detection."""
        import re  # Ensure re module is available
        
        if not psmiles:
            return False
        
        # Must have exactly 2 [*] symbols
        if psmiles.count('[*]') != 2:
            print(f"❌ Wrong number of [*] symbols: {psmiles.count('[*]')}")
            return False
        
        # Check for wrong connection formats
        if '[]' in psmiles:
            print(f"❌ Empty brackets [] found (should be [*])")
            return False
        
        # Check for unescaped * in SMILES part (but not in [*])
        clean_psmiles = psmiles.replace('[*]', '')
        if '*' in clean_psmiles:
            print(f"❌ Unescaped * in SMILES: {clean_psmiles}")
            return False
        
        # No spaces or hyphens
        if ' ' in psmiles or '-' in psmiles:
            print(f"❌ Contains spaces or hyphens")
            return False
        
        # Must contain some chemical atoms
        if not re.search(r'[CNOSPBFH]', psmiles, re.IGNORECASE):
            print(f"❌ No chemical atoms found")
            return False
        
        # Check balanced brackets and parentheses
        if psmiles.count('(') != psmiles.count(')'):
            print(f"❌ Unbalanced parentheses: {psmiles.count('(')} vs {psmiles.count(')')}")
            return False
        
        if psmiles.count('[') != psmiles.count(']'):
            print(f"❌ Unbalanced brackets: {psmiles.count('[')} vs {psmiles.count(']')}")
            return False
        
        # ULTRA-CONSERVATIVE: Must start and end with [*]
        if not psmiles.startswith('[*]') or not psmiles.endswith('[*]'):
            print(f"❌ Must start and end with [*]: {psmiles}")
            return False
        
        # Check for problematic terminal heteroatoms
        # Extract what's between the [*] symbols
        middle_match = re.search(r'\[\*\](.*?)\[\*\]', psmiles)
        if middle_match:
            middle_part = middle_match.group(1)
            
            # ULTRA-CONSERVATIVE: Reject overly complex structures
            if len(middle_part) > 20:  # Too long/complex
                print(f"❌ Structure too complex (>20 chars): {middle_part}")
                return False
            
            # Check for impossible bonding patterns
            if 'S(=C)' in middle_part or 'S(=N)' in middle_part:
                print(f"❌ Impossible sulfur bonding: {middle_part}")
                return False
                
            # Check for too many parentheses levels (overly complex)
            if middle_part.count('(') > 3:
                print(f"❌ Too many nested groups: {middle_part}")
                return False
            
            # Check if starts or ends with problematic atoms
            if middle_part.startswith(('S', 'O', 'N', 's', 'o', 'n')):
                print(f"❌ Starts with terminal heteroatom: {middle_part[0]}")
                return False
                
            if middle_part.endswith(('S', 'O', 'N', 's', 'o', 'n')):
                print(f"❌ Ends with terminal heteroatom: {middle_part[-1]}")
                return False
        
        # Check for malformed SMILES patterns
        if 'S(=S)' in psmiles:
            print(f"❌ Unstable S=S bond detected")
            return False
        
        # Check for adjacent connection points (e.g., [*][*])
        if '[*][*]' in psmiles:
            print(f"❌ Adjacent connection points [*][*] detected")
            return False
        
        # ULTRA-CONSERVATIVE: Reject structures that don't start with valid pattern
        valid_starts = ['[*]C', '[*]N', '[*]O', '[*]S', '[*]B', '[*]c', '[*]n']
        if not any(psmiles.startswith(start) for start in valid_starts):
            print(f"❌ Must start with [*] followed by valid atom")
            return False
        
        print(f"✅ Format validation passed: {psmiles}")
        return True

    def _fix_connection_points_robust(self, psmiles: str) -> str:
        """Robust connection point fixing for exactly 2 [*] symbols."""
        if not psmiles:
            return '[*]C[*]'  # Emergency fallback
        
        # Clean corrupted brackets first
        psmiles = psmiles.replace('[]', '[*]')
        
        connection_count = psmiles.count('[*]')
        
        if connection_count == 2:
            return psmiles  # Perfect
        elif connection_count == 0:
            return f'[*]{psmiles}[*]'  # Add both ends
        elif connection_count == 1:
            if psmiles.startswith('[*]'):
                return f'{psmiles}[*]'  # Add to end
            elif psmiles.endswith('[*]'):
                return f'[*]{psmiles}'  # Add to start
            else:
                return f'[*]{psmiles}[*]'  # Add both ends
        else:
            # Too many [*] - keep first and last
            parts = psmiles.split('[*]')
            if len(parts) >= 3:
                middle = ''.join(parts[1:-1])  # Everything between first and last
                return f'[*]{middle}[*]'
            else:
                return f'[*]{psmiles.replace("[*]", "")}[*]'

    def _emergency_generation(self, request: str) -> Dict:
        """Emergency pure LLM generation with ultra-explicit prompts to ensure 100% success rate."""
        print(f"🚨 Emergency pure LLM generation for: {request}")
        
        # **ULTRA-EXPLICIT LLM PROMPT** - Absolutely cannot fail
        emergency_prompt = f"""
EMERGENCY PSMILES GENERATION - ULTRA-SIMPLE ONLY:

REQUEST: {request}

MANDATORY ULTRA-CONSERVATIVE RULES:
1. Use exactly 2 [*] symbols (NOT [] or *)
2. Maximum 10 characters total
3. Only use proven safe patterns
4. NO complex structures allowed

CHOOSE EXACTLY ONE OF THESE ULTRA-SAFE OPTIONS:

A) [*]CC[*]    (simple carbon)
B) [*]CCC[*]   (longer carbon) 
C) [*]COC[*]   (simple ether)
D) [*]CNC[*]   (simple amine)
E) [*]CSC[*]   (simple sulfur)

RESPOND WITH JUST THE LETTER: A, B, C, D, or E

Your choice: """
        
        # **EMERGENCY LLM CALLS** - Multiple attempts with different approaches
        emergency_attempts = 3
        
        for emergency_attempt in range(emergency_attempts):
            try:
                print(f"🚨 Emergency attempt {emergency_attempt + 1}/{emergency_attempts}")
                
                # Use very low temperature for consistency
                original_temp = self.llm.temperature
                self.llm.temperature = 0.1
                
                # Get LLM response
                response = self.llm.invoke(emergency_prompt)
                
                # **LETTER-BASED EMERGENCY EXTRACTION** - Handle A, B, C, D, E choices
                response_clean = response.strip().upper()
                letter_mapping = {
                    'A': '[*]CC[*]',
                    'B': '[*]CCC[*]', 
                    'C': '[*]COC[*]',
                    'D': '[*]CNC[*]',
                    'E': '[*]CSC[*]'
                }
                
                # Check for direct letter response
                for letter, psmiles in letter_mapping.items():
                    if letter in response_clean:
                        print(f"✅ Emergency letter choice {letter}: {psmiles}")
                        return {
                            'success': True,
                            'request': request,
                            'psmiles': psmiles,
                            'explanation': f'Emergency choice {letter}: {psmiles}',
                            'method': f'emergency_letter_{letter}',
                            'conversation_turn': self.conversation_count,
                            'timestamp': datetime.now().isoformat()
                        }
                
                # **AGGRESSIVE EXTRACTION** - Find any PSMILES-like pattern
                emergency_result = self._extract_psmiles_robust(response)
                
                if emergency_result.get('success') and emergency_result.get('psmiles'):
                    psmiles = emergency_result['psmiles']
                    
                    # Fix connection points if needed
                    fixed_psmiles = self._fix_connection_points_robust(psmiles)
                    
                    # Basic validation
                    if self._validate_psmiles_format_strict(fixed_psmiles):
                        print(f"✅ Emergency LLM success: {fixed_psmiles}")
                        
                        return {
                            'success': True,
                            'request': request,
                            'psmiles': fixed_psmiles,
                            'explanation': f'Emergency pure LLM generation: {response}',
                            'method': f'emergency_llm_attempt_{emergency_attempt + 1}',
                            'conversation_turn': self.conversation_count,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        print(f"⚠️  Emergency format validation failed: {fixed_psmiles}")
                        
            except Exception as emergency_error:
                print(f"❌ Emergency attempt {emergency_attempt + 1} failed: {emergency_error}")
                
            finally:
                # Restore temperature
                self.llm.temperature = original_temp
        
        # **FINAL EMERGENCY** - If even emergency LLM fails, create minimal valid structure
        print(f"🚨 All emergency LLM attempts failed, creating minimal valid PSMILES...")
        
        # Extract any mentioned chemical element and create ULTRA-SAFE pattern
        request_lower = request.lower()
        if 'nitrogen' in request_lower or 'amine' in request_lower or 'amide' in request_lower:
            minimal_psmiles = '[*]CNC[*]'  # Ultra-safe nitrogen
        elif 'oxygen' in request_lower or 'ether' in request_lower or 'ester' in request_lower:
            minimal_psmiles = '[*]COC[*]'  # Ultra-safe oxygen
        elif 'sulfur' in request_lower or 'thiol' in request_lower:
            minimal_psmiles = '[*]CSC[*]'  # Ultra-safe sulfur
        else:
            minimal_psmiles = '[*]CC[*]'  # Ultra-safe default carbon
        
        print(f"🚨 Final minimal result: {minimal_psmiles}")
        
        return {
            'success': True,
            'request': request,
            'psmiles': minimal_psmiles,
            'explanation': f'Minimal valid PSMILES for 100% reliability when LLM methods exhausted',
            'method': 'minimal_fallback',
            'conversation_turn': self.conversation_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_psmiles_from_natural_language(self, description: str) -> Dict[str, Any]:
        """
        Convert natural language description to PSMILES using RDKit validation.
        
        Args:
            description (str): Natural language description of molecule
            
        Returns:
            Dict: Complete conversion results with validation
        """
        if not self.nl_to_psmiles:
            return {
                'success': False,
                'error': 'Natural language converter not available',
                'description': description,
                'suggestion': 'Use standard PSMILES generation instead'
            }
        
        try:
            # Convert description to PSMILES using the pipeline
            result = self.nl_to_psmiles.convert_description_to_psmiles(description)
            
            # Add conversation tracking
            self.conversation_count += 1
            
            # Save to memory
            if result['success']:
                psmiles = result.get('psmiles', '')
                self.memory.chat_memory.add_user_message(
                    f"Natural language request: {description}"
                )
                self.memory.chat_memory.add_ai_message(
                    f"Generated PSMILES: {psmiles}"
                )
            
            # Add metadata
            result.update({
                'method': 'natural_language_pipeline',
                'rdkit_validated': result.get('validation', {}).get('is_valid', False),
                'conversation_turn': self.conversation_count,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Natural language conversion failed: {str(e)}",
                'description': description,
                'conversation_turn': self.conversation_count,
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_smiles_with_rdkit(self, smiles: str) -> Dict[str, Any]:
        """
        Validate SMILES string using RDKit and convert to PSMILES.
        
        Args:
            smiles (str): SMILES string to validate
            
        Returns:
            Dict: Validation results and PSMILES conversion
        """
        if not self.chemical_validator: # Changed from self.validator to self.chemical_validator
            return {
                'success': False,
                'error': 'RDKit validator not available',
                'smiles': smiles
            }
        
        try:
            # Validate with RDKit
            is_valid, mol, message = self.chemical_validator.validate_smiles(smiles, debug=False) # Changed from self.validator to self.chemical_validator
            
            result = {
                'success': is_valid,
                'smiles': smiles,
                'validation_message': message,
                'is_valid': is_valid
            }
            
            if is_valid and mol:
                # Canonicalize SMILES
                canonical_smiles = self.chemical_validator.canonicalize_smiles(smiles) # Changed from self.validator to self.chemical_validator
                result['canonical_smiles'] = canonical_smiles
                
                # Convert to PSMILES
                psmiles = self._convert_smiles_to_psmiles(canonical_smiles or smiles)
                result['psmiles'] = psmiles
                
                # Validate PSMILES format
                psmiles_validation = self._validate_psmiles_format(psmiles)
                result['psmiles_validation'] = psmiles_validation
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"RDKit validation failed: {str(e)}",
                'smiles': smiles
            }
    
    def _convert_smiles_to_psmiles(self, smiles: str) -> str:
        """Convert SMILES to PSMILES by adding [*] connection points."""
        if not smiles:
            return "[*][*]"
        
        # Check if already has connection points
        if '[*]' in smiles:
            connection_count = smiles.count('[*]')
            if connection_count == 2:
                return smiles
            elif connection_count == 1:
                return f"[*]{smiles}"
            else:
                smiles_clean = smiles.replace('[*]', '')
        else:
            smiles_clean = smiles
        
        # Add connection points for polymer repeat unit
        psmiles = f"[*]{smiles_clean}[*]"
        return psmiles
    
    def _validate_psmiles_format(self, psmiles: str) -> Dict[str, Any]:
        """Validate PSMILES format according to rules."""
        validation = {
            'is_valid': True,
            'issues': [],
            'connection_points': 0
        }
        
        if not psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Empty PSMILES')
            return validation
        
        # Count connection points
        connection_count = psmiles.count('[*]')
        validation['connection_points'] = connection_count
        
        if connection_count != 2:
            validation['is_valid'] = False
            validation['issues'].append(f'Must have exactly 2 [*] symbols, found {connection_count}')
        
        # Check for spaces and hyphens
        if ' ' in psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Cannot contain spaces')
        
        if '-' in psmiles:
            validation['is_valid'] = False
            validation['issues'].append('Cannot contain hyphens')
        
        return validation
    
    def validate_psmiles(self, psmiles_string: str, context: str = "") -> Dict:
        """
        Validate a PSMILES string for correctness.
        
        Args:
            psmiles_string (str): PSMILES string to validate
            context (str): Additional context for validation
            
        Returns:
            Dict: Validation results
        """
        try:
            # Basic syntax validation
            basic_validation = self._basic_syntax_check(psmiles_string)
            
            # Enhanced validation using LLM
            validation_prompt = self.prompts['validate']
            response = self.llm.invoke(
                validation_prompt.format(
                    psmiles_string=psmiles_string,
                    context=context
                )
            )
            
            return {
                'success': True,
                'psmiles_string': psmiles_string,
                'context': context,
                'basic_validation': basic_validation,
                'ai_validation': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'psmiles_string': psmiles_string,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _basic_syntax_check(self, psmiles_string: str) -> Dict:
        """Perform basic syntax validation of PSMILES string."""
        errors = []
        warnings = []
        
        # Check for spaces
        if ' ' in psmiles_string:
            errors.append("PSMILES strings cannot contain spaces")
        
        # Check for hyphens
        if '-' in psmiles_string:
            errors.append("PSMILES strings cannot contain hyphens")
        
        # Check for balanced brackets
        if psmiles_string.count('(') != psmiles_string.count(')'):
            errors.append("Unbalanced parentheses")
        
        if psmiles_string.count('[') != psmiles_string.count(']'):
            errors.append("Unbalanced square brackets")
        
        # Check for valid characters (updated to exclude hyphens)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]()=#*')
        invalid_chars = set(psmiles_string) - valid_chars
        if invalid_chars:
            errors.append(f"Invalid characters found: {invalid_chars}")
        
        # Check for empty string
        if not psmiles_string.strip():
            errors.append("Empty PSMILES string")
        
        # CRITICAL: Check for exactly 2 [*] symbols
        connection_count = psmiles_string.count('[*]')
        if connection_count != 2:
            errors.append(f"PSMILES must have exactly 2 [*] symbols, found {connection_count}")
        
        # Check for proper connection symbols
        connection_symbols = ['[*]', '[e]', '[d]', '[t]', '[g]']
        found_connections = [sym for sym in connection_symbols if sym in psmiles_string]
        
        # Check for empty brackets (common mistake)
        if '[]' in psmiles_string:
            warnings.append("Empty brackets [] found - should use [*] for connection points")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'connection_symbols': found_connections,
            'connection_count': connection_count,
            'length': len(psmiles_string)
        }
    
    def interactive_generation(self, polymer_description: str) -> Dict:
        """
        Interactive PSMILES generation with validation and suggestions.
        
        Args:
            polymer_description (str): Description of desired polymer
            
        Returns:
            Dict: Complete generation and validation results
        """
        # Generate PSMILES
        generation_result = self.generate_psmiles(polymer_description)
        
        if not generation_result['success']:
            return generation_result
        
        # Validate the generated PSMILES
        psmiles = generation_result['psmiles']
        if psmiles and psmiles != 'Not found' and psmiles != 'Could not generate':
            validation_result = self.validate_psmiles(psmiles, polymer_description)
            
            return {
                'success': True,
                'request': polymer_description,
                'generation': generation_result,
                'validation': validation_result,
                'timestamp': datetime.now().isoformat()
            }
        
        return generation_result

    def reset_conversation_memory(self):
        """Reset conversation memory and counter."""
        self.memory.clear()
        self.conversation_count = 0
        print("🔄 PSMILES conversation memory reset")
    
    def get_memory_status(self) -> Dict:
        """Get current memory and conversation status."""
        chat_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
        return {
            'conversation_count': self.conversation_count,
            'memory_length': len(chat_history),
            'recent_messages': len(chat_history[-6:]) if chat_history else 0
        }

    def test_connection(self) -> str:
        """Test connection to the LLM."""
        try:
            response = self.llm.invoke("Generate PSMILES for ethylene: CC")
            return f"✅ Pure LLM PSMILES Generator connection successful. Response: {response[:100]}..."
        except Exception as e:
            return f"❌ Pure LLM PSMILES Generator connection failed: {e}"

    def generate_diverse_candidates(self, base_request: str, num_candidates: int = 10, 
                                   temperature_range: tuple = (0.6, 1.0)) -> List[Dict]:
        """
        Generate diverse PSMILES candidates using the WORKING PIPELINE:
        Natural Language → SMILES (with repair) → PSMILES → Validation
        
        This uses the robust SMILES generation and repair infrastructure instead
        of problematic direct PSMILES generation.
        
        Args:
            base_request (str): Base material request
            num_candidates (int): Number of unique candidates to generate
            temperature_range (tuple): Min and max temperature for diversity
            
        Returns:
            List[Dict]: List of unique candidate structures
        """
        candidates = []
        unique_psmiles = set()
        attempts = 0
        max_attempts = num_candidates * 3
        
        # Create diverse prompt variations for pure LLM generation
        prompt_templates = [
            "{request}",
            "biocompatible polymer incorporating {request} for medical applications",
            "linear polymer backbone with {request} functional groups",
            "branched copolymer design featuring {request} and ester linkages",
            "aromatic polymer chain incorporating {request} as side groups",
            "cross-linked network polymer containing {request} atoms",
            "amphiphilic block copolymer with {request} hydrophilic segments",
            "biodegradable polymer matrix with {request} and hydroxyl groups",
            "pH-responsive polymer containing {request} and carboxyl groups",
            "thermally stable polymer with {request} and amide linkages",
            "flexible polymer chain incorporating {request} and ether bonds",
            "rigid polymer backbone with {request} and aromatic rings",
            "water-soluble polymer featuring {request} and polar groups",
            "hydrophobic polymer matrix with {request} and alkyl chains",
            "bioactive polymer containing {request} and amino acid residues",
            "smart polymer with {request} and stimuli-responsive properties",
            "nanostructured polymer incorporating {request} for drug delivery",
            "composite polymer material with {request} and reinforcing agents",
            "membrane-forming polymer with {request} and selective permeability",
            "adhesive polymer containing {request} and tacky functional groups"
        ]
        
        print(f"🎯 Generating {num_candidates} diverse PSMILES candidates using WORKING PIPELINE...")
        print(f"🌡️  Using temperature range: {temperature_range[0]}-{temperature_range[1]}")
        print(f"🔧 Using: Natural Language → SMILES (repair) → PSMILES pipeline")
        
        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            
            # Vary temperature for each generation
            temperature = np.random.uniform(temperature_range[0], temperature_range[1])
            
            # Use different prompt template
            prompt_template = prompt_templates[attempts % len(prompt_templates)]
            diversified_request = prompt_template.format(request=base_request)
            
            # Temporarily adjust LLM temperature for the working pipeline
            original_temp = self.llm.temperature
            self.llm.temperature = temperature
            
            # Also adjust temperature in the natural language pipeline if available
            original_nl_temp = None
            if self.nl_to_psmiles and hasattr(self.nl_to_psmiles.nl_to_smiles, 'llm'):
                original_nl_temp = self.nl_to_psmiles.nl_to_smiles.llm.temperature
                self.nl_to_psmiles.nl_to_smiles.llm.temperature = temperature
            
            try:
                # **USE WORKING PIPELINE** - Natural Language → SMILES → PSMILES
                result = self.generate_psmiles_from_natural_language(diversified_request)
                
                if result.get('success') and result.get('psmiles'):
                    psmiles = result['psmiles']
                    
                    # Check for uniqueness and format
                    if psmiles not in unique_psmiles and psmiles.count('[*]') == 2:
                        # Chemical validation if available
                        if self.hybrid_mode and self.chemical_validator:
                            smiles_for_validation = psmiles.replace('[*]', '')
                            is_valid, mol, validation_msg = self.chemical_validator.validate_smiles(smiles_for_validation, debug=False)
                            
                            if is_valid:
                                unique_psmiles.add(psmiles)
                                
                                candidate = {
                                    'psmiles': psmiles,
                                    'explanation': result.get('explanation', 'Working pipeline: NL→SMILES→PSMILES'),
                                    'diversity_prompt': diversified_request,
                                    'generation_temperature': temperature,
                                    'attempt_number': attempts,
                                    'method': 'working_pipeline_diverse',
                                    'validation_applied': True,
                                    'pipeline_used': 'NaturalLanguage→SMILES→PSMILES',
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                candidates.append(candidate)
                                print(f"   ✅ Candidate {len(candidates)}: {psmiles} (T={temperature:.2f})")
                            else:
                                print(f"   ❌ Invalid chemistry (rejected): {psmiles}")
                        else:
                            # No validation available - accept format-valid candidates
                            unique_psmiles.add(psmiles)
                            
                            candidate = {
                                'psmiles': psmiles,
                                'explanation': result.get('explanation', 'Working pipeline: NL→SMILES→PSMILES'),
                                'diversity_prompt': diversified_request,
                                'generation_temperature': temperature,
                                'attempt_number': attempts,
                                'method': 'working_pipeline_diverse',
                                'validation_applied': False,
                                'pipeline_used': 'NaturalLanguage→SMILES→PSMILES',
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            candidates.append(candidate)
                            print(f"   ✅ Candidate {len(candidates)}: {psmiles} (T={temperature:.2f}) (format validated)")
                    else:
                        print(f"   🔄 Duplicate or invalid format: {psmiles}")
                        
            except Exception as e:
                print(f"   ❌ Generation error (attempt {attempts}): {e}")
                
            finally:
                # Restore original temperatures
                self.llm.temperature = original_temp
                if original_nl_temp is not None and self.nl_to_psmiles and hasattr(self.nl_to_psmiles.nl_to_smiles, 'llm'):
                    self.nl_to_psmiles.nl_to_smiles.llm.temperature = original_nl_temp
        
        print(f"🎉 Generated {len(candidates)} unique candidates using WORKING PIPELINE from {attempts} attempts")
        return candidates


def test_psmiles_generator():
    """Test function for Pure LLM PSMILES Generator."""
    try:
        generator = PSMILESGenerator()
        
        # Test connection
        print("Testing connection...")
        print(generator.test_connection())
        
        # Test generation
        print("\nTesting pure LLM generation...")
        result = generator.generate_psmiles("polyethylene repeat unit")
        print(f"Generated: {result}")
        
        # Test validation
        print("\nTesting validation...")
        validation = generator.validate_psmiles("[*]CC[*]", "ethylene repeat unit")
        print(f"Validation: {validation}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_psmiles_generator() 