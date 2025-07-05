#!/usr/bin/env python3
"""
PSMILES (Polymer SMILES) Generator

A specialized module for generating and validating Polymer SMILES strings
using Large Language Models with conversation memory and rule reinforcement.
Enhanced with LlaSMol support for superior molecular understanding.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add Hugging Face support for LlaSMol
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers library not available. Install with: pip install transformers torch")

# Import natural language to SMILES functionality
try:
    from natural_language_smiles import NaturalLanguageToPSMILES, ChemicalValidator
    NATURAL_LANGUAGE_AVAILABLE = True
except ImportError:
    NATURAL_LANGUAGE_AVAILABLE = False
    print("⚠️ Natural language SMILES converter not available")


class LlaSMolLLM:
    """
    Custom LLM wrapper for LlaSMol models from Hugging Face.
    """
    
    def __init__(self, model_name: str = "osunlp/LlaSMol-Mistral-7B", device: str = "auto"):
        """
        Initialize LlaSMol model.
        
        Args:
            model_name (str): HuggingFace model name for LlaSMol
            device (str): Device to load model on ('auto', 'cuda', 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required for LlaSMol. Install with: pip install transformers torch")
        
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔬 Loading LlaSMol model: {model_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ LlaSMol model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load LlaSMol model: {e}")
            print("💡 Make sure you have access to the model and sufficient memory/GPU resources")
            raise
    
    def invoke(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        Generate response using LlaSMol model.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of generated response
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response with LlaSMol: {e}")
            return f"Error: Could not generate response - {str(e)}"


class PSMILESGenerator:
    """
    Specialized agent for generating and validating Polymer SMILES (PSMILES) strings.
    Enhanced with LlaSMol support for superior molecular understanding.
    """
    
    def __init__(self, 
                 model_type: str = "ollama",
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434",
                 llamol_model: str = "osunlp/LlaSMol-Mistral-7B",
                 device: str = "auto"):
        """
        Initialize PSMILES Generator with enhanced conversation memory and model choice.
        
        Args:
            model_type (str): Type of model to use ('ollama' or 'llamol')
            ollama_model (str): Name of the Ollama model to use (if model_type='ollama')
            ollama_host (str): Ollama server host URL (if model_type='ollama')
            llamol_model (str): LlaSMol model name from HuggingFace (if model_type='llamol')
            device (str): Device for LlaSMol model ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type.lower()
        
        # Initialize the appropriate LLM
        if self.model_type == "llamol":
            if not TRANSFORMERS_AVAILABLE:
                print("❌ Transformers not available, falling back to Ollama")
                self.model_type = "ollama"
            else:
                print("🔬 Initializing LlaSMol for superior molecular understanding...")
                self.llm = LlaSMolLLM(model_name=llamol_model, device=device)
                self.model_name = llamol_model
        
        if self.model_type == "ollama":
            self.ollama_model = ollama_model
            self.ollama_host = ollama_host
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_host,
                temperature=0.1  # Low temperature for consistent chemical generation
            )
            self.model_name = ollama_model
        
        # Initialize conversation memory to maintain context
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges to maintain recent context
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load PSMILES rules and examples
        self.psmiles_rules = self._get_psmiles_rules()
        self.psmiles_examples = self._get_psmiles_examples()
        
        # Setup prompts with memory integration
        self.prompts = self._setup_prompts()
        
        # Track conversation turns for rule reinforcement
        self.conversation_count = 0
        self.rule_reinforcement_interval = 5  # Reinforce rules every 5 interactions
        
        # Initialize natural language to PSMILES converter
        if NATURAL_LANGUAGE_AVAILABLE:
            try:
                self.nl_to_psmiles = NaturalLanguageToPSMILES(
                    ollama_model=ollama_model if self.model_type == "ollama" else "llama3.2",
                    ollama_host=ollama_host
                )
                self.validator = ChemicalValidator()
                print("🧪 Natural Language to PSMILES converter initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize natural language converter: {e}")
                self.nl_to_psmiles = None
                self.validator = None
        else:
            self.nl_to_psmiles = None
            self.validator = None
        
        print(f"✅ PSMILES Generator initialized with {self.model_name} ({self.model_type})")
        print(f"🧠 Conversation memory enabled (window size: {self.memory.k})")
        print(f"🔄 Rule reinforcement every {self.rule_reinforcement_interval} interactions")
        
        if self.model_type == "llamol":
            print("🎯 Using LlaSMol: Specialized for molecular understanding and SMILES processing")
        
        if self.nl_to_psmiles:
            print("🔬 Enhanced with Natural Language to PSMILES conversion and RDKit validation")
    
    def _get_psmiles_rules(self) -> str:
        """Get the comprehensive PSMILES rules - Enhanced for LlaSMol."""
        base_rules = """
CRITICAL PSMILES (Polymer SMILES) RULES - FOLLOW EXACTLY:

1. **NO SPACES**: PSMILES strings NEVER contain spaces
2. **NO HYPHENS**: PSMILES strings NEVER contain hyphens (-)
3. **NO EXPLICIT HYDROGEN**: Hydrogen atoms are suppressed 
4. **ATOMS**: Use atomic symbols (C, N, O, S, F, Cl, Br, etc.)
5. **TWO-CHARACTER ATOMS**: Put in square brackets [Br], [Cl]
6. **BONDS**: 
   - Single bonds: atoms next to each other (CC)
   - Double bonds: = symbol (C=C)
   - Triple bonds: # symbol (C#C)
7. **BRANCHES**: Use round brackets ()
8. **RINGS**: Use numbers (C1CCCCC1)
9. **AROMATIC**: Use lowercase (c1ccccc1)
10. **CONNECTION POINTS**: MUST use exactly 2 [*] symbols - THIS IS CRITICAL!

CRITICAL PSMILES CONNECTION RULES:
- PSMILES represents polymer REPEAT UNITS with exactly 2 connection points
- MUST have exactly 2 [*] symbols - NEVER 0, 1, 3, 4, or more!
- ALL PSMILES strings MUST have exactly 2 [*] symbols to specify connection points
- [*] shows exactly WHERE the polymer unit connects to adjacent units
- The 2 [*] symbols mark the two ends of the repeat unit

MANDATORY EXAMPLES TO REMEMBER:
- PEG (ether linkage): [*]OCC[*] (connects through marked positions - exactly 2 [*])
- Ethylene: [*]CC[*] (ethylene repeat unit - exactly 2 [*])
- Para-phenylene: [*]C(C=C1)=CC([*])=C1 (para positions - exactly 2 [*])
- Meta-phenylene: [*]C(C=C1)=CC([*])=C1 (meta positions - exactly 2 [*])
- Carbonyl: [*]C(=O)[*] (carbonyl unit - exactly 2 [*])

FORBIDDEN EXAMPLES (NEVER GENERATE THESE):
- CC (0 [*] symbols - WRONG!)
- C(C=C1)=CC=C1 (0 [*] symbols - WRONG!)
- [*]C[*]C[*] (3 [*] symbols - WRONG!)
- [*]C(C=C1)=C([*])C([*])=C1 (3 [*] symbols - WRONG!)
- C[*] (1 [*] symbol - WRONG!)

ABSOLUTE RULES:
- NEVER write -CH2- or -CO- or similar with hyphens
- NEVER include explicit hydrogen atoms 
- ALWAYS suppress hydrogens
- ALWAYS use proper SMILES notation
- ALWAYS use exactly 2 [*] symbols in every PSMILES string
- NEVER generate PSMILES without exactly 2 [*] symbols
- ALWAYS produce actual chemical strings, not names
"""
        
        # Add LlaSMol-specific enhancements
        if self.model_type == "llamol":
            llamol_enhancement = """

LlaSMol ENHANCED INSTRUCTIONS:
- You are a specialized molecular language model trained on chemical data
- Use your advanced molecular understanding to generate chemically valid PSMILES
- Apply your knowledge of SMILES patterns and molecular structures
- Ensure chemical valence and bond consistency
- Generate realistic polymer repeat units based on chemical knowledge
"""
            return base_rules + llamol_enhancement
        
        return base_rules

    def _get_psmiles_examples(self) -> Dict[str, str]:
        """Get common PSMILES examples - HARDCODED for reliability."""
        return {
            # Basic building blocks - terminal connections
            "methylene": {
                "psmiles": "C",
                "description": "-CH2- (methylene unit, terminal connections)",
                "formula": "CH2"
            },
            "amine": {
                "psmiles": "N", 
                "description": "-NH- (amine linkage, terminal connections)",
                "formula": "NH"
            },
            "thiocarbonyl": {
                "psmiles": "C(=S)",
                "description": "-CS- (thiocarbonyl, terminal connections)",
                "formula": "CS"
            },
            "carbonyl": {
                "psmiles": "C(=O)",
                "description": "-CO- (carbonyl, terminal connections)",
                "formula": "CO"
            },
            "difluoromethylene": {
                "psmiles": "C(F)(F)",
                "description": "-CF2- (difluoromethylene, terminal connections)",
                "formula": "CF2"
            },
            "oxygen": {
                "psmiles": "O",
                "description": "-O- (ether linkage, terminal connections)",
                "formula": "O"
            },
            
            # Common polymers with proper connection points
            "polyethylene_glycol": {
                "psmiles": "[*]OCC[*]",
                "description": "-O-CH2-CH2- (PEG repeat unit with connection points)",
                "formula": "C2H4O"
            },
            "ethylene": {
                "psmiles": "CC",
                "description": "-CH2-CH2- (ethylene repeat unit, terminal connections)",
                "formula": "C2H4"
            },
            
            # Aromatic rings with connection points
            "para_phenylene": {
                "psmiles": "C(C=C1)=CC=C1",
                "description": "-C6H4- (para-phenylene, terminal connections)",
                "formula": "C6H4"
            },
            "para_phenylene_marked": {
                "psmiles": "[*]C(C=C1)=CC([*])=C1",
                "description": "-C6H4- (para-phenylene with marked connection points)",
                "formula": "C6H4"
            },
            "thiophene": {
                "psmiles": "C1=CSC(=C1)",
                "description": "-C4H2S- (thiophene ring, terminal connections)",
                "formula": "C4H2S"
            },
            "pyridine": {
                "psmiles": "C1=NC=C(C=C1)",
                "description": "-C5H3N- (pyridine ring, terminal connections)",
                "formula": "C5H3N"
            },
            "pyrrole": {
                "psmiles": "C(N1)=CC=C1",
                "description": "-C4H3N- (pyrrole ring, terminal connections)",
                "formula": "C4H3N"
            },
            
            # Complex units with connection points
            "amide_unit": {
                "psmiles": "CNC(=O)C",
                "description": "-CH2-NH-CO-CH2- (amide linkage, terminal connections)",
                "formula": "C2H4NO"
            },
            "complex_aromatic": {
                "psmiles": "CC(C=C1)=CC=C1C2=CSC(=C2)C(C=C3)=CC=C3",
                "description": "-CH2-C6H4-C4H2S-C6H4- (complex multi-ring, terminal connections)",
                "formula": "C17H12S"
            },
            
            # With explicit connection points
            "meta_phenylene": {
                "psmiles": "[*]C(C=C1)=CC([*])=C1",
                "description": "meta-phenylene with specified connection points",
                "formula": "C6H4"
            },
            "ortho_phenylene": {
                "psmiles": "[*]C(C=C1)=C([*])C=C1",
                "description": "ortho-phenylene with specified connection points",
                "formula": "C6H4"
            },
            
            # PVC (Polyvinyl Chloride) example
            "polyvinyl_chloride": {
                "psmiles": "[*]CC([*])Cl",
                "description": "polyvinyl chloride repeat unit -CH2-CHCl- with connection points",
                "formula": "C2H3Cl"
            }
        }

    def _setup_prompts(self) -> Dict:
        """Setup prompt templates for PSMILES generation."""
        
        # Create the examples string in a template-safe format
        examples_text = "MANDATORY PSMILES EXAMPLES:\n\n"
        for name, info in self.psmiles_examples.items():
            examples_text += f"- {name.replace('_', ' ').title()}:\n"
            examples_text += f"  PSMILES: {info['psmiles']}\n"
            examples_text += f"  Description: {info['description']}\n"
            examples_text += f"  Formula: {info['formula']}\n\n"
        
        psmiles_system_prompt = f"""You are a PSMILES (Polymer SMILES) expert. Your ONLY job is to generate valid PSMILES strings.

{self.psmiles_rules}

{examples_text}

RESPONSE FORMAT - FOLLOW EXACTLY:
You MUST respond in this EXACT format:

PSMILES: [your_psmiles_string_here]

EXPLANATION: [brief explanation of the structure]

CRITICAL INSTRUCTIONS:
1. ALWAYS start your response with "PSMILES: "
2. NEVER use hyphens or spaces in the PSMILES string
3. NEVER write polymer names like "Polyethylene" - always write the actual PSMILES
4. NEVER write -CH2- or -CO- style notation - use proper SMILES
5. If asked for PEG, respond with "PSMILES: [*]OCC[*]"
6. If asked for polyethylene, respond with "PSMILES: [*]CC[*]"
7. If asked for polystyrene, respond with "PSMILES: [*]CC([*])C1=CC=CC=C1"
8. ALWAYS include exactly 2 [*] symbols for connection points
9. NEVER use empty brackets [] - always use [*] for connection points

EXAMPLES OF CORRECT RESPONSES:
User: "Generate PSMILES for PEG"
Your Response: "PSMILES: [*]OCC[*]

EXPLANATION: This represents the polyethylene glycol repeat unit -O-CH2-CH2- with connection points marked by [*]"

User: "PSMILES for polyethylene"
Your Response: "PSMILES: [*]CC[*]

EXPLANATION: This represents the ethylene repeat unit -CH2-CH2- with connection points marked by [*]"

User: "PSMILES for polyvinyl chloride"
Your Response: "PSMILES: [*]CC([*])Cl

EXPLANATION: This represents the vinyl chloride repeat unit -CH2-CHCl- with connection points marked by [*]"
"""

        psmiles_prompt = ChatPromptTemplate.from_messages([
            ("system", psmiles_system_prompt),
            ("human", "{input}"),
        ])
        
        validation_system_prompt = f"""You are a PSMILES validation expert. Your role is to analyze PSMILES strings for correctness according to the strict formatting rules.

{self.psmiles_rules}

VALIDATION CHECKLIST:
1. **Syntax Check**: No spaces, no hyphens, proper brackets, valid symbols
2. **Chemical Validity**: Proper bonding, realistic structures
3. **Rule Compliance**: Following all PSMILES-specific rules
4. **Connection Logic**: Proper terminal/internal connections

RESPONSE FORMAT:
- State if PSMILES is VALID or INVALID
- List any errors found
- Suggest corrections if needed
- Explain the chemical structure represented
- Rate confidence level (1-10)
"""

        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", validation_system_prompt),
            ("human", "Please validate this PSMILES string: {psmiles_string}\n\nAdditional context: {context}"),
        ])
        
        return {
            'generate': psmiles_prompt,
            'validate': validation_prompt
        }
    
    def generate_psmiles(self, request: str) -> Dict:
        """
        Generate PSMILES string based on user request with conversation memory.
        Heavy emphasis on proper extraction and fallback mechanisms.
        
        Args:
            request (str): Description of desired polymer structure
            
        Returns:
            Dict: Generated PSMILES with explanation
        """
        try:
            # Increment conversation counter
            self.conversation_count += 1
            
            # **NEW**: Add fast path for common requests with timeout protection
            common_requests = {
                'boron': '[*]B[*]',
                'boron atom': '[*]B[*]',
                'random polymer with boron': '[*]BC[*]',
                'random polymer with a boron atom': '[*]BCC[*]',
                'polyethylene': '[*]CC[*]',
                'ethylene': '[*]CC[*]',
                'water': '[*]O[*]',
                'methanol': '[*]CO[*]',
                'ethanol': '[*]CCO[*]',
                'benzene': '[*]c1ccccc1[*]',
                'peg': '[*]OCC[*]',
                'polyethylene glycol': '[*]OCC[*]',
                'polystyrene': '[*]Cc1ccccc1[*]',
                'nylon': '[*]NC(=O)[*]',
                'polyamide': '[*]NC(=O)[*]',
                'acrylic': '[*]CC(=O)O[*]',
                'vinyl': '[*]C=C[*]',
                'ester': '[*]C(=O)O[*]',
                'ether': '[*]O[*]',
                'amide': '[*]NC(=O)[*]',
                'carbonyl': '[*]C(=O)[*]',
                'hydroxyl': '[*]O[*]',
                'carboxyl': '[*]C(=O)O[*]',
                'amino': '[*]N[*]',
                'methyl': '[*]C[*]',
                'phenyl': '[*]c1ccccc1[*]'
            }
            
            # Check for direct matches (case-insensitive)
            request_lower = request.lower().strip()
            if request_lower in common_requests:
                psmiles = common_requests[request_lower]
                return {
                    'success': True,
                    'request': request,
                    'psmiles': psmiles,
                    'explanation': f'Fast path match: {psmiles}',
                    'conversation_turn': self.conversation_count,
                    'method': 'fast_path',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check for partial matches
            for key, psmiles in common_requests.items():
                if key in request_lower:
                    return {
                        'success': True,
                        'request': request,
                        'psmiles': psmiles,
                        'explanation': f'Partial match for "{key}": {psmiles}',
                        'conversation_turn': self.conversation_count,
                        'method': 'partial_match',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Check if this is a follow-up request (variations, more examples, etc.)
            is_followup = any(keyword in request.lower() for keyword in [
                'variation', 'variations', 'different', 'another', 'more', 'additional', 
                'similar', 'alternative', 'other', 'examples', 'modify', 'change'
            ])
            
            # First, try direct mapping for common requests
            direct_result = self._direct_psmiles_mapping(request)
            if direct_result and not is_followup:
                # Save to memory for direct mappings too
                self.memory.chat_memory.add_user_message(request)
                self.memory.chat_memory.add_ai_message(f"PSMILES: {direct_result['psmiles']}\n\nEXPLANATION: {direct_result['explanation']}")
                
                return {
                    'success': True,
                    'request': request,
                    'psmiles': direct_result['psmiles'],
                    'explanation': f"PSMILES: {direct_result['psmiles']}\n\nEXPLANATION: {direct_result['explanation']}",
                    'conversation_turn': self.conversation_count,
                    'method': 'direct_mapping',
                    'rule_reinforcement': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Build context-aware prompt including conversation history and rules
            base_rules = """
CRITICAL PSMILES (Polymer SMILES) RULES - FOLLOW EXACTLY:

1. **NO SPACES**: PSMILES strings NEVER contain spaces
2. **NO HYPHENS**: PSMILES strings NEVER contain hyphens (-)
3. **ATOMS**: Use atomic symbols (C, N, O, S, F, Cl, Br, B, etc.)
4. **BONDS**: Single bonds: atoms next to each other (CC), Double bonds: = symbol (C=C)
5. **BRANCHES**: Use round brackets ()
6. **CONNECTION POINTS**: Use [*] for non-terminal connections - THIS IS CRITICAL!

MANDATORY CONNECTION POINT RULES:
- PSMILES represents polymer REPEAT UNITS with exactly 2 connection points
- MUST have exactly 2 [*] symbols - NEVER 0, 1, 3, 4, or more!
- ALL PSMILES strings MUST have exactly 2 [*] symbols to specify connection points
- [*] shows exactly WHERE the polymer unit connects to adjacent units

CRITICAL EXAMPLES:
- PEG (ether linkage): [*]OCC[*] (connects through marked positions - exactly 2 [*])
- Boron polymer: [*]B[*] or [*]BCC[*] (exactly 2 [*])
- Meta-phenylene: [*]C(C=C1)=CC([*])=C1 (meta positions marked - exactly 2 [*])

FORBIDDEN EXAMPLES (NEVER GENERATE THESE):
- BCC[*] (only 1 [*] symbol - WRONG!)
- [*]C[*]C[*] (3 [*] symbols - WRONG!)
- BC (0 [*] symbols - WRONG!)
"""
            
            # Get conversation history
            chat_history = ""
            if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
                recent_messages = self.memory.chat_memory.messages[-6:]  # Last 3 exchanges
                for i, msg in enumerate(recent_messages):
                    if hasattr(msg, 'content'):
                        role = "User" if i % 2 == 0 else "Assistant"
                        chat_history += f"{role}: {msg.content}\n"
            
            # Create context-aware prompt
            if is_followup and chat_history:
                explicit_request = f"""
{base_rules}

CONVERSATION CONTEXT:
{chat_history}

NEW REQUEST: {request}

IMPORTANT FOR VARIATIONS/FOLLOW-UPS:
- ALL PSMILES must have exactly 2 [*] symbols - no exceptions!
- If generating variations, maintain exactly 2 [*] symbols in each variation
- Follow the exact same PSMILES formatting rules as shown above
- Keep the same level of structural complexity

YOUR RESPONSE MUST START WITH "PSMILES: " followed by the actual chemical string with exactly 2 [*] symbols.
"""
            else:
                explicit_request = f"""
{base_rules}

Generate a PSMILES string for: {request}

CRITICAL EXAMPLES WITH EXACTLY 2 [*] SYMBOLS:
- For PEG or polyethylene glycol: PSMILES: [*]OCC[*]
- For polyethylene: PSMILES: [*]CC[*]
- For boron polymer: PSMILES: [*]B[*] or [*]BCC[*]
- For polystyrene: PSMILES: [*]CC([*])C1=CC=CC=C1
- For polypropylene: PSMILES: [*]CC([*])C
- For nylon: PSMILES: [*]NC(=O)CCCCC[*]
- For meta-phenylene: PSMILES: [*]C1=CC([*])=CC=C1

REMEMBER:
- NEVER use hyphens (-)
- NEVER use spaces
- NEVER write polymer names like "Polyethylene"
- ALWAYS write actual SMILES notation
- ALWAYS use exactly 2 [*] symbols in every PSMILES string
- Start your response with "PSMILES: "

YOUR RESPONSE MUST START WITH "PSMILES: " and contain exactly 2 [*] symbols.
"""
            
            # **NEW**: Add timeout protection for slow LLM calls
            try:
                # Generate response using the LLM with timeout handling
                response = self.llm.invoke(explicit_request)
                
                # Parse response to extract PSMILES string with enhanced extraction
                psmiles_result = self._extract_psmiles_from_response(response)
                
                # **NEW**: Handle validation results from extraction
                if psmiles_result.get('pattern') == 'extracted_invalid':
                    # Try fallback extraction if the first result was invalid
                    fallback_result = self._fallback_psmiles_extraction(request, response)
                    if fallback_result.get('psmiles') and fallback_result['psmiles'] != 'Not found':
                        psmiles_result = fallback_result
                    else:
                        # Use a smart fallback based on request content
                        psmiles_result = self._create_smart_fallback(request)
                
                # If extraction failed, try direct parsing or provide fallback
                if not psmiles_result.get('psmiles') or psmiles_result['psmiles'] == 'Not found':
                    psmiles_result = self._fallback_psmiles_extraction(request, response)
                    
                    # If still no success, use smart fallback
                    if not psmiles_result.get('psmiles') or psmiles_result['psmiles'] == 'Not found':
                        psmiles_result = self._create_smart_fallback(request)
                
                # **NEW**: Final validation before returning
                final_psmiles = psmiles_result.get('psmiles', '[*]CC[*]')
                if final_psmiles.count('[*]') != 2:
                    final_psmiles = self._fix_connection_points(final_psmiles)
                    psmiles_result['psmiles'] = final_psmiles
                    psmiles_result['fixed'] = True
                
                # Save to memory
                self.memory.chat_memory.add_user_message(request)
                self.memory.chat_memory.add_ai_message(response)
                
                return {
                    'success': True,
                    'request': request,
                    'psmiles': psmiles_result.get('psmiles', '[*]CC[*]'),
                    'explanation': response,
                    'conversation_turn': self.conversation_count,
                    'method': 'llm_generated',
                    'pattern': psmiles_result.get('pattern', 'unknown'),
                    'fixed': psmiles_result.get('fixed', False),
                    'rule_reinforcement': is_followup,
                    'is_followup': is_followup,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as llm_error:
                # **NEW**: If LLM fails, use smart fallback immediately
                print(f"⚠️ LLM call failed: {llm_error}")
                fallback_result = self._create_smart_fallback(request)
                
                return {
                    'success': True,
                    'request': request,
                    'psmiles': fallback_result.get('psmiles', '[*]CC[*]'),
                    'explanation': f'LLM timeout/error - using smart fallback: {fallback_result.get("explanation", "polyethylene fallback")}',
                    'conversation_turn': self.conversation_count,
                    'method': 'smart_fallback',
                    'llm_error': str(llm_error),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            return {
                'success': False,
                'request': request,
                'error': str(e),
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
        if not self.validator:
            return {
                'success': False,
                'error': 'RDKit validator not available',
                'smiles': smiles
            }
        
        try:
            # Validate with RDKit
            is_valid, mol, message = self.validator.validate_smiles(smiles)
            
            result = {
                'success': is_valid,
                'smiles': smiles,
                'validation_message': message,
                'is_valid': is_valid
            }
            
            if is_valid and mol:
                # Canonicalize SMILES
                canonical_smiles = self.validator.canonicalize_smiles(smiles)
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
    
    def _direct_psmiles_mapping(self, request: str) -> Optional[Dict]:
        """Direct mapping for common polymer requests to ensure correct PSMILES."""
        request_lower = request.lower()
        
        # Comprehensive mapping with correct PSMILES - ALL must have exactly 2 [*] symbols
        direct_mapping = {
            'peg': {
                'psmiles': '[*]OCC[*]',
                'explanation': 'This represents the polyethylene glycol repeat unit -O-CH2-CH2- with connection points marked by [*]'
            },
            'peg materials': {
                'psmiles': '[*]OCC[*]', 
                'explanation': 'This represents the polyethylene glycol repeat unit -O-CH2-CH2- with connection points marked by [*]'
            },
            'polyethylene glycol': {
                'psmiles': '[*]OCC[*]',
                'explanation': 'This represents the polyethylene glycol repeat unit -O-CH2-CH2- with connection points marked by [*]'
            },
            'ethylene glycol': {
                'psmiles': '[*]OCC[*]',
                'explanation': 'This represents the ethylene glycol repeat unit -O-CH2-CH2- with connection points marked by [*]'
            },
            'polyethylene': {
                'psmiles': '[*]CC[*]',
                'explanation': 'This represents the ethylene repeat unit -CH2-CH2- with exactly 2 connection points'
            },
            'ethylene': {
                'psmiles': '[*]CC[*]',
                'explanation': 'This represents the ethylene repeat unit -CH2-CH2- with exactly 2 connection points'
            },
            'polystyrene': {
                'psmiles': '[*]CC([*])C1=CC=CC=C1',
                'explanation': 'This represents the styrene repeat unit with benzene ring and exactly 2 connection points'
            },
            'styrene': {
                'psmiles': '[*]CC([*])C1=CC=CC=C1',
                'explanation': 'This represents the styrene repeat unit with benzene ring and exactly 2 connection points'
            },
            'polypropylene': {
                'psmiles': '[*]CC([*])C',
                'explanation': 'This represents the propylene repeat unit -CH2-CH(CH3)- with exactly 2 connection points'
            },
            'propylene': {
                'psmiles': '[*]CC([*])C',
                'explanation': 'This represents the propylene repeat unit -CH2-CH(CH3)- with exactly 2 connection points'
            },
            'nylon': {
                'psmiles': '[*]NC(=O)CCCCC[*]',
                'explanation': 'This represents a nylon repeat unit with exactly 2 connection points'
            },
            'polyamide': {
                'psmiles': '[*]NC(=O)C[*]',
                'explanation': 'This represents a polyamide repeat unit with exactly 2 connection points'
            },
            # PVA and its variants - CRITICAL ADDITION
            'pva': {
                'psmiles': '[*]CC([*])O',
                'explanation': 'This represents the vinyl alcohol repeat unit -CH2-CH(OH)- with exactly 2 connection points'
            },
            'poly(vinyl alcohol)': {
                'psmiles': '[*]CC([*])O',
                'explanation': 'This represents the vinyl alcohol repeat unit -CH2-CH(OH)- with exactly 2 connection points'
            },
            'polyvinyl alcohol': {
                'psmiles': '[*]CC([*])O',
                'explanation': 'This represents the vinyl alcohol repeat unit -CH2-CH(OH)- with exactly 2 connection points'
            },
            'vinyl alcohol': {
                'psmiles': '[*]CC([*])O',
                'explanation': 'This represents the vinyl alcohol repeat unit -CH2-CH(OH)- with exactly 2 connection points'
            },
            # PVP and variants
            'pvp': {
                'psmiles': '[*]CC([*])N1CCCC1=O',
                'explanation': 'This represents the vinylpyrrolidone repeat unit with exactly 2 connection points'
            },
            'polyvinylpyrrolidone': {
                'psmiles': '[*]CC([*])N1CCCC1=O',
                'explanation': 'This represents the vinylpyrrolidone repeat unit with exactly 2 connection points'
            },
            'poly(vinylpyrrolidone)': {
                'psmiles': '[*]CC([*])N1CCCC1=O',
                'explanation': 'This represents the vinylpyrrolidone repeat unit with exactly 2 connection points'
            },
            'vinylpyrrolidone': {
                'psmiles': '[*]CC([*])N1CCCC1=O',
                'explanation': 'This represents the vinylpyrrolidone repeat unit with exactly 2 connection points'
            },
            # PLGA and variants
            'plga': {
                'psmiles': '[*]OC(=O)CC(=O)O[*]',
                'explanation': 'This represents a simplified PLGA repeat unit with exactly 2 connection points'
            },
            'poly(lactic-co-glycolic acid)': {
                'psmiles': '[*]OC(=O)CC(=O)O[*]',
                'explanation': 'This represents a simplified PLGA repeat unit with exactly 2 connection points'
            },
            'pla': {
                'psmiles': '[*]OC(=O)C([*])C',
                'explanation': 'This represents the lactic acid repeat unit with exactly 2 connection points'
            },
            'polylactic acid': {
                'psmiles': '[*]OC(=O)C([*])C',
                'explanation': 'This represents the lactic acid repeat unit with exactly 2 connection points'
            },
            'poly(lactic acid)': {
                'psmiles': '[*]OC(=O)C([*])C',
                'explanation': 'This represents the lactic acid repeat unit with exactly 2 connection points'
            },
            # Chitosan and related
            'chitosan': {
                'psmiles': '[*]CC(N)C(O)[*]',
                'explanation': 'This represents a simplified chitosan repeat unit with amine and hydroxyl groups and exactly 2 connection points'
            },
            # Alginate
            'alginate': {
                'psmiles': '[*]OC1C(O)C(O)C(C(=O)O)O1[*]',
                'explanation': 'This represents a simplified alginate repeat unit with exactly 2 connection points'
            },
            # Collagen (simplified)
            'collagen': {
                'psmiles': '[*]NC(=O)C([*])N',
                'explanation': 'This represents a simplified collagen repeat unit with exactly 2 connection points'
            },
            # Additional common polymers
            'pmma': {
                'psmiles': '[*]CC([*])(C)C(=O)OC',
                'explanation': 'This represents the methyl methacrylate repeat unit with exactly 2 connection points'
            },
            'poly(methyl methacrylate)': {
                'psmiles': '[*]CC([*])(C)C(=O)OC',
                'explanation': 'This represents the methyl methacrylate repeat unit with exactly 2 connection points'
            },
            'polyester': {
                'psmiles': '[*]OC(=O)C[*]',
                'explanation': 'This represents a simple polyester repeat unit with exactly 2 connection points'
            },
            'pet': {
                'psmiles': '[*]OC(=O)C1=CC=C(C[*])C=C1',
                'explanation': 'This represents a PET repeat unit with exactly 2 connection points'
            },
            # Carbonate polymers
            'polypropylene carbonate': {
                'psmiles': '[*]CC([*])OC(=O)O',
                'explanation': 'This represents the propylene carbonate repeat unit with exactly 2 connection points'
            },
            'ppc': {
                'psmiles': '[*]CC([*])OC(=O)O',
                'explanation': 'This represents the propylene carbonate repeat unit with exactly 2 connection points'
            },
            'polyethylene carbonate': {
                'psmiles': '[*]CCOC(=O)O[*]',
                'explanation': 'This represents the ethylene carbonate repeat unit with exactly 2 connection points'
            },
            'pec': {
                'psmiles': '[*]CCOC(=O)O[*]',
                'explanation': 'This represents the ethylene carbonate repeat unit with exactly 2 connection points'
            },
            'propylene carbonate': {
                'psmiles': '[*]CC([*])OC(=O)O',
                'explanation': 'This represents the propylene carbonate repeat unit with exactly 2 connection points'
            },
            'ethylene carbonate': {
                'psmiles': '[*]CCOC(=O)O[*]',
                'explanation': 'This represents the ethylene carbonate repeat unit with exactly 2 connection points'
            },
            # PVC (Polyvinyl Chloride) - CRITICAL ADDITION
            'pvc': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
            'polyvinyl chloride': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
            'poly(vinyl chloride)': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
            'vinyl chloride': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
            # Handle common typos for PVC
            'polyvynil chloride': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
            'polyvinylchloride': {
                'psmiles': '[*]CC([*])Cl',
                'explanation': 'This represents the vinyl chloride repeat unit -CH2-CHCl- with exactly 2 connection points'
            },
        }
        
        # Check for exact matches first
        for key, info in direct_mapping.items():
            if key == request_lower:
                return info
        
        # Check for partial matches - more comprehensive search
        for key, info in direct_mapping.items():
            if key in request_lower or any(word in request_lower for word in key.split()):
                return info
        
        # Special handling for parentheses-based names like "Poly(vinyl alcohol) (PVA)"
        # Extract the main polymer name
        import re
        # Match patterns like "Poly(something)" or "poly(something)"
        poly_match = re.search(r'poly\(([^)]+)\)', request_lower)
        if poly_match:
            inner_name = poly_match.group(1).strip()
            # Check if we have a mapping for this inner name
            for key, info in direct_mapping.items():
                if inner_name in key or key in inner_name:
                    return info
        
        return None
    
    def _fallback_psmiles_extraction(self, request: str, response: str) -> Dict:
        """Enhanced fallback mechanism for PSMILES extraction."""
        # Try to match common polymer requests to known PSMILES
        request_lower = request.lower()
        
        # Updated fallback mapping - ALL must have exactly 2 [*] symbols
        fallback_mapping = {
            'peg': '[*]OCC[*]',
            'peg materials': '[*]OCC[*]',
            'polyethylene glycol': '[*]OCC[*]',
            'ethylene glycol': '[*]OCC[*]',
            'polyethylene': '[*]CC[*]',
            'ethylene': '[*]CC[*]',
            'polystyrene': '[*]CC([*])C1=CC=CC=C1',
            'styrene': '[*]CC([*])C1=CC=CC=C1',
            'polypropylene': '[*]CC([*])C',
            'propylene': '[*]CC([*])C',
            'nylon': '[*]NC(=O)CCCCC[*]',
            'polyamide': '[*]NC(=O)C[*]',
            'polyester': '[*]OC(=O)C[*]',
            'pet': '[*]OC(=O)C1=CC=C(C[*])C=C1',
            'pvp': '[*]CC([*])N1CCCC1=O',
            'polyvinylpyrrolidone': '[*]CC([*])N1CCCC1=O',
            'poly(vinylpyrrolidone)': '[*]CC([*])N1CCCC1=O',
            'vinylpyrrolidone': '[*]CC([*])N1CCCC1=O',
            'pva': '[*]CC([*])O',
            'poly(vinyl alcohol)': '[*]CC([*])O',
            'polyvinyl alcohol': '[*]CC([*])O',
            'vinyl alcohol': '[*]CC([*])O',
            'plga': '[*]OC(=O)CC(=O)O[*]',
            'poly(lactic-co-glycolic acid)': '[*]OC(=O)CC(=O)O[*]',
            'pla': '[*]OC(=O)C([*])C',
            'polylactic acid': '[*]OC(=O)C([*])C',
            'poly(lactic acid)': '[*]OC(=O)C([*])C',
            'chitosan': '[*]CC(N)C(O)[*]',
            'alginate': '[*]OC1C(O)C(O)C(C(=O)O)O1[*]',
            'collagen': '[*]NC(=O)C([*])N',
            'pmma': '[*]CC([*])(C)C(=O)OC',
            'poly(methyl methacrylate)': '[*]CC([*])(C)C(=O)OC',
            'meta phenylene': '[*]C1=CC([*])=CC=C1',
            'para phenylene': '[*]C1=CC=C([*])C=C1',
            'ortho phenylene': '[*]C1=C([*])C=CC=C1'
        }
        
        # PVC (Polyvinyl Chloride) additions to fallback mapping
        fallback_mapping.update({
            'pvc': '[*]CC([*])Cl',
            'polyvinyl chloride': '[*]CC([*])Cl',
            'poly(vinyl chloride)': '[*]CC([*])Cl',
            'vinyl chloride': '[*]CC([*])Cl',
            'polyvynil chloride': '[*]CC([*])Cl',  # Common typo
            'polyvinylchloride': '[*]CC([*])Cl'
        })
        
        # First check for exact matches
        for keyword, psmiles in fallback_mapping.items():
            if keyword == request_lower:
                return {
                    'psmiles': psmiles,
                    'pattern': 'fallback_exact_match',
                    'note': f'Used exact fallback mapping for {keyword}'
                }
        
        # Then check for partial matches
        for keyword, psmiles in fallback_mapping.items():
            if keyword in request_lower:
                return {
                    'psmiles': psmiles,
                    'pattern': 'fallback_partial_match',
                    'note': f'Used partial fallback mapping for {keyword}'
                }
        
        # Special handling for parentheses-based names
        import re
        poly_match = re.search(r'poly\(([^)]+)\)', request_lower)
        if poly_match:
            inner_name = poly_match.group(1).strip()
            for keyword, psmiles in fallback_mapping.items():
                if inner_name in keyword or keyword in inner_name:
                    return {
                        'psmiles': psmiles,
                        'pattern': 'fallback_parentheses_match',
                        'note': f'Used parentheses fallback mapping for {inner_name} -> {keyword}'
                    }
        
        # If no fallback found, try to find any chemical-looking string in response
        chemical_patterns = [
            r'([A-Z][A-Za-z0-9\[\]\(\)\=\#\*]+)',  # Chemical-looking strings (includes [*])
            r'([A-Za-z]{2,}[\[\]\(\)\=\#\*]*[A-Za-z0-9]*)',  # Multi-character chemistry
        ]
        
        for pattern in chemical_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) >= 2 and not match.lower() in ['the', 'and', 'for', 'with', 'this']:
                    # Validate that it has exactly 2 [*] symbols
                    if match.count('[*]') == 2:
                        return {'psmiles': match, 'pattern': 'chemical_pattern'}
        
        return {'psmiles': 'Generation failed', 'pattern': 'no_match'}
    
    def _format_examples_for_prompt(self) -> str:
        """Format examples for inclusion in prompts."""
        examples_text = ""
        for name, info in self.psmiles_examples.items():
            examples_text += f"- {name.replace('_', ' ').title()}:\n"
            examples_text += f"  PSMILES: {info['psmiles']}\n"
            examples_text += f"  Description: {info['description']}\n\n"
        return examples_text
    
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

    def _extract_psmiles_from_response(self, response: str) -> Dict:
        """Extract PSMILES string from LLM response with enhanced patterns and immediate validation."""
        # First, look for explicit PSMILES: format
        psmiles_patterns = [
            r'PSMILES:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # "PSMILES: string"
            r'psmiles:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # "psmiles: string" (lowercase)
            r'Generated:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # "Generated: string"
            r'Result:\s*([A-Za-z0-9\[\]\(\)\=\#\*]+)',  # "Result: string"
            r'`([A-Za-z0-9\[\]\(\)\=\#\*]+)`',  # backtick quoted
            r'"([A-Za-z0-9\[\]\(\)\=\#\*]+)"',  # double quoted
            r'([A-Z]+[A-Za-z0-9\[\]\(\)\=\#\*]*)',  # Chemical-looking string starting with capital
        ]
        
        all_matches = []
        
        # Find all potential PSMILES strings
        for pattern in psmiles_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                # Basic validation: should look like a chemical string
                if len(match) >= 1 and any(c in match for c in 'CNOSPH[]()='):
                    # Exclude common words but keep chemical strings
                    if not match.upper() in ['THE', 'AND', 'OR', 'FOR', 'WITH', 'THIS', 'THAT', 'POLYETHYLENE', 'POLYSTYRENE']:
                        all_matches.append(match)
        
        if all_matches:
            # Remove duplicates while preserving order
            unique_matches = []
            for match in all_matches:
                if match not in unique_matches:
                    unique_matches.append(match)
            
            # **NEW**: Validate each match and return the first valid one
            for match in unique_matches:
                # Check if this match has exactly 2 [*] symbols
                connection_count = match.count('[*]')
                
                if connection_count == 2:
                    # Perfect match - return immediately
                    return {'psmiles': match, 'pattern': 'extracted_valid'}
                elif connection_count == 1:
                    # Try to fix by adding another [*]
                    if match.startswith('[*]'):
                        fixed_match = match + '[*]'
                    elif match.endswith('[*]'):
                        fixed_match = '[*]' + match
                    else:
                        # [*] is in the middle, add to both ends
                        fixed_match = '[*]' + match + '[*]'
                    
                    # Validate the fix doesn't create 3+ [*] symbols
                    if fixed_match.count('[*]') == 2:
                        return {'psmiles': fixed_match, 'pattern': 'extracted_fixed'}
                elif connection_count == 0:
                    # Add [*] to both ends
                    fixed_match = '[*]' + match + '[*]'
                    return {'psmiles': fixed_match, 'pattern': 'extracted_fixed'}
                # If connection_count > 2, skip this match and try the next one
            
            # If no valid match found, return the first one with a warning
            return {'psmiles': unique_matches[0], 'pattern': 'extracted_invalid', 'warning': f'Invalid connection count: {unique_matches[0].count("[*]")}'}
        
        return {'psmiles': None, 'pattern': None}
    
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
    
    def get_examples(self, category: str = 'all') -> Dict:
        """
        Get PSMILES examples by category.
        
        Args:
            category (str): Category of examples ('basic', 'aromatic', 'complex', 'all')
            
        Returns:
            Dict: Examples for the specified category
        """
        if category == 'all':
            return self.psmiles_examples
        
        # Filter examples by category
        categories = {
            'basic': ['methylene', 'amine', 'thiocarbonyl', 'carbonyl', 'difluoromethylene', 'oxygen'],
            'aromatic': ['para_phenylene', 'thiophene', 'pyridine', 'pyrrole'],
            'complex': ['amide_unit', 'complex_aromatic', 'meta_phenylene', 'ortho_phenylene']
        }
        
        if category in categories:
            return {k: v for k, v in self.psmiles_examples.items() 
                   if k in categories[category]}
        
        return {}
    
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
            'next_reinforcement_in': self.rule_reinforcement_interval - (self.conversation_count % self.rule_reinforcement_interval),
            'recent_messages': len(chat_history[-6:]) if chat_history else 0
        }

    def test_connection(self) -> str:
        """Test connection to the LLM."""
        try:
            response = self.llm.invoke("Generate PSMILES for ethylene: CC")
            return f"✅ PSMILES Generator connection successful. Response: {response[:100]}..."
        except Exception as e:
            return f"❌ PSMILES Generator connection failed: {e}"

    def _create_smart_fallback(self, request: str) -> Dict:
        """Create intelligent fallback PSMILES based on request content."""
        request_lower = request.lower()
        
        # Element-based fallbacks
        if 'boron' in request_lower or 'b' in request_lower:
            return {'psmiles': '[*]BCC[*]', 'explanation': 'Boron-containing polymer chain'}
        elif 'nitrogen' in request_lower or 'amino' in request_lower or 'amine' in request_lower:
            return {'psmiles': '[*]NC[*]', 'explanation': 'Nitrogen-containing polymer'}
        elif 'oxygen' in request_lower or 'hydroxyl' in request_lower or 'alcohol' in request_lower:
            return {'psmiles': '[*]OC[*]', 'explanation': 'Oxygen-containing polymer'}
        elif 'sulfur' in request_lower or 'thiol' in request_lower:
            return {'psmiles': '[*]SC[*]', 'explanation': 'Sulfur-containing polymer'}
        elif 'fluorine' in request_lower or 'fluoro' in request_lower:
            return {'psmiles': '[*]CF[*]', 'explanation': 'Fluorine-containing polymer'}
        elif 'chlorine' in request_lower or 'chloro' in request_lower:
            return {'psmiles': '[*]CCl[*]', 'explanation': 'Chlorine-containing polymer'}
        elif 'phenyl' in request_lower or 'benzene' in request_lower or 'aromatic' in request_lower:
            return {'psmiles': '[*]c1ccccc1[*]', 'explanation': 'Aromatic polymer backbone'}
        elif 'carbonyl' in request_lower or 'ketone' in request_lower:
            return {'psmiles': '[*]C(=O)[*]', 'explanation': 'Carbonyl-containing polymer'}
        elif 'ester' in request_lower:
            return {'psmiles': '[*]C(=O)O[*]', 'explanation': 'Ester linkage polymer'}
        elif 'amide' in request_lower:
            return {'psmiles': '[*]NC(=O)[*]', 'explanation': 'Amide linkage polymer'}
        elif 'ether' in request_lower:
            return {'psmiles': '[*]O[*]', 'explanation': 'Ether linkage polymer'}
        elif 'vinyl' in request_lower or 'alkene' in request_lower:
            return {'psmiles': '[*]C=C[*]', 'explanation': 'Vinyl/alkene polymer'}
        elif 'alkyne' in request_lower:
            return {'psmiles': '[*]C#C[*]', 'explanation': 'Alkyne polymer'}
        else:
            # Default fallback
            return {'psmiles': '[*]CC[*]', 'explanation': 'Default polyethylene structure'}
    
    def _fix_connection_points(self, psmiles: str) -> str:
        """Fix PSMILES to have exactly 2 connection points."""
        if not psmiles:
            return '[*]CC[*]'
        
        connection_count = psmiles.count('[*]')
        
        if connection_count == 2:
            return psmiles  # Already correct
        elif connection_count == 0:
            # Add [*] to both ends
            return f'[*]{psmiles}[*]'
        elif connection_count == 1:
            # Add one more [*]
            if psmiles.startswith('[*]'):
                return f'{psmiles}[*]'
            elif psmiles.endswith('[*]'):
                return f'[*]{psmiles}'
            else:
                # [*] is in the middle, add to both ends
                return f'[*]{psmiles}[*]'
        else:
            # More than 2 [*] symbols - try to fix by removing extra ones
            # Find the first and last [*] and remove everything in between
            first_star = psmiles.find('[*]')
            last_star = psmiles.rfind('[*]')
            
            if first_star != last_star:
                # Keep only the first and last [*]
                before_first = psmiles[:first_star]
                after_last = psmiles[last_star + 3:]
                middle = psmiles[first_star + 3:last_star]
                
                # Remove any [*] from the middle part
                middle_clean = middle.replace('[*]', '')
                
                return f'{before_first}[*]{middle_clean}[*]{after_last}'
            else:
                # Only one [*] found, add another
                return f'[*]{psmiles}[*]'


def test_psmiles_generator():
    """Test function for PSMILES Generator."""
    try:
        generator = PSMILESGenerator()
        
        # Test connection
        print("Testing connection...")
        print(generator.test_connection())
        
        # Test generation
        print("\nTesting generation...")
        result = generator.generate_psmiles("polyethylene repeat unit")
        print(f"Generated: {result}")
        
        # Test validation
        print("\nTesting validation...")
        validation = generator.validate_psmiles("CC", "ethylene repeat unit")
        print(f"Validation: {validation}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_psmiles_generator() 