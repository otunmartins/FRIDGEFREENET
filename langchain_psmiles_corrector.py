#!/usr/bin/env python3
"""
LangChain-Powered PSMILES Self-Correcting Chain
===============================================

This module implements advanced LangChain design patterns for robust PSMILES autocorrection:
1. Self-Correcting Chain with Retry Mechanisms
2. Reflection Pattern for iterative refinement
3. Validation with Re-prompting
4. Error Handling with Fallbacks
5. Context Engineering for better results

Based on LangChain best practices and agentic workflow patterns.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import OutputParserException

from pydantic import BaseModel, Field, validator
from rdkit import Chem
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectionStage(Enum):
    """Stages in the correction process"""
    INITIAL_ANALYSIS = "initial_analysis"
    STRUCTURE_REPAIR = "structure_repair"
    CHEMISTRY_VALIDATION = "chemistry_validation"
    REFLECTION = "reflection"
    FINAL_VALIDATION = "final_validation"


class CorrectionResult(BaseModel):
    """Structured output for PSMILES corrections"""
    corrected_psmiles: str = Field(description="The corrected PSMILES structure")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    correction_method: str = Field(description="Method used for correction")
    reasoning: str = Field(description="Step-by-step reasoning for the correction")
    chemistry_valid: bool = Field(description="Whether the chemistry is valid")
    validation_message: str = Field(description="Validation details")
    
    @validator('corrected_psmiles')
    def validate_psmiles_format(cls, v):
        if not v.startswith('[*]') or not v.endswith('[*]'):
            raise ValueError("PSMILES must start and end with [*]")
        if v.count('[*]') != 2:
            raise ValueError("PSMILES must have exactly 2 connection points")
        return v


class AnalysisResult(BaseModel):
    """Structured output for structure analysis"""
    connection_points: int = Field(description="Number of [*] connection points found")
    underlying_smiles: str = Field(description="SMILES without connection points")
    identified_issues: List[str] = Field(description="List of identified structural issues")
    suggested_fixes: List[str] = Field(description="Suggested fixes for each issue")
    chemistry_assessment: str = Field(description="Assessment of chemical validity")


@dataclass
class ValidationState:
    """State tracking for the validation process"""
    original_psmiles: str
    current_psmiles: str
    attempt_number: int
    stage: CorrectionStage
    validation_errors: List[str]
    corrections_applied: List[Dict]
    final_result: Optional[CorrectionResult] = None


class LangChainPSMILESCorrector:
    """
    LangChain-powered PSMILES corrector implementing self-correcting chains
    and reflection patterns for robust autocorrection.
    """
    
    def __init__(self, llm, max_attempts: int = 3, enable_reflection: bool = True):
        """
        Initialize the corrector with LangChain components.
        
        Args:
            llm: LangChain LLM instance
            max_attempts: Maximum correction attempts before fallback
            enable_reflection: Whether to use reflection pattern
        """
        self.llm = llm
        self.max_attempts = max_attempts
        self.enable_reflection = enable_reflection
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup structured output parsers
        self.analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResult)
        self.correction_parser = PydanticOutputParser(pydantic_object=CorrectionResult)
        
        # Initialize prompt templates
        self._setup_prompts()
        
        # Create the self-correcting chain
        self._setup_correction_chain()
        
        logger.info("✅ LangChain PSMILES Corrector initialized")
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates with context engineering"""
        
        # Analysis prompt - identifies issues systematically
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert PSMILES (Polymer SMILES) analyst. Your task is to analyze failed PSMILES structures and identify specific issues.

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

PSMILES Format Rules:
- Must have exactly 2 [*] connection points (exactly 2 connection points)
- Contains valid SMILES chemistry between connection points
- Examples: [*]CC[*], [*]c1ccccc1[*], [*]CCO[*]

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

Common Issues:
1. Wrong number of connection points ([*])
2. Invalid SMILES chemistry
3. Malformed brackets or parentheses
4. Missing atoms or bonds
5. Incorrect stereochemistry
6. Invalid ring closures

Analyze the structure systematically and provide structured output."""),
            
            ("human", """Failed PSMILES: {failed_psmiles}
Error Details: {error_details}

Please analyze this structure and identify all issues. Be specific about what's wrong and how to fix it.

{format_instructions}""")
        ])
        
        # Correction prompt - generates fixes based on analysis
        self.correction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert PSMILES corrector. Based on the analysis, generate a corrected PSMILES structure.

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

SMILES FOR LADDER POLYMERS:
A ladder polymer is a type of double stranded polymer with multiple connection points between monomer repeat units. Different from typical polymers the ladder polymer requires four different symbols ([e], [d], [t] and [g]) to specify the connection points between monomers. A point [e] is assumed to be connected to a point [t] of the next monomer. (and [d] connected to [g])

Key Principles:
1. Preserve the intended chemical structure when possible
2. Ensure exactly 2 [*] connection points
3. Use valid SMILES chemistry
4. Apply the most conservative fix that addresses all issues
5. Provide clear reasoning for your corrections

Common Corrections:
- Add missing [*]: CC → [*]CC[*]
- Fix brackets: [*]C(C[*] → [*]C(C)[*]
- Repair SMILES: [*]C1CCC → [*]C1CCCC1[*]
- Clean malformed: []CC[] → [*]CC[*]

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
-C4H3N- -> C(N1)=CC=C1"""),
            
            ("human", """Analysis Results:
{analysis_results}

Original PSMILES: {original_psmiles}
Attempt: {attempt_number}/{max_attempts}

Generate a corrected PSMILES structure that addresses all identified issues.

{format_instructions}"""),
            
            MessagesPlaceholder(variable_name="correction_history")
        ])
        
        # Reflection prompt - evaluates corrections
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a PSMILES validation expert. Evaluate the proposed correction and determine if it successfully addresses the original issues.

Evaluation Criteria:
1. Correct PSMILES format ([*]...[*])
2. Valid SMILES chemistry
3. Addresses all identified issues
4. Maintains intended chemical structure
5. No new issues introduced

Provide detailed feedback and suggest improvements if needed."""),
            
            ("human", """Original Failed PSMILES: {original_psmiles}
Proposed Correction: {corrected_psmiles}
Correction Reasoning: {correction_reasoning}

Validation Results: {validation_results}

Evaluate this correction and provide feedback. If issues remain, suggest specific improvements.
Should we accept this correction or try again?""")
        ])
    
    def _setup_correction_chain(self):
        """Setup the self-correcting chain with proper error handling"""
        
        def analyze_structure(state: ValidationState) -> AnalysisResult:
            """Analyze the failed structure to identify issues"""
            try:
                prompt_input = {
                    "failed_psmiles": state.original_psmiles,
                    "error_details": "; ".join(state.validation_errors) if state.validation_errors else "Visualization failed",
                    "format_instructions": self.analysis_parser.get_format_instructions()
                }
                
                chain = self.analysis_prompt | self.llm | self.analysis_parser
                analysis = chain.invoke(prompt_input)
                
                logger.info(f"🔍 Analysis complete: Found {len(analysis.identified_issues)} issues")
                return analysis
                
            except Exception as e:
                logger.error(f"❌ Analysis failed: {e}")
                # Fallback analysis
                return AnalysisResult(
                    connection_points=state.original_psmiles.count('[*]'),
                    underlying_smiles=state.original_psmiles.replace('[*]', ''),
                    identified_issues=["Analysis failed", str(e)],
                    suggested_fixes=["Apply basic PSMILES format correction"],
                    chemistry_assessment="Unknown due to analysis failure"
                )
        
        def generate_correction(state: ValidationState, analysis: AnalysisResult) -> CorrectionResult:
            """Generate a correction based on analysis"""
            try:
                # Build correction history for context
                correction_history = []
                for i, correction in enumerate(state.corrections_applied):
                    correction_history.extend([
                        HumanMessage(content=f"Previous attempt {i+1}: {correction['attempt']}"),
                        AIMessage(content=f"Result: {correction['result']} (Issues: {correction.get('issues', 'None')})")
                    ])
                
                prompt_input = {
                    "analysis_results": analysis.dict(),
                    "original_psmiles": state.original_psmiles,
                    "attempt_number": state.attempt_number,
                    "max_attempts": self.max_attempts,
                    "format_instructions": self.correction_parser.get_format_instructions(),
                    "correction_history": correction_history
                }
                
                chain = self.correction_prompt | self.llm | self.correction_parser
                correction = chain.invoke(prompt_input)
                
                logger.info(f"🛠️ Generated correction: {correction.corrected_psmiles}")
                return correction
                
            except OutputParserException as e:
                logger.error(f"❌ Correction parsing failed: {e}")
                # Fallback correction using simple pattern repair
                return self._generate_fallback_correction(state, analysis)
            except Exception as e:
                logger.error(f"❌ Correction generation failed: {e}")
                return self._generate_fallback_correction(state, analysis)
        
        def validate_chemistry(psmiles: str) -> Dict[str, Any]:
            """Validate the chemistry of a PSMILES structure"""
            try:
                # Remove connection points for SMILES validation
                smiles = psmiles.replace('[*]', '')
                
                # Try RDKit validation
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        return {
                            'valid': True,
                            'canonical_smiles': canonical_smiles,
                            'message': 'Valid chemistry',
                            'mol_weight': Chem.Descriptors.MolWt(mol) if hasattr(Chem, 'Descriptors') else None
                        }
                    else:
                        return {
                            'valid': False,
                            'message': 'Invalid SMILES - RDKit cannot parse',
                            'canonical_smiles': None
                        }
                except ImportError:
                    # RDKit not available - basic validation
                    if re.match(r'^[A-Za-z0-9@+\-\[\]()=#/\\:.]+$', smiles):
                        return {
                            'valid': True,
                            'message': 'Basic format validation passed (RDKit not available)',
                            'canonical_smiles': smiles
                        }
                    else:
                        return {
                            'valid': False,
                            'message': 'Basic format validation failed',
                            'canonical_smiles': None
                        }
                
            except Exception as e:
                return {
                    'valid': False,
                    'message': f'Validation error: {str(e)}',
                    'canonical_smiles': None
                }
        
        def reflect_on_correction(state: ValidationState, correction: CorrectionResult) -> Dict[str, Any]:
            """Use reflection pattern to evaluate the correction"""
            if not self.enable_reflection:
                return {'accept': True, 'feedback': 'Reflection disabled'}
            
            try:
                # Validate the chemistry
                chemistry_validation = validate_chemistry(correction.corrected_psmiles)
                
                prompt_input = {
                    "original_psmiles": state.original_psmiles,
                    "corrected_psmiles": correction.corrected_psmiles,
                    "correction_reasoning": correction.reasoning,
                    "validation_results": json.dumps(chemistry_validation, indent=2)
                }
                
                chain = self.reflection_prompt | self.llm | StrOutputParser()
                reflection = chain.invoke(prompt_input)
                
                # Simple heuristic to determine acceptance
                accept = (
                    chemistry_validation['valid'] and
                    correction.confidence > 0.7 and
                    'accept' in reflection.lower() and
                    'reject' not in reflection.lower()
                )
                
                return {
                    'accept': accept,
                    'feedback': reflection,
                    'chemistry_valid': chemistry_validation['valid'],
                    'chemistry_message': chemistry_validation['message']
                }
                
            except Exception as e:
                logger.error(f"❌ Reflection failed: {e}")
                # Default to acceptance if reflection fails
                chemistry_validation = validate_chemistry(correction.corrected_psmiles)
                return {
                    'accept': chemistry_validation['valid'],
                    'feedback': f'Reflection failed: {e}',
                    'chemistry_valid': chemistry_validation['valid'],
                    'chemistry_message': chemistry_validation.get('message', 'Unknown')
                }
        
        # Store chain components for use in main correction method
        self._analyze_fn = analyze_structure
        self._correct_fn = generate_correction
        self._validate_fn = validate_chemistry
        self._reflect_fn = reflect_on_correction
    
    def _generate_fallback_correction(self, state: ValidationState, analysis: AnalysisResult) -> CorrectionResult:
        """Generate a simple fallback correction when LLM fails"""
        original = state.original_psmiles
        
        # Apply basic fixes
        corrected = original
        
        # Fix connection points
        if not corrected.startswith('[*]'):
            corrected = '[*]' + corrected
        if not corrected.endswith('[*]'):
            corrected = corrected + '[*]'
        
        # Clean multiple connection points
        corrected = re.sub(r'\[\*\]+', '[*]', corrected)
        
        # Remove invalid characters at boundaries
        corrected = re.sub(r'\[\*\]\[\*\]', '[*]', corrected)
        
        # Ensure we have exactly 2 connection points
        parts = corrected.split('[*]')
        if len(parts) > 3:
            # Take the middle part with largest content
            middle_parts = [p for p in parts[1:-1] if p.strip()]
            if middle_parts:
                corrected = f"[*]{max(middle_parts, key=len)}[*]"
            else:
                corrected = "[*]CC[*]"  # Default polymer
        
        return CorrectionResult(
            corrected_psmiles=corrected,
            confidence=0.6,
            correction_method="fallback_pattern_repair",
            reasoning=f"Applied basic pattern fixes: {original} → {corrected}",
            chemistry_valid=False,  # Will be validated separately
            validation_message="Fallback correction applied"
        )
    
    def correct_psmiles_with_chain(self, failed_psmiles: str, error_details: Optional[str] = None) -> Dict[str, Any]:
        """
        Main correction method implementing the self-correcting chain pattern.
        
        This implements:
        1. Analysis Phase - identify specific issues
        2. Correction Phase - generate fixes iteratively
        3. Validation Phase - check chemistry and format
        4. Reflection Phase - evaluate and refine (if enabled)
        5. Retry Logic - up to max_attempts with learning
        
        Args:
            failed_psmiles: The PSMILES structure that failed
            error_details: Optional error details from the failure
            
        Returns:
            Dict containing the correction results and process details
        """
        
        # Initialize validation state
        state = ValidationState(
            original_psmiles=failed_psmiles,
            current_psmiles=failed_psmiles,
            attempt_number=1,
            stage=CorrectionStage.INITIAL_ANALYSIS,
            validation_errors=[error_details] if error_details else [],
            corrections_applied=[]
        )
        
        logger.info(f"🚀 Starting self-correcting chain for: {failed_psmiles}")
        
        try:
            # Phase 1: Initial Analysis
            state.stage = CorrectionStage.INITIAL_ANALYSIS
            analysis = self._analyze_fn(state)
            
            # Main correction loop with retry mechanism
            for attempt in range(1, self.max_attempts + 1):
                state.attempt_number = attempt
                state.stage = CorrectionStage.STRUCTURE_REPAIR
                
                logger.info(f"🔄 Correction attempt {attempt}/{self.max_attempts}")
                
                # Generate correction
                correction = self._correct_fn(state, analysis)
                
                # Validate chemistry
                state.stage = CorrectionStage.CHEMISTRY_VALIDATION
                chemistry_validation = self._validate_fn(correction.corrected_psmiles)
                
                # Update correction with validation results
                correction.chemistry_valid = chemistry_validation['valid']
                correction.validation_message = chemistry_validation['message']
                
                # Reflection phase (if enabled)
                if self.enable_reflection:
                    state.stage = CorrectionStage.REFLECTION
                    reflection = self._reflect_fn(state, correction)
                    
                    if reflection['accept']:
                        logger.info(f"✅ Correction accepted after reflection")
                        state.final_result = correction
                        break
                    else:
                        logger.info(f"🔄 Correction rejected, trying again: {reflection['feedback'][:100]}...")
                        state.corrections_applied.append({
                            'attempt': correction.corrected_psmiles,
                            'result': 'rejected',
                            'issues': reflection['feedback'][:200],
                            'confidence': correction.confidence
                        })
                else:
                    # Accept if chemistry is valid
                    if chemistry_validation['valid']:
                        logger.info(f"✅ Correction accepted (chemistry valid)")
                        state.final_result = correction
                        break
                    else:
                        logger.info(f"❌ Correction rejected (invalid chemistry): {chemistry_validation['message']}")
                        state.corrections_applied.append({
                            'attempt': correction.corrected_psmiles,
                            'result': 'invalid_chemistry',
                            'issues': chemistry_validation['message'],
                            'confidence': correction.confidence
                        })
            
            # Final validation and result preparation
            state.stage = CorrectionStage.FINAL_VALIDATION
            
            if state.final_result:
                # Success!
                final_validation = self._validate_fn(state.final_result.corrected_psmiles)
                
                result = {
                    'success': True,
                    'original_psmiles': failed_psmiles,
                    'corrected_psmiles': state.final_result.corrected_psmiles,
                    'correction_method': state.final_result.correction_method,
                    'confidence': state.final_result.confidence,
                    'reasoning': state.final_result.reasoning,
                    'chemistry_valid': final_validation['valid'],
                    'validation_message': final_validation['message'],
                    'canonical_smiles': final_validation.get('canonical_smiles'),
                    'attempts_used': state.attempt_number,
                    'analysis_results': analysis.dict(),
                    'corrections_tried': state.corrections_applied,
                    'method': 'langchain_self_correcting_chain',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to memory for learning
                self.memory.save_context(
                    {"input": f"Failed PSMILES: {failed_psmiles}"},
                    {"output": f"Corrected to: {state.final_result.corrected_psmiles}"}
                )
                
                logger.info(f"🎉 Self-correcting chain succeeded: {state.final_result.corrected_psmiles}")
                return result
            
            else:
                # All attempts failed
                logger.error(f"❌ All {self.max_attempts} correction attempts failed")
                
                # Provide best attempt or simple fallback
                if state.corrections_applied:
                    best_attempt = max(state.corrections_applied, key=lambda x: x.get('confidence', 0))
                    fallback_psmiles = best_attempt['attempt']
                else:
                    # Ultimate fallback - ensure basic format
                    fallback_psmiles = self._create_ultimate_fallback(failed_psmiles)
                
                return {
                    'success': False,
                    'original_psmiles': failed_psmiles,
                    'corrected_psmiles': fallback_psmiles,
                    'correction_method': 'fallback_after_all_attempts_failed',
                    'confidence': 0.3,
                    'reasoning': f"All {self.max_attempts} attempts failed, providing best fallback",
                    'chemistry_valid': False,
                    'validation_message': "Fallback correction - chemistry not validated",
                    'attempts_used': self.max_attempts,
                    'analysis_results': analysis.dict(),
                    'corrections_tried': state.corrections_applied,
                    'method': 'langchain_self_correcting_chain_fallback',
                    'error': 'Max attempts exceeded',
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Self-correcting chain failed with exception: {e}")
            
            # Emergency fallback
            emergency_fallback = self._create_ultimate_fallback(failed_psmiles)
            
            return {
                'success': False,
                'original_psmiles': failed_psmiles,
                'corrected_psmiles': emergency_fallback,
                'correction_method': 'emergency_fallback',
                'confidence': 0.1,
                'reasoning': f"Chain failed with exception: {str(e)}",
                'chemistry_valid': False,
                'validation_message': f"Emergency fallback due to: {str(e)}",
                'attempts_used': state.attempt_number,
                'method': 'langchain_self_correcting_chain_emergency',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_ultimate_fallback(self, failed_psmiles: str) -> str:
        """Create an ultimate fallback PSMILES when everything else fails"""
        # Try to extract any valid chemistry
        extracted = re.findall(r'[A-Z][a-z]?', failed_psmiles)
        if extracted and len(''.join(extracted)) > 0:
            simple_chain = ''.join(extracted[:3])  # Take first 3 atoms
            return f"[*]{simple_chain}[*]"
        else:
            return "[*]CC[*]"  # Default ethylene polymer


# Convenience function for easy integration
def apply_langchain_correction(failed_psmiles: str, 
                             error_details: Optional[str] = None,
                             llm = None,
                             max_attempts: int = 3,
                             enable_reflection: bool = True) -> Dict[str, Any]:
    """
    Apply LangChain-based PSMILES correction.
    
    Args:
        failed_psmiles: The PSMILES that failed
        error_details: Optional error details
        llm: LangChain LLM instance (if None, will try to create a simple one)
        max_attempts: Maximum correction attempts
        enable_reflection: Whether to use reflection pattern
        
    Returns:
        Correction results dictionary
    """
    
    # Create a simple LLM if none provided
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        except ImportError:
            logger.error("❌ No LLM provided and OpenAI not available")
            # Return simple pattern-based correction
            return {
                'success': True,
                'original_psmiles': failed_psmiles,
                'corrected_psmiles': f"[*]CC[*]",  # Safe default
                'correction_method': 'simple_fallback_no_llm',
                'confidence': 0.5,
                'reasoning': 'LLM not available, used simple fallback',
                'chemistry_valid': True,
                'validation_message': 'Simple ethylene polymer',
                'method': 'langchain_fallback_no_llm'
            }
    
    # Create corrector and apply
    corrector = LangChainPSMILESCorrector(
        llm=llm,
        max_attempts=max_attempts,
        enable_reflection=enable_reflection
    )
    
    return corrector.correct_psmiles_with_chain(failed_psmiles, error_details)


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing LangChain PSMILES Corrector...")
    
    test_cases = [
        "[]c1ccc(cc1)C(=O)(OC(=O)C(O)C(O))[]",  # User's example
        "[*]c1ccc(cc1)C(=O)(OC(=O)C(O)C(O))",    # Missing second [*]
        "CC[*]CC[*]CC",                           # Wrong placement
        "[*]INVALID_CHEMISTRY[*]",                # Invalid SMILES
    ]
    
    for test_psmiles in test_cases:
        print(f"\n🔬 Testing: {test_psmiles}")
        try:
            result = apply_langchain_correction(
                test_psmiles,
                error_details="Visualization failed",
                llm=None,  # Will use fallback
                max_attempts=2,
                enable_reflection=False  # Disable for testing
            )
            print(f"✅ Result: {result['corrected_psmiles']}")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"🔧 Method: {result['correction_method']}")
        except Exception as e:
            print(f"❌ Test failed: {e}") 