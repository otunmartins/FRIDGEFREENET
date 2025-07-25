#!/usr/bin/env python3
"""
PSMILES Auto-Corrector using LangChain

A specialized LangChain agent that analyzes failed PSMILES structures,
identifies common failure patterns, and automatically generates corrected versions.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PSMILESPatternAnalyzer:
    """Analyzes PSMILES structures to identify common failure patterns."""
    
    def __init__(self):
        self.failure_patterns = {
            # Valency issues with heteroatoms directly connected to [*]
            'nitrogen_direct': {
                'pattern': r'N\[\*\]',
                'severity': 'high',
                'description': 'Nitrogen directly connected to connection point',
                'fixes': ['Move [*] to adjacent carbon', 'Add carbon buffer: NC -> [*]CNC']
            },
            'oxygen_direct': {
                'pattern': r'O\[\*\]',
                'severity': 'high', 
                'description': 'Oxygen directly connected to connection point',
                'fixes': ['Move [*] to adjacent carbon', 'Add carbon buffer: OC -> [*]COC']
            },
            'sulfur_direct': {
                'pattern': r'S\[\*\]',
                'severity': 'high',
                'description': 'Sulfur directly connected to connection point',
                'fixes': ['Move [*] to adjacent carbon', 'Add carbon buffer: SC -> [*]CSC']
            },
            
            # Terminal atom issues
            'terminal_sulfur': {
                'pattern': r'\[\*\].*S$',
                'severity': 'medium',
                'description': 'Structure ends with sulfur atom',
                'fixes': ['Add explicit hydrogen: S -> SH', 'Rearrange to avoid terminal S']
            },
            'terminal_oxygen': {
                'pattern': r'\[\*\].*O$',
                'severity': 'medium', 
                'description': 'Structure ends with oxygen atom',
                'fixes': ['Add explicit hydrogen: O -> OH', 'Rearrange to avoid terminal O']
            },
            'terminal_nitrogen': {
                'pattern': r'\[\*\].*N$',
                'severity': 'medium',
                'description': 'Structure ends with nitrogen atom', 
                'fixes': ['Add explicit hydrogen: N -> NH2', 'Rearrange to avoid terminal N']
            },
            
            # Syntax issues
            'unbalanced_parens': {
                'pattern': None,  # Custom check
                'severity': 'critical',
                'description': 'Unbalanced parentheses in SMILES',
                'fixes': ['Check and balance parentheses', 'Verify SMILES syntax']
            },
            
            # Double bond/aromatic issues
            'aromatic_connection': {
                'pattern': r'c\[\*\]|n\[\*\]|o\[\*\]|s\[\*\]',
                'severity': 'medium',
                'description': 'Aromatic atom directly connected to connection point',
                'fixes': ['Use uppercase atoms for aliphatic', 'Add carbon linker']
            }
        }
    
    def analyze_structure(self, psmiles: str) -> Dict[str, Any]:
        """Analyze a PSMILES structure for common failure patterns."""
        issues = []
        severity_score = 0
        
        # Remove [*] for underlying SMILES analysis
        underlying_smiles = psmiles.replace('[*]', '')
        
        # Check each pattern
        for pattern_name, pattern_info in self.failure_patterns.items():
            if pattern_name == 'unbalanced_parens':
                # Custom check for parentheses
                if underlying_smiles.count('(') != underlying_smiles.count(')'):
                    issues.append({
                        'pattern': pattern_name,
                        'severity': pattern_info['severity'],
                        'description': pattern_info['description'],
                        'fixes': pattern_info['fixes'],
                        'matches': ['Parentheses mismatch']
                    })
                    severity_score += 10  # Critical
            else:
                # Regex pattern check
                matches = re.findall(pattern_info['pattern'], psmiles)
                if matches:
                    severity_map = {'critical': 10, 'high': 7, 'medium': 4, 'low': 1}
                    severity_score += severity_map.get(pattern_info['severity'], 1) * len(matches)
                    
                    issues.append({
                        'pattern': pattern_name,
                        'severity': pattern_info['severity'],
                        'description': pattern_info['description'],
                        'fixes': pattern_info['fixes'],
                        'matches': matches
                    })
        
        return {
            'psmiles': psmiles,
            'underlying_smiles': underlying_smiles,
            'issues': issues,
            'severity_score': severity_score,
            'is_problematic': severity_score > 0,
            'analysis_timestamp': datetime.now().isoformat()
        }


class PSMILESStructuralCorrector:
    """Applies structural corrections to PSMILES based on identified patterns."""
    
    def __init__(self):
        self.correction_strategies = {
            'nitrogen_direct': self._fix_nitrogen_direct,
            'oxygen_direct': self._fix_oxygen_direct,
            'sulfur_direct': self._fix_sulfur_direct,
            'terminal_sulfur': self._fix_terminal_sulfur,
            'terminal_oxygen': self._fix_terminal_oxygen,
            'terminal_nitrogen': self._fix_terminal_nitrogen,
            'aromatic_connection': self._fix_aromatic_connection
        }
    
    def _fix_nitrogen_direct(self, psmiles: str) -> List[str]:
        """Fix nitrogen directly connected to [*]."""
        corrections = []
        
        # Strategy 1: Move [*] to adjacent carbon if possible
        if 'CN[*]' in psmiles:
            corrections.append(psmiles.replace('CN[*]', '[*]CN'))
        if 'N[*]C' in psmiles:
            corrections.append(psmiles.replace('N[*]C', 'NC[*]'))
            
        # Strategy 2: Add carbon buffer
        corrections.append(psmiles.replace('N[*]', 'NC[*]'))
        
        # Strategy 3: Rearrange to put nitrogen in middle
        if psmiles.count('[*]') == 2:
            parts = psmiles.split('[*]')
            if len(parts) == 3:  # [*]...N[*] pattern
                middle = parts[1]
                if 'N' in middle:
                    # Try to rearrange with N in center
                    corrections.append(f"[*]C{middle}C[*]")
        
        return list(set(corrections))  # Remove duplicates
    
    def _fix_oxygen_direct(self, psmiles: str) -> List[str]:
        """Fix oxygen directly connected to [*]."""
        corrections = []
        
        # Strategy 1: Move [*] to adjacent carbon
        if 'CO[*]' in psmiles:
            corrections.append(psmiles.replace('CO[*]', '[*]CO'))
        if 'O[*]C' in psmiles:
            corrections.append(psmiles.replace('O[*]C', 'OC[*]'))
            
        # Strategy 2: Add carbon buffer  
        corrections.append(psmiles.replace('O[*]', 'OC[*]'))
        
        return list(set(corrections))
    
    def _fix_sulfur_direct(self, psmiles: str) -> List[str]:
        """Fix sulfur directly connected to [*]."""
        corrections = []
        
        # Strategy 1: Move [*] to adjacent carbon
        if 'CS[*]' in psmiles:
            corrections.append(psmiles.replace('CS[*]', '[*]CS'))
        if 'S[*]C' in psmiles:
            corrections.append(psmiles.replace('S[*]C', 'SC[*]'))
            
        # Strategy 2: Add carbon buffer
        corrections.append(psmiles.replace('S[*]', 'SC[*]'))
        
        return list(set(corrections))
    
    def _fix_terminal_sulfur(self, psmiles: str) -> List[str]:
        """Fix terminal sulfur atoms."""
        corrections = []
        
        # Add explicit hydrogen
        if psmiles.endswith('S'):
            base = psmiles[:-1]
            corrections.extend([base + 'SH', base + 'S(H)'])
            
        # Rearrange if there's a clear alternative
        if '[*]' in psmiles and psmiles.endswith('S'):
            # Try to move S away from terminal position
            parts = psmiles.split('S')
            if len(parts) == 2:
                corrections.append(f"[*]S{parts[0]}{parts[1][:-3]}[*]")  # Move S to beginning
                
        return list(set(corrections))
    
    def _fix_terminal_oxygen(self, psmiles: str) -> List[str]:
        """Fix terminal oxygen atoms."""
        corrections = []
        
        if psmiles.endswith('O'):
            base = psmiles[:-1]
            corrections.extend([base + 'OH', base + 'O(H)'])
            
        return list(set(corrections))
    
    def _fix_terminal_nitrogen(self, psmiles: str) -> List[str]:
        """Fix terminal nitrogen atoms."""
        corrections = []
        
        if psmiles.endswith('N'):
            base = psmiles[:-1]
            corrections.extend([base + 'NH2', base + 'N(H)H'])
            
        return list(set(corrections))
    
    def _fix_aromatic_connection(self, psmiles: str) -> List[str]:
        """Fix aromatic atoms directly connected to [*]."""
        corrections = []
        
        # Convert aromatic to aliphatic
        aromatic_map = {'c': 'C', 'n': 'N', 'o': 'O', 's': 'S'}
        
        for aromatic, aliphatic in aromatic_map.items():
            if f"{aromatic}[*]" in psmiles:
                corrections.append(psmiles.replace(f"{aromatic}[*]", f"{aliphatic}[*]"))
                # Also try with carbon buffer
                corrections.append(psmiles.replace(f"{aromatic}[*]", f"{aromatic}C[*]"))
        
        return list(set(corrections))
    
    def generate_corrections(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate corrected PSMILES structures based on analysis."""
        psmiles = analysis['psmiles']
        issues = analysis['issues']
        
        all_corrections = []
        
        # Apply corrections for each identified issue
        for issue in issues:
            pattern_name = issue['pattern']
            if pattern_name in self.correction_strategies:
                corrections = self.correction_strategies[pattern_name](psmiles)
                
                for correction in corrections:
                    all_corrections.append({
                        'original': psmiles,
                        'corrected': correction,
                        'fix_applied': pattern_name,
                        'fix_description': issue['description'],
                        'confidence': self._calculate_confidence(psmiles, correction),
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Sort by confidence score
        all_corrections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates and invalid corrections
        seen = set()
        valid_corrections = []
        
        for correction in all_corrections:
            corrected_psmiles = correction['corrected']
            
            # Skip duplicates
            if corrected_psmiles in seen:
                continue
                
            # Skip if same as original
            if corrected_psmiles == psmiles:
                continue
                
            # Basic validation
            if corrected_psmiles.count('[*]') == 2:
                seen.add(corrected_psmiles)
                valid_corrections.append(correction)
        
        return valid_corrections[:5]  # Return top 5 corrections
    
    def _calculate_confidence(self, original: str, corrected: str) -> float:
        """Calculate confidence score for a correction."""
        confidence = 0.5  # Base confidence
        
        # Bonus for maintaining structure length similarity
        length_ratio = min(len(corrected), len(original)) / max(len(corrected), len(original))
        confidence += 0.2 * length_ratio
        
        # Bonus for maintaining chemical elements
        original_elements = set(re.findall(r'[A-Z][a-z]?', original))
        corrected_elements = set(re.findall(r'[A-Z][a-z]?', corrected))
        element_overlap = len(original_elements & corrected_elements) / len(original_elements | corrected_elements)
        confidence += 0.2 * element_overlap
        
        # Penalty for adding too many atoms
        original_atom_count = len(re.findall(r'[A-Z][a-z]?', original))
        corrected_atom_count = len(re.findall(r'[A-Z][a-z]?', corrected))
        if corrected_atom_count > original_atom_count * 1.5:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))


class PSMILESAutoCorrector:
    """
    LangChain-based PSMILES Auto-Corrector that uses pattern analysis 
    and LLM reasoning to fix failed PSMILES structures.
    """
    
    def __init__(self, 
                 ollama_model: str = "llama3.2",
                 ollama_host: str = "http://localhost:11434"):
        """Initialize the auto-corrector with LangChain components."""
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_host,
            temperature=0.3  # Lower temperature for consistent corrections
        )
        
        # Initialize analyzers
        self.pattern_analyzer = PSMILESPatternAnalyzer()
        self.structural_corrector = PSMILESStructuralCorrector()
        
        # Initialize memory for learning from corrections
        self.memory = ConversationBufferWindowMemory(
            k=20,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Setup LangChain tools
        self.tools = self._setup_tools()
        
        # Setup the main correction prompt
        self.correction_prompt = self._setup_correction_prompt()
        
        logger.info(f"✅ PSMILES Auto-Corrector initialized with {ollama_model}")
    
    def _setup_tools(self) -> List[Tool]:
        """Setup LangChain tools for PSMILES analysis and correction."""
        
        def analyze_psmiles_tool(psmiles: str) -> str:
            """Analyze PSMILES structure for failure patterns."""
            analysis = self.pattern_analyzer.analyze_structure(psmiles)
            return json.dumps(analysis, indent=2)
        
        def generate_corrections_tool(analysis_json: str) -> str:
            """Generate structural corrections based on analysis."""
            try:
                analysis = json.loads(analysis_json)
                corrections = self.structural_corrector.generate_corrections(analysis)
                return json.dumps(corrections, indent=2)
            except json.JSONDecodeError:
                return "Error: Invalid analysis format"
        
        def validate_psmiles_tool(psmiles: str) -> str:
            """Validate PSMILES structure format."""
            star_count = psmiles.count('[*]')
            underlying_smiles = psmiles.replace('[*]', '')
            
            validation = {
                'psmiles': psmiles,
                'valid_format': star_count == 2,
                'connection_points': star_count,
                'underlying_smiles': underlying_smiles,
                'has_atoms': len(re.findall(r'[A-Z][a-z]?', underlying_smiles)) > 0
            }
            return json.dumps(validation, indent=2)
        
        return [
            Tool(
                name="analyze_psmiles",
                description="Analyze a PSMILES structure to identify failure patterns and issues",
                func=analyze_psmiles_tool
            ),
            Tool(
                name="generate_corrections", 
                description="Generate corrected PSMILES structures based on pattern analysis",
                func=generate_corrections_tool
            ),
            Tool(
                name="validate_psmiles",
                description="Validate PSMILES format and basic structure",
                func=validate_psmiles_tool
            )
        ]
    
    def _setup_correction_prompt(self) -> ChatPromptTemplate:
        """Setup the main correction prompt template."""
        
        system_message = """You are an expert PSMILES (Polymer SMILES) structure corrector. 
Your job is to analyze failed PSMILES structures and generate chemically valid corrections.

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

PSMILES Rules:
1. Must have exactly 2 connection points: [*]
2. Connection points should typically be on carbon atoms, not heteroatoms (N, O, S)
3. Avoid terminal heteroatoms without explicit hydrogens
4. Maintain chemical valency rules
5. Use proper SMILES syntax

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

Common Failure Patterns:
- N[*], O[*], S[*] (heteroatoms directly connected to connection points)
- Terminal S, O, N without explicit hydrogens
- Unbalanced parentheses
- Aromatic atoms (lowercase) directly connected to [*]

Your process:
1. Analyze the failed PSMILES using analyze_psmiles tool
2. Generate corrections using generate_corrections tool  
3. Validate each correction using validate_psmiles tool
4. Provide the best corrections with explanations

Be systematic and provide clear reasoning for each correction."""

        human_template = """Failed PSMILES Structure: {psmiles}

Error Details: {error_details}

Please analyze this structure and provide corrected versions that should work for visualization.
Focus on the most likely fixes based on the error patterns."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])
    
    def correct_psmiles(self, 
                       failed_psmiles: str, 
                       error_details: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze and correct a failed PSMILES structure.
        
        Args:
            failed_psmiles: The PSMILES that failed visualization
            error_details: Optional error details from the failure
            
        Returns:
            Dict containing analysis, corrections, and recommendations
        """
        
        try:
            # Step 1: Pattern analysis
            logger.info(f"🔍 Analyzing failed PSMILES: {failed_psmiles}")
            analysis = self.pattern_analyzer.analyze_structure(failed_psmiles)
            
            # Step 2: Generate structural corrections
            logger.info("🛠️ Generating structural corrections...")
            corrections = self.structural_corrector.generate_corrections(analysis)
            
            # Step 3: LLM-based reasoning for additional corrections
            logger.info("🧠 Applying LLM reasoning...")
            llm_corrections = self._get_llm_corrections(failed_psmiles, error_details, analysis)
            
            # Step 4: Combine and rank all corrections
            all_corrections = corrections + llm_corrections
            
            # Remove duplicates and rank by confidence
            seen = set()
            final_corrections = []
            for correction in all_corrections:
                corrected_psmiles = correction['corrected']
                if corrected_psmiles not in seen and corrected_psmiles != failed_psmiles:
                    seen.add(corrected_psmiles)
                    final_corrections.append(correction)
            
            # Sort by confidence
            final_corrections.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
            
            # Store in memory for learning
            self.memory.save_context(
                {"input": f"Failed PSMILES: {failed_psmiles}"},
                {"output": f"Generated {len(final_corrections)} corrections"}
            )
            
            result = {
                'original_psmiles': failed_psmiles,
                'analysis': analysis,
                'corrections': final_corrections[:5],  # Top 5
                'correction_count': len(final_corrections),
                'success': len(final_corrections) > 0,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Generated {len(final_corrections)} corrections")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in correction process: {e}")
            return {
                'original_psmiles': failed_psmiles,
                'analysis': None,
                'corrections': [],
                'correction_count': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_llm_corrections(self, 
                           failed_psmiles: str, 
                           error_details: Optional[str],
                           analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to generate additional correction suggestions."""
        
        try:
            # Format the prompt
            prompt_input = {
                'psmiles': failed_psmiles,
                'error_details': error_details or "Visualization failed"
            }
            
            # Get LLM response
            formatted_prompt = self.correction_prompt.format(**prompt_input)
            response = self.llm.invoke(formatted_prompt)
            
            # Parse LLM suggestions (this would need more sophisticated parsing in practice)
            llm_corrections = self._parse_llm_response(response, failed_psmiles)
            
            return llm_corrections
            
        except Exception as e:
            logger.warning(f"⚠️ LLM correction failed: {e}")
            return []
    
    def _parse_llm_response(self, response: str, original_psmiles: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract correction suggestions."""
        
        corrections = []
        
        # Look for PSMILES patterns in the response
        psmiles_pattern = r'\[?\*\]?[A-Za-z0-9\(\)\[\]\*\=\#\-\+\.]*\[?\*\]?'
        potential_psmiles = re.findall(psmiles_pattern, response)
        
        for psmiles in potential_psmiles:
            # Clean up the match
            cleaned = psmiles.strip()
            
            # Must have exactly 2 [*] and be different from original
            if cleaned.count('[*]') == 2 and cleaned != original_psmiles and len(cleaned) > 5:
                corrections.append({
                    'original': original_psmiles,
                    'corrected': cleaned,
                    'fix_applied': 'llm_suggestion',
                    'fix_description': 'LLM-generated correction',
                    'confidence': 0.6,  # Medium confidence for LLM suggestions
                    'timestamp': datetime.now().isoformat()
                })
        
        return corrections[:3]  # Return top 3 LLM suggestions
    
    def batch_correct(self, failed_psmiles_list: List[str]) -> Dict[str, Any]:
        """Correct multiple failed PSMILES structures in batch."""
        
        results = {}
        
        for psmiles in failed_psmiles_list:
            logger.info(f"🔄 Processing batch correction for: {psmiles}")
            results[psmiles] = self.correct_psmiles(psmiles)
        
        # Generate batch summary
        total_corrections = sum(len(result['corrections']) for result in results.values())
        success_rate = sum(1 for result in results.values() if result['success']) / len(results)
        
        return {
            'individual_results': results,
            'batch_summary': {
                'total_structures': len(failed_psmiles_list),
                'total_corrections': total_corrections,
                'success_rate': success_rate,
                'timestamp': datetime.now().isoformat()
            }
        }


# Convenience function for integration
def create_psmiles_auto_corrector(ollama_model: str = "llama3.2", 
                                 ollama_host: str = "http://localhost:11434") -> PSMILESAutoCorrector:
    """Create and return a PSMILES auto-corrector instance."""
    return PSMILESAutoCorrector(ollama_model=ollama_model, ollama_host=ollama_host)


if __name__ == "__main__":
    # Test the system with the provided examples
    corrector = create_psmiles_auto_corrector()
    
    test_structures = [
        "C(O)C(=O)N[*]CC(=O)[*]",  # Structure 4
        "C(O)CO[*]NC(=O)COC(=O)C[*]S"  # Structure 5
    ]
    
    for psmiles in test_structures:
        print(f"\n🔧 Testing correction for: {psmiles}")
        result = corrector.correct_psmiles(psmiles)
        
        if result['success']:
            print(f"✅ Generated {result['correction_count']} corrections:")
            for i, correction in enumerate(result['corrections'][:3], 1):
                print(f"  {i}. {correction['corrected']} (confidence: {correction['confidence']:.2f})")
        else:
            print("❌ No corrections generated") 