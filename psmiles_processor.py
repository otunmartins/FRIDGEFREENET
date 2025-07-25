#!/usr/bin/env python3
"""
Enhanced PSMILES Processor
Integrates the psmiles library functionality.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import base64

try:
    from psmiles import PolymerSmiles as PS
    PSMILES_AVAILABLE = True
except ImportError:
    PSMILES_AVAILABLE = False
    print("⚠️ psmiles library not available. Install with: pip install 'psmiles[polyBERT,mordred]@git+https://github.com/Ramprasad-Group/psmiles.git'")

class PSMILESProcessor:
    """
    Enhanced PSMILES processor that integrates the full psmiles library functionality.
    Provides canonicalization, visualization, dimerization, copolymerization, 
    fingerprint generation, and random functional group addition.
    """
    
    def __init__(self):
        """Initialize the PSMILES processor."""
        self.available = PSMILES_AVAILABLE
        self.session_psmiles = {}  # Store PSMILES for each session
        self.temp_dir = tempfile.gettempdir()
        print(f"🔬 PSMILESProcessor initialized - Available: {self.available}")
        
        # Define functional groups for random addition
        self.functional_groups = {
            'hydroxyl': {'pattern': '[*]C(O)[*]', 'description': 'Hydroxyl (-OH) group'},
            'carboxyl': {'pattern': '[*]C(=O)O[*]', 'description': 'Carboxyl (-COOH) group'},
            'amine': {'pattern': '[*]N[*]', 'description': 'Amine (-NH2) group'},
            'amide': {'pattern': '[*]C(=O)N[*]', 'description': 'Amide (-CONH2) group'},
            'ester': {'pattern': '[*]C(=O)OC[*]', 'description': 'Ester (-COO-) group'},
            'ether': {'pattern': '[*]COC[*]', 'description': 'Ether (-O-) linkage'},
            'aromatic': {'pattern': '[*]c1ccccc1[*]', 'description': 'Aromatic ring'},
            'methyl': {'pattern': '[*]CC[*]', 'description': 'Methyl (-CH3) group'},
            'carbonyl': {'pattern': '[*]C(=O)[*]', 'description': 'Carbonyl (C=O) group'},
            'sulfonic': {'pattern': '[*]S(=O)(=O)[*]', 'description': 'Sulfonic acid group'},
            'phosphate': {'pattern': '[*]P(=O)(O)(O)[*]', 'description': 'Phosphate group'},
            'vinyl': {'pattern': '[*]C=C[*]', 'description': 'Vinyl (C=C) group'}
        }
        
        if not self.available:
            print("❌ PSMILES library not available. Advanced features disabled.")
        else:
            print("✅ PSMILES Processor initialized with full functionality!")
            print(f"🧪 Available functional groups: {len(self.functional_groups)}")
            print("🎯 Features: canonicalization, visualization, dimerization, copolymerization, fingerprints, random FG addition")
    
    def _validate_psmiles_format(self, psmiles: str) -> bool:
        """
        Validate PSMILES format and return boolean result.
        
        Args:
            psmiles (str): PSMILES string to validate
            
        Returns:
            bool: True if valid, False if invalid
        """
        if not psmiles:
            return False
        
        # Count connection points
        connection_count = psmiles.count('[*]')
        if connection_count != 2:
            return False
        
        # Check for spaces and hyphens
        if ' ' in psmiles or '-' in psmiles:
            return False
        
        # Check for empty brackets (common error)
        if '[]' in psmiles:
            return False
        
        # Check for balanced brackets and parentheses
        if psmiles.count('(') != psmiles.count(')'):
            return False
        
        if psmiles.count('[') != psmiles.count(']'):
            return False
        
        return True
    
    def _fix_connection_points(self, psmiles: str) -> str:
        """
        Fix connection point formatting issues.
        
        Args:
            psmiles (str): PSMILES string that may have format issues
            
        Returns:
            str: Fixed PSMILES string with proper [*] connection points
        """
        if not psmiles:
            return '[*]C[*]'  # Emergency fallback
        
        # Clean corrupted brackets first - fix common LLM errors
        import re
        
        # Replace empty brackets [] with [*] - be very explicit
        psmiles = re.sub(r'\[\]', '[*]', psmiles)
        
        # Fix naked asterisks (but be careful not to double-fix [*])
        # Only replace naked * that aren't already in [*]
        psmiles = re.sub(r'(?<!\[)\*(?!\])', '[*]', psmiles)
        
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
    
    def process_psmiles_workflow(self, psmiles_string: str, session_id: str, step: str = "initial") -> Dict:
        """
        Process PSMILES through the complete workflow.
        
        Args:
            psmiles_string (str): The PSMILES string to process
            session_id (str): Session identifier for tracking
            step (str): Current step in workflow ('initial', 'dimer', 'addition', 'copolymer', 'other')
        
        Returns:
            Dict: Processing results with workflow options
        """
        if not self.available:
            return {
                'success': False,
                'error': 'PSMILES library not available',
                'suggestion': 'Install psmiles library to enable advanced features'
            }
        
        try:
            # Check for organometallic compounds first
            metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc'}
            contains_metal = any(f'[{metal}]' in psmiles_string or f'{metal}+' in psmiles_string for metal in metal_atoms)
            
            if contains_metal:
                return self._handle_organometallic_psmiles(psmiles_string, session_id, step)
            
            # **FIX CONNECTION POINTS** - Auto-correct common format issues
            original_psmiles = psmiles_string
            psmiles_string = self._fix_connection_points(psmiles_string)
            
            if original_psmiles != psmiles_string:
                print(f"🔧 Auto-corrected connection points: {original_psmiles} → {psmiles_string}")
            
            # Validate PSMILES format
            if not self._validate_psmiles_format(psmiles_string):
                return {
                    'success': False,
                    'error': f'Invalid PSMILES format: {psmiles_string}',
                    'suggestion': 'PSMILES must have exactly 2 [*] connection points'
                }
            
            # **PREPROCESS HYDROXYL GROUPS**: Fix common OH notation issues
            # Convert C(OH) to C(O) for proper SMILES format
            psmiles_string = psmiles_string.replace('(OH)', '(O)')
            
            # Extract SMILES for validation (remove [*] markers)
            smiles_for_validation = psmiles_string.replace('[*]', '')
            
            # Quick chemistry check to prevent RDKit parse errors
            try:
                from rdkit import Chem
                from natural_language_smiles import ChemicalValidator
                
                validator = ChemicalValidator()
                is_valid, mol, validation_msg = validator.validate_smiles(smiles_for_validation, debug=False)
                
                if not is_valid:
                    print(f"⚠️ PSMILES chemistry validation failed: {validation_msg}")
                    print(f"   Original PSMILES: {psmiles_string}")
                    
                    # Try to repair the chemistry
                    if "cleaned:" in validation_msg or "autocorrected:" in validation_msg or "fallback:" in validation_msg:
                        # Extract repaired SMILES
                        import re
                        match = re.search(r'→\s*([^\)]+)', validation_msg)
                        if match:
                            repaired_smiles = match.group(1)
                            repaired_psmiles = f"[*]{repaired_smiles}[*]"
                            print(f"   🔧 Auto-repaired to: {repaired_psmiles}")
                            psmiles_string = repaired_psmiles
                        else:
                            # Repair failed, return error
                            return {
                                'success': False,
                                'error': f'Invalid PSMILES chemistry: {validation_msg}',
                                'original_psmiles': psmiles_string,
                                'suggestion': 'Use the SMILES repair system or try a different structure'
                            }
                    else:
                        # No repair available, return error
                        return {
                            'success': False,
                            'error': f'Invalid PSMILES chemistry: {validation_msg}',
                            'original_psmiles': psmiles_string,
                            'suggestion': 'Check SMILES syntax and chemical validity'
                        }
                else:
                    print(f"✅ PSMILES chemistry validated: {psmiles_string}")
                    
            except ImportError:
                print("⚠️ Chemical validation not available - proceeding without validation")
            except Exception as validation_error:
                print(f"⚠️ Chemistry validation error: {validation_error}")
            
            # Create PSMILES object for standard organic polymers
            ps = PS(psmiles_string)
            
            # Store in session for future operations
            if session_id not in self.session_psmiles:
                self.session_psmiles[session_id] = []
            
            # Canonicalize the PSMILES
            canonical_ps = ps.canonicalize
            
            # Generate and save SVG
            svg_filename = f"psmiles_{session_id}_{len(self.session_psmiles[session_id])}.svg"
            svg_path = os.path.join(self.temp_dir, svg_filename)
            
            # Save figure as SVG
            canonical_ps.savefig(svg_path)
            
            # Read SVG content
            svg_content = ""
            if os.path.exists(svg_path):
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
            
            # Store this PSMILES in session
            psmiles_data = {
                'original': psmiles_string,
                'canonical': str(canonical_ps),
                'object': ps,
                'svg_path': svg_path,
                'timestamp': datetime.now().isoformat(),
                'type': 'organic'
            }
            self.session_psmiles[session_id].append(psmiles_data)
            
            # Generate workflow options based on current step
            workflow_options = self._generate_workflow_options(ps, session_id, step)
            
            # **FIX CANONICAL PSMILES DISPLAY** - Ensure proper [*] connection points for display
            canonical_str = str(canonical_ps)
            
            # The PSMILES library sometimes converts [*] to [] in canonical form
            # Fix this for proper display while preserving chemical validity
            if '[]' in canonical_str and '[*]' not in canonical_str:
                canonical_display = canonical_str.replace('[]', '[*]')
                print(f"🔧 Fixed canonical display: {canonical_str} → {canonical_display}")
            else:
                canonical_display = canonical_str
            
            return {
                'success': True,
                'original_psmiles': psmiles_string,
                'canonical_psmiles': canonical_display,  # Use fixed display version
                'svg_content': svg_content,
                'svg_filename': svg_filename,
                'workflow_options': workflow_options,
                'session_count': len(self.session_psmiles[session_id]),
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'type': 'organic'
            }
            
        except Exception as e:
            # If standard processing fails, check if it might be an organometallic compound
            metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc'}
            contains_metal = any(f'[{metal}]' in psmiles_string or f'{metal}+' in psmiles_string for metal in metal_atoms)
            
            if contains_metal:
                return self._handle_organometallic_psmiles(psmiles_string, session_id, step)
            
            return {
                'success': False,
                'error': f"PSMILES processing error: {str(e)}",
                'original_psmiles': psmiles_string
            }
    
    def _handle_organometallic_psmiles(self, psmiles_string: str, session_id: str, step: str) -> Dict:
        """
        Handle organometallic PSMILES that can't be processed by the standard psmiles library.
        
        Args:
            psmiles_string (str): Organometallic PSMILES string
            session_id (str): Session identifier
            step (str): Current workflow step
            
        Returns:
            Dict: Results with limited functionality for organometallic compounds
        """
        try:
            # Store in session for tracking
            if session_id not in self.session_psmiles:
                self.session_psmiles[session_id] = []
            
            # Create a simple text-based representation
            svg_content = self._create_organometallic_placeholder_svg(psmiles_string)
            
            # Store this PSMILES in session (without PS object)
            psmiles_data = {
                'original': psmiles_string,
                'canonical': psmiles_string,  # Can't canonicalize organometallic
                'object': None,  # No PS object available
                'svg_path': None,  # No file generated
                'timestamp': datetime.now().isoformat(),
                'type': 'organometallic'
            }
            self.session_psmiles[session_id].append(psmiles_data)
            
            # Generate limited workflow options for organometallic compounds
            workflow_options = self._generate_organometallic_workflow_options(psmiles_string, session_id, step)
            
            return {
                'success': True,
                'original_psmiles': psmiles_string,
                'canonical_psmiles': psmiles_string,
                'svg_content': svg_content,
                'svg_filename': 'organometallic_placeholder.svg',
                'workflow_options': workflow_options,
                'session_count': len(self.session_psmiles[session_id]),
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'type': 'organometallic',
                'note': 'Organometallic compound - Limited functionality available'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Organometallic PSMILES handling error: {str(e)}",
                'original_psmiles': psmiles_string,
                'type': 'organometallic'
            }
    
    def perform_dimerization(self, session_id: str, psmiles_index: int, star_index: int) -> Dict:
        """
        Perform dimerization on a PSMILES from the session.
        
        Args:
            session_id (str): Session identifier
            psmiles_index (int): Index of PSMILES in session list
            star_index (int): Which star to connect (0 or 1)
        
        Returns:
            Dict: Dimerization results
        """
        if not self.available:
            return {'success': False, 'error': 'PSMILES library not available'}
        
        try:
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            
            # Check if this is an organometallic compound
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Dimerization not supported for organometallic compounds',
                    'suggestion': 'Try using simpler organic polymer structures for dimerization operations'
                }
            
            ps = psmiles_data['object']
            if ps is None:
                return {'success': False, 'error': 'PSMILES object not available'}
            
            # Perform dimerization
            dimer_ps = ps.dimer(star_index)
            
            # Process the dimer through the workflow
            result = self.process_psmiles_workflow(str(dimer_ps), session_id, step="dimer")
            
            if result['success']:
                result['operation'] = f'Dimerization (star {star_index})'
                result['parent_psmiles'] = psmiles_data['original']
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Dimerization error: {str(e)}"
            }
    
    def perform_copolymerization(self, session_id: str, psmiles1_index: int, psmiles2_string: str, 
                               connection_pattern: List[int]) -> Dict:
        """
        Perform alternating copolymerization between two PSMILES.
        
        Args:
            session_id (str): Session identifier
            psmiles1_index (int): Index of first PSMILES in session
            psmiles2_string (str): Second PSMILES string
            connection_pattern (List[int]): Connection pattern [star1, star2]
        
        Returns:
            Dict: Copolymerization results
        """
        if not self.available:
            return {'success': False, 'error': 'PSMILES library not available'}
        
        try:
            if session_id not in self.session_psmiles or psmiles1_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'First PSMILES not found in session'}
            
            psmiles1_data = self.session_psmiles[session_id][psmiles1_index]
            
            # Check if the first PSMILES is organometallic
            if psmiles1_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Copolymerization not supported for organometallic compounds',
                    'suggestion': 'Try using simpler organic polymer structures for copolymerization operations'
                }
            
            # Check if the second PSMILES contains metals
            metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc'}
            contains_metal = any(f'[{metal}]' in psmiles2_string or f'{metal}+' in psmiles2_string for metal in metal_atoms)
            
            if contains_metal:
                return {
                    'success': False,
                    'error': 'Copolymerization not supported with organometallic compounds',
                    'suggestion': 'Use organic polymer structures only for copolymerization operations'
                }
            
            ps1 = psmiles1_data['object']
            if ps1 is None:
                return {'success': False, 'error': 'First PSMILES object not available'}
            
            ps2 = PS(psmiles2_string)
            
            # Perform alternating copolymerization
            copolymer_ps = ps1.alternating_copolymer(ps2, connection_pattern)
            copolymer_string = str(copolymer_ps)
            
            # Validate the resulting copolymer PSMILES
            star_count = copolymer_string.count('[*]')
            if star_count != 2:
                return {
                    'success': False,
                    'error': f'Copolymerization produced invalid PSMILES with {star_count} [*] symbols instead of 2: {copolymer_string}. This is a library issue - please try different connection patterns or simpler structures.'
                }
            
            # Process the copolymer through the workflow
            result = self.process_psmiles_workflow(copolymer_string, session_id, step="copolymer")
            
            if result['success']:
                result['operation'] = f'Alternating Copolymer (pattern: {connection_pattern})'
                result['parent_psmiles1'] = psmiles1_data['original']
                result['parent_psmiles2'] = psmiles2_string
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Copolymerization error: {str(e)}"
            }
    
    def get_fingerprints(self, session_id: str, psmiles_index: int, fingerprint_types: List[str]) -> Dict:
        """
        Generate various fingerprints for a PSMILES.
        
        Args:
            session_id (str): Session identifier
            psmiles_index (int): Index of PSMILES in session
            fingerprint_types (List[str]): Types of fingerprints to generate ['ci', 'mordred', 'rdkit', 'polyBERT']
        
        Returns:
            Dict: Fingerprint results
        """
        if not self.available:
            return {'success': False, 'error': 'PSMILES library not available'}
        
        try:
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            
            # Check if this is an organometallic compound
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Fingerprint generation not supported for organometallic compounds',
                    'suggestion': 'The psmiles library cannot generate fingerprints for organometallic structures. Consider extracting organic parts only.'
                }
            
            ps = psmiles_data['object']
            if ps is None:
                return {'success': False, 'error': 'PSMILES object not available'}
            
            fingerprints = {}
            
            for fp_type in fingerprint_types:
                try:
                    if fp_type.lower() == 'mordred':
                        # Get first 10 Mordred fingerprints for display
                        fp_result = ps.fingerprint('mordred')
                        fingerprints[fp_type] = {k: v for k, v in list(fp_result.items())[:10]}
                    else:
                        fingerprints[fp_type] = ps.fingerprint(fp_type.lower())
                except Exception as e:
                    fingerprints[fp_type] = f"Error: {str(e)}"
            
            return {
                'success': True,
                'psmiles': psmiles_data['original'],
                'fingerprints': fingerprints
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Fingerprint generation error: {str(e)}"
            }
    
    def get_inchi_info(self, session_id: str, psmiles_index: int) -> Dict:
        """
        Get InChI information for a PSMILES.
        
        Args:
            session_id (str): Session identifier
            psmiles_index (int): Index of PSMILES in session
        
        Returns:
            Dict: InChI results
        """
        if not self.available:
            return {'success': False, 'error': 'PSMILES library not available'}
        
        try:
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            
            # Check if this is an organometallic compound
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'InChI generation not supported for organometallic compounds',
                    'suggestion': 'The psmiles library cannot generate InChI for organometallic structures. Consider using specialized organometallic chemistry software.'
                }
            
            ps = psmiles_data['object']
            if ps is None:
                return {'success': False, 'error': 'PSMILES object not available'}
            
            # Get InChI information
            inchi = ps.inchi
            inchi_key = ps.inchi_key
            
            return {
                'success': True,
                'psmiles': psmiles_data['original'],
                'inchi': inchi,
                'inchi_key': inchi_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"InChI generation error: {str(e)}"
            }
    
    def _generate_workflow_options(self, ps, session_id: str, current_step: str) -> Dict:
        """Generate workflow options based on current PSMILES and step."""
        options = {
            'dimerization': {
                'title': '🔗 Dimerization',
                'description': 'Create dimers by connecting star atoms',
                'options': [
                    {'id': 'dimer_0', 'label': 'Connect to First Star [*]', 'star': 0},
                    {'id': 'dimer_1', 'label': 'Connect to Second Star [*]', 'star': 1}
                ]
            },
            'addition': {
                'title': '➕ Addition',
                'description': 'Add new components to the polymer',
                'placeholder': 'Describe what you want to add (e.g., "add aromatic rings", "make it longer")'
            },
            'copolymerization': {
                'title': '🧬 Block Copolymerization',
                'description': 'Combine with another PSMILES string',
                'options': [
                    {'id': 'copolymer_11', 'label': 'Pattern [1,1] - Connect both second stars', 'pattern': [1, 1]},
                    {'id': 'copolymer_01', 'label': 'Pattern [0,1] - First->first, Second->second', 'pattern': [0, 1]},
                    {'id': 'copolymer_10', 'label': 'Pattern [1,0] - First->second, Second->first', 'pattern': [1, 0]},
                    {'id': 'copolymer_00', 'label': 'Pattern [0,0] - Connect both first stars', 'pattern': [0, 0]}
                ]
            },
            'analysis': {
                'title': '🔬 Analysis',
                'description': 'Analyze the polymer properties',
                'options': [
                    {'id': 'fingerprints', 'label': 'Generate Fingerprints (CI, RDKit, polyBERT)'},
                    {'id': 'inchi', 'label': 'Get InChI String and Key'},
                    {'id': 'properties', 'label': 'Calculate Properties'}
                ]
            },
            'other': {
                'title': '💭 Other Operations',
                'description': 'Free-form modifications and questions',
                'placeholder': 'Ask anything about this PSMILES or request specific modifications'
            }
        }
        
        # Add session history context
        session_history = []
        if session_id in self.session_psmiles:
            for i, psmiles_data in enumerate(self.session_psmiles[session_id]):
                session_history.append({
                    'index': i,
                    'psmiles': psmiles_data['original'],
                    'canonical': psmiles_data['canonical'],
                    'timestamp': psmiles_data['timestamp']
                })
        
        options['session_history'] = session_history
        options['current_step'] = current_step
        
        return options
    
    def get_session_psmiles(self, session_id: str) -> List[Dict]:
        """Get all PSMILES strings for a session."""
        if session_id not in self.session_psmiles:
            return []
        
        return [
            {
                'index': i,
                'original': data['original'],
                'canonical': data['canonical'],
                'timestamp': data['timestamp']
            }
            for i, data in enumerate(self.session_psmiles[session_id])
        ]
    
    def clear_session(self, session_id: str):
        """Clear all PSMILES data for a session."""
        if session_id in self.session_psmiles:
            # Clean up temporary SVG files
            for psmiles_data in self.session_psmiles[session_id]:
                svg_path = psmiles_data.get('svg_path')
                if svg_path and os.path.exists(svg_path):
                    try:
                        os.remove(svg_path)
                    except:
                        pass
            
            del self.session_psmiles[session_id]
    
    def get_status(self) -> Dict:
        """Get processor status and capabilities."""
        return {
            'available': self.available,
            'capabilities': {
                'canonicalization': self.available,
                'visualization': self.available,
                'dimerization': self.available,
                'copolymerization': self.available,
                'fingerprints': self.available,
                'inchi_generation': self.available
            },
            'active_sessions': len(self.session_psmiles),
            'library_info': 'psmiles[polyBERT,mordred]' if self.available else 'Not installed'
        }
    
    def _create_organometallic_placeholder_svg(self, psmiles_string: str) -> str:
        """
        Create a placeholder SVG for organometallic compounds that can't be visualized.
        
        Args:
            psmiles_string (str): The organometallic PSMILES string
            
        Returns:
            str: SVG content as a string
        """
        # Extract metal atoms for display
        metal_atoms = {'Fe', 'Ni', 'Cu', 'Zn', 'Mn', 'Co', 'Cr', 'Ti', 'V', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os', 'Re', 'W', 'Mo', 'Tc', 'Nb', 'Ta', 'Hf', 'Zr', 'Y', 'Sc'}
        detected_metals = []
        
        for metal in metal_atoms:
            if f'[{metal}]' in psmiles_string or f'{metal}+' in psmiles_string:
                detected_metals.append(metal)
        
        metals_text = ', '.join(detected_metals) if detected_metals else 'Metal'
        
        # Create a simple SVG placeholder
        svg_content = f"""
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="300" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
            
            <!-- Title -->
            <text x="200" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#495057">
                Organometallic Compound
            </text>
            
            <!-- Metal symbol in center -->
            <circle cx="200" cy="120" r="40" fill="#6c757d" stroke="#495057" stroke-width="2"/>
            <text x="200" y="130" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="white">
                {metals_text}
            </text>
            
            <!-- Organic ligands -->
            <circle cx="120" cy="120" r="25" fill="#28a745" stroke="#1e7e34" stroke-width="2"/>
            <text x="120" y="127" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="white">
                Org
            </text>
            
            <circle cx="280" cy="120" r="25" fill="#28a745" stroke="#1e7e34" stroke-width="2"/>
            <text x="280" y="127" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="white">
                Org
            </text>
            
            <!-- Connection lines -->
            <line x1="145" y1="120" x2="175" y2="120" stroke="#495057" stroke-width="2"/>
            <line x1="225" y1="120" x2="255" y2="120" stroke="#495057" stroke-width="2"/>
            
            <!-- PSMILES string -->
            <text x="200" y="180" text-anchor="middle" font-family="monospace" font-size="10" fill="#6c757d">
                {psmiles_string[:60]}
            </text>
            {f'<text x="200" y="195" text-anchor="middle" font-family="monospace" font-size="10" fill="#6c757d">{psmiles_string[60:]}</text>' if len(psmiles_string) > 60 else ''}
            
            <!-- Note -->
            <text x="200" y="230" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#dc3545">
                ⚠️ Limited functionality available
            </text>
            
            <text x="200" y="250" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#6c757d">
                The psmiles library cannot process organometallic compounds
            </text>
            
            <text x="200" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#6c757d">
                Generated by natural language method with RDKit validation
            </text>
        </svg>
        """
        return svg_content.strip()
    
    def _generate_organometallic_workflow_options(self, psmiles_string: str, session_id: str, step: str) -> Dict:
        """
        Generate limited workflow options for organometallic compounds.
        
        Args:
            psmiles_string (str): The organometallic PSMILES string
            session_id (str): Session identifier
            step (str): Current workflow step
            
        Returns:
            Dict: Limited workflow options
        """
        options = {
            'analysis': {
                'title': '🔬 Limited Analysis',
                'description': 'Basic analysis available for organometallic compounds',
                'options': [
                    {'id': 'metal_info', 'label': 'Show Metal Information'},
                    {'id': 'basic_info', 'label': 'Show Basic PSMILES Info'},
                    {'id': 'export_text', 'label': 'Export as Text'}
                ]
            },
            'conversion': {
                'title': '🔄 Conversion Options',
                'description': 'Try converting to simpler forms',
                'options': [
                    {'id': 'extract_organic', 'label': 'Extract Organic Parts Only'},
                    {'id': 'simplify', 'label': 'Create Simplified Version'},
                    {'id': 'alternative', 'label': 'Suggest Alternative Polymers'}
                ]
            },
            'limitations': {
                'title': '⚠️ Current Limitations',
                'description': 'Operations not available for organometallic compounds',
                'disabled_operations': [
                    'Dimerization (requires psmiles library support)',
                    'Copolymerization (library limitation)',
                    'Fingerprint generation (library limitation)',
                    'Structure visualization (library limitation)',
                    'InChI generation (library limitation)'
                ]
            },
            'help': {
                'title': '💡 Suggestions',
                'description': 'Alternative approaches',
                'suggestions': [
                    'Use simpler organic polymer structures for full functionality',
                    'Consider the organic ligands separately for polymer design',
                    'Try polymer builder mode for 3D structure generation',
                    'Export the PSMILES for use in specialized organometallic software'
                ]
            }
        }
        
        # Add session history context
        session_history = []
        if session_id in self.session_psmiles:
            for i, psmiles_data in enumerate(self.session_psmiles[session_id]):
                session_history.append({
                    'index': i,
                    'psmiles': psmiles_data['original'],
                    'canonical': psmiles_data['canonical'],
                    'timestamp': psmiles_data['timestamp'],
                    'type': psmiles_data.get('type', 'unknown')
                })
        
        options['session_history'] = session_history
        options['current_step'] = step
        
        return options 

    def add_random_functional_groups(self, session_id: str, psmiles_index: int, 
                                   num_groups: int = 1, specific_groups: List[str] = None,
                                   random_seed: int = None) -> Dict:
        """
        Add random functional groups to a PSMILES structure - "master of degeneration"!
        
        Args:
            session_id (str): Session identifier
            psmiles_index (int): Index of PSMILES in session
            num_groups (int): Number of functional groups to add
            specific_groups (List[str]): Specific groups to add (None for random)
            random_seed (int): Random seed for reproducibility
        
        Returns:
            Dict: Results with modified PSMILES
        """
        if not self.available:
            return {
                'success': False,
                'error': 'PSMILES library not available',
                'suggestion': 'Install psmiles library to enable functional group addition'
            }
        
        try:
            import random
            if random_seed:
                random.seed(random_seed)
            
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            base_psmiles = psmiles_data['canonical']
            
            # Check for organometallic compounds
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Random functional group addition not supported for organometallic compounds',
                    'suggestion': 'Try using organic polymer structures for functional group modification'
                }
            
            # Select functional groups
            if specific_groups:
                selected_groups = [group for group in specific_groups if group in self.functional_groups]
            else:
                available_groups = list(self.functional_groups.keys())
                selected_groups = random.sample(available_groups, min(num_groups, len(available_groups)))
            
            # Create copolymers with functional groups
            modified_psmiles = base_psmiles
            applied_groups = []
            
            for group_name in selected_groups:
                try:
                    # Create PSMILES objects
                    base_ps = PS(modified_psmiles)
                    fg_ps = PS(self.functional_groups[group_name]['pattern'])
                    
                    # Create alternating copolymer with random connection pattern
                    connection_patterns = [[0,0], [0,1], [1,0], [1,1]]
                    pattern = random.choice(connection_patterns)
                    
                    # Perform copolymerization
                    copolymer_ps = base_ps.alternating_copolymer(fg_ps, pattern)
                    modified_psmiles = str(copolymer_ps)
                    
                    applied_groups.append({
                        'name': group_name,
                        'pattern': self.functional_groups[group_name]['pattern'],
                        'description': self.functional_groups[group_name]['description'],
                        'connection_pattern': pattern
                    })
                    
                except Exception as e:
                    print(f"⚠️  Failed to add {group_name}: {e}")
                    continue
            
            if not applied_groups:
                return {
                    'success': False,
                    'error': 'No functional groups could be added',
                    'suggestion': 'Try with a simpler base polymer structure'
                }
            
            # Process the modified PSMILES through the workflow
            result = self.process_psmiles_workflow(modified_psmiles, session_id, "functional_addition")
            
            if result['success']:
                result['operation'] = f'Random functional group addition'
                result['parent_psmiles'] = base_psmiles
                result['applied_groups'] = applied_groups
                result['num_groups_added'] = len(applied_groups)
                result['random_seed'] = random_seed
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Functional group addition error: {str(e)}"
            }
    
    def generate_comprehensive_fingerprints(self, session_id: str, psmiles_index: int) -> Dict:
        """
        Generate comprehensive fingerprints using the psmiles library.
        
        Args:
            session_id (str): Session identifier
            psmiles_index (int): Index of PSMILES in session
        
        Returns:
            Dict: Comprehensive fingerprint data
        """
        if not self.available:
            return {
                'success': False,
                'error': 'PSMILES library not available',
                'suggestion': 'Install psmiles library to enable fingerprint generation'
            }
        
        try:
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            
            # Check for organometallic compounds
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Fingerprint generation not supported for organometallic compounds',
                    'suggestion': 'The psmiles library cannot generate fingerprints for organometallic structures. Consider extracting organic parts only.'
                }
            
            ps = psmiles_data['object']
            if ps is None:
                return {'success': False, 'error': 'PSMILES object not available'}
            
            fingerprints = {}
            
            # polyBERT fingerprint (main one from user's example)
            try:
                polybert_fp = ps.fingerprint("polyBERT")
                fingerprints['polyBERT'] = {
                    'type': 'polyBERT',
                    'values': polybert_fp.tolist() if hasattr(polybert_fp, 'tolist') else list(polybert_fp),
                    'description': 'Polymer-specific BERT-based molecular fingerprint'
                }
            except Exception as e:
                fingerprints['polyBERT'] = {'error': f'Failed to generate polyBERT fingerprint: {e}'}
            
            # Try other available fingerprint types
            fingerprint_types = ['morgan', 'descriptors', 'maccs']
            
            for fp_type in fingerprint_types:
                try:
                    fp = ps.fingerprint(fp_type)
                    fingerprints[fp_type] = {
                        'type': fp_type,
                        'values': fp.tolist() if hasattr(fp, 'tolist') else list(fp),
                        'description': f'{fp_type.capitalize()} fingerprint'
                    }
                except Exception as e:
                    fingerprints[fp_type] = {'error': f'Failed to generate {fp_type} fingerprint: {e}'}
            
            return {
                'success': True,
                'psmiles': psmiles_data['canonical'],
                'fingerprints': fingerprints,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Fingerprint generation error: {str(e)}"
            }
    
    def create_advanced_dimers(self, session_id: str, psmiles_index: int) -> Dict:
        """
        Create both possible dimers (star 0 and star 1) as shown in user's example.
        
        Args:
            session_id (str): Session identifier  
            psmiles_index (int): Index of PSMILES in session
        
        Returns:
            Dict: Both dimer structures
        """
        if not self.available:
            return {
                'success': False,
                'error': 'PSMILES library not available'
            }
        
        try:
            if session_id not in self.session_psmiles or psmiles_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'PSMILES not found in session'}
            
            psmiles_data = self.session_psmiles[session_id][psmiles_index]
            
            # Check for organometallic compounds
            if psmiles_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Dimerization not supported for organometallic compounds'
                }
            
            ps = psmiles_data['object']
            if ps is None:
                return {'success': False, 'error': 'PSMILES object not available'}
            
            # Create both dimers
            dimer_0 = ps.dimer(0)  # Connect via star 0
            dimer_1 = ps.dimer(1)  # Connect via star 1
            
            # Process both dimers
            results = {
                'success': True,
                'original_psmiles': psmiles_data['canonical'],
                'dimers': {
                    'star_0': {
                        'psmiles': str(dimer_0),
                        'description': 'Dimer connected via star 0 position'
                    },
                    'star_1': {
                        'psmiles': str(dimer_1), 
                        'description': 'Dimer connected via star 1 position'
                    }
                },
                'operation': 'Advanced dimerization (both positions)',
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Advanced dimerization error: {str(e)}"
            }
    
    def create_copolymer_library(self, session_id: str, psmiles1_index: int, 
                               psmiles2_string: str) -> Dict:
        """
        Create a library of copolymers with all possible connection patterns.
        
        Args:
            session_id (str): Session identifier
            psmiles1_index (int): Index of first PSMILES in session  
            psmiles2_string (str): Second PSMILES string
        
        Returns:
            Dict: Library of copolymer structures
        """
        if not self.available:
            return {
                'success': False,
                'error': 'PSMILES library not available'
            }
        
        try:
            if session_id not in self.session_psmiles or psmiles1_index >= len(self.session_psmiles[session_id]):
                return {'success': False, 'error': 'First PSMILES not found in session'}
            
            psmiles1_data = self.session_psmiles[session_id][psmiles1_index]
            
            # Check for organometallic compounds
            if psmiles1_data.get('type') == 'organometallic':
                return {
                    'success': False,
                    'error': 'Copolymerization not supported for organometallic compounds'
                }
            
            ps1 = psmiles1_data['object']
            if ps1 is None:
                return {'success': False, 'error': 'First PSMILES object not available'}
            
            # Create second PSMILES object
            ps2 = PS(psmiles2_string)
            
            # All possible connection patterns
            connection_patterns = {
                '[0,0]': [0, 0],
                '[0,1]': [0, 1], 
                '[1,0]': [1, 0],
                '[1,1]': [1, 1]
            }
            
            copolymers = {}
            
            for pattern_name, pattern in connection_patterns.items():
                try:
                    copolymer_ps = ps1.alternating_copolymer(ps2, pattern)
                    copolymers[pattern_name] = {
                        'psmiles': str(copolymer_ps),
                        'pattern': pattern,
                        'description': f'Alternating copolymer with connection pattern {pattern_name}'
                    }
                except Exception as e:
                    copolymers[pattern_name] = {
                        'error': f'Failed with pattern {pattern_name}: {e}'
                    }
            
            return {
                'success': True,
                'psmiles1': psmiles1_data['canonical'],
                'psmiles2': psmiles2_string,
                'copolymers': copolymers,
                'operation': 'Copolymer library generation',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Copolymer library error: {str(e)}"
            } 