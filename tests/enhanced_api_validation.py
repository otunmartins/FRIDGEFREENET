# Enhanced API Validation System with LangChain Integration
# Based on research and LangChain best practices for error handling

import time
import json
import logging
import requests
import requests_cache
from typing import Optional, Dict, Any, List, Union
from functools import wraps
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LangChain Integration Components
# ============================================================================

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain.schema.runnable import RunnableWithFallbacks
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Falling back to basic validation.")
    LANGCHAIN_AVAILABLE = False
    
    # Create mock classes for compatibility
    class BaseModel:
        pass
    class Field:
        def __init__(self, *args, **kwargs):
            pass
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def tool(func):
        return func

# ============================================================================
# Pydantic Models for Structured Validation
# ============================================================================

class SMILESValidationResult(BaseModel):
    """Structured output for SMILES validation operations."""
    success: bool = Field(description="Whether the validation was successful")
    validated_smiles: Optional[str] = Field(description="The validated SMILES string")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the validation (0-1)")
    error_details: Optional[str] = Field(description="Details of any validation errors")
    source: str = Field(description="Source of the SMILES (pubchem, fallback, cache)")
    corrections_applied: List[str] = Field(default_factory=list, description="List of corrections applied")

class APIRequestValidation(BaseModel):
    """Validation for API request parameters."""
    molecule_name: str = Field(min_length=1, max_length=200, description="Valid molecule name")
    search_strategies: List[str] = Field(default_factory=list, description="Strategies to try")
    max_retries: int = Field(ge=1, le=10, default=3, description="Maximum retry attempts")
    timeout_seconds: int = Field(ge=5, le=60, default=30, description="Request timeout")

# ============================================================================
# Enhanced Caching System (Based on ChemInformant approach)
# ============================================================================

class EnhancedCacheManager:
    """Advanced caching system with automatic cleanup and validation."""
    
    def __init__(self, cache_name='enhanced_pubchem_cache', backend='sqlite', expire_after=604800):
        """
        Initialize enhanced caching system.
        
        Args:
            cache_name: Name of the cache file/database
            backend: Caching backend ('sqlite', 'memory', 'redis')
            expire_after: Cache expiration time in seconds (default: 7 days)
        """
        self.cache_name = cache_name
        self.backend = backend
        self.expire_after = expire_after
        self._session = None
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup requests-cache with enhanced configuration."""
        try:
            # Configure cache based on ChemInformant best practices
            cache_config = {
                'cache_name': self.cache_name,
                'backend': self.backend,
                'expire_after': self.expire_after,
                'allowable_codes': [200, 404],  # Cache both successful and not-found responses
                'allowable_methods': ['GET', 'POST'],
                'ignored_parameters': ['timestamp', 'session_id'],  # Don't cache these params
            }
            
            if self.backend == 'sqlite':
                cache_config['fast_save'] = True
                cache_config['use_cache_dir'] = True
            
            self._session = requests_cache.CachedSession(**cache_config)
            logger.info(f"✅ Enhanced cache initialized: {self.backend} backend, {self.expire_after}s expiry")
            
        except Exception as e:
            logger.error(f"❌ Cache setup failed: {e}. Falling back to regular requests.")
            self._session = requests.Session()
    
    def get_session(self) -> requests.Session:
        """Get the cached session."""
        return self._session if self._session else requests.Session()
    
    def clear_cache(self):
        """Clear the cache."""
        try:
            if hasattr(self._session, 'cache'):
                self._session.cache.clear()
                logger.info("✅ Cache cleared successfully")
        except Exception as e:
            logger.error(f"❌ Cache clear failed: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if hasattr(self._session, 'cache'):
                return {
                    'backend': self.backend,
                    'cache_name': self.cache_name,
                    'expire_after': self.expire_after,
                    'urls_count': len(self._session.cache.urls),
                    'responses_count': len(self._session.cache.responses),
                }
        except Exception as e:
            logger.error(f"❌ Cache info failed: {e}")
        return {}

# ============================================================================
# LangChain-Based Validation Chain
# ============================================================================

class SMILESValidationChain:
    """LangChain-powered validation and correction system."""
    
    def __init__(self, model_name: str = "granite3.3:8b"):
        self.model_name = model_name
        self.cache_manager = EnhancedCacheManager()
        self._setup_validation_chains()
    
    def _setup_validation_chains(self):
        """Setup LangChain validation chains."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Using basic validation.")
            return
        
        try:
            # Validation prompt template
            self.validation_prompt = ChatPromptTemplate.from_template(
                """
                You are a chemistry expert. Review this molecule name for correctness and suggest improvements.
                
                Molecule Name: {molecule_name}
                Error Context: {error_context}
                
                Tasks:
                1. Check if the name is a valid chemical name
                2. Suggest corrections for common misspellings
                3. Identify if it's a common name vs IUPAC name
                4. Provide alternative names to search
                
                Return a JSON response with:
                {{
                    "is_valid": true/false,
                    "corrected_name": "corrected name",
                    "alternative_names": ["alt1", "alt2"],
                    "confidence": 0.0-1.0,
                    "reasoning": "explanation"
                }}
                """
            )
            
            # Correction prompt template
            self.correction_prompt = ChatPromptTemplate.from_template(
                """
                The molecule search failed. Help fix the search query.
                
                Original Query: {original_query}
                Error: {error_message}
                Previous Attempts: {previous_attempts}
                
                Provide a better search query that is more likely to work with PubChem.
                Consider:
                - Common chemical name variations
                - IUPAC vs common names
                - Spelling corrections
                - Synonym alternatives
                
                Return only the improved search query as plain text.
                """
            )
            
            logger.info("✅ LangChain validation chains initialized")
            
        except Exception as e:
            logger.error(f"❌ LangChain setup failed: {e}")
    
    def validate_and_correct_molecule_name(self, molecule_name: str, error_context: str = "") -> Dict[str, Any]:
        """
        Use LangChain to validate and correct molecule names.
        
        Args:
            molecule_name: The molecule name to validate
            error_context: Context about any errors encountered
            
        Returns:
            Dictionary with validation results and corrections
        """
        if not LANGCHAIN_AVAILABLE:
            # Fallback to basic validation
            return self._basic_name_validation(molecule_name)
        
        try:
            # Use Ollama through LangChain for validation
            from langchain_community.llms import Ollama
            
            llm = Ollama(model=self.model_name, temperature=0.1)
            
            # Create validation chain
            validation_chain = (
                self.validation_prompt | 
                llm | 
                StrOutputParser()
            )
            
            # Execute validation
            result = validation_chain.invoke({
                "molecule_name": molecule_name,
                "error_context": error_context
            })
            
            # Parse JSON response
            try:
                parsed_result = json.loads(result)
                logger.info(f"✅ LangChain validation completed for '{molecule_name}'")
                return parsed_result
            except json.JSONDecodeError:
                logger.warning(f"⚠️ LangChain returned non-JSON response: {result}")
                return self._basic_name_validation(molecule_name)
                
        except Exception as e:
            logger.error(f"❌ LangChain validation failed: {e}")
            return self._basic_name_validation(molecule_name)
    
    def _basic_name_validation(self, molecule_name: str) -> Dict[str, Any]:
        """Basic validation without LangChain."""
        # Simple heuristics for molecule name validation
        is_valid = bool(molecule_name and len(molecule_name.strip()) > 0)
        
        # Common corrections
        corrected_name = molecule_name.strip().lower()
        alternative_names = [
            corrected_name,
            corrected_name.replace('-', ''),
            corrected_name.replace(' ', ''),
        ]
        
        return {
            "is_valid": is_valid,
            "corrected_name": corrected_name,
            "alternative_names": list(set(alternative_names)),
            "confidence": 0.7 if is_valid else 0.3,
            "reasoning": "Basic validation without LangChain"
        }

# ============================================================================
# Enhanced PubChem API Client with Validation
# ============================================================================

class ValidatedPubChemClient:
    """PubChem client with LangChain validation and enhanced error handling."""
    
    def __init__(self, model_name: str = "granite3.3:8b"):
        self.model_name = model_name
        self.validation_chain = SMILESValidationChain(model_name)
        self.cache_manager = EnhancedCacheManager()
        
        # Rate limiting (NCBI guidelines: max 30 requests/minute)
        self.min_request_interval = 2.0  # seconds between requests
        self.last_request_time = 0.0
        
        # Enhanced known molecules database
        self.known_molecules = self._load_enhanced_molecule_database()
    
    def _load_enhanced_molecule_database(self) -> Dict[str, str]:
        """Load comprehensive molecule database."""
        return {
            # Amino acids (standard 20)
            'alanine': 'C[C@@H](C(=O)O)N',
            'arginine': 'C(C[C@@H](C(=O)O)N)CN=C(N)N',
            'asparagine': 'C([C@@H](C(=O)O)N)C(=O)N',
            'aspartic acid': 'C([C@@H](C(=O)O)N)C(=O)O',
            'cysteine': 'C([C@@H](C(=O)O)N)S',
            'glutamic acid': 'C(CC(=O)O)[C@@H](C(=O)O)N',
            'glutamine': 'C(CC(=O)N)[C@@H](C(=O)O)N',
            'glycine': 'C(C(=O)O)N',
            'histidine': 'c1c([nH]cn1)C[C@@H](C(=O)O)N',
            'isoleucine': 'CC[C@H](C)[C@@H](C(=O)O)N',
            'leucine': 'CC(C)C[C@@H](C(=O)O)N',
            'lysine': 'C(CCN)C[C@@H](C(=O)O)N',
            'methionine': 'CSCCC(C(=O)O)N',
            'phenylalanine': 'c1ccc(cc1)C[C@@H](C(=O)O)N',
            'proline': 'C1C[C@H](NC1)C(=O)O',
            'serine': 'C([C@@H](C(=O)O)N)O',
            'threonine': 'C[C@H]([C@@H](C(=O)O)N)O',
            'tryptophan': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
            'tyrosine': 'c1cc(ccc1C[C@@H](C(=O)O)N)O',
            'valine': 'CC(C)[C@@H](C(=O)O)N',
            
            # Common solvents
            'water': 'O',
            'ethanol': 'CCO',
            'methanol': 'CO',
            'acetone': 'CC(=O)C',
            'benzene': 'c1ccccc1',
            'toluene': 'Cc1ccccc1',
            'dmso': 'CS(=O)C',
            'thf': 'C1CCOC1',
            'dichloromethane': 'ClCCl',
            'chloroform': 'C(Cl)(Cl)Cl',
            
            # Sugars and carbohydrates
            'glucose': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O',
            'fructose': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O',
            'sucrose': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@@H](O2)CO)O)O)CO)O)O)O)O',
            
            # Common acids
            'acetic acid': 'CC(=O)O',
            'formic acid': 'C(=O)O',
            'citric acid': 'C(C(=O)O)C(CC(=O)O)(C(=O)O)O',
            
            # Polymers and monomers
            'styrene': 'c1ccc(cc1)C=C',
            'ethylene': 'C=C',
            'propylene': 'CC=C',
            'vinyl chloride': 'C=CCl',
            'acrylonitrile': 'C=CC#N',
            'methyl methacrylate': 'CC(=C)C(=O)OC',
        }
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def get_smiles_with_validation(self, molecule_name: str, max_correction_attempts: int = 3) -> SMILESValidationResult:
        """
        Get SMILES with comprehensive validation and correction.
        
        Args:
            molecule_name: Name of the molecule to search
            max_correction_attempts: Maximum attempts to correct the name
            
        Returns:
            SMILESValidationResult with detailed validation information
        """
        logger.info(f"🔍 Starting validated SMILES search for '{molecule_name}'")
        
        # Input validation
        if not molecule_name or not isinstance(molecule_name, str):
            return SMILESValidationResult(
                success=False,
                validated_smiles=None,
                confidence_score=0.0,
                error_details="Invalid input: molecule name must be a non-empty string",
                source="validation",
                corrections_applied=[]
            )
        
        molecule_name = molecule_name.strip()
        corrections_applied = []
        
        # Try multiple correction attempts
        for attempt in range(max_correction_attempts):
            logger.info(f"🔄 Attempt {attempt + 1}: Searching for '{molecule_name}'")
            
            # 1. Check known molecules first (fastest)
            result = self._check_known_molecules(molecule_name)
            if result.success:
                return result
            
            # 2. Try PubChem API
            result = self._search_pubchem_api(molecule_name)
            if result.success:
                return result
            
            # 3. Use LangChain for correction (if available and attempts remaining)
            if attempt < max_correction_attempts - 1:
                logger.info(f"🤖 Using LangChain for correction attempt {attempt + 1}")
                validation_result = self.validation_chain.validate_and_correct_molecule_name(
                    molecule_name, 
                    f"PubChem search failed after {attempt + 1} attempts"
                )
                
                if validation_result.get("is_valid", False):
                    corrected_name = validation_result.get("corrected_name", molecule_name)
                    if corrected_name != molecule_name:
                        corrections_applied.append(f"{molecule_name} -> {corrected_name}")
                        molecule_name = corrected_name
                        logger.info(f"✨ LangChain suggested correction: '{corrected_name}'")
                        continue
                
                # Try alternative names
                alternatives = validation_result.get("alternative_names", [])
                for alt_name in alternatives:
                    if alt_name != molecule_name:
                        logger.info(f"🔄 Trying alternative name: '{alt_name}'")
                        alt_result = self._search_pubchem_api(alt_name)
                        if alt_result.success:
                            alt_result.corrections_applied = corrections_applied + [f"Used alternative: {alt_name}"]
                            return alt_result
        
        # All attempts failed
        return SMILESValidationResult(
            success=False,
            validated_smiles=None,
            confidence_score=0.0,
            error_details=f"Failed to find SMILES after {max_correction_attempts} correction attempts",
            source="failed",
            corrections_applied=corrections_applied
        )
    
    def _check_known_molecules(self, molecule_name: str) -> SMILESValidationResult:
        """Check against known molecules database."""
        normalized_name = molecule_name.lower().strip()
        
        # Try exact match and common variations
        for name_variant in [
            normalized_name,
            normalized_name.replace('-', ''),
            normalized_name.replace(' ', ''),
            normalized_name.replace('_', ''),
        ]:
            if name_variant in self.known_molecules:
                smiles = self.known_molecules[name_variant]
                logger.info(f"✅ Found in known molecules: '{name_variant}' -> {smiles}")
                return SMILESValidationResult(
                    success=True,
                    validated_smiles=smiles,
                    confidence_score=1.0,
                    error_details=None,
                    source="known_molecules",
                    corrections_applied=[]
                )
        
        return SMILESValidationResult(
            success=False,
            validated_smiles=None,
            confidence_score=0.0,
            error_details="Not found in known molecules database",
            source="known_molecules",
            corrections_applied=[]
        )
    
    def _search_pubchem_api(self, molecule_name: str) -> SMILESValidationResult:
        """Search PubChem API with enhanced error handling."""
        try:
            import pubchempy as pcp
            
            self._enforce_rate_limit()
            
            # Enhanced search strategies
            search_strategies = [
                ('name', molecule_name),
                ('name', molecule_name.lower()),
                ('name', molecule_name.replace(' ', '')),
                ('name', molecule_name.replace('-', '')),
                ('synonym', molecule_name),
            ]
            
            # Only try formula search for short, alphanumeric names
            if len(molecule_name) < 10 and molecule_name.replace(' ', '').isalnum():
                search_strategies.append(('formula', molecule_name))
            
            for strategy, search_term in search_strategies:
                try:
                    logger.debug(f"🔍 Trying PubChem strategy '{strategy}' with '{search_term}'")
                    
                    compounds = pcp.get_compounds(search_term, strategy)
                    
                    if compounds and len(compounds) > 0:
                        compound = compounds[0]
                        
                        # Prefer isomeric SMILES, fallback to canonical
                        smiles = None
                        if hasattr(compound, 'isomeric_smiles') and compound.isomeric_smiles:
                            smiles = compound.isomeric_smiles
                        elif hasattr(compound, 'canonical_smiles') and compound.canonical_smiles:
                            smiles = compound.canonical_smiles
                        
                        if smiles and smiles.strip():
                            logger.info(f"✅ PubChem success via '{strategy}': {smiles}")
                            return SMILESValidationResult(
                                success=True,
                                validated_smiles=smiles.strip(),
                                confidence_score=0.9,
                                error_details=None,
                                source=f"pubchem_{strategy}",
                                corrections_applied=[]
                            )
                            
                except Exception as strategy_error:
                    logger.debug(f"⚠️ PubChem strategy '{strategy}' failed: {strategy_error}")
                    continue
            
            return SMILESValidationResult(
                success=False,
                validated_smiles=None,
                confidence_score=0.0,
                error_details="No results found in PubChem with any strategy",
                source="pubchem_failed",
                corrections_applied=[]
            )
            
        except ImportError:
            return SMILESValidationResult(
                success=False,
                validated_smiles=None,
                confidence_score=0.0,
                error_details="pubchempy not installed",
                source="import_error",
                corrections_applied=[]
            )
        except Exception as e:
            return SMILESValidationResult(
                success=False,
                validated_smiles=None,
                confidence_score=0.0,
                error_details=f"PubChem API error: {str(e)}",
                source="api_error",
                corrections_applied=[]
            )

# ============================================================================
# Usage Example and Testing
# ============================================================================

def test_enhanced_validation_system():
    """Test the enhanced validation system."""
    client = ValidatedPubChemClient()
    
    test_molecules = [
        "alanine",           # Should work immediately
        "alinine",           # Typo - should be corrected
        "aspirin",           # Common drug
        "acetylsalicylic acid",  # IUPAC name for aspirin
        "H2O",               # Formula
        "nonexistent_molecule_xyz",  # Should fail gracefully
    ]
    
    for molecule in test_molecules:
        print(f"\n{'='*50}")
        print(f"Testing: {molecule}")
        print('='*50)
        
        result = client.get_smiles_with_validation(molecule)
        
        print(f"Success: {result.success}")
        print(f"SMILES: {result.validated_smiles}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Source: {result.source}")
        print(f"Corrections: {result.corrections_applied}")
        if result.error_details:
            print(f"Error: {result.error_details}")

if __name__ == "__main__":
    test_enhanced_validation_system() 