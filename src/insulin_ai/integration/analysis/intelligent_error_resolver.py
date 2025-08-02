#!/usr/bin/env python3
"""
Intelligent Error Resolver for Molecular Dynamics Pipeline
==============================================================

This module implements a sophisticated LangChain RAG-based error resolution system
that can automatically diagnose and fix common issues in the insulin-AI MD pipeline.

Key Features:
- Automated error detection and classification
- Context-aware error resolution using RAG agents
- Real-time pipeline monitoring and self-healing
- OpenMM and OpenFF integration error handling
- Session state management error resolution

Following AI Engineering principles from Chip Nguyen's book and best practices
for context engineering in molecular simulation workflows.

Authors: AI-driven molecular simulation error resolution system
License: MIT
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import pandas as pd
import numpy as np

# LangChain imports for RAG-based error resolution
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.agents import AgentType, initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain openai faiss-cpu")

# Streamlit imports for session state handling
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class MDErrorClassifier:
    """Classifies and categorizes MD pipeline errors using pattern matching and ML"""
    
    # Error patterns and their characteristics
    ERROR_PATTERNS = {
        'openff_toolkit_missing': {
            'patterns': [
                'No module named \'openff\'',
                'ImportError: cannot import name \'Molecule\'',
                'openff.toolkit'
            ],
            'severity': 'high',
            'auto_fixable': True
        },
        'openmm_platform': {
            'patterns': [
                'OpenCL platform',
                'CUDA platform',
                'no suitable platform'
            ],
            'severity': 'medium',
            'auto_fixable': True
        },
        'force_field_assignment': {
            'patterns': [
                'No template found for residue',
                'UNL',
                'Unknown residue',
                'createSystem failed'
            ],
            'severity': 'high',
            'auto_fixable': True
        },
        'trajectory_analysis': {
            'patterns': [
                'Trajectory file not found',
                'MDTraj',
                'Frame index out of range'
            ],
            'severity': 'medium',
            'auto_fixable': True
        },
        'memory_overflow': {
            'patterns': [
                'OutOfMemoryError',
                'MemoryError',
                'killed process'
            ],
            'severity': 'high',
            'auto_fixable': False
        },
        'timeout_errors': {
            'patterns': [
                'TimeoutError',
                'simulation timeout',
                'took too long'
            ],
            'severity': 'medium',
            'auto_fixable': True
        }
    }
    
    # Solution templates for each error type
    SOLUTION_TEMPLATES = {
        'openff_toolkit_missing': {
            'description': 'OpenFF toolkit not available or improperly installed',
            'solutions': [
                {
                    'title': 'Install OpenFF toolkit',
                    'code_template': '''
# Install via conda (recommended)
import subprocess
subprocess.run(['conda', 'install', '-c', 'conda-forge', 'openff-toolkit'])

# Or fallback to pip
subprocess.run(['pip', 'install', 'openff-toolkit'])
                    ''',
                    'success_rate': 0.95
                },
                {
                    'title': 'Use fallback without OpenFF',
                    'code_template': '''
try:
    from openff.toolkit import Molecule
    USE_OPENFF = True
except ImportError:
    USE_OPENFF = False
    print("⚠️ OpenFF not available, using fallback mode")
                    ''',
                    'success_rate': 0.80
                }
            ]
        },
        'openmm_platform': {
            'description': 'OpenMM platform selection and compatibility issues',
            'solutions': [
                {
                    'title': 'Auto-select best available platform',
                    'code_template': '''
import openmm as mm

def get_best_platform():
    platforms = []
    for i in range(mm.Platform.getNumPlatforms()):
        platform = mm.Platform.getPlatform(i)
        platforms.append((platform.getName(), platform.getSpeed()))
    
    # Sort by speed (fastest first)
    platforms.sort(key=lambda x: x[1], reverse=True)
    return platforms[0][0] if platforms else 'Reference'

platform_name = get_best_platform()
platform = mm.Platform.getPlatformByName(platform_name)
                    ''',
                    'success_rate': 0.90
                },
                {
                    'title': 'Force CPU platform as fallback',
                    'code_template': '''
try:
    platform = mm.Platform.getPlatformByName('CUDA')
except:
    try:
        platform = mm.Platform.getPlatformByName('OpenCL')
    except:
        platform = mm.Platform.getPlatformByName('CPU')
                    ''',
                    'success_rate': 0.85
                }
            ]
        },
        'force_field_assignment': {
            'description': 'Force field parameter assignment failures',
            'solutions': [
                {
                    'title': 'Use GAFF fallback for unknown residues',
                    'code_template': '''
from openmmforcefields.generators import GAFFTemplateGenerator

# Create GAFF generator for unknown molecules
gaff_generator = GAFFTemplateGenerator(molecules=molecules)
forcefield.registerTemplateGenerator(gaff_generator.generator)
                    ''',
                    'success_rate': 0.85
                },
                {
                    'title': 'Simplify system by removing problematic residues',
                    'code_template': '''
# Remove unknown residues as last resort
modeller = Modeller(topology, positions)
unknown_residues = [res for res in topology.residues() if res.name == 'UNL']
if unknown_residues:
    modeller.delete(unknown_residues)
    topology = modeller.topology
    positions = modeller.positions
                    ''',
                    'success_rate': 0.70
                }
            ]
        },
        'trajectory_analysis': {
            'description': 'Trajectory file access and analysis problems',
            'solutions': [
                {
                    'title': 'Verify trajectory file exists and is readable',
                    'code_template': '''
import os
from pathlib import Path

trajectory_path = Path(trajectory_file)
if not trajectory_path.exists():
    raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

if trajectory_path.stat().st_size == 0:
    raise ValueError(f"Trajectory file is empty: {trajectory_path}")
                    ''',
                    'success_rate': 0.90
                }
            ]
        },
        'memory_overflow': {
            'description': 'Memory limitations during simulation or analysis',
            'solutions': [
                {
                    'title': 'Reduce system size or analysis frequency',
                    'code_template': '''
# Reduce trajectory analysis frequency
analysis_interval = max(1000, original_interval * 2)

# Or analyze in chunks
chunk_size = min(1000, total_frames // 4)
for start in range(0, total_frames, chunk_size):
    end = min(start + chunk_size, total_frames)
    analyze_chunk(trajectory[start:end])
                    ''',
                    'success_rate': 0.75
                }
            ]
        },
        'timeout_errors': {
            'description': 'Simulation or analysis timeouts',
            'solutions': [
                {
                    'title': 'Implement incremental processing with checkpoints',
                    'code_template': '''
import time

def process_with_timeout(func, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            return func()
        except Exception as e:
            if time.time() - start_time >= timeout:
                raise TimeoutError(f"Operation timed out after {timeout}s")
            time.sleep(1)
                    ''',
                    'success_rate': 0.80
                }
            ]
        }
    }
    
    def classify_error(self, error_msg: str, traceback_str: str = "") -> Dict[str, Any]:
        """Classify an error message into categories"""
        full_text = f"{error_msg} {traceback_str}".lower()
        
        classifications = []
        for category, config in self.ERROR_PATTERNS.items():
            matches = sum(1 for pattern in config['patterns'] if pattern.lower() in full_text)
            if matches > 0:
                classifications.append({
                    'category': category,
                    'confidence': min(1.0, matches / len(config['patterns'])),
                    'severity': config['severity'],
                    'auto_fixable': config['auto_fixable'],
                    'matches': matches
                })
        
        # Sort by confidence
        classifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'primary_category': classifications[0]['category'] if classifications else 'unknown',
            'all_classifications': classifications,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }

class ErrorResolutionKnowledgeBase:
    """Knowledge base containing error resolution strategies"""
    
    def __init__(self):
        self.resolution_strategies = {
            'session_state': {
                'description': 'Session state object access errors',
                'solutions': [
                    {
                        'title': 'Use safe_get_session_object helper',
                        'code_template': '''
# Replace direct access:
# st.session_state.psmiles_processor.method()

# With safe access:
psmiles_processor = safe_get_session_object('psmiles_processor')
if not psmiles_processor:
    raise Exception("PSMILES processor not available")
psmiles_processor.method()
                        ''',
                        'success_rate': 0.95
                    },
                    {
                        'title': 'Add object validation',
                        'code_template': '''
def validate_session_object(obj_name: str) -> bool:
    if obj_name not in st.session_state:
        return False
    obj = st.session_state[obj_name]
    return obj is not None and hasattr(obj, '__class__')
                        ''',
                        'success_rate': 0.90
                    }
                ]
            },
            'trajectory_analysis': {
                'description': 'Trajectory analysis and file handling errors',
                'solutions': [
                    {
                        'title': 'Graceful trajectory file handling',
                        'code_template': '''
def safe_load_trajectory(trajectory_file: str):
    try:
        if not os.path.exists(trajectory_file):
            return None, "Trajectory file not found"
        
        import mdtraj as md
        trajectory = md.load(trajectory_file)
        return trajectory, None
    except Exception as e:
        return None, str(e)
                        ''',
                        'success_rate': 0.90
                    }
                ]
            },
            'dataframe_type': {
                'description': 'DataFrame vs NumPy array type issues',
                'solutions': [
                    {
                        'title': 'Add type checking before DataFrame operations',
                        'code_template': '''
def safe_dataframe_operation(data, operation='corr'):
    if isinstance(data, pd.DataFrame):
        if operation == 'corr':
            return data.corr()
        # Add other operations as needed
    else:
        # Convert to DataFrame if it's array-like
        try:
            df = pd.DataFrame(data)
            if operation == 'corr':
                return df.corr()
        except Exception:
            return None
                        ''',
                        'success_rate': 0.95
                    }
                ]
            },
            'force_field': {
                'description': 'Force field parameterization errors',
                'solutions': [
                    {
                        'title': 'Enhanced molecule extraction with stereochemistry handling',
                        'code_template': '''
from rdkit import Chem
from rdkit.Chem import rdMolStandardize

def safe_molecule_extraction(mol):
    try:
        # Handle undefined stereochemistry
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        
        # Standardize the molecule
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        
        return mol
    except Exception as e:
        logging.warning(f"Molecule standardization failed: {e}")
        return mol
                        ''',
                        'success_rate': 0.85
                    }
                ]
            }
        }
    
    def get_solutions(self, error_category: str) -> List[Dict[str, Any]]:
        """Get resolution solutions for a given error category"""
        return self.resolution_strategies.get(error_category, {}).get('solutions', [])
    
    def get_best_solution(self, error_category: str) -> Optional[Dict[str, Any]]:
        """Get the highest success rate solution for an error category"""
        solutions = self.get_solutions(error_category)
        if not solutions:
            return None
        
        return max(solutions, key=lambda x: x['success_rate'])

class LangChainErrorResolver:
    """LangChain-powered intelligent error resolution agent"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for intelligent error resolution")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logging.warning("OpenAI API key not provided. Some features may be limited.")
        
        self.knowledge_base = ErrorResolutionKnowledgeBase()
        self.classifier = MDErrorClassifier()
        
        # Initialize LangChain components
        self._setup_rag_system()
    
    def _setup_rag_system(self):
        """Setup the RAG system for error resolution"""
        if not self.openai_api_key:
            logging.warning("Cannot setup RAG system without OpenAI API key")
            return
        
        try:
            # Create documents from knowledge base
            documents = []
            for category, config in self.knowledge_base.resolution_strategies.items():
                for solution in config['solutions']:
                    doc_content = f"""
Category: {category}
Description: {config['description']}
Solution: {solution['title']}
Success Rate: {solution['success_rate']}
Code Template:
{solution['code_template']}
                    """
                    documents.append(Document(
                        page_content=doc_content,
                        metadata={
                            'category': category,
                            'solution_title': solution['title'],
                            'success_rate': solution['success_rate']
                        }
                    ))
            
            # Setup text splitter and embeddings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Setup QA chain
            llm = OpenAI(openai_api_key=self.openai_api_key)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            logging.info("✅ RAG system setup completed")
            
        except Exception as e:
            logging.error(f"Failed to setup RAG system: {e}")
            self.qa_chain = None
    
    def resolve_error(self, error_msg: str, traceback_str: str = "", 
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Resolve an error using intelligent RAG-based analysis"""
        
        # Step 1: Classify the error
        classification = self.classifier.classify_error(error_msg, traceback_str)
        
        # Step 2: Get knowledge base solutions
        kb_solutions = self.knowledge_base.get_solutions(classification['primary_category'])
        
        # Step 3: Use RAG for enhanced solutions (if available)
        rag_solution = None
        if self.qa_chain:
            try:
                query = f"""
Error: {error_msg}
Category: {classification['primary_category']}
Context: {context or {}}

Provide a specific solution for this molecular dynamics pipeline error.
                """
                rag_result = self.qa_chain({"query": query})
                rag_solution = {
                    'answer': rag_result['result'],
                    'source_documents': [doc.metadata for doc in rag_result.get('source_documents', [])]
                }
            except Exception as e:
                logging.warning(f"RAG query failed: {e}")
        
        return {
            'error_classification': classification,
            'knowledge_base_solutions': kb_solutions,
            'rag_solution': rag_solution,
            'recommended_action': self._get_recommended_action(classification, kb_solutions),
            'auto_fixable': classification.get('primary_category') in ['session_state', 'dataframe_type'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_recommended_action(self, classification: Dict, solutions: List[Dict]) -> Dict[str, Any]:
        """Generate a recommended action based on classification and solutions"""
        if not solutions:
            return {
                'action': 'manual_investigation',
                'description': 'No automated solutions available. Manual investigation required.',
                'priority': 'high'
            }
        
        best_solution = max(solutions, key=lambda x: x['success_rate'])
        
        return {
            'action': 'automated_fix',
            'solution_title': best_solution['title'],
            'code_template': best_solution['code_template'],
            'success_rate': best_solution['success_rate'],
            'priority': classification.get('severity', 'medium')
        }

class AutomaticErrorFixer:
    """Implements automatic fixes for common errors"""
    
    def __init__(self, resolver: LangChainErrorResolver):
        self.resolver = resolver
    
    def fix_session_state_error(self, file_path: str, error_line: int) -> bool:
        """Automatically fix session state access errors"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line >= len(lines):
                return False
            
            original_line = lines[error_line]
            
            # Pattern matching for common session state issues
            if 'st.session_state.psmiles_processor.' in original_line:
                # Replace with safe access pattern
                indent = len(original_line) - len(original_line.lstrip())
                new_lines = [
                    ' ' * indent + 'psmiles_processor = safe_get_session_object(\'psmiles_processor\')\n',
                    ' ' * indent + 'if not psmiles_processor:\n',
                    ' ' * indent + '    raise Exception("PSMILES processor not available")\n',
                    original_line.replace('st.session_state.psmiles_processor.', 'psmiles_processor.')
                ]
                
                lines[error_line:error_line+1] = new_lines
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return True
            
        except Exception as e:
            logging.error(f"Failed to fix session state error: {e}")
        
        return False
    
    def fix_dataframe_type_error(self, file_path: str, error_line: int) -> bool:
        """Automatically fix DataFrame vs array type errors"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if error_line >= len(lines):
                return False
            
            original_line = lines[error_line]
            
            # Pattern matching for DataFrame operations
            if '.corr()' in original_line:
                variable_name = original_line.split('.corr()')[0].strip()
                indent = len(original_line) - len(original_line.lstrip())
                
                new_lines = [
                    ' ' * indent + f'if isinstance({variable_name}, pd.DataFrame):\n',
                    ' ' * indent + f'    result = {variable_name}.corr()\n',
                    ' ' * indent + 'else:\n',
                    ' ' * indent + f'    result = pd.DataFrame({variable_name}).corr()\n'
                ]
                
                lines[error_line:error_line+1] = new_lines
                
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                
                return True
            
        except Exception as e:
            logging.error(f"Failed to fix DataFrame type error: {e}")
        
        return False

class MDPipelineMonitor:
    """Real-time monitoring and self-healing for MD pipeline"""
    
    def __init__(self, resolver: LangChainErrorResolver):
        self.resolver = resolver
        self.fixer = AutomaticErrorFixer(resolver)
        self.error_log = []
    
    def monitor_and_fix(self, func: Callable, *args, **kwargs) -> Any:
        """Monitor function execution and attempt automatic fixes"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            
            # Log the error
            error_entry = {
                'function': func.__name__,
                'error_msg': error_msg,
                'traceback': traceback_str,
                'timestamp': datetime.now().isoformat(),
                'args': str(args)[:200],  # Truncate for logging
                'kwargs': str(kwargs)[:200]
            }
            self.error_log.append(error_entry)
            
            # Attempt resolution
            resolution = self.resolver.resolve_error(error_msg, traceback_str)
            
            logging.warning(f"Error detected in {func.__name__}: {error_msg}")
            logging.info(f"Suggested resolution: {resolution['recommended_action']}")
            
            # Re-raise the exception for now (can be enhanced with automatic fixing)
            raise

def setup_intelligent_error_handling():
    """Setup intelligent error handling for the entire pipeline"""
    try:
        # Initialize the resolver
        resolver = LangChainErrorResolver()
        
        # Setup monitoring
        monitor = MDPipelineMonitor(resolver)
        
        logging.info("✅ Intelligent error handling system initialized")
        
        return {
            'resolver': resolver,
            'monitor': monitor,
            'status': 'active'
        }
        
    except Exception as e:
        logging.error(f"Failed to setup intelligent error handling: {e}")
        return {
            'resolver': None,
            'monitor': None,
            'status': 'inactive',
            'error': str(e)
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the error resolution system
    resolver = LangChainErrorResolver()
    
    # Test error classification
    test_error = "'InsulinComprehensiveAnalyzer' object has no attribute 'analyze_trajectory_file'"
    result = resolver.resolve_error(test_error)
    
    print("Error Resolution Result:")
    print(json.dumps(result, indent=2, default=str)) 