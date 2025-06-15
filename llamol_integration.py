#!/usr/bin/env python3
"""
LlaSMol Integration Module

Integrates the LlaSMol chemistry-specialized language model from the LLM4Chem project
into our insulin AI chatbot system as an additional model option.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add LLM4Chem submodule to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLM4Chem'))

try:
    from LLM4Chem.generation import LlaSMolGeneration
    from LLM4Chem.config import BASE_MODELS
    LLAMOL_AVAILABLE = True
    print("✅ LlaSMol dependencies loaded successfully")
except ImportError as e:
    LLAMOL_AVAILABLE = False
    print(f"❌ LlaSMol dependencies not available: {e}")
    print("Please ensure the LLM4Chem submodule is properly initialized and dependencies are installed")


class LlaSMolChatWrapper:
    """
    Wrapper class to integrate LlaSMol models into our chatbot system.
    Provides a consistent interface compatible with our existing chatbot architecture.
    """
    
    def __init__(self, 
                 model_name: str = "osunlp/LlaSMol-Mistral-7B",
                 device: str = "auto"):
        """
        Initialize LlaSMol chat wrapper.
        
        Args:
            model_name (str): LlaSMol model name from HuggingFace
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        if not LLAMOL_AVAILABLE:
            raise ImportError("LlaSMol dependencies not available. Please install transformers, torch, and peft")
        
        self.model_name = model_name
        self.device = device
        
        # Available LlaSMol models
        self.available_models = list(BASE_MODELS.keys())
        
        if model_name not in self.available_models:
            print(f"⚠️ Model {model_name} not in predefined list. Available models: {self.available_models}")
            print("Proceeding anyway - might work if model exists on HuggingFace")
        
        try:
            print(f"🔬 Loading LlaSMol model: {model_name}")
            self.generator = LlaSMolGeneration(model_name=model_name, device=device)
            print(f"✅ LlaSMol model {model_name} loaded successfully!")
            
            # Test the model with a simple query
            test_result = self.generator.generate("What is the SMILES for water?", print_out=False)
            print(f"🧪 Model test successful")
            
        except Exception as e:
            print(f"❌ Failed to initialize LlaSMol model {model_name}: {e}")
            print("This might be due to:")
            print("1. Model not downloaded/available")
            print("2. Insufficient GPU memory")
            print("3. Missing dependencies (transformers, torch, peft)")
            print("4. Network issues downloading the model")
            raise
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Generate response using LlaSMol model.
        Compatible with LangChain LLM interface.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        try:
            # Extract generation parameters
            max_new_tokens = kwargs.get('max_new_tokens', 1024)
            temperature = kwargs.get('temperature', 0.1)
            canonicalize_smiles = kwargs.get('canonicalize_smiles', True)
            
            # Generate response using LlaSMol
            results = self.generator.generate(
                input_text=prompt,
                max_new_tokens=max_new_tokens,
                canonicalize_smiles=canonicalize_smiles,
                print_out=False,
                temperature=temperature
            )
            
            if results and len(results) > 0:
                # Extract the generated text from the first result
                result = results[0]
                if isinstance(result, dict) and 'output' in result:
                    output = result['output']
                    if output and len(output) > 0:
                        return output[0] if isinstance(output, list) else str(output)
                else:
                    return str(result)
            
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            error_msg = f"LlaSMol generation error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I encountered an error while processing your request: {error_msg}"
    
    def generate_chemistry_response(self, 
                                  query: str, 
                                  chemistry_task: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate chemistry-specific response with proper formatting.
        
        Args:
            query (str): Chemistry query
            chemistry_task (str): Specific chemistry task type
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Add chemistry-specific context if task is specified
            if chemistry_task:
                formatted_query = self._format_chemistry_query(query, chemistry_task)
            else:
                formatted_query = query
            
            # Generate response
            response = self.invoke(formatted_query)
            
            # Parse chemistry-specific tags from response
            parsed_response = self._parse_chemistry_response(response)
            
            return {
                'success': True,
                'query': query,
                'response': response,
                'parsed_response': parsed_response,
                'chemistry_task': chemistry_task,
                'model': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'model': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_chemistry_query(self, query: str, task: str) -> str:
        """Format query for specific chemistry tasks."""
        # Task-specific query formatting based on LlaSMol examples
        task_formats = {
            'name_conversion': query,  # Usually already properly formatted
            'property_prediction': query,
            'molecule_captioning': f"Describe this molecule: {query}",
            'molecule_generation': f"Generate a molecule that {query}",
            'forward_synthesis': f"Based on the reactants {query}, suggest a possible product.",
            'retrosynthesis': f"Identify possible reactants for the product {query}",
        }
        
        return task_formats.get(task, query)
    
    def _parse_chemistry_response(self, response: str) -> Dict[str, Any]:
        """Parse chemistry-specific tags from LlaSMol response."""
        import re
        
        parsed = {}
        
        # Define tag patterns from LlaSMol
        tag_patterns = {
            'smiles': r'<SMILES>(.*?)</SMILES>',
            'iupac': r'<IUPAC>(.*?)</IUPAC>',
            'molformula': r'<MOLFORMULA>(.*?)</MOLFORMULA>',
            'number': r'<NUMBER>(.*?)</NUMBER>',
            'boolean': r'<BOOLEAN>(.*?)</BOOLEAN>'
        }
        
        # Extract tagged content
        for tag, pattern in tag_patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                parsed[tag] = [match.strip() for match in matches]
        
        return parsed
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available LlaSMol models."""
        if not LLAMOL_AVAILABLE:
            return []
        return list(BASE_MODELS.keys())
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if LlaSMol is available."""
        return LLAMOL_AVAILABLE
    
    def test_connection(self) -> str:
        """Test the LlaSMol model connection."""
        try:
            test_query = "What is the SMILES representation for water?"
            response = self.invoke(test_query)
            return f"✅ LlaSMol connection successful. Test response: {response[:100]}..."
        except Exception as e:
            return f"❌ LlaSMol connection failed: {e}"


class LlaSMolModelManager:
    """
    Manager class for handling multiple LlaSMol models and model selection.
    """
    
    def __init__(self):
        self.loaded_models = {}  # model_name -> LlaSMolChatWrapper
        self.default_model = "osunlp/LlaSMol-Mistral-7B"
    
    def load_model(self, model_name: str, device: str = "auto") -> bool:
        """
        Load a LlaSMol model.
        
        Args:
            model_name (str): Model name to load
            device (str): Device to use
            
        Returns:
            bool: Success status
        """
        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded")
            return True
        
        try:
            wrapper = LlaSMolChatWrapper(model_name=model_name, device=device)
            self.loaded_models[model_name] = wrapper
            print(f"✅ Loaded LlaSMol model: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load LlaSMol model {model_name}: {e}")
            return False
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[LlaSMolChatWrapper]:
        """
        Get a loaded model or load the default.
        
        Args:
            model_name (str): Specific model name, or None for default
            
        Returns:
            LlaSMolChatWrapper: Model wrapper or None if failed
        """
        target_model = model_name or self.default_model
        
        if target_model not in self.loaded_models:
            if not self.load_model(target_model):
                return None
        
        return self.loaded_models[target_model]
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free memory.
        
        Args:
            model_name (str): Model to unload
            
        Returns:
            bool: Success status
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            print(f"🗑️ Unloaded model: {model_name}")
            return True
        return False


# Global model manager instance
llamol_manager = LlaSMolModelManager()


def test_llamol_integration():
    """Test function for LlaSMol integration."""
    if not LLAMOL_AVAILABLE:
        print("❌ LlaSMol not available for testing")
        return False
    
    try:
        print("🧪 Testing LlaSMol integration...")
        
        # Test model loading
        model = llamol_manager.get_model()
        if not model:
            print("❌ Failed to load default model")
            return False
        
        # Test basic generation
        test_queries = [
            "What is the SMILES for water?",
            "What is the molecular formula of <SMILES>CCO</SMILES>?",
            "Describe the molecule <SMILES>CC(=O)OC1=CC=CC=C1C(=O)O</SMILES>"
        ]
        
        for query in test_queries:
            print(f"\nTesting query: {query}")
            result = model.generate_chemistry_response(query)
            if result['success']:
                print(f"✅ Response: {result['response'][:100]}...")
                if result['parsed_response']:
                    print(f"📋 Parsed: {result['parsed_response']}")
            else:
                print(f"❌ Failed: {result['error']}")
        
        print("\n✅ LlaSMol integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ LlaSMol integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_llamol_integration() 