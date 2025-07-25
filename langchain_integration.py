#!/usr/bin/env python3

"""
LangChain Integration Module for Insulin AI App

Integrates the robust LangChain-based PSMILES generation agent 
with the existing Streamlit application workflow.

This module provides:
- Seamless integration between old and new generation methods
- Fallback mechanisms for reliability
- Performance comparison tools
- User-selectable generation strategies

Author: AI Agent  
Date: 2024
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import existing systems
try:
    from langchain_psmiles_agent import (
        RobustPSMILESAgent, 
        PSMILESStructure, 
        ValidationLevel,
        PSMILESValidationError
    )
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️ LangChain agent not available: {e}")

# Set up logging
logger = logging.getLogger(__name__)


class GenerationStrategy(Enum):
    """Available PSMILES generation strategies."""
    TRADITIONAL = "traditional"           # Original pipeline
    LANGCHAIN_ROBUST = "langchain_robust" # LangChain ReAct agent
    HYBRID = "hybrid"                     # Try LangChain first, fallback to traditional
    COMPARATIVE = "comparative"           # Generate with both and compare


@dataclass 
class GenerationResult:
    """Result from PSMILES generation with metadata."""
    psmiles: str
    success: bool
    method: str
    confidence: float
    generation_time: float
    validation_errors: List[str]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


class LangChainPSMILESIntegration:
    """
    Integration layer between LangChain agents and existing workflow.
    
    Provides unified interface for PSMILES generation with multiple strategies.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4",
                 validation_level: ValidationLevel = ValidationLevel.CHEMICAL,
                 enable_fallback: bool = True):
        """
        Initialize the LangChain integration.
        
        Args:
            api_key: OpenAI API key (if not in environment)
            model_name: LLM model to use for generation
            validation_level: Chemical validation strictness
            enable_fallback: Enable fallback to traditional methods
        """
        self.model_name = model_name
        self.validation_level = validation_level
        self.enable_fallback = enable_fallback
        
        # Set up API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize LangChain agent if available
        self.langchain_agent = None
        self.langchain_available = LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY")
        
        if self.langchain_available:
            try:
                self.langchain_agent = RobustPSMILESAgent(
                    model_name=model_name,
                    temperature=0.1,
                    validation_level=validation_level
                )
                logger.info("✅ LangChain agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LangChain agent: {e}")
                self.langchain_available = False
        
        # Import traditional systems
        self.traditional_processor = None
        try:
            from psmiles_processor import PSMILESProcessor
            self.traditional_processor = PSMILESProcessor()
            logger.info("✅ Traditional processor available")
        except ImportError as e:
            logger.error(f"❌ Traditional processor not available: {e}")
    
    def is_available(self, strategy: GenerationStrategy) -> bool:
        """Check if a generation strategy is available."""
        if strategy == GenerationStrategy.TRADITIONAL:
            return self.traditional_processor is not None
        
        elif strategy in [GenerationStrategy.LANGCHAIN_ROBUST, GenerationStrategy.HYBRID]:
            return self.langchain_available
        
        elif strategy == GenerationStrategy.COMPARATIVE:
            return self.langchain_available and self.traditional_processor is not None
        
        return False
    
    def generate_psmiles(self, 
                        material_request: str,
                        strategy: GenerationStrategy = GenerationStrategy.HYBRID,
                        context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """
        Generate PSMILES using specified strategy.
        
        Args:
            material_request: Description of desired material
            strategy: Generation strategy to use
            context: Additional context for generation
            
        Returns:
            GenerationResult: Generation result with metadata
        """
        start_time = time.time()
        
        try:
            if strategy == GenerationStrategy.LANGCHAIN_ROBUST:
                return self._generate_with_langchain(material_request, context, start_time)
            
            elif strategy == GenerationStrategy.TRADITIONAL:
                return self._generate_with_traditional(material_request, context, start_time)
            
            elif strategy == GenerationStrategy.HYBRID:
                return self._generate_hybrid(material_request, context, start_time)
            
            elif strategy == GenerationStrategy.COMPARATIVE:
                return self._generate_comparative(material_request, context, start_time)
            
            else:
                raise ValueError(f"Unknown generation strategy: {strategy}")
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Generation failed with strategy {strategy}: {e}")
            
            return GenerationResult(
                psmiles="[*]ERROR[*]",
                success=False,
                method=strategy.value,
                confidence=0.0,
                generation_time=generation_time,
                validation_errors=[str(e)],
                properties={},
                metadata={"error": str(e)}
            )
    
    def _generate_with_langchain(self, 
                               material_request: str, 
                               context: Optional[Dict[str, Any]], 
                               start_time: float) -> GenerationResult:
        """Generate using LangChain ReAct agent."""
        if not self.langchain_available:
            raise RuntimeError("LangChain agent not available")
        
        try:
            # Generate with LangChain agent
            structure = self.langchain_agent.generate_psmiles(material_request, context)
            generation_time = time.time() - start_time
            
            return GenerationResult(
                psmiles=structure.psmiles,
                success=True,
                method="langchain_robust",
                confidence=structure.confidence,
                generation_time=generation_time,
                validation_errors=[],
                properties={
                    "molecular_weight": structure.molecular_weight,
                    "functional_groups": structure.functional_groups,
                    "validation_level": structure.validation_level.value
                },
                metadata={
                    "generation_method": structure.generation_method,
                    "llm_model": self.model_name
                }
            )
            
        except PSMILESValidationError as e:
            generation_time = time.time() - start_time
            return GenerationResult(
                psmiles="[*]ERROR[*]",
                success=False,
                method="langchain_robust",
                confidence=0.0,
                generation_time=generation_time,
                validation_errors=[str(e)],
                properties={},
                metadata={"validation_error": str(e)}
            )
    
    def _generate_with_traditional(self, 
                                 material_request: str, 
                                 context: Optional[Dict[str, Any]], 
                                 start_time: float) -> GenerationResult:
        """Generate using traditional PSMILES generation."""
        if not self.traditional_processor:
            raise RuntimeError("Traditional processor not available")
        
        try:
            # Use existing psmiles_generation_with_llm function
            from insulin_ai_app import psmiles_generation_with_llm
            
            result = psmiles_generation_with_llm(material_request)
            generation_time = time.time() - start_time
            
            if result and result.get('success'):
                return GenerationResult(
                    psmiles=result['psmiles'],
                    success=True,
                    method="traditional",
                    confidence=0.7,  # Default confidence
                    generation_time=generation_time,
                    validation_errors=[],
                    properties=result.get('properties', {}),
                    metadata=result.get('metadata', {})
                )
            else:
                return GenerationResult(
                    psmiles="[*]ERROR[*]",
                    success=False,
                    method="traditional",
                    confidence=0.0,
                    generation_time=generation_time,
                    validation_errors=[result.get('error', 'Unknown error')],
                    properties={},
                    metadata={}
                )
                
        except Exception as e:
            generation_time = time.time() - start_time
            return GenerationResult(
                psmiles="[*]ERROR[*]",
                success=False,
                method="traditional",
                confidence=0.0,
                generation_time=generation_time,
                validation_errors=[str(e)],
                properties={},
                metadata={"error": str(e)}
            )
    
    def _generate_hybrid(self, 
                        material_request: str, 
                        context: Optional[Dict[str, Any]], 
                        start_time: float) -> GenerationResult:
        """Hybrid generation: try LangChain first, fallback to traditional."""
        
        # Try LangChain first
        if self.langchain_available:
            try:
                logger.info("🔬 Attempting LangChain generation...")
                result = self._generate_with_langchain(material_request, context, start_time)
                
                if result.success:
                    result.method = "hybrid_langchain"
                    logger.info("✅ LangChain generation successful")
                    return result
                else:
                    logger.warning("⚠️ LangChain generation failed, trying fallback...")
            except Exception as e:
                logger.warning(f"⚠️ LangChain failed: {e}, trying fallback...")
        
        # Fallback to traditional
        if self.enable_fallback and self.traditional_processor:
            logger.info("🔄 Falling back to traditional generation...")
            result = self._generate_with_traditional(material_request, context, start_time)
            result.method = "hybrid_traditional"
            result.metadata["fallback_used"] = True
            return result
        
        # Both failed
        generation_time = time.time() - start_time
        return GenerationResult(
            psmiles="[*]ERROR[*]",
            success=False,
            method="hybrid_failed",
            confidence=0.0,
            generation_time=generation_time,
            validation_errors=["Both LangChain and traditional methods failed"],
            properties={},
            metadata={"hybrid_failure": True}
        )
    
    def _generate_comparative(self, 
                            material_request: str, 
                            context: Optional[Dict[str, Any]], 
                            start_time: float) -> GenerationResult:
        """Generate with both methods and compare results."""
        
        results = {}
        
        # Try LangChain
        if self.langchain_available:
            try:
                langchain_result = self._generate_with_langchain(material_request, context, start_time)
                results["langchain"] = langchain_result
            except Exception as e:
                logger.error(f"LangChain failed in comparative: {e}")
        
        # Try traditional  
        if self.traditional_processor:
            try:
                traditional_result = self._generate_with_traditional(material_request, context, start_time)
                results["traditional"] = traditional_result
            except Exception as e:
                logger.error(f"Traditional failed in comparative: {e}")
        
        # Select best result
        if not results:
            generation_time = time.time() - start_time
            return GenerationResult(
                psmiles="[*]ERROR[*]",
                success=False,
                method="comparative_failed",
                confidence=0.0,
                generation_time=generation_time,
                validation_errors=["Both methods failed"],
                properties={},
                metadata={}
            )
        
        # Prefer successful results, then higher confidence
        best_result = None
        for method, result in results.items():
            if result.success:
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
        
        # If no successful results, take the first one
        if best_result is None:
            best_result = list(results.values())[0]
        
        # Add comparative metadata
        best_result.method = "comparative"
        best_result.metadata["comparison_results"] = {
            method: {
                "success": result.success,
                "confidence": result.confidence,
                "psmiles": result.psmiles
            }
            for method, result in results.items()
        }
        
        return best_result
    
    def batch_generate(self, 
                      requests: List[str],
                      strategy: GenerationStrategy = GenerationStrategy.HYBRID,
                      context: Optional[Dict[str, Any]] = None) -> List[GenerationResult]:
        """Generate multiple PSMILES structures in batch."""
        results = []
        
        for i, request in enumerate(requests):
            logger.info(f"🔄 Generating {i+1}/{len(requests)}: {request[:50]}...")
            result = self.generate_psmiles(request, strategy, context)
            results.append(result)
        
        return results
    
    def get_generation_statistics(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """Calculate statistics from generation results."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        
        # Method breakdown
        methods = {}
        for result in results:
            method = result.method
            if method not in methods:
                methods[method] = {"count": 0, "success": 0}
            methods[method]["count"] += 1
            if result.success:
                methods[method]["success"] += 1
        
        # Calculate success rates
        for method in methods:
            methods[method]["success_rate"] = methods[method]["success"] / methods[method]["count"]
        
        # Time statistics
        times = [r.generation_time for r in results]
        avg_time = sum(times) / len(times) if times else 0
        
        # Confidence statistics
        confidences = [r.confidence for r in results if r.success]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_requests": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_time": avg_time,
            "average_confidence": avg_confidence,
            "method_breakdown": methods,
            "failed_requests": total - successful
        }


def create_streamlit_integration() -> LangChainPSMILESIntegration:
    """Create LangChain integration for Streamlit app."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("⚠️ OPENAI_API_KEY not found - LangChain features will be limited")
    
    # Create integration with default settings
    integration = LangChainPSMILESIntegration(
        api_key=api_key,
        model_name="gpt-4",  # Use best model for production
        validation_level=ValidationLevel.CHEMICAL,
        enable_fallback=True
    )
    
    return integration


# Test function
def test_langchain_integration():
    """Test the LangChain integration."""
    
    print("🧪 Testing LangChain Integration")
    print("=" * 50)
    
    integration = create_streamlit_integration()
    
    # Test availability
    print("📋 Strategy Availability:")
    for strategy in GenerationStrategy:
        available = integration.is_available(strategy)
        status = "✅" if available else "❌"
        print(f"  {status} {strategy.value}")
    
    # Test generation if available
    test_request = "biodegradable polymer with carbonyl groups"
    
    if integration.is_available(GenerationStrategy.HYBRID):
        print(f"\n🎯 Testing Generation: {test_request}")
        result = integration.generate_psmiles(test_request, GenerationStrategy.HYBRID)
        
        print(f"✅ Result:")
        print(f"  Success: {result.success}")
        print(f"  Method: {result.method}")
        print(f"  PSMILES: {result.psmiles}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Time: {result.generation_time:.2f}s")
    else:
        print("⚠️ No generation strategies available for testing")


if __name__ == "__main__":
    test_langchain_integration() 