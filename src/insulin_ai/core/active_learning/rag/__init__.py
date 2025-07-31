#!/usr/bin/env python3
"""
RAG System Components for Active Learning - Phase 3

This package provides comprehensive RAG (Retrieval-Augmented Generation) 
capabilities for the active learning material discovery system.

Components:
- Vector Database: ChromaDB-based semantic search
- Property Database: SQLite-based material properties storage
- Similarity Matcher: Multi-metric material similarity analysis
- Improvement Analyzer: Structure-property relationship analysis
- Enhanced Web Search: Scientific literature and patent search
- Feedback Generator: Intelligent feedback and prompt generation

Author: AI-Driven Material Discovery Team
"""

# Core RAG components
from .vector_database import (
    VectorDatabase,
    MaterialDocument,
    SearchResult,
    ScientificEmbeddingModel,
    create_vector_database
)

from .property_database import (
    PropertyDatabase,
    Material,
    MaterialProperty,
    PropertyType,
    DataSource,
    PropertyQuery,
    BenchmarkData,
    create_property_database
)

from .similarity_matcher import (
    SimilarityMatcher,
    SimilarityResult,
    SimilarityMetric,
    SimilarityConfig,
    CompositionSimilarity,
    PropertySimilarity,
    StructureSimilarity,
    SemanticSimilarity,
    create_similarity_matcher
)

from .improvement_analyzer import (
    ImprovementAnalyzer,
    AnalysisResult,
    ImprovementSuggestion,
    ImprovementType,
    PropertyGap,
    StructurePropertyRelation,
    create_improvement_analyzer
)

from .enhanced_web_search import (
    EnhancedWebSearchAgent,
    WebSearchResult,
    SearchQuery,
    AggregatedResults,
    SearchDomain,
    QueryEnhancer,
    ScientificDatabaseSearcher,
    ResultProcessor,
    create_enhanced_web_search_agent
)

from .feedback_generator import (
    FeedbackGenerator,
    IterationFeedback,
    FeedbackItem,
    FeedbackType,
    FeedbackPriority,
    LiteratureInsightExtractor,
    BenchmarkComparator,
    PromptGenerator,
    create_feedback_generator
)

__all__ = [
    # Vector Database
    'VectorDatabase',
    'MaterialDocument', 
    'SearchResult',
    'ScientificEmbeddingModel',
    'create_vector_database',
    
    # Property Database
    'PropertyDatabase',
    'Material',
    'MaterialProperty',
    'PropertyType',
    'DataSource',
    'PropertyQuery', 
    'BenchmarkData',
    'create_property_database',
    
    # Similarity Matcher
    'SimilarityMatcher',
    'SimilarityResult',
    'SimilarityMetric',
    'SimilarityConfig',
    'CompositionSimilarity',
    'PropertySimilarity',
    'StructureSimilarity',
    'SemanticSimilarity',
    'create_similarity_matcher',
    
    # Improvement Analyzer
    'ImprovementAnalyzer',
    'AnalysisResult',
    'ImprovementSuggestion',
    'ImprovementType',
    'PropertyGap',
    'StructurePropertyRelation',
    'create_improvement_analyzer',
    
    # Enhanced Web Search
    'EnhancedWebSearchAgent',
    'WebSearchResult',
    'SearchQuery',
    'AggregatedResults',
    'SearchDomain',
    'QueryEnhancer',
    'ScientificDatabaseSearcher',
    'ResultProcessor',
    'create_enhanced_web_search_agent',
    
    # Feedback Generator
    'FeedbackGenerator',
    'IterationFeedback',
    'FeedbackItem',
    'FeedbackType',
    'FeedbackPriority',
    'LiteratureInsightExtractor',
    'BenchmarkComparator',
    'PromptGenerator',
    'create_feedback_generator'
]


# Convenience factory function to create complete RAG system
def create_complete_rag_system(config: dict = None) -> dict:
    """
    Create a complete RAG system with all components initialized.
    
    Args:
        config: Configuration dictionary with component settings
        
    Returns:
        Dictionary containing all initialized RAG components
    """
    config = config or {}
    
    # Initialize core databases
    vector_db = create_vector_database(config.get('vector_db', {}))
    property_db = create_property_database(config.get('property_db', {}).get('path', 'property_db.db'))
    
    # Initialize analysis components
    similarity_matcher = create_similarity_matcher(property_db, vector_db)
    improvement_analyzer = create_improvement_analyzer(property_db, similarity_matcher)
    
    # Initialize search agent
    web_search_agent = create_enhanced_web_search_agent(
        config.get('web_search', {}).get('cache_directory', 'search_cache')
    )
    
    # Initialize feedback generator
    feedback_generator = create_feedback_generator(
        vector_db, property_db, similarity_matcher, 
        improvement_analyzer, web_search_agent
    )
    
    return {
        'vector_db': vector_db,
        'property_db': property_db,
        'similarity_matcher': similarity_matcher,
        'improvement_analyzer': improvement_analyzer,
        'web_search_agent': web_search_agent,
        'feedback_generator': feedback_generator
    }


# Version information
__version__ = "1.0.0"
__author__ = "AI-Driven Material Discovery Team"
__description__ = "RAG System for Active Learning Material Discovery" 