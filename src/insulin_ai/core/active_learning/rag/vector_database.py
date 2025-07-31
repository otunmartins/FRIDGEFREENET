#!/usr/bin/env python3
"""
Vector Database Implementation for RAG System - Phase 3

This module provides vector database functionality using ChromaDB for semantic search
of scientific literature, material properties, and molecular structures.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available - install with: pip install chromadb")

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("Embeddings not available")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    LANGCHAIN_CHROMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_CHROMA_AVAILABLE = False
    logging.warning("LangChain Chroma integration not available")

# Scientific processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MaterialDocument:
    """Represents a material document in the vector database."""
    id: str
    content: str
    properties: Dict[str, Any]
    source: str
    timestamp: datetime
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Represents a search result from the vector database."""
    document: MaterialDocument
    similarity_score: float
    rank: int


class ScientificEmbeddingModel:
    """Wrapper for scientific literature-specific embedding models."""
    
    def __init__(self, model_type: str = "openai", model_name: Optional[str] = None):
        """
        Initialize scientific embedding model.
        
        Args:
            model_type: Type of embedding model ('openai', 'huggingface', 'scientific')
            model_name: Specific model name to use
        """
        self.model_type = model_type
        self.model_name = model_name
        self.embeddings = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("No embedding models available")
            return
        
        try:
            if self.model_type == "openai":
                # Check for API key before initializing
                import os
                if not os.getenv("OPENAI_API_KEY"):
                    logger.warning("OpenAI API key not found. Embeddings will not be available.")
                    self.embeddings = None
                    return
                
                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name or "text-embedding-3-small",
                    dimensions=1536  # Optimized for scientific content
                )
                logger.info(f"Initialized OpenAI embeddings: {self.model_name or 'text-embedding-3-small'}")
                
            elif self.model_type == "huggingface":
                # Use scientific literature-specific model
                model_name = self.model_name or "allenai/scibert-scivocab-uncased"
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # Can be changed to 'cuda' if available
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Initialized HuggingFace embeddings: {model_name}")
                
            elif self.model_type == "scientific":
                # Use specialized scientific embedding model
                model_name = self.model_name or "sentence-transformers/allenai-specter"
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Initialized Scientific embeddings: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embeddings = None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text."""
        if not self.embeddings:
            return None
        
        try:
            if hasattr(self.embeddings, 'embed_query'):
                return self.embeddings.embed_query(text)
            else:
                return self.embeddings.embed_documents([text])[0]
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not self.embeddings:
            return []
        
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            return []


class VectorDatabase:
    """Advanced vector database for scientific literature and material properties."""
    
    def __init__(self, 
                 storage_path: str = "vector_db",
                 embedding_model: str = "openai",
                 collection_name: str = "materials_science"):
        """
        Initialize vector database.
        
        Args:
            storage_path: Path to store vector database
            embedding_model: Type of embedding model to use
            collection_name: Name of the collection
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_model = ScientificEmbeddingModel(embedding_model)
        
        # Initialize ChromaDB
        self.client = None
        self.collection = None
        self.langchain_vectorstore = None
        
        self._initialize_database()
        
        # Cache for frequently accessed documents
        self._document_cache = {}
        self._similarity_cache = {}
        
        logger.info(f"VectorDatabase initialized at {storage_path}")
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available - vector search disabled")
            return
        
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Scientific literature and material properties"}
            )
            
            # Initialize LangChain vectorstore if available
            if LANGCHAIN_CHROMA_AVAILABLE and self.embedding_model.embeddings:
                self.langchain_vectorstore = Chroma(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model.embeddings,
                    persist_directory=str(self.storage_path)
                )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def _generate_document_id(self, content: str, source: str) -> str:
        """Generate unique document ID."""
        combined = f"{content}_{source}_{datetime.now().isoformat()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def add_document(self, document: MaterialDocument) -> bool:
        """Add a document to the vector database."""
        if not self.collection:
            logger.warning("Vector database not available")
            return False
        
        try:
            # Generate embedding if not provided
            if not document.embedding and self.embedding_model.embeddings:
                document.embedding = self.embedding_model.embed_text(document.content)
            
            if not document.embedding:
                logger.warning(f"Could not generate embedding for document {document.id}")
                return False
            
            # Prepare metadata
            metadata = {
                "source": document.source,
                "timestamp": document.timestamp.isoformat(),
                "properties": json.dumps(document.properties),
                **(document.metadata or {})
            }
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[document.embedding],
                documents=[document.content],
                metadatas=[metadata],
                ids=[document.id]
            )
            
            # Add to cache
            self._document_cache[document.id] = document
            
            logger.debug(f"Added document {document.id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector database: {e}")
            return False
    
    async def add_documents(self, documents: List[MaterialDocument]) -> int:
        """Add multiple documents to the vector database."""
        if not self.collection:
            logger.warning("Vector database not available")
            return 0
        
        success_count = 0
        batch_size = 100  # Process in batches
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings for batch
            contents = [doc.content for doc in batch]
            embeddings = self.embedding_model.embed_documents(contents)
            
            if not embeddings or len(embeddings) != len(batch):
                logger.warning(f"Failed to generate embeddings for batch {i//batch_size + 1}")
                continue
            
            try:
                # Prepare batch data
                ids = []
                batch_contents = []
                metadatas = []
                batch_embeddings = []
                
                for doc, embedding in zip(batch, embeddings):
                    if embedding:  # Only add documents with valid embeddings
                        doc.embedding = embedding
                        ids.append(doc.id)
                        batch_contents.append(doc.content)
                        metadatas.append({
                            "source": doc.source,
                            "timestamp": doc.timestamp.isoformat(),
                            "properties": json.dumps(doc.properties),
                            **(doc.metadata or {})
                        })
                        batch_embeddings.append(embedding)
                        
                        # Add to cache
                        self._document_cache[doc.id] = doc
                
                # Add batch to ChromaDB
                if ids:
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_contents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    success_count += len(ids)
                
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Successfully added {success_count}/{len(documents)} documents")
        return success_count
    
    async def semantic_search(self, 
                            query: str, 
                            n_results: int = 10,
                            filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if not self.collection:
            logger.warning("Vector database not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query)
            if not query_embedding:
                logger.warning("Could not generate embedding for query")
                return []
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if key == "source":
                        where_clause["source"] = {"$eq": value}
                    elif key == "property_range":
                        # Handle property range filtering
                        pass  # TODO: Implement property range filtering
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc_content, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (1 - normalized distance)
                    similarity_score = max(0, 1 - distance)
                    
                    # Parse metadata
                    properties = {}
                    if "properties" in metadata:
                        try:
                            properties = json.loads(metadata["properties"])
                        except:
                            pass
                    
                    # Create MaterialDocument
                    doc = MaterialDocument(
                        id=results["ids"][0][i],
                        content=doc_content,
                        properties=properties,
                        source=metadata.get("source", "unknown"),
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                        metadata=metadata
                    )
                    
                    search_results.append(SearchResult(
                        document=doc,
                        similarity_score=similarity_score,
                        rank=i + 1
                    ))
            
            logger.debug(f"Semantic search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def hybrid_search(self, 
                          query: str,
                          keywords: List[str] = None,
                          n_results: int = 10,
                          semantic_weight: float = 0.7) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search."""
        # Perform semantic search
        semantic_results = await self.semantic_search(query, n_results * 2)  # Get more for reranking
        
        # Perform keyword filtering if keywords provided
        if keywords:
            keyword_filtered = []
            for result in semantic_results:
                content_lower = result.document.content.lower()
                keyword_score = sum(1 for kw in keywords if kw.lower() in content_lower) / len(keywords)
                
                # Combine semantic and keyword scores
                combined_score = (semantic_weight * result.similarity_score + 
                                (1 - semantic_weight) * keyword_score)
                result.similarity_score = combined_score
                keyword_filtered.append(result)
            
            # Re-sort by combined score
            keyword_filtered.sort(key=lambda x: x.similarity_score, reverse=True)
            return keyword_filtered[:n_results]
        
        return semantic_results[:n_results]
    
    async def find_similar_materials(self, 
                                   material_properties: Dict[str, float],
                                   n_results: int = 5) -> List[SearchResult]:
        """Find materials with similar properties."""
        # Create a query based on material properties
        property_descriptions = []
        for prop, value in material_properties.items():
            property_descriptions.append(f"{prop}: {value}")
        
        query = f"Material with properties: {', '.join(property_descriptions)}"
        
        # Use property-based filtering if possible
        filters = {"property_range": material_properties}
        
        return await self.semantic_search(query, n_results, filters)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        if not self.collection:
            return {"error": "Vector database not available"}
        
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "storage_path": str(self.storage_path),
                "embedding_model": self.embedding_model.model_type,
                "cache_size": len(self._document_cache)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        if not self.collection:
            return False
        
        try:
            # Get all document IDs
            results = self.collection.get(include=[])
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            
            # Clear caches
            self._document_cache.clear()
            self._similarity_cache.clear()
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# Factory function for easy initialization
def create_vector_database(config: Dict[str, Any] = None) -> VectorDatabase:
    """Create vector database with configuration."""
    config = config or {}
    
    return VectorDatabase(
        storage_path=config.get("storage_path", "vector_db"),
        embedding_model=config.get("embedding_model", "openai"),
        collection_name=config.get("collection_name", "materials_science")
    ) 