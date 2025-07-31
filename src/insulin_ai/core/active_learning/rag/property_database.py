#!/usr/bin/env python3
"""
Property Database Implementation for RAG System - Phase 3

This module provides a database for storing and querying known material properties,
experimental data, and performance benchmarks for material discovery.

Author: AI-Driven Material Discovery Team
"""

import asyncio
import logging
import json
import sqlite3
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Database integration
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of material properties."""
    MECHANICAL = "mechanical"
    THERMAL = "thermal"
    TRANSPORT = "transport"
    CHEMICAL = "chemical"
    BIOLOGICAL = "biological"
    ELECTRICAL = "electrical"
    OPTICAL = "optical"


class DataSource(Enum):
    """Sources of property data."""
    EXPERIMENTAL = "experimental"
    COMPUTATIONAL = "computational"
    LITERATURE = "literature"
    DATABASE = "database"
    PREDICTED = "predicted"


@dataclass
class MaterialProperty:
    """Represents a single material property measurement."""
    material_id: str
    property_name: str
    property_type: PropertyType
    value: Union[float, int, str]
    unit: str
    uncertainty: Optional[float] = None
    conditions: Optional[Dict[str, Any]] = None
    source: DataSource = DataSource.EXPERIMENTAL
    reference: Optional[str] = None
    measurement_date: Optional[datetime] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Material:
    """Represents a material with its properties."""
    material_id: str
    name: str
    composition: Dict[str, Any]
    structure: Optional[str] = None
    synthesis_method: Optional[str] = None
    properties: List[MaterialProperty] = None
    created_date: datetime = None
    updated_date: datetime = None
    tags: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = []
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.updated_date is None:
            self.updated_date = datetime.now()
        if self.tags is None:
            self.tags = []


@dataclass
class PropertyQuery:
    """Query parameters for property search."""
    property_names: Optional[List[str]] = None
    property_types: Optional[List[PropertyType]] = None
    value_range: Optional[Tuple[float, float]] = None
    materials: Optional[List[str]] = None
    sources: Optional[List[DataSource]] = None
    min_confidence: float = 0.0
    date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class BenchmarkData:
    """Benchmark data for material property comparison."""
    property_name: str
    property_type: PropertyType
    percentiles: Dict[str, float]  # e.g., {"p25": 1.0, "p50": 2.0, "p75": 3.0}
    mean: float
    std: float
    count: int
    unit: str
    last_updated: datetime


class PropertyDatabase:
    """Database for storing and querying material properties."""
    
    def __init__(self, database_path: str = "property_database.db"):
        """
        Initialize property database.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self._initialize_database()
        
        # Cache for frequently accessed data
        self._material_cache = {}
        self._property_cache = {}
        self._benchmark_cache = {}
        
        logger.info(f"PropertyDatabase initialized at {database_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(self.database_path, timeout=10.0) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                
                cursor = conn.cursor()
                
                # Create materials table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS materials (
                        material_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        composition TEXT NOT NULL,
                        structure TEXT,
                        synthesis_method TEXT,
                        created_date TEXT NOT NULL,
                        updated_date TEXT NOT NULL,
                        tags TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create properties table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS properties (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        material_id TEXT NOT NULL,
                        property_name TEXT NOT NULL,
                        property_type TEXT NOT NULL,
                        value TEXT NOT NULL,
                        unit TEXT NOT NULL,
                        uncertainty REAL,
                        conditions TEXT,
                        source TEXT NOT NULL,
                        reference TEXT,
                        measurement_date TEXT,
                        confidence REAL DEFAULT 1.0,
                        metadata TEXT,
                        FOREIGN KEY (material_id) REFERENCES materials (material_id)
                    )
                """)
                
                # Create benchmarks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS benchmarks (
                        property_name TEXT PRIMARY KEY,
                        property_type TEXT NOT NULL,
                        percentiles TEXT NOT NULL,
                        mean REAL NOT NULL,
                        std REAL NOT NULL,
                        count INTEGER NOT NULL,
                        unit TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                """)
                
                # Create indices for faster queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_material_name ON materials(name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_name ON properties(property_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_type ON properties(property_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_material_property ON properties(material_id, property_name)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def add_material(self, material: Material) -> bool:
        """Add a material to the database."""
        try:
            # Use a separate connection for this operation
            conn = sqlite3.connect(self.database_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            
            try:
                cursor = conn.cursor()
                
                # Insert material
                cursor.execute("""
                    INSERT OR REPLACE INTO materials 
                    (material_id, name, composition, structure, synthesis_method, 
                     created_date, updated_date, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    material.material_id,
                    material.name,
                    json.dumps(material.composition),
                    material.structure,
                    material.synthesis_method,
                    material.created_date.isoformat(),
                    material.updated_date.isoformat(),
                    json.dumps(material.tags),
                    json.dumps(material.metadata) if material.metadata else None
                ))
                
                # Insert properties directly in same transaction
                for prop in material.properties:
                    cursor.execute("""
                        INSERT INTO properties 
                        (material_id, property_name, property_type, value, unit, uncertainty, 
                         conditions, source, reference, measurement_date, confidence, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prop.material_id,
                        prop.property_name,
                        prop.property_type.value,
                        str(prop.value),
                        prop.unit,
                        prop.uncertainty,
                        json.dumps(prop.conditions) if prop.conditions else None,
                        prop.source.value,
                        prop.reference,
                        prop.measurement_date.isoformat() if prop.measurement_date else None,
                        prop.confidence,
                        json.dumps(prop.metadata) if prop.metadata else None
                    ))
                
                conn.commit()
                
                # Update cache
                self._material_cache[material.material_id] = material
                
                logger.debug(f"Added material {material.material_id} to database")
                return True
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Failed to add material: {e}")
            return False
    
    async def add_property(self, property_data: MaterialProperty) -> bool:
        """Add a property measurement to the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO properties 
                    (material_id, property_name, property_type, value, unit, uncertainty, 
                     conditions, source, reference, measurement_date, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    property_data.material_id,
                    property_data.property_name,
                    property_data.property_type.value,
                    str(property_data.value),
                    property_data.unit,
                    property_data.uncertainty,
                    json.dumps(property_data.conditions) if property_data.conditions else None,
                    property_data.source.value,
                    property_data.reference,
                    property_data.measurement_date.isoformat() if property_data.measurement_date else None,
                    property_data.confidence,
                    json.dumps(property_data.metadata) if property_data.metadata else None
                ))
                
                conn.commit()
                logger.debug(f"Added property {property_data.property_name} for material {property_data.material_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add property: {e}")
            return False
    
    async def get_material(self, material_id: str) -> Optional[Material]:
        """Retrieve a material by ID."""
        # Check cache first
        if material_id in self._material_cache:
            return self._material_cache[material_id]
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get material data
                cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Get properties
                properties = await self.get_material_properties(material_id)
                
                # Create Material object
                material = Material(
                    material_id=row[0],
                    name=row[1],
                    composition=json.loads(row[2]),
                    structure=row[3],
                    synthesis_method=row[4],
                    created_date=datetime.fromisoformat(row[5]),
                    updated_date=datetime.fromisoformat(row[6]),
                    tags=json.loads(row[7]) if row[7] else [],
                    metadata=json.loads(row[8]) if row[8] else None,
                    properties=properties
                )
                
                # Cache the result
                self._material_cache[material_id] = material
                
                return material
                
        except Exception as e:
            logger.error(f"Failed to get material {material_id}: {e}")
            return None
    
    async def get_material_properties(self, material_id: str) -> List[MaterialProperty]:
        """Get all properties for a material."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM properties WHERE material_id = ?", (material_id,))
                rows = cursor.fetchall()
                
                properties = []
                for row in rows:
                    prop = MaterialProperty(
                        material_id=row[1],
                        property_name=row[2],
                        property_type=PropertyType(row[3]),
                        value=self._parse_value(row[4]),
                        unit=row[5],
                        uncertainty=row[6],
                        conditions=json.loads(row[7]) if row[7] else None,
                        source=DataSource(row[8]),
                        reference=row[9],
                        measurement_date=datetime.fromisoformat(row[10]) if row[10] else None,
                        confidence=row[11],
                        metadata=json.loads(row[12]) if row[12] else None
                    )
                    properties.append(prop)
                
                return properties
                
        except Exception as e:
            logger.error(f"Failed to get properties for material {material_id}: {e}")
            return []
    
    def _parse_value(self, value_str: str) -> Union[float, int, str]:
        """Parse property value from string."""
        try:
            # Try integer first
            if '.' not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            return value_str
    
    async def query_properties(self, query: PropertyQuery) -> List[MaterialProperty]:
        """Query properties based on criteria."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Build SQL query
                sql = "SELECT * FROM properties WHERE 1=1"
                params = []
                
                if query.property_names:
                    placeholders = ','.join(['?' for _ in query.property_names])
                    sql += f" AND property_name IN ({placeholders})"
                    params.extend(query.property_names)
                
                if query.property_types:
                    placeholders = ','.join(['?' for _ in query.property_types])
                    sql += f" AND property_type IN ({placeholders})"
                    params.extend([pt.value for pt in query.property_types])
                
                if query.materials:
                    placeholders = ','.join(['?' for _ in query.materials])
                    sql += f" AND material_id IN ({placeholders})"
                    params.extend(query.materials)
                
                if query.sources:
                    placeholders = ','.join(['?' for _ in query.sources])
                    sql += f" AND source IN ({placeholders})"
                    params.extend([s.value for s in query.sources])
                
                if query.min_confidence > 0:
                    sql += " AND confidence >= ?"
                    params.append(query.min_confidence)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # Convert to MaterialProperty objects
                properties = []
                for row in rows:
                    # Apply value range filter if specified
                    if query.value_range:
                        try:
                            value = float(row[4])
                            if not (query.value_range[0] <= value <= query.value_range[1]):
                                continue
                        except (ValueError, TypeError):
                            continue
                    
                    prop = MaterialProperty(
                        material_id=row[1],
                        property_name=row[2],
                        property_type=PropertyType(row[3]),
                        value=self._parse_value(row[4]),
                        unit=row[5],
                        uncertainty=row[6],
                        conditions=json.loads(row[7]) if row[7] else None,
                        source=DataSource(row[8]),
                        reference=row[9],
                        measurement_date=datetime.fromisoformat(row[10]) if row[10] else None,
                        confidence=row[11],
                        metadata=json.loads(row[12]) if row[12] else None
                    )
                    properties.append(prop)
                
                return properties
                
        except Exception as e:
            logger.error(f"Failed to query properties: {e}")
            return []
    
    async def calculate_benchmark(self, property_name: str) -> Optional[BenchmarkData]:
        """Calculate benchmark statistics for a property."""
        try:
            # Get all values for this property
            query = PropertyQuery(property_names=[property_name])
            properties = await self.query_properties(query)
            
            if not properties:
                return None
            
            # Extract numeric values
            values = []
            for prop in properties:
                try:
                    values.append(float(prop.value))
                except (ValueError, TypeError):
                    continue
            
            if not values:
                return None
            
            if NUMPY_AVAILABLE:
                values_array = np.array(values)
                
                benchmark = BenchmarkData(
                    property_name=property_name,
                    property_type=properties[0].property_type,
                    percentiles={
                        "p10": float(np.percentile(values_array, 10)),
                        "p25": float(np.percentile(values_array, 25)),
                        "p50": float(np.percentile(values_array, 50)),
                        "p75": float(np.percentile(values_array, 75)),
                        "p90": float(np.percentile(values_array, 90))
                    },
                    mean=float(np.mean(values_array)),
                    std=float(np.std(values_array)),
                    count=len(values),
                    unit=properties[0].unit,
                    last_updated=datetime.now()
                )
            else:
                # Manual calculation without numpy
                values.sort()
                n = len(values)
                
                benchmark = BenchmarkData(
                    property_name=property_name,
                    property_type=properties[0].property_type,
                    percentiles={
                        "p10": values[int(0.1 * n)],
                        "p25": values[int(0.25 * n)],
                        "p50": values[int(0.5 * n)],
                        "p75": values[int(0.75 * n)],
                        "p90": values[int(0.9 * n)]
                    },
                    mean=sum(values) / len(values),
                    std=(sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5,
                    count=len(values),
                    unit=properties[0].unit,
                    last_updated=datetime.now()
                )
            
            # Store benchmark
            await self._store_benchmark(benchmark)
            
            return benchmark
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark for {property_name}: {e}")
            return None
    
    async def _store_benchmark(self, benchmark: BenchmarkData) -> bool:
        """Store benchmark data in database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO benchmarks 
                    (property_name, property_type, percentiles, mean, std, count, unit, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    benchmark.property_name,
                    benchmark.property_type.value,
                    json.dumps(benchmark.percentiles),
                    benchmark.mean,
                    benchmark.std,
                    benchmark.count,
                    benchmark.unit,
                    benchmark.last_updated.isoformat()
                ))
                
                conn.commit()
                
                # Update cache
                self._benchmark_cache[benchmark.property_name] = benchmark
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store benchmark: {e}")
            return False
    
    async def get_benchmark(self, property_name: str) -> Optional[BenchmarkData]:
        """Get benchmark data for a property."""
        # Check cache first
        if property_name in self._benchmark_cache:
            return self._benchmark_cache[property_name]
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM benchmarks WHERE property_name = ?", (property_name,))
                row = cursor.fetchone()
                
                if not row:
                    # Calculate benchmark if not exists
                    return await self.calculate_benchmark(property_name)
                
                benchmark = BenchmarkData(
                    property_name=row[0],
                    property_type=PropertyType(row[1]),
                    percentiles=json.loads(row[2]),
                    mean=row[3],
                    std=row[4],
                    count=row[5],
                    unit=row[6],
                    last_updated=datetime.fromisoformat(row[7])
                )
                
                # Cache the result
                self._benchmark_cache[property_name] = benchmark
                
                return benchmark
                
        except Exception as e:
            logger.error(f"Failed to get benchmark for {property_name}: {e}")
            return None
    
    async def find_similar_materials(self, 
                                   target_properties: Dict[str, float],
                                   tolerance: float = 0.1,
                                   max_results: int = 10) -> List[Tuple[Material, float]]:
        """Find materials with similar properties."""
        try:
            similar_materials = []
            
            # Get all materials
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT material_id FROM materials")
                material_ids = [row[0] for row in cursor.fetchall()]
            
            for material_id in material_ids:
                material = await self.get_material(material_id)
                if not material:
                    continue
                
                # Calculate similarity score
                similarity_score = self._calculate_property_similarity(
                    material.properties, target_properties, tolerance
                )
                
                if similarity_score > 0:
                    similar_materials.append((material, similarity_score))
            
            # Sort by similarity score
            similar_materials.sort(key=lambda x: x[1], reverse=True)
            
            return similar_materials[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to find similar materials: {e}")
            return []
    
    def _calculate_property_similarity(self, 
                                     material_properties: List[MaterialProperty],
                                     target_properties: Dict[str, float],
                                     tolerance: float) -> float:
        """Calculate similarity score between material and target properties."""
        if not target_properties:
            return 0.0
        
        # Create property dictionary
        prop_dict = {}
        for prop in material_properties:
            try:
                prop_dict[prop.property_name] = float(prop.value)
            except (ValueError, TypeError):
                continue
        
        # Calculate similarity for each target property
        matches = 0
        total_properties = len(target_properties)
        
        for prop_name, target_value in target_properties.items():
            if prop_name in prop_dict:
                material_value = prop_dict[prop_name]
                relative_diff = abs(material_value - target_value) / max(abs(target_value), 1e-6)
                
                if relative_diff <= tolerance:
                    matches += 1
        
        return matches / total_properties if total_properties > 0 else 0.0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Count materials
                cursor.execute("SELECT COUNT(*) FROM materials")
                material_count = cursor.fetchone()[0]
                
                # Count properties
                cursor.execute("SELECT COUNT(*) FROM properties")
                property_count = cursor.fetchone()[0]
                
                # Count benchmarks
                cursor.execute("SELECT COUNT(*) FROM benchmarks")
                benchmark_count = cursor.fetchone()[0]
                
                # Property types distribution
                cursor.execute("SELECT property_type, COUNT(*) FROM properties GROUP BY property_type")
                property_type_dist = dict(cursor.fetchall())
                
                return {
                    "material_count": material_count,
                    "property_count": property_count,
                    "benchmark_count": benchmark_count,
                    "property_type_distribution": property_type_dist,
                    "cache_sizes": {
                        "materials": len(self._material_cache),
                        "properties": len(self._property_cache),
                        "benchmarks": len(self._benchmark_cache)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Factory function for easy initialization
def create_property_database(database_path: str = "property_database.db") -> PropertyDatabase:
    """Create property database with configuration."""
    return PropertyDatabase(database_path) 