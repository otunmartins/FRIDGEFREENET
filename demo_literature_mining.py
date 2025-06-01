#!/usr/bin/env python3
"""
Demo script for the Materials Literature Mining System - Milestone 1
Basic LLM-guided literature mining for insulin delivery materials.
"""

import os
from literature_mining_system import MaterialsLiteratureMiner


def demo_basic_mining():
    """Demonstrate basic literature mining functionality."""
    print("=" * 80)
    print("MILESTONE 1: BASIC LITERATURE MINING")
    print("=" * 80)
    
    # Initialize the mining system
    miner = MaterialsLiteratureMiner()
    
    # Run basic literature mining
    print("🔍 Starting basic literature search for insulin delivery materials...")
    results = miner.mine_insulin_delivery_materials(
        max_papers=30,
        recent_only=True,
        save_results=True
    )
    
    print(f"\n📊 MINING RESULTS:")
    print(f"   - Papers analyzed: {results.get('papers_analyzed', 0)}")
    print(f"   - Material candidates found: {len(results.get('material_candidates', []))}")
    print(f"   - Search queries used: {len(results.get('search_queries', []))}")
    
    # Display search queries used
    queries = results.get('search_queries', [])
    if queries:
        print(f"\n🔍 SEARCH QUERIES USED:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
    
    # Display some candidates
    candidates = results.get('material_candidates', [])
    if candidates:
        print(f"\n🧪 MATERIAL CANDIDATES FOUND:")
        for i, material in enumerate(candidates[:5], 1):  # Show top 5
            name = material.get('material_name', 'Unknown')
            composition = material.get('material_composition', 'Not specified')[:100]
            mechanism = material.get('stabilization_mechanism', 'Not specified')[:100]
            confidence = material.get('confidence_score', 'N/A')
            
            print(f"   {i}. {name}")
            print(f"      Composition: {composition}...")
            print(f"      Stabilization: {mechanism}...")
            print(f"      Confidence: {confidence}")
            print()
    
    return results


def demo_material_analysis():
    """Demonstrate detailed material analysis."""
    print("=" * 80)
    print("DETAILED MATERIAL ANALYSIS")
    print("=" * 80)
    
    miner = MaterialsLiteratureMiner()
    
    # Analyze specific materials
    materials_to_analyze = ["chitosan", "PEG hydrogel", "alginate"]
    
    for material_name in materials_to_analyze:
        print(f"\n🔬 Analyzing: {material_name}")
        print("-" * 40)
        
        details = miner.get_material_details(material_name)
        
        if 'error' not in details:
            print(f"   - Source papers: {details.get('source_papers', 0)}")
            
            analysis = details.get('detailed_analysis', '')
            if analysis:
                # Display first 300 characters of analysis
                print(f"\n   📝 ANALYSIS SUMMARY:")
                print(f"   {analysis[:300]}...")
            
            papers = details.get('papers', [])
            if papers:
                print(f"\n   📚 KEY REFERENCE PAPERS:")
                for i, paper in enumerate(papers[:2], 1):
                    title = paper.get('title', 'Unknown')[:50]
                    year = paper.get('year', 'Unknown')
                    print(f"   {i}. {title}... ({year})")
        else:
            print(f"   ❌ Error: {details['error']}")
        
        print()
    
    return details


def demo_custom_search():
    """Demonstrate searching with custom parameters."""
    print("=" * 80)
    print("CUSTOM SEARCH PARAMETERS")
    print("=" * 80)
    
    miner = MaterialsLiteratureMiner()
    
    # Test different search parameters
    print("🔍 Testing different search configurations...")
    
    # Small focused search
    print("\n1️⃣ Small focused search (10 papers, recent only):")
    results_small = miner.mine_insulin_delivery_materials(
        max_papers=10,
        recent_only=True,
        save_results=False
    )
    print(f"   Found {len(results_small.get('material_candidates', []))} candidates")
    
    # Broader search including older papers
    print("\n2️⃣ Broader search (20 papers, all years):")
    results_broad = miner.mine_insulin_delivery_materials(
        max_papers=20,
        recent_only=False,
        save_results=False
    )
    print(f"   Found {len(results_broad.get('material_candidates', []))} candidates")
    
    return results_small, results_broad


def main():
    """Run the Milestone 1 demo - basic literature mining functionality."""
    
    print("🚀 MILESTONE 1: MATERIALS LITERATURE MINING SYSTEM")
    print("Basic LLM-guided literature mining for insulin delivery patches")
    print("=" * 80)
    
    try:
        # Demo 1: Basic mining functionality
        print("\n1️⃣ Running Basic Literature Mining...")
        basic_results = demo_basic_mining()
        
        print("\n\n2️⃣ Running Detailed Material Analysis...")
        demo_material_analysis()
        
        print("\n\n3️⃣ Running Custom Search Parameters...")
        demo_custom_search()
        
        print("\n\n✅ MILESTONE 1 DEMO COMPLETE!")
        print("Check the 'mining_results/' directory for saved results.")
        
        print("\n📋 MILESTONE 1 DELIVERABLES:")
        print("✅ Functional LLM system for literature mining")
        print("✅ Material database with structured extraction")
        print("✅ Documentation and methodology")
        
        print("\n🎯 NEXT MILESTONES:")
        print("   Milestone 2: Generative Model Integration (Week 2)")
        print("   Milestone 3: UMA-ASE MD Simulation Pipeline (Week 3)")
        print("   Milestone 4: Complete Active Learning Framework (Week 4)")
        
        # Show summary statistics
        if 'material_candidates' in basic_results:
            total_candidates = len(basic_results['material_candidates'])
            print(f"\n📊 SESSION SUMMARY:")
            print(f"   - Total material candidates identified: {total_candidates}")
            print(f"   - Papers analyzed: {basic_results.get('papers_analyzed', 0)}")
            print(f"   - Search queries executed: {len(basic_results.get('search_queries', []))}")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure OLLAMA is running: ollama serve")
        print("2. Check if the required model is available: ollama list")
        print("3. Verify internet connection for Semantic Scholar API")
        print("4. Check that all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 