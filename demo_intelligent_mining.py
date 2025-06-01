#!/usr/bin/env python3
"""
Demo script for Intelligent Literature Mining System
Demonstrates how user requests are translated into targeted literature searches.
"""

import json
from literature_mining_system import MaterialsLiteratureMiner

def demo_intelligent_queries():
    """
    Demonstrate intelligent mining with various user requests.
    """
    print("🧠 INTELLIGENT LITERATURE MINING DEMO")
    print("=" * 60)
    
    # Initialize the mining system
    miner = MaterialsLiteratureMiner()
    
    # Test queries that demonstrate different scenarios
    test_requests = [
        {
            "request": "smart polymers that respond to temperature",
            "description": "User interested in temperature-responsive materials"
        },
        {
            "request": "nanotechnology for drug delivery",
            "description": "User interested in nanocarriers and nanomaterials"
        },
        {
            "request": "green chemistry and sustainable materials",
            "description": "User interested in eco-friendly biocompatible materials"
        },
        {
            "request": "pain relief patches",
            "description": "Related but different application - should adapt to insulin delivery"
        },
        {
            "request": "3D printing materials for medical devices",
            "description": "Manufacturing-focused but relevant to biomedical applications"
        },
        {
            "request": "machine learning for protein folding",
            "description": "AI/computational focus - should connect to protein stabilization"
        }
    ]
    
    for i, test_case in enumerate(test_requests, 1):
        print(f"\n🔍 TEST CASE {i}: {test_case['description']}")
        print(f"📝 Request: '{test_case['request']}'")
        print("-" * 50)
        
        try:
            # Run intelligent mining with limited papers for demo
            results = miner.intelligent_mining(
                user_request=test_case['request'],
                max_papers=15,  # Small number for demo
                save_results=False  # Don't save demo results
            )
            
            if 'error' in results:
                print(f"❌ {results['error']}")
                if 'suggestion' in results:
                    print(f"💡 Suggestion: {results['suggestion']}")
            else:
                # Display the search strategy
                strategy = results.get('search_strategy', {})
                print(f"🎯 Relevance Score: {strategy.get('relevance_score', 'N/A')}/10")
                print(f"🧠 Interpretation: {strategy.get('interpretation', 'N/A')}")
                
                print("\n📚 Generated Search Queries:")
                for j, query in enumerate(strategy.get('search_queries', []), 1):
                    print(f"  {j}. {query}")
                
                print(f"\n🔬 Extraction Focus: {strategy.get('extraction_focus', 'General')}")
                print(f"📊 Papers Analyzed: {results.get('papers_analyzed', 0)}")
                print(f"🎯 Materials Found: {len(results.get('material_candidates', []))}")
                
                # Show first material candidate if available
                candidates = results.get('material_candidates', [])
                if candidates:
                    first_material = candidates[0]
                    print(f"\n🧪 Example Material: {first_material.get('material_name', 'Unknown')}")
                    print(f"   Composition: {first_material.get('material_composition', 'Not specified')[:100]}...")
                    print(f"   Confidence: {first_material.get('confidence_score', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error during mining: {e}")
        
        print("\n" + "=" * 60)

def demo_off_topic_request():
    """
    Demonstrate how the system handles off-topic requests.
    """
    print("\n🚫 OFF-TOPIC REQUEST HANDLING DEMO")
    print("=" * 60)
    
    miner = MaterialsLiteratureMiner()
    
    off_topic_requests = [
        "best pizza recipes in Italy",
        "how to train a dog",
        "stock market analysis",
        "cooking with organic ingredients"  # This might find a connection to food-safe materials
    ]
    
    for request in off_topic_requests:
        print(f"\n📝 Testing: '{request}'")
        print("-" * 30)
        
        try:
            results = miner.intelligent_mining(
                user_request=request,
                max_papers=5,
                save_results=False
            )
            
            if 'error' in results:
                print(f"❌ {results['error']}")
                if 'suggestion' in results:
                    print(f"💡 {results['suggestion']}")
            else:
                strategy = results.get('search_strategy', {})
                print(f"✅ Found connection! Score: {strategy.get('relevance_score', 'N/A')}/10")
                print(f"🧠 {strategy.get('interpretation', 'N/A')}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_focused_extraction():
    """
    Demonstrate how extraction focus changes based on user request.
    """
    print("\n🎯 FOCUSED EXTRACTION DEMO")
    print("=" * 60)
    
    miner = MaterialsLiteratureMiner()
    
    # Test with a specific focus
    request = "biodegradable polymers for medical implants"
    print(f"📝 Request: '{request}'")
    print("This should focus on biodegradability and biocompatibility aspects")
    print("-" * 50)
    
    try:
        results = miner.intelligent_mining(
            user_request=request,
            max_papers=10,
            save_results=False
        )
        
        if 'error' not in results:
            strategy = results.get('search_strategy', {})
            print(f"🔬 Extraction Focus: {strategy.get('extraction_focus', 'N/A')}")
            
            # Show how this affects the materials found
            candidates = results.get('material_candidates', [])
            print(f"\n📊 Found {len(candidates)} materials with biodegradability focus")
            
            for i, material in enumerate(candidates[:3], 1):  # Show first 3
                print(f"\n{i}. {material.get('material_name', 'Unknown')}")
                print(f"   Focus Relevance: {material.get('biocompatibility_data', 'Not specified')[:80]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all demos."""
    print("🚀 INTELLIGENT LITERATURE MINING SYSTEM DEMO")
    print("This demo shows how user requests are intelligently translated")
    print("into targeted literature searches for insulin delivery materials.")
    print("\n" + "=" * 80)
    
    try:
        # Run the main intelligent query demo
        demo_intelligent_queries()
        
        # Demo off-topic handling
        demo_off_topic_request()
        
        # Demo focused extraction
        demo_focused_extraction()
        
        print("\n🎉 DEMO COMPLETED!")
        print("The system successfully demonstrates:")
        print("✅ Intelligent interpretation of user requests")
        print("✅ Context-aware search query generation")
        print("✅ Focused material extraction")
        print("✅ Off-topic request handling with suggestions")
        print("✅ Adaptation of various interests to insulin delivery research")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please ensure OLLAMA is running and the system is properly configured.")

if __name__ == "__main__":
    main() 