#!/usr/bin/env python3
"""
Simple test to verify the improved search logic works correctly.
"""

import asyncio
from semantic_scholar_search import initialize_client, search_papers

async def test_improved_search():
    """Test the search and filtering improvements."""
    print("🧪 Testing improved search logic...")
    
    # Initialize client
    client = initialize_client()
    
    # Test query that should find results
    test_query = "biocompatible polymers thermal stability"
    print(f"📝 Test query: {test_query}")
    print("=" * 60)
    
    try:
        # Search for papers
        results = search_papers(client, test_query, 5)
        print(f"📚 Found {len(results)} papers from search")
        
        if results:
            print("\n🔍 Paper Details:")
            for i, paper in enumerate(results, 1):
                title = paper.get('title', 'No title')
                year = paper.get('year', 'Unknown year')
                abstract = paper.get('abstract', '') or 'No abstract'
                
                print(f"\n{i}. **{title}**")
                print(f"   Year: {year}")
                print(f"   Abstract: {abstract[:200]}...")
                
                # Test the filtering logic
                title_lower = title.lower()
                abstract_lower = abstract.lower()
                
                # Primary keywords
                primary_keywords = [
                    'insulin', 'delivery', 'polymer', 'biocompatible', 'stability', 
                    'patch', 'transdermal', 'therapeutic', 'drug', 'material'
                ]
                
                # Secondary keywords
                secondary_keywords = [
                    'protein', 'preservation', 'controlled release', 'nano',
                    'hydrogel', 'micro', 'bio', 'medical', 'pharmaceutical',
                    'temperature', 'thermal', 'storage', 'formulation'
                ]
                
                has_primary = any(keyword in title_lower or keyword in abstract_lower for keyword in primary_keywords)
                has_secondary = any(keyword in title_lower or keyword in abstract_lower for keyword in secondary_keywords)
                
                if has_primary or has_secondary:
                    print(f"   ✅ Would pass filter (primary: {has_primary}, secondary: {has_secondary})")
                else:
                    print(f"   ❌ Would be filtered out")
        else:
            print("❌ No papers found")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improved_search()) 