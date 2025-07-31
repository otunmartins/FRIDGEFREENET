#!/usr/bin/env python3
"""
Demonstration script for Enhanced API Validation System
This shows how to integrate the new validation system with your existing application.
"""

import logging
import sys
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_validation_demo.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Demonstrate the enhanced API validation system."""
    print("🧪 Enhanced PubChem API with LangChain Validation Demo")
    print("=" * 60)
    
    # Import the enhanced system
    try:
        from enhanced_api_validation import ValidatedPubChemClient, SMILESValidationResult
        print("✅ Enhanced API validation system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced system: {e}")
        print("💡 Make sure to install requirements: pip install -r requirements_enhanced_api.txt")
        return
    
    # Initialize the client
    print("\n🔧 Initializing ValidatedPubChemClient...")
    client = ValidatedPubChemClient(model_name="granite3.3:8b")
    
    # Test molecules with various challenges
    test_molecules = [
        "alanine",                    # ✅ Should work immediately (known molecule)
        "alinine",                    # 🔧 Typo - should be corrected by LangChain
        "aspirin",                    # ✅ Common drug name
        "acetylsalicylic acid",       # ✅ IUPAC name for aspirin
        "methionine",                 # ✅ Amino acid (from your memory issue)
        "CC=OCCCNC",                  # ❌ Invalid SMILES (missing parentheses)
        "H2O",                        # ✅ Chemical formula
        "ethyl alcohol",              # ✅ Common name for ethanol
        "nonexistent_mol_123",        # ❌ Should fail gracefully
        "insulin",                    # 🤔 Complex protein (should handle gracefully)
    ]
    
    print(f"\n🧪 Testing {len(test_molecules)} molecules...")
    print("=" * 60)
    
    results = []
    total_start_time = time.time()
    
    for i, molecule in enumerate(test_molecules, 1):
        print(f"\n[{i:2d}/{len(test_molecules)}] Testing: '{molecule}'")
        print("-" * 40)
        
        # Time the individual request
        start_time = time.time()
        result = client.get_smiles_with_validation(molecule, max_correction_attempts=2)
        end_time = time.time()
        
        # Display results
        status_emoji = "✅" if result.success else "❌"
        print(f"{status_emoji} Success: {result.success}")
        print(f"🧬 SMILES: {result.validated_smiles or 'N/A'}")
        print(f"📊 Confidence: {result.confidence_score:.2f}")
        print(f"📍 Source: {result.source}")
        print(f"⏱️  Time: {end_time - start_time:.2f}s")
        
        if result.corrections_applied:
            print(f"🔧 Corrections: {', '.join(result.corrections_applied)}")
        
        if result.error_details:
            print(f"⚠️  Error: {result.error_details}")
        
        # Store results for summary
        results.append({
            'molecule': molecule,
            'success': result.success,
            'smiles': result.validated_smiles,
            'confidence': result.confidence_score,
            'source': result.source,
            'corrections': result.corrections_applied,
            'time': end_time - start_time
        })
    
    total_end_time = time.time()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📈 SUMMARY STATISTICS")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    corrected = [r for r in results if r['corrections']]
    
    print(f"✅ Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)}/{len(results)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"🔧 Corrected: {len(corrected)}/{len(results)} ({len(corrected)/len(results)*100:.1f}%)")
    print(f"⏱️  Total time: {total_end_time - total_start_time:.2f}s")
    print(f"⚡ Avg time/molecule: {(total_end_time - total_start_time)/len(results):.2f}s")
    
    # Source breakdown
    print(f"\n📍 Sources used:")
    sources = {}
    for result in results:
        if result['success']:
            source = result['source']
            sources[source] = sources.get(source, 0) + 1
    
    for source, count in sorted(sources.items()):
        print(f"   {source}: {count}")
    
    # Cache information
    try:
        cache_info = client.cache_manager.get_cache_info()
        if cache_info:
            print(f"\n💾 Cache statistics:")
            print(f"   Backend: {cache_info.get('backend', 'N/A')}")
            print(f"   URLs cached: {cache_info.get('urls_count', 0)}")
            print(f"   Responses cached: {cache_info.get('responses_count', 0)}")
    except Exception as e:
        print(f"⚠️  Cache info unavailable: {e}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    if len(failed) > 0:
        print("For failed molecules:")
        for result in failed:
            if 'nonexistent' in result['molecule'].lower():
                print(f"   • '{result['molecule']}': Expected failure (test molecule)")
            elif result['source'] == 'import_error':
                print(f"   • '{result['molecule']}': Install missing dependencies")
            else:
                print(f"   • '{result['molecule']}': Try alternative names or manual SMILES lookup")
    
    if len(corrected) > 0:
        print(f"\nLangChain corrections helped with {len(corrected)} molecules:")
        for result in corrected:
            print(f"   • '{result['molecule']}': {', '.join(result['corrections'])}")
    
    print(f"\n🚀 Integration tips:")
    print("   1. The enhanced system is drop-in compatible with your existing code")
    print("   2. It provides detailed error information for debugging")
    print("   3. Caching reduces API calls and improves performance")
    print("   4. LangChain integration helps with typos and alternative names")
    print("   5. Rate limiting prevents API abuse and ensures compliance")
    
    print(f"\n🔧 To integrate into your app, replace:")
    print("   OLD: smiles = get_smiles_from_pubchem(molecule_name)")
    print("   NEW: result = client.get_smiles_with_validation(molecule_name)")
    print("        smiles = result.validated_smiles if result.success else None")

def demo_integration_patterns():
    """Show different integration patterns."""
    print("\n" + "=" * 60)
    print("🔧 INTEGRATION PATTERNS")
    print("=" * 60)
    
    from enhanced_api_validation import ValidatedPubChemClient
    
    client = ValidatedPubChemClient()
    
    # Pattern 1: Simple replacement
    print("\n1️⃣ Simple Replacement Pattern:")
    print("```python")
    print("# OLD:")
    print("# smiles = get_smiles_from_pubchem('alanine')")
    print("# NEW:")
    print("result = client.get_smiles_with_validation('alanine')")
    print("smiles = result.validated_smiles if result.success else None")
    print("```")
    
    # Pattern 2: Error handling
    print("\n2️⃣ Enhanced Error Handling Pattern:")
    print("```python")
    print("result = client.get_smiles_with_validation('molecule_name')")
    print("if result.success:")
    print("    print(f'SMILES: {result.validated_smiles}')")
    print("    print(f'Confidence: {result.confidence_score}')")
    print("    print(f'Source: {result.source}')")
    print("else:")
    print("    print(f'Failed: {result.error_details}')")
    print("    if result.corrections_applied:")
    print("        print(f'Tried corrections: {result.corrections_applied}')")
    print("```")
    
    # Pattern 3: Batch processing
    print("\n3️⃣ Batch Processing Pattern:")
    print("```python")
    print("molecules = ['alanine', 'glycine', 'methionine']")
    print("results = []")
    print("for molecule in molecules:")
    print("    result = client.get_smiles_with_validation(molecule)")
    print("    results.append(result)")
    print("    if not result.success:")
    print("        logger.warning(f'Failed to get SMILES for {molecule}')")
    print("```")

if __name__ == "__main__":
    try:
        main()
        demo_integration_patterns()
        print(f"\n🎉 Demo completed successfully!")
        print("💡 Check 'api_validation_demo.log' for detailed logs")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo failed with exception")
        sys.exit(1) 