#!/usr/bin/env python3
"""
Interactive literature mining demo for insulin delivery materials.
Demonstrates the AI-guided literature search workflow.
"""

import json
import sys
from .literature_mining_system import MaterialsLiteratureMiner

def print_header():
    """Print the system header."""
    print("🧠 AI-DRIVEN INSULIN DELIVERY MATERIALS RESEARCH ASSISTANT")
    print("=" * 70)
    print("Ask me anything about materials for insulin delivery patches!")
    print("I'll intelligently interpret your request and search the literature.")
    print("\nExamples you can try:")
    print("• 'smart polymers that respond to temperature'")
    print("• 'nanotechnology for drug delivery'")
    print("• 'biodegradable materials for medical devices'")
    print("• 'green chemistry approaches'")
    print("• 'machine learning for protein folding'")
    print("\nType 'quit', 'exit', or 'q' to exit")
    print("Type 'help' for more options")
    print("=" * 70)

def print_help():
    """Print help information."""
    print("\n📚 HELP - Available Commands:")
    print("• Just type your question in natural language")
    print("• 'examples' - Show example queries")
    print("• 'config' - Show current configuration")
    print("• 'last' - Show details of last search")
    print("• 'save' - Toggle saving results to files")
    print("• 'papers N' - Set max papers to analyze (e.g., 'papers 20')")
    print("• 'recent' - Toggle recent papers only filter")
    print("• 'clear' - Clear screen")
    print("• 'quit' or 'exit' - Exit the system")

def print_examples():
    """Print example queries."""
    print("\n🎯 EXAMPLE QUERIES:")
    
    examples = [
        ("Temperature-responsive", "smart polymers that respond to temperature"),
        ("Nanotechnology", "nanoparticles for drug delivery"),
        ("Sustainability", "green chemistry and biodegradable materials"),
        ("Manufacturing", "3D printing materials for medical devices"),
        ("Computational", "machine learning for protein structure prediction"),
        ("Related applications", "pain relief patches and transdermal delivery"),
        ("Materials science", "hydrogels with controlled release properties"),
        ("Biocompatibility", "FDA approved materials for skin contact"),
    ]
    
    for category, example in examples:
        print(f"• {category}: '{example}'")

def format_results_summary(results):
    """Format and display results summary."""
    if 'error' in results:
        print(f"\n❌ {results['error']}")
        if 'suggestion' in results:
            print(f"💡 Suggestion: {results['suggestion']}")
        return
    
    strategy = results.get('search_strategy', {})
    
    print(f"\n🎯 RELEVANCE SCORE: {strategy.get('relevance_score', 'N/A')}/10")
    print(f"🧠 INTERPRETATION: {strategy.get('interpretation', 'N/A')}")
    print(f"🔬 EXTRACTION FOCUS: {strategy.get('extraction_focus', 'General')}")
    
    print(f"\n📊 SEARCH RESULTS:")
    print(f"   Papers Analyzed: {results.get('papers_analyzed', 0)}")
    print(f"   Materials Found: {len(results.get('material_candidates', []))}")
    
    # Show generated search queries
    queries = strategy.get('search_queries', [])
    if queries:
        print(f"\n📚 GENERATED SEARCH QUERIES:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")

def format_materials_preview(results, num_preview=3):
    """Show preview of found materials."""
    candidates = results.get('material_candidates', [])
    if not candidates:
        print("\n📋 No materials found matching your criteria.")
        return
    
    print(f"\n🧪 MATERIAL CANDIDATES (showing {min(len(candidates), num_preview)} of {len(candidates)}):")
    print("-" * 60)
    
    for i, material in enumerate(candidates[:num_preview], 1):
        name = material.get('material_name', 'Unknown Material')
        composition = material.get('material_composition', 'Not specified')
        confidence = material.get('confidence_score', 'N/A')
        
        print(f"\n{i}. {name}")
        print(f"   Composition: {composition[:100]}{'...' if len(composition) > 100 else ''}")
        print(f"   Confidence: {confidence}/10")
        
        # Show key properties if available
        thermal = material.get('thermal_stability_temp_range', '')
        if thermal:
            print(f"   Thermal Stability: {thermal}")
        
        mechanism = material.get('stabilization_mechanism', '')
        if mechanism and mechanism != 'Not specified':
            print(f"   Mechanism: {mechanism[:80]}{'...' if len(mechanism) > 80 else ''}")

def interactive_session():
    """Run the interactive mining session."""
    print_header()
    
    try:
        # Initialize the mining system
        print("\n🔧 Initializing AI Literature Mining System...")
        miner = MaterialsLiteratureMiner()
        print("✅ System ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please ensure OLLAMA is running and properly configured.")
        return
    
    # Configuration
    config = {
        'max_papers': 20,  # Smaller default for interactive use
        'save_results': False,  # Don't save by default in interactive mode
        'recent_only': True
    }
    
    last_results = None
    
    while True:
        try:
            user_input = input("\n🔍 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the AI Literature Mining System!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'examples':
                print_examples()
                continue
            
            elif user_input.lower() == 'config':
                print(f"\n⚙️ CURRENT CONFIGURATION:")
                print(f"   Max Papers: {config['max_papers']}")
                print(f"   Save Results: {config['save_results']}")
                print(f"   Recent Papers Only: {config['recent_only']}")
                continue
            
            elif user_input.lower().startswith('papers '):
                try:
                    num = int(user_input.split()[1])
                    config['max_papers'] = max(5, min(100, num))  # Limit between 5-100
                    print(f"✅ Max papers set to {config['max_papers']}")
                except:
                    print("❌ Invalid format. Use: papers 20")
                continue
            
            elif user_input.lower() == 'save':
                config['save_results'] = not config['save_results']
                print(f"✅ Save results: {'ON' if config['save_results'] else 'OFF'}")
                continue
            
            elif user_input.lower() == 'recent':
                config['recent_only'] = not config['recent_only']
                print(f"✅ Recent papers only: {'ON' if config['recent_only'] else 'OFF'}")
                continue
            
            elif user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                print_header()
                continue
            
            elif user_input.lower() == 'last':
                if last_results:
                    print("\n📋 LAST SEARCH DETAILS:")
                    format_results_summary(last_results)
                    format_materials_preview(last_results, num_preview=5)
                else:
                    print("❌ No previous search results available.")
                continue
            
            # Process as a research question
            print(f"\n🤖 Processing: '{user_input}'...")
            print("⏳ This may take a few moments...")
            
            results = miner.intelligent_mining(
                user_request=user_input,
                max_papers=config['max_papers'],
                recent_only=config['recent_only'],
                save_results=config['save_results']
            )
            
            last_results = results
            
            # Display results
            format_results_summary(results)
            format_materials_preview(results)
            
            # Offer additional options
            if results.get('material_candidates'):
                print(f"\n💡 Want more details? Try:")
                print(f"   • 'last' to see more materials")
                print(f"   • Ask a follow-up question")
                print(f"   • Try 'papers {config['max_papers'] * 2}' for more comprehensive results")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try rephrasing your question or check your connection.")

def quick_demo():
    """Run a quick demo with sample queries."""
    print("🚀 QUICK DEMO MODE")
    print("Running sample queries to demonstrate the system...")
    
    miner = MaterialsLiteratureMiner()
    
    sample_queries = [
        "smart polymers for temperature control",
        "green chemistry materials",
        "nanotechnology drug delivery"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'='*50}")
        print(f"DEMO {i}/3: '{query}'")
        print('='*50)
        
        results = miner.intelligent_mining(
            user_request=query,
            max_papers=10,
            save_results=False
        )
        
        format_results_summary(results)
        format_materials_preview(results, num_preview=2)

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        quick_demo()
    else:
        interactive_session()

if __name__ == "__main__":
    main() 