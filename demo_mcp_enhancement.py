#!/usr/bin/env python3
"""
Demo script to showcase the enhanced MCP literature mining system
with rich natural language feedback and AI explanations.
"""

import asyncio
import sys
from datetime import datetime
from mcp_client import EnhancedLiteratureMiner
from ollama_client import OllamaClient

def print_demo_header():
    """Print demonstration header."""
    print("🚀 ENHANCED MCP LITERATURE MINING DEMONSTRATION")
    print("=" * 70)
    print("This demo showcases the rich, interactive experience of the")
    print("enhanced MCP literature mining system with AI explanations.")
    print("=" * 70)

def print_comparison():
    """Print comparison between old and new versions."""
    print("\n📊 COMPARISON: Basic vs Enhanced MCP Experience")
    print("-" * 50)
    print("BASIC MCP (Previous):")
    print("• Simple progress messages: 'Searching...', 'Analyzing...'")
    print("• Limited feedback on AI reasoning")
    print("• Basic status updates")
    print()
    print("ENHANCED MCP (Current):")
    print("• Natural language AI explanations")
    print("• Real-time reasoning insights")
    print("• Phase-by-phase narrative")
    print("• AI shows its thinking process")
    print("• Conversational progress updates")
    print("-" * 50)

async def demonstrate_enhanced_mining():
    """Demonstrate the enhanced MCP mining with rich feedback."""
    
    print_demo_header()
    print_comparison()
    
    # Initialize the enhanced system
    try:
        ollama_client = OllamaClient()
        enhanced_miner = EnhancedLiteratureMiner(ollama_client=ollama_client)
        
        print(f"\n✅ Enhanced MCP system initialized with Ollama model: {ollama_client.model_name}")
        print("🤖 AI explanation generation: ENABLED")
        print("📡 MCP protocol integration: ACTIVE")
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced system: {e}")
        print("📝 Note: This demo requires Ollama to be running for AI explanations")
        return
    
    # Demo query
    test_query = "smart polymers for temperature-responsive insulin delivery"
    print(f"\n🧪 DEMO QUERY: '{test_query}'")
    print("\n" + "="*70)
    print("🎬 STARTING ENHANCED MCP MINING DEMONSTRATION")
    print("="*70)
    
    # Track progress messages to show the rich narrative
    progress_messages = []
    explanations = []
    
    def demo_progress_callback(message, step_type="info"):
        """Enhanced progress callback for demo."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding for different message types
        if step_type == "start":
            print(f"\n🚀 [{timestamp}] START: {message}")
        elif step_type == "explanation":
            print(f"\n🧠 [{timestamp}] AI REASONING:")
            print(f"    {message}")
            explanations.append(message)
        elif step_type == "complete":
            print(f"\n✅ [{timestamp}] COMPLETE: {message}")
        elif step_type == "info":
            print(f"📋 [{timestamp}] {message}")
        else:
            print(f"📌 [{timestamp}] {message}")
        
        progress_messages.append({
            'message': message,
            'type': step_type,
            'timestamp': timestamp
        })
    
    try:
        print("\n🔄 Initiating enhanced literature mining...")
        
        # Perform the enhanced mining
        results = await enhanced_miner.intelligent_mining_with_mcp(
            user_request=test_query,
            max_papers=15,  # Smaller number for demo
            recent_only=True,
            progress_callback=demo_progress_callback
        )
        
        print("\n" + "="*70)
        print("📈 DEMONSTRATION SUMMARY")
        print("="*70)
        
        # Show what makes this enhanced
        print(f"\n📊 Rich Interaction Statistics:")
        print(f"   • Total Progress Messages: {len(progress_messages)}")
        print(f"   • AI Explanations Generated: {len(explanations)}")
        print(f"   • Natural Language Reasoning: {'✅ ENABLED' if explanations else '❌ DISABLED'}")
        
        # Show results if successful
        if 'error' not in results:
            material_count = len(results.get('material_candidates', []))
            papers_count = results.get('papers_analyzed', 0)
            
            print(f"\n🎯 Mining Results:")
            print(f"   • Papers Analyzed: {papers_count}")
            print(f"   • Materials Found: {material_count}")
            print(f"   • MCP Enhanced: {'✅' if results.get('mcp_enhanced') else '❌'}")
            
            print("\n💭 Sample AI Reasoning Excerpts:")
            for i, explanation in enumerate(explanations[:3], 1):
                print(f"{i}. {explanation[:100]}...")
        else:
            print(f"\n❌ Demo Error: {results['error']}")
        
        print("\n🎯 KEY ENHANCEMENT FEATURES DEMONSTRATED:")
        print("   ✅ Natural language AI explanations at each phase")
        print("   ✅ Real-time reasoning insights")
        print("   ✅ Conversational progress updates")
        print("   ✅ Phase-by-phase narrative structure")
        print("   ✅ AI shows its thinking process")
        print("   ✅ Rich user interaction experience")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 This is expected if MCP server components aren't fully set up")
    
    finally:
        # Cleanup
        try:
            await enhanced_miner.cleanup()
        except:
            pass

def show_enhanced_features():
    """Show what makes the enhanced version special."""
    print("\n🌟 ENHANCED MCP FEATURES:")
    print("=" * 50)
    
    features = [
        {
            "feature": "AI-Generated Explanations",
            "description": "LLM creates natural language explanations of what it's thinking at each step",
            "example": "'I'm excited to use my enhanced MCP capabilities to help you discover...'"
        },
        {
            "feature": "Phase-by-Phase Narrative",
            "description": "Structured storytelling approach to research process",
            "example": "Phase 1: Strategy → Phase 2: Search → Phase 3: Analysis"
        },
        {
            "feature": "Rich Progress Callbacks",
            "description": "Multiple types of updates (start, explanation, info, complete)",
            "example": "Shows both technical progress AND AI reasoning"
        },
        {
            "feature": "Conversational Interface",
            "description": "AI speaks naturally about its process and reasoning",
            "example": "'Let me leverage my advanced research tools to...'"
        },
        {
            "feature": "Real-time Insights",
            "description": "User sees AI's thought process as it happens",
            "example": "Strategic thinking, analysis approach, result interpretation"
        }
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"\n{i}. {feature['feature']}")
        print(f"   📝 {feature['description']}")
        print(f"   💬 Example: {feature['example']}")

def main():
    """Main demonstration function."""
    print("🎯 This script demonstrates the enhanced MCP experience")
    print("   compared to the basic version mentioned in your question.")
    print("\nChoose demonstration mode:")
    print("1. Show enhanced features overview")
    print("2. Run live enhanced mining demo (requires Ollama)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            show_enhanced_features()
        
        if choice in ['2', '3']:
            print("\n⚠️  Starting live demo (requires Ollama running)...")
            asyncio.run(demonstrate_enhanced_mining())
        
        print("\n🎉 Demo complete!")
        print("💡 The enhanced MCP version now provides the same rich,")
        print("   interactive experience as the regular literature mining system!")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main() 