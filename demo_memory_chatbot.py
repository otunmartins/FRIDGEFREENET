#!/usr/bin/env python3
"""
Demo script showing different LangChain memory types in the Insulin AI Chatbot System.

This script demonstrates:
1. ConversationBufferMemory - Keeps full conversation history
2. ConversationSummaryMemory - Summarizes old conversations
3. ConversationBufferWindowMemory - Keeps only the last N interactions
4. Persistent memory that survives application restarts
"""

import os
import time
from chatbot_system import InsulinAIChatbot

def demo_memory_types():
    """Demonstrate different memory types available in the chatbot."""
    
    print("=== Insulin AI Chatbot Memory Demo ===\n")
    
    # Demo 1: Buffer Window Memory (default)
    print("🔷 Demo 1: Buffer Window Memory (keeps last 10 interactions)")
    print("-" * 60)
    
    chatbot_window = InsulinAIChatbot(
        memory_type="buffer_window",
        memory_dir="demo_memory_window"
    )
    
    session_id = "demo_session_1"
    
    # Have a conversation
    messages = [
        "Hello, I'm interested in insulin delivery patches.",
        "What materials are commonly used for drug delivery?",
        "How does temperature affect insulin stability?",
        "What are the key challenges in fridge-free insulin delivery?",
        "Can you recommend some research papers on this topic?"
    ]
    
    print("Having conversation with Buffer Window Memory:")
    for i, msg in enumerate(messages, 1):
        print(f"\n👤 User: {msg}")
        response = chatbot_window.chat(msg, session_id, mode='research')
        print(f"🤖 Bot: {response['message'][:200]}...")
        
        # Show memory summary
        memory_summary = chatbot_window.get_memory_summary(session_id, 'research')
        print(f"📊 Memory: {memory_summary['message_count']} messages stored")
    
    print("\n" + "="*80 + "\n")
    
    # Demo 2: Buffer Memory (keeps everything)
    print("🔷 Demo 2: Buffer Memory (keeps full conversation)")
    print("-" * 60)
    
    chatbot_buffer = InsulinAIChatbot(
        memory_type="buffer",
        memory_dir="demo_memory_buffer"
    )
    
    session_id_2 = "demo_session_2"
    
    print("Same conversation with Buffer Memory:")
    for i, msg in enumerate(messages, 1):
        print(f"\n👤 User: {msg}")
        response = chatbot_buffer.chat(msg, session_id_2, mode='research')
        print(f"🤖 Bot: {response['message'][:200]}...")
        
        memory_summary = chatbot_buffer.get_memory_summary(session_id_2, 'research')
        print(f"📊 Memory: {memory_summary['message_count']} messages stored")
    
    print("\n" + "="*80 + "\n")
    
    # Demo 3: Summary Memory (summarizes old conversations)
    print("🔷 Demo 3: Summary Memory (summarizes old conversations)")
    print("-" * 60)
    
    chatbot_summary = InsulinAIChatbot(
        memory_type="summary",
        memory_dir="demo_memory_summary"
    )
    
    session_id_3 = "demo_session_3"
    
    print("Same conversation with Summary Memory:")
    for i, msg in enumerate(messages, 1):
        print(f"\n👤 User: {msg}")
        response = chatbot_summary.chat(msg, session_id_3, mode='research')
        print(f"🤖 Bot: {response['message'][:200]}...")
        
        memory_summary = chatbot_summary.get_memory_summary(session_id_3, 'research')
        print(f"📊 Memory: {memory_summary['summary']}")
    
    print("\n" + "="*80 + "\n")

def demo_persistent_memory():
    """Demonstrate persistent memory across application restarts."""
    
    print("🔷 Demo 4: Persistent Memory (survives app restarts)")
    print("-" * 60)
    
    session_id = "persistent_demo"
    
    # First chatbot instance
    print("Creating first chatbot instance...")
    chatbot1 = InsulinAIChatbot(
        memory_type="buffer_window",
        memory_dir="demo_persistent_memory"
    )
    
    print("\n👤 User: Tell me about insulin stability mechanisms")
    response1 = chatbot1.chat(
        "Tell me about insulin stability mechanisms", 
        session_id, 
        mode='research'
    )
    print(f"🤖 Bot: {response1['message'][:200]}...")
    
    memory_summary1 = chatbot1.get_memory_summary(session_id, 'research')
    print(f"📊 Memory after first message: {memory_summary1['message_count']} messages")
    
    # Simulate application restart by creating new instance
    print("\n🔄 Simulating application restart...")
    time.sleep(1)
    
    chatbot2 = InsulinAIChatbot(
        memory_type="buffer_window", 
        memory_dir="demo_persistent_memory"
    )
    
    print("\n👤 User: What did we discuss about insulin before?")
    response2 = chatbot2.chat(
        "What did we discuss about insulin before?", 
        session_id, 
        mode='research'
    )
    print(f"🤖 Bot: {response2['message'][:200]}...")
    
    memory_summary2 = chatbot2.get_memory_summary(session_id, 'research')
    print(f"📊 Memory after restart: {memory_summary2['message_count']} messages")
    
    print("\n✅ Memory successfully persisted across application restart!")
    print("\n" + "="*80 + "\n")

def demo_multi_mode_memory():
    """Demonstrate separate memory for different conversation modes."""
    
    print("🔷 Demo 5: Multi-Mode Memory (separate context per mode)")
    print("-" * 60)
    
    chatbot = InsulinAIChatbot(
        memory_type="buffer_window",
        memory_dir="demo_multi_mode"
    )
    
    session_id = "multi_mode_demo"
    
    # General conversation
    print("Starting general conversation:")
    print("\n👤 User (General): What is this project about?")
    response_general = chatbot.chat(
        "What is this project about?", 
        session_id, 
        mode='general'
    )
    print(f"🤖 Bot (General): {response_general['message'][:200]}...")
    
    # Research conversation
    print("\nSwitching to research mode:")
    print("\n👤 User (Research): Explain insulin degradation pathways")
    response_research = chatbot.chat(
        "Explain insulin degradation pathways", 
        session_id, 
        mode='research'
    )
    print(f"🤖 Bot (Research): {response_research['message'][:200]}...")
    
    # Literature conversation
    print("\nSwitching to literature mode:")
    print("\n👤 User (Literature): How should I search for polymer studies?")
    response_literature = chatbot.chat(
        "How should I search for polymer studies?", 
        session_id, 
        mode='literature'
    )
    print(f"🤖 Bot (Literature): {response_literature['message'][:200]}...")
    
    # Show memory summaries for each mode
    print("\n📊 Memory Summary by Mode:")
    for mode in ['general', 'research', 'literature']:
        summary = chatbot.get_memory_summary(session_id, mode)
        print(f"  {mode.title()}: {summary['message_count']} messages")
    
    print("\n✅ Each mode maintains separate conversation context!")
    print("\n" + "="*80 + "\n")

def cleanup_demo_files():
    """Clean up demo memory files."""
    import shutil
    
    demo_dirs = [
        "demo_memory_window",
        "demo_memory_buffer", 
        "demo_memory_summary",
        "demo_persistent_memory",
        "demo_multi_mode"
    ]
    
    print("🧹 Cleaning up demo files...")
    for dir_name in demo_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    try:
        print("Starting Insulin AI Chatbot Memory Demos...\n")
        
        # Run all demos
        demo_memory_types()
        demo_persistent_memory()
        demo_multi_mode_memory()
        
        print("🎉 All demos completed successfully!")
        
        # Ask if user wants to clean up
        cleanup = input("\nCleanup demo files? (y/N): ").strip().lower()
        if cleanup in ['y', 'yes']:
            cleanup_demo_files()
        else:
            print("Demo files kept for inspection.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}") 