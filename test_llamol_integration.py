#!/usr/bin/env python3
"""
Test script for LlaSMol integration.

This script tests the LlaSMol integration with the insulin AI system
to ensure everything is working properly before deployment.
"""

import os
import sys
import time
from typing import Dict, List

def test_llamol_availability():
    """Test if LlaSMol dependencies are available."""
    print("🧪 Testing LlaSMol availability...")
    
    try:
        from llamol_integration import LLAMOL_AVAILABLE, LlaSMolChatWrapper
        if LLAMOL_AVAILABLE:
            print("✅ LlaSMol dependencies available")
            available_models = LlaSMolChatWrapper.get_available_models()
            print(f"📋 Available models: {available_models}")
            return True
        else:
            print("❌ LlaSMol dependencies not available")
            return False
    except Exception as e:
        print(f"❌ Error testing LlaSMol availability: {e}")
        return False


def test_llamol_model_loading():
    """Test loading a LlaSMol model."""
    print("\n🔬 Testing LlaSMol model loading...")
    
    try:
        from llamol_integration import llamol_manager
        
        # Try to load the default model
        model_name = "osunlp/LlaSMol-Mistral-7B"
        print(f"Loading model: {model_name}")
        
        success = llamol_manager.load_model(model_name)
        if success:
            print("✅ Model loaded successfully")
            loaded_models = llamol_manager.list_loaded_models()
            print(f"📋 Loaded models: {loaded_models}")
            return True
        else:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        return False


def test_llamol_generation():
    """Test LlaSMol text generation."""
    print("\n💬 Testing LlaSMol generation...")
    
    try:
        from llamol_integration import llamol_manager
        
        model = llamol_manager.get_model()
        if not model:
            print("❌ No model available for testing")
            return False
        
        # Test chemistry queries
        test_queries = [
            "What is the SMILES for water?",
            "What is the molecular formula of <SMILES>CCO</SMILES>?",
            "Is <SMILES>CC(C)Cl</SMILES> toxic?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = model.generate_chemistry_response(query)
            
            if result['success']:
                print(f"✅ Response: {result['response'][:100]}...")
                if result['parsed_response']:
                    print(f"📋 Parsed chemistry: {result['parsed_response']}")
            else:
                print(f"❌ Generation failed: {result['error']}")
                return False
        
        print("✅ All generation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing generation: {e}")
        return False


def test_chatbot_integration():
    """Test LlaSMol integration with the chatbot system."""
    print("\n🤖 Testing chatbot integration...")
    
    try:
        from chatbot_system import InsulinAIChatbot
        
        # Test with LlaSMol model
        chatbot = InsulinAIChatbot(
            model_type="llamol",
            llamol_model="osunlp/LlaSMol-Mistral-7B"
        )
        
        # Test chemistry-focused conversation
        test_messages = [
            "What are the key properties to consider for insulin stability?",
            "Can you explain the SMILES notation for glucose?",
            "What makes a good polymer for drug delivery patches?"
        ]
        
        session_id = "test_session"
        
        for message in test_messages:
            print(f"\nTesting: {message}")
            response = chatbot.chat(message, session_id, mode='research')
            
            if response['success']:
                print(f"✅ Response: {response['response'][:100]}...")
                print(f"📊 Model: {response['model_type']} - {response['model_name']}")
                if response.get('parsed_chemistry'):
                    print(f"🧪 Chemistry info: {response['parsed_chemistry']}")
            else:
                print(f"❌ Chat failed: {response['error']}")
                return False
        
        # Test model switching
        print("\n🔄 Testing model switching...")
        switch_success = chatbot.switch_model("ollama", "llama3.2")
        if switch_success:
            print("✅ Successfully switched to Ollama")
            
            # Test with Ollama
            response = chatbot.chat("Hello from Ollama", session_id)
            if response['success']:
                print(f"✅ Ollama response: {response['response'][:50]}...")
            
            # Switch back to LlaSMol
            switch_back = chatbot.switch_model("llamol")
            if switch_back:
                print("✅ Successfully switched back to LlaSMol")
            else:
                print("⚠️ Could not switch back to LlaSMol")
        else:
            print("⚠️ Model switching test skipped (Ollama may not be available)")
        
        print("✅ Chatbot integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chatbot integration: {e}")
        return False


def test_api_integration():
    """Test API integration (requires Flask app to be running)."""
    print("\n🌐 Testing API integration...")
    
    try:
        import requests
        import json
        
        base_url = "http://localhost:5000"
        
        # Test model info endpoint
        response = requests.get(f"{base_url}/api/models/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info API working")
            print(f"📊 Current model: {data.get('current_model', {})}")
            print(f"🔬 LlaSMol available: {data.get('llamol_available', False)}")
        else:
            print(f"⚠️ API not responding (status: {response.status_code})")
            print("💡 Start the Flask app with: python app.py")
            return False
        
        # Test model switching via API
        switch_data = {"model_type": "llamol"}
        response = requests.post(
            f"{base_url}/api/models/switch", 
            json=switch_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Model switching API working")
        else:
            print(f"⚠️ Model switching failed (status: {response.status_code})")
        
        # Test chat with LlaSMol
        chat_data = {
            "message": "What is the SMILES for ethanol?",
            "type": "general"
        }
        response = requests.post(
            f"{base_url}/api/chat",
            json=chat_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat API working with LlaSMol")
            print(f"🤖 Response: {data.get('message', '')[:100]}...")
            if data.get('chemistry'):
                print(f"🧪 Chemistry data: {data['chemistry']}")
        else:
            print(f"⚠️ Chat API failed (status: {response.status_code})")
            
        return True
        
    except requests.exceptions.RequestException:
        print("⚠️ API tests skipped - Flask app not running")
        print("💡 Start the Flask app with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error testing API integration: {e}")
        return False


def run_all_tests():
    """Run all LlaSMol integration tests."""
    print("🚀 Starting LlaSMol Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Availability Check", test_llamol_availability),
        ("Model Loading", test_llamol_model_loading), 
        ("Text Generation", test_llamol_generation),
        ("Chatbot Integration", test_chatbot_integration),
        ("API Integration", test_api_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n❌ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LlaSMol integration is ready.")
    elif passed >= total // 2:
        print("⚠️ Some tests failed. Check the output above for details.")
    else:
        print("❌ Many tests failed. Please check your setup.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 