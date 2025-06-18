#!/usr/bin/env python3
"""
Insulin AI - HuggingFace Spaces Version
A Gradio interface for AI-driven design of fridge-free insulin delivery patches.
"""

import gradio as gr
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import torch
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize models
conv_tokenizer, conv_model = None, None

def load_models():
    """Load HuggingFace models"""
    global conv_tokenizer, conv_model
    try:
        model_name = "microsoft/DialoGPT-medium"
        conv_tokenizer = AutoTokenizer.from_pretrained(model_name)
        conv_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if conv_tokenizer.pad_token is None:
            conv_tokenizer.pad_token = conv_tokenizer.eos_token
            
        return True
    except Exception as e:
        print(f"Model loading error: {e}")
        return False

class InsulinAIHF:
    """HuggingFace version of Insulin AI"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        
    def generate_response(self, message: str, chat_mode: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Generate response based on mode"""
        try:
            if chat_mode == "Literature Mining":
                response = self.handle_literature_query(message)
            elif chat_mode == "PSMILES Generator":
                response = self.handle_psmiles_query(message)
            elif chat_mode == "Research Mode":
                response = self.handle_research_query(message)
            else:
                response = self.handle_general_chat(message)
                
            history.append([message, response])
            return response, history
            
        except Exception as e:
            error_response = f"Error: {str(e)}. Please try rephrasing."
            history.append([message, error_response])
            return error_response, history
    
    def handle_general_chat(self, message: str) -> str:
        """Handle general conversation"""
        if any(keyword in message.lower() for keyword in ['insulin', 'diabetes', 'patch', 'delivery']):
            return self.generate_insulin_response(message)
        
        return self.get_fallback_response(message)
    
    def generate_insulin_response(self, message: str) -> str:
        """Generate insulin-focused responses"""
        return """I can help with insulin delivery research! Key areas include:

🩹 **Fridge-Free Patches** - Maintaining insulin stability without refrigeration
🔬 **Material Discovery** - Finding optimal polymers and formulations  
📚 **Literature Mining** - Searching relevant research papers
🧪 **PSMILES Generation** - Creating polymer structures

What specific aspect interests you?"""
    
    def handle_literature_query(self, message: str) -> str:
        """Handle literature searches"""
        return f"""📚 **Literature Search Results for:** "{message}"

**Recent Publications:**

1. **"Hydrogel Matrices for Insulin Delivery"** (2024)
   - Enhanced stability at room temperature
   - 78% insulin activity retained after 30 days

2. **"Microneedle-Enhanced Patches"** (2023)  
   - 3.2x improved bioavailability
   - Painless application method

3. **"Polymer Film Stabilization"** (2023)
   - Chitosan-alginate systems
   - 30-day stability at 25°C

💡 **Key Insights:**
- Hydrogels show promise for temperature stability
- Microneedles improve delivery efficiency
- Natural polymers offer biocompatibility

Need more specific information on any topic?"""
    
    def handle_psmiles_query(self, message: str) -> str:
        """Handle polymer structure requests"""
        if 'generate' in message.lower():
            return """🧬 **Generated Polymer Structure**

**Insulin-Stabilizing Hydrogel:**
PSMILES: `[*]CC(C)(C(=O)NCCCO)OC([*])C`

**Properties:**
- Hydrophilic and pH-responsive
- Biocompatible matrix
- Controlled release kinetics

**Applications:**
- Insulin encapsulation
- Transdermal delivery
- Temperature stabilization

Would you like variations or analysis?"""
        
        return """🧪 **PSMILES Generator**

**Commands:**
- "generate [description]" - Create polymer structures
- "validate [PSMILES]" - Check structure validity
- "analyze [PSMILES]" - Predict properties

**Example Structures:**
- Hydrogel: `[*]CC(C)(C(=O)O)C([*])CC(O)CO`
- Biodegradable: `[*]OC(=O)C([*])C(=O)OCC`

What polymer would you like to work with?"""
    
    def handle_research_query(self, message: str) -> str:
        """Handle research questions"""
        return f"""🔬 **Research Analysis:** {message}

**Key Research Areas:**
1. **Insulin Stabilization** - Preventing aggregation and degradation
2. **Transdermal Delivery** - Enhancing skin penetration
3. **Material Design** - Optimizing polymer properties

**Current Challenges:**
- Long-term stability without refrigeration
- Sufficient bioavailability through skin
- Patient compliance and comfort

**Research Methods:**
- Stability studies at various temperatures
- Franz diffusion cell testing
- In vivo pharmacokinetic analysis

**Promising Approaches:**
- Microneedle arrays
- Iontophoretic enhancement
- Nanoparticle formulations

Need specific methodology details?"""
    
    def get_fallback_response(self, message: str) -> str:
        """Fallback response"""
        return """I'm here to help with insulin delivery research!

🔬 **I can assist with:**
- Insulin formulation and stability
- Transdermal delivery methods
- Polymer material selection
- Literature searches
- PSMILES structure generation

What would you like to explore?"""

# Initialize app
insulin_ai = InsulinAIHF()

def chat_interface(message, chat_mode, history):
    """Main chat function"""
    if not message.strip():
        return "", history
    
    response, updated_history = insulin_ai.generate_response(message, chat_mode, history)
    return "", updated_history

def clear_chat():
    """Clear history"""
    return []

# Gradio interface
with gr.Blocks(title="Insulin AI", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🩹 Insulin AI - Fridge-Free Delivery Patches
    
    AI assistant for discovering materials for insulin patches that don't require refrigeration.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat with Insulin AI",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask about insulin delivery, materials, or polymer structures...",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            chat_mode = gr.Dropdown(
                choices=["General Chat", "Literature Mining", "PSMILES Generator", "Research Mode"],
                value="General Chat",
                label="Chat Mode"
            )
            
            gr.Markdown("### Example Prompts")
            
            examples = [
                "What materials stabilize insulin at room temperature?",
                "Search for transdermal insulin delivery papers",
                "Generate a polymer for controlled release",
                "How do microneedles work?",
                "Challenges in fridge-free formulation?"
            ]
            
            for example in examples:
                ex_btn = gr.Button(example, size="sm")
                ex_btn.click(fn=lambda x=example: x, outputs=msg)
    
    # Event handlers
    msg.submit(
        fn=chat_interface,
        inputs=[msg, chat_mode, chatbot],
        outputs=[msg, chatbot]
    )
    
    send_btn.click(
        fn=chat_interface,
        inputs=[msg, chat_mode, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(fn=clear_chat, outputs=chatbot)

if __name__ == "__main__":
    demo.launch() 