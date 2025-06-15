# 🧠 Chatbot Memory System with LangChain

This document explains the enhanced conversation memory system implemented for the Insulin AI Chatbot using LangChain's memory components.

## 🎯 Overview

The chatbot now features persistent conversation memory that:
- **Remembers context** across conversations
- **Persists memory** when the application restarts
- **Supports different memory types** for various use cases
- **Maintains separate memory** for different conversation modes
- **Automatically saves** conversation history to disk

## 🛠️ Memory Types Available

### 1. Buffer Window Memory (Default)
- **Best for**: Most conversations
- **Behavior**: Keeps only the last N interactions (default: 10)
- **Pros**: Balances context and performance, prevents token overflow
- **Cons**: May lose older context in very long conversations

```python
chatbot = InsulinAIChatbot(memory_type="buffer_window")
```

### 2. Buffer Memory
- **Best for**: Short to medium conversations where you need full history
- **Behavior**: Keeps entire conversation history
- **Pros**: Never loses context
- **Cons**: Can become expensive with very long conversations

```python
chatbot = InsulinAIChatbot(memory_type="buffer")
```

### 3. Summary Memory
- **Best for**: Very long conversations
- **Behavior**: Summarizes old conversations to save tokens
- **Pros**: Handles unlimited conversation length
- **Cons**: May lose some specific details in summaries

```python
chatbot = InsulinAIChatbot(memory_type="summary")
```

## 🚀 Getting Started

### 1. Environment Configuration

Add these variables to your `.env` file:

```bash
# Memory type: 'buffer', 'summary', or 'buffer_window'
CHATBOT_MEMORY_TYPE=buffer_window

# Directory to store memory files
CHATBOT_MEMORY_DIR=chat_memory
```

### 2. Basic Usage

```python
from chatbot_system import InsulinAIChatbot

# Initialize with memory
chatbot = InsulinAIChatbot(
    memory_type="buffer_window",
    memory_dir="my_chat_memory"
)

# Have a conversation
session_id = "user_123"
response1 = chatbot.chat("Tell me about insulin stability", session_id, mode='research')
response2 = chatbot.chat("What did we just discuss?", session_id, mode='research')

# The chatbot will remember the previous context
```

### 3. Memory Management

```python
# Get memory summary
summary = chatbot.get_memory_summary(session_id, mode='research')
print(f"Messages in memory: {summary['message_count']}")

# Clear memory for specific mode
chatbot.clear_history(session_id, mode='research')

# Clear all memory for session
chatbot.clear_history(session_id)
```

## 🌐 Web API Endpoints

### Get Memory Summary
```http
GET /api/memory/summary?mode=research
```

Response:
```json
{
  "session_id": "abc123",
  "mode": "research", 
  "message_count": 6,
  "memory_type": "buffer_window",
  "summary": "Conversation contains 6 messages using buffer_window memory."
}
```

### Clear Memory
```http
POST /api/memory/clear
Content-Type: application/json

{
  "mode": "research"  // Optional: leave empty to clear all modes
}
```

### Get Memory Configuration
```http
GET /api/memory/config
```

Response:
```json
{
  "memory_type": "buffer_window",
  "memory_dir": "chat_memory",
  "available_types": ["buffer", "summary", "buffer_window"],
  "description": {
    "buffer": "Keeps full conversation history",
    "summary": "Summarizes old conversations to save tokens", 
    "buffer_window": "Keeps only the last N interactions (default: 10)"
  }
}
```

## 🔄 Persistent Memory

Memory is automatically saved to disk and restored when the application starts:

```
chat_memory/
├── user123_general.pkl      # General conversation memory
├── user123_research.pkl     # Research conversation memory
├── user123_literature.pkl   # Literature conversation memory
└── user456_general.pkl      # Another user's memory
```

## 🎭 Multi-Mode Memory

Each conversation mode maintains separate memory:

```python
# These conversations are kept separate
chatbot.chat("Hello", session_id, mode='general')      # Separate memory
chatbot.chat("Explain polymers", session_id, mode='research')  # Separate memory
chatbot.chat("Find papers", session_id, mode='literature')    # Separate memory
```

## 🧪 Demo Script

Run the demo to see all memory types in action:

```bash
python demo_memory_chatbot.py
```

This demonstrates:
- Different memory types
- Persistent memory across restarts
- Multi-mode conversations
- Memory management functions

## 🔧 Advanced Configuration

### Custom Memory Settings

```python
# Custom buffer window size
from langchain.memory import ConversationBufferWindowMemory

chatbot = InsulinAIChatbot(memory_type="buffer_window")
# The buffer window size is set to 10 by default
# You can modify this in the _create_memory method
```

### Memory Directory Structure

```python
# Custom memory directory structure
chatbot = InsulinAIChatbot(
    memory_type="buffer_window",
    memory_dir="custom_memory_path"
)
```

## 📊 Performance Considerations

### Memory Type Recommendations

| Conversation Length | Recommended Type | Reason |
|-------------------|------------------|---------|
| Short (< 20 messages) | `buffer` | Full context, low overhead |
| Medium (20-100 messages) | `buffer_window` | Good balance |
| Long (100+ messages) | `summary` | Prevents token overflow |

### Token Usage

- **Buffer**: Linear growth with conversation length
- **Buffer Window**: Constant token usage (last N messages)
- **Summary**: Grows slowly, summarizes old content

## 🛡️ Error Handling

The system gracefully handles:
- Corrupted memory files (creates new ones)
- Missing memory directories (creates them automatically)
- Memory serialization errors (falls back to new memory)

## 🔍 Troubleshooting

### Memory Not Persisting
1. Check file permissions in memory directory
2. Verify `memory_dir` path is writable
3. Check for disk space issues

### Memory Files Too Large
1. Switch to `summary` memory type
2. Reduce buffer window size
3. Periodically clear old sessions

### Performance Issues
1. Use `buffer_window` for most cases
2. Avoid `buffer` memory for very long conversations
3. Clean up unused memory files periodically

## 🔮 Future Enhancements

Planned improvements:
- Redis/database backend for distributed systems
- Conversation search and indexing
- Memory compression and archiving
- Smart memory pruning based on importance
- Integration with vector databases for semantic memory

## 📝 Example Usage Patterns

### Research Assistant Pattern
```python
# Long research conversation with context retention
session = "research_session_1"
chatbot.chat("What are the challenges in insulin delivery?", session, mode='research')
chatbot.chat("How do hydrogels help?", session, mode='research')  # Remembers context
chatbot.chat("What about polymer degradation?", session, mode='research')  # Still remembers
```

### Multi-User Support Pattern
```python
# Different users get separate memory
user1_session = "user_alice_2024"
user2_session = "user_bob_2024"

chatbot.chat("My research focus is polymers", user1_session, mode='research')
chatbot.chat("I'm interested in nanoparticles", user2_session, mode='research')
# Each user's context is separate
```

### Mode Switching Pattern
```python
session = "user_session"
# Start with general questions
chatbot.chat("What is this project about?", session, mode='general')

# Switch to research mode for technical discussion
chatbot.chat("Explain insulin degradation pathways", session, mode='research')

# Switch to literature mode for paper search
chatbot.chat("Find papers on PLGA", session, mode='literature')

# Each mode remembers its own conversation context
```

This memory system transforms the chatbot from a stateless question-answering system into a truly conversational AI that builds understanding over time! 🎉 