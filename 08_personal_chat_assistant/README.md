# 🧠 Personal Chat Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Memory-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Memory-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

> A memory-enabled assistant combining **retrieval**, **profile context**, and **conversation history** for personalized, context-aware responses across recurring interactions.

## 📌 What It Does

- **Long-term memory** — stores past conversations in ChromaDB vector store, persists across sessions
- **User profiles** — tracks name, preferences, and topics of interest
- **Short-term memory** — rolling conversation window with auto-summarization for long chats
- **Interest tracking** — automatically detects and remembers user topics (tech, finance, health, etc.)
- **Multi-user** — isolated memory and profiles per user ID
- Streamlit chat UI for interactive testing

## 🏗️ Architecture

```
User Message
      │
      ▼
  Retrieve relevant long-term memories (ChromaDB similarity search)
      │
      ▼
  Build context: [User Profile] + [Memories] + [Recent History]
      │
      ▼
  LLM generates personalized response
      │
      ▼
  Store exchange in long-term memory
  Update user profile (interests, interaction count)
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Start API
python api.py          # runs on port 8007

# Or launch the Streamlit chat UI
streamlit run app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message, get personalized response |
| POST | `/profile` | Update user name and preferences |
| GET | `/memory/{user_id}` | View what the assistant remembers |
| GET | `/history/{user_id}` | View conversation history |
| DELETE | `/session/{user_id}` | Clear session and memory |

**Example:**
```bash
# Set profile
curl -X POST http://localhost:8007/profile \
  -d '{"user_id": "alex", "name": "Alex", "preferences": {"style": "concise"}}'

# Chat
curl -X POST http://localhost:8007/chat \
  -d '{"user_id": "alex", "message": "What are good Python libraries for data science?"}'

# Check what it remembers about you
curl http://localhost:8007/memory/alex
```

**Memory response:**
```json
{
  "name": "Alex",
  "interests": ["technology"],
  "total_interactions": 5,
  "long_term_memories": 12
}
```

## 🛠️ Tech Stack

`Python` · `LangChain` · `ChromaDB` · `Vector DB` · `FastAPI` · `Streamlit` · `sentence-transformers`
