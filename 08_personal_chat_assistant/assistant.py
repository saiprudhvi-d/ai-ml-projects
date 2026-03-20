"""
Project 08: Personal Chat Assistant
Memory-enabled assistant combining:
  - Retrieval: vector search over past conversations + knowledge
  - Profile context: user preferences, name, history
  - Conversation history: rolling window + long-term memory
Personalized, context-aware responses across recurring interactions.
"""

import os
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
import torch

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-large")
CHROMA_DIR = "./chroma_memory"
PROFILE_PATH = "./user_profiles.json"
MAX_HISTORY_TOKENS = 1000   # Summarize beyond this
MEMORY_TOP_K = 3            # Retrieved long-term memories per query


@dataclass
class UserProfile:
    user_id: str
    name: str = "User"
    preferences: dict = field(default_factory=dict)
    topics_of_interest: list[str] = field(default_factory=list)
    interaction_count: int = 0
    last_seen: str = ""
    summary: str = ""       # LLM-generated profile summary


@dataclass
class Message:
    role: str               # "user" | "assistant"
    content: str
    timestamp: str = ""
    session_id: str = ""


class PersonalChatAssistant:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())[:8]

        print("⏳ Loading components...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        # Long-term memory vector store (ChromaDB, persistent)
        self.memory_store = Chroma(
            collection_name=f"memory_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        self.llm = self._load_llm()
        self.profile = self._load_profile()
        self.conversation_history: list[Message] = []

        # Short-term memory with auto-summarization
        self.short_term = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=MAX_HISTORY_TOKENS,
            return_messages=True,
        )

        print(f"✅ Assistant ready for {self.profile.name} (session {self.session_id})")

    def _load_llm(self) -> HuggingFacePipeline:
        gen = hf_pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=gen)

    def _load_profile(self) -> UserProfile:
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH) as f:
                profiles = json.load(f)
            if self.user_id in profiles:
                return UserProfile(**profiles[self.user_id])
        return UserProfile(user_id=self.user_id)

    def _save_profile(self):
        profiles = {}
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH) as f:
                profiles = json.load(f)
        self.profile.last_seen = datetime.now().isoformat()
        profiles[self.user_id] = asdict(self.profile)
        with open(PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)

    # ─── Memory Operations ───────────────────────────────────────────────────────
    def store_memory(self, text: str, metadata: dict = None):
        """Store a message or fact in long-term vector memory."""
        meta = metadata or {}
        meta.update({"user_id": self.user_id, "session_id": self.session_id, "timestamp": datetime.now().isoformat()})
        self.memory_store.add_texts([text], metadatas=[meta])

    def retrieve_memories(self, query: str, k: int = MEMORY_TOP_K) -> list[str]:
        """Retrieve relevant long-term memories for a given query."""
        results = self.memory_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def build_context(self, user_input: str) -> str:
        """Compose full context: profile + long-term memories + short-term history."""
        memories = self.retrieve_memories(user_input)
        memory_block = "\n".join(f"- {m}" for m in memories) if memories else "No prior relevant memories."

        profile_block = (
            f"User: {self.profile.name}\n"
            f"Interests: {', '.join(self.profile.topics_of_interest) or 'Unknown'}\n"
            f"Preferences: {json.dumps(self.profile.preferences) or 'None set'}\n"
            f"Interactions: {self.profile.interaction_count}"
        )

        history_block = ""
        for msg in self.conversation_history[-6:]:   # Last 3 turns
            prefix = "User" if msg.role == "user" else "Assistant"
            history_block += f"{prefix}: {msg.content}\n"

        return (
            f"[USER PROFILE]\n{profile_block}\n\n"
            f"[RELEVANT MEMORIES]\n{memory_block}\n\n"
            f"[RECENT CONVERSATION]\n{history_block}"
        )

    # ─── Core Chat ───────────────────────────────────────────────────────────────
    def chat(self, user_input: str) -> str:
        """Process a user message and return a personalized response."""
        user_msg = Message(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
        )
        self.conversation_history.append(user_msg)

        context = self.build_context(user_input)
        prompt = (
            f"{context}\n\n"
            f"You are a helpful, personalized assistant. "
            f"Respond naturally and use the user's name ({self.profile.name}) occasionally. "
            f"Be concise but thorough.\n\n"
            f"User: {user_input}\nAssistant:"
        )

        response = self.llm(prompt)
        # Extract just the new text after "Assistant:"
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        response = response.split("\nUser:")[0].strip()

        assistant_msg = Message(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
        )
        self.conversation_history.append(assistant_msg)

        # Store this exchange in long-term memory
        self.store_memory(
            f"User asked: {user_input}. Assistant responded: {response[:300]}",
            metadata={"type": "conversation"},
        )

        # Update profile
        self.profile.interaction_count += 1
        self._update_interests(user_input)
        self._save_profile()

        return response

    def _update_interests(self, text: str):
        """Simple keyword-based interest tracking."""
        topics = {
            "technology": ["code", "software", "api", "programming", "python", "ai", "ml"],
            "finance": ["money", "invest", "budget", "stock", "crypto"],
            "health": ["exercise", "diet", "sleep", "fitness", "mental health"],
            "travel": ["trip", "vacation", "flight", "hotel", "destination"],
            "food": ["recipe", "cooking", "restaurant", "eat"],
        }
        lower = text.lower()
        for topic, keywords in topics.items():
            if any(kw in lower for kw in keywords):
                if topic not in self.profile.topics_of_interest:
                    self.profile.topics_of_interest.append(topic)

    def set_profile(self, name: str = None, preferences: dict = None):
        """Update user profile."""
        if name:
            self.profile.name = name
        if preferences:
            self.profile.preferences.update(preferences)
        self._save_profile()
        return f"Profile updated: {self.profile.name}"

    def get_memory_summary(self) -> dict:
        """Return a summary of what the assistant knows about the user."""
        return {
            "name": self.profile.name,
            "interests": self.profile.topics_of_interest,
            "preferences": self.profile.preferences,
            "total_interactions": self.profile.interaction_count,
            "long_term_memories": self.memory_store._collection.count(),
        }


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assistant = PersonalChatAssistant(user_id="laaz")
    assistant.set_profile(name="Laaz", preferences={"response_style": "concise", "language": "English"})

    conversations = [
        "Hi! Can you help me understand how transformers work?",
        "What are some good Python libraries for machine learning?",
        "Can you remind me what we discussed about transformers?",
    ]

    for msg in conversations:
        print(f"\n👤 User: {msg}")
        response = assistant.chat(msg)
        print(f"🤖 Assistant: {response}")

    print("\n📊 Memory Summary:", assistant.get_memory_summary())
