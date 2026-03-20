"""
Project 08: Personal Chat Assistant — FastAPI Endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from assistant import PersonalChatAssistant
import uvicorn

app = FastAPI(title="Personal Chat Assistant", version="1.0")

# In-memory sessions: {user_id: PersonalChatAssistant}
_sessions: dict[str, PersonalChatAssistant] = {}


def get_or_create(user_id: str) -> PersonalChatAssistant:
    if user_id not in _sessions:
        _sessions[user_id] = PersonalChatAssistant(user_id=user_id)
    return _sessions[user_id]


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ProfileRequest(BaseModel):
    user_id: str
    name: str = None
    preferences: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message is required")
    bot = get_or_create(req.user_id)
    response = bot.chat(req.message)
    return {
        "user_id": req.user_id,
        "message": req.message,
        "response": response,
        "interaction_count": bot.profile.interaction_count,
    }


@app.post("/profile")
def update_profile(req: ProfileRequest):
    bot = get_or_create(req.user_id)
    result = bot.set_profile(name=req.name, preferences=req.preferences)
    return {"status": result}


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    bot = get_or_create(user_id)
    return bot.get_memory_summary()


@app.get("/history/{user_id}")
def get_history(user_id: str, last_n: int = 10):
    bot = get_or_create(user_id)
    history = bot.conversation_history[-last_n:]
    return {"history": [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in history]}


@app.delete("/session/{user_id}")
def clear_session(user_id: str):
    if user_id in _sessions:
        del _sessions[user_id]
    return {"status": "session cleared"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8007, reload=True)
