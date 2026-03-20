"""
Project 04: Empathetic Response AI — FastAPI Endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from response_generator import EmpatheticPipeline
import uvicorn

app = FastAPI(title="Empathetic Response AI", version="1.0")
bot = EmpatheticPipeline()


class MessageRequest(BaseModel):
    customer_message: str
    agent_context: str = ""


class BatchRequest(BaseModel):
    messages: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond")
def respond(req: MessageRequest):
    if not req.customer_message.strip():
        raise HTTPException(400, "customer_message is required")
    return bot.generate_response(req.customer_message, req.agent_context)


@app.post("/batch_respond")
def batch_respond(req: BatchRequest):
    if not req.messages:
        raise HTTPException(400, "messages list is empty")
    return bot.batch_generate(req.messages)


@app.post("/classify_only")
def classify_only(req: MessageRequest):
    emotion, confidence = bot.classify_emotion(req.customer_message)
    return {"emotion": emotion, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)
