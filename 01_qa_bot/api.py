"""
Project 01: Domain-Specific Q&A Bot — FastAPI Inference Endpoint
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os

app = FastAPI(title="Domain Q&A Bot", version="1.0")

MODEL_DIR = os.getenv("MODEL_DIR", "./model_output")

# Load pipeline once at startup
qa_pipeline = None

@app.on_event("startup")
async def load_model():
    global qa_pipeline
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
        )
        print(f"✅ Model loaded from {MODEL_DIR}")
    except Exception as e:
        # Fallback to base model for demo
        print(f"⚠️  Custom model not found ({e}). Using base DistilBERT.")
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
        )


class QARequest(BaseModel):
    question: str
    context: str


class QAResponse(BaseModel):
    answer: str
    score: float
    start: int
    end: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": qa_pipeline is not None}


@app.post("/answer", response_model=QAResponse)
def answer_question(req: QARequest):
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.question.strip() or not req.context.strip():
        raise HTTPException(status_code=400, detail="question and context must not be empty")

    result = qa_pipeline(question=req.question, context=req.context)
    return QAResponse(
        answer=result["answer"],
        score=round(result["score"], 4),
        start=result["start"],
        end=result["end"],
    )


@app.post("/batch_answer")
def batch_answer(items: list[QARequest]):
    """Answer multiple Q&A pairs in one call."""
    if qa_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for item in items:
        r = qa_pipeline(question=item.question, context=item.context)
        results.append({
            "question": item.question,
            "answer": r["answer"],
            "score": round(r["score"], 4),
        })
    return results


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
