"""
Project 07: Knowledge Base RAG — FastAPI Endpoint
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_pipeline import KnowledgeBaseRAG
import uvicorn

app = FastAPI(title="Knowledge Base RAG", version="1.0")
rag = KnowledgeBaseRAG()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class TextIngestionRequest(BaseModel):
    text: str
    source: str = "manual"
    category: str = ""


class JsonIngestionRequest(BaseModel):
    items: list[dict]
    text_key: str = "content"
    meta_keys: list[str] = []


def result_to_dict(r) -> dict:
    return {
        "question": r.question,
        "answer": r.answer,
        "sources": r.sources,
        "cached": r.cached,
    }


@app.get("/health")
def health():
    return {"status": "ok", **rag.get_stats()}


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    result = rag.query(req.question, req.top_k)
    return result_to_dict(result)


@app.post("/ingest/text")
def ingest_text(req: TextIngestionRequest):
    n = rag.ingest_text(req.text, metadata={"source": req.source, "category": req.category})
    return {"chunks_ingested": n}


@app.post("/ingest/json")
def ingest_json_data(req: JsonIngestionRequest):
    import json, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req.items, f)
        tmp = f.name
    try:
        n = rag.ingest_json(tmp, text_key=req.text_key, meta_keys=req.meta_keys)
        return {"chunks_ingested": n}
    finally:
        os.unlink(tmp)


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    import tempfile, os
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        n = rag.ingest_pdf(tmp_path)
        return {"chunks_ingested": n, "filename": file.filename}
    finally:
        os.unlink(tmp_path)


@app.post("/save_index")
def save_index():
    rag.save_index()
    return {"status": "saved"}


@app.post("/load_index")
def load_index():
    rag.load_index()
    return {"status": "loaded", **rag.get_stats()}


@app.get("/stats")
def stats():
    return rag.get_stats()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8006, reload=True)
