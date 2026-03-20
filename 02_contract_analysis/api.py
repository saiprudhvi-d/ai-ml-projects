"""
Project 02: Contract Analysis AI — FastAPI Endpoint
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pipeline import ContractAnalysisPipeline, CLAUSE_TYPES
import uvicorn

app = FastAPI(title="Contract Analysis AI", version="1.0")
analyzer = ContractAnalysisPipeline()


class TextRequest(BaseModel):
    text: str
    doc_id: str = "contract"


class ClauseRequest(BaseModel):
    clause_type: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "supported_clauses": CLAUSE_TYPES}


@app.post("/ingest")
def ingest_text(req: TextRequest):
    n = analyzer.ingest_contract(req.text, req.doc_id)
    return {"status": "ingested", "chunks": n, "doc_id": req.doc_id}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        from pypdf import PdfReader
        import io
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() for page in reader.pages)
        n = analyzer.ingest_contract(text, doc_id=file.filename)
        return {"status": "ingested", "chunks": n, "pages": len(reader.pages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_clause")
def extract_clause(req: ClauseRequest):
    if req.clause_type not in CLAUSE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown clause type. Choose from: {CLAUSE_TYPES}",
        )
    return analyzer.extract_clause(req.clause_type, req.top_k)


@app.post("/summarize")
def summarize(req: TextRequest):
    summary = analyzer.summarize_contract(req.text)
    return {"summary": summary}


@app.post("/analyze_risk")
def analyze_risk(req: TextRequest):
    risks = analyzer.analyze_risk(req.text)
    return {"risks": risks, "doc_id": req.doc_id}


@app.post("/full_analysis")
def full_analysis(req: TextRequest):
    """Run clause extraction + risk analysis in one call."""
    analyzer.ingest_contract(req.text, req.doc_id)
    clauses = {ct: analyzer.extract_clause(ct)["extraction"] for ct in CLAUSE_TYPES[:4]}
    risks = analyzer.analyze_risk(req.text)
    return {"doc_id": req.doc_id, "clauses": clauses, "risks": risks}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
