"""
Project 05: Meeting Notes Auto Summarizer — FastAPI + Streamlit
"""

import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pipeline import MeetingSummarizer
import uvicorn

app = FastAPI(title="Meeting Notes Summarizer", version="1.0")
summarizer = MeetingSummarizer()


class TranscriptRequest(BaseModel):
    transcript: str


def summary_to_dict(s) -> dict:
    return {
        "summary": s.summary,
        "action_items": s.action_items,
        "blockers": s.blockers,
        "next_steps": s.next_steps,
        "word_count": s.word_count,
        "duration_seconds": s.duration_seconds,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize_audio")
async def summarize_audio(file: UploadFile = File(...)):
    """Upload an audio file (mp3/wav/m4a) and get structured meeting notes."""
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use: {allowed}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = summarizer.process_audio(tmp_path)
        return JSONResponse(summary_to_dict(result))
    finally:
        os.unlink(tmp_path)


@app.post("/summarize_transcript")
def summarize_transcript(req: TranscriptRequest):
    """Submit raw transcript text and get structured meeting notes."""
    if not req.transcript.strip():
        raise HTTPException(400, "transcript is required")
    result = summarizer.process_text(req.transcript)
    return summary_to_dict(result)


@app.post("/extract_actions")
def extract_actions(req: TranscriptRequest):
    """Extract only action items from a transcript."""
    items = summarizer.extract_action_items(req.transcript)
    next_steps = summarizer.extract_next_steps(items)
    return {"action_items": items, "next_steps": next_steps}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8004, reload=True)
