"""
Project 06: Review Intelligence Engine — FastAPI Endpoint
"""

import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pipeline import ReviewIntelligenceEngine
import pandas as pd
import uvicorn

app = FastAPI(title="Review Intelligence Engine", version="1.0")
engine = ReviewIntelligenceEngine()


class ReviewsRequest(BaseModel):
    reviews: list[str]


def analysis_to_dict(a) -> dict:
    return {
        "total_reviews": a.total_reviews,
        "avg_sentiment_score": a.avg_sentiment_score,
        "sentiment_distribution": a.sentiment_distribution,
        "top_themes": a.top_themes,
        "theme_sentiment": a.theme_sentiment,
        "recommendations": a.recommendations,
        "critical_issues": a.critical_issues,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: ReviewsRequest):
    if not req.reviews:
        raise HTTPException(400, "reviews list is empty")
    result = engine.analyze(req.reviews)
    return analysis_to_dict(result)


@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...), text_column: str = "review_text"):
    """Upload a CSV with a review text column for bulk analysis."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if text_column not in df.columns:
        raise HTTPException(400, f"Column '{text_column}' not found. Available: {list(df.columns)}")
    result = engine.analyze_dataframe(df, text_column)
    return analysis_to_dict(result)


@app.post("/sentiment_only")
def sentiment_only(req: ReviewsRequest):
    sentiments = engine.analyze_sentiment(req.reviews)
    return {"sentiments": sentiments}


@app.post("/themes_only")
def themes_only(req: ReviewsRequest):
    themes = engine.extract_themes(req.reviews)
    return {"themes": {k: len(v) for k, v in themes.items()}}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8005, reload=True)
