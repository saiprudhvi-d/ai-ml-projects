# ⭐ Review Intelligence Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![RoBERTa](https://img.shields.io/badge/RoBERTa-Sentiment-orange)
![FlanT5](https://img.shields.io/badge/Flan--T5-Recommendations-green)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blue?logo=plotly)

> NLP pipeline combining **sentiment analysis**, **theme extraction**, and **LLM-generated recommendations** from large-scale customer review datasets.

## 📌 What It Does

- Runs **RoBERTa sentiment analysis** across thousands of reviews (positive/negative/neutral)
- Extracts recurring **themes** (delivery, quality, pricing, customer service, etc.)
- Computes **per-theme sentiment scores** to identify exactly what's driving dissatisfaction
- Uses **Flan-T5** to generate actionable product improvement recommendations
- Surfaces **critical issues** — the most negative reviews ranked by severity
- Accepts CSV upload for bulk analysis

## 🏗️ Architecture

```
Customer Reviews (list or CSV)
        │
        ▼
  RoBERTa Sentiment Analysis (batch)
        │
        ▼
  Theme Extraction (keyword + embedding matching)
        │
        ▼
  Per-Theme Sentiment Aggregation
        │
        ▼
  Flan-T5 Recommendation Generation
        │
        ▼
  Structured Report: sentiment + themes + recommendations + critical issues
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python api.py   # runs on port 8005
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Full pipeline on a list of reviews |
| POST | `/analyze_csv` | Upload CSV file for bulk analysis |
| POST | `/sentiment_only` | Just sentiment scores |
| POST | `/themes_only` | Just theme extraction |

**Example:**
```bash
curl -X POST http://localhost:8005/analyze \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product, fast delivery!", "Terrible service, item was broken on arrival.", "Price is high but quality is good."]}'
```

**Response:**
```json
{
  "total_reviews": 3,
  "avg_sentiment_score": 0.12,
  "sentiment_distribution": {"positive": 2, "negative": 1},
  "top_themes": [{"theme": "quality", "count": 2}, {"theme": "delivery", "count": 1}],
  "recommendations": ["[QUALITY] Address reports of items arriving damaged..."],
  "critical_issues": ["Terrible service, item was broken on arrival."]
}
```

## 🛠️ Tech Stack

`Python` · `BERT/RoBERTa` · `LLMs (Flan-T5)` · `Pandas` · `Scikit-learn` · `Plotly` · `FastAPI`
