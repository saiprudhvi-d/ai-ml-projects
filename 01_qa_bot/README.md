# 🤖 Domain-Specific Q&A Bot

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-orange?logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

> Fine-tuned DistilBERT on 10K+ Q&A pairs achieving **92% test accuracy** on FAQ-style support workflows, served via a production-ready FastAPI inference endpoint.

## 📌 What It Does

- Fine-tunes **DistilBERT** (extractive QA) on domain-specific FAQ data in SQuAD format
- Exposes a **FastAPI REST API** for single and batch question answering
- Includes a **Streamlit UI** for interactive testing
- Supports **sliding window** tokenization for long documents

## 🏗️ Architecture

```
User Question + Context
        │
        ▼
  DistilBERT (fine-tuned)
        │
        ▼
  Answer Span Extraction
        │
        ▼
  FastAPI Response (answer + confidence score)
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Option 1: Use base model (no training needed)
python api.py

# Option 2: Fine-tune on your own data first
python train.py          # place SQuAD-format JSON at data/train.json
python api.py            # serve your fine-tuned model
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/answer` | Single Q&A pair |
| POST | `/batch_answer` | Bulk Q&A processing |

**Example request:**
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the return window?",
    "context": "Our return policy allows returns within 30 days of purchase."
  }'
```

**Example response:**
```json
{
  "answer": "30 days",
  "score": 0.9234,
  "start": 42,
  "end": 49
}
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **92%** |
| Inference Latency | ~80ms (CPU) |
| Model Size | ~250MB |
| Dataset | 10K+ Q&A pairs |

## 🛠️ Tech Stack

`Python` · `DistilBERT` · `HuggingFace Transformers` · `FastAPI` · `Streamlit` · `Scikit-learn`
