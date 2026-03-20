# 💬 Empathetic Response AI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![RoBERTa](https://img.shields.io/badge/RoBERTa-28_Emotions-orange)
![LLM](https://img.shields.io/badge/LLM-Tone_Conditioned-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)

> A two-stage pipeline: **RoBERTa emotion classification** (28 categories) → **tone-conditioned LLM response generation**, aligned to customer support standards.

## 📌 What It Does

- **Stage 1:** Fine-tunes RoBERTa on GoEmotions (28 emotion categories) to detect customer sentiment
- **Stage 2:** Uses detected emotion to condition LLM response tone (calm for anger, warm for gratitude, etc.)
- Processes customer messages in real-time with empathy-aware responses
- Supports batch processing for large support queues

## 🏗️ Architecture

```
Customer Message
        │
        ▼
  Stage 1: RoBERTa Emotion Classifier
  (28 emotions: anger, gratitude, confusion, fear...)
        │
        ▼
  Tone Instruction Mapping
  e.g. anger → "Acknowledge frustration, offer resolution"
        │
        ▼
  Stage 2: LLM Response Generator
  (conditioned on tone + agent context)
        │
        ▼
  Empathetic, on-brand support response
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train emotion classifier
python emotion_classifier.py

# Start API
python api.py   # runs on port 8003
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/respond` | Full pipeline: classify + generate response |
| POST | `/classify_only` | Just emotion detection |
| POST | `/batch_respond` | Process multiple messages |

**Example:**
```bash
curl -X POST http://localhost:8003/respond \
  -H "Content-Type: application/json" \
  -d '{
    "customer_message": "I have been waiting 3 weeks! This is completely unacceptable!",
    "agent_context": "Order delayed due to supplier issue"
  }'
```

**Response:**
```json
{
  "detected_emotion": "anger",
  "emotion_confidence": 0.91,
  "tone_guideline": "Respond calmly, acknowledge frustration, offer a resolution.",
  "generated_response": "I completely understand your frustration and sincerely apologize..."
}
```

## 📊 Emotion Categories

The classifier covers all 28 GoEmotions categories, grouped for support use cases:

| Group | Emotions |
|-------|---------|
| Negative | anger, annoyance, disappointment, disapproval, disgust |
| Concern | fear, nervousness, sadness, grief, remorse |
| Positive | joy, gratitude, admiration, love, optimism, pride |
| Neutral | neutral, curiosity, surprise, realization, confusion |

## 🛠️ Tech Stack

`Python` · `BERT/RoBERTa` · `LLMs` · `FastAPI` · `NLP Preprocessing` · `GoEmotions Dataset`
