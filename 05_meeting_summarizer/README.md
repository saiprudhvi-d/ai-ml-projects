# 📋 Meeting Notes Auto Summarizer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai)
![BART](https://img.shields.io/badge/Facebook-BART-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)

> Automated summarization pipeline using **Whisper (speech-to-text) + BART (summarization)** — producing structured outputs for summaries, action items, blockers, and owner-based next steps.

## 📌 What It Does

- Transcribes meeting audio (MP3/WAV/M4A) using **OpenAI Whisper**
- Summarizes transcripts using **Facebook BART** with chunking for long meetings
- Extracts **action items** with owner attribution and priority detection
- Identifies **blockers** and dependencies mentioned in the meeting
- Groups output as **next steps per owner** — ready to paste into Jira/Notion

## 🏗️ Architecture

```
Audio File (MP3/WAV/M4A)    OR    Raw Transcript Text
        │                                  │
        ▼                                  │
  Whisper (speech-to-text)                 │
        │                                  │
        └──────────────────────────────────┘
                          │
                          ▼
              BART Summarization (chunked)
                          │
                          ▼
              Structured Extraction
         ┌────────────────┬───────────────┐
         ▼                ▼               ▼
    Summary         Action Items      Blockers
                   (owner + priority)
```

## 🚀 Quick Start

```bash
# System requirement
brew install ffmpeg   # macOS
apt install ffmpeg    # Linux

pip install -r requirements.txt
python api.py   # runs on port 8004
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/summarize_audio` | Upload audio file → structured notes |
| POST | `/summarize_transcript` | Process raw text transcript |
| POST | `/extract_actions` | Extract only action items + owners |

**Example:**
```bash
curl -X POST http://localhost:8004/summarize_transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript": "John will fix the auth bug by Friday. We are blocked on the payment gateway vendor..."}'
```

**Response:**
```json
{
  "summary": "Sprint planning covered auth bug fix and payment gateway blocker.",
  "action_items": [
    {"owner": "John", "action": "fix the auth bug by Friday", "priority": "high"}
  ],
  "blockers": ["blocked on payment gateway vendor response"],
  "next_steps": [
    {"owner": "John", "tasks": ["fix the auth bug by Friday"]}
  ]
}
```

## 🛠️ Tech Stack

`Python` · `OpenAI Whisper` · `BART/T5` · `HuggingFace` · `FastAPI` · `Streamlit` · `ffmpeg`
