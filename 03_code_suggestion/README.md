# 💻 AI-Powered Code Suggestion System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CodeLlama](https://img.shields.io/badge/Code_Llama--7B-Fine--tuned-red)
![VSCode](https://img.shields.io/badge/VS_Code-Extension-007ACC?logo=visualstudiocode)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)

> Fine-tuned **Code Llama-7B** on 50K+ Python functions achieving a **28% relevance improvement**, with inference latency reduced from 800ms to **250ms** via caching — served as a VS Code extension.

## 📌 What It Does

- Fine-tunes **Code Llama-7B** on Python function completion (CodeSearchNet dataset)
- Provides **inline code completions** in VS Code with <250ms latency
- **Caches** repeated completions for near-instant responses
- Supports function completion, code explanation, and multi-suggestion ranking
- Includes a full **VS Code extension** with keyboard shortcuts

## 🏗️ Architecture

```
VS Code (typing...)
        │
        ▼
  Extension (debounced keystroke)
        │  HTTP POST /suggest
        ▼
  FastAPI Server (port 8002)
        │
        ▼
  Cache check → hit: <5ms
        │ miss
        ▼
  Code Llama-7B (LoRA fine-tuned)
        │
        ▼
  Top-3 completions ranked by relevance → 250ms
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Fine-tune on Python functions
python train.py

# Start inference API
python api.py   # runs on port 8002
```

**Install VS Code Extension:**
```bash
# Copy extension folder to VS Code extensions directory
cp -r vscode_extension ~/.vscode/extensions/ai-code-suggest

# Reload VS Code — completions appear automatically in .py files
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/suggest` | Get N code completions with latency |
| POST | `/complete_function` | Full function completion |
| POST | `/explain` | Plain-English code explanation |
| DELETE | `/cache` | Clear completion cache |

**Example:**
```bash
curl -X POST http://localhost:8002/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "def binary_search(arr, target",
    "num_suggestions": 3,
    "temperature": 0.2
  }'
```

## 📊 Results

| Metric | Before | After |
|--------|--------|-------|
| Relevance score | baseline | **+28%** |
| Inference latency | 800ms | **250ms** |
| Cache hit latency | — | **<5ms** |
| Training data | — | 50K Python functions |

## 🛠️ Tech Stack

`Python` · `Code Llama` · `CodeT5` · `VS Code API` · `FastAPI` · `Tokenizers` · `PEFT`
