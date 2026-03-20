# 📄 Contract Analysis AI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LLaMA](https://img.shields.io/badge/LLaMA--7B-LoRA%2FPEFT-purple)
![LangChain](https://img.shields.io/badge/LangChain-FAISS-yellow?logo=chainlink)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)

> Fine-tuned **LLaMA-7B with LoRA/PEFT** on 5K+ legal documents for automated clause extraction and summarization — reducing manual contract review effort by **40%**.

## 📌 What It Does

- Fine-tunes **LLaMA-2-7B** using **LoRA adapters** (4-bit quantized, runs on consumer GPUs)
- Extracts specific clause types: termination, payment, confidentiality, liability, IP, and more
- Performs **risk analysis** — flags high/medium/low risk language automatically
- Semantic search over ingested contracts using **LangChain + FAISS**
- Accepts raw text or **PDF uploads**

## 🏗️ Architecture

```
Contract (Text / PDF)
        │
        ▼
  Text Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
  FAISS Vector Store (sentence-transformers embeddings)
        │
        ▼
  LLaMA-7B + LoRA (clause extraction / summarization)
        │
        ▼
  Structured Output: clauses + risk flags + summary
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Fine-tune (requires HuggingFace access to meta-llama/Llama-2-7b-hf)
export HF_TOKEN=your_token_here
python train_lora.py

# Start API
python api.py   # runs on port 8001
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Add contract text to vector store |
| POST | `/ingest_pdf` | Upload a PDF contract |
| POST | `/extract_clause` | Extract a specific clause type |
| POST | `/analyze_risk` | Flag risky contract language |
| POST | `/summarize` | Generate structured contract summary |
| POST | `/full_analysis` | All-in-one: ingest + extract + risk |

**Supported clause types:** `termination` · `payment` · `confidentiality` · `liability` · `indemnification` · `governing_law` · `intellectual_property` · `dispute_resolution`

**Example:**
```bash
curl -X POST http://localhost:8001/analyze_risk \
  -H "Content-Type: application/json" \
  -d '{"text": "Vendor liability is unlimited. Client may terminate at sole discretion..."}'
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Manual review reduction | **40%** |
| Training dataset | 5K+ legal documents |
| LoRA trainable params | ~4M (vs 7B base) |
| GPU memory (4-bit) | ~8GB VRAM |

## 🛠️ Tech Stack

`Python` · `LLaMA 2/3` · `LoRA/PEFT` · `LangChain` · `FAISS` · `FastAPI` · `BitsAndBytes`
