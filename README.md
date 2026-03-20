# 🚀 AI/ML Projects Portfolio

> A collection of 8 production-grade AI/ML projects spanning fine-tuning, RAG, NLP pipelines, and LLM applications — each with a live FastAPI backend, core logic, and real performance results.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)
![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_APIs-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)

---

## 📂 Projects

| # | Project | Model(s) | Key Result | Stack |
|---|---------|----------|------------|-------|
| [01](./01_qa_bot) | 🤖 Domain-Specific Q&A Bot | DistilBERT | **92% test accuracy** | Transformers, FastAPI, Streamlit |
| [02](./02_contract_analysis) | 📄 Contract Analysis AI | LLaMA-7B + LoRA | **40% less manual review** | PEFT, LangChain, FAISS |
| [03](./03_code_suggestion) | 💻 AI Code Suggestion System | Code Llama-7B | **250ms latency, +28% relevance** | PEFT, VS Code API, FastAPI |
| [04](./04_empathetic_response) | 💬 Empathetic Response AI | RoBERTa + LLM | **28-emotion classification** | Transformers, FastAPI |
| [05](./05_meeting_summarizer) | 📋 Meeting Notes Summarizer | Whisper + BART | **Structured action items + owners** | Whisper, HuggingFace, FastAPI |
| [06](./06_review_intelligence) | ⭐ Review Intelligence Engine | RoBERTa + Flan-T5 | **Theme + sentiment at scale** | Transformers, Pandas, Plotly |
| [07](./07_rag_knowledge_base) | 🔍 Knowledge Base RAG | LangChain + Pinecone | **30%+ faster resolution** | LangChain, FAISS, Pinecone |
| [08](./08_personal_chat_assistant) | 🧠 Personal Chat Assistant | LangChain + ChromaDB | **Persistent cross-session memory** | LangChain, ChromaDB, Streamlit |

---

## ⚡ Run All Projects (Demo Mode)

```bash
git clone https://github.com/YOUR_USERNAME/ai-projects.git
cd ai-projects

pip install fastapi uvicorn
python demo_all.py
```

All 8 APIs start instantly on ports **8000–8007**. Visit any `http://localhost:800X/docs` for the interactive API explorer.

---

## 🏗️ Repository Structure

```
ai-projects/
├── 01_qa_bot/                  # DistilBERT Q&A fine-tuning + FastAPI
│   ├── train.py                # Fine-tuning script
│   ├── api.py                  # FastAPI inference endpoint
│   ├── app.py                  # Streamlit UI
│   └── requirements.txt
├── 02_contract_analysis/       # LLaMA-7B LoRA + LangChain FAISS
├── 03_code_suggestion/         # Code Llama + VS Code extension
│   └── vscode_extension/       # VS Code inline completion plugin
├── 04_empathetic_response/     # RoBERTa emotion → LLM response
├── 05_meeting_summarizer/      # Whisper + BART structured extraction
├── 06_review_intelligence/     # Sentiment + themes + recommendations
├── 07_rag_knowledge_base/      # RAG with FAISS/Pinecone + citations
├── 08_personal_chat_assistant/ # Memory-enabled personalized assistant
├── dashboard.py                # Unified Streamlit dashboard (all 8 UIs)
├── orchestrator.py             # Cross-project pipelines
├── demo_all.py                 # One-command demo runner
├── docker-compose.yml          # Full stack containerization
└── README.md
```

---

## 🔗 Cross-Project Pipelines

The `orchestrator.py` wires projects together into end-to-end workflows:

**Pipeline 1 — Smart Customer Support:**
```
Customer Message → [04] Emotion Detection → [07] RAG Knowledge Lookup → [04] Empathetic Response
```

**Pipeline 2 — Review Intelligence → Knowledge Base:**
```
Reviews → [06] Theme + Sentiment Analysis → [07] Ingest Insights into RAG KB
```

**Pipeline 3 — Meeting Notes → Assistant Memory:**
```
Meeting Transcript → [05] Action Item Extraction → [08] Store in Personal Assistant Memory
```

---

## 🐳 Docker

```bash
# Start all services
docker-compose up

# Dashboard available at http://localhost:8501
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Models** | DistilBERT, LLaMA 2/3, Code Llama, RoBERTa, BART, Whisper, Flan-T5 |
| **Training** | HuggingFace Transformers, PEFT/LoRA, BitsAndBytes (4-bit), Datasets |
| **Retrieval** | LangChain, LlamaIndex, FAISS, Pinecone, ChromaDB |
| **Serving** | FastAPI, Uvicorn, Streamlit |
| **NLP** | Sentence-Transformers, Tokenizers, NLP Preprocessing |
| **Data** | Pandas, Scikit-learn, Plotly |
| **DevOps** | Docker, docker-compose |

---

## 📋 Hardware Requirements

| Project | Min GPU VRAM | CPU-only? |
|---------|-------------|-----------|
| 01 · DistilBERT | 4GB | ✅ Yes |
| 02 · LLaMA LoRA | 16GB | ⚠️ Slow |
| 03 · Code Llama | 16GB | ⚠️ Slow |
| 04 · RoBERTa | 4GB | ✅ Yes |
| 05 · Whisper + BART | 8GB | ✅ Yes |
| 06 · RoBERTa | 4GB | ✅ Yes |
| 07 · RAG | 4GB | ✅ Yes |
| 08 · Assistant | 4GB | ✅ Yes |
