"""
Demo Runner — All 8 AI Projects with mock inference responses.
Starts all 8 FastAPI servers on ports 8000-8007 in-process using threads.
Real model behavior is mocked so everything runs instantly with no GPU/disk required.
Swap mock functions for real model calls once you run on your own hardware.
"""

import threading
import time
import random
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ─── Shared Pydantic models ───────────────────────────────────────────────────
class TextReq(BaseModel):
    text: str = ""
    doc_id: str = "doc"
    source: str = "manual"
    category: str = ""

class QAReq(BaseModel):
    question: str
    context: str

class MsgReq(BaseModel):
    customer_message: str
    agent_context: str = ""

class ReviewsReq(BaseModel):
    reviews: list[str]

class TranscriptReq(BaseModel):
    transcript: str

class QueryReq(BaseModel):
    question: str
    top_k: int = 3

class ChatReq(BaseModel):
    user_id: str
    message: str

class SuggestReq(BaseModel):
    prefix: str
    max_tokens: int = 128
    temperature: float = 0.2
    num_suggestions: int = 3
    language: str = "python"

class ClauseReq(BaseModel):
    clause_type: str
    top_k: int = 3

# ─── Mock helpers ─────────────────────────────────────────────────────────────
EMOTIONS = ["anger","gratitude","confusion","fear","joy","disappointment","neutral","curiosity"]
THEMES = ["delivery","quality","customer_service","pricing","usability","features"]

def mock_latency():
    return round(random.uniform(40, 240), 1)

# ─── Project 01: Q&A Bot ──────────────────────────────────────────────────────
app01 = FastAPI(title="01 · Q&A Bot")

@app01.get("/health")
def h01(): return {"status":"ok","model":"DistilBERT (mock)"}

@app01.post("/answer")
def answer(req: QAReq):
    words = req.context.split()
    excerpt = " ".join(words[:8]) if words else "N/A"
    return {"answer": excerpt, "score": round(random.uniform(0.75, 0.97), 3),
            "start": 0, "end": len(excerpt)}

@app01.post("/batch_answer")
def batch_answer(items: list[QAReq]):
    return [{"question": i.question,
             "answer": " ".join(i.context.split()[:6]),
             "score": round(random.uniform(0.75, 0.97), 3)} for i in items]

# ─── Project 02: Contract Analysis ────────────────────────────────────────────
app02 = FastAPI(title="02 · Contract Analysis")
_ingested = {}

@app02.get("/health")
def h02(): return {"status":"ok","model":"LLaMA-7B+LoRA (mock)"}

@app02.post("/ingest")
def ingest02(req: TextReq):
    _ingested[req.doc_id] = req.text
    return {"status":"ingested","chunks":max(1,len(req.text)//512),"doc_id":req.doc_id}

@app02.post("/extract_clause")
def extract_clause(req: ClauseReq):
    return {"clause_type": req.clause_type,
            "extraction": f"[MOCK] Extracted {req.clause_type} clause: standard terms apply with 30-day notice period.",
            "source_chunks": 2}

@app02.post("/analyze_risk")
def risk(req: TextReq):
    text = req.text.lower()
    high = [w for w in ["unlimited liability","irrevocable","indemnify"] if w in text]
    medium = [w for w in ["auto-renew","sole discretion","unilateral"] if w in text]
    return {"risks":{"high":high,"medium":medium,"low":["best efforts" if "best efforts" in text else ""]}}

@app02.post("/summarize")
def summarize02(req: TextReq):
    return {"summary": f"[MOCK] Contract between two parties. Key terms: payment on net-30 basis, 1-year initial term with auto-renewal, standard confidentiality obligations, liability capped at 3 months fees."}

@app02.post("/full_analysis")
def full02(req: TextReq):
    ingest02(req)
    return {"doc_id": req.doc_id,
            "clauses": {t: f"Standard {t} clause found." for t in ["termination","payment","confidentiality","liability"]},
            "risks": {"high":[],"medium":["auto-renew"],"low":["best efforts"]}}

# ─── Project 03: Code Suggestion ──────────────────────────────────────────────
app03 = FastAPI(title="03 · Code Suggestion")
_code_cache = {}

@app03.get("/health")
def h03(): return {"status":"ok","model":"CodeLlama-7B (mock)"}

@app03.post("/suggest")
def suggest(req: SuggestReq):
    key = hash(req.prefix)
    cached = key in _code_cache
    suggestions = [
        f's: str) -> str:\n    return s[::-1]',
        f's: str) -> bool:\n    return len(s) > 0',
        f's: str) -> int:\n    return len(s)',
    ][:req.num_suggestions]
    _code_cache[key] = suggestions
    return {"suggestions": suggestions, "latency_ms": mock_latency(), "cached": cached}

@app03.post("/complete_function")
def complete_fn(req: SuggestReq):
    return {"suggestions": [f"s: str) -> str:\n    \"\"\"Auto-completed function.\"\"\"\n    return s.strip()"],
            "latency_ms": mock_latency(), "cached": False}

@app03.post("/explain")
def explain(req: SuggestReq):
    return {"suggestions": [f"[MOCK] This code performs string manipulation. It takes an input, processes it, and returns a result."],
            "latency_ms": mock_latency(), "cached": False}

@app03.delete("/cache")
def clear_cache03(): _code_cache.clear(); return {"status":"cache cleared"}

# ─── Project 04: Empathetic Response ──────────────────────────────────────────
app04 = FastAPI(title="04 · Empathetic Response")

TONE_MAP = {
    "anger":      "Respond calmly, acknowledge frustration, offer a resolution.",
    "gratitude":  "Acknowledge their kind words warmly.",
    "confusion":  "Clarify things clearly and patiently.",
    "fear":       "Reassure them with specific information.",
    "joy":        "Match their positive tone.",
    "neutral":    "Respond professionally and clearly.",
}

def detect_emotion(text: str) -> tuple[str, float]:
    t = text.lower()
    if any(w in t for w in ["angry","frustrat","unacceptable","terrible"]): return "anger", 0.91
    if any(w in t for w in ["thank","grateful","appreciate","great"]): return "gratitude", 0.88
    if any(w in t for w in ["confus","don't understand","unclear","help"]): return "confusion", 0.82
    if any(w in t for w in ["worried","concern","afraid","scared"]): return "fear", 0.79
    if any(w in t for w in ["happy","love","excellent","amazing"]): return "joy", 0.85
    return "neutral", 0.71

@app04.get("/health")
def h04(): return {"status":"ok","model":"RoBERTa+LLM (mock)"}

@app04.post("/respond")
def respond(req: MsgReq):
    emotion, conf = detect_emotion(req.customer_message)
    tone = TONE_MAP.get(emotion, TONE_MAP["neutral"])
    responses = {
        "anger": "I completely understand your frustration, and I sincerely apologize for this experience. Let me look into this right away and make sure we resolve it for you as quickly as possible.",
        "gratitude": "Thank you so much for your kind words! It's wonderful to hear that we were able to help. Please don't hesitate to reach out if you need anything else.",
        "confusion": "I'd be happy to walk you through this step by step. Let's start from the beginning — could you tell me exactly where you got stuck so I can guide you through it?",
        "fear": "I completely understand your concern, and I want to reassure you that we're on top of this. Here's exactly what's happening and what we'll do to fix it.",
        "joy": "That's wonderful to hear! We're so glad you're having a great experience. We're always here if you need anything!",
        "neutral": "Thank you for reaching out. I've reviewed your message and I'm happy to help. Could you provide a bit more detail so I can assist you most effectively?",
    }
    return {"customer_message": req.customer_message, "detected_emotion": emotion,
            "emotion_confidence": conf, "tone_guideline": tone,
            "generated_response": responses.get(emotion, responses["neutral"])}

@app04.post("/classify_only")
def classify(req: MsgReq):
    emotion, conf = detect_emotion(req.customer_message)
    return {"emotion": emotion, "confidence": conf}

@app04.post("/batch_respond")
def batch_respond(req: ReviewsReq):
    return [respond(MsgReq(customer_message=m)) for m in req.reviews]

# ─── Project 05: Meeting Summarizer ──────────────────────────────────────────
app05 = FastAPI(title="05 · Meeting Summarizer")

import re

def extract_actions(text: str) -> list[dict]:
    actions = []
    patterns = [r"(\w+)\s+will\s+([^.]+)", r"(\w+)\s+to\s+([^.]+\btomorrow\b[^.]*|[^.]+\btoday\b[^.]*)"]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            owner, action = m.group(1), m.group(2).strip()
            if len(action) > 5:
                actions.append({"owner": owner.title(), "action": action[:120],
                                 "priority": "high" if any(w in action.lower() for w in ["urgent","asap","today"]) else "normal"})
    return actions[:10]

def extract_blockers(text: str) -> list[str]:
    blockers = []
    for m in re.finditer(r"(?:blocked|waiting|stuck|can't|cannot)\s+(?:on\s+)?([^.]+)", text, re.IGNORECASE):
        blockers.append(m.group(0).strip()[:120])
    return blockers[:5]

@app05.get("/health")
def h05(): return {"status":"ok","model":"Whisper+BART (mock)"}

@app05.post("/summarize_transcript")
def summarize_transcript(req: TranscriptReq):
    sentences = [s.strip() for s in req.transcript.split(".") if len(s.strip()) > 20]
    summary = ". ".join(sentences[:3]) + "." if sentences else "Meeting covered sprint planning and team updates."
    actions = extract_actions(req.transcript)
    blockers = extract_blockers(req.transcript)
    by_owner = {}
    for a in actions:
        by_owner.setdefault(a["owner"], []).append(a["action"])
    next_steps = [{"owner": k, "tasks": v} for k, v in by_owner.items()]
    return {"summary": summary, "action_items": actions, "blockers": blockers,
            "next_steps": next_steps, "word_count": len(req.transcript.split()), "duration_seconds": None}

@app05.post("/extract_actions")
def extract_actions_ep(req: TranscriptReq):
    actions = extract_actions(req.transcript)
    by_owner = {}
    for a in actions:
        by_owner.setdefault(a["owner"], []).append(a["action"])
    return {"action_items": actions, "next_steps": [{"owner": k, "tasks": v} for k, v in by_owner.items()]}

# ─── Project 06: Review Intelligence ─────────────────────────────────────────
app06 = FastAPI(title="06 · Review Intelligence")

def score_review(text: str) -> tuple[str, float]:
    pos = sum(text.lower().count(w) for w in ["great","excellent","love","good","amazing","fast","helpful","perfect"])
    neg = sum(text.lower().count(w) for w in ["terrible","bad","broken","slow","awful","damaged","poor","late","waste"])
    if pos > neg: return "positive", round(0.7 + pos*0.05, 2)
    if neg > pos: return "negative", round(0.7 + neg*0.05, 2)
    return "neutral", round(random.uniform(0.6, 0.75), 2)

@app06.get("/health")
def h06(): return {"status":"ok","model":"RoBERTa+FlanT5 (mock)"}

@app06.post("/analyze")
def analyze_reviews(req: ReviewsReq):
    sentiments = [score_review(r) for r in req.reviews]
    from collections import Counter
    dist = dict(Counter(s[0] for s in sentiments))
    avg = sum((1 if s[0]=="positive" else -1 if s[0]=="negative" else 0) * s[1] for s in sentiments) / max(len(sentiments),1)
    theme_counts = {t: sum(1 for r in req.reviews if any(k in r.lower() for k in t.split("_"))) for t in THEMES}
    top_themes = sorted([{"theme":t,"count":c} for t,c in theme_counts.items() if c > 0], key=lambda x:-x["count"])
    neg_themes = [t for t,c in theme_counts.items() if c > 0 and avg < 0]
    recs = [f"[{t.upper()}] Improve {t.replace('_',' ')} based on negative feedback patterns." for t in neg_themes[:3]] or ["Maintain current quality across all areas."]
    critical = [r[:100] for r, s in zip(req.reviews, sentiments) if s[0]=="negative"][:5]
    return {"total_reviews": len(req.reviews), "avg_sentiment_score": round(avg,3),
            "sentiment_distribution": dist, "top_themes": top_themes[:6],
            "theme_sentiment": {t: round(avg * random.uniform(0.7,1.3),3) for t in THEMES},
            "recommendations": recs, "critical_issues": critical}

@app06.post("/sentiment_only")
def sentiment_only(req: ReviewsReq):
    return {"sentiments": [{"sentiment":s,"score":c,"numeric":c if s=="positive" else -c} for s,c in [score_review(r) for r in req.reviews]]}

@app06.post("/themes_only")
def themes_only(req: ReviewsReq):
    return {"themes": {t: sum(1 for r in req.reviews if t.split("_")[0] in r.lower()) for t in THEMES}}

# ─── Project 07: RAG Knowledge Base ──────────────────────────────────────────
app07 = FastAPI(title="07 · RAG Knowledge Base")
_kb: list[dict] = []

@app07.get("/health")
def h07(): return {"status":"ok","indexed_chunks":len(_kb),"cached_queries":0}

@app07.get("/stats")
def stats07(): return {"indexed_chunks":len(_kb),"cached_queries":0}

@app07.post("/ingest/text")
def ingest07(req: TextReq):
    chunks = max(1, len(req.text)//256)
    for i in range(chunks):
        _kb.append({"text": req.text[i*256:(i+1)*256], "source": req.source, "category": req.category})
    return {"chunks_ingested": chunks}

@app07.post("/query")
def query07(req: QueryReq):
    if not _kb:
        return {"question":req.question,"answer":"Knowledge base is empty. Please ingest documents first.","sources":[],"cached":False}
    q = req.question.lower()
    relevant = [c for c in _kb if any(w in c["text"].lower() for w in q.split() if len(w) > 3)][:req.top_k]
    if not relevant: relevant = _kb[:req.top_k]
    excerpt = relevant[0]["text"][:150] if relevant else ""
    answer = f"Based on the knowledge base: {excerpt}..."
    sources = [{"source":c["source"],"page":"","excerpt":c["text"][:100]} for c in relevant]
    return {"question":req.question,"answer":answer,"sources":sources,"cached":False}

@app07.post("/ingest/json")
def ingest_json07(body: dict):
    items = body.get("items", [])
    key = body.get("text_key", "content")
    n = 0
    for item in items:
        if key in item:
            _kb.append({"text": item[key], "source": item.get("source","json"), "category": item.get("category","")})
            n += 1
    return {"chunks_ingested": n}

@app07.post("/save_index")
def save07(): return {"status":"saved (mock)"}

@app07.post("/load_index")
def load07(): return {"status":"loaded (mock)", "indexed_chunks": len(_kb)}

# ─── Project 08: Personal Assistant ──────────────────────────────────────────
app08 = FastAPI(title="08 · Personal Assistant")
_profiles: dict[str, dict] = {}
_histories: dict[str, list] = {}
_memories: dict[str, list] = {}

RESPONSES = [
    "That's a great point! Based on what I know about you, I'd suggest focusing on the most impactful items first.",
    "I remember you've mentioned this topic before. Here's what I think would help most in your situation.",
    "Absolutely! Let me help you with that. Given your preferences, here's my recommendation.",
    "Thanks for sharing that with me. I'll keep this in mind for our future conversations.",
    "Great question! From our previous discussions, I can see this connects to what you mentioned earlier.",
]

@app08.get("/health")
def h08(): return {"status":"ok","active_sessions":len(_profiles)}

@app08.post("/chat")
def chat08(req: ChatReq):
    if req.user_id not in _profiles:
        _profiles[req.user_id] = {"name": req.user_id, "interaction_count": 0, "interests": []}
    _profiles[req.user_id]["interaction_count"] += 1
    _histories.setdefault(req.user_id, []).append({"role":"user","content":req.message,"timestamp":""})
    _memories.setdefault(req.user_id, []).append(req.message[:100])
    response = random.choice(RESPONSES)
    _histories[req.user_id].append({"role":"assistant","content":response,"timestamp":""})
    return {"user_id": req.user_id, "message": req.message,
            "response": response, "interaction_count": _profiles[req.user_id]["interaction_count"]}

@app08.post("/profile")
def profile08(body: dict):
    uid = body.get("user_id","default")
    if uid not in _profiles: _profiles[uid] = {"name":uid,"interaction_count":0,"interests":[]}
    if body.get("name"): _profiles[uid]["name"] = body["name"]
    return {"status": f"Profile updated for {_profiles[uid]['name']}"}

@app08.get("/memory/{user_id}")
def memory08(user_id: str):
    p = _profiles.get(user_id, {})
    return {"name": p.get("name",user_id), "interests": p.get("interests",[]),
            "preferences": {}, "total_interactions": p.get("interaction_count",0),
            "long_term_memories": len(_memories.get(user_id,[]))}

@app08.get("/history/{user_id}")
def history08(user_id: str, last_n: int = 10):
    return {"history": _histories.get(user_id,[])[-last_n:]}

@app08.delete("/session/{user_id}")
def clear08(user_id: str):
    _profiles.pop(user_id, None); _histories.pop(user_id, None); _memories.pop(user_id, None)
    return {"status":"session cleared"}

# ─── Launch all servers ───────────────────────────────────────────────────────
SERVERS = [
    (app01, 8000), (app02, 8001), (app03, 8002), (app04, 8003),
    (app05, 8004), (app06, 8005), (app07, 8006), (app08, 8007),
]

def run_server(app, port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  🚀  All 8 AI Projects — Demo Mode           ║")
    print("╠══════════════════════════════════════════════╣")
    threads = []
    for app, port in SERVERS:
        t = threading.Thread(target=run_server, args=(app, port), daemon=True)
        t.start()
        threads.append(t)
        print(f"║  ▶  {app.title:30s} :{ port}  ║")
        time.sleep(0.3)

    print("╚══════════════════════════════════════════════╝")
    print("\n✅ All services running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
