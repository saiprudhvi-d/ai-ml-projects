"""
Orchestrator — Connects all 8 AI Projects into end-to-end pipelines.

Available Pipelines:
  1. support_pipeline(message)     → Emotion → Empathetic Response + RAG answer
  2. review_pipeline(reviews)      → Reviews → Insights → RAG ingestion
  3. meeting_pipeline(transcript)  → Meeting → Summary → Assistant Memory
  4. code_review_pipeline(code)    → Code → Explain → Q&A context
  5. full_support_pipeline(msg)    → Combines all relevant projects for support

Run standalone: python orchestrator.py
Or import:      from orchestrator import Orchestrator
"""

import requests
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

# On Windows, the default console encoding can be CP1252; this file prints
# unicode/emoji symbols in CLI output. Force UTF-8 (best-effort) so execution
# doesn't crash with UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─── API base URLs ─────────────────────────────────────────────────────────────
SERVICES = {
    "qa":        "http://localhost:8000",
    "contract":  "http://localhost:8001",
    "code":      "http://localhost:8002",
    "empathetic":"http://localhost:8003",
    "meeting":   "http://localhost:8004",
    "reviews":   "http://localhost:8005",
    "rag":       "http://localhost:8006",
    "assistant": "http://localhost:8007",
}


@dataclass
class PipelineResult:
    pipeline: str
    steps: list[dict] = field(default_factory=list)
    final_output: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    def add_step(self, name: str, result: dict, ok: bool = True):
        self.steps.append({"step": name, "ok": ok, "result": result})

    def summary(self) -> str:
        lines = [f"Pipeline: {self.pipeline}"]
        for s in self.steps:
            status = "✅" if s["ok"] else "❌"
            lines.append(f"  {status} {s['step']}")
        if self.final_output:
            lines.append(f"  → Output: {self.final_output[:200]}")
        if self.errors:
            lines.append(f"  ⚠️  Errors: {'; '.join(self.errors)}")
        return "\n".join(lines)


class Orchestrator:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def _post(self, service: str, path: str, body: dict) -> tuple[bool, dict]:
        try:
            r = requests.post(
                f"{SERVICES[service]}{path}", json=body, timeout=self.timeout
            )
            if r.ok:
                return True, r.json()
            return False, {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except requests.ConnectionError:
            port_map = {'qa':8000,'contract':8001,'code':8002,'empathetic':8003,'meeting':8004,'reviews':8005,'rag':8006,'assistant':8007}
            return False, {"error": f"Service '{service}' is offline (:{port_map[service]})"}
        except Exception as e:
            return False, {"error": str(e)}

    # ─── Pipeline 1: Smart Customer Support ───────────────────────────────────
    def support_pipeline(
        self, customer_message: str, context_doc: str = "", user_id: str = "user"
    ) -> PipelineResult:
        """
        customer_message → emotion detection → RAG knowledge lookup → empathetic response
        Steps: Empathetic(classify) + RAG(query) + Empathetic(respond)
        """
        result = PipelineResult(pipeline="Smart Customer Support")

        # Step 1: Classify emotion
        ok, data = self._post("empathetic", "/classify_only", {"customer_message": customer_message})
        result.add_step("Emotion Classification", data, ok)
        emotion = data.get("emotion", "neutral") if ok else "neutral"

        # Step 2: Retrieve relevant knowledge
        ok2, rag_data = self._post("rag", "/query", {"question": customer_message, "top_k": 3})
        result.add_step("RAG Knowledge Lookup", rag_data, ok2)
        rag_answer = rag_data.get("answer", "") if ok2 else ""
        sources = [s["source"] for s in rag_data.get("sources", [])] if ok2 else []

        # Step 3: Generate empathetic response enriched with RAG context
        agent_context = f"Relevant knowledge: {rag_answer}" if rag_answer else context_doc
        ok3, resp_data = self._post("empathetic", "/respond", {
            "customer_message": customer_message,
            "agent_context": agent_context,
        })
        result.add_step("Empathetic Response Generation", resp_data, ok3)

        response = resp_data.get("generated_response", "") if ok3 else "Unable to generate response."
        result.final_output = response

        if sources:
            result.final_output += f"\n\n📚 Based on: {', '.join(sources)}"

        return result

    # ─── Pipeline 2: Reviews → Knowledge Base ──────────────────────────────────
    def review_pipeline(self, reviews: list[str]) -> PipelineResult:
        """
        reviews → analysis → extract issues → ingest insights into RAG KB
        """
        result = PipelineResult(pipeline="Review Intelligence → Knowledge Base")

        # Step 1: Analyze reviews
        ok, data = self._post("reviews", "/analyze", {"reviews": reviews})
        result.add_step("Review Analysis", data, ok)
        if not ok:
            result.errors.append(data.get("error", ""))
            return result

        # Step 2: Build knowledge articles from insights
        themes = [t["theme"] for t in data.get("top_themes", [])[:5]]
        recs = data.get("recommendations", [])
        critical = data.get("critical_issues", [])

        kb_article = (
            f"Customer Feedback Summary\n"
            f"Top Issues: {', '.join(themes)}\n"
            f"Critical Problems: {'; '.join(critical[:3])}\n"
            f"Improvement Recommendations: {'; '.join(recs[:3])}"
        )

        # Step 3: Ingest into RAG
        ok2, ingest_data = self._post("rag", "/ingest/text", {
            "text": kb_article,
            "source": "review_analysis",
            "category": "product_feedback",
        })
        result.add_step("Ingest into RAG Knowledge Base", ingest_data, ok2)
        result.final_output = f"Ingested insights from {len(reviews)} reviews. Themes: {', '.join(themes)}"
        return result

    # ─── Pipeline 3: Meeting → Assistant Memory ────────────────────────────────
    def meeting_pipeline(self, transcript: str, user_id: str = "user") -> PipelineResult:
        """
        transcript → structured summary → store in personal assistant memory
        """
        result = PipelineResult(pipeline="Meeting Notes → Assistant Memory")

        # Step 1: Summarize meeting
        ok, data = self._post("meeting", "/summarize_transcript", {"transcript": transcript})
        result.add_step("Meeting Summarization", data, ok)
        if not ok:
            result.errors.append(data.get("error", ""))
            return result

        summary = data.get("summary", "")
        actions = data.get("action_items", [])
        blockers = data.get("blockers", [])

        # Step 2: Store in assistant memory via chat
        action_str = '; '.join(a["owner"] + ": " + a["action"][:60] for a in actions[:5])
        blocker_str = '; '.join(blockers[:3])
        memory_message = (
            f"Please remember these meeting notes:\n"
            f"Summary: {summary}\n"
            f"My action items: {action_str}\n"
            f"Blockers to watch: {blocker_str}"
        )
        ok2, chat_data = self._post("assistant", "/chat", {
            "user_id": user_id,
            "message": memory_message,
        })
        result.add_step("Store in Assistant Memory", chat_data, ok2)
        result.final_output = f"Meeting stored. {len(actions)} action items. Response: {chat_data.get('response','')[:150]}"
        return result

    # ─── Pipeline 4: Contract → RAG → Q&A ────────────────────────────────────
    def contract_qa_pipeline(self, contract_text: str, question: str) -> PipelineResult:
        """
        contract → risk analysis + ingest → RAG-powered Q&A on the contract
        """
        result = PipelineResult(pipeline="Contract Analysis → RAG Q&A")

        # Step 1: Risk analysis
        ok, risk_data = self._post("contract", "/analyze_risk", {"text": contract_text})
        result.add_step("Contract Risk Analysis", risk_data, ok)

        # Step 2: Ingest contract into RAG
        ok2, ingest = self._post("rag", "/ingest/text", {
            "text": contract_text,
            "source": "contract",
            "category": "legal",
        })
        result.add_step("Ingest Contract into RAG", ingest, ok2)

        # Step 3: Answer question using RAG
        ok3, qa_data = self._post("rag", "/query", {"question": question, "top_k": 3})
        result.add_step("RAG Q&A on Contract", qa_data, ok3)
        result.final_output = qa_data.get("answer", "") if ok3 else ""
        return result

    # ─── Pipeline 5: Full Support Desk ────────────────────────────────────────
    def full_support_pipeline(
        self, customer_message: str, user_id: str = "user"
    ) -> PipelineResult:
        """
        Full pipeline: emotion → RAG lookup → empathetic response → log to assistant memory
        """
        result = PipelineResult(pipeline="Full Support Desk")

        # Step 1: Smart support response
        support = self.support_pipeline(customer_message, user_id=user_id)
        for step in support.steps:
            result.add_step(step["step"], step["result"], step["ok"])

        # Step 2: Log interaction to personal assistant memory
        if support.final_output:
            log_msg = f"Support interaction: Customer said '{customer_message[:100]}'. We responded: '{support.final_output[:150]}'"
            ok, mem = self._post("assistant", "/chat", {
                "user_id": user_id,
                "message": f"Remember this support interaction: {log_msg}",
            })
            result.add_step("Log to Assistant Memory", mem, ok)

        result.final_output = support.final_output
        return result

    # ─── Utilities ────────────────────────────────────────────────────────────
    def health_check(self) -> dict:
        """Check which services are online."""
        status = {}
        for name, url in SERVICES.items():
            try:
                r = requests.get(f"{url}/health", timeout=2)
                status[name] = "🟢 online" if r.ok else f"🔴 error {r.status_code}"
            except Exception:
                status[name] = "🔴 offline"
        return status


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orch = Orchestrator()

    print("🔍 Checking service health...")
    health = orch.health_check()
    for svc, status in health.items():
        print(f"  {svc:15s}: {status}")

    print("\n" + "="*60)
    print("Pipeline 1: Smart Customer Support")
    print("="*60)
    result = orch.support_pipeline(
        customer_message="I've been waiting 3 weeks for my order and nobody is responding!",
        user_id="demo_user"
    )
    print(result.summary())

    print("\n" + "="*60)
    print("Pipeline 2: Reviews → Knowledge Base")
    print("="*60)
    result2 = orch.review_pipeline([
        "Delivery was super slow, took 4 weeks.",
        "Great product quality but packaging was damaged.",
        "Customer service is unresponsive. Very frustrating.",
        "Setup was confusing, needs better instructions.",
        "Love the features! Works exactly as described.",
    ])
    print(result2.summary())

    print("\n" + "="*60)
    print("Pipeline 3: Meeting → Assistant Memory")
    print("="*60)
    result3 = orch.meeting_pipeline(
        transcript="John will fix the auth bug by Friday — urgent. Sarah is reviewing the API docs. Blocked on payment gateway vendor response. Alice to update staging today.",
        user_id="demo_user"
    )
    print(result3.summary())
