"""
Unified Dashboard — All 8 AI Projects
Run: streamlit run dashboard.py
Requires all API servers to be running (use start_all.sh)
"""

import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="AI Projects Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API Config ────────────────────────────────────────────────────────────────
APIS = {
    "01 · Q&A Bot":            "http://localhost:8000",
    "02 · Contract Analysis":  "http://localhost:8001",
    "03 · Code Suggestion":    "http://localhost:8002",
    "04 · Empathetic Response":"http://localhost:8003",
    "05 · Meeting Summarizer": "http://localhost:8004",
    "06 · Review Intelligence":"http://localhost:8005",
    "07 · RAG Knowledge Base": "http://localhost:8006",
    "08 · Personal Assistant": "http://localhost:8007",
}

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.status-ok   { color: #22c55e; font-weight: bold; }
.status-down { color: #ef4444; font-weight: bold; }
.project-card { background: #1e1e2e; border-radius: 12px; padding: 16px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar: Status Panel ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 AI Projects Hub")
    st.caption("8 live AI projects · unified control")
    st.divider()

    st.subheader("🟢 Service Status")
    statuses = {}
    for name, url in APIS.items():
        try:
            r = requests.get(f"{url}/health", timeout=1.5)
            ok = r.status_code == 200
        except Exception:
            ok = False
        statuses[name] = ok
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} **{name}**")

    live = sum(statuses.values())
    st.metric("Services Online", f"{live} / {len(APIS)}")
    st.divider()
    st.caption("Run `./start_all.sh` to start all services.")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🤖 Q&A Bot",
    "📄 Contracts",
    "💻 Code AI",
    "💬 Empathetic",
    "📋 Meetings",
    "⭐ Reviews",
    "🔍 RAG",
    "🧠 Assistant",
    "🔗 Orchestrate",
])

# ── Tab 1: Q&A Bot ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("🤖 Domain-Specific Q&A Bot")
    st.caption("DistilBERT · 92% test accuracy · FastAPI on :8000")
    context = st.text_area("📄 Context passage", height=150,
        value="Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Contact support@example.com for assistance.")
    question = st.text_input("❓ Question", value="How many days do I have to return an item?")
    if st.button("Get Answer", key="qa"):
        if not statuses["01 · Q&A Bot"]:
            st.error("Service offline — run `python 01_qa_bot/api.py`")
        else:
            with st.spinner("Querying..."):
                r = requests.post(f"{APIS['01 · Q&A Bot']}/answer",
                    json={"question": question, "context": context})
                if r.ok:
                    d = r.json()
                    st.success(f"**Answer:** {d['answer']}")
                    st.metric("Confidence", f"{d['score']:.1%}")

# ── Tab 2: Contract Analysis ───────────────────────────────────────────────────
with tabs[1]:
    st.header("📄 Contract Analysis AI")
    st.caption("LLaMA-7B + LoRA · LangChain + FAISS · FastAPI on :8001")
    contract_text = st.text_area("📝 Paste contract text", height=200,
        value="This Agreement between Acme Corp and TechVendor Inc. PAYMENT: $15,000/month due by the 5th. CONFIDENTIALITY: 3 years post-termination. TERMINATION: 60-day notice required. LIABILITY: Capped at 3 months of fees paid.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Risk Analysis"):
            if not statuses["02 · Contract Analysis"]:
                st.error("Service offline")
            else:
                r = requests.post(f"{APIS['02 · Contract Analysis']}/analyze_risk",
                    json={"text": contract_text})
                st.json(r.json())
    with col2:
        clause = st.selectbox("Clause type", ["termination","payment","confidentiality","liability"])
        if st.button("📌 Extract Clause"):
            if not statuses["02 · Contract Analysis"]:
                st.error("Service offline")
            else:
                r = requests.post(f"{APIS['02 · Contract Analysis']}/ingest",
                    json={"text": contract_text})
                r2 = requests.post(f"{APIS['02 · Contract Analysis']}/extract_clause",
                    json={"clause_type": clause})
                st.write(r2.json().get("extraction",""))
    with col3:
        if st.button("📃 Summarize"):
            if not statuses["02 · Contract Analysis"]:
                st.error("Service offline")
            else:
                r = requests.post(f"{APIS['02 · Contract Analysis']}/summarize",
                    json={"text": contract_text})
                st.write(r.json().get("summary",""))

# ── Tab 3: Code Suggestion ─────────────────────────────────────────────────────
with tabs[2]:
    st.header("💻 AI Code Suggestion")
    st.caption("Code Llama-7B · <250ms latency · FastAPI on :8002")
    code_prefix = st.text_area("✏️ Code prefix", height=120,
        value="# Function to find the longest palindrome in a string\ndef longest_palindrome(")
    col1, col2 = st.columns(2)
    with col1:
        n_suggestions = st.slider("# Suggestions", 1, 3, 2)
        temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    with col2:
        if st.button("⚡ Get Suggestions", type="primary"):
            if not statuses["03 · Code Suggestion"]:
                st.error("Service offline — run `python 03_code_suggestion/api.py`")
            else:
                t0 = time.time()
                r = requests.post(f"{APIS['03 · Code Suggestion']}/suggest",
                    json={"prefix": code_prefix, "num_suggestions": n_suggestions, "temperature": temp})
                if r.ok:
                    d = r.json()
                    st.metric("Latency", f"{d['latency_ms']}ms", delta="cached ✓" if d["cached"] else None)
                    for i, s in enumerate(d["suggestions"], 1):
                        st.code(code_prefix + s, language="python")

# ── Tab 4: Empathetic Response ─────────────────────────────────────────────────
with tabs[3]:
    st.header("💬 Empathetic Response AI")
    st.caption("RoBERTa (28 emotions) → LLM tone conditioning · FastAPI on :8003")
    customer_msg = st.text_input("📩 Customer message",
        value="I've been waiting 3 weeks! This is completely unacceptable. Nobody is responding to my emails!")
    agent_ctx = st.text_input("🗒️ Agent context (optional)", value="Order #12345, delayed due to supplier issue")
    if st.button("Generate Response", type="primary", key="empathetic"):
        if not statuses["04 · Empathetic Response"]:
            st.error("Service offline")
        else:
            r = requests.post(f"{APIS['04 · Empathetic Response']}/respond",
                json={"customer_message": customer_msg, "agent_context": agent_ctx})
            if r.ok:
                d = r.json()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Emotion", d["detected_emotion"])
                    st.metric("Confidence", f"{d['emotion_confidence']:.0%}")
                with col2:
                    st.info(f"**Tone guide:** {d['tone_guideline']}")
                st.success(f"**Generated Response:** {d['generated_response']}")

# ── Tab 5: Meeting Summarizer ──────────────────────────────────────────────────
with tabs[4]:
    st.header("📋 Meeting Notes Auto Summarizer")
    st.caption("Whisper + BART · Action items, blockers, next steps · FastAPI on :8004")
    transcript_input = st.text_area("📝 Paste meeting transcript", height=180,
        value="Sprint planning today. John will fix the login bug by Friday — urgent. Sarah needs to review the API docs. We're blocked on the payment gateway because the vendor hasn't responded. Mike is following up with them asap. Alice to update staging environment today. Stuck on database migration until DevOps approves.")
    if st.button("📋 Summarize Meeting", type="primary"):
        if not statuses["05 · Meeting Summarizer"]:
            st.error("Service offline")
        else:
            r = requests.post(f"{APIS['05 · Meeting Summarizer']}/summarize_transcript",
                json={"transcript": transcript_input})
            if r.ok:
                d = r.json()
                st.subheader("Summary")
                st.write(d["summary"])
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Action Items")
                    for item in d["action_items"]:
                        priority = "🔴" if item["priority"] == "high" else "🟡"
                        st.markdown(f"{priority} **{item['owner']}**: {item['action'][:80]}")
                with col2:
                    st.subheader("🚧 Blockers")
                    for b in d["blockers"]:
                        st.markdown(f"- {b[:80]}")
                st.subheader("👣 Next Steps by Owner")
                for ns in d["next_steps"]:
                    st.markdown(f"**{ns['owner']}**: {', '.join(t[:40] for t in ns['tasks'])}")

# ── Tab 6: Review Intelligence ─────────────────────────────────────────────────
with tabs[5]:
    st.header("⭐ Review Intelligence Engine")
    st.caption("RoBERTa sentiment + theme extraction + Flan-T5 recommendations · :8005")
    reviews_raw = st.text_area("📝 Paste reviews (one per line)", height=180,
        value="Great quality, fast delivery!\nTerrible customer service, waited 3 weeks and package was damaged.\nPrice is reasonable but setup was confusing.\nExcellent performance, exactly what I needed.\nVery slow shipping, product looks cheap.\nSupport team was super helpful!")
    if st.button("🔍 Analyze Reviews", type="primary"):
        if not statuses["06 · Review Intelligence"]:
            st.error("Service offline")
        else:
            reviews = [r.strip() for r in reviews_raw.strip().split("\n") if r.strip()]
            r = requests.post(f"{APIS['06 · Review Intelligence']}/analyze",
                json={"reviews": reviews})
            if r.ok:
                d = r.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", d["total_reviews"])
                col2.metric("Avg Sentiment", f"{d['avg_sentiment_score']:+.2f}")
                dist = d["sentiment_distribution"]
                col3.metric("Positive %", f"{dist.get('positive',0)/d['total_reviews']:.0%}")
                st.subheader("🏷️ Top Themes")
                st.bar_chart({t["theme"]: t["count"] for t in d["top_themes"]})
                st.subheader("💡 Recommendations")
                for rec in d["recommendations"]:
                    st.markdown(f"• {rec}")

# ── Tab 7: RAG Knowledge Base ──────────────────────────────────────────────────
with tabs[6]:
    st.header("🔍 Knowledge Base RAG")
    st.caption("LangChain + FAISS · Citation retrieval · Reduced hallucinations · :8006")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📥 Ingest Document")
        ingest_text = st.text_area("Knowledge base article", height=120,
            value="To reset your password, go to Settings > Security > Reset Password. Enter your registered email. The link expires in 24 hours.")
        ingest_src = st.text_input("Source label", value="KB-001")
        if st.button("➕ Add to Knowledge Base"):
            if not statuses["07 · RAG Knowledge Base"]:
                st.error("Service offline")
            else:
                r = requests.post(f"{APIS['07 · RAG Knowledge Base']}/ingest/text",
                    json={"text": ingest_text, "source": ingest_src})
                if r.ok:
                    st.success(f"Ingested {r.json()['chunks_ingested']} chunks")
    with col2:
        st.subheader("❓ Ask a Question")
        rag_q = st.text_input("Your question", value="How do I reset my password?")
        top_k = st.slider("Sources to retrieve", 1, 5, 3)
        if st.button("🔎 Search & Answer"):
            if not statuses["07 · RAG Knowledge Base"]:
                st.error("Service offline")
            else:
                r = requests.post(f"{APIS['07 · RAG Knowledge Base']}/query",
                    json={"question": rag_q, "top_k": top_k})
                if r.ok:
                    d = r.json()
                    st.success(d["answer"])
                    if d["sources"]:
                        st.caption(f"📚 Sources: {', '.join(s['source'] for s in d['sources'])}")
                    if d["cached"]:
                        st.caption("⚡ Cached result")

# ── Tab 8: Personal Assistant ──────────────────────────────────────────────────
with tabs[7]:
    st.header("🧠 Personal Chat Assistant")
    st.caption("Memory-enabled · Profile-aware · ChromaDB · FastAPI on :8007")
    user_id = st.text_input("User ID", value="laaz", key="chat_uid")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Chat with your personal AI..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if not statuses["08 · Personal Assistant"]:
            st.error("Service offline")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    r = requests.post(f"{APIS['08 · Personal Assistant']}/chat",
                        json={"user_id": user_id, "message": prompt})
                    if r.ok:
                        response = r.json()["response"]
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

# ── Tab 9: Orchestration ───────────────────────────────────────────────────────
with tabs[8]:
    st.header("🔗 Connected Pipelines")
    st.caption("Projects wired together for end-to-end workflows")

    st.subheader("Pipeline 1: Reviews → Knowledge Base → Q&A")
    st.markdown("""
    1. **Review Intelligence (06)** analyzes customer reviews and extracts recurring issues
    2. Issues are auto-ingested into the **RAG Knowledge Base (07)** as support articles
    3. Support agents query the **Q&A Bot (01)** which now answers based on real review patterns
    """)
    pipeline1_reviews = st.text_area("Paste reviews to run pipeline", height=100,
        value="Customers keep asking about password reset. Many users confused about export feature. Billing questions about refund timeline are common.")
    if st.button("▶️ Run Pipeline 1", type="primary"):
        if not all(statuses.get(k) for k in ["06 · Review Intelligence", "07 · RAG Knowledge Base"]):
            st.error("Requires services 06 and 07 to be running.")
        else:
            with st.spinner("Step 1: Analyzing reviews..."):
                reviews = [r.strip() for r in pipeline1_reviews.split("\n") if r.strip()]
                r1 = requests.post(f"{APIS['06 · Review Intelligence']}/analyze", json={"reviews": reviews})
            with st.spinner("Step 2: Ingesting insights into RAG..."):
                if r1.ok:
                    summary_text = "Common customer issues: " + "; ".join(
                        t["theme"] for t in r1.json().get("top_themes", [])[:5])
                    r2 = requests.post(f"{APIS['07 · RAG Knowledge Base']}/ingest/text",
                        json={"text": summary_text, "source": "review_insights"})
                    st.success(f"✅ Insights ingested: {r2.json().get('chunks_ingested', 0)} chunks")
                    st.json(r1.json().get("recommendations", []))

    st.divider()
    st.subheader("Pipeline 2: Meeting Notes → Personal Assistant Memory")
    st.markdown("""
    1. **Meeting Summarizer (05)** extracts action items and decisions
    2. Action items are stored in the **Personal Assistant's (08)** long-term memory
    3. The **Personal Assistant** can now remind you of commitments and follow up
    """)
    meeting_for_pipeline = st.text_area("Paste meeting notes", height=80,
        value="John agreed to fix the auth bug by Friday. Team decided to migrate to PostgreSQL next sprint. Alice will present the new dashboard on Monday.")
    pipeline2_uid = st.text_input("Your user ID for memory", value="laaz", key="p2uid")
    if st.button("▶️ Run Pipeline 2", type="primary", key="p2"):
        if not all(statuses.get(k) for k in ["05 · Meeting Summarizer", "08 · Personal Assistant"]):
            st.error("Requires services 05 and 08 to be running.")
        else:
            with st.spinner("Step 1: Summarizing meeting..."):
                r1 = requests.post(f"{APIS['05 · Meeting Summarizer']}/summarize_transcript",
                    json={"transcript": meeting_for_pipeline})
            with st.spinner("Step 2: Storing in assistant memory..."):
                if r1.ok:
                    d = r1.json()
                    memory_text = f"Meeting summary: {d['summary']}. Action items: " + \
                        "; ".join(f"{i['owner']} - {i['action'][:60]}" for i in d["action_items"][:5])
                    r2 = requests.post(f"{APIS['08 · Personal Assistant']}/chat",
                        json={"user_id": pipeline2_uid,
                              "message": f"Remember these action items from our meeting: {memory_text}"})
                    st.success("✅ Meeting context stored in assistant memory!")
                    st.write(r2.json().get("response", ""))
