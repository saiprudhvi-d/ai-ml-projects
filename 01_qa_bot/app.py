"""
Project 01: Domain-Specific Q&A Bot — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Domain Q&A Bot", page_icon="🤖")
st.title("🤖 Domain-Specific Q&A Bot")
st.caption("Powered by fine-tuned DistilBERT · 92% test accuracy on FAQ workflows")

with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model**: DistilBERT fine-tuned on 10K+ Q&A pairs
    - **Task**: Extractive question answering
    - **Use case**: FAQ support workflow automation
    - **Inference**: FastAPI endpoint on port 8000
    """)
    api_url = st.text_input("API URL", value=API_URL)

tab1, tab2 = st.tabs(["Single Q&A", "Batch Q&A"])

with tab1:
    context = st.text_area(
        "📄 Context / Knowledge Base Passage",
        height=200,
        placeholder="Paste your support document or FAQ passage here...",
    )
    question = st.text_input("❓ Your Question", placeholder="e.g. What is the return window?")

    if st.button("Get Answer", type="primary"):
        if not context or not question:
            st.warning("Please provide both context and a question.")
        else:
            with st.spinner("Querying model..."):
                try:
                    resp = requests.post(
                        f"{api_url}/answer",
                        json={"question": question, "context": context},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"**Answer:** {data['answer']}")
                        st.metric("Confidence Score", f"{data['score']:.2%}")
                        st.markdown(f"*Span: characters {data['start']}–{data['end']}*")
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.ConnectionError:
                    st.error("⚠️ Cannot connect to API. Make sure `api.py` is running.")

with tab2:
    st.markdown("Upload a JSON file with `[{'question': '...', 'context': '...'}]` format.")
    uploaded = st.file_uploader("Upload batch JSON", type="json")
    if uploaded and st.button("Run Batch", type="primary"):
        import json
        items = json.load(uploaded)
        with st.spinner(f"Processing {len(items)} items..."):
            try:
                resp = requests.post(
                    f"{api_url}/batch_answer",
                    json=items,
                    timeout=30,
                )
                if resp.status_code == 200:
                    import pandas as pd
                    df = pd.DataFrame(resp.json())
                    st.dataframe(df)
                    st.download_button("Download Results", df.to_csv(index=False), "results.csv")
                else:
                    st.error(f"API error: {resp.text}")
            except requests.ConnectionError:
                st.error("⚠️ Cannot connect to API.")
