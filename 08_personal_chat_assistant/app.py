"""
Project 08: Personal Chat Assistant — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8007"

st.set_page_config(page_title="Personal AI Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Personal Chat Assistant")
st.caption("Memory-enabled · Profile-aware · Context-persistent across sessions")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Your Profile")
    user_id = st.text_input("User ID", value="user_001")
    user_name = st.text_input("Name", value="")
    if st.button("Update Profile"):
        resp = requests.post(f"{API_URL}/profile", json={"user_id": user_id, "name": user_name})
        st.success(resp.json().get("status", "Updated"))

    st.divider()
    st.header("🧠 Memory")
    if st.button("View Memory Summary"):
        resp = requests.get(f"{API_URL}/memory/{user_id}")
        if resp.status_code == 200:
            data = resp.json()
            st.json(data)

    if st.button("View History"):
        resp = requests.get(f"{API_URL}/history/{user_id}?last_n=20")
        if resp.status_code == 200:
            for msg in resp.json()["history"]:
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"{role_icon} **{msg['role'].title()}**: {msg['content']}")

    if st.button("Clear Session", type="secondary"):
        requests.delete(f"{API_URL}/session/{user_id}")
        st.session_state.messages = []
        st.success("Session cleared.")

# ─── Chat Interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Talk to your personal assistant..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"user_id": user_id, "message": prompt},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    response = data["response"]
                    st.markdown(response)
                    st.caption(f"Interaction #{data['interaction_count']}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Error: {resp.text}")
            except requests.ConnectionError:
                st.error("⚠️ Cannot connect to backend. Run `python api.py` first.")
