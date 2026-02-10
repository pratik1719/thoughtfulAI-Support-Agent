import os
import streamlit as st

from src.utils import load_json
from src.retriever import FAQRetriever
from src.fallback import gemini_fallback


# App Setup
st.set_page_config(page_title="Thoughtful AI Support Agent", page_icon="🤖")
st.title("🤖 Thoughtful AI — Customer Support Agent")
st.caption("FAQ retrieval (TF-IDF) + Gemini fallback when no strong FAQ match exists.")



# Load FAQ Data
DATA_PATH = os.path.join("data", "faq.json")

try:
    data = load_json(DATA_PATH)
    qa_items = data.get("questions", [])
    retriever = FAQRetriever(qa_items)
except Exception as e:
    st.error(f"Failed to load FAQ dataset: {e}")
    st.stop()



# Sidebar Settings
with st.sidebar:
    st.subheader("Settings")
    threshold = st.slider("FAQ match threshold", 0.10, 0.90, 0.60, 0.01)
    st.write("Higher = stricter matching, fewer wrong matches.")
    st.divider()
    st.write("Gemini Fallback (optional)")
    st.code(
        "export GOOGLE_API_KEY=...\n"
        "export GEMINI_MODEL=gemini-2.0-flash\n"
        "streamlit run app.py",
        language="bash",
    )


# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about Thoughtful AI’s agents (EVA, CAM, PHIL) or their benefits."}
    ]


# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Input
user_text = st.chat_input("Ask a question about Thoughtful AI...")
if user_text:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Retrieve answer
    best_item, score = retriever.retrieve(user_text, threshold=threshold)

    if best_item:
        answer = best_item["answer"]
        meta = f" Matched FAQ: **{best_item['question']}**  \nConfidence: `{score:.2f}`"
        final = f"{answer}\n\n---\n{meta}"
    else:
        answer = gemini_fallback(user_text)
        meta = f" No strong FAQ match (best confidence: `{score:.2f}`) → using fallback."
        final = f"{answer}\n\n---\n{meta}"

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": final})
    with st.chat_message("assistant"):
        st.markdown(final)
