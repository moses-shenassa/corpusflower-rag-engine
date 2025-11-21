from __future__ import annotations

import json
import logging
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BOB_BACKEND_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask_bob(question: str) -> dict:
    resp = requests.post(f"{BACKEND_URL}/api/ask", json={"question": question})
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="CorpusFlower — Occult Knowledge Engine", page_icon="📚")

    st.title("🧠 CorpusFlower — Occult Knowledge Engine")
    st.caption("Dresden Files–inspired RAG over your private occult library.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown("### Connection")
        st.write(f"Backend URL: `{BACKEND_URL}`")
        st.markdown("---")
        st.markdown("### Tips")
        st.write(
            """Try questions like:
- "What does this corpus say about planetary pentacles of Saturn?"
- "Compare how hoodoo and Solomonic grimoires treat protective amulets."
- "Summarize the role of psalms in these texts."""  # noqa: E501
        )

    question = st.text_area("Ask CorpusFlower a question about your corpus:", height=100)
    if st.button("Ask CorpusFlower", type="primary") and question.strip():
        with st.spinner("Consulting grimoires..."):
            try:
                data = ask_bob(question.strip())
                answer = data.get("answer", "")
                raw_context = data.get("raw_context", {})
            except Exception as e:  # noqa: BLE001
                st.error(f"Error contacting backend: {e}")
                return

        st.session_state.history.append({"q": question, "a": answer, "ctx": raw_context})

    for turn in reversed(st.session_state.history):
        st.markdown("#### You asked:")
        st.markdown(turn["q"])
        st.markdown("#### CorpusFlower answered:")
        st.markdown(turn["a"])

        with st.expander("Show raw retrieval context"):
            st.code(json.dumps(turn["ctx"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
