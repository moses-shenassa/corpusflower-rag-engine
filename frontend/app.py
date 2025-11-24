from __future__ import annotations

import json
import logging
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("CORPUSFLOWER_BACKEND_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ask_corpusflower(question: str) -> dict:
    """Call the backend /api/ask endpoint and return the JSON response."""
    resp = requests.post(f"{BACKEND_URL}/api/ask", json={"question": question})
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="CorpusFlower — Document Intelligence Engine", page_icon="📚")

    st.title("🧠 CorpusFlower — Document Intelligence Engine")
    st.caption("An enterprise-ready RAG interface over your own PDF collections.")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.markdown("### Connection")
        st.write(f"Backend URL: `{BACKEND_URL}`")
        st.markdown("---")
        st.markdown("### Tips")
        tips = (
            "Try questions like:\n"
            '- "What are the main themes that appear across these documents?"\n'
            '- "Summarize how this corpus discusses data privacy and risk."\n'
            '- "Compare how different documents describe the same concept or process."'
        )
        st.write(tips)

    question = st.text_area("Ask CorpusFlower a question about your indexed documents:", height=100)
    if st.button("Ask CorpusFlower", type="primary") and question.strip():
        with st.spinner("Analyzing documents and assembling an answer..."):
            try:
                data = ask_corpusflower(question.strip())
                answer = data.get("answer", "")
                raw_context = data.get("raw_context", {})
            except Exception as e:  # noqa: BLE001
                st.error(f"Error contacting backend: {e}")
                logger.exception("Error contacting backend")
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
