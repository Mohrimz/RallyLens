"""
src/app.py - Streamlit dashboard for the badminton analytics pipeline.

Upload a clip -> current shot + confidence + top-3 -> rally history -> next-shot
prediction + coaching line. The 4-slot history persists across uploads so you
can walk through a rally clip by clip.

Run (from the repo root):
    streamlit run src/app.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import config
from pipeline import (
    load_models, predict_current, predict_next, coaching_line, RallyHistory,
)

st.set_page_config(page_title="Badminton Analytics", page_icon="🏸", layout="centered")


@st.cache_resource
def _models():
    return load_models()


def main():
    st.title("🏸 Badminton Stroke & Rally Analytics")
    st.caption("Upload a shot clip to classify the stroke and predict the next one.")

    video_model, seq_model = _models()

    if video_model is None:
        st.error(f"Video model not found at {config.VIDEO_CKPT}. "
                 f"Train it first: `python src/train_video.py`")
        st.stop()

    # Persistent 4-slot rally history.
    if "history" not in st.session_state:
        st.session_state.history = RallyHistory()
    history = st.session_state.history

    with st.sidebar:
        st.header("Rally")
        st.write("History:", history.labels() or "—")
        if st.button("Reset rally"):
            history.reset()
            st.rerun()

    uploaded = st.file_uploader("Shot clip", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        clip_path = tmp.name

    st.video(clip_path)

    try:
        with st.spinner("Analyzing..."):
            current = predict_current(video_model, clip_path)
    finally:
        try:
            os.remove(clip_path)
        except OSError:
            pass

    # ---- Current shot -----------------------------------------------------
    st.subheader("Current shot")
    st.metric(current["label"], f"{current['confidence']*100:.1f}% confidence")
    st.caption("Top-3")
    for p in current["top"]:
        st.progress(p["prob"], text=f"{p['label']} — {p['prob']*100:.1f}%")

    history.append(current["pred_id"])
    st.caption(f"Rally history: {history.labels()}")

    # ---- Next shot --------------------------------------------------------
    if seq_model is not None:
        nxt = predict_next(seq_model, history.ids())["top"]
        st.subheader("Predicted next shot")
        for p in nxt:
            st.progress(p["prob"], text=f"{p['label']} — {p['prob']*100:.1f}%")
        st.info("💡 " + coaching_line(current["label"], nxt[0]["label"]))
    else:
        st.warning("Sequence model not found - next-shot prediction disabled. "
                   "Train it with `python src/train_sequence.py`.")


if __name__ == "__main__":
    main()
