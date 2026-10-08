"""EmotionLens — facial-expression & text emotion recognition.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import emotionlens  # noqa: F401  (imports torch first — see emotionlens/__init__.py)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from emotionlens import models
from emotionlens.analyze import (analyze_image, analyze_video, annotate, calibrate, classify_text, crop_face,
                                 smooth_timeline)
from emotionlens.explain import occlusion_map, overlay
from emotionlens.models import COLORS, EMOJI, EMOTIONS

st.set_page_config(page_title="EmotionLens", page_icon="🙂", layout="wide")
ROOT = Path(__file__).parent
ss = st.session_state
EVAL = json.loads((ROOT / "docs" / "eval.json").read_text()) if (ROOT / "docs" / "eval.json").exists() else {}
TEMPERATURE = float(EVAL.get("temperature", 1.0))


def prob_bars(probs: dict[str, float], height: int = 230) -> go.Figure:
    order = sorted(EMOTIONS, key=lambda e: probs.get(e, 0))
    fig = go.Figure(go.Bar(x=[probs.get(e, 0) for e in order], y=[f"{EMOJI[e]} {e}" for e in order],
                           orientation="h", marker_color=[COLORS[e] for e in order],
                           text=[f"{probs.get(e, 0):.0%}" for e in order], textposition="outside"))
    fig.update_layout(height=height, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 1.15],
                      visible=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


@st.cache_data(show_spinner=False)
def sample_faces(n: int = 8, seed: int = 1) -> list[tuple[np.ndarray, str]]:
    """A few held-out FER-2013 test faces, fetched from the Hugging Face Hub."""
    models.expression_classifier()  # load torch weights before pyarrow (see evaluate.py)
    from datasets import load_dataset

    ds = load_dataset("AutumnQiu/fer2013", split="test")
    names = [x.lower() for x in ds.features["label"].names]
    rng = np.random.default_rng(seed)
    picks = []
    for emo in EMOTIONS:  # one per emotion, plus a random extra
        idx = [i for i, y in enumerate(ds["label"]) if names[y] == emo]
        picks.append(int(rng.choice(idx)))
    picks += [int(i) for i in rng.choice(len(ds), max(0, n - len(picks)), replace=False)]
    return [(np.array(ds[i]["image"].convert("RGB")), names[ds[i]["label"]]) for i in picks[:n]]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🙂 EmotionLens")
    st.caption("Find faces → read expressions → explain → evaluate")
    st.subheader("The process")
    for i, s in enumerate(["Import models from Hugging Face", "Detect faces (YOLOv8-face)",
                           "Classify expression (ViT)", "Aggregate: people, video timelines, text",
                           "Explain with occlusion maps", "Evaluate on held-out FER-2013"], 1):
        st.markdown(f"**{i}.** {s}")
    st.divider()
    st.markdown("**Models** (Hugging Face)\n\n"
                f"- `{models.FACE_REPO}`\n- `{models.EXPRESSION_MODEL}`\n- `{models.TEXT_MODEL}`")
    use_cal = st.toggle(f"Calibrated confidence (T = {TEMPERATURE:.2f})", value=TEMPERATURE != 1.0,
                        disabled=TEMPERATURE == 1.0,
                        help="The raw model is over-confident. Temperature scaling (fitted on held-out data) makes "
                             "its percentages honest. The top emotion never changes.")
    st.caption("Expressions aren't feelings. A smile can be polite, and a neutral face can hide anything. "
               "Use for fun and research, not for judging people.")

st.title("Emotion detection")
tab_photo, tab_video, tab_text, tab_eval, tab_how = st.tabs(
    ["📸 Photo", "🎬 Video", "💬 Text", "📊 How accurate is it?", "🧭 How it works"])

# ---------------------------------------------------------------- photo
with tab_photo:
    src = st.radio("Image source", ["Upload a photo", "Take a photo", "Sample faces (FER-2013 test set)"],
                   horizontal=True)
    img = None
    if src == "Upload a photo":
        up = st.file_uploader("Photo with one or more faces", type=["jpg", "jpeg", "png", "webp"])
        img = Image.open(up) if up else None
    elif src == "Take a photo":
        shot = st.camera_input("Make a face 😄")
        img = Image.open(shot) if shot else None
    else:
        with st.spinner("Fetching sample faces from Hugging Face…"):
            faces_ = sample_faces()
        cols = st.columns(len(faces_))
        for i, (c, (arr, label)) in enumerate(zip(cols, faces_)):
            c.image(arr, caption=f"true: {label}", use_container_width=True)
            if c.button("Analyse", key=f"sample_{i}"):
                ss.sample_pick = i
        if "sample_pick" in ss:
            img = Image.fromarray(faces_[ss.sample_pick][0])
            st.caption(f"Ground-truth label: **{faces_[ss.sample_pick][1]}**")

    if img is not None:
        with st.spinner("Detecting faces and reading expressions…"):
            faces, rgb = analyze_image(img)
        if use_cal:
            for f in faces:
                f.probs = calibrate(f.probs, TEMPERATURE)
        if not faces:
            st.warning("No face found. Try a closer, front-facing photo.")
        else:
            c1, c2 = st.columns([3, 2])
            c1.image(annotate(rgb, faces), use_container_width=True,
                     caption=f"{len(faces)} face(s) · boxes coloured by top emotion")
            with c2:
                for i, f in enumerate(faces, 1):
                    st.markdown(f"#### Face {i}: {EMOJI[f.top]} **{f.top}** · {f.confidence:.0%}")
                    st.plotly_chart(prob_bars(f.probs), use_container_width=True, key=f"bars_{i}")
            st.divider()
            st.subheader("🔍 Why? Occlusion map")
            st.caption("We hide one square of the face at a time and re-ask the model. Bright areas are where "
                       "hiding the face changed the answer most, i.e. what the model was looking at. "
                       "(49 extra model calls, a few seconds on CPU.)")
            pick = st.selectbox("Face", [f"Face {i}" for i in range(1, len(faces) + 1)]) if len(faces) > 1 else "Face 1"
            f = faces[int(pick.split()[-1]) - 1]
            if st.button(f"Explain the '{f.top}' prediction"):
                crop = crop_face(rgb, f.box)
                with st.spinner("Occluding…"):
                    heat = occlusion_map(crop, f.top)
                e1, e2, _ = st.columns([1, 1, 2])
                e1.image(crop.resize((224, 224)), caption="face crop")
                e2.image(overlay(crop, heat), caption=f"what drove '{f.top}'")
            st.divider()
            st.subheader("💬 Face vs. words")
            said = st.text_input("What did they say? (optional)", placeholder="e.g. 'Oh great, another Monday.'")
            if said:
                tp = classify_text([said])[0]
                top_text = max(tp, key=tp.get)
                face_top = faces[0].top
                a, b = st.columns(2)
                a.metric("Face says", f"{EMOJI[face_top]} {face_top}")
                b.metric("Words say", f"{EMOJI.get(top_text, '')} {top_text}")
                if top_text == face_top:
                    st.success("Face and words agree.")
                elif {top_text, face_top} & {"happy"} and {top_text, face_top} & {"sad", "angry", "disgust"}:
                    st.warning("Mixed signals: a happy face with negative words (or vice versa). Sarcasm? 🙃")
                else:
                    st.info("Face and words disagree, which is common: expressions and language carry "
                            "different signals.")

# ---------------------------------------------------------------- video
with tab_video:
    st.markdown("Upload a clip (an interview, a reaction video, a game-day celebration…) to get an "
                "**emotion timeline for each person**.")
    up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"], key="vid")
    c1, c2 = st.columns(2)
    sps = c1.slider("Samples per second", 1.0, 8.0, 3.0, 0.5, help="Frames analysed per second of video")
    max_s = c2.slider("Analyse the first … seconds", 5, 300, 60, 5)
    if up is not None and st.button("▶ Analyse video", type="primary"):
        path = Path(tempfile.gettempdir()) / f"emotionlens_{up.name}"
        path.write_bytes(up.getbuffer())
        bar = st.progress(0.0, "Starting…")
        ss.timeline = analyze_video(str(path), sps, max_s, progress=lambda f, m: bar.progress(f, m))
        ss.video_path = str(path)
        bar.empty()
    tl = ss.get("timeline")
    if tl is not None:
        if tl.empty:
            st.warning("No faces found in the sampled frames.")
        else:
            sm = smooth_timeline(tl)
            people = sorted(sm["person"].unique())
            counts = sm["person"].value_counts()
            st.caption(f"{len(people)} person track(s). Short tracks are usually brief detections. Pick the "
                       "main people below.")
            chosen = st.multiselect("People", people, default=[p for p in counts.index[:3]],
                                    format_func=lambda p: f"Person {p} ({counts[p]} samples)")
            for p in chosen:
                g = sm[sm["person"] == p]
                m = g.melt(id_vars=["t"], value_vars=EMOTIONS, var_name="emotion", value_name="probability")
                fig = px.area(m, x="t", y="probability", color="emotion", color_discrete_map=COLORS,
                              labels={"t": "seconds"}, title=f"Person {p}")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)
            share = (sm[sm["person"].isin(chosen)].groupby("person")["top"].value_counts(normalize=True)
                     .unstack(fill_value=0).reindex(columns=EMOTIONS, fill_value=0) * 100).round(0)
            st.subheader("Share of time in each emotion (%)")
            st.dataframe(share.rename(columns=lambda e: f"{EMOJI[e]} {e}"), use_container_width=True)
            st.download_button("Download timeline (CSV)", sm.to_csv(index=False), "emotion_timeline.csv")

# ---------------------------------------------------------------- text
with tab_text:
    st.markdown("One message per line. Each line gets the same 7 emotions, from a DistilRoBERTa model.")
    default = ("We just won the championship!!!\nI can't believe you forgot my birthday again.\n"
               "Please don't leave me alone in this house tonight.\nUgh, the fridge smells like rotten eggs.\n"
               "Wait, you're telling me the test is TODAY?\nThe meeting is at 3pm in room B.")
    txt = st.text_area("Text", default, height=170)
    lines = [l for l in txt.splitlines() if l.strip()]
    if lines and st.button("Classify text", type="primary"):
        res = classify_text(lines)
        rows = []
        for line, p in zip(lines, res):
            top = max(p, key=p.get)
            rows.append({"text": line, "emotion": f"{EMOJI.get(top, '')} {top}", "confidence": p[top], **p})
        df = pd.DataFrame(rows)
        st.dataframe(df[["text", "emotion", "confidence"]], hide_index=True, use_container_width=True,
                     column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0,
                                                                                  max_value=1, format="%.0f%%")})
        m = df.melt(id_vars=["text"], value_vars=EMOTIONS, var_name="emotion", value_name="p")
        fig = px.bar(m, y="text", x="p", color="emotion", orientation="h", color_discrete_map=COLORS,
                     labels={"p": "probability", "text": ""})
        fig.update_layout(height=60 + 45 * len(lines), margin=dict(l=10, r=10, t=10, b=10), barmode="stack",
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- evaluation
with tab_eval:
    path = ROOT / "docs" / "eval.json"
    if not path.exists():
        st.info("Run `python -m emotionlens.evaluate` to score the model on the FER-2013 test split.")
    else:
        r = json.loads(path.read_text())
        st.markdown(f"Scored on **{r['n']:,} held-out faces** from `{r['dataset']}`. The model never saw "
                    "these during training.")
        k = st.columns(4)
        k[0].metric("Accuracy", f"{r['accuracy']:.1%}")
        k[1].metric("Macro-F1", f"{r['macro_f1']:.3f}", help="Average F1 over the 7 emotions, so rare ones count equally")
        k[2].metric("Always-'happy' baseline", f"{r['majority_baseline']:.1%}")
        if "ece_calibrated" in r:
            k[3].metric("Calibration error (ECE)", f"{r['ece_calibrated']:.3f}", f"{r['ece_calibrated'] - r['ece']:+.3f} "
                        f"vs raw", delta_color="inverse",
                        help=f"0 = confidence matches accuracy. Raw model: {r['ece']:.3f}. After temperature "
                             f"scaling (T = {r['temperature']:.2f}, fitted on {r.get('n_calibration', '?')} "
                             "validation faces).")
        else:
            k[3].metric("Calibration error (ECE)", f"{r['ece']:.3f}", help="0 = confidence matches accuracy exactly")
        st.caption("For context, humans agree with FER-2013's labels only about 65% of the time: the faces are "
                   "48×48 greyscale and some labels are ambiguous.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion matrix")
            cm = np.array(r["confusion"], dtype=float)
            norm = cm / cm.sum(1, keepdims=True).clip(min=1)
            fig = px.imshow(norm, x=r["labels"], y=r["labels"], color_continuous_scale="Blues", zmin=0, zmax=1,
                            text_auto=".0%", labels=dict(x="predicted", y="true", color="share"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Per-emotion scores")
            pc = pd.DataFrame({e: r["per_class"][e] for e in r["labels"]}).T[["precision", "recall", "f1-score",
                                                                            "support"]]
            pc.index = [f"{EMOJI[e]} {e}" for e in pc.index]
            st.dataframe(pc.round(3), use_container_width=True)
            st.subheader("Is the confidence honest?")
            rel = pd.DataFrame(r["reliability"])
            traces = [go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#8a96a3"),
                                 name="perfect"),
                      go.Scatter(x=rel["confidence"], y=rel["accuracy"], mode="lines+markers",
                                 marker=dict(size=np.sqrt(rel["n"]) + 4, color="#d64545"), name="raw model")]
            if "reliability_calibrated" in r:
                rc = pd.DataFrame(r["reliability_calibrated"])
                traces.append(go.Scatter(x=rc["confidence"], y=rc["accuracy"], mode="lines+markers",
                                         marker=dict(size=np.sqrt(rc["n"]) + 4, color="#3d7bd9"),
                                         name=f"calibrated (T={r['temperature']:.2f})"))
            fig = go.Figure(traces)
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="stated confidence",
                              yaxis_title="actual accuracy")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown((ROOT / "docs" / "HOW_IT_WORKS.md").read_text())
