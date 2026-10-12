"""🎓 Learn tab: emotion recognition one step at a time, with sliders, on a real photo."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from emotionlens import models
from emotionlens.analyze import classify_crops, crop_face, detect_faces
from emotionlens.models import COLORS, EMOJI, EMOTIONS

ROOT = Path(__file__).parent
STEPS = ["1 · Pixels", "2 · Find faces", "3 · Patches (how a ViT sees)", "4 · Logits → softmax → temperature",
         "5 · Occlusion playground", "6 · Is it any good?"]


@st.cache_data(show_spinner=False)
def _crew() -> np.ndarray:
    return np.array(Image.open(ROOT / "samples" / "sts125_crew.jpg").convert("RGB"))


@st.cache_data(show_spinner=False)
def _faces(rgb: np.ndarray):
    return detect_faces(rgb, min_conf=0.05)


def _logits(img: Image.Image) -> np.ndarray:
    import torch

    clf = models.expression_classifier()
    inputs = clf.image_processor(img, return_tensors="pt")
    with torch.no_grad():
        z = clf.model(**inputs).logits[0].numpy()
    idx = {v.lower(): int(k) for k, v in clf.model.config.id2label.items()}
    return z[[idx[e] for e in EMOTIONS]]


def _bars(p: dict, height=250):
    order = sorted(EMOTIONS, key=lambda e: p[e])
    fig = go.Figure(go.Bar(x=[p[e] for e in order], y=[f"{EMOJI[e]} {e}" for e in order], orientation="h",
                           marker_color=[COLORS[e] for e in order], text=[f"{p[e]:.0%}" for e in order],
                           textposition="outside"))
    fig.update_layout(height=height, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 1.35], visible=False))
    return fig


def learn_tab():
    st.markdown("### 🎓 Learn computer vision with faces")
    st.caption("Each step runs the project's real models. Move the sliders and see what changes. "
               "The written guide is in `docs/LEARN.md`.")
    step = st.radio("Step", STEPS, horizontal=True, label_visibility="collapsed")
    rgb = _crew()
    faces = _faces(rgb)
    face_box = faces[3][0] if len(faces) > 3 else (faces[0][0] if faces else (0, 0, rgb.shape[1], rgb.shape[0]))
    face = crop_face(rgb, face_box)

    if step == STEPS[0]:
        st.markdown("**What the model actually receives.** A face is cropped, turned grey (FER-2013, the "
                    "training set, is greyscale), shrunk to **48×48** like the dataset, then resized to the "
                    "**224×224** the Vision Transformer expects and normalised to roughly −1…1.")
        c = st.columns(4)
        c[0].image(face, caption=f"Crop: {face.size[0]}×{face.size[1]} colour")
        g = face.convert("L")
        c[1].image(g, caption="Greyscale: 1 number per pixel")
        s48 = g.resize((48, 48))
        c[2].image(s48.resize((192, 192), Image.NEAREST), caption="48×48: what FER-2013 faces look like")
        c[3].image(face.resize((224, 224)), caption="224×224: the model's input size")
        st.dataframe(pd.DataFrame(np.array(s48)[20:28, 18:30]), use_container_width=True)
        st.caption("Those are real pixel values from the middle of the 48×48 face (0 = black, 255 = white).")

    elif step == STEPS[1]:
        st.markdown("**Face detection** puts a box around each face, with a confidence score. The YOLOv8 model "
                    "looks at the whole photo once and predicts every box together.")
        thr = st.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
        keep = [(b, c) for b, c in faces if c >= thr]
        vis = rgb.copy()
        for b, c in keep:
            cv2.rectangle(vis, b[:2], b[2:], (124, 77, 255), 3)
            cv2.putText(vis, f"{c:.2f}", (b[0], b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124, 77, 255), 2)
        st.image(vis, caption=f"{len(keep)} face(s) at threshold {thr:.2f} (there are 7 astronauts)",
                 use_container_width=True)

    elif step == STEPS[2]:
        st.markdown("**A Vision Transformer reads an image like a sentence.** It cuts the 224×224 face into a grid "
                    "of 16×16-pixel **patches** (14 × 14 = 196 of them). Each patch becomes a vector, a bit like a "
                    "word, and *attention* lets every patch look at every other one. That's how it connects a "
                    "raised eyebrow with an open mouth.")
        k = st.slider("Highlight patch #", 0, 195, 104)
        img = np.array(face.resize((224, 224)))
        for i in range(0, 225, 16):
            cv2.line(img, (i, 0), (i, 224), (255, 255, 255), 1)
            cv2.line(img, (0, i), (224, i), (255, 255, 255), 1)
        r, c_ = divmod(k, 14)
        cv2.rectangle(img, (c_ * 16, r * 16), (c_ * 16 + 16, r * 16 + 16), (255, 60, 60), 2)
        col1, col2 = st.columns(2)
        col1.image(img, caption="The 14×14 patch grid", width=336)
        patch = np.array(face.resize((224, 224)))[r * 16:(r + 1) * 16, c_ * 16:(c_ + 1) * 16]
        col2.image(cv2.resize(patch, (160, 160), interpolation=cv2.INTER_NEAREST),
                   caption=f"Patch {k} (row {r}, column {c_}): 16×16×3 = 768 numbers → one 'token'")

    elif step == STEPS[3]:
        st.markdown("**From scores to probabilities.** The model's last layer outputs 7 raw scores called "
                    "**logits**. **Softmax** turns them into probabilities: p = e^(z/T) / Σ e^(z/T). The "
                    "**temperature T** controls confidence. T = 1 is the raw model. Our evaluation found "
                    "T ≈ 1.8 makes its confidence honest.")
        with st.spinner("Running the model…"):
            z = _logits(face)
        T = st.slider("Temperature T", 0.3, 5.0, 1.0, 0.1)
        p = np.exp((z - z.max()) / T)
        p /= p.sum()
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(dict(emotion=EMOTIONS, logit=z.round(2), probability=p.round(3))),
                     hide_index=True, use_container_width=True)
        c2.plotly_chart(_bars(dict(zip(EMOTIONS, p))), use_container_width=True)
        st.caption("Low T sharpens (more certain), high T flattens (less certain). The top emotion never "
                   "changes, only how sure the model sounds.")

    elif step == STEPS[4]:
        st.markdown("**Explainability by experiment.** Hide part of the face and see if the answer changes. "
                    "Drag the grey square over the eyes, then the mouth.")
        c1, c2, c3 = st.columns(3)
        x = c1.slider("Patch x", 0, 200, 70, 4)
        y = c2.slider("Patch y", 0, 200, 140, 4)
        size = c3.slider("Patch size", 16, 140, 70, 4)
        img = np.array(face.resize((224, 224)))
        img[y:y + size, x:x + size] = img.mean(axis=(0, 1)).astype(np.uint8)
        with st.spinner("Re-asking the model…"):
            p0 = classify_crops([face])[0]
            p1 = classify_crops([Image.fromarray(img)])[0]
        a, b, c = st.columns([1, 1, 1])
        a.image(img, caption="What the model sees now", use_container_width=True)
        b.markdown("**Before**")
        b.plotly_chart(_bars(p0, 230), use_container_width=True, key="occ_before")
        c.markdown("**After hiding that patch**")
        c.plotly_chart(_bars(p1, 230), use_container_width=True, key="occ_after")
        top = max(p0, key=p0.get)
        st.metric(f"Change in '{top}'", f"{p1[top]:.0%}", f"{p1[top] - p0[top]:+.0%}")

    elif step == STEPS[5]:
        st.markdown("**Measuring a model honestly.** Accuracy alone hides a lot. The confusion matrix shows "
                    "*which* emotions get mixed up, and calibration shows whether the confidence can be "
                    "trusted. One practical use: **only answer when confident**.")
        path = ROOT / "docs" / "eval.json"
        if not path.exists():
            st.info("Run `python -m emotionlens.evaluate` first.")
            return
        r = json.loads(path.read_text())
        cm = np.array(r["confusion"], dtype=float)
        true = st.selectbox("True emotion", r["labels"], index=r["labels"].index("fear"))
        row = cm[r["labels"].index(true)]
        order = np.argsort(-row)
        st.write(", ".join(f"{EMOJI[r['labels'][j]]} {r['labels'][j]} {row[j] / row.sum():.0%}" for j in order[:4]))
        st.caption(f"Of {int(row.sum())} '{true}' faces, this is what the model said.")
        rel = pd.DataFrame(r["reliability"])
        thr = st.slider("Only answer when the model is at least … sure", 0.2, 0.95, 0.6, 0.05)
        lows = rel["bin"].str.split("-").str[0].astype(float)
        sel = rel[lows >= thr - 1e-9]
        coverage = sel["n"].sum() / rel["n"].sum()
        acc = (sel["accuracy"] * sel["n"]).sum() / max(sel["n"].sum(), 1)
        a, b = st.columns(2)
        a.metric("Faces it answers", f"{coverage:.0%}")
        b.metric("Accuracy on those", f"{acc:.0%}", f"{acc - r['accuracy']:+.0%} vs answering everything")
        st.caption("Trading coverage for accuracy is called *selective prediction*. Temperature scaling makes "
                   "this threshold mean what it says.")
