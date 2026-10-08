"""Steps 2–4 — find faces, classify each expression, and summarise.

    image ──► face detector ──► crop + margin ──► ViT expression model ──► 7 probabilities
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import models
from .models import EMOTIONS


@dataclass
class Face:
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in pixels
    probs: dict[str, float] = field(default_factory=dict)
    det_conf: float = 1.0

    @property
    def top(self) -> str:
        return max(self.probs, key=self.probs.get) if self.probs else "neutral"

    @property
    def confidence(self) -> float:
        return self.probs.get(self.top, 0.0)


def to_rgb(img) -> np.ndarray:
    """Accept a PIL image, an RGB array or a file path; return an RGB uint8 array."""
    if isinstance(img, (str, bytes)) or hasattr(img, "read"):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    return np.asarray(img)


def detect_faces(rgb: np.ndarray, min_conf: float = 0.4, min_size: int = 24) -> list[tuple[tuple[int, int, int, int], float]]:
    """Step 2 — boxes for every face. Tries YOLOv8-face, falls back to a Haar cascade."""
    det = models.face_detector()
    h, w = rgb.shape[:2]
    out = []
    if det is not None:
        res = det(rgb[:, :, ::-1], conf=min_conf, verbose=False)[0]  # ultralytics expects BGR arrays
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = b.astype(int)
            if min(x2 - x1, y2 - y1) >= min_size:
                out.append(((max(0, x1), max(0, y1), min(w, x2), min(h, y2)), float(c)))
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for x, y, fw, fh in models.haar_cascade().detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size)):
            out.append(((x, y, x + fw, y + fh), 1.0))
    return sorted(out, key=lambda f: f[0][0])  # left to right, so "Face 1" is the leftmost


def crop_face(rgb: np.ndarray, box, margin: float = 0.2) -> Image.Image:
    """FER-2013 faces are tightly framed but include the whole head — add a little margin."""
    x1, y1, x2, y2 = box
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    h, w = rgb.shape[:2]
    return Image.fromarray(rgb[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)])


def classify_crops(crops: list[Image.Image]) -> list[dict[str, float]]:
    """Step 3 — the ViT model returns a probability for each of the 7 emotions."""
    if not crops:
        return []
    clf = models.expression_classifier()
    results = clf(crops, batch_size=8)
    if crops and isinstance(results[0], dict):  # a single image comes back un-nested
        results = [results]
    return [{r["label"].lower(): float(r["score"]) for r in res} for res in results]


def analyze_image(img, whole_image_if_no_face: bool = True) -> tuple[list[Face], np.ndarray]:
    rgb = to_rgb(img)
    found = detect_faces(rgb)
    if not found and whole_image_if_no_face:
        # Already a face crop (e.g. a dataset image) — classify the whole picture.
        found = [((0, 0, rgb.shape[1], rgb.shape[0]), 0.0)]
    crops = [crop_face(rgb, b) for b, _ in found]
    faces = [Face(box=b, probs=p, det_conf=c) for (b, c), p in zip(found, classify_crops(crops))]
    return faces, rgb


def annotate(rgb: np.ndarray, faces: list[Face]) -> np.ndarray:
    up = max(1.0, 360 / rgb.shape[1])  # tiny images (48 px dataset faces) get upscaled for drawing
    img = cv2.resize(rgb, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else rgb.copy()
    scale = max(0.5, img.shape[1] / 900)
    for i, f in enumerate(faces, 1):
        x1, y1, x2, y2 = (int(v * up) for v in f.box)
        hexc = models.COLORS[f.top].lstrip("#")
        color = tuple(int(hexc[j:j + 2], 16) for j in (0, 2, 4))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
        label = f"{i} {f.top} {f.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, max(1, int(1.5 * scale)))
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255),
                    max(1, int(1.5 * scale)), cv2.LINE_AA)
    return img


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def analyze_video(path: str, samples_per_second: float = 4.0, max_seconds: float | None = None,
                  progress=None) -> pd.DataFrame:
    """Step 4 (video) — an emotion timeline per person.

    Faces are linked across sampled frames by box overlap (IoU), so each
    person gets one `person` id. Returns one row per person per sample.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / samples_per_second)))
    last = n if max_seconds is None else min(n, int(max_seconds * fps))
    tracks: dict[int, tuple[int, int, int, int]] = {}
    next_id, rows = 1, []
    for idx in range(0, last, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, _ = analyze_image(rgb, whole_image_if_no_face=False)
        for f in faces:
            best = max(tracks.items(), key=lambda kv: _iou(kv[1], f.box), default=(None, None))
            if best[0] is not None and _iou(best[1], f.box) > 0.3:
                pid = best[0]
            else:
                pid, next_id = next_id, next_id + 1
            tracks[pid] = f.box
            rows.append(dict(t=idx / fps, frame=idx, person=pid, top=f.top, confidence=f.confidence,
                             **{e: f.probs.get(e, 0.0) for e in EMOTIONS}))
        if progress:
            progress(min(1.0, (idx + step) / last), f"{idx / fps:.1f}s / {last / fps:.1f}s")
    cap.release()
    return pd.DataFrame(rows)


def smooth_timeline(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Rolling mean of each emotion per person — single-frame flickers wash out."""
    if df.empty:
        return df
    out = df.sort_values("t").copy()
    out[EMOTIONS] = out.groupby("person")[EMOTIONS].transform(lambda s: s.rolling(window, min_periods=1).mean())
    out["top"] = out[EMOTIONS].idxmax(axis=1)
    return out


def calibrate(probs: dict[str, float], T: float) -> dict[str, float]:
    """Temperature scaling: p ∝ p_raw^(1/T). T > 1 softens an over-confident model; the ranking never changes."""
    if not probs or T == 1:
        return probs
    logp = {k: np.log(max(v, 1e-12)) / T for k, v in probs.items()}
    m = max(logp.values())
    z = sum(np.exp(v - m) for v in logp.values())
    return {k: float(np.exp(v - m) / z) for k, v in logp.items()}


def classify_text(texts: list[str]) -> list[dict[str, float]]:
    """Text emotion, mapped onto the same 7 labels as the face model."""
    if not texts:
        return []
    res = models.text_classifier()(texts, truncation=True)
    if isinstance(res[0], dict):
        res = [res]
    return [{models.TEXT_TO_FACE.get(r["label"], r["label"]): float(r["score"]) for r in one} for one in res]
