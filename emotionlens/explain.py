"""Step 5 — Why did the model say that? Occlusion sensitivity.

Slide a grey patch over the face, one square at a time, and re-ask the model.
If hiding the mouth makes "happy" drop from 95% to 20%, the mouth mattered.

We measure the drop in *log-odds* (logit of the target minus log-sum-exp of
the others) rather than raw probability: a confident model sits at 99.9%,
where probabilities barely move but log-odds still do.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from . import models


def _log_odds(images: list[Image.Image], target: str) -> np.ndarray:
    import torch

    clf = models.expression_classifier()
    label2id = {v.lower(): int(k) for k, v in clf.model.config.id2label.items()}
    j = label2id[target]
    out = []
    for i in range(0, len(images), 16):
        inputs = clf.image_processor(images[i:i + 16], return_tensors="pt")
        with torch.no_grad():
            logits = clf.model(**inputs).logits
        others = torch.cat([logits[:, :j], logits[:, j + 1:]], dim=1)
        out.append((logits[:, j] - torch.logsumexp(others, dim=1)).numpy())
    return np.concatenate(out)


def occlusion_map(face: Image.Image, target: str, patch: int = 72, stride: int = 24) -> np.ndarray:
    """(224, 224) map: average drop in the target's log-odds when a patch covering that pixel is hidden.

    Overlapping patches (72 px, every 24 px → 7×7 = 49 model calls) catch features such
    as a whole smile that a small square would only partly cover.
    """
    img = np.array(face.convert("RGB").resize((224, 224)))
    base = _log_odds([Image.fromarray(img)], target)[0]
    fill = img.mean(axis=(0, 1)).astype(np.uint8)
    starts = list(range(0, 224 - patch + 1, stride))
    variants, boxes = [], []
    for y in starts:
        for x in starts:
            v = img.copy()
            v[y:y + patch, x:x + patch] = fill
            variants.append(Image.fromarray(v))
            boxes.append((y, x))
    drops = base - _log_odds(variants, target)
    total, count = np.zeros((224, 224)), np.zeros((224, 224))
    for (y, x), d in zip(boxes, drops):
        total[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    return np.clip(total / np.maximum(count, 1), 0, None)


def overlay(face: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img = np.array(face.convert("RGB").resize((224, 224)))
    h = heat / heat.max() if heat.max() > 0 else heat
    h = cv2.GaussianBlur(cv2.resize(h.astype(np.float32), (224, 224)), (0, 0), 9)
    h = h / h.max() if h.max() > 0 else h
    colored = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    return (img * (1 - alpha) + colored * alpha).astype(np.uint8)
