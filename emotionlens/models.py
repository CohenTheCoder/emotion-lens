"""Step 1 — Import the models from Hugging Face.

Three pretrained models, no training required:

| Job | Model | Notes |
|---|---|---|
| Find faces | `arnabdhar/YOLOv8-Face-Detection` | YOLOv8 fine-tuned on WIDER FACE (falls back to OpenCV's Haar cascade) |
| Read the expression | `trpakov/vit-face-expression` | Vision Transformer fine-tuned on FER-2013 (7 emotions) |
| Read emotion in text | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa, the same 7 Ekman-style emotions |

Everything is cached in ~/.cache/huggingface after the first download.
"""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # skip TensorFlow if it happens to be installed
os.environ.setdefault("USE_TF", "0")

FACE_REPO, FACE_FILE = "arnabdhar/YOLOv8-Face-Detection", "model.pt"
EXPRESSION_MODEL = "trpakov/vit-face-expression"
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😮"}
COLORS = {"angry": "#d64545", "disgust": "#6f9a3a", "fear": "#8a5cc2", "happy": "#f2a900", "neutral": "#8a96a3",
          "sad": "#3d7bd9", "surprise": "#e8743b"}
# The text model uses slightly different names for the same emotions.
TEXT_TO_FACE = {"anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy", "neutral": "neutral",
                "sadness": "sad", "surprise": "surprise"}


@lru_cache(maxsize=1)
def face_detector():
    """YOLOv8 face detector from the Hub, or None if ultralytics/the download is unavailable."""
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        return YOLO(hf_hub_download(FACE_REPO, FACE_FILE))
    except Exception:  # noqa: BLE001 — any failure just means "use the fallback"
        return None


@lru_cache(maxsize=1)
def haar_cascade():
    import cv2

    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@lru_cache(maxsize=1)
def expression_classifier():
    from transformers import pipeline

    return pipeline("image-classification", model=EXPRESSION_MODEL, top_k=len(EMOTIONS))  # all 7, not the default 5


@lru_cache(maxsize=1)
def text_classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=TEXT_MODEL, top_k=None)
