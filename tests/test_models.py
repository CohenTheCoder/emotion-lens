"""Integration checks with the real Hugging Face models (downloads on first run)."""
import numpy as np
import pytest

pytest.importorskip("transformers")

from emotionlens.analyze import analyze_image, classify_text  # noqa: E402


def test_text_model_reads_obvious_emotions():
    joy, anger = classify_text(["We won the championship! Best day ever!", "I am furious, you lied to me."])
    assert max(joy, key=joy.get) == "happy"
    assert max(anger, key=anger.get) == "angry"


def test_blank_image_returns_valid_distribution():
    faces, _ = analyze_image(np.full((96, 96, 3), 128, np.uint8))
    assert len(faces) == 1  # no face found → whole image classified
    assert abs(sum(faces[0].probs.values()) - 1) < 1e-3
