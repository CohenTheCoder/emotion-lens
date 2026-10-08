import numpy as np
import pandas as pd

from emotionlens.analyze import _iou, smooth_timeline
from emotionlens.evaluate import expected_calibration_error
from emotionlens.models import EMOTIONS, TEXT_TO_FACE


def test_text_labels_map_onto_face_labels():
    assert sorted(TEXT_TO_FACE.values()) == sorted(EMOTIONS)


def test_iou():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert abs(_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.array([0.95] * 100)
    correct = np.array([1] * 95 + [0] * 5, dtype=float)
    ece, _ = expected_calibration_error(conf, correct)
    assert ece < 1e-9


def test_ece_detects_overconfidence():
    ece, _ = expected_calibration_error(np.full(100, 0.9), np.r_[np.ones(50), np.zeros(50)])
    assert abs(ece - 0.4) < 1e-9


def test_smoothing_removes_one_frame_flicker():
    rows = []
    for t in range(9):
        p = {e: 0.0 for e in EMOTIONS}
        p["sad" if t == 4 else "happy"] = 1.0  # a single 'sad' blip
        rows.append(dict(t=float(t), person=1, **p))
    out = smooth_timeline(pd.DataFrame(rows), window=3)
    assert (out["top"] == "happy").all()


def test_temperature_softens_but_keeps_ranking():
    from emotionlens.analyze import calibrate

    raw = {"happy": 0.97, "surprise": 0.02, "neutral": 0.01}
    soft = calibrate(raw, 2.0)
    assert max(soft, key=soft.get) == "happy"
    assert soft["happy"] < raw["happy"]
    assert abs(sum(soft.values()) - 1) < 1e-9
