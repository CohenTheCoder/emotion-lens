"""Step 6 — How good is it, really? Evaluate on held-out FER-2013 faces.

The expression model was fine-tuned on FER-2013's *training* split, so we score
it on the *test* split (PrivateTest, 3,589 faces) that it never saw:

* accuracy and macro-F1 (macro so the rare "disgust" class counts equally)
* per-class precision / recall
* a confusion matrix (which emotions get mixed up)
* expected calibration error (when it says 80%, is it right 80% of the time?)
* **temperature scaling**: one number T, fitted on the separate *validation*
  split, that softens over-confident probabilities: p ∝ p_raw^(1/T)

    python -m emotionlens.evaluate --n 600        # quick, stratified sample
    python -m emotionlens.evaluate --n 0          # the full test + validation splits
Results land in docs/eval.json + docs/confusion_matrix.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import models
from .analyze import classify_crops
from .models import EMOTIONS

DATASET, SPLIT, CAL_SPLIT = "AutumnQiu/fer2013", "test", "valid"
CACHE = Path("outputs")


def load_sample(n: int, seed: int = 0, split: str = SPLIT):
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    names = [x.lower() for x in ds.features["label"].names]
    df = pd.DataFrame({"i": range(len(ds)), "label": [names[y] for y in ds["label"]]})
    if n and n < len(df):
        # Stratified: keep each emotion's share of the test set.
        frac = n / len(df)
        df = pd.concat([g.sample(max(1, round(frac * len(g))), random_state=seed) for _, g in df.groupby("label")])
    images = [ds[int(i)]["image"].convert("RGB") for i in df["i"]]
    return images, df["label"].tolist()


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list]:
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            table.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()), confidence=float(conf[m].mean()),
                              accuracy=float(correct[m].mean())))
    return float(ece), table


def score_split(split: str, n: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """Probabilities for every face in a split (cached in outputs/, it takes a while on CPU)."""
    cache = CACHE / f"probs_{split}_{n}_{seed}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["P"], list(z["y"])
    images, y_true = load_sample(n, seed, split)
    probs = []
    for i in range(0, len(images), 32):
        probs += classify_crops(images[i:i + 32])
        print(f"\r{split}: scored {min(i + 32, len(images))}/{len(images)}", end="", flush=True)
    print()
    P = np.array([[p.get(e, 0.0) for e in EMOTIONS] for p in probs])
    CACHE.mkdir(exist_ok=True)
    np.savez(cache, P=P, y=np.array(y_true))
    return P, y_true


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(P, 1e-12, 1)) / T
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(P: np.ndarray, y: list[str]) -> float:
    """The T that minimises negative log-likelihood (the standard recipe, Guo et al. 2017)."""
    from scipy.optimize import minimize_scalar

    idx = np.array([EMOTIONS.index(v) for v in y])

    def nll(T):
        return -np.mean(np.log(apply_temperature(P, T)[np.arange(len(idx)), idx] + 1e-12))

    return float(minimize_scalar(nll, bounds=(0.5, 20), method="bounded").x)


def evaluate(n: int = 600, seed: int = 0, out_dir: str | Path = "docs", cal_n: int = 1200) -> dict:
    # Load the model *before* `datasets` imports pyarrow: on some Macs loading torch
    # weights after pyarrow's OpenMP runtime is in memory crashes the process.
    models.expression_classifier()
    P, y_true = score_split(SPLIT, n, seed)
    # One parameter doesn't need thousands of faces: a stratified sample of the validation split is plenty.
    P_val, y_val = score_split(CAL_SPLIT, cal_n if n == 0 else min(n, cal_n), seed)
    T = fit_temperature(P_val, y_val)
    y_pred = [EMOTIONS[j] for j in P.argmax(1)]
    correct = np.array([a == b for a, b in zip(y_true, y_pred)])
    ece, rel = expected_calibration_error(P.max(1), correct)
    ece_cal, rel_cal = expected_calibration_error(apply_temperature(P, T).max(1), correct)
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    result = dict(
        dataset=f"{DATASET} ({SPLIT} split)", n=len(y_true), n_calibration=len(y_val),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=EMOTIONS, average="macro")),
        ece=ece, reliability=rel, temperature=T, ece_calibrated=ece_cal, reliability_calibrated=rel_cal,
        per_class=classification_report(y_true, y_pred, labels=EMOTIONS, output_dict=True, zero_division=0),
        confusion=cm.tolist(), labels=EMOTIONS,
        majority_baseline=float(pd.Series(y_true).value_counts(normalize=True).iloc[0]),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    _plot_confusion(cm, out / "confusion_matrix.png")
    return result


def _plot_confusion(cm: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=35, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "#0f1b2d")
    ax.set_title("FER-2013 test — row-normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="faces to score (0 = whole test split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal-n", type=int, default=1200, help="validation faces used to fit the temperature")
    args = ap.parse_args()
    r = evaluate(args.n, args.seed, cal_n=args.cal_n)
    print(f"accuracy {r['accuracy']:.3f} · macro-F1 {r['macro_f1']:.3f} · majority baseline "
          f"{r['majority_baseline']:.3f} (n={r['n']})\nECE {r['ece']:.3f} → {r['ece_calibrated']:.3f} after "
          f"temperature scaling (T = {r['temperature']:.2f}, fitted on the validation split)")
