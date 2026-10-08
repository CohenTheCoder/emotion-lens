"""Regenerate docs/assets for the README:  python scripts/make_readme_assets.py

Uses a public-domain NASA crew portrait (downloaded on first run) and docs/eval.json.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emotionlens  # noqa: E402,F401  (torch first)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from emotionlens.analyze import analyze_image, annotate, crop_face  # noqa: E402
from emotionlens.explain import occlusion_map, overlay  # noqa: E402

OUT = ROOT / "docs" / "assets"
CREW_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/STS-125_crew_portrait.jpg/1280px-STS-125_crew_portrait.jpg"


def crew():
    path = ROOT / "outputs" / "sts125_crew.jpg"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        req = urllib.request.Request(CREW_URL, headers={"User-Agent": "emotion-lens-readme/0.1"})
        path.write_bytes(urllib.request.urlopen(req).read())
    faces, rgb = analyze_image(str(path))
    Image.fromarray(annotate(rgb, faces)).save(OUT / "crew_annotated.jpg", quality=86)
    strips = []
    for k in (0, 5):
        c = crop_face(rgb, faces[k].box)
        strips.append(np.hstack([np.array(c.resize((224, 224))), overlay(c, occlusion_map(c, faces[k].top))]))
    Image.fromarray(np.hstack([strips[0], np.full((224, 16, 3), 255, np.uint8), strips[1]])).save(
        OUT / "occlusion.jpg", quality=88)


def calibration():
    r = json.loads((ROOT / "docs" / "eval.json").read_text())
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=160)
    ax.plot([0, 1], [0, 1], ":", color="#8a96a3", label="perfectly honest")
    for key, label, color in (("reliability", f"raw (ECE {r['ece']:.3f})", "#d64545"),
                              ("reliability_calibrated",
                               f"temperature-scaled, T={r['temperature']:.2f} (ECE {r['ece_calibrated']:.3f})",
                               "#3d7bd9")):
        rel = r[key]
        ax.plot([b["confidence"] for b in rel], [b["accuracy"] for b in rel], "o-", color=color, label=label,
                markersize=4)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title("Is the model's confidence honest?", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    crew()
    calibration()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
