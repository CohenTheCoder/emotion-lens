# 🎓 Learn computer vision with EmotionLens

**Who this is for:** you know a little Python, and you've never done computer vision. By the end you'll be
able to explain, step by step, how a program goes from a photo to *"face 3 looks surprised, 82 %
confident"*, and how to tell whether that number deserves your trust.

**How to use it:** read a section here, then run the matching cell in
**[`notebooks/learn_cv_step_by_step.ipynb`](../notebooks/learn_cv_step_by_step.ipynb)**. The notebook shows
the image at every stage. Budget about an hour.

```
photo ──► ① pixels ──► ② find faces ──► ③ crop ──► ④ resize + normalize ──► ⑤ model ──► ⑥ probabilities
                                                                                     │
                                         ⑨ is it honest? ◄── ⑧ is it right? ◄── ⑦ why?
```

<details>
<summary>⚙️ Setup (5 minutes)</summary>

```bash
git clone https://github.com/CohenTheCoder/emotion-lens.git && cd emotion-lens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook notebooks/learn_cv_step_by_step.ipynb
```
The first run downloads the models (~400 MB) from Hugging Face. After that they load from
`~/.cache/huggingface` in seconds.
</details>

---

## Part 1 · The five ideas everything else is built on

### 1. An image is a grid of numbers
🧠 **The idea.** A digital photo is a table of **pixels**. Each pixel stores three numbers: how much **R**ed,
**G**reen and **B**lue light, each from 0 (none) to 255 (full). A 1280×720 photo is a NumPy array with
shape `(720, 1280, 3)`: **height first**, then width, then the 3 colour channels.

💻 **Try it.**
```python
import numpy as np
from PIL import Image
img = np.array(Image.open("photo.jpg").convert("RGB"))
img.shape        # (720, 1280, 3)
img.dtype        # uint8 → whole numbers 0..255
img[0, 0]        # top-left pixel, e.g. [ 34  51 120]  (a dark blue)
img[:, :, 0]     # just the red channel: a 720×1280 grid
```

⚠️ **Gotcha #1: BGR vs RGB.** OpenCV (`cv2.imread`) loads images as **B**lue-**G**reen-**R**ed, while
almost everything else uses RGB. Swap them and people turn blue. That's why
[`analyze.detect_faces`](../emotionlens/analyze.py) passes `rgb[:, :, ::-1]` (reversed channels) to
YOLO, which expects BGR.

⚠️ **Gotcha #2: rows vs x/y.** Arrays are indexed `img[row, col]` = `img[y, x]`. Boxes are written
`(x1, y1, x2, y2)`. The origin is the **top-left** corner, and **y grows downward**.

### 2. A model only understands one input size
🧠 Neural networks are built for a fixed input size. This project's expression model wants
**224×224** pixels, and its training faces (FER-2013) were 48×48 greyscale. So every face is **cropped,
then resized**, and resizing a 48-px face up to 224 can't invent detail that was never there.

### 3. Normalization: speak the model's language
🧠 Models are trained on numbers in a particular range. Before inference, pixels (0–255) are scaled to
0–1 and then shifted by the training set's mean and standard deviation (here: `(x − 0.5) / 0.5`, giving −1…1).
You almost never do this by hand. The model's **image processor** does it:

```python
from transformers import AutoImageProcessor
proc = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
batch = proc(face_image, return_tensors="pt")
batch["pixel_values"].shape     # torch.Size([1, 3, 224, 224])  → N, C, H, W
```
⚠️ Notice the shape changed from `(H, W, 3)` to **`(N, C, H, W)`**: batch size first, channels second.
This "channels-first **tensor**" layout is how PyTorch wants it.

### 4. Pretrained models + Hugging Face = no training needed
🧠 Training a vision model from scratch takes millions of images and lots of GPU time. Instead we download
models other people trained and shared on the **[Hugging Face Hub](https://huggingface.co/models)**.
Using a pretrained model for your task is called **transfer learning**. This project uses three, all in
[`emotionlens/models.py`](../emotionlens/models.py):

| Job | Model | Architecture |
|---|---|---|
| Find faces | `arnabdhar/YOLOv8-Face-Detection` | **YOLO** (a convolutional object detector) |
| Read the expression | `trpakov/vit-face-expression` | **ViT** (Vision Transformer) |
| Read emotion in text | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa (language) |

```python
from transformers import pipeline
clf = pipeline("image-classification", model="trpakov/vit-face-expression", top_k=7)
clf(face_image)   # [{'label': 'happy', 'score': 0.93}, {'label': 'neutral', 'score': 0.04}, ...]
```

### 5. What a model actually outputs: logits → softmax → probabilities
🧠 The last layer outputs one raw number per class, called a **logit** (any value, e.g. `[-1.2, 0.3, 4.1, …]`).
**Softmax** turns logits into probabilities that are all positive and sum to 1:
`p_i = e^(logit_i) / Σ e^(logit_j)`. The biggest probability is the prediction, and its value is the
**confidence**. Hold onto that word, because Part 3 shows confidence can lie.

---

## Part 2 · The EmotionLens pipeline, step by step

### Step ① · Load the photo
🔍 [`analyze.to_rgb`](../emotionlens/analyze.py) accepts a file path, a PIL image or an array and always
returns an **RGB uint8 array**. Normalizing inputs at the front door prevents the BGR bug above.

### Step ② · Find the faces (object detection)
🧠 **Object detection** answers "what is where?" with **bounding boxes**. Each box comes with a
**confidence score**. YOLO ("You Only Look Once") looks at the whole image in a single pass and predicts
boxes for every region at once. That makes it fast enough for video.

🔍 [`analyze.detect_faces`](../emotionlens/analyze.py) keeps boxes with confidence ≥ 0.4 and size ≥ 24 px,
then sorts them left to right so "Face 1" is always the leftmost.

💻 In the notebook: draw the raw boxes and move `min_conf`. At 0.1 you'll see false faces (patterns, logos);
at 0.9 you'll lose small or turned faces. **Every detector has this precision/recall knob.**

⚠️ **Fallback.** If YOLO can't load, OpenCV's **Haar cascade** (a 2001-era method built on hand-designed
light/dark patterns) takes over. It's worse but always available. Good systems degrade gracefully.

✅ <details><summary><b>Check yourself:</b> why sort faces left to right?</summary>
Detectors return boxes in arbitrary order. Sorting makes "Face 1, Face 2…" stable, so the cards in the
app line up with what you see in the photo.</details>

### Step ③ · Crop with a margin
🧠 The expression model was trained on FER-2013, where each image shows the **whole head** loosely
framed. Face detectors box the face *tightly*. To make our input look like the training data, we add a
**20 % margin** on every side. *Matching your input to the training distribution* is one of the most
important habits in applied ML.

🔍 [`analyze.crop_face`](../emotionlens/analyze.py), which clips the margin at the image border so the
crop never goes out of bounds.

### Step ④ · Resize + normalize
🔍 Done inside the Hugging Face **image processor** (Part 1 · idea 3): crop → 224×224 → scaled and normalized
→ tensor `(1, 3, 224, 224)`. The notebook prints the min/max/mean of the tensor so you can see the
numbers change from 0…255 to roughly −1…1.

### Step ⑤ · The Vision Transformer
🧠 A **ViT** cuts the 224×224 image into a grid of **16×16-pixel patches** (14 × 14 = 196 patches),
turns each patch into a vector, and lets every patch "look at" every other patch with **attention**.
That way a mouth patch can combine with an eye patch to recognise a smile. A CNN (like YOLO) instead
slides small filters across the image and builds up from edges → textures → parts → objects.

### Step ⑥ · Probabilities
🔍 [`analyze.classify_crops`](../emotionlens/analyze.py) returns 7 probabilities per face (angry, disgust,
fear, happy, neutral, sad, surprise), which sum to 1. For **video**, the same thing runs on a few frames per
second. Faces are linked across frames by **IoU** (intersection-over-union: how much two boxes overlap,
0 = none, 1 = identical), and a rolling average smooths out single-frame flicker.

### Step ⑦ · Why did it say that? (explainability)
🧠 **Occlusion sensitivity:** cover one patch of the face with grey, re-run the model, and measure how much
the prediction drops. Repeat across a 7×7 grid. Big drops mean that region **mattered**. It works for any
model, because it only needs inputs and outputs.

🔍 [`explain.occlusion_map`](../emotionlens/explain.py). It measures the drop in **log-odds** rather than
probability, because at 99.9 % the probability barely moves even when the model's internal evidence
changes a lot.

💻 The notebook draws the heat map for a smiling face. The mouth should light up. If the background lit
up instead, the model would be relying on something irrelevant (a "shortcut"), and that's exactly what this
check is for.

### Step ⑧ · Is it right? (evaluation)
🧠 Never judge a model on the data it learned from. FER-2013 has a **train** split (the model learned from
it), a **validation** split (for tuning) and a **test** split (only for the final score). We report on
**test: 3,589 faces the model never saw**.

| Metric | Plain English | EmotionLens |
|---|---|---|
| **Accuracy** | % of faces labeled correctly | 71.8 % |
| **Baseline** | accuracy of always guessing the most common class ("happy") | 24.5 % |
| **Precision** (per class) | when it says *fear*, how often is it really fear? | 0.61 |
| **Recall** (per class) | of all real *fear* faces, how many did it catch? | 0.56 |
| **Macro-F1** | precision & recall combined, averaged so rare classes count equally | 0.711 |
| **Confusion matrix** | a grid of *true label × predicted label*, which shows **which** mistakes happen | fear ↔ sad, angry ↔ disgust |

✅ <details><summary><b>Check yourself:</b> why compare with a baseline at all?</summary>
"72 % accurate" means nothing alone. If 70 % of faces were happy, guessing "happy" every time would score
70 %. The baseline tells you how much the model actually adds.</details>

### Step ⑨ · Is it honest? (calibration)
🧠 A model is **calibrated** if, when it says "80 %", it's right about 80 % of the time. Group predictions
by confidence and compare each group's average confidence with its actual accuracy. The average gap is the
**expected calibration error (ECE)**.

🔍 [`evaluate.py`](../emotionlens/evaluate.py) found this model **over-confident**: predictions above
90 % confidence were right only 82 % of the time (ECE 0.183). The fix is **temperature scaling**: divide
the logits by one number *T* before softmax. *T* > 1 softens the probabilities. *T* is fitted on the
**validation** split (never test!) and comes out at **T = 1.81**, cutting ECE to **0.061**. Accuracy doesn't
change, because dividing by T never changes which class is biggest.

---

## Part 3 · Using vision responsibly
A facial expression is a **signal, not a feeling**: people smile politely and hide sadness, and expression
norms vary across cultures. FER-2013 also has demographic imbalances. Good computer-vision work includes
saying what a model **shouldn't** be used for, here hiring, grading or policing.

## 📖 Glossary
| Term | Meaning |
|---|---|
| **Pixel / channel** | one cell of the image grid / one of its R, G, B values |
| **Bounding box** | `(x1, y1, x2, y2)` rectangle around an object |
| **IoU** | overlap of two boxes ÷ their combined area |
| **Tensor** | a multi-dimensional array; images are `(N, C, H, W)` in PyTorch |
| **Logit / softmax** | raw class score / the function that turns scores into probabilities |
| **Transfer learning** | reusing a model trained on another (usually bigger) dataset |
| **CNN / ViT** | convolution-based / attention-based vision architectures |
| **Train / val / test split** | data to learn from / to tune on / to report on, kept separate |
| **Calibration / ECE** | whether confidence matches accuracy / the average mismatch |
| **Occlusion sensitivity** | hide parts of the input to see which parts drive the prediction |

## 🧭 Where to go next
* **Hugging Face Computer Vision Course**: free, hands-on, uses the same libraries as this repo.
* **fast.ai *Practical Deep Learning for Coders***: train your own image classifier in lesson 1.
* **Stanford CS231n notes**: the classic, deeper explanation of CNNs.
* **Exercise:** swap `trpakov/vit-face-expression` in `models.py` for another image-classification model
  from the Hub, re-run `python -m emotionlens.evaluate --n 600`, and compare accuracy *and* ECE.
