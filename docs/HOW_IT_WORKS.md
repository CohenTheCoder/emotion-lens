### The pipeline, one step at a time

**1 · Import models from Hugging Face.** Nothing is trained here. [`emotionlens/models.py`](../emotionlens/models.py)
pulls three pretrained models from the Hub:
a YOLOv8 face detector, a Vision Transformer (ViT) fine-tuned on FER-2013 for 7 facial expressions, and a
DistilRoBERTa text model for the same 7 emotions.

**2 · Find the faces.** YOLOv8-face draws a box around every face. If the photo is already a tight face crop
(like the dataset images), the whole picture is used. If the detector can't load, OpenCV's Haar cascade
takes over.

**3 · Read each expression.** Each face is cropped with a 20 % margin (FER-2013 faces include the whole
head), resized to 224×224, and passed through the ViT. Out come 7 probabilities that sum to 1.

**4 · Aggregate.**
* *Photos*: one card per face, numbered left to right.
* *Video*: sample a few frames per second and link faces over time by box overlap (IoU), giving each
  person one ID. A rolling mean smooths out single-frame flicker, and the result is an emotion timeline per
  person plus time-share stats.
* *Text*: the language model scores each line, with its labels mapped onto the face labels
  (joy→happy, sadness→sad…). That lets the app compare *what a face shows* with *what the words say*.

**5 · Explain (occlusion sensitivity).** Grey out one of 49 squares, re-run the model, and measure how much
the predicted emotion's probability drops. Big drops mean that region mattered. It works with any model
and needs no access to its internals.

**6 · Evaluate.** [`emotionlens/evaluate.py`](../emotionlens/evaluate.py) scores the model on FER-2013's
**test** split (the model was trained on the train split). It reports accuracy, macro-F1, a confusion
matrix, per-class precision/recall, and **calibration**: when the model says 80 %, is it right 80 % of the
time?

### Things to know
* FER-2013 faces are 48×48 greyscale. Bright, high-resolution photos work, but the model's world is small.
* *Disgust* is rare in the training data (~1.5 %), so it's the weakest class.
* A facial expression is a signal, not a feeling. Expression recognition shouldn't be used to make
  decisions about people (hiring, policing, grading).
