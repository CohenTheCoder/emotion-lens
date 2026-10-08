"""EmotionLens — facial-expression and text emotion recognition with Hugging Face models."""
# Import torch before scikit-learn / pyarrow: on macOS they ship different OpenMP
# runtimes, and loading torch second can crash the interpreter.
import torch  # noqa: F401
