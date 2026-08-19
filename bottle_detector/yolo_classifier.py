from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLEAN_CLASS_NAME = "clean"
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD = 0.90


@dataclass
class YoloVerdict:
    is_clean: bool
    confidence: float
    predicted_class: str


class YoloCleanClassifier:
    """Fast binary pre-filter that runs before Claude: is this bottle clean, or does it
    need Claude's full diagnosis?

    This is a two-class YOLOv8 classification model (classes "clean" and "defective"),
    trained separately from this codebase — see the Phase 2 training script, which
    consumes labeled crops accumulated from real runs (crop_path + defect verdicts in
    outputs/detections.json). There is no bundled trained model; a missing model file
    is treated as a hard configuration error rather than silently falling back to
    Claude-for-everything, since this pre-filter is opt-in and expected to be ready
    when enabled.

    Deliberately conservative: only reports is_clean=True when the model's "clean"
    prediction clears clean_confidence_threshold. Anything else — a "defective"
    prediction, or a "clean" prediction the model isn't confident about — is left
    for Claude to diagnose, so an uncertain YOLO call never lets a real defect through
    unseen.
    """

    def __init__(
        self,
        model_path: Path,
        clean_confidence_threshold: float = DEFAULT_CLEAN_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"YOLO pre-filter model not found at {model_path}. "
                "Train a two-class (clean/defective) YOLOv8 classification model first "
                "(see the Phase 2 training script) or run without --use-yolo-prefilter "
                "to send every bottle straight to Claude."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The ultralytics package is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(str(model_path))
        self.clean_confidence_threshold = clean_confidence_threshold

    def predict(self, crop_bgr: Any) -> YoloVerdict:
        results = self.model.predict(source=crop_bgr, verbose=False)
        probs = results[0].probs
        top_index = int(probs.top1)
        confidence = float(probs.top1conf)
        predicted_class = results[0].names[top_index]

        is_clean = predicted_class == CLEAN_CLASS_NAME and confidence >= self.clean_confidence_threshold
        return YoloVerdict(is_clean=is_clean, confidence=confidence, predicted_class=predicted_class)
