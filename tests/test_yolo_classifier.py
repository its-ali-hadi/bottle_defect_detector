from pathlib import Path

import pytest

from bottle_detector.yolo_classifier import YoloCleanClassifier


def test_missing_model_file_fails_fast(tmp_path: Path) -> None:
    missing_model_path = tmp_path / "yolo_bottle_classifier.pt"

    with pytest.raises(RuntimeError, match="YOLO pre-filter model not found"):
        YoloCleanClassifier(model_path=missing_model_path)
