"""Trains the YOLOv8 binary classifier (clean vs defective) consumed by
bottle_detector/yolo_classifier.py::YoloCleanClassifier.

Run scripts/prepare_yolo_dataset.py first to build training_data/ from the
project's labeled images.

Trains yolov8n-cls (nano), yolov8s-cls (small), and yolov8m-cls (medium) with
dropout/label-smoothing regularization plus extra live augmentation on top of
prepare_yolo_dataset.py's offline augmentation, then keeps whichever scores
higher on the held-out val split (the smaller/earlier model in
CANDIDATE_BASE_MODELS wins ties, since it's faster/lighter for a real-time
pre-filter and a val split this small can't meaningfully separate close scores
anyway). The medium candidate was added on the chance its larger capacity
helps it pick up the subtler real defects (small dirty specks, faint neck
deformation) that the smaller models were missing -- it is not guaranteed to
win, and given how small the dataset is, there's no guarantee any amount of
architecture/hyperparameter tuning meaningfully moves real-world accuracy;
the bottleneck is data volume, not model capacity (see below).

PROTOTYPE WARNING: as of this writing there are ~135 base labeled images (see
prepare_yolo_dataset.py's module docstring for where they come from and why
that's still below the commonly-cited 150-300+ per class). No amount of extra
epochs, regularization, augmentation, or larger model candidates changes that
ceiling -- the model this script produces is a working proof-of-concept that
validates the whole YOLO-then-Claude pipeline end to end, not a
production-grade classifier on its own. Re-run this script (no code changes
needed) once more real, Claude-labeled crops have accumulated in outputs/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "training_data"
OUTPUT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_bottle_classifier.pt"

CANDIDATE_BASE_MODELS = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt"]
EPOCHS = 150
IMAGE_SIZE = 224
PATIENCE = 40

TRAIN_KWARGS = dict(
    dropout=0.3,
    label_smoothing=0.1,
    weight_decay=0.001,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    shear=5.0,
    hsv_h=0.02,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    erasing=0.2,
)


def train_one(base_model: str) -> tuple[Path, float]:
    run_name = f"yolo_bottle_classifier_{Path(base_model).stem}"
    model = YOLO(base_model)
    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(PROJECT_ROOT / "training_runs"),
        name=run_name,
        exist_ok=True,
        patience=PATIENCE,
        **TRAIN_KWARGS,
    )
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"Training finished but {best_weights} was not produced.")

    metrics = YOLO(str(best_weights)).val(
        data=str(DATASET_DIR),
        split="val",
        project=str(PROJECT_ROOT / "training_runs"),
        name=f"{run_name}_val",
        exist_ok=True,
    )
    top1_acc = float(metrics.top1)
    return best_weights, top1_acc


def train() -> Path:
    if not DATASET_DIR.exists():
        raise RuntimeError(f"{DATASET_DIR} not found. Run scripts/prepare_yolo_dataset.py first.")

    results: list[tuple[str, Path, float]] = []
    for base_model in CANDIDATE_BASE_MODELS:
        print(f"\n=== Training {base_model} ===")
        best_weights, top1_acc = train_one(base_model)
        print(f"{base_model}: val top1_acc={top1_acc:.4f}")
        results.append((base_model, best_weights, top1_acc))

    # On ties (common with a val split this small), prefer the smaller/earlier
    # model in CANDIDATE_BASE_MODELS -- faster and lighter for a real-time
    # pre-filter, and a handful of val images can't reliably separate close
    # scores anyway.
    best_base_model, best_weights, best_acc = max(
        results, key=lambda item: (item[2], -CANDIDATE_BASE_MODELS.index(item[0]))
    )
    print(f"\nSelected {best_base_model} (val top1_acc={best_acc:.4f}) as the better candidate.")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_weights, OUTPUT_MODEL_PATH)
    return OUTPUT_MODEL_PATH


if __name__ == "__main__":
    model_path = train()
    print(f"\nTrained model copied to {model_path}")
    print("This is a PROTOTYPE trained on a very small dataset -- see the module")
    print("docstrings here and in prepare_yolo_dataset.py before relying on it for")
    print("real production defect screening.")
