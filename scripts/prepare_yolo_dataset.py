"""Builds a YOLOv8 classification dataset (clean vs defective) from real project images.

Labeled data comes from two places:
1. assets/reference_bottles/ -- the few-shot reference photos already used to steer
   Claude (clean_ok.jpg is the one clean example; every other file is a confirmed
   defective example, regardless of which specific defect type it shows).
2. outputs/crops/ + outputs/detections.json -- real crops from live runs. If
   detections.json was produced by an actual Claude run (model != "no-ai"), each
   crop's label comes from whether Claude flagged any defects for it. Otherwise
   (e.g. the current snapshot, which is from a --no-ai smoke test and carries no
   real verdicts) labels fall back to scripts/manual_crop_labels.json, a small
   hand-verified manifest.

IMPORTANT — data-size caveat: as of this writing there are ~142 labeled images
(9 reference photos + 133 real crops from outputs/crops/, the latter now including
120 crops pulled from 6 real conveyor videos in videos/ and individually visually
inspected under a zero-tolerance standard -- see scripts/manual_crop_labels.json's
_comment: any visible mark on the bottle at all, no matter how small, is labeled
defective/"dirty"; a bottle is only "clean" if its surface is pure unbroken white).
That is still below what a production-grade classifier needs (commonly cited
guidance is 150-300+ images per class), though the classes are much closer to
balanced now (86 clean / 56 defective from real crops) than in an earlier version
of this dataset, which under-flagged small marks and ended up ~7x skewed toward
clean. This script exists to produce a working *prototype* dataset now, structured
so re-running it later -- once more real, Claude-labeled crops have accumulated in
outputs/ -- produces a properly-sized dataset with zero code changes.

To partially compensate for the small, unbalanced source set, this script generates
augmented variants of every training image -- a random mix of horizontal flip,
rotation, perspective warp, brightness/contrast/color/sharpness jitter, zoom+crop,
Gaussian blur, Gaussian noise, and random erasing (occlusion patches) -- and boosts
the variant count for whichever class has fewer real photos so both classes end up
with a similar number of final training images (see the class-balancing block in
build_dataset()), instead of letting the majority class dominate what the model
learns to predict. This does not manufacture real data diversity (lighting/
background/camera angle are still shared across most images of a given source) --
it only stretches what's already here so the model sees more distinct pixel
patterns per real photo and overfits a bit less easily. It is not a substitute for
collecting more real photos, especially more defective ones.

Output layout (consumed directly by scripts/train_yolo_classifier.py):
    training_data/
        train/clean/*.jpg
        train/defective/*.jpg
        val/clean/*.jpg
        val/defective/*.jpg
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ProjectLabel = Literal["clean", "defective"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference_bottles"
CROPS_DIR = PROJECT_ROOT / "outputs" / "crops"
DETECTIONS_JSON = PROJECT_ROOT / "outputs" / "detections.json"
MANUAL_LABELS_PATH = PROJECT_ROOT / "scripts" / "manual_crop_labels.json"
DATASET_DIR = PROJECT_ROOT / "training_data"

REFERENCE_CLEAN_BASENAME = "clean_ok"
AUGMENTED_VARIANTS_PER_IMAGE = 20
MAX_AUGMENTED_VARIANTS_PER_IMAGE = 60  # cap on the class-balancing boost below
VAL_HOLDOUT_PER_CLASS = 10
RANDOM_SEED = 42


def collect_reference_labels() -> dict[Path, ProjectLabel]:
    labels: dict[Path, ProjectLabel] = {}
    if not REFERENCE_DIR.exists():
        return labels
    for path in sorted(REFERENCE_DIR.glob("*.jpg")):
        label: ProjectLabel = "clean" if path.stem == REFERENCE_CLEAN_BASENAME else "defective"
        labels[path] = label
    return labels


def load_manual_labels() -> dict[str, ProjectLabel]:
    if not MANUAL_LABELS_PATH.exists():
        return {}
    payload = json.loads(MANUAL_LABELS_PATH.read_text(encoding="utf-8"))
    return payload.get("labels", {})


def load_detections_labels() -> dict[str, ProjectLabel] | None:
    """Real AI-derived labels, keyed by crop filename. None if detections.json isn't
    from a real Claude run (e.g. a --no-ai smoke test), since those carry no verdict."""
    if not DETECTIONS_JSON.exists():
        return None
    payload = json.loads(DETECTIONS_JSON.read_text(encoding="utf-8"))
    if payload.get("model") in (None, "no-ai"):
        return None

    labels: dict[str, ProjectLabel] = {}
    for detection in payload.get("detections", []):
        crop_path = detection.get("crop_path", "")
        filename = Path(crop_path).name
        if not filename:
            continue
        labels[filename] = "defective" if detection.get("defects") else "clean"
    return labels


def collect_crop_labels() -> dict[Path, ProjectLabel]:
    if not CROPS_DIR.exists():
        return {}

    detections_labels = load_detections_labels()
    manual_labels = load_manual_labels()
    source = "outputs/detections.json (real AI verdicts)" if detections_labels else "scripts/manual_crop_labels.json (fallback)"
    print(f"Crop labels source: {source}")

    labels: dict[Path, ProjectLabel] = {}
    for path in sorted(CROPS_DIR.glob("*.jpg")):
        label = (detections_labels or {}).get(path.name) or manual_labels.get(path.name)
        if label is None:
            print(f"  skipping {path.name}: no label available")
            continue
        labels[path] = label
    return labels


def apply_perspective_warp(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    width, height = image.size
    jitter = lambda: rng.uniform(-strength, strength)  # noqa: E731
    src = [(0, 0), (width, 0), (width, height), (0, height)]
    dst = [
        (jitter() * width, jitter() * height),
        (width + jitter() * width, jitter() * height),
        (width + jitter() * width, height + jitter() * height),
        (jitter() * width, height + jitter() * height),
    ]
    # Solve the 8 perspective coefficients PIL's QUAD->PERSPECTIVE transform needs.
    matrix = []
    for (x, y), (px, py) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -px * x, -px * y])
        matrix.append([0, 0, 0, x, y, 1, -py * x, -py * y])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    try:
        coeffs = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return image
    return image.transform((width, height), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def apply_random_erasing(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    patch = image.copy()
    erase_w = int(width * rng.uniform(0.08, 0.22))
    erase_h = int(height * rng.uniform(0.08, 0.22))
    left = rng.randint(0, max(0, width - erase_w))
    top = rng.randint(0, max(0, height - erase_h))
    fill = tuple(rng.randint(150, 255) for _ in range(3))
    block = Image.new("RGB", (erase_w, erase_h), fill)
    patch.paste(block, (left, top))
    return patch


def apply_gaussian_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(3, 12)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def augment_variants(image: Image.Image, count: int, rng: random.Random) -> list[Image.Image]:
    """Wide, aggressive augmentation pool -- given only ~25 base photos, squeezing
    real signal out of them matters far more than it would with a normal-sized
    dataset. Each variant applies a random subset of these transforms so the
    trainer sees a genuinely varied mix, not the same fixed pipeline every time."""
    variants: list[Image.Image] = []
    for _ in range(count):
        variant = image

        if rng.random() < 0.5:
            variant = variant.transpose(Image.FLIP_LEFT_RIGHT)

        angle = rng.uniform(-25, 25)
        variant = variant.rotate(angle, expand=False, fillcolor=(255, 255, 255))

        if rng.random() < 0.6:
            variant = apply_perspective_warp(variant, rng.uniform(0.03, 0.09), rng)

        variant = ImageEnhance.Brightness(variant).enhance(rng.uniform(0.7, 1.3))
        variant = ImageEnhance.Contrast(variant).enhance(rng.uniform(0.75, 1.25))
        variant = ImageEnhance.Color(variant).enhance(rng.uniform(0.7, 1.3))
        variant = ImageEnhance.Sharpness(variant).enhance(rng.uniform(0.5, 1.8))

        width, height = variant.size
        zoom = rng.uniform(1.0, 1.25)
        crop_w, crop_h = int(width / zoom), int(height / zoom)
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        variant = variant.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))

        if rng.random() < 0.35:
            variant = variant.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.8)))
        if rng.random() < 0.35:
            variant = apply_gaussian_noise(variant, rng)
        if rng.random() < 0.3:
            variant = apply_random_erasing(variant, rng)

        variants.append(variant)
    return variants


def build_dataset() -> None:
    # Separate RNGs on purpose: the train/val split must stay stable and reproducible
    # regardless of how augmentation logic changes (more/fewer random calls per image
    # would otherwise silently shift a shared RNG's state and pick different held-out
    # images between runs -- which happened here once before this was split out).
    split_rng = random.Random(RANDOM_SEED)
    augment_rng = random.Random(RANDOM_SEED + 1)

    all_labels: dict[Path, ProjectLabel] = {}
    all_labels.update(collect_reference_labels())
    all_labels.update(collect_crop_labels())

    by_class: dict[ProjectLabel, list[Path]] = {"clean": [], "defective": []}
    for path, label in all_labels.items():
        by_class[label].append(path)

    print(f"Base labeled images: {len(by_class['clean'])} clean, {len(by_class['defective'])} defective")
    if len(by_class["clean"]) < 3 or len(by_class["defective"]) < 3:
        raise RuntimeError(
            "Not enough labeled images to build even a prototype dataset "
            f"(clean={len(by_class['clean'])}, defective={len(by_class['defective'])}). "
            "Check assets/reference_bottles/ and outputs/crops/."
        )

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        for label in ("clean", "defective"):
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)

    splits_by_class: dict[ProjectLabel, tuple[list[Path], list[Path]]] = {}
    for label, paths in by_class.items():
        shuffled = paths[:]
        split_rng.shuffle(shuffled)
        val_count = min(VAL_HOLDOUT_PER_CLASS, max(1, len(shuffled) // 5))
        splits_by_class[label] = (shuffled[val_count:], shuffled[:val_count])

    # Real photo counts aren't balanced between classes (far more "clean" crops
    # accumulate naturally than "defective" ones), so a fixed augmentation count
    # per image would leave the model trained on a lopsided training set that
    # skews it toward predicting "clean". Instead, boost the minority class's
    # variant count so both classes end up with roughly the same number of final
    # training images, capped so a handful of defective photos aren't stretched
    # into an absurd number of near-duplicate variants.
    base_train_totals = {
        label: len(train_paths) * (1 + AUGMENTED_VARIANTS_PER_IMAGE)
        for label, (train_paths, _val_paths) in splits_by_class.items()
    }
    target_total = max(base_train_totals.values())
    variants_per_class: dict[ProjectLabel, int] = {}
    for label, (train_paths, _val_paths) in splits_by_class.items():
        if not train_paths:
            continue
        needed = -(-target_total // len(train_paths)) - 1  # ceil division, minus the 1 real image
        variants_per_class[label] = max(AUGMENTED_VARIANTS_PER_IMAGE, min(needed, MAX_AUGMENTED_VARIANTS_PER_IMAGE))
    print(f"Augmented variants per training image, by class: {variants_per_class}")

    for label, (train_paths, val_paths) in splits_by_class.items():
        for split, split_paths in (("train", train_paths), ("val", val_paths)):
            for source_path in split_paths:
                dest_dir = DATASET_DIR / split / label
                base_image = Image.open(source_path).convert("RGB")
                base_image.save(dest_dir / source_path.name, quality=92)
                # Only augment the training split -- val must stay real, unmodified images.
                if split == "train":
                    for index, variant in enumerate(
                        augment_variants(base_image, variants_per_class[label], augment_rng), start=1
                    ):
                        variant.save(dest_dir / f"{source_path.stem}_aug{index}.jpg", quality=92)

    for split in ("train", "val"):
        for label in ("clean", "defective"):
            count = len(list((DATASET_DIR / split / label).glob("*.jpg")))
            print(f"{split}/{label}: {count} images")


if __name__ == "__main__":
    build_dataset()
    print(f"\nDataset ready at {DATASET_DIR}")
