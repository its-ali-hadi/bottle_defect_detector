# models/

This directory holds the trained YOLOv8 classification model used by the optional
YOLO pre-filter (`--use-yolo-prefilter`), which screens out confidently-clean bottles
before they ever reach Claude.

Expected file: `yolo_bottle_classifier.pt` — a two-class YOLOv8 classification model
(`yolov8n-cls` or `yolov8s-cls`) with classes named exactly `clean` and `defective`.

Weight files (`*.pt`) are git-ignored on purpose (they're large binaries produced by
training, not source) — this directory itself is tracked so the expected path always
exists.

A prototype `yolo_bottle_classifier.pt` has been trained locally via
`scripts/prepare_yolo_dataset.py` + `scripts/train_yolo_classifier.py` (class-balanced
augmented variants per training image — more variants for whichever class has fewer
real photos — dropout/label-smoothing regularization, nano vs small model
comparison), but it is **not production-grade**: the underlying data is ~135 base
labeled images (85 clean, 50 defective — see `scripts/manual_crop_labels.json`),
mostly from real conveyor footage but still from a small number of camera setups.
That is close to but still under the ~150-300+ images per class a real classifier
needs, and it risks having learned incidental scene details rather than the bottle
itself. Labels use a **zero-tolerance standard**: any visible mark on the bottle at
all, no matter how small, is labeled defective — a bottle is only "clean" if its
surface is pure unbroken white. (An earlier pass on this dataset treated some faint
1-3px specks as camera noise and left them labeled clean; that was wrong — even a
tiny visible spot is a real defect for this line's inspection purposes — and was
corrected after a specific miss was flagged during review. Re-check any future
labeling pass against this standard before retraining.) Current held-out validation
accuracy is 95%, but with only 20 held-out images that number is indicative, not
statistically strong. Treat this model as a proof that the YOLO-then-Claude wiring
works end to end and applies a strict, consistent cleanliness standard, not as
something to trust unsupervised for real defect screening yet.

Every crop in this dataset was captured with the full-bottle-height fix in
`bottle_detector/tracking.py::project_bottle_bases` (see `CLAUDE.md`'s Detection
section) — the crop spans the bottle's full extent from base through the
screw-thread neck/opening, not just the cylindrical body. An earlier version of
this pipeline cropped only the body, which hid neck-area deformities entirely
(a real defect category on this line) and occasionally missed defects sitting
near the top edge of the old, shorter crop. If `outputs/crops/*.jpg` ever looks
noticeably shorter than the full bottle again (missing the neck), that fix has
regressed — re-verify `project_bottle_bases` against a raw video frame before
trusting new labels or a new training run.

To retrain as more real, Claude-labeled crops accumulate in `outputs/` (no code
changes needed):
```bash
.venv/bin/python scripts/prepare_yolo_dataset.py
.venv/bin/python scripts/train_yolo_classifier.py
```

If this file is ever missing (e.g. a fresh clone, before any training has run),
`--use-yolo-prefilter` fails immediately with a clear error instead of silently
falling back to sending every bottle to Claude — this is intentional (see
`bottle_detector/yolo_classifier.py::YoloCleanClassifier`).
