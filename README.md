# Python AI Bottle Defect Detector

Detect white bottles on a conveyor, crop one clear image per bottle, analyze the crop with Anthropic Claude, and export JSON plus Excel reports.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` includes `ultralytics` (for the optional YOLO pre-filter below), which pulls in `torch`/`torchvision`. If you hit `RuntimeError: operator torchvision::nms does not exist`, it means pip resolved mismatched builds (e.g. a `+cpu` torch with a non-`+cpu` torchvision) — fix it by reinstalling both from the same index:
```bash
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The app reads the Anthropic key from `.env`. It supports both names:

```env
anthropic_api_key=...
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-5
```

## Run With The Test Video

Smoke test without AI:

```bash
.venv/bin/python -m bottle_detector --source assets/test.mp4 --no-ai --no-display
```

Limited AI run:

```bash
.venv/bin/python -m bottle_detector --source assets/test.mp4 --max-bottles 3
```

Full video run:

```bash
.venv/bin/python -m bottle_detector --source assets/test.mp4
```

## Run With A Camera

Camera `0` is the default source:

```bash
.venv/bin/python -m bottle_detector
```

Or explicitly:

```bash
.venv/bin/python -m bottle_detector --source 0
```

Press `q` in the preview window to stop and write the final JSON.

## Run The GUI

Install GUI dependencies:

```bash
.venv/bin/python -m pip install -r requirements-gui.txt
```

Start the desktop app:

```bash
.venv/bin/python run_gui.py
```

The GUI lets you refresh/select a camera, choose a max bottle count or run unlimited until you press stop, start analysis, stop the camera safely, and open the generated Excel report or result folder.

## Output

Default output:

```text
outputs/detections.json
outputs/crops/bottle_0001.jpg
outputs/crops/bottle_0002.jpg
result/detections_YYYYMMDD_HHMMSS.xlsx
result/detections_latest.xlsx
```

Each JSON detection includes the bottle sequence, video timestamp, crop path, the compact defect list, Arabic summary, and a confidence score. The AI is asked only for `defects.type`, `defects.description_ar`, `summary_ar`, and `confidence` (Arabic labels are filled in locally) to keep output tokens minimal. Three defect categories are inspected — this line manufactures bottles without caps, so cap defects are never inspected:
- `body_defect`: dents, scratches, cracks, or a warped/crushed neck-opening ring — moderate, localized flaws.
- `dirty`: any visible mud/dirt staining, flagged at any size down to a single small spot, distinguished from lighting shadows by color tone/edge shape.
- `factory_defect`: a large, prominent, unmistakable manufacturing malformation (e.g. a big lump of excess plastic) that dominates the bottle's shape at a glance — reserved for gross structural flaws, not small/moderate ones (those stay `body_defect`).

Every analysis call also sends a few reference photos from `assets/reference_bottles/` alongside the target crop, so Claude has a real visual anchor for what this bottle model's defects look like instead of judging from a text description alone: `clean_ok.jpg` (a known-good bottle), `dent_defect_1.jpg`/`dent_defect_2.jpg` and `body_defect.jpg` (confirmed `body_defect` examples — body dents and a neck-opening warp respectively), `dirty_1.jpg`–`dirty_4.jpg` (`dirty` examples), and `factorial_damage.jpg` (a confirmed `factory_defect` example). If a defect type is being missed or a normal design feature is being flagged as a defect, add/replace the images in that folder with clearer examples — no code changes needed, `ai.py` picks up any file present under those names automatically.

Each detection also carries `accuracy_pct`, `precision_pct`, and `recall_pct`, derived from the AI's own confidence score (there's no manual ground-truth inspection in this pipeline, so these are a confidence-based proxy, not statistically rigorous metrics): `accuracy_pct` reflects confidence in the overall verdict, `precision_pct` is set when a defect was flagged, and `recall_pct` is set when the bottle was verdict-clean. The JSON's top-level `statistics` object aggregates these same proxies across the whole run (`accuracy_pct` averaged over all bottles, `precision_pct` over defective bottles only, `recall_pct` over clean bottles only), and the Excel report surfaces both the per-bottle and run-level numbers.

After every completed detector run, Excel export runs automatically. A timestamped report is always created; `detections_latest.xlsx` is updated when it is not locked/open.

Every run also writes to `logs/bottle_detector.log` (rotated at 5MB, 5 backups kept) alongside the console — this is the first place to check if a run behaved unexpectedly, especially for the GUI/EXE where there's no visible terminal on the actual production line.

Each detection also carries `analysis_stage`, either `"claude"` (full diagnosis) or `"yolo"` (see below — the fast pre-filter was confident enough to skip Claude for this bottle).

## Optional YOLO Pre-Filter

`--use-yolo-prefilter` inserts a fast binary classifier (clean vs. defective) ahead of Claude in the pipeline: bottles YOLO is confident are clean skip the Claude call entirely (saving cost and latency), while anything YOLO flags as defective — or isn't confident about — still gets Claude's exact same full diagnosis as today. This is deliberately conservative: a low-confidence "clean" guess is treated the same as "defective" and forwarded to Claude, so an uncertain YOLO call never lets a real defect through unseen.

```bash
.venv/bin/python -m bottle_detector --source assets/test.mp4 --use-yolo-prefilter
```

This requires a trained two-class YOLOv8 classification model (classes named exactly `clean` and `defective`) at `models/yolo_bottle_classifier.pt` (override with `--yolo-model-path`). Running `--use-yolo-prefilter` without a trained model present fails immediately with a clear error (see `bottle_detector/yolo_classifier.py::YoloCleanClassifier`) — it does not silently fall back to sending everything to Claude, since this flag is opt-in and only meant to be turned on once a model actually exists.

**A prototype model is already trained and checked in working end-to-end, but it is not production-grade** — it was trained on ~135 base labeled images (85 clean, 50 defective; see `scripts/manual_crop_labels.json`), most of them real crops pulled from 6 conveyor videos and individually visually inspected, padded out with class-balanced augmented variants per training image (flip/rotation/perspective-warp/brightness/contrast/color/sharpness jitter/zoom/blur/noise/random-erasing — more variants for whichever class has fewer real photos) and trained with dropout/label-smoothing regularization, auto-picking the better of `yolov8n-cls`/`yolov8s-cls` on the held-out split. That's close to but still under the ~150-300+ images per class a real classifier needs, and it risks having learned incidental scene details rather than the bottle itself.

Labels use a **zero-tolerance standard**: any visible mark on the bottle — a dark speck, a faint discoloration, anything — no matter how small, is labeled `defective`. A bottle is only `clean` if its surface is pure, unbroken white with nothing visible on it at all. Current held-out validation accuracy is 95%, though with only 20 held-out images that number is indicative rather than statistically strong. Treat this as a working proof that the YOLO-then-Claude wiring is correct and applies a strict, consistent cleanliness standard, not as something to trust unsupervised for real defect screening yet.

Every crop this model was trained on captures the bottle's **full height**, base through the screw-thread neck/opening — not just the cylindrical body. An earlier version of the detector cropped only the body, which hid neck-area deformities (a real `factory_defect`/`body_defect` category on this line) and occasionally missed defects near the top edge. If crops start looking noticeably shorter again, check `bottle_detector/tracking.py::project_bottle_bases` before trusting new labels.

To retrain (no code changes needed, just more/better labeled images):
```bash
.venv/bin/python scripts/prepare_yolo_dataset.py   # rebuilds training_data/ from assets/reference_bottles/ + outputs/crops/
.venv/bin/python scripts/train_yolo_classifier.py  # trains yolov8n-cls + yolov8s-cls, keeps the better one at models/yolo_bottle_classifier.pt
```
Labels for `outputs/crops/*.jpg` come from `outputs/detections.json` automatically whenever that file is from a real Claude run (not `--no-ai`); `scripts/manual_crop_labels.json` is a hand-verified fallback for crops without real AI verdicts. The more real, varied bottles the main pipeline analyzes over time, the better a re-trained model gets — this is the intended path to a production-grade model.

## Convert JSON To Excel

Convert the latest JSON result into an Excel report under `result/`:

```bash
.venv/bin/python json_to_excel.py
```

Default paths:

```text
input:  outputs/detections.json
output: result/detections_YYYYMMDD_HHMMSS.xlsx
latest: result/detections_latest.xlsx
```

The workbook opens on a full Arabic report sheet with the summary at the top and bottle-by-bottle details below it. It also includes `Summary` and `Bottle Details` sheets only. Both the `Report` and `Bottle Details` sheets embed a thumbnail of the bottle's crop image next to its analysis in the last column, resolved from each detection's `crop_path`; if the crop file is missing, the cell shows "لا توجد صورة" instead.

## Build Windows EXE

Build on a Windows machine with Python installed:

```bat
build_windows.bat
```

The executable will be created at:

```text
dist\BottleDefectDetector\BottleDefectDetector.exe
```

Copy `.env.example` to `.env` inside `dist\BottleDefectDetector`, then add the real Anthropic API key. The EXE writes `outputs\` and `result\` beside itself, so the operator machine does not need Python installed.

If `.env` already exists in the project when you run `build_windows.bat`, it is copied automatically into `dist\BottleDefectDetector\.env`. This makes the final folder ready to send without extra setup. Anyone who receives that folder can read/use the included API key, so share it only with trusted operators.

## Useful Options

```bash
.venv/bin/python -m bottle_detector --help
```

- `--source`: camera index such as `0`, or a video file path.
- `--output`: final JSON path.
- `--crops-dir`: folder for analyzed bottle crops.
- `--model`: Anthropic model override. Defaults to `.env` `ANTHROPIC_MODEL` or `claude-sonnet-5`.
- `--no-ai`: save crops and JSON without calling Claude.
- `--max-bottles`: stop after N captured bottles, useful for API testing.
- `--no-display`: run without opening an OpenCV preview window.
- `--frame-limit`: stop after N frames, useful for quick debugging.
- `--use-yolo-prefilter`: screen bottles with a trained YOLOv8 binary classifier before Claude; confidently-clean bottles skip Claude entirely. Requires a trained model at `--yolo-model-path` (default `models/yolo_bottle_classifier.pt`) — fails immediately if none exists, it does not silently fall back to Claude-for-everything.
- `--yolo-model-path`: path to the trained YOLO classification model (`.pt`). Only used with `--use-yolo-prefilter`.
- `--yolo-confidence`: minimum confidence (default `0.90`) for YOLO to mark a bottle clean without forwarding it to Claude. Deliberately conservative — anything less confident, or predicted defective, still gets Claude's full diagnosis.
