# Python AI Bottle Defect Detector

Detect white bottles on a conveyor, crop one clear image per bottle, analyze the crop with Anthropic Claude, and export JSON plus Excel reports.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
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
- `--model`: Anthropic model override. Defaults to `.env` `ANTHROPIC_MODEL` or `claude-sonnet-4-6`.
- `--no-ai`: save crops and JSON without calling Claude.
- `--max-bottles`: stop after N captured bottles, useful for API testing.
- `--no-display`: run without opening an OpenCV preview window.
- `--frame-limit`: stop after N frames, useful for quick debugging.
