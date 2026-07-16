# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI-assisted quality-inspection tool: it watches a conveyor belt (camera or video file) for white bottles, crops one clear image per bottle, sends the crop to Anthropic Claude for defect analysis, and writes results to JSON + Excel (with Arabic-language defect labels/summaries). Ships as both a CLI (`bottle_detector`) and a PySide6 desktop GUI (`run_gui.py`), the latter packaged into a Windows EXE via PyInstaller.

## Commands

Setup:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # add requirements-gui.txt for the desktop GUI
```

Run the detector (CLI):
```bash
.venv/bin/python -m bottle_detector --source assets/test.mp4 --no-ai --no-display   # smoke test, no API calls
.venv/bin/python -m bottle_detector --source assets/test.mp4 --max-bottles 3        # limited AI run
.venv/bin/python -m bottle_detector                                                 # camera 0, full run
.venv/bin/python -m bottle_detector --help                                          # see all tuning flags
```

Run the GUI:
```bash
.venv/bin/python run_gui.py
```

Convert an existing detections JSON to an Excel report:
```bash
.venv/bin/python json_to_excel.py
```

Tests (pytest, configured via `pyproject.toml` `testpaths = ["tests"]`):
```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_models.py
.venv/bin/python -m pytest tests/test_models.py::test_normalize_detection_payload_fills_arabic_labels
```

Build the Windows EXE (must run on Windows, uses a separate `.venv-win`):
```bat
build_windows.bat
```
Output: `dist\BottleDefectDetector\BottleDefectDetector.exe`. Note: if a `.env` exists in the project root when this script runs, it is copied automatically into the dist folder — anyone receiving that folder gets the embedded API key, so only hand it to trusted operators.

## Configuration

The Anthropic key/model are read from `.env` (see `.env.example`), supporting both `anthropic_api_key` and `ANTHROPIC_API_KEY`; model defaults to `ANTHROPIC_MODEL` or `claude-sonnet-5` (`bottle_detector/config.py`). `.env` is loaded twice in `AppConfig.__post_init__`: once resolved relative to the app base dir (`paths.resolve_app_path`, which uses `sys.executable`'s directory when frozen by PyInstaller, else cwd) and once via default `load_dotenv()` — this dual load is what lets the same code find `.env` both in dev and next to the frozen EXE.

## Architecture

Pipeline, wired together in `bottle_detector/runner.py::run_detector`, which both the CLI (`__main__.py`) and GUI (`run_gui.py`) call into:

1. **Frame source** (`cameras.py`) — opens a camera index (platform-specific backend: `CAP_DSHOW` on Windows, `CAP_V4L2` on Linux) or a video file via plain `cv2.VideoCapture`.
2. **Detection** (`tracking.py::BottleDetector`) — per-frame HSV color thresholding to find white blobs against the blue conveyor belt, locating the belt line first (`find_blue_belt_top`) then projecting bottle bases within a band around it (`project_bottle_bases`). Pure geometry/color heuristics, no ML model for detection itself.
3. **Tracking** (`tracking.py::BottleTracker`) — greedy nearest-centroid + IoU matching frame-to-frame to keep a stable `Track` per physical bottle; a track becomes a "capture candidate" once it's been seen a few frames and its centroid sits inside the configured `capture_start`/`capture_end` horizontal zone (avoids capturing bottles still entering/exiting frame).
4. **Async AI analysis** (`ai.py`) — each captured bottle crop is submitted to a single-worker `ThreadPoolExecutor` calling `AnthropicBottleAnalyzer.analyze` (or `NoAiAnalyzer` under `--no-ai`) so frame reading isn't blocked waiting on the API. Results are collected non-blockingly each loop iteration (`collect_finished`) and drained at the end (`wait_for_pending`). Analyzer failures are caught per-crop and turned into a synthetic "couldn't analyze" defect rather than aborting the run — a bad API call never kills the conveyor loop. Since the Anthropic API has no fine-tuning/persistent memory across calls, `AnthropicBottleAnalyzer.__init__` loads a small set of few-shot reference photos once (via `build_reference_content_blocks`, from `assets/reference_bottles/`) and prepends them to every single analysis call's message content, so the model has a real visual anchor for this bottle model's actual defects instead of judging from the text prompt alone. `REFERENCE_DEFECT_GROUPS` in `ai.py` maps each defect type to its own list of reference filenames + description text (`dent_defect_1/2.jpg` and `body_defect.jpg` for `body_defect`, `dirty_1-4.jpg` for `dirty`, `factorial_damage.jpg` for `factory_defect`) — any supported image extension (`.jpg`/`.jpeg`/`.png`, resolved via `find_reference_file`) present under those exact base names is picked up automatically, no code changes needed, missing files are silently skipped. The prompt draws a deliberate severity line between `body_defect` (moderate, localized: dents, scratches, a warped neck-opening ring) and `factory_defect` (large, gross, unmistakable structural malformation, e.g. a big lump of excess plastic) — this distinction was verified to hold under real API calls (the neck-warp example stays `body_defect` even with `factory_defect` available as a competing category). If detection is inconsistent on a particular defect look, the fix is almost always to swap in a clearer/more representative example image under `assets/reference_bottles/`, not to tweak the prompt further — this is exactly how the `body_defect` under-detection bug (bottles with a visible dent inconsistently marked "OK") got fixed. `build_windows.bat` copies this whole folder beside the packaged EXE the same way it copies `.env`, since `paths.resolve_app_path` resolves it relative to the frozen exe's own directory, not PyInstaller's bundle.
5. **Validation/normalization** (`models.py`) — Claude is asked to return minimal compact JSON (`defects[].type/description_ar`, `summary_ar`, `confidence`, no markdown) to keep output tokens low; `label_ar` is never requested from the model — it's always filled in locally from the fixed `LABELS_AR` map. `normalize_detection_payload` maps loose/aliased defect-type strings (e.g. "scratch", "stain", "molding") onto the three canonical `DefectType`s (`body_defect`, `dirty`, `factory_defect` — this conveyor line manufactures bottles without caps, so there is no cap-defect category) and clamps `confidence` into `[0, 1]`. Pydantic models enforce non-empty Arabic text via validators.
6. **Output** (`exporter.py`, `json_to_excel.py`) — detections are sorted by sequence and written as JSON; `write_run_output` also computes a run-level `RunStatistics` block via `compute_run_statistics`. `run_detector` then always calls `export_excel_report` to produce a timestamped `result/detections_TIMESTAMP.xlsx` plus `result/detections_latest.xlsx` (skipped/best-effort if that file is currently open/locked). The Excel workbook opens on a combined Arabic summary+details sheet, with `Summary` and `Bottle Details` as separate sheets. The `Report` and `Bottle Details` sheets both embed a resized JPEG thumbnail of each bottle's crop (via `write_bottle_rows`/`build_thumbnail` in `json_to_excel.py`) in the last column, anchored to that bottle's row with a matching row height; `resolve_crop_path` handles both absolute `crop_path` values (the normal case from a live run) and relative ones (e.g. hand-edited/legacy JSON) by trying them against cwd and against the input JSON's grandparent directory.

### Accuracy/Precision/Recall are confidence proxies, not real ML metrics

There is no manual ground-truth inspection step in this pipeline, so `accuracy_pct`/`precision_pct`/`recall_pct` are **not** statistically rigorous — they're derived entirely from the AI's own self-reported `confidence` (0-1) for its verdict on a bottle, both per-detection (`DetectionResult` computed fields in `models.py`) and aggregated per-run (`RunStatistics`/`compute_run_statistics`):
- `accuracy_pct` = confidence in the overall verdict — set for every bottle/run.
- `precision_pct` = confidence restricted to bottles where a defect *was* flagged — `None` when nothing was flagged.
- `recall_pct` = confidence restricted to bottles verdict-clean (no defects) — `None` when the bottle was flagged defective.
Per bottle, exactly one of `precision_pct`/`recall_pct` is populated (mutually exclusive by definition above) while `accuracy_pct` always is. If real precision/recall against ground truth is ever needed, it requires adding a manual-inspection input to compare against — there's no way to derive it from AI confidence alone.

Cross-cutting concerns:
- `progress_callback`/`preview_callback` in `run_detector` are how the GUI observes detector state (status text, live frame preview) without the runner knowing about Qt; the CLI just doesn't pass them and instead uses `--no-display`/OpenCV's own window.
- `stop_event` (a `threading.Event`) is the cooperative-cancellation mechanism — GUI's stop button and CLI's `q` keypress both feed into it, and on stop the runner still waits for in-flight AI analyses before writing final output.
- `paths.resolve_app_path` is the one place that needs to know about PyInstaller's frozen-exe layout; anything reading/writing files relative to the app root should go through it rather than assuming cwd.
