# Python AI Bottle Defect Detector

Detect white bottles on a conveyor, crop one clear image per bottle, analyze the crop with Anthropic Claude, and export a final JSON report.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The app reads the Anthropic key from `.env`. It supports both names:

```env
anthropic_api_key=...
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6
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

## Output

Default output:

```text
outputs/detections.json
outputs/crops/bottle_0001.jpg
outputs/crops/bottle_0002.jpg
```

Each JSON detection includes the bottle sequence, video timestamp, crop path, the compact defect list, and Arabic summary. The AI is asked only for `defects.type`, `defects.label_ar`, `defects.description_ar`, and `summary_ar` to reduce output tokens.

## Convert JSON To Excel

Convert the latest JSON result into an Excel report under `result/`:

```bash
.venv/bin/python json_to_excel.py
```

Default paths:

```text
input:  outputs/detections.json
output: result/detections.xlsx
```

The workbook opens on a full Arabic report sheet with the summary at the top and bottle-by-bottle details below it. It also includes separate summary, compact detection, and defect detail sheets.

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
