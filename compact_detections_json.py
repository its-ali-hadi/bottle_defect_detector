from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("outputs/detections.json")
DEFAULT_OUTPUT = Path("outputs/detections.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove unused AI fields from bottle detector JSON output.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSON path. Default: outputs/detections.json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path. Default overwrites outputs/detections.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    compacted = compact_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(compacted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote compact JSON: {output_path}")
    return 0


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": payload.get("source", ""),
        "model": payload.get("model", ""),
        "created_at": payload.get("created_at", ""),
        "detections": [
            compact_detection(item)
            for item in payload.get("detections", [])
            if isinstance(item, dict)
        ],
    }


def compact_detection(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": item.get("sequence", ""),
        "frame_index": item.get("frame_index", ""),
        "timestamp_sec": item.get("timestamp_sec", ""),
        "crop_path": item.get("crop_path", ""),
        "defects": [
            compact_defect(defect)
            for defect in item.get("defects", [])
            if isinstance(defect, dict)
        ],
        "summary_ar": item.get("summary_ar", ""),
    }


def compact_defect(defect: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": defect.get("type", ""),
        "label_ar": defect.get("label_ar", ""),
        "description_ar": defect.get("description_ar", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())

