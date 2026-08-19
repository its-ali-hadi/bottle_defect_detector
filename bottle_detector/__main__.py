from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .runner import run_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bottle_detector",
        description="Detect conveyor bottles, analyze each crop with Claude, and write JSON.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index such as 0, or a video path such as assets/test.mp4.",
    )
    parser.add_argument(
        "--output",
        default="outputs/detections.json",
        help="Final JSON output path.",
    )
    parser.add_argument(
        "--crops-dir",
        default="outputs/crops",
        help="Directory where analyzed bottle crops are saved.",
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Directory where Excel reports are saved.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Anthropic model override. Defaults to ANTHROPIC_MODEL or claude-sonnet-5.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Do not call Anthropic; write placeholder JSON results for smoke testing.",
    )
    parser.add_argument(
        "--max-bottles",
        type=int,
        default=None,
        help="Stop after capturing this many bottles.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Stop after reading this many frames.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="Optional live camera capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Optional live camera capture height.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.012,
        help="Minimum white component area as a fraction of frame area.",
    )
    parser.add_argument(
        "--capture-start",
        type=float,
        default=0.08,
        help="Left edge of capture zone as a fraction of frame width.",
    )
    parser.add_argument(
        "--capture-end",
        type=float,
        default=0.92,
        help="Right edge of capture zone as a fraction of frame width.",
    )
    parser.add_argument(
        "--use-yolo-prefilter",
        action="store_true",
        help=(
            "Screen bottles with a trained YOLOv8 binary classifier before Claude; "
            "confidently-clean bottles skip Claude entirely. Requires a trained model "
            "(see the Phase 2 training script) — fails fast if none is found."
        ),
    )
    parser.add_argument(
        "--yolo-model-path",
        default="models/yolo_bottle_classifier.pt",
        help="Path to the trained YOLO classification model (.pt). Only used with --use-yolo-prefilter.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.90,
        help="Minimum confidence for YOLO to mark a bottle clean without forwarding to Claude.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AppConfig(
        source=args.source,
        output_path=Path(args.output),
        crops_dir=Path(args.crops_dir),
        result_dir=Path(args.result_dir),
        model=args.model,
        use_ai=not args.no_ai,
        display=not args.no_display,
        max_bottles=args.max_bottles,
        frame_limit=args.frame_limit,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        min_area_ratio=args.min_area_ratio,
        capture_start=args.capture_start,
        capture_end=args.capture_end,
        use_yolo_prefilter=args.use_yolo_prefilter,
        yolo_model_path=Path(args.yolo_model_path),
        yolo_clean_confidence_threshold=args.yolo_confidence,
    )
    run_detector(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
