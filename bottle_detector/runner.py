from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .ai import AnalyzerProtocol, AnthropicBottleAnalyzer, NoAiAnalyzer
from .config import AppConfig
from .exporter import write_run_output
from .models import DetectionResult
from .tracking import BottleDetector, BottleTracker, Track, crop_with_padding


@dataclass
class PendingAnalysis:
    sequence: int
    future: Future[DetectionResult]


def run_detector(config: AppConfig) -> list[DetectionResult]:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.crops_dir.mkdir(parents=True, exist_ok=True)

    analyzer: AnalyzerProtocol
    analyzer = AnthropicBottleAnalyzer(config) if config.use_ai else NoAiAnalyzer()

    capture = cv2.VideoCapture(config.parsed_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {config.source}")

    if isinstance(config.parsed_source, int):
        if config.camera_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        if config.camera_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    detector = BottleDetector(min_area_ratio=config.min_area_ratio)
    tracker = BottleTracker(
        capture_start=config.capture_start,
        capture_end=config.capture_end,
    )

    detections: list[DetectionResult] = []
    pending: list[PendingAnalysis] = []
    sequence = 0
    display_enabled = should_display(config.display)
    window_name = "Bottle Defect Detector"
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    executor = ThreadPoolExecutor(max_workers=1)
    stop_requested = False

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            if config.frame_limit and frame_index > config.frame_limit:
                break

            raw_detections = detector.detect(frame)
            tracks, capture_candidates = tracker.update(
                raw_detections,
                frame_index=frame_index,
                frame_shape=frame.shape,
            )

            for track in capture_candidates:
                if config.max_bottles is not None and sequence >= config.max_bottles:
                    continue
                sequence += 1
                track.sequence = sequence
                track.analysis_state = "pending"
                crop = crop_with_padding(frame, track.bbox, padding=0.18)
                crop_path = config.crops_dir / f"bottle_{sequence:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                timestamp_sec = get_timestamp_sec(capture, frame_index, fps)
                pending.append(
                    PendingAnalysis(
                        sequence=sequence,
                        future=executor.submit(
                            analyzer.analyze,
                            sequence=sequence,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            crop_path=crop_path,
                            crop_bgr=crop.copy(),
                        ),
                    )
                )

            collect_finished(pending, detections, tracker)

            if display_enabled:
                preview = draw_preview(
                    frame=frame,
                    tracks=tracks,
                    pending=pending,
                    captured_count=sequence,
                    model=analyzer.model,
                    capture_start=config.capture_start,
                    capture_end=config.capture_end,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                    break

            if config.max_bottles is not None and sequence >= config.max_bottles:
                break

        if stop_requested:
            print("Stop requested; waiting for pending AI analysis before writing JSON.")

        wait_for_pending(pending, detections, tracker)
        output = write_run_output(
            source=config.source,
            model=analyzer.model,
            output_path=config.output_path,
            detections=detections,
        )
        print(
            f"Wrote {len(output.detections)} detections to {config.output_path} "
            f"and crops to {config.crops_dir}"
        )
        return output.detections
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        capture.release()
        if display_enabled:
            cv2.destroyAllWindows()


def should_display(requested: bool) -> bool:
    if not requested:
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("No display server detected; running without OpenCV preview window.")
        return False
    return True


def get_timestamp_sec(capture: Any, frame_index: int, fps: float) -> float:
    pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec and pos_msec > 0:
        return round(pos_msec / 1000.0, 3)
    if fps > 0:
        return round(frame_index / fps, 3)
    return 0.0


def collect_finished(
    pending: list[PendingAnalysis],
    detections: list[DetectionResult],
    tracker: BottleTracker,
) -> None:
    still_pending: list[PendingAnalysis] = []
    for item in pending:
        if item.future.done():
            result = item.future.result()
            detections.append(result)
            tracker.mark_analysis_complete(item.sequence, preview_state_for(result))
        else:
            still_pending.append(item)
    pending[:] = still_pending


def wait_for_pending(
    pending: list[PendingAnalysis],
    detections: list[DetectionResult],
    tracker: BottleTracker,
) -> None:
    for item in pending:
        result = item.future.result()
        detections.append(result)
        tracker.mark_analysis_complete(item.sequence, preview_state_for(result))
    pending.clear()


def preview_state_for(result: DetectionResult) -> str:
    return "defective" if result.defects else "ok"


def draw_preview(
    *,
    frame: Any,
    tracks: list[Track],
    pending: list[PendingAnalysis],
    captured_count: int,
    model: str,
    capture_start: float,
    capture_end: float,
) -> Any:
    preview = frame.copy()
    height, width = preview.shape[:2]
    x1 = int(width * capture_start)
    x2 = int(width * capture_end)
    cv2.line(preview, (x1, 0), (x1, height), (0, 180, 255), 2)
    cv2.line(preview, (x2, 0), (x2, height), (0, 180, 255), 2)

    pending_sequences = {item.sequence for item in pending}
    for track in tracks:
        x, y, w, h = track.bbox
        color = (120, 120, 120)
        label = f"T{track.track_id}"
        if track.sequence:
            if track.sequence in pending_sequences:
                color = (0, 180, 255)
                label = f"#{track.sequence} pending"
            else:
                color = status_color(track.analysis_state)
                label = f"#{track.sequence} {track.analysis_state}"
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            preview,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(preview, (8, 8), (min(width - 8, 560), 72), (0, 0, 0), -1)
    cv2.putText(
        preview,
        f"Captured: {captured_count} | Model: {model} | q: stop",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Orange lines = capture zone",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return preview


def status_color(status: str) -> tuple[int, int, int]:
    if status == "ok":
        return (60, 220, 60)
    if status == "defective":
        return (60, 60, 255)
    if status == "uncertain":
        return (0, 180, 255)
    return (180, 180, 180)
