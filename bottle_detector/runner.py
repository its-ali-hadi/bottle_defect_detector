from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import cv2

from json_to_excel import export_excel_report

from .ai import AnalyzerProtocol, AnthropicBottleAnalyzer, NoAiAnalyzer, YoloThenClaudeAnalyzer
from .cameras import open_camera
from .config import AppConfig
from .exporter import write_run_output
from .log_config import get_logger
from .models import DetectionResult
from .tracking import BottleDetector, BottleTracker, Track, crop_with_padding

logger = get_logger(__name__)

# A live camera can drop a frame read transiently (USB hiccup, driver glitch,
# momentary bandwidth stall) without the feed actually being gone -- unlike a
# video file, where a failed read reliably means end-of-file. Retrying a bounded
# number of times before giving up avoids ending an entire production run over
# one bad frame. At a typical 15-30fps this window is a few seconds.
CAMERA_READ_MAX_CONSECUTIVE_FAILURES = 30
CAMERA_READ_RETRY_DELAY_SEC = 0.1


@dataclass
class PendingAnalysis:
    sequence: int
    future: Future[DetectionResult]


ProgressCallback = Callable[[dict[str, Any]], None]
PreviewCallback = Callable[[Any], None]


def run_detector(
    config: AppConfig,
    *,
    stop_event: Event | None = None,
    preview_callback: PreviewCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[DetectionResult]:
    output_path = config.resolved_output_path
    crops_dir = config.resolved_crops_dir
    result_dir = config.resolved_result_dir

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        progress_callback,
        event="started",
        source=config.source,
        output_path=str(output_path),
        result_dir=str(result_dir),
    )

    analyzer: AnalyzerProtocol
    if not config.use_ai:
        analyzer = NoAiAnalyzer()
    elif config.use_yolo_prefilter:
        analyzer = YoloThenClaudeAnalyzer(config)
    else:
        analyzer = AnthropicBottleAnalyzer(config)

    parsed_source = config.parsed_source
    capture = open_camera(parsed_source) if isinstance(parsed_source, int) else cv2.VideoCapture(parsed_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {config.source}")

    if isinstance(parsed_source, int):
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
    is_live_camera = isinstance(parsed_source, int)

    try:
        frame_index = 0
        consecutive_read_failures = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                stop_requested = True
                break

            ok, frame = capture.read()
            if not ok:
                if is_live_camera and consecutive_read_failures < CAMERA_READ_MAX_CONSECUTIVE_FAILURES:
                    consecutive_read_failures += 1
                    if consecutive_read_failures == 1:
                        logger.warning("Camera frame read failed; retrying (source=%s).", config.source)
                    time.sleep(CAMERA_READ_RETRY_DELAY_SEC)
                    continue
                if is_live_camera:
                    logger.error(
                        "Camera frame read failed %d times in a row; stopping run (source=%s).",
                        consecutive_read_failures,
                        config.source,
                    )
                break
            consecutive_read_failures = 0
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
                crop_path = crops_dir / f"bottle_{sequence:04d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                timestamp_sec = get_timestamp_sec(capture, frame_index, fps)
                emit_progress(
                    progress_callback,
                    event="captured",
                    sequence=sequence,
                    frame_index=frame_index,
                    crop_path=str(crop_path),
                )
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
            emit_progress(
                progress_callback,
                event="progress",
                captured_count=sequence,
                completed_count=len(detections),
                pending_count=len(pending),
            )

            if display_enabled or preview_callback is not None:
                preview = draw_preview(
                    frame=frame,
                    tracks=tracks,
                    pending=pending,
                    captured_count=sequence,
                    model=analyzer.model,
                    capture_start=config.capture_start,
                    capture_end=config.capture_end,
                )
                emit_preview(preview_callback, preview)
            if display_enabled:
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_requested = True
                    break

            if config.max_bottles is not None and sequence >= config.max_bottles:
                break

        if stop_requested:
            logger.info("Stop requested; waiting for pending AI analysis before writing JSON.")

        wait_for_pending(pending, detections, tracker)
        output = write_run_output(
            source=config.source,
            model=analyzer.model,
            output_path=output_path,
            detections=detections,
        )
        excel_result = export_excel_report(input_path=output_path, output_dir=result_dir)
        emit_progress(
            progress_callback,
            event="finished",
            detections_count=len(output.detections),
            output_path=str(output_path),
            report_path=str(excel_result["report_path"]),
            latest_path=str(excel_result["latest_path"]) if excel_result["latest_path"] else None,
        )
        logger.info(
            "Wrote %d detections to %s, crops to %s, and Excel to %s",
            len(output.detections),
            output_path,
            crops_dir,
            excel_result["report_path"],
        )
        return output.detections
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        capture.release()
        if display_enabled:
            cv2.destroyAllWindows()


def emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        pass


def emit_preview(callback: PreviewCallback | None, frame: Any) -> None:
    if callback is None:
        return
    try:
        callback(frame)
    except Exception:
        pass


def should_display(requested: bool) -> bool:
    if not requested:
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        logger.info("No display server detected; running without OpenCV preview window.")
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
