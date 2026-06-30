from __future__ import annotations

import sys
from pathlib import Path

import cv2


def discover_cameras(max_index: int = 6) -> list[int]:
    available: list[int] = []
    previous_log_level = silence_opencv_logs()
    for index in candidate_camera_indices(max_index):
        capture = open_camera(index)
        try:
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    available.append(index)
        finally:
            capture.release()
    restore_opencv_logs(previous_log_level)
    return available


def candidate_camera_indices(max_index: int) -> list[int]:
    if sys.platform.startswith("linux"):
        indices: list[int] = []
        for path in sorted(Path("/dev").glob("video*")):
            suffix = path.name.removeprefix("video")
            if suffix.isdigit():
                indices.append(int(suffix))
        return indices or list(range(max_index))
    return list(range(max_index))


def open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def silence_opencv_logs() -> int | None:
    if not hasattr(cv2, "getLogLevel") or not hasattr(cv2, "setLogLevel"):
        return None
    previous = cv2.getLogLevel()
    cv2.setLogLevel(0)
    return int(previous)


def restore_opencv_logs(level: int | None) -> None:
    if level is not None and hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(level)
