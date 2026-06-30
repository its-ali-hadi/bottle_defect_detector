from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


@dataclass
class CandidateDetection:
    bbox: BBox
    area: float

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    first_frame: int
    last_frame: int
    seen_count: int = 1
    missed_count: int = 0
    captured: bool = False
    sequence: int | None = None
    analysis_state: str = "tracking"

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def update(self, detection: CandidateDetection, frame_index: int) -> None:
        self.bbox = smooth_bbox(self.bbox, detection.bbox)
        self.last_frame = frame_index
        self.seen_count += 1
        self.missed_count = 0


class BottleDetector:
    def __init__(self, *, min_area_ratio: float = 0.012) -> None:
        self.min_area_ratio = min_area_ratio

    def detect(self, frame: Any) -> list[CandidateDetection]:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        belt_top = find_blue_belt_top(hsv, frame_width=width, frame_height=height)

        # White bottles have low saturation and high value; the blue conveyor has high saturation.
        lower_white = np.array([0, 0, 125], dtype=np.uint8)
        upper_white = np.array([180, 95, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        min_area = height * width * self.min_area_ratio
        detections: list[CandidateDetection] = []
        for bbox, pixel_area in project_bottle_bases(mask, belt_top=belt_top):
            x, y, w, h = bbox
            if pixel_area < min_area:
                continue
            if not self._looks_like_bottle(x, y, w, h, frame_width=width, frame_height=height):
                continue
            detections.append(CandidateDetection(bbox=bbox, area=float(pixel_area)))

        return sorted(detections, key=lambda item: item.bbox[0])

    @staticmethod
    def _looks_like_bottle(
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if w < frame_width * 0.055 or h < frame_height * 0.24:
            return False
        aspect = w / max(1, h)
        if aspect < 0.40 or aspect > 1.45:
            return False
        bottom = y + h
        if bottom < frame_height * 0.42:
            return False
        if y > frame_height * 0.72:
            return False
        return True


def find_blue_belt_top(hsv: Any, *, frame_width: int, frame_height: int) -> int:
    blue_mask = cv2.inRange(
        hsv,
        np.array([85, 50, 70], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    )
    row_counts = (blue_mask > 0).sum(axis=1)
    rows = np.where(row_counts > frame_width * 0.45)[0]
    if len(rows):
        return int(rows[0])
    return int(frame_height * 0.52)


def project_bottle_bases(mask: Any, *, belt_top: int) -> list[tuple[BBox, int]]:
    height, width = mask.shape[:2]
    band_y1 = max(0, belt_top - 35)
    band_y2 = min(height, belt_top + 22)
    if band_y2 <= band_y1:
        return []

    band = mask[band_y1:band_y2, :]
    counts = (band > 0).sum(axis=0)
    pixel_threshold = max(8, int((band_y2 - band_y1) * 0.62))
    intervals = merge_active_columns(
        counts > pixel_threshold,
        min_width=max(34, int(width * 0.04)),
        gap=max(10, int(width * 0.017)),
    )

    boxes: list[tuple[BBox, int]] = []
    search_y1 = max(0, belt_top - 210)
    search_y2 = min(height, belt_top + 28)
    x_pad = max(6, int(width * 0.01))
    for x1, x2 in intervals:
        sx1 = max(0, x1 - x_pad)
        sx2 = min(width, x2 + x_pad)
        submask = mask[search_y1:search_y2, sx1:sx2]
        ys, xs = np.where(submask > 0)
        if len(xs) == 0:
            continue
        bx1 = sx1 + int(xs.min())
        bx2 = sx1 + int(xs.max())
        by1 = search_y1 + int(ys.min())
        by2 = search_y1 + int(ys.max())
        boxes.append(((bx1, by1, bx2 - bx1 + 1, by2 - by1 + 1), int(len(xs))))
    return boxes


def merge_active_columns(active: Any, *, min_width: int, gap: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            last_active = index
        elif start is not None and last_active is not None and index - last_active > gap:
            if last_active - start + 1 >= min_width:
                intervals.append((start, last_active))
            start = None
            last_active = None
    if start is not None and last_active is not None and last_active - start + 1 >= min_width:
        intervals.append((start, last_active))
    return intervals


class BottleTracker:
    def __init__(
        self,
        *,
        capture_start: float = 0.08,
        capture_end: float = 0.92,
        min_seen_before_capture: int = 3,
        max_missed: int = 10,
    ) -> None:
        self.capture_start = capture_start
        self.capture_end = capture_end
        self.min_seen_before_capture = min_seen_before_capture
        self.max_missed = max_missed
        self._next_track_id = 1
        self._tracks: list[Track] = []

    def update(
        self,
        detections: list[CandidateDetection],
        *,
        frame_index: int,
        frame_shape: tuple[int, int, int],
    ) -> tuple[list[Track], list[Track]]:
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))

        pairs = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = match_score(track, detection)
                if score < 95:
                    pairs.append((score, track_index, detection_index))
        pairs.sort(key=lambda item: item[0])

        for _, track_index, detection_index in pairs:
            if track_index not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self._tracks[track_index].update(detections[detection_index], frame_index)
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed_count += 1

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self._tracks.append(
                Track(
                    track_id=self._next_track_id,
                    bbox=detection.bbox,
                    first_frame=frame_index,
                    last_frame=frame_index,
                )
            )
            self._next_track_id += 1

        self._tracks = [track for track in self._tracks if track.missed_count <= self.max_missed]
        capture_candidates = [
            track
            for track in self._tracks
            if self._should_capture(track, frame_shape=frame_shape)
        ]
        for track in capture_candidates:
            track.captured = True

        capture_candidates.sort(key=lambda item: item.bbox[0])
        return list(self._tracks), capture_candidates

    def mark_analysis_complete(self, sequence: int, status: str) -> None:
        for track in self._tracks:
            if track.sequence == sequence:
                track.analysis_state = status
                return

    def _should_capture(self, track: Track, *, frame_shape: tuple[int, int, int]) -> bool:
        if track.captured or track.seen_count < self.min_seen_before_capture:
            return False
        height, width = frame_shape[:2]
        x, y, w, h = track.bbox
        center_x = x + w / 2.0
        if center_x < width * self.capture_start or center_x > width * self.capture_end:
            return False
        edge_margin = max(4, int(width * 0.015))
        if x <= edge_margin or x + w >= width - edge_margin:
            return False
        if y <= 1 or y + h >= height - 1:
            return False
        return True


def match_score(track: Track, detection: CandidateDetection) -> float:
    tx, ty = track.centroid
    dx, dy = detection.centroid
    centroid_distance = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
    iou_bonus = bbox_iou(track.bbox, detection.bbox) * 35.0
    return centroid_distance - iou_bonus


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def smooth_bbox(old: BBox, new: BBox, alpha: float = 0.72) -> BBox:
    return tuple(
        int(round(alpha * old_value + (1.0 - alpha) * new_value))
        for old_value, new_value in zip(old, new)
    )  # type: ignore[return-value]


def crop_with_padding(frame: Any, bbox: BBox, *, padding: float = 0.18) -> Any:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return frame[y1:y2, x1:x2].copy()
