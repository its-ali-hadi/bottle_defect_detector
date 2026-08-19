"""Regression tests for bottle_detector/tracking.py.

Every case here locks in a real bug found and fixed during development against
actual conveyor footage (see CLAUDE.md's Detection section for the full story):
  - rounded-base width under-measurement + touching-bottle merging
  - crop cutting off the bottle's neck/cap-thread area
  - neck-to-body connectivity flickering frame-to-frame (motion blur/compression),
    fragmenting tracking into duplicate captures of the same physical bottle
  - a fallen/sideways bottle reading as an oversized, malformed detection

These use synthetic masks instead of real video frames so they run fast and
deterministically in CI, without needing the (large, not-checked-in) video
files this project was actually debugged against.
"""

from __future__ import annotations

import numpy as np
import pytest

from bottle_detector.tracking import (
    BBox,
    BottleDetector,
    BottleTracker,
    CandidateDetection,
    Track,
    bbox_iou,
    find_blue_belt_top,
    merge_active_columns,
    project_bottle_bases,
    smooth_bbox,
)


def draw_rect(mask: np.ndarray, *, x1: int, x2: int, y1: int, y2: int) -> None:
    mask[y1:y2, x1:x2] = 255


def make_single_bottle_mask(
    *, width: int = 500, height: int = 1000, belt_top: int = 700
) -> np.ndarray:
    """One bottle: narrow neck on top of a full-width body/base, no gaps."""
    mask = np.zeros((height, width), dtype=np.uint8)
    draw_rect(mask, x1=210, x2=290, y1=150, y2=350)  # neck
    draw_rect(mask, x1=150, x2=350, y1=350, y2=760)  # body + base
    return mask


class TestMergeActiveColumns:
    def test_single_run(self) -> None:
        active = np.array([False] * 5 + [True] * 20 + [False] * 5)
        assert merge_active_columns(active, min_width=10, gap=2) == [(5, 24)]

    def test_short_run_dropped(self) -> None:
        active = np.array([False] * 5 + [True] * 3 + [False] * 5)
        assert merge_active_columns(active, min_width=10, gap=2) == []

    def test_two_runs_split_by_large_gap(self) -> None:
        active = np.array([True] * 20 + [False] * 20 + [True] * 20)
        assert merge_active_columns(active, min_width=10, gap=5) == [(0, 19), (40, 59)]

    def test_small_gap_bridged(self) -> None:
        active = np.array([True] * 20 + [False] * 3 + [True] * 20)
        assert merge_active_columns(active, min_width=10, gap=5) == [(0, 42)]


class TestFindBlueBeltTop:
    def test_finds_belt_row(self) -> None:
        height, width = 400, 600
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[250:, :, 0] = 100  # blue hue across the full width below row 250
        hsv[250:, :, 1] = 200
        hsv[250:, :, 2] = 200
        assert find_blue_belt_top(hsv, frame_width=width, frame_height=height) == 250

    def test_falls_back_when_no_belt_visible(self) -> None:
        height, width = 400, 600
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        assert find_blue_belt_top(hsv, frame_width=width, frame_height=height) == int(height * 0.52)


class TestProjectBottleBases:
    def test_single_bottle_full_height_recovered(self) -> None:
        """The crop must span the true neck-to-base extent, not just the body
        (the original bug this whole detection pipeline was rebuilt around)."""
        mask = make_single_bottle_mask()
        boxes = project_bottle_bases(mask, belt_top=700)
        assert len(boxes) == 1
        (x, y, w, h), _area = boxes[0]
        assert y == pytest.approx(150, abs=5), "top should reach the neck, not stop at the body"
        assert y + h == pytest.approx(759, abs=5), "bottom should reach the base"
        assert x == pytest.approx(150, abs=5)
        assert x + w == pytest.approx(349, abs=5), "width should match the true (wider) body, not the narrow neck"

    def test_touching_bottles_split_via_belt_gap(self) -> None:
        """Two bottles whose bodies come close but keep a small real gap (15px
        here, out of a 700px-wide frame -- comparable to a real, if faint,
        shadow/highlight line between two physically separate objects), plus
        a wider gap between their (narrower, rounded) bases at the belt line,
        must be split into two detections, each recovering its true (wider)
        body width -- not merged into one, and not clamped to the narrow base
        reading. NOTE: a *mathematically zero-pixel* gap between bodies (two
        bottles perfectly touching with no distinguishing pixel at all) is
        NOT handled correctly -- the morphological closing step (needed to
        bridge the neck/shoulder connectivity flicker, see the test below)
        also bridges tiny body gaps, so both boxes bleed a little into each
        other's territory when there's truly nothing to tell them apart. This
        matches the one real touching-bottle-merge case found in production
        data (IMG_9792, 3 bottles fused with zero visible gap) -- a genuine,
        narrow, accepted limitation, not something this test tries to fix."""
        width, height, belt_top = 700, 1000, 700
        mask = np.zeros((height, width), dtype=np.uint8)
        # Bottle A: body 100-284, base narrower (120-279)
        draw_rect(mask, x1=175, x2=225, y1=150, y2=350)
        draw_rect(mask, x1=100, x2=285, y1=350, y2=590)
        draw_rect(mask, x1=120, x2=280, y1=590, y2=760)
        # Bottle B: body 315-499 (30px real gap from A), base narrower (321-478)
        draw_rect(mask, x1=350, x2=450, y1=150, y2=350)
        draw_rect(mask, x1=315, x2=500, y1=350, y2=590)
        draw_rect(mask, x1=321, x2=479, y1=590, y2=760)

        boxes = sorted(project_bottle_bases(mask, belt_top=belt_top), key=lambda item: item[0][0])
        assert len(boxes) == 2, "a real gap at the belt line must split touching bottles"
        (ax, ay, aw, ah), _ = boxes[0]
        (bx, by, bw, bh), _ = boxes[1]
        assert ax == pytest.approx(100, abs=5)
        assert ax + aw == pytest.approx(284, abs=5), "bottle A width must match its true body, not its narrow base"
        assert bx == pytest.approx(315, abs=5)
        assert bx + bw == pytest.approx(499, abs=5), "bottle B width must match its true body, not its narrow base"

    def test_neck_body_gap_bridged_by_closing(self) -> None:
        """A single-row break between neck and body (simulating motion blur /
        compression noise on one frame) must still be treated as one bottle
        reaching the neck -- not silently clipped down to just the body.
        This is what previously fragmented tracking into duplicate captures
        of the same physical bottle."""
        width, height, belt_top = 500, 1000, 700
        mask = np.zeros((height, width), dtype=np.uint8)
        draw_rect(mask, x1=210, x2=290, y1=150, y2=299)  # neck, stops at 298
        # row 299 intentionally left blank -- the broken bridge
        draw_rect(mask, x1=150, x2=350, y1=300, y2=760)  # body + base

        boxes = project_bottle_bases(mask, belt_top=belt_top)
        assert len(boxes) == 1
        (x, y, w, h), _area = boxes[0]
        assert y == pytest.approx(150, abs=8), "neck must still be included despite the one-row gap"

    def test_no_bottle_returns_empty(self) -> None:
        mask = np.zeros((1000, 500), dtype=np.uint8)
        assert project_bottle_bases(mask, belt_top=700) == []


class TestLooksLikeBottle:
    def test_normal_standing_bottle_accepted(self) -> None:
        # width/height ratio ~0.65, well within the accepted band
        assert BottleDetector._looks_like_bottle(
            100, 150, 200, 610, frame_width=500, frame_height=1000
        )

    def test_fallen_sideways_bottle_rejected(self) -> None:
        # A bottle lying on its side reads much wider than tall (observed
        # 1.46-1.50 on real footage) -- must be rejected, not captured.
        assert not BottleDetector._looks_like_bottle(
            50, 400, 730, 500, frame_width=800, frame_height=1000
        )

    def test_too_narrow_sliver_rejected(self) -> None:
        assert not BottleDetector._looks_like_bottle(
            100, 150, 50, 610, frame_width=500, frame_height=1000
        )

    def test_too_small_rejected(self) -> None:
        assert not BottleDetector._looks_like_bottle(
            100, 150, 10, 10, frame_width=500, frame_height=1000
        )


class TestBottleDetectorEndToEnd:
    def test_detects_single_bottle_in_bgr_frame(self) -> None:
        width, height, belt_top = 500, 1000, 700
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[belt_top:, :] = (235, 160, 70)  # BGR-ish blue belt
        white_mask = make_single_bottle_mask(width=width, height=height, belt_top=belt_top)
        frame[white_mask > 0] = (255, 255, 255)

        detector = BottleDetector(min_area_ratio=0.01)
        detections = detector.detect(frame)
        assert len(detections) == 1
        x, y, w, h = detections[0].bbox
        assert y == pytest.approx(150, abs=8)
        assert x == pytest.approx(150, abs=8)


class TestBboxHelpers:
    def test_bbox_iou_identical(self) -> None:
        box: BBox = (10, 10, 50, 50)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_bbox_iou_disjoint(self) -> None:
        assert bbox_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0

    def test_smooth_bbox_blends_toward_new(self) -> None:
        old: BBox = (0, 0, 100, 100)
        new: BBox = (10, 10, 100, 100)
        smoothed = smooth_bbox(old, new, alpha=0.5)
        assert smoothed == (5, 5, 100, 100)


class TestBottleTrackerCapture:
    def test_bottle_captured_once_after_enough_frames_in_zone(self) -> None:
        tracker = BottleTracker(capture_start=0.1, capture_end=0.9, min_seen_before_capture=3)
        frame_shape = (1000, 500, 3)
        bbox: BBox = (150, 150, 200, 600)  # centered, well inside the capture zone

        captured_counts = []
        for frame_index in range(1, 6):
            detection = CandidateDetection(bbox=bbox, area=10000.0)
            _tracks, candidates = tracker.update(
                [detection], frame_index=frame_index, frame_shape=frame_shape
            )
            captured_counts.append(len(candidates))

        assert sum(captured_counts) == 1, "a bottle sitting still in the capture zone must be captured exactly once"

    def test_bottle_at_frame_edge_not_captured(self) -> None:
        tracker = BottleTracker(capture_start=0.1, capture_end=0.9, min_seen_before_capture=1)
        frame_shape = (1000, 500, 3)
        bbox: BBox = (0, 150, 200, 600)  # touches the left frame edge

        detection = CandidateDetection(bbox=bbox, area=10000.0)
        _tracks, candidates = tracker.update([detection], frame_index=1, frame_shape=frame_shape)
        assert candidates == []
