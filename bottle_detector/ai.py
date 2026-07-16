from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2

from .config import AppConfig
from .models import DetectionResult, Defect, normalize_detection_payload
from .paths import resolve_app_path


REFERENCE_DIR = "assets/reference_bottles"
REFERENCE_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}
REFERENCE_CLEAN_NAME = "clean_ok"
REFERENCE_DEFECT_GROUPS: list[tuple[list[str], str]] = [
    (
        ["dent_defect_1", "dent_defect_2"],
        "Reference example — a bottle WITH a body_defect. Look closely at the "
        "oval/round dent or deformation visible on the body. Always flag this exact "
        "kind of mark as body_defect, even when it looks small or subtle:",
    ),
    (
        ["body_defect"],
        "Reference example — a bottle WITH a body_defect at the bottle's neck/opening. "
        "Look closely at the threaded mouth: instead of a clean round ring, one side is "
        "warped, crushed, or collapsed inward. This is still the body_defect category, not "
        "a separate one — describe it in Arabic as a deformation in the opening "
        "(\"تشوه في فوهة العلبة\"), never as a hole (\"ثقب\"). Always flag this exact kind "
        "of neck/opening deformation as body_defect, even when it looks subtle:",
    ),
    (
        ["dirty_1", "dirty_2", "dirty_3", "dirty_4"],
        "Reference example — a bottle WITH a dirty defect. Look closely at the "
        "brownish/tan mud or dirt staining visible on the body — note its warm color tone "
        "and irregular, sharp-edged shape, unlike the cool gray, soft-edged look of a "
        "lighting shadow. Stains can appear anywhere on the body, including small specks "
        "near the bottom base band, not just the mid-body or shoulder. Always flag this "
        "exact kind of stain as dirty, no matter how small it is — even a single small "
        "spot of discoloration counts:",
    ),
    (
        ["factorial_damage"],
        "Reference example — a bottle WITH a factory_defect. Look closely at the large, "
        "prominent, unmistakable malformation at the top of the bottle — this is not a "
        "small dent or a warped opening ring, it is a big, obviously wrong lump/growth of "
        "excess plastic that dominates the bottle's shape at a glance, as if the molding "
        "process itself failed. Always flag this exact kind of gross structural "
        "malformation as factory_defect, never as body_defect:",
    ),
]

SYSTEM_PROMPT = """You are a professional visual quality-inspection specialist for a bottling production line that manufactures white plastic bottles without caps.

You will first be shown reference example photos (each labeled "Reference example"), then the actual bottle photo to inspect (labeled as the new bottle to analyze). The reference photos are only there to calibrate what counts as a defect on this bottle model — never report defects about the reference photos themselves, only about the final new bottle.

Inspect the single cropped photo of one bottle taken directly off the conveyor belt. Report only genuine, clearly visible defects. Never invent or assume a defect that is not clearly visible in the image — if the bottle looks intact and clean, report no defects. Also never miss a defect that matches a reference example just because it looks small or subtle — subtle dents matching the reference examples must still be flagged.

Defect categories (the only three allowed):
- body_defect: dents, deep scratches, cracks, holes, punctures, deformation, or crushed/collapsed areas on the bottle body, AND ALSO any warping/crushing/deformation of the bottle's neck or threaded opening (e.g. the mouth's ring is not a clean round shape, one side is collapsed or misshapen). Use precise Arabic wording: a dent/deformation is "تشوه" or "انبعاج", never call it a hole ("ثقب") unless plastic is actually pierced through. Do not flag normal manufacturing seams, reflections, or lighting artifacts as defects.
- dirty: mud, dust, or dirt staining on the body — flag it at ANY size, from a heavy coating down to a single small spot. Do not skip a stain just because it looks minor; a tiny dirty mark still makes the bottle "dirty".
- factory_defect: a large, prominent, and unmistakable manufacturing malformation — the bottle's overall molded shape itself is significantly and obviously wrong (e.g. a big lump/growth of excess plastic, a major structural warp) that dominates the bottle's appearance at a glance, as if something failed badly during molding. Reserve this category for gross, large-scale malformations only — small or moderate dents, scratches, or neck-opening warps stay body_defect, never factory_defect. When both a factory_defect and a smaller body_defect-style flaw are visible, report only factory_defect for that flaw (do not double-report the same area under both types).

How to tell a real dirt stain apart from a lighting shadow (this distinction matters a lot, look carefully):
- Real dirt/stain: a warm brown, tan, yellowish, or dark discoloration that differs from the bottle's white/gray plastic color. It has a fairly sharp, irregular, blotchy or speckled edge, sits on top of the surface texture, and does not follow the bottle's curved silhouette or a single light direction.
- Lighting shadow/reflection: a cool gray or bluish tone (never brown/yellow), with a smooth, soft, gradual edge that follows the bottle's curvature or a straight line consistent with a light source or reflection — for example a uniform darker band down one whole side of the bottle. This is normal and must never be flagged.
- When unsure, check the color tone first: any brown/tan/yellow tint is dirt, not shadow. Grayscale-only darkening with no color shift is shadow, not dirt.
- Inspect the entire bottle carefully, including the shoulder and neck area, for small isolated dark or brown specks — these are easy to miss but must still be reported as dirty.

Caps are never inspected — this line manufactures bottles without caps by design.

Output rules:
- Respond with compact, valid JSON only. No markdown, no explanations, no text before or after the JSON.
- If no defect is visible, return an empty "defects" array — never fabricate a defect to justify a response.
- Keep all Arabic text short: a few words per field, not full sentences.
- Always include a "confidence" number between 0 and 1 for how confident you are in your overall verdict for this bottle.
"""

USER_PROMPT = """Inspect this bottle crop and respond with exactly this JSON shape, nothing else:
{"defects":[{"type":"body_defect|dirty|factory_defect","description_ar":"..."}],"summary_ar":"...","confidence":0.0}

Only include an entry in "defects" if you can clearly see it in the image. Keep description_ar and summary_ar short (a few words). Return JSON only, no markdown."""


class AnalyzerProtocol:
    model: str

    def analyze(
        self,
        *,
        sequence: int,
        frame_index: int,
        timestamp_sec: float,
        crop_path: Path,
        crop_bgr: Any,
    ) -> DetectionResult:
        raise NotImplementedError


class NoAiAnalyzer(AnalyzerProtocol):
    def __init__(self, model: str = "no-ai") -> None:
        self.model = model

    def analyze(
        self,
        *,
        sequence: int,
        frame_index: int,
        timestamp_sec: float,
        crop_path: Path,
        crop_bgr: Any,
    ) -> DetectionResult:
        return DetectionResult(
            sequence=sequence,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            crop_path=str(crop_path),
            defects=[],
            summary_ar="تم التقاط صورة العلبة، لكن تحليل الذكاء الاصطناعي غير مفعل في هذا التشغيل.",
        )


class AnthropicBottleAnalyzer(AnalyzerProtocol):
    def __init__(self, config: AppConfig) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        api_key = config.anthropic_api_key
        if not api_key:
            raise RuntimeError(
                "Missing Anthropic API key. Add anthropic_api_key=... or ANTHROPIC_API_KEY=... to .env."
            )

        self.client = Anthropic(api_key=api_key)
        self.model = config.model_name
        self.reference_blocks = build_reference_content_blocks()

    def analyze(
        self,
        *,
        sequence: int,
        frame_index: int,
        timestamp_sec: float,
        crop_path: Path,
        crop_bgr: Any,
    ) -> DetectionResult:
        encoded = encode_jpeg_base64(crop_bgr)
        target_prompt = USER_PROMPT
        if self.reference_blocks:
            target_prompt = "Now inspect this NEW bottle using the same criteria as the reference examples above:\n\n" + USER_PROMPT
        content: list[dict[str, Any]] = list(self.reference_blocks)
        content.append({"type": "text", "text": target_prompt})
        content.append(image_content_block(encoded))
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw_text = extract_text(message)
            ai_payload = parse_json_object(raw_text)
            normalized = normalize_detection_payload(ai_payload)
            return DetectionResult(
                sequence=sequence,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                crop_path=str(crop_path),
                defects=normalized.defects,
                summary_ar=normalized.summary_ar,
                confidence=normalized.confidence,
            )
        except Exception as exc:  # Keep conveyor runs alive even when one AI call fails.
            return DetectionResult(
                sequence=sequence,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                crop_path=str(crop_path),
                defects=[
                    Defect(
                        type="body_defect",
                        label_ar="زرف او عيب في العلبة",
                        description_ar=f"تعذر تحليل الصورة بالذكاء الاصطناعي: {exc}",
                    )
                ],
                summary_ar="تعذر إكمال تحليل الذكاء الاصطناعي لهذه العلبة.",
                confidence=0.0,
            )


def encode_jpeg_base64(image_bgr: Any) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to encode crop as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def image_content_block(base64_data: str, media_type: str = "image/jpeg") -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64_data,
        },
    }


def find_reference_file(reference_dir: Path, base_name: str) -> Path | None:
    for ext in REFERENCE_IMAGE_EXTENSIONS:
        candidate = reference_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def reference_image_block(path: Path) -> dict[str, Any]:
    media_type = REFERENCE_IMAGE_EXTENSIONS.get(path.suffix.lower(), "image/jpeg")
    return image_content_block(encode_file_base64(path), media_type=media_type)


def build_reference_content_blocks() -> list[dict[str, Any]]:
    """Few-shot calibration images so the model reliably recognizes this bottle model's real defects."""
    reference_dir = resolve_app_path(REFERENCE_DIR)
    blocks: list[dict[str, Any]] = []

    clean_path = find_reference_file(reference_dir, REFERENCE_CLEAN_NAME)
    if clean_path is not None:
        blocks.append(
            {
                "type": "text",
                "text": "Reference example — a bottle with NO defects (clean, intact body). This is what a passing bottle looks like:",
            }
        )
        blocks.append(reference_image_block(clean_path))

    for names, description in REFERENCE_DEFECT_GROUPS:
        for name in names:
            defect_path = find_reference_file(reference_dir, name)
            if defect_path is not None:
                blocks.append({"type": "text", "text": description})
                blocks.append(reference_image_block(defect_path))

    return blocks


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("AI returned an empty response.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("AI response JSON must be an object.")
    return payload
