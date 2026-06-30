from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2

from .config import AppConfig
from .models import DetectionResult, Defect, normalize_detection_payload


SYSTEM_PROMPT = """You are an expert visual quality-inspection system for white plastic bottles on a conveyor.

Return compact JSON only. No markdown or extra text.

Inspect the visible bottle crop for these defect categories:
- body_defect: Arabic label "زرف او عيب في العلبة"; dents, scratches, deformation, holes, cracks, crushed areas, manufacturing defects on the bottle body.
- cap_defect: Arabic label "غطاء علبة بيه مشكلة"; missing cap, broken cap, wrong cap position, damaged cap, open cap, cap deformation.
- dirty: Arabic label "العلبة متسخة"; dirt, stains, grease, marks, dust, contamination, or discoloration.

Rules:
- Multiple defects are allowed.
- If no defect is visible, return an empty defects array.
- Keep Arabic descriptions short and direct.
"""

USER_PROMPT = """Analyze this single bottle crop and return exactly this JSON shape:
{
  "defects": [
    {
      "type": "body_defect | cap_defect | dirty",
      "label_ar": "زرف او عيب في العلبة | غطاء علبة بيه مشكلة | العلبة متسخة",
      "description_ar": "وصف مختصر للمشكلة"
    }
  ],
  "summary_ar": "ملخص قصير لحالة العلبة"
}

Return JSON only."""


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
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    }
                ],
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
            )


def encode_jpeg_base64(image_bgr: Any) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to encode crop as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


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
