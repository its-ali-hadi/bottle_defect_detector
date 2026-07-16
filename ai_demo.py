"""
نسخة توضيحية مبسطة لآلية عمل تحليل عيوب العلب بالذكاء الاصطناعي.

هذا الملف لغرض العرض والتوضيح فقط، وليس النسخة الكاملة المستخدمة فعلياً
في تشغيل النظام (الموجودة في bottle_detector/ai.py). الهدف منه إعطاء فكرة
عامة عن آلية العمل: كل صورة علبة تُرفق مع بعض الأمثلة المرجعية (صور سليمة/
معيبة) وتُرسل لنموذج Claude ليحلّلها ويرجّع تصنيفاً بصيغة JSON، بدلاً من
تدريب نموذج رؤية حاسوبية مخصص من الصفر.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2


# مثال مبسط جداً لأمر النظام الموجه للنموذج.
# النسخة الفعلية المستخدمة في الإنتاج أكثر تفصيلاً بكثير: تحدد معايير دقيقة
# لكل نوع عيب، وتفرّق بين حالات متشابهة بصرياً (مثل بقعة اتساخ مقابل ظل
# ضوئي)، وتضبط المصطلحات المستخدمة بدقة — وهذا الجزء لا يُشارك هنا.
SYSTEM_PROMPT = """You are a visual quality-inspection assistant for bottles
moving on a conveyor belt. Look at the bottle image and report any visible
defect you can see. Return compact JSON only."""

USER_PROMPT = """Inspect this bottle and respond in exactly this JSON shape:
{"defects": [{"type": "...", "description": "..."}], "summary": "...", "confidence": 0.0}"""


# مجلد يحوي صوراً مرجعية بسيطة تُرفق مع كل طلب تحليل، لتعليم النموذج شكل
# الحالة السليمة وشكل العيب المطلوب رصده دون أي حاجة لتدريب النموذج
# (Few-shot Learning). في النسخة الفعلية توجد عدة مجموعات مرجعية لكل نوع
# عيب على حدة، أما هنا فمثال واحد مبسط فقط لتوضيح الفكرة العامة.
REFERENCE_DIR = "reference_examples"
REFERENCE_EXAMPLES: dict[str, str] = {
    "clean_example.jpg": "Example of a bottle with no defects.",
    "defect_example.jpg": "Example of a bottle with a visible defect.",
}


class BottleAnalyzerDemo:
    """توضيح مبسط لفكرة إرسال صورة علبة + أمثلة مرجعية لنموذج Claude لتحليلها."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.reference_blocks = self._build_reference_blocks()

    def _build_reference_blocks(self) -> list[dict[str, Any]]:
        """يبني قائمة الصور المرجعية المرفقة مع كل طلب، لتوضيح شكل الحالة السليمة والمعيبة للنموذج."""
        blocks: list[dict[str, Any]] = []
        reference_dir = Path(REFERENCE_DIR)
        for filename, description in REFERENCE_EXAMPLES.items():
            image_path = reference_dir / filename
            if image_path.exists():
                blocks.append({"type": "text", "text": description})
                blocks.append(self._image_block(image_path.read_bytes()))
        return blocks

    def analyze(self, crop_bgr: Any) -> dict[str, Any]:
        """يرسل صورة علبة واحدة للنموذج مع الأمثلة المرجعية، ويرجع نتيجة التحليل."""
        ok, buffer = cv2.imencode(".jpg", crop_bgr)
        if not ok:
            raise ValueError("Failed to encode image")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")

        content: list[dict[str, Any]] = list(self.reference_blocks)
        content.append({"type": "text", "text": USER_PROMPT})
        content.append(self._image_block_from_base64(encoded))

        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        raw_text = "".join(block.text for block in message.content if hasattr(block, "text"))
        return json.loads(raw_text)

    @staticmethod
    def _image_block(image_bytes: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return BottleAnalyzerDemo._image_block_from_base64(encoded)

    @staticmethod
    def _image_block_from_base64(data: str) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        }


if __name__ == "__main__":
    import os

    # مثال تشغيل بسيط: تحليل صورة علبة واحدة من القرص
    analyzer = BottleAnalyzerDemo(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    image = cv2.imread("example_bottle.jpg")
    result = analyzer.analyze(image)
    print(result)
