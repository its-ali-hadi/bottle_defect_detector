from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator


DefectType = Literal["body_defect", "dirty", "factory_defect"]


LABELS_AR: dict[str, str] = {
    "body_defect": "زرف او عيب في العلبة",
    "dirty": "العلبة متسخة بالطين او التراب",
    "factory_defect": "عيب تصنيعي واضح وكبير في شكل العلبة",
}


class Defect(BaseModel):
    type: DefectType
    label_ar: str
    description_ar: str

    @field_validator("label_ar")
    @classmethod
    def label_must_not_be_empty(cls, value: str) -> str:
        return value.strip() or "مشكلة غير محددة"

    @field_validator("description_ar")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        return value.strip() or "لا يوجد وصف إضافي."


class DetectionResult(BaseModel):
    sequence: int
    frame_index: int
    timestamp_sec: float
    crop_path: str
    defects: list[Defect] = Field(default_factory=list)
    summary_ar: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Which stage produced this verdict: "yolo" (fast pre-filter, skipped Claude) or "claude" (full diagnosis).
    analysis_stage: str = Field(default="claude")

    @field_validator("summary_ar")
    @classmethod
    def summary_must_not_be_empty(cls, value: str) -> str:
        return value.strip() or "لا يوجد ملخص."

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy_pct(self) -> float:
        """AI's own confidence in this bottle's verdict, as a proxy accuracy score."""
        return round(self.confidence * 100, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision_pct(self) -> float | None:
        """Confidence in the flagged defects; only meaningful when defects were found."""
        if not self.defects:
            return None
        return round(self.confidence * 100, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall_pct(self) -> float | None:
        """Confidence that no defect was missed; only meaningful for clean verdicts."""
        if self.defects:
            return None
        return round(self.confidence * 100, 1)


class AiPayload(BaseModel):
    defects: list[Defect] = Field(default_factory=list)
    summary_ar: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class RunStatistics(BaseModel):
    total_bottles: int
    defective_count: int
    ok_count: int
    accuracy_pct: float | None
    precision_pct: float | None
    recall_pct: float | None


class RunOutput(BaseModel):
    source: str
    model: str
    created_at: str
    statistics: RunStatistics
    detections: list[DetectionResult]


def compute_run_statistics(detections: list[DetectionResult]) -> RunStatistics:
    defective = [item for item in detections if item.defects]
    ok = [item for item in detections if not item.defects]
    return RunStatistics(
        total_bottles=len(detections),
        defective_count=len(defective),
        ok_count=len(ok),
        accuracy_pct=average_confidence_pct(detections),
        precision_pct=average_confidence_pct(defective),
        recall_pct=average_confidence_pct(ok),
    )


def average_confidence_pct(items: list[DetectionResult]) -> float | None:
    if not items:
        return None
    return round(sum(item.confidence for item in items) / len(items) * 100, 1)


def normalize_detection_payload(payload: dict[str, Any]) -> AiPayload:
    defects_raw = payload.get("defects") or []
    defects: list[dict[str, Any]] = []
    for defect in defects_raw:
        if not isinstance(defect, dict):
            continue
        defect_type = normalize_defect_type(defect.get("type"))
        if not defect_type:
            continue
        defects.append(
            {
                "type": defect_type,
                "label_ar": LABELS_AR[defect_type],
                "description_ar": defect.get("description_ar") or LABELS_AR[defect_type],
            }
        )

    return AiPayload(
        defects=defects,
        summary_ar=payload.get("summary_ar") or default_summary(defects),
        confidence=normalize_confidence(payload.get("confidence")),
    )


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.8
    return min(1.0, max(0.0, confidence))


def normalize_defect_type(value: Any) -> DefectType | None:
    if value in LABELS_AR:
        return value
    if isinstance(value, str):
        lowered = value.lower().strip()
        aliases = {
            "body": "body_defect",
            "bottle": "body_defect",
            "scratch": "body_defect",
            "dent": "body_defect",
            "mud": "dirty",
            "dirt": "dirty",
            "dirty_bottle": "dirty",
            "stain": "dirty",
            "factory": "factory_defect",
            "manufacturing": "factory_defect",
            "manufacturing_defect": "factory_defect",
            "molding_defect": "factory_defect",
            "molding": "factory_defect",
        }
        return aliases.get(lowered)  # type: ignore[return-value]
    return None


def default_summary(defects: list[dict[str, Any]]) -> str:
    if defects:
        return "تم رصد مشكلة في العلبة."
    return "العلبة تبدو سليمة."
