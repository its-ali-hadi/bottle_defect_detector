from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


DefectType = Literal["body_defect", "cap_defect", "dirty"]


LABELS_AR: dict[str, str] = {
    "body_defect": "زرف او عيب في العلبة",
    "cap_defect": "غطاء علبة بيه مشكلة",
    "dirty": "العلبة متسخة",
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

    @field_validator("summary_ar")
    @classmethod
    def summary_must_not_be_empty(cls, value: str) -> str:
        return value.strip() or "لا يوجد ملخص."


class AiPayload(BaseModel):
    defects: list[Defect] = Field(default_factory=list)
    summary_ar: str


class RunOutput(BaseModel):
    source: str
    model: str
    created_at: str
    detections: list[DetectionResult]


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
                "label_ar": defect.get("label_ar") or LABELS_AR[defect_type],
                "description_ar": defect.get("description_ar") or LABELS_AR[defect_type],
            }
        )

    return AiPayload(
        defects=defects,
        summary_ar=payload.get("summary_ar") or default_summary(defects),
    )


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
            "cap": "cap_defect",
            "lid": "cap_defect",
            "dirt": "dirty",
            "dirty_bottle": "dirty",
            "stain": "dirty",
        }
        return aliases.get(lowered)  # type: ignore[return-value]
    return None


def default_summary(defects: list[dict[str, Any]]) -> str:
    if defects:
        return "تم رصد مشكلة في العلبة."
    return "العلبة تبدو سليمة."
