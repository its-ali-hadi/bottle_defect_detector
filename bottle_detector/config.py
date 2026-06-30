from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .paths import resolve_app_path


DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class AppConfig:
    source: str = "0"
    output_path: Path = Path("outputs/detections.json")
    crops_dir: Path = Path("outputs/crops")
    result_dir: Path = Path("result")
    model: str | None = None
    use_ai: bool = True
    display: bool = True
    max_bottles: int | None = None
    frame_limit: int | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    min_area_ratio: float = 0.012
    capture_start: float = 0.08
    capture_end: float = 0.92

    def __post_init__(self) -> None:
        load_dotenv(resolve_app_path(".env"))
        load_dotenv()

    @property
    def model_name(self) -> str:
        return self.model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropic_api_key")

    @property
    def parsed_source(self) -> int | str:
        value = str(self.source)
        if value.isdigit():
            return int(value)
        return value

    @property
    def resolved_output_path(self) -> Path:
        return resolve_app_path(self.output_path)

    @property
    def resolved_crops_dir(self) -> Path:
        return resolve_app_path(self.crops_dir)

    @property
    def resolved_result_dir(self) -> Path:
        return resolve_app_path(self.result_dir)
