from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import DetectionResult, RunOutput


def write_run_output(
    *,
    source: str,
    model: str,
    output_path: Path,
    detections: list[DetectionResult],
) -> RunOutput:
    output = RunOutput(
        source=source,
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        detections=sorted(detections, key=lambda item: item.sequence),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output

