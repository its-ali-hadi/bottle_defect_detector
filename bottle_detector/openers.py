from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_file(path: str | Path) -> None:
    target = Path(path).resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
        return
    subprocess.Popen(["xdg-open", str(target)])


def open_folder(path: str | Path) -> None:
    target = Path(path).resolve()
    if target.is_file():
        target = target.parent
    open_file(target)


def reveal_in_file_manager(path: str | Path) -> None:
    target = Path(path).resolve()
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", f"/select,{target}"])
        return
    open_folder(target.parent if target.is_file() else target)
