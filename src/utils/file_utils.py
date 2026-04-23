from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import json


APP_NAME = "FraudShield"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_resource_root() -> Path:
    if is_frozen_app():
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root)
        return Path(sys.executable).resolve().parent
    return PROJECT_ROOT


def get_user_data_root() -> Path:
    if not is_frozen_app():
        return PROJECT_ROOT
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


RESOURCE_ROOT = get_resource_root()
USER_DATA_DIR = get_user_data_root()
ASSETS_DIR = RESOURCE_ROOT / "assets"
DATA_DIR = RESOURCE_ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
EXAMPLES_DIR = RESOURCE_ROOT / "examples"
MODELS_DIR = USER_DATA_DIR / "models"
OUTPUTS_DIR = USER_DATA_DIR / "outputs"
LOGS_DIR = USER_DATA_DIR / "logs"


def resource_path(relative_path: Path | str) -> Path:
    return RESOURCE_ROOT / Path(relative_path)


def get_dialog_start_dir(preferred: Path | str | None = None) -> str:
    candidates = [
        Path(preferred) if preferred else None,
        USER_DATA_DIR,
        RESOURCE_ROOT,
        Path.home(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return str(RESOURCE_ROOT)


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def timestamp_slug(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_timestamped_dir(base_dir: Path | str, prefix: str = "run") -> Path:
    base = ensure_dir(base_dir)
    target = base / timestamp_slug(prefix)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shorten_path(path: Path | str, max_length: int = 60) -> str:
    value = str(path)
    if len(value) <= max_length:
        return value
    return f"...{value[-(max_length - 3):]}"
