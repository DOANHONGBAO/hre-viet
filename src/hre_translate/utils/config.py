from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    root = project_root()
    config_path = Path(path) if path else root / "configs" / "data.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, root


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
