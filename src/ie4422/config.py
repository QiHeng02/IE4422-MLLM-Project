"""Load the project config so every script reads the same settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config. Relative paths inside it resolve from the repo root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def repo_path(p: str | Path) -> Path:
    """Resolve a path from the config against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
