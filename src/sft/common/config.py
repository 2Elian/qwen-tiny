"""Config loading utilities."""

import os
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    return cfg


def resolve_path(path: str, base_dir: str = None) -> str:
    """Resolve a path relative to base_dir or expand ~."""
    path = os.path.expanduser(path)
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return path
