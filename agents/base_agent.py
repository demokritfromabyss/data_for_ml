from __future__ import annotations

from pathlib import Path
import yaml


class BaseAgent:
    def __init__(self, config: str | dict | None = None):
        self.config = self._load_config(config)

    @staticmethod
    def _load_config(config: str | dict | None) -> dict:
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
