"""
Configuration loading utilities.

Design:
- Single source of truth: configs/default.yaml
- Nested dict access via AttributeDict for clean notebook usage
- Paths resolved relative to the project root (parent of configs/)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


class AttributeDict(dict):
    """Dict that supports attribute-style access for nested configs."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, AttributeDict):
            value = AttributeDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to a plain dict (for JSON / experiment logs)."""
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, AttributeDict):
                out[key] = value.to_dict()
            elif isinstance(value, dict):
                out[key] = AttributeDict(value).to_dict()
            else:
                out[key] = value
        return out


def get_project_root() -> Path:
    """
    Return the repository root.

    Resolved as the parent of the ``configs/`` directory so notebooks work
    regardless of the current working directory when launched from Jupyter.
    """
    return Path(__file__).resolve().parents[1]


def _resolve_paths(cfg: MutableMapping[str, Any], root: Path) -> None:
    """Convert relative path strings under ``paths`` to absolute Path objects."""
    paths = cfg.get("paths", {})
    resolved: dict[str, Any] = {}
    for key, value in paths.items():
        if key == "project_root":
            resolved[key] = root
        elif isinstance(value, str):
            p = Path(value)
            resolved[key] = p if p.is_absolute() else (root / p)
        else:
            resolved[key] = value
    cfg["paths"] = resolved


def load_config(config_path: str | Path | None = None) -> AttributeDict:
    """
    Load the YAML experiment configuration.

    Parameters
    ----------
    config_path:
        Optional path to a YAML file. Defaults to ``configs/default.yaml``.

    Returns
    -------
    AttributeDict
        Nested configuration with resolved absolute paths.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: Mapping[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping, got {type(raw)}")

    cfg = AttributeDict(raw)
    _resolve_paths(cfg, root)

    # Mirror commonly used flat aliases expected by the project brief
    cfg.MODEL_NAME = cfg.model.model_name
    cfg.MAX_LENGTH = cfg.model.max_length
    cfg.NUM_LABELS = cfg.model.num_labels
    cfg.LEARNING_RATE = cfg.training.learning_rate
    cfg.WEIGHT_DECAY = cfg.training.weight_decay
    cfg.WARMUP_RATIO = cfg.training.warmup_ratio
    cfg.LABEL_SMOOTHING = cfg.training.label_smoothing
    cfg.DROPOUT = cfg.model.dropout
    cfg.BATCH_SIZE = cfg.training.batch_size
    cfg.EPOCHS = cfg.training.epochs
    cfg.PATIENCE = cfg.training.patience
    cfg.SEED = cfg.runtime.seed
    cfg.DEVICE = cfg.runtime.device
    cfg.USE_MIXED_PRECISION = cfg.training.use_mixed_precision
    cfg.SAVE_PATH = cfg.paths.save_path
    cfg.RESULTS_PATH = cfg.paths.results_path
    cfg.MODEL_PATH = cfg.paths.model_path

    return cfg
