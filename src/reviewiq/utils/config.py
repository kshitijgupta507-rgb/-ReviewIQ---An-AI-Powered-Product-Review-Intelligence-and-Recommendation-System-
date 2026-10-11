"""Configuration loader for ReviewIQ.

Uses OmegaConf to load YAML configs, merge with environment overrides,
and expose a typed DictConfig object to all modules.

Usage
-----
    from reviewiq.utils.config import load_config
    cfg = load_config()          # loads configs/default.yaml
    cfg = load_config("configs/model.yaml")   # merge additional file
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _find_repo_root() -> Path:
    """Walk up from this file until we find a directory containing 'configs/'.

    This works whether the package is installed in editable mode or not.
    Falls back to the current working directory if not found.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "configs" / "default.yaml").exists():
            return parent
    # Fallback: assume CWD is the repo root (works when running from repo root)
    return Path.cwd()


_DEFAULT_CONFIG = _find_repo_root() / "configs" / "default.yaml"


def load_config(extra_config: str | Path | None = None) -> DictConfig:
    """Load the default config and optionally merge an extra config file.

    Parameters
    ----------
    extra_config:
        Path to a secondary YAML file whose values override the default.

    Returns
    -------
    DictConfig
        Merged configuration object.
    """
    cfg: DictConfig = OmegaConf.load(_DEFAULT_CONFIG)  # type: ignore[assignment]

    if extra_config is not None:
        override = OmegaConf.load(extra_config)  # type: ignore[assignment]
        cfg = OmegaConf.merge(cfg, override)

    # Allow environment variables to override specific keys, e.g.:
    #   REVIEWIQ_SEED=0 overrides cfg.seed
    env_seed = os.getenv("REVIEWIQ_SEED")
    if env_seed is not None:
        cfg.seed = int(env_seed)  # type: ignore[assignment]

    env_device = os.getenv("REVIEWIQ_DEVICE")
    if env_device is not None:
        OmegaConf.update(cfg, "device", env_device, merge=True)

    return cfg
