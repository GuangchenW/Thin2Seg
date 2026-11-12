"""A config file is a YAML file with ``task`` (see ``thin2seg.main``), a section specific to the
task, and these common sections:

``data``
    ``labeled_csv``    manifest of annotated recordings (``image_path``, ``channel``, ``instance_map_path``)
    ``pool_csv``       manifest of unannotated recordings (``image_path``, ``channel``)
    ``counts_dir``     cache directory for decoded photon-count images (optional)
    ``reference_csv``  reference table written by ``task: photon_reference`` (optional; 
                        required for resolving exact absolute budgets)
    ``budget_mode``    ``absolute`` (default) or ``ratio``

``device``
    ``auto`` (default: GPU if there is one), ``cuda`` or ``cpu``.

``cellpose``
    ``pretrained_model``  starting weights (default ``cpsam_v2``)
    ``eval``              ``flow_threshold`` (0.8), ``cellprob_threshold`` (0.0), ``min_size`` (15),
                          ``batch_size`` (8)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER = "thin2seg"

EVAL_DEFAULTS = {"flow_threshold": 0.8, "cellprob_threshold": 0.0, "min_size": 15, "batch_size": 8}


def setup_logging(out_dir: str | Path, name: str = "run.log") -> logging.Logger:
    """Log to stdout and append to ``out_dir/name`` with timestamp."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(ROOT_LOGGER)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / name, mode="a")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def eval_settings(params: dict) -> dict:
    """Mask-extraction settings (``cellpose.eval``), uses default if not provided."""
    return {**EVAL_DEFAULTS, **((params.get("cellpose") or {}).get("eval") or {})}


def pretrained_model(params: dict) -> str:
    return str((params.get("cellpose") or {}).get("pretrained_model", "cpsam_v2"))


def data_settings(params: dict) -> dict:
    d = dict(params.get("data") or {})
    d.setdefault("budget_mode", "absolute")
    return d


def section(params: dict, task: str) -> dict:
    """The task specific config section."""
    if task not in params or params[task] is None:
        raise KeyError(f"The config has task: {task} but no '{task}:' section")
    return params[task]
