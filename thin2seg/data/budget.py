"""The photon budget is the photon per cell pixel at which FLIM images may be acquired and analysed.
During training, a range of photon budgets may be selected and evaluated to determine an ideal or
minimum budget to perform experiments with.

Every budget resolves, per image, to a thinning rate ``r`` in (0, 1]. This may be expressed in two ways:

* ``absolute`` (default): the target **median photon count per cell pixel** (ppp). An image is 
  thinned at ``r = min(1, target / ref)``, where ``ref`` is its own median photon count per cell pixel
  at full photons. Every image then lands at the same signal level. An image already dimmer than 
  the target cannot be brightened, so it stays at ``r = 1`` and is flagged ``capped``.
* ``ratio``: ``r`` is the budget itself, a fixed fraction of each recording. I.e. the budget is equal to
  acquisitions that are N times shorter than the reference acquisitions.

``full`` is the unthinned recording in either unit.

``ref`` comes from a reference table (CSV written by ``task: photon_reference``), containing the median
over cell pixels. See data/reference.py for details.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thin2seg.data.io import stem

logger = logging.getLogger(__name__)

MODES = ("absolute", "ratio")


@dataclass(frozen=True)
class Budget:
    """
    ``value``: photons per cell pixel (absolute) or a fraction of the recording (ratio);
    At ``inf`` (absolute) or ``1.0`` (ratio), full recording is recovered.
    """
    value: float
    mode: str = "absolute"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"budget mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "ratio" and not 0.0 < self.value <= 1.0:
            raise ValueError(f"a ratio budget must be in (0, 1], got {self.value}")
        if self.mode == "absolute" and not self.value > 0:
            raise ValueError(f"an absolute budget must be > 0 photons per pixel, got {self.value}")

    @classmethod
    def full(cls, mode: str = "absolute") -> "Budget":
        return cls(math.inf if mode == "absolute" else 1.0, mode)

    @property
    def is_full(self) -> bool:
        return math.isinf(self.value) if self.mode == "absolute" else self.value >= 1.0

    @property
    def label(self) -> str:
        """Name used in files and tables: ``full``, ``p<photons>`` (absolute) or ``r<fraction>``."""
        if self.is_full:
            return "full"
        return f"r{self.value:g}" if self.mode == "ratio" else f"p{self.value:.3g}"

    def rate(self, ref: float | None = None) -> float:
        """Thinning rate for an image whose full-photon median count per cell pixel is ``ref``."""
        if self.is_full: return 1.0
        if self.mode == "ratio": return float(self.value)
        if ref is None or not np.isfinite(ref) or ref <= 0:
            raise ValueError(f"absolute budget {self.value} needs a valid reference level, got {ref}")
        return float(min(1.0, self.value / ref))

    def capped(self, ref: float | None) -> bool:
        """Returns ``true`` if the image is too dim to reach this budget."""
        return self.mode == "absolute" and not self.is_full and ref is not None and ref <= self.value


def parse_budgets(values, mode: str = "absolute") -> list[Budget]:
    """Parse budgets from a config list. Entries are numbers or ``full``."""
    if mode not in MODES:
        raise ValueError(f"budget_mode must be one of {MODES}, got {mode!r}")
    out = []
    for v in values:
        if isinstance(v, str) and v.strip().lower() == "full":
            out.append(Budget.full(mode))
        elif mode == "ratio" and float(v) >= 1.0:
            out.append(Budget.full(mode))
        else:
            out.append(Budget(float(v), mode))
    return out


def reference_level(counts: np.ndarray, cells: np.ndarray) -> float:
    """Median photon count over cell pixels (indicated by ``cells``) of a full-photon count image."""
    m = np.asarray(cells) > 0
    if not m.any(): return float("nan")
    return float(np.median(np.asarray(counts)[m]))


def proxy_foreground(counts: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Model-free foreground: Otsu threshold on the smoothed log counts. Approximate, and used
    only when no cell mask is available."""
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu

    x = gaussian_filter(np.log1p(np.asarray(counts, np.float32)), sigma)
    return x > threshold_otsu(x)


class ReferenceTable:
    """Stores name -> median photons per cell pixel at full photons, from reference CSV (``name``, ``ref_median``)."""

    def __init__(self, path: str | Path | None = None):
        import pandas as pd

        self.path = Path(path) if path else None
        self.table: dict[str, float] = {}
        if self.path is not None:
            if not self.path.exists():
                raise FileNotFoundError(f"reference table {self.path} not found "
                                        f"(build it with task: photon_reference)")
            df = pd.read_csv(self.path)
            self.table = {str(r["name"]): float(r["ref_median"]) for _, r in df.iterrows()}

    def get(self, name: str, counts: np.ndarray | None = None, cells: np.ndarray | None = None) -> float:
        """Reference level of image ``name`` (a stem or a path). Falls back to ``cells`` if given,
        else to :func:`proxy_foreground` on ``counts``, with warning."""
        name = stem(name) if ("/" in str(name) or "\\" in str(name)) else str(name)
        if name in self.table: return self.table[name]
        if counts is None:
            raise KeyError(f"{name}: no reference level in {self.path} and no counts to compute one")
        if cells is not None:
            ref = reference_level(counts, cells)
        else:
            ref = reference_level(counts, proxy_foreground(counts))
            logger.warning(f"{name}: not in the reference table; using the model-free foreground "
                           f"estimate ({ref:.2f} photons per foreground pixel)")
        self.table[name] = ref
        return ref


def resolve(budget: Budget, name: str, refs: ReferenceTable, counts: np.ndarray | None = None) -> dict:
    """Resolves the pair (image, budget) and returns budget and photon information in a dictionary."""
    ref = None
    if budget.mode == "absolute" and not budget.is_full:
        ref = refs.get(name, counts)
    else:
        try:
            ref = refs.get(name, counts)
        except KeyError:
            ref = None
    r = budget.rate(ref)
    return {"budget_mode": budget.mode, "budget": budget.value, "budget_label": budget.label, "rate": r,
            "ref_photons_px": ref, "photons_px": (r * ref) if ref is not None else None,
            "capped": budget.capped(ref)}
