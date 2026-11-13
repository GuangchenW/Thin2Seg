"""Rendering photon-count images and masks.

* :func:`display`: a count image as grey RGB, on a log scale under a 1st-99th percentile stretch;
* :func:`instance_overlay`: masks in random colours with outlines;
* :func:`status_overlay`: masks coloured by whether they match an annotated cell;
* :func:`save_overlay`: input + masks figure for one image (used by ``task: predict``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rgb(hex_: str) -> np.ndarray:
    """``'#RRGGBB'`` as a float RGB array in [0, 1]."""
    h = hex_.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


MATCHED = _rgb("#2A78D6")     # predicted cell matched to an annotated cell
EXTRA = _rgb("#FF7D4A")       # predicted cell without a match
MISSED = _rgb("#FF0000")      # annotated cell nothing matches (outline)
OUTLINE = _rgb("#FF00FF")


def stretch(v: np.ndarray) -> np.ndarray:
    """1st-99th percentile stretch, clipped to [0, 1]."""
    lo, hi = np.percentile(v, (1, 99))
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def display(counts: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """``log1p(counts / rate)`` under :func:`stretch`, as grey RGB [H, W, 3]. Dividing by the
    thinning rate so copies at different budgets are on one intensity scale."""
    v = stretch(np.log1p(np.clip(np.asarray(counts, np.float32), 0, None) / max(rate, 1e-9)))
    return np.repeat(v[..., None], 3, 2)


def instance_overlay(base: np.ndarray, labels: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    """Filled instances (random colours) with magenta outlines on ``base`` [H, W, 3].
    Optionally renders the annotation outlines in green."""
    from skimage.segmentation import find_boundaries

    rng = np.random.default_rng(0)
    n = max(labels.max(), 1)
    lut = np.concatenate([[[0, 0, 0]], rng.random((n, 3)) * 0.7 + 0.3])
    over = np.where((labels > 0)[..., None], 0.45 * lut[np.clip(labels, 0, n)] + 0.55 * base, base)
    if gt is not None:
        over[find_boundaries(gt, mode="inner")] = _rgb("#1AE61A")
    over[find_boundaries(labels, mode="inner")] = OUTLINE
    return over


def status_overlay(base: np.ndarray, pred: np.ndarray, gt: np.ndarray, iou: float = 0.5) -> tuple[np.ndarray, dict]:
    """Predicted cells filled blue where they match an annotated cell at ``iou`` and orange
    where they do not; missed annotated cells outlined in red. Also returns the counts and F1."""
    from skimage.segmentation import find_boundaries
    from thin2seg.eval.metrics import best_match

    bid, biou = best_match(pred, gt)          # index = predicted id (0 = background)
    matched = biou >= iou                     # one-to-one for iou >= 0.5
    matched[0] = False
    hit = set(bid[matched].tolist())
    missed_ids = [i for i in range(1, int(gt.max()) + 1) if i not in hit]
    out = base.copy()
    m = (pred > 0) & matched[pred]
    e = (pred > 0) & ~matched[pred]
    out[m] = 0.5 * out[m] + 0.5 * MATCHED
    out[e] = 0.45 * out[e] + 0.55 * EXTRA
    out[find_boundaries(pred, mode="inner")] = _rgb("#0D0D0D")
    out[find_boundaries(np.where(np.isin(gt, missed_ids), gt, 0), mode="thick")] = MISSED
    tp, fp, fn = int(matched[1:].sum()), int((~matched[1:]).sum()), len(missed_ids)
    return out, {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                 "matched_iou": biou[matched].mean() if tp else 0.0}


def save_overlay(path: str | Path, counts: np.ndarray, labels: np.ndarray, title: str = "",
                 gt: np.ndarray | None = None, dpi: int = 110):
    """Renders and saves two-panel figure: the count image, and the same image with the masks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = display(counts)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    ax[0].imshow(base)
    ax[0].set_title("input (photon counts, log scale)", fontsize=10)
    ax[1].imshow(instance_overlay(base, labels, gt))
    ax[1].set_title(title or f"{int(labels.max())} cells", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
