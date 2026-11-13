"""Conformal selection with false-discovery-rate control (Jin and Candes, JMLR 24, 2023).

Each predicted cell carries a reliability score ``r`` (higher = more reliable). Calibration
units additionally carry a binary outcome ``y`` (1 = reliable). For a test unit with score 
``r_j`` the conformal p-value of the null "y_j = 0" is

    p_j = (1 + #{calibration i : y_i = 0 and r_i >= r_j}) / (n + 1), where n = calibration size

Benjamini-Hochberg at level ``alpha`` over the test p-values selects a set whose expected 
fraction of units with y = 0 (the FDR) is at most ``alpha`` whenever calibration and test units
are exchangeable.

Cells of one image are not exchangeable with each other, so the BH guarantee holds on average
over images. :func:`ltt_threshold` instead treats whole acquisitions as the exchangeable units
and bounds the probability that a new acquisition's FDP exceeds ``alpha``.
"""
from __future__ import annotations

import numpy as np


def conformal_pvalues(r_cal, y_cal, r_test) -> np.ndarray:
    """Conformal p-values of the test units for the null "unreliable"."""
    r_cal = np.asarray(r_cal, float)
    y_cal = np.asarray(y_cal, bool)
    r_test = np.asarray(r_test, float)
    if r_cal.shape != y_cal.shape:
        raise ValueError(f"r_cal {r_cal.shape} and y_cal {y_cal.shape} differ")
    null = np.sort(r_cal[~y_cal])
    n_ge = len(null) - np.searchsorted(null, r_test, side="left")   # null scores >= r_j
    return (1.0 + n_ge) / (len(r_cal) + 1.0)


def bh_select(p, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg at level ``alpha``.
    Returns boolean mask of the rejected (selected) units."""
    p = np.asarray(p, float)
    m = len(p)
    sel = np.zeros(m, bool)
    if m == 0:
        return sel
    order = np.argsort(p, kind="stable")
    ok = np.nonzero(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    if len(ok):
        sel[order[: ok[-1] + 1]] = True
    return sel


def certify(r_cal, y_cal, r_test, alpha: float) -> np.ndarray:
    """Select the test units certified reliable at FDR level ``alpha``."""
    return bh_select(conformal_pvalues(r_cal, y_cal, r_test), alpha)


def selection_metrics(selected, y) -> dict:
    """False-discovery proportion and power."""
    selected = np.asarray(selected, bool)
    y = np.asarray(y, bool)
    n_sel = int(selected.sum())
    n_good = int(y.sum())
    tp = int((selected & y).sum())
    return {"n_sel": n_sel, "n_units": int(len(y)), "n_good": n_good,
            "fdp": (n_sel - tp) / n_sel if n_sel else 0.0,
            "power": tp / n_good if n_good else float("nan")}


# ---------------------------------------------------------------------------------------
# per-acquisition guarantee
# ---------------------------------------------------------------------------------------

def group_fdp(groups, lam: float) -> np.ndarray:
    """False-discovery proportion of the selection ``r >= lam`` in each group of (r, y) arrays
    (0 for an empty selection)."""
    out = np.empty(len(groups))
    for i, (r, y) in enumerate(groups):
        s = np.asarray(r) >= lam
        n = s.sum()
        out[i] = (s & ~np.asarray(y, bool)).sum() / n if n else 0.0
    return out


def ltt_threshold(groups, alpha: float, delta: float, eps: float, n_grid: int = 101,
                  grid_scores=None) -> float:
    """Returns score threshold with a per-group guarantee, by Learn-then-Test.

    For each threshold on a grid, from strict to lenient, P(group FDP > alpha) > delta is
    tested with the exact binomial test, stopping at the first threshold that is not rejected.
    With probability >= (1 - eps) over the calibration set, a new acquisition then has 
    P(FDP > alpha) <= delta at the returned threshold.
    
    Returns +inf (certify nothing) when even the strictest threshold fails.
    The grid is the score quantiles of ``grid_scores`` (pass scores from data disjoint from 
    the calibration set, so the grid is fixed in advance), else of the calibration scores."""
    from scipy.stats import binom

    r_all = (np.asarray(grid_scores, float) if grid_scores is not None
             else np.concatenate([np.asarray(r, float) for r, _ in groups]))
    grid = np.unique(np.quantile(r_all, np.linspace(0.0, 1.0, n_grid)))[::-1]
    lam_hat, n = np.inf, len(groups)
    for lam in grid:
        k = int((group_fdp(groups, lam) > alpha).sum())
        if binom.cdf(k, n, delta) <= eps:
            lam_hat = lam
        else:
            break
    return float(lam_hat)
