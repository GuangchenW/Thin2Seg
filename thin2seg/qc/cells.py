"""Per-cell measurements and reliability scores for label-free certification.

A predicted cell on a short acquisition is stable across budgets if its mask agrees with the 
mask the same model gives on the full recording. However, whether it is stable is only known where 
the full recording exists (calibration); the features below are available on any acquisition,
and a reliability score built from them ranks cells by how likely they are stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thin2seg.eval.metrics import best_match
from thin2seg.model.cellpose import flow_errors

#: test-time features of the learned score
FEATURES = ["flow_error", "cellprob", "log_area", "log_photons", "photons_per_px"]
SCORES = ("learned", "flow", "cellprob")


def cell_table(lab: np.ndarray, dP: np.ndarray, cellprob: np.ndarray, counts: np.ndarray,
               ref: np.ndarray | None = None, gt: np.ndarray | None = None, device=None) -> pd.DataFrame:
    """One row per predicted cell:
        area, photons, mean cell-probability logit, flow error;
        when given, also the best-match IoU against ``ref`` (``iou_ref``) and
        annotation ``gt`` (``iou_gt``).
    """
    n = int(lab.max()) + 1
    ids = np.unique(lab)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return pd.DataFrame()
    flat = lab.ravel()
    area = np.bincount(flat, minlength=n).astype(np.float64)
    phot = np.bincount(flat, weights=counts.ravel(), minlength=n)
    cp = np.bincount(flat, weights=cellprob.ravel(), minlength=n) / np.maximum(area, 1)
    fe = flow_errors(lab, dP, device)
    out = {"cell": ids, "area": area[ids], "photons": phot[ids], "cellprob": cp[ids],
           "flow_error": [fe.get(int(i), np.nan) for i in ids]}
    if ref is not None:
        ref_id, iou_ref = best_match(lab, ref)
        out["ref_id"], out["iou_ref"] = ref_id[ids], iou_ref[ids]
    if gt is not None:
        out["iou_gt"] = best_match(lab, gt)[1][ids]
    return pd.DataFrame(out)


def add_features(d: pd.DataFrame, flow_error_fill: float | None = None) -> pd.DataFrame:
    """Add derived features in place."""
    d["log_area"] = np.log(d["area"])
    d["log_photons"] = np.log1p(d["photons"])
    d["photons_per_px"] = d["photons"] / d["area"]
    fill = d["flow_error"].max() if flow_error_fill is None else flow_error_fill
    d["flow_error"] = d["flow_error"].fillna(fill)
    return d


def photon_level(cells: pd.DataFrame) -> float:
    """Returns the median of the mean photon count per pixel of predicted cells."""
    return (cells["photons"] / cells["area"].clip(lower=1.0)).median()


class LearnedScore:
    """Logistic regression on standardised :data:`FEATURES` against the probability that a cell is
    photon-stable. Stored as plain arrays."""

    def __init__(self, mean, scale, coef, intercept: float):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept = np.asarray(coef), intercept

    @classmethod
    def fit(cls, cells: pd.DataFrame, stable: np.ndarray) -> "LearnedScore":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(cells[FEATURES].to_numpy())
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(cells[FEATURES].to_numpy()), stable)
        return cls(scaler.mean_, scaler.scale_, clf.coef_[0], clf.intercept_[0])

    def __call__(self, cells: pd.DataFrame) -> np.ndarray:
        z = ((cells[FEATURES].to_numpy() - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


def score_functions(fit_cells: pd.DataFrame | None = None, stable: np.ndarray | None = None) -> dict:
    """Returns reliability scores (higher = more reliable):
        ``flow`` (negative flow error),
        ``cellprob``
        ``learned`` (when ``fit_cells`` are given).
    """
    scores = {"flow": lambda df: -df["flow_error"].to_numpy(),
              "cellprob": lambda df: df["cellprob"].to_numpy()}
    if fit_cells is not None:
        scores["learned"] = LearnedScore.fit(fit_cells, stable)
    return scores
