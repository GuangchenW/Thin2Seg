"""Instance matching and segmentation metrics.

* :func:`best_match`: find for every instance of a label image, its best-IoU instance in
  another. Used wherever a cell is followed across acquisitions (photon stability, pseudo-label
  filtering, certification outcomes, etc.)
* :func:`f1_scores`: F1 at IoU thresholds with optimal one-to-one matching (Hungarian on the IoU matrix).
* :func:`extent_metrics`: how much of each annotated cell the prediction covers. At low photon
  counts, a mask can shrink onto the bright nucleus and still count as a match if the nucleus is large enough
  to result in sufficiently high IoU.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


def contingency(a: np.ndarray, b: np.ndarray):
    """Returns (sparse overlap counts [na+1, nb+1], areas of a, areas of b). IDs must be >= 0."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    c = sparse.coo_matrix((np.ones(a.size, np.int32), (a.ravel(), b.ravel())), shape=(na, nb)).tocsr()
    return c, np.bincount(a.ravel(), minlength=na), np.bincount(b.ravel(), minlength=nb)


def best_match(a: np.ndarray, b: np.ndarray):
    """For each instance in ``a``, find the best-IoU instance in ``b`` and that IoU.
    Returns 0 when nothing overlaps.
    """
    c, aa, ab = contingency(a, b)
    c = c.tocoo()
    keep = (c.row > 0) & (c.col > 0)
    r, k, v = c.row[keep], c.col[keep], c.data[keep].astype(np.float64)
    iou = v / (aa[r] + ab[k] - v)
    best_id = np.zeros(len(aa), np.int64)
    best_iou = np.zeros(len(aa))
    order = np.lexsort((iou, r)) # per row, the largest IoU comes last
    best_id[r[order]] = k[order]
    best_iou[r[order]] = iou[order]
    return best_id, best_iou


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Returns dense IoU [P, G] between every predicted and every annotated instance."""
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return np.zeros((len(pred_ids), len(gt_ids)), np.float64)
    c, ap, ag = contingency(pred, gt)
    inter = np.asarray(c[pred_ids][:, gt_ids].todense(), np.float64)
    union = ap[pred_ids][:, None] + ag[gt_ids][None, :] - inter
    return inter / np.maximum(union, 1e-9)


def f1_scores(pred: np.ndarray, gt: np.ndarray, thresholds=(0.5, 0.75)) -> dict:
    """Computes F1 at each IoU threshold: ``f1_50``, ``f1_75``, etc. plus ``n_pred``, ``n_gt``
    and ``matched_iou``.
    """
    iou = iou_matrix(pred, gt)
    n_pred, n_gt = iou.shape
    out = {"n_pred": int(n_pred), "n_gt": int(n_gt)}
    r, c = linear_sum_assignment(-iou) if iou.size else (np.zeros(0, np.int32), np.zeros(0, np.int32))
    matched = iou[r, c] if iou.size else np.zeros(0)
    for t in thresholds:
        tp = int((matched >= t).sum())
        out[f"f1_{int(round(t * 100))}"] = 2 * tp / (n_pred + n_gt) if (n_pred + n_gt) else 0.0
    ok = matched >= min(thresholds)
    out["matched_iou"] = matched[ok].mean() if ok.any() else 0.0
    return out


def extent_metrics(pred: np.ndarray, gt: np.ndarray, shrink: float = 0.6) -> dict:
    """Report the mask extent metrics against the annotation.
        ``area_ratio_med``  : the median over annotated cells of the area of the 
            best-IoU predicted cell over the annotated area (0 if none);
        ``shrunk_or_missed``: the fraction of annotated cells whose best match 
            covers less than ``shrink`` of their area or is missing.
    """
    bid, _ = best_match(gt, pred)
    a_gt = np.bincount(gt.ravel(), minlength=int(gt.max()) + 1).astype(np.float64)
    a_pr = np.bincount(pred.ravel(), minlength=int(pred.max()) + 1).astype(np.float64)
    ids = np.nonzero(a_gt[1:])[0] + 1
    ratio = np.where(bid[ids] > 0, a_pr[bid[ids]] / a_gt[ids], 0.0)
    return {"area_ratio_med": float(np.median(ratio)) if len(ratio) else np.nan,
            "shrunk_or_missed": float(np.mean(ratio < shrink)) if len(ratio) else np.nan}
