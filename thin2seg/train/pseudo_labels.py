"""Pseudo-labels for unannotated recordings.

A teacher network segments each unannotated recording at full photons and produce a reference mask.
Cells the teacher itself is unsure of are ignored in the loss. Uncertain cells are:

* cells whose mask changes between two disjoint half-photon splits of the recording
  (photon-split sensitivity = 1 - mean best-match IoU with the two halves, above ``tau``);
* cells that fail the flow-error check.
"""

from __future__ import annotations

import numpy as np

from thin2seg.data.thinning import disjoint, rng_for
from thin2seg.eval.metrics import best_match
from thin2seg.model.cellpose import flow_errors, predict


def split_sensitivity(labels: np.ndarray, halves: list[np.ndarray]) -> np.ndarray:
    """Per label id (array index; 0 unused): 1 - mean best-match IoU of the cell with the
    predictions on photon-split halves of the same recording."""
    return 1.0 - np.mean([best_match(labels, h)[1] for h in halves], axis=0)


def unreliable(labels: np.ndarray, errors: dict[int, float], sens: np.ndarray,
               flow_threshold: float, tau: float) -> np.ndarray:
    """For each labelled mask, check whether the cell fail the flow check or it is 
    photon-split unstable above ``tau``. Index 0 (background) is always False."""
    n = int(labels.max()) + 1
    fe = np.array([errors.get(i, 0.0) for i in range(n)])
    bad = (fe > flow_threshold) | (np.asarray(sens)[:n] > tau)
    bad[0] = False
    return bad


def teacher_labels(model, counts: np.ndarray, ecfg: dict, name: str, device, tau: float = 0.3):
    """Full-photon masks of the teacher on one recording, and the pixels to ignore.

    Returns: int32 labels, unreliable cell label mask, number of unreliable cells, number of teacher cells."""
    halves = disjoint(counts, [0.5, 0.5], rng_for(name, "teacher_halves"))
    (lab, dP, _), (h1, _, _), (h2, _, _) = predict(model, [counts, *halves], {**ecfg, "flow_threshold": 0.0})
    thr = float(ecfg.get("flow_threshold", 0.8))
    ferr = flow_errors(lab, dP, device)
    bad = unreliable(lab, ferr, split_sensitivity(lab, [h1, h2]), thr, tau)
    n = int(lab.max()) + 1
    fe = np.array([ferr.get(i, 0.0) for i in range(n)])
    ignore = bad[lab]
    kept = np.where(fe[lab] > thr, 0, lab)
    return kept.astype(np.int32), ignore, bad[1:].sum(), n - 1
