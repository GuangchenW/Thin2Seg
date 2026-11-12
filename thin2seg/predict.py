"""Segmenting FLIM recording with a trained model (``task: predict``).

Example config definition:

    task: predict
    predict:
      ckpt: path/to/ckpt        # or cpsam_v2 for zero-shot Cellpose-SAM
      images_csv: data/short_acquisitions.csv    # manifest: image_path, channel
      out_dir: runs/predictions
      overlays: true            # also write <name>_overlay.png
      budgets: null             # optional, e.g. [1, 0.5]: segment thinned copies instead,
                                # to preview how the model behaves at shorter acquisitions

Writes ``<name>_masks.tif`` (int32 instance labels) per recording and ``summary.csv``.
With ``budgets``, files are named ``<name>_<budget>_masks.tif``.

The network needs no denoising or other preprocessing: it is applied to the photon counts.
To certify the masks of short acquisitions, use ``task: certify``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from thin2seg.config import data_settings, eval_settings, section, setup_logging
from thin2seg.data.budget import ReferenceTable, parse_budgets, resolve
from thin2seg.data.io import load_counts, read_manifest
from thin2seg.data.thinning import rng_for, thin

logger = logging.getLogger(__name__)


def main(params: dict):
    from thin2seg.model.cellpose import load_model, predict

    cfg = section(params, "predict")
    data, ecfg = data_settings(params), eval_settings(params)
    out = Path(cfg["out_dir"])
    setup_logging(out, "predict.log")
    budgets = parse_budgets(cfg["budgets"], data["budget_mode"]) if cfg.get("budgets") else None
    refs = ReferenceTable(data.get("reference_csv")) if budgets else None
    model = load_model(cfg["ckpt"])
    rows = []
    for _, r in read_manifest(cfg["images_csv"]).iterrows():
        counts = load_counts(r["image_path"], int(r["channel"]), data.get("counts_dir"))
        jobs = [("", counts, {})]
        if budgets:
            jobs = []
            for b in budgets:
                info = resolve(b, r["name"], refs, counts)
                jobs.append((f"_{b.label}", thin(counts, info["rate"], rng_for(r["name"], b.label, "predict")),
                             {"budget": b.label, "rate": info["rate"], "photons_px": info["photons_px"]}))
        for (tag, x, info), (lab, _, _) in zip(jobs, predict(model, [j[1] for j in jobs], ecfg)):
            tifffile.imwrite(out / f"{r['name']}{tag}_masks.tif", lab, compression="zlib")
            if cfg.get("overlays", True):
                from thin2seg.visual.overlays import save_overlay
                save_overlay(out / f"{r['name']}{tag}_overlay.png", x, lab,
                             title=f"{r['name']}{tag.replace('_', ' ')}: {int(lab.max())} cells")
            rows.append({"name": r["name"], **info, "n_cells": int(lab.max()),
                         "photons_total": int(np.asarray(x).sum())})
            logger.info(f"{r['name']}{tag}: {int(lab.max())} cells")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    logger.info(f"{len(rows)} predictions -> {out}")
