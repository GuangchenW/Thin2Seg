"""Compute model error and photon-induced error of a model under thinning (``task: error_budget``).

The full recording is a reference for every thinned copy of it, so the error of a model at a
short acquisition splits comprises two components:

* **model error**: the mask on the full recording is already wrong;
* **photon-induced error**: the mask is right at full photons but wrong at budget b. Reported
  as ``lost``.

The photon-induced error does not require labels to evaluate, so it can be evaluated against
unannotated recordings. ``labelfree_changed``: the fraction of the full-recording masks of 
the model that change at the budget (best-match IoU < tau).

Every number is reported at two cell-probability thresholds ``c``: the default (0), and the
calibrated one that maximises the F1 of the model on the annotated cells of the image at full
photons. This removes any image-wide bias in mask extent, but can only be done during evaluation
since in practice there is no annotation to maximise the thresholds against.

``mode: collect`` (GPU) runs one network pass per input (the full recording and ``draws`` thinned
copies per budget) and computes the masks for every threshold ``c`` from that one output. It writes,
per model, ``cells/<model>.csv`` (IoU against every annotated cell), ``refcells/<model>.csv`` (IoU
against every full-recording prediction) and ``grid/<model>.csv`` (cell counts).

``mode: analyse`` (CPU) writes ``analysis/f1.csv``, ``analysis/decomposition.csv`` and ``analysis/summary.csv``.

Example task definition:
    task: error_budget
    error_budget:
      out_dir: runs/organoid/error_budget
      mode: both
      budgets: [2, 1, 0.5]
      draws: 3
      taus: [0.5, 0.75]
      models:
        - name: thin2seg_foldA
          ckpt: runs/organoid/thin2seg/foldA/thin2seg_foldA
          images: [stem, ...]        # annotated images this model did not train on
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thin2seg.model.cellpose import get_device, load_model, masks_for_thresholds, predict
from thin2seg.config import data_settings, eval_settings, section, setup_logging
from thin2seg.data.budget import ReferenceTable, parse_budgets
from thin2seg.data.io import load_counts, load_labels, read_manifest
from thin2seg.data.thinning import disjoint, rng_for
from thin2seg.eval.metrics import best_match

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3]


def collect(cfg: dict, data: dict, ecfg: dict):
    out = Path(cfg["out_dir"])
    for sub in ("cells", "refcells", "grid"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ft, min_size = float(ecfg["flow_threshold"]), int(ecfg["min_size"])
    refs = ReferenceTable(data.get("reference_csv"))
    lab = read_manifest(data["labeled_csv"], labeled=True).set_index("name")
    budgets = parse_budgets(cfg.get("budgets", [2, 1, 0.5]), data["budget_mode"])
    draws = int(cfg.get("draws", 3))
    cs = [float(c) for c in cfg.get("cellprob_thresholds", DEFAULT_THRESHOLDS)]
    device = get_device()
    for m in cfg["models"]:
        if (out / "cells" / f"{m['name']}.csv").exists():
            logger.info(f"model {m['name']}: already collected")
            continue
        model = load_model(m["ckpt"])
        cells, refcells, grid = [], [], []
        for name in m["images"]:
            row = lab.loc[name]
            full = load_counts(row["image_path"], int(row["channel"]), data.get("counts_dir"))
            gt = load_labels(row["instance_map_path"])
            ref_level = refs.get(name, full, cells=gt)
            gids = np.unique(gt)
            gids = gids[gids > 0]
            inputs = [("full", 0, full)]
            for b in budgets:
                inputs += [(b.label, d, disjoint(full, [b.rate(ref_level)], rng_for(name, "error_budget", b.label, d))[0])
                           for d in range(draws)]
            preds = predict(model, [x for _, _, x in inputs], {**ecfg, "flow_threshold": 0.0})
            masks = [masks_for_thresholds(dP, cp, cs, ft, min_size, device) for _, dP, cp in preds]
            for (bl, d, _), mc in zip(inputs, masks):
                for c, lb in mc.items():
                    grid.append({"name": name, "budget": bl, "draw": d, "c": c, "n_pred": int((np.unique(lb) > 0).sum()),
                                 "n_gt": len(gids)})
                    cells.append(pd.DataFrame({"name": name, "budget": bl, "draw": d, "c": c, "gt_cell": gids,
                                               "iou": best_match(gt, lb)[1][gids]}))
                    if bl != "full":
                        ref = masks[0][c]
                        rids = np.unique(ref)
                        rids = rids[rids > 0]
                        refcells.append(pd.DataFrame({"name": name, "budget": bl, "draw": d, "c": c, "ref_cell": rids,
                                                      "iou_ref": best_match(ref, lb)[1][rids]}))
            logger.info(f"  {m['name']} {name}: {len(gids)} annotated cells, {len(inputs)} inputs")
        pd.concat(cells, ignore_index=True).to_csv(out / "cells" / f"{m['name']}.csv", index=False)
        pd.concat(refcells, ignore_index=True).to_csv(out / "refcells" / f"{m['name']}.csv", index=False)
        pd.DataFrame(grid).to_csv(out / "grid" / f"{m['name']}.csv", index=False)
        del model
        torch.cuda.empty_cache()


def f1_from(cells: pd.DataFrame, grid: pd.DataFrame, tau: float) -> pd.Series:
    """F1 at IoU ``tau`` per (name, budget, draw, c) against the annotated and predicted counts."""
    tp = cells.assign(ok=cells["iou"] >= tau).groupby(["name", "budget", "draw", "c"])["ok"].sum()
    g = grid.set_index(["name", "budget", "draw", "c"])
    return 2 * tp / (g["n_gt"] + g["n_pred"])


def decompose(iou_full: np.ndarray, iou_b: np.ndarray, tau: float) -> dict:
    """Returns fractions of annotated cells: 
    correct at both, photon-induced (correct at full photons only), model error (wrong at full photons),
    recovered (wrong at full, right at the budget).
    Also returns ``lost``, fraction of cells lost due to photon-induced errors."""
    f, b = np.asarray(iou_full) >= tau, np.asarray(iou_b) >= tau
    n = max(len(f), 1)
    return {"ok_both": (f & b).sum() / n, "photon_induced": (f & ~b).sum() / n,
            "model_error": (~f).sum() / n, "recovered": (~f & b).sum() / n,
            "lost": (f & ~b).sum() / max(f.sum(), 1)}


def analyse(cfg: dict):
    out = Path(cfg["out_dir"])
    taus = [float(t) for t in cfg.get("taus", [0.5, 0.75])]
    f1_rows, dec_rows = [], []
    for m in cfg["models"]:
        name_m = m["name"]
        cells = pd.read_csv(out / "cells" / f"{name_m}.csv")
        refc = pd.read_csv(out / "refcells" / f"{name_m}.csv")
        grid = pd.read_csv(out / "grid" / f"{name_m}.csv")
        for tau in taus:
            f1 = f1_from(cells, grid, tau).groupby(["name", "budget", "c"]).mean()       # mean over draws
            for img in m["images"]:
                cf = cells[(cells["name"] == img) & (cells["budget"] == "full")].pivot(index="gt_cell", columns="c", values="iou")
                c_cal = float(f1.loc[img, "full"].idxmax())
                for b in f1.loc[img].index.get_level_values(0).unique():
                    fb = f1.loc[img, b]
                    f1_rows.append({"model": name_m, "name": img, "tau": tau, "budget": b,
                                    "f1_default": float(fb.loc[0.0]), "f1_calibrated": float(fb.loc[c_cal]),
                                    "c_calibrated": c_cal, "f1_best": float(fb.max()), "c_best": float(fb.idxmax())})
                    if b == "full":
                        continue
                    for which, c in (("default", 0.0), ("calibrated", c_cal)):
                        low = cells[(cells["name"] == img) & (cells["budget"] == b) & (cells["c"] == c)]
                        for d, g in low.groupby("draw"):
                            dd = decompose(cf[c].loc[g["gt_cell"]].to_numpy(), g["iou"].to_numpy(), tau)
                            rc = refc[(refc["name"] == img) & (refc["budget"] == b) & (refc["c"] == c) & (refc["draw"] == d)]
                            dec_rows.append({"model": name_m, "name": img, "tau": tau, "budget": b, "draw": d,
                                             "threshold": which, "c": c, **dd,
                                             "labelfree_changed": (rc["iou_ref"] < tau).mean() if len(rc) else np.nan})
    (out / "analysis").mkdir(exist_ok=True)
    pd.DataFrame(f1_rows).to_csv(out / "analysis" / "f1.csv", index=False)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(out / "analysis" / "decomposition.csv", index=False)
    if len(dec):
        s = dec.groupby(["model", "tau", "threshold", "budget"], sort=False)[
            ["model_error", "lost", "photon_induced", "ok_both", "recovered", "labelfree_changed"]].mean().reset_index()
        s.to_csv(out / "analysis" / "summary.csv", index=False)
        logger.info("mean over images and draws:\n" + s.round(3).to_string(index=False))


def main(params: dict):
    cfg = section(params, "error_budget")
    setup_logging(cfg["out_dir"], "error_budget.log")
    mode = cfg.get("mode", "both")
    if mode in ("collect", "both"):
        collect(cfg, data_settings(params), eval_settings(params))
    if mode in ("analyse", "both"):
        analyse(cfg)
