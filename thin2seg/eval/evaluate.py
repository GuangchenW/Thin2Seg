"""Scoring a model on annotated images across budgets (``task: evaluate``).

Every annotated test image is segmented at full photons and at each evaluation budget, using
``eval_draws`` disjoint thinned copies. The copies depend only on the image and the budget, so
all models are scored on identical inputs.

Example task definition:
    task: evaluate
    evaluate:
      out_dir: runs/organoid/eval
      eval_budgets: [full, 2, 1, 0.5]
      eval_draws: 3
      images: null                 # file stems from data.labeled_csv; null = all of them
      models:
        - {name: zero_shot, ckpt: cpsam_v2}
        - {name: thin2seg, ckpt: path/to/ckpt}
      save_masks: false            # also write every predicted label image

Writes ``eval.csv`` (one row per model, image, budget and draw) and ``summary.csv`` (means per
model and budget).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from thin2seg.config import data_settings, eval_settings, section, setup_logging
from thin2seg.data.budget import ReferenceTable, parse_budgets, resolve
from thin2seg.data.io import load_counts, load_labels, read_manifest
from thin2seg.data.thinning import evaluation_copies
from thin2seg.eval.metrics import extent_metrics, f1_scores
from thin2seg.model.cellpose import predict

logger = logging.getLogger(__name__)

METRICS = ["f1_50", "f1_75", "matched_iou", "area_ratio_med", "shrunk_or_missed", "n_pred", "n_gt"]


def evaluate_image(model, counts: np.ndarray, gt: np.ndarray, budgets, draws: int, ecfg: dict, name: str,
                   refs: ReferenceTable, save_dir: Path | None = None) -> list[dict]:
    """Returns a list of evauation results.
    Each result contains info for one image at every budget and draw, with the budget, 
    the thinning rate and the photons per cell pixel it corresponds to. 
    ``save_dir``: if provided, write each prediction as ``<name>_<budget>_d<draw>.tif``.
    """
    rows = []
    for b in budgets:
        info = resolve(b, name, refs, counts)
        acqs = evaluation_copies(counts, b, info["rate"], name, draws)
        for d, (lab, _, _) in enumerate(predict(model, acqs, ecfg)):
            rows.append({**info, "draw": d, **f1_scores(lab, gt), **extent_metrics(lab, gt)})
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(save_dir / f"{name}_{info['budget_label']}_d{d}.tif", lab.astype(np.int32),
                                 compression="zlib")
    return rows


def load_annotated(manifest: pd.DataFrame, names: list[str], cache, refs: ReferenceTable) -> list[tuple]:
    """Returns list of (name, counts, annotation) for each named image.
    Also registers its reference level from the annotation when the reference table does not list it."""
    rows = manifest.set_index("name")
    out = []
    for n in names:
        if n not in rows.index:
            raise KeyError(f"{n} is not in the annotated manifest")
        counts = load_counts(rows.loc[n, "image_path"], int(rows.loc[n, "channel"]), cache)
        gt = load_labels(rows.loc[n, "instance_map_path"])
        refs.get(n, counts, cells=gt)
        out.append((n, counts, gt))
    return out


def summarise(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean of every metric per ``by`` and budget. First over draws, then over images."""
    keys = by + ["budget_label"]
    per_image = results.groupby(keys + ["name"], sort=False)[METRICS].mean()
    return per_image.groupby(keys, sort=False).mean().reset_index()


def main(params: dict):
    from thin2seg.model.cellpose import load_model

    cfg = section(params, "evaluate")
    data, ecfg = data_settings(params), eval_settings(params)
    out = Path(cfg["out_dir"])
    setup_logging(out, "evaluate.log")
    budgets = parse_budgets(cfg.get("eval_budgets", ["full", 2, 1, 0.5]), data["budget_mode"])
    draws = int(cfg.get("eval_draws", 3))
    refs = ReferenceTable(data.get("reference_csv"))
    manifest = read_manifest(data["labeled_csv"], labeled=True)
    names = list(cfg.get("images") or manifest["name"])
    images = load_annotated(manifest, names, data.get("counts_dir"), refs)

    res_f = out / "eval.csv"
    results = pd.read_csv(res_f).to_dict("records") if res_f.exists() else []
    done = {r["model"] for r in results}
    for m in cfg["models"]:
        if m["name"] in done:
            logger.info(f"{m['name']}: already in {res_f}, skipped")
            continue
        model = load_model(m["ckpt"])
        for name, counts, gt in images:
            sd = out / "masks" / m["name"] if cfg.get("save_masks") else None
            results += [{"model": m["name"], "name": name, **r}
                        for r in evaluate_image(model, counts, gt, budgets, draws, ecfg, name, refs, sd)]
        pd.DataFrame(results).to_csv(res_f, index=False)
        logger.info(f"{m['name']}: evaluated on {len(images)} images")
        del model
    summary = summarise(pd.DataFrame(results), ["model"])
    summary.to_csv(out / "summary.csv", index=False)
    logger.info("mean over images:\n" + summary.round(3).to_string(index=False))
