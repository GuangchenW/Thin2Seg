"""Certifying the predicted masks of new short acquisitions (``task: certify``).

``mode: calibrate`` (calibrate model output against dataset). 
Every recording of a calibration pool is thinned to each budget and segmented; a cell
is stable if its mask agrees with the model's mask on the full recording (IoU >= ``tau``). Half
of the pool fits the reliability score, the other half provides the calibration scores. The
result is the calibrator file.

Example task definition:
    task: certify
    certify:
      mode: calibrate
      ckpt: path/to/ckpt
      out_dir: runs/organoid/certify
      pool_csv: data/organoid/pool_calibration.csv
      budgets: [2, 1, 0.5]           # photon levels the short acquisitions will have
      draws: 2
      tau: 0.5                       # IoU at which a mask counts as unchanged
      score: learned                 # learned | flow | cellprob
      image_guarantee: {alpha: 0.15, delta: 0.2, eps: 0.1}   # optional per-acquisition threshold

``mode: apply``. The model segments the acquisition; each cell gets a conformal p-value against
the calibration cells of the nearest calibrated photon level, and the Benjamini-Hochberg procedure
selects the certified cells at false-discovery rate ``alpha``.

Example task definition:
    task: certify
    certify:
      mode: apply
      ckpt: path/to/ckpt
      calibrator: runs/organoid/certify/calibrator.npz
      images_csv: data/organoid/short_acquisitions.csv
      out_dir: runs/organoid/certified
      alpha: 0.1
      budget: auto                   # or the label of a calibrated budget, e.g. p1

For each acquisition it writes ``<name>_masks.tif`` (candidate masks), ``<name>_certified.tif``
(certified only) and ``<name>_cells.csv`` (features, score, p-value and decision of every cell).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from thin2seg.config import data_settings, eval_settings, section, setup_logging
from thin2seg.data.budget import ReferenceTable, parse_budgets
from thin2seg.data.io import load_counts, read_manifest
from thin2seg.qc.cells import SCORES, LearnedScore, add_features, cell_table, photon_level
from thin2seg.qc.conformal import bh_select, conformal_pvalues, ltt_threshold

logger = logging.getLogger(__name__)


class Calibrator:
    """Calibration data per photon level.
    ``levels[label]`` holds the calibration scores ``r_cal`` and outcomes ``y_cal``,
    the typical test-time photon level (``qc.cells.photon_level``), the largest 
    calibration flow error ``flow_error_fill``, the learned-score parameters and 
    the per-acquisition threshold``pac_threshold``."""

    def __init__(self, meta: dict, levels: dict):
        self.meta, self.levels = meta, levels

    def save(self, path: str | Path):
        arrays = {"meta": json.dumps(self.meta)}
        for label, lv in self.levels.items():
            for k, v in lv.items():
                arrays[f"{label}/{k}"] = v
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        z = np.load(path, allow_pickle=False)
        levels: dict = {}
        for key in z.files:
            if key == "meta":
                continue
            label, k = key.split("/", 1)
            levels.setdefault(label, {})[k] = z[key]
        return cls(json.loads(str(z["meta"])), levels)

    def nearest(self, level: float) -> str:
        """The calibrated photon level closest to ``level``."""
        return min(self.levels, key=lambda b: abs(np.log(max(float(self.levels[b]["level"]), 1e-6))
                                                  - np.log(max(level, 1e-6))))

    def score(self, cells: pd.DataFrame, label: str) -> np.ndarray:
        lv, kind = self.levels[label], self.meta["score"]
        if kind == "flow":
            return -cells["flow_error"].to_numpy()
        if kind == "cellprob":
            return cells["cellprob"].to_numpy()
        return LearnedScore(lv["mean"], lv["scale"], lv["coef"], lv["intercept"])(cells)

    def certify(self, cells: pd.DataFrame, label: str, alpha: float) -> pd.DataFrame:
        """Certify ``cells`` at the given ``alpha``."""
        lv = self.levels[label]
        cells = add_features(cells.copy(), float(lv["flow_error_fill"]))
        r = self.score(cells, label)
        cells["score"] = r
        cells["p_value"] = conformal_pvalues(lv["r_cal"], lv["y_cal"], r)
        cells["certified"] = bh_select(cells["p_value"].to_numpy(), alpha)
        if "pac_threshold" in lv:
            cells["certified_per_acquisition"] = r >= lv["pac_threshold"]
        return cells


def fit_calibrator(d: pd.DataFrame, labels: list[str], tau: float, score: str, seed: int,
                   image_guarantee: dict | None = None, meta: dict | None = None) -> Calibrator:
    """Fit a :class:`Calibrator` using the per-cell tables ``d`` of a calibration pool"""
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    d = add_features(d.copy())
    names = np.array(sorted(set(d["name"])))
    perm = np.random.default_rng(seed).permutation(names)
    fit_n = set(perm[: len(perm) // 2]) if score == "learned" else set()
    if score == "learned" and (len(fit_n) == 0 or len(fit_n) == len(names)):
        raise ValueError("the learned score needs at least two calibration recordings")
    levels = {}
    for label in labels:
        g = d[d["budget"] == label]
        if g.empty:
            raise ValueError(f"no calibration cells at budget {label}")
        fit, cal = g[g["name"].isin(fit_n)], g[~g["name"].isin(fit_n)]
        lv = {"level": np.median([photon_level(a) for _, a in g.groupby(["name", "draw"])]),
              "flow_error_fill": g["flow_error"].max()}
        if score == "learned":
            ls = LearnedScore.fit(fit, (fit["iou_ref"] >= tau).to_numpy())
            lv.update(mean=ls.mean, scale=ls.scale, coef=ls.coef, intercept=ls.intercept)
        cal_obj = Calibrator({"score": score}, {label: lv})
        r_cal = cal_obj.score(cal, label)
        y_cal = (cal["iou_ref"] >= tau).to_numpy()
        lv.update(r_cal=r_cal, y_cal=y_cal)
        if image_guarantee:
            ig = image_guarantee
            groups = [(r_cal[ix], y_cal[ix])
                      for ix in cal.reset_index(drop=True).groupby(["name", "draw"]).indices.values()]
            grid = cal_obj.score(fit, label) if score == "learned" else None
            lv["pac_threshold"] = ltt_threshold(groups, float(ig["alpha"]), float(ig["delta"]), float(ig["eps"]),
                                                grid_scores=grid)
        levels[label] = lv
        logger.info(f"  {label}: {len(cal)} calibration cells from {cal['name'].nunique()} recordings, "
                    f"{y_cal.mean():.1%} stable, typical level {lv['level']:.2f} photons per cell pixel")
    return Calibrator({"score": score, "tau": tau, "seed": seed, "image_guarantee": image_guarantee,
                       **(meta or {})}, levels)


def calibrate(cfg: dict, data: dict, ecfg: dict):
    import torch

    from thin2seg.model.cellpose import get_device, load_model
    from thin2seg.qc.certification import collect_recording

    out = Path(cfg["out_dir"])
    pool_csv = cfg.get("pool_csv") or data.get("pool_csv")
    if not pool_csv:
        raise KeyError("certify (calibrate) needs pool_csv: full recordings the model was not trained on")
    pool = read_manifest(pool_csv)
    if cfg.get("max_recordings"):
        pool = pool.head(int(cfg["max_recordings"]))
    budgets = parse_budgets(cfg.get("budgets", [2, 1, 0.5]), data["budget_mode"])
    if any(b.is_full for b in budgets):
        raise ValueError("calibration budgets must be below the full recording")
    refs = ReferenceTable(data.get("reference_csv"))
    draws = int(cfg.get("draws", 2))
    cdir = out / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in pool.iterrows() if not (cdir / f"{r['name']}.csv").exists()]
    if todo:
        device = get_device()
        model = load_model(cfg["ckpt"])
        for i, r in enumerate(todo):
            full = load_counts(r["image_path"], int(r["channel"]), data.get("counts_dir"))
            t = collect_recording(model, r["name"], full, refs.get(r["name"], full), budgets, draws, ecfg, device)
            t.to_csv(cdir / f"{r['name']}.csv", index=False)
            logger.info(f"  {r['name']} [{i + 1}/{len(todo)}] {len(t)} cells")
        del model
        torch.cuda.empty_cache()
    tables = [pd.read_csv(cdir / f"{n}.csv") for n in pool["name"] if (cdir / f"{n}.csv").stat().st_size > 2]
    d = pd.concat([t for t in tables if len(t)], ignore_index=True)
    cal = fit_calibrator(d, [b.label for b in budgets], float(cfg.get("tau", 0.5)), cfg.get("score", "learned"),
                         int(cfg.get("seed", 0)), cfg.get("image_guarantee"),
                         meta={"ckpt": str(cfg["ckpt"]), "eval": ecfg, "n_recordings": d["name"].nunique()})
    path = Path(cfg.get("calibrator") or out / "calibrator.npz")
    cal.save(path)
    logger.info(f"calibrator -> {path}")


def apply(cfg: dict, data: dict, ecfg: dict):
    import tifffile

    from thin2seg.model.cellpose import get_device, load_model, predict

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cal = Calibrator.load(cfg["calibrator"])
    alpha = float(cfg.get("alpha", 0.1))
    want = str(cfg.get("budget", "auto"))
    if want != "auto" and want not in cal.levels:
        raise KeyError(f"budget {want!r} is not calibrated; available: {list(cal.levels)}")
    if cal.meta.get("eval") and {k: cal.meta["eval"].get(k) for k in ("cellprob_threshold", "min_size")} != \
            {k: ecfg.get(k) for k in ("cellprob_threshold", "min_size")}:
        logger.warning(f"mask settings differ from calibration ({cal.meta['eval']} vs {ecfg}); "
                       f"the guarantee assumes the same settings")
    device = get_device()
    model = load_model(cfg["ckpt"])
    rows = []
    for _, r in read_manifest(cfg["images_csv"]).iterrows():
        counts = load_counts(r["image_path"], int(r["channel"]), data.get("counts_dir"))
        lab, dP, cp = predict(model, [counts], {**ecfg, "flow_threshold": 0.0})[0]
        cells = cell_table(lab, dP, cp, counts, device=device)
        certified = np.zeros_like(lab)
        rec = {"name": r["name"], "n_cells": 0, "n_certified": 0, "photon_level": np.nan, "calibrated_budget": ""}
        if len(cells):
            level = photon_level(cells)
            label = cal.nearest(level) if want == "auto" else want
            cells = cal.certify(cells, label, alpha)
            keep = cells.loc[cells["certified"], "cell"].to_numpy()
            certified = np.where(np.isin(lab, keep), lab, 0)
            rec.update(n_cells=len(cells), n_certified=cells["certified"].sum(), photon_level=level,
                       calibrated_budget=label)
            lo, hi = min(float(v["level"]) for v in cal.levels.values()), max(float(v["level"]) for v in cal.levels.values())
            if want == "auto" and not (0.5 * lo <= level <= 2.0 * hi):
                logger.warning(f"{r['name']}: photon level {level:.2f} is outside the calibrated range "
                               f"({lo:.2f}-{hi:.2f}); recalibrate with matching budgets")
            cells.to_csv(out / f"{r['name']}_cells.csv", index=False)
        tifffile.imwrite(out / f"{r['name']}_masks.tif", lab, compression="zlib")
        tifffile.imwrite(out / f"{r['name']}_certified.tif", certified, compression="zlib")
        rows.append(rec)
        logger.info(f"{r['name']}: {rec['n_certified']}/{rec['n_cells']} cells certified at FDR {alpha:g} "
                    f"(level {rec['photon_level']:.2f}, calibrated at {rec['calibrated_budget']})")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


def main(params: dict):
    cfg = section(params, "certify")
    data, ecfg = data_settings(params), eval_settings(params)
    setup_logging(cfg["out_dir"], "certify.log")
    mode = cfg.get("mode")
    if mode == "calibrate":
        calibrate(cfg, data, ecfg)
    elif mode == "apply":
        apply(cfg, data, ecfg)
    else:
        raise ValueError(f"certify.mode must be 'calibrate' or 'apply', got {mode!r}")
