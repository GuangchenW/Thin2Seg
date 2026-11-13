"""Training Thin2Seg or its variants or the fine-tuning baselines (``task: train``).

Every variant starts from the same Cellpose-SAM weights and runs the same number of optimiser
steps with the same augmentation.

Variants differ in what they train on:

``feature_target``  ``null``    segmentation loss only, used for a fine-tuning baseline
                    ``ema``     Thin2Seg: features of each thinned copy are trained towards the
                                features of the full photon recording under an EMA of the network
                    ``frozen``  the same, but against a frozen ``teacher`` network (distillation)
``pseudo_labels``   if ``true``, add the ``teacher``'s filtered full-photon masks as labels of the
                    unannotated pool (``train.pseudo_labels``)
``noise_model``     ``binomial`` (photon thinning, default) or ``gaussian`` (ablation only)
``train_budgets``   photon budgets of the training copies (``[full]`` gives plain fine-tuning)
``labeled_repeat``  oversampling of annotated images: an integer, or ``balanced`` to
                    give the annotated images as many training items as the unlabelled pool
``teacher``         required when using a teacher network: the name of another variant of this run
                    (checkpoint for the same fold), or a path that contains ``{fold}``

The unannotated pool (``data.pool_csv``) is used by variants with a feature target or
pseudo-labels. With a feature target and no pseudo-labels, pool images carry the feature term
only.

A *fold* names the annotated images to train on; the model is then evaluated on the other
annotated images (or on ``test``) at every evaluation budget (``eval.evaluate``).

Example task definition:

    task: train
    train:
      out_dir: runs/organoid
      folds:
        A: {train: [image_stem]}
      variants:
        - {name: thin2seg, feature_target: ema, labeled_repeat: balanced}

Writes to ``out_dir``: ``<variant>/fold<fold>/<variant>_fold<fold>`` (the model weights), 
``spec.json`` and ``history.json`` next to it, ``eval.csv`` (one row per variant, fold, image, budget and draw)
and ``summary.csv``. Finished variants are skipped, so an interrupted run can be restarted.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thin2seg.config import data_settings, eval_settings, pretrained_model, section, setup_logging
from thin2seg.data.budget import ReferenceTable, parse_budgets
from thin2seg.data.io import load_counts, load_labels, read_manifest
from thin2seg.data.thinning import NOISE_MODELS, training_copy
from thin2seg.eval.evaluate import evaluate_image, load_annotated, summarise
from thin2seg.model.cellpose import get_device, load_model, network_input
from thin2seg.train.trainer import TrainItem, finetune

logger = logging.getLogger(__name__)

FEATURE_TARGETS = (None, "ema", "frozen")
DEFAULT_TRAIN_BUDGETS = ["full", 8, 4, 2, 1, 0.5]


def variant_spec(v: dict) -> dict:
    """Validates and returns a variant spec definition."""
    spec = {"feature_target": None, "pseudo_labels": False, "noise_model": "binomial",
            "train_budgets": None, "labeled_repeat": 1, "teacher": None, **v}
    if "name" not in spec:
        raise ValueError(f"a variant needs a name: {v}")
    if spec["feature_target"] not in FEATURE_TARGETS:
        raise ValueError(f"{spec['name']}: feature_target must be one of {FEATURE_TARGETS}")
    if spec["noise_model"] not in NOISE_MODELS:
        raise ValueError(f"{spec['name']}: noise_model must be one of {NOISE_MODELS}")
    if (spec["feature_target"] == "frozen" or spec["pseudo_labels"]) and not spec["teacher"]:
        raise ValueError(f"{spec['name']}: frozen target and pseudo-labels need a teacher")
    if spec["labeled_repeat"] != "balanced" and int(spec["labeled_repeat"]) < 1:
        raise ValueError(f"{spec['name']}: labeled_repeat must be 'balanced' or an integer >= 1")
    spec["uses_pool"] = spec["feature_target"] is not None or bool(spec["pseudo_labels"])
    return spec


def checkpoint_path(out: Path, variant: str, fold) -> Path:
    return out / variant / f"fold{fold}" / f"{variant}_fold{fold}"


def teacher_path(spec: dict, out: Path, fold, variant_names: list[str]) -> Path:
    """The teacher checkpoint of ``spec`` for ``fold``."""
    t = str(spec["teacher"])
    path = checkpoint_path(out, t, fold) if t in variant_names else Path(t.format(fold=fold))
    if not path.exists():
        raise FileNotFoundError(f"{spec['name']}: teacher checkpoint {path} not found"
                                + (f" (list variant '{t}' before '{spec['name']}')" if t in variant_names else ""))
    return path


def repeat_annotated(labelled: list, pool: list, repeat: str|int) -> list:
    """Set up training samples: the annotated images repeated ``k`` times, then the unlabelled pool.
    ``balanced`` picks ``k`` so that the annotated images have about as many items as the pool."""
    k = max(1, round(len(pool) / max(len(labelled), 1))) if repeat == "balanced" else int(repeat)
    return labelled * k + pool


def build_items(sources: list, budgets, refs: ReferenceTable, fold, noise: str = "binomial",
                with_target: bool = False) -> tuple[list[TrainItem], int]:
    """Build a list of TrainItem for each source over the training budgets. Also returns 
    the number of pairs skipped because the image is already dimmer than the budget 
    (these pairs would duplicate the full-photon source so they should not be included).

    ``sources``: (name, counts, labels or None, ignore mask or None).
    ``with_target`` attaches additional data for the feature term: the full-photon network input and 
    the photon levels (log1p photons per cell pixel) of the thinned copy and of the full recording.
    """
    items, n_capped = [], 0
    for name, c, labels, ign in sources:
        ref = refs.get(name, c)
        ref_img = network_input(c) if with_target else None
        for b in budgets:
            if b.capped(ref):
                n_capped += 1
                continue
            x = training_copy(c, b, ref if b.mode == "absolute" else None, (name, fold), noise=noise)
            bud = None
            if with_target:
                kept = x.sum() / max(c.sum(), 1.0)
                bud = (np.log1p(ref * kept), np.log1p(ref))
            items.append(TrainItem(network_input(x), labels, ign, f"{name}_{b.label}", ref_img, bud))
    return items, n_capped


def pool_sources(spec: dict, pool: list, teacher_ckpt: Path | None, ecfg: dict, cache_dir: Path,
                 tau: float, device) -> list:
    """Build the unlablled pool. With pseudo-labels, each image is paired with 
    the teacher's filtered masks (cached under ``cache_dir``)."""
    if not spec["pseudo_labels"]:
        return [(name, c, None, None) for name, c in pool]
    from thin2seg.train.pseudo_labels import teacher_labels

    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [name for name, _ in pool if not (cache_dir / f"{name}.npz").exists()]
    teacher = load_model(teacher_ckpt) if todo else None
    out, n_bad, n_all = [], 0, 0
    for name, c in pool:
        f = cache_dir / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            lbl, ign, nb, na = z["labels"], z["ignore"], int(z["n_bad"]), int(z["n_all"])
        else:
            lbl, ign, nb, na = teacher_labels(teacher, c, ecfg, name, device, tau)
            np.savez_compressed(f, labels=lbl, ignore=ign, n_bad=nb, n_all=na)
        out.append((name, c, lbl, ign))
        n_bad, n_all = n_bad + nb, n_all + na
    del teacher
    torch.cuda.empty_cache()
    logger.info(f"  pseudo-labels for {len(out)} pool images: {n_bad}/{n_all} teacher cells ignored as unreliable")
    return out


def main(params: dict):
    cfg = section(params, "train")
    data, ecfg = data_settings(params), eval_settings(params)
    out = Path(cfg["out_dir"])
    setup_logging(out, "train.log")
    device = get_device()
    mode = data["budget_mode"]
    cache = data.get("counts_dir")
    seed = int(cfg.get("seed", 42))
    n_epochs, steps = int(cfg.get("n_epochs", 100)), int(cfg.get("steps_per_epoch", 31))
    eval_budgets = parse_budgets(cfg.get("eval_budgets", ["full", 2, 1, 0.5]), mode)
    draws = int(cfg.get("eval_draws", 3))
    variants = [variant_spec(v) for v in cfg["variants"]]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"variant names must be unique: {names}")

    refs = ReferenceTable(data.get("reference_csv"))
    lab = read_manifest(data["labeled_csv"], labeled=True)
    pool_rows = None
    if any(v["uses_pool"] for v in variants):
        if not data.get("pool_csv"):
            raise KeyError("variants with a feature target or pseudo-labels need data.pool_csv")
        pool_rows = read_manifest(data["pool_csv"])
        pool_rows = pool_rows[~pool_rows["name"].isin(set(lab["name"]))]   # never treat an annotated image as unlabelled
        if cfg.get("max_pool"):
            pool_rows = pool_rows.head(int(cfg["max_pool"]))
    logger.info(f"device {device}; {len(lab)} annotated images, "
                f"{0 if pool_rows is None else len(pool_rows)} pool images; variants {names}")

    res_f = out / "eval.csv"
    results = pd.read_csv(res_f).to_dict("records") if res_f.exists() else []
    done = {(str(r["fold"]), r["variant"]) for r in results}
    pool = None # (name, counts), loaded once

    for fold, fcfg in cfg["folds"].items():
        fold = str(fold)
        train_names = list(fcfg["train"])
        unknown = set(train_names) - set(lab["name"])
        if unknown:
            raise KeyError(f"fold {fold}: training images not in the annotated manifest: {unknown}")
        test_names = list(fcfg.get("test") or [n for n in lab["name"] if n not in train_names])
        if set(test_names) & set(train_names):
            raise ValueError(f"fold {fold}: an image is both trained on and tested")
        labelled = [(n, c, gt, None) for n, c, gt in load_annotated(lab, train_names, cache, refs)]
        tests = load_annotated(lab, test_names, cache, refs) if cfg.get("evaluate", True) else []
        logger.info(f"fold {fold}: train on {train_names}, test on {len(tests)} annotated images")

        for spec in variants:
            name = spec["name"]
            ckpt = checkpoint_path(out, name, fold)
            if not ckpt.exists():
                if spec["uses_pool"] and pool is None:
                    pool = [(r["name"], load_counts(r["image_path"], int(r["channel"]), cache))
                            for _, r in pool_rows.iterrows()]
                t_ckpt = teacher_path(spec, out, fold, names) if spec["teacher"] else None
                psrc = []
                if spec["uses_pool"]:
                    psrc = pool_sources(spec, pool, t_ckpt, ecfg, out / "pseudo_labels" / f"fold{fold}" / str(spec["teacher"]),
                                        float(cfg.get("pseudo_label_tau", 0.3)), device)
                sources = repeat_annotated(labelled, psrc, spec["labeled_repeat"])
                budgets = parse_budgets(spec["train_budgets"] or cfg.get("train_budgets", DEFAULT_TRAIN_BUDGETS), mode)
                with_target = spec["feature_target"] is not None
                items, n_capped = build_items(sources, budgets, refs, fold, noise=spec["noise_model"],
                                              with_target=with_target)
                n_sup = sum(it.supervised for it in items)
                logger.info(f"fold {fold} {name}: {len(items)} training items ({n_sup} with labels)"
                            + (f", {n_capped} (image, budget) pairs skipped: image dimmer than the budget" if n_capped else ""))
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = load_model(pretrained_model(params), train=True)
                target = None
                if spec["feature_target"] == "ema":
                    target = "ema"
                elif spec["feature_target"] == "frozen":
                    target = load_model(t_ckpt, train=True).net
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                json.dump({**spec, "fold": fold, "seed": seed, "train_images": train_names, "n_items": len(items),
                           "n_epochs": n_epochs, "steps_per_epoch": steps, "train_budgets": [b.label for b in budgets],
                           "teacher_ckpt": str(t_ckpt) if t_ckpt else None, "feature": cfg.get("feature"),
                           "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                          open(ckpt.parent / "spec.json", "w"), indent=1)
                t0, hist = time.time(), {}
                _, losses = finetune(model.net, items, ckpt, n_epochs=n_epochs,
                                     learning_rate=float(cfg.get("learning_rate", 1e-5)),
                                     weight_decay=float(cfg.get("weight_decay", 0.1)),
                                     steps_per_epoch=steps, device=device, target=target,
                                     feature=cfg.get("feature"), history=hist)
                json.dump({"loss": losses, **hist, "train_s": round(time.time() - t0)},
                          open(ckpt.parent / "history.json", "w"))
                logger.info(f"fold {fold} {name}: trained in {time.time() - t0:.0f}s -> {ckpt}")
                del model, target, items
                torch.cuda.empty_cache()
            if tests and (fold, name) not in done:
                model = load_model(ckpt)
                sd = out / "masks" / f"fold{fold}" / name if cfg.get("save_masks") else None
                ev = [{"fold": fold, "variant": name, "name": n, **r}
                      for n, counts, gt in tests
                      for r in evaluate_image(model, counts, gt, eval_budgets, draws, ecfg, n, refs, sd)]
                results += ev
                pd.DataFrame(results).to_csv(res_f, index=False)
                f1 = pd.DataFrame(ev).groupby("budget_label", sort=False)["f1_50"].mean().round(3).to_dict()
                logger.info(f"fold {fold} {name}: F1@0.5 by budget {f1}")
                del model
                torch.cuda.empty_cache()
    if results:
        summary = summarise(pd.DataFrame(results), ["variant"])
        summary.to_csv(out / "summary.csv", index=False)
        logger.info("mean over folds and held-out images:\n" + summary.round(3).to_string(index=False))
    logger.info("Done.")
