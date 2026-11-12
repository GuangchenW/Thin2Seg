"""Building the photon reference table (``task: photon_reference``).

Absolute photon budgets are stated in median photons per *cell* pixel, so every recording needs
its own median photon count per cell pixel at full photons. Cell pixels come from, in order:

1. the annotation (``data.labeled_csv``), ``source = annotation``;
2. full-photon mask prediction (``ckpt``: any Cellpose-SAM weights, default ``cpsam_v2``), ``source = model``;
3. with ``ckpt: null``, Otsu foreground, ``source = foreground``.

Example definition:

    task: photon_reference
    photon_reference:
      out: data/organoid/photon_reference.csv
      ckpt: cpsam_v2

Run this task once per dataset, before training, and point ``data.reference_csv`` at the result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from thin2seg.config import data_settings, eval_settings, section, setup_logging
from thin2seg.data.budget import proxy_foreground, reference_level
from thin2seg.data.io import load_counts, load_labels, read_manifest

logger = logging.getLogger(__name__)


def main(params: dict):
    cfg = section(params, "photon_reference")
    data, ecfg = data_settings(params), eval_settings(params)
    out = Path(cfg["out"])
    setup_logging(out.parent, "photon_reference.log")
    cache = data.get("counts_dir")
    model = None
    rows, seen = [], set()
    manifests = []
    if data.get("labeled_csv"):
        manifests.append(read_manifest(data["labeled_csv"], labeled=True))
    if data.get("pool_csv"):
        manifests.append(read_manifest(data["pool_csv"]))
    for extra in cfg.get("extra_csv") or []:
        manifests.append(read_manifest(extra))
    for df in manifests:
        for _, row in df.iterrows():
            name = row["name"]
            if name in seen:
                continue
            seen.add(name)
            counts = load_counts(row["image_path"], int(row["channel"]), cache)
            gt_path = row.get("instance_map_path")
            if isinstance(gt_path, str) and gt_path:
                cells, source = load_labels(gt_path) > 0, "annotation"
            elif cfg.get("ckpt", "cpsam_v2"):
                from thin2seg.model.cellpose import load_model, predict
                if model is None:
                    model = load_model(cfg.get("ckpt", "cpsam_v2"))
                cells, source = predict(model, [counts], ecfg)[0][0] > 0, "model"
                if not cells.any():
                    cells, source = proxy_foreground(counts), "foreground"
            else:
                cells, source = proxy_foreground(counts), "foreground"
            rows.append({"name": name, "ref_median": reference_level(counts, cells),
                         "ref_mean": float(counts[cells].mean()), "n_cell_px": int(cells.sum()), "source": source})
            logger.info(f"{name:<32} {source:<11} median {rows[-1]['ref_median']:6.1f} photons per cell pixel")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info(
        f"{len(df)} recordings -> {out};\n"
        f"sources {df['source'].value_counts().to_dict()};\n"
        f"median of medians {df['ref_median'].median():.2f}"
    )
