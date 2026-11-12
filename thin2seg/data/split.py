"""Splitting the unannotated pool in two (``task: split_pool``).

Thin2Seg trains on one part of the unannotated samples because certification must be calibrated on
samples the model has not seen. This task writes the two manifests.

Example definition:

    task: split_pool
    split_pool:
      pool_csv: data/organoid/unlabeled.csv
      train_csv: data/organoid/pool_train.csv
      calibration_csv: data/organoid/pool_calibration.csv
      train_fraction: 0.5
      seed: 0

Recordings listed in ``data.labeled_csv`` are left out of both parts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thin2seg.config import data_settings, section
from thin2seg.data.io import read_manifest, stem


def main(params: dict):
    cfg = section(params, "split_pool")
    data = data_settings(params)
    df = pd.read_csv(cfg["pool_csv"])
    read_manifest(cfg["pool_csv"])  # validates columns and unique names
    names = df["image_path"].map(stem)
    if data.get("labeled_csv"):
        df = df[~names.isin(set(read_manifest(data["labeled_csv"])["name"]))]
    perm = np.random.default_rng(int(cfg.get("seed", 0))).permutation(len(df))
    k = round(float(cfg.get("train_fraction", 0.5)) * len(df))
    for key, idx in (("train_csv", np.sort(perm[:k])), ("calibration_csv", np.sort(perm[k:]))):
        Path(cfg[key]).parent.mkdir(parents=True, exist_ok=True)
        df.iloc[idx].to_csv(cfg[key], index=False)
        print(f"{len(idx)} recordings -> {cfg[key]}")
