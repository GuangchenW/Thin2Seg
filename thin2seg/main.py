"""Task entry point for Thin2Seg. Config required.

    Example usage: python -m thin2seg --config configs/train_thin2seg.yaml

The config defines its ``task`` and holds the corresponding section with the task's settings.
Each task module documents its section in its docstring.
"""

from __future__ import annotations

import argparse
import importlib

import yaml

#: task -> module responsible
TASKS = {
    "photon_reference": "thin2seg.data.reference",      # per-image photon levels, needed for absolute budgets
    "split_pool": "thin2seg.data.split",                # pool -> training half and calibration half
    "train": "thin2seg.train.run",                      # Thin2Seg, its variants and the fine-tuning baselines
    "evaluate": "thin2seg.eval.evaluate",               # evaluate F1 across photon budgets against annotated images
    "predict": "thin2seg.predict",                      # predict masks for images
    "error_budget": "thin2seg.eval.error_budget",       # model error vs photon-induced error
    "certification": "thin2seg.qc.certification",       # FDR and power of label-free certification
    "certify": "thin2seg.qc.certify",                   # calibrate once, then certify new acquisitions
    "compare_overlays": "thin2seg.visual.compare",      # generate models x budgets figure for one image
    "plot_curves": "thin2seg.visual.curves",            # generate metrics against budget figure
}


def run(argv=None):
    assert __doc__ is not None
    parser = argparse.ArgumentParser(prog="thin2seg", description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path of a YAML config file")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    task = (params or {}).get("task")
    if task not in TASKS:
        raise SystemExit(f"Config {args.config} has unknown task {task!r}!\n"
                         f"Valid tasks: {', '.join(TASKS)}")
    if params.get("device"):
        from thin2seg.model.cellpose import set_device
        set_device(params["device"])
    importlib.import_module(TASKS[task]).main(params)


if __name__ == "__main__":
    run()
