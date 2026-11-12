"""Set up a Cellpose-SAM model with extra harness.

The network input is the photon-count image under a 1st-99th percentile scaling, in channel 0 of
a three-channel image (the other two are zero). The scaling is scale-free, so a thinned copy
needs no rescaling by its thinning rate.
"""

from __future__ import annotations

import numpy as np
import torch
import fastremap
from cellpose import dynamics


_DEVICE = "auto"


def set_device(name: str | None):
    """Sets the device for tasks: ``auto`` (default, GPU if there is one), ``cuda`` or ``cpu``."""
    global _DEVICE
    name = "auto" if name is None else str(name).lower()
    if name not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested but no CUDA device is available")
    _DEVICE = name


def get_device() -> torch.device:
    if _DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(_DEVICE)


def load_model(pretrained_model: str = "cpsam_v2", train: bool = False):
    """Returns ``cellpose.models.CellposeModel``. ``pretrained_model`` is a built-in name or the path of
    saved weights. Inference on a GPU uses bfloat16 weights, the Cellpose default. ``train=True``
    loads float32 weights, which fine-tuning needs, also when using CPU since bfloat16 is very slow for CPU."""
    from cellpose import models

    device = get_device()
    return models.CellposeModel(
        gpu=device.type == "cuda", device=device,
        pretrained_model=str(pretrained_model),
        use_bfloat16=not train and device.type == "cuda"
    )


def network_input(counts: np.ndarray) -> np.ndarray:
    """Photon counts [Y, X] -> float32 input [3, Y, X]."""
    x = np.asarray(counts, np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    out = np.zeros((3, *x.shape), np.float32)
    out[0] = (x - lo) / max(hi - lo, 1e-6)
    return out


def predict(model, images: list[np.ndarray], ecfg: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Segment photon-count images; returns a list of (instance labels [Y, X] int32, flow field
    [2, Y, X], cell-probability logits [Y, X]). ``ecfg`` is ``config.eval_settings``."""
    out = []
    for im in images:
        labels, flows, _ = model.eval(network_input(im), channel_axis=0, normalize=False,
                                      batch_size=int(ecfg.get("batch_size", 8)),
                                      flow_threshold=float(ecfg.get("flow_threshold", 0.8)),
                                      cellprob_threshold=float(ecfg.get("cellprob_threshold", 0.0)),
                                      min_size=int(ecfg.get("min_size", 15)))
        out.append((labels.astype(np.int32), np.asarray(flows[1], np.float32),
                    np.asarray(flows[2], np.float32)))
    return out


def flow_errors(labels: np.ndarray, dP: np.ndarray, device=None) -> dict[int, float]:
    """Cellpose's flow error of every mask: the mean squared difference between the predicted
    flow and the flow rebuilt from the mask."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return {}
    ren, remap = fastremap.renumber(labels.astype(np.int32).copy(), in_place=False)
    inv = {v: k for k, v in remap.items()}
    err, _ = dynamics.flow_error(ren, dP, device=device) # type: ignore
    return {int(inv[j + 1]): float(e) for j, e in enumerate(err)}


def masks_for_thresholds(dP, cellprob, thresholds, flow_threshold: float, min_size: int, device) -> dict:
    """Masks for each cell-probability threshold from one network output, as ``model.eval`` would compute."""
    return {float(c): dynamics.resize_and_compute_masks(dP, cellprob, niter=200, cellprob_threshold=float(c),
                                                        flow_threshold=flow_threshold, min_size=min_size,
                                                        device=device).astype(np.int32)
            for c in thresholds}
