"""Read FLIM recording, annotations and manifests.

A *recording* is reduced to its photon-count image, i.e. the number of photons detected in every
pixel, summed over the delay-time bins.

Supported files:

* ``.ptu``, read with ``ptufile``;
* ``.tif`` / ``.tiff`` / ``.npy`` holding photon counts, either ``[Y, X]`` or ``[H, Y, X]`` with
  the delay-time (or frame) axis first, which is summed.

A *manifest* is a CSV with one row per recording: ``image_path``, optionally ``channel``
(0-based indexing, default 0) and, for annotated recordings, ``instance_map_path``
(label TIFF, 0 = background). Samples are referred to by the stem of ``image_path``.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from phasorpy.io import signal_from_ptu
from ptufile import PtuFile
import tifffile

# Upper bound on delay-time bins read from a PTU. Files with more bins per laser period are
# rebinned while decoding. This is to prevent memory problems when dealing with multi-frame
# FLIM recordings with man time bins, which is often sparse and consumes too much memory if
# loaded as-is. The photon-count image does not depend on it, apart from the few photons 
# past the period end that the last (wider) bin collects.
DEFAULT_MAX_BINS = 128


def stem(path: str|Path) -> str:
    return PureWindowsPath(path).stem


def _bin_factor(n_bins: int, max_bins: int) -> int:
    """Smallest integer factor that brings ``n_bins`` down to at most ``max_bins`. 0 disables."""
    if max_bins < 0:
        raise ValueError(f"max_bins must be 0 (no rebinning) or >= 1, got {max_bins}")
    if max_bins == 0:
        return 1
    return -(-n_bins // max_bins)


def _signal_from_ptu(path: Path, channel: int, max_bins: int):
    """Returns frame-integrated TCSPC signal [Y, X, H] over one laser period, rebinned to <= max_bins."""
    with PtuFile(path) as f:
        n = f.number_bins_in_period
    k = _bin_factor(n, max_bins)
    if k == 1:
        return signal_from_ptu(path, frame=-1, channel=channel)
    else:
        return signal_from_ptu(path, frame=-1, channel=channel, dtime=None, dtype=np.uint32,
                               selection=(..., slice(0, n, k)))


def read_histogram(path: str|Path, channel: int = 0, max_bins: int = DEFAULT_MAX_BINS) -> np.ndarray:
    """Returns photons histogram [H, Y, X] of a PTU recording (all frames summed)."""
    _bin_factor(1, max_bins)
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    if path.suffix.lower() != ".ptu":
        raise ValueError(f"{path}: decay histograms are read from .ptu files only")
    return _signal_from_ptu(path, channel, max_bins).transpose("H", "Y", "X").to_numpy()


def read_counts(path: str|Path, channel: int = 0) -> np.ndarray:
    """Returns photon-count image [Y, X] (int32) of one recording."""
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".ptu":
        arr = np.asarray(read_histogram(path, channel), np.int32).sum(0)
    elif suffix in (".tif", ".tiff", ".npy"):
        arr = np.load(path) if suffix == ".npy" else tifffile.imread(path)
        if arr.ndim == 3: arr = arr.sum(0) # Sum along the H axis
        if arr.ndim != 2:
            raise ValueError(f"{path}: expected photon counts [Y, X] or [H, Y, X], got shape {arr.shape}")
        if (arr < 0).any() or not np.allclose(arr, np.round(arr)):
            raise ValueError(f"{path}: photon counts must be non-negative integers "
                             f"(thinning draws from the counts themselves)")
    else:
        raise ValueError(f"{path}: unsupported format {suffix!r} (use .ptu, .tif or .npy)")
    return np.asarray(arr).astype(np.int32)


def load_counts(path: str | Path, channel: int = 0, cache_dir: str | Path | None = None) -> np.ndarray:
    """Load counts from cache. If no cache exists, apply :func:`read_counts` and cached as 
    ``<cache_dir>/<stem>_ch<channel>_counts.npy`` when a cache directory is given."""
    if cache_dir is None: return read_counts(path, channel)
    f = Path(cache_dir) / f"{stem(path)}_ch{channel}_counts.npy"
    if f.exists(): return np.load(f)
    counts = read_counts(path, channel)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, counts)
    return counts


def read_frames(path: str | Path, channel: int = 0) -> np.ndarray:
    """Returns per-frame photon counts [T, Y, X] of a multi-frame PTU recording.
    each frame represents a genuine shorter acquisition of the same field."""
    with PtuFile(path) as f:
        img = np.asarray(f.decode_image(channel=channel, dtime=-1, dtype=np.uint32))
    return img.reshape(img.shape[0], img.shape[1], img.shape[2])


def load_labels(path: str | Path) -> np.ndarray:
    """Returns Instance map [Y, X] (int32, consecutive ids from 1) from a label TIFF.
    Multi-page files give their first non-empty page."""
    from skimage.segmentation import relabel_sequential

    arr = tifffile.imread(path)
    while arr.ndim > 2:
        nz = [i for i in range(arr.shape[0]) if (arr[i] > 0).any()]
        arr = arr[nz[0]] if nz else arr[0]
    return relabel_sequential(arr.astype(np.int32))[0].astype(np.int32)


def read_manifest(csv: str | Path, labeled: bool = False) -> pd.DataFrame:
    """A manifest as a table with ``name`` (stem), ``image_path``, ``channel`` and, if
    ``labeled``, ``instance_map_path``. Names must be unique."""
    df = pd.read_csv(csv)
    if "image_path" not in df:
        raise ValueError(f"{csv}: a manifest needs an image_path column")
    if "channel" not in df: df["channel"] = 0 # Default channel 0
    df["channel"] = df["channel"].fillna(0).astype(int)
    df["name"] = df["image_path"].map(stem)
    dup = df["name"][df["name"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{csv}: recordings must have unique file stems, repeated: {dup}")
    if labeled and "instance_map_path" not in df:
        raise ValueError(f"{csv}: an annotated manifest needs an instance_map_path column")
    cols = ["name", "image_path", "channel"] + (["instance_map_path"] if "instance_map_path" in df else [])
    return df[cols].reset_index(drop=True)
