"""Simulating a shorter acquisition from a FLIM recording using photon thinning.

Each detected photon is kept independently with probability ``p``. This turns the mean of the 
Poisson count, ``lambda``, into ``p * lambda``. A thinned copy therefore has the statistics of a
``p``-fold shorter acquisition and stays pixel-aligned with the full recording.

Every draw is seeded by a key (see :func:`rng_for`), so the same image, budget and purpose always
give the same acquisition.

The ``gaussian`` noise model is for ablation purposes and should not be used in deployment,
because it performs worse than binomial thinning overall.
"""

from __future__ import annotations

import hashlib

import numpy as np

NOISE_MODELS = ("binomial", "gaussian")


def rng_for(*key) -> np.random.Generator:
    """A generator seeded by the hash of ``key``."""
    return np.random.default_rng(int(hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:8], 16))


def disjoint(counts: np.ndarray, fracs: list[float], rng: np.random.Generator) -> list[np.ndarray]:
    """Disjoint sub-acquisitions holding fractions ``fracs`` (sum <= 1) of the photons.

    Sequential binomial thinning is an exact multinomial split. I.e. part j keeps each remaining
    photon with probability f_j / (1 - sum of earlier f). If the recording is Poisson, the
    parts are independent Poisson acquisitions at f_j of its photons. When the fractions sum
    to 1 the last part takes the remainder exactly.
    """
    if sum(fracs) > 1 + 1e-9:
        raise ValueError(f"photon split fractions sum to {sum(fracs)} > 1")
    rem, left, out = counts, 1.0, []
    for j, f in enumerate(fracs):
        if j == len(fracs) - 1:
            part = rem
        else:
            part = rng.binomial(rem, min(f / left, 1.0))
        out.append(part.astype(np.int32))
        rem, left = rem - part, left - f
    return out


def thin(counts: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """One acquisition keeping ``rate`` of the photons (or the recording itself at rate >= 1)."""
    return counts if rate >= 1.0 else disjoint(counts, [rate], rng)[0]


def training_copy(counts: np.ndarray, budget, ref: float | None, key, noise: str = "binomial") -> np.ndarray:
    """The training acquisition of one image at ``budget`` (a :class:`thin2seg.data.budget.Budget`).

    ``ref`` is the median photon count per cell pixel of the image at full photons (needed for
    absolute budgets).
    """
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {noise!r}")
    r = budget.rate(ref)
    if r >= 1.0: return counts
    tag = r if budget.mode == "ratio" else budget.label
    if noise == "gaussian":
        level = r * (ref if ref is not None else float(np.median(counts[counts > 0])))
        rng = rng_for(*key, tag, "train_gauss")
        return (
            counts.astype(np.float32) * r + rng.normal(0.0, np.sqrt(level), counts.shape).astype(np.float32)
        )
    return disjoint(counts, [r], rng_for(*key, tag, "train"))[0]


def evaluation_copies(counts: np.ndarray, budget, rate: float, name: str, draws: int) -> list[np.ndarray]:
    """The evaluation acquisitions of image ``name`` at ``budget``, up to ``draws`` disjoint thinned copies."""
    if rate >= 1.0: return [counts]
    n = min(draws, int(1 / rate))
    tag = rate if budget.mode == "ratio" else budget.label
    return disjoint(counts, [rate] * n, rng_for(name, tag, "eval"))
