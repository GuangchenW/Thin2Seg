"""Model training with a per-pixel loss mask and an optional representation target.

The flow and cell probability loss is averaged over the pixels an item marks as supervised.
An annotated image supervises every pixel and trains exactly as ``train_seg`` would. 
An unannotated image supervises none (or, with pseudo-labels, every pixel except the unreliable cells).

* targets from ``cellpose.dynamics.labels_to_flows``, cell probability ``= dist > 0.5``;
* ``cellpose.transforms.random_rotate_and_resize`` for augmentation (random 256 px crops,
  rotation, flips, rescaling by 0.75-1.25). The loss mask goes through the same;
* loss = MSE(flow, 5 x target flow) / 2 + BCE(cell probability);

With a representation target (``target="ema"`` or a frozen network), every step also runs the
target network on the full-photon copy of the item, with the same augmentation as the thinned copy,
and adds the Thin2Seg feature term (``thin2seg.model.features``):

    loss = L_seg + lambda(epoch) * [2 - 2 cos(q(h(X_b); b, b_X), sg h_target(X))] + w_var * L_var

where ``q`` is the budget-conditioned predictor and ``L_var`` the variance floor on ``h(X_b)``.
An EMA target starts from the initial network and follows it after every optimiser step.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from thin2seg.model.features import (Predictor, channel_std, cosine_loss, ema_momentum, feature_statistics,
                                     forward_features, ramp_weight, variance_floor)

logger = logging.getLogger(__name__)

#: defaults of the feature term (``feature:`` section of a training config)
FEATURE_DEFAULTS = {
    "weight": 1.0,                   # final weight of the cosine term
    "ramp": [10, 20],                # epochs over which that weight rises linearly from 0
    "var_weight": 1.0,               # weight of the variance floor
    "gamma": 0.5,                    # the floor - this fraction of each channel's initial spread
    "hidden": 1024,                  # predictor hidden width
    "lr_mult": 10.0,                 # predictor learning rate, as a multiple of the network
    "ema_schedule": [0.99, 0.999],   # EMA momentum at the first and the last step
}


@dataclass
class TrainItem:
    """One training image.

    ``image``      the [3, H, W] network input (``model.cellpose.network_input``)
    ``labels``     instance map, or None for an image without any supervision
    ``ignore``     True where the segmentation loss ignores (None: every pixel supervised)
    ``ref_image``  input of the target network: the full-photon copy of the same image
    ``budget``     (log1p photons per cell pixel of ``image``, of ``ref_image``), for the predictor
    """
    image: np.ndarray
    labels: np.ndarray | None
    ignore: np.ndarray | None = None
    name: str = ""
    ref_image: np.ndarray | None = None
    budget: tuple | None = None

    @property
    def supervised(self) -> bool:
        return self.labels is not None and not (self.ignore is not None and self.ignore.all())


def lr_schedule(learning_rate: float, n_epochs: int) -> np.ndarray:
    """Per-epoch learning rate. 10-epoch linear warm-up from 0, constant, then (n_epochs > 99)
    for the last 50 epochs halve every 5, or (n_epochs > 300) the last 100 halve every 10."""
    lr = np.linspace(0, learning_rate, 10)
    lr = np.append(lr, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        lr = lr[:-100]
        for _ in range(10):
            lr = np.append(lr, lr[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        lr = lr[:-50]
        for _ in range(10):
            lr = np.append(lr, lr[-1] / 2 * np.ones(5))
    return lr


def masked_seg_loss(lbl: torch.Tensor, y: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    """The Cellpose segmentation loss averaged over pixels.

    lbl [N, 3, h, w] (dist, dY, dX); y [N, 3, h, w] (dY, dX, cell-probability logit);
    keep [N, h, w] in {0, 1} or None (all pixels).
    With keep all ones this equals ``cellpose.train._loss_fn_seg``.
    """
    veci = 5.0 * lbl[:, -2:]
    se = (y[:, -3:-1] - veci) ** 2
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype), reduction="none")
    if keep is None:
        return se.mean() / 2.0 + bce.mean()
    keep = keep.to(y.dtype)
    n = keep.sum().clamp_min(1.0)
    return (se * keep[:, None]).sum() / (2.0 * n) / 2.0 + (bce * keep).sum() / n


def make_targets(labels: np.ndarray, device=None) -> np.ndarray:
    """[3, H, W] (dist, dY, dX) training target of an instance map."""
    from cellpose import dynamics
    return dynamics.labels_to_flows([labels.astype(np.int32)], device=device)[0][1:].astype(np.float32)


def augment(imgs: list[np.ndarray], tgts: list[np.ndarray], ignores: list[np.ndarray | None],
            bsize: int = 256, scale_range: float = 0.5, device=torch.device("cpu")):
    """Augmentation for images and targets, and the same affine map for the ignore
    masks. Returns (X, lbl, keep) tensors; keep is None if no mask is set."""
    from cellpose.transforms import random_rotate_and_resize

    state = np.random.get_state()
    X, lbl, _ = random_rotate_and_resize(imgs, lbls=tgts, bsize=bsize, scale_range=scale_range, device=device)
    if all(ig is None for ig in ignores): return X, lbl, None
    np.random.set_state(state)
    masks = [(np.zeros(t.shape[-2:], np.float32) if ig is None else ig.astype(np.float32))[None]
             for ig, t in zip(ignores, tgts)]
    M, _, _ = random_rotate_and_resize(masks, lbls=None, bsize=bsize, scale_range=scale_range,
                                       device=device)
    # pixels padded in from outside the image are unsupervised too. 
    # Push a ones-image through the same map and drop what it does not cover.
    np.random.set_state(state)
    ones = [np.ones((1, *t.shape[-2:]), np.float32) for t in tgts]
    O, _, _ = random_rotate_and_resize(ones, lbls=None, bsize=bsize, scale_range=scale_range, device=device)
    keep = ((M[:, 0] < 0.5) & (O[:, 0] > 0.5)).float()
    return X, lbl, keep


def finetune(net, items: list[TrainItem], save_path: str | Path, n_epochs: int = 100,
             learning_rate: float = 1e-5, weight_decay: float = 0.1, batch_size: int = 1,
             steps_per_epoch: int | None = None, bsize: int = 256, scale_range: float = 0.5,
             device: torch.device | None = None, target=None, feature: dict | None = None,
             history: dict | None = None) -> tuple[Path, list[float]]:
    """Fine-tune ``net`` in place and save it. Returns (weights path, per-epoch loss).

    ``target``
        None    : trains the segmentation loss only (baselines). 
        ``ema`` : adds the feature term against an exponential moving average of ``net``;
        network : adds it against that frozen network. Items then need ``ref_image`` and 
        ``budget``. ``feature`` overrides :data:`FEATURE_DEFAULTS`.
    
    The predictor is saved next to the weights as ``<save_path>_predictor.pt``.
    ``history`` (dict) records per-epoch ``seg``, ``feat``, ``var``, ``lam``, ``momentum`` and ``probe``.
    """
    device = device or next(net.parameters()).device
    fc = {**FEATURE_DEFAULTS, **(feature or {})}
    use_ema = isinstance(target, str)
    if use_ema and target != "ema":
        raise ValueError(f"target must be None, 'ema' or a network, got {target!r}")
    teacher = copy.deepcopy(net) if use_ema else target
    pred = None
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        missing = [it.name for it in items if it.ref_image is None or it.budget is None]
        if missing:
            raise ValueError(f"the feature term needs ref_image and budget on every item; missing on {missing[:3]}")
        pred = Predictor(hidden=int(fc["hidden"])).to(device)
        buds = [it.budget for it in items]
        refs = [it.ref_image.astype(np.float32) for it in items]

    tgts = [make_targets(it.labels, device) if it.labels is not None
            else np.zeros((3, *it.image.shape[-2:]), np.float32) for it in items]
    imgs = [it.image.astype(np.float32) for it in items]
    ignores = [np.ones(it.image.shape[-2:], bool) if it.labels is None else it.ignore for it in items]
    nimg = len(items)
    steps_per_epoch = nimg if steps_per_epoch is None else int(steps_per_epoch)
    lr = lr_schedule(learning_rate, n_epochs)
    groups = [{"params": list(net.parameters()), "lr_mult": 1.0}]
    if pred is not None:
        groups.append({"params": list(pred.parameters()), "lr_mult": float(fc["lr_mult"])})
    opt = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    s0, probe = None, []
    if teacher is not None:
        s0, probe = _feature_baseline(net, items, imgs, tgts, steps_per_epoch, bsize, scale_range, device)
        logger.info(f"  feature term: {'EMA' if use_ema else 'frozen'} target, predictor "
                    f"{sum(q.numel() for q in pred.parameters())} parameters, initial neck std "
                    f"(median over channels) {float(s0.median()):.4f}, settings {fc}")
    total_steps = n_epochs * ((steps_per_epoch + batch_size - 1) // batch_size)
    step = 0
    losses, seg_l, feat_l = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
    var_l, lam_l, mom_l, probes = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), []
    t0 = time.time()
    for ep in range(n_epochs):
        np.random.seed(ep)
        perm = (np.random.permutation(nimg) if nimg == steps_per_epoch
                else np.random.choice(nimg, size=steps_per_epoch))
        for g in opt.param_groups:
            g["lr"] = lr[ep] * g.get("lr_mult", 1.0)
        lam = float(fc["weight"]) * (ramp_weight(ep, *fc["ramp"]) if fc.get("ramp") else 1.0)
        net.train()
        for k in range(0, steps_per_epoch, batch_size):
            inds = perm[k:k + batch_size]
            if teacher is not None:   # thinned and full-photon inputs as one stack so identical augmentation
                nch = imgs[inds[0]].shape[0]
                X, lbl, keep = augment([np.concatenate([imgs[i], refs[i]], 0) for i in inds],
                                       [tgts[i] for i in inds], [ignores[i] for i in inds],
                                       bsize=bsize, scale_range=scale_range, device=device)
                X, Xt = X[:, :nch], X[:, nch:]
                with torch.no_grad():
                    _, ft = forward_features(teacher, Xt)
                with torch.autocast(device_type=device.type, dtype=net.dtype, enabled=device.type == "cuda"):
                    y, fs = forward_features(net, X)
                lseg = masked_seg_loss(lbl, y.float(), keep)
                cond = torch.tensor([buds[i] for i in inds], dtype=torch.float32, device=device)
                lfeat = cosine_loss(pred(fs.float(), cond), ft)
                lvar = (variance_floor(fs, s0, float(fc["gamma"])) if fc.get("var_weight")
                        else torch.zeros((), device=device))
                loss = lseg + lam * lfeat + float(fc.get("var_weight") or 0.0) * lvar
                seg_l[ep] += lseg.item() * len(inds)
                feat_l[ep] += lfeat.item() * len(inds)
                var_l[ep] += lvar.item() * len(inds)
            else:
                X, lbl, keep = augment([imgs[i] for i in inds], [tgts[i] for i in inds],
                                       [ignores[i] for i in inds], bsize=bsize,
                                       scale_range=scale_range, device=device)
                with torch.autocast(device_type=device.type, dtype=net.dtype, enabled=device.type == "cuda"):
                    y = net(X)[0]
                loss = masked_seg_loss(lbl, y.float(), keep)
                seg_l[ep] += loss.item() * len(inds)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if use_ema:
                m = ema_momentum(step, total_steps, *fc["ema_schedule"])
                mom_l[ep] = m
                with torch.no_grad():
                    for pt, ps in zip(teacher.parameters(), net.parameters()):
                        pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            step += 1
            losses[ep] += loss.item() * len(inds)
        for arr in (losses, seg_l, feat_l, var_l):
            arr[ep] /= steps_per_epoch
        lam_l[ep] = lam
        if probe:
            probes.append({"epoch": ep, **_probe(net, probe, device)})
        if ep == 5 or ep % 10 == 0 or ep == n_epochs - 1:
            extra = ""
            if teacher is not None:
                extra = f" (seg {seg_l[ep]:.4f} feat {feat_l[ep]:.4f} var {var_l[ep]:.4f} lam {lam:.2f}"
                extra += f" m {mom_l[ep]:.4f})" if use_ema else ")"
                if probes:
                    pr = probes[-1]
                    extra += f" | neck var {pr['total_var']:.2f} rank {pr['eff_rank']:.1f} sep {pr['fg_bg_sep']:.2f}"
            logger.info(f"  epoch {ep} loss {losses[ep]:.4f}{extra} lr {lr[ep]:.2e} [{time.time() - t0:.0f}s]")
    if history is not None:
        history["seg"] = seg_l.tolist()
        if teacher is not None:
            history.update({"feat": feat_l.tolist(), "var": var_l.tolist(), "lam": lam_l.tolist(),
                            "momentum": mom_l.tolist(), "probe": probes})
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_model(str(save_path))
    if pred is not None:
        torch.save(pred.state_dict(), str(save_path) + "_predictor.pt")
    return save_path, losses.tolist()


def _feature_baseline(net, items, imgs, tgts, n_crops, bsize, scale_range, device):
    """Returns (s0, probe).
        s0      : the per-channel neck feature std of the initial network, averaged over ``n_crops``;
        probe   : up to 4 fixed full-photon annotated crops (centre ``bsize`` px) with their labels,
                for the per-epoch collapse probe.
    Uses a separate numpy seed and restores the global state."""
    state = np.random.get_state()
    np.random.seed(12345)
    stds = []
    net.eval()
    with torch.no_grad():
        for i in np.random.choice(len(items), size=n_crops):
            X, _, _ = augment([imgs[i]], [tgts[i]], [None], bsize=bsize, scale_range=scale_range, device=device)
            stds.append(channel_std(forward_features(net, X)[1]))
    np.random.set_state(state)
    s0 = torch.stack(stds).mean(0)
    probe = []
    for it in items:
        if it.labels is not None and it.ignore is None and len(probe) < 4:
            H, W = it.labels.shape
            if H < bsize or W < bsize:
                continue
            y, x = (H - bsize) // 2, (W - bsize) // 2
            probe.append((it.ref_image[:, y:y + bsize, x:x + bsize].astype(np.float32),
                          it.labels[y:y + bsize, x:x + bsize]))
    return s0, probe


def _probe(net, probe, device) -> dict:
    """Collapse statistics of the neck features of ``net`` on the probe crops."""
    was = net.training
    net.eval()
    feats, labs = [], []
    with torch.no_grad():
        for img, lab in probe:
            f = forward_features(net, torch.from_numpy(img[None]).to(device))[1]
            n = f.shape[-1] # tokens per side
            ps = lab.shape[-1] // n
            feats.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).float().cpu().numpy())
            labs.append(((lab > 0).reshape(n, ps, n, ps).mean((1, 3)) > 0.5).ravel())
    net.train(was)
    return feature_statistics(np.concatenate(feats), np.concatenate(labs))
