"""The representation objective of Thin2Seg.

The online network sees a thinned copy of a recording, the target network the full recording
under the same augmentation. A predictor, conditioned on the photon budgets of the two inputs,
maps the online features onto the target features. Everything here is used in training only.
"""

from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn.functional as F

NECK_DIM = 256


def forward_features(net, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``cellpose.vit_sam.Transformer.forward`` that also returns the neck features:
    (flow-head output [N, 3, H, W], neck [N, 256, H/8, W/8]). Same random layer drop in training
    mode and the same detached patch embedding as the original forward."""
    enc = net.encoder
    x = F.conv2d(x, enc.patch_embed.proj.weight.data[:, :x.shape[1]],
                 bias=enc.patch_embed.proj.bias.data, stride=net.ps)
    x = x.permute(0, 2, 3, 1)
    if enc.pos_embed is not None:
        x = x + enc.pos_embed
    nlay = len(enc.blocks)
    rdrop = None
    if net.training and net.rdrop > 0:
        rdrop = (torch.rand((len(x), nlay), device=x.device) <
                 torch.linspace(0, net.rdrop, nlay, device=x.device)).to(x.dtype)
    for i, blk in enumerate(enc.blocks):
        if rdrop is not None:
            mask = rdrop[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = x * mask + blk(x) * (1 - mask)
        else:
            x = blk(x)
    neck = enc.neck(x.permute(0, 3, 1, 2))
    y = F.conv_transpose2d(net.out(neck), net.W2, stride=net.ps, padding=0)
    return y, neck


class Predictor(torch.nn.Module):
    """Per-token residual MLP on the neck, q(z) = z + W2 GELU(FiLM(W1 LN(z))).

    The hidden layer is modulated (FiLM) by sinusoidal encodings of the log photon budgets of the
    online input and of the target input. ``W2`` and the FiLM output layer start at zero, so the
    predictor is the identity at step 0 and training starts as plain feature matching. There is
    no spatial mixing: each token is mapped from its low-photon to its full-photon version.
    """

    def __init__(self, dim: int = NECK_DIM, hidden: int = 1024, n_freq: int = 4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.n_freq = n_freq
        self.film = torch.nn.Sequential(torch.nn.Linear(2 * 2 * n_freq, 128), torch.nn.GELU(),
                                        torch.nn.Linear(128, 2 * hidden))
        torch.nn.init.zeros_(self.film[-1].weight) # type: ignore
        torch.nn.init.zeros_(self.film[-1].bias) # type: ignore

    def encode(self, cond: torch.Tensor) -> torch.Tensor:
        """[N, 2] log photon levels (online, target) -> sinusoidal features [N, 4 n_freq]."""
        f = cond[..., None] * (2.0 ** torch.arange(self.n_freq, device=cond.device, dtype=cond.dtype))
        return torch.cat([torch.sin(f), torch.cos(f)], -1).flatten(1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z [N, C, H, W]
        cond [N, 2] = log1p photons per cell pixel of (online, target) input.
        """
        h = z.permute(0, 2, 3, 1)
        h = self.fc1(F.layer_norm(h, (h.shape[-1],)))
        g, b = self.film(self.encode(cond)).chunk(2, dim=-1)
        h = h * (1 + g[:, None, None, :]) + b[:, None, None, :]
        h = self.fc2(F.gelu(h))
        return z + h.permute(0, 3, 1, 2)


def cosine_loss(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mean over tokens of 2 - 2 cos(s, t); [N, C, H, W] inputs. Scale-invariant, so shrinking
    the features cannot lower it."""
    # TODO: Double check against manuscript
    return (2.0 - 2.0 * F.cosine_similarity(s.float(), t.float(), dim=1)).mean()


def channel_std(z: torch.Tensor) -> torch.Tensor:
    """Per-channel standard deviation over all tokens of the batch: [C]."""
    return z.float().permute(1, 0, 2, 3).reshape(z.shape[1], -1).std(dim=1)


def variance_floor(z: torch.Tensor, s0: torch.Tensor, gamma: float) -> torch.Tensor:
    """Hinge on mean over channels of relu(1 - std_c / (gamma * s0_c)). Zero when 
    every channel keeps at least ``gamma`` times its reference spread ``s0`` (that of the
    initial network), so it should only acts against collapse."""
    return F.relu(1.0 - channel_std(z) / (gamma * s0).clamp_min(1e-6)).mean()


def ema_momentum(step: int, total: int, lo: float, hi: float) -> float:
    """Momentum of the target network. ``lo`` -> ``hi`` via cosine."""
    return hi - (hi - lo) * (math.cos(math.pi * step / max(total, 1)) + 1) / 2


def ramp_weight(epoch: int, start: int, end: int) -> float:
    """Feature-loss weight. 0 at ``start``, linear to 1 at ``end``."""
    if epoch < start: return 0.0
    return 1.0 if end <= start else min(1.0, (epoch - start) / (end - start))


def feature_statistics(tokens, is_cell) -> dict:
    """Collapse diagnostics of neck tokens [N, C] with a boolean cell/background label per token.
    Includes total variance, effective rank of the covariance, and the distance between the mean cell
    and mean background token over the mean within-class distribution."""
    f, lab = np.asarray(tokens), np.asarray(is_cell, bool)
    ev = np.clip(np.linalg.eigvalsh(np.cov(f, rowvar=False))[::-1], 0, None)
    p = ev / ev.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    out = {"total_var": float(ev.sum()), "eff_rank": eff, "fg_bg_sep": float("nan")}
    if lab.any() and (~lab).any():
        mu1, mu0 = f[lab].mean(0), f[~lab].mean(0)
        spread = 0.5 * (np.sqrt(((f[lab] - mu1) ** 2).sum(1)).mean() + np.sqrt(((f[~lab] - mu0) ** 2).sum(1)).mean())
        out["fg_bg_sep"] = float(np.linalg.norm(mu1 - mu0) / max(spread, 1e-9))
    return out
