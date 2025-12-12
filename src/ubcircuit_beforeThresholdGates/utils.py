# src/ubcircuit/utils.py
# Practical utilities for training & ablation studies:
#  - temperature annealing (sync/async)
#  - entropy regularizers for soft selections
#  - diversity penalties (between units; between row-mixers)
#  - misc helpers (clamp, seeding)

from __future__ import annotations
from typing import Iterable, List, Sequence
import math
import random
import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------
# Misc helpers
# ---------------------------

def clamp01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable clamp to (0,1)."""
    return x.clamp(min=eps, max=1.0 - eps)

def seed_all(seed: int = 42) -> None:
    """Set Python/NumPy/PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with a small clamp to avoid log(0)."""
    return F.binary_cross_entropy(clamp01(pred), target)


# ---------------------------
# Entropy regularization
# ---------------------------

def entropy_from_probs(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Shannon entropy of a prob. vector or batch of vectors.
    Args:
        p: (..., K) simplex vectors
    Returns:
        scalar entropy (sum over all entries)
    """
    p = p + eps
    return -(p * p.log()).sum()

def softmax_entropy(vec: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Entropy of softmax(vec / temperature).
    Returns scalar tensor.
    """
    p = F.softmax(vec / max(temperature, 1e-8), dim=-1)
    return entropy_from_probs(p)


# ---------------------------
# Diversity penalties
# ---------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Cosine similarity between two vectors (1D).
    Higher => more similar.
    """
    a = a.view(-1)
    b = b.view(-1)
    num = torch.dot(a, b)
    den = (a.norm(p=2) * b.norm(p=2)).clamp_min(eps)
    return num / den

def unit_mix_diversity(unitWs: Sequence[torch.Tensor], tau: float) -> torch.Tensor:
    """
    Encourage *different* primitive mixes across units within the same layer.
    We sum pairwise cosine similarities of the softmaxed mixes and use it as a penalty.
    Lower is better (we'll add lambda * this to the loss).
    """
    mixes = [F.softmax(W / max(tau, 1e-8), dim=0) for W in unitWs]
    pen = torch.zeros((), device=mixes[0].device)
    for i in range(len(mixes)):
        for j in range(i + 1, len(mixes)):
            pen = pen + cosine_similarity(mixes[i], mixes[j])
    return pen

def row_diversity_penalty(L_rows: torch.Tensor) -> torch.Tensor:
    """
    Encourage different row-wise mixtures (avoid routing all wires through the same unit).
    Args:
        L_rows: (out_bits, S) row-softmax weights
    Returns:
        scalar penalty (sum of pairwise cosine similarities)
    """
    if L_rows.dim() != 2:
        raise ValueError(f"Expected (out_bits, S), got {tuple(L_rows.shape)}")
    out_bits = L_rows.size(0)
    pen = torch.zeros((), device=L_rows.device)
    for i in range(out_bits):
        for j in range(i + 1, out_bits):
            pen = pen + cosine_similarity(L_rows[i], L_rows[j])
    return pen


# ---------------------------
# Annealing schedules
# ---------------------------

def anneal_linear(step: int, total: int, T0: float, Tmin: float, start_frac: float = 0.0) -> float:
    """
    Linear temperature anneal from T0 -> Tmin starting at start_frac * total.
    Before start, returns T0; after end, returns Tmin.
    """
    total = max(total, 1)
    s = (step / total - start_frac)
    s = 0.0 if s < 0 else (1.0 if s > 1.0 - start_frac else s / (1.0 - start_frac))
    return float(Tmin + (T0 - Tmin) * (1.0 - s))

def anneal_cosine(step: int, total: int, T0: float, Tmin: float, start_frac: float = 0.0) -> float:
    """
    Cosine anneal from T0 -> Tmin; smoother than linear.
    """
    total = max(total, 1)
    s = (step / total - start_frac)
    s = 0.0 if s < 0 else (1.0 if s > 1.0 - start_frac else s / (1.0 - start_frac))
    cos_term = 0.5 * (1.0 + math.cos(math.pi * s))  # 1 -> 0 over s in [0,1]
    return float(Tmin + (T0 - Tmin) * cos_term)

def make_async_taus(
    L: int,
    step: int,
    total: int,
    T0: float,
    Tmin: float,
    direction: str = "bottom_up",
    schedule: str = "linear",
    phase_scale: float = 0.4,
    start_frac: float = 0.0,
) -> List[float]:
    """
    Produce per-layer temperatures for asynchronous sharpening.
    Args:
        L: number of layers
        step/total: progress
        T0/Tmin: start/end temps
        direction: "bottom_up" or "top_down"
        schedule: "linear" or "cosine"
        phase_scale: how much to offset phases (0=no async, 1=full)
        start_frac: global delay before annealing begins
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    taus: List[float] = []
    for li in range(L):
        if direction == "bottom_up":
            phase = li / max(1, L - 1)  # 0 (bottom) .. 1 (top)
        else:  # "top_down"
            phase = (L - 1 - li) / max(1, L - 1)

        # Phase reduces the effective progress for earlier (or later) layers
        phased_start = start_frac + phase_scale * phase
        if schedule == "cosine":
            t = anneal_cosine(step, total, T0, Tmin, start_frac=phased_start)
        else:
            t = anneal_linear(step, total, T0, Tmin, start_frac=phased_start)
        taus.append(t)
    return taus

def const_gate_penalty_16(unitWs, tau: float, li: int, L_total: int, lam_const: float = 1e-3) -> torch.Tensor:
    """
    Penalize picking constant gates (indices 0 and 15 in 16-gate head) on non-final layers.
    unitWs: list of W tensors for units in this layer.
    li: current layer index (0-based), L_total: total #layers in the stack.
    """
    if li == L_total - 1:  # allow constants on final layer
        return torch.zeros((), device=unitWs[0].device)
    pen = torch.zeros((), device=unitWs[0].device)
    for W in unitWs:
        if W.numel() == 16:  # only for 16-gate head
            p = F.softmax(W / max(tau, 1e-8), dim=0)
            pen = pen + (p[0] + p[15])  # FALSE + TRUE
    return lam_const * pen



# ---------------------------
# Regularizer bundle (easy to plug into training)
# ---------------------------

def regularizers_bundle(
    dbg: List,                      # list of (outs, L_rows, unitWs) per layer from model forward
    taus: Sequence[float],          # per-layer temperatures used that step
    lam_entropy: float = 1e-3,
    lam_div_units: float = 5e-4,
    lam_div_rows: float = 5e-4,
    lam_const16: float = 1e-3,
) -> torch.Tensor:
    """
    Compute summed regularizer given per-layer diagnostics.
    - entropy on layer row mixtures
    - entropy on unit primitive mixes
    - diversity across unit primitive mixes
    - diversity across layer row mixtures
    """
    reg = torch.zeros((), device=dbg[0][0].device)
    for li, (outs, L_rows, unitWs) in enumerate(dbg):
        tau_l = float(taus[li])

        # Entropy on each row mixture (encourage softness early in training)
        for k in range(L_rows.shape[0]):
            reg = reg + lam_entropy * entropy_from_probs(L_rows[k])

        # Entropy on each unit's primitive mixing
        for W in unitWs:
            reg = reg + lam_entropy * softmax_entropy(W, tau_l)

        # Diversity across units' primitive mixes
        if lam_div_units != 0.0 and len(unitWs) > 1:
            reg = reg + lam_div_units * unit_mix_diversity(unitWs, tau_l)

        # Diversity across row mixtures (avoid both wires selecting the same unit)
        if lam_div_rows != 0.0 and L_rows.shape[0] > 1:
            reg = reg + lam_div_rows * row_diversity_penalty(L_rows)
        
        # NEW: constant-gate penalty for 16-way heads (non-final layers)
        reg = reg + const_gate_penalty_16(unitWs, tau_l, li, len(dbg), lam_const=lam_const16)

    return reg
