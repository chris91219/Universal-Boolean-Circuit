# src/ubcircuit/boolean_prims16.py
# Smooth (C^∞) 16-gate feature head using bump interpolation over {0,1}^2.

from __future__ import annotations
from typing import List
import torch

# ---- Enumerate the 16 binary 2-input Boolean gates by truth table order:
# inputs order: (0,0), (0,1), (1,0), (1,1)
# Each row is a length-4 vector (Z) of outputs for that gate.
Z_TABLE = torch.tensor([
    [0,0,0,0],  # g1  : FALSE
    [0,0,0,1],  # g2  : AND
    [0,0,1,0],  # g3  : A AND NOT B
    [0,0,1,1],  # g4  : A
    [0,1,0,0],  # g5  : NOT A AND B
    [0,1,0,1],  # g6  : B
    [0,1,1,0],  # g7  : XOR
    [0,1,1,1],  # g8  : OR
    [1,0,0,0],  # g9  : NOR
    [1,0,0,1],  # g10 : XNOR
    [1,0,1,0],  # g11 : NOT B
    [1,0,1,1],  # g12 : A OR NOT B
    [1,1,0,0],  # g13 : NOT A
    [1,1,0,1],  # g14 : NOT A OR B
    [1,1,1,0],  # g15 : NAND
    [1,1,1,1],  # g16 : TRUE
], dtype=torch.float32)  # (16,4)

PRIMS16: List[str] = [
    "FALSE", "AND", "A&~B", "A", "~A&B", "B", "XOR", "OR",
    "NOR", "XNOR", "~B", "A|~B", "~A", "~A|B", "NAND", "TRUE"
]

# --- add a global default mode and radius (optional helpers) ---
_S16_MODE = "rbf"     # "rbf" or "bump"
_S16_RADIUS = 0.75    # only used when mode == "bump"
def set_sigma16_mode(mode: str) -> None:
    global _S16_MODE
    if mode not in {"rbf", "bump", "lagrange"}:
        raise ValueError("mode must be 'rbf', 'bump', or 'lagrange'")
    _S16_MODE = mode

def set_sigma16_radius(r: float) -> None:
    global _S16_RADIUS
    _S16_RADIUS = float(r)

_S16_BANDWIDTH = 0.15
def set_sigma16_bandwidth(s: float) -> None:
    global _S16_BANDWIDTH
    _S16_BANDWIDTH = float(s)

# Smooth bump φ(x; μ, s) = exp(-||x-μ||^2 / (2 s^2)), centers μ in {(0,0),(0,1),(1,0),(1,1)}.
# s is a small bandwidth (default 0.15) to keep outputs sharp near Boolean corners.
def _rbf(x: torch.Tensor, mu: torch.Tensor, s: float) -> torch.Tensor:
    # x: (B,2), mu: (4,2) or (1,2)
    diff = x.unsqueeze(1) - mu.unsqueeze(0)  # (B,4,2)
    d2 = (diff * diff).sum(dim=-1)           # (B,4)
    return torch.exp(- d2 / (2.0 * (s ** 2)))

# --- with a compactly supported bump ---
def _bump(x: torch.Tensor, mu: torch.Tensor, r: float) -> torch.Tensor:
    # x: (B,2), mu: (4,2) or (1,2)
    diff = x.unsqueeze(1) - mu.unsqueeze(0)         # (B,4,2)
    d = diff.norm(dim=-1)                           # (B,4)
    z = (d / r).clamp(min=0.0)                      # (B,4)
    inside = (z < 1.0)
    # exp(-1/(1-z^2)) on support, 0 outside
    val = torch.zeros_like(z)
    z2 = z[inside] * z[inside]
    val[inside] = torch.exp(-1.0 / (1.0 - z2))
    return val

# --- new: multilinear / Lagrange basis on {0,1}^2 ---
def _lagrange_weights(x2: torch.Tensor) -> torch.Tensor:
    """
    Bilinear basis values at x2 for corners ordered as:
    (0,0), (0,1), (1,0), (1,1)  -> matches your Z_TABLE column order.
    Returns: w: (B,4)
    """
    x1 = x2[:, 0:1]  # (B,1)
    x2c = x2[:, 1:2]
    phi00 = (1.0 - x1) * (1.0 - x2c)
    phi01 = (1.0 - x1) * x2c
    phi10 = x1 * (1.0 - x2c)
    phi11 = x1 * x2c
    # stack in the same corner order as Z columns: (0,0),(0,1),(1,0),(1,1)
    return torch.cat([phi00, phi01, phi10, phi11], dim=1)  # (B,4)

# Centers for (0,0),(0,1),(1,0),(1,1)
_CENTERS = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype=torch.float32)  # (4,2)

def sigma16(
    x2: torch.Tensor,
    s: float | None = 0.15,
    *,
    mode: str | None = None,
    r: float | None = None,
) -> torch.Tensor:
    """
    Interpolates all 16 gates around {0,1}^2.

    Args:
        x2  : (B,2) in [0,1]^2
        s   : RBF bandwidth if mode='rbf' (ignored otherwise). If None, uses global.
        mode: 'rbf' (Gaussian), 'bump' (compact C^∞), or 'lagrange' (bilinear). If None, uses global.
        r   : bump radius if mode='bump' (suggest 0.75). If None, uses global.

    Returns:
        feats: (B,16)
    """
    if x2.dim() != 2 or x2.size(-1) != 2:
        raise ValueError(f"sigma16 expects (B,2), got {tuple(x2.shape)}")

    # resolve defaults
    s = _S16_BANDWIDTH if s is None else float(s)
    mode = _S16_MODE if mode is None else mode
    if mode not in {"rbf", "bump", "lagrange"}:
        raise ValueError("mode must be 'rbf', 'bump', or 'lagrange'")
    r = _S16_RADIUS if (r is None) else float(r)

    device = x2.device
    centers = _CENTERS.to(device)   # (4,2)
    Z = Z_TABLE.to(device)          # (16,4)

    if mode == "rbf":
        w = _rbf(x2, centers, s)                        # (B,4)
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    elif mode == "bump":
        w = _bump(x2, centers, r)                       # (B,4)
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    else:  # 'lagrange'
        w = _lagrange_weights(x2)                       # (B,4) sums to 1 on [0,1]^2 by construction

    feats = w @ Z.t()                                   # (B,16)
    return feats