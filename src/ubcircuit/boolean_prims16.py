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

# Smooth bump φ(x; μ, s) = exp(-||x-μ||^2 / (2 s^2)), centers μ in {(0,0),(0,1),(1,0),(1,1)}.
# s is a small bandwidth (default 0.15) to keep outputs sharp near Boolean corners.
def _rbf(x: torch.Tensor, mu: torch.Tensor, s: float) -> torch.Tensor:
    # x: (B,2), mu: (4,2) or (1,2)
    diff = x.unsqueeze(1) - mu.unsqueeze(0)  # (B,4,2)
    d2 = (diff * diff).sum(dim=-1)           # (B,4)
    return torch.exp(- d2 / (2.0 * (s ** 2)))

# Centers for (0,0),(0,1),(1,0),(1,1)
_CENTERS = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype=torch.float32)  # (4,2)

def sigma16(x2: torch.Tensor, s: float = 0.15) -> torch.Tensor:
    """
    Smoothly interpolate all 16 gates at once via RBF bumps around {0,1}^2.
    Args:
        x2: (B,2) in [0,1]^2
        s : bump bandwidth
    Returns:
        feats: (B,16) where feats[b,i] ≈ g_i(x_b) near corners and is C^∞ in x.
    """
    if x2.dim() != 2 or x2.size(-1) != 2:
        raise ValueError(f"sigma16 expects (B,2), got {tuple(x2.shape)}")
    device = x2.device
    centers = _CENTERS.to(device)         # (4,2)
    Z = Z_TABLE.to(device)                # (16,4)

    w = _rbf(x2, centers, s)              # (B,4), unnormalized weights around corners
    # Normalize across the 4 corners to form a partition of unity
    w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-12))  # (B,4)

    # For each gate i, output is sum_j w_j * Z[i,j].
    # => feats = w @ Z^T
    feats = w @ Z.t()                     # (B,16)
    return feats
