# src/ubcircuit/boolean_prims.py
# Primitive Boolean projections and features for 2-bit inputs.

from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn.functional as F

# Matrices from the draft (Rep2)
#   A1 x = [[a, b],
#           [a, 0],
#           [b, 0]]
#   A2 x = [[0, 1],
#           [0, 0],
#           [0, 0]]
A1 = torch.tensor([[1.0, 0.0],
                   [1.0, 0.0],
                   [0.0, 1.0]])
A2 = torch.tensor([[0.0, 1.0],
                   [0.0, 0.0],
                   [0.0, 0.0]])

PRIMS: List[str] = [
    "AND(a,b)=min",   # index 0
    "OR(a,b)=max",    # index 1
    "NOT(a)",         # index 2
    "NOT(b)",         # index 3
    "a (skip)",       # index 4
    "b (skip)"        # index 5
]

def NOT(u: torch.Tensor) -> torch.Tensor:
    """Elementwise logical NOT for {0,1}-valued tensors represented as floats."""
    return 1.0 - u

def Rep2(x: torch.Tensor) -> torch.Tensor:
    """
    Lift 2-bit inputs to a (3,2) 'matrix' per sample, stacking A1x and A2x.
    Args:
        x: (B,2) tensor with entries in [0,1]
    Returns:
        X: (B,3,2) tensor
    """
    if x.dim() != 2 or x.size(-1) != 2:
        raise ValueError(f"Rep2 expects (B,2), got {tuple(x.shape)}")
    # Use CPU tensors for A1/A2; x device may vary. Move to x.device at runtime.
    X1 = x @ A1.to(x.device).t()
    X2 = x @ A2.to(x.device).t()
    # We want shape (B,3,2): concat as last dim
    return torch.stack([X1, X2], dim=-1)

def sigma(X: torch.Tensor) -> torch.Tensor:
    """
    Compute 6 primitive features from (B,3,2) lifted inputs:
      [min(row1), max(row1), NOT(max(row2)), NOT(max(row3)), max(row2), max(row3)]
    Args:
        X: (B,3,2)
    Returns:
        feats: (B,6)
    """
    if X.dim() != 3 or X.size(1) != 3 or X.size(2) != 2:
        raise ValueError(f"sigma expects (B,3,2), got {tuple(X.shape)}")
    r1 = X[:, 0, :]  # (B,2)
    r2 = X[:, 1, :]
    r3 = X[:, 2, :]

    f_AND  = torch.min(r1, dim=-1).values       # min(a,b)
    f_OR   = torch.max(r1, dim=-1).values       # max(a,b)
    f_NOTa = NOT(torch.max(r2, dim=-1).values)  # NOT(a)
    f_NOTb = NOT(torch.max(r3, dim=-1).values)  # NOT(b)
    f_a    = torch.max(r2, dim=-1).values       # a (skip)
    f_b    = torch.max(r3, dim=-1).values       # b (skip)

    return torch.stack([f_AND, f_OR, f_NOTa, f_NOTb, f_a, f_b], dim=-1)  # (B,6)
