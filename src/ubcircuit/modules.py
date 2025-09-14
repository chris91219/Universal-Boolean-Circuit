# src/ubcircuit/modules.py
# Model modules: BooleanUnit, PairSelector, ReasoningLayer, GeneralLayer, DepthStack.

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .boolean_prims import Rep2, sigma


class BooleanUnit(nn.Module):
    """
    One probabilistic Boolean primitive mixer:
        p = softmax(W / tau) in R^6
        y = <p, sigma(Rep2(x2))>
    """
    def __init__(self, tau: float = 0.3):
        super().__init__()
        self.W = nn.Parameter(1e-3 * torch.randn(6))
        self.tau = float(tau)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x2: (BATCH,2) tensor (continuous relaxations in [0,1] are fine)
        Returns:
            y: (BATCH,1)
        """
        feats = sigma(Rep2(x2))                               # (B,6)
        p = F.softmax(self.W / max(self.tau, 1e-8), dim=0)    # (6,)
        return torch.sum(feats * p, dim=-1, keepdim=True)     # (B,1)


class PairSelector(nn.Module):
    """
    Differentiable selection of S pairs (a_s, b_s) from B-bit input x.
    Two row-softmax matrices: PL, PR in R^{S x B}.
    """
    def __init__(self, B: int, S: int, tau: float = 0.3):
        super().__init__()
        self.PL = nn.Parameter(1e-3 * torch.randn(S, B))
        self.PR = nn.Parameter(1e-3 * torch.randn(S, B))
        self.S = int(S)
        self.B = int(B)
        self.tau = float(tau)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, xB: torch.Tensor) -> torch.Tensor:
        """
        xB: (BATCH, B)
        returns pairs2: (BATCH, S, 2) with pairs2[:, s, 0]=a_s, pairs2[:, s, 1]=b_s
        """
        PL = F.softmax(self.PL / max(self.tau, 1e-8), dim=-1)  # (S,B)
        PR = F.softmax(self.PR / max(self.tau, 1e-8), dim=-1)  # (S,B)
        a = xB @ PL.t()                                        # (batch,S)
        b = xB @ PR.t()                                        # (batch,S)
        return torch.stack([a, b], dim=-1)                     # (batch,S,2)


class ReasoningLayer(nn.Module):
    """
    Parallel S BooleanUnits followed by a row-wise softmax mixing to produce 'out_bits' wires.
    - outs: (BATCH,S) are unit outputs.
    - WL: (out_bits, S) with row-softmax to mix S units into each next-bit wire.
    """
    def __init__(self, S: int = 2, out_bits: int = 2, tau: float = 0.3):
        super().__init__()
        if S < 1 or out_bits < 1:
            raise ValueError("S and out_bits must be >= 1")
        self.units = nn.ModuleList([BooleanUnit(tau=tau) for _ in range(S)])
        self.WL = nn.Parameter(1e-3 * torch.randn(out_bits, S))  # row-softmax over S
        self.tau = float(tau)
        self.out_bits = int(out_bits)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)
        for u in self.units:
            u.set_tau(tau)

    def forward(self, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x2: (BATCH,2)
        Returns:
            x_next: (BATCH,out_bits)
            outs:   (BATCH,S) unit outputs before mixing
            L_rows: (out_bits,S) row-softmax weights
            unitWs: list of each unit.param W (for diagnostics/regularizers)
        """
        outs = torch.cat([u(x2) for u in self.units], dim=-1)      # (B,S)
        L_rows = F.softmax(self.WL / max(self.tau, 1e-8), dim=-1)  # (out_bits,S)
        next_bits = [(outs * L_rows[k]).sum(-1, keepdim=True) for k in range(self.out_bits)]
        x_next = torch.cat(next_bits, dim=-1)                      # (B,out_bits)
        unitWs = [u.W for u in self.units]
        return x_next, outs, L_rows, unitWs


class GeneralLayer(nn.Module):
    """
    First layer when in_bits=B>=2:
      - If in_bits==2: behave like ReasoningLayer.
      - If in_bits>2: PairSelector picks S pairs, apply S BooleanUnits on each pair.
      Output mixed into out_bits via row-softmax WL.
    """
    def __init__(self, in_bits: int, S: int, out_bits: int, tau: float = 0.3):
        super().__init__()
        self.in_bits = int(in_bits)
        self.S = int(S)
        self.out_bits = int(out_bits)
        self.tau = float(tau)
        self.units = nn.ModuleList([BooleanUnit(tau=tau) for _ in range(S)])
        self.WL = nn.Parameter(1e-3 * torch.randn(out_bits, S))
        self.selector = PairSelector(B=in_bits, S=S, tau=tau) if in_bits > 2 else None

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)
        for u in self.units:
            u.set_tau(tau)
        if self.selector is not None:
            self.selector.set_tau(tau)

    def forward(self, x: torch.Tensor):
        # x: (BATCH, in_bits)
        if self.selector is None:
            # in_bits==2
            x2 = x  # (B,2)
            outs = torch.cat([u(x2) for u in self.units], dim=-1)  # (B,S)
        else:
            pairs = self.selector(x)                                # (B,S,2)
            outs = torch.cat([self.units[s](pairs[:, s, :]) for s in range(self.S)], dim=-1)  # (B,S)
        L_rows = F.softmax(self.WL / max(self.tau, 1e-8), dim=-1)   # (out_bits,S)
        next_bits = [(outs * L_rows[k]).sum(-1, keepdim=True) for k in range(self.out_bits)]
        x_next = torch.cat(next_bits, dim=-1)                        # (B,out_bits)
        unitWs = [u.W for u in self.units]
        dbg = {
            "outs": outs, "L_rows": L_rows, "unitWs": unitWs,
            "PL": (self.selector.PL if self.selector is not None else None),
            "PR": (self.selector.PR if self.selector is not None else None),
        }
        return x_next, dbg


class DepthStack(nn.Module):
    """
    L layers: first is GeneralLayer(in_bits=B, out_bits=2), middle ReasoningLayer(2->2),
    final ReasoningLayer(2->1).
    """
    def __init__(self, B: int, L: int = 2, S: int = 2, tau: float = 0.3):
        super().__init__()
        if L < 1:
            raise ValueError("L must be >= 1")
        self.layers = nn.ModuleList([])
        # First layer maps B -> 2 bits
        self.layers.append(GeneralLayer(in_bits=B, S=S, out_bits=2, tau=tau))
        # Middle (L-2) keep 2 -> 2
        for _ in range(max(L - 2, 0)):
            self.layers.append(ReasoningLayer(S=S, out_bits=2, tau=tau))
        # Final 2 -> 1
        if L >= 2:
            self.layers.append(ReasoningLayer(S=S, out_bits=1, tau=tau))
        self.tau = float(tau)
        self.L = int(L)
        self.B = int(B)

    def set_layer_taus(self, taus: List[float]) -> None:
        if len(taus) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} taus, got {len(taus)}")
        for lyr, t in zip(self.layers, taus):
            lyr.set_tau(float(t))

    def forward(self, xB: torch.Tensor):
        dbg_list = []
        x = xB
        for li, layer in enumerate(self.layers):
            out = layer(x)
            if isinstance(layer, GeneralLayer):
                x, dbg = out
                dbg_list.append((dbg["outs"], dbg["L_rows"], dbg["unitWs"], dbg["PL"], dbg["PR"]))
            else:
                x, outs, L_rows, unitWs = out
                dbg_list.append((outs, L_rows, unitWs, None, None))
        return x, dbg_list  # final x is (BATCH,1)
