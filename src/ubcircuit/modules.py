# src/ubcircuit/modules.py
# Model modules: BooleanUnit, PairSelector, ReasoningLayer, GeneralLayer, DepthStack.

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .boolean_prims import Rep2, sigma as sigma6
from .boolean_prims16 import sigma16
from .utils import clamp01


class BooleanUnit(nn.Module):
    """
    Probabilistic Boolean primitive mixer.
      - gate_set = "6"  -> features in R^6 via sigma6(Rep2(x))
      - gate_set = "16" -> features in R^16 via sigma16(x)
    """
    def __init__(self, tau: float = 0.3, gate_set: str = "6", s16: float = 0.15):
        super().__init__()
        assert gate_set in {"6", "16"}
        self.gate_set = gate_set
        self.K = 16 if gate_set == "16" else 6
        self.W = nn.Parameter(1e-3 * torch.randn(self.K))
        self.tau = float(tau)
        self.s16 = float(s16)

    def set_sigma16_bandwidth(self, s: float) -> None:
        self.s16 = float(s)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, x2: torch.Tensor) -> torch.Tensor:
        if self.gate_set == "16":
            feats = sigma16(x2, s=self.s16)                    # (B,16)  smooth in x
        else:
            feats = sigma6(Rep2(x2))                           # (B,6)   piecewise-smooth
        p = F.softmax(self.W / max(self.tau, 1e-8), dim=0)     # (K,)
        return torch.sum(feats * p, dim=-1, keepdim=True)      # (B,1)



class PairSelector(nn.Module):
    """
    Differentiable selection of S pairs (a_s, b_s) from B-bit input x.
    Two row-softmax matrices: PL, PR in R^{S x B}.
    """
    def __init__(self, B: int, S: int, tau: float = 0.3,
                 repel: bool = True, repel_eta: float = 1.0, repel_mode: str = "log"):
        super().__init__()
        self.PL = nn.Parameter(1e-3 * torch.randn(S, B))
        self.PR = nn.Parameter(1e-3 * torch.randn(S, B))
        self.S = int(S)
        self.B = int(B)
        self.tau = float(tau)
        self.repel = bool(repel)
        self.repel_eta = float(repel_eta)
        assert repel_mode in {"log", "mul"}
        self.repel_mode = repel_mode

    def set_repulsion(self, repel: bool = None, eta: float = None, mode: str = None):
        if repel is not None: self.repel = bool(repel)
        if eta   is not None: self.repel_eta = float(eta)
        if mode  is not None:
            assert mode in {"log", "mul"}
            self.repel_mode = mode


    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, xB: torch.Tensor) -> torch.Tensor:
        """
         xB: (BATCH, B)
         returns pairs2: (BATCH, S, 2) with pairs2[:, s, 0]=a_s, pairs2[:, s, 1]=b_s
        """        
        temp = max(self.tau, 1e-8)
        # Left pick (standard)
        PL = F.softmax(self.PL / temp, dim=-1)                 # (S,B)

        # --- HARD MASK to avoid doubled edges ---
        left_idx = PL.argmax(dim=-1)            # (S,)
        mask = torch.full_like(self.PR, 0.0)    # (S,B)
        mask.scatter_(1, left_idx.view(-1,1), float('-inf'))  # -inf at PL argmax per row

        # Right pick: repulsive by default
        if self.repel:
            if self.repel_mode == "log":
                # log-space bias: logits_R + eta * log(1 - pL)
                # gate = torch.log(clamp01(1.0 - PL))            # (S,B)
                # adj_logits = self.PR / temp + self.repel_eta * gate
                # PR = F.softmax(adj_logits, dim=-1)

                # Add hard mask to avoid exact duplicates
                gate = torch.log(clamp01(1.0 - PL))
                adj_logits = (self.PR / temp) + self.repel_eta * gate + mask
                PR = F.softmax(adj_logits, dim=-1)

            else:  # "mul" 
                # Prob-space multiplicative gate 
                # w2 = F.softmax(self.PR / temp, dim=-1)     # interpret PR as logits -> probs
                # gate = (1.0 - PL)                          # (S,B)
                # pR_unnorm = self.repel_eta * gate * w2     # elementwise
                # pR = pR_unnorm / pR_unnorm.sum(dim=-1, keepdim=True)

                # Add hard mask to avoid exact duplicates
                w2 = F.softmax(self.PR / temp, dim=-1)
                gate = (1.0 - PL)
                pR_unnorm = self.repel_eta * gate * w2
                # zero out the masked column:
                pR_unnorm.scatter_(1, left_idx.view(-1,1), 0.0)
                PR = pR_unnorm / pR_unnorm.sum(dim=-1, keepdim=True)

        else:
            PR = F.softmax(self.PR / temp, dim=-1)             # (S,B)

        a = xB @ PL.t()                                        # (batch,S)
        b = xB @ PR.t()                                        # (batch,S)
        return torch.stack([a, b], dim=-1)                     # (batch,S,2)


class ReasoningLayer(nn.Module):
    """
    Parallel S BooleanUnits followed by a row-wise softmax mixing to produce 'out_bits' wires.
    - outs: (BATCH,S) are unit outputs.
    - WL: (out_bits, S) with row-softmax to mix S units into each next-bit wire.
    """
    def __init__(self, S: int = 2, out_bits: int = 2, tau: float = 0.3, gate_set: str = "6"):
        super().__init__()
        if S < 1 or out_bits < 1:
            raise ValueError("S and out_bits must be >= 1")
        self.units = nn.ModuleList([BooleanUnit(tau=tau, gate_set=gate_set) for _ in range(S)])
        self.gate_set = gate_set
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
    def __init__(self, in_bits: int, S: int, out_bits: int, tau: float = 0.3,
                 pair: dict | None = None, gate_set: str = "6"):
        super().__init__()
        self.in_bits = int(in_bits)
        self.S = int(S)
        self.out_bits = int(out_bits)
        self.tau = float(tau)
        self.units = nn.ModuleList([BooleanUnit(tau=tau, gate_set=gate_set) for _ in range(S)])
        self.gate_set = gate_set
        self.WL = nn.Parameter(1e-3 * torch.randn(out_bits, S))
        if in_bits > 2:
            pair = pair or {}
            self.selector = PairSelector(
                B=in_bits, S=S, tau=tau,
                repel=pair.get("repel", True),
                repel_eta=pair.get("eta", 1.0),
                repel_mode=pair.get("mode", "log"),
            )
        else:
            self.selector = None

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
            "outs": outs,
            "L_rows": L_rows,
            "unitWs": unitWs,
            "PL": (
                F.softmax(self.selector.PL / max(self.tau, 1e-8), dim=-1)
                if self.selector is not None else None
            ),
            "PR": (
                F.softmax(
                    self.selector.PR / max(self.tau, 1e-8)
                    + self.selector.repel_eta * torch.log(
                        clamp01(1.0 - F.softmax(self.selector.PL / max(self.tau, 1e-8), dim=-1))
                    ),
                    dim=-1,
                )
                if (self.selector is not None and self.selector.repel and self.selector.repel_mode == "log")
                else (
                    F.softmax(self.selector.PR / max(self.tau, 1e-8), dim=-1)
                    if self.selector is not None else None
                )
            ),
        }
        return x_next, dbg


class DepthStack(nn.Module):
    """
    L layers: first is GeneralLayer(in_bits=B, out_bits=2), middle ReasoningLayer(2->2),
    final ReasoningLayer(2->1).
    """
    def __init__(self, B: int, L: int = 2, S: int = 2, tau: float = 0.3,
                 pair: dict | None = None, gate_set: str = "6"):
        super().__init__()
        if L < 1:
            raise ValueError("L must be >= 1")
        self.layers = nn.ModuleList([])
        # First layer maps B -> 2 bits  
        self.layers.append(GeneralLayer(in_bits=B, S=S, out_bits=2, tau=tau, pair=pair, gate_set=gate_set))
        # Middle (L-2) keep 2 -> 2
        for _ in range(max(L - 2, 0)):
            self.layers.append(ReasoningLayer(S=S, out_bits=2, tau=tau, gate_set=gate_set))
        # Final 2 -> 1
        if L >= 2:
            self.layers.append(ReasoningLayer(S=S, out_bits=1, tau=tau, gate_set=gate_set))
        self.gate_set = gate_set
        
        self.tau = float(tau)
        self.L = int(L)
        self.B = int(B)

    def set_layer_taus(self, taus: List[float]) -> None:
        if len(taus) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} taus, got {len(taus)}")
        for lyr, t in zip(self.layers, taus):
            lyr.set_tau(float(t))

    def set_layer_taus_and_bandwidths(self, taus: List[float], s_start=0.25, s_end=0.10):
        assert len(taus) == len(self.layers)
        for li, lyr in enumerate(self.layers):
            lyr.set_tau(taus[li])
            # linear interp bandwidth
            s = s_start + (s_end - s_start) * (li / max(1, len(self.layers)-1))
            # apply if 16
            if hasattr(lyr, "units"):
                for u in lyr.units:
                    if getattr(u, "gate_set", "6") == "16":
                        u.set_sigma16_bandwidth(s)


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
