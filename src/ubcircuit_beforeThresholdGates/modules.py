# src/ubcircuit/modules.py
# Model modules: BooleanUnit, PairSelector, ReasoningLayer, GeneralLayer, DepthStack.

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .boolean_prims import Rep2, sigma as sigma6
from .boolean_prims16 import sigma16
from .fixed_selector import FixedPairSelector
from .utils import clamp01
from .lifting import BitLifting   # NEW: lifting layer


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
                 repel: bool = True, repel_eta: float = 1.0, repel_mode: str = "hard-log",
                 PL_prior: torch.Tensor | None = None,
                 PR_prior: torch.Tensor | None = None,
                 prior_strength: float = 0.0):
        super().__init__()
        self.PL = nn.Parameter(1e-3 * torch.randn(S, B))
        self.PR = nn.Parameter(1e-3 * torch.randn(S, B))
        self.S = int(S)
        self.B = int(B)
        self.tau = float(tau)
        self.repel = bool(repel)
        self.repel_eta = float(repel_eta)
        assert repel_mode in {"log", "mul", "hard-log", "hard-mul"}
        self.repel_mode = repel_mode
        self.prior_strength = float(prior_strength)
        if PL_prior is not None:
            assert PL_prior.shape == (S, B)
            self.register_buffer("PL_prior", PL_prior)
        else:
            self.PL_prior = None
        if PR_prior is not None:
            assert PR_prior.shape == (S, B)
            self.register_buffer("PR_prior", PR_prior)
        else:
            self.PR_prior = None

    def set_repulsion(self, repel: bool = None, eta: float = None, mode: str = None):
        if repel is not None: self.repel = bool(repel)
        if eta   is not None: self.repel_eta = float(eta)
        if mode  is not None:
            assert mode in {"log", "mul", "hard-log", "hard-mul"}
            self.repel_mode = mode

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, xB: torch.Tensor) -> torch.Tensor:
        """
        xB: (BATCH, B)
        returns pairs2: (BATCH, S, 2) with pairs2[:, s, 0]=a_s, pairs2[:, s, 1]=b_s
        """
        temp = max(self.tau, 1e-8)

        # -------- Left pick: PL (row-softmax over B inputs) --------
        PL_logits = self.PL / temp
        # (optional) soft MI prior in log-space
        if getattr(self, "PL_prior", None) is not None and self.prior_strength > 0.0:
            PL_logits = PL_logits + self.prior_strength * torch.log(clamp01(self.PL_prior))
        PL = F.softmax(PL_logits, dim=-1)                               # (S,B)

        # Indices + masks used by "hard-*" modes to forbid doubled edges
        left_idx = PL.argmax(dim=-1)                                    # (S,)
        mask_logit = torch.full_like(self.PR, 0.0)                      # (S,B)
        mask_logit.scatter_(1, left_idx.view(-1,1), float('-inf'))      # for log-space
        mask_prob = torch.zeros_like(self.PR)                           # (S,B)
        mask_prob.scatter_(1, left_idx.view(-1,1), 1.0)                 # for prob-space zeroing

        # -------- Right pick: PR --------
        if self.repel:
            mode = self.repel_mode
            if mode in {"log", "hard-log"}:
                # log-space repulsion against PL: add eta * log(1 - PL)
                gate = torch.log(clamp01(1.0 - PL))                     # (S,B)
                PR_logits = (self.PR / temp) + self.repel_eta * gate

                # (optional) soft MI prior in log-space
                if getattr(self, "PR_prior", None) is not None and self.prior_strength > 0.0:
                    PR_logits = PR_logits + self.prior_strength * torch.log(clamp01(self.PR_prior))

                # hard mask only in "hard-log"
                if mode == "hard-log":
                    PR_logits = PR_logits + mask_logit

                PR = F.softmax(PR_logits, dim=-1)                       # (S,B)

            elif mode in {"mul", "hard-mul"}:
                # prob-space repulsion: scale probs by (1 - PL)
                w2 = F.softmax(self.PR / temp, dim=-1)                  # (S,B)
                # (optional) soft MI prior in prob-space
                if getattr(self, "PR_prior", None) is not None and self.prior_strength > 0.0:
                    # raise prior to strength, then renorm later
                    w2 = w2 * (clamp01(self.PR_prior) ** self.prior_strength)

                gate = (1.0 - PL)                                       # (S,B)
                pR_unnorm = self.repel_eta * gate * w2                  # (S,B)

                if mode == "hard-mul":
                    # zero the PL argmax column
                    pR_unnorm = pR_unnorm * (1.0 - mask_prob)

                PR = pR_unnorm / pR_unnorm.sum(dim=-1, keepdim=True)    # (S,B)

            else:
                # should not happen (assert in __init__), but keep a safe fallback
                PR = F.softmax(self.PR / temp, dim=-1)
        else:
            PR = F.softmax(self.PR / temp, dim=-1)

        # -------- Gather pairs and return --------
        a = xB @ PL.t()                                                 # (batch,S)
        b = xB @ PR.t()                                                 # (batch,S)
        return torch.stack([a, b], dim=-1)                              # (batch,S,2)



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
            route = pair.get("route", "learned")
            if route == "mi_hard" and ("fixed_pairs" in pair):
                self.selector = FixedPairSelector(B=in_bits, pairs=pair["fixed_pairs"])
            elif route == "mi_soft" and ("PL_prior" in pair) and ("PR_prior" in pair):
                PLp = torch.tensor(pair["PL_prior"], dtype=torch.float32)
                PRp = torch.tensor(pair["PR_prior"], dtype=torch.float32)
                self.selector = PairSelector(
                    B=in_bits, S=S, tau=tau,
                    repel=pair.get("repel", True),
                    repel_eta=pair.get("eta", 1.0),
                    repel_mode=pair.get("mode", "hard-log"),
                    PL_prior=PLp, PR_prior=PRp,
                    prior_strength=float(pair.get("prior_strength", 2.0)),
                )
            else:
                # default learned
                self.selector = PairSelector(
                    B=in_bits, S=S, tau=tau,
                    repel=pair.get("repel", True),
                    repel_eta=pair.get("eta", 1.0),
                    repel_mode=pair.get("mode", "hard-log"),
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

        temp = max(self.tau, 1e-8)
        PL_dbg = F.softmax(self.selector.PL / temp, dim=-1) if self.selector is not None else None
        if self.selector is None:
            PR_dbg = None
        elif not self.selector.repel:
            PR_dbg = F.softmax(self.selector.PR / temp, dim=-1)
        else:
            mode = self.selector.repel_mode
            if mode in {"log", "hard-log"}:
                gate = torch.log(clamp01(1.0 - PL_dbg))
                PR_logits = (self.selector.PR / temp) + self.selector.repel_eta * gate
                if mode == "hard-log":
                    left_idx_dbg = PL_dbg.argmax(dim=-1)
                    mask_logit_dbg = torch.full_like(self.selector.PR, 0.0)
                    mask_logit_dbg.scatter_(1, left_idx_dbg.view(-1,1), float('-inf'))
                    PR_logits = PR_logits + mask_logit_dbg
                PR_dbg = F.softmax(PR_logits, dim=-1)
            else:  # "mul" or "hard-mul"
                w2 = F.softmax(self.selector.PR / temp, dim=-1)
                gate = (1.0 - PL_dbg)
                pR_unnorm = self.selector.repel_eta * gate * w2
                if mode == "hard-mul":
                    left_idx_dbg = PL_dbg.argmax(dim=-1)
                    mask_prob_dbg = torch.zeros_like(self.selector.PR)
                    mask_prob_dbg.scatter_(1, left_idx_dbg.view(-1,1), 1.0)
                    pR_unnorm = pR_unnorm * (1.0 - mask_prob_dbg)
                PR_dbg = pR_unnorm / pR_unnorm.sum(dim=-1, keepdim=True)

        dbg = {
            "outs": outs,
            "L_rows": L_rows,
            "unitWs": unitWs,
            "PL": PL_dbg,
            "PR": PR_dbg,
        }

        return x_next, dbg


class DepthStack(nn.Module):
    """
    L layers: optional BitLifting first (B -> B_eff),
    then:
      - first GeneralLayer(in_bits=B_eff, out_bits=2),
      - (L-2) ReasoningLayer(2->2),
      - final ReasoningLayer(2->1).

    If use_lifting=False, behaves exactly like the original implementation (no lifting).
    """
    def __init__(self, B: int, L: int = 2, S: int = 2, tau: float = 0.3,
                 pair: dict | None = None, gate_set: str = "6",
                 use_lifting: bool = True, lift_factor: float = 2.0):
        super().__init__()
        if L < 1:
            raise ValueError("L must be >= 1")

        self.gate_set = gate_set
        self.tau = float(tau)
        self.L = int(L)
        self.B_in = int(B)                    # original input bit-width
        self.use_lifting = bool(use_lifting)

        # --- Optional lifting: B_in -> B_eff ---
        if self.use_lifting:
            # simple heuristic: B_eff ≈ factor * B_in, but at least B_in
            B_eff = int(round(self.B_in * float(lift_factor)))
            if B_eff < self.B_in:
                B_eff = self.B_in
            self.lift = BitLifting(in_bits=self.B_in, out_bits=B_eff)
            self.B_effective = B_eff
        else:
            self.lift = None
            self.B_effective = self.B_in

        # --- Stack of layers ---
        self.layers = nn.ModuleList([])
        # First layer maps B_effective -> 2 bits
        self.layers.append(
            GeneralLayer(in_bits=self.B_effective, S=S, out_bits=2, tau=tau, pair=pair, gate_set=gate_set)
        )
        # Middle (L-2) keep 2 -> 2
        for _ in range(max(L - 2, 0)):
            self.layers.append(ReasoningLayer(S=S, out_bits=2, tau=tau, gate_set=gate_set))
        # Final 2 -> 1
        if L >= 2:
            self.layers.append(ReasoningLayer(S=S, out_bits=1, tau=tau, gate_set=gate_set))

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
        """
        xB: (BATCH, B_in) in {0,1}
        """
        if xB.size(-1) != self.B_in:
            raise ValueError(f"DepthStack expected input dim {self.B_in}, got {xB.size(-1)}")

        # Optional bit lifting
        if self.lift is not None:
            x = self.lift(xB)        # (BATCH, B_effective)
        else:
            x = xB

        dbg_list = []
        for li, layer in enumerate(self.layers):
            out = layer(x)
            if isinstance(layer, GeneralLayer):
                x, dbg = out
                dbg_list.append((dbg["outs"], dbg["L_rows"], dbg["unitWs"], dbg["PL"], dbg["PR"]))
            else:
                x, outs, L_rows, unitWs = out
                dbg_list.append((outs, L_rows, unitWs, None, None))
        return x, dbg_list  # final x is (BATCH,1)
