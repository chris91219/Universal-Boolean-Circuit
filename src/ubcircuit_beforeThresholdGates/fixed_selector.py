from __future__ import annotations
import torch
import torch.nn as nn

class FixedPairSelector(nn.Module):
    def __init__(self, B: int, pairs: list[tuple[int,int]]):
        super().__init__()
        self.B = int(B); self.S = len(pairs)
        PL = torch.zeros(self.S, B); PR = torch.zeros(self.S, B)
        for s,(i,j) in enumerate(pairs):
            PL[s,i] = 1.0; PR[s,j] = 1.0
        self.register_buffer("PL_fix", PL)
        self.register_buffer("PR_fix", PR)

    def forward(self, xB: torch.Tensor) -> torch.Tensor:
        a = xB @ self.PL_fix.t()          # (batch,S)
        b = xB @ self.PR_fix.t()          # (batch,S)
        return torch.stack([a, b], dim=-1)  # (batch,S,2)
