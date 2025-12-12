# src/ubcircuit/lifting.py

from __future__ import annotations
import torch
import torch.nn as nn

class BitLifting(nn.Module):
    """
    Boolean bit-lifting layer:
      x in R^B  ->  y in R^{B_up}
    where each lifted channel is a softmax-weighted copy/negation of some input bit.

    y = softmax(W) @ [x, 1-x]
    """
    def __init__(self, in_bits: int, out_bits: int):
        super().__init__()
        self.in_bits = int(in_bits)
        self.out_bits = int(out_bits)
        # W: (B_up, 2B)
        self.W = nn.Parameter(torch.zeros(self.out_bits, 2 * self.in_bits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, B) with entries in [0,1]
        Returns:
            y: (batch, B_up)
        """
        if x.dim() != 2 or x.size(-1) != self.in_bits:
            raise ValueError(f"BitLifting expects (BATCH,{self.in_bits}), got {tuple(x.shape)}")

        # v = [x, 1-x]  \in R^{2B}
        v = torch.cat([x, 1.0 - x], dim=-1)      # (batch, 2B)
        P = torch.softmax(self.W, dim=-1)        # (B_up, 2B)
        y = v @ P.t()                            # (batch, B_up)
        return y
