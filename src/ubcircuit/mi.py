from __future__ import annotations
from typing import List, Tuple
import torch

@torch.no_grad()
def _pmf(cnt: torch.Tensor) -> torch.Tensor:
    p = cnt.float()
    return p / p.sum().clamp_min(1e-12)

@torch.no_grad()
def mi_pair_y(X: torch.Tensor, y: torch.Tensor, i: int, j: int) -> float:
    """
    I((ai,aj); y) from full truth table. X:(N,B) in {0,1}, y:(N,1) in {0,1}
    """
    ai = X[:, i].long(); aj = X[:, j].long(); yy = y[:, 0].long()
    code = (ai << 1) | aj  # 00->0, 01->1, 10->2, 11->3
    cnt = torch.zeros(4, 2, dtype=torch.float64, device=X.device)
    for c in (0,1,2,3):
        mask = (code == c)
        cnt[c,0] = (mask & (yy == 0)).sum()
        cnt[c,1] = (mask & (yy == 1)).sum()
    p_xy = _pmf(cnt)           # (4,2)
    p_x  = p_xy.sum(1)         # (4,)
    p_y  = p_xy.sum(0)         # (2,)
    den = (p_x.unsqueeze(1) * p_y.unsqueeze(0)).clamp_min(1e-12)
    mi = (p_xy * (p_xy.clamp_min(1e-12) / den).log()).sum().item()
    return mi

@torch.no_grad()
def top_mi_pairs(X: torch.Tensor, y: torch.Tensor, S: int, disjoint: bool = True) -> List[Tuple[int,int]]:
    B = X.size(1)
    scores = []
    for i in range(B):
        for j in range(i+1, B):
            scores.append((mi_pair_y(X, y, i, j), i, j))
    scores.sort(reverse=True)
    pairs, used = [], set()
    for _, i, j in scores:
        if disjoint and (i in used or j in used):
            continue
        pairs.append((i, j)); used.add(i); used.add(j)
        if len(pairs) >= S: break
    k = 0
    while len(pairs) < S and k < len(scores):
        _, i, j = scores[k]; k += 1
        pairs.append((i, j))
    return pairs[:S]

@torch.no_grad()
def priors_from_pairs(pairs: List[Tuple[int,int]], B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-hot priors for PL/PR given a (potentially disjoint) pair list.
    Returns PL_prior, PR_prior in R^{S x B} (probabilities, sum=1 per row).
    """
    S = len(pairs)
    PL = torch.zeros(S, B); PR = torch.zeros(S, B)
    for s, (i, j) in enumerate(pairs):
        PL[s, i] = 1.0; PR[s, j] = 1.0
    return PL, PR
