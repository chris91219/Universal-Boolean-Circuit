import torch

from ubcircuit import tasks as T
from ubcircuit.boolean_prims16 import PRIMS16
from ubcircuit.readout import compose_readout_expr, decoded_readout_metrics, normalize_expr


def _argmax_vec(idx: int, size: int) -> torch.Tensor:
    v = torch.full((size,), -10.0)
    v[idx] = 10.0
    return v


def test_decoded_readout_metrics_match_hard_or_readout():
    B = 2
    S = 2

    unitWs0 = [
        _argmax_vec(PRIMS16.index("A"), len(PRIMS16)),
        _argmax_vec(PRIMS16.index("B"), len(PRIMS16)),
    ]
    Lrows0 = torch.stack([
        _argmax_vec(0, S),
        _argmax_vec(1, S),
    ], dim=0)
    PL = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    PR = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    unitWsF = [
        _argmax_vec(PRIMS16.index("OR"), len(PRIMS16)),
        _argmax_vec(PRIMS16.index("FALSE"), len(PRIMS16)),
    ]
    LrowsF = torch.stack([_argmax_vec(0, S)], dim=0)

    dbg = [
        (torch.empty(0), Lrows0, unitWs0, PL, PR),
        (torch.empty(0), LrowsF, unitWsF, None, None),
    ]

    pred_expr_raw = compose_readout_expr(B, dbg, [0.1, 0.1], PRIMS16)
    assert normalize_expr(pred_expr_raw) == "a0 | a1"

    _, y_true = T.truth_table_from_formula(B, "(a0 | a1)")
    decoded_row_acc, decoded_em = decoded_readout_metrics(B, pred_expr_raw, y_true)

    assert decoded_row_acc == 1.0
    assert decoded_em == 1


def test_compose_readout_expr_uses_lifted_symbols():
    B = 2
    S = 2

    lift_W = torch.stack([
        _argmax_vec(0, 2 * B),
        _argmax_vec(B + 0, 2 * B),
        _argmax_vec(1, 2 * B),
    ], dim=0)

    unitWs0 = [
        _argmax_vec(PRIMS16.index("A"), len(PRIMS16)),
        _argmax_vec(PRIMS16.index("B"), len(PRIMS16)),
    ]
    Lrows0 = torch.stack([
        _argmax_vec(0, S),
        _argmax_vec(1, S),
    ], dim=0)
    PL = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    PR = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    unitWsF = [
        _argmax_vec(PRIMS16.index("OR"), len(PRIMS16)),
        _argmax_vec(PRIMS16.index("FALSE"), len(PRIMS16)),
    ]
    LrowsF = torch.stack([_argmax_vec(0, S)], dim=0)

    dbg = [
        (torch.empty(0), Lrows0, unitWs0, PL, PR),
        (torch.empty(0), LrowsF, unitWsF, None, None),
    ]

    pred_expr_raw = compose_readout_expr(B, dbg, [0.1, 0.1], PRIMS16, lift_W=lift_W)
    assert normalize_expr(pred_expr_raw) == "(~a0) | a1"

    _, y_true = T.truth_table_from_formula(B, "((1 - a0) | a1)")
    decoded_row_acc, decoded_em = decoded_readout_metrics(B, pred_expr_raw, y_true)

    assert decoded_row_acc == 1.0
    assert decoded_em == 1
