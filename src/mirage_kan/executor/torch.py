"""Torch batch execution for the typed factor AST."""

from __future__ import annotations

import pandas as pd
import torch

from mirage_kan.data import PitPanel
from mirage_kan.dsl import AstNode, Evaluation
from mirage_kan.dsl.core import LEAF_TYPES


def evaluate_torch(
    program: AstNode, panel: PitPanel, *, device: str | torch.device = "cpu"
) -> Evaluation:
    """Evaluate a validated AST with Torch and return panel-indexed results."""
    program.validate()
    target = torch.device(device)
    lag_cache: dict[int, torch.Tensor] = {}

    def lag_indices(periods: int) -> torch.Tensor:
        if periods not in lag_cache:
            indices = [-1] * len(panel.raw)
            history: dict[object, list[int]] = {}
            instruments = panel.raw.index.get_level_values("instrument")
            for position, instrument in enumerate(instruments):
                prior = history.setdefault(instrument, [])
                if len(prior) >= periods:
                    indices[position] = prior[-periods]
                prior.append(position)
            lag_cache[periods] = torch.tensor(
                indices, dtype=torch.long, device=target
            )
        return lag_cache[periods]

    def lagged(
        values: torch.Tensor, support: torch.Tensor, periods: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = lag_indices(periods)
        valid = indices.ge(0)
        gather = indices.clamp_min(0)
        return (
            values[gather].masked_fill(~valid, torch.nan),
            support[gather] & valid,
        )

    def result_name(node: AstNode) -> str | None:
        if node.op in LEAF_TYPES:
            return node.op.lower()
        if node.op == "Constant":
            return None
        if node.op in {"Add", "Sub", "SafeDiv"}:
            left = result_name(node.children[0])
            return left if left == result_name(node.children[1]) else None
        return result_name(node.children[0])

    def visit(node: AstNode) -> tuple[torch.Tensor, torch.Tensor]:
        if node.op in LEAF_TYPES:
            field = node.op.lower()
            values = torch.as_tensor(
                panel.field(field).to_numpy(dtype=float, copy=True),
                dtype=torch.float64,
                device=target,
            )
            support = torch.as_tensor(
                panel.observed[field].to_numpy(dtype=bool, copy=True),
                dtype=torch.bool,
                device=target,
            )
        elif node.op == "Constant":
            values = torch.full(
                (len(panel.raw),),
                float(node.params["value"]),
                dtype=torch.float64,
                device=target,
            )
            support = torch.ones(len(panel.raw), dtype=torch.bool, device=target)
        else:
            children = tuple(visit(child) for child in node.children)
            if node.op in {"Add", "Sub"}:
                left, right = children
                values = left[0] + right[0] if node.op == "Add" else left[0] - right[0]
                support = left[1] & right[1]
            elif node.op == "SafeDiv":
                numerator, denominator = children
                domain = torch.isfinite(denominator[0]) & denominator[0].ne(0)
                support = numerator[1] & denominator[1] & domain
                values = numerator[0] / torch.where(
                    domain, denominator[0], torch.ones_like(denominator[0])
                )
            elif node.op in {"Delay", "Delta", "Return"}:
                child = children[0]
                previous = lagged(child[0], child[1], int(node.params["window"]))
                if node.op == "Delay":
                    values, support = previous
                elif node.op == "Delta":
                    values = child[0] - previous[0]
                    support = child[1] & previous[1]
                else:
                    domain = torch.isfinite(previous[0]) & previous[0].ne(0)
                    support = child[1] & previous[1] & domain
                    values = child[0] / torch.where(
                        domain, previous[0], torch.ones_like(previous[0])
                    ) - 1.0
            elif node.op == "TsMean":
                child = children[0]
                window = int(node.params["window"])
                indices = torch.stack(
                    [
                        torch.arange(len(panel.raw), device=target),
                        *(lag_indices(period) for period in range(1, window)),
                    ],
                    dim=1,
                )
                valid = indices.ge(0)
                gather = indices.clamp_min(0)
                values = child[0][gather].mean(dim=1)
                support = valid.all(dim=1) & child[1][gather].all(dim=1)
            elif node.op == "CSRank":
                child = children[0]
                values = torch.full_like(child[0], torch.nan)
                support = torch.zeros_like(child[1])
                date_groups: dict[object, list[int]] = {}
                dates = panel.raw.index.get_level_values("datetime")
                for position, date in enumerate(dates):
                    date_groups.setdefault(date, []).append(position)
                for positions in date_groups.values():
                    group = torch.tensor(positions, dtype=torch.long, device=target)
                    eligible = group[child[1][group]]
                    if eligible.numel() == 0:
                        continue
                    ranked = child[0][eligible]
                    less = (ranked.unsqueeze(0) < ranked.unsqueeze(1)).sum(
                        dim=1
                    ).to(ranked.dtype)
                    equal = (ranked.unsqueeze(0) == ranked.unsqueeze(1)).sum(
                        dim=1
                    ).to(ranked.dtype)
                    ranks = (less + 1 + (equal - 1) / 2) / eligible.numel()
                    values[eligible] = ranks
                    support[eligible] = True
            else:
                raise AssertionError(
                    f"validated but unimplemented Torch operator: {node.op}"
                )
        support = support & torch.isfinite(values)
        return values.masked_fill(~support, torch.nan), support

    values, support = visit(program)
    return Evaluation(
        pd.Series(
            values.cpu().numpy(), index=panel.raw.index, name=result_name(program)
        ),
        pd.Series(support.cpu().numpy(), index=panel.raw.index, name="support"),
    )
