"""Power schedule utilities for scheduled Muon experiments.

The schedule controls the singular-value exponent p in transforms of the form:
    G_p = U diag(s_i^p) V^T
where G = U diag(s_i) V^T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class AnnealingPowerSchedule:
    p_start: float
    p_end: float
    total_steps: int
    mode: str = "linear"  # linear | cosine

    def value(self, step: int, entropy: float | None = None) -> float:
        del entropy
        if self.total_steps <= 1:
            return self.p_end
        t = max(0.0, min(1.0, step / float(self.total_steps - 1)))
        if self.mode == "cosine":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        return self.p_start + (self.p_end - self.p_start) * t


@dataclass
class FixedAlternatingPowerSchedule:
    p_low: float
    p_high: float
    period_steps: int

    def value(self, step: int, entropy: float | None = None) -> float:
        del entropy
        if self.period_steps <= 0:
            return self.p_low
        phase = (step // self.period_steps) % 2
        return self.p_low if phase == 0 else self.p_high


class EntropyAlternatingPowerSchedule:
    """Hysteresis switching between p_low and p_high using entropy thresholds.

    - If currently at p_low and entropy >= entropy_high, switch to p_high.
    - If currently at p_high and entropy <= entropy_low, switch to p_low.
    """

    def __init__(
        self,
        p_low: float,
        p_high: float,
        entropy_low: float,
        entropy_high: float,
        initial_mode: str = "low",
    ) -> None:
        if entropy_low >= entropy_high:
            raise ValueError("entropy_low must be < entropy_high")
        self.p_low = p_low
        self.p_high = p_high
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.current_p = p_low if initial_mode == "low" else p_high

    def value(self, step: int, entropy: float | None = None) -> float:
        del step
        if entropy is None:
            return self.current_p
        if self.current_p == self.p_low and entropy >= self.entropy_high:
            self.current_p = self.p_high
        elif self.current_p == self.p_high and entropy <= self.entropy_low:
            self.current_p = self.p_low
        return self.current_p


def normalized_svd_entropy(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """Return entropy in [0, 1] from singular-value distribution."""
    if matrix.ndim == 4:
        matrix = matrix.view(matrix.size(0), -1)
    if matrix.ndim != 2:
        raise ValueError("normalized_svd_entropy expects a 2D matrix")

    s = torch.linalg.svdvals(matrix.float())
    if s.numel() == 0:
        return 0.0
    probs = s.clamp_min(eps)
    probs = probs / probs.sum()
    entropy = -(probs * probs.log()).sum()
    entropy = entropy / math.log(float(probs.numel()))
    return float(entropy)


def mean_grad_svd_entropy(params: Iterable[torch.nn.Parameter], max_matrices: int = 8) -> float | None:
    """Average SVD entropy over up to max_matrices gradient matrices."""
    values: list[float] = []
    for p in params:
        if p.grad is None or p.grad.ndim < 2:
            continue
        g = p.grad
        if g.ndim == 4:
            g = g.view(g.size(0), -1)
        if g.ndim != 2:
            continue
        try:
            values.append(normalized_svd_entropy(g))
        except RuntimeError:
            # Some backends may not support SVD for all shapes/dtypes.
            continue
        if len(values) >= max_matrices:
            break
    if not values:
        return None
    return float(sum(values) / len(values))


def build_power_schedule(
    schedule_type: str,
    *,
    total_steps: int,
    p_start: float,
    p_end: float,
    p_low: float,
    p_high: float,
    alternation_period: int,
    entropy_low: float,
    entropy_high: float,
    entropy_initial_mode: str,
):
    schedule_type = schedule_type.lower()
    if schedule_type == "anneal":
        return AnnealingPowerSchedule(p_start=p_start, p_end=p_end, total_steps=total_steps, mode="linear")
    if schedule_type == "anneal_cosine":
        return AnnealingPowerSchedule(p_start=p_start, p_end=p_end, total_steps=total_steps, mode="cosine")
    if schedule_type == "fixed_alternating":
        return FixedAlternatingPowerSchedule(p_low=p_low, p_high=p_high, period_steps=alternation_period)
    if schedule_type == "entropy_alternating":
        return EntropyAlternatingPowerSchedule(
            p_low=p_low,
            p_high=p_high,
            entropy_low=entropy_low,
            entropy_high=entropy_high,
            initial_mode=entropy_initial_mode,
        )
    if schedule_type == "constant":
        return AnnealingPowerSchedule(p_start=p_start, p_end=p_start, total_steps=max(2, total_steps), mode="linear")
    raise ValueError(f"unknown schedule_type: {schedule_type}")
