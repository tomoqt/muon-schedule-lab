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


class EntropyLawPowerSchedule:
    """Continuous entropy->p mapping between p_low and p_high.

    Entropy is normalized into [0, 1] using [entropy_low, entropy_high], then
    transformed by a selected law and mapped onto [p_low, p_high].
    """

    def __init__(
        self,
        p_low: float,
        p_high: float,
        entropy_low: float,
        entropy_high: float,
        law: str = "linear",
        gamma: float = 1.0,
        sigmoid_temp: float = 8.0,
        linear_coeff: float = 1.0,
        osc_amp: float = 0.0,
        osc_period: int = 0,
        ema_beta: float = 0.0,
    ) -> None:
        if entropy_low >= entropy_high:
            raise ValueError("entropy_low must be < entropy_high")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        if sigmoid_temp <= 0.0:
            raise ValueError("sigmoid_temp must be > 0")
        if osc_amp < 0.0:
            raise ValueError("osc_amp must be >= 0")
        if osc_period < 0:
            raise ValueError("osc_period must be >= 0")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0, 1)")
        self.p_low = p_low
        self.p_high = p_high
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high
        self.law = law.lower()
        if self.law not in {"linear", "power", "sigmoid"}:
            raise ValueError(f"unknown entropy law: {law}")
        self.gamma = gamma
        self.sigmoid_temp = sigmoid_temp
        self.linear_coeff = linear_coeff
        self.osc_amp = osc_amp
        self.osc_period = osc_period
        self.ema_beta = ema_beta
        self.current_p = p_low

    def _apply_law(self, t: float) -> float:
        # t is already normalized to [0, 1].
        if self.law == "linear":
            return t
        if self.law == "power":
            return t ** self.gamma
        # law == "sigmoid"
        z = 2.0 * t - 1.0
        lo = 1.0 / (1.0 + math.exp(self.sigmoid_temp))
        hi = 1.0 / (1.0 + math.exp(-self.sigmoid_temp))
        s = 1.0 / (1.0 + math.exp(-self.sigmoid_temp * z))
        if hi <= lo:
            return t
        return (s - lo) / (hi - lo)

    def value(self, step: int, entropy: float | None = None) -> float:
        if entropy is None:
            return self.current_p
        t_raw = (entropy - self.entropy_low) / (self.entropy_high - self.entropy_low)
        t_raw = max(0.0, min(1.0, t_raw))
        # linear_coeff<0 flips the entropy->p direction around the midpoint.
        t = 0.5 + self.linear_coeff * (t_raw - 0.5)
        t = max(0.0, min(1.0, t))
        mapped = self._apply_law(t)
        if self.osc_amp > 0.0 and self.osc_period > 0:
            phase = 2.0 * math.pi * (step / float(self.osc_period))
            mapped = mapped + self.osc_amp * math.sin(phase)
            mapped = max(0.0, min(1.0, mapped))
        target_p = self.p_low + (self.p_high - self.p_low) * mapped
        if self.ema_beta > 0.0:
            self.current_p = self.ema_beta * self.current_p + (1.0 - self.ema_beta) * target_p
        else:
            self.current_p = target_p
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
    entropy_law: str,
    entropy_gamma: float,
    entropy_sigmoid_temp: float,
    entropy_linear_coeff: float,
    entropy_osc_amp: float,
    entropy_osc_period: int,
    entropy_ema_beta: float,
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
    if schedule_type == "entropy_law":
        return EntropyLawPowerSchedule(
            p_low=p_low,
            p_high=p_high,
            entropy_low=entropy_low,
            entropy_high=entropy_high,
            law=entropy_law,
            gamma=entropy_gamma,
            sigmoid_temp=entropy_sigmoid_temp,
            linear_coeff=entropy_linear_coeff,
            osc_amp=entropy_osc_amp,
            osc_period=entropy_osc_period,
            ema_beta=entropy_ema_beta,
        )
    if schedule_type == "constant":
        return AnnealingPowerSchedule(p_start=p_start, p_end=p_start, total_steps=max(2, total_steps), mode="linear")
    raise ValueError(f"unknown schedule_type: {schedule_type}")
