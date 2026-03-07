"""Small local smoke test for scheduled Muon power schedules.

Runs a tiny MLP on random data for a few steps with:
- annealing schedule
- fixed alternating schedule
- entropy alternating schedule
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muon import SingleDeviceMuonWithAuxAdam
from power_schedule import build_power_schedule, mean_grad_svd_entropy


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 32, d_hidden: int = 64, d_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_optimizer(model: nn.Module) -> tuple[SingleDeviceMuonWithAuxAdam, list[nn.Parameter]]:
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    matrix_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    non_matrix_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {
            "params": matrix_params,
            "use_muon": True,
            "lr": 0.03,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "weight_decay": 0.0,
            "use_power": True,
            "power_p": 1.0,
            "svd_eps": 1e-8,
        },
        {
            "params": non_matrix_params,
            "use_muon": False,
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
    ]
    return SingleDeviceMuonWithAuxAdam(optim_groups), matrix_params


def run_schedule(name: str, schedule_type: str, steps: int = 30) -> None:
    torch.manual_seed(123)
    model = TinyMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer, matrix_params = make_optimizer(model)

    schedule = build_power_schedule(
        schedule_type,
        total_steps=steps,
        p_start=1.0,
        p_end=0.0,
        p_low=0.0,
        p_high=1.0,
        alternation_period=5,
        entropy_low=0.45,
        entropy_high=0.65,
        entropy_initial_mode="low",
    )

    print(f"\n=== {name} ({schedule_type}) ===")
    for step in range(steps):
        x = torch.randn(32, 32)
        y = torch.randint(0, 8, (32,))

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        entropy = None
        if schedule_type == "entropy_alternating":
            entropy = mean_grad_svd_entropy(matrix_params, max_matrices=8)
        p_val = schedule.value(step, entropy)

        for group in optimizer.param_groups:
            if group.get("use_muon", False):
                group["power_p"] = p_val
                group["use_power"] = True

        optimizer.step()

        if step % 5 == 0:
            extra = f", entropy={entropy:.3f}" if entropy is not None else ""
            print(f"step={step:02d}, loss={loss.item():.4f}, p={p_val:.3f}{extra}")


def main() -> None:
    run_schedule("Annealing", "anneal")
    run_schedule("Fixed alternating", "fixed_alternating")
    run_schedule("Entropy alternating", "entropy_alternating")


if __name__ == "__main__":
    main()
