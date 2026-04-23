from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int = 30) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def save_model_checkpoint(path: Path | str, model: FraudMLP, config: dict, feature_columns: list[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": len(feature_columns),
        "feature_columns": feature_columns,
        "config": config,
    }
    torch.save(checkpoint, target)
    return target


def load_model_checkpoint(path: Path | str, device: str = "cpu") -> tuple[FraudMLP, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = FraudMLP(input_dim=checkpoint.get("input_dim", 30))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
