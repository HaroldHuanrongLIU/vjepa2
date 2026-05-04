from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
from torch import nn

from src.models.surgwmbench_vjepa_ac import SurgVJEPA2AC
from surgwmbench_benchmark.future_model_helpers import FutureFrameDecoder
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class VJEPA2ACFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the SurgWMBench V-JEPA2-AC core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.core = SurgVJEPA2AC(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            action_dim=3,
            freeze_encoder=False,
        )
        self.frame_decoder = FutureFrameDecoder(config.latent_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        _, _, _, height, width = frames.shape
        rollout = self.core.rollout(
            frames_context=frames,
            coords_context=batch["context_coords_norm"],
            sampled_indices=batch["context_frame_indices"],
            horizon=batch["future_frame_indices"].shape[1],
            mode="policy_rollout",
        )
        pred_coords = rollout["pred_coords_norm"][:, 1:]
        pred_frames = self.frame_decoder(rollout["pred_latents"], (height, width))
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return VJEPA2ACFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("vjepa2_ac", "VJEPA2ACFuturePredictionCore", "src.datasets.surgwmbench", make_model))
