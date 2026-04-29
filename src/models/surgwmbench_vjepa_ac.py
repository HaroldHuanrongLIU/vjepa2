"""SurgWMBench V-JEPA 2-AC-style model skeleton."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from src.utils.checkpoint_loader import robust_checkpoint_loader

RolloutMode = Literal["teacher_forced_actions", "policy_rollout"]


class DummyFrameEncoder(nn.Module):
    """Small deterministic frame encoder for CPU tests and smoke runs."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.proj = nn.Linear(3, latent_dim)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``frames`` with shape ``[B, T, 3, H, W]`` into ``[B, T, D]``."""

        if frames.ndim != 5:
            raise ValueError(f"frames must have shape [B, T, 3, H, W], got {tuple(frames.shape)}")
        pooled = frames.mean(dim=(-1, -2))
        return self.proj(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DummyFrameEncoder.forward expects [B, 3, H, W], got {tuple(x.shape)}")
        return self.proj(x.mean(dim=(-1, -2))).unsqueeze(1)


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _checkpoint_encoder_state(checkpoint: Any, checkpoint_key: str = "target_encoder") -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in (checkpoint_key, "encoder", "target_encoder", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return _clean_state_dict(checkpoint)
    raise ValueError("Checkpoint does not contain an encoder-compatible state_dict")


def load_surgwmbench_encoder(
    encoder_name: str = "dummy",
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_key: str = "target_encoder",
    image_size: int = 384,
    patch_size: int = 16,
    latent_dim: int = 64,
    freeze: bool = True,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Load a SurgWMBench visual encoder without requiring network access."""

    if encoder_name == "dummy":
        encoder: nn.Module = DummyFrameEncoder(latent_dim=latent_dim)
    else:
        from src.models import vision_transformer as vit_encoder

        if encoder_name not in vit_encoder.__dict__:
            raise ValueError(f"Unknown encoder_name={encoder_name!r}")
        encoder = vit_encoder.__dict__[encoder_name](
            img_size=image_size,
            patch_size=patch_size,
            num_frames=1,
            tubelet_size=1,
        )

    if checkpoint_path is not None:
        checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location="cpu")
        state_dict = _checkpoint_encoder_state(checkpoint, checkpoint_key=checkpoint_key)
        msg = encoder.load_state_dict(state_dict, strict=False)
        if getattr(msg, "unexpected_keys", None):
            warnings.warn(
                "Loaded encoder with unexpected keys ignored. "
                "If this is a V-JEPA 2-AC checkpoint, SurgWMBench initializes a new action predictor.",
                stacklevel=2,
            )

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

    if device is not None:
        encoder = encoder.to(device)
    return encoder


class SurgVJEPA2AC(nn.Module):
    """Minimal V-JEPA 2-AC-style world model for SurgWMBench sparse anchors."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        *,
        latent_dim: int = 64,
        coord_dim: int = 2,
        action_dim: int = 3,
        hidden_dim: int = 128,
        freeze_encoder: bool = True,
        pooling: Literal["mean", "first"] = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else DummyFrameEncoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.coord_dim = coord_dim
        self.action_dim = action_dim
        self.pooling = pooling

        encoder_dim = int(getattr(self.encoder, "embed_dim", latent_dim))
        self.latent_proj = nn.Identity() if encoder_dim == latent_dim else nn.Linear(encoder_dim, latent_dim)
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.predictor = nn.GRUCell(latent_dim * 3, latent_dim)
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, coord_dim), nn.Sigmoid()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frame batch ``[B, T, 3, H, W]`` into per-frame latents ``[B, T, D]``."""

        if hasattr(self.encoder, "encode_frames"):
            latents = self.encoder.encode_frames(frames)
            return self.latent_proj(latents)

        if frames.ndim != 5:
            raise ValueError(f"frames must have shape [B, T, 3, H, W], got {tuple(frames.shape)}")
        batch_size, timesteps = frames.shape[:2]
        flat_frames = frames.flatten(0, 1)
        tokens = self.encoder(flat_frames)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens.ndim == 3:
            if self.pooling == "first":
                pooled = tokens[:, 0]
            elif self.pooling == "mean":
                pooled = tokens.mean(dim=1)
            else:
                raise ValueError(f"Unsupported pooling={self.pooling!r}")
        elif tokens.ndim == 2:
            pooled = tokens
        else:
            raise ValueError(f"Unsupported encoder output shape: {tuple(tokens.shape)}")
        return self.latent_proj(pooled.view(batch_size, timesteps, -1))

    def _predict_next(self, latent: torch.Tensor, coord: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        coord_emb = self.coord_encoder(coord)
        action_emb = self.action_encoder(action)
        return self.predictor(torch.cat([latent, coord_emb, action_emb], dim=-1), latent)

    def predict_actions(self, latents: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        coord_emb = self.coord_encoder(coords)
        return self.policy_head(torch.cat([latents, coord_emb], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["frames"]
        coords = batch["coords_norm"]
        actions = batch["actions_delta_dt"]
        target_latents = self.encode_frames(frames)

        latent = target_latents[:, 0]
        pred_latents = []
        pred_next_coords = []
        for step in range(actions.shape[1]):
            latent = self._predict_next(latent, coords[:, step], actions[:, step])
            pred_latents.append(latent)
            pred_next_coords.append(self.coord_head(latent))

        pred_latents_tensor = torch.stack(pred_latents, dim=1)
        pred_next_coords_tensor = torch.stack(pred_next_coords, dim=1)
        pred_coords = torch.cat([coords[:, :1], pred_next_coords_tensor], dim=1)
        pred_actions = self.predict_actions(target_latents[:, :-1], coords[:, :-1])

        return {
            "target_latents": target_latents,
            "pred_latents": pred_latents_tensor,
            "pred_next_coords_norm": pred_next_coords_tensor,
            "pred_coords_norm": pred_coords,
            "pred_actions_delta_dt": pred_actions,
        }

    def rollout(
        self,
        frames_context: torch.Tensor,
        coords_context: torch.Tensor,
        *,
        sampled_indices: torch.Tensor | None = None,
        horizon: int = 19,
        mode: RolloutMode = "policy_rollout",
        actions: torch.Tensor | None = None,
        goal_coord: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del sampled_indices, goal_coord
        if mode == "teacher_forced_actions" and actions is None:
            raise ValueError("teacher_forced_actions rollout requires actions")
        if frames_context.ndim != 5 or coords_context.ndim != 3:
            raise ValueError("frames_context must be [B, C, 3, H, W] and coords_context must be [B, C, 2]")

        context_latents = self.encode_frames(frames_context)
        latent = context_latents[:, -1]
        coord = coords_context[:, -1]
        pred_coords = [coord]
        pred_latents = []
        pred_actions = []

        for step in range(horizon):
            if mode == "teacher_forced_actions":
                assert actions is not None
                action = actions[:, step]
            elif mode == "policy_rollout":
                action = self.predict_actions(latent.unsqueeze(1), coord.unsqueeze(1)).squeeze(1)
            else:
                raise ValueError(f"Unsupported rollout mode: {mode}")
            latent = self._predict_next(latent, coord, action)
            coord = self.coord_head(latent)
            pred_actions.append(action)
            pred_latents.append(latent)
            pred_coords.append(coord)

        return {
            "pred_coords_norm": torch.stack(pred_coords, dim=1),
            "pred_latents": torch.stack(pred_latents, dim=1),
            "pred_actions": torch.stack(pred_actions, dim=1),
        }
