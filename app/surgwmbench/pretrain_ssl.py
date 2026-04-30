"""Small SurgWMBench SSL pretraining adapter for raw videos or clip frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.surgwmbench_collators import collate_ssl_video
from src.datasets.surgwmbench_video import SurgWMBenchRawVideoDataset
from src.models.surgwmbench_vjepa_ac import load_surgwmbench_encoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _maybe_subset(dataset: SurgWMBenchRawVideoDataset, max_samples: int | None) -> SurgWMBenchRawVideoDataset | Subset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


class SurgWMBenchSSLModel(nn.Module):
    """Temporal latent next-step SSL objective for SurgWMBench smoke pretraining."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        latent_dim: int,
        hidden_dim: int,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        encoder_dim = int(getattr(encoder, "embed_dim", latent_dim))
        self.latent_proj = nn.Identity() if encoder_dim == latent_dim else nn.Linear(encoder_dim, latent_dim)
        self.predictor = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "encode_frames"):
            latents = self.encoder.encode_frames(frames)
            return self.latent_proj(latents)
        if frames.ndim != 5:
            raise ValueError(f"frames must have shape [B, T, 3, H, W], got {tuple(frames.shape)}")
        batch_size, timesteps = frames.shape[:2]
        tokens = self.encoder(frames.flatten(0, 1))
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens.ndim == 3:
            pooled = tokens.mean(dim=1)
        elif tokens.ndim == 2:
            pooled = tokens
        else:
            raise ValueError(f"Unsupported encoder output shape: {tuple(tokens.shape)}")
        return self.latent_proj(pooled.view(batch_size, timesteps, -1))

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.encode_frames(frames)
        pred_sequence, _ = self.predictor(latents[:, :-1])
        pred_latents = self.prediction_head(pred_sequence)
        return {
            "latents": latents,
            "pred_latents": pred_latents,
            "target_latents": latents[:, 1:].detach(),
        }


def build_model(args: argparse.Namespace, device: torch.device) -> SurgWMBenchSSLModel:
    encoder = load_surgwmbench_encoder(
        encoder_name=args.encoder_name,
        checkpoint_path=args.encoder_checkpoint,
        checkpoint_key=args.encoder_checkpoint_key,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        freeze=args.freeze_encoder,
        device=device,
    )
    return SurgWMBenchSSLModel(
        encoder=encoder,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        freeze_encoder=args.freeze_encoder,
    ).to(device)


def _run_epoch(
    model: SurgWMBenchSSLModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if args.freeze_encoder:
        model.encoder.eval()
    use_amp = bool(args.precision == "amp" and device.type == "cuda")
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch["frames"])
                loss = F.smooth_l1_loss(outputs["pred_latents"], outputs["target_latents"])
            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
        batch_size = int(batch["frames"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return {"loss": total_loss / max(count, 1)}


def pretrain_ssl(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = _device(args.device)
    backend = "opencv_or_frames" if args.ssl_source == "raw_videos" else "frames"
    dataset = SurgWMBenchRawVideoDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        manifest=args.manifest,
        source_video_manifest=args.source_video_manifest,
        clip_length=args.clip_length,
        stride=args.stride,
        image_size=args.image_size,
        backend=backend,
        max_videos=args.max_videos,
        max_clips_per_video=args.max_clips_per_video,
    )
    dataset = _maybe_subset(dataset, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ssl_video,
    )
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=bool(args.precision == "amp" and device.type == "cuda"))

    history: list[dict[str, float]] = []
    for _epoch in range(args.epochs):
        history.append(_run_epoch(model, loader, device, args, optimizer=optimizer, scaler=scaler))

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "model_state": model.state_dict(),
            "model_config": {
                "encoder_name": args.encoder_name,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "image_size": args.image_size,
                "freeze_encoder": args.freeze_encoder,
            },
            "train_args": vars(args),
            "history": history,
        },
        output,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", default="manifests/train.jsonl")
    parser.add_argument("--source-video-manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ssl-source", choices=("raw_videos", "extracted_frames"), default="raw_videos")
    parser.add_argument("--encoder-name", default="dummy")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--encoder-checkpoint-key", default="target_encoder")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="amp")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-clips-per-video", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    output = pretrain_ssl(build_parser().parse_args())
    print(f"Saved SSL checkpoint to {output}")


if __name__ == "__main__":
    main()
