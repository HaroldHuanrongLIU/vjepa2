"""Small SurgWMBench SSL pretraining adapter for raw videos or clip frames."""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.datasets.surgwmbench_collators import collate_ssl_video
from src.datasets.surgwmbench_video import SurgWMBenchRawVideoDataset
from src.models.surgwmbench_vjepa_ac import load_surgwmbench_encoder


def _setup_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed when launched via torchrun; otherwise no-op."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return world_size, rank, local_rank, True
    return 1, 0, 0, False


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


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
    """V-JEPA-flavored SSL: EMA target encoder + temporal frame masking on the online branch.

    The online encoder sees a randomly frame-masked clip and predicts each next-step latent
    from the prior context. The target encoder is an EMA copy of the online encoder, sees the
    full unmasked clip, and produces stop-gradient targets. EMA + masking together break the
    trivial constant-latent collapse of pure self-distillation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        latent_dim: int,
        hidden_dim: int,
        freeze_encoder: bool = False,
        ema_decay: float = 0.996,
        mask_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.ema_decay = float(ema_decay)
        self.mask_ratio = float(mask_ratio)
        encoder_dim = int(getattr(encoder, "embed_dim", latent_dim))
        self.latent_proj = nn.Identity() if encoder_dim == latent_dim else nn.Linear(encoder_dim, latent_dim)
        self.predictor = nn.GRU(input_size=latent_dim, hidden_size=latent_dim, batch_first=True)
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.target_encoder = copy.deepcopy(encoder)
        self.target_latent_proj = copy.deepcopy(self.latent_proj)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_latent_proj.parameters():
            param.requires_grad = False
        self.target_encoder.eval()
        self.target_latent_proj.eval()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    @staticmethod
    def _encode_with(encoder: nn.Module, proj: nn.Module, frames: torch.Tensor) -> torch.Tensor:
        if hasattr(encoder, "encode_frames"):
            return proj(encoder.encode_frames(frames))
        if frames.ndim != 5:
            raise ValueError(f"frames must have shape [B, T, 3, H, W], got {tuple(frames.shape)}")
        batch_size, timesteps = frames.shape[:2]
        tokens = encoder(frames.flatten(0, 1))
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens.ndim == 3:
            pooled = tokens.mean(dim=1)
        elif tokens.ndim == 2:
            pooled = tokens
        else:
            raise ValueError(f"Unsupported encoder output shape: {tuple(tokens.shape)}")
        return proj(pooled.view(batch_size, timesteps, -1))

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode_with(self.encoder, self.latent_proj, frames)

    def _mask_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0.0 or not self.training:
            return frames
        batch_size, timesteps = frames.shape[:2]
        keep = torch.rand(batch_size, timesteps, device=frames.device) >= self.mask_ratio
        # Guarantee at least the first frame stays so the predictor has a real context start.
        keep[:, 0] = True
        mask = keep.view(batch_size, timesteps, 1, 1, 1).to(frames.dtype)
        return frames * mask

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        masked = self._mask_frames(frames)
        online_latents = self._encode_with(self.encoder, self.latent_proj, masked)
        with torch.no_grad():
            target_latents = self._encode_with(self.target_encoder, self.target_latent_proj, frames)
        pred_sequence, _ = self.predictor(online_latents[:, :-1])
        pred_latents = self.prediction_head(pred_sequence)
        return {
            "latents": online_latents,
            "pred_latents": F.normalize(pred_latents, dim=-1),
            "target_latents": F.normalize(target_latents[:, 1:].detach(), dim=-1),
        }

    @torch.no_grad()
    def update_target_ema(self) -> None:
        decay = self.ema_decay
        for online_p, target_p in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)
        for online_b, target_b in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            target_b.data.copy_(online_b.data)
        for online_p, target_p in zip(self.latent_proj.parameters(), self.target_latent_proj.parameters()):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)


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
        ema_decay=getattr(args, "ema_decay", 0.996),
        mask_ratio=getattr(args, "mask_ratio", 0.25),
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
        _unwrap(model).encoder.eval()
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
                _unwrap(model).update_target_ema()
        batch_size = int(batch["frames"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return {"loss": total_loss / max(count, 1)}


def pretrain_ssl(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    world_size, rank, local_rank, distributed = _setup_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
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
    sampler: DistributedSampler | None = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_ssl_video,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = build_model(args, device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=bool(args.precision == "amp" and device.type == "cuda"))

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        stats = _run_epoch(model, loader, device, args, optimizer=optimizer, scaler=scaler)
        if _is_main_process():
            print(f"[epoch {epoch}] {stats}", flush=True)
        history.append(stats)

    output = Path(args.output).expanduser()
    if _is_main_process():
        output.parent.mkdir(parents=True, exist_ok=True)
        base = _unwrap(model)
        torch.save(
            {
                "encoder": base.encoder.state_dict(),
                "model_state": base.state_dict(),
                "model_config": {
                    "encoder_name": args.encoder_name,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "image_size": args.image_size,
                    "freeze_encoder": args.freeze_encoder,
                    "ema_decay": getattr(args, "ema_decay", 0.996),
                    "mask_ratio": getattr(args, "mask_ratio", 0.25),
                },
                "train_args": vars(args),
                "history": history,
                "world_size": world_size,
            },
            output,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
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
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    return parser


def main() -> None:
    output = pretrain_ssl(build_parser().parse_args())
    if _is_main_process():
        print(f"Saved SSL checkpoint to {output}")


if __name__ == "__main__":
    main()
