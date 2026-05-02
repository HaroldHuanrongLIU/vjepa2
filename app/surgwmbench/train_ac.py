"""Sparse 20-anchor SurgWMBench action-conditioned training."""

from __future__ import annotations

import argparse
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

from src.datasets.surgwmbench import SurgWMBenchClipDataset
from src.datasets.surgwmbench_collators import collate_sparse_anchors
from src.models.surgwmbench_vjepa_ac import SurgVJEPA2AC, load_surgwmbench_encoder


def _setup_distributed() -> tuple[int, int, int, bool]:
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


def _maybe_subset(dataset: SurgWMBenchClipDataset, max_samples: int | None) -> SurgWMBenchClipDataset | Subset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def build_model(args: argparse.Namespace, device: torch.device) -> SurgVJEPA2AC:
    encoder = load_surgwmbench_encoder(
        encoder_name=args.encoder_name,
        checkpoint_path=args.encoder_checkpoint,
        checkpoint_key=args.encoder_checkpoint_key,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        freeze=args.freeze_encoder,
        device=device,
    )
    model = SurgVJEPA2AC(
        encoder=encoder,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        freeze_encoder=args.freeze_encoder,
    )
    return model.to(device)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def smoothness_loss(coords: torch.Tensor) -> torch.Tensor:
    if coords.shape[1] < 3:
        return coords.new_zeros(())
    accel = coords[:, 2:] - 2.0 * coords[:, 1:-1] + coords[:, :-2]
    return accel.square().sum(dim=-1).mean()


def compute_losses(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace
) -> dict[str, torch.Tensor]:
    latent_loss = F.smooth_l1_loss(outputs["pred_latents"], outputs["target_latents"][:, 1:].detach())
    sparse_coord_loss = F.smooth_l1_loss(outputs["pred_next_coords_norm"], batch["coords_norm"][:, 1:])
    action_loss = F.smooth_l1_loss(outputs["pred_actions_delta_dt"], batch["actions_delta_dt"])
    smooth_loss = smoothness_loss(outputs["pred_coords_norm"])
    total = (
        args.latent_weight * latent_loss
        + args.sparse_coord_weight * sparse_coord_loss
        + args.action_weight * action_loss
        + args.smoothness_weight * smooth_loss
    )
    return {
        "loss": total,
        "latent_loss": latent_loss,
        "sparse_coord_loss": sparse_coord_loss,
        "action_loss": action_loss,
        "smoothness_loss": smooth_loss,
    }


def _run_epoch(
    model: SurgVJEPA2AC,
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
    totals: dict[str, float] = {}
    count = 0
    use_amp = bool(args.precision == "amp" and device.type == "cuda")

    for batch in loader:
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch)
                losses = compute_losses(outputs, batch, args)
            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["loss"]).backward()
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
        batch_size = int(batch["frames"].shape[0])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def train_ac(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    world_size, rank, local_rank, distributed = _setup_distributed()
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _device(args.device)

    train_dataset = SurgWMBenchClipDataset(
        dataset_root=args.dataset_root,
        manifest=args.train_manifest,
        interpolation_method=args.interpolation_method,
        image_size=args.image_size,
        frame_sampling="sparse_anchors",
    )
    train_dataset = _maybe_subset(train_dataset, args.max_train_samples)
    train_sampler: DistributedSampler | None = None
    train_shuffle = True
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        train_shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_sparse_anchors,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    val_sampler: DistributedSampler | None = None
    if args.val_manifest:
        val_dataset = SurgWMBenchClipDataset(
            dataset_root=args.dataset_root,
            manifest=args.val_manifest,
            interpolation_method=args.interpolation_method,
            image_size=args.image_size,
            frame_sampling="sparse_anchors",
        )
        val_dataset = _maybe_subset(val_dataset, args.max_val_samples)
        if distributed:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_sparse_anchors,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    model = build_model(args, device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.freeze_encoder,
        )
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=bool(args.precision == "amp" and device.type == "cuda"))

    history: list[dict[str, float]] = []
    best_val_loss: float | None = None
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = _run_epoch(model, train_loader, device, args, optimizer=optimizer, scaler=scaler)
        stats = {f"train_{key}": value for key, value in train_stats.items()}
        if val_loader is not None:
            with torch.inference_mode():
                val_stats = _run_epoch(model, val_loader, device, args)
            stats.update({f"val_{key}": value for key, value in val_stats.items()})
            best_val_loss = stats["val_loss"] if best_val_loss is None else min(best_val_loss, stats["val_loss"])
        if _is_main_process():
            print(f"[epoch {epoch}] {stats}", flush=True)
        history.append(stats)

    output = Path(args.output).expanduser()
    if _is_main_process():
        output.parent.mkdir(parents=True, exist_ok=True)
        base = _unwrap(model)
        torch.save(
            {
                "model_state": base.state_dict(),
                "model_config": {
                    "encoder_name": args.encoder_name,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "image_size": args.image_size,
                    "freeze_encoder": args.freeze_encoder,
                },
                "train_args": vars(args),
                "history": history,
                "best_val_loss": best_val_loss,
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
    parser.add_argument("--train-manifest", default="manifests/train.jsonl")
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interpolation-method", default="linear")
    parser.add_argument("--encoder-name", default="dummy")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--encoder-checkpoint-key", default="target_encoder")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=384)
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
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--sparse-coord-weight", type=float, default=10.0)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    return parser


def main() -> None:
    output = train_ac(build_parser().parse_args())
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
