"""Sparse human-anchor evaluation for SurgWMBench action-conditioned models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from app.surgwmbench.train_ac import _device
from src.datasets.surgwmbench import SurgWMBenchClipDataset
from src.datasets.surgwmbench_collators import collate_sparse_anchors
from src.models.surgwmbench_vjepa_ac import SurgVJEPA2AC, load_surgwmbench_encoder
from src.utils.surgwmbench_metrics import sparse_anchor_metrics


def _maybe_subset(dataset: SurgWMBenchClipDataset, max_samples: int | None) -> SurgWMBenchClipDataset | Subset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def build_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> SurgVJEPA2AC:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    encoder_name = config.get("encoder_name", "dummy")
    latent_dim = int(config.get("latent_dim", 64))
    hidden_dim = int(config.get("hidden_dim", 128))
    image_size = int(config.get("image_size", 384))
    freeze_encoder = bool(config.get("freeze_encoder", True))
    encoder = load_surgwmbench_encoder(
        encoder_name=encoder_name,
        image_size=image_size,
        latent_dim=latent_dim,
        freeze=freeze_encoder,
        device=device,
    )
    model = SurgVJEPA2AC(
        encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        freeze_encoder=freeze_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _cat(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(values, dim=0) if values else torch.empty(0, 0, 2)


def _metric_dict(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, horizons: list[int]
) -> dict[str, float | None]:
    if pred.numel() == 0:
        return sparse_anchor_metrics(torch.empty(0, 0, 2), torch.empty(0, 0, 2), torch.empty(0, 0), horizons=horizons)
    return sparse_anchor_metrics(pred, target, mask, horizons=horizons)


def evaluate_ac(args: argparse.Namespace) -> Path:
    device = _device(args.device)
    dataset = SurgWMBenchClipDataset(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        interpolation_method=args.interpolation_method,
        image_size=args.image_size,
        frame_sampling="sparse_anchors",
    )
    dataset = _maybe_subset(dataset, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sparse_anchors,
    )

    model = build_model_from_checkpoint(args.checkpoint, device)
    context = int(args.context_anchors)
    horizons = [int(value) for value in args.horizons]

    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []
    by_difficulty: dict[str, dict[str, list[torch.Tensor]]] = {}

    with torch.inference_mode():
        for batch in loader:
            raw_difficulty = batch["difficulty"]
            batch = _move_batch(batch, device)
            total_future = batch["coords_norm"].shape[1] - context
            horizon = min(int(args.horizon), total_future)
            actions = None
            if args.rollout_mode == "teacher_forced_actions":
                actions = batch["actions_delta_dt"][:, context - 1 : context - 1 + horizon]
            rollout = model.rollout(
                batch["frames"][:, :context],
                batch["coords_norm"][:, :context],
                horizon=horizon,
                mode=args.rollout_mode,
                actions=actions,
            )
            pred = rollout["pred_coords_norm"][:, 1 : horizon + 1].detach().cpu()
            target = batch["coords_norm"][:, context : context + horizon].detach().cpu()
            mask = torch.ones(pred.shape[:2], dtype=torch.bool)
            all_pred.append(pred)
            all_target.append(target)
            all_mask.append(mask)

            for row, difficulty in enumerate(raw_difficulty):
                key = "null" if difficulty is None else str(difficulty)
                store = by_difficulty.setdefault(key, {"pred": [], "target": [], "mask": []})
                store["pred"].append(pred[row : row + 1])
                store["target"].append(target[row : row + 1])
                store["mask"].append(mask[row : row + 1])

    overall = _metric_dict(_cat(all_pred), _cat(all_target), torch.cat(all_mask, dim=0), horizons)
    difficulty_metrics = {
        difficulty: _metric_dict(
            _cat(values["pred"]), _cat(values["target"]), torch.cat(values["mask"], dim=0), horizons
        )
        for difficulty, values in sorted(by_difficulty.items())
    }

    output_payload = {
        "dataset_name": "SurgWMBench",
        "manifest": args.manifest,
        "checkpoint": str(args.checkpoint),
        "interpolation_method": args.interpolation_method,
        "primary_target": "sparse_human_anchors",
        "dense_target": None,
        "rollout_mode": args.rollout_mode,
        "context_anchors": context,
        "metrics_overall": overall,
        "metrics_by_difficulty": difficulty_metrics,
        "num_clips": int(sum(tensor.shape[0] for tensor in all_pred)),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/test.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interpolation-method", default="linear")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--rollout-mode", choices=("policy_rollout", "teacher_forced_actions"), default="policy_rollout"
    )
    parser.add_argument("--context-anchors", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=19)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    output = evaluate_ac(build_parser().parse_args())
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
