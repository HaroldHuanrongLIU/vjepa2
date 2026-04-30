from __future__ import annotations

import argparse

import torch

from app.surgwmbench.pretrain_ssl import pretrain_ssl
from src.datasets.surgwmbench_collators import collate_ssl_video
from src.datasets.surgwmbench_video import SurgWMBenchRawVideoDataset
from tests.surgwmbench_test_utils import make_surgwmbench_root


def test_raw_video_dataset_falls_back_to_clip_frames(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchRawVideoDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        clip_length=4,
        stride=2,
        image_size=32,
        backend="opencv_or_frames",
        max_videos=1,
        max_clips_per_video=2,
    )

    sample = dataset[0]

    assert sample["frames"].shape == (4, 3, 32, 32)
    assert sample["backend"] == "frames"
    assert sample["frame_indices"].shape == (4,)

    batch = collate_ssl_video([dataset[0], dataset[1]])
    assert batch["frames"].shape == (2, 4, 3, 32, 32)
    assert batch["frame_indices"].shape == (2, 4)


def test_pretrain_ssl_smoke(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    checkpoint_path = tmp_path / "checkpoints" / "ssl.pt"
    args = argparse.Namespace(
        dataset_root=str(root),
        split="train",
        manifest="manifests/train.jsonl",
        source_video_manifest=None,
        output=str(checkpoint_path),
        ssl_source="raw_videos",
        encoder_name="dummy",
        encoder_checkpoint=None,
        encoder_checkpoint_key="target_encoder",
        freeze_encoder=False,
        image_size=32,
        clip_length=4,
        stride=2,
        latent_dim=16,
        hidden_dim=32,
        batch_size=2,
        num_workers=0,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        precision="fp32",
        grad_clip_norm=1.0,
        seed=9,
        device="cpu",
        max_videos=1,
        max_clips_per_video=2,
        max_samples=2,
    )

    output = pretrain_ssl(args)

    assert output == checkpoint_path
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "encoder" in checkpoint
    assert "model_state" in checkpoint
    assert checkpoint["model_config"]["encoder_name"] == "dummy"
    assert checkpoint["history"]
