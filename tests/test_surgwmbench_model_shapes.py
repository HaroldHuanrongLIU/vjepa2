from __future__ import annotations

import torch

from src.datasets.surgwmbench import SurgWMBenchClipDataset
from src.datasets.surgwmbench_collators import collate_sparse_anchors
from src.models.surgwmbench_vjepa_ac import DummyFrameEncoder, SurgVJEPA2AC, load_surgwmbench_encoder
from tests.surgwmbench_test_utils import make_surgwmbench_root


def _batch(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )
    return collate_sparse_anchors([dataset[0], dataset[1]])


def test_surg_vjepa2_ac_forward_and_rollout_shapes(tmp_path):
    batch = _batch(tmp_path)
    model = SurgVJEPA2AC(encoder=DummyFrameEncoder(latent_dim=32), latent_dim=32, hidden_dim=48)

    outputs = model(batch)

    assert outputs["target_latents"].shape == (2, 20, 32)
    assert outputs["pred_latents"].shape == (2, 19, 32)
    assert outputs["pred_next_coords_norm"].shape == (2, 19, 2)
    assert outputs["pred_coords_norm"].shape == (2, 20, 2)
    assert outputs["pred_actions_delta_dt"].shape == (2, 19, 3)

    teacher = model.rollout(
        batch["frames"][:, :1],
        batch["coords_norm"][:, :1],
        horizon=19,
        mode="teacher_forced_actions",
        actions=batch["actions_delta_dt"],
    )
    assert teacher["pred_coords_norm"].shape == (2, 20, 2)
    assert teacher["pred_latents"].shape == (2, 19, 32)
    assert teacher["pred_actions"].shape == (2, 19, 3)

    policy = model.rollout(batch["frames"][:, :1], batch["coords_norm"][:, :1], horizon=19, mode="policy_rollout")
    assert policy["pred_coords_norm"].shape == (2, 20, 2)
    assert torch.isfinite(policy["pred_coords_norm"]).all()


def test_encoder_loader_dummy_and_repo_native_tiny_encoder(tmp_path):
    dummy = load_surgwmbench_encoder("dummy", latent_dim=16, freeze=True)
    assert isinstance(dummy, DummyFrameEncoder)
    assert all(not param.requires_grad for param in dummy.parameters())

    encoder = load_surgwmbench_encoder(
        "vit_tiny",
        image_size=32,
        patch_size=16,
        latent_dim=8,
        freeze=True,
    )
    model = SurgVJEPA2AC(encoder=encoder, latent_dim=8, hidden_dim=16, freeze_encoder=True)
    frames = torch.rand(1, 2, 3, 32, 32)
    latents = model.encode_frames(frames)
    assert latents.shape == (1, 2, 8)
