from __future__ import annotations

import pytest
import torch

from src.datasets.surgwmbench import SOURCE_TO_CODE, SurgWMBenchClipDataset
from tests.surgwmbench_test_utils import make_surgwmbench_root


def test_sparse_dataset_loads_exactly_20_anchors(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (20, 3, 32, 32)
    assert sample["sampled_indices"].shape == (20,)
    assert sample["human_anchor_coords_px"].shape == (20, 2)
    assert sample["human_anchor_coords_norm"].shape == (20, 2)
    assert sample["selected_coords_norm"].shape == (20, 2)
    assert torch.equal(sample["frame_indices"], sample["human_anchor_local_indices"])
    assert torch.all(sample["selected_coord_sources"] == SOURCE_TO_CODE["human"])
    assert sample["dense_coords_norm"] is None


def test_dense_dataset_loads_variable_length_coordinates_and_sources(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (25, 3, 32, 32)
    assert sample["dense_coords_norm"].shape == (25, 2)
    assert sample["selected_coords_px"].shape == (25, 2)
    assert int((sample["selected_coord_sources"] == SOURCE_TO_CODE["human"]).sum()) == 20
    assert int((sample["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]).sum()) == 5
    assert torch.all(
        sample["selected_label_weights"][sample["selected_coord_sources"] == SOURCE_TO_CODE["human"]] == 1.0
    )
    assert torch.all(
        sample["selected_label_weights"][sample["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]] == 0.5
    )


def test_interpolation_method_switching_loads_selected_file(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    linear = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )[0]
    pchip = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
    )[0]

    non_anchor = torch.nonzero(linear["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"])[0].item()
    assert linear["interpolation_method"] == "linear"
    assert pchip["interpolation_method"] == "pchip"
    assert not torch.allclose(linear["selected_coords_px"][non_anchor], pchip["selected_coords_px"][non_anchor])


def test_private_use_path_alias_resolves_star_directories(tmp_path):
    root = make_surgwmbench_root(tmp_path, private_use_path_alias=True)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample = dataset[0]

    assert sample["trajectory_id"].endswith("\uf021")
    assert sample["frames"].shape == (25, 3, 32, 32)
    assert sample["selected_coords_norm"].shape == (25, 2)


def test_strict_loader_rejects_missing_interpolation_file(tmp_path):
    root = make_surgwmbench_root(tmp_path, missing_interpolation_method="pchip")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
        strict=True,
    )

    with pytest.raises(FileNotFoundError, match="Interpolation file not found"):
        _ = dataset[0]


def test_strict_loader_rejects_wrong_dataset_version(tmp_path):
    root = make_surgwmbench_root(tmp_path, bad_version=True)

    with pytest.raises(ValueError, match="dataset_version"):
        SurgWMBenchClipDataset(
            dataset_root=root,
            manifest="manifests/train.jsonl",
            image_size=32,
            frame_sampling="sparse_anchors",
            strict=True,
        )


def test_dataset_loader_does_not_create_random_splits(tmp_path):
    root = make_surgwmbench_root(tmp_path)
    manifests_dir = root / "manifests"
    before = sorted(path.name for path in manifests_dir.iterdir())

    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )
    _ = dataset[0]

    after = sorted(path.name for path in manifests_dir.iterdir())
    assert after == before == ["all.jsonl", "test.jsonl", "train.jsonl", "val.jsonl"]
