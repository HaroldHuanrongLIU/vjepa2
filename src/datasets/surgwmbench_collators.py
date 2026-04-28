"""Collators for SurgWMBench final-layout datasets."""

from __future__ import annotations

from typing import Any

import torch


def _metadata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "source_video_id": [item["source_video_id"] for item in batch],
        "source_video_path": [item["source_video_path"] for item in batch],
        "trajectory_id": [item["trajectory_id"] for item in batch],
        "difficulty": [item["difficulty"] for item in batch],
        "annotation_path": [item["annotation_path"] for item in batch],
        "interpolation_path": [item["interpolation_path"] for item in batch],
        "interpolation_method": [item["interpolation_method"] for item in batch],
        "image_size_original": [item["image_size_original"] for item in batch],
    }


def collate_sparse_anchors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples loaded with ``frame_sampling='sparse_anchors'``."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    frames = torch.stack([item["frames"] for item in batch], dim=0)
    coords_norm = torch.stack([item["selected_coords_norm"] for item in batch], dim=0)
    coords_px = torch.stack([item["selected_coords_px"] for item in batch], dim=0)
    sampled_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    frame_indices = torch.stack([item["frame_indices"] for item in batch], dim=0)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    if frames.shape[1] != 20:
        raise ValueError(f"Sparse SurgWMBench batches must have 20 anchor frames, got {frames.shape[1]}")
    if coords_norm.shape[1] != 20 or sampled_indices.shape[1] != 20:
        raise ValueError("Sparse SurgWMBench batches must have 20 coordinates and sampled indices.")

    actions_delta = coords_norm[:, 1:] - coords_norm[:, :-1]
    denom = torch.clamp(num_frames.to(torch.float32) - 1.0, min=1.0).unsqueeze(1)
    anchor_dt = (sampled_indices[:, 1:] - sampled_indices[:, :-1]).to(torch.float32) / denom
    actions_delta_dt = torch.cat([actions_delta, anchor_dt.unsqueeze(-1)], dim=-1)
    mask = torch.ones(coords_norm.shape[:2], dtype=torch.bool)

    result = {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "sampled_indices": sampled_indices,
        "frame_indices": frame_indices,
        "num_frames": num_frames,
        "anchor_dt": anchor_dt,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "mask": mask,
        "coord_source": torch.stack([item["selected_coord_sources"] for item in batch], dim=0),
        "label_weight": torch.stack([item["selected_label_weights"] for item in batch], dim=0),
    }
    result.update(_metadata(batch))
    return result


def collate_dense_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length dense or windowed SurgWMBench samples."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)
    max_t = max(int(item["frames"].shape[0]) for item in batch)
    channels, height, width = batch[0]["frames"].shape[1:]

    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    coords_norm = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    coords_px = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    coord_source = torch.zeros(batch_size, max_t, dtype=torch.long)
    label_weight = torch.zeros(batch_size, max_t, dtype=torch.float32)
    confidence = torch.zeros(batch_size, max_t, dtype=torch.float32)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    actions_delta = torch.zeros(batch_size, max(max_t - 1, 0), 2, dtype=torch.float32)
    actions_delta_dt = torch.zeros(batch_size, max(max_t - 1, 0), 3, dtype=torch.float32)
    action_mask = torch.zeros(batch_size, max(max_t - 1, 0), dtype=torch.bool)

    for row, item in enumerate(batch):
        t = int(item["frames"].shape[0])
        frames[row, :t] = item["frames"]
        coords_norm[row, :t] = item["selected_coords_norm"]
        coords_px[row, :t] = item["selected_coords_px"]
        frame_mask[row, :t] = True
        coord_source[row, :t] = item["selected_coord_sources"]
        label_weight[row, :t] = item["selected_label_weights"]
        confidence[row, :t] = item["selected_confidence"]
        frame_indices[row, :t] = item["frame_indices"]

        if t > 1:
            actions_delta[row, : t - 1] = item["selected_coords_norm"][1:] - item["selected_coords_norm"][:-1]
            denom = max(float(item["num_frames"] - 1), 1.0)
            dt = (item["frame_indices"][1:] - item["frame_indices"][:-1]).to(torch.float32) / denom
            actions_delta_dt[row, : t - 1] = torch.cat([actions_delta[row, : t - 1], dt.unsqueeze(-1)], dim=-1)
            action_mask[row, : t - 1] = True

    result = {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "frame_mask": frame_mask,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "num_frames": num_frames,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "action_mask": action_mask,
    }
    result.update(_metadata(batch))
    return result
