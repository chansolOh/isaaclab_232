#!/usr/bin/env python3
"""Adapter for the multi-object Dataset_2026/dataset_v2 grasp format.

The source dataset is organised by environment/section/platform, not by
object.  This module builds a small per-gripper scene index, extracts records
for one object, reconstructs the missing 3-D finger boxes (or matches the hand
pre-grasp boxes), and converts everything to the object's zero-pose frame.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np

from grasp.mesh_geometry import (
    euler_xyz_degrees_to_matrix,
    object_pose_matrix,
    quaternion_wxyz_to_matrix,
)
from merge_grasps import (
    matrix_to_euler_xyz_degrees,
    matrix_to_quaternion,
    transform_points,
    transform_pose_dict,
    zero_pose_object_config,
)


DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/dataset_v2")
MODEL_ROOT = Path("/nas/ochansol/3d_model/peel3_scan_data_2026")
FINGER_INFO_PATH = Path("/nas/ochansol/gripper_info/gripper_info_new_2026.json")
HAND_INFO_PATH = Path("/nas/ochansol/gripper_info/gripper_info_hand_2026.json")
CACHE_ROOT = Path(__file__).resolve().with_name(".dataset_v2_cache")
MESH_UNIT_SCALE = 0.01


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def usage_rows(dataset_root: Path, gripper: str | None = None) -> list[dict]:
    path = dataset_root / "pregrasp_statistics" / "scene_gripper_usage.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if gripper:
        rows = [row for row in rows if row.get("gripper_model") == gripper]
    return rows


def gripper_names(dataset_root: Path) -> list[str]:
    return sorted(
        {row["gripper_model"] for row in usage_rows(dataset_root) if row.get("gripper_model")},
        key=str.casefold,
    )


def object_names(model_root: Path) -> list[str]:
    if not model_root.is_dir():
        return []
    return sorted(
        [
            path.name
            for path in model_root.iterdir()
            if path.is_dir() and (path / "edited").is_dir()
        ],
        key=str.casefold,
    )


def object_name_from_conf(obj: dict) -> str:
    usd_path = Path(str(obj.get("usd_path", "")))
    if usd_path.parent.name == "edited":
        return usd_path.parent.parent.name
    return usd_path.stem.removesuffix("_rigid")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_gripper_scene_index(
    dataset_root: Path,
    gripper: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Return object -> source scenes, using a persistent lightweight cache."""
    csv_path = dataset_root / "pregrasp_statistics" / "scene_gripper_usage.csv"
    cache_path = CACHE_ROOT / f"{_safe_name(gripper)}.json"
    signature = {
        "dataset_root": str(dataset_root.resolve()),
        "usage_mtime_ns": csv_path.stat().st_mtime_ns,
        "gripper": gripper,
    }
    if cache_path.is_file():
        cached = load_json(cache_path)
        if cached.get("signature") == signature and isinstance(cached.get("objects"), dict):
            return cached["objects"]

    rows = usage_rows(dataset_root, gripper)
    result: dict[str, list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        relative = row["platform_path"]
        scene_id = row["scene_id"]
        base = dataset_root / relative
        conf_path = base / "conf" / f"{scene_id}.json"
        output_path = base / "output_grasp" / f"{scene_id}.json"
        if not conf_path.is_file() or not output_path.is_file():
            continue
        try:
            objects = load_json(conf_path).get("objects", [])
        except (OSError, ValueError, TypeError):
            continue
        reference = {"platform_path": relative, "scene_id": scene_id}
        for obj in objects:
            name = object_name_from_conf(obj)
            if name:
                result.setdefault(name, []).append(reference)
        if progress and (index % 250 == 0 or index + 1 == len(rows)):
            progress(f"{gripper} scene index 생성 중 · {index + 1}/{len(rows)}")

    atomic_json(cache_path, {"signature": signature, "objects": result})
    return result


def _flatten_pregrasp(payload) -> list[dict]:
    groups = payload if isinstance(payload, list) else [payload]
    records: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        model = str(group.get("gripper_model", ""))
        for record in group.get("data", []):
            if isinstance(record, dict):
                copied = record
                if model and "gripper_model" not in copied:
                    copied = dict(record, gripper_model=model)
                records.append(copied)
    return records


def _angle_error(lhs, rhs) -> float:
    a = np.asarray(lhs, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    if a.shape != b.shape:
        return 1.0e6
    return float(np.linalg.norm((a - b + 180.0) % 360.0 - 180.0))


def _match_hand_pregrasp(output: dict, pregrasps: list[dict]) -> dict | None:
    best = None
    best_cost = math.inf
    for candidate in pregrasps:
        if candidate.get("target_object") != output.get("target_object"):
            continue
        if output.get("preset_name") and candidate.get("preset_name") != output.get("preset_name"):
            continue
        try:
            cost = float(
                np.linalg.norm(
                    np.asarray(candidate["target_points"], dtype=np.float64)
                    - np.asarray(output["target_points"], dtype=np.float64)
                )
            )
            cost += 1.0e-4 * _angle_error(
                candidate.get("target_orientation", []), output.get("target_orientation", [])
            )
        except (KeyError, TypeError, ValueError):
            continue
        if cost < best_cost:
            best, best_cost = candidate, cost
    # Output records are copied from pre-grasp; a large distance means that the
    # match is unsafe and displaying another hand trajectory would be worse.
    return best if best is not None and best_cost < 1.0e-3 else None


def _finger_boxes_world(record: dict, info: dict) -> np.ndarray:
    half_height = float(info["height"]) * 0.5
    width = float(record["target_width"])
    gripper_type = str(record.get("gripper_type", info.get("type", ""))).lower()
    if gripper_type.startswith("finger2"):
        boxes = np.asarray(
            [[
                [half_height, -width / 2.0, 0.0],
                [-half_height, -width / 2.0, 0.0],
                [-half_height, width / 2.0, 0.0],
                [half_height, width / 2.0, 0.0],
            ]],
            dtype=np.float64,
        )
    else:
        base = np.asarray(
            [
                [half_height, -width / 2.0, 0.0],
                [-half_height, -width / 2.0, 0.0],
                [-half_height, 0.0, 0.0],
                [half_height, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        boxes = []
        for finger in info.get("finger_bbox_info", []):
            angle = math.radians(float(finger["rot"]))
            rotation = np.asarray(
                [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            )
            box = base.copy()
            box[:, :2] = box[:, :2] @ rotation.T + np.asarray(finger["pos"][:2])
            boxes.append(box)
        if not boxes:
            raise ValueError(f"finger3 gripper has no finger_bbox_info: {record.get('gripper_model')}")
        boxes = np.asarray(boxes, dtype=np.float64)
    yaw = math.radians(float(record["target_orientation"][2]))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    boxes[:, :, :2] = boxes[:, :, :2] @ rotation.T
    boxes += np.asarray(record["target_points"], dtype=np.float64)[None, None, :]
    return boxes


def _pose_matrix(record: dict) -> np.ndarray:
    poses = record.get("target_base_tf", {})
    pose = poses.get("end") or poses.get("start")
    if not isinstance(pose, dict):
        rotation = euler_xyz_degrees_to_matrix(record.get("target_orientation", [0, 0, 0]))
        position = np.asarray(record["target_points"], dtype=np.float64)
    else:
        rotation = quaternion_wxyz_to_matrix(pose["orientation_wxyz"])
        position = np.asarray(pose["position"], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def _zero_pose_record(
    raw: dict,
    obj: dict,
    boxes_world: np.ndarray,
    *,
    source_scene: str,
    output_index: int,
) -> dict:
    inverse = np.linalg.inv(object_pose_matrix(obj))
    boxes = transform_points(inverse, boxes_world)
    grasp_mat = inverse @ _pose_matrix(raw)
    approach = grasp_mat[:3, 2].copy()
    approach /= max(float(np.linalg.norm(approach)), 1.0e-12)
    points = transform_points(
        inverse, np.asarray(raw["target_points"], dtype=np.float64)[None]
    )[0]
    disturbed = int(raw.get("disturbed_object_count", 1))
    collateral = max(0, disturbed - 1)
    score = 1.0 / (1.0 + collateral)

    result = {
        "scene_id": source_scene,
        "source_output_index": int(output_index),
        "target_object": raw.get("target_object", obj.get("class", "")),
        "gripper_model": raw.get("gripper_model", ""),
        "gripper_type": raw.get("gripper_type", ""),
        "grasp_box": boxes[0].round(7).tolist(),
        "grasp_boxes": boxes.round(7).tolist(),
        "grasp_mat": grasp_mat.round(7).tolist(),
        "approach_vector": approach.round(7).tolist(),
        "target_points": points.round(7).tolist(),
        "target_width": float(raw.get("target_width", 0.0)),
        "disturbed_object_count": disturbed,
        "collateral_object_count": collateral,
        "disturbance_score": score,
        "score": score,
        "coordinate_frame": "object_zero_pose",
    }
    orientation = np.asarray(raw.get("target_orientation", []), dtype=np.float64)
    if orientation.shape == (3,):
        rotation = inverse[:3, :3] @ euler_xyz_degrees_to_matrix(orientation)
        result["target_orientation"] = matrix_to_euler_xyz_degrees(rotation).round(7).tolist()
    for key in ("preset_name", "target_joint_pos", "joint_unit", "transition"):
        if key in raw:
            result[key] = copy.deepcopy(raw[key])
    target_base = raw.get("target_base_tf")
    if isinstance(target_base, dict):
        result["target_base_tf"] = {
            phase: transform_pose_dict(pose, inverse)
            for phase, pose in target_base.items()
            if isinstance(pose, dict) and "position" in pose and "orientation_wxyz" in pose
        }
    return result


def load_object_grasps(
    dataset_root: Path,
    object_name: str,
    gripper: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict, list[str]]:
    """Load and convert every successful record for one object/gripper pair."""
    index = build_gripper_scene_index(dataset_root, gripper, progress=progress)
    references = index.get(object_name, [])
    if not references:
        raise ValueError(f"{object_name} / {gripper} 조합이 포함된 scene이 없습니다")

    finger_info = load_json(FINGER_INFO_PATH)
    gripper_info = finger_info.get(gripper)
    candidates: list[dict] = []
    reference_object = None
    used_scenes: list[str] = []
    source_order = 0
    for ref_index, ref in enumerate(references):
        relative = ref["platform_path"]
        scene_id = ref["scene_id"]
        base = dataset_root / relative
        conf = load_json(base / "conf" / f"{scene_id}.json")
        obj = next(
            (item for item in conf.get("objects", []) if object_name_from_conf(item) == object_name),
            None,
        )
        if obj is None:
            continue
        reference_object = reference_object or obj
        raw_records = load_json(base / "output_grasp" / f"{scene_id}.json")
        target_records = [
            (index_value, item)
            for index_value, item in enumerate(raw_records if isinstance(raw_records, list) else [])
            if item.get("target_object") == obj.get("class")
            and item.get("gripper_model", gripper) == gripper
        ]
        if not target_records:
            continue

        pregrasps: list[dict] = []
        is_hand = str(target_records[0][1].get("gripper_type", "")).lower() == "hand"
        if is_hand:
            pre_path = base / "pre_grasp" / f"{scene_id}.json"
            if pre_path.is_file():
                pregrasps = _flatten_pregrasp(load_json(pre_path))

        scene_ref = f"{relative}/{scene_id}"
        for output_index, raw in target_records:
            try:
                if is_hand:
                    matched = _match_hand_pregrasp(raw, pregrasps)
                    if matched is None:
                        continue
                    boxes_world = np.asarray(matched.get("grasp_bbox", []), dtype=np.float64)
                else:
                    if gripper_info is None:
                        raise KeyError(f"그리퍼 정보가 없습니다: {gripper}")
                    boxes_world = _finger_boxes_world(raw, gripper_info)
                if boxes_world.ndim != 3 or boxes_world.shape[1:] != (4, 3) or not len(boxes_world):
                    continue
                item = _zero_pose_record(
                    raw, obj, boxes_world, source_scene=scene_ref, output_index=output_index
                )
                item["_filter_gui_id"] = f"{scene_ref}:{output_index}"
                item["_filter_gui_source_order"] = source_order
                candidates.append(item)
                source_order += 1
            except (KeyError, TypeError, ValueError):
                continue
        if target_records:
            used_scenes.append(scene_ref)
        if progress and (ref_index % 10 == 0 or ref_index + 1 == len(references)):
            progress(
                f"{object_name} / {gripper} 읽는 중 · "
                f"{ref_index + 1}/{len(references)} scenes · {len(candidates)} grasps"
            )

    if reference_object is None:
        raise ValueError(f"{object_name} object config를 찾지 못했습니다")
    return candidates, zero_pose_object_config(reference_object), used_scenes


def compact_v2_item(item: dict) -> dict:
    """Keep geometry plus the minimum pose/preset fields needed downstream."""
    keys = (
        "scene_id",
        "source_output_index",
        "target_object",
        "gripper_model",
        "gripper_type",
        "grasp_box",
        "grasp_boxes",
        "grasp_mat",
        "approach_vector",
        "target_points",
        "target_orientation",
        "target_width",
        "disturbed_object_count",
        "collateral_object_count",
        "disturbance_score",
        "score",
        "target_base_tf",
        "preset_name",
        "target_joint_pos",
        "joint_unit",
        "transition",
    )
    return {key: copy.deepcopy(item[key]) for key in keys if key in item}


def save_v2_npz(path: Path, metadata: dict, items: list[dict]) -> None:
    groups = [np.asarray(item["grasp_boxes"], dtype=np.float32) for item in items]
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(group) for group in groups])
    flat = np.concatenate(groups) if groups else np.zeros((0, 4, 3), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        scene_id=np.asarray([item["scene_id"] for item in items]),
        source_output_index=np.asarray([item["source_output_index"] for item in items]),
        grasp_box=np.asarray([item["grasp_box"] for item in items], dtype=np.float32).reshape(-1, 4, 3),
        grasp_boxes_flat=flat,
        grasp_boxes_offsets=offsets,
        grasp_mat=np.asarray([item["grasp_mat"] for item in items], dtype=np.float32).reshape(-1, 4, 4),
        approach_vector=np.asarray([item["approach_vector"] for item in items], dtype=np.float32).reshape(-1, 3),
        target_points=np.asarray([item["target_points"] for item in items], dtype=np.float32).reshape(-1, 3),
        target_orientation=np.asarray([item.get("target_orientation", [np.nan] * 3) for item in items], dtype=np.float32),
        target_width=np.asarray([item.get("target_width", 0.0) for item in items], dtype=np.float32),
        disturbed_object_count=np.asarray([item["disturbed_object_count"] for item in items], dtype=np.int32),
        score=np.asarray([item["score"] for item in items], dtype=np.float32),
        replay_json=np.asarray([
            json.dumps(
                {key: item[key] for key in ("target_base_tf", "preset_name", "target_joint_pos", "joint_unit", "transition") if key in item},
                separators=(",", ":"),
            )
            for item in items
        ]),
    )
    os.replace(temporary, path)

