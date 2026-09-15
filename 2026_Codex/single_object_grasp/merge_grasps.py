#!/usr/bin/env python3
"""Merge scene grasp results into one object-zero-pose JSON and NPZ index."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from grasp.mesh_geometry import (
    load_obj_triangles,
    object_pose_matrix,
    quaternion_wxyz_to_matrix,
    resolve_obj_path,
    transform_object_vertices,
)


# -----------------------------------------------------------------------------
# 실행 설정: 이 부분만 수정해서 사용한다.
# ROOT는 <output-root>/<object>/<gripper> 폴더다.
# -----------------------------------------------------------------------------
ROOT = Path(
    "/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/"
    "black_pepper_shaker/Robotiq_2f140"
)
SCENE_START: int | None = None
SCENE_END: int | None = None

SCORE_THRESHOLD = 0.25
MESH_UNIT_SCALE = 0.01
BOX_THICKNESS = 0.02
BOX_MARGIN = 0.002
NMS_CENTER = 0.015
NMS_ROTATION_DEG = 20.0
EMPTY_BOX_FILTER = True
USE_NMS = True

# None이면 ROOT 아래에 기본 파일명으로 저장한다.
OUTPUT_JSON: Path | None = None
OUTPUT_NPZ: Path | None = None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def selected_scene_ids() -> list[str]:
    conf_ids = {path.stem for path in (ROOT / "conf").glob("[0-9][0-9][0-9][0-9].json")}
    output_ids = {
        path.stem for path in (ROOT / "output_grasp").glob("[0-9][0-9][0-9][0-9].json")
    }
    result = sorted(conf_ids & output_ids)
    if SCENE_START is not None:
        result = [value for value in result if int(value) >= SCENE_START]
    if SCENE_END is not None:
        result = [value for value in result if int(value) <= SCENE_END]
    return result


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points.reshape(-1, 3), np.ones(points.size // 3)))
    return (homogeneous @ matrix.T)[:, :3].reshape(points.shape)


def transform_pose_dict(pose: dict, inverse: np.ndarray) -> dict:
    result = copy.deepcopy(pose)
    position = transform_points(inverse, np.asarray(pose["position"])[None])[0]
    rotation = inverse[:3, :3] @ quaternion_wxyz_to_matrix(pose["orientation_wxyz"])
    result["position"] = position.round(7).tolist()
    result["orientation_wxyz"] = matrix_to_quaternion(rotation).round(8).tolist()
    result.pop("rpy_deg", None)
    return result


def to_zero_pose(item: dict, obj_conf: dict, scene_id: str) -> dict | None:
    box = np.asarray(item.get("grasp_box", []), dtype=np.float64)
    if box.shape != (4, 3):
        return None
    inverse = np.linalg.inv(object_pose_matrix(obj_conf))
    result = copy.deepcopy(item)
    result["grasp_box"] = transform_points(inverse, box).round(7).tolist()
    if "grasp_mat" in item:
        matrix = np.asarray(item["grasp_mat"], dtype=np.float64)
        if matrix.shape == (4, 4):
            result["grasp_mat"] = (inverse @ matrix).round(7).tolist()
    points = np.asarray(item.get("target_points", []), dtype=np.float64)
    if points.shape == (3,):
        result["target_points"] = transform_points(inverse, points[None])[0].round(7).tolist()
    normal = np.asarray(item.get("normal", []), dtype=np.float64)
    if normal.shape == (3,):
        normal = inverse[:3, :3] @ normal
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        result["normal"] = normal.round(7).tolist()
    target_base = item.get("target_base_tf")
    if isinstance(target_base, dict):
        result["target_base_tf"] = {
            phase: transform_pose_dict(pose, inverse)
            for phase, pose in target_base.items()
            if isinstance(pose, dict) and "position" in pose
        }
    result["scene_id"] = scene_id
    return result


def mesh_in_box(points: np.ndarray, box: np.ndarray, thickness: float, margin: float) -> bool:
    center = box.mean(axis=0)
    edge_x = box[1] - box[0]
    edge_y = box[3] - box[0]
    length_x = float(np.linalg.norm(edge_x))
    x_axis = edge_x / max(length_x, 1e-12)
    orthogonal_y = edge_y - np.dot(edge_y, x_axis) * x_axis
    length_y = float(np.linalg.norm(orthogonal_y))
    if length_x <= 1e-8 or length_y <= 1e-8:
        return False
    y_axis = orthogonal_y / length_y
    z_axis = np.cross(x_axis, y_axis)
    local = points - center
    return bool(
        np.any(
            (np.abs(local @ x_axis) <= length_x / 2 + margin)
            & (np.abs(local @ y_axis) <= length_y / 2 + margin)
            & (np.abs(local @ z_axis) <= thickness / 2 + margin)
        )
    )


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def nms(items: list[dict], center_threshold: float, rotation_threshold: float) -> list[dict]:
    kept, frames = [], []
    for item in sorted(items, key=lambda value: float(value.get("score", 0.0)), reverse=True):
        box = np.asarray(item["grasp_box"], dtype=np.float64)
        center = box.mean(axis=0)
        matrix = np.asarray(item.get("grasp_mat", np.eye(4)), dtype=np.float64)
        rotation = matrix[:3, :3]
        duplicate = any(
            np.linalg.norm(center - old_center) <= center_threshold
            and rotation_distance(rotation, old_rotation) <= rotation_threshold
            for old_center, old_rotation in frames
        )
        if not duplicate:
            kept.append(item)
            frames.append((center, rotation))
    return kept


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def save_npz(path: Path, metadata: dict, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = np.asarray(
        [item["grasp_box"] for item in items], dtype=np.float32
    ).reshape(-1, 4, 3)
    matrices = np.asarray(
        [item["grasp_mat"] for item in items], dtype=np.float32
    ).reshape(-1, 4, 4)
    centers = boxes.mean(axis=1)
    scores = np.asarray([item.get("score", np.nan) for item in items], dtype=np.float32)
    rotation_scores = np.asarray(
        [item.get("rotation_score", np.nan) for item in items], dtype=np.float32
    )
    normals = np.asarray(
        [item.get("normal", [np.nan] * 3) for item in items], dtype=np.float32
    ).reshape(-1, 3)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata)),
        grasp_box=boxes,
        grasp_mat=matrices,
        grasp_center=centers,
        rotation_matrix=matrices[:, :3, :3],
        score=scores,
        rotation_score=rotation_scores,
        normal=normals,
        scene_id=np.asarray([item["scene_id"] for item in items]),
        gripper_model=np.asarray([item["gripper_model"] for item in items]),
    )
    os.replace(temporary, path)


def main() -> None:
    scene_ids = selected_scene_ids()
    if not scene_ids:
        raise RuntimeError("No scene has both conf and output_grasp JSON")
    merged = []
    reference_obj = None
    for scene_id in scene_ids:
        conf = load_json(ROOT / "conf" / f"{scene_id}.json")
        if len(conf.get("objects", [])) != 1:
            raise ValueError(f"scene {scene_id} is not single-object")
        obj = conf["objects"][0]
        reference_obj = reference_obj or obj
        for item in load_json(ROOT / "output_grasp" / f"{scene_id}.json"):
            if float(item.get("score", 0.0)) < SCORE_THRESHOLD:
                continue
            transformed = to_zero_pose(item, obj, scene_id)
            if transformed is not None:
                merged.append(transformed)
    after_score = len(merged)
    if EMPTY_BOX_FILTER:
        obj_path = resolve_obj_path(reference_obj["usd_path"])
        vertices, _ = load_obj_triangles(obj_path)
        vertices = transform_object_vertices(
            vertices,
            reference_obj,
            MESH_UNIT_SCALE,
            apply_pose=False,
        )
        merged = [
            item
            for item in merged
            if mesh_in_box(
                vertices,
                np.asarray(item["grasp_box"]),
                BOX_THICKNESS,
                BOX_MARGIN,
            )
        ]
    after_empty = len(merged)
    if USE_NMS:
        merged = nms(merged, NMS_CENTER, NMS_ROTATION_DEG)
    object_name = reference_obj["class"]
    output_json = OUTPUT_JSON or ROOT / f"{object_name}_merged_grasp_zero_pose.json"
    output_npz = OUTPUT_NPZ or ROOT / f"{object_name}_merged_grasp_zero_pose.npz"
    metadata = {
        "object": object_name,
        "scene_ids": scene_ids,
        "score_threshold": SCORE_THRESHOLD,
        "counts": {
            "after_score": after_score,
            "after_empty_box": after_empty,
            "final": len(merged),
        },
    }
    atomic_json(output_json, {"metadata": metadata, "data": merged})
    save_npz(output_npz, metadata, merged)
    print(f"Merge > json={output_json}")
    print(f"Merge > npz={output_npz}")
    print(f"Merge > counts={metadata['counts']}")


if __name__ == "__main__":
    main()
