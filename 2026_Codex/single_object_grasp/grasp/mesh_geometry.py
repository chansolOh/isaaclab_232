"""Mesh and pose helpers used when merging evaluated grasps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def quaternion_wxyz_to_matrix(value: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(value), dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"WXYZ quaternion must have shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion cannot have zero length")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_degrees_to_matrix(value: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(tuple(value), dtype=np.float64))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def object_pose_matrix(obj_conf: dict) -> np.ndarray:
    """Return the rigid object pose (scale is deliberately excluded)."""
    orient = obj_conf.get("orient", [0.0, 0.0, 0.0])
    if len(orient) == 3:
        rotation = euler_xyz_degrees_to_matrix(orient)
    elif len(orient) == 4:
        rotation = quaternion_wxyz_to_matrix(orient)
    else:
        raise ValueError("object orient must be Euler XYZ degrees or WXYZ quaternion")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(obj_conf["translate"], dtype=np.float64)
    return matrix


def resolve_obj_path(usd_path: str | Path) -> Path:
    """Resolve the mesh source used to build the USD."""
    usd_path = Path(usd_path)
    candidates = [
        usd_path.with_suffix(".obj"),
        usd_path.with_name(f"{usd_path.stem.removesuffix('_rigid')}.obj"),
        usd_path.parent.parent / f"{usd_path.stem.removesuffix('_rigid')}.obj",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find OBJ beside {usd_path}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_obj_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and triangulated faces without a third-party mesh package."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                fields = line.split()[1:]
                face = []
                for field in fields:
                    raw_index = int(field.split("/", 1)[0])
                    face.append(raw_index - 1 if raw_index > 0 else len(vertices) + raw_index)
                for index in range(1, len(face) - 1):
                    triangles.append([face[0], face[index], face[index + 1]])
    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,) or not len(vertex_array):
        raise ValueError(f"OBJ contains no valid vertices: {path}")
    if triangle_array.ndim != 2 or triangle_array.shape[1:] != (3,) or not len(triangle_array):
        raise ValueError(f"OBJ contains no valid faces: {path}")
    return vertex_array, triangle_array


def transform_object_vertices(
    vertices: np.ndarray,
    obj_conf: dict,
    mesh_unit_scale: float = 0.01,
    *,
    apply_pose: bool = True,
) -> np.ndarray:
    scale = np.asarray(obj_conf.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.shape != (3,):
        raise ValueError(f"Object scale must contain 1 or 3 values, got {scale}")
    transformed = np.asarray(vertices, dtype=np.float64) * float(mesh_unit_scale) * scale
    if apply_pose:
        pose = object_pose_matrix(obj_conf)
        transformed = transformed @ pose[:3, :3].T + pose[:3, 3]
    return transformed

