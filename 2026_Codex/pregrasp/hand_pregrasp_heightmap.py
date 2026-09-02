"""Hand pre-grasp generation using collision-geometry bottom height maps.

This module is intentionally separate from the legacy BBox samplers.  Isaac
Sim imports are kept inside stage-facing functions, so the height-map matching
and depth conversion functions can also be tested with ordinary Python.

Coordinate convention
---------------------
The center of the global min/max extent of every preset ``grasp_bbox`` is the
TCP pivot.  The stored BBoxes are expressed in the START gripper-base frame,
then reconstructed in world coordinates before their common extent center is
calculated.  The TCP frame used here is world-axis aligned; yaw therefore
rotates the START collision geometry about the TCP's world Z axis. All stored
bottom heights are relative to TCP Z. Consequently a top-view scene height
and a gripper bottom height can be compared directly:

    first_contact_z = nanmax(scene_height - gripper_bottom_height)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_HAND_INFO_PATH = "/nas/ochansol/gripper_info/gripper_info_hand_2026.json"
DEFAULT_YAWS = tuple(range(0, 360, 10))
HEIGHT_MAP_ARCHIVE_SCHEMA_VERSION = 3


def load_hand_database(path: str | os.PathLike = DEFAULT_HAND_INFO_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        database = json.load(stream)
    if not isinstance(database, dict):
        raise ValueError("Hand gripper database root must be a JSON object")
    return database


def get_gripper_and_preset(database: dict, gripper_key: str, preset_name: str) -> tuple[dict, dict]:
    if gripper_key not in database:
        raise KeyError(f"Unknown hand gripper: {gripper_key}")
    gripper = database[gripper_key]
    for preset in gripper.get("preset", []):
        if preset.get("name") == preset_name:
            return gripper, preset
    raise KeyError(f"Preset {preset_name!r} was not found under {gripper_key!r}")


def _normalize_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must contain [w, x, y, z]")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("Quaternion cannot have zero length")
    return quaternion / norm


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
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
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def rotation_z(yaw_deg: float) -> np.ndarray:
    angle = math.radians(float(yaw_deg))
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def pose_matrix(pose: dict) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(pose["orientation_wxyz"])
    matrix[:3, 3] = np.asarray(pose["position"], dtype=float)
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
    return (np.asarray(matrix, dtype=float) @ homogeneous.T).T[:, :3]


def _preset_bbox_world_points(preset: dict) -> np.ndarray:
    """Reconstruct every stored grasp-BBox corner in the preset world frame."""
    start_tf = pose_matrix(preset["start_base_tf"])
    fingertips = preset.get("fingertip_points", {})
    if not isinstance(fingertips, dict) or not fingertips:
        raise ValueError(
            f"Preset {preset.get('name', '<unnamed>')!r} has no fingertip grasp_bbox"
        )

    world_bboxes: list[np.ndarray] = []
    for fingertip_name, fingertip in fingertips.items():
        if not isinstance(fingertip, dict):
            raise ValueError(f"Invalid fingertip entry: {fingertip_name!r}")
        bbox = fingertip.get("grasp_bbox", {})
        if "points" in bbox:
            corners = np.asarray(bbox["points"], dtype=float)
            if corners.shape != (4, 3):
                raise ValueError(
                    f"{fingertip_name}.grasp_bbox.points must be 4x3, got "
                    f"{corners.shape}"
                )
        elif "min" in bbox and "max" in bbox:
            minimum = np.asarray(bbox["min"], dtype=float)
            maximum = np.asarray(bbox["max"], dtype=float)
            z_value = float(min(minimum[2], maximum[2]))
            corners = np.asarray(
                [
                    [minimum[0], minimum[1], z_value],
                    [maximum[0], minimum[1], z_value],
                    [maximum[0], maximum[1], z_value],
                    [minimum[0], maximum[1], z_value],
                ],
                dtype=float,
            )
        else:
            raise ValueError(
                f"{fingertip_name}.grasp_bbox requires points or min/max"
            )

        frame = str(fingertip.get("frame", "world")).lower()
        if frame == "gripper_base":
            if str(fingertip.get("base_pose", "start")).lower() != "start":
                raise ValueError(
                    f"{fingertip_name}.base_pose must be 'start'"
                )
            corners = transform_points(start_tf, corners)
        elif frame != "world":
            raise ValueError(
                f"Unsupported fingertip frame {fingertip.get('frame')!r}"
            )

        if "world_min_z" in bbox:
            corners[:, 2] = float(bbox["world_min_z"])
        if not np.all(np.isfinite(corners)):
            raise ValueError(f"{fingertip_name}.grasp_bbox contains non-finite values")
        world_bboxes.append(corners)

    return np.concatenate(world_bboxes, axis=0)


def preset_tcp_world_point(preset: dict) -> np.ndarray:
    """Return ``(global bbox minimum + maximum) / 2`` in world coordinates."""
    points = _preset_bbox_world_points(preset)
    return (np.min(points, axis=0) + np.max(points, axis=0)) * 0.5


def preset_tcp_local_point(preset: dict) -> np.ndarray:
    """Return the common grasp-BBox extent center in START base-local coordinates."""
    world_point = preset_tcp_world_point(preset)
    start_tf_inverse = np.linalg.inv(pose_matrix(preset["start_base_tf"]))
    return transform_points(start_tf_inverse, world_point[None])[0]


def disable_dangling_joints(stage, gripper_root_prim_path: str) -> list[str]:
    """Deactivate referenced joints whose body relationships target absent prims."""
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(gripper_root_prim_path)
    if not root.IsValid():
        raise ValueError(f"Invalid gripper root prim: {gripper_root_prim_path}")
    disabled: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
        if any(not stage.GetPrimAtPath(target).IsValid() for target in targets):
            prim.SetActive(False)
            disabled.append(str(prim.GetPath()))
    return disabled


def _set_world_aligned_tcp(stage, tcp_prim_path: str, position: Sequence[float]) -> None:
    from isaacsim.core.prims import SingleXFormPrim

    tcp = SingleXFormPrim(tcp_prim_path, reset_xform_properties=True)
    tcp.set_world_pose(
        position=np.asarray(position, dtype=float),
        orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )


def _set_gripper_root_pose(gripper_root_prim_path: str, pose: dict) -> None:
    from isaacsim.core.prims import SingleXFormPrim

    root = SingleXFormPrim(gripper_root_prim_path, reset_xform_properties=True)
    root.set_world_pose(
        position=np.asarray(pose["position"], dtype=float),
        orientation=_normalize_quaternion_wxyz(pose["orientation_wxyz"]),
    )


def _apply_start_joint_state_to_usd(stage, root_path: str, joint_positions: dict) -> list[str]:
    """Set drive targets and initial JointState values before PhysX initialization."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    applied: list[str] = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in joint_positions:
            continue
        value_rad = float(joint_positions[name])
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_name = "angular"
            usd_value = math.degrees(value_rad)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_name = "linear"
            usd_value = value_rad
        else:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
        if not drive or not drive.GetTargetPositionAttr().IsValid():
            continue
        drive.GetTargetPositionAttr().Set(usd_value)
        state = PhysxSchema.JointStateAPI.Apply(prim, drive_name)
        state.CreatePositionAttr().Set(usd_value)
        applied.append(name)
    return applied


def find_articulation_root_path(stage, gripper_root_prim_path: str) -> str:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(gripper_root_prim_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"No ArticulationRootAPI found below {gripper_root_prim_path}")


def instantiate_preset_start_gripper(
    stage,
    gripper: dict,
    preset: dict,
    gripper_root_prim_path: str,
    tcp_prim_path: str,
    *,
    replace_existing: bool = False,
) -> dict:
    """Load/reuse a hand USD and author its preset START pose on the current stage.

    Call ``world.reset()`` after this function, then call
    :func:`synchronize_articulation_start_pose` before extracting geometry.
    """
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(gripper_root_prim_path)
    if root.IsValid() and replace_existing:
        stage.RemovePrim(gripper_root_prim_path)
        root = stage.GetPrimAtPath(gripper_root_prim_path)

    asset_path = gripper_root_prim_path
    if not root.IsValid():
        UsdGeom.Xform.Define(stage, gripper_root_prim_path)
        asset_path = f"{gripper_root_prim_path}/Asset"
        asset = UsdGeom.Xform.Define(stage, asset_path).GetPrim()
        if not asset.GetReferences().AddReference(gripper["usd_path"]):
            raise RuntimeError(f"Could not reference hand USD: {gripper['usd_path']}")

    _set_gripper_root_pose(gripper_root_prim_path, preset["start_base_tf"])
    disabled = disable_dangling_joints(stage, gripper_root_prim_path)
    applied = _apply_start_joint_state_to_usd(
        stage, gripper_root_prim_path, preset["start_joint_pos"]
    )
    missing = sorted(set(preset["start_joint_pos"]) - set(applied))
    if missing:
        raise RuntimeError(f"START joints not found as driven USD joints: {', '.join(missing)}")

    tcp_world = preset_tcp_world_point(preset)
    _set_world_aligned_tcp(stage, tcp_prim_path, tcp_world)
    return {
        "asset_path": asset_path,
        "articulation_root_path": find_articulation_root_path(stage, gripper_root_prim_path),
        "tcp_world_point": tcp_world,
        "disabled_dangling_joints": disabled,
    }


def synchronize_articulation_start_pose(articulation_root_path: str, preset: dict):
    """Teleport initialized PhysX DOFs to the exact preset START positions."""
    from isaacsim.core.prims import SingleArticulation

    articulation = SingleArticulation(
        prim_path=articulation_root_path,
        name=f"heightmap_{preset['name']}",
        reset_xform_properties=False,
    )
    articulation.initialize()
    missing = [name for name in preset["start_joint_pos"] if name not in articulation.dof_names]
    if missing:
        raise RuntimeError(f"Articulation is missing START DOFs: {', '.join(missing)}")
    names = list(preset["start_joint_pos"])
    indices = np.asarray([articulation.get_dof_index(name) for name in names], dtype=np.int32)
    positions = np.asarray([preset["start_joint_pos"][name] for name in names], dtype=np.float32)
    articulation.set_joint_positions(positions, joint_indices=indices)
    articulation.set_joint_velocities(np.zeros_like(positions), joint_indices=indices)
    return articulation


def _collision_enabled_on_mesh(prim) -> bool:
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False
    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    return enabled is not False


def collision_mesh_prims(stage, gripper_root_prim_path: str) -> list:
    """Return enabled collision Mesh prims below the requested gripper root."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(gripper_root_prim_path)
    if not root.IsValid():
        raise ValueError(f"Invalid gripper root prim: {gripper_root_prim_path}")
    meshes = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdGeom.Mesh) and _collision_enabled_on_mesh(prim)
    ]
    if not meshes:
        raise RuntimeError(f"No enabled collision Mesh found below {gripper_root_prim_path}")
    return meshes


def _triangulate_mesh(mesh_prim, world_matrix) -> list[np.ndarray]:
    from pxr import Usd, UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if not points or not counts or not indices:
        return []
    world_points = np.asarray(
        [
            tuple(world_matrix.Transform(point))
            for point in points
        ],
        dtype=float,
    )
    triangles: list[np.ndarray] = []
    offset = 0
    for count in counts:
        count = int(count)
        face = [int(index) for index in indices[offset:offset + count]]
        offset += count
        for index in range(1, count - 1):
            triangle = world_points[[face[0], face[index], face[index + 1]]]
            if np.all(np.isfinite(triangle)):
                triangles.append(triangle)
    return triangles


def extract_collision_triangles_tcp(
    stage,
    gripper_root_prim_path: str,
    tcp_prim_path: str,
) -> np.ndarray:
    """Read collision triangles at the current joint pose and express them relative to TCP."""
    import omni.usd

    tcp_prim = stage.GetPrimAtPath(tcp_prim_path)
    if not tcp_prim.IsValid():
        raise ValueError(f"Invalid TCP prim: {tcp_prim_path}")
    tcp_matrix = omni.usd.get_world_transform_matrix(tcp_prim)
    tcp_position = np.asarray(tuple(tcp_matrix.ExtractTranslation()), dtype=float)
    tcp_rotation = np.asarray(tcp_matrix.ExtractRotationMatrix(), dtype=float)
    if not np.allclose(tcp_rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(
            "TCP must be world-axis aligned so top-view XY/Z can be compared directly"
        )

    triangles: list[np.ndarray] = []
    for mesh_prim in collision_mesh_prims(stage, gripper_root_prim_path):
        world_matrix = omni.usd.get_world_transform_matrix(mesh_prim)
        triangles.extend(_triangulate_mesh(mesh_prim, world_matrix))
    if not triangles:
        raise RuntimeError("Collision meshes contained no readable triangles")
    return np.asarray(triangles, dtype=float) - tcp_position[None, None, :]


def _update_grid_at_points(
    height_map: np.ndarray,
    points: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    reducer: str,
) -> None:
    x_index = np.rint((points[:, 0] - x_min) / resolution).astype(np.int64)
    y_index = np.rint((points[:, 1] - y_min) / resolution).astype(np.int64)
    valid = (
        (x_index >= 0)
        & (x_index < height_map.shape[1])
        & (y_index >= 0)
        & (y_index < height_map.shape[0])
        & np.isfinite(points[:, 2])
    )
    if reducer == "minimum":
        np.minimum.at(height_map, (y_index[valid], x_index[valid]), points[valid, 2])
    elif reducer == "maximum":
        np.maximum.at(height_map, (y_index[valid], x_index[valid]), points[valid, 2])
    else:
        raise ValueError(f"Unknown grid reducer: {reducer}")


def sample_triangle_surfaces(
    collision_triangles_tcp: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Sample collision triangle surfaces once at sub-cell XY spacing.

    The resulting cloud is reused for all yaw angles.  This is substantially
    faster for detailed hand meshes than rasterizing every triangle 36 times.
    """
    triangles = np.asarray(collision_triangles_tcp, dtype=float)
    sampled: list[np.ndarray] = []
    target_spacing = resolution * 0.7
    for triangle in triangles:
        edges_xy = np.asarray(
            [
                np.linalg.norm((triangle[1] - triangle[0])[:2]),
                np.linalg.norm((triangle[2] - triangle[1])[:2]),
                np.linalg.norm((triangle[0] - triangle[2])[:2]),
            ]
        )
        subdivisions = max(1, int(math.ceil(float(edges_xy.max()) / target_spacing)))
        if subdivisions == 1:
            sampled.append(np.vstack((triangle, triangle.mean(axis=0, keepdims=True))))
            continue
        weights = []
        inverse = 1.0 / subdivisions
        for first in range(subdivisions + 1):
            for second in range(subdivisions + 1 - first):
                third = subdivisions - first - second
                weights.append((first * inverse, second * inverse, third * inverse))
        sampled.append(np.asarray(weights, dtype=float) @ triangle)
    if not sampled:
        raise ValueError("No collision triangle surface samples were generated")
    return np.vstack(sampled)


def build_bottom_height_maps(
    collision_triangles_tcp: np.ndarray,
    resolution: float,
    yaw_degrees: Iterable[int] = DEFAULT_YAWS,
    padding: float = 0.002,
) -> dict:
    """Build a common-grid TCP-relative bottom map for every requested yaw."""
    triangles = np.asarray(collision_triangles_tcp, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("collision_triangles_tcp must have shape (N, 3, 3)")
    if resolution <= 0.0:
        raise ValueError("Height-map resolution must be positive")
    yaw_values = np.asarray(list(yaw_degrees), dtype=np.int16)
    if len(yaw_values) == 0:
        raise ValueError("At least one yaw angle is required")

    radial_extent = float(np.linalg.norm(triangles[..., :2], axis=-1).max()) + float(padding)
    half_extent = math.ceil(radial_extent / resolution) * resolution
    grid_size = int(round(2.0 * half_extent / resolution)) + 1
    x_min = y_min = -half_extent
    maps = np.full((len(yaw_values), grid_size, grid_size), np.nan, dtype=np.float32)
    # Sample once, then reuse the exact same surface samples for every yaw.
    # Do not pre-collapse points in XY: two close samples can separate into
    # different cells after rotation, especially on thin finger links.
    surface_points = sample_triangle_surfaces(triangles, resolution)

    for yaw_index, yaw in enumerate(yaw_values):
        rotation = rotation_z(float(yaw))
        rotated = surface_points @ rotation.T
        bottom = np.full((grid_size, grid_size), np.inf, dtype=float)
        _update_grid_at_points(
            bottom, rotated, x_min, y_min, resolution, "minimum"
        )
        bottom[np.isinf(bottom)] = np.nan
        maps[yaw_index] = bottom.astype(np.float32)

    return {
        "bottom_height_maps": maps,
        "yaw_degrees": yaw_values,
        "resolution": float(resolution),
        "x_min": float(x_min),
        "y_min": float(y_min),
        "surface_sample_count": int(len(surface_points)),
    }


def save_height_map_archive(
    output_path: str | os.PathLike,
    height_maps: dict,
    gripper_model: str,
    preset: dict,
    gripper_root_prim_path: str,
    tcp_prim_path: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_joint_names = sorted(preset.get("start_joint_pos", {}))
    start_joint_positions = [
        float(preset["start_joint_pos"][name]) for name in start_joint_names
    ]
    np.savez_compressed(
        output_path,
        archive_schema_version=np.asarray(
            HEIGHT_MAP_ARCHIVE_SCHEMA_VERSION, dtype=np.int64
        ),
        bottom_height_maps=np.asarray(height_maps["bottom_height_maps"], dtype=np.float32),
        yaw_degrees=np.asarray(height_maps["yaw_degrees"], dtype=np.int16),
        resolution=np.asarray(height_maps["resolution"], dtype=np.float64),
        x_min=np.asarray(height_maps["x_min"], dtype=np.float64),
        y_min=np.asarray(height_maps["y_min"], dtype=np.float64),
        surface_sample_count=np.asarray(
            height_maps.get("surface_sample_count", 0), dtype=np.int64
        ),
        gripper_model=np.asarray(gripper_model),
        preset_name=np.asarray(preset["name"]),
        gripper_root_prim_path=np.asarray(gripper_root_prim_path),
        tcp_prim_path=np.asarray(tcp_prim_path),
        tcp_local_point=np.asarray(preset_tcp_local_point(preset), dtype=np.float64),
        start_base_position=np.asarray(preset["start_base_tf"]["position"], dtype=np.float64),
        start_base_orientation_wxyz=np.asarray(
            preset["start_base_tf"]["orientation_wxyz"], dtype=np.float64
        ),
        start_joint_names=np.asarray(start_joint_names, dtype=np.str_),
        start_joint_positions=np.asarray(start_joint_positions, dtype=np.float64),
    )
    return output_path


def load_height_map_archive(path: str | os.PathLike) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _quaternion_equivalent(
    first: Sequence[float], second: Sequence[float], atol: float
) -> bool:
    first = _normalize_quaternion_wxyz(first)
    second = _normalize_quaternion_wxyz(second)
    return bool(
        np.allclose(first, second, atol=atol, rtol=0.0)
        or np.allclose(first, -second, atol=atol, rtol=0.0)
    )


def validate_height_map_archive_preset(
    archive: dict,
    preset: dict,
    *,
    archive_path: str | os.PathLike | None = None,
    atol: float = 1.0e-7,
) -> None:
    """Reject a height map generated from a different/stale preset START pose."""
    label = str(archive_path) if archive_path is not None else "height-map archive"
    required = {
        "archive_schema_version",
        "tcp_local_point",
        "start_base_position",
        "start_base_orientation_wxyz",
        "start_joint_names",
        "start_joint_positions",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise ValueError(
            f"Stale {label}: missing preset signature fields {missing}. "
            "Regenerate height maps with Generate_Hand_Heightmap.py."
        )

    version = int(np.asarray(archive["archive_schema_version"]).item())
    if version != HEIGHT_MAP_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"Stale {label}: schema={version}, expected="
            f"{HEIGHT_MAP_ARCHIVE_SCHEMA_VERSION}. Regenerate height maps."
        )

    differences: list[str] = []
    stored_tcp = np.asarray(archive["tcp_local_point"], dtype=float)
    current_tcp = preset_tcp_local_point(preset)
    if not np.allclose(stored_tcp, current_tcp, atol=atol, rtol=0.0):
        differences.append(
            f"bbox_extent_center max_delta="
            f"{float(np.max(np.abs(stored_tcp - current_tcp))):.9g}m"
        )

    stored_position = np.asarray(archive["start_base_position"], dtype=float)
    current_position = np.asarray(preset["start_base_tf"]["position"], dtype=float)
    if not np.allclose(stored_position, current_position, atol=atol, rtol=0.0):
        differences.append(
            f"start_base_position max_delta="
            f"{float(np.max(np.abs(stored_position - current_position))):.9g}m"
        )

    if not _quaternion_equivalent(
        archive["start_base_orientation_wxyz"],
        preset["start_base_tf"]["orientation_wxyz"],
        atol,
    ):
        differences.append("start_base_orientation")

    stored_names = [str(name) for name in np.asarray(archive["start_joint_names"]).tolist()]
    current_names = sorted(preset.get("start_joint_pos", {}))
    if stored_names != current_names:
        differences.append("start_joint_names")
    else:
        stored_positions = np.asarray(archive["start_joint_positions"], dtype=float)
        current_positions = np.asarray(
            [preset["start_joint_pos"][name] for name in current_names], dtype=float
        )
        if not np.allclose(stored_positions, current_positions, atol=atol, rtol=0.0):
            differences.append(
                f"start_joint_positions max_delta="
                f"{float(np.max(np.abs(stored_positions - current_positions))):.9g}rad"
            )

    if differences:
        raise ValueError(
            f"Stale {label}: preset signature mismatch ({'; '.join(differences)}). "
            "Regenerate height maps with Generate_Hand_Heightmap.py."
        )


def select_yaw_height_map(archive: dict, yaw_deg: float) -> tuple[np.ndarray, float]:
    yaws = np.asarray(archive["yaw_degrees"], dtype=float)
    requested = float(yaw_deg) % 360.0
    circular_distance = np.abs((yaws - requested + 180.0) % 360.0 - 180.0)
    index = int(np.argmin(circular_distance))
    return np.asarray(archive["bottom_height_maps"][index], dtype=float), float(yaws[index])


def top_depth_to_world_points(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    camera_world_transform: np.ndarray,
    *,
    isaac_top_camera_flip_x_180: bool = True,
) -> np.ndarray:
    """Back-project an Isaac top-view depth image into world XYZ points."""
    depth = np.asarray(depth_image, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    camera_world_transform = np.asarray(camera_world_transform, dtype=float)
    rows, columns = np.nonzero(np.isfinite(depth) & (depth > 0.0))
    z = depth[rows, columns]
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
    points = np.vstack((x, y, z, np.ones_like(z)))
    if isaac_top_camera_flip_x_180:
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        camera_world_transform = camera_world_transform @ flip
    return (camera_world_transform @ points)[:3].T


def scene_height_map_aligned_to_gripper(
    scene_world_points: np.ndarray,
    grasp_x: float,
    grasp_y: float,
    gripper_height_map: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
) -> np.ndarray:
    """Rasterize top-view scene points onto a gripper map aligned at grasp X/Y."""
    scene_world_points = np.asarray(scene_world_points, dtype=float)
    x_max = x_min + (gripper_height_map.shape[1] - 1) * float(resolution)
    y_max = y_min + (gripper_height_map.shape[0] - 1) * float(resolution)
    mask = (
        (scene_world_points[:, 0] >= float(grasp_x) + x_min)
        & (scene_world_points[:, 0] <= float(grasp_x) + x_max)
        & (scene_world_points[:, 1] >= float(grasp_y) + y_min)
        & (scene_world_points[:, 1] <= float(grasp_y) + y_max)
    )
    points = scene_world_points[mask].copy()
    points[:, 0] -= float(grasp_x)
    points[:, 1] -= float(grasp_y)
    scene_map = np.full(gripper_height_map.shape, -np.inf, dtype=float)
    if len(points) > 0:
        _update_grid_at_points(scene_map, points, x_min, y_min, resolution, "maximum")
    scene_map[np.isneginf(scene_map)] = np.nan
    return scene_map


def calculate_first_contact_tcp_height(
    scene_height_map: np.ndarray,
    gripper_bottom_height_map: np.ndarray,
    safety_margin: float,
) -> tuple[float, float]:
    """Return (first contact TCP Z, final safe approach TCP Z)."""
    scene = np.asarray(scene_height_map, dtype=float)
    gripper = np.asarray(gripper_bottom_height_map, dtype=float)
    if scene.shape != gripper.shape:
        raise ValueError("Scene and gripper height maps must have identical shapes")
    overlap = np.isfinite(scene) & np.isfinite(gripper)
    if not np.any(overlap):
        raise ValueError("Scene and gripper height maps have no valid XY overlap")
    difference = np.full(scene.shape, np.nan, dtype=float)
    difference[overlap] = scene[overlap] - gripper[overlap]
    first_contact_z = float(np.nanmax(difference))
    return first_contact_z, first_contact_z + float(safety_margin)


def _matrix_to_rpy_degrees(matrix: np.ndarray) -> list[float]:
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw]).tolist()


def target_pose_from_tcp(
    pose: dict,
    template_tcp_world: Sequence[float],
    target_tcp_world: Sequence[float],
    yaw_deg: float,
) -> dict:
    """Place one preset base pose so its grasp-center TCP reaches the target."""
    target_tcp = np.asarray(target_tcp_world, dtype=float)
    template_tcp = np.asarray(template_tcp_world, dtype=float)
    yaw_rotation = rotation_z(yaw_deg)
    target_position = target_tcp + yaw_rotation @ (
        np.asarray(pose["position"], dtype=float) - template_tcp
    )
    target_rotation = yaw_rotation @ quaternion_wxyz_to_matrix(pose["orientation_wxyz"])
    return {
        "frame": "world",
        "position": target_position.tolist(),
        "orientation_wxyz": matrix_to_quaternion_wxyz(target_rotation).tolist(),
        "rpy_deg": _matrix_to_rpy_degrees(target_rotation),
    }


def target_base_tf_from_tcp(
    preset: dict,
    target_tcp_world: Sequence[float],
    yaw_deg: float,
) -> dict:
    """Return direct START/END base transforms about the same target TCP."""
    template_tcp = preset_tcp_world_point(preset)
    return {
        "start": target_pose_from_tcp(
            preset["start_base_tf"], template_tcp, target_tcp_world, yaw_deg
        ),
        "end": target_pose_from_tcp(
            preset["end_base_tf"], template_tcp, target_tcp_world, yaw_deg
        ),
    }


def preset_contact_sensor_sets(preset: dict) -> dict[str, list[str]]:
    """Group fingertip rigid-body sensor paths for saved preset metadata.

    Presets saved before ``contact_sensor_path`` was introduced are supported
    by using the parent link of their selected mesh path.  Missing legacy
    ``contact_set`` values each receive a distinct set, matching the pre-grasp
    BBox filter's compatibility rule. The grasp-time lost-contact check treats
    the resulting active sensor collection as one global OR, independent of
    these groups.
    """
    fingertips = preset.get("fingertip_points", {})
    if not isinstance(fingertips, dict) or not fingertips:
        raise ValueError(
            f"Preset {preset.get('name', '<unnamed>')!r} has no fingertip_points"
        )

    fingertip_values = [
        value for value in fingertips.values() if isinstance(value, dict)
    ]
    explicit_sets = {
        value.get("contact_set")
        for value in fingertip_values
        if isinstance(value.get("contact_set"), int)
        and not isinstance(value.get("contact_set"), bool)
        and value.get("contact_set") >= 1
    }
    used_sets = set(explicit_sets)
    next_set = 1
    grouped: dict[str, list[str]] = {}
    for fingertip in fingertip_values:
        raw_set = fingertip.get("contact_set")
        if isinstance(raw_set, int) and not isinstance(raw_set, bool) and raw_set >= 1:
            set_id = raw_set
        else:
            while next_set in used_sets:
                next_set += 1
            set_id = next_set
            used_sets.add(set_id)
            next_set += 1

        sensor_path = fingertip.get("contact_sensor_path")
        if not isinstance(sensor_path, str) or not sensor_path:
            mesh_path = fingertip.get("mesh_path")
            if not isinstance(mesh_path, str) or not mesh_path.startswith("/"):
                raise ValueError(
                    f"Preset {preset.get('name', '<unnamed>')!r} has a fingertip "
                    "without contact_sensor_path or a usable mesh_path"
                )
            stripped = mesh_path.rstrip("/")
            sensor_path = stripped.rsplit("/", 1)[0] if "/" in stripped[1:] else stripped

        group = grouped.setdefault(str(set_id), [])
        if sensor_path not in group:
            group.append(sensor_path)
    return grouped


def make_pregrasp_record(
    archive: dict,
    preset: dict,
    scene_world_points: np.ndarray,
    grasp_x: float,
    grasp_y: float,
    yaw_deg: float,
    safety_margin: float,
    target_object: str,
    gripper: dict | None = None,
) -> dict:
    """Match one stored gripper map and return the legacy-compatible pre-grasp record."""
    stored_preset = str(np.asarray(archive["preset_name"]).item())
    if stored_preset != preset.get("name"):
        raise ValueError(
            f"Height-map preset {stored_preset!r} does not match requested "
            f"preset {preset.get('name')!r}"
        )
    gripper_map, selected_yaw = select_yaw_height_map(archive, yaw_deg)
    resolution = float(np.asarray(archive["resolution"]).item())
    x_min = float(np.asarray(archive["x_min"]).item())
    y_min = float(np.asarray(archive["y_min"]).item())
    scene_map = scene_height_map_aligned_to_gripper(
        scene_world_points,
        grasp_x,
        grasp_y,
        gripper_map,
        x_min,
        y_min,
        resolution,
    )
    first_contact_z, target_z = calculate_first_contact_tcp_height(
        scene_map, gripper_map, safety_margin
    )
    target_tcp = np.asarray([grasp_x, grasp_y, target_z], dtype=float)
    target_base_tf = target_base_tf_from_tcp(preset, target_tcp, selected_yaw)
    start_pose = target_base_tf["start"]
    gripper_model = str(np.asarray(archive["gripper_model"]).item())
    return {
        "gripper_model": gripper_model,
        "gripper_type": "Hand",
        "preset_name": preset["name"],
        "target_object": target_object,
        "target_points": target_tcp.tolist(),
        "target_height": target_z,
        # Compatibility field. Hand execution should use target_base_tf.
        "target_orientation": start_pose["rpy_deg"],
        "target_width": 0.0,
        "target_base_tf": target_base_tf,
        "target_joint_pos": {
            "start": dict(preset.get("start_joint_pos", {})),
            "end": dict(preset.get("end_joint_pos", {})),
        },
        "joint_unit": preset.get("joint_unit", "rad"),
        "transition": dict(preset.get("transition", {})),
        "contact_sensor_sets": preset_contact_sensor_sets(preset),
        "gripper_usd_path": None if gripper is None else gripper.get("usd_path"),
        "selected_heightmap_yaw": selected_yaw,
        "first_contact_z": first_contact_z,
        "safety_margin": float(safety_margin),
    }


def save_pregrasp_records(
    output_path: str | os.PathLike,
    gripper_model: str,
    records: Sequence[dict],
) -> Path:
    """Save the same outer/data layout used by the existing pre-grasp pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"gripper_model": gripper_model, "data": list(records)}]
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)
    return output_path


def build_archive_from_current_stage(
    stage,
    gripper_root_prim_path: str,
    tcp_prim_path: str,
    output_path: str | os.PathLike,
    gripper_model: str,
    preset: dict,
    resolution: float,
    yaw_degrees: Iterable[int] = DEFAULT_YAWS,
) -> Path:
    triangles = extract_collision_triangles_tcp(stage, gripper_root_prim_path, tcp_prim_path)
    height_maps = build_bottom_height_maps(triangles, resolution, yaw_degrees)
    return save_height_map_archive(
        output_path,
        height_maps,
        gripper_model,
        preset,
        gripper_root_prim_path,
        tcp_prim_path,
    )


# Minimal in-process usage after Isaac Sim and the current Stage are ready:
#
# database = load_hand_database()
# gripper, preset = get_gripper_and_preset(database, "Inspire-F1", "3f_grip")
# setup = instantiate_preset_start_gripper(
#     stage, gripper, preset, "/World/HeightMapHand", "/World/HeightMapTCP"
# )
# world.reset()
# synchronize_articulation_start_pose(setup["articulation_root_path"], preset)
# world.step(render=False)
# build_archive_from_current_stage(
#     stage, "/World/HeightMapHand", "/World/HeightMapTCP",
#     "heightmaps/Inspire-F1__3f_grip.npz", "Inspire-F1", preset, 0.002,
# )
