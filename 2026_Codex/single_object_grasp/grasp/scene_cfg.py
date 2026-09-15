"""Isaac Lab scene config containing exactly one object and no platform."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    obj00 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/obj00",
        spawn=sim_utils.UsdFileCfg(usd_path="", scale=(0.1, 0.1, 0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="",
        update_period=0.0,
        history_length=8,
        debug_vis=False,
    )


def rigid_usd_path(path: str | Path) -> str:
    path = Path(path)
    if path.stem.endswith("_rigid"):
        return str(path)
    candidate = path.with_name(f"{path.stem}_rigid{path.suffix}")
    return str(candidate if candidate.exists() else path)


def orientation_wxyz(orientation) -> tuple[float, float, float, float]:
    if len(orientation) == 4:
        return tuple(float(value) for value in orientation)
    if len(orientation) == 3:
        quaternion = euler_angles_to_quat(orientation, degrees=True)
        return tuple(float(value) for value in quaternion)
    raise ValueError("Object orient must be Euler XYZ degrees or WXYZ quaternion")


def configure_object(scene: SceneCfg, objects: list[dict]) -> None:
    if len(objects) != 1:
        raise ValueError(f"Single-object grasp requires one object, got {len(objects)}")
    obj = objects[0]
    scene.obj00.spawn.usd_path = rigid_usd_path(obj["usd_path"])
    scene.obj00.spawn.scale = tuple(float(value) for value in obj.get("scale", [1, 1, 1]))
    scene.obj00.class_name = str(obj["class"])
    scene.obj00.init_state.pos = tuple(float(value) for value in obj["translate"])
    scene.obj00.init_state.rot = orientation_wxyz(obj.get("orient", [0, 0, 0]))
