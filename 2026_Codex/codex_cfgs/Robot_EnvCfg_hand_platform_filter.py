"""Isaac Lab environment for hand grippers with robot-platform filtered pairs.

The legacy ``Robot_EnvCfg.py`` remains untouched and is still used for all
two/three-finger grippers.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import posixpath
import re

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from cfgs import scene_cfg as SC
from .action_policy_hand_new import HandActionPolicy, _quaternion_slerp
from .grasp_physics_settings import (
    ARTICULATION_POSITION_ITERATION_COUNT,
    ARTICULATION_VELOCITY_ITERATION_COUNT,
    SIM_DT,
    configure_scene_object_rigid_props,
    make_grasp_simulation_cfg,
)


EMPTY_HAND_ENV_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=ARTICULATION_POSITION_ITERATION_COUNT,
            solver_velocity_iteration_count=ARTICULATION_VELOCITY_ITERATION_COUNT,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
    actuators={},
)


class EnvGroupedContactSensor(ContactSensor):
    """Group nested wildcard sensor parents back into their IsaacLab envs.

    A pattern such as ``env_.*/Robot/.../F.*/finger`` makes the stock contact
    sensor treat every F parent as a separate environment.  PhysX therefore
    exposes ``num_envs * fingers`` rows with one body each.  This class keeps
    that valid PhysX view while recording which physical rows belong to each
    logical ``env_N`` and expands reset indices accordingly.
    """

    _ENV_PATH_PATTERN = re.compile(r"/env_(\d+)(?:/|$)")

    def _initialize_impl(self):
        super()._initialize_impl()
        rows_by_env: dict[int, list[int]] = {}
        for row, prim in enumerate(self._parent_prims):
            match = self._ENV_PATH_PATTERN.search(prim.GetPath().pathString)
            if match is None:
                continue
            rows_by_env.setdefault(int(match.group(1)), []).append(row)

        if not rows_by_env:
            # Standalone assets do not have an env_N namespace. Preserve the
            # stock interpretation for diagnostic scripts.
            self._logical_env_rows = torch.arange(
                self._num_envs, dtype=torch.long, device=self._device
            )[:, None]
            return

        logical_env_ids = sorted(rows_by_env)
        if logical_env_ids != list(range(len(logical_env_ids))):
            raise RuntimeError(
                "Contact sensor env namespaces must be contiguous from env_0; "
                f"found {logical_env_ids}"
            )
        row_counts = {len(rows_by_env[env_id]) for env_id in logical_env_ids}
        if len(row_counts) != 1:
            raise RuntimeError(
                "Every logical environment must have the same number of contact "
                f"sensor parents; found {[len(rows_by_env[i]) for i in logical_env_ids]}"
            )
        self._logical_env_rows = torch.tensor(
            [rows_by_env[env_id] for env_id in logical_env_ids],
            dtype=torch.long,
            device=self._device,
        )
        print(
            "Contact sensor grouping: "
            f"logical_envs={len(logical_env_ids)} "
            f"rows_per_env={self._logical_env_rows.shape[1]} "
            f"physical_rows={self._num_envs} "
            f"bodies_per_row={self.num_bodies}"
        )

    @property
    def logical_env_rows(self) -> torch.Tensor:
        return self._logical_env_rows

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or not hasattr(self, "_logical_env_rows"):
            return super().reset(env_ids)
        logical_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self._logical_env_rows.device
        )
        physical_rows = self._logical_env_rows[logical_ids].reshape(-1)
        return super().reset(physical_rows)


def _contact_sensor_relative_pattern(gripper_info: dict) -> str:
    """Build one ContactSensorCfg pattern from the hand gripper-info paths."""
    configured_paths = gripper_info.get("contact_sensor_paths")
    if not isinstance(configured_paths, list) or not configured_paths:
        raise KeyError(
            f"{gripper_info.get('gripper_name')} needs non-empty "
            "contact_sensor_paths in the hand gripper info JSON"
        )

    sensor_paths: list[str] = []
    for path in configured_paths:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f"Invalid contact sensor path for {gripper_info.get('gripper_name')}: "
                f"{path!r}"
            )
        normalized = "/" + path.strip().strip("/")
        if normalized not in sensor_paths:
            sensor_paths.append(normalized)

    relative_paths = [path.strip("/") for path in sensor_paths]
    parent_paths = {posixpath.dirname(path) for path in relative_paths}
    if len(parent_paths) != 1:
        raise ValueError(
            "Configured contact sensor rigid bodies must share one USD parent for "
            f"Isaac Lab ContactSensorCfg, got: {sensor_paths}"
        )
    parent_path = parent_paths.pop()
    body_names = sorted({posixpath.basename(path) for path in relative_paths})
    if any(not name for name in body_names):
        raise ValueError(f"Invalid contact sensor rigid-body paths: {sensor_paths}")
    body_pattern = (
        re.escape(body_names[0])
        if len(body_names) == 1
        else "(?:" + "|".join(re.escape(name) for name in body_names) + ")"
    )
    return f"{parent_path}/{body_pattern}" if parent_path else body_pattern


def Set_RobotEnvCFG(envcfg, gripper_info, pre_grasp_data):
    """Configure a hand articulation without consulting preset data again."""
    if str(gripper_info.get("type", "")).lower() != "hand":
        raise ValueError("Robot_EnvCfg_hand_new only accepts a Hand gripper")
    if not pre_grasp_data:
        raise ValueError("Hand pre-grasp data is empty")

    first_start = pre_grasp_data[0].get("target_joint_pos", {}).get("start", {})
    joint_cfg = gripper_info.get("joint_cfg", {})
    if not joint_cfg:
        raise KeyError("Hand gripper_info is missing joint_cfg")

    missing = [name for name in joint_cfg if name not in first_start]
    if missing:
        raise KeyError(f"Hand pre-grasp START pose is missing joints: {missing}")

    actuators = {}
    for name in joint_cfg:
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=None,
            velocity_limit_sim=None,
            stiffness=None,
            damping=None,
        )

    joint_unit = str(pre_grasp_data[0].get("joint_unit", "rad")).lower()
    if joint_unit in {"rad", "radian", "radians"}:
        to_radians = float
    elif joint_unit in {"deg", "degree", "degrees"}:
        to_radians = math.radians
    else:
        raise ValueError(f"Unsupported hand joint unit: {joint_unit!r}")

    envcfg.robot_cfg.spawn.usd_path = gripper_info["usd_path"]
    envcfg.robot_cfg.actuator_value_resolution_debug_print = False
    envcfg.robot_cfg.init_state.joint_pos = {
        name: to_radians(first_start[name]) for name in joint_cfg
    }
    envcfg.robot_cfg.actuators = actuators
    envcfg.joint_names = list(joint_cfg)
    envcfg.gripper_info = gripper_info
    envcfg.contact_sensor_prim_path = _contact_sensor_relative_pattern(gripper_info)
    envcfg.scene.contact_sensor.prim_path = (
        f"{envcfg.robot_prim_path}/{envcfg.contact_sensor_prim_path}"
    )
    # Do not cap passive/mimic DOFs by default.  PhysX mimic constraints use
    # those DOFs internally, so forcing the same low gripper effort limit onto
    # them can make the hand feel weak when mimic damping is tuned in the USD.
    envcfg.passive_effort_limit = gripper_info.get("passive_effort_limit")
    configure_scene_object_rigid_props(envcfg.scene)


@configclass
class RobotEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 100.0
    action_space = 1
    observation_space = 4
    state_space = 0
    envs = 150
    dt = SIM_DT

    robot_prim_path = "/World/envs/env_.*/Robot"
    root_max_linear_speed = 0.5
    root_max_angular_speed = math.radians(180.0)
    z_compliance_force_threshold = 1.0
    z_compliance_stiffness = 2.0
    z_compliance_max_offset = 0.1
    z_compliance_spring = 0.0
    z_compliance_damping = 0.1
    z_compliance_max_speed = 0.08
    z_compliance_force_filter = 0.08
    # Maximum allowed hand-object penetration depth. PhysX reports penetration
    # as negative contact separation; separation < -threshold fails the grasp.
    # Unit: metres. Kept at the value previously selected in the wrapper.
    contact_penetration_threshold = 0.005
    print_contact_separation = False
    contact_separation_print_delta = 0.0001
    platform_drop_height = 0.50
    target_drop_failure_distance = 0.15
    passive_effort_limit = None
    filter_robot_platform_collision = True

    sim: SimulationCfg = make_grasp_simulation_cfg()
    robot_cfg: ArticulationCfg = EMPTY_HAND_ENV_CFG.replace(prim_path=robot_prim_path)
    scene: SC.SceneCfg = SC.SceneCfg(num_envs=envs, env_spacing=2, replicate_physics=True)
    scene.platform = RigidObjectCfg(
        prim_path="/World/envs/env_.*/platform",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            scale=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    scene.contact_sensor.debug_vis = False


class RobotEnv(DirectRLEnv):
    cfg: RobotEnvCfg

    def __init__(
        self,
        cfg: RobotEnvCfg,
        render_mode: str | None = None,
        pre_grasp_data=None,
        conf_data=None,
        debug=False,
        **kwargs,
    ):
        try:
            super().__init__(cfg, render_mode, **kwargs)
        except SystemExit as error:
            self._is_closed = True
            raise RuntimeError(
                "Isaac Lab exited while constructing the hand environment"
            ) from error
        except BaseException:
            # A partially initialized DirectRLEnv calls close() from __del__,
            # which can hide the original constructor exception.
            self._is_closed = True
            raise

        self._joint_indexes = torch.tensor(
            [self.robot.find_joints(name)[0][0] for name in self.cfg.joint_names],
            dtype=torch.long,
            device=self.device,
        )
        mimic_reset_joint_names = list(
            getattr(self.cfg, "mimic_reset_joint_names", [])
        )
        self._mimic_reset_joint_indexes = torch.tensor(
            [
                self.robot.find_joints(name)[0][0]
                for name in mimic_reset_joint_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._apply_passive_joint_effort_limit()
        self._print_hand_joint_limits_once()
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.pre_grasp_data = [] if pre_grasp_data is None else pre_grasp_data
        self.conf_data = {} if conf_data is None else conf_data
        self.debug = debug
        self._contact_sensor_rows_by_env = self._resolve_contact_sensor_rows_by_env()
        self.act_pol = self._make_action_policy()
        self._last_root_pose_command = self.robot.data.default_root_state[:, :7].clone()
        self._last_root_pose_command[:, :3] += self.scene.env_origins
        self._z_compliance_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._z_compliance_velocity = torch.zeros_like(self._z_compliance_offset)
        self._z_compliance_force = torch.zeros_like(self._z_compliance_offset)
        self._setup_platform_drop_handles()

        self.scene_var_dict = {}
        for index in range(10):
            item = getattr(self.scene.cfg, f"obj{index:02d}", None)
            if item is not None:
                self.scene_var_dict[item.class_name] = item

    def _setup_platform_drop_handles(self) -> None:
        self._last_platform_drop_offset = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _apply_platform_drop(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self.platform, "write_root_pose_to_sim"):
            return

        offsets = self.act_pol.platform_drop_offset[env_ids]
        root_state = self.platform.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += offsets
        velocity = torch.zeros_like(root_state[:, 7:])
        # The platform is teleported out of the way. Keep its velocity zero so
        # the 50 cm jump does not inject a large downward impulse into objects.
        self.platform.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.platform.write_root_velocity_to_sim(velocity, env_ids)
        self._last_platform_drop_offset[env_ids] = offsets

    def _apply_passive_joint_effort_limit(self) -> None:
        if self.cfg.passive_effort_limit is None:
            return
        all_joint_ids = torch.arange(self.robot.num_joints, dtype=torch.long, device=self.device)
        passive_mask = torch.ones(self.robot.num_joints, dtype=torch.bool, device=self.device)
        passive_mask[self._joint_indexes] = False
        passive_joint_ids = all_joint_ids[passive_mask]
        if len(passive_joint_ids) == 0:
            return
        self.robot.write_joint_effort_limit_to_sim(
            float(self.cfg.passive_effort_limit), joint_ids=passive_joint_ids
        )

    def _print_hand_joint_limits_once(self) -> None:
        effort_limits = self.robot.data.joint_effort_limits[0]
        stiffness = self.robot.data.default_joint_stiffness[0]
        damping = self.robot.data.default_joint_damping[0]
        controlled = set(int(index) for index in self._joint_indexes.cpu().tolist())
        print("Hand joint limits from PhysX:")
        for joint_id, name in enumerate(self.robot.joint_names):
            label = "controlled" if joint_id in controlled else "passive/mimic"
            print(
                f"  {name} ({label}): effort={float(effort_limits[joint_id]):.6g}, "
                f"stiffness={float(stiffness[joint_id]):.6g}, "
                f"damping={float(damping[joint_id]):.6g}"
            )

    def _resolve_contact_sensor_rows_by_env(self) -> torch.Tensor:
        rows = getattr(self.contact_sensor, "logical_env_rows", None)
        if rows is None:
            if self.contact_sensor._num_envs != self.num_envs:
                raise RuntimeError(
                    "Contact sensor physical rows do not match logical environments: "
                    f"sensor={self.contact_sensor._num_envs}, env={self.num_envs}"
                )
            rows = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )[:, None]
        if rows.shape[0] != self.num_envs:
            raise RuntimeError(
                "Contact sensor grouping does not match the environment count: "
                f"rows={tuple(rows.shape)}, env={self.num_envs}"
            )
        return rows

    def _make_action_policy(self):
        return HandActionPolicy(
            env_num=self.num_envs,
            pre_grasp_data=self.pre_grasp_data,
            conf_data=self.conf_data,
            gripper_info=self.cfg.gripper_info,
            joint_names=self.cfg.joint_names,
            joint_index=self._joint_indexes,
            joint_pos=self.joint_pos,
            step_dt=self.step_dt,
            device=self.device,
            contact_sensor=self.contact_sensor,
            frame_transformer=self.transformer,
            env_origin=self.scene.env_origins,
            debug=self.debug,
            contact_penetration_threshold=self.cfg.contact_penetration_threshold,
            print_contact_separation=self.cfg.print_contact_separation,
            contact_separation_print_delta=self.cfg.contact_separation_print_delta,
            platform_drop_height=self.cfg.platform_drop_height,
            target_drop_failure_distance=self.cfg.target_drop_failure_distance,
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        for index in range(10):
            object_cfg = getattr(self.scene.cfg, f"obj{index:02d}")
            if object_cfg is None:
                continue
            rigid_object = RigidObject(
                cfg=RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/obj{index:02d}",
                    spawn=object_cfg.spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=object_cfg.pos,
                        rot=object_cfg.quat,
                    ),
                )
            )
            rigid_object.class_name = object_cfg.class_name
            setattr(self, f"obj{index:02d}", rigid_object)

        self.platform = self.scene["platform"]
        self._define_platform_rigid_body_schema()
        self.transformer = self.scene["transformer"]
        self.contact_sensor = self.scene["contact_sensor"]
        self.scene.clone_environments(copy_from_source=False)
        self._filter_robot_platform_collision_pairs()
        self.scene.articulations["robot"] = self.robot

    def _define_platform_rigid_body_schema(self) -> None:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        )
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        )
        for env_index in range(self.num_envs):
            platform_path = f"/World/envs/env_{env_index}/platform"
            try:
                sim_utils.define_rigid_body_properties(platform_path, rigid_props)
                sim_utils.modify_collision_properties(platform_path, collision_props)
            except Exception as error:
                print(
                    f"[WARN] Could not force platform rigid-body schema on "
                    f"{platform_path}: {error}"
                )

    def _collision_prims_below(self, root_path: str) -> list:
        from pxr import Usd, UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return []
        prims = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                prims.append(prim)
        return prims

    def _filter_robot_platform_collision_pairs(self) -> None:
        if not self.cfg.filter_robot_platform_collision:
            return

        from pxr import UsdPhysics

        total_pairs = 0
        for env_index in range(self.num_envs):
            env_root = f"/World/envs/env_{env_index}"
            robot_collisions = self._collision_prims_below(f"{env_root}/Robot")
            platform_collisions = self._collision_prims_below(f"{env_root}/platform")
            if not robot_collisions or not platform_collisions:
                continue

            for robot_prim in robot_collisions:
                filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(robot_prim)
                relation = filtered_pairs.CreateFilteredPairsRel()
                for platform_prim in platform_collisions:
                    relation.AddTarget(platform_prim.GetPath())
                    total_pairs += 1

        print(f"Hand platform collision filter pairs: {total_pairs}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        del actions
        # ArticulationData replaces its cached joint-position tensor whenever
        # it refreshes from PhysX.  Keep the policy reference synchronized so
        # width-ready and closing checks use the current master joint rather
        # than the tensor captured when the policy was constructed.
        self.joint_pos = self.robot.data.joint_pos
        self.act_pol.joint_pos = self.joint_pos
        self.actions = self.act_pol.step()

    @staticmethod
    def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lw, lx, ly, lz = left.unbind(dim=1)
        rw, rx, ry, rz = right.unbind(dim=1)
        return torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=1,
        )

    def _limited_root_motion(
        self, desired_pose: torch.Tensor, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_pose = self._last_root_pose_command[env_ids]
        limited_pose = desired_pose.clone()

        delta_pos = desired_pose[:, :3] - previous_pose[:, :3]
        distance = torch.linalg.vector_norm(delta_pos, dim=1, keepdim=True)
        max_delta = float(self.cfg.root_max_linear_speed) * float(self.step_dt)
        pos_scale = (max_delta / distance.clamp_min(1.0e-8)).clamp(max=1.0)
        limited_pose[:, :3] = previous_pose[:, :3] + delta_pos * pos_scale

        dot = torch.abs(torch.sum(previous_pose[:, 3:7] * desired_pose[:, 3:7], dim=1))
        angle = 2.0 * torch.acos(dot.clamp(max=1.0))
        max_angle = float(self.cfg.root_max_angular_speed) * float(self.step_dt)
        rot_scale = (max_angle / angle.clamp_min(1.0e-8)).clamp(max=1.0)
        limited_pose[:, 3:7] = _quaternion_slerp(
            previous_pose[:, 3:7], desired_pose[:, 3:7], rot_scale[:, None]
        )

        root_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float, device=self.device)
        root_velocity[:, :3] = (limited_pose[:, :3] - previous_pose[:, :3]) / float(self.step_dt)
        previous_inv = previous_pose[:, 3:7].clone()
        previous_inv[:, 1:] *= -1.0
        delta_quat = self._quat_multiply(limited_pose[:, 3:7], previous_inv)
        delta_quat = torch.where(delta_quat[:, :1] < 0.0, -delta_quat, delta_quat)
        axis_norm = torch.linalg.vector_norm(delta_quat[:, 1:], dim=1, keepdim=True)
        angle = 2.0 * torch.atan2(axis_norm, delta_quat[:, :1].clamp_min(1.0e-8))
        root_velocity[:, 3:] = (
            delta_quat[:, 1:] / axis_norm.clamp_min(1.0e-8)
        ) * (angle / float(self.step_dt))
        return limited_pose, root_velocity

    def _apply_z_compliance(
        self, desired_pose: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self.contact_sensor.data, "net_forces_w"):
            return desired_pose

        sensor_rows = self._contact_sensor_rows_by_env[env_ids]
        forces = self.contact_sensor.data.net_forces_w[sensor_rows]
        force_z = forces[..., 2].clamp_min(0.0).sum(dim=(1, 2))
        force_alpha = (
            float(self.step_dt)
            / max(float(self.step_dt), float(self.cfg.z_compliance_force_filter))
        )
        self._z_compliance_force[env_ids] = torch.lerp(
            self._z_compliance_force[env_ids], force_z, force_alpha
        )
        active = self.act_pol.stage_num[env_ids] <= 1
        target_offset = (
            (self._z_compliance_force[env_ids] - float(self.cfg.z_compliance_force_threshold)).clamp_min(0.0)
            / float(self.cfg.z_compliance_stiffness)
        ).clamp(max=float(self.cfg.z_compliance_max_offset))
        target_offset = torch.where(active, target_offset, torch.zeros_like(target_offset))

        acceleration = (
            float(self.cfg.z_compliance_spring) * (target_offset - self._z_compliance_offset[env_ids])
            - float(self.cfg.z_compliance_damping) * self._z_compliance_velocity[env_ids]
        )
        self._z_compliance_velocity[env_ids] = (
            self._z_compliance_velocity[env_ids] + acceleration * float(self.step_dt)
        ).clamp(
            min=-float(self.cfg.z_compliance_max_speed),
            max=float(self.cfg.z_compliance_max_speed),
        )
        self._z_compliance_offset[env_ids] = (
            self._z_compliance_offset[env_ids]
            + self._z_compliance_velocity[env_ids] * float(self.step_dt)
        ).clamp(
            min=0.0,
            max=float(self.cfg.z_compliance_max_offset),
        )

        compliant_pose = desired_pose.clone()
        compliant_pose[:, 2] += self._z_compliance_offset[env_ids]
        return compliant_pose

    def _apply_action(self) -> None:
        # Include stage 3 here so the final lift pose produced immediately
        # before the done check is actually written to PhysX once.
        env_ids = torch.where(self.act_pol.action_enable == 1)[0]
        if len(env_ids) == 0:
            return

        joint_env_ids = torch.where(
            (self.act_pol.action_enable == 1) & self.act_pol.joint_command_dirty
        )[0]
        if len(joint_env_ids):
            self.robot.set_joint_position_target(
                self.actions[joint_env_ids],
                joint_ids=self._joint_indexes,
                env_ids=joint_env_ids,
            )
            self.act_pol.joint_command_dirty[joint_env_ids] = False
        root_pose = self.act_pol.root_pose[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose = self._apply_z_compliance(root_pose, env_ids)
        root_pose, root_velocity = self._limited_root_motion(root_pose, env_ids)
        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self._last_root_pose_command[env_ids] = root_pose
        self._apply_platform_drop(env_ids)

    def _get_observations(self) -> dict:
        return {"policy": self.act_pol.action_enable}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.act_pol.joint_pos = self.joint_pos
        time_out = self.episode_length_buf >= 1000 - 1
        return self.act_pol.get_done_idx(), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        self.act_pol.reset(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        assigned = self.act_pol.assigned_pregrasp[env_ids] >= 0
        assigned_envs = env_ids[assigned]
        if len(assigned_envs):
            default_root_state[assigned, :7] = self.act_pol.root_pose[assigned_envs]
            default_root_state[assigned, :3] += self.scene.env_origins[assigned_envs]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        if len(assigned_envs):
            assigned_rows = torch.where(assigned)[0]
            joint_pos[
                assigned_rows[:, None], self._joint_indexes[None, :]
            ] = self.act_pol.target_joint[assigned_envs]
            if len(self._mimic_reset_joint_indexes):
                # A mimic follower has no drive target of its own, but writing
                # the full articulation reset state with its default value can
                # violate the mimic constraint before the first physics step.
                # Initialize it consistently with the master START-width pose;
                # subsequent commands still target only the selected master.
                master_start = self.act_pol.target_joint[assigned_envs, :1]
                joint_pos[
                    assigned_rows[:, None],
                    self._mimic_reset_joint_indexes[None, :],
                ] = master_start.expand(
                    -1, len(self._mimic_reset_joint_indexes)
                )
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._last_root_pose_command[env_ids] = default_root_state[:, :7]
        self._z_compliance_offset[env_ids] = 0.0
        self._z_compliance_velocity[env_ids] = 0.0
        self._z_compliance_force[env_ids] = 0.0
        self._last_platform_drop_offset[env_ids] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._apply_platform_drop(env_ids)

        print("cur_grasp:", self.act_pol.current_grasp_num)

        for index in range(10):
            name = f"obj{index:02d}"
            if not hasattr(self, name):
                continue
            item = getattr(self, name)
            default = item.data.default_root_state[env_ids].clone()
            default[:, :3] += self.scene.env_origins[env_ids]
            item.write_root_pose_to_sim(default[:, :7], env_ids)
            item.write_root_velocity_to_sim(default[:, 7:], env_ids)

    def factory_reset(self):
        self.act_pol = self._make_action_policy()
        self._reset_idx(None)
        print("#############################    Hand factory reset complete")

    def _get_rewards(self):
        return None
