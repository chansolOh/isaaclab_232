"""Isaac Lab environment for one object and one platform-free gripper."""

from __future__ import annotations

from collections.abc import Sequence
import math
import posixpath
import re

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from . import calibration
from . import scene_cfg as SC
from .policy import APPROACH, STRESS, GraspPolicy, quaternion_multiply, quaternion_slerp


class EnvGroupedContactSensor(ContactSensor):
    """Map nested wildcard sensor rows back to their logical env_N."""

    _ENV_PATTERN = re.compile(r"/env_(\d+)(?:/|$)")

    def _initialize_impl(self):
        super()._initialize_impl()
        rows: dict[int, list[int]] = {}
        for row, prim in enumerate(self._parent_prims):
            match = self._ENV_PATTERN.search(prim.GetPath().pathString)
            if match:
                rows.setdefault(int(match.group(1)), []).append(row)
        if not rows:
            self._logical_env_rows = torch.arange(
                self._num_envs, device=self._device, dtype=torch.long
            )[:, None]
            return
        env_ids = sorted(rows)
        counts = {len(rows[env_id]) for env_id in env_ids}
        if env_ids != list(range(len(env_ids))) or len(counts) != 1:
            raise RuntimeError(f"Invalid contact sensor env grouping: {rows.keys()}")
        self._logical_env_rows = torch.tensor(
            [rows[env_id] for env_id in env_ids],
            device=self._device,
            dtype=torch.long,
        )

    @property
    def logical_env_rows(self) -> torch.Tensor:
        return self._logical_env_rows

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or not hasattr(self, "_logical_env_rows"):
            return super().reset(env_ids)
        logical = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        return super().reset(self._logical_env_rows[logical].reshape(-1))


EMPTY_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=4,
        ),
    ),
    # Keep the gripper away from the origin while PhysX creates the scene.
    # The policy reset moves it to the actual pre-grasp before collection.
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), joint_pos={}),
    actuators={},
)


def optional_number(value):
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def _contact_paths(gripper: dict) -> list[str]:
    if str(gripper.get("type", "")).lower() == "hand":
        paths = gripper.get("contact_sensor_paths", [])
    else:
        paths = [gripper.get("contact_sensor_prim_path", "")]
    result = [str(path).strip().strip("/") for path in paths if str(path).strip()]
    if not result:
        raise KeyError(f"{gripper.get('gripper_name')} has no contact sensor paths")
    return result


def _aggregate_contact_path(paths: list[str]) -> str:
    """Build one ContactSensor regex containing every configured sensor body."""
    parents = {posixpath.dirname(path) for path in paths}
    if len(parents) != 1:
        raise ValueError(
            "Contact sensor rigid bodies must share one USD parent, got: "
            f"{paths}"
        )
    parent = parents.pop()
    bodies = sorted({posixpath.basename(path) for path in paths})
    body_pattern = (
        re.escape(bodies[0])
        if len(bodies) == 1
        else "(?:" + "|".join(re.escape(body) for body in bodies) + ")"
    )
    return f"{parent}/{body_pattern}" if parent else body_pattern


def _matching_rigid_body_paths(usd_path: str, expression: str) -> list[str]:
    """Resolve a broad contact expression to one exact rigid body per sensor."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open gripper USD for contact discovery: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"Gripper USD has no default prim: {usd_path}")
    root_path = str(default_prim.GetPath()).rstrip("/")
    expressions = [expression]
    # One legacy gripper entry uses shell-style `/*` instead of regex `/.*`.
    if "/*" in expression and "/.*" not in expression:
        expressions.append(expression.replace("/*", "/.*"))
    patterns = [re.compile(pattern) for pattern in expressions]
    result: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath())
        if not path.startswith(f"{root_path}/"):
            continue
        relative = path[len(root_path) + 1 :]
        candidates = (relative, f"{default_prim.GetName()}/{relative}")
        if any(
            pattern.fullmatch(candidate)
            for pattern in patterns
            for candidate in candidates
        ):
            # UsdFileCfg remaps the default prim itself to cfg.prim_path, so
            # sensor paths must be relative to that prim.
            result.append(relative)
    if not result:
        raise RuntimeError(
            f"No rigid body in {usd_path} matches contact path {expression!r}"
        )
    return sorted(set(result))


def _usd_rigid_body_relative_path(usd_path: str) -> str:
    """Return the only rigid body's path relative to the USD default prim."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open object USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"Object USD has no default prim: {usd_path}")
    root_path = str(default_prim.GetPath()).rstrip("/")
    rigid_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and (
            str(prim.GetPath()) == root_path
            or str(prim.GetPath()).startswith(f"{root_path}/")
        )
    ]
    if len(rigid_paths) != 1:
        raise RuntimeError(
            f"Single-object grasp requires exactly one rigid body in {usd_path}, "
            f"got {rigid_paths}"
        )
    rigid_path = rigid_paths[0]
    return "" if rigid_path == root_path else rigid_path[len(root_path) + 1 :]


def configure_gripper(cfg, gripper: dict, pre_grasps: list[dict]) -> None:
    if cfg.penetration_backend != "contact_sensor":
        raise ValueError(
            "Only penetration_backend='contact_sensor' is currently active; "
            "the policy boundary is reserved for a future Warp backend"
        )
    is_hand = str(gripper.get("type", "")).lower() == "hand"
    if is_hand:
        names = list(gripper.get("joint_cfg", {}))
        if not names:
            raise KeyError("Hand gripper has no joint_cfg")
        start = pre_grasps[0].get("target_joint_pos", {}).get("start", {})
        missing = [name for name in names if name not in start]
        if missing:
            raise KeyError(f"Hand pre-grasp START pose missing joints: {missing}")
    else:
        calibration.validate(gripper)
        names = calibration.joint_names(gripper)

    actuators = {}
    for name in names:
        setting = gripper["joint_cfg"][name]
        if is_hand:
            effort = velocity = stiffness = damping = None
        else:
            effort = optional_number(setting.get("effort_limit"))
            velocity = optional_number(setting.get("velocity_limit"))
            stiffness = optional_number(setting.get("stiffness"))
            damping = optional_number(setting.get("damping"))
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=effort,
            velocity_limit_sim=velocity,
            stiffness=stiffness,
            damping=damping,
        )
    cfg.robot_cfg.spawn.usd_path = gripper["usd_path"]
    if is_hand:
        unit = str(pre_grasps[0].get("joint_unit", "rad")).lower()
        convert = math.radians if unit.startswith("deg") else float
        cfg.robot_cfg.init_state.joint_pos = {
            name: convert(start[name]) for name in names
        }
    else:
        cfg.robot_cfg.init_state.joint_pos = {
            name: float(value) for name, value in gripper.get("init_joint_pos", {}).items()
        }
    cfg.robot_cfg.actuators = actuators
    cfg.joint_names = names
    cfg.gripper_info = gripper

    cfg.contact_sensor_names = []
    cfg.penetration_contact_sensor_names = []
    cfg.penetration_contact_sensor_paths = []
    paths = _contact_paths(gripper)
    # Resolve the actual RigidBodyAPI prim. Some assets use /World as their
    # default prim and place the object one level below it.
    object_body_path = _usd_rigid_body_relative_path(cfg.scene.obj00.spawn.usd_path)
    object_filter = f"{{ENV_REGEX_NS}}/obj00"
    if object_body_path:
        object_filter = f"{object_filter}/{object_body_path}"
    all_gripper_body_paths = _matching_rigid_body_paths(gripper["usd_path"], ".*")
    if not is_hand:
        cfg.scene.contact_sensor.prim_path = f"{cfg.robot_prim_path}/{paths[0]}"
        cfg.scene.contact_sensor.class_type = EnvGroupedContactSensor
        # Keep a broad unfiltered sensor for aggregate force. A filtered PhysX
        # view cannot reliably pair multiple wildcard bodies with cloned envs.
        cfg.scene.contact_sensor.filter_prim_paths_expr = []
        cfg.contact_sensor_names.append("contact_sensor")
        if not cfg.enable_object_contact_sensor:
            # Fallback for assets where an object-side report is unavailable.
            # Monitor every rigid body, not only nominal fingertip names.
            for index, path in enumerate(all_gripper_body_paths):
                name = f"penetration_contact_sensor_{index:02d}"
                setattr(
                    cfg.scene,
                    name,
                    ContactSensorCfg(
                        prim_path=f"{cfg.robot_prim_path}/{path}",
                        update_period=0.0,
                        history_length=0,
                        debug_vis=False,
                        filter_prim_paths_expr=[object_filter],
                        max_contact_data_count_per_prim=int(
                            cfg.contact_max_data_count_per_prim
                        ),
                        class_type=EnvGroupedContactSensor,
                    ),
                )
                cfg.penetration_contact_sensor_names.append(name)
                cfg.penetration_contact_sensor_paths.append(path)
    else:
        # One unfiltered aggregate sensor reads force from every configured
        # fingertip. Detailed fingertip sensors below are fallback-only.
        cfg.scene.contact_sensor.prim_path = (
            f"{cfg.robot_prim_path}/{_aggregate_contact_path(paths)}"
        )
        cfg.scene.contact_sensor.class_type = EnvGroupedContactSensor
        cfg.scene.contact_sensor.filter_prim_paths_expr = []
        cfg.scene.contact_sensor.history_length = 8
        cfg.contact_sensor_names.append("contact_sensor")
        if not cfg.enable_object_contact_sensor:
            for index, path in enumerate(paths):
                name = f"penetration_contact_sensor_{index:02d}"
                setattr(
                    cfg.scene,
                    name,
                    ContactSensorCfg(
                        prim_path=f"{cfg.robot_prim_path}/{path}",
                        update_period=0.0,
                        history_length=0,
                        debug_vis=False,
                        filter_prim_paths_expr=[object_filter],
                        max_contact_data_count_per_prim=int(
                            cfg.contact_max_data_count_per_prim
                        ),
                        class_type=EnvGroupedContactSensor,
                    ),
                )
                cfg.penetration_contact_sensor_names.append(name)
                cfg.penetration_contact_sensor_paths.append(path)

    if cfg.enable_object_contact_sensor:
        # One object-side detailed view observes every articulation rigid body.
        # This replaces the many gripper-side detailed views; the unfiltered
        # aggregate fingertip sensor above remains solely for force detection.
        cfg.scene.obj00.spawn.activate_contact_sensors = True
        robot_filters = [
            f"{{ENV_REGEX_NS}}/Robot/{path}" for path in all_gripper_body_paths
        ]
        name = "object_penetration_contact_sensor"
        setattr(
            cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=object_filter,
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                filter_prim_paths_expr=robot_filters,
                max_contact_data_count_per_prim=int(
                    cfg.contact_max_data_count_per_prim
                ),
                class_type=EnvGroupedContactSensor,
            ),
        )
        cfg.penetration_contact_sensor_names.append(name)
        cfg.penetration_contact_sensor_paths.append(
            f"object:{object_body_path or '<root>'}->robot({len(robot_filters)} bodies)"
        )

    cfg.scene.obj00.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5,
        linear_damping=0,
    )
    object_collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=float(cfg.contact_offset),
        rest_offset=float(cfg.rest_offset),
    )
    cfg.scene.obj00.spawn.collision_props = object_collision_props
    if cfg.override_gripper_collision_offsets:
        cfg.robot_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=float(cfg.contact_offset),
            rest_offset=float(cfg.rest_offset),
        )


@configclass
class RobotEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 120.0
    action_space = 1
    observation_space = 1
    state_space = 0
    envs = 64
    robot_prim_path = "/World/envs/env_.*/Robot"
    max_stress_acceleration = 60.0
    contact_offset = 0.005
    rest_offset = 0.0
    contact_penetration_threshold = 0.005
    enable_object_contact_sensor = True
    penetration_backend = "contact_sensor"
    pre_stress_object_motion_threshold = 0.1
    max_relative_translation = 0.010
    max_relative_rotation_deg = 20.0
    contact_lost_duration = 0.10
    contact_max_data_count_per_prim = 256
    override_gripper_collision_offsets = False
    print_contact_separation = False
    contact_separation_print_delta = 0.0001
    root_max_linear_speed = 0.5
    root_max_angular_speed = math.radians(180.0)
    apply_finger_z_hop = False

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 800.0,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            enable_ccd=True,
            enable_external_forces_every_iteration=True,
            # For this fixed-root, position-controlled gripper, solving
            # articulation contacts last increases visible penetration.
            # Keep contact solving in the normal solver order by default.
            solve_articulation_contact_last=False,
            min_position_iteration_count=1,
            max_position_iteration_count=255,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=255,
            # Use standard PhysX buffer capacities for this single-object scene.
        ),
    )
    robot_cfg: ArticulationCfg = EMPTY_GRIPPER_CFG.replace(prim_path=robot_prim_path)
    scene: SC.SceneCfg = SC.SceneCfg(
        num_envs=envs,
        env_spacing=0.7,
        replicate_physics=True,
    )


class RobotEnv(DirectRLEnv):
    cfg: RobotEnvCfg

    def __init__(
        self,
        cfg: RobotEnvCfg,
        *,
        pre_grasp_data: list[dict],
        conf_data: dict,
        debug: bool = False,
        hold_completed: bool = False,
        **kwargs,
    ):
        self.pre_grasp_data = pre_grasp_data
        self.conf_data = conf_data
        self.debug = bool(debug)
        self.hold_completed = bool(hold_completed)
        super().__init__(cfg, **kwargs)
        self.replay_finished = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._joint_indexes = torch.tensor(
            [self.robot.find_joints(name)[0][0] for name in cfg.joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.contact_rows = [self._logical_rows(sensor) for sensor in self.contact_sensors]
        for index, (sensor, rows) in enumerate(
            zip(self.contact_sensors, self.contact_rows)
        ):
            self._validate_contact_sensor_rows(
                sensor, rows, f"aggregate_contact_sensor_{index:02d}"
            )
        self.penetration_contact_rows = [
            self._logical_rows(sensor) for sensor in self.penetration_contact_sensors
        ]
        self.contact_physical_row_to_env = [
            self._validate_contact_sensor_rows(
                sensor,
                rows,
                cfg.penetration_contact_sensor_names[index],
                require_filters=True,
            )
            for index, (sensor, rows) in enumerate(zip(
                self.penetration_contact_sensors, self.penetration_contact_rows
            ))
        ]
        self._printed_sensor_separation = torch.full(
            (len(self.penetration_contact_sensors), self.num_envs),
            torch.inf,
            dtype=torch.float,
            device=self.device,
        )
        self._substep_minimum_contact_separation = torch.full(
            (self.num_envs,), torch.inf, dtype=torch.float, device=self.device
        )
        self._physics_substep_index = 0
        self.policy = GraspPolicy(
            env_num=self.num_envs,
            pre_grasp_data=pre_grasp_data,
            conf_data=conf_data,
            gripper_info=cfg.gripper_info,
            joint_names=cfg.joint_names,
            joint_index=self._joint_indexes,
            joint_pos=self.robot.data.joint_pos,
            step_dt=self.step_dt,
            device=self.device,
            env_origin=self.scene.env_origins,
            object_initial_pos=(
                self.obj00.data.default_root_state[:, :3] + self.scene.env_origins
            ),
            contact_penetration_threshold=cfg.contact_penetration_threshold,
            pre_stress_object_motion_threshold=(
                cfg.pre_stress_object_motion_threshold
            ),
            max_relative_translation=cfg.max_relative_translation,
            max_relative_rotation_deg=cfg.max_relative_rotation_deg,
            contact_lost_duration=cfg.contact_lost_duration,
            apply_finger_z_hop=cfg.apply_finger_z_hop,
        )
        # Compatibility with the legacy main loop.
        self.act_pol = self.policy
        self.applied_force_n = torch.zeros(self.num_envs, device=self.device)
        self._last_root_pose_command = self.robot.data.default_root_state[:, :7].clone()
        self._last_root_pose_command[:, :3] += self.scene.env_origins
        # PhysX drive targets persist until they are changed. Cache the last
        # value so unchanged targets are not rewritten every physics substep.
        self._last_joint_position_target = torch.full_like(
            self.policy.target_joint, torch.nan
        )
        self._pending_joint_target_env_ids = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        if self.debug:
            details = ", ".join(
                f"{name}({path}):rows={int(sensor._num_envs)},"
                f"bodies={int(sensor.num_bodies)}"
                for name, path, sensor in zip(
                    cfg.penetration_contact_sensor_names,
                    cfg.penetration_contact_sensor_paths,
                    self.penetration_contact_sensors,
                )
            )
            print(
                f"ContactSensors > envs={self.num_envs} "
                f"aggregate={len(self.contact_sensors)} "
                f"penetration={len(self.penetration_contact_sensors)} [{details}]",
                flush=True,
            )

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.obj00 = self.scene["obj00"]
        self.obj00.class_name = self.scene.cfg.obj00.class_name
        self.contact_sensors = [
            self.scene[name] for name in self.cfg.contact_sensor_names
        ]
        self.penetration_contact_sensors = [
            self.scene[name] for name in self.cfg.penetration_contact_sensor_names
        ]
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    def _logical_rows(self, sensor) -> torch.Tensor:
        rows = getattr(sensor, "logical_env_rows", None)
        if rows is None:
            if sensor._num_envs != self.num_envs:
                raise RuntimeError(
                    f"Contact sensor rows={sensor._num_envs}, logical envs={self.num_envs}"
                )
            rows = torch.arange(self.num_envs, device=self.device)[:, None]
        return rows

    def _contact_force(self) -> torch.Tensor:
        result = torch.zeros(self.num_envs, device=self.device)
        for sensor, rows in zip(self.contact_sensors, self.contact_rows):
            history = sensor.data.net_forces_w_history[rows]
            magnitude = torch.linalg.vector_norm(history, dim=-1)
            result += magnitude.reshape(self.num_envs, -1).amax(dim=1)
        return result

    def _validate_contact_sensor_rows(
        self,
        sensor: ContactSensor,
        rows: torch.Tensor,
        label: str,
        *,
        require_filters: bool = False,
    ) -> torch.Tensor:
        """Validate and invert all physical rows, including multi-row envs."""
        if rows.ndim != 2 or rows.shape[0] != self.num_envs:
            raise ValueError(
                f"{label} rows must have shape ({self.num_envs}, rows_per_env), "
                f"got {tuple(rows.shape)}"
            )
        physical_row_count = int(sensor._num_envs)
        flat_rows = rows.reshape(-1)
        if (
            flat_rows.numel() != physical_row_count
            or torch.unique(flat_rows).numel() != physical_row_count
            or int(flat_rows.min()) != 0
            or int(flat_rows.max()) != physical_row_count - 1
        ):
            raise ValueError(
                f"{label} must map every physical row exactly once: "
                f"physical_rows={physical_row_count}, mapping={tuple(rows.shape)}"
            )
        mapping = torch.empty(
            physical_row_count, dtype=torch.long, device=self.device
        )
        logical = torch.arange(self.num_envs, device=self.device)[:, None]
        mapping[flat_rows] = logical.expand_as(rows).reshape(-1)
        view = sensor.contact_physx_view
        expected = physical_row_count * int(sensor.num_bodies)
        invalid_filters = require_filters and int(view.filter_count) < 1
        if int(view.sensor_count) != expected or invalid_filters:
            raise RuntimeError(
                f"Invalid detailed contact view for {label}: "
                f"sensors={int(view.sensor_count)}/{expected}, "
                f"filters={int(view.filter_count)}"
            )
        return mapping

    def _minimum_contact_separation(
        self, *, emit_debug: bool = False
    ) -> torch.Tensor:
        """Return deepest gripper-object contact separation per logical env."""
        minimum = torch.full(
            (self.num_envs,), torch.inf, dtype=torch.float, device=self.device
        )
        for sensor_index, (sensor, physical_row_to_env) in enumerate(zip(
            self.penetration_contact_sensors, self.contact_physical_row_to_env
        )):
            sensor_minimum = torch.full_like(minimum, torch.inf)
            view = sensor.contact_physx_view
            filter_count = int(view.filter_count)
            if filter_count < 1:
                continue
            _, _, _, separations, counts, starts = view.get_contact_data(
                dt=self.physics_dt
            )
            counts = counts.reshape(-1).to(device=self.device, dtype=torch.long)
            starts = starts.reshape(-1).to(device=self.device, dtype=torch.long)
            total_contacts = int(counts.sum().item())
            if total_contacts == 0:
                continue

            pair_ids = torch.repeat_interleave(
                torch.arange(counts.numel(), device=self.device), counts
            )
            packed_starts = counts.cumsum(0) - counts
            offsets = torch.arange(total_contacts, device=self.device) - (
                packed_starts.repeat_interleave(counts)
            )
            contact_indices = starts[pair_ids] + offsets
            valid_separations = separations.reshape(-1).index_select(
                0, contact_indices
            )
            sensor_ids = pair_ids // filter_count
            physical_row_ids = sensor_ids // int(sensor.num_bodies)
            env_ids = physical_row_to_env[physical_row_ids]
            sensor_minimum.scatter_reduce_(
                0, env_ids, valid_separations, reduce="amin", include_self=True
            )
            minimum = torch.minimum(minimum, sensor_minimum)
            if self.cfg.print_contact_separation and emit_debug:
                # Reset/unassigned envs can overlap at their default poses.
                # They are irrelevant to grasp validation and would otherwise
                # produce misleading source=-1 debug messages.
                finite = torch.isfinite(sensor_minimum) & (
                    self.policy.action_enable == 1
                )
                deeper = sensor_minimum < (
                    self._printed_sensor_separation[sensor_index]
                    - float(self.cfg.contact_separation_print_delta)
                )
                for env_id in torch.where(finite & deeper)[0].tolist():
                    value = float(sensor_minimum[env_id])
                    source = int(self.policy.assigned_pregrasp[env_id])
                    print(
                        "ContactSeparation > "
                        f"sensor={self.cfg.penetration_contact_sensor_names[sensor_index]} "
                        f"body={self.cfg.penetration_contact_sensor_paths[sensor_index]} "
                        f"env={env_id} source={source} "
                        f"separation={value:.7f}m "
                        f"penetration={max(0.0, -value) * 1000.0:.3f}mm",
                        flush=True,
                    )
                update = finite & deeper
                self._printed_sensor_separation[sensor_index, update] = (
                    sensor_minimum[update]
                )
        return minimum

    def _minimum_penetration_separation(
        self, *, emit_debug: bool = False
    ) -> torch.Tensor:
        """Backend boundary; a Warp query can replace this without policy changes."""
        if self.cfg.penetration_backend == "contact_sensor":
            return self._minimum_contact_separation(emit_debug=emit_debug)
        raise RuntimeError(
            f"Unsupported penetration backend: {self.cfg.penetration_backend!r}"
        )

    def _limited_root_motion(
        self, desired_pose: torch.Tensor, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Limit kinematic root jumps so contact solving can respond."""
        previous_pose = self._last_root_pose_command[env_ids]
        limited_pose = desired_pose.clone()

        delta_pos = desired_pose[:, :3] - previous_pose[:, :3]
        distance = torch.linalg.vector_norm(delta_pos, dim=1, keepdim=True)
        # _apply_action runs once per physics substep, not once per policy
        # step. Using step_dt here multiplies the configured speed by
        # decimation and can drive the fixed root through contacts.
        max_delta = float(self.cfg.root_max_linear_speed) * float(self.physics_dt)
        scale = (max_delta / distance.clamp_min(1.0e-8)).clamp(max=1.0)
        limited_pose[:, :3] = previous_pose[:, :3] + delta_pos * scale

        dot = torch.abs(
            torch.sum(previous_pose[:, 3:7] * desired_pose[:, 3:7], dim=1)
        )
        angle = 2.0 * torch.acos(dot.clamp(max=1.0))
        max_angle = float(self.cfg.root_max_angular_speed) * float(self.physics_dt)
        rotation_scale = (max_angle / angle.clamp_min(1.0e-8)).clamp(max=1.0)
        limited_pose[:, 3:7] = quaternion_slerp(
            previous_pose[:, 3:7], desired_pose[:, 3:7], rotation_scale[:, None]
        )

        velocity = torch.zeros((len(env_ids), 6), dtype=torch.float, device=self.device)
        velocity[:, :3] = (
            limited_pose[:, :3] - previous_pose[:, :3]
        ) / self.physics_dt
        previous_inv = previous_pose[:, 3:7].clone()
        previous_inv[:, 1:] *= -1.0
        delta_quat = quaternion_multiply(limited_pose[:, 3:7], previous_inv)
        delta_quat = torch.where(delta_quat[:, :1] < 0.0, -delta_quat, delta_quat)
        axis_norm = torch.linalg.vector_norm(delta_quat[:, 1:], dim=1, keepdim=True)
        rotation_angle = 2.0 * torch.atan2(
            axis_norm, delta_quat[:, :1].clamp_min(1.0e-8)
        )
        velocity[:, 3:] = (
            delta_quat[:, 1:] / axis_norm.clamp_min(1.0e-8)
        ) * (rotation_angle / self.physics_dt)
        return limited_pose, velocity

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        del actions
        self._substep_minimum_contact_separation.fill_(torch.inf)
        self._physics_substep_index = 0
        self.policy.joint_pos = self.robot.data.joint_pos
        self.actions = self.policy.step()
        changed = torch.any(
            torch.abs(self.actions - self._last_joint_position_target) > 1.0e-8,
            dim=1,
        ) | torch.any(torch.isnan(self._last_joint_position_target), dim=1)
        self._pending_joint_target_env_ids = torch.where(
            (self.policy.action_enable == 1) & changed
        )[0]

    def _apply_action(self) -> None:
        # scene.update() has refreshed sensor buffers after every preceding
        # physics substep. Accumulate them here so transient penetration is
        # not lost across DirectRLEnv decimation.
        if self._physics_substep_index > 0:
            self._substep_minimum_contact_separation = torch.minimum(
                self._substep_minimum_contact_separation,
                self._minimum_penetration_separation(),
            )
        self._physics_substep_index += 1
        env_ids = torch.where(self.policy.action_enable == 1)[0]
        if len(env_ids):
            # set_joint_position_target updates a persistent drive target.
            # Only submit a changed target on the first physics substep.
            command_env_ids = self._pending_joint_target_env_ids
            if self._physics_substep_index == 1 and len(command_env_ids):
                self.robot.set_joint_position_target(
                    self.actions[command_env_ids],
                    joint_ids=self._joint_indexes,
                    env_ids=command_env_ids,
                )
                self._last_joint_position_target[command_env_ids] = self.actions[
                    command_env_ids
                ]
            pose = self.policy.root_pose[env_ids].clone()
            pose[:, :3] += self.scene.env_origins[env_ids]
            pose, velocity = self._limited_root_motion(pose, env_ids)
            self.robot.write_root_pose_to_sim(pose, env_ids)
            self.robot.write_root_velocity_to_sim(velocity, env_ids)
            self._last_root_pose_command[env_ids] = pose
        self._apply_object_test_force()

    def _apply_object_test_force(self) -> None:
        view = self.obj00.root_physx_view
        forces = torch.zeros((view.count, 3), dtype=torch.float, device=self.device)
        self.applied_force_n.zero_()
        stress = torch.where(
            (self.policy.action_enable == 1) & (self.policy.stage_num == STRESS)
        )[0]
        if len(stress):
            progress = (
                self.policy.stage_step[stress].float() / self.policy.stress_steps
            ).clamp(0.0, 1.0)
            acceleration = 9.81 + (
                float(self.cfg.max_stress_acceleration) - 9.81
            ) * progress
            masses = self.obj00.data.default_mass.to(self.device)[stress].reshape(-1)
            force_n = masses * acceleration
            forces[stress] = self.policy.force_direction[stress] * force_n[:, None]
            self.applied_force_n[stress] = force_n
        all_ids = torch.arange(view.count, dtype=torch.int32, device=self.device)
        active = stress.to(dtype=torch.int32, device="cpu")
        if len(active):
            view.wake_up(active)
        view.apply_forces_and_torques_at_position(
            force_data=forces,
            torque_data=None,
            position_data=None,
            indices=all_ids,
            is_global=True,
        )

    def _get_observations(self) -> dict:
        return {"policy": self.policy.action_enable}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pose = self.obj00.data.root_link_pose_w
        robot_pose = self.robot.data.root_link_pose_w
        current_separation = self._minimum_penetration_separation(emit_debug=True)
        minimum_separation = torch.minimum(
            self._substep_minimum_contact_separation, current_separation
        )
        terminated = self.policy.observe(
            object_pos=object_pose[:, :3],
            object_quat=object_pose[:, 3:7],
            robot_pos=robot_pose[:, :3],
            robot_quat=robot_pose[:, 3:7],
            joint_pos=self.robot.data.joint_pos,
            contact_force=self._contact_force(),
            minimum_contact_separation=minimum_separation,
            applied_force=self.applied_force_n,
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.hold_completed:
            # Interactive replay must keep the final simulated pose on screen.
            # Suppress DirectRLEnv's automatic reset only for this opt-in mode.
            self.replay_finished |= terminated | time_out
            terminated = torch.zeros_like(terminated)
            time_out = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if hasattr(self, "replay_finished"):
            self.replay_finished[env_ids] = False
        self.policy.reset(env_ids)
        if hasattr(self, "_printed_sensor_separation"):
            self._printed_sensor_separation[:, env_ids] = torch.inf
        assigned = self.policy.assigned_pregrasp[env_ids] >= 0
        assigned_envs = env_ids[assigned]

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        if len(assigned_envs):
            root_state[assigned, :7] = self.policy.root_pose[assigned_envs]
            root_state[assigned, :3] += self.scene.env_origins[assigned_envs]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        if len(assigned_envs):
            rows = torch.where(assigned)[0]
            joint_pos[rows[:, None], self._joint_indexes[None]] = self.policy.target_joint[
                assigned_envs
            ]
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._last_root_pose_command[env_ids] = root_state[:, :7]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        reset_joint_target = joint_pos[:, self._joint_indexes]
        self.robot.set_joint_position_target(
            reset_joint_target,
            joint_ids=self._joint_indexes,
            env_ids=env_ids,
        )
        if hasattr(self, "_last_joint_position_target"):
            self._last_joint_position_target[env_ids] = reset_joint_target

        object_state = self.obj00.data.default_root_state[env_ids].clone()
        object_state[:, :3] += self.scene.env_origins[env_ids]
        self.obj00.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.obj00.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids
        )
        self.applied_force_n[env_ids] = 0.0
        self.scene.write_data_to_sim()
        self.sim.forward()

    def factory_reset(self) -> None:
        self.policy = GraspPolicy(
            env_num=self.num_envs,
            pre_grasp_data=self.pre_grasp_data,
            conf_data=self.conf_data,
            gripper_info=self.cfg.gripper_info,
            joint_names=self.cfg.joint_names,
            joint_index=self._joint_indexes,
            joint_pos=self.robot.data.joint_pos,
            step_dt=self.step_dt,
            device=self.device,
            env_origin=self.scene.env_origins,
            object_initial_pos=(
                self.obj00.data.default_root_state[:, :3] + self.scene.env_origins
            ),
            contact_penetration_threshold=self.cfg.contact_penetration_threshold,
            pre_stress_object_motion_threshold=(
                self.cfg.pre_stress_object_motion_threshold
            ),
            max_relative_translation=self.cfg.max_relative_translation,
            max_relative_rotation_deg=self.cfg.max_relative_rotation_deg,
            contact_lost_duration=self.cfg.contact_lost_duration,
            apply_finger_z_hop=self.cfg.apply_finger_z_hop,
        )
        self.act_pol = self.policy

    def _get_rewards(self):
        return None
