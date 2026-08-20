"""Isaac Lab environment for calibrated two/three-finger grippers.

It reuses the proven hand environment's root-pose limiting, kinematic platform
drop, object reset, and robot/platform filtered-pair implementation.  Finger
closing and calibrated Z-hop are supplied by ``FingerActionPolicy``.
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from cfgs import scene_cfg as SC
from .Robot_EnvCfg_hand_platform_filter import RobotEnv as _PlatformRobotEnv
from .Robot_EnvCfg_hand_platform_filter import RobotEnvCfg as _PlatformRobotEnvCfg
from .action_policy_finger_new import FingerActionPolicy
from .finger_gripper_calibration import (
    infer_calibration_joint_names,
    validate_calibration,
)
from .grasp_physics_settings import (
    ARTICULATION_POSITION_ITERATION_COUNT,
    ARTICULATION_VELOCITY_ITERATION_COUNT,
    SIM_DT,
    configure_scene_object_rigid_props,
    make_grasp_simulation_cfg,
)


EMPTY_FINGER_ENV_CFG = ArticulationCfg(
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


def _optional_number(value):
    if value is None or str(value).lower() == "none":
        return None
    return float(value)


def Set_RobotEnvCFG(envcfg, gripper_info):
    gripper_type = str(gripper_info.get("type", "")).lower()
    if not gripper_type.startswith(("finger2", "finger3")):
        raise ValueError(
            "Robot_EnvCfg_finger_platform_filter only accepts finger2/finger3 grippers"
        )
    validate_calibration(gripper_info)
    if "total_length" not in gripper_info:
        raise KeyError(
            f"{gripper_info.get('gripper_name')} needs total_length for base-TF approach"
        )

    controlled_names = infer_calibration_joint_names(gripper_info)
    joint_cfg = gripper_info.get("joint_cfg", {})
    actuators = {}
    for name in controlled_names:
        cfg = joint_cfg[name]
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=_optional_number(cfg.get("effort_limit")),
            velocity_limit_sim=_optional_number(cfg.get("velocity_limit")),
            stiffness=_optional_number(cfg.get("stiffness")),
            damping=_optional_number(cfg.get("damping")),
        )

    envcfg.robot_cfg.spawn.usd_path = gripper_info["usd_path"]
    envcfg.robot_cfg.init_state.joint_pos = {
        name: float(value)
        for name, value in gripper_info.get("init_joint_pos", {}).items()
    }
    envcfg.robot_cfg.actuators = actuators
    envcfg.joint_names = controlled_names
    envcfg.gripper_info = gripper_info
    envcfg.contact_sensor_prim_path = gripper_info.get(
        "contact_sensor_prim_path",
        f"{gripper_info['gripper_name']}/.*[Ff]inger.*",
    )
    envcfg.scene.contact_sensor.prim_path = (
        f"{envcfg.robot_prim_path}/{envcfg.contact_sensor_prim_path}"
    )
    configure_scene_object_rigid_props(envcfg.scene)


@configclass
class RobotEnvCfg(_PlatformRobotEnvCfg):
    envs = 200
    dt = SIM_DT
    robot_prim_path = "/World/envs/env_.*/Robot"
    contact_sensor_prim_path = "Robotiq_2f140/.*[Ff]inger.*"
    filter_robot_platform_collision = True

    sim: SimulationCfg = make_grasp_simulation_cfg()
    robot_cfg: ArticulationCfg = EMPTY_FINGER_ENV_CFG.replace(
        prim_path=robot_prim_path
    )
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
    scene.contact_sensor.prim_path = (
        f"{robot_prim_path}/{contact_sensor_prim_path}"
    )
    scene.contact_sensor.debug_vis = False


class RobotEnv(_PlatformRobotEnv):
    cfg: RobotEnvCfg

    def _make_action_policy(self):
        return FingerActionPolicy(
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

    def _apply_z_compliance(
        self, desired_pose: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        del env_ids
        return desired_pose

    def _print_hand_joint_limits_once(self) -> None:
        effort_limits = self.robot.data.joint_effort_limits[0]
        stiffness = self.robot.data.default_joint_stiffness[0]
        damping = self.robot.data.default_joint_damping[0]
        controlled = set(int(index) for index in self._joint_indexes.cpu().tolist())
        print("Finger joint limits from PhysX:")
        for joint_id, name in enumerate(self.robot.joint_names):
            label = "controlled" if joint_id in controlled else "passive/mimic"
            print(
                f"  {name} ({label}): effort={float(effort_limits[joint_id]):.6g}, "
                f"stiffness={float(stiffness[joint_id]):.6g}, "
                f"damping={float(damping[joint_id]):.6g}"
            )

    def factory_reset(self):
        self.act_pol = self._make_action_policy()
        self._reset_idx(None)
        print("#############################    Finger factory reset complete")
