import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg
from collections.abc import Sequence

from isaaclab.actuators import ImplicitActuatorCfg

import isaacsim.core.utils.stage as stage_utils

import sys
import getpass
sys.path.append(f"/home/{getpass.getuser()}/ochansol/IsaacLab/scripts/chansol/cfgs")
import scene_cfg as SC
from action_policy_revised import ActionPolicy
from direct_rl_env_custom import DirectRLEnv_custom

EMPTY_ROBOT_ENV_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=0
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={},
        ),
        actuators={}
    )



def Set_RobotEnvCFG(envcfg, json):
    joint_dict = {}
    for name, cfg in json["joint_cfg"].items():
        joint_dict[name] = ImplicitActuatorCfg(
                                joint_names_expr=name,
                                effort_limit=cfg["effort_limit"] if cfg["effort_limit"] is not None else None,
                                # velocity_limit=cfg["velocity_limit"] if cfg["velocity_limit"] is not "None" else None,
                                stiffness=cfg["stiffness"] if cfg["stiffness"] is not None else None,
                                damping=cfg["damping"] if cfg["damping"] is not None else None,
                        )
                
    envcfg.robot_cfg.spawn.usd_path = json["usd_path"]
    envcfg.robot_cfg.init_state.joint_pos = json["init_joint_pos"]
    envcfg.robot_cfg.actuators = joint_dict

    envcfg.joint_names = list(json["joint_cfg"].keys())
    envcfg.gripper_info = json



@configclass
class RobotEnvCfg(DirectRLEnvCfg):


    # env
    decimation = 4
    episode_length_s = 100.0
    # action_scale = 1.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0
    envs = 30
    dt = 1 / 400

    robot_prim_path = "/World/envs/env_.*/Robot"

    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation,
                                    #    gravity=(0,0,0),
                                       physx=PhysxCfg())
    
    robot_cfg: ArticulationCfg = EMPTY_ROBOT_ENV_CFG.replace(prim_path=robot_prim_path)

    scene: SC.SceneCfg = SC.SceneCfg(num_envs=envs, env_spacing=0.5, replicate_physics=True)






class RobotEnv(DirectRLEnv_custom):
    cfg: RobotEnvCfg

    def __init__(self, cfg: RobotEnvCfg, render_mode: str | None = None, pre_grasp_data = [],
                 conf_data = {},
                 debug = False,
                  **kwargs):
        rigid_patch_count                       = 2**18  * 2**4
        gpu_temp_buffer_capacity                = 2**24  * 2**2
        gpu_max_rigid_contact_count             = 2**23  * 2**2
        gpu_heap_capacity                       = 2**26  * 2**2
        gpu_found_lost_pairs_capacity           = 2**21  * 2**2
        gpu_found_lost_aggregate_pairs_capacity = 2**25  * 2**2
        gpu_total_aggregate_pairs_capacity      = 2**21  * 2**2
        super().__init__(cfg, render_mode, 
                         rigid_patch_count=rigid_patch_count,
                         gpu_temp_buffer_capacity=gpu_temp_buffer_capacity,
                         gpu_max_rigid_contact_count=gpu_max_rigid_contact_count,
                         gpu_heap_capacity=gpu_heap_capacity,
                         gpu_found_lost_pairs_capacity=gpu_found_lost_pairs_capacity,
                         gpu_found_lost_aggregate_pairs_capacity=gpu_found_lost_aggregate_pairs_capacity,
                         gpu_total_aggregate_pairs_capacity=gpu_total_aggregate_pairs_capacity,
                         **kwargs)

        self._joint_indexes = torch.tensor([self.robot.find_joints(i)[0][0] for i in self.cfg.joint_names] )
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.pre_grasp_data = pre_grasp_data
        self.conf_data = conf_data
        self.debug = debug
        self.act_pol = ActionPolicy(env_num= self.num_envs, 
                                    pre_grasp_data = self.pre_grasp_data, 
                                    conf_data = self.conf_data,
                                    gripper_info = self.cfg.gripper_info,
                                    joint_index = self._joint_indexes,
                                    joint_pos = self.joint_pos,
                                    step_dt = self.cfg.dt,
                                    device = self.device,
                                    frame_transformer=self.transformer,
                                    env_origin = self.scene.env_origins,
                                    debug = self.debug,
                                    )
        self.grasp_root_pose_targets = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.object_gravity_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stage0_root_lift_start = 0.08
        self.stage0_root_lift = torch.full((self.num_envs,), self.stage0_root_lift_start, dtype=torch.float, device=self.device)
        self.stage0_root_lift_step = 0.001
        self.object_gravity = torch.tensor((0.0, 0.0, -9.81), dtype=torch.float, device=self.device)

        self.scene_var_dict = {}
        for i in range(10):
            if hasattr(self.scene.cfg, f"obj{i:02d}") and getattr(self.scene.cfg, f"obj{i:02d}") is not None:
                item = getattr(self.scene.cfg, f"obj{i:02d}")
                self.scene_var_dict[item.class_name] = item





    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        for i in range(1):
            if getattr(self.scene.cfg, f"obj{i:02d}") is not None:
                obj = self.scene[f"obj{i:02d}"]
                obj.class_name = getattr(self.scene.cfg, f"obj{i:02d}").class_name
                setattr(self, f"obj{i:02d}", obj)
                
        self.transformer = self.scene["transformer"]
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot




    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.act_pol.step()
        # not_action_idx = self.act_pol.action_enable!=1
        # self.actions[not_action_idx]*=0


    def _apply_action(self) -> None:

        # self.actions = self.act_pol.step(self.joint_pos)
        action_env_idx = torch.where(self.act_pol.stage_num != 4)[0]
        # self.actions[not_action_idx]*=0
       
        if len(action_env_idx) > 0:
            self.robot.set_joint_position_target(self.actions[action_env_idx], joint_ids=self._joint_indexes, env_ids=action_env_idx)

        stage0_idx = torch.where(self.act_pol.stage_num == 0)[0]
        if len(stage0_idx) > 0:
            self.stage0_root_lift[stage0_idx] = torch.clamp(
                self.stage0_root_lift[stage0_idx] - self.stage0_root_lift_step,
                min=0.0,
            )
            stage0_root_pose = self.grasp_root_pose_targets[stage0_idx].clone()
            stage0_root_pose[:, 2] += self.stage0_root_lift[stage0_idx]
            zero_root_vel = torch.zeros((len(stage0_idx), 6), dtype=torch.float, device=self.device)
            self.robot.write_root_pose_to_sim(stage0_root_pose, stage0_idx)
            self.robot.write_root_velocity_to_sim(zero_root_vel, stage0_idx)

            stage0_done_idx = stage0_idx[self.stage0_root_lift[stage0_idx] <= 0.0]
            if len(stage0_done_idx) > 0:
                self.act_pol.stage_num[stage0_done_idx] = 1
                self.act_pol.stage_time_out[stage0_done_idx] = 0


        stage2_idx = torch.where((self.act_pol.stage_num == 2) & (self.act_pol.stage_time_out > 0))[0]
        stage3_idx = torch.where(self.act_pol.stage_num == 3)[0]
        force_env_idx = torch.cat((stage2_idx, stage3_idx))
        base_force_magnitude = torch.linalg.norm(self.object_gravity)
        force_magnitude = torch.ones((len(force_env_idx),), dtype=torch.float, device=self.device) * base_force_magnitude
        if len(stage3_idx) > 0:
            stress_progress = torch.clamp(
                self.act_pol.stage_time_out[stage3_idx] / self.act_pol.stress_wait_th,
                min=0.0,
                max=1.0,
            )
            force_magnitude[len(stage2_idx):] = base_force_magnitude + (60.0 - base_force_magnitude) * stress_progress
        self._disable_builtin_object_gravity()
        self._apply_object_gravity_force(force_env_idx, force_magnitude)


    def _get_observations(self) -> dict:

        observations = {"policy": self.act_pol.action_enable}
        return observations


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = self.episode_length_buf >= 1000 - 1
        object_pos = self._get_target_object_positions()
        object_quat = self._get_target_object_quaternions()
        stage_done_idx = self.act_pol.get_done_idx(
            object_pos=object_pos,
            object_quat=object_quat,
            joint_vel=self.robot.data.joint_vel,
        )

        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._right_joint_idx]) > math.pi / 2, dim=1)
        return stage_done_idx, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        self.act_pol.reset(env_ids)
        self.object_gravity_enabled[env_ids] = False
        self.stage0_root_lift[env_ids] = self.stage0_root_lift_start
        self.act_pol.stage_time_out[env_ids] = 0

        robot_root_pos = self.robot.data.default_root_state[env_ids].clone()
        robot_root_pos[:, :3] += self.scene.env_origins[env_ids]
        robot_root_pos[:, :3] += self.act_pol.target_pos[env_ids]
        robot_root_pos[:, 3:7] = self.act_pol.target_rot[env_ids]
        self.grasp_root_pose_targets[env_ids] = robot_root_pos[:, :7]
        robot_root_pos[:, 2] += self.stage0_root_lift[env_ids]

        # print("Reset complete", env_ids)
        # print("target_width : ",self.act_pol.target_width[env_ids])
        # print("joint : ",self.act_pol.stage_action_arr[env_ids,0,1]/torch.pi*180, self.act_pol.stage_action_arr[env_ids,0,2]/torch.pi*180)


        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]


        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.robot.write_root_pose_to_sim(robot_root_pos[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root_pos[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # joints_idx = torch.tensor([0], device=self.device)
        # self.robot.write_joint_velocity_limit_to_sim(torch.ones((self.num_envs,1), dtype=torch.float, device=self.device)*0.1, joints_idx, None)

        for i in range(10):
            if hasattr(self,f"obj{i:02d}"):
                item = getattr(self,f"obj{i:02d}")
                default = item.data.default_root_state[env_ids].clone()
                default[:, :3] += self.scene.env_origins[env_ids]
                item.write_root_pose_to_sim(default[:, :7], env_ids)
                # item.write_root_velocity_to_sim(default[:, 7:], env_ids)
        self._set_object_gravity(env_ids, enabled=False, zero_vel=True)
        self.scene.write_data_to_sim()
        self.sim.forward()




    def factory_reset(self):
        self.act_pol = ActionPolicy(env_num= self.num_envs, 
                            pre_grasp_data = self.pre_grasp_data, 
                            conf_data = self.conf_data,
                            gripper_info = self.cfg.gripper_info,
                            joint_index = self._joint_indexes,
                            joint_pos = self.joint_pos,
                            step_dt = self.cfg.dt,
                            device = self.device,
                            frame_transformer=self.transformer,
                            env_origin = self.scene.env_origins,
                            debug = self.debug,
                            )
        self._reset_idx(None)
        print("#############################    Factory reset complete")


    def _get_target_object_positions(self):
        object_pos_list = []
        for i in range(1):
            item = getattr(self, f"obj{i:02d}")
            object_pos_list.append(item.data.root_link_pose_w[:, :3])
        return torch.stack(object_pos_list, dim=1)[torch.arange(self.num_envs), self.act_pol.target_class]

    def _get_target_object_quaternions(self):
        object_quat_list = []
        for i in range(1):
            item = getattr(self, f"obj{i:02d}")
            object_quat_list.append(item.data.root_link_pose_w[:, 3:7])
        return torch.stack(object_quat_list, dim=1)[torch.arange(self.num_envs), self.act_pol.target_class]


    def _set_object_gravity(self, env_ids, enabled: bool, zero_vel=False):
        if env_ids is None:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        for i in range(1):
            item = getattr(self, f"obj{i:02d}", None)
            if item is None:
                continue

            self._disable_builtin_object_gravity(item)

            if zero_vel:
                zero_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float, device=self.device)
                item.write_root_velocity_to_sim(zero_velocity, env_ids)

    def _disable_builtin_object_gravity(self, item=None):
        if item is None:
            item = self.obj00
        view = item.root_physx_view
        all_indices = torch.arange(view.count, dtype=torch.int32, device="cpu")
        disable_gravity_flags = torch.ones((view.count, 1), dtype=torch.uint8, device="cpu")
        view.set_disable_gravities(disable_gravity_flags, all_indices)

    def _apply_object_gravity_force(self, env_ids, force_magnitude=None):
        item = self.obj00
        view = item.root_physx_view
        forces = torch.zeros((view.count, 3), dtype=torch.float, device=self.device)
        if len(env_ids) > 0:
            masses = item.data.default_mass.to(device=self.device, dtype=torch.float)
            masses = masses[env_ids].reshape(-1, 1)
            if force_magnitude is None:
                force_magnitude = torch.ones((len(env_ids),), dtype=torch.float, device=self.device) * torch.linalg.norm(self.object_gravity)
            force_direction = self.act_pol.force_direction[env_ids]
            forces[env_ids] = masses * force_direction * force_magnitude.reshape(-1, 1)

        all_indices = torch.arange(view.count, dtype=torch.int32, device=self.device)
        if len(env_ids) > 0:
            view.wake_up(env_ids.to(dtype=torch.int32, device="cpu"))
        view.apply_forces_and_torques_at_position(
            force_data=forces,
            torque_data=None,
            position_data=None,
            indices=all_indices,
            is_global=True,
        )


    def _get_rewards(self):
        return None
    
    

    
