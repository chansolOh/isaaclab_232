import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg , RigidObject, RigidObjectCfg
from isaaclab.sensors import FrameTransformer, ContactSensor
from collections.abc import Sequence
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from isaaclab.actuators import ImplicitActuatorCfg


import sys
sys.path.append("/home/cubox/ochansol/IsaacLab/scripts/chansol/cfgs")
import scene_cfg as SC
import Robot_CFG as CRC
from action_policy_revised import ActionPolicy
from direct_rl_env_custom import DirectRLEnv_custom

import carb
import json


EMPTY_ROBOT_ENV_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
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
    episode_length_s = 10.0
    # action_scale = 1.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0
    envs = 1
    dt = 1 / 500

    robot_prim_path = "/World/envs/env_.*/Robot"
    contact_sensor_prim_path = "Body/gripper/Finger_.*"



    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation,
                                       physx=PhysxCfg())
    
    robot_cfg: ArticulationCfg = EMPTY_ROBOT_ENV_CFG.replace(prim_path=robot_prim_path)

    scene: SC.SceneCfg = SC.SceneCfg(num_envs=envs, env_spacing=1, replicate_physics=True)
    scene.contact_sensor.prim_path = f"{robot_prim_path}/{contact_sensor_prim_path}"






class RobotEnv(DirectRLEnv_custom):
    cfg: RobotEnvCfg

    def __init__(self, cfg: RobotEnvCfg, render_mode: str | None = None, pre_grasp_data = [],
                 conf_data = {},
                  **kwargs):
        rigid_patch_count                       = 2**18  * 2**4
        gpu_temp_buffer_capacity                = 2**24  
        gpu_max_rigid_contact_count             = 2**23  
        gpu_heap_capacity                       = 2**26  
        gpu_found_lost_pairs_capacity           = 2**21  * 2**2
        gpu_found_lost_aggregate_pairs_capacity = 2**25  
        gpu_total_aggregate_pairs_capacity      = 2**21  
        super().__init__(cfg, render_mode, 
                         rigid_patch_count=rigid_patch_count,
                         gpu_temp_buffer_capacity=gpu_temp_buffer_capacity,
                         gpu_max_rigid_contact_count=gpu_max_rigid_contact_count,
                         gpu_heap_capacity=gpu_heap_capacity,
                         gpu_found_lost_pairs_capacity=gpu_found_lost_pairs_capacity,
                         gpu_found_lost_aggregate_pairs_capacity=gpu_found_lost_aggregate_pairs_capacity,
                         gpu_total_aggregate_pairs_capacity=gpu_total_aggregate_pairs_capacity,
                         **kwargs)


        self._joint_indexes = torch.tensor([self.robot.find_joints(i)[0][0] for i in self.cfg.joint_names])
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.grasp_num = 0

        self.act_pol = ActionPolicy(env_num= self.num_envs, 
                                    pre_grasp_data = pre_grasp_data, 
                                    conf_data = conf_data,
                                    gripper_info = self.cfg.gripper_info,
                                    joint_index = self._joint_indexes,
                                    joint_pos = self.joint_pos,
                                    step_dt = self.step_dt,
                                    device = self.device, 
                                    contact_sensor = self.contact_sensor,
                                    frame_transformer = self.transformer,
                                    env_origin = self.scene.env_origins,
                                    debug = True,
                                    stage_time_out = 0.6)

        self.scene_var_dict = {}
        for i in range(10):
            if hasattr(self.scene.cfg, f"obj{i:02d}") and getattr(self.scene.cfg, f"obj{i:02d}") is not None:
                item = getattr(self.scene.cfg, f"obj{i:02d}")
                self.scene_var_dict[item.class_name] = item






    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        for i in range(10):
            if getattr(self.scene.cfg, f"obj{i:02d}") is not None:
                obj = getattr(self.scene.cfg, f"obj{i:02d}")
                setattr(self,f"obj{i:02d}", 
                        RigidObject(cfg = RigidObjectCfg(
                                            prim_path=f"/World/envs/env_.*/obj{i:02d}",
                                            spawn=obj.spawn,
                                            init_state=RigidObjectCfg.InitialStateCfg(pos=obj.pos, rot=obj.quat),
                )))
                getattr(self,f"obj{i:02d}").class_name = obj.class_name
        self.transformer = self.scene["transformer"]
        self.contact_sensor = self.scene["contact_sensor"]


        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot




    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.act_pol.step()
        
        # not_action_idx = self.act_pol.action_enable!=1
        # self.actions[not_action_idx]*=0


    def _apply_action(self) -> None:
        # self.actions = self.act_pol.step(self.joint_pos)
        action_env_idx = torch.where(self.act_pol.stage_num!=3)[0]
        # self.actions[not_action_idx]*=0
        self.robot.set_joint_position_target(self.actions[action_env_idx], joint_ids=self._joint_indexes, env_ids=action_env_idx)

    def _get_observations(self) -> dict:


        # observations = {"policy": self.scene["transformer"].data.target_pos_source.squeeze(dim=1)}
        observations = {"policy": self.act_pol.action_enable,
                        "grasp_num": self.act_pol.current_grasp_num,}
        return observations


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = self.episode_length_buf >= 850 - 1
        stage_done_idx = self.act_pol.get_done_idx()


        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._right_joint_idx]) > math.pi / 2, dim=1)

        return stage_done_idx, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self.act_pol.current_grasp_num = self.grasp_num
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.act_pol.reset(env_ids)

        # for i in env_ids:
        #     SC.frame_transformer_set(self.scene.cfg,self.scene_var_dict[self.act_pol.target_class[i]])

        robot_root_pos = self.robot.data.default_root_state[env_ids]
        robot_root_pos[:, :3] += self.scene.env_origins[env_ids]
        robot_root_pos[:, :3] += self.act_pol.target_pos[env_ids]

        robot_root_pos[:, 3:7] = self.act_pol.target_rot[env_ids]

        print("Reset complete", env_ids)
        print(quat_to_euler_angles(self.act_pol.target_rot[env_ids].cpu().numpy()[0], degrees=True)) 


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
                default = item.data.default_root_state[env_ids]
                default[:, :3] += self.scene.env_origins[env_ids]
                item.write_root_pose_to_sim(default[:, :7], env_ids)
                item.write_root_velocity_to_sim(default[:, 7:], env_ids)


        self.contact_sensor_body_names= self.contact_sensor.body_names.copy()
        self.transformer_body_names = self.transformer._target_frame_body_names.copy()
        
    def _get_rewards(self):
        return None
    

    

