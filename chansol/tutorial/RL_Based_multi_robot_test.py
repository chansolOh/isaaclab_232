# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment for a quadruped robot with height-scan sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The height-scan sensor is used to detect the height of
the terrain.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/04_envs/quadruped_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

from isaaclab.sensors import ContactSensorCfg

####### custom import
import scripts.chansol.cfgs.Onrobot_CFG as CRC
from isaaclab.managers import TerminationTermCfg as DoneTerm





class ChansolSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    usd_prim = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object", spawn=sim_utils.UsdFileCfg(
            usd_path="/nas/ochansol/3d_model/peel3_scan_data/aircon_remote/edited/aircon_remote.usd",
            scale = (0.1,0.1,0.1),
        )
    )

    # aircon_remote_cfg = RigidObjectCfg(
    #     prim_path="/World/Object",
    #     spawn=usd_prim.spawn,
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )



    chansol_robot = CRC.CHANSOL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Body/gripper/onrobot_2fg_14/Left", update_period=0.0, history_length=6, debug_vis=True
    )


##
# Custom observation terms
##


# def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
#     """The generated command from the command generator."""
#     return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # pass
    joint_pos = mdp.JointPositionActionCfg(asset_name="chansol_robot", joint_names=[".*"], scale=1, use_default_offset=True)



def get_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[sensor_cfg.name]
    return asset.data.net_forces_w



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass
    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("chansol_robot", joint_names=["right_joint"]), "bounds": (-3.0, 3.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("chansol_robot")}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("chansol_robot")}, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        contact_info = ObsTerm(
            func=get_contact, # get_contact함수의 매개변수 를 params로 넘겨줌
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class ChansolEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: ChansolSceneCfg = ChansolSceneCfg(num_envs=args_cli.num_envs, env_spacing=1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.episode_length_s = 10
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.decimation * self.sim.dt  # 50 Hz


def main():
    """Main function."""
    # setup base environment
    env_cfg = ChansolEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # load level policy
    # policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # # check if policy file exists
    # if not check_file_path(policy_path):
    #     raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    # file_bytes = read_file(policy_path)
    # jit load the policy
    # policy = torch.jit.load(file_bytes).to(env.device).eval()

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            
            if count < 250:
                positions = torch.tile(
                    torch.tensor([-0.9,  0.0, -0.0]),
                    (env.action_manager.action.shape[0], 1)
                )
                print("stage 1")
            elif count >= 250 and count < 500:
                positions = torch.tile(
                    torch.tensor([-0.9,  -0.03, 0.03]),
                    (env.action_manager.action.shape[0], 1)
                )
                print("stage 2")
            elif count >= 500 and count < 750:
                positions = torch.tile(
                    torch.tensor([-0.7,  -0.03, 0.03]),
                    (env.action_manager.action.shape[0], 1)
                )
                print("stage 3")
            # step env
            obs, rew, term, trunc, info = env.step(positions)
            # update counter
            count += 1
            print(count)

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
