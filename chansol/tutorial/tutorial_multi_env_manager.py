
import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, AssetBaseCfg

import isaacsim.core.utils.prims as prim_utils
import torch

from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg




CHANSOL_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/nas/ochansol/isaac/sanjabu/Robot/Custom_onrobot/Custom_onrobot.usd",
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
        joint_pos={
            "PrismaticJoint": 0.0,
            "right_joint": 0.0,
            "left_joint": 0.0,
        },
    ),
    actuators={
        "Right_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_joint"],
            effort_limit=50.0,
            # velocity_limit=2.175,
            stiffness=2000.0,
            damping=150.0,
        ),
        "Left_figner": ImplicitActuatorCfg(
            joint_names_expr=["left_joint"],
            effort_limit=50.0,
            # velocity_limit=2.61,
            stiffness=2000.0,
            damping=150.0,
        ),
        "Lift": ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint"],
            effort_limit=200.0,
            # velocity_limit=0.2,
            stiffness=2000,
            damping=200,
        ),
    },
)
"""Configuration of Franka Emika Panda robot."""


# CHANSOL_ROBOT = CHANSOL_ROBOT_CFG.copy()
# CHANSOL_ROBOT.spawn.rigid_props.disable_gravity = True





@configclass
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



    chansol_robot = CHANSOL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # chansol_robot = Articulation(cfg=robot_cfg)



@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointPositionActionCfg(asset_name="chansol_robot", joint_names=["PrismaticJoint", "right_joint", "left_joint"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,  params={"asset_cfg": SceneEntityCfg("chansol_robot")})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,  params={"asset_cfg": SceneEntityCfg("chansol_robot")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on reset
    reset_rigth_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("chansol_robot", joint_names=["right_joint"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1, 1),
        },
    )

    reset_left_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("chansol_robot", joint_names=["left_joint"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1, 1),
        },
    )

    reset_lift_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("chansol_robot", joint_names=["PrismaticJoint"]),
            "position_range": (-2, 1),
            "velocity_range": (-1, 1),
        },
    )


@configclass
class ChansolEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = ChansolSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz







def main():
    """Main function."""

    env_cfg = ChansolEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # setup base environment
    # import pdb; pdb.set_trace()
    env = ManagerBasedEnv(cfg=env_cfg)

    print("[INFO]: Setup complete...")

    count = 0

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset
            if count % 800 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")


            if count < 250:
                positions = torch.tile(
                    torch.tensor([-0.9,  0.0, -0.0]),
                    (env.action_manager.action.shape[0], 1)
                )
            elif count >= 250 and count < 500:
                positions = torch.tile(
                    torch.tensor([-0.9,  -0.03, 0.03]),
                    (env.action_manager.action.shape[0], 1)
                )
            elif count >= 500 and count < 750:
                positions = torch.tile(
                    torch.tensor([-0.7,  -0.03, 0.03]),
                    (env.action_manager.action.shape[0], 1)
                )

            obs, _ = env.step(positions)

            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()