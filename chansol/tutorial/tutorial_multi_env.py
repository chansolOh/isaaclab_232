
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



    robot_cfg = CHANSOL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # chansol_robot = Articulation(cfg=robot_cfg)



def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    

    scene_cfg = ChansolSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)


    origins = [[0.0, 0.0, 0.0]]

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    # while simulation_app.is_running():
    #     # perform step
    #     sim.step()

    
    sim_dt = sim.get_physics_dt()
    count = 0
    repeat = 0
    chansol_robot = scene["robot_cfg"]
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 800 == 0:
            repeat += 1
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = chansol_robot.data.default_root_state.clone()
            # import pdb; pdb.set_trace() 
            root_state[:, :3] += scene.env_origins
            root_state[:, 3:7] += torch.tensor(euler_angles_to_quat([0,0,10*repeat]), device=root_state.device)
            chansol_robot.write_root_pose_to_sim(root_state[:, :7])
            chansol_robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = chansol_robot.data.default_joint_pos.clone(), chansol_robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            chansol_robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            # aircon_remote.reset()
            print("[INFO]: Resetting chansol_robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(chansol_robot.data.joint_pos) * 5.0
        # -- apply action to the chansol_robot

        if count < 250:
            positions = torch.tile(
                torch.tensor([-0.9,  0.0, -0.0], device=root_state.device),
                (chansol_robot.data.joint_pos.shape[0], 1)
            )
        elif count >= 250 and count < 500:
            positions = torch.tile(
                torch.tensor([-0.9,  -0.03, 0.03], device=root_state.device),
                (chansol_robot.data.joint_pos.shape[0], 1)
            )
        elif count >= 500 and count < 750:
            positions = torch.tile(
                torch.tensor([-0.7,  -0.03, 0.03], device=root_state.device),
                (chansol_robot.data.joint_pos.shape[0], 1)
            )

        chansol_robot.set_joint_position_target(positions)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()