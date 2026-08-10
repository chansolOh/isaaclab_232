
import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
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
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

import isaacsim.core.utils.prims as prim_utils
import torch

from isaacsim.core.utils.rotations import euler_angles_to_quat


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
            velocity_limit=0.2,
            stiffness=2000,
            damping=200,
        ),
    },
)
"""Configuration of Franka Emika Panda robot."""


CHANSOL_ROBOT = CHANSOL_ROBOT_CFG.copy()
CHANSOL_ROBOT.spawn.rigid_props.disable_gravity = True



def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))
    



    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    cfg = sim_utils.UsdFileCfg(
        usd_path="/nas/ochansol/3d_model/peel3_scan_data/aircon_remote/edited/aircon_remote.usd",
        scale = (0.1,0.1,0.1),
        )
    cfg.func("/World/Object/Table", cfg, translation=(0.0, 0.0, 0), orientation=(0.70711, 0.0, 0.0, 0.70711))
    aircon_remote_cfg = RigidObjectCfg(
        prim_path="/World/Object/Table",
        spawn=cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    aircon_remote = RigidObject(cfg=aircon_remote_cfg)


    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])


    robot_cfg = CHANSOL_ROBOT.copy()
    robot_cfg.prim_path = "/World/Origin.*/Robot"
    chansol_robot = Articulation(cfg=robot_cfg)


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
            root_state[:, :3] += torch.tensor(origins,device=root_state.device)
            root_state[:, 3:7] += torch.tensor(euler_angles_to_quat([0,0,10*repeat]), device=root_state.device)
            chansol_robot.write_root_pose_to_sim(root_state[:, :7])
            chansol_robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = chansol_robot.data.default_joint_pos.clone(), chansol_robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            chansol_robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            chansol_robot.reset()
            # aircon_remote.reset()
            print("[INFO]: Resetting chansol_robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(chansol_robot.data.joint_pos) * 5.0
        # -- apply action to the chansol_robot

        if count < 250:
            positions = torch.tensor([[-0.9,  0.0, -0.0],
                                    [-0.5, -0.01,  0.01]], device=root_state.device)
        elif count >= 250 and count < 500:
            positions = torch.tensor([[-0.9,  -0.03, 0.03],
                                    [-0.5, -0.01,  0.01]], device=root_state.device)
        elif count >= 500 and count < 750:
            positions = torch.tensor([[-0.7,  -0.03, 0.03],
                                    [-0.5, -0.01,  0.01]], device=root_state.device)

        chansol_robot.set_joint_position_target(positions)
        # -- write data to sim
        chansol_robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        chansol_robot.update(sim_dt)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()