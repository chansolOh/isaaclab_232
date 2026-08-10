import argparse
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--scene_num", type=int, default=0, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, UsdPhysics
import cfgs.Robot_EnvCfg as cs_robot
from cfgs.scene_cfg import scene_obj_setup, object_frame_transformer_setup
import torch
import time
import json
import os
root_path = "/nas/Dataset/Dataset_2024/scene_rev20241209/Home"
pre_grasp_path = "pre_grasp"
output_grasp_path = "output_grasp"
conf_path = "conf"

if not os.path.isdir(os.path.join(root_path, output_grasp_path)):
    os.makedirs(os.path.join(root_path, output_grasp_path))

scene_num= args_cli.scene_num

# target_object = ["headphone","broom","mouse","cream","pouch"]




#1500 ro 3:30


# stage = simulation_app.context.get_stage()
# stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuMaxRigidPatchCount").Set(563840)




def main(scene_num):

    with open(os.path.join(root_path, pre_grasp_path, f"{scene_num:04d}.json"), "r") as f:
        pre_grasp_data = json.load(f)
    with open(os.path.join(root_path, conf_path, f"{scene_num:04d}.json"), "r") as f:
        conf_data = json.load(f)


    env_cfg = cs_robot.RobotEnvCfg()

    scene_obj_setup(env_cfg.scene,conf_data["objects"])
    object_list = [i["class"] for i in conf_data["objects"]]
    object_frame_transformer_setup(env_cfg.scene,object_list)

    env = cs_robot.RobotEnv(cfg=env_cfg, pre_grasp_data=pre_grasp_data, conf_data=conf_data)

    # stage = simulation_app.context.get_stage()
    # stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuMaxRigidPatchCount").Set(1563840)
    # stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuTempBufferCapacity").Set(26777216)

    
    ####### robot control matrix definition #######
    count = 0
    obs, _ = env.reset()
    old_time = time.time()

    while simulation_app.is_running():
        with torch.inference_mode():
            operation_tensor = torch.ones((env.cfg.envs))
            obs, rew, term, trunc, info = env.step(operation_tensor)
            if obs["policy"].sum()==0:
                with open(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json", "w") as f:
                    json.dump(env.act_pol.output_list, f, indent=4)
                break


    # close the environment
    print(f"Time : {time.time()-old_time}")
    # import pdb; pdb.set_trace()
    env.close()




if __name__ == "__main__":
    # run the main function
    main(scene_num)
    # close sim app
    simulation_app.close()
