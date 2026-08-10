import argparse
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import PhysxSchema, UsdPhysics


import sys
sys.path.append("/home/cubox/ochansol/IsaacLab/scripts/chansol/cfgs")
import Robot_EnvCfg_debug as cs_robot
from scene_cfg import scene_obj_setup, object_frame_transformer_setup, platform_setup
import torch
import time
import json
import os
from Grasp_Bounding_Box import bbox_display





output_grasp_path = "output_grasp"
conf_path = "conf"

root_path = "/nas/Dataset/Dataset_2025"
img_sampling_num = 1000
all_data_num = 1000


###### data path ######
dir_path = "sim2real/demo/default"
rgb_path = os.path.join(root_path, dir_path, "rgb/top_view_camera")
bbox_path  = os.path.join(root_path, dir_path, "output_grasp")
inst_seg_path = os.path.join(root_path, dir_path, "inst_seg/top_view_camera")

scene_num = 0

graphics = bbox_display(rgb_path, bbox_path, inst_seg_path, img_sampling_num, all_data_num, scene_num=scene_num)


def main():

    
    with open(os.path.join(root_path, dir_path, output_grasp_path, f"{scene_num:04d}.json"), "r") as f:
        output_grasp_data = json.load(f)
    with open(os.path.join(root_path, dir_path, conf_path, f"{scene_num:04d}.json"), "r") as f:
        conf_data = json.load(f)
    with open("/nas/ochansol/gripper_info/gripper_info.json", "r") as f:
        gripper_info = json.load(f)


    gripper_name = output_grasp_data[0]["gripper_model"]


    env_cfg = cs_robot.RobotEnvCfg()

    cs_robot.Set_RobotEnvCFG(env_cfg,gripper_info[gripper_name])
    scene_obj_setup(env_cfg.scene,conf_data["objects"])
    platform_setup(env_cfg.scene, conf_data["platform"])
    object_list = [i["class"] for i in conf_data["objects"]]
    object_frame_transformer_setup(env_cfg.scene,object_list)

    env = cs_robot.RobotEnv(cfg=env_cfg, pre_grasp_data=output_grasp_data, conf_data=conf_data)


    ####### robot control matrix definition #######
    count = 0
    obs, _ = env.reset()
    old_time = time.time()

    while simulation_app.is_running():
        with torch.inference_mode():
            graphics.update()
            env.grasp_num = max(graphics.grasp_num,0)

            operation_tensor = torch.ones((env.cfg.envs))
            obs, rew, term, trunc, info = env.step(operation_tensor)
            if scene_num != graphics.scene_num:
                break
            

    # close the environment
    print(f"Time : {time.time()-old_time}")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
