


import argparse
from isaaclab.app import AppLauncher
import sys

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")

AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--scene-num", type=int, default=0, help="Scene index to process.")
args_cli = parser.parse_args()


args_cli.headless =False
debug = not args_cli.headless

app_launcher = AppLauncher(args_cli)


simulation_app = app_launcher.app
print("Grasp > App_start")
sys.stdout.flush()

import Robot_EnvCfg as cs_robot
from scene_cfg import scene_obj_setup, object_frame_transformer_setup
import torch
import json
import os
import time

# object_name = "dslr_camera"
object_name = "wireless_charging_stand"
gripper_info_path = "/nas/ochansol/gripper_info/gripper_info.json"    
root_path = f"/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/{object_name}"
pre_grasp_path = "pre_grasp"
output_grasp_path = "output_grasp"
conf_path = "conf"
scene_num = args_cli.scene_num

if not os.path.isdir(os.path.join(root_path, output_grasp_path)):
    os.makedirs(os.path.join(root_path, output_grasp_path))

output_list = []
output_list_tmp = []
if os.path.exists(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json"):
    with open(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json", "r") as f:
        output_list = json.load(f)

with open(os.path.join(root_path, pre_grasp_path, f"{scene_num:04d}.json"), "r") as f:
    pre_grasp_data = json.load(f)
with open(os.path.join(root_path, conf_path, f"{scene_num:04d}.json"), "r") as f:
    conf_data = json.load(f)
with open(gripper_info_path, "r") as f:
    gripper_info = json.load(f)
if isinstance(pre_grasp_data, list):
    pre_grasp_data = pre_grasp_data[0]
gripper_name = pre_grasp_data["gripper_model"]


env_cfg = cs_robot.RobotEnvCfg()

cs_robot.Set_RobotEnvCFG(env_cfg,gripper_info[gripper_name])
scene_obj_setup(env_cfg.scene,conf_data["objects"])
object_list = [i["class"] for i in conf_data["objects"]]
object_frame_transformer_setup(env_cfg.scene, object_list)
env = cs_robot.RobotEnv(cfg=env_cfg, pre_grasp_data=pre_grasp_data["data"], conf_data=conf_data, debug=debug)





####### robot control matrix definition #######
count = 0
obs, _ = env.reset()
print("Grasp > START")
print(f"Grasp > SCENE:{scene_num}")


old_time = time.time()

while simulation_app.is_running():
    with torch.inference_mode():
        operation_tensor = torch.ones((env.cfg.envs))
        obs, rew, term, trunc, info = env.step(operation_tensor)
        if obs["policy"].sum()==0:
            output_list_tmp += env.act_pol.output_list  ###### test [0]
            print("output_list_tmp : ", len(output_list_tmp))
            if len(output_list_tmp)<5:
                env.factory_reset()
                print("Grasp > factory_reset")
                obs["policy"] = torch.zeros(1)
                count += 1
                if count <4:
                    continue

            output_list += output_list_tmp
            sorted_output_list = sorted(output_list, key=lambda x: x["target_object"])
            with open(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json", "w") as f:
                json.dump(sorted_output_list, f, indent=4)
            break


print("Grasp > data_gen_time : ", time.time()-old_time)
sys.stdout.flush()
print(f"Time : {time.time()-old_time}")
env.close()
simulation_app.close()
