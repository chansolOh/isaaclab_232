


from isaaclab.app import AppLauncher
import sys
from pathlib import Path

# 새 수집기와 동일한 <dataset>/<object>/<gripper>/<data-type> 구조를 사용한다.
DATASET_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
OBJECT_NAME = "black_pepper_shaker"
GRIPPER_NAME = "Robotiq_2f140"
SCENE_NUM = 0

# 기존 물리/그리퍼 설정은 유지한다. 비교 결과는 새 수집 결과와 섞이지 않게
# output_grasp_legacy에 저장한다.
GRIPPER_INFO = Path("/nas/ochansol/gripper_info/gripper_info.json")
PRE_GRASP_DIR = "pre_grasp"
CONF_DIR = "conf"
OUTPUT_GRASP_DIR = "output_grasp_legacy"
OVERWRITE = True

HEADLESS = False
DEVICE = "cuda:0"

debug = not HEADLESS
app_launcher = AppLauncher(
    headless=HEADLESS,
    device=DEVICE,
    enable_cameras=False,
)


simulation_app = app_launcher.app
print("Grasp > App_start")
sys.stdout.flush()

import Robot_EnvCfg as cs_robot
from scene_cfg import scene_obj_setup, object_frame_transformer_setup
import torch
import json
import time

root_path = DATASET_ROOT / OBJECT_NAME / GRIPPER_NAME
pre_grasp_path = root_path / PRE_GRASP_DIR / f"{SCENE_NUM:04d}.json"
conf_path = root_path / CONF_DIR / f"{SCENE_NUM:04d}.json"
output_path = root_path / OUTPUT_GRASP_DIR / f"{SCENE_NUM:04d}.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

for required in (pre_grasp_path, conf_path, GRIPPER_INFO):
    if not required.is_file():
        raise FileNotFoundError(required)

output_list = []
output_list_tmp = []
if output_path.exists() and not OVERWRITE:
    with output_path.open("r", encoding="utf-8") as f:
        output_list = json.load(f)

with pre_grasp_path.open("r", encoding="utf-8") as f:
    pre_grasp_data = json.load(f)
with conf_path.open("r", encoding="utf-8") as f:
    conf_data = json.load(f)
with GRIPPER_INFO.open("r", encoding="utf-8") as f:
    gripper_info = json.load(f)
if isinstance(pre_grasp_data, list):
    pre_grasp_data = pre_grasp_data[0]
gripper_name = pre_grasp_data["gripper_model"]
if gripper_name != GRIPPER_NAME:
    raise ValueError(
        f"Folder gripper {GRIPPER_NAME!r} does not match pregrasp {gripper_name!r}"
    )


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
print(f"Grasp > SCENE:{SCENE_NUM}")


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
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(sorted_output_list, f, indent=4)
            break


print("Grasp > data_gen_time : ", time.time()-old_time)
sys.stdout.flush()
print(f"Time : {time.time()-old_time}")
env.close()
simulation_app.close()
