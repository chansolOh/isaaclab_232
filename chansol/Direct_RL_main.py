
def main(root_path,
         scene_num,
         pre_grasp_index):

    import argparse
    from isaaclab.app import AppLauncher
    import sys

    parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()


    debug = False
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)


    simulation_app = app_launcher.app
    print("Grasp > App_start")
    sys.stdout.flush()

    import cfgs.Robot_EnvCfg as cs_robot
    from cfgs.scene_cfg import scene_obj_setup,platform_setup, object_frame_transformer_setup
    import torch
    import json
    import os
    import time


    gripper_info_path = "/nas/ochansol/gripper_info/gripper_info.json"    
    pre_grasp_path = "pre_grasp"
    output_grasp_path = "output_grasp"
    conf_path = "conf"

    if not os.path.isdir(os.path.join(root_path, output_grasp_path)):
        os.makedirs(os.path.join(root_path, output_grasp_path))

    output_list = []
    output_list_tmp = []
    if os.path.exists(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json"):
        with open(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json", "r") as f:
            output_list = json.load(f)

    with open(os.path.join(root_path, pre_grasp_path, f"{scene_num:04d}.json"), "r") as f:
        pre_grasp_data = json.load(f)[pre_grasp_index]
    with open(os.path.join(root_path, conf_path, f"{scene_num:04d}.json"), "r") as f:
        conf_data = json.load(f)
    with open(gripper_info_path, "r") as f:
        gripper_info = json.load(f)
    gripper_name = pre_grasp_data["gripper_model"]


    env_cfg = cs_robot.RobotEnvCfg()

    cs_robot.Set_RobotEnvCFG(env_cfg,gripper_info[gripper_name])
    scene_obj_setup(env_cfg.scene,conf_data["objects"])
    platform_setup(env_cfg.scene, conf_data["platform"])
    object_list = [i["class"] for i in conf_data["objects"]]
    object_frame_transformer_setup(env_cfg.scene,object_list)

    env = cs_robot.RobotEnv(cfg=env_cfg, pre_grasp_data=pre_grasp_data["data"], conf_data=conf_data, debug=debug)





    ####### robot control matrix definition #######
    count = 0
    obs, _ = env.reset()
    print("Grasp > START")
    sys.stdout.flush()
    print(f"Grasp > SCENE:{scene_num}")
    sys.stdout.flush()
    print(f"Grasp > PreGrasp_index:{pre_grasp_index}")
    sys.stdout.flush()
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
                # with open(f"{root_path}/{output_grasp_path}/{scene_num:04d}.json", "w") as f:
                #     json.dump(sorted_output_list, f, indent=4)
                break

    
    print("Grasp > data_gen_time : ", time.time()-old_time)
    sys.stdout.flush()
    print(f"Time : {time.time()-old_time}")
    env.close()
    simulation_app.close()




if __name__ == "__main__":
    main(root_path="",scene_num=0, pre_grasp_index=0)
    # simulation_app.close()
