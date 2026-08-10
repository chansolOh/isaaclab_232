
def main(scene_num, env, section, platform, root_path):
    import os
    import json
    import numpy as np

    from PIL import Image
    import ast
    import matplotlib.pyplot as plt
    import time
    import sys
    import getpass
    sys.path.append(f"/home/{getpass.getuser()}/ochansol/isaac_code/python/utils")
    import cs_utils as cs

    print("PreGrasp > App_start")
    sys.stdout.flush()
    print("PreGrasp > START")
    sys.stdout.flush()
    print(f"PreGrasp > SCENE:{scene_num}")
    sys.stdout.flush()
    data_gen_time = time.time()
    # from collections import defaultdict

    # def find_dup(arr, tolerance=1e-10):
    #     """
    #     방법 2: 딕셔너리 사용 (더 많은 정보 제공)
        
    #     Returns:
    #     duplicate_info: 중복 벡터와 그 위치들의 상세 정보
    #     """
    #     if tolerance > 0:
    #         # 부동소수점 오차 처리
    #         rounded_arr = np.round(arr / tolerance) * tolerance
    #     else:
    #         rounded_arr = arr
        
    #     vector_positions = defaultdict(list)
        
    #     # 각 벡터의 위치 기록
    #     for i, vector in enumerate(rounded_arr):
    #         # 벡터를 tuple로 변환하여 딕셔너리 키로 사용
    #         key = tuple(vector)
    #         vector_positions[key].append(i)
        
    #     # 중복된 벡터들만 필터링
    #     duplicates = {k: v for k, v in vector_positions.items() if len(v) > 1}
        
    #     return duplicates


    root_path = f"/nas/Dataset/Dataset_2025/dataset_v1/{env}/{section}/{platform}"
    gripper_info_path = "/nas/ochansol/gripper_info/gripper_info.json" 

    with open(gripper_info_path, 'r') as f:
            gripper_info = json.load(f)


    print("@@@@@@@@@@@@@@@@@@@@@@@@ scene_num : ",scene_num)
    grasp_point_ann_path = os.path.join(root_path,"pre_grasp")
    if not os.path.exists(grasp_point_ann_path):
        os.makedirs(grasp_point_ann_path)
    with open( os.path.join(root_path,"conf",f"{scene_num:04d}"+".json"), 'r') as f:
        config= json.load(f)
    obj_conf = config["objects"]
    cam_conf = config["cameras"]
    physx_conf = config["physics_scene"]
    top_cam_config = [ i for i in cam_conf if i["name"] == "top_view_camera"][0]
    side_cam_config = [ i for i in cam_conf if i["name"] == "side_view_camera"][0]


    depth_img_cam00 = np.load(os.path.join(root_path,"depth","top_view_camera",f"{scene_num:04d}"+".npy"))
    rgb_img_cam00 = Image.open(os.path.join(root_path,"rgb","top_view_camera",f"{scene_num:04d}"+".png"))
    inst_img_cam00 = np.array(Image.open(os.path.join(root_path,"inst_seg","top_view_camera",f"{scene_num:04d}"+".png")))
    with open( os.path.join(root_path,"inst_seg","top_view_camera","semantics_mapping_"+f"{scene_num:04d}"+".json"), 'r') as f:
        inst_label_cam00= json.load(f)

    output_json_list = []
    gripper_type_list =["finger2", "finger3"]
    for gripper_type in gripper_type_list:

        while True:
            random_gripper_name = np.random.choice(list(gripper_info.keys()))
            if gripper_type in gripper_info[random_gripper_name]["type"]:
                break


        # random_gripper_name = "Custom_schunk_parallel"
        # random_gripper_name = "Custom_GEP2016IO"
        # random_gripper_name = "Custom_onrobot"
        # random_gripper_name = "Hitbot_z_efg_100"
        # random_gripper_name = "OnRobot_RG2FTv2"
        # random_gripper_name = "OnRobot_RG6_v1_2"
        # random_gripper_name = "UON_Robotics_Jamin_Gripper"
        # random_gripper_name = "DH_Robotics_DH3"
        # random_gripper_name = "Robotiq_3Finger_Adaptive_Gripper"
        gripper = gripper_info[random_gripper_name]
        # if gripper["type"] not in ["finger3", "finger3_par+allel"]:continue
        output_json = {
            "gripper_model": random_gripper_name,
            "data": [],
        }


        class temp_rep:
            def __init__(self,class_name):
                self.class_name = class_name
        obj_rep_list = []
        for obj in obj_conf:
            obj_rep_list.append(temp_rep(obj["class"]))
            
        cx,cy = np.array(top_cam_config["intrinsic_isaac"])[0,2], np.array(top_cam_config["intrinsic_isaac"])[1,2]
        focal_length = np.array(top_cam_config["intrinsic_isaac"])[0,0]

        cam_tf = np.array(top_cam_config["cam_poses"])

        gripper_height = gripper["height"]
        gripper_depth = gripper["depth"]
        finger_thickness = gripper["finger_thickness"]
        gripper_width_list = np.round(np.array([1.0,0.8,0.6,0.4])*gripper["width"], 2).tolist()



        ###### point cloud sampling
        IDX = np.argwhere(depth_img_cam00==depth_img_cam00).T
        Z_all = depth_img_cam00[IDX[0],IDX[1]]
        sample_idx = np.random.choice(len(Z_all),int(len(Z_all)*0.08),replace=False)
        SAMPLED_IDX = IDX[:,sample_idx]
        Z_sample = Z_all[sample_idx]
        X_sample = (SAMPLED_IDX[1]-cx)/focal_length*Z_sample
        Y_sample = (SAMPLED_IDX[0]-cy)/focal_length*Z_sample
        # PCD = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
        cam_to_pt_all = np.array([X_sample,Y_sample,Z_sample,np.ones_like(X_sample)])
        PCD = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_pt_all)[:3]

        ###### grasp point sampling
        labeled_depth_dict = {}
        for obj in obj_rep_list:
            idx_part = None
            for color_key in inst_label_cam00.keys():
                if obj.class_name == inst_label_cam00[color_key]["class"]:
                    color_arr = np.array(ast.literal_eval(color_key))

                    idx_part = np.argwhere( (inst_img_cam00[SAMPLED_IDX[0], SAMPLED_IDX[1], 0] == color_arr[0]) &\
                                            (inst_img_cam00[SAMPLED_IDX[0], SAMPLED_IDX[1], 1] == color_arr[1]) &\
                                            (inst_img_cam00[SAMPLED_IDX[0], SAMPLED_IDX[1], 2] == color_arr[2]) &\
                                            (inst_img_cam00[SAMPLED_IDX[0], SAMPLED_IDX[1], 3] == color_arr[3])).T
            if idx_part is None:
                continue
            labeled_depth_dict[obj.class_name] = idx_part
            
        # if os.path.exists(os.path.join(grasp_point_ann_path,f"{scene_num:04d}"+".json")):
        #     with open(os.path.join(grasp_point_ann_path,f"{scene_num:04d}"+".json"), 'r') as f:
        #         grasp_output_dict = json.load(f)
        #         if grasp_output_dict.keys() == []:
        #             for obj in obj_rep_list:
        #                 grasp_output_dict[obj.class_name] = []
        # else:
        #     with open(os.path.join(grasp_point_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
        #         grasp_output_dict = {}
        #         for obj in obj_rep_list:
        #             grasp_output_dict[obj.class_name] = []
        #         json.dump(grasp_output_dict,f, indent = 4)

        all_obj_idx = np.hstack([labeled_depth_dict[key] for key in labeled_depth_dict.keys()])[0]   ##### x,y -90+


        old_time = time.time()

        for obj in obj_rep_list:
            print("####################### obj : ",obj.class_name)
            class_part_idx= labeled_depth_dict[obj.class_name][0]  ##### shape = 1 * idx 
            

            ##### distance based sampling in IMG
            sample_num = 2000
            idx = class_part_idx[ np.random.choice(np.arange(len(class_part_idx)),sample_num if len(class_part_idx)>sample_num else len(class_part_idx) )   ]
            sampled_class_part_idx = SAMPLED_IDX[:,idx]
            sampled_class_part_idx = cs.dist_based_sampling(sampled_class_part_idx, dist_th = 40)
            # print("sample_ num : ",len(sampled_class_part_idx.T))


            ###### convert to world coordinate
            temp_depth = depth_img_cam00[sampled_class_part_idx[0],sampled_class_part_idx[1]]
            X = (sampled_class_part_idx[1]-cx)/focal_length*temp_depth
            Y = (sampled_class_part_idx[0]-cy)/focal_length*temp_depth
            cam_to_grasp = np.array([X,Y,temp_depth,np.ones_like(X)])
            world_to_grasp = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_grasp)[:3].T
            world_to_grasp = np.unique(world_to_grasp.round(6), axis=0)

            # gripper_width_list = [0.15,0.11,0.05]#0.15,0.11,0.05
            for gripper_width in gripper_width_list:
                # print("gripper_width : ",gripper_width)

                if gripper["type"] in ["finger2", "finger2_parallel"]:
                    ## 우측 하단 꼭지에서 반시계 방향
                    ## 하단 finger , 상단 finger 순으로
                    gripper_finger_bbox = np.array([[ gripper_height/2, -gripper_width/2 -  finger_thickness],
                                                    [ gripper_height/2, -gripper_width/2 ],
                                                    [-gripper_height/2, -gripper_width/2 ],
                                                    [-gripper_height/2, -gripper_width/2 -  finger_thickness],

                                                    [ gripper_height/2,  gripper_width/2 ],
                                                    [ gripper_height/2,  gripper_width/2 +  finger_thickness],
                                                    [-gripper_height/2,  gripper_width/2 +  finger_thickness],
                                                    [-gripper_height/2,  gripper_width/2 ]])
                    ## 우측 하단 꼭지에서 반시계 방향
                    gripper_width_bbox = np.array([[ gripper_height/2, -gripper_width/2],
                                                [ gripper_height/2,  gripper_width/2],
                                                [-gripper_height/2,  gripper_width/2],
                                                [-gripper_height/2, -gripper_width/2]]) 
                    
                elif gripper["type"] in ["finger3", "finger3_parallel"]:

                    def tf_bbox(bbox,gripper):
                        shifted_bbox = []
                        for info in gripper["finger_bbox_info"]:
                            trans = cs.trans(info["pos"][0], info["pos"][1], 0)
                            rot = cs.rot_z(info["rot"])
                            shifted_bbox.append( cs.dot([trans,rot, np.vstack((bbox,np.ones_like(bbox[0])))])[:3] )
                        return np.concatenate(shifted_bbox, axis=1)[:2].T
                    

                    tmp_bbox = np.array([[ gripper_height/2, -gripper_width/2 -  finger_thickness],
                                            [ gripper_height/2, -gripper_width/2 ],
                                            [-gripper_height/2, -gripper_width/2 ],
                                            [-gripper_height/2, -gripper_width/2 -  finger_thickness]])
                    tmp_bbox = tmp_bbox.T
                    tmp_bbox = np.vstack((tmp_bbox,np.zeros_like(tmp_bbox[0])))
                    gripper_finger_bbox = tf_bbox(tmp_bbox,gripper)


                    tmp_bbox = np.array([[ gripper_height/2, -gripper_width/2],
                                            [ gripper_height/2,  0],
                                            [-gripper_height/2,  0],
                                            [-gripper_height/2, -gripper_width/2]]) 
                    tmp_bbox = tmp_bbox.T
                    tmp_bbox = np.vstack((tmp_bbox,np.zeros_like(tmp_bbox[0])))
                    gripper_width_bbox = tf_bbox(tmp_bbox,gripper)
            



                gripper_center_point = np.array([[ 0,0 ]])
                gripper_bbox = np.vstack((gripper_finger_bbox,gripper_width_bbox,gripper_center_point)).T
                gripper_bbox = np.vstack((gripper_bbox,np.zeros_like(gripper_bbox[0])))


                #### point in bbox

                rot_deg_range = range(gripper["yaw_max"],0,-10)
        

                for rot_deg in rot_deg_range:
                    # print("gripper_rot : ",rot_deg)
                    gripper_bbox_3d = cs.rot_z(rot_deg).dot(np.vstack((gripper_bbox,np.ones_like(gripper_bbox[0]))))[:3]
                    gripper_bbox_3d = np.tile(world_to_grasp[...,None], (1,1,gripper_bbox_3d.shape[1])) + np.tile(gripper_bbox_3d[None,...],(len(world_to_grasp),1,1))
                    bbx_xmin, bbx_ymin, bbx_xmax, bbx_ymax =   [np.min(gripper_bbox_3d[:,0,:]),
                                                                np.min(gripper_bbox_3d[:,1,:]),
                                                                np.max(gripper_bbox_3d[:,0,:]),
                                                                np.max(gripper_bbox_3d[:,1,:])]


                
                    X_all = (SAMPLED_IDX[1]-cx)/focal_length*depth_img_cam00[SAMPLED_IDX[0],SAMPLED_IDX[1]]
                    Y_all = (SAMPLED_IDX[0]-cy)/focal_length*depth_img_cam00[SAMPLED_IDX[0],SAMPLED_IDX[1]]
                    cam_to_obj_part = np.array([X_all,Y_all,depth_img_cam00[SAMPLED_IDX[0],SAMPLED_IDX[1]],np.ones_like(X_all)])
                    world_to_obj_part = np.array(cam_tf).dot(cs.rot_x(180)).dot(cam_to_obj_part)[:3]
                    world_to_obj_part_idx = np.where( (world_to_obj_part[0]>bbx_xmin) &(world_to_obj_part[0]<bbx_xmax) & (world_to_obj_part[1]>bbx_ymin) & (world_to_obj_part[1]<bbx_ymax) )[0]
                    world_to_obj_part = world_to_obj_part[:,world_to_obj_part_idx]

                    #random sampling
                    point_in_bboxes_idx = cs.select_points(gripper_bbox_3d,world_to_obj_part)





                    # cnt00 = 0
                    # for grasp in point_in_bboxes_idx:
                    #     cnt11 = 0
                    #     for gripper_b in grasp:
                    #         print(f"{cnt00} : {cnt11} : {gripper_b.shape}")
                    #         cnt11 += 1
                    #     cnt00 += 1
                    # point_in_bboxes_idx = cs.select_points(gripper_bbox_3d,world_to_obj_part, debug=True)



                    for bbox_num in range(len(gripper_bbox_3d)):
                        if gripper["type"] in ["finger2", "finger2_parallel"]:
                            finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                                        0 if len(point_in_bboxes_idx[bbox_num][1])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][1]].max()])
                            palm_max_depth   = np.max( 0 if len(point_in_bboxes_idx[bbox_num][2])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][2]].max())
                        elif gripper["type"] in ["finger3", "finger3_parallel"]:
                            finger_max_depth = np.max([0 if len(point_in_bboxes_idx[bbox_num][0])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][0]].max(),  
                                                        0 if len(point_in_bboxes_idx[bbox_num][1])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][1]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][2])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][2]].max()])
                            
                            palm_max_depth   = np.max([ 0 if len(point_in_bboxes_idx[bbox_num][3])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][3]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][4])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][4]].max(),
                                                        0 if len(point_in_bboxes_idx[bbox_num][5])==0 else world_to_obj_part[2][point_in_bboxes_idx[bbox_num][5]].max()])
                        
                        ########### visualization
                        ##############
                        crit = palm_max_depth - finger_max_depth
                        margin = 0.002 # default = 0.005
                        if crit<0.01:
                            continue

                        if crit>gripper_depth/10*9:
                            target_z = palm_max_depth + margin - gripper_depth/10*9
                        else:
                            target_z = max(palm_max_depth + margin -0.04, finger_max_depth + margin)
                        
                        world_points = world_to_grasp[bbox_num].copy()
                        world_points[2] = target_z
                        output_json["data"].append(
                            {
                                "gripper_model" : random_gripper_name,
                                "target_object": obj.class_name,
                                "target_points": world_points.tolist(),
                                "target_orientation": [0,0,rot_deg],
                                "target_width": gripper_width,
                            }
                        )
        output_json_list.append(output_json)

    
    with open(os.path.join(grasp_point_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
        json.dump(output_json_list,f, indent=4)

    print("PreGrasp > data_gen_time : ", time.time()-data_gen_time)
    sys.stdout.flush()


        # print("time : ",time.time()-old_time)
        # print("Grasp Data count : ",grasp_output_list.__len__())
        # old_time = time.time()
                        # print(crit)
                        ##################         2finger, parallel       ################
                        # plt.figure(figsize=(10,10))
                        # plt.scatter(PCD[0],PCD[1], c = PCD[2], cmap = "jet", s=0.1)
                        # plt.scatter(world_to_grasp.T[0],world_to_grasp.T[1], c = 'yellow', s=10)
                        # pt = gripper_bbox_3d[bbox_num]
                        # plt.plot(pt[0,[0,1,2,3,0]],pt[1,[0,1,2,3,0]],c = "g")
                        # plt.plot(pt[0,[4,5,6,7,4]],pt[1,[4,5,6,7,4]], c = "g")
                        # plt.plot(pt[0,[8,9,10,11,8]],pt[1,[8,9,10,11,8]], c = "r")
                        # plt.scatter(pt[0,12],pt[1,12], c = "b",s=5)

                        # pt_in_bbx = point_in_bboxes_idx[bbox_num]
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[0]], world_to_obj_part[1][pt_in_bbx[0]], c = "g",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[1]], world_to_obj_part[1][pt_in_bbx[1]], c = "g",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[2]], world_to_obj_part[1][pt_in_bbx[2]], c = "r",s=2)
                        # plt.axis('equal')
                        # plt.show()


                        ##################     3finger         ######################
                        # plt.figure(figsize=(10,10))
                        # plt.scatter(PCD[0],PCD[1], c = PCD[2], cmap = "jet", s=0.1)
                        # plt.scatter(world_to_grasp.T[0],world_to_grasp.T[1], c = 'yellow', s=10)
                        # pt = gripper_bbox_3d[bbox_num]
                        # plt.plot(pt[0,[0,1,2,3,0]],pt[1,[0,1,2,3,0]],c = "g")
                        # plt.plot(pt[0,[4,5,6,7,4]],pt[1,[4,5,6,7,4]], c = "g")
                        # plt.plot(pt[0,[8,9,10,11,8]],pt[1,[8,9,10,11,8]], c = "g")
                        # plt.plot(pt[0,[12,13,14,15,12]],pt[1,[12,13,14,15,12]], c = "r")
                        # plt.plot(pt[0,[16,17,18,19,16]],pt[1,[16,17,18,19,16]], c = "r")
                        # plt.plot(pt[0,[20,21,22,23,20]],pt[1,[20,21,22,23,20]], c = "r")

                        # pt_in_bbx = point_in_bboxes_idx[bbox_num]
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[0]], world_to_obj_part[1][pt_in_bbx[0]], c = "g",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[1]], world_to_obj_part[1][pt_in_bbx[1]], c = "g",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[2]], world_to_obj_part[1][pt_in_bbx[2]], c = "g",s=2)

                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[3]], world_to_obj_part[1][pt_in_bbx[3]], c = "r",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[4]], world_to_obj_part[1][pt_in_bbx[4]], c = "r",s=2)
                        # plt.scatter(world_to_obj_part[0][pt_in_bbx[5]], world_to_obj_part[1][pt_in_bbx[5]], c = "r",s=2)
                        # plt.scatter(pt[0,24],pt[1,24], c = "hotpink",s=5)
                        # plt.axis('equal')
                        # plt.show()

                        
                        ########################################################
                        # grasp_success_dict = robot_task.picking(target_points = world_points, 
                        #                                 target_orientation = [0,0,rot_deg], 
                        #                                 target_width = gripper_width,
                        #                                 target_prim_path=str(obj.prim.GetPath()))

                        
                        # print("grasp_success : ",grasp_success_dict["success"])
                        # if grasp_success_dict["success"]:
                        #     grasp_success_dict["width"] +=0.005
                        #     org_grasp_bbox = np.array([[  gripper_height/2, -grasp_success_dict["width"]/2 -grasp_success_dict["center"] -  finger_thickness, 0],
                        #                                 [-gripper_height/2, -grasp_success_dict["width"]/2 -grasp_success_dict["center"] -  finger_thickness, 0],
                        #                                 [-gripper_height/2,  grasp_success_dict["width"]/2 -grasp_success_dict["center"] +  finger_thickness, 0],
                        #                                 [ gripper_height/2,  grasp_success_dict["width"]/2 -grasp_success_dict["center"] +  finger_thickness, 0] ]).T

                            # org_grasp_bbox = cs.rot_z(rot_deg).dot(np.vstack((org_grasp_bbox,np.ones_like(org_grasp_bbox[0]))) )[:3].T
                            # ##### no restore
                            # org_grasp_bbox_shifted = world_points + org_grasp_bbox
                            # org_grasp_bbox_shifted = np.vstack((org_grasp_bbox_shifted,np.zeros_like(org_grasp_bbox_shifted[0])))

                            # ##### resetore
                            # # suc_gripper_bbx = org_grasp_bbox + grasp_success_dict["last_position"]
                            # # obj_tf = np.array(csr.find_parents_tf(obj.prim, include_self=True, include_scale=False)).T
                            # # obj_init_tf = rot_utils.euler_to_rot_matrix(obj.init_rotation,degrees=True)
                            # # obj_init_tf = np.vstack((np.hstack((obj_init_tf,np.array(obj.init_position)[:,None])),[0,0,0,1]))
                            # # suc_gripper_bbx_3d = obj_init_tf.dot(np.linalg.inv(obj_tf)).dot(np.vstack((suc_gripper_bbx.T,np.ones_like(suc_gripper_bbx.T[0]))))[:3].T
                            




                            # ## edited
                            # bbx_cnt = suc_gripper_bbx_3d.mean(axis=0)
                            # width_cnt = np.array([(suc_gripper_bbx_3d[0] + suc_gripper_bbx_3d[1])/2 , (suc_gripper_bbx_3d[2] + suc_gripper_bbx_3d[3])/2])
                            # width_cnt[:,2] = bbx_cnt[2]
                            # width_sub = (width_cnt[1] - width_cnt[0])[:2]
                            # edited_width = np.sqrt(np.sum(width_sub**2))
                            # yaw = -np.arctan(width_sub[0]/width_sub[1])
                            
                            # #### option
                            # org_grasp_bbox = np.array([ [ gripper_height/2, -edited_width/2 , 0],
                            #                             [-gripper_height/2, -edited_width/2 , 0],
                            #                             [-gripper_height/2,  edited_width/2 , 0],
                            #                             [ gripper_height/2,  edited_width/2 , 0] ]).T
                            # edited_grasp_bbox = np.vstack((org_grasp_bbox, np.ones_like(org_grasp_bbox[0]) ))
                            # edited_grasp_bbox =cs.rot_z(yaw/np.pi*180).dot(edited_grasp_bbox)[:3].T + bbx_cnt 
                            

                            
                            # top_cam_tf = np.array(cam_tf).dot(cs.rot_x(180))
                            # bbox_3d_to_2d = np.array(top_cam_config["intrinsic_isaac"]).dot(cs.dot([np.linalg.inv(top_cam_tf), np.vstack((edited_grasp_bbox.T, np.ones_like(edited_grasp_bbox.T[0]))) ])[:3])
                            # bbox_3d_to_2d = (bbox_3d_to_2d[:2]/bbox_3d_to_2d[2]).T
                            # grasp_output_dict[obj.class_name].append(
                            #     {
                            #         "bbox_3d_point":suc_gripper_bbx_3d.tolist(),
                            #         "bbox_3d_point_aligned":edited_grasp_bbox.tolist(),
                            #         "bbox_2d":{
                            #             "bbox":bbox_3d_to_2d.tolist(),
                            #             "center":bbox_3d_to_2d.mean(axis=0).tolist(),
                            #             "width":np.sqrt(np.sum((bbox_3d_to_2d[1] - bbox_3d_to_2d[2])**2) ).tolist(),
                            #             "height":np.sqrt(np.sum((bbox_3d_to_2d[0] - bbox_3d_to_2d[1])**2) ).tolist(),
                            #             "angle":yaw,
                            #             },
                            #         "quality":grasp_success_dict["quality"],
                            #         "isaac_env":{
                            #             "target_points": world_points.tolist(),
                            #             "target_orientation": [0,0,rot_deg],
                            #             "target_width": gripper_width,
                            #             "target_prim_path":str(obj.prim.GetPath())
                            #         },
                            #         "gripper_model":robot_task.gripper_model,
                                    
                            #     }
                            # )
            # with open(os.path.join(grasp_ann_path,f"{scene_num:04d}"+".json"), 'w') as f:
            #     json.dump(grasp_output_dict,f, indent=4)

if __name__ == "__main__":
    main(scene_num=0, 
         env="Manufactory",
         section="FOODnamoo_poultry_plant", 
         platform="catch_table", 
         root_path="/nas/Dataset/Dataset_2025/dataset_v1")


                




