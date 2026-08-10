import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix

import time
import sys
sys.path.append("/home/cubox/ochansol/isaac_code/python/utils")
import cs_utils as cs

import carb

class ActionPolicy:
    def __init__(self, env_num,
                 pre_grasp_data,
                 conf_data,
                 gripper_info,
                 joint_index,  
                 joint_pos,
                 step_dt,
                 device, 
                 contact_sensor, 
                 frame_transformer,
                 env_origin,
                 debug,
                 stage_time_out=0.4):


        self.env_num = env_num
        self.pre_grasp_data = pre_grasp_data
        self.obj_conf_data = conf_data["objects"]
        self.conf_data = conf_data
        self.device = device
        self.gripper_info = gripper_info
        self.type = gripper_info["type"]
        self.down_lift_joint = gripper_info["down_lift_joint"]
        self.close_r_joint = gripper_info["close_r_joint"]
        self.close_l_joint = gripper_info["close_l_joint"]
        self.gripper_max_width = gripper_info["width"]
        self.gripper_height = gripper_info["height"]
        self.gripper_finger_thickness = gripper_info["finger_thickness"]
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = step_dt
        self.contact_sensor = contact_sensor
        self.frame_transformer = frame_transformer
        self.env_origin = env_origin
        self.debug = debug
        if self.debug:
            from isaacsim.util.debug_draw import _debug_draw

        self.z_lift_up = 0.20
        self.criterion_th = 0.001 
        self.picking_point_offset = 0.01
        self.steady_state_th = 0.5
        self.stage_time_out_th = stage_time_out * (1/self.step_dt) ## 250 step

        self.total_grasp_num = len(self.pre_grasp_data)
        self.current_grasp_num = 0
        self.output_list = []

        self.setup()
        if self.type == "parallel_jaw":
            self.stage_action_arr = torch.tile(torch.tensor([
                [self.down_lift_joint, 0, 0],
                [self.down_lift_joint, self.close_r_joint, self.close_l_joint],
                [self.down_lift_joint+self.z_lift_up, self.close_r_joint, self.close_l_joint],
                [0,0,0],
                [0,0,0]
            ]),(self.env_num,1,1)).float().to(self.device)
        elif self.type == "finger2":
            self.stage_action_arr = torch.tile(torch.tensor([
                [self.down_lift_joint, 0, 0, 0, 0],
                #[self.down_lift_joint,                  self.close_r_joint/180*torch.pi, self.close_l_joint/180*torch.pi, self.gripper_info["close_rf_joint"]/180*torch.pi, self.gripper_info["close_lf_joint"]/180*torch.pi],
                [self.down_lift_joint,                  self.close_r_joint/180*torch.pi, self.close_l_joint/180*torch.pi, 0,0],
                #[self.down_lift_joint+self.z_lift_up,   self.close_r_joint/180*torch.pi, self.close_l_joint/180*torch.pi, self.gripper_info["close_rf_joint"]/180*torch.pi, self.gripper_info["close_lf_joint"]/180*torch.pi],
                [self.down_lift_joint+self.z_lift_up,   self.close_r_joint/180*torch.pi, self.close_l_joint/180*torch.pi, 0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]
            ]),(self.env_num,1,1)).float().to(self.device)
        self.stage_num      = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        self.stage_time_out = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.target_pos     = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)
        self.target_rot     = torch.zeros((self.env_num, 4), dtype=torch.float, device=self.device)
        self.target_width   = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.target_class   = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        self.action_enable  = torch.ones(self.env_num, dtype=torch.int, device=self.device)

        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.collision_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)

        self.gripper_bbox = torch.tensor([[ self.gripper_height/2, -  self.gripper_finger_thickness, 0],
                                          [-self.gripper_height/2, -  self.gripper_finger_thickness, 0],
                                          [-self.gripper_height/2, +  self.gripper_finger_thickness, 0],
                                          [ self.gripper_height/2, +  self.gripper_finger_thickness, 0] ], dtype=torch.float, device=self.device)
        if debug: self.draw = _debug_draw.acquire_debug_draw_interface()

        self.object_pos_org = torch.tensor([self.obj_conf_data[i]["translate"] for i in range(len(self.obj_conf_data))], dtype=torch.float, device=self.device)
        self.object_rot_org = torch.tensor([self.obj_conf_data[i]["orient"] for i in range(len(self.obj_conf_data))], dtype=torch.float, device=self.device)
        self.top_cam_config = [i for i in self.conf_data["cameras"] if i["name"]=="top_view_camera"][0]


    def setup(self):
        self.total_pos = []
        self.total_rot = []
        self.total_width = []
        self.total_class = []
        class_name_idx = {name:i for i,name in enumerate(self.frame_transformer.data.target_frame_names)}
        for data in self.pre_grasp_data:
            self.total_pos.append(data["target_points"])
            self.total_rot.append(euler_angles_to_quat(data["target_orientation"], degrees=True))
            self.total_width.append(data["target_width"])
            self.total_class.append( class_name_idx[data["target_object"]])
        self.total_pos      = torch.tensor(self.total_pos, dtype=torch.float, device=self.device)
        self.total_rot      = torch.tensor(self.total_rot, dtype=torch.float, device=self.device)
        self.total_width    = torch.tensor(self.total_width, dtype=torch.float, device=self.device).clip(0,self.gripper_max_width)
        self.total_class    = torch.tensor(self.total_class, dtype=torch.int, device=self.device)

    def pos_rot_to_matrix(self, pos, rot):
        rot_mat = torch.tensor([quat_to_rot_matrix(i) for i in rot.cpu()], dtype=torch.float, device=self.device)
        rot_mat = torch.concat((rot_mat,pos[:,None,:].transpose(1,2)),dim=2)
        rot_mat = torch.concat((rot_mat,torch.tile( torch.tensor([[[0,0,0,1]]],dtype=torch.float, device=self.device), (len(rot_mat),1,1)) )  ,dim=1)

        return rot_mat


    def restore_bbox(self, idx,obj_class_idx, bbox):
        pos = self.frame_transformer.data.target_pos_source[idx,obj_class_idx]
        pos[:,-1] -= self.picking_point_offset
        rot = self.frame_transformer.data.target_quat_source[idx, obj_class_idx]
        obj_rot_mat = self.pos_rot_to_matrix(pos,rot)
        obj_rot_mat_org = self.pos_rot_to_matrix(self.object_pos_org[obj_class_idx], self.object_rot_org[obj_class_idx])
        bbox = torch.concat((bbox, torch.ones((len(idx),4,1),device=self.device)), dim=2)
        result_bbox = torch.bmm(torch.linalg.inv(obj_rot_mat), bbox.transpose(1,2))# .transpose(1,2)[...,:3]
        result_bbox = torch.bmm(obj_rot_mat_org, result_bbox).transpose(1,2)[...,:3]

        if self.debug:
            for box in result_bbox + self.env_origin[idx][:,None,:]:
                self.draw.draw_lines([carb.Float3(i) for i in box[[0,1,2,3]].cpu().numpy() ],
                                        [carb.Float3(i) for i in box[[1,2,3,0]].cpu().numpy() ],
                                        [carb.ColorRgba(0.0,1.0,0.0,1.0),
                                        carb.ColorRgba(1.0,0.0,0.0,1.0),
                                        carb.ColorRgba(0.0,1.0,0.0,1.0),
                                        carb.ColorRgba(1.0,0.0,0.0,1.0)],
                                        [1]*4     )
        return result_bbox


    def make_gripper_bbox(self,idx):
        lift_joint_diff = self.joint_pos[idx,self.joint_index[0]] - self.down_lift_joint 
        pos = self.target_pos[idx]
        pos[:,-1] += lift_joint_diff
        rot = self.target_rot[idx]
        joints = self.joint_pos[idx][...,self.joint_index].clone()
        if self.type == "parallel_jaw":
            close_joint = torch.tensor([self.down_lift_joint,self.close_r_joint, self.close_l_joint],dtype=torch.float, device=self.device)
            gripper_joint_idx = torch.where((self.joint_index ==1) | (self.joint_index ==2))[0]
            width = (close_joint[gripper_joint_idx] - joints[...,gripper_joint_idx]).abs()
        elif self.type == "finger2":
            close_joint = torch.tensor([self.down_lift_joint,
                                        self.gripper_info["zero_deg_r_joint"], self.gripper_info["zero_deg_l_joint"], 
                                        self.gripper_info["close_rf_joint"], self.gripper_info["close_lf_joint"] ],
                                    dtype=torch.float, device=self.device)

            gripper_joint_idx = torch.where((self.joint_index ==1) | (self.joint_index ==2))[0]
            # gripper_joint_idx = torch.tensor([torch.where(self.joint_index==1)[0], torch.where(self.joint_index==2)[0] ])

            width = (close_joint[gripper_joint_idx]/180*torch.pi - joints[...,gripper_joint_idx]).abs()
            width = torch.sin(width)*self.gripper_info["outer_link_length"] - self.gripper_info["finger_joint_thickness"] + self.gripper_info["outer_link_offset"]
        # import pdb; pdb.set_trace()
        width_tensor = torch.zeros((len(idx),4,3),device=self.device)

        width_tensor[:,[0,1],1] = -width[:,None,1]
        width_tensor[:,[2,3],1] =  width[:,None,0]

        gripper_bbox = torch.tile(self.gripper_bbox, (len(idx),1,1)) + width_tensor
        gripper_bbox = torch.concat((gripper_bbox, torch.ones((len(idx),4,1),device=self.device)), dim=2).transpose(1,2)  ### env_num, 4(x,y,z,1), 4(bbox 1,2,3,4)
        rot_mat = torch.tensor([cs.rot_z(quat_to_euler_angles(i, degrees=True)[-1] ) for i in rot.cpu()], dtype=torch.float, device=self.device) ### env_num, 4,4
        rotated_bbox = torch.bmm(rot_mat, gripper_bbox).transpose(1,2)[...,:3]  ### env_num,4(bbox 1,2,3,4), 4(x,y,z,1)


        bbox = rotated_bbox + pos[:,None,:]


        if self.debug:
            for box in bbox + self.env_origin[idx][:,None,:]:
                self.draw.draw_lines([carb.Float3(i) for i in box[[0,1,2,3]].cpu().numpy() ],
                                        [carb.Float3(i) for i in box[[1,2,3,0]].cpu().numpy() ],
                                        [carb.ColorRgba(0.0,1.0,1.0,1.0),
                                        carb.ColorRgba(1.0,0.0,1.0,1.0),
                                        carb.ColorRgba(0.0,1.0,1.0,1.0),
                                        carb.ColorRgba(1.0,0.0,1.0,1.0)],
                                        [1]*4     )
        return bbox
        
    def align_bbox(self, bbox): ### bbox = env_num * 4(bbox) * 3(x,y,z)
        bbx_cnt = bbox.mean(dim=1)
        width_cnt = torch.concat( ( ((bbox[:,0] + bbox[:,1])/2).unsqueeze(1), ((bbox[:,2] + bbox[:,3])/2).unsqueeze(1)), dim=1)
        width_cnt[...,2] = bbx_cnt[:,None,:][...,2]
        width_sub = (width_cnt[:,1] - width_cnt[:,0])[:,:2]
        edited_width = torch.sqrt(torch.sum(width_sub**2 ,dim=1) )
        yaw = -torch.arctan(width_sub[:,0]/width_sub[:,1])
        
        #### option

        width_tensor = torch.zeros((len(bbox),4,3),device=self.device)
        width_tensor[:,[0,1],1] = -edited_width[:,None]/2 + self.gripper_finger_thickness
        width_tensor[:,[2,3],1] =  edited_width[:,None]/2 - self.gripper_finger_thickness


        gripper_bbox = torch.tile(self.gripper_bbox, (len(bbox),1,1)) + width_tensor
        edited_grasp_bbox = torch.concat((gripper_bbox, torch.ones((len(bbox),4,1), device=self.device)), dim=2 ).transpose(1,2)
        rot_mat = torch.tensor([cs.rot_z(deg.cpu()/np.pi*180) for deg in yaw ],dtype=torch.float, device=self.device)
        edited_grasp_bbox = torch.bmm(rot_mat, edited_grasp_bbox).transpose(1,2)[...,:3] + bbx_cnt.unsqueeze(1)


        return edited_grasp_bbox
    
    def bbox_3d_to_2d(self, bbox):
        bbox = torch.concat((bbox, torch.ones((len(bbox),4,1), device=self.device)), dim=2 ).transpose(1,2)
        cam_tf = np.linalg.inv(np.array(self.top_cam_config["cam_poses"]).dot(cs.rot_x(180)))
        cam_intrinsic_mat = cs.mat_to_tf(torch.tensor(self.top_cam_config["intrinsic_isaac"],device=self.device ))
        tf = torch.bmm(cam_intrinsic_mat[None,...],torch.tensor(cam_tf,dtype=torch.float,device=self.device)[None,...])
        bbox_2d = torch.bmm(torch.tile(tf,(len(bbox),1,1)), bbox)
        bbox_2d = bbox_2d[:,:2]/bbox_2d[:,2][:,None,:]

        return bbox_2d.transpose(1,2)


    def check_object_lift_up(self, idx):
        if len(idx)==0:
            return
        object_pos = self.frame_transformer.data.target_pos_source[idx]
        # only_one_item_idx = torch.sum( (object_pos - self.object_pos_org)[...,2] >0.02, dim=1) == 1
        # if len(idx[only_one_item_idx])>0:

        #     idx = idx[only_one_item_idx]
        #     objects_pos = self.frame_transformer.data.target_pos_source[idx,self.target_class[idx]]
        #     objects_pos_org = torch.tensor([self.obj_conf_data[i]["translate"] for i in self.target_class[idx]], dtype=torch.float, device=self.device)
        #     success_idx = (objects_pos - objects_pos_org)[:,2] >0.02
        #     if len(idx[success_idx])>0:
        #         bbox = self.make_gripper_bbox(idx[success_idx])
        #         restored_bbox_3d = self.restore_bbox(idx[success_idx],self.target_class[idx[success_idx]], bbox) 
        #         aligned_bbox_3d = self.align_bbox(restored_bbox_3d)
        #         bbox_2d = self.bbox_3d_to_2d(aligned_bbox_3d).cpu()

        #         for i in range(len(bbox)):
        #             bbox_yaw = torch.atan2(bbox_2d[i][2][1] - bbox_2d[i][1][1], bbox_2d[i][2][0] - bbox_2d[i][1][0])
        #             self.output_list.append({
        #                 "bbox_3d_point" : restored_bbox_3d[i].cpu().tolist(),
        #                 "bbox_3d_aligned" : aligned_bbox_3d[i].cpu().tolist(),
        #                 "bbox_2d" :{
        #                     "bbox":bbox_2d[i].tolist(),
        #                     "center":bbox_2d[i].mean(dim=0).tolist(),
        #                     "width":torch.sqrt(torch.sum((bbox_2d[i][1] - bbox_2d[i][2])**2) ).tolist(),
        #                     "height":torch.sqrt(torch.sum((bbox_2d[i][0] - bbox_2d[i][1])**2) ).tolist(),
        #                     "angle":bbox_yaw.tolist(),
        #                 },
        #                 "target_points" : self.target_pos[idx[success_idx][i]].cpu().tolist(),
        #                 "target_orientation" : quat_to_euler_angles(self.target_rot[idx[success_idx][i]].cpu(), degrees=True).tolist(),
        #                 "target_width" : self.target_width[idx[success_idx][i]].cpu().tolist(),
        #                 "target_object" : self.frame_transformer.data.target_frame_names[self.target_class[idx[success_idx][i]]],
        #                 # "gripper_model": self.pre_grasp_data[0]["gripper_model"],
        #                 "gripper_model": self.gripper_info["gripper_name"],
        #                 "disturbed_object_count" : 0,

        #             })



        #         if self.debug:
        #             for box in aligned_bbox_3d + self.env_origin[idx][:,None,:] :
        #                 self.draw.draw_lines([carb.Float3(i) for i in box[[0,1,2,3]].cpu().numpy() ],
        #                                         [carb.Float3(i) for i in box[[1,2,3,0]].cpu().numpy() ],
        #                                         [carb.ColorRgba(1.0,1.0,0.0,1.0),
        #                                         carb.ColorRgba(0.0,0.0,1.0,1.0),
        #                                         carb.ColorRgba(1.0,1.0,0.0,1.0),
        #                                         carb.ColorRgba(0.0,0.0,1.0,1.0)],
        #                                         [1]*4     )
        # import pdb; pdb.set_trace()
        disturbed_object_count = torch.sum( (object_pos - self.object_pos_org) >0.02, dim=(1,2)).cpu()


        objects_pos = self.frame_transformer.data.target_pos_source[idx,self.target_class[idx]]
        objects_pos_org = torch.tensor([self.obj_conf_data[i]["translate"] for i in self.target_class[idx]], dtype=torch.float, device=self.device)
        success_idx = (objects_pos - objects_pos_org)[:,2] >0.02
        if len(idx[success_idx])>0:
            bbox = self.make_gripper_bbox(idx[success_idx])
            restored_bbox_3d = self.restore_bbox(idx[success_idx],self.target_class[idx[success_idx]], bbox) 
            aligned_bbox_3d = self.align_bbox(restored_bbox_3d)
            bbox_2d = self.bbox_3d_to_2d(aligned_bbox_3d).cpu()

            for i,index in enumerate(torch.where(success_idx==success_idx)[0][success_idx].cpu()) :
                bbox_yaw = torch.atan2(bbox_2d[i][2][1] - bbox_2d[i][1][1], bbox_2d[i][2][0] - bbox_2d[i][1][0])
                self.output_list.append({
                    "bbox_3d_point" : restored_bbox_3d[i].cpu().tolist(),
                    "bbox_3d_aligned" : aligned_bbox_3d[i].cpu().tolist(),
                    "bbox_2d" :{
                        "bbox":bbox_2d[i].tolist(),
                        "center":bbox_2d[i].mean(dim=0).tolist(),
                        "width":torch.sqrt(torch.sum((bbox_2d[i][1] - bbox_2d[i][2])**2) ).tolist(),
                        "height":torch.sqrt(torch.sum((bbox_2d[i][0] - bbox_2d[i][1])**2) ).tolist(),
                        "angle":bbox_yaw.tolist(),
                    },
                    "target_points" : self.target_pos[idx[success_idx][i]].cpu().tolist(),
                    "target_orientation" : quat_to_euler_angles(self.target_rot[idx[success_idx][i]].cpu(), degrees=True).tolist(),
                    "target_width" : self.target_width[idx[success_idx][i]].cpu().tolist(),
                    "target_object" : self.frame_transformer.data.target_frame_names[self.target_class[idx[success_idx][i]]],
                    # "gripper_model": self.pre_grasp_data[0]["gripper_model"],
                    "gripper_model": self.gripper_info["gripper_name"],
                    "disturbed_object_count" : int(disturbed_object_count[index]),
                })



            if self.debug:
                for box in aligned_bbox_3d + self.env_origin[idx][:,None,:] :
                    self.draw.draw_lines([carb.Float3(i) for i in box[[0,1,2,3]].cpu().numpy() ],
                                            [carb.Float3(i) for i in box[[1,2,3,0]].cpu().numpy() ],
                                            [carb.ColorRgba(1.0,1.0,0.0,1.0),
                                            carb.ColorRgba(0.0,0.0,1.0,1.0),
                                            carb.ColorRgba(1.0,1.0,0.0,1.0),
                                            carb.ColorRgba(0.0,0.0,1.0,1.0)],
                                            [1]*4     )
                




    def get_done_idx(self):
        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.collision_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)

        idx = torch.where(self.stage_num==0)[0]
        if len(idx)>0:
            contact_collision_idx     = self.contact_sensor.data.net_forces_w[idx,:,2].abs().sum(dim=1)  >3
            self.collision_idx[idx[contact_collision_idx]] = True
            # if torch.sum(self.collision_idx)>0:
        
        idx = torch.where(self.stage_num==1)[0]
        if len(idx)>0:
            contact_detect_idx = self.contact_sensor.data.net_forces_w[idx].abs().sum(dim=(1,2)) >2
            if len(idx[contact_detect_idx])>0:
                steady_state_idx = torch.std(self.contact_sensor.data.net_forces_w_history[idx[contact_detect_idx]], dim=1, unbiased=False).sum(dim=(1,2)) < self.steady_state_th
                self.stage_num[idx[contact_detect_idx][steady_state_idx]] +=1

            criterion_idx = torch.sqrt(torch.sum((self.target_joint[idx,1:] - self.joint_pos[idx][...,self.joint_index[1:]])**2, dim=1)) < self.criterion_th
            self.stage_num[idx[criterion_idx]] +=1
            
            # steady_state_idx = torch.std(self.contact_sensor.data.net_forces_w_history[idx], dim=1, unbiased=False).sum(dim=(1,2)) <0.05
            # self.act_pol.stage_num[idx[steady_state_idx]] +=1

        idx = torch.where(self.stage_num==2)[0]
        if len(idx)>0:

            contact_grasp_fail_idx = torch.sum(torch.max(self.contact_sensor.data.net_forces_w_history[idx],dim = 1)[0].abs(),dim=(1,2)) <0.2
            self.grasp_fail_idx[idx[contact_grasp_fail_idx]] = True

        stage_end_idx = self.stage_num == 3

        idx = torch.where(stage_end_idx)[0]
        if len(idx)>0:
            self.check_object_lift_up(idx)

        return stage_end_idx | self.grasp_fail_idx | self.collision_idx




    def step(self):
        
        self.target_joint = self.stage_action_arr[torch.arange(len(self.stage_num)),self.stage_num]
        # print("target_joint",self.target_joint)
        # print("joint_pos",self.joint_pos[...,self.joint_index])
        criterion_idx = torch.sqrt(torch.sum((self.target_joint - self.joint_pos[...,self.joint_index])**2, dim=1)) < self.criterion_th
        
        stage2_idx = torch.where(self.stage_num==2)[0]
        if len(stage2_idx)>0:
            tmp_idx = (self.target_joint[stage2_idx,0] - self.joint_pos[stage2_idx,self.joint_index[0]]).abs()< 0.015
            criterion_idx[stage2_idx] = tmp_idx

        if self.type == "finger2" :
            stage1_idx = torch.where(self.stage_num==1)[0]
            if len(stage1_idx)>0:

                self.stage_action_arr[stage1_idx,1,self.joint_index[3]] = -1*self.joint_pos[stage1_idx,1]
                self.stage_action_arr[stage1_idx,1,self.joint_index[4]] = -1*self.joint_pos[stage1_idx,2]
                self.stage_action_arr[stage2_idx,2,self.joint_index[3]] = -1*self.joint_pos[stage2_idx,1]
                self.stage_action_arr[stage2_idx,2,self.joint_index[4]] = -1*self.joint_pos[stage2_idx,2]
            if len(stage2_idx)>0:
                self.stage_action_arr[stage2_idx,2,self.joint_index[3]] = -1*self.joint_pos[stage2_idx,1]
                self.stage_action_arr[stage2_idx,2,self.joint_index[4]] = -1*self.joint_pos[stage2_idx,2]


 

        time_out_idx = self.stage_time_out > self.stage_time_out_th
        enable_idx = self.action_enable==1
        idx = (criterion_idx | time_out_idx) & enable_idx
        

        self.stage_num[idx] += 1
        self.stage_time_out[enable_idx] += 1
        self.stage_time_out[idx] = 0

        self.target_joint = self.stage_action_arr[torch.arange(len(self.stage_num)),self.stage_num]

        return self.target_joint
    



    def reset(self, idx_arr):
        if self.current_grasp_num + len(idx_arr) > self.total_grasp_num:
            if self.current_grasp_num >= self.total_grasp_num:
                self.action_enable[idx_arr] = 0
                self.stage_num[idx_arr] = 4
                return
            else:
                self.action_enable[idx_arr[self.total_grasp_num - self.current_grasp_num:]] = 0
                self.stage_num[idx_arr[self.total_grasp_num - self.current_grasp_num:]] = 4
                idx_arr = idx_arr[:self.total_grasp_num - self.current_grasp_num]



        idx = np.arange(len(idx_arr)) + self.current_grasp_num
        self.stage_num[idx_arr] = 0
        self.target_pos[idx_arr] = self.total_pos[idx]
        self.target_rot[idx_arr] = self.total_rot[idx]
        self.target_width[idx_arr] = self.total_width[idx]

        if self.type == "parallel_jaw":
            r_joint_arr = (1 if self.close_r_joint<0 else -1)* self.total_width[idx]/2 + self.close_r_joint
            l_joint_arr = (1 if self.close_l_joint<0 else -1)* self.total_width[idx]/2 + self.close_l_joint
            self.stage_action_arr[idx_arr,0,1] = r_joint_arr
            self.stage_action_arr[idx_arr,0,2] = l_joint_arr

        elif self.type == "finger2":
            width = torch.arcsin( ( self.total_width[idx]/2 +self.gripper_info["finger_joint_thickness"]- self.gripper_info["outer_link_offset"]) /self.gripper_info["outer_link_length"]).abs()
            r_joint_arr = (1 if self.close_r_joint<0 else -1)* width + self.gripper_info["zero_deg_r_joint"]/180*torch.pi
            l_joint_arr = (1 if self.close_l_joint<0 else -1)* width + self.gripper_info["zero_deg_l_joint"]/180*torch.pi

            self.stage_action_arr[idx_arr,0,0] = self.target_pos[idx_arr][:,2] + self.down_lift_joint

            self.stage_action_arr[idx_arr,0,1] = r_joint_arr
            self.stage_action_arr[idx_arr,0,2] = l_joint_arr
            self.stage_action_arr[idx_arr,0,3] = -1 * r_joint_arr
            self.stage_action_arr[idx_arr,0,4] = -1 * l_joint_arr

            self.stage_action_arr[idx_arr,1,3] = -1 * r_joint_arr
            self.stage_action_arr[idx_arr,1,4] = -1 * l_joint_arr
            import pdb; pdb.set_trace()

        elif self.type == "finger3":
            width = torch.arcsin( ( self.total_width[idx]/2 +self.gripper_info["finger_joint_thickness"]- self.gripper_info["outer_link_offset"]) /self.gripper_info["outer_link_length"]).abs()
            r_joint_arr = (1 if self.close_r_joint<0 else -1)* width + self.gripper_info["zero_deg_r_joint"]/180*torch.pi
            l_joint_arr = (1 if self.close_l_joint<0 else -1)* width + self.gripper_info["zero_deg_l_joint"]/180*torch.pi

            self.stage_action_arr[idx_arr,0,1] = r_joint_arr
            self.stage_action_arr[idx_arr,0,2] = l_joint_arr
            self.stage_action_arr[idx_arr,0,3] = -1 * r_joint_arr
            self.stage_action_arr[idx_arr,0,4] = -1 * l_joint_arr

            self.stage_action_arr[idx_arr,1,3] = -1 * r_joint_arr
            self.stage_action_arr[idx_arr,1,4] = -1 * l_joint_arr



        self.target_class[idx_arr] = self.total_class[idx]

        self.stage_time_out[idx_arr] = 0

        self.current_grasp_num += len(idx_arr)

        # self.draw.clear_points()
        # self.draw.draw_points([carb.Float3(i) for i in self.target_pos[idx_arr].cpu().numpy()], 
        #                       [carb.ColorRgba(1.0,0.0,0.0,1.0)]*len(idx_arr), 
        #                       [3]*len(idx_arr))



