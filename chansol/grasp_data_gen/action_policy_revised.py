import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix

import time
import sys
import getpass
sys.path.append(f"/home/uon/ochansol/isaac_code/isaac_chansol")
from Utils.general_utils import mat_utils
sys.path.append("/home/uon/ochansol/isaac_code/isaac_chansol/example/gripper_calibration")
from gripper_cal_z import GripperTool

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
        self.joint_names = list(gripper_info["joint_cfg"].keys())
        self.has_lift_joint = (
            gripper_info.get("down_lift_joint") is not None
            and len(self.joint_names) > 0
            and "lift" in self.joint_names[0].lower()
        )
        self.down_lift_joint = gripper_info.get("down_lift_joint", 0.0) or 0.0
        self.gripper_max_width = gripper_info["width"]
        self.gripper_height = gripper_info["height"]
        self.gripper_depth = gripper_info["depth"]
        self.gripper_finger_thickness = gripper_info["finger_thickness"]
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = step_dt
        self.env_origin = env_origin
        self.debug = debug
        self.frame_transformer = frame_transformer
        if self.debug:
            from isaacsim.util.debug_draw import _debug_draw

        self.z_lift_up = 0.20
        self.z_lift_up_th = 0.015
        self.criterion_th = 0.001 
        self.grasp_blocked_th = 0.005
        self.grasp_blocked_vel_th = 0.15
        self.grasp_stall_delta_th = 0.003
        self.grasp_stall_count_th = 3
        self.picking_point_offset = 0.01
        self.steady_state_th = 0.5
        self.stage_time_out_th = stage_time_out * (1/self.step_dt) ## 250 step
        self.stage1_min_wait_th = 5
        self.release_wait_th = max(1, int(0.45 / self.step_dt))
        self.stress_wait_th = max(1, int(self.release_wait_th * 1.3))
        self.object_fall_th = 0.03
        self.object_reset_dist_th = 0.10

        self.total_grasp_num = len(self.pre_grasp_data)
        self.current_grasp_num = 0
        self.output_list = []

        self.gripper_tool = GripperTool(gripper_info = gripper_info)

        self.setup()
        if self.type == "finger2":
            self.close_r_joint = gripper_info["close_r_joint"]
            self.close_l_joint = gripper_info["close_l_joint"]
            self.gripper_bbox = torch.tensor([[ self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [ self.gripper_height/2, 0, 0] ], dtype=torch.float, device=self.device)
            self.stage_action_arr = torch.tile(
                torch.zeros((5, len(self.joint_index))), (self.env_num, 1, 1)
            ).float().to(self.device)
            if self.has_lift_joint:
                self.stage_action_arr[:, 0, 0] = self.down_lift_joint
                self.stage_action_arr[:, 1, 0] = self.down_lift_joint
                self.stage_action_arr[:, 2, 0] = self.down_lift_joint
        elif self.type == "finger2_parallel":
            self.close_r_joint = gripper_info["close_r_joint"]
            self.close_l_joint = gripper_info["close_l_joint"]
            self.gripper_bbox = torch.tensor([[ self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [ self.gripper_height/2, 0, 0] ], dtype=torch.float, device=self.device)
            self.stage_action_arr = torch.tile(torch.zeros((5, len(self.joint_index))),(self.env_num,1,1)).float().to(self.device)

            self.grasp_box = torch.tensor([ [ 0, 0, 0],
                                            [ 0, 0, self.gripper_depth],
                                            [ 0, 0, self.gripper_depth],
                                            [ 0, 0, 0] ], dtype=torch.float, device=self.device)

        elif self.type == "finger3_parallel":
            self.gripper_bbox = torch.tensor([[ self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, +  0, 0],
                                            [ self.gripper_height/2, +  0, 0] ], dtype=torch.float, device=self.device)
            self.stage_action_arr = torch.tile(torch.zeros((5, len(self.joint_index))),(self.env_num,1,1)).float().to(self.device)

        elif self.type == "finger3":
            self.gripper_bbox = torch.tensor([[ self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, 0, 0],
                                            [-self.gripper_height/2, +  0, 0],
                                            [ self.gripper_height/2, +  0, 0] ], dtype=torch.float, device=self.device)
            self.stage_action_arr = torch.tile(torch.zeros((5, len(self.joint_index))),(self.env_num,1,1)).float().to(self.device)
 


        self.stage_num      = torch.ones(self.env_num, dtype=torch.int, device=self.device)
        self.stage_time_out = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.target_pos     = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)
        self.target_rot     = torch.zeros((self.env_num, 4), dtype=torch.float, device=self.device)
        self.target_width   = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.target_class   = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        self.action_enable  = torch.ones(self.env_num, dtype=torch.int, device=self.device)

        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.collision_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.stage1_close_error_prev = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.stage1_close_error_valid = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.stage1_stall_count = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        self.grasp_score = torch.zeros(self.env_num, dtype=torch.float, device=self.device)
        self.grasp_rotation_score = torch.ones(self.env_num, dtype=torch.float, device=self.device)
        self.force_direction = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)
        self.force_direction[:, 2] = -1.0


        if debug: self.draw = _debug_draw.acquire_debug_draw_interface()

        self.top_cam_config = [i for i in self.conf_data["cameras"] if i["name"]=="top_view_camera"][0]

        self.object_pos_org = torch.tensor([obj["translate"] for obj in self.obj_conf_data], dtype=torch.float, device=self.device)
        self.object_rot_org = torch.tensor([mat_utils.euler_to_quat(self.obj_conf_data[i]["orient"], degrees=True) for i in range(len(self.obj_conf_data))], dtype=torch.float, device=self.device)
        self.grasp_object_pos = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)
        self.grasp_object_quat = torch.zeros((self.env_num, 4), dtype=torch.float, device=self.device)
        self.grasp_object_quat[:, 0] = 1.0
        self.restored_grasp_box = torch.zeros((self.env_num, 4, 3), dtype=torch.float, device=self.device)
        self.restored_grasp_mat = torch.eye(4, dtype=torch.float, device=self.device).repeat(self.env_num, 1, 1)
        self.restored_force_normal = self.force_direction.clone()


    def setup(self):
        self.total_pos = []
        self.total_rot = []
        self.total_width = []
        self.total_class = []
        class_name_idx = {obj["class"]: i for i, obj in enumerate(self.obj_conf_data)}
        
        for data in self.pre_grasp_data:
            self.total_pos.append(data["target_points"])
            target_orientation = list(data["target_orientation"])
            target_orientation[0] += 180.0
            self.total_rot.append(euler_angles_to_quat(target_orientation, degrees=True))
            self.total_width.append(data["target_width"])
            self.total_class.append( class_name_idx[data["target_object"]])
        self.total_pos      = torch.tensor(self.total_pos, dtype=torch.float, device=self.device)
        self.total_rot      = torch.tensor(self.total_rot, dtype=torch.float, device=self.device)
        self.total_width    = torch.tensor(self.total_width, dtype=torch.float, device=self.device).clip(0,self.gripper_max_width)
        self.total_class    = torch.tensor(self.total_class, dtype=torch.int, device=self.device)

    def make_gripper_bbox(self,idx):

        pos = self.target_pos[idx]
        pos[...,2] -= self.gripper_info["total_length"]
        rot = self.target_rot[idx]
        joints = self.joint_pos[idx][...,self.joint_index].clone()

        if self.type == "finger2_parallel":


            width = torch.tensor([[self.gripper_tool.get_width(joint_angle_deg = joint_i/torch.pi*180)/2 for joint_i in joint] for joint in joints], dtype=torch.float, device=self.device)

            width_tensor = torch.zeros((len(idx),4,3),device=self.device)

            width_tensor[:,[0,1],1] =  (1 if self.close_r_joint<0 else -1)*width[:,None,0]
            width_tensor[:,[2,3],1] =  (-1 if self.close_l_joint<0 else 1)*width[:,None,1]

            gripper_bbox = torch.tile(self.gripper_bbox, (len(idx),1,1)) + width_tensor
            gripper_bbox = torch.concat((gripper_bbox, torch.ones((len(idx),4,1),device=self.device)), dim=2).transpose(1,2)  ### env_num, 4(x,y,z,1), 4(bbox 1,2,3,4)
            gripper_bbox = gripper_bbox[:,None,...]

        elif self.type == "finger3":
            width = torch.tensor([ ( 1 if self.gripper_info["close_f0_joint"]<0 else -1) * self.gripper_info["zero_deg_f0_joint"], 
                                   ( 1 if self.gripper_info["close_f1_joint"]<0 else -1) * self.gripper_info["zero_deg_f1_joint"], 
                                   ( 1 if self.gripper_info["close_f2_joint"]<0 else -1) * self.gripper_info["zero_deg_f2_joint"]],
                                 dtype=torch.float, device=self.device) - joints[...,[1,2,3]]
            width = torch.sin(width/180*torch.pi)*self.gripper_info["outer_link_length"] - self.gripper_info["finger_joint_thickness"] + self.gripper_info["outer_link_offset"]

            gripper_bbox = torch.tile(self.gripper_bbox, (len(idx),3,1,1))

            gripper_bbox[:,:,:2,1] -= torch.abs(width[...,None])
            
            tmp_bbox = gripper_bbox.transpose(2,3)
            rot_arr = torch.tensor([mat_utils.rot_z(0), mat_utils.rot_z(120), mat_utils.rot_z(240)], device = self.device, dtype= torch.float32)[None,...].tile(len(idx),1,1,1)
            gripper_bbox = torch.matmul(rot_arr, torch.concat((tmp_bbox, torch.tile(torch.ones(1),(tmp_bbox.shape[0],3,1,4)).to(self.device)),dim=2 ))






        rot_mat = torch.tensor([mat_utils.rot_z(quat_to_euler_angles(i, degrees=True)[-1] ) for i in rot.cpu()], dtype=torch.float, device=self.device)[:,None,...].tile((1,gripper_bbox.shape[1],1,1)) ### env_num, gripper_bbox_num, 4, 4
        rotated_bbox = torch.einsum('...ij,...jk->...ik', rot_mat, gripper_bbox).transpose(2,3)[...,:3]  ### env_num, gripper_bbox_num, 4(bbox 1,2,3,4), 4(x,y,z,1)

        bbox = rotated_bbox + pos[:,None,None,:]

        if self.debug:
            
            for bbx in bbox + self.env_origin[idx][:,None,None,:]:
                for bx in bbx:
                    self.draw.draw_lines([carb.Float3(i) for i in bx[[0,1,2,3]].cpu().numpy() ],
                                        [carb.Float3(i) for i in bx[[1,2,3,0]].cpu().numpy() ],
                                        [carb.ColorRgba(0.0,1.0,1.0,1.0),
                                        carb.ColorRgba(1.0,0.0,1.0,1.0),
                                        carb.ColorRgba(0.0,1.0,1.0,1.0),
                                        carb.ColorRgba(1.0,0.0,1.0,1.0)],
                                        [1]*4     )
        return bbox




    def make_grasp_box(self,idx):

        pos = self.target_pos[idx]
        pos[...,2] -= self.gripper_info["total_length"]
        rot = self.target_rot[idx]
        joints = self.joint_pos[idx][...,self.joint_index].clone()

        if self.type == "finger2_parallel":
            z_hop = self.gripper_tool.get_z(joint_angle_deg = joints[...,0]/torch.pi*180)

            width = torch.tensor([[self.gripper_tool.get_width(joint_angle_deg = joint_i/torch.pi*180)/2 for joint_i in joint] for joint in joints], dtype=torch.float, device=self.device)

            width_tensor = torch.zeros((len(idx),4,3),device=self.device)

            width_tensor[:,[0,1],1] =  (1 if self.close_r_joint<0 else -1)*width[:,None,0]
            width_tensor[:,[2,3],1] =  (-1 if self.close_l_joint<0 else 1)*width[:,None,1]
            width_tensor[...,2] -= z_hop[:,None]


            grasp_box = torch.tile(self.grasp_box, (len(idx),1,1)) + width_tensor
            grasp_box = torch.concat((grasp_box, torch.ones((len(idx),4,1),device=self.device)), dim=2).transpose(1,2)  ### env_num, 4(x,y,z,1), 4(bbox 1,2,3,4)
            grasp_box = grasp_box[:,None,...]





        rot_mat = torch.tensor([mat_utils.rot_z(quat_to_euler_angles(i, degrees=True)[-1] ) for i in rot.cpu()], dtype=torch.float, device=self.device)[:,None,...].tile((1,grasp_box.shape[1],1,1)) ### env_num, gripper_bbox_num, 4, 4
        rotated_bbox = torch.einsum('...ij,...jk->...ik', rot_mat, grasp_box).transpose(2,3)[...,:3]  ### env_num, gripper_bbox_num, 4(bbox 1,2,3,4), 4(x,y,z,1)

        bbox = rotated_bbox + pos[:,None,None,:]

        # if self.debug:
            
        #     for bbx in bbox + self.env_origin[idx][:,None,None,:]:
        #         for bx in bbx:
        #             self.draw.draw_lines([carb.Float3(i) for i in bx[[0,1,2,3]].cpu().numpy() ],
        #                                 [carb.Float3(i) for i in bx[[1,2,3,0]].cpu().numpy() ],
        #                                 [carb.ColorRgba(0.0,1.0,1.0,1.0),
        #                                 carb.ColorRgba(1.0,0.0,1.0,1.0),
        #                                 carb.ColorRgba(0.0,1.0,1.0,1.0),
        #                                 carb.ColorRgba(1.0,0.0,1.0,1.0)],
        #                                 [3]*4     )
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
        rot_mat = torch.tensor([mat_utils.rot_z(deg.cpu()/np.pi*180) for deg in yaw ],dtype=torch.float, device=self.device)
        edited_grasp_bbox = torch.bmm(rot_mat, edited_grasp_bbox).transpose(1,2)[...,:3] + bbx_cnt.unsqueeze(1)


        return edited_grasp_bbox
    
    def pos_rot_to_matrix(self, pos, rot):
        rot_mat = torch.tensor([quat_to_rot_matrix(i) for i in rot.cpu()], dtype=torch.float, device=self.device)
        rot_mat = torch.concat((rot_mat,pos[:,None,:].transpose(1,2)),dim=2)
        rot_mat = torch.concat((rot_mat,torch.tile( torch.tensor([[[0,0,0,1]]],dtype=torch.float, device=self.device), (len(rot_mat),1,1)) )  ,dim=1)
        return rot_mat
    
    def restore_bbox(self, idx, bbox):
        pos = self.frame_transformer.data.target_pos_source[idx,0]
        pos[:,-1] -= self.picking_point_offset
        rot = self.frame_transformer.data.target_quat_source[idx,0]
        obj_rot_mat = self.pos_rot_to_matrix(pos,rot)
        obj_rot_mat_org = self.pos_rot_to_matrix(self.object_pos_org, self.object_rot_org)
        obj_rot_mat_org = torch.tile(obj_rot_mat_org,(len(idx),1,1))
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
    
    def restore_grasp_box(self, idx, bbox):
        pos = self.grasp_object_pos[idx].clone()
        pos[:,-1] -= self.picking_point_offset
        rot = self.grasp_object_quat[idx]
        obj_rot_mat = self.pos_rot_to_matrix(pos,rot)
        obj_rot_mat_org = self.pos_rot_to_matrix(self.object_pos_org, self.object_rot_org)
        obj_rot_mat_org = torch.tile(obj_rot_mat_org,(len(idx),1,1))
        bbox = torch.concat((bbox, torch.ones((len(idx),4,1),device=self.device)), dim=2)
        result_bbox = torch.bmm(torch.linalg.inv(obj_rot_mat), bbox.transpose(1,2))# .transpose(1,2)[...,:3]
        result_bbox = torch.bmm(obj_rot_mat_org, result_bbox).transpose(1,2)[...,:3]
        rot_mat_only = torch.bmm(obj_rot_mat_org, torch.linalg.inv(obj_rot_mat))
        return result_bbox, rot_mat_only

    def cache_grasp_result_at_stage1_end(self, idx):
        if len(idx) == 0:
            return

        grasp_box = self.make_grasp_box(idx).transpose(0, 1)[0]
        restored_grasp_box, restored_grasp_mat = self.restore_grasp_box(idx, grasp_box)
        restored_force_normal = torch.bmm(
            restored_grasp_mat[:, :3, :3],
            self.force_direction[idx].unsqueeze(-1),
        ).squeeze(-1)
        restored_force_normal = restored_force_normal / torch.clamp(
            torch.linalg.norm(restored_force_normal, dim=1, keepdim=True),
            min=1.0e-6,
        )

        self.restored_grasp_box[idx] = restored_grasp_box
        self.restored_grasp_mat[idx] = restored_grasp_mat
        self.restored_force_normal[idx] = restored_force_normal
    
    def bbox_3d_to_2d(self, bbox):
        bbox = torch.concat((bbox, torch.ones((len(bbox),4,1), device=self.device)), dim=2 ).transpose(1,2)
        cam_tf = np.linalg.inv(np.array(self.top_cam_config["cam_poses"]).dot(mat_utils.rot_x(180)))
        cam_intrinsic_mat = mat_utils.mat_to_tf(torch.tensor(self.top_cam_config["intrinsic_isaac"],device=self.device ))
        tf = torch.bmm(cam_intrinsic_mat[None,...],torch.tensor(cam_tf,dtype=torch.float,device=self.device)[None,...])
        bbox_2d = torch.bmm(torch.tile(tf,(len(bbox),1,1)), bbox)
        bbox_2d = bbox_2d[:,:2]/bbox_2d[:,2][:,None,:]

        return bbox_2d.transpose(1,2)


    def record_grasp_result(self, idx):
        if len(idx)==0:
            return
        # bbox = self.make_gripper_approach_bbox(idx).transpose(1,0)
        # bbox = self.make_gripper_bbox(idx).transpose(0,1)
        # restored_bbox_3d    = torch.stack([self.restore_bbox(idx, bbx) for bbx in bbox])
        restored_grasp_box = self.restored_grasp_box[idx]
        restored_grasp_mat = self.restored_grasp_mat[idx]
        restored_force_normal = self.restored_force_normal[idx]
        if self.debug:
            for box in restored_grasp_box + self.env_origin[idx][:,None,:]:
                self.draw.draw_lines(
                    [carb.Float3(i) for i in box[[0,1,2]].cpu().numpy()],
                    [carb.Float3(i) for i in box[[1,2,3]].cpu().numpy()],
                    [
                        carb.ColorRgba(0.0,0.0,0.0,1.0),
                        carb.ColorRgba(1.0,1.0,0.0,1.0),
                        carb.ColorRgba(0.0,0.0,0.0,1.0),
                    ],
                    [3]*3,
                )
            normal_start = restored_grasp_box.mean(dim=1) + self.env_origin[idx]
            normal_end = normal_start + restored_force_normal * 0.1
            self.draw.draw_lines(
                [carb.Float3(i) for i in normal_start.cpu().numpy()],
                [carb.Float3(i) for i in normal_end.cpu().numpy()],
                [carb.ColorRgba(1.0, 0.05, 0.05, 1.0)] * len(idx),
                [6] * len(idx),
            )

        # bbox_2d = torch.stack([self.bbox_3d_to_2d(bbx).cpu() for bbx in bbox]).transpose(1,0)
        self.target_pos[idx,...,2] -= self.gripper_info["total_length"]
        for i in range(len(idx)):
            rot_euler = quat_to_euler_angles(self.target_rot[idx[i]].cpu(), degrees=True).round().tolist()
            rot_euler[0] -= 180.0
            self.output_list.append({
                    # "bbox_2d" :{
                    #     "bbox":torch.round(bbox_2d[i]).to(torch.int32).tolist(),
                    #     "center":torch.round(bbox_2d[i].mean(dim=(0,1))).to(torch.int32).tolist(),
                    #     "width":int(torch.sqrt(torch.sum((bbox_2d[i][0][1] - bbox_2d[i][0][2])**2)).round().to(torch.int32)),
                    #     "height": int(torch.sqrt(torch.sum((bbox_2d[i][0][0] - bbox_2d[i][0][1])**2) ).round().to(torch.int32)),
                    #     "angle":-quat_to_euler_angles(self.target_rot[idx[i]].cpu(), degrees=False)[-1].round(6)
                    # },
                    "grasp_box" : restored_grasp_box[i].cpu().numpy().astype(float).round(6).tolist(),
                    "grasp_mat" : restored_grasp_mat[i].cpu().numpy().astype(float).round(6).tolist(),
                    "target_points" : self.target_pos[idx[i]].cpu().numpy().astype(float).round(6).tolist(),
                    "target_orientation" : rot_euler,
                    "target_width" : self.target_width[idx[i]].cpu().numpy().astype(float).round(6).tolist(),
                    "target_object" : self.obj_conf_data[self.target_class[idx[i]].item()]["class"],
                    "gripper_model": self.gripper_info["gripper_name"],
                    "gripper_type" : self.type,
                    "score": round(float(self.grasp_score[idx[i]].detach().cpu()), 6),
                    "rotation_score": round(float(self.grasp_rotation_score[idx[i]].detach().cpu()), 6),
                    "normal": restored_force_normal[i].cpu().numpy().astype(float).round(6).tolist(),
            })
                

    def _compute_rotation_score(self, env_ids, object_quat):
        if object_quat is None or len(env_ids) == 0:
            return torch.ones(len(env_ids), dtype=torch.float, device=self.device)

        current_quat = object_quat[env_ids]
        initial_quat = self.object_rot_org[self.target_class[env_ids]]
        current_quat = current_quat / torch.clamp(torch.linalg.norm(current_quat, dim=1, keepdim=True), min=1.0e-6)
        initial_quat = initial_quat / torch.clamp(torch.linalg.norm(initial_quat, dim=1, keepdim=True), min=1.0e-6)
        quat_dot = torch.abs(torch.sum(current_quat * initial_quat, dim=1)).clamp(max=1.0)
        angle = 2.0 * torch.acos(quat_dot)
        return torch.clamp(1.0 - angle / torch.pi, min=0.0, max=1.0)

    def get_done_idx(self, object_pos=None, object_quat=None, joint_vel=None):
        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.collision_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)

        object_far_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        if object_pos is not None:
            init_pos = self.object_pos_org[self.target_class] + self.env_origin
            object_dist = torch.linalg.norm(object_pos - init_pos, dim=1)
            object_far_idx = object_dist > self.object_reset_dist_th
            self.collision_idx |= object_far_idx & (self.stage_num != 3) & (self.action_enable == 1)

        idx = torch.where(self.stage_num == 1)[0]
        if len(idx) > 0:
            close_joint_slice = slice(1, None) if self.has_lift_joint else slice(None)
            close_error = torch.sqrt(
                torch.sum(
                    (self.target_joint[idx, close_joint_slice] - self.joint_pos[idx][..., self.joint_index[close_joint_slice]]) ** 2,
                    dim=1,
                )
            )
            closed_idx = close_error < self.criterion_th
            grasped_idx = close_error > self.grasp_blocked_th
            if joint_vel is not None:
                close_speed = torch.sqrt(
                    torch.sum(joint_vel[idx][..., self.joint_index[close_joint_slice]] ** 2, dim=1)
                )
                slow_idx = close_speed < self.grasp_blocked_vel_th
            else:
                slow_idx = torch.zeros(len(idx), dtype=torch.bool, device=self.device)

            prev_error = self.stage1_close_error_prev[idx]
            valid_prev = self.stage1_close_error_valid[idx]
            close_error_delta = torch.abs(prev_error - close_error)
            stalled_now = valid_prev & (close_error_delta < self.grasp_stall_delta_th) & grasped_idx
            self.stage1_stall_count[idx] = torch.where(
                stalled_now,
                self.stage1_stall_count[idx] + 1,
                torch.zeros_like(self.stage1_stall_count[idx]),
            )
            self.stage1_close_error_prev[idx] = close_error
            self.stage1_close_error_valid[idx] = True

            stalled_idx = self.stage1_stall_count[idx] >= self.grasp_stall_count_th
            grasped_idx = grasped_idx & (slow_idx | stalled_idx)

            ready_idx = self.stage_time_out[idx] >= self.stage1_min_wait_th
            stage2_start_idx = idx[ready_idx & (closed_idx | grasped_idx)]
            self.stage_num[stage2_start_idx] = 2
            self.stage_time_out[stage2_start_idx] = 0
            self.stage1_stall_count[stage2_start_idx] = 0
            self.stage1_close_error_valid[stage2_start_idx] = False
            if len(stage2_start_idx) > 0 and object_pos is not None and object_quat is not None:
                self.grasp_object_pos[stage2_start_idx] = object_pos[stage2_start_idx] - self.env_origin[stage2_start_idx]
                grasp_object_quat = object_quat[stage2_start_idx]
                self.grasp_object_quat[stage2_start_idx] = grasp_object_quat / torch.clamp(
                    torch.linalg.norm(grasp_object_quat, dim=1, keepdim=True),
                    min=1.0e-6,
                )
                self.cache_grasp_result_at_stage1_end(stage2_start_idx)

        idx = torch.where(self.stage_num == 2)[0]
        if len(idx) > 0:
            close_joint_slice = slice(1, None) if self.has_lift_joint else slice(None)
            close_error = torch.sqrt(
                torch.sum(
                    (self.stage_action_arr[idx, 1, close_joint_slice] - self.joint_pos[idx][..., self.joint_index[close_joint_slice]]) ** 2,
                    dim=1,
                )
            )
            blocked_idx = close_error > self.grasp_blocked_th
            drop_fail_idx = torch.zeros(len(idx), dtype=torch.bool, device=self.device)
            if object_pos is not None:
                init_pos = self.object_pos_org[self.target_class[idx]] + self.env_origin[idx]
                force_drop_dist = torch.sum((object_pos[idx] - init_pos) * self.force_direction[idx], dim=1)
                drop_fail_idx = force_drop_dist > self.object_fall_th

            done_wait_idx = self.stage_time_out[idx] >= self.release_wait_th
            success_idx = done_wait_idx & blocked_idx & ~drop_fail_idx
            fail_idx = drop_fail_idx | (done_wait_idx & ~blocked_idx)
            stage3_start_idx = idx[success_idx]
            self.stage_num[stage3_start_idx] = 3
            self.stage_time_out[stage3_start_idx] = 0
            self.grasp_score[stage3_start_idx] = 0.0
            self.grasp_fail_idx[idx[fail_idx]] = True

        stress_done_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        idx = torch.where(self.stage_num == 3)[0]
        if len(idx) > 0:
            drop_fail_idx = object_far_idx[idx]
            if object_pos is not None:
                init_pos = self.object_pos_org[self.target_class[idx]] + self.env_origin[idx]
                force_drop_dist = torch.sum((object_pos[idx] - init_pos) * self.force_direction[idx], dim=1)
                drop_fail_idx = drop_fail_idx | (force_drop_dist > self.object_fall_th)

            stress_progress = torch.clamp(self.stage_time_out[idx] / self.stress_wait_th, min=0.0, max=1.0)
            stress_timeout_idx = self.stage_time_out[idx] >= self.stress_wait_th
            done_idx = drop_fail_idx | stress_timeout_idx
            done_env_idx = idx[done_idx]
            if len(done_env_idx) > 0:
                self.grasp_score[done_env_idx] = stress_progress[done_idx]
                self.grasp_score[idx[stress_timeout_idx]] = 1.0
                self.grasp_rotation_score[done_env_idx] = self._compute_rotation_score(done_env_idx, object_quat)
                self.record_grasp_result(done_env_idx)
                stress_done_idx[done_env_idx] = True

        return stress_done_idx | self.grasp_fail_idx | self.collision_idx




    def step(self):
        action_stage_num = torch.clamp(self.stage_num, max=2)
        self.target_joint = self.stage_action_arr[torch.arange(len(self.stage_num)), action_stage_num]
        criterion_idx = torch.sqrt(torch.sum((self.target_joint - self.joint_pos[...,self.joint_index])**2, dim=1)) < self.criterion_th

        stage0_idx = torch.where(self.stage_num==0)[0]
        if len(stage0_idx)>0:
            criterion_idx[stage0_idx] = False

        stage1_idx = torch.where(self.stage_num==1)[0]
        if len(stage1_idx)>0:
            criterion_idx[stage1_idx] &= self.stage_time_out[stage1_idx] >= self.stage1_min_wait_th

        stage2_idx = torch.where(self.stage_num==2)[0]
        if len(stage2_idx)>0:
            criterion_idx[stage2_idx] = False

        stage3_idx = torch.where(self.stage_num==3)[0]
        if len(stage3_idx)>0:
            criterion_idx[stage3_idx] = False
        
            # self.robot.write_root_pose_to_sim(robot_root_pos[:, :7], env_ids)

        if self.type == "finger2_parallel" :
            stage1_idx = torch.where(self.stage_num==1)[0]
            if len(stage1_idx)>0 and self.has_lift_joint and self.stage_action_arr.shape[-1] >= 6:
            
                self.stage_action_arr[stage1_idx,1,3] = -1*self.joint_pos[stage1_idx,self.joint_index[1]]
                self.stage_action_arr[stage1_idx,1,4] = -1*self.joint_pos[stage1_idx,self.joint_index[2]]
                self.stage_action_arr[stage2_idx,2,3] = -1*self.joint_pos[stage2_idx,self.joint_index[1]]
                self.stage_action_arr[stage2_idx,2,4] = -1*self.joint_pos[stage2_idx,self.joint_index[2]]

                self.stage_action_arr[stage1_idx,1,5] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_r_joint"]/180*torch.pi - self.joint_pos[stage1_idx,self.joint_index[1]])
                self.stage_action_arr[stage1_idx,2,5] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_r_joint"]/180*torch.pi - self.joint_pos[stage1_idx,self.joint_index[1]])  


            if len(stage2_idx)>0 and self.has_lift_joint and self.stage_action_arr.shape[-1] >= 5:
                self.stage_action_arr[stage2_idx,2,3] = -1*self.joint_pos[stage2_idx,self.joint_index[1]]
                self.stage_action_arr[stage2_idx,2,4] = -1*self.joint_pos[stage2_idx,self.joint_index[2]]
        if "finger3_parallel" in self.type and self.has_lift_joint and self.stage_action_arr.shape[-1] >= 8:
            stage1_idx = torch.where(self.stage_num==1)[0]
            if len(stage1_idx)>0:
                
                self.stage_action_arr[stage1_idx,1,4] = -1*self.joint_pos[stage1_idx,self.joint_index[1]]
                self.stage_action_arr[stage1_idx,1,5] = -1*self.joint_pos[stage1_idx,self.joint_index[2]]
                self.stage_action_arr[stage1_idx,1,6] = -1*self.joint_pos[stage1_idx,self.joint_index[3]]

                self.stage_action_arr[stage2_idx,2,4] = -1*self.joint_pos[stage2_idx,self.joint_index[1]]
                self.stage_action_arr[stage2_idx,2,5] = -1*self.joint_pos[stage2_idx,self.joint_index[2]]
                self.stage_action_arr[stage2_idx,2,6] = -1*self.joint_pos[stage2_idx,self.joint_index[3]]

                self.stage_action_arr[stage1_idx,1,7] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_f0_joint"]/180*torch.pi - self.joint_pos[stage1_idx,self.joint_index[1]])
                self.stage_action_arr[stage1_idx,2,7] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_f0_joint"]/180*torch.pi - self.joint_pos[stage1_idx,self.joint_index[1]])  


            if len(stage2_idx)>0:
                self.stage_action_arr[stage2_idx,2,4] = -1*self.joint_pos[stage2_idx,self.joint_index[1]]
                self.stage_action_arr[stage2_idx,2,5] = -1*self.joint_pos[stage2_idx,self.joint_index[2]]
                self.stage_action_arr[stage2_idx,2,6] = -1*self.joint_pos[stage2_idx,self.joint_index[3]]


 

        time_out_idx = (self.stage_time_out > self.stage_time_out_th) & (self.stage_num == 1)
        enable_idx = self.action_enable==1
        idx = (criterion_idx | time_out_idx) & enable_idx
        

        self.stage_num[idx] += 1
        self.stage_time_out[enable_idx] += 1
        self.stage_time_out[idx] = 0

        action_stage_num = torch.clamp(self.stage_num, max=2)
        self.target_joint = self.stage_action_arr[torch.arange(len(self.stage_num)), action_stage_num]

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
        self.stage1_close_error_prev[idx_arr] = 0.0
        self.stage1_close_error_valid[idx_arr] = False
        self.stage1_stall_count[idx_arr] = 0
        self.grasp_score[idx_arr] = 0.0
        self.grasp_rotation_score[idx_arr] = 1.0
        random_force_direction = torch.randn((len(idx_arr), 3), dtype=torch.float, device=self.device)
        random_force_direction = random_force_direction / torch.clamp(
            torch.linalg.norm(random_force_direction, dim=1, keepdim=True),
            min=1.0e-6,
        )
        random_force_direction[:, 2] = -torch.abs(random_force_direction[:, 2])
        self.force_direction[idx_arr] = random_force_direction
        self.restored_grasp_box[idx_arr] = 0.0
        self.restored_grasp_mat[idx_arr] = torch.eye(4, dtype=torch.float, device=self.device).repeat(len(idx_arr), 1, 1)
        self.restored_force_normal[idx_arr] = random_force_direction
        self.target_pos[idx_arr] = self.total_pos[idx]
        self.target_pos[idx_arr,...,2] += self.gripper_info["total_length"]
        self.target_rot[idx_arr] = self.total_rot[idx]
        self.target_width[idx_arr] = self.total_width[idx]

        if self.type == "finger2":
            r_joint_arr = (1 if self.close_r_joint<0 else -1)* self.total_width[idx]/2 + self.close_r_joint
            l_joint_arr = (1 if self.close_l_joint<0 else -1)* self.total_width[idx]/2 + self.close_l_joint
            joint_offset = 1 if self.has_lift_joint else 0
            self.stage_action_arr[idx_arr,0,joint_offset] = r_joint_arr
            self.stage_action_arr[idx_arr,0,joint_offset + 1] = l_joint_arr
            self.stage_action_arr[idx_arr,1,joint_offset] = self.close_r_joint
            self.stage_action_arr[idx_arr,1,joint_offset + 1] = self.close_l_joint
            self.stage_action_arr[idx_arr,2,joint_offset] = self.close_r_joint
            self.stage_action_arr[idx_arr,2,joint_offset + 1] = self.close_l_joint

        elif self.type == "finger2_parallel":
            joint_deg = self.gripper_tool.get_joint_angle(self.total_width[idx])
            # width = torch.arcsin( ( self.total_width[idx]/2 +self.gripper_info["finger_joint_thickness"]- self.gripper_info["outer_link_offset"]) /self.gripper_info["outer_link_length"]).abs()
            # r_joint_arr = (1 if self.close_r_joint<0 else -1)* width + self.gripper_info["zero_deg_r_joint"]/180*torch.pi
            # l_joint_arr = (1 if self.close_l_joint<0 else -1)* width + self.gripper_info["zero_deg_l_joint"]/180*torch.pi
            close_r = self.close_r_joint/180*torch.pi
            close_l = self.close_l_joint/180*torch.pi


            self.stage_action_arr[idx_arr,0,0] = joint_deg/180*torch.pi
            self.stage_action_arr[idx_arr,0,1] = joint_deg/180*torch.pi
            self.stage_action_arr[idx_arr,1,0] = close_r
            self.stage_action_arr[idx_arr,1,1] = close_l
            self.stage_action_arr[idx_arr,2,0] = close_r
            self.stage_action_arr[idx_arr,2,1] = close_l



        elif self.type == "finger3_parallel":
            width = torch.arcsin( ( self.total_width[idx]/2 +self.gripper_info["finger_joint_thickness"]- self.gripper_info["outer_link_offset"]) /self.gripper_info["outer_link_length"]).abs()
            f0_joint_arr = (1 if self.gripper_info["close_f0_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f0_joint"]/180*torch.pi
            f1_joint_arr = (1 if self.gripper_info["close_f1_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f1_joint"]/180*torch.pi
            f2_joint_arr = (1 if self.gripper_info["close_f2_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f2_joint"]/180*torch.pi


            if self.has_lift_joint:
                self.stage_action_arr[idx_arr,0,0] = self.down_lift_joint
                self.stage_action_arr[idx_arr,1,0] = self.down_lift_joint
                self.stage_action_arr[idx_arr,2,0] = self.down_lift_joint
                joint_offset = 1
            else:
                joint_offset = 0
            

            self.stage_action_arr[idx_arr,0,joint_offset] = f0_joint_arr
            self.stage_action_arr[idx_arr,0,joint_offset + 1] = f1_joint_arr
            self.stage_action_arr[idx_arr,0,joint_offset + 2] = f2_joint_arr
            self.stage_action_arr[idx_arr,1,joint_offset] = self.gripper_info["close_f0_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,1,joint_offset + 1] = self.gripper_info["close_f1_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,1,joint_offset + 2] = self.gripper_info["close_f2_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,2,joint_offset] = self.gripper_info["close_f0_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,2,joint_offset + 1] = self.gripper_info["close_f1_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,2,joint_offset + 2] = self.gripper_info["close_f2_joint"]/180*torch.pi

            if self.has_lift_joint and self.stage_action_arr.shape[-1] >= 8:
                self.stage_action_arr[idx_arr,0,4] = -1 * f0_joint_arr
                self.stage_action_arr[idx_arr,0,5] = -1 * f1_joint_arr
                self.stage_action_arr[idx_arr,0,6] = -1 * f2_joint_arr
                self.stage_action_arr[idx_arr,0,7] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_f0_joint"]/180*torch.pi - f0_joint_arr)  

                self.stage_action_arr[idx_arr,1,4] = -1 * f0_joint_arr
                self.stage_action_arr[idx_arr,1,5] = -1 * f1_joint_arr
                self.stage_action_arr[idx_arr,1,6] = -1 * f2_joint_arr
                self.stage_action_arr[idx_arr,1,7] = self.gripper_info["outer_link_length"]*torch.cos(self.gripper_info["zero_deg_f0_joint"]/180*torch.pi - f0_joint_arr)  

        elif self.type == "finger3":
            width = torch.arcsin( ( self.total_width[idx]/2 +self.gripper_info["finger_joint_thickness"]- self.gripper_info["outer_link_offset"]) /self.gripper_info["outer_link_length"]).abs()
            f0_joint_arr = (1 if self.gripper_info["close_f0_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f0_joint"]/180*torch.pi
            f1_joint_arr = (1 if self.gripper_info["close_f1_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f1_joint"]/180*torch.pi
            f2_joint_arr = (1 if self.gripper_info["close_f2_joint"]<0 else -1)* width + self.gripper_info["zero_deg_f2_joint"]/180*torch.pi


            if self.has_lift_joint:
                self.stage_action_arr[idx_arr,0,0] = self.down_lift_joint
                self.stage_action_arr[idx_arr,1,0] = self.down_lift_joint
                self.stage_action_arr[idx_arr,2,0] = self.down_lift_joint
                joint_offset = 1
            else:
                joint_offset = 0
            

            self.stage_action_arr[idx_arr,0,joint_offset] = f0_joint_arr
            self.stage_action_arr[idx_arr,0,joint_offset + 1] = f1_joint_arr
            self.stage_action_arr[idx_arr,0,joint_offset + 2] = f2_joint_arr

            self.stage_action_arr[idx_arr,1,joint_offset] = self.gripper_info["close_f0_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,1,joint_offset + 1] = self.gripper_info["close_f1_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,1,joint_offset + 2] = self.gripper_info["close_f2_joint"]/180*torch.pi

            self.stage_action_arr[idx_arr,2,joint_offset] = self.gripper_info["close_f0_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,2,joint_offset + 1] = self.gripper_info["close_f1_joint"]/180*torch.pi
            self.stage_action_arr[idx_arr,2,joint_offset + 2] = self.gripper_info["close_f2_joint"]/180*torch.pi


        self.target_class[idx_arr] = self.total_class[idx]
        self.grasp_object_pos[idx_arr] = self.object_pos_org[self.target_class[idx_arr]]
        self.grasp_object_quat[idx_arr] = self.object_rot_org[self.target_class[idx_arr]]

        self.stage_time_out[idx_arr] = 0

        self.current_grasp_num += len(idx_arr)
        print(f"current grasp num : {self.current_grasp_num}")

        # self.draw.clear_points()
        # self.draw.draw_points([carb.Float3(i) for i in self.target_pos[idx_arr].cpu().numpy()], 
        #                       [carb.ColorRgba(1.0,0.0,0.0,1.0)]*len(idx_arr), 
        #                       [3]*len(idx_arr))
