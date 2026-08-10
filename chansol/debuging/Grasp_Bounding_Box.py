import numpy as np
import json 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import os
import tkinter as tk
import ast
import utils

# root_path = "../Data_samples"
root_path = "/nas/Dataset/Dataset_2024"
img_sampling_num = 1000
all_data_num = 1000


###### data path ######
dir_path = "scene_rev20241209/Home"
rgb_path = os.path.join(root_path, dir_path, "rgb/cam_00")
bbox_path  = os.path.join(root_path, dir_path, "output_grasp")
inst_seg_path = os.path.join(root_path, dir_path, "inst_seg/cam_00")


def rot_z(rad):

    return np.array([
        [np.cos(rad), -np.sin(rad) ],
        [np.sin(rad),  np.cos(rad)],
    ])


class bbox_display():
    def __init__(self, rgb_path, bbox_path, inst_seg_path,img_sampling_num, all_data_num, scene_num = -1):
        
        self.rgb_path = rgb_path
        self.bbox_path = bbox_path
        self.img_sampling_num = img_sampling_num
        self.inst_seg_path = inst_seg_path
        self.all_data_num = all_data_num
        
        self.sample_num_list = np.arange(1000)#np.random.choice(range(all_data_num),size = img_sampling_num, replace=False)
        self.grasp_sample_num_list = []
        self.acc_dict = {}

        for sample_num in self.sample_num_list:
            self.acc_dict[sample_num] = {}


        
        
        
        self.window = tk.Tk()
        self.window.title("Bounding Box")
        self.window.geometry("1000x1300")

        
        
        ####### plt #######
        self.fig, self.ax = plt.subplots(figsize = (14,7))
        self.fig.tight_layout()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        
        
        
        
        ##### Frame #####
        self.img_frame = tk.Frame(self.window)
        self.img_frame.pack(side=tk.TOP, pady=(0,30) ,fill=tk.X)
        
        self.left_frame = tk.Frame(self.window)
        self.left_frame.pack(side=tk.LEFT,fill=tk.X)
        
        self.right_frame = tk.Frame(self.window)
        self.right_frame.pack(side=tk.RIGHT,fill=tk.X)
        
        self.eval_frame = tk.Frame(self.left_frame)
        self.eval_frame.pack(side=tk.TOP,pady=30,fill=tk.X)
        
        self.eval_top_frame = tk.Frame(self.eval_frame)
        self.eval_top_frame.pack(side=tk.TOP,fill=tk.X)
        self.eval_L_frame = tk.Frame(self.eval_frame)
        self.eval_L_frame.pack(side=tk.LEFT,fill=tk.X)
        self.eval_R_frame = tk.Frame(self.eval_frame)
        self.eval_R_frame.pack(side=tk.RIGHT,fill=tk.X)
        
        self.move_frame = tk.Frame(self.left_frame)
        self.move_frame.pack(side=tk.TOP,fill=tk.X)
        
        self.scene_frame = tk.Frame(self.move_frame)
        self.scene_frame.pack(side=tk.BOTTOM,fill=tk.X)
        
        self.grasp_option_frame = tk.Frame(self.right_frame)
        self.grasp_option_frame.pack(side=tk.TOP,fill=tk.X)
        
        self.grasp_move_frame = tk.Frame(self.right_frame)
        self.grasp_move_frame.pack(side=tk.TOP,fill=tk.X)
        
    
        
        
        ######  layout ######
        
        self.file_name_label = tk.Label(self.img_frame,  font = ("Arial",20), text="file_name")
        self.file_name_label.pack(side=tk.TOP)
        
        self.can_plt = FigureCanvasTkAgg(self.fig,master = self.img_frame)
        self.can_plt.draw()
        self.can_plt.get_tk_widget().pack(side = tk.TOP, fill=tk.BOTH, expand=1)
        self.plt_bbox = []
        
        self.toolbar = NavigationToolbar2Tk(self.can_plt, self.img_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.slider = tk.Scale(self.img_frame, orient=tk.HORIZONTAL, width = 30, length=400, sliderlength=100, from_=0, to=100, command=self.slider_move)
        self.slider.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        self.slider_label = tk.Label(self.img_frame, font = ("Arial",20), text="Line Width")
        self.slider_label.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        
        self.acc_label = tk.Label(self.eval_top_frame, font = ("Arial",20,"bold"),fg="#FF0000", text="Accuracy")
        self.acc_label.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        
        self.correct_label = tk.Label(self.eval_L_frame, font = ("Arial",20), text="Correct")
        self.correct_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.incorrect_label = tk.Label(self.eval_R_frame, font = ("Arial",20), text="Incorrect")
        self.incorrect_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        
        self.correct_btn = tk.Button(self.eval_L_frame, width = 3, text="O", font = ("Arial",80), command=self.correct, bg="#FFFFFF" , activebackground="#50FF50")
        self.correct_btn.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        self.incorrect_btn = tk.Button(self.eval_R_frame, width = 3, text="X", font = ("Arial",80),command=self.incorrect, bg="#FFFFFF", activebackground="#FF5050")
        self.incorrect_btn.pack(side=tk.TOP,fill=tk.BOTH, expand=True)
        

        self.prev_btn = tk.Button(self.move_frame, width =10, font = ("Arial",30), text="Prev", command=self.prev)
        self.prev_btn.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.next_btn = tk.Button(self.move_frame, width = 10,font = ("Arial",30), text="Next", command=self.next)
        self.next_btn.pack(side=tk.RIGHT,fill=tk.BOTH, expand=True)
        
        self.scene_num_label = tk.Label(self.scene_frame, font = ("Arial",22), text="Scene Num")
        self.scene_num_label.pack(side=tk.LEFT, padx=10,fill=tk.BOTH, expand=True)
        self.scene_entry = tk.Entry(self.scene_frame, width=5, font = ("Arial",20))
        self.scene_entry.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.scene_entry.bind("<Return>", self.go_to_scene)
        self.scene_entry.bind("<KP_Enter>", self.go_to_scene)
        self.scene_entry.insert(0, f"{scene_num+1}")
        self.scene_change_btn = tk.Button(self.scene_frame, width = 3, font = ("Arial",20), text="Go", command= lambda : self.go_to_scene(None))
        self.scene_change_btn.pack(side=tk.RIGHT,fill=tk.BOTH, expand=True)
        
        self.window.bind("<Button-1>", lambda event: event.widget.focus_set())
        self.window.bind("<Left>", lambda x:self.prev() if self.window.focus_get() != self.scene_entry else None)
        self.window.bind("<Right>", lambda x: self.next() if self.window.focus_get() != self.scene_entry else None)
        
        
        self.inst_seg_var = tk.IntVar()
        self.inst_seg_vis_check = tk.Checkbutton(self.grasp_option_frame, width = 15, font = ("Arial",20), text="Viz Segmentation", bg = "white",activebackground="#eeeeee", variable=self.inst_seg_var, command=self.inst_seg_vis)
        self.inst_seg_vis_check.pack(side=tk.TOP, pady=(0,20),fill=tk.BOTH, expand=True)
        self.grasp_auto_eval_btn = tk.Button(self.grasp_option_frame, width = 8, font = ("Arial",25), text="Auto Eval", bg = "#90F060",command=self.grasp_auto_eval)
        self.grasp_auto_eval_btn.pack(side=tk.LEFT, pady=(0,30),fill=tk.BOTH, expand=True)
        self.grasp_auto_eval_all_data_btn = tk.Button(self.grasp_option_frame, width = 5, font = ("Arial",10), text="Auto Eval All", bg = "#90F060",command=self.auto_eval_all)
        self.grasp_auto_eval_all_data_btn.pack(side=tk.RIGHT, pady=(0,30),fill=tk.BOTH, expand=True)
        
        self.grasp_show_all_btn = tk.Button(self.grasp_move_frame, width = 10,font = ("Arial",30), text="Show All", command=self.grasp_show_all)
        self.grasp_show_all_btn.pack(side=tk.BOTTOM,pady=(10,0),fill=tk.BOTH, expand=True)
        self.grasp_slider = tk.Scale(self.grasp_move_frame, orient=tk.HORIZONTAL, width = 30, length=400, sliderlength=100, from_=1, to=100, command=self.grasp_slider_move)
        self.grasp_slider.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.grasp_num_label = tk.Label(self.grasp_move_frame, font = ("Arial",22), text="Grasp Num")
        self.grasp_num_label.pack(side=tk.BOTTOM,fill=tk.BOTH, expand=True)
        self.grasp_prev_btn = tk.Button(self.grasp_move_frame, width = 5, font = ("Arial",50), text="\u2190", command=self.grasp_prev)
        self.grasp_prev_btn.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        self.grasp_next_btn = tk.Button(self.grasp_move_frame, width = 5,font = ("Arial",50), text="\u2192", command=self.grasp_next)
        self.grasp_next_btn.pack(side=tk.RIGHT,fill=tk.BOTH, expand=True)





        ###### init ######
        self.scene_num = scene_num
        self.grasp_num = -1
        self.bbox_line_width = 2

        self.acc_label.config(text=f"Accuracy : {self.evaluate()}")

        self.go_to_scene(None)
        # self.window.mainloop()
        
    def update(self):
        self.window.update()

        
        
    def next(self):
        if self.scene_num<self.img_sampling_num:
            self.scene_num += 1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            self.can_plt.draw()
            self.eval_btn_update()

    def prev(self):
        if self.scene_num>0:
            self.scene_num -= 1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            self.scene_entry.delete(0, tk.END)
            self.scene_entry.insert(0, f"{self.scene_num+1}")
            self.can_plt.draw()
            self.eval_btn_update()

    def correct(self):
        if self.scene_num != -1:
            self.acc_dict[self.sample_num_list[self.scene_num]][self.grasp_num] = True
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            self.plt_bbox[self.grasp_num][0].set_color("#0000FF")
            self.plt_bbox[self.grasp_num][1].set_color("#0000FF")
            self.plt_bbox[self.grasp_num][2].set_color("#FF0000")
            self.plt_bbox[self.grasp_num][3].set_color("#FF0000")
            self.can_plt.draw()
            self.eval_btn_update()


    def incorrect(self):
        if self.scene_num != -1:
            self.acc_dict[self.sample_num_list[self.scene_num]][self.grasp_num] = False
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            self.plt_bbox[self.grasp_num][0].set_color("#FF00FF")
            self.plt_bbox[self.grasp_num][1].set_color("#FF00FF")
            self.plt_bbox[self.grasp_num][2].set_color("#00FF00")
            self.plt_bbox[self.grasp_num][3].set_color("#00FF00")
            self.can_plt.draw()
            self.eval_btn_update()
    
    def eval_btn_update(self):
        self.correct_btn.config(bg="#FFFFFF")
        self.incorrect_btn.config(bg="#FFFFFF")
        try:
            if self.acc_dict[self.sample_num_list[self.scene_num]][self.grasp_num] == True:
                self.correct_btn.config(bg="#00FF00")
            elif self.acc_dict[self.sample_num_list[self.scene_num]][self.grasp_num] == False:
                self.incorrect_btn.config(bg="#FF0000")
        except:
            pass

    def draw_bbox(self,bbox, gripper_type = "parallel", color=True ):
        if color:
            line1_color = "#0000FF"
            line2_color = "#FF0000"
        else:
            line1_color = "#FF00FF"
            line2_color = "#00FF00"
        
        if gripper_type in ["finger2_parallel", "finger2"]:
            self.plt_bbox+=[
                self.ax.plot( bbox.T[0,[1,2]], bbox.T[1,[1,2]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox.T[0,[3,0]], bbox.T[1,[3,0]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox.T[0,[0,1]], bbox.T[1,[0,1]],c=line2_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox.T[0,[2,3]], bbox.T[1,[2,3]],c=line2_color, linewidth=self.bbox_line_width)
                ]
        elif gripper_type in ["finger3"]:
            self.plt_bbox+=[
                self.ax.plot( bbox[0].T[0,[1,2]], bbox[0].T[1,[1,2]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[0].T[0,[3,0]], bbox[0].T[1,[3,0]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[0].T[0,[0,1]], bbox[0].T[1,[0,1]],c=line2_color, linewidth=self.bbox_line_width)+\
                
                self.ax.plot( bbox[1].T[0,[1,2]], bbox[1].T[1,[1,2]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[1].T[0,[3,0]], bbox[1].T[1,[3,0]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[1].T[0,[0,1]], bbox[1].T[1,[0,1]],c=line2_color, linewidth=self.bbox_line_width)+\
                
                self.ax.plot( bbox[2].T[0,[1,2]], bbox[2].T[1,[1,2]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[2].T[0,[3,0]], bbox[2].T[1,[3,0]],c=line1_color, linewidth=self.bbox_line_width)+\
                self.ax.plot( bbox[2].T[0,[0,1]], bbox[2].T[1,[0,1]],c=line2_color, linewidth=self.bbox_line_width)
                # self.ax.plot( bbx.T[0,[2,3]], bbx.T[1,[2,3]],c=line2_color, linewidth=self.bbox_line_width)
                ]


        


    def data_load(self, scene_num):
        self.file_name_label.config(text=f"File Name : {scene_num:04d}.png")
        self.rgb_img = Image.open(os.path.join(self.rgb_path, f"{scene_num:04d}.png"))
        self.inst_seg_img = Image.open(os.path.join(self.inst_seg_path, f"{scene_num:04d}.png"))
        self.inst_seg_arr = np.array(self.inst_seg_img)
        with open(os.path.join(self.inst_seg_path, f"semantics_mapping_{scene_num:04d}.json"),"r") as f:
            self.inst_seg_label = json.load(f)
        with open(os.path.join(self.bbox_path, f"{scene_num:04d}.json"), "r") as f:
            self.bbox = json.load(f)
            
        self.ax.cla()
        self.ax.imshow(self.rgb_img)
        self.plt_inst_seg = self.ax.imshow(self.inst_seg_img)
        self.plt_inst_seg.set_visible(self.inst_seg_var.get())
        
        self.plt_bbox = []
        cnt=0
        for grasp in self.bbox:
            bbox_2d = grasp["bbox_2d"]["bbox"]
            center = grasp["bbox_2d"]["center"]
            width = grasp["bbox_2d"]["width"]
            height = grasp["bbox_2d"]["height"]
            angle = grasp["bbox_2d"]["angle"]
            gripper_type = grasp["gripper_type"]

            if gripper_type in ["finger2_parallel", "finger2"]:
                bbox = rot_z(angle).dot(np.array([[-width/2, -height/2],
                                [-width/2, +height/2],
                                [+width/2, +height/2],
                                [+width/2, -height/2]]).T).T + center
            elif gripper_type in ["finger3"]:
                bbox = np.array(bbox_2d)

            
            try:
                if self.acc_dict[scene_num][cnt] == True:
                    self.draw_bbox(bbox,gripper_type = gripper_type, color=True)
                elif self.acc_dict[scene_num][cnt] == False:
                    self.draw_bbox(bbox,gripper_type = gripper_type, color=False)
            except:
                self.draw_bbox(bbox,gripper_type = gripper_type, color=True)

            
            cnt+=1
        self.grasp_slider.config(to=len(self.plt_bbox))
        self.grasp_num = 0
        self.grasp_slider.set(self.grasp_num+1)
        # self.grasp_slider_move(1)
        self.grasp_num_label.config(text=f"Grasp Num : {self.grasp_num+1}/{len(self.plt_bbox)}")
                
            
        
    
    def slider_move(self, val):
        if self.plt_bbox is not []:
            self.bbox_line_width = int(val)/100 *15
            for bbox in self.plt_bbox:
                for bbx in bbox:
                    bbx.set_linewidth(self.bbox_line_width)
            self.can_plt.draw()
    
    def evaluate(self):
        correct = 0
        incorrect = 0
        for key in self.acc_dict.keys():
            for grasp_key in self.acc_dict[key].keys():
                if self.acc_dict[key][grasp_key] == True:
                    correct += 1
                elif self.acc_dict[key][grasp_key] == False:
                    incorrect += 1
        print(correct, incorrect)
        self.correct_label.config(text=f"Correct : {correct}")
        self.incorrect_label.config(text=f"Incorrect : {incorrect}")
        
        if correct==0:
            return 0
        return correct/(correct+incorrect)
    
    def go_to_scene(self, event):
        if int(self.scene_entry.get()) <= self.img_sampling_num and int(self.scene_entry.get()) > 0:
            self.scene_num = int(self.scene_entry.get())-1
            self.data_load(self.sample_num_list[self.scene_num])
            self.scene_num_label.config(text=f"{self.scene_num+1}/{self.img_sampling_num}")
            self.can_plt.draw()
            self.eval_btn_update()
            
    def grasp_next(self):
        if self.grasp_num<len(self.plt_bbox):
            self.grasp_num += 1
            self.grasp_slider.set(self.grasp_num+1)
            self.eval_btn_update()
        

    def grasp_prev(self):
        if self.grasp_num>0:
            self.grasp_num -= 1
            self.grasp_slider.set(self.grasp_num+1)
            self.eval_btn_update()
            
    def grasp_slider_move(self,val):
        if self.plt_bbox is not []:
            self.grasp_num = int(val)-1
            for i,plt_bbox in enumerate(self.plt_bbox):
                if i != self.grasp_num:
                    for bbox in plt_bbox:
                        bbox.set_visible(False)
                else:
                    for bbox in plt_bbox:
                        bbox.set_visible(True)
            self.can_plt.draw()
            self.grasp_num_label.config(text=f"Grasp Num : {self.grasp_num+1}/{len(self.plt_bbox)}")
            self.eval_btn_update()
    def grasp_show_all(self):
        if self.plt_bbox is not []:
            for plt_bbox in self.plt_bbox:
                for bbox in plt_bbox:
                    bbox.set_visible(True)
            self.can_plt.draw()
            
    def grasp_auto_eval(self):
        self.grasp_auto_eval_btn.config(state="disabled")
        try:
            points = []
            for key in self.bbox.keys():
                bboxes = []
                
                for grasp in self.bbox[key]:
                    center = grasp["center"]
                    width = grasp["width"]
                    height = grasp["height"]
                    angle = grasp["angle"]
                    bbox = rot_z(angle).dot(np.array([[-width/2, -height/2],
                                    [-width/2, +height/2],
                                    [+width/2, +height/2],
                                    [+width/2, -height/2]]).T).T + center
                    bboxes.append(bbox.T)

                for inst_key in self.inst_seg_label.keys():
                    if self.inst_seg_label[inst_key]["class"] == key:
                        rgba = ast.literal_eval(inst_key)
                        idx = np.where(np.all(self.inst_seg_arr==np.array(rgba) ,axis=2) == True)   # idx = 2 x N
                        idx = utils.select_points(bboxes, idx)
                        points.append(idx)
            cnt = 0
            for pt_c in points:
                for pt_b in pt_c:
                    if len(pt_b) == 0:
                        print("Incorrect")
                        self.plt_bbox[cnt][0].set_color("#FF00FF")
                        self.plt_bbox[cnt][1].set_color("#FF00FF")
                        self.plt_bbox[cnt][2].set_color("#00FF00")
                        self.plt_bbox[cnt][3].set_color("#00FF00")
                        
                    self.acc_dict[self.sample_num_list[self.scene_num]][cnt] = len(pt_b) != 0
                    cnt+=1
                    # self.ax.scatter(pt_b.T[1], pt_b.T[0], c="g", s=1)

            self.can_plt.draw()
            self.acc_label.config(text=f"Accuracy : {round(self.evaluate(),3)}")
            self.eval_btn_update()
            self.grasp_auto_eval_btn.config(state="normal")
        except : 
            print("Error")
            self.grasp_auto_eval_btn.config(state="normal")

    
    def inst_seg_vis(self):
        if self.inst_seg_var.get() == 1:
            self.inst_seg_vis_check.config(bg="yellow")
            self.inst_seg_vis_check.config(font = ("Arial",20,"bold"))
            self.plt_inst_seg.set_visible(True)
            self.can_plt.draw()
        else:
            self.inst_seg_vis_check.config(bg="white")
            self.inst_seg_vis_check.config(font = ("Arial",20))
            self.plt_inst_seg.set_visible(False)
            self.can_plt.draw()
            
    def auto_eval_all(self):
        self.scene_num = -1
        for num in range(self.img_sampling_num):
            self.next()
            self.grasp_auto_eval()
        
            

    



            

if __name__ == "__main__":
    bbox_display(rgb_path, bbox_path, inst_seg_path, img_sampling_num, all_data_num)