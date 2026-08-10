import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import glob
import json
import csv
import numpy as np
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def rot_z(angle):
    """Z축 회전 행렬"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a],
                     [sin_a, cos_a]])

# CustomTkinter 설정
ctk.set_appearance_mode("dark")  # 다크 모드
ctk.set_default_color_theme("blue")  # 블루 테마

class ImageGraspViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Output Grasp Validator")
        self.root.geometry("1800x1000")
        
        # 데이터 경로 및 변수 초기화
        self.root_path = "/nas/Dataset/Dataset_2026/dataset_v2"
        # self.root_path = "/nas/Dataset/Dataset_2025"
        self.current_images = []
        self.current_index = 0
        self.current_image_path = None
        self.current_grasp_data = None
        self.current_inst_seg_data = None  # instance segmentation 데이터 추가
        self.current_bbox_index = 0
        self.processed_bboxes = []
        self.clicked_target_object = None  # 클릭한 물체 추적
        
        # 성능 최적화용 캐시
        self.image_cache = {}
        self.grasp_cache = {}
        self.inst_seg_cache = {}  # instance segmentation 캐시 추가
        self.is_updating_scale = False
        
        # Object filtering
        self.object_list = []  # 전체 물체 목록
        self.checked_objects = set()  # 체크된 물체들
        self.filtered_images = []  # 필터링된 이미지 목록
        self.object_check_window = None  # Object Check 창
        
        # 현재 표시된 이미지 객체
        self.display_image = None
        self.display_photo = None
        
        # 상태 메시지 표시를 위한 변수
        self.status_message = ""

        # Isaac Lab replay runs in a separate process so Tk remains responsive.
        self.sim_process = None
        self.sim_enabled = False
        self.sim_restart_key = None
        self.sim_request_id = 0
        self.sim_last_status_stamp = None
        self.sim_sync_after_id = None
        self.sim_runtime_dir = Path(tempfile.mkdtemp(prefix="output_grasp_replay_"))
        self.sim_control_file = self.sim_runtime_dir / "control.json"
        self.sim_status_file = self.sim_runtime_dir / "status.json"
        
        # GUI 구성 요소 생성
        self.create_widgets()
        
        # 키보드 바인딩 (스크롤 영역에도 적용)
        self.root.bind('<Key>', self.on_key_press)
        self.left_canvas.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
        # 스크롤 영역 초기화
        self.root.after(100, self.configure_scroll_region)
        
        # 초기 폴더 구조 로드
        self.load_folder_structure()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(500, self.poll_sim_status)
    
    def show_status(self, message, level="info"):
        """상태 메시지를 콘솔과 GUI에 표시"""
        print(f"[{level.upper()}] {message}")
        self.status_message = message
        # GUI 상태 표시 업데이트 (필요시)
        if hasattr(self, 'info_label'):
            current_text = self.info_label.cget("text")
            if "STATUS:" not in current_text:
                self.info_label.configure(text=f"{current_text} | STATUS: {message}")
    
    def create_widgets(self):
        """GUI 위젯 생성 - 좌우 분할 레이아웃으로 개선"""
        # 메인 컨테이너
        main_container = ctk.CTkFrame(self.root, corner_radius=0)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 좌우 분할 - 왼쪽: 제어판, 오른쪽: 이미지
        # 왼쪽 제어판 프레임 (고정 너비)
        left_panel_frame = ctk.CTkFrame(main_container, corner_radius=15, width=450)
        left_panel_frame.pack(side="left", fill="y", padx=(5, 2.5), pady=5)
        left_panel_frame.pack_propagate(False)
        
        # 왼쪽 제어판 스크롤 영역
        self.left_canvas = tk.Canvas(left_panel_frame, bg='#212121', highlightthickness=0, width=430)
        left_scrollbar = ctk.CTkScrollbar(left_panel_frame, orientation="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # 스크롤 가능한 프레임
        left_panel = ctk.CTkFrame(self.left_canvas, fg_color="transparent")
        
        # 캔버스에 프레임 추가
        self.canvas_frame_id = self.left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        # 스크롤바와 캔버스 배치
        self.left_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        left_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # 마우스 휠 스크롤 바인딩
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.left_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # 프레임 크기 업데이트 이벤트 바인딩
        left_panel.bind("<Configure>", self.configure_scroll_region)
        self.left_canvas.bind("<Configure>", self.configure_scroll_region)
        
        # 오른쪽 이미지 영역 (확장)
        right_panel = ctk.CTkFrame(main_container, corner_radius=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(2.5, 5), pady=5)
        
        # === 왼쪽 제어판 구성 ===
        # 제목
        title_label = ctk.CTkLabel(left_panel, text="🎯 Output Grasp Validator", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(15, 20))
        
        # 경로 선택 섹션
        path_section = ctk.CTkFrame(left_panel, fg_color="transparent")
        path_section.pack(fill="x", padx=15, pady=(0, 15))
        
        # Environment 선택
        ctk.CTkLabel(path_section, text="Environment", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.env_combo = ctk.CTkComboBox(path_section, command=self.on_env_selected,
                                        font=ctk.CTkFont(size=11))
        self.env_combo.pack(fill="x", pady=(3, 8))
        
        # Section 선택
        ctk.CTkLabel(path_section, text="Section", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.section_combo = ctk.CTkComboBox(path_section, command=self.on_section_selected,
                                            font=ctk.CTkFont(size=11))
        self.section_combo.pack(fill="x", pady=(3, 8))
        
        # Platform 선택
        ctk.CTkLabel(path_section, text="Platform", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        self.platform_combo = ctk.CTkComboBox(path_section, command=self.on_platform_selected,
                                             font=ctk.CTkFont(size=11))
        self.platform_combo.pack(fill="x", pady=(3, 8))
        
        # 로드 버튼
        self.load_btn = ctk.CTkButton(path_section, text="🚀 Load Images", 
                                     command=self.load_images, height=35,
                                     font=ctk.CTkFont(size=13, weight="bold"),
                                     fg_color=("#1f538d", "#14375e"), hover_color=("#14375e", "#1f538d"))
        self.load_btn.pack(fill="x", pady=(5, 8))

        self.load_sim_btn = ctk.CTkButton(
            path_section,
            text="▶ Load to Sim",
            command=self.load_to_sim,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#2E8B57", "#1F6040"),
            hover_color=("#256F46", "#174B31"),
        )
        self.load_sim_btn.pack(fill="x", pady=(0, 8))

        self.sim_status_label = ctk.CTkLabel(
            path_section,
            text="SIM: stopped",
            font=ctk.CTkFont(size=11),
            anchor="w",
            text_color=("#555555", "#BBBBBB"),
        )
        self.sim_status_label.pack(fill="x", pady=(0, 8))

        self.sim_debug_legend = ctk.CTkLabel(
            path_section,
            text=(
                "SIM lines: yellow=target, magenta=START→END, "
                "RGB=pose axes, cyan=grasp bbox"
            ),
            font=ctk.CTkFont(size=10),
            anchor="w",
            justify="left",
            wraplength=390,
            text_color=("#555555", "#AAAAAA"),
        )
        self.sim_debug_legend.pack(fill="x", pady=(0, 8))
        
        # Object Check 버튼
        self.object_check_btn = ctk.CTkButton(path_section, text="🔍 Object Check", 
                                             command=self.open_object_check_window, height=35,
                                             font=ctk.CTkFont(size=13, weight="bold"),
                                             fg_color=("#8B4513", "#654321"), hover_color=("#654321", "#8B4513"))
        self.object_check_btn.pack(fill="x", pady=(0, 8))
        
        # Clear Filter 버튼
        self.clear_filter_btn = ctk.CTkButton(path_section, text="🚫 Clear Filter", 
                                             command=self.clear_object_filter, height=35,
                                             font=ctk.CTkFont(size=13, weight="bold"),
                                             fg_color=("#DC143C", "#B22222"), hover_color=("#B22222", "#DC143C"))
        self.clear_filter_btn.pack(fill="x", pady=(0, 0))
        
        # 구분선
        separator1 = ctk.CTkFrame(left_panel, height=2, fg_color=("#CCCCCC", "#333333"))
        separator1.pack(fill="x", padx=15, pady=12)
        
        # 이미지 네비게이션 섹션
        img_nav_section = ctk.CTkFrame(left_panel, fg_color="transparent")
        img_nav_section.pack(fill="x", padx=15, pady=(0, 12))
        
        ctk.CTkLabel(img_nav_section, text="📷 Image Navigation", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0, 8))
        
        # 이미지 정보
        self.info_label = ctk.CTkLabel(img_nav_section, text="Load images to start", 
                                      font=ctk.CTkFont(size=12), anchor="w")
        self.info_label.pack(fill="x", pady=(0, 6))
        
        # 이미지 버튼들
        img_btn_frame = ctk.CTkFrame(img_nav_section, fg_color="transparent")
        img_btn_frame.pack(fill="x", pady=(0, 6))
        
        self.prev_img_btn = ctk.CTkButton(img_btn_frame, text="◀ Prev", command=self.prev_image, 
                                         width=80, height=32, font=ctk.CTkFont(size=12))
        self.prev_img_btn.pack(side="left", padx=(0, 5))
        
        self.next_img_btn = ctk.CTkButton(img_btn_frame, text="Next ▶", command=self.next_image, 
                                         width=80, height=32, font=ctk.CTkFont(size=12))
        self.next_img_btn.pack(side="right")
        
        # 이미지 슬라이더
        ctk.CTkLabel(img_nav_section, text="Image Slider", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.image_slider = ctk.CTkSlider(img_nav_section, from_=0, to=0, number_of_steps=1,
                                         command=self.on_image_slider_change, height=20)
        self.image_slider.pack(fill="x", pady=(5, 0))
        
        # 구분선
        separator2 = ctk.CTkFrame(left_panel, height=2, fg_color=("#CCCCCC", "#333333"))
        separator2.pack(fill="x", padx=15, pady=12)
        
        # BBox 네비게이션 섹션
        bbox_nav_section = ctk.CTkFrame(left_panel, fg_color="transparent")
        bbox_nav_section.pack(fill="x", padx=15, pady=(0, 12))
        
        ctk.CTkLabel(bbox_nav_section, text="🎯 BBox Navigation", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0, 8))
        
        # BBox 정보
        self.bbox_info_label = ctk.CTkLabel(bbox_nav_section, text="", 
                                           font=ctk.CTkFont(size=12, weight="bold"), anchor="w")
        self.bbox_info_label.pack(fill="x", pady=(0, 4))
        
        # BBox 모델 정보
        self.bbox_model_label = ctk.CTkLabel(bbox_nav_section, text="", 
                                            font=ctk.CTkFont(size=11), anchor="w",
                                            text_color=("#7CC7FF", "#4A9EFF"))
        self.bbox_model_label.pack(fill="x", pady=(0, 8))
        
        # BBox 버튼들
        bbox_btn_frame = ctk.CTkFrame(bbox_nav_section, fg_color="transparent")
        bbox_btn_frame.pack(fill="x", pady=(0, 6))
        
        self.prev_bbox_btn = ctk.CTkButton(bbox_btn_frame, text="◀ Prev", command=self.prev_bbox, 
                                          width=80, height=32, font=ctk.CTkFont(size=12))
        self.prev_bbox_btn.pack(side="left", padx=(0, 5))
        
        self.next_bbox_btn = ctk.CTkButton(bbox_btn_frame, text="Next ▶", command=self.next_bbox, 
                                          width=80, height=32, font=ctk.CTkFont(size=12))
        self.next_bbox_btn.pack(side="right")
        
        # BBox 슬라이더
        ctk.CTkLabel(bbox_nav_section, text="BBox Slider", font=ctk.CTkFont(size=11)).pack(anchor="w")
        self.bbox_slider = ctk.CTkSlider(bbox_nav_section, from_=0, to=0, number_of_steps=1,
                                        command=self.on_bbox_slider_change, height=20)
        self.bbox_slider.pack(fill="x", pady=(5, 0))
        
        # 구분선
        separator3 = ctk.CTkFrame(left_panel, height=2, fg_color=("#CCCCCC", "#333333"))
        separator3.pack(fill="x", padx=15, pady=12)
        
        # BBox 옵션 섹션
        bbox_options_section = ctk.CTkFrame(left_panel, fg_color="transparent")
        bbox_options_section.pack(fill="x", padx=15, pady=(0, 20))  # 마지막 섹션이므로 아래 패딩 추가
        
        ctk.CTkLabel(bbox_options_section, text="⚙️ Display Options", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0, 8))
        
        # 옵션 스위치들
        self.show_bbox_var = ctk.BooleanVar(value=True)
        self.bbox_switch = ctk.CTkSwitch(bbox_options_section, text="Show BBox", variable=self.show_bbox_var,
                                        command=self.on_bbox_option_change, font=ctk.CTkFont(size=12))
        self.bbox_switch.pack(fill="x", pady=(0, 6))
        
        self.show_all_bbox_var = ctk.BooleanVar(value=False)
        self.all_bbox_switch = ctk.CTkSwitch(bbox_options_section, text="Show All BBoxes", 
                                            variable=self.show_all_bbox_var,
                                            command=self.on_show_all_bbox_change, font=ctk.CTkFont(size=12))
        self.all_bbox_switch.pack(fill="x", pady=(0, 6))
        
        # Instance Segmentation 옵션 추가
        self.show_inst_seg_var = ctk.BooleanVar(value=False)
        self.inst_seg_switch = ctk.CTkSwitch(bbox_options_section, text="Show Instance Segmentation", 
                                            variable=self.show_inst_seg_var,
                                            command=self.on_bbox_option_change, font=ctk.CTkFont(size=12))
        self.inst_seg_switch.pack(fill="x", pady=(0, 6))
        
        # Clicked Object BBoxes 옵션 추가
        self.show_clicked_object_var = ctk.BooleanVar(value=False)
        self.clicked_object_switch = ctk.CTkSwitch(bbox_options_section, text="Show Clicked Object BBoxes", 
                                                  variable=self.show_clicked_object_var,
                                                  command=self.on_bbox_option_change, font=ctk.CTkFont(size=12))
        self.clicked_object_switch.pack(fill="x", pady=(0, 6))
        
        self.color_var = ctk.BooleanVar(value=True)
        self.color_switch = ctk.CTkSwitch(bbox_options_section, text="Color Mode", variable=self.color_var,
                                         command=self.on_bbox_option_change, font=ctk.CTkFont(size=12))
        self.color_switch.pack(fill="x", pady=(0, 10))
        
        # 선 두께 조절
        ctk.CTkLabel(bbox_options_section, text="Line Width", font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.line_width_slider = ctk.CTkSlider(bbox_options_section, from_=1, to=8, number_of_steps=7,
                                              command=self.on_display_option_change, height=20)
        self.line_width_slider.set(3)
        self.line_width_slider.pack(fill="x", pady=(5, 6))
        
        # Segmentation 투명도 조절
        ctk.CTkLabel(bbox_options_section, text="Segmentation Opacity", font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.seg_opacity_slider = ctk.CTkSlider(bbox_options_section, from_=0.1, to=0.8, number_of_steps=7,
                                               command=self.on_display_option_change, height=20)
        self.seg_opacity_slider.set(0.3)
        self.seg_opacity_slider.pack(fill="x", pady=(5, 10))
        
        # === 오른쪽 이미지 영역 구성 ===
        # 파일명 라벨
        self.filename_label = ctk.CTkLabel(right_panel, text="", 
                                          font=ctk.CTkFont(size=16, weight="bold"))
        self.filename_label.pack(pady=(15, 10))
        
        # 이미지 표시를 위한 프레임
        canvas_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        canvas_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 이미지 캔버스 (크게!)
        self.image_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=0)
        
        # 스크롤바
        v_scrollbar = ctk.CTkScrollbar(canvas_frame, orientation="vertical", command=self.image_canvas.yview)
        h_scrollbar = ctk.CTkScrollbar(canvas_frame, orientation="horizontal", command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # 그리드 레이아웃으로 스크롤바 배치
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # 마우스 클릭 이벤트 바인딩 (segmentation 영역 클릭으로 bbox 찾기)
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
    
    def load_folder_structure(self):
        """폴더 구조 로드"""
        try:
            if not os.path.exists(self.root_path):
                self.show_status(f"Root path not found: {self.root_path}. Using current directory.", "warning")
                self.root_path = "."
            
            env_folders = [item for item in os.listdir(self.root_path) 
                          if os.path.isdir(os.path.join(self.root_path, item))]
            
            if env_folders:
                self.env_combo.configure(values=sorted(env_folders))
                self.env_combo.set(sorted(env_folders)[0])
                self.on_env_selected(sorted(env_folders)[0])
                self.show_status(f"Loaded {len(env_folders)} environments", "info")
            else:
                self.show_status("No environment folders found", "warning")
                
        except Exception as e:
            self.show_status(f"Failed to load folder structure: {str(e)}", "error")
    
    def on_env_selected(self, choice):
        """Environment 선택 시 호출"""
        if not choice:
            return
            
        env_path = os.path.join(self.root_path, choice)
        sections = []
        
        try:
            sections = [item for item in os.listdir(env_path) 
                       if os.path.isdir(os.path.join(env_path, item))]
        except:
            pass
        
        if sections:
            self.section_combo.configure(values=sorted(sections))
            self.section_combo.set(sorted(sections)[0])
            self.on_section_selected(sorted(sections)[0])
        else:
            self.section_combo.configure(values=[])
            self.section_combo.set("")
            self.platform_combo.configure(values=[])
            self.platform_combo.set("")
    
    def on_section_selected(self, choice):
        """Section 선택 시 호출"""
        if not choice:
            return
            
        section_path = os.path.join(self.root_path, self.env_combo.get(), choice)
        platforms = []
        
        try:
            platforms = [item for item in os.listdir(section_path) 
                        if os.path.isdir(os.path.join(section_path, item))]
        except:
            pass
        
        if platforms:
            self.platform_combo.configure(values=sorted(platforms))
            self.platform_combo.set(sorted(platforms)[0])
        else:
            self.platform_combo.configure(values=[])
            self.platform_combo.set("")
    
    def on_platform_selected(self, choice):
        """Platform 선택 시 호출"""
        pass

    def current_dataset_path(self):
        if not all([self.env_combo.get(), self.section_combo.get(), self.platform_combo.get()]):
            return None
        return Path(
            self.root_path,
            self.env_combo.get(),
            self.section_combo.get(),
            self.platform_combo.get(),
        )

    def current_sim_selection(self):
        if not self.current_image_path or not isinstance(self.current_grasp_data, list):
            raise ValueError("먼저 output_grasp가 있는 이미지를 선택하세요.")
        if not 0 <= self.current_bbox_index < len(self.processed_bboxes):
            raise ValueError("재현할 grasp를 선택하세요.")

        scene_num = int(Path(self.current_image_path).stem)
        source_index = int(
            self.processed_bboxes[self.current_bbox_index].get(
                "source_index", self.current_bbox_index
            )
        )
        if not 0 <= source_index < len(self.current_grasp_data):
            raise IndexError(f"output_grasp index가 잘못되었습니다: {source_index}")
        record = self.current_grasp_data[source_index]
        dataset_path = self.current_dataset_path()
        if dataset_path is None:
            raise ValueError("Dataset 경로를 먼저 선택하세요.")
        return dataset_path, scene_num, source_index, record

    def write_sim_command(self, dataset_path, scene_num, grasp_index, record):
        self.sim_request_id += 1
        command = {
            "request_id": self.sim_request_id,
            "root_path": str(dataset_path),
            "scene_num": int(scene_num),
            "grasp_index": int(grasp_index),
            "gripper_model": str(record.get("gripper_model", "")),
            "gripper_type": str(record.get("gripper_type", "")),
        }
        temporary = self.sim_control_file.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(command, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, self.sim_control_file)

    def stop_sim(self, update_ui=True):
        process = self.sim_process
        self.sim_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        self.sim_restart_key = None
        if update_ui and hasattr(self, "sim_status_label"):
            self.sim_status_label.configure(text="SIM: stopped")

    def load_to_sim(self):
        try:
            dataset_path, scene_num, grasp_index, record = self.current_sim_selection()
            restart_key = (
                str(dataset_path),
                int(scene_num),
                str(record.get("gripper_model", "")),
                str(record.get("gripper_type", "")).lower(),
            )
            running = self.sim_process is not None and self.sim_process.poll() is None
            must_restart = not running or restart_key != self.sim_restart_key

            if must_restart:
                self.stop_sim(update_ui=False)
                for path in (self.sim_status_file, self.sim_control_file):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                self.write_sim_command(dataset_path, scene_num, grasp_index, record)

                isaaclab_root = Path(__file__).resolve().parents[2]
                python_path = isaaclab_root / ".venv" / "bin" / "python"
                replay_script = Path(__file__).with_name("Replay_Output_Grasp.py")
                if not python_path.exists():
                    raise FileNotFoundError(f"Isaac Lab Python not found: {python_path}")

                command = [
                    str(python_path),
                    str(replay_script),
                    "--root_path",
                    str(dataset_path),
                    "--scene_num",
                    str(scene_num),
                    "--grasp_index",
                    str(grasp_index),
                    "--control_file",
                    str(self.sim_control_file),
                    "--status_file",
                    str(self.sim_status_file),
                ]
                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"
                self.sim_process = subprocess.Popen(
                    command,
                    cwd=str(isaaclab_root),
                    env=child_env,
                )
                self.sim_restart_key = restart_key
                action_text = "starting"
            else:
                self.write_sim_command(dataset_path, scene_num, grasp_index, record)
                action_text = "switching"

            self.sim_enabled = True
            self.sim_status_label.configure(
                text=f"SIM: {action_text} scene={scene_num:04d}, grasp={grasp_index}"
            )
            self.show_status(
                f"Isaac Lab {action_text}: scene {scene_num:04d}, grasp {grasp_index}",
                "info",
            )
        except Exception as error:
            self.show_status(f"Load to Sim failed: {error}", "error")
            messagebox.showerror("Load to Sim", str(error))

    def schedule_sim_sync(self):
        if not self.sim_enabled:
            return
        if self.sim_sync_after_id is not None:
            self.root.after_cancel(self.sim_sync_after_id)
        self.sim_sync_after_id = self.root.after(120, self._sync_sim_selection)

    def _sync_sim_selection(self):
        self.sim_sync_after_id = None
        if self.sim_enabled:
            self.load_to_sim()

    def poll_sim_status(self):
        try:
            if self.sim_process is not None:
                return_code = self.sim_process.poll()
                if return_code is not None:
                    self.sim_process = None
                    self.sim_restart_key = None
                    self.sim_enabled = False
                    detail = ""
                    if self.sim_status_file.exists():
                        try:
                            detail = json.loads(
                                self.sim_status_file.read_text(encoding="utf-8")
                            ).get("message", "")
                        except Exception:
                            pass
                    self.sim_status_label.configure(
                        text=f"SIM: exited ({return_code}) {detail}".strip()
                    )
                elif self.sim_status_file.exists():
                    stamp = self.sim_status_file.stat().st_mtime_ns
                    if stamp != self.sim_last_status_stamp:
                        self.sim_last_status_stamp = stamp
                        status = json.loads(self.sim_status_file.read_text(encoding="utf-8"))
                        state = status.get("state", "unknown")
                        scene = int(status.get("scene_num", 0))
                        grasp = int(status.get("grasp_index", 0))
                        cycles = int(status.get("cycle_count", 0))
                        matched = status.get("matched_pre_grasp")
                        bbox_matched = status.get("matched_grasp_bbox")
                        if bbox_matched:
                            match_text = "pregrasp+bbox matched"
                        elif matched:
                            match_text = "pregrasp matched, no 3D bbox"
                        else:
                            match_text = "output-only"
                        self.sim_status_label.configure(
                            text=(
                                f"SIM: {state} scene={scene:04d}, grasp={grasp}, "
                                f"loop={cycles}, {match_text}"
                            )
                        )
        except Exception as error:
            self.sim_status_label.configure(text=f"SIM status error: {error}")
        finally:
            try:
                self.root.after(500, self.poll_sim_status)
            except tk.TclError:
                pass

    def on_close(self):
        self.sim_enabled = False
        if self.sim_sync_after_id is not None:
            try:
                self.root.after_cancel(self.sim_sync_after_id)
            except tk.TclError:
                pass
        self.stop_sim(update_ui=False)
        shutil.rmtree(self.sim_runtime_dir, ignore_errors=True)
        self.root.destroy()

    def load_images(self):
        """이미지 로드"""
        if not all([self.env_combo.get(), self.section_combo.get(), self.platform_combo.get()]):
            self.show_status("Please select all paths (Environment, Section, Platform)", "warning")
            return
        
        image_path = os.path.join(
            self.root_path, self.env_combo.get(), self.section_combo.get(), 
            self.platform_combo.get(), "rgb", "top_view_camera"
        )
        
        if not os.path.exists(image_path):
            self.show_status(f"Image path not found: {image_path}", "error")
            return
        
        png_files = glob.glob(os.path.join(image_path, "*.png"))
        
        if not png_files:
            self.show_status("No PNG files found in the specified path", "warning")
            return
        
        self.current_images = sorted(png_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.current_index = 0
        self.current_bbox_index = 0
        
        # 캐시 초기화
        self.image_cache.clear()
        self.grasp_cache.clear()
        self.inst_seg_cache.clear()  # instance segmentation 캐시도 초기화
        
        # 필터링 초기화
        self.filtered_images = []
        self.checked_objects.clear()
        
        # 슬라이더 범위 설정
        if len(self.current_images) > 0:
            max_val = len(self.current_images) - 1
            if max_val > 0:
                self.image_slider.configure(from_=0, to=max_val, number_of_steps=max_val)
            else:
                self.image_slider.configure(from_=0, to=1, number_of_steps=1)
            self.image_slider.set(0)
        else:
            self.image_slider.configure(from_=0, to=1, number_of_steps=1)
            self.image_slider.set(0)
        
        self.show_current_image()
        self.show_status(f"Successfully loaded {len(self.current_images)} images", "info")
    
    def update_image_slider_range(self):
        """이미지 슬라이더 범위 업데이트 (필터링 고려)"""
        active_images = self.get_current_image_list()
        if len(active_images) > 0:
            max_val = len(active_images) - 1
            if max_val > 0:
                self.image_slider.configure(from_=0, to=max_val, number_of_steps=max_val)
            else:
                self.image_slider.configure(from_=0, to=1, number_of_steps=1)
        else:
            self.image_slider.configure(from_=0, to=1, number_of_steps=1)
    
    def load_grasp_data(self, image_filename):
        """grasp JSON 데이터 로드 (캐시 사용)"""
        if image_filename in self.grasp_cache:
            return self.grasp_cache[image_filename]
        
        try:
            base_name = os.path.splitext(image_filename)[0]
            json_filename = f"{base_name}.json"
            
            json_path = os.path.join(
                self.root_path, self.env_combo.get(), self.section_combo.get(), 
                self.platform_combo.get(), "output_grasp", json_filename
            )
            
            if not os.path.exists(json_path):
                self.grasp_cache[image_filename] = None
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.grasp_cache[image_filename] = data
            return data
            
        except Exception as e:
            print(f"JSON load error: {str(e)}")
            self.grasp_cache[image_filename] = None
            return None
    
    def load_object_list(self):
        """물체 목록 JSON 파일에서 로드 (size_rank 정보 포함)"""
        object_info = {}  # {name: {"size_rank": value}} 형태로 저장
        
        json_paths = [
            # "/nas/ochansol/3d_model/peel3_scan_data_2024/objects_conf.json",
            # "/nas/ochansol/3d_model/peel3_scan_data_2025/objects_conf.json",
            "/nas/ochansol/3d_model/peel3_scan_data_2026/objects_conf.json"
        ]
        # csv_path ="/nas/ochansol/3d_model/2024_2025_objects_cat_attr.csv"
        csv_path ="/nas/ochansol/3d_model/2026_objects_cat_attr.csv"
        self.csv_dict = {}

        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV header의 대소문자/공백 차이를 흡수한다.
                normalized_row = {
                    str(key).strip().lower(): str(value).strip()
                    for key, value in row.items()
                    if key is not None and value is not None
                }
                class_name = normalized_row.get("class_name", "")
                object_name = normalized_row.get("object_name", "")
                if not object_name:
                    continue

                # 예전 conf: class="obj_001"
                if class_name:
                    self.csv_dict[class_name] = object_name
                # 현재 conf: class="adjustable_shower_head_holder"
                self.csv_dict[object_name] = object_name

            
        
        for json_path in json_paths:
            try:
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # list 내부의 dict에서 "name"과 "size_rank" 키 추출
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "category" in item:
                                name = item["category"]["object_name"]
                                size_rank = item.get("size_rank", "Unknown")
                                object_info[name] = {"size_rank": size_rank}
                    
                    print(f"Loaded {len(object_info)} objects from {json_path}")
                else:
                    print(f"Object config file not found: {json_path}")
            
            except Exception as e:
                print(f"Error loading object config {json_path}: {str(e)}")
        

        
        self.object_info = object_info  # 전체 정보 저장
        object_list = sorted(list(object_info.keys()))
        print(f"Total unique objects loaded: {len(object_list)}")
        return object_list

    def _on_wheel_generic(self, event, canvas):
        """모든 OS에서 동작하도록 휠 이벤트를 정규화해서 canvas에 스크롤 전달"""
        # Windows / macOS: event.delta 값 사용 (120/-120 단위)
        if event.delta:
            canvas.yview_scroll(int(-event.delta / 120), "units")
            return
        # Linux(X11): Button-4 / Button-5 이벤트 사용
        if hasattr(event, "num"):
            if event.num == 4:  # wheel up
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # wheel down
                canvas.yview_scroll(1, "units")
    def _bind_wheel_recursive(self, root_widget, canvas):
        """root_widget 및 모든 자식 위젯에 휠 이벤트 바인딩 (재귀)"""
        # 공용 람다 (late-binding 문제 피하기 위해 디폴트 인수 사용)
        handler = lambda e, c=canvas: self._on_wheel_generic(e, c)

        # Windows/macOS
        root_widget.bind("<MouseWheel>", handler, add=True)
        # Linux
        root_widget.bind("<Button-4>", handler, add=True)
        root_widget.bind("<Button-5>", handler, add=True)

        for child in root_widget.winfo_children():
            try:
                self._bind_wheel_recursive(child, canvas)
            except Exception:
                pass
    def _norm(self, s: str):
        return str(s).strip()

    def _to_object_name(self, value: str):
        """Class_name과 Object_name 입력을 모두 표준 Object_name으로 변환한다."""
        normalized = self._norm(value)
        return self.csv_dict.get(normalized, normalized)

    def open_object_check_window(self):
        """Object Check 창 열기"""
        if self.object_check_window is not None and self.object_check_window.winfo_exists():
            self.object_check_window.lift()
            return
        
        if not self.current_images:
            self.show_status("Please load images first before using object filter", "warning")
            return
        
        # 실제 데이터에서 target_object 목록 추출
        print("Scanning images for conf objects...")
        actual_objects = self.extract_actual_target_objects(scan_all=True)

        # JSON에서 물체 목록도 로드
        json_objects = self.load_object_list()
        actual_objects = {self._to_object_name(name) for name in actual_objects}

        json_norm = [self._norm(x) for x in json_objects]
        actual_norm = set(self._norm(x) for x in actual_objects)
 

        self.object_list = sorted(set(json_norm) | actual_norm)
        actual_objects_list = sorted([o for o in self.object_list if o in actual_norm])
        json_only_objects   = sorted([o for o in self.object_list if o not in actual_norm])
        
        # 두 목록 합치기 (실제 데이터 우선)
        if actual_objects:
            self.object_list = sorted(list(actual_objects | set(json_objects)))
            print(f"Using actual conf objects: {len(actual_objects)} found in scene data + {len(json_objects)} from JSON")
        else:
            self.object_list = json_objects
            print(f"Using JSON objects only: {len(json_objects)}")
        
        if not self.object_list:
            self.show_status("No objects found in conf data or config files", "warning")
            return
        
        # 새 창 생성
        self.object_check_window = ctk.CTkToplevel(self.root)
        self.object_check_window.title("🔍 Object Check - Select Objects to Filter")
        self.object_check_window.geometry("600x700")
        
        # 메인 프레임
        main_frame = ctk.CTkFrame(self.object_check_window, corner_radius=0)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 상단 제어 패널
        control_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        control_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        # 제목과 설명
        title_label = ctk.CTkLabel(control_frame, text="🔍 Object Filter", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(15, 5))
        
        desc_label = ctk.CTkLabel(control_frame, 
                                 text="Check objects to filter images based on scene conf data\nAND mode: Show images with ALL selected objects in scene\nOR mode: Show images with ANY selected object in scene",
                                 font=ctk.CTkFont(size=11), text_color=("#666666", "#AAAAAA"))
        desc_label.pack(pady=(0, 5))
        
        # Filter Mode 선택
        mode_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        mode_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.filter_mode_var = ctk.StringVar(value="AND")
        
        and_radio = ctk.CTkRadioButton(mode_frame, text="AND (All objects)", variable=self.filter_mode_var, 
                                      value="AND", font=ctk.CTkFont(size=11))
        and_radio.pack(side="left", padx=(0, 20))
        
        or_radio = ctk.CTkRadioButton(mode_frame, text="OR (Any object)", variable=self.filter_mode_var, 
                                     value="OR", font=ctk.CTkFont(size=11))
        or_radio.pack(side="left")
        
        # 실제 데이터 정보 표시
        if actual_objects:
            data_info_label = ctk.CTkLabel(control_frame, 
                                          text=f"✅ {len(actual_objects)} objects found in current scene conf data",
                                          font=ctk.CTkFont(size=10), text_color=("#4CAF50", "#66BB6A"))
            data_info_label.pack(pady=(0, 10))
        
        # 버튼 프레임
        button_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # 전체 선택/해제 버튼
        select_all_btn = ctk.CTkButton(button_frame, text="Select All", command=self.select_all_objects,
                                      width=100, height=30, font=ctk.CTkFont(size=11))
        select_all_btn.pack(side="left", padx=(0, 10))
        
        clear_all_btn = ctk.CTkButton(button_frame, text="Clear All", command=self.clear_all_objects,
                                     width=100, height=30, font=ctk.CTkFont(size=11))
        clear_all_btn.pack(side="left", padx=(0, 10))
        
        # 적용 버튼
        apply_btn = ctk.CTkButton(button_frame, text="Apply Filter", command=self.apply_object_filter,
                                 width=120, height=30, font=ctk.CTkFont(size=12, weight="bold"),
                                 fg_color=("#1f538d", "#14375e"), hover_color=("#14375e", "#1f538d"))
        apply_btn.pack(side="right")
        
        # 상태 라벨
        self.filter_status_label = ctk.CTkLabel(control_frame, text=f"Total objects: {len(self.object_list)} | Selected: 0",
                                               font=ctk.CTkFont(size=11), text_color=("#333333", "#CCCCCC"))
        self.filter_status_label.pack(pady=(0, 10))
        
        # 스크롤 가능한 체크박스 영역
        scroll_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

                
        # 스크롤바가 있는 캔버스
        canvas = tk.Canvas(scroll_frame, bg='#212121', highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(scroll_frame, orientation="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        scrollable_frame.grid_columnconfigure(0, weight=1)   # checkbox column stretches
        scrollable_frame.grid_columnconfigure(1, weight=0) 
        header_font = ctk.CTkFont(size=11, weight="bold")
        ctk.CTkLabel(scrollable_frame, text="object", font=header_font).grid(row=0, column=0, sticky="w", padx=15, pady=(8,4))
        ctk.CTkLabel(scrollable_frame, text="size_rank", font=header_font).grid(row=0, column=1, sticky="e", padx=10, pady=(8,4))
        row_idx = 1
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # ===== 마우스 휠 스크롤 개선된 바인딩 =====

        
        self._bind_wheel_recursive(canvas, canvas)                 # 캔버스 자신
        self._bind_wheel_recursive(scrollable_frame, canvas)       # 스크롤되는 내부 프레임
        self._bind_wheel_recursive(self.object_check_window, canvas)  # 창 전체

        # (선택) 포커스 강제 (일부 WM에서 필요)
        scrollable_frame.focus_set()
        
        # 캔버스와 스크롤바 배치
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # 체크박스 변수들 저장
        self.object_checkboxes = {}
        
        # 체크박스 생성 (실제 데이터에 있는 객체들을 상단에 표시)
        actual_objects_list = sorted(list(actual_objects)) if actual_objects else []
        json_only_objects = sorted(list(set(json_objects) - actual_objects)) if actual_objects else []
        
        # 실제 데이터의 객체들 먼저 표시
        if actual_objects_list:
        # section title spans 2 columns
            title1 = ctk.CTkLabel(scrollable_frame, text="🎯 objects in current scene conf data", 
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=("#4CAF50", "#66BB6A"))
            title1.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=15, pady=(10,5))
            row_idx += 1

            for obj_name in actual_objects_list:
                var = ctk.BooleanVar(value=obj_name in self.checked_objects)
                cb = ctk.CTkCheckBox(
                    scrollable_frame,
                    text=obj_name,
                    variable=var,
                    command=lambda name=obj_name: self.on_object_checkbox_change(name),
                    font=ctk.CTkFont(size=11)
                )
                cb.grid(row=row_idx, column=0, sticky="w", padx=25, pady=1)

                size_lbl = ctk.CTkLabel(scrollable_frame, text=self.get_size_rank(obj_name), font=ctk.CTkFont(size=11))
                size_lbl.grid(row=row_idx, column=1, sticky="e", padx=10, pady=1)

                self.object_checkboxes[obj_name] = var
                row_idx += 1
        
        # JSON에만 있는 객체들 표시
        if json_only_objects:
            if actual_objects_list:
                sep = ctk.CTkFrame(scrollable_frame, height=2, fg_color=("#CCCCCC", "#333333"))
                sep.grid(row=row_idx, column=0, columnspan=2, sticky="ew", padx=15, pady=10)
                row_idx += 1

            title2 = ctk.CTkLabel(scrollable_frame, text="📋 additional objects from config",
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=("#666666", "#AAAAAA"))
            title2.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=15, pady=(5,5))
            row_idx += 1

            for obj_name in json_only_objects:
                var = ctk.BooleanVar(value=obj_name in self.checked_objects)
                cb = ctk.CTkCheckBox(
                    scrollable_frame,
                    text=obj_name,
                    variable=var,
                    command=lambda name=obj_name: self.on_object_checkbox_change(name),
                    font=ctk.CTkFont(size=11),
                    text_color=("#888888", "#AAAAAA")
                )
                cb.grid(row=row_idx, column=0, sticky="w", padx=25, pady=1)

                size_lbl = ctk.CTkLabel(scrollable_frame, text=self.get_size_rank(obj_name),
                                        font=ctk.CTkFont(size=11),
                                        text_color=("#888888", "#AAAAAA"))
                size_lbl.grid(row=row_idx, column=1, sticky="e", padx=10, pady=1)

                self.object_checkboxes[obj_name] = var
                row_idx += 1
        
        # 모든 자식 위젯에 마우스 휠 이벤트 바인딩 (재귀적으로)

        
        # 창 닫기 이벤트 바인딩
        self.object_check_window.protocol("WM_DELETE_WINDOW", self.close_object_check_window)
    def get_size_rank(self, name: str):
        return self.object_info.get(name, {}).get("size_rank", "Unknown")
    
    def extract_actual_target_objects(self, scan_all: bool = True):
        """return set of object names found in conf data.
        scan_all=True -> check every image; False -> first 50 (faster)
        """
        actual_objects = set()
        imgs = self.current_images if scan_all else (self.current_images[:50] if len(self.current_images) > 50 else self.current_images)
        for image_path in imgs:
            filename = os.path.basename(image_path)
            conf_objs = self.load_conf_data(filename)

            if conf_objs:
                for n in conf_objs:
                    actual_objects.add(self._norm(n))

        return actual_objects
    
    def select_all_objects(self):
        """모든 물체 선택"""
        for obj_name, var in self.object_checkboxes.items():
            var.set(True)
        self.checked_objects = set(self.object_list)
        self.update_filter_status()
    
    def clear_all_objects(self):
        """모든 물체 선택 해제"""
        for obj_name, var in self.object_checkboxes.items():
            var.set(False)
        self.checked_objects.clear()
        self.update_filter_status()
    
    def on_object_checkbox_change(self, obj_name):
        """체크박스 상태 변경 시"""
        if self.object_checkboxes[obj_name].get():
            self.checked_objects.add(obj_name)
        else:
            self.checked_objects.discard(obj_name)
        self.update_filter_status()
    
    def update_filter_status(self):
        """필터 상태 업데이트"""
        if hasattr(self, 'filter_status_label'):
            self.filter_status_label.configure(
                text=f"Total objects: {len(self.object_list)} | Selected: {len(self.checked_objects)}"
            )
    
    def load_conf_data(self, image_filename):
        """이미지별 conf 데이터 로드 (씬에 존재하는 물체 목록)"""
        try:
            base_name = os.path.splitext(image_filename)[0]
            
            # conf 파일 경로들 시도
            possible_paths = [
                # 각 이미지별 conf 파일
                os.path.join(self.root_path, self.env_combo.get(), self.section_combo.get(), 
                           self.platform_combo.get(), "conf", f"{base_name}.json"),
                # 씬 전체 conf 파일
                os.path.join(self.root_path, self.env_combo.get(), self.section_combo.get(), 
                           self.platform_combo.get(), "conf", "scene_objects.json"),
                # 다른 가능한 위치들
                os.path.join(self.root_path, self.env_combo.get(), self.section_combo.get(), 
                           self.platform_combo.get(), "objects_conf.json"),
            ]
            
            for conf_path in possible_paths:
                if os.path.exists(conf_path):
                    with open(conf_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 물체 이름들 추출
                    object_names = set()
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # 다양한 키 이름 시도
                                for key in ['name', 'object_name', 'class', 'label']:
                                    if key in item and item[key]:
                                        object_names.add(item[key])
                                        break
                    elif isinstance(data, dict):
                        # dict 형태의 경우
                        if 'objects' in data:
                            for obj in data['objects']:
                                if isinstance(obj, dict):
                                    for key in ['name', 'object_name', 'class', 'label']:
                                        if key in obj and obj[key]:
                                            object_names.add(obj[key])
                                            break
                                elif isinstance(obj, str):
                                    object_names.add(obj)
                        else:
                            # 직접 물체 이름들이 있는 경우
                            for key, value in data.items():
                                if isinstance(value, str):
                                    object_names.add(value)
                    
                    if object_names:
                        print(f"Found conf data for {image_filename}: {list(object_names)[:5]}...")
                        return object_names
            
            # conf 데이터가 없으면 None 반환
            return None
            
        except Exception as e:
            print(f"Conf data load error for {image_filename}: {str(e)}")
            return None

    def apply_object_filter(self):
        """선택된 물체들로 데이터 필터링 적용 (conf 데이터 기준)"""
        if not self.current_images:
            self.show_status("Please load images first", "warning")
            return
        
        if not self.checked_objects:
            # 체크된 물체가 없으면 필터 해제
            self.filtered_images = []
            self.current_index = 0
            self.update_image_slider_range()
            self.show_current_image()
            self.show_status("Filter cleared. Showing all images", "info")
            # Object Check 창은 닫지 않음!
            return
        
        # 필터링된 이미지 찾기
        filtered_paths = []
        checked_list = list(self.checked_objects)
        filter_mode = getattr(self, 'filter_mode_var', None)
        is_or_mode = filter_mode and filter_mode.get() == "OR"
        
        # 디버깅 정보
        all_conf_objects = set()
        images_with_conf = 0
        images_without_conf = 0
        
        print(f"\n=== Object Filter Debug (Conf Data Based) ===")
        print(f"Looking for objects: {checked_list}")
        print(f"Filter mode: {'OR (Any object)' if is_or_mode else 'AND (All objects)'}")
        print(f"Total images to check: {len(self.current_images)}")
        
        for i, image_path in enumerate(self.current_images):
            filename = os.path.basename(image_path)
            
            # 해당 이미지의 conf 데이터 로드 (씬에 존재하는 물체들)
            loaded_conf_objects = self.load_conf_data(filename) or set()
            conf_objects = {
                self._to_object_name(name)
                for name in loaded_conf_objects
            }
            
            if not conf_objects:
                images_without_conf += 1
                # conf 데이터가 없으면 제외
                continue
            
            images_with_conf += 1
            all_conf_objects.update(conf_objects)
            
            # 디버깅: 처음 몇 개 이미지의 정보 출력
            if i < 5:
                print(f"Image {filename}: conf objects = {list(conf_objects)[:5]}...")
            
            # 필터링 조건 확인 (conf 데이터 기준)
            if is_or_mode:
                # OR 모드: 선택한 물체 중 하나라도 씬에 존재하면 포함
                if self.checked_objects & conf_objects:  # 교집합이 있으면
                    filtered_paths.append(image_path)
                    if len(filtered_paths) <= 3:  # 처음 몇 개만 출력
                        matched_objects = list(self.checked_objects & conf_objects)
                        print(f"✅ OR Match: {filename} scene contains {matched_objects}")
            else:
                # AND 모드: 선택한 모든 물체가 씬에 존재해야 포함
                if self.checked_objects.issubset(conf_objects):
                    filtered_paths.append(image_path)
                    if len(filtered_paths) <= 3:  # 처음 몇 개만 출력
                        print(f"✅ AND Match: {filename} scene contains all {list(self.checked_objects)}")
        
        self.filtered_images = filtered_paths
        
        print(f"\nFilter Results:")
        print(f"- Images with conf data: {images_with_conf}/{len(self.current_images)}")
        print(f"- Images without conf data: {images_without_conf}/{len(self.current_images)}")
        print(f"- Unique conf objects found: {len(all_conf_objects)}")
        print(f"- Sample conf objects: {list(sorted(all_conf_objects))[:10]}")
        print(f"- Filtered images ({filter_mode.get() if filter_mode else 'AND'} mode): {len(self.filtered_images)}")
        print(f"=== End Debug ===\n")
        
        if self.filtered_images:
            self.current_index = 0
            self.update_image_slider_range()
            self.show_current_image()
            
            mode_text = "OR (Any object)" if is_or_mode else "AND (All objects)"
            
            self.show_status(
                f"Filter applied: {len(self.filtered_images)} images found with {mode_text} mode. "
                f"Objects: {', '.join(checked_list)}", "info"
            )
        else:
            # 부분 매치 제안 (OR 모드에서만)
            mode_text = "OR (Any object)" if is_or_mode else "AND (All objects)"
            suggestion = ""
            if not is_or_mode:
                suggestion = " Try 'OR (Any object)' mode."
            
            self.show_status(
                f"No images found using {mode_text} mode for objects: {', '.join(checked_list)}.{suggestion}", 
                "warning"
            )
        
        # Object Check 창은 닫지 않음!
    
    def clear_object_filter(self):
        """Object 필터 해제"""
        self.filtered_images = []
        self.checked_objects.clear()
        self.clicked_target_object = None
        
        if self.current_images:
            self.current_index = 0
            self.update_image_slider_range()
            self.show_current_image()
            self.show_status(f"Filter cleared. Now showing all {len(self.current_images)} images", "info")
        else:
            self.show_status("Filter cleared. No images loaded", "info")
    def get_size_rank(self, name: str):
        """return size_rank string from self.object_info if exists"""
        try:
            return self.object_info.get(name, {}).get("size_rank", "Unknown")
        except Exception:
            return "Unknown"
    def close_object_check_window(self):
        """Object Check 창 닫기"""
        if self.object_check_window:
            self.object_check_window.destroy()
            self.object_check_window = None
    
    def load_inst_seg_data(self, image_filename):
        """instance segmentation 데이터 로드 (캐시 사용)"""
        if image_filename in self.inst_seg_cache:
            return self.inst_seg_cache[image_filename]
        
        try:
            base_name = os.path.splitext(image_filename)[0]
            
            # PNG 파일 경로
            png_filename = f"{base_name}.png"
            png_path = os.path.join(
                self.root_path, self.env_combo.get(), self.section_combo.get(), 
                self.platform_combo.get(), "inst_seg", "top_view_camera", png_filename
            )
            
            # JSON 파일 경로
            json_filename = f"semantics_mapping_{base_name}.json"
            json_path = os.path.join(
                self.root_path, self.env_combo.get(), self.section_combo.get(), 
                self.platform_combo.get(), "inst_seg", "top_view_camera", json_filename
            )
            
            if not os.path.exists(png_path) or not os.path.exists(json_path):
                self.inst_seg_cache[image_filename] = None
                return None
            
            # PNG 이미지 로드
            seg_image = Image.open(png_path).convert('RGBA')
            
            # JSON 매핑 로드
            with open(json_path, 'r', encoding='utf-8') as f:
                color_mapping = json.load(f)
            
            # 색상 문자열을 튜플로 변환
            processed_mapping = {}
            for color_str, obj_info in color_mapping.items():
                # "(r, g, b, a)" 형태의 문자열을 튜플로 변환
                color_tuple = eval(color_str)
                processed_mapping[color_tuple] = obj_info
            
            result = {
                'seg_image': seg_image,
                'color_mapping': processed_mapping
            }
            
            self.inst_seg_cache[image_filename] = result
            return result
            
        except Exception as e:
            print(f"Instance segmentation load error: {str(e)}")
            self.inst_seg_cache[image_filename] = None
            return None
    
    def load_cached_image(self, image_path):
        """이미지 로드 (캐시 사용)"""
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        try:
            image = Image.open(image_path)
            if len(self.image_cache) > 15:  # 캐시 크기 증가
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            
            self.image_cache[image_path] = image
            return image
        except Exception as e:
            print(f"Image load error: {str(e)}")
            return None
    
    def process_grasp_data(self, grasp_data):
        """grasp 데이터 처리하여 bbox 생성"""
        processed_bboxes = []
        
        if not grasp_data or not isinstance(grasp_data, list):
            return processed_bboxes
        
        for source_index, grasp in enumerate(grasp_data):
            try:
                bbox_2d_data = grasp["bbox_2d"]
                center = np.array(bbox_2d_data["center"])
                width = bbox_2d_data["width"]
                height = bbox_2d_data["height"]
                angle = bbox_2d_data["angle"]
                # angle = (bbox_2d_data["angle"]) /np.pi*180
                gripper_type = grasp["gripper_type"]
                
                # gripper_model 정보 추가
                gripper_model = grasp.get("gripper_model", "Unknown")
                target_object = grasp.get("target_object", "Unknown")  # target_object 정보 추가
                
                if gripper_type in ["finger2_parallel", "finger2"]:
                    bbox = rot_z(angle).dot(np.array([[-width/2, -height/2],
                                                     [-width/2, +height/2],
                                                     [+width/2, +height/2],
                                                     [+width/2, -height/2]]).T).T + center
                elif gripper_type in ["finger3", "finger3_parallel", "Hand", "hand"]:
                    bbox = np.array(bbox_2d_data["bbox"])
                else:
                    continue
                # bbox = bbox_2d_data["bbox"]
                
                processed_bboxes.append({
                    'bbox': bbox,
                    'gripper_type': gripper_type,
                    'gripper_model': gripper_model,
                    'target_object': target_object,  # target_object 정보 추가
                    'center': center,
                    'width': width,
                    'height': height,
                    'angle': angle,
                    'source_index': source_index,
                })
                
            except Exception as e:
                print(f"Grasp data processing error: {str(e)}")
                continue
        
        return processed_bboxes
    
    def draw_bbox_on_image(self, image, bbox, gripper_type="finger2", color=True, line_width=3):
        """이미지에 bbox 그리기 (finger3는 2,3 꼭지점 제외)"""
        draw = ImageDraw.Draw(image)
        
        # 색상 설정
        if color:
            line1_color = "#1E90FF"  # 도지블루 
            line2_color = "#FF4500"  # 오렌지레드
        else:
            line1_color = "#FF1493"  # 딥핑크
            line2_color = "#00FF7F"  # 스프링그린
        
        if gripper_type in ["finger2", "finger2_parallel"]:
            bbox_array = np.array(bbox)
            
            # [1,2] 연결, [3,0] 연결 - line1_color
            draw.line([tuple(bbox_array[1]), tuple(bbox_array[2])], fill=line1_color, width=line_width)
            draw.line([tuple(bbox_array[3]), tuple(bbox_array[0])], fill=line1_color, width=line_width)
            
            # [0,1] 연결, [2,3] 연결 - line2_color
            draw.line([tuple(bbox_array[0]), tuple(bbox_array[1])], fill=line2_color, width=line_width)
            draw.line([tuple(bbox_array[2]), tuple(bbox_array[3])], fill=line2_color, width=line_width)
        
        elif gripper_type in ["finger3", "finger3_parallel", "Hand", "hand"]:
            if len(bbox.shape) == 3:  # (N, 4, 2)
                for i in range(len(bbox)):
                    bbox_single = bbox[i]
                    
                    # finger3는 2,3 꼭지점 연결 제외 - [1,2] 연결과 [3,0] 연결만
                    draw.line([tuple(bbox_single[1]), tuple(bbox_single[2])], fill=line1_color, width=line_width)
                    draw.line([tuple(bbox_single[3]), tuple(bbox_single[0])], fill=line1_color, width=line_width)
                    
                    # [0,1] 연결만 그리기 (2,3 연결 제외)
                    draw.line([tuple(bbox_single[0]), tuple(bbox_single[1])], fill=line2_color, width=line_width)
            else:  # (4, 2)
                bbox_array = np.array(bbox)
                
                # finger3는 2,3 꼭지점 연결 제외 - [1,2] 연결과 [3,0] 연결만
                draw.line([tuple(bbox_array[1]), tuple(bbox_array[2])], fill=line1_color, width=line_width)
                draw.line([tuple(bbox_array[3]), tuple(bbox_array[0])], fill=line1_color, width=line_width)
                
                # [0,1] 연결만 그리기 (2,3 연결 제외)
                draw.line([tuple(bbox_array[0]), tuple(bbox_array[1])], fill=line2_color, width=line_width)
    
    def create_segmentation_mask(self, inst_seg_data, target_object, opacity=0.3):
        """특정 target_object에 해당하는 segmentation 마스크 생성"""
        if not inst_seg_data:
            return None
        
        try:
            seg_image = inst_seg_data['seg_image']
            color_mapping = inst_seg_data['color_mapping']
            
            # target_object에 해당하는 색상 찾기
            target_colors = []
            for color_tuple, obj_info in color_mapping.items():
                if obj_info.get('class', '') == target_object:
                    target_colors.append(color_tuple)
            
            if not target_colors:
                return None
            
            # 원본 이미지와 같은 크기의 투명 마스크 생성
            mask = Image.new('RGBA', seg_image.size, (0, 0, 0, 0))
            seg_array = np.array(seg_image)
            mask_array = np.array(mask)
            
            # target_object에 해당하는 픽셀들만 마스크에 추가
            for color_tuple in target_colors:
                r, g, b, a = color_tuple
                # 해당 색상과 일치하는 픽셀 찾기
                matches = np.all(seg_array == [r, g, b, a], axis=2)
                if np.any(matches):
                    # 반투명한 색상으로 마스크 설정 (원래 색상 유지, 투명도 조절)
                    mask_array[matches] = [r, g, b, int(255 * opacity)]
            
            return Image.fromarray(mask_array, 'RGBA')
            
        except Exception as e:
            print(f"Segmentation mask creation error: {str(e)}")
            return None
    
    def get_current_image_list(self):
        """현재 사용할 이미지 목록 반환 (필터링된 목록 또는 전체 목록)"""
        return self.filtered_images if self.filtered_images else self.current_images
    
    def show_current_image(self):
        """현재 인덱스의 이미지 표시"""
        active_images = self.get_current_image_list()
        if not active_images:
            return
        
        try:
            self.current_image_path = active_images[self.current_index]
            filename = os.path.basename(self.current_image_path)
            
            # grasp 데이터 로드 및 처리 (실패해도 이미지는 표시)
            grasp_data_available = False
            try:
                self.current_grasp_data = self.load_grasp_data(filename)
                self.processed_bboxes = self.process_grasp_data(self.current_grasp_data)
                grasp_data_available = self.current_grasp_data is not None
            except Exception as e:
                print(f"BBox data loading failed: {str(e)}")
                self.current_grasp_data = None
                self.processed_bboxes = []
                grasp_data_available = False
            
            # instance segmentation 데이터 로드
            try:
                self.current_inst_seg_data = self.load_inst_seg_data(filename)
            except Exception as e:
                print(f"Instance segmentation data loading failed: {str(e)}")
                self.current_inst_seg_data = None
            
            self.current_bbox_index = 0
            
            # BBox 슬라이더 범위 설정
            if len(self.processed_bboxes) > 0:
                max_bbox = len(self.processed_bboxes) - 1
                if max_bbox > 0:
                    self.bbox_slider.configure(from_=0, to=max_bbox, number_of_steps=max_bbox)
                else:
                    self.bbox_slider.configure(from_=0, to=1, number_of_steps=1)
                self.bbox_slider.set(0)
            else:
                self.bbox_slider.configure(from_=0, to=1, number_of_steps=1)
                self.bbox_slider.set(0)
            
            # 이미지 슬라이더 업데이트
            if not self.is_updating_scale:
                self.is_updating_scale = True
                self.image_slider.set(self.current_index)
                self.is_updating_scale = False
            
            # 이미지 표시 (항상 실행)
            self.refresh_image()
            self.schedule_sim_sync()
            
            # 정보 업데이트 (필터링 상태 표시)
            active_images = self.get_current_image_list()
            if self.filtered_images:
                filter_info = f" [🔍 FILTERED: {len(self.filtered_images)}/{len(self.current_images)}]"
                if self.checked_objects:
                    filter_objects = ', '.join(list(self.checked_objects)[:3])
                    if len(self.checked_objects) > 3:
                        filter_objects += f" +{len(self.checked_objects)-3} more"
                    filter_info += f" Objects: {filter_objects}"
            else:
                filter_info = f" [📷 ALL IMAGES]"
            
            # grasp 데이터 상태 표시
            data_status = " [✅ Grasp Data]" if grasp_data_available else " [❌ No Grasp Data]"
            
            info_text = f"📷 {self.current_index + 1} / {len(active_images)}{filter_info}{data_status} - {filename}"
            self.info_label.configure(text=info_text)
            
        except Exception as e:
            # 이미지 로드에 실패했을 때 에러 메시지만 표시
            print(f"Image loading error: {str(e)}")
            self.show_status(f"Failed to load image: {str(e)}", "error")
    
    def refresh_image(self):
        """이미지 새로고침 (큰 화면에 최적화)"""
        if not self.current_image_path:
            return
        
        try:
            # 캐시된 이미지 로드
            base_image = self.load_cached_image(self.current_image_path)
            if base_image is None:
                return
            
            # 이미지 복사 (원본 보존)
            self.display_image = base_image.copy()
            
            # 파일명 표시
            filename = os.path.basename(self.current_image_path)
            self.filename_label.configure(text=f"📁 {filename}")
            
            # 현재 선 두께 가져오기
            current_line_width = int(self.line_width_slider.get())
            
            # Segmentation 오버레이 적용 (bbox 그리기 전에)
            if self.show_inst_seg_var.get() and hasattr(self, 'current_inst_seg_data') and self.current_inst_seg_data:
                try:
                    opacity = self.seg_opacity_slider.get()
                    if self.show_bbox_var.get() and self.processed_bboxes:
                        if self.show_all_bbox_var.get():
                            # 모든 bbox의 target_object에 대한 segmentation 표시 (최우선순위, 최적화: unique objects만)
                            unique_target_objects = set()
                            for bbox_data in self.processed_bboxes:
                                target_object = bbox_data.get('target_object', '')
                                if target_object and target_object != 'Unknown':
                                    unique_target_objects.add(target_object)
                            
                            print(f"Show All Segmentation: {len(unique_target_objects)} unique objects (optimization applied)")
                            
                            # 각 unique target_object에 대해 한 번씩만 segmentation 생성
                            for target_object in unique_target_objects:
                                seg_mask = self.create_segmentation_mask(
                                    self.current_inst_seg_data, target_object, opacity
                                )
                                if seg_mask:
                                    self.display_image = Image.alpha_composite(
                                        self.display_image.convert('RGBA'), seg_mask
                                    ).convert('RGB')
                        elif self.show_clicked_object_var.get() and self.clicked_target_object:
                            # 클릭한 물체의 segmentation 표시
                            seg_mask = self.create_segmentation_mask(
                                self.current_inst_seg_data, self.clicked_target_object, opacity
                            )
                            if seg_mask:
                                self.display_image = Image.alpha_composite(
                                    self.display_image.convert('RGBA'), seg_mask
                                ).convert('RGB')
                        else:
                            # 현재 bbox의 target_object에 대한 segmentation만 표시
                            if 0 <= self.current_bbox_index < len(self.processed_bboxes):
                                bbox_data = self.processed_bboxes[self.current_bbox_index]
                                target_object = bbox_data.get('target_object', '')
                                if target_object and target_object != 'Unknown':
                                    seg_mask = self.create_segmentation_mask(
                                        self.current_inst_seg_data, target_object, opacity
                                    )
                                    if seg_mask:
                                        self.display_image = Image.alpha_composite(
                                            self.display_image.convert('RGBA'), seg_mask
                                        ).convert('RGB')
                except Exception as e:
                    print(f"Segmentation overlay error: {str(e)}")
            
            # BBox 그리기
            bbox_info = "🎯 "
            model_info = ""
            if self.show_bbox_var.get() and self.processed_bboxes:
                if self.show_all_bbox_var.get():
                    # 모든 bbox 표시 (최우선순위)
                    for i, bbox_data in enumerate(self.processed_bboxes):
                        color_mode = self.color_var.get()
                        self.draw_bbox_on_image(self.display_image, bbox_data['bbox'], 
                                               bbox_data['gripper_type'], color_mode, current_line_width)
                    bbox_info += f"All BBoxes: {len(self.processed_bboxes)} displayed"
                    model_info = "💡 Multiple models displayed"
                elif self.show_clicked_object_var.get() and self.clicked_target_object:
                    # 클릭한 물체의 모든 bbox 표시
                    clicked_bbox_indices = self.find_all_bboxes_by_target_object(self.clicked_target_object)
                    for bbox_idx in clicked_bbox_indices:
                        bbox_data = self.processed_bboxes[bbox_idx]
                        color_mode = self.color_var.get()
                        self.draw_bbox_on_image(self.display_image, bbox_data['bbox'], 
                                               bbox_data['gripper_type'], color_mode, current_line_width)
                    bbox_info += f"'{self.clicked_target_object}' BBoxes: {len(clicked_bbox_indices)} displayed"
                    if 0 <= self.current_bbox_index < len(self.processed_bboxes):
                        current_bbox = self.processed_bboxes[self.current_bbox_index]
                        model_info = f"🤖 Model: {current_bbox['gripper_model']}\n🎯 Target: {self.clicked_target_object}"
                else:
                    # 현재 bbox만 표시
                    if 0 <= self.current_bbox_index < len(self.processed_bboxes):
                        bbox_data = self.processed_bboxes[self.current_bbox_index]
                        color_mode = self.color_var.get()
                        self.draw_bbox_on_image(self.display_image, bbox_data['bbox'], 
                                               bbox_data['gripper_type'], color_mode, current_line_width)
                        bbox_info += f"BBox {self.current_bbox_index + 1}/{len(self.processed_bboxes)} ({bbox_data['gripper_type']})"
                        # target_object 정보도 표시
                        target_obj = bbox_data.get('target_object', 'Unknown')
                        model_info = f"🤖 Model: {bbox_data['gripper_model']}\n🎯 Target: {target_obj}"
                    else:
                        bbox_info += "No BBox available"
                        model_info = ""
            elif not self.show_bbox_var.get():
                bbox_info += "BBox display disabled"
                model_info = ""
            else:
                bbox_info += "No BBox data found"
                model_info = ""
            
            # bbox 정보가 없어도 이미지는 표시
            self.bbox_info_label.configure(text=bbox_info)
            self.bbox_model_label.configure(text=model_info)
            
            # 캔버스 크기에 맞게 이미지 조정 (더 큰 화면 활용)
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 100 and canvas_height > 100:  # 최소 크기 확인
                # 이미지 리사이즈 (종횡비 유지, 여백 최소화)
                img_ratio = self.display_image.width / self.display_image.height
                canvas_ratio = canvas_width / canvas_height
                
                # 여백을 20픽셀로 줄여서 이미지를 더 크게 표시
                if img_ratio > canvas_ratio:
                    new_width = min(canvas_width - 20, self.display_image.width)
                    new_height = int(new_width / img_ratio)
                else:
                    new_height = min(canvas_height - 20, self.display_image.height)
                    new_width = int(new_height * img_ratio)
                
                # 너무 작은 경우 최소 크기 보장
                min_size = 400
                if new_width < min_size or new_height < min_size:
                    if img_ratio > 1:  # 가로가 더 긴 경우
                        new_width = min_size
                        new_height = int(min_size / img_ratio)
                    else:  # 세로가 더 긴 경우
                        new_height = min_size
                        new_width = int(min_size * img_ratio)
                
                # 고품질 리사이즈
                display_resized = self.display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                display_resized = self.display_image
            
            # PhotoImage로 변환
            self.display_photo = ImageTk.PhotoImage(display_resized)
            
            # 캔버스에 이미지 표시
            self.image_canvas.delete("all")
            canvas_center_x = self.image_canvas.winfo_width() // 2
            canvas_center_y = self.image_canvas.winfo_height() // 2
            
            self.image_canvas.create_image(canvas_center_x, canvas_center_y, 
                                          image=self.display_photo, anchor=tk.CENTER)
            
            # 스크롤 영역 설정
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            # 에러가 발생해도 기본 이미지는 표시하려고 시도
            try:
                base_image = self.load_cached_image(self.current_image_path)
                if base_image is not None:
                    self.display_image = base_image.copy()
                    self.display_photo = ImageTk.PhotoImage(self.display_image)
                    self.image_canvas.delete("all")
                    self.image_canvas.create_image(self.image_canvas.winfo_width() // 2, 
                                                  self.image_canvas.winfo_height() // 2, 
                                                  image=self.display_photo, anchor=tk.CENTER)
                    self.bbox_info_label.configure(text="🎯 Error loading BBox data, showing image only")
                    self.bbox_model_label.configure(text="")
            except:
                self.show_status(f"Failed to display image: {str(e)}", "error")
    
    # 슬라이더 이벤트 핸들러들
    def on_image_slider_change(self, value):
        """이미지 슬라이더 변경 시 호출"""
        active_images = self.get_current_image_list()
        if self.is_updating_scale or not active_images:
            return
        
        try:
            new_index = int(value)
            if 0 <= new_index < len(active_images) and new_index != self.current_index:
                self.current_index = new_index
                self.show_current_image()
        except (ValueError, IndexError):
            pass
    
    def on_bbox_slider_change(self, value):
        """BBox 슬라이더 변경 시 호출"""
        if not self.processed_bboxes:
            return
        
        try:
            new_index = int(value)
            if 0 <= new_index < len(self.processed_bboxes) and new_index != self.current_bbox_index:
                # Clicked Object BBoxes 모드에서 슬라이더 사용 시 개별 모드로 전환
                if self.show_clicked_object_var.get():
                    self.show_clicked_object_var.set(False)
                    self.clicked_target_object = None
                    print("Switched to individual bbox mode")
                
                self.current_bbox_index = new_index
                self.refresh_image()
                self.schedule_sim_sync()
        except (ValueError, IndexError):
            pass
    
    def on_bbox_option_change(self):
        """BBox 옵션 변경 시 호출"""
        self.refresh_image()
    
    def on_display_option_change(self, value=None):
        """표시 옵션 변경 시 호출 (선 두께, segmentation 투명도 등)"""
        if hasattr(self, 'processed_bboxes'):
            self.refresh_image()
    
    # 네비게이션 메서드들
    def prev_image(self):
        """이전 이미지"""
        active_images = self.get_current_image_list()
        if active_images and self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def next_image(self):
        """다음 이미지"""
        active_images = self.get_current_image_list()
        if active_images and self.current_index < len(active_images) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def prev_bbox(self):
        """이전 BBox"""
        if self.processed_bboxes and self.current_bbox_index > 0:
            # Clicked Object BBoxes 모드에서 버튼 사용 시 개별 모드로 전환
            if self.show_clicked_object_var.get():
                self.show_clicked_object_var.set(False)
                self.clicked_target_object = None
                print("Switched to individual bbox mode")
            
            self.current_bbox_index -= 1
            if len(self.processed_bboxes) > 1:  # 슬라이더 업데이트는 2개 이상일 때만
                self.bbox_slider.set(self.current_bbox_index)
            self.refresh_image()
            self.schedule_sim_sync()
    
    def next_bbox(self):
        """다음 BBox"""
        if self.processed_bboxes and self.current_bbox_index < len(self.processed_bboxes) - 1:
            # Clicked Object BBoxes 모드에서 버튼 사용 시 개별 모드로 전환
            if self.show_clicked_object_var.get():
                self.show_clicked_object_var.set(False)
                self.clicked_target_object = None
                print("Switched to individual bbox mode")
            
            self.current_bbox_index += 1
            if len(self.processed_bboxes) > 1:  # 슬라이더 업데이트는 2개 이상일 때만
                self.bbox_slider.set(self.current_bbox_index)
            self.refresh_image()
            self.schedule_sim_sync()
    
    def on_show_all_bbox_change(self):
        """Show All BBoxes 변경 시 호출 - 성능을 위해 segmentation 자동 비활성화"""
        if self.show_all_bbox_var.get():
            # Show All BBoxes가 활성화되면 segmentation 비활성화
            if self.show_inst_seg_var.get():
                self.show_inst_seg_var.set(False)
                print("Performance optimization: Segmentation disabled when showing all bboxes")
        self.refresh_image()
    
    def on_canvas_click(self, event):
        """캔버스 클릭 시 해당 위치의 segmentation 객체에 해당하는 bbox 찾기"""
        if not hasattr(self, 'current_inst_seg_data') or not self.current_inst_seg_data:
            return
        if not self.processed_bboxes:
            return
        
        try:
            # 클릭 좌표를 실제 이미지 좌표로 변환
            image_coords = self.canvas_to_image_coords(event.x, event.y)
            if not image_coords:
                return
            
            img_x, img_y = image_coords
            
            # 해당 좌표의 segmentation 색상 확인
            clicked_object = self.get_object_at_coords(img_x, img_y)
            if not clicked_object or clicked_object in ['BACKGROUND', 'UNLABELLED']:
                return
            
            # 해당 object를 target_object로 가지는 모든 bbox 찾기
            bbox_indices = self.find_all_bboxes_by_target_object(clicked_object)
            if bbox_indices:
                # 클릭한 물체 설정
                self.clicked_target_object = clicked_object
                
                # Show All BBoxes와 Single BBox 모드 비활성화
                self.show_all_bbox_var.set(False)
                
                # Clicked Object BBoxes 모드 활성화
                self.show_clicked_object_var.set(True)
                
                # 첫 번째 bbox로 이동 (정보 표시용)
                self.current_bbox_index = bbox_indices[0]
                self.bbox_slider.set(bbox_indices[0])
                
                # Segmentation 활성화 (클릭한 객체 표시)
                if not self.show_inst_seg_var.get():
                    self.show_inst_seg_var.set(True)
                
                self.refresh_image()
                self.schedule_sim_sync()
                print(f"Found {len(bbox_indices)} bboxes for '{clicked_object}': {bbox_indices}")
            else:
                print(f"No bbox found for object '{clicked_object}'")
                
        except Exception as e:
            print(f"Canvas click error: {str(e)}")
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """캔버스 좌표를 실제 이미지 좌표로 변환"""
        try:
            if not hasattr(self, 'display_photo') or not self.display_photo:
                return None
            
            # 캔버스에서 이미지의 실제 위치 찾기
            canvas_items = self.image_canvas.find_all()
            if not canvas_items:
                return None
            
            # 이미지 아이템의 좌표와 크기 가져오기
            image_item = canvas_items[0]  # 첫 번째 아이템이 이미지
            bbox = self.image_canvas.bbox(image_item)
            if not bbox:
                return None
            
            img_left, img_top, img_right, img_bottom = bbox
            
            # 클릭이 이미지 영역 내부인지 확인
            if canvas_x < img_left or canvas_x > img_right or canvas_y < img_top or canvas_y > img_bottom:
                return None
            
            # 캔버스 상의 이미지 크기
            display_width = img_right - img_left
            display_height = img_bottom - img_top
            
            # 원본 이미지 크기
            if hasattr(self, 'display_image') and self.display_image:
                orig_width, orig_height = self.display_image.size
            else:
                return None
            
            # 좌표 변환
            rel_x = (canvas_x - img_left) / display_width
            rel_y = (canvas_y - img_top) / display_height
            
            img_x = int(rel_x * orig_width)
            img_y = int(rel_y * orig_height)
            
            # 이미지 범위 내 확인
            if 0 <= img_x < orig_width and 0 <= img_y < orig_height:
                return (img_x, img_y)
            
            return None
            
        except Exception as e:
            print(f"Coordinate conversion error: {str(e)}")
            return None
    
    def get_object_at_coords(self, x, y):
        """특정 좌표의 segmentation 객체 클래스 반환"""
        try:
            if not hasattr(self, 'current_inst_seg_data') or not self.current_inst_seg_data:
                return None
            
            seg_image = self.current_inst_seg_data['seg_image']
            color_mapping = self.current_inst_seg_data['color_mapping']
            
            # 해당 좌표의 픽셀 색상 가져오기
            pixel_color = seg_image.getpixel((x, y))
            
            # 색상이 매핑에 있는지 확인
            if pixel_color in color_mapping:
                return color_mapping[pixel_color].get('class', None)
            
            return None
            
        except Exception as e:
            print(f"Object detection error: {str(e)}")
            return None
    
    def find_bbox_by_target_object(self, target_object):
        """특정 target_object를 가지는 bbox의 인덱스 반환"""
        try:
            for i, bbox_data in enumerate(self.processed_bboxes):
                if bbox_data.get('target_object', '') == target_object:
                    return i
            return -1
            
        except Exception as e:
            print(f"BBox search error: {str(e)}")
            return -1
    
    def find_all_bboxes_by_target_object(self, target_object):
        """특정 target_object를 가지는 모든 bbox의 인덱스 리스트 반환"""
        try:
            indices = []
            for i, bbox_data in enumerate(self.processed_bboxes):
                if bbox_data.get('target_object', '') == target_object:
                    indices.append(i)
            return indices
            
        except Exception as e:
            print(f"BBox search error: {str(e)}")
            return []
    
    def configure_scroll_region(self, event=None):
        """스크롤 영역 설정"""
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        # 캔버스 너비에 맞춰 내부 프레임 너비 조정
        canvas_width = self.left_canvas.winfo_width()
        if canvas_width > 1:  # 최소 크기 확인
            self.left_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)
    
    def on_key_press(self, event):
        """키보드 이벤트 처리"""
        if event.keysym == 'Left':
            if event.state & 0x1:  # Shift 키
                self.prev_bbox()
            else:
                self.prev_image()
        elif event.keysym == 'Right':
            if event.state & 0x1:  # Shift 키
                self.next_bbox()
            else:
                self.next_image()

def main():
    root = ctk.CTk()
    app = ImageGraspViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
