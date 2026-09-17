#!/usr/bin/env python3
"""Interactive filter, review, and final export tool for collected grasps.

No command-line arguments are used. Edit the constants below or choose a root
directory from the GUI. The final JSON/NPZ use the same zero-pose conversion
and serialization functions as merge_grasps.py.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import open3d as o3d

from scipy.spatial import cKDTree

from grasp.mesh_geometry import (
    load_obj_triangles,
    resolve_obj_path,
    transform_object_vertices,
)
from merge_grasps import (
    atomic_json,
    grasp_boxes,
    load_json,
    mesh_in_box,
    rotation_distance,
    save_npz,
    to_zero_pose,
)


# -----------------------------------------------------------------------------
# 실행 기본값: argparse 없이 여기 또는 GUI에서 변경한다.
# ROOT는 <output-root>/<object>/<gripper> 폴더다.
# -----------------------------------------------------------------------------
ROOT = Path(
    "/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen/"
    "black_pepper_shaker/Robotiq_2f140"
)
SCENE_START: int | None = None
SCENE_END: int | None = None
GRASP_DIR_NAME = "output_grasp"
MESH_UNIT_SCALE = 0.01

DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_BOX_THICKNESS_M = 0.02
DEFAULT_BOX_MARGIN_M = 0.002
DEFAULT_MIN_OCCUPIED_BOXES = 1
DEFAULT_NMS_CENTER_M = 0.015
DEFAULT_NMS_ROTATION_DEG = 20.0
AUTO_LOAD = True
OPEN3D_WINDOW_WIDTH = 1100
OPEN3D_WINDOW_HEIGHT = 850


SCORE_FIELDS = (
    ("score", "총점", DEFAULT_SCORE_THRESHOLD),
    ("force_score", "외력 점수", 0.0),
    ("pregrasp_pose_score", "파지 자세 점수", 0.0),
    ("stress_pose_score", "외력 중 자세 점수", 0.0),
    ("contact_area_score", "접촉면 점수", 0.0),
)
SORT_FIELDS = (
    ("score", "총점"),
    ("force_score", "외력 점수"),
    ("pregrasp_pose_score", "파지 자세 점수"),
    ("stress_pose_score", "외력 중 자세 점수"),
    ("contact_area_score", "접촉면 점수"),
    ("scene_id", "Scene ID"),
)


def normalize_grasp_payload(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "grasps"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def parse_optional_int(value: str) -> int | None:
    text = value.strip()
    return None if not text else int(text)


def finite_score(item: dict, key: str, fallback: float = -math.inf) -> float:
    try:
        value = float(item.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def review_id(item: dict) -> str:
    return str(item["_filter_gui_id"])


def clean_item(item: dict) -> dict:
    result = copy.deepcopy(item)
    result.pop("_filter_gui_id", None)
    result.pop("_filter_gui_source_order", None)
    return result


def item_center(item: dict) -> np.ndarray:
    boxes = grasp_boxes(item)
    if len(boxes):
        return boxes.reshape(-1, 3).mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def item_approach_vector(item: dict) -> np.ndarray | None:
    """Return the gripper +Z approach axis, never the external-force normal."""
    vector = np.asarray(item.get("approach_vector", []), dtype=np.float64)
    if vector.shape != (3,):
        matrix = np.asarray(item.get("grasp_mat", []), dtype=np.float64)
        if matrix.shape != (4, 4):
            return None
        vector = matrix[:3, 2]
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1.0e-12 else vector / norm


def approach_bbox_error_degrees(item: dict) -> float | None:
    approach = item_approach_vector(item)
    boxes = grasp_boxes(item)
    if approach is None or not len(boxes):
        return None
    first = boxes[0]
    normal = np.cross(first[1] - first[0], first[3] - first[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        return None
    cosine = float(np.clip(abs(np.dot(normal / norm, approach)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def nms_by_score(
    items: list[dict], center_threshold: float, rotation_threshold: float, score_key: str
) -> list[dict]:
    kept: list[dict] = []
    spatial_cells: dict[tuple[int, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    ordered = sorted(items, key=lambda value: finite_score(value, score_key), reverse=True)
    cell_size = max(float(center_threshold), 1.0e-12)
    for item in ordered:
        center = item_center(item)
        matrix = np.asarray(item.get("grasp_mat", np.eye(4)), dtype=np.float64)
        rotation = matrix[:3, :3] if matrix.shape == (4, 4) else np.eye(3)
        cell = tuple(np.floor(center / cell_size).astype(np.int64).tolist())
        nearby = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby.extend(
                        spatial_cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
                    )
        duplicate = any(
            np.linalg.norm(center - old_center) <= center_threshold
            and rotation_distance(rotation, old_rotation) <= rotation_threshold
            for old_center, old_rotation in nearby
        )
        if not duplicate:
            kept.append(item)
            spatial_cells.setdefault(cell, []).append((center, rotation))
    return kept


def occupied_box_count(
    item: dict,
    mesh_vertices: np.ndarray,
    mesh_tree: cKDTree,
    thickness: float,
    margin: float,
) -> int:
    """Count occupied finger boxes after a cheap spherical broad-phase query."""
    count = 0
    for box in grasp_boxes(item):
        center = box.mean(axis=0)
        radius = float(np.linalg.norm(box - center, axis=1).max()) + thickness * 0.5 + margin
        indices = mesh_tree.query_ball_point(center, max(radius, 1.0e-9))
        if indices and mesh_in_box(mesh_vertices[indices], box, thickness, margin):
            count += 1
    return count


from PySide6 import QtCore, QtGui, QtWidgets


APP_STYLE = r"""
QMainWindow, QWidget {
    background: #0f172a;
    color: #e5e7eb;
}
QWidget {
    font-size: 10.5pt;
}
QFrame#Card, QGroupBox {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 12px;
}
QGroupBox {
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
    color: #f8fafc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #cbd5e1;
}
QLabel#Title {
    font-size: 24pt;
    font-weight: 800;
    color: #f8fafc;
}
QLabel#Subtitle {
    color: #94a3b8;
    font-size: 10.5pt;
}
QLabel#SectionTitle {
    color: #f8fafc;
    font-size: 12pt;
    font-weight: 700;
}
QLabel#Muted {
    color: #94a3b8;
}
QLabel#StatusLabel {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 10px;
    padding: 10px 12px;
    color: #cbd5e1;
}
QLabel#ItemLabel {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
    color: #f8fafc;
    font-weight: 650;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0b1220;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 7px 9px;
    min-height: 22px;
    color: #f8fafc;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #60a5fa;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox QAbstractItemView {
    background: #111827;
    border: 1px solid #334155;
    color: #f8fafc;
    selection-background-color: #1d4ed8;
    padding: 4px;
}
QPushButton {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 24px;
    color: #e5e7eb;
    font-weight: 650;
}
QPushButton:hover {
    background: #273449;
    border-color: #475569;
}
QPushButton:pressed {
    background: #172033;
}
QPushButton#PrimaryButton {
    background: #2563eb;
    border-color: #2563eb;
    color: white;
}
QPushButton#PrimaryButton:hover { background: #3b82f6; }
QPushButton#SuccessButton {
    background: #059669;
    border-color: #059669;
    color: white;
}
QPushButton#SuccessButton:hover { background: #10b981; }
QPushButton#DangerButton {
    background: #dc2626;
    border-color: #dc2626;
    color: white;
}
QPushButton#DangerButton:hover { background: #ef4444; }
QPushButton#NeutralButton {
    background: #475569;
    border-color: #475569;
    color: white;
}
QPushButton#SegmentButton {
    background: #0b1220;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 10px;
}
QPushButton#SegmentButton:checked {
    background: #1d4ed8;
    border-color: #60a5fa;
    color: white;
}
QCheckBox {
    spacing: 8px;
    color: #e5e7eb;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #475569;
    border-radius: 5px;
    background: #0b1220;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #60a5fa;
}
QTabWidget::pane {
    border: 1px solid #263244;
    background: #0f172a;
    border-radius: 10px;
    top: -1px;
}
QTabBar::tab {
    background: #111827;
    color: #94a3b8;
    border: 1px solid #263244;
    border-bottom: none;
    padding: 10px 18px;
    margin-right: 3px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: #172033;
    color: #f8fafc;
    border-color: #3b82f6;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    width: 10px;
    background: #0f172a;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QToolTip {
    background: #111827;
    color: #f8fafc;
    border: 1px solid #475569;
    padding: 5px;
}
"""


def choose_ui_font() -> str:
    preferred = [
        "Pretendard",
        "Noto Sans KR",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    try:
        families = set(QtGui.QFontDatabase.families())
    except Exception:
        families = set()
    for name in preferred:
        if name in families:
            return name
    return "Sans Serif"


class GraspFilterGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grasp Filter & Review")
        self.resize(980, 940)
        self.setMinimumSize(820, 720)

        self.candidates: list[dict] = []
        self.filtered: list[dict] = []
        self.visible: list[dict] = []
        self.decisions: dict[str, str] = {}
        self.current_index = 0
        self.reference_object: dict | None = None
        self.scene_ids: list[str] = []
        self.mesh_vertices = np.zeros((0, 3), dtype=np.float64)
        self.mesh_triangles = np.zeros((0, 3), dtype=np.int64)
        self.mesh_tree: cKDTree | None = None
        self.object_name = "object"
        self.last_filter_counts: dict[str, int] = {}
        self.loaded_root: Path | None = None
        self.loaded_grasp_dir_name: str | None = None
        self.visualizer = None
        self.object_mesh = None
        self.grasp_lines = None
        self.target_cloud = None

        self.score_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.score_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.view_buttons: dict[str, QtWidgets.QPushButton] = {}

        self._build_ui()
        self._bind_shortcuts()

        self.open3d_timer = QtCore.QTimer(self)
        self.open3d_timer.setInterval(16)
        self.open3d_timer.timeout.connect(self._poll_open3d)
        self.open3d_timer.start()

        self.open_visualizer()
        if AUTO_LOAD:
            QtCore.QTimer.singleShot(120, self.load_dataset)

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _card(title: str | None = None):
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        if title:
            label = QtWidgets.QLabel(title)
            label.setObjectName("SectionTitle")
            layout.addWidget(label)
        return frame, layout

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float, step: float, decimals: int = 4):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _int_spin(value: int, minimum: int, maximum: int, step: int = 1):
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(14)

        # Header
        title = QtWidgets.QLabel("Grasp Filter")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("파지 데이터 필터링 · Open3D 검수 · 최종 병합")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # Dataset card
        dataset_card, dataset_layout = self._card("Dataset")
        path_row = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit(str(ROOT))
        self.root_edit.setPlaceholderText("<output-root>/<object>/<gripper>")
        choose_btn = QtWidgets.QPushButton("폴더 선택")
        choose_btn.clicked.connect(self.choose_root)
        load_btn = QtWidgets.QPushButton("데이터 불러오기")
        load_btn.setObjectName("PrimaryButton")
        load_btn.clicked.connect(self.load_dataset)
        path_row.addWidget(self.root_edit, 1)
        path_row.addWidget(choose_btn)
        path_row.addWidget(load_btn)
        dataset_layout.addLayout(path_row)

        settings_row = QtWidgets.QGridLayout()
        settings_row.setHorizontalSpacing(10)
        settings_row.setVerticalSpacing(6)
        self.scene_start_edit = QtWidgets.QLineEdit("" if SCENE_START is None else str(SCENE_START))
        self.scene_start_edit.setPlaceholderText("전체")
        self.scene_start_edit.setMaximumWidth(120)
        self.scene_end_edit = QtWidgets.QLineEdit("" if SCENE_END is None else str(SCENE_END))
        self.scene_end_edit.setPlaceholderText("전체")
        self.scene_end_edit.setMaximumWidth(120)
        self.grasp_dir_edit = QtWidgets.QLineEdit(GRASP_DIR_NAME)
        self.grasp_dir_edit.setPlaceholderText("output_grasp")
        self.grasp_dir_edit.setMaximumWidth(210)
        self.mesh_scale_spin = self._double_spin(MESH_UNIT_SCALE, 0.000001, 1000.0, 0.001, 6)
        self.mesh_scale_spin.setMaximumWidth(140)
        settings_row.addWidget(QtWidgets.QLabel("Scene 시작"), 0, 0)
        settings_row.addWidget(self.scene_start_edit, 0, 1)
        settings_row.addWidget(QtWidgets.QLabel("Scene 끝"), 0, 2)
        settings_row.addWidget(self.scene_end_edit, 0, 3)
        settings_row.addWidget(QtWidgets.QLabel("Mesh scale"), 0, 4)
        settings_row.addWidget(self.mesh_scale_spin, 0, 5)
        settings_row.addWidget(QtWidgets.QLabel("Grasp 폴더"), 0, 6)
        settings_row.addWidget(self.grasp_dir_edit, 0, 7)
        settings_row.setColumnStretch(8, 1)
        dataset_layout.addLayout(settings_row)
        root.addWidget(dataset_card)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_filter_tab(), "FILTER")
        tabs.addTab(self._build_review_tab(), "REVIEW & EXPORT")
        root.addWidget(tabs, 1)

        # Bottom info
        self.status_label = QtWidgets.QLabel("데이터를 불러오세요.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.item_label = QtWidgets.QLabel("표시할 grasp가 없습니다.")
        self.item_label.setObjectName("ItemLabel")
        self.item_label.setWordWrap(True)
        root.addWidget(self.item_label)

        viewer_row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open3D 창 열기")
        open_btn.clicked.connect(self.open_visualizer)
        reset_btn = QtWidgets.QPushButton("시점 초기화")
        reset_btn.clicked.connect(self.reset_view)
        viewer_row.addWidget(open_btn)
        viewer_row.addWidget(reset_btn)
        root.addLayout(viewer_row)

        legend = QtWidgets.QLabel(
            "Open3D  ·  cyan 현재 bbox  ·  orange gripper approach (+Z)  ·  "
            "magenta 외력 방향  ·  green 승인  ·  red 제외"
        )
        legend.setObjectName("Muted")
        legend.setWordWrap(True)
        root.addWidget(legend)

    def _scroll_tab(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(12)
        scroll.setWidget(body)
        return scroll, layout

    def _build_filter_tab(self):
        scroll, layout = self._scroll_tab()

        quality = QtWidgets.QGroupBox("Quality")
        ql = QtWidgets.QVBoxLayout(quality)
        self.require_completed_check = QtWidgets.QCheckBox("quality.result == completed만 사용")
        self.require_completed_check.setChecked(True)
        ql.addWidget(self.require_completed_check)
        layout.addWidget(quality)

        score_box = QtWidgets.QGroupBox("최소 점수")
        score_grid = QtWidgets.QGridLayout(score_box)
        score_grid.setColumnStretch(1, 1)
        for row, (key, label, default) in enumerate(SCORE_FIELDS):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(key == "score")
            spin = self._double_spin(default, 0.0, 1.0, 0.01, 3)
            spin.setMaximumWidth(130)
            check.toggled.connect(spin.setEnabled)
            spin.setEnabled(check.isChecked())
            score_grid.addWidget(check, row, 0)
            score_grid.addWidget(spin, row, 1, alignment=QtCore.Qt.AlignRight)
            self.score_checks[key] = check
            self.score_spins[key] = spin
        layout.addWidget(score_box)

        empty_box = QtWidgets.QGroupBox("Empty bbox")
        empty_grid = QtWidgets.QGridLayout(empty_box)
        self.empty_filter_check = QtWidgets.QCheckBox("물체가 없는 bbox 제거")
        self.empty_filter_check.setChecked(True)
        empty_grid.addWidget(self.empty_filter_check, 0, 0, 1, 2)
        self.box_thickness_spin = self._double_spin(DEFAULT_BOX_THICKNESS_M, 0.0, 10.0, 0.001, 4)
        self.box_margin_spin = self._double_spin(DEFAULT_BOX_MARGIN_M, 0.0, 10.0, 0.0005, 4)
        self.min_occupied_spin = self._int_spin(DEFAULT_MIN_OCCUPIED_BOXES, 1, 1000)
        self._form_row(empty_grid, 1, "두께 (m)", self.box_thickness_spin)
        self._form_row(empty_grid, 2, "여유 (m)", self.box_margin_spin)
        self._form_row(empty_grid, 3, "필요 bbox 수", self.min_occupied_spin)
        layout.addWidget(empty_box)

        nms_box = QtWidgets.QGroupBox("NMS")
        nms_grid = QtWidgets.QGridLayout(nms_box)
        self.nms_check = QtWidgets.QCheckBox("중복 파지 제거")
        self.nms_check.setChecked(True)
        nms_grid.addWidget(self.nms_check, 0, 0, 1, 2)
        self.nms_center_spin = self._double_spin(DEFAULT_NMS_CENTER_M, 0.0, 10.0, 0.001, 4)
        self.nms_rotation_spin = self._double_spin(DEFAULT_NMS_ROTATION_DEG, 0.0, 360.0, 1.0, 2)
        self.nms_score_combo = QtWidgets.QComboBox()
        for key, label in SORT_FIELDS:
            if key != "scene_id":
                self.nms_score_combo.addItem(label, key)
        self._form_row(nms_grid, 1, "중심 거리 (m)", self.nms_center_spin)
        self._form_row(nms_grid, 2, "회전 거리 (deg)", self.nms_rotation_spin)
        self._form_row(nms_grid, 3, "우선 점수", self.nms_score_combo)
        layout.addWidget(nms_box)

        apply_btn = QtWidgets.QPushButton("필터 적용")
        apply_btn.setObjectName("PrimaryButton")
        apply_btn.clicked.connect(self.apply_filters)
        layout.addWidget(apply_btn)

        self.count_label = QtWidgets.QLabel("")
        self.count_label.setObjectName("Muted")
        self.count_label.setWordWrap(True)
        layout.addWidget(self.count_label)
        layout.addStretch(1)
        return scroll

    def _build_review_tab(self):
        scroll, layout = self._scroll_tab()

        sort_box = QtWidgets.QGroupBox("표시 순서")
        sort_layout = QtWidgets.QGridLayout(sort_box)
        self.sort_combo = QtWidgets.QComboBox()
        for key, label in SORT_FIELDS:
            self.sort_combo.addItem(label, key)
        self.sort_descending_check = QtWidgets.QCheckBox("높은 값부터")
        self.sort_descending_check.setChecked(True)
        self.sort_descending_check.toggled.connect(lambda _checked: self.refresh_visible())
        sort_btn = QtWidgets.QPushButton("정렬 적용")
        sort_btn.clicked.connect(self.refresh_visible)
        sort_layout.addWidget(self.sort_combo, 0, 0, 1, 2)
        sort_layout.addWidget(self.sort_descending_check, 1, 0)
        sort_layout.addWidget(sort_btn, 1, 1)
        layout.addWidget(sort_box)

        view_box = QtWidgets.QGroupBox("3D View")
        view_layout = QtWidgets.QVBoxLayout(view_box)
        modes = QtWidgets.QHBoxLayout()
        self.view_group = QtWidgets.QButtonGroup(self)
        self.view_group.setExclusive(True)
        for label, mode in (
            ("현재 1개", "single"),
            ("필터 전체", "all"),
            ("승인만", "accepted"),
            ("제외만", "rejected"),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("SegmentButton")
            button.setCheckable(True)
            button.setProperty("mode", mode)
            button.clicked.connect(lambda _checked=False, m=mode: self.set_view_mode(m))
            self.view_group.addButton(button)
            self.view_buttons[mode] = button
            modes.addWidget(button)
        self.view_buttons["all"].setChecked(True)
        view_layout.addLayout(modes)
        hint = QtWidgets.QLabel("전체 보기는 필터 통과 bbox를 한 번에 표시합니다.")
        hint.setObjectName("Muted")
        view_layout.addWidget(hint)
        layout.addWidget(view_box)

        nav_box = QtWidgets.QGroupBox("Individual review")
        nav_layout = QtWidgets.QVBoxLayout(nav_box)
        nav_row = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("◀ 이전")
        prev_btn.clicked.connect(lambda: self.move(-1))
        next_btn = QtWidgets.QPushButton("다음 ▶")
        next_btn.clicked.connect(lambda: self.move(1))
        nav_row.addWidget(prev_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(next_btn)
        nav_layout.addLayout(nav_row)

        jump_row = QtWidgets.QHBoxLayout()
        jump_row.addWidget(QtWidgets.QLabel("번호"))
        self.jump_spin = self._int_spin(0, 0, 10_000_000)
        self.jump_spin.setMaximumWidth(130)
        jump_btn = QtWidgets.QPushButton("이동")
        jump_btn.clicked.connect(self.jump_to)
        jump_row.addWidget(self.jump_spin)
        jump_row.addWidget(jump_btn)
        jump_row.addStretch(1)
        nav_layout.addLayout(jump_row)

        self.unreviewed_only_check = QtWidgets.QCheckBox("미검수 항목만 보기")
        self.unreviewed_only_check.toggled.connect(lambda _checked: self.refresh_visible())
        nav_layout.addWidget(self.unreviewed_only_check)

        decision_row = QtWidgets.QHBoxLayout()
        accept_btn = QtWidgets.QPushButton("승인   A")
        accept_btn.setObjectName("SuccessButton")
        accept_btn.clicked.connect(lambda: self.set_decision("accept"))
        reject_btn = QtWidgets.QPushButton("제외   R")
        reject_btn.setObjectName("DangerButton")
        reject_btn.clicked.connect(lambda: self.set_decision("reject"))
        hold_btn = QtWidgets.QPushButton("보류   U")
        hold_btn.setObjectName("NeutralButton")
        hold_btn.clicked.connect(lambda: self.set_decision(""))
        decision_row.addWidget(accept_btn)
        decision_row.addWidget(reject_btn)
        decision_row.addWidget(hold_btn)
        nav_layout.addLayout(decision_row)
        layout.addWidget(nav_box)

        session_box = QtWidgets.QGroupBox("검수 상태")
        session_layout = QtWidgets.QHBoxLayout(session_box)
        save_state_btn = QtWidgets.QPushButton("상태 저장")
        save_state_btn.clicked.connect(self.save_review_state)
        load_state_btn = QtWidgets.QPushButton("상태 불러오기")
        load_state_btn.clicked.connect(self.load_review_state)
        session_layout.addWidget(save_state_btn)
        session_layout.addWidget(load_state_btn)
        layout.addWidget(session_box)

        export_box = QtWidgets.QGroupBox("최종 저장")
        export_layout = QtWidgets.QVBoxLayout(export_box)
        self.accepted_only_save_check = QtWidgets.QCheckBox("승인한 항목만 저장")
        export_layout.addWidget(self.accepted_only_save_check)
        export_hint = QtWidgets.QLabel("체크하지 않으면 필터 통과 항목 중 제외 표시만 제거합니다.")
        export_hint.setObjectName("Muted")
        export_hint.setWordWrap(True)
        export_layout.addWidget(export_hint)
        save_btn = QtWidgets.QPushButton("최종 JSON + NPZ 저장   Ctrl+S")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self.save_final)
        export_layout.addWidget(save_btn)
        layout.addWidget(export_box)

        layout.addStretch(1)
        return scroll

    @staticmethod
    def _form_row(layout: QtWidgets.QGridLayout, row: int, label: str, widget: QtWidgets.QWidget):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(widget, row, 1)
        layout.setColumnStretch(1, 1)

    # --------------------------------------------------------------- shortcuts
    def _bind_shortcuts(self) -> None:
        self._shortcuts = []
        bindings = [
            ("Left", self.move, (-1,)),
            ("Right", self.move, (1,)),
            ("PgUp", self.move, (-10,)),
            ("PgDown", self.move, (10,)),
            ("A", self.set_decision, ("accept",)),
            ("R", self.set_decision, ("reject",)),
            ("U", self.set_decision, ("",)),
            ("1", self.set_view_mode, ("single",)),
            ("2", self.set_view_mode, ("all",)),
            ("3", self.set_view_mode, ("accepted",)),
            ("4", self.set_view_mode, ("rejected",)),
            ("Ctrl+S", self.save_final, ()),
        ]
        for sequence, callback, args in bindings:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.activated.connect(lambda cb=callback, a=args: self._shortcut(cb, *a))
            self._shortcuts.append(shortcut)

    def _shortcut(self, callback, *args):
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(
            focus,
            (QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QComboBox),
        ):
            return
        callback(*args)

    # ------------------------------------------------------------ data loading
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        QtWidgets.QApplication.processEvents()

    def choose_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "데이터 폴더 선택",
            self.root_edit.text(),
        )
        if selected:
            self.root_edit.setText(selected)
            self.load_dataset()

    def _grasp_dir_name(self) -> str:
        value = self.grasp_dir_edit.text().strip()
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("Grasp 폴더에는 ROOT 바로 아래의 폴더명만 입력하세요.")
        return value

    def _selected_scene_ids(self, root: Path, grasp_dir_name: str) -> list[str]:
        conf_ids = {path.stem for path in (root / "conf").glob("*.json")}
        grasp_ids = {path.stem for path in (root / grasp_dir_name).glob("*.json")}
        scene_ids = sorted(value for value in conf_ids & grasp_ids if value.isdigit())
        start = parse_optional_int(self.scene_start_edit.text())
        end = parse_optional_int(self.scene_end_edit.text())
        if start is not None:
            scene_ids = [value for value in scene_ids if int(value) >= start]
        if end is not None:
            scene_ids = [value for value in scene_ids if int(value) <= end]
        return scene_ids

    def load_dataset(self) -> None:
        try:
            root = Path(self.root_edit.text()).expanduser().resolve()
            grasp_dir_name = self._grasp_dir_name()
            scene_ids = self._selected_scene_ids(root, grasp_dir_name)
            if not scene_ids:
                raise ValueError(
                    f"conf와 {grasp_dir_name}가 함께 있는 scene이 없습니다."
                )

            self._set_status("데이터와 mesh를 읽는 중...")
            candidates: list[dict] = []
            reference_object = None
            source_order = 0

            for scene_id in scene_ids:
                conf = load_json(root / "conf" / f"{scene_id}.json")
                objects = conf.get("objects", [])
                if len(objects) != 1:
                    raise ValueError(f"scene {scene_id}: single-object conf가 아닙니다.")
                obj = objects[0]
                reference_object = reference_object or obj
                payload = load_json(root / grasp_dir_name / f"{scene_id}.json")
                for local_index, item in enumerate(normalize_grasp_payload(payload)):
                    transformed = to_zero_pose(item, obj, scene_id)
                    if transformed is None:
                        continue
                    transformed["_filter_gui_id"] = f"{scene_id}:{local_index}"
                    transformed["_filter_gui_source_order"] = source_order
                    candidates.append(transformed)
                    source_order += 1

            if reference_object is None:
                raise ValueError("object 정보를 찾지 못했습니다.")

            obj_path = resolve_obj_path(reference_object["usd_path"])
            vertices, triangles = load_obj_triangles(obj_path)
            vertices = transform_object_vertices(
                vertices,
                reference_object,
                float(self.mesh_scale_spin.value()),
                apply_pose=False,
            )

            self.candidates = candidates
            if (
                self.loaded_root is not None
                and (self.loaded_root != root or self.loaded_grasp_dir_name != grasp_dir_name)
            ):
                self.decisions = {}
            self.loaded_root = root
            self.loaded_grasp_dir_name = grasp_dir_name
            self.reference_object = reference_object
            self.scene_ids = scene_ids
            self.mesh_vertices = vertices
            self.mesh_triangles = triangles
            self.mesh_tree = cKDTree(vertices)
            self.object_name = str(reference_object.get("class", root.parent.name))
            self.current_index = 0
            self._reset_visualizer()
            self._set_status(
                f"로드 완료  ·  {len(scene_ids)} scenes  ·  {len(candidates)} grasps  ·  mesh {obj_path.name}"
            )
            self.apply_filters()
        except Exception as exc:
            self._set_status(f"로드 실패: {exc}")
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))

    # --------------------------------------------------------------- filtering
    def _current_view_mode(self) -> str:
        for mode, button in self.view_buttons.items():
            if button.isChecked():
                return mode
        return "all"

    def _filter_settings(self) -> dict:
        return {
            "mesh_unit_scale": float(self.mesh_scale_spin.value()),
            "require_completed": bool(self.require_completed_check.isChecked()),
            "score_thresholds": {
                key: float(self.score_spins[key].value())
                for key, _label, _default in SCORE_FIELDS
                if self.score_checks[key].isChecked()
            },
            "empty_box": {
                "enabled": bool(self.empty_filter_check.isChecked()),
                "thickness_m": max(0.0, float(self.box_thickness_spin.value())),
                "margin_m": max(0.0, float(self.box_margin_spin.value())),
                "min_occupied_boxes": max(1, int(self.min_occupied_spin.value())),
            },
            "nms": {
                "enabled": bool(self.nms_check.isChecked()),
                "center_m": max(0.0, float(self.nms_center_spin.value())),
                "rotation_deg": max(0.0, float(self.nms_rotation_spin.value())),
                "score_key": str(self.nms_score_combo.currentData()),
            },
            "sort": {
                "key": str(self.sort_combo.currentData()),
                "descending": bool(self.sort_descending_check.isChecked()),
            },
            "view_mode": self._current_view_mode(),
        }

    def apply_filters(self) -> None:
        if not self.candidates:
            self.filtered = []
            self.refresh_visible()
            return
        try:
            settings = self._filter_settings()
            items = list(self.candidates)
            counts = {"loaded": len(items)}

            if settings["require_completed"]:
                items = [
                    item
                    for item in items
                    if isinstance(item.get("quality"), dict)
                    and item["quality"].get("result") == "completed"
                ]
            counts["completed"] = len(items)

            for key, threshold in settings["score_thresholds"].items():
                items = [item for item in items if finite_score(item, key) >= threshold]
                counts[f"after_{key}"] = len(items)

            empty = settings["empty_box"]
            if empty["enabled"]:
                thickness = max(0.0, empty["thickness_m"])
                margin = max(0.0, empty["margin_m"])
                required = max(1, empty["min_occupied_boxes"])
                self._set_status(f"Empty bbox 검사 중  ·  {len(items)}개")
                kept = []
                if self.mesh_tree is None:
                    raise RuntimeError("Mesh 공간 인덱스가 준비되지 않았습니다.")
                for item in items:
                    occupied = occupied_box_count(
                        item,
                        self.mesh_vertices,
                        self.mesh_tree,
                        thickness,
                        margin,
                    )
                    if occupied >= required:
                        kept.append(item)
                items = kept
            counts["after_empty_box"] = len(items)

            nms_settings = settings["nms"]
            if nms_settings["enabled"]:
                items = nms_by_score(
                    items,
                    max(0.0, nms_settings["center_m"]),
                    max(0.0, nms_settings["rotation_deg"]),
                    nms_settings["score_key"],
                )
            counts["after_nms"] = len(items)

            self.filtered = items
            self.last_filter_counts = counts
            self.current_index = 0
            self.refresh_visible()
            self._set_status("필터 적용 완료")
        except Exception as exc:
            self._set_status(f"필터 실패: {exc}")
            QtWidgets.QMessageBox.critical(self, "Filter failed", str(exc))

    def refresh_visible(self) -> None:
        old_index = self.current_index
        current_id = None
        if self.visible and 0 <= self.current_index < len(self.visible):
            current_id = review_id(self.visible[self.current_index])

        visible = list(self.filtered)
        if self.unreviewed_only_check.isChecked():
            visible = [item for item in visible if not self.decisions.get(review_id(item))]
        key = str(self.sort_combo.currentData())
        reverse = bool(self.sort_descending_check.isChecked())
        if key == "scene_id":
            visible.sort(key=lambda item: str(item.get("scene_id", "")), reverse=reverse)
        else:
            visible.sort(key=lambda item: finite_score(item, key), reverse=reverse)
        self.visible = visible

        if current_id:
            matching = [i for i, item in enumerate(visible) if review_id(item) == current_id]
            self.current_index = matching[0] if matching else old_index
        self.current_index = min(max(0, self.current_index), max(0, len(visible) - 1))
        self._update_count_text()
        self.render()

    def _update_count_text(self) -> None:
        decisions = Counter(self.decisions.get(review_id(item), "") for item in self.filtered)
        stage_text = "  →  ".join(
            f"{key.removeprefix('after_')} {value}"
            for key, value in self.last_filter_counts.items()
        )
        self.count_label.setText(
            f"{stage_text}\n"
            f"최종 필터 {len(self.filtered)}   ·   표시 {len(self.visible)}   ·   "
            f"승인 {decisions['accept']}   ·   제외 {decisions['reject']}   ·   미검수 {decisions['']}"
        )

    # ---------------------------------------------------------------- review
    def current_item(self) -> dict | None:
        if not self.visible:
            return None
        return self.visible[self.current_index]

    def move(self, amount: int) -> None:
        if not self.visible:
            return
        self.current_index = (self.current_index + amount) % len(self.visible)
        self.render()

    def jump_to(self) -> None:
        if not self.visible:
            return
        self.current_index = min(max(int(self.jump_spin.value()), 0), len(self.visible) - 1)
        self.render()

    def set_decision(self, decision: str) -> None:
        item = self.current_item()
        if item is None:
            return
        item_id = review_id(item)
        if decision:
            self.decisions[item_id] = decision
        else:
            self.decisions.pop(item_id, None)
        if self.unreviewed_only_check.isChecked() and decision:
            self.refresh_visible()
        else:
            self._update_count_text()
            self.render()

    # ------------------------------------------------------------- Open3D
    def open_visualizer(self) -> None:
        if self.visualizer is not None:
            return
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        created = visualizer.create_window(
            window_name="grasp filter review",
            width=OPEN3D_WINDOW_WIDTH,
            height=OPEN3D_WINDOW_HEIGHT,
        )
        if not created:
            self._set_status("Open3D 창을 만들지 못했습니다.")
            return
        self.visualizer = visualizer
        visualizer.register_key_callback(263, lambda _vis: self._open3d_action(self.move, -1))
        visualizer.register_key_callback(262, lambda _vis: self._open3d_action(self.move, 1))
        visualizer.register_key_callback(266, lambda _vis: self._open3d_action(self.move, -10))
        visualizer.register_key_callback(267, lambda _vis: self._open3d_action(self.move, 10))
        visualizer.register_key_callback(
            ord("A"), lambda _vis: self._open3d_action(self.set_decision, "accept")
        )
        visualizer.register_key_callback(
            ord("R"), lambda _vis: self._open3d_action(self.set_decision, "reject")
        )
        visualizer.register_key_callback(
            ord("U"), lambda _vis: self._open3d_action(self.set_decision, "")
        )
        for key, mode in (
            (ord("1"), "single"),
            (ord("2"), "all"),
            (ord("3"), "accepted"),
            (ord("4"), "rejected"),
        ):
            visualizer.register_key_callback(
                key,
                lambda _vis, selected=mode: self._open3d_action(self.set_view_mode, selected),
            )
        option = visualizer.get_render_option()
        option.background_color = np.asarray([0.03, 0.03, 0.035])
        option.line_width = 3.0
        option.point_size = 9.0
        self._reset_visualizer()
        self.render()

    @staticmethod
    def _open3d_action(callback, *args):
        callback(*args)
        return False

    def _poll_open3d(self) -> None:
        if self.visualizer is not None:
            try:
                if self.visualizer.poll_events():
                    self.visualizer.update_renderer()
                else:
                    self.visualizer.destroy_window()
                    self.visualizer = None
                    self.object_mesh = None
                    self.grasp_lines = None
                    self.target_cloud = None
            except Exception:
                self.visualizer = None
                self.object_mesh = None
                self.grasp_lines = None
                self.target_cloud = None

    def _reset_visualizer(self) -> None:
        if self.visualizer is None:
            return
        self.visualizer.clear_geometries()
        self.object_mesh = None
        self.grasp_lines = None
        self.target_cloud = None
        if len(self.mesh_vertices) and len(self.mesh_triangles):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(self.mesh_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(self.mesh_triangles.astype(np.int32))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.72, 0.72, 0.74])
            self.object_mesh = mesh
            self.visualizer.add_geometry(mesh, reset_bounding_box=True)
            self.reset_view()

    def reset_view(self) -> None:
        if self.visualizer is None or not len(self.mesh_vertices):
            return
        center = (self.mesh_vertices.min(axis=0) + self.mesh_vertices.max(axis=0)) * 0.5
        control = self.visualizer.get_view_control()
        control.set_lookat(center)
        control.set_front([0.4, -0.7, 0.55])
        control.set_up([0.0, 0.0, 1.0])
        control.set_zoom(0.7)

    @staticmethod
    def _append_segment(points, lines, colors, start, end, color) -> None:
        base = len(points)
        points.extend((np.asarray(start, dtype=float), np.asarray(end, dtype=float)))
        lines.append((base, base + 1))
        colors.append(color)

    def _append_grasp_boxes(self, points, lines, colors, item: dict, color) -> None:
        for box in grasp_boxes(item):
            for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
                self._append_segment(points, lines, colors, box[start], box[end], color)

    def _score_color(self, item: dict) -> list[float]:
        key = str(self.sort_combo.currentData())
        if key == "scene_id":
            key = "score"
        score = float(np.clip(finite_score(item, key, 0.0), 0.0, 1.0))
        blue = np.asarray([0.0, 0.25, 1.0])
        green = np.asarray([0.0, 0.95, 0.2])
        yellow = np.asarray([1.0, 0.9, 0.0])
        if score < 0.5:
            color = blue * (1.0 - score * 2.0) + green * (score * 2.0)
        else:
            amount = (score - 0.5) * 2.0
            color = green * (1.0 - amount) + yellow * amount
        return color.tolist()

    def set_view_mode(self, mode: str) -> None:
        button = self.view_buttons.get(mode)
        if button is not None:
            button.setChecked(True)
        self.render()

    def _view_mode_items(self) -> list[dict]:
        mode = self._current_view_mode()
        if mode == "accepted":
            return [
                item
                for item in self.filtered
                if self.decisions.get(review_id(item)) == "accept"
            ]
        if mode == "rejected":
            return [
                item
                for item in self.filtered
                if self.decisions.get(review_id(item)) == "reject"
            ]
        if mode == "all":
            return self.filtered
        current = self.current_item()
        return [current] if current is not None else []

    def _make_review_geometry(self, item: dict | None):
        points, lines, colors = [], [], []
        mode = self._current_view_mode()
        current_decision = self.decisions.get(review_id(item), "") if item is not None else ""
        current_is_visible = (
            mode in ("single", "all")
            or (mode == "accepted" and current_decision == "accept")
            or (mode == "rejected" and current_decision == "reject")
        )
        if mode != "single":
            candidates = self._view_mode_items()
            for candidate in candidates:
                if (
                    current_is_visible
                    and item is not None
                    and review_id(candidate) == review_id(item)
                ):
                    continue
                decision = self.decisions.get(review_id(candidate), "")
                color = self._score_color(candidate)
                if decision == "accept":
                    color = [0.12, 0.9, 0.35]
                elif decision == "reject":
                    color = [0.9, 0.18, 0.18]
                self._append_grasp_boxes(points, lines, colors, candidate, color)

        target = None
        if item is not None and current_is_visible:
            decision = self.decisions.get(review_id(item), "")
            color = [0.0, 0.85, 1.0]
            if mode == "single":
                color = {"accept": [0.1, 1.0, 0.3], "reject": [1.0, 0.1, 0.1]}.get(
                    decision, color
                )
            self._append_grasp_boxes(points, lines, colors, item, color)
            center = item_center(item)
            target_array = np.asarray(item.get("target_points", center), dtype=np.float64)
            if target_array.shape == (3,):
                target = target_array

            normal = np.asarray(item.get("normal", []), dtype=np.float64)
            if normal.shape == (3,) and np.linalg.norm(normal) > 1.0e-9:
                normal /= np.linalg.norm(normal)
                self._append_segment(
                    points, lines, colors, center, center + normal * 0.08, [1.0, 0.0, 0.75]
                )

            approach = item_approach_vector(item)
            if approach is not None:
                tip = center + approach * 0.08
                self._append_segment(points, lines, colors, center, tip, [1.0, 0.42, 0.0])
                reference = np.asarray([0.0, 0.0, 1.0])
                if abs(float(np.dot(reference, approach))) > 0.9:
                    reference = np.asarray([0.0, 1.0, 0.0])
                side = np.cross(approach, reference)
                side /= max(float(np.linalg.norm(side)), 1.0e-12)
                arrow_base = tip - approach * 0.018
                self._append_segment(
                    points,
                    lines,
                    colors,
                    tip,
                    arrow_base + side * 0.008,
                    [1.0, 0.42, 0.0],
                )
                self._append_segment(
                    points,
                    lines,
                    colors,
                    tip,
                    arrow_base - side * 0.008,
                    [1.0, 0.42, 0.0],
                )

        line_set = o3d.geometry.LineSet()
        if lines:
            line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
            line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        point_cloud = o3d.geometry.PointCloud()
        if target is not None:
            point_cloud.points = o3d.utility.Vector3dVector(target[None])
            point_cloud.colors = o3d.utility.Vector3dVector(np.asarray([[1.0, 0.85, 0.0]]))
        return line_set, point_cloud

    def render(self) -> None:
        item = self.current_item()
        mode = self._current_view_mode()
        mode_label = {
            "single": "현재 1개",
            "all": "필터 전체",
            "accepted": "승인만",
            "rejected": "제외만",
        }.get(mode, mode)
        drawn_count = len(self._view_mode_items())
        if item is not None:
            decision = self.decisions.get(review_id(item), "")
            values = "   ".join(
                f"{label} {finite_score(item, key, math.nan):.3f}"
                for key, label, _default in SCORE_FIELDS
            )
            bbox_error = approach_bbox_error_degrees(item)
            alignment_text = (
                "bbox⊥approach 오차 없음"
                if bbox_error is None
                else f"bbox⊥approach 오차 {bbox_error:.3f}°"
            )
            decision_label = {"accept": "승인", "reject": "제외"}.get(decision, "미검수")
            self.item_label.setText(
                f"{mode_label} · {drawn_count}개   |   "
                f"[{self.current_index}/{len(self.visible) - 1}]   "
                f"scene {item.get('scene_id')}   source {item.get('source_pregrasp_index', '?')}   "
                f"상태 {decision_label}   ·   {alignment_text}\n{values}"
            )
            self.jump_spin.blockSignals(True)
            self.jump_spin.setMaximum(max(0, len(self.visible) - 1))
            self.jump_spin.setValue(self.current_index)
            self.jump_spin.blockSignals(False)
        else:
            self.item_label.setText("표시할 grasp가 없습니다.")

        if self.visualizer is None:
            return
        if self.grasp_lines is not None:
            self.visualizer.remove_geometry(self.grasp_lines, reset_bounding_box=False)
        if self.target_cloud is not None:
            self.visualizer.remove_geometry(self.target_cloud, reset_bounding_box=False)
        self.grasp_lines, self.target_cloud = self._make_review_geometry(item)
        if self.grasp_lines.has_lines():
            self.visualizer.add_geometry(self.grasp_lines, reset_bounding_box=False)
        if self.target_cloud.has_points():
            self.visualizer.add_geometry(self.target_cloud, reset_bounding_box=False)
        self.visualizer.update_renderer()

    # ------------------------------------------------------------ state/export
    def _review_state_path(self) -> Path:
        grasp_dir = self._grasp_dir_name()
        label = grasp_dir.removeprefix("output_grasp").strip("_")
        suffix = "" if not label else f"_{label}"
        return (
            Path(self.root_edit.text())
            / f"{self.object_name}_grasp_review{suffix}.json"
        )

    def save_review_state(self) -> None:
        try:
            path = self._review_state_path()
            payload = {
                "root": str(Path(self.root_edit.text()).resolve()),
                "grasp_directory": self._grasp_dir_name(),
                "object": self.object_name,
                "scene_ids": self.scene_ids,
                "filters": self._filter_settings(),
                "decisions": self.decisions,
            }
            atomic_json(path, payload)
            self._set_status(f"검수 상태 저장: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save review failed", str(exc))

    def load_review_state(self) -> None:
        default = self._review_state_path()
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "검수 상태 불러오기",
            str(default),
            "JSON (*.json)",
        )
        if not selected:
            return
        try:
            payload = load_json(Path(selected))
            saved_root = Path(payload.get("root", "")).expanduser().resolve()
            current_root = Path(self.root_edit.text()).expanduser().resolve()
            if saved_root != current_root:
                raise ValueError(
                    f"검수 상태의 root가 현재 데이터와 다릅니다.\n"
                    f"saved: {saved_root}\ncurrent: {current_root}"
                )
            saved_grasp_dir = str(payload.get("grasp_directory", GRASP_DIR_NAME))
            if saved_grasp_dir != self._grasp_dir_name():
                raise ValueError(
                    "검수 상태의 grasp 폴더가 현재 선택과 다릅니다.\n"
                    f"saved: {saved_grasp_dir}\ncurrent: {self._grasp_dir_name()}"
                )
            previous_scale = float(self.mesh_scale_spin.value())
            self._restore_filter_settings(payload.get("filters", {}))
            if not math.isclose(previous_scale, float(self.mesh_scale_spin.value())):
                self.load_dataset()
            decisions = payload.get("decisions", {})
            valid_ids = {review_id(item) for item in self.candidates}
            self.decisions = {
                str(key): value
                for key, value in decisions.items()
                if key in valid_ids and value in ("accept", "reject")
            }
            self.apply_filters()
            self._set_status(f"검수 상태 불러옴: {selected}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load review failed", str(exc))

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if str(combo.itemData(i)) == str(value):
                combo.setCurrentIndex(i)
                return

    def _restore_filter_settings(self, settings: dict) -> None:
        if not isinstance(settings, dict):
            return
        if "mesh_unit_scale" in settings:
            self.mesh_scale_spin.setValue(float(settings["mesh_unit_scale"]))
        self.require_completed_check.setChecked(bool(settings.get("require_completed", True)))
        thresholds = settings.get("score_thresholds", {})
        if isinstance(thresholds, dict):
            for key, _label, _default in SCORE_FIELDS:
                enabled = key in thresholds
                self.score_checks[key].setChecked(enabled)
                if enabled:
                    self.score_spins[key].setValue(float(thresholds[key]))
        empty = settings.get("empty_box", {})
        if isinstance(empty, dict):
            self.empty_filter_check.setChecked(bool(empty.get("enabled", True)))
            self.box_thickness_spin.setValue(float(empty.get("thickness_m", DEFAULT_BOX_THICKNESS_M)))
            self.box_margin_spin.setValue(float(empty.get("margin_m", DEFAULT_BOX_MARGIN_M)))
            self.min_occupied_spin.setValue(int(empty.get("min_occupied_boxes", 1)))
        nms_settings = settings.get("nms", {})
        if isinstance(nms_settings, dict):
            self.nms_check.setChecked(bool(nms_settings.get("enabled", True)))
            self.nms_center_spin.setValue(float(nms_settings.get("center_m", DEFAULT_NMS_CENTER_M)))
            self.nms_rotation_spin.setValue(float(nms_settings.get("rotation_deg", DEFAULT_NMS_ROTATION_DEG)))
            self._set_combo_data(self.nms_score_combo, str(nms_settings.get("score_key", "score")))
        sort = settings.get("sort", {})
        if isinstance(sort, dict):
            self._set_combo_data(self.sort_combo, str(sort.get("key", "score")))
            self.sort_descending_check.setChecked(bool(sort.get("descending", True)))
        mode = settings.get("view_mode")
        if mode in {"single", "all", "accepted", "rejected"}:
            self.set_view_mode(mode)

    def final_items(self) -> list[dict]:
        result = []
        for item in self.filtered:
            decision = self.decisions.get(review_id(item), "")
            if decision == "reject":
                continue
            if self.accepted_only_save_check.isChecked() and decision != "accept":
                continue
            result.append(clean_item(item))
        key = str(self.sort_combo.currentData())
        reverse = bool(self.sort_descending_check.isChecked())
        if key == "scene_id":
            result.sort(key=lambda item: str(item.get("scene_id", "")), reverse=reverse)
        else:
            result.sort(key=lambda item: finite_score(item, key), reverse=reverse)
        return result

    def save_final(self) -> None:
        if self.reference_object is None:
            QtWidgets.QMessageBox.warning(self, "No data", "먼저 데이터를 불러오세요.")
            return
        items = self.final_items()
        grasp_dir = self._grasp_dir_name()
        label = grasp_dir.removeprefix("output_grasp").strip("_")
        suffix = "" if not label else f"_{label}"
        default = (
            Path(self.root_edit.text())
            / f"{self.object_name}_merged_grasp_reviewed{suffix}_zero_pose.json"
        )
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "최종 JSON 저장",
            str(default),
            "JSON (*.json)",
        )
        if not selected:
            return
        try:
            json_path = Path(selected)
            if json_path.suffix.lower() != ".json":
                json_path = json_path.with_suffix(".json")
            npz_path = json_path.with_suffix(".npz")
            decisions = Counter(self.decisions.get(review_id(item), "") for item in self.filtered)
            metadata = {
                "object": self.object_name,
                "gripper_root": str(Path(self.root_edit.text()).resolve()),
                "source_grasp_directory": self._grasp_dir_name(),
                "scene_ids": self.scene_ids,
                "approach_axis_index": 2,
                "approach_axis_sign": 1.0,
                "filters": self._filter_settings(),
                "manual_review": {
                    "accepted": decisions["accept"],
                    "rejected": decisions["reject"],
                    "accepted_only_save": bool(self.accepted_only_save_check.isChecked()),
                },
                "counts": {**self.last_filter_counts, "final": len(items)},
            }
            atomic_json(json_path, {"metadata": metadata, "data": items})
            save_npz(npz_path, metadata, items)
            self.save_review_state()
            self._set_status(f"최종 저장: {len(items)}개\n{json_path}\n{npz_path}")
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"{len(items)} grasps 저장 완료\n{json_path}\n{npz_path}",
            )
        except Exception as exc:
            self._set_status(f"저장 실패: {exc}")
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.visualizer is not None:
            try:
                self.visualizer.destroy_window()
            except Exception:
                pass
            self.visualizer = None
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setApplicationName("Grasp Filter & Review")
    app.setStyle("Fusion")
    font = QtGui.QFont(choose_ui_font(), 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(APP_STYLE)

    window = GraspFilterGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
