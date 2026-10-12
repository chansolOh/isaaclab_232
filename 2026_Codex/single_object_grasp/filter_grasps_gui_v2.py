#!/usr/bin/env python3
"""Filter/review/merge GUI for Dataset_2026/dataset_v2.

No command-line arguments are used.  Paths and defaults are constants below
and can also be changed through the GUI.
"""

from __future__ import annotations

import copy
from pathlib import Path

from scipy.spatial import cKDTree

import filter_grasps_gui as legacy
from dataset_v2_adapter import (
    DATASET_ROOT,
    MESH_UNIT_SCALE,
    MODEL_ROOT,
    atomic_json,
    compact_v2_item,
    gripper_names,
    load_object_grasps,
    object_names,
    save_v2_npz,
)
from grasp.mesh_geometry import load_obj_triangles, resolve_obj_path, transform_object_vertices


# -----------------------------------------------------------------------------
# 실행 기본값
# -----------------------------------------------------------------------------
ROOT = DATASET_ROOT
OUTPUT_MODEL_ROOT = MODEL_ROOT
OUTPUT_SUBDIR = ""
DEFAULT_OBJECT = "black_pepper_shaker"
DEFAULT_GRIPPER = "Inspire-F1_right"
AUTO_LOAD = False


# Reuse the mature Open3D review UI, but expose only quality values that really
# exist in dataset_v2.  The source has no force/contact/pose scores.
legacy.ROOT = ROOT
legacy.DEFAULT_OBJECT = DEFAULT_OBJECT
legacy.DEFAULT_GRIPPER = DEFAULT_GRIPPER
legacy.AUTO_LOAD = AUTO_LOAD
legacy.MESH_UNIT_SCALE = MESH_UNIT_SCALE
legacy.SCORE_FIELDS = (
    ("score", "비방해 점수", 0.5),
)
legacy.SORT_FIELDS = (
    ("score", "비방해 점수"),
    ("collateral_sort_score", "추가 방해 적은 순"),
    ("scene_id", "Scene"),
)

QtCore = legacy.QtCore
QtGui = legacy.QtGui
QtWidgets = legacy.QtWidgets


class DatasetV2FilterGUI(legacy.GraspFilterGUI):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset v2 · Grasp Filter & Review")
        self.root_edit.setText(str(ROOT))
        self.grasp_dir_edit.setText("output_grasp")
        self.grasp_dir_edit.setEnabled(False)
        self.scene_start_edit.setEnabled(False)
        self.scene_end_edit.setEnabled(False)
        self.require_completed_check.setChecked(False)
        self.require_completed_check.setEnabled(False)
        self.require_completed_check.setText("dataset_v2 output_grasp는 성공 결과만 포함")
        self.refresh_dataset_choices(DEFAULT_OBJECT, DEFAULT_GRIPPER)
        self._set_status(
            "dataset_v2를 선택했습니다. 최초 그리퍼 로드는 scene index를 한 번 생성합니다."
        )

    @staticmethod
    def _set_combo(combo, values: list[str], preferred: str) -> None:
        current = preferred or combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        if current in values:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def refresh_dataset_choices(self, preferred_object="", preferred_gripper="") -> None:
        if not hasattr(self, "object_combo"):
            return
        dataset_root = Path(self.root_edit.text()).expanduser()
        preferred_object = preferred_object if isinstance(preferred_object, str) else ""
        preferred_gripper = preferred_gripper if isinstance(preferred_gripper, str) else ""
        self._set_combo(self.object_combo, object_names(OUTPUT_MODEL_ROOT), preferred_object)
        try:
            names = gripper_names(dataset_root)
        except Exception:
            names = []
        self._set_combo(self.gripper_combo, names, preferred_gripper)

    def refresh_gripper_choices(self, preferred="") -> None:
        # Every v2 gripper may occur for an object; availability is resolved by
        # the per-gripper scene index during load.
        if not hasattr(self, "gripper_combo"):
            return
        dataset_root = Path(self.root_edit.text()).expanduser()
        current = preferred if isinstance(preferred, str) else self.gripper_combo.currentText()
        try:
            self._set_combo(self.gripper_combo, gripper_names(dataset_root), current)
        except Exception:
            pass

    def _dataset_path(self) -> Path:
        object_name = self.object_combo.currentText().strip()
        gripper = self.gripper_combo.currentText().strip()
        if not object_name or not gripper:
            raise ValueError("물체와 그리퍼를 선택하세요")
        return OUTPUT_MODEL_ROOT / object_name / OUTPUT_SUBDIR / gripper

    def _dataset_load_signature(self) -> tuple:
        return (
            str(Path(self.root_edit.text()).expanduser().resolve()),
            self.object_combo.currentText(),
            self.gripper_combo.currentText(),
            round(float(self.mesh_scale_spin.value()), 9),
        )

    def _progress(self, text: str) -> None:
        self._set_status(text)
        QtWidgets.QApplication.processEvents()

    def load_dataset(self) -> bool:
        try:
            dataset_root = Path(self.root_edit.text()).expanduser().resolve()
            object_name = self.object_combo.currentText().strip()
            gripper = self.gripper_combo.currentText().strip()
            if not dataset_root.is_dir():
                raise FileNotFoundError(dataset_root)
            if not object_name or not gripper:
                raise ValueError("물체와 그리퍼를 선택하세요")

            candidates, reference_object, scenes = load_object_grasps(
                dataset_root, object_name, gripper, progress=self._progress
            )
            if not candidates:
                raise ValueError(
                    f"{object_name} / {gripper}: 성공 output grasp 또는 hand bbox match가 없습니다"
                )
            # Lower collateral counts should sort first with the base GUI's
            # descending numeric sorter.
            for item in candidates:
                item["collateral_sort_score"] = -float(item["collateral_object_count"])

            obj_path = resolve_obj_path(reference_object["usd_path"])
            vertices, triangles = load_obj_triangles(obj_path)
            vertices = transform_object_vertices(
                vertices,
                reference_object,
                float(self.mesh_scale_spin.value()),
                apply_pose=False,
            )

            signature = self._dataset_load_signature()
            if self.loaded_dataset_signature is not None and self.loaded_dataset_signature != signature:
                self.decisions = {}
            self.candidates = candidates
            self.filtered = []
            self.visible = []
            self._invalidate_pre_nms_cache()
            self.loaded_root = dataset_root
            self.loaded_grasp_dir_name = "output_grasp"
            self.loaded_dataset_signature = signature
            self.reference_object = reference_object
            self.scene_ids = scenes
            self.mesh_vertices = vertices
            self.mesh_triangles = triangles
            self.mesh_tree = cKDTree(vertices)
            self.object_name = object_name
            self.current_index = 0
            self._reset_visualizer()
            self._set_status(
                f"로드 완료 · {len(scenes)} source scenes · {len(candidates)} grasps · "
                f"zero-pose mesh {obj_path.name}"
            )
            self.apply_filters()
            return True
        except Exception as exc:
            self._set_status(f"로드 실패: {exc}")
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return False

    def render(self) -> None:
        super().render()
        item = self.current_item()
        if item is not None:
            base = self.item_label.text()
            self.item_label.setText(
                base
                + f"\n원본 이동 물체 {item['disturbed_object_count']} · "
                f"추가 방해 물체 {item['collateral_object_count']} · "
                f"비방해 점수 {item['score']:.3f}"
            )

    def _review_state_path(self) -> Path:
        return self._dataset_path() / f"{self.object_name}_grasp_review.json"

    def start_isaac_replay(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Dataset v2 replay",
            "기존 replay는 single-object scene 전용입니다.\n"
            "dataset_v2는 이 GUI의 Open3D 개별/전체 검수를 사용하세요.",
        )

    def load_isaac_review_result(self) -> None:
        self.start_isaac_replay()

    def final_items(self) -> list[dict]:
        return [compact_v2_item(item) for item in self._selected_filtered_items()]

    def save_final(self) -> None:
        if self.reference_object is None:
            QtWidgets.QMessageBox.warning(self, "No data", "먼저 데이터를 불러오세요.")
            return
        items = self.final_items()
        gripper = self.gripper_combo.currentText().strip()
        output_dir = self._dataset_path()
        try:
            # The destination is part of the dataset contract.  A native Qt
            # save dialog silently falls back to another existing directory
            # when this nested directory has not been created yet, so do not
            # let dialog history change where merged data is written.
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / (
                f"{self.object_name}_{gripper}_merged_grasp_zero_pose.json"
            )
            npz_path = json_path.with_suffix(".npz")
            metadata = {
                "schema": "dataset_v2_object_grasp_compact",
                "schema_version": 1,
                "coordinate_frame": "object_zero_pose",
                "object": self.object_name,
                "gripper_model": gripper,
                "gripper_type": items[0].get("gripper_type", "") if items else "",
                "source": {
                    "dataset_root": str(Path(self.root_edit.text()).expanduser().resolve()),
                    "format": "environment/section/platform/output_grasp",
                },
                "score_definition": {
                    "disturbance_score": "1 / (1 + max(0, disturbed_object_count - 1))",
                    "note": "target object itself accounts for the first moved object",
                },
                "filters": self._filter_settings(),
                "object_config_zero_pose": copy.deepcopy(self.reference_object),
                "source_scene_count": len(self.scene_ids),
                "count": len(items),
            }
            atomic_json(json_path, {"metadata": metadata, "data": items})
            save_v2_npz(npz_path, metadata, items)
            self.save_review_state()
            self._set_status(f"최종 저장: {len(items)}개\n{json_path}\n{npz_path}")
            QtWidgets.QMessageBox.information(
                self, "Saved", f"{len(items)} grasps 저장 완료\n{json_path}\n{npz_path}"
            )
        except Exception as exc:
            self._set_status(f"저장 실패: {exc}")
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))


def main() -> None:
    app = QtWidgets.QApplication([])
    app.setApplicationName("Dataset v2 Grasp Filter & Review")
    app.setStyle("Fusion")
    font = QtGui.QFont(legacy.choose_ui_font(), 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(legacy.APP_STYLE)
    window = DatasetV2FilterGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
