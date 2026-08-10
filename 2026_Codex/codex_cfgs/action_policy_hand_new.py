"""Action policy for preset-based hand grippers.

This module is intentionally separate from ``action_policy_revised.py``.  The
legacy two/three-finger policy converts a requested width into joint targets,
whereas a hand pre-grasp already contains the exact START/END base transforms
and joint positions required for execution.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch


def _normalise_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quaternion_slerp(
    start: torch.Tensor, end: torch.Tensor, amount: torch.Tensor
) -> torch.Tensor:
    """Batch SLERP for Isaac's WXYZ quaternion convention."""
    start = _normalise_quaternion(start)
    end = _normalise_quaternion(end)
    dot = torch.sum(start * end, dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = torch.abs(dot).clamp(max=1.0)

    linear = _normalise_quaternion(start + amount * (end - start))
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1.0e-8)
    theta = theta_0 * amount
    spherical = (
        torch.sin(theta_0 - theta) / sin_theta_0 * start
        + torch.sin(theta) / sin_theta_0 * end
    )
    return torch.where(dot > 0.9995, linear, spherical)


class HandActionPolicy:
    """Batched hand-grasp controller following the legacy staged policy style."""

    def __init__(
        self,
        env_num,
        pre_grasp_data,
        conf_data,
        gripper_info,
        joint_names,
        joint_index,
        joint_pos,
        step_dt,
        device,
        frame_transformer,
        env_origin,
        debug=False,
        approach_duration_sec=1.0,
        lift_duration_sec=1.0,
        lift_height=0.20,
        grasp_joint_error_th=0.03,
        grasp_timeout_sec=2.0,
        early_fail_check_drop_ratio=0.25,
        early_fail_success_fraction=0.5,
    ):
        self.env_num = int(env_num)
        self.pre_grasp_data = pre_grasp_data
        self.obj_conf_data = conf_data["objects"]
        self.conf_data = conf_data
        self.gripper_info = gripper_info
        self.joint_names = list(joint_names)
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = float(step_dt)
        self.device = device
        self.frame_transformer = frame_transformer
        self.env_origin = env_origin
        self.debug = debug
        self._debug_carb = None
        self._debug_draw = None
        if self.debug:
            try:
                import carb
                from isaacsim.util.debug_draw import _debug_draw

                self._debug_carb = carb
                self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as error:
                print(f"HandActionPolicy > debug draw disabled: {error}")
                self.debug = False

        self.approach_steps = max(1, int(round(approach_duration_sec / self.step_dt)))
        self.lift_steps = max(1, int(round(lift_duration_sec / self.step_dt)))
        self.lift_height = float(lift_height)
        self.lift_success_threshold = 0.02
        self.early_fail_check_drop = max(
            0.01, self.lift_height * float(early_fail_check_drop_ratio)
        )
        self.early_fail_height_threshold = (
            self.lift_success_threshold * float(early_fail_success_fraction)
        )
        self.grasp_joint_error_th = float(grasp_joint_error_th)
        self.grasp_timeout_steps = max(1, int(round(grasp_timeout_sec / self.step_dt)))

        self.total_grasp_num = len(pre_grasp_data)
        self.current_grasp_num = 0
        self.output_list = []

        self._prepare_pre_grasps()

        # 0: approach START from above, 1: interpolate START->END, 2: lift,
        # 3: evaluate/done, 4: disabled.
        self.stage_num = torch.zeros(self.env_num, dtype=torch.long, device=self.device)
        self.stage_step = torch.zeros(self.env_num, dtype=torch.long, device=self.device)
        self.action_enable = torch.ones(self.env_num, dtype=torch.long, device=self.device)
        self.recorded = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.grasp_fail_idx = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)
        self.joint_command_dirty = torch.zeros(self.env_num, dtype=torch.bool, device=self.device)

        joint_count = len(self.joint_names)
        self.target_joint = torch.zeros((self.env_num, joint_count), dtype=torch.float, device=self.device)
        self.root_pose = torch.zeros((self.env_num, 7), dtype=torch.float, device=self.device)
        self.root_pose[:, 3] = 1.0
        self.target_class = torch.zeros(self.env_num, dtype=torch.long, device=self.device)
        self.assigned_pregrasp = torch.full(
            (self.env_num,), -1, dtype=torch.long, device=self.device
        )
        self.target_points = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)

        self.start_pos = torch.zeros((self.env_num, 3), dtype=torch.float, device=self.device)
        self.end_pos = torch.zeros_like(self.start_pos)
        self.start_quat = torch.zeros((self.env_num, 4), dtype=torch.float, device=self.device)
        self.end_quat = torch.zeros_like(self.start_quat)
        self.start_joint = torch.zeros_like(self.target_joint)
        self.end_joint = torch.zeros_like(self.target_joint)
        self.transition_steps = torch.ones(self.env_num, dtype=torch.long, device=self.device)
        self.use_smoothstep = torch.ones(self.env_num, dtype=torch.bool, device=self.device)
        self.platform_drop_offset = torch.zeros(self.env_num, dtype=torch.float, device=self.device)

        self.object_pos_org = torch.tensor(
            [item["translate"] for item in self.obj_conf_data],
            dtype=torch.float,
            device=self.device,
        )
        self.top_cam_config = next(
            (
                camera
                for camera in self.conf_data.get("cameras", [])
                if camera.get("name") == "top_view_camera"
            ),
            None,
        )

    def _clear_debug_draw(self) -> None:
        if not self.debug or self._debug_draw is None:
            return
        try:
            self._debug_draw.clear_points()
            self._debug_draw.clear_lines()
        except Exception as error:
            print(f"HandActionPolicy > debug draw clear failed: {error}")

    def _draw_points(self, points: torch.Tensor, color, size: float = 10.0) -> None:
        if not self.debug or self._debug_draw is None or len(points) == 0:
            return
        points_np = points.detach().cpu().numpy()
        self._debug_draw.draw_points(
            [self._debug_carb.Float3(*point.tolist()) for point in points_np],
            [color] * len(points_np),
            [float(size)] * len(points_np),
        )

    def _draw_bboxes(self, bboxes, colors, thickness: float = 3.0) -> None:
        if not self.debug or self._debug_draw is None or len(bboxes) == 0:
            return
        if isinstance(bboxes, torch.Tensor):
            bboxes_np = bboxes.detach().cpu().numpy()
        else:
            bboxes_np = np.asarray(bboxes, dtype=np.float64)
        for index, bbox in enumerate(bboxes_np):
            if not np.isfinite(bbox).all():
                continue
            color = colors[index] if isinstance(colors, list) else colors
            starts = bbox[[0, 1, 2, 3]]
            ends = bbox[[1, 2, 3, 0]]
            self._debug_draw.draw_lines(
                [self._debug_carb.Float3(*point.tolist()) for point in starts],
                [self._debug_carb.Float3(*point.tolist()) for point in ends],
                [color] * 4,
                [float(thickness)] * 4,
            )

    def _draw_active_pregrasps(self, env_ids: torch.Tensor) -> None:
        # Keep the scene clean while sampling.  Only successful final grasps
        # are drawn and left visible for inspection.
        return

    def _draw_grasp_results(self, env_ids: torch.Tensor, success: torch.Tensor) -> None:
        if not self.debug or len(env_ids) == 0:
            return
        valid = self.assigned_pregrasp[env_ids] >= 0
        env_ids = env_ids[valid]
        success = success[valid]
        env_ids = env_ids[success]
        if len(env_ids) == 0:
            return
        points = self.target_points[env_ids] + self.env_origin[env_ids]
        bboxes = self._assigned_bboxes_with_env_origin(env_ids)
        env_colors = [self._debug_carb.ColorRgba(0.0, 1.0, 0.0, 1.0)] * len(env_ids)
        bbox_counts = self._assigned_bbox_counts(env_ids)
        colors = [
            env_color
            for env_color, bbox_count in zip(env_colors, bbox_counts)
            for _ in range(bbox_count)
        ]
        self._draw_points(points, self._debug_carb.ColorRgba(0.0, 1.0, 0.0, 1.0), size=16.0)
        self._draw_bboxes(bboxes, colors, thickness=5.0)

    def _assigned_bbox_counts(self, env_ids: torch.Tensor) -> list[int]:
        counts: list[int] = []
        for env_id in env_ids.detach().cpu().tolist():
            source_id = int(self.assigned_pregrasp[env_id])
            if source_id < 0:
                counts.append(0)
            else:
                counts.append(len(self.total_grasp_bboxes[source_id]))
        return counts

    def _assigned_bboxes_with_env_origin(self, env_ids: torch.Tensor) -> np.ndarray:
        bboxes: list[np.ndarray] = []
        for env_id in env_ids.detach().cpu().tolist():
            source_id = int(self.assigned_pregrasp[env_id])
            if source_id < 0:
                continue
            origin = self.env_origin[env_id].detach().cpu().numpy()
            for bbox in self.total_grasp_bboxes[source_id]:
                bboxes.append(np.asarray(bbox, dtype=np.float64) + origin)
        if not bboxes:
            return np.zeros((0, 4, 3), dtype=np.float64)
        return np.asarray(bboxes, dtype=np.float64)

    @staticmethod
    def _require_pose(record: dict, phase: str) -> tuple[list[float], list[float]]:
        try:
            pose = record["target_base_tf"][phase]
            position = pose["position"]
            orientation = pose["orientation_wxyz"]
        except KeyError as error:
            raise KeyError(
                f"Hand pre-grasp is missing target_base_tf.{phase}.{error.args[0]}"
            ) from error
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError(f"Invalid target_base_tf.{phase} in hand pre-grasp")
        return position, orientation

    def _joint_vector(self, record: dict, phase: str) -> list[float]:
        try:
            values = record["target_joint_pos"][phase]
        except KeyError as error:
            raise KeyError(
                f"Hand pre-grasp is missing target_joint_pos.{phase}"
            ) from error
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise KeyError(
                f"Hand pre-grasp target_joint_pos.{phase} is missing joints: {missing}"
            )
        result = [float(values[name]) for name in self.joint_names]
        unit = str(record.get("joint_unit", "rad")).lower()
        if unit in {"deg", "degree", "degrees"}:
            result = [math.radians(value) for value in result]
        elif unit not in {"rad", "radian", "radians"}:
            raise ValueError(f"Unsupported hand joint unit: {unit!r}")
        return result

    @staticmethod
    def _rot_x_matrix(degrees: float) -> np.ndarray:
        radians = math.radians(float(degrees))
        return np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, math.cos(radians), -math.sin(radians), 0.0],
                [0.0, math.sin(radians), math.cos(radians), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _intrinsic_to_tf(intrinsic: list[list[float]] | np.ndarray) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        matrix[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        return matrix

    def _bbox_3d_to_2d_np(self, bbox_3d: np.ndarray) -> np.ndarray:
        if self.top_cam_config is None:
            raise KeyError("top_view_camera is missing from conf_data.cameras")

        bbox_3d = np.asarray(bbox_3d, dtype=np.float64)
        homogeneous = np.concatenate(
            (bbox_3d, np.ones((*bbox_3d.shape[:2], 1), dtype=np.float64)),
            axis=2,
        )
        camera_pose = np.asarray(self.top_cam_config["cam_poses"], dtype=np.float64)
        camera_tf = np.linalg.inv(camera_pose @ self._rot_x_matrix(180.0))
        projection_tf = self._intrinsic_to_tf(self.top_cam_config["intrinsic_isaac"]) @ camera_tf
        projected = np.einsum("ij,nkj->nki", projection_tf, homogeneous)
        pixels = projected[..., :2] / np.clip(projected[..., 2:3], 1.0e-8, None)
        return np.rint(pixels).astype(int)

    @staticmethod
    def _bbox_2d_summary(bbox_2d: np.ndarray) -> dict:
        bbox_2d = np.asarray(bbox_2d, dtype=np.float64)
        first_bbox = bbox_2d[0]
        center = bbox_2d.reshape(-1, 2).mean(axis=0)
        width = np.linalg.norm(first_bbox[1] - first_bbox[2])
        height = np.linalg.norm(first_bbox[0] - first_bbox[1])
        angle = math.atan2(
            first_bbox[2, 1] - first_bbox[1, 1],
            first_bbox[2, 0] - first_bbox[1, 0],
        )
        return {
            "bbox": bbox_2d.astype(int).tolist(),
            "center": np.rint(center).astype(int).tolist(),
            "width": int(round(float(width))),
            "height": int(round(float(height))),
            "angle": round(float(angle), 6),
        }

    def _prepare_pre_grasps(self) -> None:
        class_name_idx = {
            name: index
            for index, name in enumerate(self.frame_transformer.data.target_frame_names)
        }
        start_pos = []
        end_pos = []
        start_quat = []
        end_quat = []
        start_joint = []
        end_joint = []
        transition_steps = []
        smoothstep = []
        target_class = []
        target_points = []
        grasp_bbox = []

        for record in self.pre_grasp_data:
            gripper_type = str(record.get("gripper_type", "Hand")).lower()
            if gripper_type != "hand":
                raise ValueError(
                    f"Hand policy received non-hand pre-grasp type {gripper_type!r}"
                )
            start_position, start_orientation = self._require_pose(record, "start")
            end_position, end_orientation = self._require_pose(record, "end")
            target_object = record["target_object"]
            if target_object not in class_name_idx:
                raise KeyError(f"Target object {target_object!r} is not present in the scene")

            duration = float(record.get("transition", {}).get("duration_sec", 2.0))
            interpolation = str(
                record.get("transition", {}).get("interpolation", "smoothstep")
            ).lower()
            if interpolation not in {"linear", "smoothstep"}:
                raise ValueError(
                    f"Unsupported hand transition interpolation: {interpolation!r}"
                )

            start_pos.append(start_position)
            end_pos.append(end_position)
            start_quat.append(start_orientation)
            end_quat.append(end_orientation)
            start_joint.append(self._joint_vector(record, "start"))
            end_joint.append(self._joint_vector(record, "end"))
            transition_steps.append(max(1, int(round((duration * 0.5) / self.step_dt))))
            smoothstep.append(interpolation == "smoothstep")
            target_class.append(class_name_idx[target_object])
            target_points.append(record["target_points"])
            if "grasp_bbox" not in record:
                raise KeyError(
                    "Hand pre-grasp is missing grasp_bbox. "
                    "Regenerate pre_grasp with Collect_Hand_PreGrasp_dataset.py."
                )
            bbox = np.asarray(record["grasp_bbox"], dtype=np.float64)
            if bbox.ndim != 3 or bbox.shape[1:] != (4, 3):
                raise ValueError(
                    f"Invalid hand pre-grasp grasp_bbox shape: {bbox.shape}. "
                    "Expected (bbox_count, 4, 3)."
                )
            grasp_bbox.append(bbox.tolist())

        def tensor(values, *, dtype=torch.float):
            return torch.tensor(values, dtype=dtype, device=self.device)

        self.total_start_pos = tensor(start_pos)
        self.total_end_pos = tensor(end_pos)
        self.total_start_quat = _normalise_quaternion(tensor(start_quat))
        self.total_end_quat = _normalise_quaternion(tensor(end_quat))
        self.total_start_joint = tensor(start_joint)
        self.total_end_joint = tensor(end_joint)
        self.total_transition_steps = tensor(transition_steps, dtype=torch.long)
        self.total_smoothstep = tensor(smoothstep, dtype=torch.bool)
        self.total_class = tensor(target_class, dtype=torch.long)
        self.total_target_points = tensor(target_points)
        self.total_grasp_bboxes = grasp_bbox

    def _set_disabled(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.action_enable[env_ids] = 0
        self.stage_num[env_ids] = 4
        self.stage_step[env_ids] = 0
        self.assigned_pregrasp[env_ids] = -1
        self.joint_command_dirty[env_ids] = False
        self.platform_drop_offset[env_ids] = 0.0
        self.target_points[env_ids] = 0.0

    def reset(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        remaining = self.total_grasp_num - self.current_grasp_num
        if remaining <= 0:
            self._set_disabled(env_ids)
            return

        active_count = min(len(env_ids), remaining)
        active_envs = env_ids[:active_count]
        self._set_disabled(env_ids[active_count:])
        source_ids = torch.arange(
            self.current_grasp_num,
            self.current_grasp_num + active_count,
            dtype=torch.long,
            device=self.device,
        )

        self.action_enable[active_envs] = 1
        self.stage_num[active_envs] = 0
        self.stage_step[active_envs] = 0
        self.recorded[active_envs] = False
        self.grasp_fail_idx[active_envs] = False
        self.assigned_pregrasp[active_envs] = source_ids
        self.platform_drop_offset[active_envs] = 0.0

        self.start_pos[active_envs] = self.total_start_pos[source_ids]
        self.end_pos[active_envs] = self.total_end_pos[source_ids]
        self.start_quat[active_envs] = self.total_start_quat[source_ids]
        self.end_quat[active_envs] = self.total_end_quat[source_ids]
        self.start_joint[active_envs] = self.total_start_joint[source_ids]
        self.end_joint[active_envs] = self.total_end_joint[source_ids]
        self.transition_steps[active_envs] = self.total_transition_steps[source_ids]
        self.use_smoothstep[active_envs] = self.total_smoothstep[source_ids]
        self.target_class[active_envs] = self.total_class[source_ids]
        self.target_points[active_envs] = self.total_target_points[source_ids]

        self.root_pose[active_envs, :3] = self.start_pos[active_envs]
        self.root_pose[active_envs, 2] += self.lift_height
        self.root_pose[active_envs, 3:7] = self.start_quat[active_envs]
        self.target_joint[active_envs] = self.start_joint[active_envs]
        self.joint_command_dirty[active_envs] = True
        self.current_grasp_num += active_count
        self._draw_active_pregrasps(active_envs)

    def step(self) -> torch.Tensor:
        active = self.action_enable == 1

        stage0 = torch.where(active & (self.stage_num == 0))[0]
        if len(stage0):
            progress = (
                (self.stage_step[stage0] + 1).float() / self.approach_steps
            ).clamp(0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.root_pose[stage0, :3] = self.start_pos[stage0]
            self.root_pose[stage0, 2] += self.lift_height * (1.0 - smooth)
            self.root_pose[stage0, 3:7] = self.start_quat[stage0]
            self.stage_step[stage0] += 1
            done = stage0[self.stage_step[stage0] >= self.approach_steps]
            if len(done):
                self.stage_num[done] = 1
                self.stage_step[done] = 0
                # Hand grasp closing is delegated to the joint drive/controller.
                # Do not interpolate START->END every frame: issue the END pose
                # once on stage entry and let the USD/PhysX controller handle it.
                self.target_joint[done] = self.end_joint[done]
                self.joint_command_dirty[done] = True

        stage1 = torch.where(active & (self.stage_num == 1))[0]
        if len(stage1):
            progress = (
                (self.stage_step[stage1] + 1).float()
                / self.transition_steps[stage1].float()
            ).clamp(0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            amount = torch.where(self.use_smoothstep[stage1], smooth, progress)
            self.root_pose[stage1, :3] = torch.lerp(
                self.start_pos[stage1], self.end_pos[stage1], amount[:, None]
            )
            self.root_pose[stage1, 3:7] = _quaternion_slerp(
                self.start_quat[stage1], self.end_quat[stage1], amount[:, None]
            )
            self.target_joint[stage1] = self.end_joint[stage1]
            self.stage_step[stage1] += 1
            joint_error = torch.linalg.vector_norm(
                self.joint_pos[stage1][:, self.joint_index] - self.end_joint[stage1],
                dim=1,
            )
            root_motion_done = self.stage_step[stage1] >= self.transition_steps[stage1]
            joint_reached = joint_error < self.grasp_joint_error_th
            timed_out = self.stage_step[stage1] >= (
                self.transition_steps[stage1] + self.grasp_timeout_steps
            )
            done = stage1[root_motion_done & (joint_reached | timed_out)]
            if len(done):
                self.stage_num[done] = 2
                self.stage_step[done] = 0

        stage2 = torch.where(active & (self.stage_num == 2))[0]
        if len(stage2):
            progress = ((self.stage_step[stage2] + 1).float() / self.lift_steps).clamp(0.0, 1.0)
            self.root_pose[stage2, :3] = self.end_pos[stage2]
            self.root_pose[stage2, 3:7] = self.end_quat[stage2]
            self.target_joint[stage2] = self.end_joint[stage2]
            # Do not lift/teleport the hand root after closing.  Lower the
            # support platform instead so the grasp can be evaluated without
            # forcing the hand through contact constraints.
            self.platform_drop_offset[stage2] = -self.lift_height * progress
            self.stage_step[stage2] += 1
            done = stage2[self.stage_step[stage2] >= self.lift_steps]
            self.stage_num[done] = 3
            self.stage_step[done] = 0

        return self.target_joint

    def _target_height_delta(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_positions = self.frame_transformer.data.target_pos_source[env_ids]
        target_positions = object_positions[
            torch.arange(len(env_ids), device=self.device), self.target_class[env_ids]
        ]
        original_target_positions = self.object_pos_org[self.target_class[env_ids]]
        platform_drop = -self.platform_drop_offset[env_ids].clamp(max=0.0)
        height_delta = target_positions[:, 2] - (
            original_target_positions[:, 2] - platform_drop
        )
        return height_delta, target_positions, platform_drop

    def _mark_stage2_early_failures(self) -> None:
        idx = torch.where(
            (self.action_enable == 1)
            & (self.stage_num == 2)
            & (-self.platform_drop_offset.clamp(max=0.0) >= self.early_fail_check_drop)
        )[0]
        if len(idx) == 0:
            return

        height_delta, target_positions, platform_drop = self._target_height_delta(idx)
        fail = height_delta < self.early_fail_height_threshold
        fail_env_ids = idx[fail]
        if len(fail_env_ids) == 0:
            return

        self.grasp_fail_idx[fail_env_ids] = True
        self.stage_num[fail_env_ids] = 3
        self.stage_step[fail_env_ids] = 0

        if self.debug:
            for local_index, env_id in enumerate(fail_env_ids.detach().cpu().tolist()):
                source_id = int(self.assigned_pregrasp[int(env_id)])
                print(
                    "HandActionPolicy > early_fail "
                    f"env={int(env_id)} source={source_id} "
                    f"z_delta={float(height_delta[fail][local_index]):.5f} "
                    f"threshold={self.early_fail_height_threshold:.5f} "
                    f"target_z={float(target_positions[fail][local_index, 2]):.5f} "
                    f"platform_drop={float(platform_drop[fail][local_index]):.5f}"
                )

    def _append_success_records(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        
        object_positions = self.frame_transformer.data.target_pos_source[env_ids]
        height_delta, target_positions, platform_drop = self._target_height_delta(env_ids)
        # The hand no longer lifts the object.  A successful grasp keeps the
        # target near its original world height while the platform drops away.
        success = height_delta > self.lift_success_threshold
        if self.debug:
            for local_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
                source_id = int(self.assigned_pregrasp[int(env_id)])
                print(
                    "HandActionPolicy > grasp_result "
                    f"env={int(env_id)} source={source_id} "
                    f"success={bool(success[local_index])} "
                    f"z_delta={float(height_delta[local_index]):.5f} "
                    f"threshold={self.lift_success_threshold:.5f} "
                    f"target_z={float(target_positions[local_index, 2]):.5f} "
                    f"early_fail={bool(self.grasp_fail_idx[int(env_id)])} "
                    f"platform_drop={float(platform_drop[local_index]):.5f}"
                )
        self._draw_grasp_results(env_ids, success)

        compensated_object_positions = object_positions.clone()
        compensated_object_positions[:, :, 2] += platform_drop[:, None]
        moved = torch.linalg.vector_norm(
            compensated_object_positions - self.object_pos_org[None, :, :], dim=2
        ) > self.lift_success_threshold
        disturbed_counts = moved.sum(dim=1)

        for local_index in torch.where(success)[0].tolist():
            env_id = int(env_ids[local_index])
            source_id = int(self.assigned_pregrasp[env_id])
            source = self.pre_grasp_data[source_id]
            bbox_3d = np.asarray(source["grasp_bbox"], dtype=np.float64)
            bbox_2d = self._bbox_3d_to_2d_np(bbox_3d)
            record = {
                "bbox_2d": self._bbox_2d_summary(bbox_2d),
                "target_points": copy.deepcopy(source["target_points"]),
                "target_orientation": copy.deepcopy(source["target_orientation"]),
                "target_width": 0.0,
                "target_object": source["target_object"],
                "gripper_model": self.gripper_info["gripper_name"],
                "gripper_type": "Hand",
                "preset_name": source["preset_name"],
                "target_base_tf": copy.deepcopy(source["target_base_tf"]),
                "target_joint_pos": copy.deepcopy(source["target_joint_pos"]),
                "joint_unit": source.get("joint_unit", "rad"),
                "transition": copy.deepcopy(source.get("transition", {})),
                "disturbed_object_count": int(disturbed_counts[local_index]),
            }
            for optional_key in (
                "selected_heightmap_yaw",
                "first_contact_z",
                "safety_margin",
            ):
                if optional_key in source:
                    record[optional_key] = copy.deepcopy(source[optional_key])
            self.output_list.append(record)

    def get_done_idx(self) -> torch.Tensor:
        self.grasp_fail_idx[:] = False
        self._mark_stage2_early_failures()
        done = (self.stage_num == 3) & (self.action_enable == 1)
        new_done = done & ~self.recorded
        env_ids = torch.where(new_done)[0]
        self._append_success_records(env_ids)
        self.recorded[env_ids] = True
        return done
