"""Batched platform-free approach, grasp and direct stress-test policy."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from isaacsim.core.utils.rotations import euler_angles_to_quat

from . import calibration


APPROACH = 0
CLOSE = 1
STRESS = 2
DONE = 3
DISABLED = 4

FORCE_SCORE_WEIGHT = 0.3
PREGRASP_POSE_SCORE_WEIGHT = 0.4
STRESS_POSE_SCORE_WEIGHT = 0.2
CENTER_IN_POSE_WEIGHT = 0.5
ROTATION_IN_POSE_WEIGHT = 0.5
CENTER_SCORE_RANGE_M = 0.010
ROTATION_SCORE_RANGE_RAD = math.pi
CONTACT_AREA_SCORE_WEIGHT = 0.30
CONTACT_AREA_SCORE_RANGE_MM2 = 500.0
# 1.0 keeps each component linear. Values > 1 emphasize already-high scores;
# a logarithm is intentionally not used because log(total_score) cannot change rank.
SCORE_COMPONENT_POWER = 1.5


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((*vector.shape[:-1], 1), dtype=vector.dtype, device=vector.device)
    pure = torch.cat((zeros, vector), dim=-1)
    return quaternion_multiply(
        quaternion_multiply(quaternion, pure), quaternion_conjugate(quaternion)
    )[..., 1:]


def quaternion_slerp(
    start: torch.Tensor, end: torch.Tensor, amount: torch.Tensor
) -> torch.Tensor:
    start, end = normalize_quaternion(start), normalize_quaternion(end)
    dot = torch.sum(start * end, dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = torch.abs(dot).clamp(max=1.0)
    linear = normalize_quaternion(start + amount * (end - start))
    theta = torch.acos(dot)
    spherical = (
        torch.sin(theta * (1.0 - amount)) / torch.sin(theta).clamp_min(1e-8) * start
        + torch.sin(theta * amount) / torch.sin(theta).clamp_min(1e-8) * end
    )
    return torch.where(dot > 0.9995, linear, spherical)


def quaternion_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(
        torch.sum(normalize_quaternion(first) * normalize_quaternion(second), dim=-1)
    ).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize_quaternion(quaternion).unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


class GraspPolicy:
    """One state machine shared by calibrated fingers and preset hands."""

    def __init__(
        self,
        *,
        env_num: int,
        pre_grasp_data: list[dict],
        conf_data: dict,
        gripper_info: dict,
        joint_names: list[str],
        joint_index: torch.Tensor,
        joint_pos: torch.Tensor,
        step_dt: float,
        device: str,
        env_origin: torch.Tensor,
        object_initial_pos: torch.Tensor,
        object_initial_quat: torch.Tensor,
        contact_point_names: list[str] | None = None,
        approach_height: float = 0.12,
        approach_duration: float = 0.8,
        close_timeout: float = 1.5,
        stress_duration: float = 1.2,
        grasp_blocked_error: float = 0.003,
        min_contact_force: float = 0.12,
        contact_lost_duration: float = 0.10,
        contact_penetration_threshold: float = 0.005,
        pre_stress_object_motion_threshold: float = 0.005,
        apply_finger_z_hop: bool = False,
        grasp_record_mode: str = "attempt",
    ):
        if not pre_grasp_data:
            raise ValueError("pre_grasp_data is empty")
        if len(conf_data.get("objects", [])) != 1:
            raise ValueError("GraspPolicy requires exactly one scene object")
        self.env_num = int(env_num)
        self.records = pre_grasp_data
        self.conf_data = conf_data
        self.gripper = gripper_info
        self.gripper_type = str(gripper_info.get("type", "")).lower()
        self.is_hand = self.gripper_type == "hand"
        self.joint_names = list(joint_names)
        self.joint_index = joint_index
        self.joint_pos = joint_pos
        self.step_dt = float(step_dt)
        self.device = device
        self.env_origin = env_origin
        self.object_initial_pos = object_initial_pos.clone()
        self.object_initial_quat = normalize_quaternion(object_initial_quat.clone())
        self.contact_point_names = list(contact_point_names or [])
        self.approach_height = float(approach_height)
        self.approach_steps = max(1, round(approach_duration / self.step_dt))
        self.close_timeout_steps = max(1, round(close_timeout / self.step_dt))
        self.stress_steps = max(1, round(stress_duration / self.step_dt))
        self.grasp_blocked_error = float(grasp_blocked_error)
        self.min_contact_force = float(min_contact_force)
        self.contact_lost_steps = max(
            1, round(float(contact_lost_duration) / self.step_dt)
        )
        self.contact_penetration_threshold = float(contact_penetration_threshold)
        if self.contact_penetration_threshold < 0.0:
            raise ValueError("contact_penetration_threshold must be non-negative")
        self.pre_stress_object_motion_threshold = float(
            pre_stress_object_motion_threshold
        )
        if self.pre_stress_object_motion_threshold < 0.0:
            raise ValueError(
                "pre_stress_object_motion_threshold must be non-negative"
            )
        self.apply_finger_z_hop = bool(apply_finger_z_hop)
        self.grasp_record_mode = str(grasp_record_mode).strip().lower()
        valid_record_modes = {"attempt", "grasp_complete_restored"}
        if self.grasp_record_mode not in valid_record_modes:
            raise ValueError(
                f"grasp_record_mode must be one of {sorted(valid_record_modes)}, "
                f"got {grasp_record_mode!r}"
            )
        self.output_list: list[dict] = []
        self.attempt_history: list[dict] = []
        self.debug_success_records: list[tuple[int, dict]] = []
        self.total_grasp_num = len(pre_grasp_data)
        self.current_grasp_num = 0

        if not self.is_hand:
            calibration.validate(gripper_info)
            self.direct_prismatic = calibration.direct_prismatic(gripper_info)
            self.total_length = float(gripper_info["total_length"])
            self.max_width = float(gripper_info["width"])
        self._prepare_records()
        self._allocate()

    def _allocate(self) -> None:
        e, j = self.env_num, len(self.joint_names)
        kwargs = {"device": self.device}
        self.stage_num = torch.full((e,), DISABLED, dtype=torch.long, **kwargs)
        self.stage_step = torch.zeros(e, dtype=torch.long, **kwargs)
        self.action_enable = torch.ones(e, dtype=torch.long, **kwargs)
        self.assigned_pregrasp = torch.full((e,), -1, dtype=torch.long, **kwargs)
        self.recorded = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.failed_reason = [""] * e
        self.target_joint = torch.zeros((e, j), dtype=torch.float, **kwargs)
        self.open_joint = torch.zeros_like(self.target_joint)
        self.close_joint = torch.zeros_like(self.target_joint)
        self.start_joint = torch.zeros_like(self.target_joint)
        self.end_joint = torch.zeros_like(self.target_joint)
        self.root_pose = torch.zeros((e, 7), dtype=torch.float, **kwargs)
        self.root_pose[:, 3] = 1.0
        self.start_pos = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.end_pos = torch.zeros_like(self.start_pos)
        self.approach_axis = torch.zeros_like(self.start_pos)
        self.approach_axis[:, 2] = -1.0
        self.start_quat = torch.zeros((e, 4), dtype=torch.float, **kwargs)
        self.end_quat = torch.zeros_like(self.start_quat)
        self.transition_steps = torch.ones(e, dtype=torch.long, **kwargs)
        self.smooth_transition = torch.ones(e, dtype=torch.bool, **kwargs)
        self.open_z = torch.zeros(e, dtype=torch.float, **kwargs)
        self.applied_z_hop = torch.zeros(e, dtype=torch.float, **kwargs)
        self.close_error_previous = torch.zeros(e, dtype=torch.float, **kwargs)
        self.close_error_valid = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.close_stall_count = torch.zeros(e, dtype=torch.long, **kwargs)
        self.close_contact_seen = torch.zeros(e, dtype=torch.bool, **kwargs)
        self.contact_lost_count = torch.zeros(e, dtype=torch.long, **kwargs)
        self.reference_rel_pos = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.reference_rel_quat = torch.zeros((e, 4), dtype=torch.float, **kwargs)
        self.reference_rel_quat[:, 0] = 1.0
        self.relative_translation_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.relative_rotation_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.last_contact_force = torch.zeros(e, dtype=torch.float, **kwargs)
        self.max_test_force = torch.zeros(e, dtype=torch.float, **kwargs)
        self.force_direction = torch.zeros((e, 3), dtype=torch.float, **kwargs)
        self.force_direction[:, 2] = -1.0
        self.stress_survival = torch.zeros(e, dtype=torch.float, **kwargs)
        self.minimum_contact_separation = torch.full(
            (e,), torch.inf, dtype=torch.float, **kwargs
        )
        self.minimum_close_contact_separation = torch.full(
            (e,), torch.inf, dtype=torch.float, **kwargs
        )
        self.maximum_pre_stress_object_motion = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.pregrasp_rotation_error = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.stress_rotation_error = torch.zeros(
            e, dtype=torch.float, **kwargs
        )
        self.pregrasp_center_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.stress_center_error = torch.zeros(e, dtype=torch.float, **kwargs)
        self.stress_start_object_pos = torch.zeros(
            (e, 3), dtype=torch.float, **kwargs
        )
        self.stress_start_object_quat = torch.zeros(
            (e, 4), dtype=torch.float, **kwargs
        )
        self.stress_start_object_quat[:, 0] = 1.0
        self.grasp_contact_area = torch.zeros(e, dtype=torch.float, **kwargs)
        contact_channel_count = len(self.contact_point_names)
        self.grasp_contact_points_object_local = torch.full(
            (e, contact_channel_count, 3),
            torch.nan,
            dtype=torch.float,
            **kwargs,
        )
        self.grasp_contact_point_counts = torch.zeros(
            (e, contact_channel_count), dtype=torch.long, **kwargs
        )
        self.grasp_complete_joint = torch.zeros((e, j), dtype=torch.float, **kwargs)
        self.grasp_complete_width = torch.zeros(e, dtype=torch.float, **kwargs)

    def _prepare_records(self) -> None:
        starts, ends, start_quat, end_quat = [], [], [], []
        approach_axes = []
        start_joint, end_joint, transition, smooth = [], [], [], []
        for record in self.records:
            if record.get("target_object") != self.conf_data["objects"][0]["class"]:
                raise ValueError(
                    f"Pre-grasp target {record.get('target_object')!r} does not match scene object"
                )
            if self.is_hand:
                base = record.get("target_base_tf", {})
                joints = record.get("target_joint_pos", {})
                for phase in ("start", "end"):
                    if phase not in base or phase not in joints:
                        raise KeyError(f"Hand pre-grasp needs target_base_tf/joint_pos.{phase}")
                starts.append(base["start"]["position"])
                ends.append(base["end"]["position"])
                start_quat.append(base["start"]["orientation_wxyz"])
                end_quat.append(base["end"]["orientation_wxyz"])
                approach_axes.append([0.0, 0.0, -1.0])
                start_joint.append(self._hand_joint_vector(record, "start"))
                end_joint.append(self._hand_joint_vector(record, "end"))
                setting = record.get("transition", {})
                transition.append(max(1, round(float(setting.get("duration_sec", 1.0)) / self.step_dt)))
                smooth.append(str(setting.get("interpolation", "smoothstep")).lower() == "smoothstep")
            else:
                position = list(record["target_points"])
                orientation = list(record["target_orientation"])
                orientation[0] += 180.0
                quaternion = euler_angles_to_quat(orientation, degrees=True)
                if record.get("use_3d_approach_axis", False):
                    approach_axis = np.asarray(
                        record.get("approach_vector", []), dtype=np.float64
                    )
                    if approach_axis.shape != (3,) or np.linalg.norm(
                        approach_axis
                    ) < 1.0e-9:
                        roll, pitch, yaw = map(math.radians, orientation)
                        approach_axis = np.asarray(
                            [
                                math.cos(yaw) * math.sin(pitch) * math.cos(roll)
                                + math.sin(yaw) * math.sin(roll),
                                math.sin(yaw) * math.sin(pitch) * math.cos(roll)
                                - math.cos(yaw) * math.sin(roll),
                                math.cos(pitch) * math.cos(roll),
                            ],
                            dtype=np.float64,
                        )
                    approach_axis /= np.linalg.norm(approach_axis)
                    position = [
                        position[axis] - approach_axis[axis] * self.total_length
                        for axis in range(3)
                    ]
                else:
                    approach_axis = [0.0, 0.0, -1.0]
                    position[2] += self.total_length
                starts.append(position)
                ends.append(position)
                start_quat.append(quaternion)
                end_quat.append(quaternion)
                approach_axes.append(list(approach_axis))
                start_joint.append([0.0] * len(self.joint_names))
                end_joint.append([0.0] * len(self.joint_names))
                transition.append(1)
                smooth.append(True)
        tensor = lambda value, dtype=torch.float: torch.tensor(value, dtype=dtype, device=self.device)
        self.total_start_pos = tensor(starts)
        self.total_end_pos = tensor(ends)
        self.total_approach_axis = tensor(approach_axes)
        self.total_start_quat = normalize_quaternion(tensor(np.asarray(start_quat)))
        self.total_end_quat = normalize_quaternion(tensor(np.asarray(end_quat)))
        self.total_start_joint = tensor(start_joint)
        self.total_end_joint = tensor(end_joint)
        self.total_transition_steps = tensor(transition, torch.long)
        self.total_smooth = tensor(smooth, torch.bool)

    def _hand_joint_vector(self, record: dict, phase: str) -> list[float]:
        values = record["target_joint_pos"][phase]
        missing = [name for name in self.joint_names if name not in values]
        if missing:
            raise KeyError(f"Hand {phase} pose missing joints: {missing}")
        result = [float(values[name]) for name in self.joint_names]
        unit = str(record.get("joint_unit", "rad")).lower()
        if unit.startswith("deg"):
            result = [math.radians(value) for value in result]
        elif not unit.startswith("rad"):
            raise ValueError(f"Unsupported joint unit: {unit}")
        return result

    def _initial_joint(self, count: int) -> torch.Tensor:
        values = [float(self.gripper.get("init_joint_pos", {}).get(name, 0.0)) for name in self.joint_names]
        return torch.tensor(values, dtype=torch.float, device=self.device).repeat(count, 1)

    def _finger_open_joint(self, widths: torch.Tensor) -> torch.Tensor:
        if self.direct_prismatic:
            closed = float(self.gripper["close_joint_deg"])
            opened = float(self.gripper["open_joint_deg"])
            direction = 1.0 if opened >= closed else -1.0
            return closed + direction * widths[:, None] * 0.5
        degrees = calibration.polynomial(self.gripper["width_to_joint_coeffs"], widths)
        return degrees[:, None] * (math.pi / 180.0)

    def _finger_close_joint(self, count: int) -> torch.Tensor:
        value = float(self.gripper["close_joint_deg"])
        if not self.direct_prismatic:
            value = math.radians(value)
        return torch.full((count, len(self.joint_names)), value, device=self.device)

    def _finger_width(self, joint: torch.Tensor) -> torch.Tensor:
        """Convert the actual driven joint position at grasp completion to width."""
        driven = joint[:, 0]
        if self.direct_prismatic:
            closed = float(self.gripper["close_joint_deg"])
            opened = float(self.gripper["open_joint_deg"])
            direction = 1.0 if opened >= closed else -1.0
            width = (driven - closed) * (2.0 / direction)
        else:
            degrees = driven * (180.0 / math.pi)
            width = calibration.polynomial(
                self.gripper["joint_to_width_coeffs"], degrees
            )
        return width.clamp(0.0, self.max_width)

    def _finger_joint_z(self, joint: torch.Tensor) -> torch.Tensor:
        if self.direct_prismatic:
            return torch.zeros(len(joint), dtype=torch.float, device=self.device)
        degrees = joint[:, 0] * (180.0 / math.pi)
        return calibration.polynomial(self.gripper["joint_to_z_coeffs"], degrees)

    def _disable(self, env_ids: torch.Tensor) -> None:
        if not len(env_ids):
            return
        self.stage_num[env_ids] = DISABLED
        self.stage_step[env_ids] = 0
        self.action_enable[env_ids] = 0
        self.assigned_pregrasp[env_ids] = -1

    def reset(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        remaining = self.total_grasp_num - self.current_grasp_num
        if remaining <= 0:
            self._disable(env_ids)
            return
        count = min(len(env_ids), remaining)
        active = env_ids[:count]
        self._disable(env_ids[count:])
        source = torch.arange(
            self.current_grasp_num,
            self.current_grasp_num + count,
            dtype=torch.long,
            device=self.device,
        )
        self.action_enable[active] = 1
        self.assigned_pregrasp[active] = source
        self.stage_num[active] = APPROACH
        self.stage_step[active] = 0
        self.recorded[active] = False
        self.close_error_valid[active] = False
        self.close_stall_count[active] = 0
        self.close_contact_seen[active] = False
        self.contact_lost_count[active] = 0
        self.stress_survival[active] = 0.0
        self.max_test_force[active] = 0.0
        self.relative_translation_error[active] = 0.0
        self.relative_rotation_error[active] = 0.0
        self.minimum_contact_separation[active] = torch.inf
        self.minimum_close_contact_separation[active] = torch.inf
        self.maximum_pre_stress_object_motion[active] = 0.0
        self.pregrasp_rotation_error[active] = 0.0
        self.stress_rotation_error[active] = 0.0
        self.pregrasp_center_error[active] = 0.0
        self.stress_center_error[active] = 0.0
        self.stress_start_object_pos[active] = 0.0
        self.stress_start_object_quat[active] = 0.0
        self.stress_start_object_quat[active, 0] = 1.0
        self.grasp_contact_area[active] = 0.0
        self.grasp_contact_points_object_local[active] = torch.nan
        self.grasp_contact_point_counts[active] = 0
        self.grasp_complete_joint[active] = 0.0
        self.grasp_complete_width[active] = 0.0
        self.applied_z_hop[active] = 0.0
        for env_id in active.tolist():
            self.failed_reason[env_id] = ""
        direction = torch.randn((count, 3), dtype=torch.float, device=self.device)
        direction[:, 2] = -torch.abs(direction[:, 2])
        self.force_direction[active] = direction / torch.linalg.vector_norm(
            direction, dim=1, keepdim=True
        ).clamp_min(1e-8)

        self.start_pos[active] = self.total_start_pos[source]
        self.end_pos[active] = self.total_end_pos[source]
        self.approach_axis[active] = self.total_approach_axis[source]
        self.start_quat[active] = self.total_start_quat[source]
        self.end_quat[active] = self.total_end_quat[source]
        self.transition_steps[active] = self.total_transition_steps[source]
        self.smooth_transition[active] = self.total_smooth[source]
        if self.is_hand:
            self.start_joint[active] = self.total_start_joint[source]
            self.end_joint[active] = self.total_end_joint[source]
            self.open_joint[active] = self.start_joint[active]
            self.close_joint[active] = self.end_joint[active]
        else:
            widths = torch.tensor(
                [float(self.records[index]["target_width"]) for index in source.tolist()],
                dtype=torch.float,
                device=self.device,
            ).clamp(0.0, self.max_width)
            initial = self._initial_joint(count)
            self.start_joint[active] = initial
            self.open_joint[active] = self._finger_open_joint(widths)
            self.close_joint[active] = self._finger_close_joint(count)
            self.end_joint[active] = self.close_joint[active]
            self.open_z[active] = self._finger_joint_z(self.open_joint[active])
            self.start_pos[active, 2] += self.open_z[active]
            self.end_pos[active] = self.start_pos[active]
        self.target_joint[active] = self.start_joint[active]
        self.root_pose[active, :3] = self.start_pos[active]
        self.root_pose[active, 2] += self.approach_height
        self.root_pose[active, 3:7] = self.start_quat[active]
        self.current_grasp_num += count
        print(f"GraspPolicy > assigned={self.current_grasp_num}/{self.total_grasp_num}")

    def step(self) -> torch.Tensor:
        active = self.action_enable == 1
        approaching = torch.where(active & (self.stage_num == APPROACH))[0]
        if len(approaching):
            progress = ((self.stage_step[approaching] + 1) / self.approach_steps).clamp(0.0, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.root_pose[approaching, :3] = (
                self.start_pos[approaching]
                - self.approach_axis[approaching]
                * (self.approach_height * (1.0 - smooth))[:, None]
            )
            self.root_pose[approaching, 3:7] = self.start_quat[approaching]
            self.target_joint[approaching] = torch.lerp(
                self.start_joint[approaching], self.open_joint[approaching], smooth[:, None]
            )
            self.stage_step[approaching] += 1
            done = approaching[self.stage_step[approaching] >= self.approach_steps]
            self.stage_num[done] = CLOSE
            self.stage_step[done] = 0
            # Submit the final finger close target once at the transition.
            # The persistent PhysX drive keeps this target afterwards.
            if not self.is_hand and len(done):
                self.target_joint[done] = self.close_joint[done]

        closing = torch.where(active & (self.stage_num == CLOSE))[0]
        if len(closing):
            if self.is_hand:
                progress = ((self.stage_step[closing] + 1) / self.transition_steps[closing]).clamp(0.0, 1.0)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                amount = torch.where(self.smooth_transition[closing], smooth, progress)
                self.root_pose[closing, :3] = torch.lerp(
                    self.start_pos[closing], self.end_pos[closing], amount[:, None]
                )
                self.root_pose[closing, 3:7] = quaternion_slerp(
                    self.start_quat[closing], self.end_quat[closing], amount[:, None]
                )
                self.target_joint[closing] = torch.lerp(
                    self.start_joint[closing], self.end_joint[closing], amount[:, None]
                )
            else:
                if self.apply_finger_z_hop:
                    current = self.joint_pos[closing][:, self.joint_index]
                    z_hop = self._finger_joint_z(current) - self.open_z[closing]
                else:
                    z_hop = torch.zeros(
                        len(closing), dtype=torch.float, device=self.device
                    )
                self.applied_z_hop[closing] = z_hop
                self.root_pose[closing, :3] = (
                    self.end_pos[closing]
                    - self.approach_axis[closing] * z_hop[:, None]
                )
                self.root_pose[closing, 3:7] = self.end_quat[closing]
            self.stage_step[closing] += 1

        stressing = torch.where(active & (self.stage_num == STRESS))[0]
        if len(stressing):
            self.root_pose[stressing, :3] = self.end_pos[stressing]
            self.root_pose[stressing, 3:7] = self.end_quat[stressing]
            self.target_joint[stressing] = self.end_joint[stressing]
            self.stage_step[stressing] += 1
        return self.target_joint

    def _relative_pose(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_inv = quaternion_conjugate(normalize_quaternion(robot_quat))
        relative_pos = quaternion_rotate(robot_inv, object_pos - robot_pos)
        relative_quat = quaternion_multiply(robot_inv, normalize_quaternion(object_quat))
        return relative_pos, normalize_quaternion(relative_quat)

    def _start_stress(
        self,
        env_ids: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        contact_area: torch.Tensor,
        contact_points_w: torch.Tensor,
        contact_point_counts: torch.Tensor,
    ) -> None:
        if not len(env_ids):
            return
        rel_pos, rel_quat = self._relative_pose(
            object_pos[env_ids], object_quat[env_ids], robot_pos[env_ids], robot_quat[env_ids]
        )
        self.reference_rel_pos[env_ids] = rel_pos
        self.reference_rel_quat[env_ids] = rel_quat
        grasped_object_pos = object_pos[env_ids]
        grasped_object_quat = normalize_quaternion(object_quat[env_ids])
        self.stress_start_object_pos[env_ids] = grasped_object_pos
        self.stress_start_object_quat[env_ids] = grasped_object_quat
        self.pregrasp_center_error[env_ids] = torch.linalg.vector_norm(
            grasped_object_pos - self.object_initial_pos[env_ids], dim=1
        )
        self.pregrasp_rotation_error[env_ids] = quaternion_angle(
            grasped_object_quat, self.object_initial_quat[env_ids]
        )
        self.stress_center_error[env_ids] = 0.0
        self.stress_rotation_error[env_ids] = 0.0
        self.grasp_contact_area[env_ids] = contact_area[env_ids]
        if self.grasp_contact_points_object_local.shape[1] > 0:
            points_w = contact_points_w[env_ids]
            counts = contact_point_counts[env_ids]
            object_q = normalize_quaternion(object_quat[env_ids])[:, None, :]
            object_q = object_q.expand(-1, points_w.shape[1], -1)
            relative_w = points_w - object_pos[env_ids, None, :]
            points_local = quaternion_rotate(
                quaternion_conjugate(object_q), relative_w
            )
            valid = (counts > 0) & torch.all(torch.isfinite(points_w), dim=-1)
            self.grasp_contact_points_object_local[env_ids] = torch.where(
                valid[:, :, None], points_local, torch.nan
            )
            self.grasp_contact_point_counts[env_ids] = counts
        completed_joint = self.joint_pos[env_ids][:, self.joint_index]
        self.grasp_complete_joint[env_ids] = completed_joint
        if not self.is_hand:
            self.grasp_complete_width[env_ids] = self._finger_width(completed_joint)
        # Pin the stress test to the pose PhysX actually reached. Root motion
        # is speed-limited, so this can differ slightly from the command pose.
        self.end_pos[env_ids] = robot_pos[env_ids] - self.env_origin[env_ids]
        self.end_quat[env_ids] = robot_quat[env_ids]
        self.root_pose[env_ids, :3] = self.end_pos[env_ids]
        self.root_pose[env_ids, 3:7] = self.end_quat[env_ids]
        self.end_joint[env_ids] = self.target_joint[env_ids]
        self.stage_num[env_ids] = STRESS
        self.stage_step[env_ids] = 0
        self.contact_lost_count[env_ids] = 0

    def _finish_without_record(self, env_ids: torch.Tensor, reason: str) -> None:
        env_ids = env_ids[~self.recorded[env_ids]]
        for env_id in env_ids.tolist():
            self.failed_reason[env_id] = reason
            self.attempt_history.append(
                {
                    "source_pregrasp_index": int(self.assigned_pregrasp[env_id]),
                    "result": reason,
                    "failure_stage": int(self.stage_num[env_id]),
                    "failure_stage_step": int(self.stage_step[env_id]),
                    "relative_translation_error_m": float(
                        self.relative_translation_error[env_id]
                    ),
                    "relative_rotation_error_deg": math.degrees(
                        float(self.relative_rotation_error[env_id])
                    ),
                    "contact_force_n": float(self.last_contact_force[env_id]),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": round(
                        float(self.maximum_pre_stress_object_motion[env_id]), 7
                    ),
                    "finger_z_hop_m": round(float(self.applied_z_hop[env_id]), 7),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                }
            )
        self.stage_num[env_ids] = DONE
        self.stage_step[env_ids] = 0
        self.recorded[env_ids] = True

    def observe(
        self,
        *,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        contact_force: torch.Tensor,
        minimum_contact_separation: torch.Tensor,
        contact_area: torch.Tensor,
        contact_points_w: torch.Tensor,
        contact_point_counts: torch.Tensor,
        applied_force: torch.Tensor,
    ) -> torch.Tensor:
        """Advance contact-dependent stages and return episode-done mask."""
        self.joint_pos = joint_pos
        self.last_contact_force = contact_force
        self.max_test_force = torch.maximum(self.max_test_force, applied_force)
        finite_separation = torch.isfinite(minimum_contact_separation)
        closing_separation = (
            (self.action_enable == 1)
            & (self.stage_num == CLOSE)
            & finite_separation
        )
        self.minimum_close_contact_separation = torch.where(
            closing_separation,
            torch.minimum(
                self.minimum_close_contact_separation,
                minimum_contact_separation,
            ),
            self.minimum_close_contact_separation,
        )
        self.close_contact_seen |= (
            (self.action_enable == 1)
            & (self.stage_num == CLOSE)
            & (contact_force >= self.min_contact_force)
        )
        self.minimum_contact_separation = torch.where(
            finite_separation,
            torch.minimum(
                self.minimum_contact_separation, minimum_contact_separation
            ),
            self.minimum_contact_separation,
        )
        penetrating = torch.where(
            (self.action_enable == 1)
            & finite_separation
            & (
                minimum_contact_separation
                < -self.contact_penetration_threshold
            )
        )[0]
        self._finish_without_record(penetrating, "penetration")
        pre_stress = (
            (self.action_enable == 1)
            & ((self.stage_num == APPROACH) | (self.stage_num == CLOSE))
        )
        object_motion = torch.linalg.vector_norm(
            object_pos - self.object_initial_pos, dim=1
        )
        self.maximum_pre_stress_object_motion = torch.where(
            pre_stress,
            torch.maximum(self.maximum_pre_stress_object_motion, object_motion),
            self.maximum_pre_stress_object_motion,
        )
        moved_before_stress = torch.where(
            pre_stress
            & (object_motion > self.pre_stress_object_motion_threshold)
        )[0]
        self._finish_without_record(
            moved_before_stress, "pre_stress_object_motion"
        )
        closing = torch.where((self.action_enable == 1) & (self.stage_num == CLOSE))[0]
        if len(closing):
            current = joint_pos[closing][:, self.joint_index]
            error = torch.linalg.vector_norm(self.close_joint[closing] - current, dim=1)
            if self.is_hand:
                motion_done = self.stage_step[closing] >= self.transition_steps[closing]
                timed_out = self.stage_step[closing] >= (
                    self.transition_steps[closing] + self.close_timeout_steps
                )
                ready = motion_done & ((error < 0.03) | timed_out)
                # Net contact force can be zero for one frame when opposing
                # fingertip forces cancel. The direct stress test verifies
                # retention, so do not reject from this aggregate value alone.
                grasped = ready
                empty = torch.zeros_like(ready)
            else:
                delta = torch.abs(error - self.close_error_previous[closing])
                stalled = self.close_error_valid[closing] & (delta < 2.0e-4) & (
                    error > self.grasp_blocked_error
                )
                self.close_stall_count[closing] = torch.where(
                    stalled,
                    self.close_stall_count[closing] + 1,
                    torch.zeros_like(self.close_stall_count[closing]),
                )
                self.close_error_previous[closing] = error
                self.close_error_valid[closing] = True
                waited = self.stage_step[closing] >= max(5, int(0.08 / self.step_dt))
                timed_out = self.stage_step[closing] >= self.close_timeout_steps
                blocked = error > self.grasp_blocked_error
                contact_confirmed = self.close_contact_seen[closing]
                grasped = waited & blocked & contact_confirmed & (
                    (self.close_stall_count[closing] >= 4) | timed_out
                )
                # A mimic/limited articulation can stall a few milliradians
                # before its target even with no object present.  Joint error
                # alone must never start the external-force test.
                empty = timed_out & (~blocked | ~contact_confirmed)
            self._start_stress(
                closing[grasped],
                object_pos,
                object_quat,
                robot_pos,
                robot_quat,
                contact_area,
                contact_points_w,
                contact_point_counts,
            )
            self._finish_without_record(closing[empty], "empty_close")

        tested = torch.where(
            (self.action_enable == 1) & (self.stage_num == STRESS)
        )[0]
        if len(tested):
            rel_pos, rel_quat = self._relative_pose(
                object_pos[tested], object_quat[tested], robot_pos[tested], robot_quat[tested]
            )
            position_error = torch.linalg.vector_norm(
                rel_pos - self.reference_rel_pos[tested], dim=1
            )
            rotation_error = quaternion_angle(rel_quat, self.reference_rel_quat[tested])
            self.relative_translation_error[tested] = position_error
            self.relative_rotation_error[tested] = rotation_error
            self.stress_center_error[tested] = torch.linalg.vector_norm(
                object_pos[tested] - self.stress_start_object_pos[tested], dim=1
            )
            self.stress_rotation_error[tested] = quaternion_angle(
                object_quat[tested], self.stress_start_object_quat[tested]
            )
            lost_contact = contact_force[tested] < self.min_contact_force
            self.contact_lost_count[tested] = torch.where(
                lost_contact,
                self.contact_lost_count[tested] + 1,
                torch.zeros_like(self.contact_lost_count[tested]),
            )
            contact_lost = (
                self.contact_lost_count[tested] >= self.contact_lost_steps
            )
            stress_envs = tested
            if len(stress_envs):
                progress = (self.stage_step[stress_envs].float() / self.stress_steps).clamp(0, 1)
                self.stress_survival[stress_envs] = progress
                failed_contact = stress_envs[contact_lost]
                # Pose drift is a quality score only. A grasp fails the stress
                # test only after contact is continuously lost for the grace
                # duration. If the final frame briefly loses contact, keep the
                # test alive until contact recovers or the grace time expires.
                contact_confirmed = self.contact_lost_count[stress_envs] == 0
                completed = stress_envs[
                    ~contact_lost
                    & contact_confirmed
                    & (self.stage_step[stress_envs] >= self.stress_steps)
                ]
                self._record(failed_contact, "contact_lost")
                self.stress_survival[completed] = 1.0
                self._record(completed, "completed")

        return (self.action_enable == 1) & (self.stage_num == DONE)

    def _grasp_boxes(
        self,
        source: dict,
        *,
        width: float | None = None,
        center: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return every world-frame contact bbox for this gripper grasp."""
        if center is None:
            center = np.asarray(source["target_points"], dtype=np.float64)
        else:
            center = np.asarray(center, dtype=np.float64)
        yaw = math.radians(float(source["target_orientation"][2]))
        if self.is_hand:
            boxes = np.asarray(source.get("grasp_bbox", []), dtype=np.float64)
            if boxes.ndim != 3 or boxes.shape[1:] != (4, 3) or len(boxes) == 0:
                raise ValueError(
                    "Hand pre-grasp grasp_bbox must have shape (N, 4, 3), "
                    f"got {boxes.shape}"
                )
            return boxes

        half_height = float(self.gripper["height"]) * 0.5
        attempted_width = (
            float(source["target_width"]) if width is None else float(width)
        )
        if self.gripper_type.startswith("finger2"):
            boxes = np.asarray(
                [[
                    [half_height, -attempted_width * 0.5, 0.0],
                    [-half_height, -attempted_width * 0.5, 0.0],
                    [-half_height, attempted_width * 0.5, 0.0],
                    [half_height, attempted_width * 0.5, 0.0],
                ]],
                dtype=np.float64,
            )
        elif self.gripper_type.startswith("finger3"):
            infos = self.gripper.get("finger_bbox_info")
            if not isinstance(infos, list) or not infos:
                raise ValueError(
                    f"{self.gripper.get('gripper_name')} requires finger_bbox_info"
                )
            base = np.asarray(
                [
                    [half_height, -attempted_width * 0.5, 0.0],
                    [-half_height, -attempted_width * 0.5, 0.0],
                    [-half_height, 0.0, 0.0],
                    [half_height, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            boxes = []
            for info in infos:
                finger_yaw = math.radians(float(info["rot"]))
                finger_rotation = np.asarray(
                    [
                        [math.cos(finger_yaw), -math.sin(finger_yaw)],
                        [math.sin(finger_yaw), math.cos(finger_yaw)],
                    ],
                    dtype=np.float64,
                )
                box = base.copy()
                box[:, :2] = (
                    box[:, :2] @ finger_rotation.T
                    + np.asarray(info["pos"][:2], dtype=np.float64)
                )
                boxes.append(box)
            boxes = np.asarray(boxes, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported gripper type for bbox: {self.gripper_type}")

        grasp_rotation = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw)],
                [math.sin(yaw), math.cos(yaw)],
            ],
            dtype=np.float64,
        )
        boxes[:, :, :2] = boxes[:, :, :2] @ grasp_rotation.T
        boxes += center[None, None, :]
        return boxes

    def _enclosing_grasp_box(
        self,
        source: dict,
        boxes: np.ndarray,
        *,
        center: np.ndarray | None = None,
    ) -> np.ndarray:
        """One compatibility rectangle enclosing all multi-finger bboxes."""
        if len(boxes) == 1:
            return boxes[0]
        if center is None:
            center = np.asarray(source["target_points"], dtype=np.float64)
        else:
            center = np.asarray(center, dtype=np.float64)
        yaw = math.radians(float(source["target_orientation"][2]))
        x_axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        y_axis = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0])
        relative = boxes.reshape(-1, 3) - center
        x_values = relative @ x_axis
        y_values = relative @ y_axis
        minimum_x, maximum_x = float(x_values.min()), float(x_values.max())
        minimum_y, maximum_y = float(y_values.min()), float(y_values.max())
        return np.asarray(
            [
                center + maximum_x * x_axis + minimum_y * y_axis,
                center + minimum_x * x_axis + minimum_y * y_axis,
                center + minimum_x * x_axis + maximum_y * y_axis,
                center + maximum_x * x_axis + maximum_y * y_axis,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _transform_points_np(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        homogeneous = np.concatenate(
            (points.reshape(-1, 3), np.ones((points.size // 3, 1))), axis=1
        )
        return (homogeneous @ matrix.T)[:, :3].reshape(points.shape)

    @staticmethod
    def _pose_matrix_np(position: torch.Tensor, quaternion: torch.Tensor) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = (
            quaternion_to_matrix(quaternion).detach().cpu().numpy().astype(np.float64)
        )
        matrix[:3, 3] = position.detach().cpu().numpy().astype(np.float64)
        return matrix

    @staticmethod
    def _matrix_to_quaternion_np(rotation: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a canonical WXYZ quaternion."""
        rotation = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = math.sqrt(
                    1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
                ) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = math.sqrt(
                    1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
                ) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = math.sqrt(
                    1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
                ) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
        return quaternion if quaternion[0] >= 0.0 else -quaternion

    @staticmethod
    def _matrix_to_euler_xyz_deg_np(rotation: np.ndarray) -> np.ndarray:
        """Return XYZ Euler angles for R = Rz(yaw) Ry(pitch) Rx(roll)."""
        rotation = np.asarray(rotation, dtype=np.float64)
        horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
        if horizontal > 1.0e-9:
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            pitch = math.atan2(-float(rotation[2, 0]), horizontal)
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
            pitch = math.atan2(-float(rotation[2, 0]), horizontal)
            yaw = 0.0
        return np.degrees(np.asarray([roll, pitch, yaw], dtype=np.float64))

    def _orientation_fields(self, matrix: np.ndarray) -> dict:
        rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
        grasp_rpy = self._matrix_to_euler_xyz_deg_np(rotation)
        if self.is_hand:
            target_rotation = rotation
        else:
            # Finger pregrasp target_orientation excludes the fixed 180-degree
            # base flip added before commanding the articulation.
            target_rotation = rotation @ np.diag([1.0, -1.0, -1.0])
        return {
            "target_orientation": self._matrix_to_euler_xyz_deg_np(target_rotation),
            "grasp_orientation_wxyz": self._matrix_to_quaternion_np(rotation),
            "grasp_orientation_rpy_deg": grasp_rpy,
            "gripper_yaw_deg": float(grasp_rpy[2]),
        }

    def _contact_point_records(self, env_id: int) -> list[dict]:
        """Serialize the grasp-complete mean of each contact-body channel."""
        points = (
            self.grasp_contact_points_object_local[env_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        counts = self.grasp_contact_point_counts[env_id].detach().cpu().tolist()
        return [
            {
                "sensor": name,
                "position": point.round(7).tolist(),
                "sample_count": int(count),
            }
            for name, point, count in zip(self.contact_point_names, points, counts)
            if int(count) > 0 and np.all(np.isfinite(point))
        ]

    @staticmethod
    def _apply_contact_height(
        boxes: np.ndarray,
        box: np.ndarray,
        contact_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Move bbox planes to mean contact Z without changing orientation."""
        points = np.asarray(contact_points, dtype=np.float64).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            return boxes, box, None
        mean_z = float(points[:, 2].mean())
        adjusted_boxes = np.asarray(boxes, dtype=np.float64).copy()
        adjusted_box = np.asarray(box, dtype=np.float64).copy()
        # A restored grasp plane may be tilted. Flattening every vertex to one
        # world-Z value destroys its normal and breaks bbox/approach alignment.
        # Translate each representation as a rigid plane instead.
        adjusted_boxes[:, :, 2] += mean_z - float(
            adjusted_boxes[:, :, 2].mean()
        )
        adjusted_box[:, 2] += mean_z - float(adjusted_box[:, 2].mean())
        return adjusted_boxes, adjusted_box, mean_z

    @staticmethod
    def _hand_descriptor_matrix(
        matrix: np.ndarray,
        boxes: np.ndarray,
        approach: np.ndarray,
    ) -> np.ndarray:
        """Build a hand grasp frame whose +Z is normal to the saved bboxes."""
        result = np.asarray(matrix, dtype=np.float64).copy()
        z_axis = np.asarray(approach, dtype=np.float64)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)

        # Hand BBoxes are generated in the world-aligned height-map TCP plane.
        # Use one in-plane edge for X so yaw is retained while wrist pitch/roll
        # cannot tilt the descriptor's approach axis away from the BBoxes.
        x_axis = None
        for candidate in np.asarray(boxes, dtype=np.float64):
            for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                edge = candidate[second] - candidate[first]
                edge -= np.dot(edge, z_axis) * z_axis
                norm = float(np.linalg.norm(edge))
                if norm > 1.0e-9:
                    x_axis = edge / norm
                    break
            if x_axis is not None:
                break
        if x_axis is None:
            x_axis = result[:3, 0] - np.dot(result[:3, 0], z_axis) * z_axis
            norm = float(np.linalg.norm(x_axis))
            if norm <= 1.0e-9:
                reference = np.asarray([1.0, 0.0, 0.0])
                if abs(float(np.dot(reference, z_axis))) > 0.9:
                    reference = np.asarray([0.0, 1.0, 0.0])
                x_axis = reference - np.dot(reference, z_axis) * z_axis
                norm = float(np.linalg.norm(x_axis))
            x_axis /= max(norm, 1.0e-12)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1.0e-12)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
        result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        return result

    def _record_geometry(self, env_id: int, source: dict) -> dict:
        """Build either attempted or grasp-complete/object-restored geometry."""
        force_direction = (
            self.force_direction[env_id].detach().cpu().numpy().astype(np.float64)
        )
        initial_object_pos = self.object_initial_pos[env_id] - self.env_origin[env_id]
        initial_object_matrix = self._pose_matrix_np(
            initial_object_pos, self.object_initial_quat[env_id]
        )
        contact_points_local = (
            self.grasp_contact_points_object_local[env_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        valid_contact_points_local = contact_points_local[
            np.all(np.isfinite(contact_points_local), axis=1)
        ]
        contact_points_record = self._transform_points_np(
            initial_object_matrix, valid_contact_points_local
        )
        if self.grasp_record_mode == "attempt":
            boxes = self._grasp_boxes(source)
            box = self._enclosing_grasp_box(source, boxes)
            boxes, box, bbox_contact_mean_z = self._apply_contact_height(
                boxes, box, contact_points_record
            )
            matrix = self._pose_matrix_np(
                torch.tensor(source["target_points"], device=self.device),
                self.end_quat[env_id],
            )
            if self.is_hand:
                approach = (
                    self.approach_axis[env_id]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                matrix = self._hand_descriptor_matrix(matrix, boxes, approach)
            geometry = {
                "grasp_boxes": boxes,
                "grasp_box": box,
                "grasp_mat": matrix,
                "target_points": np.asarray(source["target_points"], dtype=np.float64),
                "target_width": float(source.get("target_width", 0.0)),
                "force_direction": force_direction,
                "restore_transform": np.eye(4, dtype=np.float64),
                "contact_points_object_local": self._contact_point_records(env_id),
                "bbox_contact_mean_z": bbox_contact_mean_z,
            }
            geometry.update(self._orientation_fields(matrix))
            # Preserve exact legacy Euler values in attempt mode.
            geometry["target_orientation"] = np.asarray(
                source["target_orientation"], dtype=np.float64
            )
            return geometry

        robot_matrix = self._pose_matrix_np(
            self.end_pos[env_id], self.end_quat[env_id]
        )
        if self.is_hand:
            boxes_at_completion = self._grasp_boxes(source)
            grasp_center = boxes_at_completion.reshape(-1, 3).mean(axis=0)
            recorded_width = float(source.get("target_width", 0.0))
        else:
            # Restored geometry keeps the width of the original pregrasp attempt.
            # Actual completion width is retained in quality diagnostics only.
            recorded_width = float(source.get("target_width", 0.0))
            joint = self.grasp_complete_joint[env_id : env_id + 1]
            finger_z = float(self._finger_joint_z(joint)[0])
            grasp_center = (
                robot_matrix[:3, 3]
                + robot_matrix[:3, 2] * (self.total_length + finger_z)
            )
            boxes_at_completion = self._grasp_boxes(
                source, width=recorded_width, center=grasp_center
            )
        box_at_completion = self._enclosing_grasp_box(
            source, boxes_at_completion, center=grasp_center
        )
        grasp_matrix_at_completion = robot_matrix.copy()
        grasp_matrix_at_completion[:3, 3] = grasp_center

        complete_object_pos = (
            self.stress_start_object_pos[env_id] - self.env_origin[env_id]
        )
        complete_object_matrix = self._pose_matrix_np(
            complete_object_pos, self.stress_start_object_quat[env_id]
        )
        restore = initial_object_matrix @ np.linalg.inv(complete_object_matrix)
        restored_matrix = restore @ grasp_matrix_at_completion
        restored_boxes = self._transform_points_np(restore, boxes_at_completion)
        restored_box = self._transform_points_np(restore, box_at_completion)
        restored_boxes, restored_box, bbox_contact_mean_z = (
            self._apply_contact_height(
                restored_boxes, restored_box, contact_points_record
            )
        )
        if self.is_hand:
            completed_approach = (
                self.approach_axis[env_id]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            restored_approach = restore[:3, :3] @ completed_approach
            restored_matrix = self._hand_descriptor_matrix(
                restored_matrix, restored_boxes, restored_approach
            )
        restored_force = restore[:3, :3] @ force_direction
        restored_force /= max(float(np.linalg.norm(restored_force)), 1.0e-12)
        geometry = {
            "grasp_boxes": restored_boxes,
            "grasp_box": restored_box,
            "grasp_mat": restored_matrix,
            "target_points": restored_matrix[:3, 3],
            "target_width": recorded_width,
            "force_direction": restored_force,
            "restore_transform": restore,
            "initial_object_matrix": initial_object_matrix,
            "complete_object_matrix": complete_object_matrix,
            "contact_points_object_local": self._contact_point_records(env_id),
            "bbox_contact_mean_z": bbox_contact_mean_z,
        }
        geometry.update(self._orientation_fields(restored_matrix))
        return geometry

    def _minimum_separation_value(self, env_id: int) -> float | None:
        value = float(self.minimum_contact_separation[env_id])
        return round(value, 7) if math.isfinite(value) else None

    def _minimum_close_separation_value(self, env_id: int) -> float | None:
        value = float(self.minimum_close_contact_separation[env_id])
        return round(value, 7) if math.isfinite(value) else None

    @staticmethod
    def _penetration_mm(separation_m: float | None) -> float | None:
        if separation_m is None:
            return None
        return round(max(0.0, -float(separation_m)) * 1000.0, 4)

    def _record(self, env_ids: torch.Tensor, result: str) -> None:
        for env_id in env_ids.tolist():
            if self.recorded[env_id]:
                continue
            source_id = int(self.assigned_pregrasp[env_id])
            source = self.records[source_id]
            pregrasp_rotation_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.pregrasp_rotation_error[env_id])
                    / ROTATION_SCORE_RANGE_RAD,
                ),
            )
            stress_rotation_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.stress_rotation_error[env_id])
                    / ROTATION_SCORE_RANGE_RAD,
                ),
            )
            survival = min(
                1.0,
                max(0.0, float(self.stress_survival[env_id])),
            )
            pregrasp_center_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.pregrasp_center_error[env_id])
                    / CENTER_SCORE_RANGE_M,
                ),
            )
            stress_center_score = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - float(self.stress_center_error[env_id])
                    / CENTER_SCORE_RANGE_M,
                ),
            )
            pregrasp_pose_score = (
                CENTER_IN_POSE_WEIGHT * pregrasp_center_score
                + ROTATION_IN_POSE_WEIGHT * pregrasp_rotation_score
            )
            stress_pose_score = (
                CENTER_IN_POSE_WEIGHT * stress_center_score
                + ROTATION_IN_POSE_WEIGHT * stress_rotation_score
            )
            contact_area_score = min(
                1.0,
                max(
                    0.0,
                    float(self.grasp_contact_area[env_id]) * 1.0e6
                    / CONTACT_AREA_SCORE_RANGE_MM2,
                ),
            )
            score_weight_sum = (
                FORCE_SCORE_WEIGHT
                + PREGRASP_POSE_SCORE_WEIGHT
                + STRESS_POSE_SCORE_WEIGHT
                + CONTACT_AREA_SCORE_WEIGHT
            )
            if score_weight_sum <= 0.0:
                raise ValueError("At least one score weight must be positive")
            score = (
                FORCE_SCORE_WEIGHT * survival**SCORE_COMPONENT_POWER
                + PREGRASP_POSE_SCORE_WEIGHT
                * pregrasp_pose_score**SCORE_COMPONENT_POWER
                + STRESS_POSE_SCORE_WEIGHT
                * stress_pose_score**SCORE_COMPONENT_POWER
                + CONTACT_AREA_SCORE_WEIGHT
                * contact_area_score**SCORE_COMPONENT_POWER
            ) / score_weight_sum
            # Compatibility field for readers that still visualize rotation only.
            rotation_score = (
                PREGRASP_POSE_SCORE_WEIGHT * pregrasp_rotation_score
                + STRESS_POSE_SCORE_WEIGHT * stress_rotation_score
            ) / (
                PREGRASP_POSE_SCORE_WEIGHT
                + STRESS_POSE_SCORE_WEIGHT
            )
            pose_score = (
                PREGRASP_POSE_SCORE_WEIGHT * pregrasp_pose_score
                + STRESS_POSE_SCORE_WEIGHT * stress_pose_score
            ) / (PREGRASP_POSE_SCORE_WEIGHT + STRESS_POSE_SCORE_WEIGHT)
            geometry = self._record_geometry(env_id, source)
            grasp_boxes = geometry["grasp_boxes"]
            grasp_box = geometry["grasp_box"]
            matrix = geometry["grasp_mat"]
            approach_vector = matrix[:3, 2].copy()
            approach_vector /= max(float(np.linalg.norm(approach_vector)), 1.0e-12)
            record = {
                "grasp_box": grasp_box.round(7).tolist(),
                "grasp_boxes": grasp_boxes.round(7).tolist(),
                "grasp_mat": matrix.round(7).tolist(),
                # Gripper approach is local +Z. Keep it separate from `normal`,
                # which is the randomized external-force test direction.
                "approach_vector": approach_vector.round(7).tolist(),
                "target_points": geometry["target_points"].round(7).tolist(),
                "target_orientation": geometry["target_orientation"].round(7).tolist(),
                "grasp_orientation_wxyz": geometry[
                    "grasp_orientation_wxyz"
                ].round(8).tolist(),
                "grasp_orientation_rpy_deg": geometry[
                    "grasp_orientation_rpy_deg"
                ].round(7).tolist(),
                "gripper_yaw_deg": round(float(geometry["gripper_yaw_deg"]), 7),
                "target_width": round(float(geometry["target_width"]), 7),
                "attempt_target_points": copy.deepcopy(source["target_points"]),
                "attempt_target_orientation": copy.deepcopy(
                    source["target_orientation"]
                ),
                "attempt_target_width": copy.deepcopy(source.get("target_width", 0.0)),
                "target_object": source["target_object"],
                "source_pregrasp_index": source_id,
                "gripper_model": self.gripper["gripper_name"],
                "gripper_type": self.gripper["type"],
                "contact_points_object_local": copy.deepcopy(
                    geometry["contact_points_object_local"]
                ),
                "score": round(score, 6),
                "rotation_score": round(rotation_score, 6),
                "pose_score": round(pose_score, 6),
                "force_score": round(survival, 6),
                "contact_area_score": round(contact_area_score, 6),
                "pregrasp_pose_score": round(pregrasp_pose_score, 6),
                "stress_pose_score": round(stress_pose_score, 6),
                "pregrasp_center_score": round(pregrasp_center_score, 6),
                "stress_center_score": round(stress_center_score, 6),
                "pregrasp_rotation_score": round(pregrasp_rotation_score, 6),
                "stress_rotation_score": round(stress_rotation_score, 6),
                "normal": geometry["force_direction"].round(7).tolist(),
                "quality": {
                    "result": result,
                    "test_sequence": "close_then_stress",
                    "grasp_record_mode": self.grasp_record_mode,
                    "grasp_complete_width_m": round(
                        float(self.grasp_complete_width[env_id]), 7
                    ) if not self.is_hand else None,
                    "grasp_complete_joint_pos_rad": {
                        name: round(float(value), 7)
                        for name, value in zip(
                            self.joint_names,
                            self.grasp_complete_joint[env_id].tolist(),
                        )
                    },
                    "object_motion_restore_mat": geometry[
                        "restore_transform"
                    ].round(7).tolist(),
                    "score_weights": {
                        "force": FORCE_SCORE_WEIGHT,
                        "pregrasp_pose": PREGRASP_POSE_SCORE_WEIGHT,
                        "stress_pose": STRESS_POSE_SCORE_WEIGHT,
                        "contact_area": CONTACT_AREA_SCORE_WEIGHT,
                    },
                    "score_component_power": SCORE_COMPONENT_POWER,
                    "pose_component_weights": {
                        "center": CENTER_IN_POSE_WEIGHT,
                        "rotation": ROTATION_IN_POSE_WEIGHT,
                    },
                    "contact_area_score_range_mm2": CONTACT_AREA_SCORE_RANGE_MM2,
                    "contact_area_estimator": "physx_manifold_footprint",
                    "contact_point_estimator": (
                        "mean_physx_manifold_point_per_contact_body"
                    ),
                    "contact_point_frame": "object_local",
                    "contact_point_sensor_count": len(
                        geometry["contact_points_object_local"]
                    ),
                    "bbox_height_source": (
                        "mean_contact_point_z"
                        if geometry["bbox_contact_mean_z"] is not None
                        else "legacy_grasp_center_z_no_contact_points"
                    ),
                    "grasp_contact_area_m2": round(
                        float(self.grasp_contact_area[env_id]), 9
                    ),
                    "grasp_contact_area_mm2": round(
                        float(self.grasp_contact_area[env_id]) * 1.0e6, 4
                    ),
                    "center_score_range_m": CENTER_SCORE_RANGE_M,
                    "rotation_score_range_deg": math.degrees(
                        ROTATION_SCORE_RANGE_RAD
                    ),
                    "stress_survival": round(survival, 6),
                    "max_test_force_n": round(float(self.max_test_force[env_id]), 6),
                    "relative_translation_error_m": round(
                        float(self.relative_translation_error[env_id]), 7
                    ),
                    "relative_rotation_error_deg": round(
                        math.degrees(float(self.relative_rotation_error[env_id])), 5
                    ),
                    "pregrasp_rotation_error_deg": round(
                        math.degrees(
                            float(self.pregrasp_rotation_error[env_id])
                        ),
                        5,
                    ),
                    "stress_rotation_error_deg": round(
                        math.degrees(
                            float(self.stress_rotation_error[env_id])
                        ),
                        5,
                    ),
                    "pregrasp_center_error_m": round(
                        float(self.pregrasp_center_error[env_id]), 7
                    ),
                    "stress_center_error_m": round(
                        float(self.stress_center_error[env_id]), 7
                    ),
                    "contact_force_n": round(float(self.last_contact_force[env_id]), 6),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": round(
                        float(self.maximum_pre_stress_object_motion[env_id]), 7
                    ),
                    "finger_z_hop_m": round(float(self.applied_z_hop[env_id]), 7),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                },
            }
            if "initial_object_matrix" in geometry:
                record["quality"]["object_initial_mat"] = geometry[
                    "initial_object_matrix"
                ].round(7).tolist()
                record["quality"]["object_grasp_complete_mat"] = geometry[
                    "complete_object_matrix"
                ].round(7).tolist()
            for key in (
                "preset_name", "target_base_tf", "target_joint_pos", "joint_unit",
                "transition", "selected_heightmap_yaw", "first_contact_z",
                "safety_margin", "grasp_bbox_sets", "contact_sensor_sets",
            ):
                if key in source:
                    record[key] = copy.deepcopy(source[key])
            # output_grasp is a success dataset. Failed stress attempts belong
            # only in attempt_history and must not become low-score grasps.
            if result == "completed":
                self.output_list.append(record)
                self.debug_success_records.append((env_id, record))
            self.attempt_history.append(
                {
                    "source_pregrasp_index": source_id,
                    "result": result,
                    "grasp_record_mode": self.grasp_record_mode,
                    "grasp_complete_width_m": (
                        float(self.grasp_complete_width[env_id])
                        if not self.is_hand else None
                    ),
                    "grasp_complete_joint_pos_rad": {
                        name: float(value)
                        for name, value in zip(
                            self.joint_names,
                            self.grasp_complete_joint[env_id].tolist(),
                        )
                    },
                    "score": score,
                    "force_score": survival,
                    "contact_area_score": contact_area_score,
                    "grasp_contact_area_m2": float(
                        self.grasp_contact_area[env_id]
                    ),
                    "contact_points_object_local": copy.deepcopy(
                        geometry["contact_points_object_local"]
                    ),
                    "pregrasp_pose_score": pregrasp_pose_score,
                    "stress_pose_score": stress_pose_score,
                    "pregrasp_center_score": pregrasp_center_score,
                    "stress_center_score": stress_center_score,
                    "pregrasp_rotation_score": pregrasp_rotation_score,
                    "stress_rotation_score": stress_rotation_score,
                    "relative_translation_error_m": float(
                        self.relative_translation_error[env_id]
                    ),
                    "relative_rotation_error_deg": math.degrees(
                        float(self.relative_rotation_error[env_id])
                    ),
                    "pregrasp_rotation_error_deg": math.degrees(
                        float(self.pregrasp_rotation_error[env_id])
                    ),
                    "stress_rotation_error_deg": math.degrees(
                        float(self.stress_rotation_error[env_id])
                    ),
                    "pregrasp_center_error_m": float(
                        self.pregrasp_center_error[env_id]
                    ),
                    "stress_center_error_m": float(
                        self.stress_center_error[env_id]
                    ),
                    "contact_force_n": float(self.last_contact_force[env_id]),
                    "minimum_contact_separation_m": self._minimum_separation_value(
                        env_id
                    ),
                    "minimum_close_contact_separation_m": (
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_contact_penetration_mm": self._penetration_mm(
                        self._minimum_separation_value(env_id)
                    ),
                    "maximum_close_contact_penetration_mm": self._penetration_mm(
                        self._minimum_close_separation_value(env_id)
                    ),
                    "maximum_pre_stress_object_motion_m": float(
                        self.maximum_pre_stress_object_motion[env_id]
                    ),
                    "finger_z_hop_m": float(self.applied_z_hop[env_id]),
                    "finger_z_hop_enabled": self.apply_finger_z_hop,
                }
            )
            self.recorded[env_id] = True
            self.stage_num[env_id] = DONE
            self.stage_step[env_id] = 0
            self.failed_reason[env_id] = result
