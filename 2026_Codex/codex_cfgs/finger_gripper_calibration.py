"""Calibration helpers shared by the 2026 finger policy and environment."""

from __future__ import annotations

from collections.abc import Iterable


CALIBRATION_KEYS = (
    "joint_to_z_coeffs",
    "joint_to_width_coeffs",
    "width_to_joint_coeffs",
)


def uses_direct_prismatic_width(gripper_info: dict) -> bool:
    """Return whether width is controlled directly by a prismatic joint.

    The exact ``finger2`` type is the legacy two-finger prismatic mechanism.
    Other finger types use the polynomial calibration stored in gripper info.
    """
    return str(gripper_info.get("type", "")).strip().lower() == "finger2"


def validate_calibration(gripper_info: dict) -> None:
    infer_calibration_joint_names(gripper_info)
    if uses_direct_prismatic_width(gripper_info):
        for key in ("open_joint_deg", "close_joint_deg"):
            if key not in gripper_info:
                raise KeyError(
                    f"{gripper_info.get('gripper_name', '<unnamed>')} requires {key} "
                    "for direct prismatic control"
                )
        return

    invalid = [
        key
        for key in CALIBRATION_KEYS
        if not isinstance(gripper_info.get(key), list)
        or len(gripper_info[key]) != 4
    ]
    if invalid:
        raise ValueError(
            f"{gripper_info.get('gripper_name', '<unnamed>')} requires four-value "
            f"calibration fields: {invalid}"
        )


def evaluate_polynomial(coefficients: Iterable[float], value):
    """Evaluate coefficients ordered cubic, quadratic, linear, constant."""
    result = value * 0.0
    for coefficient in coefficients:
        result = result * value + float(coefficient)
    return result


def infer_calibration_joint_names(gripper_info: dict) -> list[str]:
    """Return the single actuator joint whose USD mimic drives all fingers."""
    joint_names = list(gripper_info.get("joint_cfg", {}))
    family = str(gripper_info.get("type", "")).strip().lower()
    if not family.startswith(("finger2", "finger3")):
        raise ValueError(f"Unsupported calibrated gripper type: {family!r}")
    if len(joint_names) != 1:
        raise ValueError(
            f"{gripper_info.get('gripper_name', '<unnamed>')} must define exactly "
            f"one mimic actuator joint in joint_cfg; found {joint_names}"
        )
    return joint_names


def close_joint_degrees(gripper_info: dict, joint_names: list[str]) -> list[float]:
    if uses_direct_prismatic_width(gripper_info):
        raise ValueError("Direct prismatic joint values are metres, not degrees")
    return [float(gripper_info["close_joint_deg"])] * len(joint_names)
