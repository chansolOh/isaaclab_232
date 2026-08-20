"""Calibration helpers shared by the 2026 finger policy and environment."""

from __future__ import annotations

import re
from collections.abc import Iterable


CALIBRATION_KEYS = (
    "joint_to_z_coeffs",
    "joint_to_width_coeffs",
    "width_to_joint_coeffs",
)


def validate_calibration(gripper_info: dict) -> None:
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


def _explicit_joint_names(gripper_info: dict) -> list[str] | None:
    names = gripper_info.get("calibration_joint_names")
    if names is None:
        return None
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("calibration_joint_names must be a non-empty string list")
    return names


def infer_calibration_joint_names(gripper_info: dict) -> list[str]:
    """Find the physical closing joints, excluding legacy lift/z-hop helpers.

    Ambiguous future grippers can provide ``calibration_joint_names`` in the
    JSON.  The current Robotiq_2f140 is inferred as its two finger joints.
    """
    explicit = _explicit_joint_names(gripper_info)
    joint_names = list(gripper_info.get("joint_cfg", {}))
    if explicit is not None:
        missing = [name for name in explicit if name not in joint_names]
        if missing:
            raise KeyError(f"calibration_joint_names not found in joint_cfg: {missing}")
        return explicit

    candidates = [
        name
        for name in joint_names
        if "lift" not in name.lower() and "z_hop" not in name.lower()
    ]
    family = str(gripper_info.get("type", "")).lower()
    expected = 2 if family.startswith("finger2") else 3 if family.startswith("finger3") else 0
    if expected == 0:
        raise ValueError(f"Unsupported calibrated gripper type: {family!r}")
    if len(candidates) == expected:
        return candidates

    if expected == 2:
        direct = [
            name for name in candidates
            if re.fullmatch(r"(?:left|right)_joint", name, flags=re.IGNORECASE)
        ]
        if len(direct) == expected:
            return direct
    else:
        direct = [
            name for name in candidates
            if re.fullmatch(r"f[0-2]_joint", name, flags=re.IGNORECASE)
        ]
        if len(direct) == expected:
            return sorted(direct)

    raise ValueError(
        f"Cannot infer {expected} calibration joints from {joint_names}. "
        "Add calibration_joint_names to this gripper entry."
    )


def close_joint_degrees(gripper_info: dict, joint_names: list[str]) -> list[float]:
    family = str(gripper_info.get("type", "")).lower()
    if family.startswith("finger2"):
        right = float(gripper_info["close_r_joint"])
        left = float(gripper_info["close_l_joint"])
        values = []
        for index, name in enumerate(joint_names):
            lowered = name.lower()
            values.append(left if "left" in lowered else right if "right" in lowered else (right, left)[index])
        return values
    if family.startswith("finger3"):
        closes = [
            float(gripper_info[f"close_f{finger}_joint"])
            for finger in range(3)
        ]
        values = []
        for index, name in enumerate(joint_names):
            match = re.search(r"f([0-2])", name.lower())
            values.append(closes[int(match.group(1))] if match else closes[index])
        return values
    raise ValueError(f"Unsupported calibrated gripper type: {family!r}")


def signed_joint_degrees(gripper_info: dict, joint_names: list[str], magnitude):
    """Apply each physical closing joint's sign to a calibrated magnitude."""
    closes = close_joint_degrees(gripper_info, joint_names)
    values = []
    for close in closes:
        sign = -1.0 if close < 0.0 else 1.0
        values.append(magnitude * sign)
    return values
