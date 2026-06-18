# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SoccerFieldConfig:
    """Soccer field dimensions and marking sizes."""

    field_length: float
    field_width: float
    goal_depth: float
    goal_width: float
    goal_area_length: float
    goal_area_width: float
    penalty_area_length: float
    penalty_area_width: float
    penalty_mark_dist: float
    center_circle_dia: float
    border_strip_width: float
    corner_arc_radius: float
    goal_height: float
    post_diameter: float
    line_width: float
    mark_size: float


# Copied from soccerLab field presets and used as default scene dimensions.
S_FIELD = SoccerFieldConfig(
    field_length=9.0,
    field_width=6.0,
    goal_depth=0.6,
    goal_width=1.8,
    goal_area_length=1.0,
    goal_area_width=3.0,
    penalty_area_length=2.0,
    penalty_area_width=4.0,
    penalty_mark_dist=1.5,
    center_circle_dia=1.5,
    border_strip_width=0.0,
    corner_arc_radius=0.0,
    goal_height=1.1,
    post_diameter=0.1,
    line_width=0.05,
    mark_size=0.10,
)

M_FIELD = SoccerFieldConfig(
    field_length=14.0,
    field_width=9.0,
    goal_depth=1.0,
    goal_width=2.4,
    goal_area_length=1.0,
    goal_area_width=4.0,
    penalty_area_length=3.0,
    penalty_area_width=6.0,
    penalty_mark_dist=2.0,
    center_circle_dia=3.0,
    border_strip_width=1.0,
    corner_arc_radius=0.5,
    goal_height=1.8,
    post_diameter=0.1,
    line_width=0.07,
    mark_size=0.10,
)

L_FIELD = SoccerFieldConfig(
    field_length=22.0,
    field_width=14.0,
    goal_depth=1.5,
    goal_width=3.0,
    goal_area_length=1.0,
    goal_area_width=5.0,
    penalty_area_length=3.5,
    penalty_area_width=7.0,
    penalty_mark_dist=2.5,
    center_circle_dia=4.0,
    border_strip_width=1.0,
    corner_arc_radius=1.0,
    goal_height=2.0,
    post_diameter=0.12,
    line_width=0.12,
    mark_size=0.15,
)

FIELD_PRESETS = {
    "S": S_FIELD,
    "M": M_FIELD,
    "L": L_FIELD,
}


def get_field_preset(preset_name: str) -> SoccerFieldConfig:
    return FIELD_PRESETS.get(preset_name.upper(), M_FIELD)


@dataclass(frozen=True)
class FieldLineSpec:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GoalPostSpec:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class GoalAssetSpec:
    name: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def build_field_line_specs(field: SoccerFieldConfig, line_height: float = 0.01, z_offset: float = 0.005) -> list[FieldLineSpec]:
    """Build thin cuboid line segments to mimic soccerLab's field-line rendering."""

    lw = field.line_width
    length = field.field_length
    width = field.field_width
    center_circle_radius = field.center_circle_dia * 0.5
    center_circle_segments = 40
    ga_length = field.goal_area_length
    ga_width = field.goal_area_width
    pa_length = field.penalty_area_length
    pa_width = field.penalty_area_width
    half_length = length * 0.5

    line_specs = [
        FieldLineSpec("side_top", (length + 2 * lw, lw, line_height), (0.0, width * 0.5 + lw * 0.5, z_offset)),
        FieldLineSpec("side_btm", (length + 2 * lw, lw, line_height), (0.0, -(width * 0.5 + lw * 0.5), z_offset)),
        FieldLineSpec("goal_left", (lw, width, line_height), (-(length * 0.5 + lw * 0.5), 0.0, z_offset)),
        FieldLineSpec("goal_right", (lw, width, line_height), (length * 0.5 + lw * 0.5, 0.0, z_offset)),
        FieldLineSpec("center_mid", (lw, width, line_height), (0.0, 0.0, z_offset)),
        FieldLineSpec(
            "ga_left_top",
            (ga_length, lw, line_height),
            (-(length * 0.5 - ga_length * 0.5), ga_width * 0.5 + lw * 0.5, z_offset),
        ),
        FieldLineSpec(
            "ga_left_btm",
            (ga_length, lw, line_height),
            (-(length * 0.5 - ga_length * 0.5), -(ga_width * 0.5 + lw * 0.5), z_offset),
        ),
        FieldLineSpec("ga_left_front", (lw, ga_width + 2 * lw, line_height), (-(length * 0.5 - ga_length), 0.0, z_offset)),
        FieldLineSpec(
            "ga_right_top",
            (ga_length, lw, line_height),
            (length * 0.5 - ga_length * 0.5, ga_width * 0.5 + lw * 0.5, z_offset),
        ),
        FieldLineSpec(
            "ga_right_btm",
            (ga_length, lw, line_height),
            (length * 0.5 - ga_length * 0.5, -(ga_width * 0.5 + lw * 0.5), z_offset),
        ),
        FieldLineSpec("ga_right_front", (lw, ga_width + 2 * lw, line_height), (length * 0.5 - ga_length, 0.0, z_offset)),
        FieldLineSpec(
            "pa_left_top",
            (pa_length, lw, line_height),
            (-(length * 0.5 - pa_length * 0.5), pa_width * 0.5 + lw * 0.5, z_offset),
        ),
        FieldLineSpec(
            "pa_left_btm",
            (pa_length, lw, line_height),
            (-(length * 0.5 - pa_length * 0.5), -(pa_width * 0.5 + lw * 0.5), z_offset),
        ),
        FieldLineSpec("pa_left_front", (lw, pa_width + 2 * lw, line_height), (-(length * 0.5 - pa_length), 0.0, z_offset)),
        FieldLineSpec(
            "pa_right_top",
            (pa_length, lw, line_height),
            (length * 0.5 - pa_length * 0.5, pa_width * 0.5 + lw * 0.5, z_offset),
        ),
        FieldLineSpec(
            "pa_right_btm",
            (pa_length, lw, line_height),
            (length * 0.5 - pa_length * 0.5, -(pa_width * 0.5 + lw * 0.5), z_offset),
        ),
        FieldLineSpec("pa_right_front", (lw, pa_width + 2 * lw, line_height), (length * 0.5 - pa_length, 0.0, z_offset)),
        FieldLineSpec(
            "penalty_mark_left",
            (field.mark_size, field.mark_size, line_height),
            (-(half_length - field.penalty_mark_dist), 0.0, z_offset),
        ),
        FieldLineSpec(
            "penalty_mark_right",
            (field.mark_size, field.mark_size, line_height),
            ((half_length - field.penalty_mark_dist), 0.0, z_offset),
        ),
    ]

    if center_circle_radius > 0.0:
        segment_angle = 2.0 * math.pi / center_circle_segments
        segment_length = max(2.0 * center_circle_radius * math.sin(0.5 * segment_angle), lw)
        for segment_idx in range(center_circle_segments):
            theta = segment_idx * segment_angle
            line_specs.append(
                FieldLineSpec(
                    name=f"center_circle_{segment_idx:02d}",
                    size=(segment_length, lw, line_height),
                    position=(
                        center_circle_radius * math.cos(theta),
                        center_circle_radius * math.sin(theta),
                        z_offset,
                    ),
                    orientation=yaw_to_quat_wxyz(theta + math.pi * 0.5),
                )
            )

    return line_specs


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def build_goal_post_specs(field: SoccerFieldConfig) -> list[GoalPostSpec]:
    """Build goal-frame cuboid specs."""

    post_diameter = field.post_diameter
    height = field.goal_height
    goal_width = field.goal_width
    half_length = field.field_length * 0.5
    post_z = height * 0.5
    post_size = (post_diameter, post_diameter, height)
    crossbar_size = (post_diameter, goal_width, post_diameter)

    return [
        GoalPostSpec("left_post_1", post_size, (-half_length, goal_width * 0.5, post_z)),
        GoalPostSpec("left_post_2", post_size, (-half_length, -goal_width * 0.5, post_z)),
        GoalPostSpec("left_crossbar", crossbar_size, (-half_length, 0.0, height)),
        GoalPostSpec("right_post_1", post_size, (half_length, goal_width * 0.5, post_z)),
        GoalPostSpec("right_post_2", post_size, (half_length, -goal_width * 0.5, post_z)),
        GoalPostSpec("right_crossbar", crossbar_size, (half_length, 0.0, height)),
    ]


def build_goal_asset_specs(field: SoccerFieldConfig, z_offset: float = 0.0) -> list[GoalAssetSpec]:
    """Build left/right goal asset poses from a single right-goal USD asset."""

    half_length = field.field_length * 0.5
    quarter_goal_depth = field.goal_depth * 0.25
    return [
        GoalAssetSpec("left_goal_asset", (-(half_length + quarter_goal_depth), 0.0, z_offset), (1.0, 0.0, 0.0, 0.0)),
        GoalAssetSpec("right_goal_asset", (half_length + quarter_goal_depth, 0.0, z_offset), (0.0, 0.0, 0.0, 1.0)),
    ]
