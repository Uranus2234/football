from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.observations import get_target_point_world
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_inv(quat), vec)


def _heading_aligned_vec(command: MotionCommand, vec: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(command.cfg, "align_motion_reference_to_initial_heading", False)):
        return vec
    heading_delta = command.initial_heading_yaw_delta.to(device=vec.device, dtype=vec.dtype)
    delta_quat = quat_from_euler_xyz(
        torch.zeros_like(heading_delta),
        torch.zeros_like(heading_delta),
        heading_delta,
    )
    if vec.ndim > 2:
        expand_shape = vec.shape[:-1] + (4,)
        delta_quat = delta_quat.view(delta_quat.shape[0], *([1] * (vec.ndim - 2)), 4).expand(expand_shape)
    return quat_apply(delta_quat.reshape(-1, 4), vec.reshape(-1, 3)).view_as(vec)


def _heading_aligned_quat(command: MotionCommand, quat: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(command.cfg, "align_motion_reference_to_initial_heading", False)):
        return quat
    heading_delta = command.initial_heading_yaw_delta.to(device=quat.device, dtype=quat.dtype)
    delta_quat = quat_from_euler_xyz(
        torch.zeros_like(heading_delta),
        torch.zeros_like(heading_delta),
        heading_delta,
    )
    if quat.ndim > 2:
        expand_shape = quat.shape[:-1] + (4,)
        delta_quat = delta_quat.view(delta_quat.shape[0], *([1] * (quat.ndim - 2)), 4).expand(expand_shape)
    return quat_mul(delta_quat.reshape(-1, 4), quat.reshape(-1, 4)).view_as(quat)


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _map_names_to_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    target_list = list(target_names)
    name_to_index = {name: idx for idx, name in enumerate(target_list)}
    indices: list[int] = []
    # Iterate all source names to map.
    for name in source_names:
        # Prefer exact matching for deterministic mapping.
        if name in name_to_index:
            indices.append(name_to_index[name])
            continue
        # If exact matching fails, attempt unique suffix matching.
        suffix_matches = [idx for idx, candidate in enumerate(target_list) if candidate.endswith(name)]
        # Accept only unique suffix matches to avoid ambiguity.
        if len(suffix_matches) == 1:
            indices.append(suffix_matches[0])
    return indices


def action_rate_l2_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward.clamp(max=100.0)


def waist_action_rate_l2_clip(env: ManagerBasedRLEnv, waist_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if waist_cfg is None:
        raise ValueError("waist_cfg cannot be None")
    robot = env.scene[waist_cfg.name]
    idx = torch.as_tensor(robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0], device=env.device)
    return torch.sum(torch.square(env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]), dim=1).clamp(max=100.0)


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand is missing kick_contact_tracker; ensure command setup is up to date.")
    return tracker


def goal_gate_crossing(
    prev_ball_xy: torch.Tensor,
    curr_ball_xy: torch.Tensor,
    gate_center_xy: torch.Tensor,
    gate_dir_xy: torch.Tensor,
    gate_half_width: torch.Tensor | float,
    dt: float,
    min_forward_speed: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized crossing test for a confidence gate perpendicular to ``gate_dir_xy``.

    Returns ``crossed``, ``inside_gate``, signed ``lateral_error`` and
    ``forward_speed``.  All positions are in the same 2D field frame.
    """
    gate_dir_xy = gate_dir_xy / torch.linalg.norm(gate_dir_xy, dim=-1, keepdim=True).clamp(min=1e-6)
    lateral_dir_xy = torch.stack((-gate_dir_xy[:, 1], gate_dir_xy[:, 0]), dim=-1)

    rel_prev = prev_ball_xy - gate_center_xy
    rel_curr = curr_ball_xy - gate_center_xy
    prev_forward = torch.sum(rel_prev * gate_dir_xy, dim=-1)
    curr_forward = torch.sum(rel_curr * gate_dir_xy, dim=-1)
    forward_speed = (curr_forward - prev_forward) / max(float(dt), 1e-6)
    forward_delta = curr_forward - prev_forward
    alpha = torch.where(torch.abs(forward_delta) > 1e-6, -prev_forward / forward_delta.clamp(min=1e-6), torch.zeros_like(forward_delta))
    alpha = alpha.clamp(0.0, 1.0)
    cross_xy = prev_ball_xy + alpha.unsqueeze(-1) * (curr_ball_xy - prev_ball_xy)
    lateral_error = torch.sum((cross_xy - gate_center_xy) * lateral_dir_xy, dim=-1)

    if not isinstance(gate_half_width, torch.Tensor):
        gate_half_width = torch.full_like(lateral_error, float(gate_half_width))
    else:
        gate_half_width = gate_half_width.to(device=lateral_error.device, dtype=lateral_error.dtype)

    crossed = (prev_forward <= 0.0) & (curr_forward > 0.0) & (forward_speed > float(min_forward_speed))
    inside_gate = torch.abs(lateral_error) <= gate_half_width
    return crossed, inside_gate, lateral_error, forward_speed


def _goal_gate_curriculum_params(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = command.target_point_pos.device
    dtype = command.target_point_pos.dtype
    direction_xy = command.target_destination_pos[:, :2] - command.initial_target_point_pos[:, :2]
    direction_norm = torch.linalg.norm(direction_xy, dim=-1, keepdim=True)
    fallback_direction = torch.zeros_like(direction_xy)
    fallback_direction[:, 0] = 1.0
    direction_xy = torch.where(direction_norm > 1e-6, direction_xy / direction_norm.clamp(min=1e-6), fallback_direction)

    local_center = command.initial_target_point_pos[:, :2] + direction_xy * float(command.cfg.goal_gate_local_distance)
    mid_center = command.initial_target_point_pos[:, :2] + direction_xy * float(command.cfg.goal_gate_mid_distance)
    real_center = command.target_destination_pos[:, :2]

    steps = getattr(command.cfg, "goal_gate_curriculum_steps", (24000, 72000, 144000))
    stage_a_end, stage_b_end, stage_c_end = [int(x) for x in steps]
    step_counter = getattr(env, "common_step_counter", 0)
    if isinstance(step_counter, torch.Tensor):
        step = int(step_counter.item())
    else:
        step = int(step_counter)

    local_half_width = float(command.cfg.goal_gate_local_half_width)
    mid_half_width = float(command.cfg.goal_gate_mid_half_width)
    real_half_width = float(command.cfg.goal_gate_real_half_width)

    if step <= stage_a_end:
        gate_center = local_center
        half_width = local_half_width
        stage_value = 0.0
    elif step <= stage_b_end:
        alpha = (step - stage_a_end) / max(float(stage_b_end - stage_a_end), 1.0)
        gate_center = torch.lerp(local_center, mid_center, alpha)
        half_width = local_half_width + alpha * (mid_half_width - local_half_width)
        stage_value = 1.0 + alpha
    elif step <= stage_c_end:
        alpha = (step - stage_b_end) / max(float(stage_c_end - stage_b_end), 1.0)
        gate_center = torch.lerp(mid_center, real_center, alpha)
        half_width = mid_half_width + alpha * (real_half_width - mid_half_width)
        stage_value = 2.0 + alpha
    else:
        gate_center = real_center
        half_width = real_half_width
        stage_value = 3.0

    half_width_tensor = torch.full((env.num_envs,), float(half_width), dtype=dtype, device=device)
    stage_tensor = torch.full((env.num_envs,), float(stage_value), dtype=dtype, device=device)
    return gate_center, direction_xy, half_width_tensor, stage_tensor


def _ensure_command_metric(command: MotionCommand, name: str) -> torch.Tensor:
    metric = command.metrics.get(name)
    if metric is None or metric.shape[0] != command.num_envs:
        metric = torch.zeros(command.num_envs, device=command.device, dtype=torch.float32)
        command.metrics[name] = metric
    return metric


def _reward_state_name(command_name: str, suffix: str) -> str:
    return f"_{command_name}_{suffix}"


def _ensure_reward_bool_state(env: ManagerBasedRLEnv, command_name: str, suffix: str) -> torch.Tensor:
    name = _reward_state_name(command_name, suffix)
    state = getattr(env, name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        state = state.to(device=env.device, dtype=torch.bool)
    setattr(env, name, state)
    return state


def _ensure_reward_int_state(env: ManagerBasedRLEnv, command_name: str, suffix: str, default: int = -1) -> torch.Tensor:
    name = _reward_state_name(command_name, suffix)
    state = getattr(env, name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.full((env.num_envs,), int(default), device=env.device, dtype=torch.int32)
    else:
        state = state.to(device=env.device, dtype=torch.int32)
    setattr(env, name, state)
    return state


def _ensure_reward_float_state(
    env: ManagerBasedRLEnv, command_name: str, suffix: str, default: float = 0.0
) -> torch.Tensor:
    name = _reward_state_name(command_name, suffix)
    state = getattr(env, name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.full((env.num_envs,), float(default), device=env.device, dtype=torch.float32)
    else:
        state = state.to(device=env.device, dtype=torch.float32)
    setattr(env, name, state)
    return state


def _ensure_reward_vec2_state(env: ManagerBasedRLEnv, command_name: str, suffix: str) -> torch.Tensor:
    name = _reward_state_name(command_name, suffix)
    state = getattr(env, name, None)
    if state is None or state.shape != (env.num_envs, 2):
        state = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    else:
        state = state.to(device=env.device, dtype=torch.float32)
    setattr(env, name, state)
    return state


def _ensure_reward_vec3_state(env: ManagerBasedRLEnv, command_name: str, suffix: str) -> torch.Tensor:
    name = _reward_state_name(command_name, suffix)
    state = getattr(env, name, None)
    if state is None or state.shape != (env.num_envs, 3):
        state = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    else:
        state = state.to(device=env.device, dtype=torch.float32)
    setattr(env, name, state)
    return state


def _command_body_z(command: MotionCommand, body_name: str, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    body_names = list(command.cfg.body_names)
    if body_name in body_names:
        z = command.robot_body_pos_w[:, body_names.index(body_name), 2].to(device=env.device, dtype=torch.float32)
        return z, torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32), torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )


def _resolve_robot_joints(robot, joint_names: list[str], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    result = robot.find_joints(list(joint_names), preserve_order=True)
    joint_ids = torch.as_tensor(result[0], device=device, dtype=torch.long)
    resolved_names = list(result[1]) if len(result) > 1 else []
    if len(resolved_names) != int(joint_ids.numel()):
        robot_joint_names = list(getattr(robot, "joint_names", []))
        resolved_names = [robot_joint_names[int(idx)] for idx in joint_ids.detach().cpu().tolist()]
    return joint_ids, resolved_names


def _joint_target_tensor(
    robot,
    joint_ids: torch.Tensor,
    resolved_names: list[str],
    targets: dict[str, float] | None,
    device: torch.device,
) -> torch.Tensor:
    if joint_ids.numel() == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    default_target = robot.data.default_joint_pos[0, joint_ids].to(device=device, dtype=torch.float32)
    if not targets:
        return default_target
    values = []
    for idx, name in enumerate(resolved_names):
        values.append(float(targets.get(name, default_target[idx].item())))
    return torch.tensor(values, device=device, dtype=torch.float32)


def _quat_apply_inverse_batched(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 2:
        return _quat_apply_inverse(quat, vec)
    quat_expanded = quat.view(quat.shape[0], *([1] * (vec.ndim - 2)), 4).expand(vec.shape[:-1] + (4,))
    return _quat_apply_inverse(quat_expanded.reshape(-1, 4), vec.reshape(-1, 3)).view_as(vec)


def _initial_ball_base_xy(command: MotionCommand, env: ManagerBasedRLEnv) -> torch.Tensor:
    ball_base_xy = getattr(command, "goal_aware_ball_base_xy", None)
    if isinstance(ball_base_xy, torch.Tensor) and ball_base_xy.shape == (env.num_envs, 2):
        return ball_base_xy.to(device=env.device, dtype=torch.float32)

    delta_w = command.initial_target_point_pos[:, :2] - command.robot_anchor_pos_w[:, :2]
    yaw = torch.atan2(
        quat_apply(
            command.robot_anchor_quat_w,
            torch.tensor([1.0, 0.0, 0.0], dtype=command.robot_anchor_quat_w.dtype, device=env.device).expand(env.num_envs, -1),
        )[:, 1],
        quat_apply(
            command.robot_anchor_quat_w,
            torch.tensor([1.0, 0.0, 0.0], dtype=command.robot_anchor_quat_w.dtype, device=env.device).expand(env.num_envs, -1),
        )[:, 0],
    )
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack((c * delta_w[:, 0] + s * delta_w[:, 1], -s * delta_w[:, 0] + c * delta_w[:, 1]), dim=-1)


def _expected_foot_from_ball_y(ball_base_y: torch.Tensor, center_deadband: float) -> torch.Tensor:
    # left=0, right=1.  Positive base-y means ball is on robot-left; the center band defaults to right foot.
    return torch.where(
        ball_base_y > float(center_deadband),
        torch.zeros_like(ball_base_y, dtype=torch.int8),
        torch.ones_like(ball_base_y, dtype=torch.int8),
    )


def _side_foot_contact_terms(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    inside_y_range: tuple[float, float] = (0.035, 0.145),
    side_x_range: tuple[float, float] = (-0.08, 0.11),
    z_abs_max: float = 0.16,
    side_y_target: float = 0.085,
    side_y_std: float = 0.045,
    toe_x_min: float = 0.12,
    toe_y_abs_max: float = 0.075,
    instep_x_range: tuple[float, float] = (-0.05, 0.15),
    instep_y_abs_max: float = 0.045,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> dict[str, torch.Tensor]:
    if foot_cfg is None:
        raise ValueError("side-foot contact rewards require foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    step = int(step_counter.item()) if isinstance(step_counter, torch.Tensor) else int(step_counter)
    cache_name = _reward_state_name(command_name, "side_foot_terms_cache")
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    lateral_state = _ensure_reward_bool_state(env, command_name, "lateral_foot_contact_awarded")
    instep_state = _ensure_reward_bool_state(env, command_name, "instep_contact_awarded")
    toe_state = _ensure_reward_bool_state(env, command_name, "toe_contact_awarded")
    expected_contact_state = _ensure_reward_bool_state(env, command_name, "ball_side_expected_contact_awarded")
    wrong_foot_state = _ensure_reward_bool_state(env, command_name, "ball_side_wrong_foot_contact_awarded")

    expected_foot = _expected_foot_from_ball_y(_initial_ball_base_xy(command, env)[:, 1], center_deadband)
    _ensure_command_metric(command, "expected_ball_left_foot_rate")[:] = (expected_foot == 0).to(torch.float32)
    _ensure_command_metric(command, "expected_ball_right_foot_rate")[:] = (expected_foot == 1).to(torch.float32)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    expected_contact = torch.zeros_like(reward)
    wrong_foot = torch.zeros_like(reward)
    instep = torch.zeros_like(reward)
    toe = torch.zeros_like(reward)
    lateral = torch.zeros_like(reward)
    non_side_expected = torch.zeros_like(reward)
    local_x = torch.zeros_like(reward)
    local_y = torch.zeros_like(reward)
    medial_y_metric = torch.zeros_like(reward)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)
            robot = command.robot
            foot_pos_w = robot.data.body_pos_w[foot_info.env_ids, foot_info.body_indices]
            foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]
            ball_pos = command.soccer_ball_pos[foot_info.env_ids]
            env_origins = getattr(env.scene, "env_origins", None)
            if env_origins is not None:
                ball_pos = ball_pos + env_origins[foot_info.env_ids]
            ball_local = quat_apply(quat_inv(foot_quat_w), ball_pos - foot_pos_w)

            actual_leg = foot_info.sides.to(device=env.device, dtype=torch.int8)
            expected = expected_foot[foot_info.env_ids].to(device=env.device, dtype=torch.int8)
            valid_leg = actual_leg >= 0
            correct = valid_leg & (actual_leg == expected)
            desired_sign = torch.where(
                actual_leg == 0,
                torch.full((foot_info.env_ids.numel(),), float(medial_sign_left), device=env.device),
                torch.full((foot_info.env_ids.numel(),), float(medial_sign_right), device=env.device),
            )
            inside_y = ball_local[:, 1] * desired_sign
            in_side_y = (inside_y >= float(inside_y_range[0])) & (inside_y <= float(inside_y_range[1]))
            in_lateral_y = ((-inside_y) >= float(inside_y_range[0])) & ((-inside_y) <= float(inside_y_range[1]))
            in_side_x = (ball_local[:, 0] >= float(side_x_range[0])) & (ball_local[:, 0] <= float(side_x_range[1]))
            in_z = torch.abs(ball_local[:, 2]) <= float(z_abs_max)
            side = correct & in_side_y & in_side_x & in_z
            lateral_contact = correct & in_lateral_y & in_side_x & in_z
            toe_contact = correct & (ball_local[:, 0] > float(toe_x_min)) & (torch.abs(inside_y) < float(toe_y_abs_max)) & in_z
            instep_contact = (
                correct
                & (ball_local[:, 0] >= float(instep_x_range[0]))
                & (ball_local[:, 0] <= float(instep_x_range[1]))
                & (torch.abs(inside_y) < float(instep_y_abs_max))
                & in_z
            )
            wrong = valid_leg & (~correct)
            expected_non_side = correct & (~side)
            shaped = torch.exp(-((inside_y - float(side_y_target)) ** 2) / max(float(side_y_std) ** 2, 1e-6))

            reward[foot_info.env_ids] = side.to(torch.float32) * shaped
            expected_contact[foot_info.env_ids] = correct.to(torch.float32)
            wrong_foot[foot_info.env_ids] = wrong.to(torch.float32)
            toe[foot_info.env_ids] = toe_contact.to(torch.float32)
            instep[foot_info.env_ids] = instep_contact.to(torch.float32)
            lateral[foot_info.env_ids] = lateral_contact.to(torch.float32)
            non_side_expected[foot_info.env_ids] = expected_non_side.to(torch.float32)
            local_x[foot_info.env_ids] = ball_local[:, 0].to(torch.float32)
            local_y[foot_info.env_ids] = ball_local[:, 1].to(torch.float32)
            medial_y_metric[foot_info.env_ids] = inside_y.to(torch.float32)

            side_state[foot_info.env_ids] |= side
            lateral_state[foot_info.env_ids] |= lateral_contact
            instep_state[foot_info.env_ids] |= instep_contact
            toe_state[foot_info.env_ids] |= toe_contact
            expected_contact_state[foot_info.env_ids] |= correct
            wrong_foot_state[foot_info.env_ids] |= wrong

            strict_expected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            strict_expected[foot_info.env_ids] = side
            tracker.record_expected_success(event.new_contact, strict_expected)

    _ensure_command_metric(command, "inside_foot_contact_rate")[:] = side_state.to(torch.float32)
    _ensure_command_metric(command, "medial_foot_contact_rate")[:] = side_state.to(torch.float32)
    _ensure_command_metric(command, "lateral_foot_contact_rate")[:] = lateral_state.to(torch.float32)
    _ensure_command_metric(command, "instep_contact_rate")[:] = instep_state.to(torch.float32)
    _ensure_command_metric(command, "toe_contact_rate")[:] = toe_state.to(torch.float32)
    _ensure_command_metric(command, "ball_side_expected_contact_rate")[:] = expected_contact_state.to(torch.float32)
    _ensure_command_metric(command, "ball_side_wrong_foot_contact_rate")[:] = wrong_foot_state.to(torch.float32)
    _ensure_command_metric(command, "foot_local_ball_x_mean")[:] = local_x
    _ensure_command_metric(command, "foot_local_ball_y_mean")[:] = local_y
    _ensure_command_metric(command, "foot_local_ball_medial_y_mean")[:] = medial_y_metric

    cache = {
        "step": step,
        "side_foot_contact": reward,
        "lateral_foot_contact": lateral,
        "expected_foot_contact": expected_contact,
        "wrong_foot_contact": wrong_foot,
        "instep_contact": instep,
        "toe_contact": toe,
        "non_side_expected_contact": non_side_expected,
        "side_foot_state": side_state,
        "lateral_foot_state": lateral_state,
    }
    setattr(env, cache_name, cache)
    return cache


def _geometric_side_foot_contact_terms(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    medial_projection_range: tuple[float, float] = (0.035, 0.155),
    side_x_range: tuple[float, float] = (-0.10, 0.12),
    z_abs_max: float = 0.16,
    projection_target: float = 0.09,
    projection_std: float = 0.05,
    toe_x_min: float = 0.12,
    toe_projection_abs_max: float = 0.075,
    instep_x_range: tuple[float, float] = (-0.05, 0.15),
    instep_projection_abs_max: float = 0.045,
) -> dict[str, torch.Tensor]:
    if foot_cfg is None:
        raise ValueError("geometric side-foot contact rewards require foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    step = int(step_counter.item()) if isinstance(step_counter, torch.Tensor) else int(step_counter)
    cache_name = _reward_state_name(command_name, "geometric_side_foot_terms_cache")
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    medial_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    geometric_medial_state = _ensure_reward_bool_state(env, command_name, "geometric_medial_contact_awarded")
    lateral_state = _ensure_reward_bool_state(env, command_name, "lateral_foot_contact_awarded")
    instep_state = _ensure_reward_bool_state(env, command_name, "instep_contact_awarded")
    toe_state = _ensure_reward_bool_state(env, command_name, "toe_contact_awarded")

    expected_foot = _expected_foot_from_ball_y(_initial_ball_base_xy(command, env)[:, 1], center_deadband)
    _ensure_command_metric(command, "expected_ball_left_foot_rate")[:] = (expected_foot == 0).to(torch.float32)
    _ensure_command_metric(command, "expected_ball_right_foot_rate")[:] = (expected_foot == 1).to(torch.float32)

    medial_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    lateral = torch.zeros_like(medial_reward)
    instep = torch.zeros_like(medial_reward)
    toe = torch.zeros_like(medial_reward)
    projection_metric = torch.zeros_like(medial_reward)
    local_x = torch.zeros_like(medial_reward)
    local_z = torch.zeros_like(medial_reward)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)
            robot = command.robot
            foot_pos_w = robot.data.body_pos_w[foot_info.env_ids, foot_info.body_indices]
            foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]
            if "pelvis" in robot.body_names:
                pelvis_pos_w = robot.data.body_pos_w[foot_info.env_ids, robot.body_names.index("pelvis")]
            else:
                pelvis_pos_w = command.robot_anchor_pos_w[foot_info.env_ids]

            ball_pos = command.soccer_ball_pos[foot_info.env_ids]
            env_origins = getattr(env.scene, "env_origins", None)
            if env_origins is not None:
                ball_pos = ball_pos + env_origins[foot_info.env_ids]
            ball_local = quat_apply(quat_inv(foot_quat_w), ball_pos - foot_pos_w)

            foot_to_pelvis_xy = pelvis_pos_w[:, :2] - foot_pos_w[:, :2]
            medial_axis_xy = foot_to_pelvis_xy / torch.linalg.norm(foot_to_pelvis_xy, dim=-1, keepdim=True).clamp(min=1e-6)
            foot_to_ball_xy = ball_pos[:, :2] - foot_pos_w[:, :2]
            projection = torch.sum(foot_to_ball_xy * medial_axis_xy, dim=-1)

            valid_leg = foot_info.sides.to(device=env.device, dtype=torch.int8) >= 0
            in_medial_projection = (
                (projection >= float(medial_projection_range[0]))
                & (projection <= float(medial_projection_range[1]))
            )
            in_lateral_projection = (
                ((-projection) >= float(medial_projection_range[0]))
                & ((-projection) <= float(medial_projection_range[1]))
            )
            in_side_x = (ball_local[:, 0] >= float(side_x_range[0])) & (ball_local[:, 0] <= float(side_x_range[1]))
            in_z = torch.abs(ball_local[:, 2]) <= float(z_abs_max)
            medial = valid_leg & in_medial_projection & in_side_x & in_z
            lateral_contact = valid_leg & in_lateral_projection & in_side_x & in_z
            toe_contact = valid_leg & (ball_local[:, 0] > float(toe_x_min)) & (torch.abs(projection) < float(toe_projection_abs_max)) & in_z
            instep_contact = (
                valid_leg
                & (ball_local[:, 0] >= float(instep_x_range[0]))
                & (ball_local[:, 0] <= float(instep_x_range[1]))
                & (torch.abs(projection) < float(instep_projection_abs_max))
                & in_z
            )
            shaped = torch.exp(-((projection - float(projection_target)) ** 2) / max(float(projection_std) ** 2, 1e-6))

            medial_reward[foot_info.env_ids] = medial.to(torch.float32) * shaped
            lateral[foot_info.env_ids] = lateral_contact.to(torch.float32)
            toe[foot_info.env_ids] = toe_contact.to(torch.float32)
            instep[foot_info.env_ids] = instep_contact.to(torch.float32)
            projection_metric[foot_info.env_ids] = projection.to(torch.float32)
            local_x[foot_info.env_ids] = ball_local[:, 0].to(torch.float32)
            local_z[foot_info.env_ids] = ball_local[:, 2].to(torch.float32)

            medial_state[foot_info.env_ids] |= medial
            geometric_medial_state[foot_info.env_ids] |= medial
            lateral_state[foot_info.env_ids] |= lateral_contact
            instep_state[foot_info.env_ids] |= instep_contact
            toe_state[foot_info.env_ids] |= toe_contact

            strict_expected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            strict_expected[foot_info.env_ids] = medial
            tracker.record_expected_success(event.new_contact, strict_expected)

    _ensure_command_metric(command, "inside_foot_contact_rate")[:] = medial_state.to(torch.float32)
    _ensure_command_metric(command, "medial_foot_contact_rate")[:] = medial_state.to(torch.float32)
    _ensure_command_metric(command, "geometric_medial_contact_rate")[:] = geometric_medial_state.to(torch.float32)
    _ensure_command_metric(command, "geometric_lateral_contact_rate")[:] = lateral_state.to(torch.float32)
    _ensure_command_metric(command, "lateral_foot_contact_rate")[:] = lateral_state.to(torch.float32)
    _ensure_command_metric(command, "instep_contact_rate")[:] = instep_state.to(torch.float32)
    _ensure_command_metric(command, "toe_contact_rate")[:] = toe_state.to(torch.float32)
    _ensure_command_metric(command, "foot_to_pelvis_ball_projection_mean")[:] = projection_metric
    _ensure_command_metric(command, "foot_local_ball_x_mean")[:] = local_x
    _ensure_command_metric(command, "foot_local_ball_z_mean")[:] = local_z

    cache = {
        "step": step,
        "medial_foot_contact": medial_reward,
        "lateral_foot_contact": lateral,
        "instep_contact": instep,
        "toe_contact": toe,
        "medial_foot_state": medial_state,
        "geometric_medial_state": geometric_medial_state,
        "lateral_foot_state": lateral_state,
    }
    setattr(env, cache_name, cache)
    return cache


def geometric_medial_foot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    medial_projection_range: tuple[float, float] = (0.035, 0.155),
    side_x_range: tuple[float, float] = (-0.10, 0.12),
    z_abs_max: float = 0.16,
    projection_target: float = 0.09,
    projection_std: float = 0.05,
) -> torch.Tensor:
    """Reward first contact on the side of the foot facing the robot pelvis."""
    terms = _geometric_side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        medial_projection_range=medial_projection_range,
        side_x_range=side_x_range,
        z_abs_max=z_abs_max,
        projection_target=projection_target,
        projection_std=projection_std,
    )
    return terms["medial_foot_contact"]


def geometric_lateral_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    medial_projection_range: tuple[float, float] = (0.035, 0.155),
    side_x_range: tuple[float, float] = (-0.10, 0.12),
    z_abs_max: float = 0.16,
) -> torch.Tensor:
    """Penalty for first contact on the outside of the actual kicking foot."""
    terms = _geometric_side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        medial_projection_range=medial_projection_range,
        side_x_range=side_x_range,
        z_abs_max=z_abs_max,
    )
    return terms["lateral_foot_contact"]


def geometric_toe_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    toe_x_min: float = 0.12,
    toe_projection_abs_max: float = 0.075,
) -> torch.Tensor:
    """Penalty for toe-like first contact under the pelvis-facing side-foot rule."""
    terms = _geometric_side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        toe_x_min=toe_x_min,
        toe_projection_abs_max=toe_projection_abs_max,
    )
    return terms["toe_contact"]


def geometric_instep_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    instep_x_range: tuple[float, float] = (-0.05, 0.15),
    instep_projection_abs_max: float = 0.045,
) -> torch.Tensor:
    """Penalty for central instep contact instead of pelvis-facing side contact."""
    terms = _geometric_side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        instep_x_range=instep_x_range,
        instep_projection_abs_max=instep_projection_abs_max,
    )
    return terms["instep_contact"]


def _goal_gate_event(env: ManagerBasedRLEnv, command_name: str) -> dict[str, torch.Tensor]:
    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    if isinstance(step_counter, torch.Tensor):
        step = int(step_counter.item())
    else:
        step = int(step_counter)

    cache_name = f"_{command_name}_goal_gate_event_cache"
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    gate_center, gate_dir, gate_half_width, gate_stage = _goal_gate_curriculum_params(env, command)
    dt = float(getattr(env, "step_dt", 0.02))
    crossed, inside_gate, lateral_error, forward_speed = goal_gate_crossing(
        command.goal_gate_prev_ball_pos[:, :2],
        command.target_point_pos[:, :2],
        gate_center,
        gate_dir,
        gate_half_width,
        dt,
        min_forward_speed=float(getattr(command.cfg, "goal_gate_min_cross_speed", 0.0)),
    )

    reset_terminated = getattr(env, "reset_terminated", None)
    if isinstance(reset_terminated, torch.Tensor) and reset_terminated.shape[0] == env.num_envs:
        stable_crossing = ~reset_terminated.to(device=command.device, dtype=torch.bool)
    else:
        stable_crossing = torch.ones(env.num_envs, dtype=torch.bool, device=command.device)

    eligible = ~(command.goal_gate_success_awarded | command.goal_gate_miss_awarded)
    success = crossed & inside_gate & stable_crossing & eligible
    unstable_inside = crossed & inside_gate & (~stable_crossing) & eligible
    miss = crossed & ((~inside_gate) | (~stable_crossing)) & eligible
    event = success | miss
    lateral_ratio = torch.abs(lateral_error) / gate_half_width.clamp(min=1e-6)
    center_score = torch.square(torch.clamp(1.0 - torch.square(lateral_ratio), min=0.0, max=1.0))
    center_score = torch.where(success, center_score, torch.zeros_like(center_score))
    edge_hit = success & (lateral_ratio > 0.75)

    if torch.any(event):
        command.goal_gate_success_awarded |= success
        command.goal_gate_miss_awarded |= miss
        command.goal_gate_lateral_error[event] = torch.abs(lateral_error[event]).to(command.goal_gate_lateral_error.dtype)
        command.goal_gate_cross_speed[event] = forward_speed[event].to(command.goal_gate_cross_speed.dtype)
        command.goal_gate_last_event_step[event] = step
        if hasattr(command, "goal_gate_center_score"):
            command.goal_gate_center_score[event] = center_score[event].to(command.goal_gate_center_score.dtype)
        if hasattr(command, "goal_gate_edge_hit"):
            command.goal_gate_edge_hit[event] = edge_hit[event]

    _ensure_command_metric(command, "goal_success_rate")[:] = command.goal_gate_success_awarded.to(torch.float32)
    _ensure_command_metric(command, "goal_gate_miss_rate")[:] = command.goal_gate_miss_awarded.to(torch.float32)
    _ensure_command_metric(command, "curriculum_gate_success_rate")[:] = command.goal_gate_success_awarded.to(torch.float32)
    _ensure_command_metric(command, "curriculum_gate_miss_rate")[:] = command.goal_gate_miss_awarded.to(torch.float32)
    _ensure_command_metric(command, "gate_lateral_error")[:] = command.goal_gate_lateral_error
    _ensure_command_metric(command, "gate_cross_speed")[:] = command.goal_gate_cross_speed
    _ensure_command_metric(command, "goal_gate_stage")[:] = gate_stage.to(torch.float32)
    _ensure_command_metric(command, "unstable_goal_cross_rate")[:] = unstable_inside.to(torch.float32)
    if hasattr(command, "goal_gate_center_score"):
        _ensure_command_metric(command, "goal_center_score")[:] = command.goal_gate_center_score
    else:
        _ensure_command_metric(command, "goal_center_score")[:] = center_score.to(torch.float32)
    _ensure_command_metric(command, "goal_lateral_error_signed")[:] = lateral_error.to(torch.float32)
    if hasattr(command, "goal_gate_edge_hit"):
        _ensure_command_metric(command, "goal_edge_hit_rate")[:] = command.goal_gate_edge_hit.to(torch.float32)
    else:
        _ensure_command_metric(command, "goal_edge_hit_rate")[:] = edge_hit.to(torch.float32)

    cache = {
        "step": step,
        "success": success,
        "miss": miss,
        "lateral_error": lateral_error,
        "forward_speed": forward_speed,
        "gate_stage": gate_stage,
        "center_score": center_score,
        "gate_half_width": gate_half_width,
    }
    setattr(env, cache_name, cache)
    return cache


def goal_gate_success(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot reward when the ball crosses the current goal gate inside the confidence width."""
    event = _goal_gate_event(env, command_name)
    return event["success"].to(device=env.device, dtype=torch.float32)


def goal_gate_miss(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot penalty when the ball crosses the gate plane outside the confidence width."""
    event = _goal_gate_event(env, command_name)
    return event["miss"].to(device=env.device, dtype=torch.float32)


def goal_gate_center_success(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot gate reward graded by how close the crossing is to the goal center."""
    event = _goal_gate_event(env, command_name)
    return event["center_score"].to(device=env.device, dtype=torch.float32)


def goal_cross_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
) -> torch.Tensor:
    """Reward forward crossing speed only when the ball crosses inside the goal gate."""
    event = _goal_gate_event(env, command_name)
    speed = torch.clamp(event["forward_speed"] / max(float(speed_scale), 1e-6), min=0.0, max=1.5)
    reward = event["success"].to(device=env.device, dtype=torch.float32) * speed.to(device=env.device)
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "goal_cross_speed_reward")[:] = reward
    return reward


def side_foot_goal_gate_center_success(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    non_side_scale: float = 0.05,
) -> torch.Tensor:
    """Gate-center reward that pays full value only after an inside-foot first contact."""
    event = _goal_gate_event(env, command_name)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    scale = torch.where(
        side_state,
        torch.ones(env.num_envs, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), float(non_side_scale), device=env.device, dtype=torch.float32),
    )
    reward = event["center_score"].to(device=env.device, dtype=torch.float32) * scale
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "side_foot_goal_center_score")[:] = reward
    return reward


def side_foot_goal_cross_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    non_side_scale: float = 0.05,
) -> torch.Tensor:
    """Forward crossing-speed reward gated by inside-foot contact quality."""
    event = _goal_gate_event(env, command_name)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    speed = torch.clamp(event["forward_speed"] / max(float(speed_scale), 1e-6), min=0.0, max=1.5)
    scale = torch.where(
        side_state,
        torch.ones(env.num_envs, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), float(non_side_scale), device=env.device, dtype=torch.float32),
    )
    reward = event["success"].to(device=env.device, dtype=torch.float32) * speed.to(device=env.device) * scale
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "goal_cross_speed_reward")[:] = reward
    return reward


def _real_goal_gate_event(
    env: ManagerBasedRLEnv,
    command_name: str,
    goal_half_width: float | None = None,
    min_forward_speed: float = 0.0,
) -> dict[str, torch.Tensor]:
    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    step = int(step_counter.item()) if isinstance(step_counter, torch.Tensor) else int(step_counter)

    cache_name = f"_{command_name}_real_goal_gate_event_cache"
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    # Keep curriculum-gate metrics available, but V4.2 real-goal metrics below
    # intentionally become the primary goal_success_rate.
    _goal_gate_event(env, command_name)

    gate_center = command.target_destination_pos[:, :2]
    goal_sign = torch.where(gate_center[:, 0] >= 0.0, torch.ones_like(gate_center[:, 0]), -torch.ones_like(gate_center[:, 0]))
    gate_dir = torch.stack((goal_sign, torch.zeros_like(goal_sign)), dim=-1)
    half_width = float(command.cfg.goal_gate_real_half_width) if goal_half_width is None else float(goal_half_width)
    half_width_tensor = torch.full((env.num_envs,), half_width, dtype=gate_center.dtype, device=env.device)
    dt = float(getattr(env, "step_dt", 0.02))

    crossed, inside_goal, lateral_error, forward_speed = goal_gate_crossing(
        command.goal_gate_prev_ball_pos[:, :2],
        command.target_point_pos[:, :2],
        gate_center,
        gate_dir,
        half_width_tensor,
        dt,
        min_forward_speed=float(min_forward_speed),
    )

    reset_terminated = getattr(env, "reset_terminated", None)
    if isinstance(reset_terminated, torch.Tensor) and reset_terminated.shape[0] == env.num_envs:
        stable_crossing = ~reset_terminated.to(device=env.device, dtype=torch.bool)
    else:
        stable_crossing = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    success_state = _ensure_reward_bool_state(env, command_name, "real_goal_success_awarded")
    miss_state = _ensure_reward_bool_state(env, command_name, "real_goal_miss_awarded")
    lateral_state = _ensure_reward_float_state(env, command_name, "real_goal_lateral_error")
    speed_state = _ensure_reward_float_state(env, command_name, "real_goal_cross_speed")
    score_state = _ensure_reward_float_state(env, command_name, "real_goal_center_score")
    edge_state = _ensure_reward_bool_state(env, command_name, "real_goal_edge_hit")

    eligible = ~(success_state | miss_state)
    success = crossed & inside_goal & stable_crossing & eligible
    miss = crossed & ((~inside_goal) | (~stable_crossing)) & eligible
    event = success | miss
    lateral_ratio = torch.abs(lateral_error) / half_width_tensor.clamp(min=1e-6)
    center_score = torch.square(torch.clamp(1.0 - torch.square(lateral_ratio), min=0.0, max=1.0))
    center_score = torch.where(success, center_score, torch.zeros_like(center_score))
    edge_hit = success & (lateral_ratio > 0.75)

    if torch.any(event):
        success_state |= success
        miss_state |= miss
        lateral_state[event] = torch.abs(lateral_error[event]).to(lateral_state.dtype)
        speed_state[event] = forward_speed[event].to(speed_state.dtype)
        score_state[event] = center_score[event].to(score_state.dtype)
        edge_state[event] = edge_hit[event]

    _ensure_command_metric(command, "goal_success_rate")[:] = success_state.to(torch.float32)
    _ensure_command_metric(command, "real_goal_success_rate")[:] = success_state.to(torch.float32)
    _ensure_command_metric(command, "real_goal_miss_rate")[:] = miss_state.to(torch.float32)
    _ensure_command_metric(command, "real_goal_lateral_error")[:] = lateral_state
    _ensure_command_metric(command, "real_goal_lateral_error_signed")[:] = lateral_error.to(torch.float32)
    _ensure_command_metric(command, "real_goal_cross_speed")[:] = speed_state
    _ensure_command_metric(command, "real_goal_center_score")[:] = score_state
    _ensure_command_metric(command, "real_goal_edge_hit_rate")[:] = edge_state.to(torch.float32)

    cache = {
        "step": step,
        "success": success,
        "miss": miss,
        "lateral_error": lateral_error,
        "forward_speed": forward_speed,
        "center_score": center_score,
        "state_success": success_state,
        "state_miss": miss_state,
        "state_cross_speed": speed_state,
        "state_center_score": score_state,
    }
    setattr(env, cache_name, cache)
    return cache


def real_goal_success(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot reward when the ball crosses the real goal mouth."""
    event = _real_goal_gate_event(env, command_name)
    return event["success"].to(device=env.device, dtype=torch.float32)


def real_goal_miss(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot penalty when the ball crosses the real goal line outside the mouth."""
    event = _real_goal_gate_event(env, command_name)
    return event["miss"].to(device=env.device, dtype=torch.float32)


def real_goal_center_success(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """One-shot real-goal reward graded by crossing distance from the goal center."""
    event = _real_goal_gate_event(env, command_name)
    return event["center_score"].to(device=env.device, dtype=torch.float32)


def real_goal_cross_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 4.0,
) -> torch.Tensor:
    """Reward real-goal crossing speed only for balls that enter the goal mouth."""
    event = _real_goal_gate_event(env, command_name)
    speed = torch.clamp(event["forward_speed"] / max(float(speed_scale), 1e-6), min=0.0, max=1.5)
    reward = event["success"].to(device=env.device, dtype=torch.float32) * speed.to(device=env.device)
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "real_goal_cross_speed_reward")[:] = reward
    return reward


def side_foot_real_goal_center_success(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    non_side_scale: float = 0.05,
) -> torch.Tensor:
    """Real-goal center reward that pays full value only after a side-foot first contact."""
    event = _real_goal_gate_event(env, command_name)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    scale = torch.where(
        side_state,
        torch.ones(env.num_envs, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), float(non_side_scale), device=env.device, dtype=torch.float32),
    )
    reward = event["center_score"].to(device=env.device, dtype=torch.float32) * scale
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "side_foot_real_goal_center_score")[:] = reward
    return reward


def side_foot_real_goal_cross_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 4.0,
    non_side_scale: float = 0.05,
) -> torch.Tensor:
    """Real-goal crossing-speed reward gated by side-foot first contact quality."""
    event = _real_goal_gate_event(env, command_name)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    speed = torch.clamp(event["forward_speed"] / max(float(speed_scale), 1e-6), min=0.0, max=1.5)
    scale = torch.where(
        side_state,
        torch.ones(env.num_envs, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), float(non_side_scale), device=env.device, dtype=torch.float32),
    )
    reward = event["success"].to(device=env.device, dtype=torch.float32) * speed.to(device=env.device) * scale
    command: MotionCommand = env.command_manager.get_term(command_name)
    _ensure_command_metric(command, "side_foot_real_goal_cross_speed_reward")[:] = reward
    return reward


def _goal_direction_xy(command: MotionCommand) -> torch.Tensor:
    direction_xy = command.target_destination_pos[:, :2] - command.initial_target_point_pos[:, :2]
    direction_norm = torch.linalg.norm(direction_xy, dim=-1, keepdim=True)
    fallback = torch.zeros_like(direction_xy)
    fallback[:, 0] = 1.0
    return torch.where(direction_norm > 1e-6, direction_xy / direction_norm.clamp(min=1e-6), fallback)


def _ball_trajectory_shaping_terms(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float,
    corridor_half_width: float,
) -> dict[str, torch.Tensor]:
    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    if isinstance(step_counter, torch.Tensor):
        step = int(step_counter.item())
    else:
        step = int(step_counter)

    cache_name = f"_{command_name}_ball_trajectory_shaping_cache"
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    soccer_ball = env.scene["soccer_ball"]
    vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    direction_xy = _goal_direction_xy(command)
    lateral_xy = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=-1)

    prev_delta = command.goal_gate_prev_ball_pos[:, :2] - command.initial_target_point_pos[:, :2]
    curr_delta = command.target_point_pos[:, :2] - command.initial_target_point_pos[:, :2]
    delta_xy = command.target_point_pos[:, :2] - command.goal_gate_prev_ball_pos[:, :2]

    progress_delta = torch.sum(delta_xy * direction_xy, dim=-1)
    forward_vel = torch.sum(vel_xy * direction_xy, dim=-1)
    lateral_error = torch.sum(curr_delta * lateral_xy, dim=-1)
    prev_progress = torch.sum(prev_delta * direction_xy, dim=-1)
    curr_progress = torch.sum(curr_delta * direction_xy, dim=-1)

    tracker = _get_kick_tracker(command)
    contact_awarded = tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    moving_or_progressed = (torch.linalg.norm(vel_xy, dim=-1) > 0.05) | (curr_progress > 0.02) | (prev_progress > 0.02)
    gate_pending = ~(command.goal_gate_success_awarded | command.goal_gate_miss_awarded)
    active = contact_awarded & moving_or_progressed & gate_pending

    speed_den = max(float(speed_scale), 1e-6)
    corridor_den = max(float(corridor_half_width), 1e-6)

    forward_progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    velocity_to_goal = torch.zeros_like(forward_progress)
    lateral_corridor = torch.zeros_like(forward_progress)
    wrong_way = torch.zeros_like(forward_progress)

    if torch.any(active):
        forward_progress[active] = torch.clamp(progress_delta[active], min=0.0, max=0.15)
        velocity_to_goal[active] = torch.clamp(forward_vel[active] / speed_den, min=0.0, max=1.0)
        lateral_corridor[active] = torch.clamp(torch.abs(lateral_error[active]) / corridor_den, min=0.0, max=2.0)
        wrong_way[active] = torch.clamp(-forward_vel[active] / speed_den, min=0.0, max=1.0)

    _ensure_command_metric(command, "ball_forward_progress_mean")[:] = forward_progress
    _ensure_command_metric(command, "ball_velocity_to_goal_mean")[:] = velocity_to_goal
    _ensure_command_metric(command, "ball_lateral_corridor_error")[:] = torch.abs(lateral_error).to(torch.float32)
    _ensure_command_metric(command, "ball_wrong_way_rate")[:] = ((wrong_way > 0.0) & active).to(torch.float32)

    cache = {
        "step": step,
        "ball_forward_progress": forward_progress,
        "ball_velocity_to_goal": velocity_to_goal,
        "ball_lateral_corridor_penalty": lateral_corridor,
        "ball_wrong_way_penalty": wrong_way,
    }
    setattr(env, cache_name, cache)
    return cache


def ball_forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    corridor_half_width: float = 0.5,
) -> torch.Tensor:
    """Reward post-contact ball displacement along the target-goal direction."""
    terms = _ball_trajectory_shaping_terms(env, command_name, speed_scale, corridor_half_width)
    return terms["ball_forward_progress"]


def ball_velocity_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    corridor_half_width: float = 0.5,
) -> torch.Tensor:
    """Reward post-contact ball velocity projected onto the target-goal direction."""
    terms = _ball_trajectory_shaping_terms(env, command_name, speed_scale, corridor_half_width)
    return terms["ball_velocity_to_goal"]


def side_foot_ball_velocity_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    corridor_half_width: float = 0.5,
    non_side_scale: float = 0.03,
) -> torch.Tensor:
    """Post-contact velocity-to-goal reward gated by corrected medial side-foot contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    terms = _ball_trajectory_shaping_terms(env, command_name, speed_scale, corridor_half_width)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    scale = torch.where(
        side_state,
        torch.ones(env.num_envs, device=env.device, dtype=torch.float32),
        torch.full((env.num_envs,), float(non_side_scale), device=env.device, dtype=torch.float32),
    )
    reward = terms["ball_velocity_to_goal"] * scale
    _ensure_command_metric(command, "side_foot_ball_velocity_to_goal_mean")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "actual_medial_speed_gate_rate")[:] = side_state.to(torch.float32)
    return reward


def ball_lateral_corridor_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    corridor_half_width: float = 0.5,
) -> torch.Tensor:
    """Penalty magnitude for post-contact lateral deviation from the shot line."""
    terms = _ball_trajectory_shaping_terms(env, command_name, speed_scale, corridor_half_width)
    return terms["ball_lateral_corridor_penalty"]


def ball_wrong_way_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    speed_scale: float = 3.0,
    corridor_half_width: float = 0.5,
) -> torch.Tensor:
    """Penalty magnitude when the ball travels opposite the target-goal direction."""
    terms = _ball_trajectory_shaping_terms(env, command_name, speed_scale, corridor_half_width)
    return terms["ball_wrong_way_penalty"]


def non_timeout_termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty indicator for non-timeout episode termination."""
    terminated = getattr(env, "reset_terminated", None)
    if isinstance(terminated, torch.Tensor) and terminated.shape[0] == env.num_envs:
        return terminated.to(device=env.device, dtype=torch.float32)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def post_kick_alive(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Reward staying alive after the ball has been contacted."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    alive = 1.0 - non_timeout_termination_penalty(env)
    return tracker.get_contact_awarded().to(device=env.device, dtype=torch.float32) * alive


def post_goal_alive(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Reward staying alive after a stable goal-gate success."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    alive = 1.0 - non_timeout_termination_penalty(env)
    return command.goal_gate_success_awarded.to(device=env.device, dtype=torch.float32) * alive


def goal_aware_root_trajectory_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.35,
    decay_after_contact: bool = True,
) -> torch.Tensor:
    """Track the heading-aligned reference anchor displacement before ball contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_idx = command.motion_anchor_body_index
    current_ref = command.motion.body_pos_w[command.motion_idx, command.time_steps, anchor_idx]
    first_steps = torch.zeros_like(command.time_steps)
    initial_ref = command.motion.body_pos_w[command.motion_idx, first_steps, anchor_idx]
    aligned_delta = _heading_aligned_vec(command, current_ref - initial_ref)

    initial_anchor = getattr(command, "reference_initial_anchor_pos_w", None)
    if initial_anchor is None or initial_anchor.shape[0] != env.num_envs:
        initial_anchor = command.robot_anchor_pos_w.detach()
    expected_anchor = initial_anchor.to(device=env.device, dtype=aligned_delta.dtype) + aligned_delta

    error_xy = torch.linalg.norm(expected_anchor[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    reward = torch.exp(-(error_xy**2) / (float(std) ** 2))
    if decay_after_contact:
        tracker = _get_kick_tracker(command)
        pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
        reward = torch.where(pre_contact, reward, 0.25 * reward)

    _ensure_command_metric(command, "goal_aware_root_traj_error")[:] = error_xy.to(torch.float32)
    return reward


def pre_contact_double_air_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    sensor_name: str = "contact_forces",
    contact_force_threshold: float = 5.0,
    min_air_height: float = 0.04,
    grace_steps: int = 5,
) -> torch.Tensor:
    """Penalty for jumping with both feet off the ground before the kick contact."""
    if foot_cfg is None:
        raise ValueError("pre_contact_double_air_penalty requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        after_grace = step_buf.to(device=env.device, dtype=torch.long) > int(grace_steps)
    else:
        after_grace = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    sensors = getattr(env.scene, "sensors", None)
    contact_sensor = None
    if sensors is not None:
        try:
            contact_sensor = sensors[sensor_name] if isinstance(sensors, dict) else getattr(sensors, sensor_name, None)
        except (KeyError, AttributeError, TypeError):
            contact_sensor = None

    robot = env.scene[foot_cfg.name]
    foot_indices = torch.as_tensor(robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0], device=env.device)
    foot_z = robot.data.body_pos_w[:, foot_indices, 2]
    feet_high = torch.all(foot_z > float(min_air_height), dim=-1)

    if contact_sensor is None:
        both_air = feet_high
    else:
        forces = getattr(contact_sensor.data, "net_forces_w_history", None)
        if forces is not None and forces.numel() > 0:
            forces = forces.to(env.device)
            if forces.ndim >= 4:
                forces = forces.amax(dim=1)
        else:
            forces = getattr(contact_sensor.data, "net_forces_w", None)
            if forces is not None and forces.numel() > 0:
                forces = forces.to(env.device)

        if forces is None or forces.ndim < 3:
            both_air = feet_high
        else:
            if not hasattr(contact_sensor, "_v3_foot_indices_cache"):
                contact_sensor._v3_foot_indices_cache = {}
            key = tuple(foot_cfg.body_names)
            if key not in contact_sensor._v3_foot_indices_cache:
                sensor_indices = contact_sensor.find_bodies(foot_cfg.body_names, preserve_order=True)[0]
                contact_sensor._v3_foot_indices_cache[key] = torch.as_tensor(
                    sensor_indices, device=env.device, dtype=torch.long
                )
            sensor_foot_indices = contact_sensor._v3_foot_indices_cache[key]
            vertical_force = forces[:, sensor_foot_indices, 2]
            foot_contact = vertical_force > float(contact_force_threshold)
            both_air = (~torch.any(foot_contact, dim=-1)) & feet_high

    penalty = (pre_contact & after_grace & both_air).to(torch.float32)
    _ensure_command_metric(command, "pre_contact_double_air_rate")[:] = penalty
    return penalty


def far_ball_pre_contact_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    far_ball_x: float = 0.65,
    progress_scale: float = 0.18,
) -> torch.Tensor:
    """Reward moving the root closer to far balls before first contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    ball_base_x = _initial_ball_base_xy(command, env)[:, 0]
    far = ball_base_x > float(far_ball_x)

    initial_anchor = getattr(command, "reference_initial_anchor_pos_w", command.robot_anchor_pos_w).to(env.device)
    initial_dist = torch.linalg.norm(command.initial_target_point_pos[:, :2] - initial_anchor[:, :2], dim=-1)
    current_dist = torch.linalg.norm(command.target_point_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    progress = torch.clamp(initial_dist - current_dist, min=0.0)
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    active = far & pre_contact
    reward[active] = torch.clamp(progress[active] / max(float(progress_scale), 1e-6), min=0.0, max=1.0)

    _ensure_command_metric(command, "far_ball_rate")[:] = far.to(torch.float32)
    _ensure_command_metric(command, "far_ball_approach_progress")[:] = progress.to(torch.float32)
    return reward


def far_ball_early_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    far_ball_x: float = 0.65,
    min_root_progress: float = 0.12,
) -> torch.Tensor:
    """Penalty for touching a far ball before the root has stepped closer."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    ball_base_x = _initial_ball_base_xy(command, env)[:, 0]
    initial_anchor = getattr(command, "reference_initial_anchor_pos_w", command.robot_anchor_pos_w).to(env.device)
    initial_dist = torch.linalg.norm(command.initial_target_point_pos[:, :2] - initial_anchor[:, :2], dim=-1)
    current_dist = torch.linalg.norm(command.target_point_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    progress = initial_dist - current_dist

    penalty = event.new_contact.to(device=env.device, dtype=torch.bool) & (ball_base_x > float(far_ball_x)) & (
        progress < float(min_root_progress)
    )
    _ensure_command_metric(command, "far_ball_early_contact_rate")[:] = penalty.to(torch.float32)
    return penalty.to(torch.float32)


def _support_step_terms(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    far_ball_x: float = 0.70,
    support_forward_target: float = 0.18,
    support_forward_min: float = 0.10,
    support_lateral_max: float = 0.16,
) -> dict[str, torch.Tensor]:
    """Track whether the support foot steps forward before a far-ball kick."""
    if foot_cfg is None:
        raise ValueError("support step rewards require foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    step_counter = getattr(env, "common_step_counter", 0)
    step = int(step_counter.item()) if isinstance(step_counter, torch.Tensor) else int(step_counter)
    cache_name = _reward_state_name(command_name, "support_step_terms_cache")
    cache = getattr(env, cache_name, None)
    if isinstance(cache, dict) and cache.get("step") == step:
        return cache

    initial_support_pos = _ensure_reward_vec3_state(env, command_name, "support_step_initial_pos")
    initial_valid = _ensure_reward_bool_state(env, command_name, "support_step_initial_valid")
    completed_state = _ensure_reward_bool_state(env, command_name, "support_step_completed")

    resample_flags = getattr(env, _reward_state_name(command_name, "motion_resampled"), None)
    if isinstance(resample_flags, torch.Tensor) and resample_flags.shape[0] == env.num_envs:
        reset_mask = resample_flags.to(device=env.device, dtype=torch.bool)
    else:
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene[foot_cfg.name]
    foot_indices = torch.as_tensor(robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0], device=env.device)
    foot_sides = torch.tensor(
        [0 if "left" in name.lower() else 1 if "right" in name.lower() else -1 for name in foot_cfg.body_names],
        dtype=torch.int8,
        device=env.device,
    )
    foot_pos = robot.data.body_pos_w[:, foot_indices]

    ball_base = _initial_ball_base_xy(command, env)
    expected_kick_foot = _expected_foot_from_ball_y(ball_base[:, 1], center_deadband)
    support_foot = 1 - expected_kick_foot
    support_slot = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for side in (0, 1):
        matches = torch.nonzero(foot_sides == side, as_tuple=False).squeeze(-1)
        if matches.numel() > 0:
            support_slot = torch.where(support_foot == side, matches[0].to(support_slot.dtype), support_slot)
    env_ids = torch.arange(env.num_envs, device=env.device)
    support_pos = foot_pos[env_ids, support_slot]

    init_mask = reset_mask | (~initial_valid)
    if torch.any(init_mask):
        initial_support_pos[init_mask] = support_pos[init_mask].detach()
        initial_valid[init_mask] = True
        completed_state[init_mask] = False

    direction_xy = _goal_direction_xy(command)
    lateral_xy = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=-1)
    delta_xy = support_pos[:, :2] - initial_support_pos[:, :2]
    forward_progress = torch.sum(delta_xy * direction_xy, dim=-1)
    lateral_drift = torch.abs(torch.sum(delta_xy * lateral_xy, dim=-1))

    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    far = ball_base[:, 0] > float(far_ball_x)
    currently_complete = (
        far
        & pre_contact
        & (forward_progress >= float(support_forward_min))
        & (lateral_drift <= float(support_lateral_max))
    )
    completed_state[:] |= currently_complete

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    active = far & pre_contact
    if torch.any(active):
        reward[active] = torch.clamp(
            forward_progress[active] / max(float(support_forward_target), 1e-6),
            min=0.0,
            max=1.0,
        )
        reward[active] *= (lateral_drift[active] <= float(support_lateral_max)).to(torch.float32)

    _ensure_command_metric(command, "far_ball_support_step_rate")[:] = completed_state.to(torch.float32)
    _ensure_command_metric(command, "far_ball_support_step_progress")[:] = forward_progress.to(torch.float32)
    _ensure_command_metric(command, "far_ball_support_step_lateral")[:] = lateral_drift.to(torch.float32)

    cache = {
        "step": step,
        "far": far,
        "pre_contact": pre_contact,
        "completed": completed_state,
        "reward": reward,
    }
    setattr(env, cache_name, cache)
    return cache


def far_ball_support_step_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    far_ball_x: float = 0.70,
    support_forward_target: float = 0.18,
    support_forward_min: float = 0.10,
    support_lateral_max: float = 0.16,
) -> torch.Tensor:
    """Reward the opposite support foot stepping forward before far-ball contact."""
    terms = _support_step_terms(
        env,
        command_name=command_name,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        far_ball_x=far_ball_x,
        support_forward_target=support_forward_target,
        support_forward_min=support_forward_min,
        support_lateral_max=support_lateral_max,
    )
    return terms["reward"]


def far_ball_no_support_step_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    far_ball_x: float = 0.70,
    support_forward_target: float = 0.18,
    support_forward_min: float = 0.10,
    support_lateral_max: float = 0.16,
) -> torch.Tensor:
    """Penalty when a far ball is touched before the support foot step is complete."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    terms = _support_step_terms(
        env,
        command_name=command_name,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        far_ball_x=far_ball_x,
        support_forward_target=support_forward_target,
        support_forward_min=support_forward_min,
        support_lateral_max=support_lateral_max,
    )
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    penalty = event.new_contact.to(device=env.device, dtype=torch.bool) & terms["far"] & (~terms["completed"])
    _ensure_command_metric(command, "far_ball_no_support_step_contact_rate")[:] = penalty.to(torch.float32)
    return penalty.to(torch.float32)


def torso_pitch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    body_name: str = "torso_link",
    pitch_threshold: float = 0.22,
    pitch_scale: float = 0.35,
    far_ball_x: float = 0.65,
    far_extra_scale: float = 1.0,
) -> torch.Tensor:
    """Penalty for excessive forward/backward trunk pitch, especially before far-ball contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_names = list(command.cfg.body_names)
    if body_name in body_names:
        body_idx = body_names.index(body_name)
        quat_w = command.robot_body_quat_w[:, body_idx]
    else:
        quat_w = command.robot_pelvis_quat_w

    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=quat_w.dtype, device=env.device).expand(env.num_envs, -1)
    local_gravity = _quat_apply_inverse(quat_w, gravity_vec_w)
    pitch_mag = torch.abs(local_gravity[:, 0])
    base_penalty = torch.clamp((pitch_mag - float(pitch_threshold)) / max(float(pitch_scale), 1e-6), min=0.0, max=2.0)

    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    far = _initial_ball_base_xy(command, env)[:, 0] > float(far_ball_x)
    multiplier = torch.where(
        pre_contact & far,
        torch.full_like(base_penalty, 1.0 + float(far_extra_scale)),
        torch.ones_like(base_penalty),
    )
    penalty = base_penalty * multiplier
    _ensure_command_metric(command, "torso_pitch_penalty")[:] = penalty.to(torch.float32)
    _ensure_command_metric(command, "torso_pitch_abs")[:] = pitch_mag.to(torch.float32)
    return penalty


def pre_contact_motion_foot_style(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.22,
    foot_body_names: list[str] | None = None,
    far_ball_x: float = 0.70,
    far_extra_scale: float = 0.5,
) -> torch.Tensor:
    """Mimic reference foot placement before contact, with extra emphasis for far balls."""
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    body_indexes = _get_body_indexes(command, foot_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    ).mean(-1)
    reward = torch.exp(-error / max(float(std) ** 2, 1e-6))
    far = _initial_ball_base_xy(command, env)[:, 0] > float(far_ball_x)
    reward = torch.where(pre_contact, reward, torch.zeros_like(reward))
    reward = reward * torch.where(far, torch.full_like(reward, 1.0 + float(far_extra_scale)), torch.ones_like(reward))
    _ensure_command_metric(command, "pre_contact_motion_foot_style")[:] = reward.to(torch.float32)
    return reward


def pre_contact_motion_style_lite(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    root_std: float = 0.35,
    foot_std: float = 0.24,
    torso_pitch_threshold: float = 0.18,
    torso_pitch_scale: float = 0.24,
    foot_body_names: list[str] | None = None,
    torso_body_name: str = "torso_link",
    root_weight: float = 0.40,
    foot_weight: float = 0.40,
    torso_weight: float = 0.20,
    post_contact_scale: float = 0.08,
) -> torch.Tensor:
    """Compact pre-contact style prior for V4 LitePower."""
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)

    anchor_idx = command.motion_anchor_body_index
    current_ref = command.motion.body_pos_w[command.motion_idx, command.time_steps, anchor_idx]
    first_steps = torch.zeros_like(command.time_steps)
    initial_ref = command.motion.body_pos_w[command.motion_idx, first_steps, anchor_idx]
    aligned_delta = _heading_aligned_vec(command, current_ref - initial_ref)
    initial_anchor = getattr(command, "reference_initial_anchor_pos_w", None)
    if initial_anchor is None or initial_anchor.shape[0] != env.num_envs:
        initial_anchor = command.robot_anchor_pos_w.detach()
    expected_anchor = initial_anchor.to(device=env.device, dtype=aligned_delta.dtype) + aligned_delta
    root_error = torch.linalg.norm(expected_anchor[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    root_score = torch.exp(-(root_error**2) / max(float(root_std) ** 2, 1e-6))

    foot_indexes = _get_body_indexes(command, foot_body_names)
    foot_error = torch.sum(
        torch.square(command.body_pos_relative_w[:, foot_indexes] - command.robot_body_pos_w[:, foot_indexes]),
        dim=-1,
    ).mean(-1)
    foot_score = torch.exp(-foot_error / max(float(foot_std) ** 2, 1e-6))

    body_names = list(command.cfg.body_names)
    if torso_body_name in body_names:
        torso_quat_w = command.robot_body_quat_w[:, body_names.index(torso_body_name)]
    else:
        torso_quat_w = command.robot_pelvis_quat_w
    gravity_vec_w = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=torso_quat_w.dtype,
        device=env.device,
    ).expand(env.num_envs, -1)
    local_gravity = _quat_apply_inverse(torso_quat_w, gravity_vec_w)
    torso_pitch_abs = torch.abs(local_gravity[:, 0])
    torso_score = torch.exp(
        -(torch.clamp(torso_pitch_abs - float(torso_pitch_threshold), min=0.0) ** 2)
        / max(float(torso_pitch_scale) ** 2, 1e-6)
    )

    total_weight = max(float(root_weight) + float(foot_weight) + float(torso_weight), 1e-6)
    style = (
        float(root_weight) * root_score
        + float(foot_weight) * foot_score
        + float(torso_weight) * torso_score
    ) / total_weight
    reward = torch.where(pre_contact, style, style * float(post_contact_scale))

    _ensure_command_metric(command, "pre_contact_motion_style_lite")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "pre_contact_motion_style_lite_root_error")[:] = root_error.to(torch.float32)
    _ensure_command_metric(command, "pre_contact_motion_style_lite_foot_error")[:] = foot_error.to(torch.float32)
    _ensure_command_metric(command, "pre_contact_motion_style_lite_torso_pitch_abs")[:] = torso_pitch_abs.to(torch.float32)
    return reward


def post_kick_stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    delay_s: float = 0.5,
    ang_vel_std: float = 1.2,
    joint_vel_std: float = 8.0,
    tilt_std: float = 0.22,
    drift_std: float = 0.25,
    foot_height_max: float = 0.085,
) -> torch.Tensor:
    """Reward being upright, slow and planted after the kick follow-through."""
    if foot_cfg is None:
        raise ValueError("post_kick_stand_still requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    counter = _ensure_reward_int_state(env, command_name, "post_kick_stand_still_counter", default=-1)
    contact_anchor_xy = _ensure_reward_vec2_state(env, command_name, "post_kick_contact_anchor_xy")

    if torch.any(event.new_contact):
        counter[event.new_contact] = 0
        contact_anchor_xy[event.new_contact] = command.robot_anchor_pos_w[event.new_contact, :2].detach()

    active_counter = counter >= 0
    delay_steps = max(0, int(round(float(delay_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    active = active_counter & (counter >= delay_steps)

    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=command.robot_pelvis_quat_w.dtype, device=env.device).expand(
        env.num_envs, -1
    )
    pelvis_gravity = _quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    tilt = torch.linalg.norm(pelvis_gravity[:, :2], dim=-1)
    anchor_ang_vel = torch.linalg.norm(command.robot_anchor_ang_vel_w, dim=-1)
    joint_vel = torch.linalg.norm(command.robot.data.joint_vel[:, command.controlled_joint_ids], dim=-1) / max(
        float(len(command.controlled_joint_ids)) ** 0.5, 1.0
    )
    drift = torch.linalg.norm(command.robot_anchor_pos_w[:, :2] - contact_anchor_xy, dim=-1)

    robot = env.scene[foot_cfg.name]
    foot_indices = torch.as_tensor(robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0], device=env.device)
    foot_z = robot.data.body_pos_w[:, foot_indices, 2]
    feet_planted = torch.all(foot_z < float(foot_height_max), dim=-1)

    score = (
        torch.exp(-(anchor_ang_vel**2) / max(float(ang_vel_std) ** 2, 1e-6))
        * torch.exp(-(joint_vel**2) / max(float(joint_vel_std) ** 2, 1e-6))
        * torch.exp(-(tilt**2) / max(float(tilt_std) ** 2, 1e-6))
        * torch.exp(-(drift**2) / max(float(drift_std) ** 2, 1e-6))
        * feet_planted.to(torch.float32)
    )
    reward = torch.where(active, score, torch.zeros_like(score))

    metric_mask = active_counter.to(torch.float32)
    _ensure_command_metric(command, "post_kick_stand_still")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "post_kick_drift")[:] = drift.to(torch.float32) * metric_mask
    _ensure_command_metric(command, "post_kick_joint_vel")[:] = joint_vel.to(torch.float32) * metric_mask
    _ensure_command_metric(command, "post_kick_tilt")[:] = tilt.to(torch.float32) * metric_mask
    _ensure_command_metric(command, "post_kick_stand_still_active")[:] = metric_mask

    counter = torch.where(active_counter, counter + 1, counter)
    setattr(env, _reward_state_name(command_name, "post_kick_stand_still_counter"), counter)
    setattr(env, _reward_state_name(command_name, "post_kick_contact_anchor_xy"), contact_anchor_xy)
    return reward


def post_kick_motion_tail_recovery_style(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    delay_s: float = 0.35,
    tail_frames: int = 40,
    joint_std: float = 0.45,
    joint_vel_std: float = 2.5,
    body_std: float = 0.22,
    tilt_std: float = 0.16,
    ang_vel_std: float = 1.0,
    foot_cfg: SceneEntityCfg | None = None,
    foot_height_max: float = 0.075,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward recovery toward the sampled kick motion's final standing tail.

    The comparison is local to the motion/robot anchor body.  Global root
    position is intentionally ignored so the policy is not asked to return to
    the mocap clip's original world coordinates.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    counter = _ensure_reward_int_state(env, command_name, "post_kick_stand_still_counter", default=-1)
    delay_steps = max(0, int(round(float(delay_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    active = (counter >= 0) & (counter >= delay_steps)

    lengths = command.motion_length.to(device=env.device, dtype=torch.long).clamp(min=1)
    tail_count = max(1, int(tail_frames))
    tail_start = torch.clamp(lengths - tail_count, min=0)
    rel_step = torch.clamp(counter.to(device=env.device, dtype=torch.long) - delay_steps, min=0)
    ref_steps = torch.minimum(tail_start + rel_step, lengths - 1)
    tail_denom = torch.clamp(lengths - tail_start - 1, min=1).to(dtype=torch.float32)
    recovery_phase = (ref_steps - tail_start).to(dtype=torch.float32) / tail_denom

    motion_idx = command.motion_idx.to(device=env.device, dtype=torch.long)
    ref_joint_pos = command.motion.joint_pos[motion_idx, ref_steps].to(device=env.device, dtype=torch.float32)
    ref_joint_vel = command.motion.joint_vel[motion_idx, ref_steps].to(device=env.device, dtype=torch.float32)
    robot_joint_pos = command.robot.data.joint_pos[:, command.controlled_joint_ids].to(device=env.device, dtype=torch.float32)
    robot_joint_vel = command.robot.data.joint_vel[:, command.controlled_joint_ids].to(device=env.device, dtype=torch.float32)

    joint_dim = max(float(command.controlled_joint_ids.numel()) ** 0.5, 1.0)
    joint_error = torch.linalg.norm(robot_joint_pos - ref_joint_pos, dim=-1) / joint_dim
    joint_vel_error = torch.linalg.norm(robot_joint_vel - ref_joint_vel, dim=-1) / joint_dim
    joint_score = torch.exp(-(joint_error**2) / max(float(joint_std) ** 2, 1e-6))
    joint_vel_score = torch.exp(-(joint_vel_error**2) / max(float(joint_vel_std) ** 2, 1e-6))

    body_indexes = _get_body_indexes(command, body_names)
    if body_indexes:
        ref_body_pos = command.motion.body_pos_w[motion_idx, ref_steps][:, body_indexes].to(
            device=env.device, dtype=torch.float32
        )
        ref_anchor_pos = command.motion.body_pos_w[motion_idx, ref_steps, command.motion_anchor_body_index].to(
            device=env.device, dtype=torch.float32
        )
        ref_anchor_quat = command.motion.body_quat_w[motion_idx, ref_steps, command.motion_anchor_body_index].to(
            device=env.device, dtype=torch.float32
        )
        cur_body_pos = command.robot_body_pos_w[:, body_indexes].to(device=env.device, dtype=torch.float32)
        cur_anchor_pos = command.robot_anchor_pos_w.to(device=env.device, dtype=torch.float32)
        cur_anchor_quat = command.robot_anchor_quat_w.to(device=env.device, dtype=torch.float32)

        ref_body_local = _quat_apply_inverse_batched(ref_anchor_quat, ref_body_pos - ref_anchor_pos[:, None, :])
        cur_body_local = _quat_apply_inverse_batched(cur_anchor_quat, cur_body_pos - cur_anchor_pos[:, None, :])
        body_error = torch.sqrt(torch.sum(torch.square(cur_body_local - ref_body_local), dim=-1).mean(dim=-1))
        body_score = torch.exp(-(body_error**2) / max(float(body_std) ** 2, 1e-6))
    else:
        body_error = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        body_score = torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=command.robot_pelvis_quat_w.dtype, device=env.device).expand(
        env.num_envs, -1
    )
    pelvis_gravity = _quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    tilt = torch.linalg.norm(pelvis_gravity[:, :2], dim=-1)
    anchor_ang_vel = torch.linalg.norm(command.robot_anchor_ang_vel_w, dim=-1)
    upright_score = torch.exp(-(tilt**2) / max(float(tilt_std) ** 2, 1e-6)) * torch.exp(
        -(anchor_ang_vel**2) / max(float(ang_vel_std) ** 2, 1e-6)
    )

    if foot_cfg is not None:
        robot = env.scene[foot_cfg.name]
        foot_indices = torch.as_tensor(robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0], device=env.device)
        foot_z = robot.data.body_pos_w[:, foot_indices, 2]
        feet_planted = torch.all(foot_z < float(foot_height_max), dim=-1)
    else:
        feet_planted = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    style_score = 0.45 * joint_score + 0.35 * body_score + 0.20 * joint_vel_score
    score = style_score * upright_score * feet_planted.to(torch.float32)
    reward = torch.where(active, score, torch.zeros_like(score))

    _ensure_command_metric(command, "post_kick_recovery_style")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "post_kick_recovery_joint_error")[:] = torch.where(
        active, joint_error, torch.zeros_like(joint_error)
    )
    _ensure_command_metric(command, "post_kick_recovery_joint_vel_error")[:] = torch.where(
        active, joint_vel_error, torch.zeros_like(joint_vel_error)
    )
    _ensure_command_metric(command, "post_kick_recovery_body_error")[:] = torch.where(
        active, body_error, torch.zeros_like(body_error)
    )
    _ensure_command_metric(command, "post_kick_recovery_active")[:] = active.to(torch.float32)
    _ensure_command_metric(command, "post_kick_recovery_ref_phase")[:] = torch.where(
        active, recovery_phase, torch.zeros_like(recovery_phase)
    ).to(torch.float32)
    return reward


def post_kick_drift_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    delay_s: float = 0.5,
    drift_limit: float = 0.18,
    drift_scale: float = 0.30,
) -> torch.Tensor:
    """Penalty for continuing to drift after the kick should have settled."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    counter = _ensure_reward_int_state(env, command_name, "post_kick_stand_still_counter", default=-1)
    contact_anchor_xy = _ensure_reward_vec2_state(env, command_name, "post_kick_contact_anchor_xy")
    delay_steps = max(0, int(round(float(delay_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    active = (counter >= 0) & (counter >= delay_steps)
    drift = torch.linalg.norm(command.robot_anchor_pos_w[:, :2] - contact_anchor_xy, dim=-1)
    penalty = torch.clamp((drift - float(drift_limit)) / max(float(drift_scale), 1e-6), min=0.0, max=2.0)
    penalty = torch.where(active, penalty, torch.zeros_like(penalty))
    _ensure_command_metric(command, "post_kick_drift_penalty")[:] = penalty.to(torch.float32)
    return penalty


def arm_raise_penalty_during_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    post_contact_s: float = 0.5,
    elbow_height_margin: float = 0.08,
    wrist_height_margin: float = 0.03,
    height_scale: float = 0.20,
    joint_margin: float = 0.65,
    joint_scale: float = 0.85,
    arm_joint_names: list[str] | None = None,
    arm_joint_targets: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize obvious high-arm and extreme arm-joint postures during the kick window."""
    if arm_joint_names is None:
        arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    counter = _ensure_reward_int_state(env, command_name, "arm_raise_kick_counter", default=-1)
    if torch.any(event.new_contact):
        counter[event.new_contact] = 0

    post_steps = max(0, int(round(float(post_contact_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)
    active = pre_contact | ((counter >= 0) & (counter <= post_steps))

    left_shoulder_z, left_shoulder_ok = _command_body_z(command, "left_shoulder_roll_link", env)
    left_elbow_z, left_elbow_ok = _command_body_z(command, "left_elbow_link", env)
    left_wrist_z, left_wrist_ok = _command_body_z(command, "left_wrist_yaw_link", env)
    right_shoulder_z, right_shoulder_ok = _command_body_z(command, "right_shoulder_roll_link", env)
    right_elbow_z, right_elbow_ok = _command_body_z(command, "right_elbow_link", env)
    right_wrist_z, right_wrist_ok = _command_body_z(command, "right_wrist_yaw_link", env)

    height_terms = []
    if torch.any(left_elbow_ok & left_shoulder_ok):
        height_terms.append(torch.clamp(left_elbow_z - left_shoulder_z - float(elbow_height_margin), min=0.0))
    if torch.any(left_wrist_ok & left_shoulder_ok):
        height_terms.append(torch.clamp(left_wrist_z - left_shoulder_z - float(wrist_height_margin), min=0.0))
    if torch.any(right_elbow_ok & right_shoulder_ok):
        height_terms.append(torch.clamp(right_elbow_z - right_shoulder_z - float(elbow_height_margin), min=0.0))
    if torch.any(right_wrist_ok & right_shoulder_ok):
        height_terms.append(torch.clamp(right_wrist_z - right_shoulder_z - float(wrist_height_margin), min=0.0))
    if height_terms:
        height_excess = torch.stack(height_terms, dim=-1).amax(dim=-1)
    else:
        height_excess = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot = command.robot
    joint_ids, resolved_names = _resolve_robot_joints(robot, arm_joint_names, env.device)
    if joint_ids.numel() > 0:
        targets = _joint_target_tensor(robot, joint_ids, resolved_names, arm_joint_targets, env.device)
        joint_pos = robot.data.joint_pos[:, joint_ids].to(device=env.device, dtype=torch.float32)
        joint_excess = torch.clamp(torch.abs(joint_pos - targets.unsqueeze(0)) - float(joint_margin), min=0.0).mean(dim=-1)
    else:
        joint_excess = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    penalty = torch.clamp(
        height_excess / max(float(height_scale), 1e-6) + joint_excess / max(float(joint_scale), 1e-6),
        min=0.0,
        max=3.0,
    )
    penalty = torch.where(active, penalty, torch.zeros_like(penalty))

    _ensure_command_metric(command, "arm_raise_penalty")[:] = penalty.to(torch.float32)
    _ensure_command_metric(command, "arm_raise_rate")[:] = ((penalty > 0.05) & active).to(torch.float32)
    _ensure_command_metric(command, "arm_height_excess")[:] = torch.where(active, height_excess, torch.zeros_like(height_excess))
    _ensure_command_metric(command, "arm_joint_deviation")[:] = torch.where(active, joint_excess, torch.zeros_like(joint_excess))

    counter = torch.where(counter >= 0, counter + 1, counter)
    setattr(env, _reward_state_name(command_name, "arm_raise_kick_counter"), counter)
    return penalty


def post_kick_arm_neutral(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    delay_s: float = 0.5,
    pos_std: float = 0.45,
    vel_std: float = 3.0,
    arm_joint_names: list[str] | None = None,
    arm_joint_targets: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reward arms returning near the default G1 posture after the follow-through."""
    if arm_joint_names is None:
        arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

    command: MotionCommand = env.command_manager.get_term(command_name)
    counter = _ensure_reward_int_state(env, command_name, "post_kick_stand_still_counter", default=-1)
    delay_steps = max(0, int(round(float(delay_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    active = (counter >= 0) & (counter >= delay_steps)

    robot = command.robot
    joint_ids, resolved_names = _resolve_robot_joints(robot, arm_joint_names, env.device)
    if joint_ids.numel() == 0:
        reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        _ensure_command_metric(command, "post_kick_arm_neutral")[:] = reward
        return reward

    targets = _joint_target_tensor(robot, joint_ids, resolved_names, arm_joint_targets, env.device)
    joint_pos = robot.data.joint_pos[:, joint_ids].to(device=env.device, dtype=torch.float32)
    joint_vel = robot.data.joint_vel[:, joint_ids].to(device=env.device, dtype=torch.float32)
    pos_error = torch.linalg.norm(joint_pos - targets.unsqueeze(0), dim=-1) / max(float(joint_ids.numel()) ** 0.5, 1.0)
    vel_error = torch.linalg.norm(joint_vel, dim=-1) / max(float(joint_ids.numel()) ** 0.5, 1.0)
    score = torch.exp(-(pos_error**2) / max(float(pos_std) ** 2, 1e-6)) * torch.exp(
        -(vel_error**2) / max(float(vel_std) ** 2, 1e-6)
    )
    reward = torch.where(active, score, torch.zeros_like(score))

    _ensure_command_metric(command, "post_kick_arm_neutral")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "post_kick_arm_joint_error")[:] = torch.where(active, pos_error, torch.zeros_like(pos_error))
    _ensure_command_metric(command, "post_kick_arm_joint_vel")[:] = torch.where(active, vel_error, torch.zeros_like(vel_error))
    return reward


def post_kick_upright_feet_planted(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    delay_s: float = 0.5,
    tilt_std: float = 0.16,
    ang_vel_std: float = 0.9,
    drift_std: float = 0.18,
    foot_height_max: float = 0.075,
) -> torch.Tensor:
    """Reward an upright, low-drift, feet-planted finish after the kick."""
    if foot_cfg is None:
        raise ValueError("post_kick_upright_feet_planted requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    counter = _ensure_reward_int_state(env, command_name, "post_kick_stand_still_counter", default=-1)
    contact_anchor_xy = _ensure_reward_vec2_state(env, command_name, "post_kick_contact_anchor_xy")
    delay_steps = max(0, int(round(float(delay_s) / max(float(getattr(env, "step_dt", 0.02)), 1e-6))))
    active = (counter >= 0) & (counter >= delay_steps)

    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], dtype=command.robot_pelvis_quat_w.dtype, device=env.device).expand(
        env.num_envs, -1
    )
    pelvis_gravity = _quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    tilt = torch.linalg.norm(pelvis_gravity[:, :2], dim=-1)
    anchor_ang_vel = torch.linalg.norm(command.robot_anchor_ang_vel_w, dim=-1)
    drift = torch.linalg.norm(command.robot_anchor_pos_w[:, :2] - contact_anchor_xy, dim=-1)

    robot = env.scene[foot_cfg.name]
    foot_indices = torch.as_tensor(robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0], device=env.device)
    foot_z = robot.data.body_pos_w[:, foot_indices, 2]
    feet_planted = torch.all(foot_z < float(foot_height_max), dim=-1)

    score = (
        torch.exp(-(tilt**2) / max(float(tilt_std) ** 2, 1e-6))
        * torch.exp(-(anchor_ang_vel**2) / max(float(ang_vel_std) ** 2, 1e-6))
        * torch.exp(-(drift**2) / max(float(drift_std) ** 2, 1e-6))
        * feet_planted.to(torch.float32)
    )
    reward = torch.where(active, score, torch.zeros_like(score))

    _ensure_command_metric(command, "post_kick_upright_feet_planted")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "post_kick_feet_planted_rate")[:] = (feet_planted & active).to(torch.float32)
    _ensure_command_metric(command, "post_kick_upright_tilt")[:] = torch.where(active, tilt, torch.zeros_like(tilt))
    _ensure_command_metric(command, "post_kick_upright_ang_vel")[:] = torch.where(active, anchor_ang_vel, torch.zeros_like(anchor_ang_vel))
    return reward


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_quat = _heading_aligned_quat(command, command.anchor_quat_w)
    error = quat_error_magnitude(reference_quat, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_foot_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, foot_body_names: list[str] | None = None
) -> torch.Tensor:
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, foot_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    reference_vel = _heading_aligned_vec(command, command.body_lin_vel_w[:, body_indexes])
    error = torch.sum(
        torch.square(reference_vel - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    reference_vel = _heading_aligned_vec(command, command.body_ang_vel_w[:, body_indexes])
    error = torch.sum(
        torch.square(reference_vel - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def foot_distance(env: ManagerBasedRLEnv, threshold: float, std: float, foot_cfg: SceneEntityCfg | None = None,) -> torch.Tensor:
    """Encourage a minimum separation between both feet to avoid crossing/overlap."""
    if foot_cfg is None:
        raise ValueError("foot_distance requires foot_cfg to identify feet.")
    robot = env.scene[foot_cfg.name]
    left_foot_idx = foot_cfg.body_ids[0]
    right_foot_idx = foot_cfg.body_ids[1]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]
    distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)  # [num_envs]
    reward = torch.where(
        distance >= threshold,
        torch.tensor(1., device=distance.device),
        1.0 * torch.exp(-((distance / threshold - 1)**2) / (std ** 2))
    )
    return reward


def _get_scene_sensor(env: ManagerBasedRLEnv, sensor_name: str):
    sensors = getattr(env.scene, "sensors", None)
    if sensors is None:
        return None
    try:
        return sensors[sensor_name] if isinstance(sensors, dict) else getattr(sensors, sensor_name, None)
    except (KeyError, AttributeError, TypeError):
        return None


def _contact_forces_w(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor | None:
    contact_sensor = _get_scene_sensor(env, sensor_name)
    if contact_sensor is None:
        return None
    forces_data = contact_sensor.data
    forces = None
    if hasattr(forces_data, "net_forces_w_history"):
        forces_hist = forces_data.net_forces_w_history
        if forces_hist is not None and forces_hist.numel() > 0:
            forces = forces_hist.to(env.device)
            if forces.ndim >= 4:
                forces = forces.amax(dim=1)
    if forces is None and hasattr(forces_data, "net_forces_w"):
        forces = forces_data.net_forces_w
        if forces is not None and forces.numel() > 0:
            forces = forces.to(env.device)
    if forces is None or forces.ndim < 3:
        return None
    return forces


def _foot_contact_mask(
    env: ManagerBasedRLEnv,
    foot_cfg: SceneEntityCfg,
    sensor_name: str,
    contact_force_threshold: float,
) -> torch.Tensor | None:
    contact_sensor = _get_scene_sensor(env, sensor_name)
    forces = _contact_forces_w(env, sensor_name)
    if contact_sensor is None or forces is None:
        return None

    key = (sensor_name, tuple(foot_cfg.body_names))
    if not hasattr(contact_sensor, "_reward_foot_indices_cache"):
        contact_sensor._reward_foot_indices_cache = {}
    if key not in contact_sensor._reward_foot_indices_cache:
        sensor_indices = contact_sensor.find_bodies(foot_cfg.body_names, preserve_order=True)[0]
        contact_sensor._reward_foot_indices_cache[key] = torch.as_tensor(
            sensor_indices, device=env.device, dtype=torch.long
        )
    foot_indices = contact_sensor._reward_foot_indices_cache[key]
    if foot_indices.numel() == 0 or forces.shape[1] <= int(foot_indices.max()):
        return None
    return forces[:, foot_indices, 2] > float(contact_force_threshold)


def feet_slip_penalty(env: ManagerBasedRLEnv, foot_cfg: SceneEntityCfg, slip_force_threshold: float,) -> torch.Tensor:
    """Penalize foot linear velocity when the foot is in contact.

    A contact is detected when the contact force sensor reports an upward (positive Z)
    force larger than ``slip_force_threshold`` on the foot bodies provided by
    ``foot_cfg``. The penalty mirrors the Isaac Gym style reward, summing the squared
    linear velocity of feet that are currently in contact.
    """

    if foot_cfg is None:
        raise ValueError("foot_cfg cannot be None for _reward_feet_slip_penalty")
    contact_sensor = None
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            contact_sensor = sensors["contact_forces"] if isinstance(sensors, dict) else getattr(sensors, "contact_forces", None)
        except (KeyError, AttributeError, TypeError):
            contact_sensor = None
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    device = env.device
    num_envs = env.num_envs
    forces = None
    forces_data = contact_sensor.data
    if hasattr(forces_data, "net_forces_w_history"):
        forces_hist = forces_data.net_forces_w_history
        if forces_hist.numel() > 0:
            forces = forces_hist.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            forces = forces_data.net_forces_w
            if forces is not None and forces.numel() > 0:
                forces = forces.to(device)
            else:
                return torch.zeros(num_envs, device=device, dtype=torch.float32)
        else:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)
    if forces.ndim < 3:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    robot = env.scene[foot_cfg.name]

    foot_indices_key = tuple(foot_cfg.body_names)
    if not hasattr(contact_sensor, '_foot_indices_cache'):
        contact_sensor._foot_indices_cache = {}
    if foot_indices_key not in contact_sensor._foot_indices_cache:
        foot_sensor_indices = contact_sensor.find_bodies(foot_cfg.body_names, preserve_order=True)[0]
        contact_sensor._foot_indices_cache[foot_indices_key] = torch.as_tensor(
            foot_sensor_indices, device=device, dtype=torch.long
        )
    foot_indices = contact_sensor._foot_indices_cache[foot_indices_key]

    max_foot_idx = int(foot_indices.max()) if len(foot_indices) > 0 else -1
    if forces.shape[1] <= max_foot_idx:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)
    vertical_forces = forces[:, foot_indices, 2]
    contact_mask = vertical_forces > slip_force_threshold
    foot_vel_w = robot.data.body_lin_vel_w[:, foot_indices]
    penalize = torch.where(
        contact_mask.unsqueeze(-1), 
        torch.square(foot_vel_w), 
        torch.zeros_like(foot_vel_w)
    )
    if penalize.numel() > 10000:  # Heuristic threshold; tune if needed.
        return penalize.reshape(num_envs, -1).sum(dim=1)
    else:
        return torch.sum(penalize, dim=(1, 2))


def pre_contact_feet_slip_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    sensor_name: str = "contact_forces",
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize horizontal foot speed while the foot is planted before ball contact."""
    if foot_cfg is None:
        raise ValueError("pre_contact_feet_slip_penalty requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)

    contact_mask = _foot_contact_mask(env, foot_cfg, sensor_name, contact_force_threshold)
    if contact_mask is None:
        slip = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    else:
        robot = env.scene[foot_cfg.name]
        robot_foot_indices = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            device=env.device,
            dtype=torch.long,
        )
        foot_vel_xy = robot.data.body_lin_vel_w[:, robot_foot_indices, :2]
        planted_vel_xy = torch.where(contact_mask.unsqueeze(-1), foot_vel_xy, torch.zeros_like(foot_vel_xy))
        slip = torch.sum(torch.square(planted_vel_xy), dim=(1, 2))

    penalty = torch.where(pre_contact, slip, torch.zeros_like(slip))
    _ensure_command_metric(command, "pre_contact_foot_slip")[:] = penalty.to(torch.float32)
    _ensure_command_metric(command, "pre_contact_feet_slip_penalty")[:] = penalty.to(torch.float32)
    return penalty


def pre_contact_swing_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    sensor_name: str = "contact_forces",
    contact_force_threshold: float = 5.0,
    target_clearance: float = 0.085,
    cap: float = 0.14,
) -> torch.Tensor:
    """Reward swing-foot clearance relative to a planted support foot before contact."""
    if foot_cfg is None:
        raise ValueError("pre_contact_swing_foot_clearance_reward requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)

    contact_mask = _foot_contact_mask(env, foot_cfg, sensor_name, contact_force_threshold)
    robot = env.scene[foot_cfg.name]
    robot_foot_indices = torch.as_tensor(
        robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
        device=env.device,
        dtype=torch.long,
    )
    foot_z = robot.data.body_pos_w[:, robot_foot_indices, 2]

    if contact_mask is None:
        has_support = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        swing_clearance = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    else:
        has_support = torch.any(contact_mask, dim=-1)
        support_count = contact_mask.to(torch.float32).sum(dim=-1).clamp(min=1.0)
        support_z = torch.sum(torch.where(contact_mask, foot_z, torch.zeros_like(foot_z)), dim=-1) / support_count
        swing_mask = ~contact_mask
        clearance = torch.clamp(foot_z - support_z.unsqueeze(-1), min=0.0)
        clearance = torch.where(swing_mask, clearance, torch.zeros_like(clearance))
        swing_clearance = clearance.max(dim=-1).values

    active = pre_contact & has_support
    capped_clearance = torch.clamp(swing_clearance, min=0.0, max=float(cap))
    reward = torch.where(
        active,
        capped_clearance / max(float(target_clearance), 1e-6),
        torch.zeros_like(capped_clearance),
    )

    _ensure_command_metric(command, "pre_contact_swing_foot_clearance")[:] = torch.where(
        pre_contact, swing_clearance, torch.zeros_like(swing_clearance)
    ).to(torch.float32)
    _ensure_command_metric(command, "pre_contact_has_support")[:] = (pre_contact & has_support).to(torch.float32)
    _ensure_command_metric(command, "pre_contact_swing_foot_clearance_reward")[:] = reward.to(torch.float32)
    return reward


def pre_contact_step_length_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    min_ball_distance: float = 0.30,
    target_step_length: float = 0.28,
    cap: float = 0.42,
) -> torch.Tensor:
    """Reward a slightly longer sagittal step before the robot reaches the ball."""
    if foot_cfg is None:
        raise ValueError("pre_contact_step_length_reward requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = ~tracker.get_contact_awarded().to(device=env.device, dtype=torch.bool)

    robot = env.scene[foot_cfg.name]
    robot_foot_indices = torch.as_tensor(
        robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
        device=env.device,
        dtype=torch.long,
    )
    foot_pos_xy = robot.data.body_pos_w[:, robot_foot_indices, :2]
    foot_delta_xy = foot_pos_xy[:, 0] - foot_pos_xy[:, 1]
    direction_xy = _goal_direction_xy(command).to(device=env.device, dtype=foot_delta_xy.dtype)
    step_length = torch.abs(torch.sum(foot_delta_xy * direction_xy, dim=-1))

    ball_distance = torch.linalg.norm(command.target_point_pos[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    active = pre_contact & (ball_distance > float(min_ball_distance))
    capped_length = torch.clamp(step_length, min=0.0, max=float(cap))
    reward = torch.where(
        active,
        capped_length / max(float(target_step_length), 1e-6),
        torch.zeros_like(capped_length),
    )

    _ensure_command_metric(command, "pre_contact_step_length")[:] = torch.where(
        pre_contact, step_length, torch.zeros_like(step_length)
    ).to(torch.float32)
    _ensure_command_metric(command, "pre_contact_step_length_reward")[:] = reward.to(torch.float32)
    return reward
    

def target_point_proximity(env: ManagerBasedRLEnv, std: float, command_name: str = "motion",) -> torch.Tensor:
    """Reward proximity to the target point (ball) and freeze at first kick contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    
    # Compute current proximity reward.
    base_xy = command.robot_anchor_pos_w[..., :2]
    target = get_target_point_world(env, command_name).to(device=base_xy.device, dtype=base_xy.dtype)
    diff_xy = base_xy - target[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)
    
    # Query kick-contact status.
    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Freeze reward for environments that just kicked this step.
    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Return frozen reward after contact; otherwise return current reward.
    reward = torch.where(contact_awarded, frozen_reward, proximity_reward)
    return reward


def target_point_contact(env: ManagerBasedRLEnv, 
        horizontal_force_threshold: float = 0.0,
        command_name: str = "motion",
        ball_sensor_name: str = "soccer_ball_contact",
        foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """One-shot reward for contacting the ball at first valid touch."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward
    # print(event.new_contact.to(reward.dtype))
    reward_scale = torch.zeros_like(reward)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
            correct_mask[foot_info.env_ids] = correct

    tracker.record_expected_success(event.new_contact, correct_mask)
    # print("contact", event.new_contact.to(reward.dtype) * reward_scale)
    return event.new_contact.to(reward.dtype) * reward_scale


def autonomous_target_point_contact(
    env: ManagerBasedRLEnv,
    horizontal_force_threshold: float = 0.0,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """One-shot ball contact reward without hidden expected-foot conditioning."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = event.new_contact.to(device=env.device, dtype=torch.float32)
    if foot_cfg is not None and torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)
    return reward


def wrong_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """One-shot penalty when first ball contact comes from the non-expected foot."""
    if foot_cfg is None:
        raise ValueError("wrong_foot_contact_penalty requires foot_cfg to identify kicking feet.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return penalty

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return penalty
    tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)

    valid_expectation = foot_info.expected >= 0
    correct = (foot_info.sides == foot_info.expected) & valid_expectation
    wrong = (~correct) & valid_expectation
    penalty[foot_info.env_ids] = wrong.to(penalty.dtype)

    correct_metric = _ensure_command_metric(command, "correct_foot_contact_rate")
    wrong_metric = _ensure_command_metric(command, "wrong_foot_contact_rate")
    correct_metric[:] = 0.0
    wrong_metric[:] = 0.0
    correct_metric[foot_info.env_ids] = correct.to(correct_metric.dtype)
    wrong_metric[foot_info.env_ids] = wrong.to(wrong_metric.dtype)
    return penalty


def _first_contact_ball_local(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
    foot_cfg: SceneEntityCfg,
):
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if not torch.any(event.new_contact):
        return command, None, None

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return command, foot_info, None

    tracker.record_contact_foot(foot_info.env_ids, foot_info.sides)
    robot = command.robot
    foot_pos_w = robot.data.body_pos_w[foot_info.env_ids, foot_info.body_indices]
    foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]
    ball_pos = command.soccer_ball_pos[foot_info.env_ids]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        ball_pos = ball_pos + env_origins[foot_info.env_ids]
    ball_local = quat_apply(quat_inv(foot_quat_w), ball_pos - foot_pos_w)
    return command, foot_info, ball_local


def inside_foot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    inside_y_range: tuple[float, float] = (0.02, 0.18),
    x_range: tuple[float, float] = (-0.12, 0.18),
    z_abs_max: float = 0.16,
    y_target: float = 0.09,
    y_std: float = 0.08,
) -> torch.Tensor:
    """One-shot reward for contacting the ball with the expected inside-foot region."""
    if foot_cfg is None:
        raise ValueError("inside_foot_contact_reward requires foot_cfg.")

    command, foot_info, ball_local = _first_contact_ball_local(
        env, command_name, ball_sensor_name, horizontal_force_threshold, foot_cfg
    )
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    inside_metric = _ensure_command_metric(command, "inside_foot_contact_rate")
    toe_metric = _ensure_command_metric(command, "toe_contact_rate")
    local_x_metric = _ensure_command_metric(command, "foot_local_ball_x_mean")
    local_y_metric = _ensure_command_metric(command, "foot_local_ball_y_mean")
    inside_metric[:] = 0.0
    toe_metric[:] = 0.0
    local_x_metric[:] = 0.0
    local_y_metric[:] = 0.0
    if foot_info is None or ball_local is None or foot_info.env_ids.numel() == 0:
        return reward

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.where(
        expected_leg == 0,
        torch.full((foot_info.env_ids.numel(),), -1.0, device=env.device),
        torch.full((foot_info.env_ids.numel(),), 1.0, device=env.device),
    )
    inside_y = ball_local[:, 1] * desired_sign
    correct_foot = (foot_info.sides == foot_info.expected) & (expected_leg >= 0)
    in_lateral_band = (inside_y >= float(inside_y_range[0])) & (inside_y <= float(inside_y_range[1]))
    in_x_band = (ball_local[:, 0] >= float(x_range[0])) & (ball_local[:, 0] <= float(x_range[1]))
    in_z_band = torch.abs(ball_local[:, 2]) <= float(z_abs_max)
    inside = correct_foot & in_lateral_band & in_x_band & in_z_band
    shaped = torch.exp(-((inside_y - float(y_target)) ** 2) / (float(y_std) ** 2))
    reward[foot_info.env_ids] = inside.to(torch.float32) * shaped

    inside_metric[foot_info.env_ids] = inside.to(inside_metric.dtype)
    local_x_metric[foot_info.env_ids] = ball_local[:, 0].to(local_x_metric.dtype)
    local_y_metric[foot_info.env_ids] = ball_local[:, 1].to(local_y_metric.dtype)
    return reward


def autonomous_inside_foot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    inside_y_range: tuple[float, float] = (0.02, 0.18),
    x_range: tuple[float, float] = (-0.12, 0.18),
    z_abs_max: float = 0.16,
    y_target: float = 0.09,
    y_std: float = 0.08,
) -> torch.Tensor:
    """Inside-foot reward based on the actual first-contact foot, not a hidden motion label."""
    if foot_cfg is None:
        raise ValueError("autonomous_inside_foot_contact_reward requires foot_cfg.")

    command, foot_info, ball_local = _first_contact_ball_local(
        env, command_name, ball_sensor_name, horizontal_force_threshold, foot_cfg
    )
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    inside_metric = _ensure_command_metric(command, "inside_foot_contact_rate")
    toe_metric = _ensure_command_metric(command, "toe_contact_rate")
    local_x_metric = _ensure_command_metric(command, "foot_local_ball_x_mean")
    local_y_metric = _ensure_command_metric(command, "foot_local_ball_y_mean")
    inside_metric[:] = 0.0
    toe_metric[:] = 0.0
    local_x_metric[:] = 0.0
    local_y_metric[:] = 0.0
    if foot_info is None or ball_local is None or foot_info.env_ids.numel() == 0:
        return reward

    actual_leg = foot_info.sides.to(device=env.device, dtype=torch.int8)
    valid_leg = actual_leg >= 0
    desired_sign = torch.where(
        actual_leg == 0,
        torch.full((foot_info.env_ids.numel(),), -1.0, device=env.device),
        torch.full((foot_info.env_ids.numel(),), 1.0, device=env.device),
    )
    inside_y = ball_local[:, 1] * desired_sign
    in_lateral_band = (inside_y >= float(inside_y_range[0])) & (inside_y <= float(inside_y_range[1]))
    in_x_band = (ball_local[:, 0] >= float(x_range[0])) & (ball_local[:, 0] <= float(x_range[1]))
    in_z_band = torch.abs(ball_local[:, 2]) <= float(z_abs_max)
    inside = valid_leg & in_lateral_band & in_x_band & in_z_band
    shaped = torch.exp(-((inside_y - float(y_target)) ** 2) / (float(y_std) ** 2))
    reward[foot_info.env_ids] = inside.to(torch.float32) * shaped

    inside_metric[foot_info.env_ids] = inside.to(inside_metric.dtype)
    local_x_metric[foot_info.env_ids] = ball_local[:, 0].to(local_x_metric.dtype)
    local_y_metric[foot_info.env_ids] = ball_local[:, 1].to(local_y_metric.dtype)
    return reward


def toe_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    toe_x_min: float = 0.16,
    inside_y_abs_max: float = 0.05,
) -> torch.Tensor:
    """One-shot penalty when the first contact is concentrated near the toe region."""
    if foot_cfg is None:
        raise ValueError("toe_contact_penalty requires foot_cfg.")

    command, foot_info, ball_local = _first_contact_ball_local(
        env, command_name, ball_sensor_name, horizontal_force_threshold, foot_cfg
    )
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    toe_metric = _ensure_command_metric(command, "toe_contact_rate")
    if foot_info is None or ball_local is None or foot_info.env_ids.numel() == 0:
        return penalty

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.where(
        expected_leg == 0,
        torch.full((foot_info.env_ids.numel(),), -1.0, device=env.device),
        torch.full((foot_info.env_ids.numel(),), 1.0, device=env.device),
    )
    inside_y = ball_local[:, 1] * desired_sign
    correct_foot = (foot_info.sides == foot_info.expected) & (expected_leg >= 0)
    toe_contact = correct_foot & (ball_local[:, 0] > float(toe_x_min)) & (torch.abs(inside_y) < float(inside_y_abs_max))
    penalty[foot_info.env_ids] = toe_contact.to(penalty.dtype)
    toe_metric[foot_info.env_ids] = toe_contact.to(toe_metric.dtype)
    return penalty


def autonomous_toe_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    toe_x_min: float = 0.16,
    inside_y_abs_max: float = 0.05,
) -> torch.Tensor:
    """Toe-contact penalty based on the actual first-contact foot."""
    if foot_cfg is None:
        raise ValueError("autonomous_toe_contact_penalty requires foot_cfg.")

    command, foot_info, ball_local = _first_contact_ball_local(
        env, command_name, ball_sensor_name, horizontal_force_threshold, foot_cfg
    )
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    toe_metric = _ensure_command_metric(command, "toe_contact_rate")
    if foot_info is None or ball_local is None or foot_info.env_ids.numel() == 0:
        return penalty

    actual_leg = foot_info.sides.to(device=env.device, dtype=torch.int8)
    valid_leg = actual_leg >= 0
    desired_sign = torch.where(
        actual_leg == 0,
        torch.full((foot_info.env_ids.numel(),), -1.0, device=env.device),
        torch.full((foot_info.env_ids.numel(),), 1.0, device=env.device),
    )
    inside_y = ball_local[:, 1] * desired_sign
    toe_contact = valid_leg & (ball_local[:, 0] > float(toe_x_min)) & (torch.abs(inside_y) < float(inside_y_abs_max))
    penalty[foot_info.env_ids] = toe_contact.to(penalty.dtype)
    toe_metric[foot_info.env_ids] = toe_contact.to(toe_metric.dtype)
    return penalty


def ball_side_expected_target_point_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Low-value contact reward only when the first-contact foot matches ball-side selection."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["expected_foot_contact"]


def autonomous_side_foot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    inside_y_range: tuple[float, float] = (0.035, 0.145),
    side_x_range: tuple[float, float] = (-0.08, 0.11),
    z_abs_max: float = 0.16,
    side_y_target: float = 0.085,
    side_y_std: float = 0.045,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Reward a correct-foot first touch on the medial side of that foot."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        inside_y_range=inside_y_range,
        side_x_range=side_x_range,
        z_abs_max=z_abs_max,
        side_y_target=side_y_target,
        side_y_std=side_y_std,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["side_foot_contact"]


def ball_side_wrong_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Penalty when first contact uses the foot opposite to the ball-side rule."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["wrong_foot_contact"]


def side_foot_toe_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    toe_x_min: float = 0.12,
    toe_y_abs_max: float = 0.075,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Penalty for toe-like contact under the ball-side foot-selection rule."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        toe_x_min=toe_x_min,
        toe_y_abs_max=toe_y_abs_max,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["toe_contact"]


def side_foot_instep_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    instep_x_range: tuple[float, float] = (-0.05, 0.15),
    instep_y_abs_max: float = 0.045,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Penalty for central instep/foot-back contact instead of medial-side contact."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        instep_x_range=instep_x_range,
        instep_y_abs_max=instep_y_abs_max,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["instep_contact"]


def lateral_side_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    inside_y_range: tuple[float, float] = (0.035, 0.145),
    side_x_range: tuple[float, float] = (-0.08, 0.11),
    z_abs_max: float = 0.16,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
) -> torch.Tensor:
    """Penalty when the correct foot touches with the outside/lateral side."""
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        inside_y_range=inside_y_range,
        side_x_range=side_x_range,
        z_abs_max=z_abs_max,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )
    return terms["lateral_foot_contact"]


def sideways_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Single-shot reward encouraging foot swing along the expected lateral axis.
    Left kick expects foot velocity along local -Y; right kick expects local +Y.
    """
    if foot_cfg is None:
        raise ValueError("sideways_kick_reward requires foot_cfg to identify kicking feet.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return reward

    robot = command.robot
    foot_vel_w = robot.data.body_lin_vel_w[foot_info.env_ids, foot_info.body_indices]
    foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]

    vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
    vel_norm = torch.norm(vel_local, dim=-1)

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.zeros(expected_leg.shape, device=env.device, dtype=torch.float32)
    desired_sign = torch.where(expected_leg == 0, torch.full_like(desired_sign, -1.0), desired_sign)
    desired_sign = torch.where(expected_leg == 1, torch.full_like(desired_sign, 1.0), desired_sign)

    directional_component = vel_local[:, 1] * desired_sign
    axis_component = torch.clamp(directional_component, min=0.0)

    alignment = torch.where(vel_norm > 1e-6, axis_component / vel_norm, torch.zeros_like(vel_norm))
    reward[foot_info.env_ids] = alignment.to(reward.dtype)

    # Reward only when expected leg is valid and contact leg matches expectation.
    valid_expectation = expected_leg >= 0
    correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
    wrong_mask = ~correct_foot
    if torch.any(wrong_mask):
        reward[foot_info.env_ids[wrong_mask]] = 0.0
    # print("sideways_kick reward:", reward)
    return reward


def side_foot_contact_leg_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    inside_y_range: tuple[float, float] = (0.035, 0.145),
    side_x_range: tuple[float, float] = (-0.08, 0.11),
    z_abs_max: float = 0.16,
    side_y_target: float = 0.085,
    side_y_std: float = 0.045,
    medial_sign_left: float = -1.0,
    medial_sign_right: float = 1.0,
    leg_speed_scale: float = 2.0,
    cap: float = 1.5,
) -> torch.Tensor:
    """One-shot leg-speed reward gated by correct side-foot first contact."""
    if foot_cfg is None:
        raise ValueError("side_foot_contact_leg_speed_reward requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
        inside_y_range=inside_y_range,
        side_x_range=side_x_range,
        z_abs_max=z_abs_max,
        side_y_target=side_y_target,
        side_y_std=side_y_std,
        medial_sign_left=medial_sign_left,
        medial_sign_right=medial_sign_right,
    )

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    speed_state = _ensure_reward_float_state(env, command_name, "side_foot_leg_speed", default=0.0)
    reward_state = _ensure_reward_float_state(env, command_name, "side_foot_leg_speed_reward", default=0.0)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            env_ids = foot_info.env_ids
            speed_state[env_ids] = 0.0
            reward_state[env_ids] = 0.0

            robot = command.robot
            foot_vel_w = robot.data.body_lin_vel_w[env_ids, foot_info.body_indices]
            foot_quat_w = robot.data.body_quat_w[env_ids, foot_info.body_indices]
            vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)

            actual_leg = foot_info.sides.to(device=env.device, dtype=torch.int8)
            expected_foot = _expected_foot_from_ball_y(_initial_ball_base_xy(command, env)[:, 1], center_deadband)
            expected = expected_foot[env_ids].to(device=env.device, dtype=torch.int8)
            valid_leg = actual_leg >= 0
            correct_foot = valid_leg & (actual_leg == expected)
            side_contact = terms["side_foot_contact"][env_ids] > 0.0

            desired_sign = torch.where(
                actual_leg == 0,
                torch.full((env_ids.numel(),), -1.0, device=env.device),
                torch.full((env_ids.numel(),), 1.0, device=env.device),
            )
            leg_speed = torch.clamp(vel_local[:, 1] * desired_sign, min=0.0)
            active = correct_foot & side_contact
            if torch.any(active):
                shaped = torch.clamp(
                    leg_speed[active] / max(float(leg_speed_scale), 1e-6),
                    min=0.0,
                    max=float(cap),
                )
                reward[env_ids[active]] = shaped.to(reward.dtype)
                speed_state[env_ids[active]] = leg_speed[active].to(speed_state.dtype)
                reward_state[env_ids[active]] = shaped.to(reward_state.dtype)

    _ensure_command_metric(command, "side_foot_leg_speed")[:] = speed_state.to(torch.float32)
    _ensure_command_metric(command, "side_foot_leg_speed_reward")[:] = reward_state.to(torch.float32)
    return reward



def ball_velocity_direction_alignment(
    env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward alignment between ball velocity direction and pre-kick target-to-destination direction.

    Active only for a short window after contact with the expected foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    vel_xy = vel[:, :2]  # x-y plane projection
    vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
    vel_norm = torch.norm(vel, dim=-1, keepdim=True)
    
    # Direction vector from pre-kick target point (ball) to destination.
    direction = command.target_destination_pos - command.initial_target_point_pos  # [num_envs, 3]
    direction_xy = direction[:, :2]
    dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)

    valid_mask = (vel_norm.squeeze(-1) > velocity_threshold) & (vel_xy_norm.squeeze(-1) > 1e-6) & (
        dir_norm.squeeze(-1) > 1e-6
    )

    # Track average angle based on initial direction vectors.
    avg_angle = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    if torch.any(valid_mask):
        dir_unit_valid = direction_xy[valid_mask] / dir_norm[valid_mask]
        vel_unit_valid = vel_xy[valid_mask] / vel_xy_norm[valid_mask]
        cos_theta_valid = torch.sum(vel_unit_valid * dir_unit_valid, dim=-1).clamp(-1.0, 1.0)
        theta_valid = torch.acos(cos_theta_valid)
        avg_angle = theta_valid.mean()
    if hasattr(command, "metrics"):
        command.metrics["ball_velocity_dir_alignment_angle"] = torch.full(
            (env.num_envs,), avg_angle.item(), device=env.device, dtype=torch.float32
        )
    
    # Reward window.
    timer_name = f"_{command_name}_dir_align_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speeds in active_mask to avoid division by zero.
    speed_valid = (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        dir_unit = direction_xy[active_mask] / dir_norm[active_mask]
        vel_unit = vel_xy[active_mask] / vel_xy_norm[active_mask]
        cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cos_theta) ** 2
        reward[active_mask] = torch.exp(-error / (std ** 2))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_velocity_direction_alignment reward:", timer,reward)
    return reward


def ball_speed_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """Reward ball speed within a short window after expected-foot contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    speed_xy = torch.norm(vel[:, :2], dim=-1)  # x-y plane speed

    timer_name = f"_{command_name}_speed_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speed in active_mask to avoid division by zero.
    speed_valid = speed_xy > 1e-6
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        reward_active = 1.0 - torch.exp(-(speed_xy[active_mask] ** 2) / (std ** 2))
        reward[active_mask] = reward_active

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_speed_reward:", reward)
    return reward


def autonomous_ball_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    window_steps: int = 12,
) -> torch.Tensor:
    """Reward horizontal ball speed after any valid first kick contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    speed_xy = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)

    timer_name = f"_{command_name}_autonomous_speed_timer"
    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if torch.any(event.new_contact):
        timer[event.new_contact] = int(window_steps)

    active = (timer > 0) & (speed_xy > float(velocity_threshold))
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active):
        reward[active] = 1.0 - torch.exp(-(speed_xy[active] ** 2) / max(float(std) ** 2, 1e-6))

    _ensure_command_metric(command, "autonomous_ball_speed")[:] = reward
    _ensure_command_metric(command, "autonomous_ball_speed_xy")[:] = speed_xy.to(torch.float32)

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    return reward


def side_foot_ball_speed_lite_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 2.4,
    velocity_threshold: float = 0.15,
) -> torch.Tensor:
    """Light post-contact speed reward paid only after side-foot contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    side_state = _ensure_reward_bool_state(env, command_name, "side_foot_contact_awarded")
    soccer_ball = env.scene["soccer_ball"]
    vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    forward_vel = torch.sum(vel_xy * _goal_direction_xy(command), dim=-1)
    active = side_state & (forward_vel > float(velocity_threshold))

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active):
        reward[active] = 1.0 - torch.exp(-(forward_vel[active] ** 2) / max(float(std) ** 2, 1e-6))

    _ensure_command_metric(command, "side_foot_ball_speed_lite")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "side_foot_ball_speed_lite_forward_vel")[:] = forward_vel.to(torch.float32)
    return reward


def style_gated_side_foot_ball_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 2.2,
    velocity_threshold: float = 0.15,
    horizontal_force_threshold: float = 10.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
    center_deadband: float = 0.08,
    window_steps: int = 12,
    torso_pitch_threshold: float = 0.18,
    torso_pitch_scale: float = 0.22,
) -> torch.Tensor:
    """Reward ball speed only when the kick preserves the side-foot style."""
    if foot_cfg is None:
        raise ValueError("style_gated_side_foot_ball_speed_reward requires foot_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    terms = _side_foot_contact_terms(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=horizontal_force_threshold,
        foot_cfg=foot_cfg,
        center_deadband=center_deadband,
    )

    timer_name = f"_{command_name}_style_gated_speed_timer"
    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    if torch.any(event.new_contact):
        timer[event.new_contact] = int(window_steps)

    side_state = terms["side_foot_state"].to(device=env.device, dtype=torch.bool)
    toe_state = _ensure_reward_bool_state(env, command_name, "toe_contact_awarded")
    instep_state = _ensure_reward_bool_state(env, command_name, "instep_contact_awarded")

    gravity_vec_w = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=command.robot_pelvis_quat_w.dtype,
        device=env.device,
    ).expand(env.num_envs, -1)
    pelvis_gravity = _quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    pitch_mag = torch.abs(pelvis_gravity[:, 0])
    torso_gate = torch.exp(
        -(torch.clamp(pitch_mag - float(torso_pitch_threshold), min=0.0) ** 2)
        / max(float(torso_pitch_scale) ** 2, 1e-6)
    )

    soccer_ball = env.scene["soccer_ball"]
    vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
    forward_vel = torch.sum(vel_xy * _goal_direction_xy(command), dim=-1)
    active = (timer > 0) & (forward_vel > float(velocity_threshold))
    style_ok = side_state & (~toe_state) & (~instep_state)

    base_reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active):
        base_reward[active] = 1.0 - torch.exp(-(forward_vel[active] ** 2) / max(float(std) ** 2, 1e-6))
    reward = base_reward * style_ok.to(torch.float32) * torso_gate.to(torch.float32)

    _ensure_command_metric(command, "style_gated_ball_speed")[:] = reward.to(torch.float32)
    _ensure_command_metric(command, "style_gated_ball_speed_raw")[:] = base_reward.to(torch.float32)
    _ensure_command_metric(command, "style_gated_ball_speed_gate")[:] = style_ok.to(torch.float32) * torso_gate.to(torch.float32)

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    return reward


def ball_z_speed_penalty_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    ) -> torch.Tensor:
    """Penalize excessive vertical ball speed in a short post-activation window."""
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    z_speed = vel[:, 2]  # vertical speed
    speed = torch.norm(vel, dim=-1)

    valid_mask = speed > velocity_threshold

    timer_name = f"_{command_name}_z_speed_timer"
    prev_name = f"_{command_name}_z_speed_prev"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    prev_valid = getattr(env, prev_name, None)
    if prev_valid is None or prev_valid.shape[0] != env.num_envs:
        prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        prev_valid = prev_valid.to(device=env.device, dtype=torch.bool)

    rising_mask = valid_mask & (~prev_valid)
    timer[rising_mask] = 5
    active_mask = timer > 0

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        scale = std if std > 0 else 1.0
        reward[active_mask] = torch.tanh(torch.abs(z_speed[active_mask]) / (scale + 1e-8))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    setattr(env, prev_name, valid_mask.to(dtype=torch.bool))
    # print("ball_z_speed_penalty_reward:", reward)
    return reward


def pelvis_orientation(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Penalize pelvis pitch/roll tilt to keep the robot upright."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    gravity_vec_w = robot.data.GRAVITY_VEC_W
    
    # Project gravity vector to pelvis local frame.
    pelvis_proj_gravity = _quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    # print("pelvis_proj_gravity:", gravity_vec_w, pelvis_proj_gravity)
    return torch.sum(torch.square(pelvis_proj_gravity[:, :2]), dim=1)
