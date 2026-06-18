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
    lateral_error = torch.sum(rel_curr * lateral_dir_xy, dim=-1)
    forward_speed = (curr_forward - prev_forward) / max(float(dt), 1e-6)

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
