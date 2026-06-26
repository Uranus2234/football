from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_anchor_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.joint_vel.view(env.num_envs, -1)


def _get_motion_command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    command: MotionCommand | None = env.command_manager.get_term(command_name)
    if command is None:
        raise RuntimeError(f"motion command '{command_name}' not found in env.command_manager")
    if not hasattr(command, "target_point_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_point_pos attribute")
    return command


def get_target_point_world(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_local = command.target_point_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def get_target_destination_world(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    if not hasattr(command, "target_destination_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_destination_pos attribute")
    target_local = command.target_destination_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def get_target_point_base(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_world = get_target_point_world(env, command_name)
    # delta = target_world - command.robot_anchor_pos_w
    delta = target_world - command.robot_pelvis_pos_w
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def _positional_encoding(vec: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Apply sinusoidal positional encoding to a target tensor of shape (E, 3).

    The encoding follows Transformer-style frequencies: for each coordinate x,
    compute sin(2^k*pi*x) and cos(2^k*pi*x) for k=0..num_freqs-1, then
    concatenate with the original coordinates.
    """
    if num_freqs <= 0:
        return vec.view(vec.shape[0], -1)

    device = vec.device
    dtype = vec.dtype
    # freqs: [num_freqs]
    freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
    # vec: [E, 3] -> vec_exp: [E, 3, num_freqs]
    vec_exp = vec.unsqueeze(-1) * freqs
    sin = torch.sin(vec_exp)
    cos = torch.cos(vec_exp)
    # sin_cos: [E, 3, 2*num_freqs] -> flatten per-sample
    sin_cos = torch.cat([sin, cos], dim=-1).view(vec.shape[0], -1)
    # Concatenate original coordinates in front.
    return torch.cat([vec.view(vec.shape[0], -1), sin_cos], dim=-1)


def target_point_pos_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_point_cache"
    target_local = get_target_point_base(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target point caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # Return cached target vector.
    return getattr(env, cache_name)
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)


def constant_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    # Constant observation path keeps the same representation as policy inputs.
    base = get_target_point_base(env, command_name)
    return base
    return _positional_encoding(base, num_freqs=6)


def blind_zone_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return target point in robot base frame with blind-zone simulation.
    
    If robot-ball (x, y) distance is outside [blind_distance_min, blind_distance_max],
    return the last visible position to emulate limited visibility.
    Thresholds are resampled from MotionCommandCfg ranges at each resample.
    """
    command = _get_motion_command(env, command_name)
    
    # Current target in robot base frame.
    target_base = get_target_point_base(env, command_name)

    # Initialize the cache at the first frame of each episode.  This keeps the
    # observation dimension unchanged and avoids returning a fake zero vector if
    # the ball is already outside the visible range when the episode starts.
    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        first_step_mask = step_buf == 0
        if torch.any(first_step_mask):
            command.last_visible_target_point_base[first_step_mask] = target_base[first_step_mask]
            command.is_in_blind_zone[first_step_mask] = False
    
    # Compute robot-target (x, y) distance in world coordinates.
    target_world = get_target_point_world(env, command_name)
    robot_pos = command.robot_pelvis_pos_w
    # Horizontal distance only.
    distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)
    
    # Visible only when distance is within [min, max].
    in_visible_range = (distance_xy >= command.blind_distance_min) & (distance_xy <= command.blind_distance_max)

    # Additional random perception dropout while the ball is otherwise visible.
    # The first frame is kept visible so the cache always starts from a valid observation.
    dropout_prob = float(getattr(command.cfg, "blind_dropout_prob", 0.0))
    if dropout_prob > 0.0:
        dropout_prob = min(max(dropout_prob, 0.0), 1.0)
        random_visible = torch.rand(env.num_envs, device=target_base.device) >= dropout_prob
        if step_buf is not None:
            random_visible = torch.where(first_step_mask, torch.ones_like(random_visible), random_visible)
        in_visible_range = in_visible_range & random_visible
    
    # Update last visible target for visible environments.
    if torch.any(in_visible_range):
        command.last_visible_target_point_base[in_visible_range] = target_base[in_visible_range]
        command.is_in_blind_zone[in_visible_range] = False
    
    # Mark blind-zone environments.
    command.is_in_blind_zone[~in_visible_range] = True
    
    # Return last visible position in blind zone, otherwise current target.
    result = torch.where(
        command.is_in_blind_zone.unsqueeze(-1),
        command.last_visible_target_point_base,
        target_base
    )
    # print("blind zone target point:", command.blind_distance_min, command.blind_distance_max, result)
    return result


def target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_world = get_target_destination_world(env, command_name)
    delta = target_world - command.robot_pelvis_pos_w
    # print("position:", quat_apply(quat_inv(command.robot_pelvis_quat_w), delta))
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def desired_kick_direction_base(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return the desired field-goal kick direction in robot/pelvis coordinates.

    The destination comes from field/global localization, while the ball point is
    the YOLO/depth-like latched perception when available.  This keeps training
    aligned with deployment: after kick latch or visual dropout, direction is
    computed against the held ball observation instead of live sim truth.
    """
    command = _get_motion_command(env, command_name)
    destination_base = target_destination_pos_local(env, command_name)
    truth_ball_base = get_target_point_base(env, command_name)

    latched_ball_base = getattr(command, "perception_latched_target_point_base", None)
    if latched_ball_base is None:
        ball_base = truth_ball_base
    else:
        latched_ball_base = latched_ball_base.to(device=truth_ball_base.device, dtype=truth_ball_base.dtype)
        last_update_step = getattr(command, "perception_last_update_step", None)
        if last_update_step is None:
            has_latch = torch.ones(env.num_envs, dtype=torch.bool, device=truth_ball_base.device)
        else:
            has_latch = last_update_step.to(device=truth_ball_base.device) >= 0
        ball_base = torch.where(has_latch.unsqueeze(-1), latched_ball_base, truth_ball_base)

    direction_xy = destination_base[:, :2] - ball_base[:, :2]
    norm = torch.linalg.norm(direction_xy, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_b = direction_xy / norm

    yaw_noise = getattr(command, "kick_direction_yaw_noise", None)
    if yaw_noise is not None:
        yaw_noise = yaw_noise.to(device=direction_b.device, dtype=direction_b.dtype)
        c = torch.cos(yaw_noise)
        s = torch.sin(yaw_noise)
        x = direction_b[:, 0] * c - direction_b[:, 1] * s
        y = direction_b[:, 0] * s + direction_b[:, 1] * c
        direction_b = torch.stack([x, y], dim=-1)

    direction_b = direction_b / torch.linalg.norm(direction_b, dim=-1, keepdim=True).clamp(min=1e-6)

    phase = command.time_steps.to(dtype=direction_b.dtype) / (
        command.motion_length.to(dtype=direction_b.dtype).clamp(min=2.0) - 1.0
    )
    latch_phase = getattr(command, "kick_latch_start_phase", torch.ones(env.num_envs, device=direction_b.device))
    after_latch = phase >= latch_phase.to(device=direction_b.device, dtype=direction_b.dtype)
    latched_direction = getattr(command, "latched_kick_direction_base", None)
    direction_latched = getattr(command, "kick_direction_latched", None)
    if latched_direction is None or direction_latched is None:
        return direction_b

    direction_latched = direction_latched.to(device=direction_b.device, dtype=torch.bool)
    latch_now = after_latch & (~direction_latched)
    if torch.any(latch_now):
        command.latched_kick_direction_base[latch_now] = direction_b[latch_now]
        command.kick_direction_latched[latch_now] = True

    return torch.where(command.kick_direction_latched.unsqueeze(-1), command.latched_kick_direction_base, direction_b)


def near_field_latched_ball_observation(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return `[ball_x, ball_y, ball_z, valid, age_s]` for the near-field kicker.

    The actor receives a YOLO/depth-like low-rate observation.  Before the kick
    phase the ball is periodically refreshed with noise/dropout.  After the
    sampled kick latch phase, the observation can disappear while the position
    remains latched, matching the real occlusion during the swing.
    """
    command = _get_motion_command(env, command_name)
    target_base = get_target_point_base(env, command_name)
    device = target_base.device
    dtype = target_base.dtype
    step_buf = getattr(env, "episode_length_buf", None)
    first_step_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if step_buf is not None:
        first_step_mask = step_buf.to(device=device) == 0

    if hasattr(command, "perception_target_point_base_history"):
        step_counter = getattr(env, "common_step_counter", 0)
        if isinstance(step_counter, torch.Tensor):
            step_counter = int(step_counter.item())
        else:
            step_counter = int(step_counter)
        needs_history_init = first_step_mask | (~command.perception_history_valid.to(device=device))
        if torch.any(needs_history_init):
            command.perception_target_point_base_history[needs_history_init] = target_base[needs_history_init].unsqueeze(1)
            command.perception_history_write_index[needs_history_init] = 0
            command.perception_history_last_step[needs_history_init] = step_counter
            command.perception_history_valid[needs_history_init] = True

        update_history = command.perception_history_last_step.to(device=device) != step_counter
        if torch.any(update_history):
            history_len = command.perception_target_point_base_history.shape[1]
            next_index = (command.perception_history_write_index[update_history] + 1) % history_len
            update_ids = torch.nonzero(update_history, as_tuple=False).squeeze(-1)
            command.perception_target_point_base_history[update_ids, next_index] = target_base[update_ids]
            command.perception_history_write_index[update_ids] = next_index
            command.perception_history_last_step[update_ids] = step_counter

        history_len = command.perception_target_point_base_history.shape[1]
        latency_steps = command.perception_ball_latency_steps.to(device=device).clamp(0, history_len - 1)
        read_index = (command.perception_history_write_index.to(device=device) - latency_steps) % history_len
        env_ids = torch.arange(env.num_envs, device=device)
        target_base = command.perception_target_point_base_history[env_ids, read_index].to(dtype=dtype)

    update_period = max(1, int(getattr(command.cfg, "perception_ball_update_period_steps", 1)))
    update_due = (command.time_steps % update_period) == 0
    update_due = update_due | first_step_mask

    min_dist, max_dist = getattr(command.cfg, "near_field_ball_visible_distance_range", (0.0, 1000.0))
    distance_xy = torch.linalg.norm(target_base[:, :2], dim=-1)
    visible = (distance_xy >= float(min_dist)) & (distance_xy <= float(max_dist))

    phase = command.time_steps.to(dtype=dtype) / (command.motion_length.to(dtype=dtype).clamp(min=2.0) - 1.0)
    latch_phase = getattr(command, "kick_latch_start_phase", torch.ones(env.num_envs, device=device, dtype=dtype))
    after_latch = phase >= latch_phase.to(device=device, dtype=dtype)

    pre_dropout = min(max(float(getattr(command.cfg, "blind_dropout_prob", 0.0)), 0.0), 1.0)
    post_dropout = getattr(command, "post_trigger_ball_dropout_prob", torch.zeros(env.num_envs, device=device, dtype=dtype))
    post_dropout = post_dropout.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    dropout_prob = torch.where(after_latch, post_dropout, torch.full_like(post_dropout, pre_dropout))
    visible = visible & (torch.rand(env.num_envs, device=device) >= dropout_prob)
    visible = torch.where(first_step_mask, torch.ones_like(visible), visible)

    fresh_update = update_due & visible & (~after_latch)
    lost_update = update_due & ~visible
    noise_std = torch.as_tensor(getattr(command.cfg, "perception_ball_noise_std", (0.0, 0.0, 0.0)), device=device, dtype=dtype)
    noisy_target = target_base + torch.randn_like(target_base) * noise_std.view(1, 3)

    if torch.any(first_step_mask):
        command.perception_latched_target_point_base[first_step_mask] = noisy_target[first_step_mask]
        command.perception_ball_age_s[first_step_mask] = 0.0
        command.perception_last_update_step[first_step_mask] = command.time_steps[first_step_mask]

    if torch.any(fresh_update):
        command.perception_latched_target_point_base[fresh_update] = noisy_target[fresh_update]
        command.perception_ball_age_s[fresh_update] = 0.0
        command.perception_last_update_step[fresh_update] = command.time_steps[fresh_update]

    dt = float(getattr(env, "step_dt", 0.02))
    command.perception_ball_age_s = torch.where(
        fresh_update,
        torch.zeros_like(command.perception_ball_age_s),
        (command.perception_ball_age_s + dt).clamp(max=5.0),
    )
    command.perception_ball_valid = torch.where(
        fresh_update,
        torch.ones_like(command.perception_ball_valid),
        torch.where(lost_update, torch.zeros_like(command.perception_ball_valid), command.perception_ball_valid),
    )

    return torch.cat(
        [
            command.perception_latched_target_point_base,
            command.perception_ball_valid.to(dtype=dtype).unsqueeze(-1),
            command.perception_ball_age_s.to(dtype=dtype).unsqueeze(-1),
        ],
        dim=-1,
    )


def kick_elapsed_phase(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return normalized kick elapsed time as a deployable phase scalar.

    Unlike ``generated_commands`` this does not expose any reference motion
    state.  Deployment can reproduce it with ``policy_step / max_steps``.
    """
    command = _get_motion_command(env, command_name)
    dtype = command.target_point_pos.dtype
    phase = command.time_steps.to(dtype=dtype) / (command.motion_length.to(dtype=dtype).clamp(min=2.0) - 1.0)
    return phase.clamp(0.0, 1.0).unsqueeze(-1)


def hold_target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return destination in robot base frame, holding the last visible input.

    This keeps the same observation dimension as ``target_destination_pos_local``.
    The destination is treated as visible at episode start, then follows the
    ball-visibility state maintained by ``blind_zone_target_point_pos``: when
    the current perception state is blind, return the cached destination input;
    otherwise refresh the cache with the current destination input.
    """
    command = _get_motion_command(env, command_name)
    target_local = target_destination_pos_local(env, command_name)
    cache_name = f"_{command_name}_target_destination_hold_cache"

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        first_step_mask = step_buf == 0
        if torch.any(first_step_mask):
            cache = getattr(env, cache_name)
            cache[first_step_mask] = target_local[first_step_mask]
            setattr(env, cache_name, cache)

    blind_mask = getattr(command, "is_in_blind_zone", None)
    if blind_mask is None:
        return target_local

    visible_mask = ~blind_mask.to(device=target_local.device, dtype=torch.bool)
    if torch.any(visible_mask):
        cache = getattr(env, cache_name)
        cache[visible_mask] = target_local[visible_mask]
        setattr(env, cache_name, cache)

    return getattr(env, cache_name)


def dropout_target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return destination in robot base frame with independent frame dropout.

    The destination/goal point itself is fixed in the environment frame, but
    its policy observation is expressed in the robot base frame, so it normally
    changes as the robot moves.  On dropout frames, keep the previous policy
    input instead of updating it.  The first frame is always visible so the
    cache starts from a valid observation.
    """
    command = _get_motion_command(env, command_name)
    target_local = target_destination_pos_local(env, command_name)
    cache_name = f"_{command_name}_target_destination_dropout_cache"

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    dropout_prob = float(getattr(command.cfg, "target_destination_dropout_prob", 0.0))
    dropout_prob = min(max(dropout_prob, 0.0), 1.0)
    visible_mask = torch.rand(env.num_envs, device=target_local.device) >= dropout_prob

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        first_step_mask = step_buf == 0
        visible_mask = torch.where(first_step_mask, torch.ones_like(visible_mask), visible_mask)

    if torch.any(visible_mask):
        cache = getattr(env, cache_name)
        cache[visible_mask] = target_local[visible_mask]
        setattr(env, cache_name, cache)

    return getattr(env, cache_name)


def target_destination_pos_local_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_destination_local_cache"
    target_local = target_destination_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target destination caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # print("cache:", getattr(env, cache_name))
    return getattr(env, cache_name)
    # Positional encoding path is intentionally disabled here.
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)
    


def foot_target_point_distance(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, command_name: str = "motion",) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    robot = env.scene[robot_cfg.name]
    foot_pos = robot.data.body_pos_w[:, robot_cfg.body_ids]
    target_world = get_target_point_world(env, command_name)
    diff = foot_pos - target_world.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    return dist.view(env.num_envs, -1)
