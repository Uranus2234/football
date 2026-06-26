from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_multi_motion_soccer import MotionCommand


@dataclass
class KickContactEvent:
    """Container for kick contact detection results produced once per step."""

    new_contact: torch.Tensor
    kick_detected: torch.Tensor
    peak_force: torch.Tensor
    force_norm: Optional[torch.Tensor] = None


@dataclass
class ContactFootInfo:
    """Resolved foot metadata for environments with an active kick contact."""

    env_ids: torch.Tensor
    body_indices: torch.Tensor
    sides: torch.Tensor
    expected: torch.Tensor


class KickContactTracker:
    """Shared kick contact detection logic reusable across reward terms."""

    def __init__(self, env: ManagerBasedRLEnv, state_prefix: str):
        self._env = env
        self._state_prefix = state_prefix
        self._device = env.device
        self._num_envs = env.num_envs
        self._cache_valid = False
        self._cached_event: Optional[KickContactEvent] = None
        self._foot_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def begin_step(self, command: MotionCommand):
        """Reset per-step cache and handle envs that just resampled motions."""
        self._cache_valid = False
        self._cached_event = None
        self._handle_resample(command)

    def detect(
        self,
        command: MotionCommand,
        ball_sensor_name: str,
        horizontal_force_threshold: float,
    ) -> KickContactEvent:
        """Detect new kick contacts while ensuring single evaluation per step."""
        if self._cache_valid and self._cached_event is not None:
            return self._cached_event

        ball_sensor = self._get_contact_sensor(ball_sensor_name)
        if ball_sensor is None:
            empty_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            zero_force = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
            event = KickContactEvent(empty_mask, empty_mask, zero_force, zero_force)
            self._cached_event = event
            self._cache_valid = True
            return event

        forces = getattr(ball_sensor.data, "net_forces_w_history", ball_sensor.data.net_forces_w)
        if forces is None or forces.numel() == 0:
            empty_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            zero_force = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
            event = KickContactEvent(empty_mask, empty_mask, zero_force, zero_force)
            self._cached_event = event
            self._cache_valid = True
            return event

        forces = forces.to(device=self._device)
        if forces.ndim > 2:
            forces = forces.amax(dim=1)

        force_norm = torch.linalg.norm(forces, dim=-1)
        peak_force = force_norm.amax(dim=-1) if force_norm.ndim > 1 else force_norm
        kick_detected = peak_force > horizontal_force_threshold

        contact_awarded = self._get_or_init_bool_tensor("target_contact_awarded", default=False)
        new_contact = (~contact_awarded) & kick_detected
        if torch.any(new_contact):
            contact_awarded[new_contact] = True
        self._update_detection_state(new_contact)

        event = KickContactEvent(new_contact, kick_detected, peak_force, force_norm)
        self._cached_event = event
        self._cache_valid = True
        return event

    def record_expected_success(self, mask: torch.Tensor, expected_mask: torch.Tensor):
        """Store whether a detected kick matched the expected leg selection."""
        if expected_mask.dtype != torch.bool:
            expected_mask = expected_mask.to(dtype=torch.bool)
        expected_state = self._get_or_init_bool_tensor("expected_kick_success", default=False)
        expected_state[mask] = expected_mask[mask]

    def record_contact_foot(self, env_ids: torch.Tensor, hit_sides: torch.Tensor):
        """Store the resolved first-contact foot side for episode-level metrics."""
        if env_ids.numel() == 0:
            return
        hit_sides = hit_sides.to(device=self._device, dtype=torch.int8)
        left_state = self._get_or_init_bool_tensor("actual_left_foot_contact", default=False)
        right_state = self._get_or_init_bool_tensor("actual_right_foot_contact", default=False)
        known_state = self._get_or_init_bool_tensor("known_foot_contact", default=False)

        left_state[env_ids] = hit_sides == 0
        right_state[env_ids] = hit_sides == 1
        known_state[env_ids] = (hit_sides == 0) | (hit_sides == 1)

    def get_contact_awarded(self) -> torch.Tensor:
        """Return kick status: False means not kicked yet, True means kicked."""
        return self._get_or_init_bool_tensor("target_contact_awarded", default=False)

    def freeze_proximity_reward(self, env_ids: torch.Tensor, reward_values: torch.Tensor):
        """Freeze proximity reward values at the kick-contact moment."""
        frozen = self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)
        frozen[env_ids] = reward_values

    def get_frozen_proximity_reward(self) -> torch.Tensor:
        """Get frozen proximity reward values."""
        return self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)

    def _get_or_init_float_tensor(self, suffix: str, default: float) -> torch.Tensor:
        name = self._tensor_name(suffix)
        tensor = getattr(self._env, name, None)
        if tensor is None or tensor.shape[0] != self._num_envs:
            tensor = torch.full((self._num_envs,), default, dtype=torch.float32, device=self._device)
            setattr(self._env, name, tensor)
            return tensor
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        setattr(self._env, name, tensor)
        return tensor

    def resolve_contact_foot(
        self,
        command: MotionCommand,
        foot_cfg: SceneEntityCfg,
        mask: torch.Tensor,
    ) -> ContactFootInfo:
        """Determine which foot most likely produced the contact for each env."""
        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self._device)
            zeros_i8 = torch.zeros(0, dtype=torch.int8, device=self._device)
            return ContactFootInfo(empty, empty, zeros_i8, zeros_i8)

        body_indices, sides = self._get_foot_metadata(command, foot_cfg)
        robot = command.robot

        foot_pos = robot.data.body_pos_w[env_ids][:, body_indices]
        ball_pos = command.soccer_ball_pos[env_ids]
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is not None:
            ball_pos = ball_pos + env_origins[env_ids]

        diff = torch.norm(foot_pos - ball_pos.unsqueeze(1), dim=-1)
        closest_idx = torch.argmin(diff, dim=-1)
        selected_body_indices = body_indices[closest_idx]
        hit_sides = sides[closest_idx]

        expected = command.kick_leg[env_ids].to(torch.int8)
        expected = expected.clamp(min=0)

        return ContactFootInfo(env_ids, selected_body_indices, hit_sides, expected)

    def _handle_resample(self, command: MotionCommand):
        flag_name = self._tensor_name("motion_resampled")
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self._num_envs:
            return

        resample_flags = resample_flags.to(device=self._device, dtype=torch.bool)
        if not torch.any(resample_flags):
            return

        contact_state = self._get_or_init_bool_tensor("target_contact_awarded", default=False)
        kick_success_state = self._get_or_init_bool_tensor("kick_success", default=False)
        expected_state = self._get_or_init_bool_tensor("expected_kick_success", default=False)
        left_foot_state = self._get_or_init_bool_tensor("actual_left_foot_contact", default=False)
        right_foot_state = self._get_or_init_bool_tensor("actual_right_foot_contact", default=False)
        known_foot_state = self._get_or_init_bool_tensor("known_foot_contact", default=False)
        side_foot_state = self._get_or_init_bool_tensor("side_foot_contact_awarded", default=False)
        geometric_medial_state = self._get_or_init_bool_tensor("geometric_medial_contact_awarded", default=False)
        lateral_foot_state = self._get_or_init_bool_tensor("lateral_foot_contact_awarded", default=False)
        instep_state = self._get_or_init_bool_tensor("instep_contact_awarded", default=False)
        toe_state = self._get_or_init_bool_tensor("toe_contact_awarded", default=False)
        expected_side_state = self._get_or_init_bool_tensor("ball_side_expected_contact_awarded", default=False)
        wrong_side_state = self._get_or_init_bool_tensor("ball_side_wrong_foot_contact_awarded", default=False)
        support_step_valid_state = self._get_or_init_bool_tensor("support_step_initial_valid", default=False)
        support_step_completed_state = self._get_or_init_bool_tensor("support_step_completed", default=False)
        real_goal_success_state = self._get_or_init_bool_tensor("real_goal_success_awarded", default=False)
        real_goal_miss_state = self._get_or_init_bool_tensor("real_goal_miss_awarded", default=False)
        real_goal_edge_state = self._get_or_init_bool_tensor("real_goal_edge_hit", default=False)
        real_goal_lateral_state = self._get_or_init_float_tensor("real_goal_lateral_error", default=0.0)
        real_goal_speed_state = self._get_or_init_float_tensor("real_goal_cross_speed", default=0.0)
        real_goal_score_state = self._get_or_init_float_tensor("real_goal_center_score", default=0.0)
        side_foot_leg_speed_state = self._get_or_init_float_tensor("side_foot_leg_speed", default=0.0)
        side_foot_leg_speed_reward_state = self._get_or_init_float_tensor("side_foot_leg_speed_reward", default=0.0)

        # Skip resamples triggered by imminent episode termination to avoid skewed stats.
        eligible_mask = resample_flags.clone()
        step_buf = getattr(self._env, "episode_length_buf", None)
        if step_buf is not None:
            step_buf = step_buf.to(device=self._device, dtype=torch.long)
            cutoff_value = 1
            # max_episode_length = getattr(self._env, "max_episode_length", None)
            # if isinstance(max_episode_length, torch.Tensor):
            #     cutoff_value = int(max_episode_length.item()) - 1
            # elif isinstance(max_episode_length, (int, float)):
            #     cutoff_value = int(max_episode_length) - 1
            cutoff_value = max(cutoff_value, 0)
            eligible_mask = eligible_mask & (cutoff_value < step_buf)

        num_resampled = int(resample_flags.sum().item())
        num_eligible = int(eligible_mask.sum().item())
        if num_resampled > 0 and hasattr(command, "metrics"):
            if "kick_success_rate" not in command.metrics or command.metrics["kick_success_rate"].shape[0] != self._num_envs:
                command.metrics["kick_success_rate"] = torch.zeros(
                    self._num_envs, device=self._device, dtype=torch.float32
                )
            if "expected_kick_success_rate" not in command.metrics or command.metrics["expected_kick_success_rate"].shape[0] != self._num_envs:
                command.metrics["expected_kick_success_rate"] = torch.zeros(
                    self._num_envs, device=self._device, dtype=torch.float32
                )
            for metric_name in (
                "actual_left_foot_contact_rate",
                "actual_right_foot_contact_rate",
                "correct_foot_episode_rate",
                "wrong_foot_episode_rate",
            ):
                if metric_name not in command.metrics or command.metrics[metric_name].shape[0] != self._num_envs:
                    command.metrics[metric_name] = torch.zeros(
                        self._num_envs, device=self._device, dtype=torch.float32
                    )

            if num_eligible > 0:
                success_rate = kick_success_state[eligible_mask].float().mean()
                expected_rate = expected_state[eligible_mask].float().mean()
                left_rate = left_foot_state[eligible_mask].float().mean()
                right_rate = right_foot_state[eligible_mask].float().mean()
                wrong_rate = (known_foot_state[eligible_mask] & (~expected_state[eligible_mask])).float().mean()
                command.metrics["kick_success_rate"].fill_(success_rate.item())
                command.metrics["expected_kick_success_rate"].fill_(expected_rate.item())
                command.metrics["actual_left_foot_contact_rate"].fill_(left_rate.item())
                command.metrics["actual_right_foot_contact_rate"].fill_(right_rate.item())
                command.metrics["correct_foot_episode_rate"].fill_(expected_rate.item())
                command.metrics["wrong_foot_episode_rate"].fill_(wrong_rate.item())

        contact_state[resample_flags] = False
        kick_success_state[resample_flags] = False
        expected_state[resample_flags] = False
        left_foot_state[resample_flags] = False
        right_foot_state[resample_flags] = False
        known_foot_state[resample_flags] = False
        side_foot_state[resample_flags] = False
        geometric_medial_state[resample_flags] = False
        lateral_foot_state[resample_flags] = False
        instep_state[resample_flags] = False
        toe_state[resample_flags] = False
        expected_side_state[resample_flags] = False
        wrong_side_state[resample_flags] = False
        support_step_valid_state[resample_flags] = False
        support_step_completed_state[resample_flags] = False
        real_goal_success_state[resample_flags] = False
        real_goal_miss_state[resample_flags] = False
        real_goal_edge_state[resample_flags] = False
        real_goal_lateral_state[resample_flags] = 0.0
        real_goal_speed_state[resample_flags] = 0.0
        real_goal_score_state[resample_flags] = 0.0
        side_foot_leg_speed_state[resample_flags] = 0.0
        side_foot_leg_speed_reward_state[resample_flags] = 0.0
        post_kick_counter = getattr(self._env, self._tensor_name("post_kick_stand_still_counter"), None)
        if post_kick_counter is not None and post_kick_counter.shape[0] == self._num_envs:
            post_kick_counter = post_kick_counter.to(device=self._device, dtype=torch.int32)
            post_kick_counter[resample_flags] = -1
            setattr(self._env, self._tensor_name("post_kick_stand_still_counter"), post_kick_counter)
        post_kick_anchor = getattr(self._env, self._tensor_name("post_kick_contact_anchor_xy"), None)
        if post_kick_anchor is not None and post_kick_anchor.shape == (self._num_envs, 2):
            post_kick_anchor = post_kick_anchor.to(device=self._device, dtype=torch.float32)
            post_kick_anchor[resample_flags] = 0.0
            setattr(self._env, self._tensor_name("post_kick_contact_anchor_xy"), post_kick_anchor)
        arm_raise_counter = getattr(self._env, self._tensor_name("arm_raise_kick_counter"), None)
        if arm_raise_counter is not None and arm_raise_counter.shape[0] == self._num_envs:
            arm_raise_counter = arm_raise_counter.to(device=self._device, dtype=torch.int32)
            arm_raise_counter[resample_flags] = -1
            setattr(self._env, self._tensor_name("arm_raise_kick_counter"), arm_raise_counter)
        support_initial_pos = getattr(self._env, self._tensor_name("support_step_initial_pos"), None)
        if support_initial_pos is not None and support_initial_pos.shape == (self._num_envs, 3):
            support_initial_pos = support_initial_pos.to(device=self._device, dtype=torch.float32)
            support_initial_pos[resample_flags] = 0.0
            setattr(self._env, self._tensor_name("support_step_initial_pos"), support_initial_pos)
        
        # Reset frozen proximity reward.
        frozen_proximity = self._get_or_init_float_tensor("frozen_proximity_reward", default=0.0)
        frozen_proximity[resample_flags] = 0.0
        
        # Reset reward timers to avoid stale window logic after resampling.
        self._reset_reward_timers(resample_flags)
        
        resample_flags[resample_flags] = False
        setattr(self._env, flag_name, resample_flags)

    def _reset_reward_timers(self, resample_flags: torch.Tensor):
        """Reset reward timer states for environments that have been resampled."""
        timer_suffixes = [
            "dir_align_timer",
            "dir_align_prev",
            "speed_timer",
            "speed_prev",
            "autonomous_speed_timer",
            "style_gated_speed_timer",
            "z_speed_timer",
            "z_speed_prev",
        ]
        for suffix in timer_suffixes:
            timer_name = f"_{self._state_prefix}_{suffix}"
            timer = getattr(self._env, timer_name, None)
            if timer is not None and timer.shape[0] == self._num_envs:
                timer[resample_flags] = 0
                setattr(self._env, timer_name, timer)

    def _tensor_name(self, suffix: str) -> str:
        return f"{self._state_prefix}_{suffix}"

    def _get_contact_sensor(self, name: str) -> Optional[ContactSensor]:
        sensors = getattr(self._env.scene, "sensors", None)
        if sensors is None:
            return None
        if isinstance(sensors, dict):
            return sensors.get(name)
        try:
            return sensors[name]
        except (KeyError, TypeError):
            return None

    def _get_or_init_bool_tensor(self, suffix: str, default: bool) -> torch.Tensor:
        name = self._tensor_name(suffix)
        tensor = getattr(self._env, name, None)
        if tensor is None or tensor.shape[0] != self._num_envs:
            tensor = torch.full((self._num_envs,), default, dtype=torch.bool, device=self._device)
            setattr(self._env, name, tensor)
            return tensor

        tensor = tensor.to(device=self._device, dtype=torch.bool)
        setattr(self._env, name, tensor)
        return tensor

    def _update_detection_state(self, new_contact: torch.Tensor):
        if not torch.any(new_contact):
            return
        kick_success_state = self._get_or_init_bool_tensor("kick_success", default=False)
        kick_success_state[new_contact] = True

    def _get_foot_metadata(
        self,
        command: MotionCommand,
        foot_cfg: SceneEntityCfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._foot_cache is not None:
            return self._foot_cache

        robot = self._env.scene[foot_cfg.name]
        indices = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self._device,
        )
        sides = torch.tensor(
            [
                0 if "left" in name.lower() else 1 if "right" in name.lower() else -1
                for name in foot_cfg.body_names
            ],
            dtype=torch.int8,
            device=self._device,
        )
        self._foot_cache = (indices, sides)
        return self._foot_cache
