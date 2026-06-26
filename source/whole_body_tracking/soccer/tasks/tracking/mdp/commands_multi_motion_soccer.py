from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .kick_detection import KickContactTracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiMotionLoader:
    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu"):
        assert len(motion_files) > 0, "motion_files must not be empty"
        self.num_files = len(motion_files)
        self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        self.device = device

        # Temporarily store data from each file.
        self.motion_name = []
        self.motion_lengths = []

        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        kick_leg_labels = []

        self.fps_list = []

        max_T = 0  # Track maximum frame count.

        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            self.fps_list.append(data["fps"])
            self.motion_name.append(motion_file.split("/")[-1].split(".")[0])  # Store filename without suffix.
            self.motion_lengths.append(data["joint_pos"].shape[0])

            jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)

            label_value: str | None = None
            if "kick_leg" in data.files:
                raw_label = data["kick_leg"]
                try:
                    label_str = str(raw_label.item()).strip().lower()
                except Exception:
                    label_str = str(raw_label).strip().lower()
                if label_str in {"left", "right"}:
                    label_value = label_str
            kick_leg_labels.append(label_value)

            max_T = max(max_T, jp.shape[0])

        # Pad all files to max_T and stack into tensors.
        def pad_tensor_list(tensor_list, pad_value=0.0):
            padded = []
            for t in tensor_list:
                T, *rest = t.shape
                pad_size = [max_T - T] + rest
                pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device)], dim=0)
                # pad_tensor = torch.cat([t, torch.full([*pad_size], pad_value, device=self.device, dtype=t.dtype)], dim=0)
                padded.append(pad_tensor)
            return torch.stack(padded, dim=0)  # shape: (num_files, max_T, ...)

        self.joint_pos = pad_tensor_list(joint_pos_list)
        self.joint_vel = pad_tensor_list(joint_vel_list)
        self._body_pos_w = pad_tensor_list(body_pos_w_list)
        self._body_quat_w = pad_tensor_list(body_quat_w_list)
        self._body_lin_vel_w = pad_tensor_list(body_lin_vel_w_list)
        self._body_ang_vel_w = pad_tensor_list(body_ang_vel_w_list)
        if self._body_indexes.numel() > 0:
            max_body_index = int(torch.max(self._body_indexes).item())
            if max_body_index >= self._body_pos_w.shape[2]:
                raise ValueError(
                    f"motion body index {max_body_index} is out of range for motion body dimension "
                    f"{self._body_pos_w.shape[2]}"
                )

        self.time_step_total = max_T  # Maximum frame count.
        self.file_lengths = torch.tensor([jp.shape[0] for jp in joint_pos_list],
                                         dtype=torch.long,
                                         device=self.device)
        self.fps = self.fps_list[0]  # Can be adjusted if needed.
        self._kick_leg_labels = tuple(kick_leg_labels)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]

    @property
    def kick_leg_labels(self) -> tuple[str | None, ...]:
        return self._kick_leg_labels
    
    def get_last_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int, motion_length: int) -> torch.Tensor:
        """Get the anchor position at the last frame of the specified motion."""
        last_frame_idx = motion_length - 1
        return self.body_pos_w[motion_idx, last_frame_idx, anchor_body_idx]

    def get_first_frame_anchor_pos(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor position at the first frame of the specified motion."""
        return self.body_pos_w[motion_idx, 0, anchor_body_idx]

    def get_first_frame_anchor_quat(self, motion_idx: int, anchor_body_idx: int) -> torch.Tensor:
        """Get the anchor orientation at the first frame of the specified motion."""
        return self.body_quat_w[motion_idx, 0, anchor_body_idx]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.soccer_ball: RigidObject | None = None
        # Try to get the soccer-ball object.
        if hasattr(env.scene, "__getitem__"):
            try:
                self.soccer_ball = env.scene["soccer_ball"]
            except KeyError:
                self.soccer_ball = None

        # Determine whether the motion sequence has ended.
        term_name = getattr(cfg, "term_name", None)
        if term_name is None:
            term_name = getattr(cfg, "name", None)
        if term_name is None:
            term_name = "motion"
            self._state_prefix = f"_{term_name}"
            self.kick_contact_tracker = KickContactTracker(env, self._state_prefix)

        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        robot_joint_names = list(getattr(self.robot, "joint_names", []))
        controlled_joint_names = self.cfg.controlled_joint_names
        if controlled_joint_names is None:
            controlled_joint_names = robot_joint_names
        self.controlled_joint_names = tuple(controlled_joint_names)
        self.controlled_joint_ids = torch.tensor(
            self.robot.find_joints(list(self.controlled_joint_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        if self.controlled_joint_ids.numel() != len(self.controlled_joint_names):
            resolved = [robot_joint_names[i] for i in self.controlled_joint_ids.detach().cpu().tolist()]
            missing = sorted(set(self.controlled_joint_names) - set(resolved))
            raise RuntimeError(f"could not resolve controlled robot joints: {missing}")

        sensor_hold_pos = dict(getattr(self.cfg, "sensor_joint_hold_pos", {}) or {})
        self.sensor_joint_names = tuple(name for name in sensor_hold_pos if name in robot_joint_names)
        self.sensor_joint_ids = torch.tensor(
            self.robot.find_joints(list(self.sensor_joint_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        ) if self.sensor_joint_names else torch.empty(0, dtype=torch.long, device=self.device)
        self.sensor_joint_hold_pos = (
            torch.tensor([sensor_hold_pos[name] for name in self.sensor_joint_names], dtype=torch.float32, device=self.device)
            if self.sensor_joint_names
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )

        motion_body_indexes = self.cfg.motion_body_indexes
        if motion_body_indexes is None:
            motion_body_indexes = self.body_indexes
        self.motion = MultiMotionLoader(self.cfg.motion_files, motion_body_indexes, device=self.device)
        motion_joint_dim = int(self.motion.joint_pos.shape[-1])
        if motion_joint_dim != int(self.controlled_joint_ids.numel()):
            raise RuntimeError(
                f"motion joint dimension {motion_joint_dim} does not match controlled joints "
                f"{int(self.controlled_joint_ids.numel())}; set MotionCommandCfg.controlled_joint_names explicitly"
            )
        kick_leg_to_id = {"left": 0, "right": 1}
        self._kick_leg_id_to_name = {v: k for k, v in kick_leg_to_id.items()}
        self._kick_leg_id_to_name[-1] = "unknown"
        self.motion_kick_leg = torch.full((self.motion.num_files,), -1, dtype=torch.int8, device=self.device)
        self.motion_kick_leg_names = []
        for idx, label in enumerate(self.motion.kick_leg_labels):
            normalized = label.lower() if isinstance(label, str) else None
            if normalized in kick_leg_to_id:
                self.motion_kick_leg[idx] = kick_leg_to_id[normalized]
                self.motion_kick_leg_names.append(normalized)
            else:
                self.motion_kick_leg_names.append("unknown")
        self.motion_indices_by_kick_leg = {
            leg_id: torch.nonzero(self.motion_kick_leg == leg_id, as_tuple=False).squeeze(-1)
            for leg_id in kick_leg_to_id.values()
        }

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Randomly assign initial motions.
        if self.motion.num_files > 1:
            self.motion_idx = torch.randint(0, self.motion.num_files, (self.num_envs,), 
                                           dtype=torch.long, device=self.device)
        # Initialize per-environment motion lengths.
        self.motion_length[:] = self.motion.file_lengths[self.motion_idx]

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling settings.
        # Compute bin count: decimation * dt is one simulation step duration.
        # Thus each bin corresponds to ~1 second and bin_count is the total number of bins.
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(
            (self.motion.num_files, self.bin_count), dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_gate_miss_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_lateral_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gate_cross_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_gate_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_robot_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_robot_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_ball_base_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_ball_base_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_init_yaw_error_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_ball_bucket_enabled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_ball_bucket_index"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["expected_left_foot_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["expected_right_foot_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sim2real_perception_latency_steps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sim2real_perception_latency_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sim2real_actuator_delay_steps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sim2real_actuator_delay_max_steps"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_foot_slip"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_feet_slip_penalty"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_swing_foot_clearance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_has_support"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_swing_foot_clearance_reward"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_step_length"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_contact_step_length_reward"] = torch.zeros(self.num_envs, device=self.device)

        # Target-point and soccer-ball generation logic.
        self.target_point_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.soccer_ball_pos = torch.zeros_like(self.target_point_pos)
        self.target_destination_pos = torch.zeros_like(self.target_point_pos)
        # Save initial target position at resample for kick-direction computation.
        self.initial_target_point_pos = torch.zeros_like(self.target_point_pos)
        self.goal_gate_prev_ball_pos = torch.zeros_like(self.target_point_pos)
        self.goal_gate_success_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_gate_miss_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_gate_lateral_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_gate_cross_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_gate_last_event_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.goal_gate_center_score = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_gate_edge_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reference_initial_anchor_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.initial_heading_yaw_delta = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_aware_root_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.goal_aware_root_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_aware_yaw_error_abs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_aware_ball_base_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.goal_aware_init_stage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.goal_aware_root_override_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Blind-zone logic: ball is invisible when robot-ball (x, y) distance is out of range.
        self.blind_distance_min = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.blind_distance_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # Target position at last visible frame (robot base frame).
        self.last_visible_target_point_base = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        # Whether currently in blind zone.
        self.is_in_blind_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.perception_latched_target_point_base = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.perception_ball_age_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.perception_ball_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.perception_last_update_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        step_dt = float(getattr(env, "step_dt", env.cfg.decimation * env.cfg.sim.dt))
        latency_range = getattr(cfg, "perception_ball_latency_range_s", (0.0, 0.0))
        self.perception_max_latency_steps = max(0, int(math.ceil(max(latency_range) / max(step_dt, 1e-6))))
        self.perception_ball_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.perception_history_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.perception_history_last_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.perception_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.perception_target_point_base_history = torch.zeros(
            self.num_envs,
            self.perception_max_latency_steps + 1,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        self.kick_latch_start_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.post_trigger_ball_dropout_prob = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.kick_direction_yaw_noise = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.kick_direction_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.latched_kick_direction_base = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        
        # Height for target_destination.
        self.destination_height = 0.11
        
        # target_destination generation parameters (world-frame based).
        self.destination_center = torch.tensor(self.cfg.target_destination_center, dtype=torch.float32, device=self.device)
        self.destination_length = float(self.cfg.target_destination_length)  # Rectangle length (x-axis).
        self.destination_width = float(self.cfg.target_destination_width)  # Rectangle width (y-axis).
        
        self.curve_radius_offset = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._radius_offset_min = None
        self._radius_offset_max = None
        curve_cfg = cfg.curve_offset_range or {}
        radius_range = curve_cfg.get("radius")
        if isinstance(radius_range, Sequence) and not isinstance(radius_range, (str, bytes)) and len(radius_range) >= 2:
            self._radius_offset_min = float(radius_range[0])
            self._radius_offset_max = float(radius_range[1])
        elif radius_range is not None:
            value = float(radius_range)
            self._radius_offset_min = value
            self._radius_offset_max = value
        self._target_arc_angle = float(curve_cfg.get("arc_angle", math.pi / 18.0))
        self._target_height = float(curve_cfg.get("height", 0.11))
        marker_cfg = cfg.target_point_marker_cfg
        self.target_point_marker = VisualizationMarkers(marker_cfg) if marker_cfg is not None else None
        dest_marker_cfg = getattr(cfg, "target_destination_marker_cfg", None)
        self.target_destination_marker = VisualizationMarkers(dest_marker_cfg) if dest_marker_cfg is not None else None

        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_soccer_offset(all_env_ids)
        self._update_destination_points(all_env_ids)
        if bool(getattr(self.cfg, "enable_goal_aware_initialization", False)):
            self._sample_goal_aware_initial_layout(all_env_ids)
        else:
            self._compute_soccer_ball_positions(all_env_ids)
            self._align_initial_layout_to_destination(all_env_ids)
        self._update_soccer_ball(all_env_ids)
        self._update_target_points(all_env_ids)
        self._reset_perception_randomization(all_env_ids)
        self._reset_goal_gate_state(all_env_ids)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_idx, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_idx, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_idx, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_idx, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.controlled_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.controlled_joint_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_pelvis_pos_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_pos_w[:, pelvis_index]
    
    @property
    def robot_pelvis_quat_w(self) -> torch.Tensor:
        pelvis_index = self.robot.body_names.index("pelvis")
        return self.robot.data.body_quat_w[:, pelvis_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    @property
    def kick_leg(self) -> torch.Tensor:
        return self.motion_kick_leg[self.motion_idx]

    @property
    def kick_leg_name(self) -> list[str]:
        ids = self.motion_kick_leg[self.motion_idx].tolist()
        return [self._kick_leg_id_to_name.get(i, "unknown") for i in ids]

    def _to_env_id_tensor(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def _sample_soccer_offset(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        if self._radius_offset_min is None or self._radius_offset_max is None:
            self.curve_radius_offset[ids] = 0.0
            return
        if abs(self._radius_offset_max - self._radius_offset_min) < 1e-6:
            self.curve_radius_offset[ids] = self._radius_offset_min
            return

        rand = torch.rand(ids.numel(), device=self.device)
        span = self._radius_offset_max - self._radius_offset_min
        self.curve_radius_offset[ids] = self._radius_offset_min + rand * span

    def _goal_aware_curriculum_alpha(self) -> tuple[float, float]:
        steps = getattr(self.cfg, "goal_aware_curriculum_steps", (48000, 144000, 288000))
        stage_a_end, stage_b_end, stage_c_end = [int(x) for x in steps]
        step_counter = getattr(self._env, "common_step_counter", 0)
        if isinstance(step_counter, torch.Tensor):
            step = int(step_counter.item())
        else:
            step = int(step_counter)

        if step <= stage_a_end:
            return 0.0, 0.0
        if step <= stage_b_end:
            alpha = (step - stage_a_end) / max(float(stage_b_end - stage_a_end), 1.0)
            return alpha, 1.0 + alpha
        if step <= stage_c_end:
            alpha = (step - stage_b_end) / max(float(stage_c_end - stage_b_end), 1.0)
            return 1.0 + alpha, 2.0 + alpha
        return 2.0, 3.0

    @staticmethod
    def _lerp_pair(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, float]:
        return (
            float(a[0]) + alpha * (float(b[0]) - float(a[0])),
            float(a[1]) + alpha * (float(b[1]) - float(a[1])),
        )

    def _goal_aware_range(self, name: str, alpha: float) -> tuple[float, float]:
        ranges = getattr(self.cfg, name)
        if alpha <= 0.0:
            return float(ranges[0][0]), float(ranges[0][1])
        if alpha <= 1.0:
            return self._lerp_pair(ranges[0], ranges[1], alpha)
        if alpha <= 2.0:
            return self._lerp_pair(ranges[1], ranges[2], alpha - 1.0)
        return float(ranges[2][0]), float(ranges[2][1])

    def _sample_uniform_range(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        if abs(high - low) < 1e-8:
            return torch.full(shape, low, dtype=torch.float32, device=self.device)
        return low + torch.rand(shape, dtype=torch.float32, device=self.device) * (high - low)

    def _sample_motion_ball_bucket_base_xy(
        self,
        ids: torch.Tensor,
        default_x_range: tuple[float, float],
        default_y_range: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample base-frame ball xy from the bucket assigned to each selected motion."""
        ranges = getattr(self.cfg, "motion_ball_bucket_base_xy_ranges", None)
        enabled = bool(getattr(self.cfg, "enable_motion_ball_bucket_sampling", False)) and ranges is not None
        fallback = bool(getattr(self.cfg, "motion_ball_bucket_fallback_to_goal_aware", True))
        if not enabled:
            return (
                self._sample_uniform_range(default_x_range, (ids.numel(),)),
                self._sample_uniform_range(default_y_range, (ids.numel(),)),
                torch.full((ids.numel(),), -1, dtype=torch.long, device=self.device),
            )

        if len(ranges) != int(self.motion.num_files):
            if not fallback:
                raise RuntimeError(
                    "MotionCommandCfg.motion_ball_bucket_base_xy_ranges must have one entry per motion file "
                    f"({len(ranges)} ranges for {int(self.motion.num_files)} files)."
                )
            return (
                self._sample_uniform_range(default_x_range, (ids.numel(),)),
                self._sample_uniform_range(default_y_range, (ids.numel(),)),
                torch.full((ids.numel(),), -1, dtype=torch.long, device=self.device),
            )

        motion_indices = self.motion_idx[ids].to(dtype=torch.long)
        ball_x = torch.empty(ids.numel(), dtype=torch.float32, device=self.device)
        ball_y = torch.empty_like(ball_x)
        for motion_i in torch.unique(motion_indices).tolist():
            motion_i = int(motion_i)
            mask = motion_indices == motion_i
            bucket = ranges[motion_i]
            if isinstance(bucket, dict):
                x_range = bucket.get("x", default_x_range)
                y_range = bucket.get("y", default_y_range)
            else:
                x_range, y_range = bucket
            count = int(mask.sum().item())
            ball_x[mask] = self._sample_uniform_range((float(x_range[0]), float(x_range[1])), (count,))
            ball_y[mask] = self._sample_uniform_range((float(y_range[0]), float(y_range[1])), (count,))
        return ball_x, ball_y, motion_indices

    def _sample_goal_aware_initial_layout(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        alpha, stage = self._goal_aware_curriculum_alpha()
        x_range = self._goal_aware_range("goal_aware_robot_x_ranges", alpha)
        y_range = self._goal_aware_range("goal_aware_robot_y_ranges", alpha)
        yaw_error_range = self._goal_aware_range("goal_aware_yaw_error_ranges", alpha)
        ball_x_range = self._goal_aware_range("goal_aware_ball_x_front_ranges", alpha)
        if bool(getattr(self.cfg, "goal_aware_ball_lateral_by_kick_leg", False)):
            ball_y_abs_range = self._goal_aware_range("goal_aware_ball_y_lat_abs_ranges", alpha)
            ball_y_range = (-ball_y_abs_range[1], ball_y_abs_range[1])
        else:
            ball_y_range = self._goal_aware_range("goal_aware_ball_y_lat_ranges", alpha)

        robot_x = self._sample_uniform_range(x_range, (ids.numel(),))
        robot_y = self._sample_uniform_range(y_range, (ids.numel(),))

        # Mirror the attack-half x ranges automatically if the configured goal is on -x.
        goal_x = self.target_destination_pos[ids, 0]
        attack_sign = torch.where(goal_x >= 0.0, torch.ones_like(goal_x), -torch.ones_like(goal_x))
        robot_xy = torch.stack((robot_x * attack_sign, robot_y), dim=-1)
        goal_xy = self.target_destination_pos[ids, :2]

        yaw_to_goal = torch.atan2(goal_xy[:, 1] - robot_xy[:, 1], goal_xy[:, 0] - robot_xy[:, 0])
        yaw_error = self._sample_uniform_range(yaw_error_range, (ids.numel(),))
        robot_yaw = yaw_to_goal + yaw_error

        root_quat = self.motion.body_quat_w[self.motion_idx[ids], self.time_steps[ids], 0]
        root_forward = quat_apply(
            root_quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=root_quat.dtype, device=self.device).expand(ids.numel(), -1),
        )
        root_yaw = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        self.initial_heading_yaw_delta[ids] = robot_yaw - root_yaw

        bucket_ball_x, bucket_ball_y, bucket_index = self._sample_motion_ball_bucket_base_xy(
            ids, ball_x_range, ball_y_range
        )
        bucket_enabled = bucket_index >= 0
        ball_base_x = bucket_ball_x
        has_bucket_fallback = not bool(torch.all(bucket_enabled).item())
        if bool(getattr(self.cfg, "goal_aware_ball_lateral_by_kick_leg", False)) and has_bucket_fallback:
            ball_y_abs = self._sample_uniform_range(ball_y_abs_range, (ids.numel(),))
            kick_leg = self.motion_kick_leg[self.motion_idx[ids]]
            y_sign = torch.where(kick_leg == 0, torch.ones_like(ball_y_abs), -torch.ones_like(ball_y_abs))
            known_leg = (kick_leg == 0) | (kick_leg == 1)
            random_y = self._sample_uniform_range(ball_y_range, (ids.numel(),))
            fallback_y = torch.where(known_leg, y_sign * ball_y_abs, random_y)
            ball_base_y = torch.where(bucket_enabled, bucket_ball_y, fallback_y)
        else:
            ball_base_y = bucket_ball_y
        c = torch.cos(robot_yaw)
        s = torch.sin(robot_yaw)
        ball_xy = robot_xy + torch.stack(
            (
                ball_base_x * c - ball_base_y * s,
                ball_base_x * s + ball_base_y * c,
            ),
            dim=-1,
        )

        self.goal_aware_root_pos_xy[ids] = robot_xy
        self.goal_aware_root_yaw[ids] = robot_yaw
        self.goal_aware_yaw_error_abs[ids] = torch.abs(yaw_error)
        self.goal_aware_ball_base_xy[ids] = torch.stack((ball_base_x, ball_base_y), dim=-1)
        self.goal_aware_init_stage[ids] = stage
        self.goal_aware_root_override_valid[ids] = True

        self.soccer_ball_pos[ids, :2] = ball_xy
        self.soccer_ball_pos[ids, 2] = float(self._target_height)
        self.initial_target_point_pos[ids] = self.soccer_ball_pos[ids]
        self.target_point_pos[ids] = self.soccer_ball_pos[ids]

        self.metrics["goal_init_stage"][ids] = stage
        self.metrics["goal_init_robot_x"][ids] = robot_xy[:, 0]
        self.metrics["goal_init_robot_y"][ids] = robot_xy[:, 1]
        self.metrics["goal_init_ball_base_x"][ids] = ball_base_x
        self.metrics["goal_init_ball_base_y"][ids] = ball_base_y
        self.metrics["goal_init_yaw_error_abs"][ids] = torch.abs(yaw_error)
        self.metrics["motion_ball_bucket_enabled"][ids] = bucket_enabled.to(torch.float32)
        self.metrics["motion_ball_bucket_index"][ids] = bucket_index.to(torch.float32)
        expected_leg = self.motion_kick_leg[self.motion_idx[ids]]
        self.metrics["expected_left_foot_rate"][ids] = (expected_leg == 0).to(torch.float32)
        self.metrics["expected_right_foot_rate"][ids] = (expected_leg == 1).to(torch.float32)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if isinstance(episode_failed, torch.Tensor):
            episode_failed = episode_failed.to(device=self.device, dtype=torch.bool)
        else:
            episode_failed = torch.tensor(episode_failed, dtype=torch.bool, device=self.device)
        # Clear failure histogram for the current update.
        self._current_bin_failed.zero_()
        # import ipdb; ipdb.set_trace()
        if torch.any(episode_failed):
            # import ipdb; ipdb.set_trace()
            # For failed environments, count the corresponding motion bins.
            failed_env_mask = episode_failed
            failed_motion_idx = self.motion_idx[env_ids][failed_env_mask]                       # [K]
            failed_lengths = self.motion_length[env_ids][failed_env_mask].clamp(min=1).float() # [K]
            failed_steps = self.time_steps[env_ids][failed_env_mask].float()                    # [K]
            # Map time_steps to normalized phase [0, 1], then to bins.
            failed_phase = failed_steps / (failed_lengths - 1.0 + 1e-6)
            failed_bins = torch.clamp((failed_phase * self.bin_count).long(), 0, self.bin_count - 1)  # [K]
            # Accumulate into a 2D histogram via flattened indices.
            flat_idx = failed_motion_idx * self.bin_count + failed_bins                          # [K]
            flat_size = int(self.motion.num_files * self.bin_count)

            # Accumulate safely on GPU to avoid CPU fallback and sync overhead.
            flat_counts = torch.zeros(flat_size, dtype=self._current_bin_failed.dtype, device=self.device)
            if flat_idx.numel() > 0:
                # Ensure indices are on the same device and in long dtype.
                flat_idx = flat_idx.to(self.device).long()
                ones = torch.ones_like(flat_idx, dtype=flat_counts.dtype, device=self.device)
                flat_counts.index_add_(0, flat_idx, ones)

            flat_counts = flat_counts.float()
            # In-place write to keep dtype/device stable.
            self._current_bin_failed[:] = flat_counts.view(self.motion.num_files, self.bin_count)

        # Probability: EMA failure counts plus a uniform prior.
        # Add self.cfg.adaptive_uniform_ratio / (M * B) per element to keep total mass consistent.
        M = max(1, int(self.motion.num_files))
        B = max(1, int(self.bin_count))
        uniform_per_pair = self.cfg.adaptive_uniform_ratio / float(M * B)
        probs = self.bin_failed_count + self._current_bin_failed + uniform_per_pair  # [M, B]
        # Non-causal padding + convolution to smooth along bins per motion.
        probs = torch.nn.functional.pad(
            probs.unsqueeze(1),  # [M, 1, B]
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).squeeze(1)         # [M, B]

        # Flatten and sample from joint (motion, bin) distribution.
        probs = probs.view(-1)                                                                    # [M*B]
        probs = probs / (probs.sum() + 1e-12)

        sampled_flat = torch.multinomial(probs, len(env_ids), replacement=True)                   # [E]
        sampled_motion = sampled_flat // self.bin_count                                           # [E]
        sampled_bins = sampled_flat % self.bin_count                                              # [E]

        # Map sampled bins to per-motion time_steps with small random offsets.
        self.motion_idx[env_ids] = sampled_motion
        self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
        rand_offset = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device).float()       # [E]
        sampled_phase = (sampled_bins.float() + rand_offset) / float(self.bin_count)              # [E]
        self.time_steps[env_ids] = (sampled_phase * (self.motion_length[env_ids].float() - 1)).long()

        # Metrics for the joint distribution.
        H = -(probs * (probs + 1e-12).log()).sum()
        denom = math.log(self.bin_count * max(1, int(self.motion.num_files)))
        H_norm = H / denom if denom > 1e-12 else torch.tensor(0.0, device=probs.device)
        pmax, imax = probs.max(dim=0)
        top1_motion = (imax // self.bin_count).float()
        top1_bin = (imax % self.bin_count).float() / self.bin_count
        # import ipdb; ipdb.set_trace()

        # Create metric entries only when needed.
        if "sampling_entropy" not in self.metrics or self.metrics["sampling_entropy"].shape[0] != self.num_envs:
            self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_prob" not in self.metrics or self.metrics["sampling_top1_prob"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_bin" not in self.metrics or self.metrics["sampling_top1_bin"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        if "sampling_top1_motion" not in self.metrics or self.metrics["sampling_top1_motion"].shape[0] != self.num_envs:
            self.metrics["sampling_top1_motion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = top1_bin
        self.metrics["sampling_top1_motion"][:] = top1_motion

    def _uniform_sampling(self, env_ids: Sequence[int]):
        # Sample motion and time-step separately to avoid out-of-range issues.
        # First, sample motions.
        if bool(getattr(self.cfg, "balance_motion_kick_leg_sampling", False)):
            requested_legs = torch.randint(0, 2, (len(env_ids),), device=self.device)
            motion_indices = torch.empty(len(env_ids), dtype=torch.long, device=self.device)
            fallback = torch.randint(0, self.motion.num_files, (len(env_ids),), device=self.device)
            motion_indices[:] = fallback
            for leg_id, candidates in self.motion_indices_by_kick_leg.items():
                leg_mask = requested_legs == int(leg_id)
                if torch.any(leg_mask) and candidates.numel() > 0:
                    choice = torch.randint(0, candidates.numel(), (int(leg_mask.sum().item()),), device=self.device)
                    motion_indices[leg_mask] = candidates[choice]
        else:
            motion_indices = torch.randint(0, self.motion.num_files, (len(env_ids),), device=self.device)
        self.motion_idx[env_ids] = motion_indices
        self.motion_length[env_ids] = self.motion.file_lengths[motion_indices]
        
        # Then sample a time-step for each selected motion.
        # time_phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        # Start each selected motion from frame 0.
        time_phase = torch.zeros(len(env_ids), device=self.device)

        self.time_steps[env_ids] = (time_phase * (self.motion_length[env_ids].float() - 1)).long()
        

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        if ids.numel() == 0:
            return

        arc_limit = float(self._target_arc_angle)
        base_height = float(self._target_height)

        for env_id in ids:
            motion_idx = int(self.motion_idx[env_id].item())
            motion_len = max(1, int(self.motion_length[env_id].item()))

            first_anchor = self.motion.get_first_frame_anchor_pos(motion_idx, self.motion_anchor_body_index,)
            last_anchor = self.motion.get_last_frame_anchor_pos(motion_idx, self.motion_anchor_body_index, motion_len,)

            radius_vec = last_anchor[:2] - first_anchor[:2]
            radius_sq = torch.dot(radius_vec, radius_vec)
            target_xy = last_anchor[:2]

            radius = torch.sqrt(radius_sq) if float(radius_sq) > 1e-12 else torch.tensor(0.0, device=self.device)
            if float(radius_sq) > 1e-12:
                base_direction = radius_vec / radius
            else:
                base_direction = torch.tensor([1.0, 0.0], device=self.device)

            if arc_limit > 0.0 and float(radius_sq) > 1e-12:
                base_angle = torch.atan2(radius_vec[1], radius_vec[0])
                angle_offset = sample_uniform(-arc_limit, arc_limit, (1,), device=self.device).squeeze(0)
                new_angle = base_angle + angle_offset
                direction = torch.stack((torch.cos(new_angle), torch.sin(new_angle)))
            else:
                direction = base_direction

            radius = torch.clamp(radius + self.curve_radius_offset[env_id], min=0.0)
            target_xy = first_anchor[:2] + radius * direction

            ball_pos = self.soccer_ball_pos.new_empty(3)
            ball_pos[:2] = target_xy
            ball_pos[2] = base_height
            self.soccer_ball_pos[env_id] = ball_pos

    def _update_target_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        # Also save initial target point for kick-direction computation.
        self.initial_target_point_pos[ids] = self.soccer_ball_pos[ids].clone()

        if self.target_point_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_positions = self.target_point_pos + env_origins
            else:
                world_positions = self.target_point_pos
            self.target_point_marker.visualize(world_positions)

    def _update_target_points_from_sim(self):
        """Read soccer-ball position from simulation each step and update target_point_pos."""
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        self.goal_gate_prev_ball_pos[:] = self.target_point_pos
        
        # Read world-space soccer-ball position from simulation.
        ball_world_pos = self.soccer_ball.data.root_pos_w  # [num_envs, 3]
        # Convert to local position relative to env origin.
        self.soccer_ball_pos = ball_world_pos - env_origins
        self.target_point_pos = self.soccer_ball_pos.clone()
        
        # Update visualization marker.
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_world_pos)



    def _update_destination_points(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        
        # Generate target_destination in world coordinates.
        # Sample destination uniformly within the rectangle.
        rand_x = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_length
        rand_y = (torch.rand(ids.numel(), device=self.device) - 0.5) * self.destination_width
        destination = self.destination_center.expand(ids.numel(), -1) + torch.stack([rand_x, rand_y, torch.zeros_like(rand_x)], dim=1)
        self.target_destination_pos[ids] = destination

        if self.target_destination_marker is not None:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is not None:
                world_destination = self.target_destination_pos + env_origins
            else:
                world_destination = self.target_destination_pos
            self.target_destination_marker.visualize(world_destination)

    def _align_initial_layout_to_destination(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        if not bool(getattr(self.cfg, "align_initial_heading_to_destination", False)):
            self.initial_heading_yaw_delta[ids] = 0.0
            return

        root_pos_local = self.motion.body_pos_w[self.motion_idx[ids], self.time_steps[ids], 0]
        root_quat = self.motion.body_quat_w[self.motion_idx[ids], self.time_steps[ids], 0]
        forward = quat_apply(
            root_quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=root_pos_local.dtype, device=self.device).expand(ids.numel(), -1),
        )
        current_yaw = torch.atan2(forward[:, 1], forward[:, 0])

        destination_delta = self.target_destination_pos[ids, :2] - root_pos_local[:, :2]
        destination_yaw = torch.atan2(destination_delta[:, 1], destination_delta[:, 0])
        yaw_delta = destination_yaw - current_yaw + float(getattr(self.cfg, "align_initial_heading_yaw_offset", 0.0))
        self.initial_heading_yaw_delta[ids] = yaw_delta

        c = torch.cos(yaw_delta)
        s = torch.sin(yaw_delta)
        rel_ball = self.soccer_ball_pos[ids, :2] - root_pos_local[:, :2]
        rotated_ball_xy = torch.stack(
            (
                rel_ball[:, 0] * c - rel_ball[:, 1] * s,
                rel_ball[:, 0] * s + rel_ball[:, 1] * c,
            ),
            dim=-1,
        )
        self.soccer_ball_pos[ids, :2] = root_pos_local[:, :2] + rotated_ball_xy

    def _reset_perception_randomization(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        latch_low, latch_high = self.cfg.kick_latch_start_phase_range
        dropout_low, dropout_high = self.cfg.post_trigger_ball_dropout_prob_range
        yaw_low, yaw_high = self.cfg.kick_direction_yaw_noise_range

        self.kick_latch_start_phase[ids] = latch_low + torch.rand(ids.numel(), device=self.device) * (latch_high - latch_low)
        self.post_trigger_ball_dropout_prob[ids] = dropout_low + torch.rand(ids.numel(), device=self.device) * (dropout_high - dropout_low)
        self.kick_direction_yaw_noise[ids] = yaw_low + torch.rand(ids.numel(), device=self.device) * (yaw_high - yaw_low)
        self.perception_latched_target_point_base[ids] = 0.0
        self.perception_ball_age_s[ids] = 0.0
        self.perception_ball_valid[ids] = False
        self.perception_last_update_step[ids] = -1
        latency_low, latency_high = getattr(self.cfg, "perception_ball_latency_range_s", (0.0, 0.0))
        step_dt = float(getattr(self._env, "step_dt", self._env.cfg.decimation * self._env.cfg.sim.dt))
        min_steps = max(0, int(round(float(latency_low) / max(step_dt, 1e-6))))
        max_steps = max(min_steps, int(round(float(latency_high) / max(step_dt, 1e-6))))
        max_steps = min(max_steps, self.perception_max_latency_steps)
        if max_steps > min_steps:
            self.perception_ball_latency_steps[ids] = torch.randint(
                min_steps,
                max_steps + 1,
                (ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.perception_ball_latency_steps[ids] = min_steps
        self.perception_history_write_index[ids] = 0
        self.perception_history_last_step[ids] = -1
        self.perception_history_valid[ids] = False
        self.perception_target_point_base_history[ids] = 0.0
        self.kick_direction_latched[ids] = False
        self.latched_kick_direction_base[ids] = 0.0
        step_dt = float(getattr(self._env, "step_dt", self._env.cfg.decimation * self._env.cfg.sim.dt))
        self.metrics["sim2real_perception_latency_steps"][ids] = self.perception_ball_latency_steps[ids].to(torch.float32)
        self.metrics["sim2real_perception_latency_s"][ids] = (
            self.perception_ball_latency_steps[ids].to(torch.float32) * step_dt
        )
        self._refresh_sim2real_actuator_delay_metrics(ids)

    def _refresh_sim2real_actuator_delay_metrics(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        delay_sum = torch.zeros(ids.numel(), dtype=torch.float32, device=self.device)
        delay_count = 0
        max_delay = 0.0
        for actuator in getattr(self.robot, "actuators", {}).values():
            cfg = getattr(actuator, "cfg", None)
            max_delay = max(max_delay, float(getattr(cfg, "max_delay", 0)))
            delay_buffer = getattr(actuator, "positions_delay_buffer", None)
            time_lags = getattr(delay_buffer, "time_lags", None)
            if isinstance(time_lags, torch.Tensor) and time_lags.shape[0] == self.num_envs:
                delay_sum += time_lags.to(device=self.device, dtype=torch.float32)[ids]
                delay_count += 1

        if delay_count > 0:
            self.metrics["sim2real_actuator_delay_steps"][ids] = delay_sum / float(delay_count)
        else:
            self.metrics["sim2real_actuator_delay_steps"][ids] = 0.0
        self.metrics["sim2real_actuator_delay_max_steps"][ids] = max_delay

    def _compose_full_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_joint_vel = torch.zeros_like(full_joint_pos)
        full_joint_pos[:, self.controlled_joint_ids] = joint_pos
        full_joint_vel[:, self.controlled_joint_ids] = joint_vel
        if self.sensor_joint_ids.numel() > 0:
            hold = self.sensor_joint_hold_pos.to(device=full_joint_pos.device, dtype=full_joint_pos.dtype)
            full_joint_pos[:, self.sensor_joint_ids] = hold.unsqueeze(0).expand(env_ids.numel(), -1)
            full_joint_vel[:, self.sensor_joint_ids] = 0.0
        return full_joint_pos, full_joint_vel

    def _reset_goal_gate_state(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.goal_gate_prev_ball_pos[ids] = self.target_point_pos[ids]
        self.goal_gate_success_awarded[ids] = False
        self.goal_gate_miss_awarded[ids] = False
        self.goal_gate_lateral_error[ids] = 0.0
        self.goal_gate_cross_speed[ids] = 0.0
        self.goal_gate_last_event_step[ids] = -1
        self.goal_gate_center_score[ids] = 0.0
        self.goal_gate_edge_hit[ids] = False
        for metric_name in (
            "goal_success_rate",
            "goal_gate_miss_rate",
            "gate_lateral_error",
            "gate_cross_speed",
            "goal_gate_stage",
            "goal_center_score",
            "goal_lateral_error_signed",
            "goal_edge_hit_rate",
            "goal_cross_speed_reward",
        ):
            metric = self.metrics.get(metric_name)
            if metric is not None and metric.shape[0] == self.num_envs:
                metric[ids] = 0.0

    def _refresh_relative_body_cache(self):
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        

    def _update_soccer_ball(self, env_ids: Sequence[int] | torch.Tensor):
        if self.soccer_ball is None or not hasattr(self.soccer_ball, "write_root_state_to_sim"):
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        ball_pos = self.soccer_ball_pos[ids] + env_origins[ids]
        ball_quat = ball_pos.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        
        # Sample initial linear velocity based on config.
        if self.cfg.enable_soccer_ball_init_vel:
            lin_vel_range = self.cfg.soccer_ball_init_lin_vel_range or {}
            lin_vel_ranges = torch.tensor(
                [lin_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]],
                device=self.device
            )  # [3, 2]
            ball_lin_vel = sample_uniform(
                lin_vel_ranges[:, 0], lin_vel_ranges[:, 1], (ids.numel(), 3), device=self.device
            )
        else:
            ball_lin_vel = ball_pos.new_zeros((ids.numel(), 3))
        
        # Set angular velocity to zero.
        ball_ang_vel = ball_pos.new_zeros((ids.numel(), 3))

        ball_state = torch.cat([ball_pos, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = self._to_env_id_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        self._sample_soccer_offset(env_ids)
        sampling_strategy = str(self.cfg.sampling_strategy).lower()
        if sampling_strategy == "adaptive":
            self._adaptive_sampling(env_ids)
        elif sampling_strategy == "uniform":
            self._uniform_sampling(env_ids)
        else:
            raise ValueError(f"Unsupported sampling_strategy: {self.cfg.sampling_strategy}")
        self._update_destination_points(env_ids)
        if bool(getattr(self.cfg, "enable_goal_aware_initialization", False)):
            self._sample_goal_aware_initial_layout(env_ids)
        else:
            self._compute_soccer_ball_positions(env_ids)
            self._align_initial_layout_to_destination(env_ids)
        self._update_soccer_ball(env_ids)
        self._update_target_points(env_ids)
        self._reset_perception_randomization(env_ids)
        self._reset_goal_gate_state(env_ids)
        
        # Sample blind-zone min/max thresholds and reset blind-zone state.
        blind_min_low, blind_min_high = self.cfg.blind_distance_min_range
        blind_max_low, blind_max_high = self.cfg.blind_distance_max_range
        self.blind_distance_min[env_ids] = blind_min_low + torch.rand(env_ids.numel(), device=self.device) * (blind_min_high - blind_min_low)
        self.blind_distance_max[env_ids] = blind_max_low + torch.rand(env_ids.numel(), device=self.device) * (blind_max_high - blind_max_low)
        self.is_in_blind_zone[env_ids] = False
        self.last_visible_target_point_base[env_ids] = 0.0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        goal_aware_enabled = bool(getattr(self.cfg, "enable_goal_aware_initialization", False))
        if goal_aware_enabled:
            env_origins = getattr(self._env.scene, "env_origins", None)
            if env_origins is None:
                raise RuntimeError("goal-aware initialization requires env.scene.env_origins")

            root_pos[env_ids, :2] = self.goal_aware_root_pos_xy[env_ids] + env_origins[env_ids, :2]
            heading_delta = self.initial_heading_yaw_delta[env_ids]
        else:
            heading_delta = self.initial_heading_yaw_delta[env_ids]

        if torch.any(torch.abs(heading_delta) > 1e-6):
            heading_delta_quat = quat_from_euler_xyz(
                torch.zeros_like(heading_delta),
                torch.zeros_like(heading_delta),
                heading_delta,
            )
            root_ori[env_ids] = quat_mul(heading_delta_quat, root_ori[env_ids])
            root_lin_vel[env_ids] = quat_apply(heading_delta_quat, root_lin_vel[env_ids])
            root_ang_vel[env_ids] = quat_apply(heading_delta_quat, root_ang_vel[env_ids])

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        if goal_aware_enabled:
            # V2 curriculum owns x/y/yaw so sampled poses stay inside the attack-half and face-goal bounds.
            range_list[0] = (0.0, 0.0)
            range_list[1] = (0.0, 0.0)
            range_list[5] = (0.0, 0.0)
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.controlled_joint_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        full_joint_pos, full_joint_vel = self._compose_full_joint_state(joint_pos[env_ids], joint_vel[env_ids], env_ids)
        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        self._refresh_relative_body_cache()
        self.reference_initial_anchor_pos_w[env_ids] = self.robot_anchor_pos_w[env_ids].detach()

        # Set resample flag so env can refresh observations on next step.
        flag_name = f"{self._state_prefix}_motion_resampled"
        resample_flags = getattr(self._env, flag_name, None)
        if resample_flags is None or resample_flags.shape[0] != self.num_envs:
            resample_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            resample_flags = resample_flags.to(device=self.device, dtype=torch.bool)
        resample_flags[env_ids] = True
        setattr(self._env, flag_name, resample_flags)

    # Called every step in the IsaacLab main loop.
    def _update_command(self):
        self.kick_contact_tracker.begin_step(self)
        # Increment time_steps; if a sequence ends, resample based on failure statistics.
        self.time_steps += 1
        # env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        env_ids = torch.where(self.time_steps >= self.motion_length)[0]
        self._resample_command(env_ids)
        
        # Update target point each step using current ball position.
        self._update_target_points_from_sim()

        # Continuously refresh pre-kick target until contact occurs; then keep it frozen.
        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        self._refresh_relative_body_cache()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    #motion_file: str = MISSING
    motion_files: list[str] = MISSING

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    motion_body_indexes: list[int] | None = None
    controlled_joint_names: list[str] | None = None
    sensor_joint_hold_pos: dict[str, float] = {}

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    align_initial_heading_to_destination: bool = False
    align_initial_heading_yaw_offset: float = 0.0
    align_motion_reference_to_initial_heading: bool = False
    balance_motion_kick_leg_sampling: bool = False

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_strategy: str = "uniform"

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.1
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.4

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Target-point marker config; typically overridden in subclasses.
    target_point_marker_cfg: VisualizationMarkersCfg | None = None
    target_destination_marker_cfg: VisualizationMarkersCfg | None = None
    # Offset configuration for arc distribution and destination height.
    curve_offset_range: dict[str, float | tuple[float, float]] | None = None
    
    # Initial soccer-ball velocity configuration.
    enable_soccer_ball_init_vel: bool = False
    soccer_ball_init_lin_vel_range: dict[str, tuple[float, float]] | None = None
    
    # Blind-zone config: ball is invisible when robot-ball (x, y) distance is outside [min, max].
    blind_distance_min_range: tuple[float, float] = (0.3, 0.5)  # Minimum distance sampling range.
    blind_distance_max_range: tuple[float, float] = (1.5, 2.0)  # Maximum distance sampling range.
    # Extra per-step random dropout probability for ball perception while it is otherwise visible.
    blind_dropout_prob: float = 0.0
    # Per-step random dropout probability for destination/goal perception.
    target_destination_dropout_prob: float = 0.0

    # Near-field kick perception/command randomization.
    near_field_ball_visible_distance_range: tuple[float, float] = (0.15, 1.6)
    perception_ball_update_period_steps: int = 5
    perception_ball_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    perception_ball_latency_range_s: tuple[float, float] = (0.0, 0.0)
    kick_latch_start_phase_range: tuple[float, float] = (1.0, 1.0)
    post_trigger_ball_dropout_prob_range: tuple[float, float] = (0.0, 0.0)
    kick_direction_yaw_noise_range: tuple[float, float] = (0.0, 0.0)

    # Goal-gate curriculum for near-field goal kicking.
    goal_gate_curriculum_steps: tuple[int, int, int] = (24000, 72000, 144000)
    goal_gate_local_distance: float = 1.5
    goal_gate_mid_distance: float = 3.0
    goal_gate_local_half_width: float = 0.45
    goal_gate_mid_half_width: float = 0.65
    goal_gate_real_half_width: float = 0.9
    goal_gate_min_cross_speed: float = 0.0

    # Goal-aware initial state curriculum for near-field goal kicking.  These
    # ranges are field-frame x/y for the robot and base-frame x/y for the ball.
    enable_goal_aware_initialization: bool = False
    goal_aware_curriculum_steps: tuple[int, int, int] = (48000, 144000, 288000)
    goal_aware_robot_x_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (3.5, 6.2),
        (2.0, 6.2),
        (0.5, 6.2),
    )
    goal_aware_robot_y_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-2.2, 2.2),
        (-3.0, 3.0),
        (-3.8, 3.8),
    )
    goal_aware_yaw_error_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-math.radians(20.0), math.radians(20.0)),
        (-math.radians(45.0), math.radians(45.0)),
        (-math.radians(90.0), math.radians(90.0)),
    )
    goal_aware_ball_x_front_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.35, 0.70),
        (0.30, 0.85),
        (0.25, 0.95),
    )
    goal_aware_ball_y_lat_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.18, 0.18),
        (-0.25, 0.25),
        (-0.35, 0.35),
    )
    goal_aware_ball_lateral_by_kick_leg: bool = False
    goal_aware_ball_y_lat_abs_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.06, 0.18),
        (0.04, 0.25),
        (0.02, 0.32),
    )
    enable_motion_ball_bucket_sampling: bool = False
    motion_ball_bucket_base_xy_ranges: list | None = None
    motion_ball_bucket_fallback_to_goal_aware: bool = True

    # Destination/goal sampling in field coordinates.  Soccer_Lab/Firmware use
    # field center as origin, long axis as x, and goal centers at x=+/-length/2.
    target_destination_center: tuple[float, float, float] = (0.0, -5.0, 0.11)
    target_destination_length: float = 1.0
    target_destination_width: float = 0.5
