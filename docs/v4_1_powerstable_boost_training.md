# V4.1 Power-Stable Boost Training

This repository contains the training code for:

`Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0`

The task is a V4.1 side-foot stable fine-tune that increases kick power while
keeping the deploy student interface unchanged:

- policy observation: 101
- action dimension: 29
- motion set: `motions/soccer-standard`
- recommended start checkpoint: `checkpoints/v4_1_sidefoot_stable_model_25000.pt`

The 25k checkpoint is intentionally used instead of 30k+ because 30k is already
close to the converged slow-kick policy. At 25k the policy still has usable
side-foot/correct-foot signal and leaves more room for power shaping.

## Prerequisites

Set up the same Isaac Lab / Isaac Sim environment used by the base project, then
install the local extension:

```bash
cd /path/to/HumanoidSoccer
pip install -e source/whole_body_tracking
```

If your Isaac Lab environment is managed by conda, activate it before running
the commands below.

## Main Fine-Tune

```bash
cd /path/to/HumanoidSoccer

python scripts/rsl_rl/train_student.py \
  --task Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0 \
  --num_envs 4096 \
  --motion_path motions/soccer-standard \
  --load_checkpoint_path checkpoints/v4_1_sidefoot_stable_model_25000.pt \
  --run_name nearfield_goalkick_v4_1_powerstable_boost_ft25000_4096 \
  --max_iterations 45000 \
  --headless
```

Equivalent helper:

```bash
bash shell/train_v4_1_powerstable_boost_ft25000.sh
```

## Playback

After checkpoints are saved, play the latest checkpoint with a close follow
camera:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0 \
  --num_envs 1 \
  --load_run <timestamp>_nearfield_goalkick_v4_1_powerstable_boost_ft25000_4096 \
  --checkpoint model_<iter>.pt \
  --play_goal_init_stage 3 \
  --play_face_goal \
  --play_follow_robot_camera \
  --video \
  --video_length 500 \
  --headless
```

## Acceptance Metrics

Use TensorBoard or `scripts/audit_hmsc_versions.py --include-scalars`.

Minimum gates:

- `Train/mean_episode_length >= 480`
- `Episode_Termination/time_out >= 0.94`
- `Metrics/motion/kick_success_rate >= 0.90`
- `Metrics/motion/correct_foot_episode_rate >= 0.40`
- `Metrics/motion/inside_foot_contact_rate >= 0.10`
- `Metrics/motion/toe_contact_rate <= 0.02`
- `Metrics/motion/instep_contact_rate <= 0.01`
- `Metrics/motion/ball_velocity_to_goal_mean >= 0.057`

Watch these speed-specific metrics:

- `Metrics/motion/side_foot_leg_speed`
- `Metrics/motion/side_foot_leg_speed_reward`
- `Metrics/motion/style_gated_ball_speed_raw`
- `Metrics/motion/style_gated_ball_speed`
- `Metrics/motion/gate_cross_speed`
