"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHOLE_BODY_TRACKING_SRC = REPO_ROOT / "source" / "whole_body_tracking"
if str(WHOLE_BODY_TRACKING_SRC) not in sys.path:
    sys.path.insert(0, str(WHOLE_BODY_TRACKING_SRC))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--robot_variant",
    type=str,
    choices=("main", "mode15"),
    default="main",
    help="G1 robot variant to use for play. Use mode15 for checkpoints trained with --robot_variant mode15.",
)
parser.add_argument(
    "--motion_path",
    type=str,
    default=None,
    help="Path to a directory containing motion .npz files. If omitted for V4, motions/soccer-standard is used.",
)
parser.add_argument("--export_student_policy", action="store_true", default=False, help="Export deploy-native policy without embedded motion reference tensors.")
parser.add_argument(
    "--play_robot_pose_range",
    type=float,
    nargs=3,
    metavar=("X_M", "Y_M", "YAW_DEG"),
    default=None,
    help="Play-only symmetric root randomization override: x/y in meters and yaw in degrees.",
)
parser.add_argument(
    "--play_face_goal",
    action="store_true",
    default=False,
    help="Play-only reset override that rotates the initial robot+ball layout so the robot faces the goal.",
)
parser.add_argument(
    "--play_face_goal_yaw_offset_deg",
    type=float,
    default=0.0,
    help="Extra yaw offset in degrees applied after --play_face_goal alignment.",
)
parser.add_argument(
    "--play_goal_init_stage",
    type=int,
    default=None,
    help="Play-only goal-aware init stage override. Use 3 for final-stage ranges and real goal gate.",
)
parser.add_argument(
    "--play_midfield_kick",
    action="store_true",
    default=False,
    help="Play-only reset override for midfield/mid-range shots instead of near-goal Stage-A starts.",
)
parser.add_argument(
    "--play_follow_robot_camera",
    action="store_true",
    default=False,
    help="Keep the viewer camera attached to the robot during play video capture.",
)
parser.add_argument(
    "--play_no_soccer_field",
    action="store_true",
    default=False,
    help="Disable Soccer_Lab field visuals during play.",
)
parser.add_argument(
    "--play_goalpost_usd",
    action="store_true",
    default=False,
    help="Use the original Soccer_Lab goalpost USD during play. By default play uses lightweight cuboid posts.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import glob
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.robots.g1 import apply_g1_robot_variant
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import install_soccer_lab_field
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx, export_student_policy_as_onnx


class RslRlVecEnvWrapper:
    """Compatibility wrapper for the TienKung/RSL-RL runner."""

    def __init__(self, env):
        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        self.env.reset()

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def _split_obs(self, obs_dict, extras=None):
        if extras is None:
            extras = {}
        extras["observations"] = {key: value for key, value in obs_dict.items() if key != "policy"}
        return obs_dict["policy"], extras

    def reset(self):
        obs_dict, extras = self.env.reset()
        return self._split_obs(obs_dict, extras)

    def get_observations(self):
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._split_obs(obs_dict, {})

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        obs, extras = self._split_obs(obs_dict, extras)
        return obs, rew, dones, extras

    def close(self):
        return self.env.close()


def _set_motion_file(env_cfg, motion_file: str | None) -> None:
    if motion_file is None:
        return
    if hasattr(env_cfg.commands.motion, "motion_files"):
        env_cfg.commands.motion.motion_files = [motion_file]
    if hasattr(env_cfg.commands.motion, "motion_file"):
        env_cfg.commands.motion.motion_file = motion_file


def _motion_files_from_path(motion_path: str) -> list[str]:
    path = pathlib.Path(motion_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        motion_files = sorted(glob.glob(str(path / "*.npz")))
        if motion_files:
            return motion_files
    raise ValueError(f"No motion .npz files found at: {motion_path}")


def _set_motion_files(env_cfg, motion_files: list[str]) -> None:
    if not motion_files:
        return
    if hasattr(env_cfg.commands.motion, "motion_files"):
        env_cfg.commands.motion.motion_files = motion_files
    if hasattr(env_cfg.commands.motion, "motion_file"):
        env_cfg.commands.motion.motion_file = motion_files[0]


def _set_motion_from_cli_or_v4_default(env_cfg, args_cli) -> None:
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        _set_motion_file(env_cfg, args_cli.motion_file)
        return
    if args_cli.motion_path is not None:
        motion_files = _motion_files_from_path(args_cli.motion_path)
        print(f"[INFO]: Using {len(motion_files)} motion files from CLI path: {args_cli.motion_path}")
        _set_motion_files(env_cfg, motion_files)
        return
    if args_cli.task is not None and "NearFieldGoalKickV4" in args_cli.task:
        default_motion_path = REPO_ROOT / "motions" / "soccer-standard"
        motion_files = _motion_files_from_path(str(default_motion_path))
        print(f"[INFO]: V4 play default motion path: {default_motion_path} ({len(motion_files)} files)")
        _set_motion_files(env_cfg, motion_files)


def _apply_play_robot_pose_range(env_cfg, pose_range: list[float] | None) -> None:
    if pose_range is None:
        return
    x_range, y_range, yaw_range_deg = [abs(float(value)) for value in pose_range]
    pose_cfg = dict(getattr(env_cfg.commands.motion, "pose_range", {}) or {})
    pose_cfg["x"] = (-x_range, x_range)
    pose_cfg["y"] = (-y_range, y_range)
    pose_cfg["yaw"] = (-math.radians(yaw_range_deg), math.radians(yaw_range_deg))
    env_cfg.commands.motion.pose_range = pose_cfg
    print(
        "[INFO]: Play robot pose randomization override: "
        f"x=+/-{x_range:.3f} m, y=+/-{y_range:.3f} m, yaw=+/-{yaw_range_deg:.1f} deg"
    )


def _apply_play_face_goal(env_cfg, enabled: bool, yaw_offset_deg: float) -> None:
    if not enabled:
        return
    yaw_offset = math.radians(float(yaw_offset_deg))
    env_cfg.commands.motion.align_initial_heading_to_destination = True
    env_cfg.commands.motion.align_initial_heading_yaw_offset = yaw_offset
    if hasattr(env_cfg.commands.motion, "goal_aware_yaw_error_ranges"):
        env_cfg.commands.motion.goal_aware_yaw_error_ranges = ((yaw_offset, yaw_offset),) * 3
    pose_cfg = dict(getattr(env_cfg.commands.motion, "pose_range", {}) or {})
    pose_cfg["yaw"] = (0.0, 0.0)
    env_cfg.commands.motion.pose_range = pose_cfg
    print(
        "[INFO]: Play face-goal override enabled: robot and ball layout will be yaw-aligned "
        f"to the target goal with extra yaw offset {float(yaw_offset_deg):.1f} deg"
    )


def _repeat_goal_range(value_range):
    return (tuple(value_range), tuple(value_range), tuple(value_range))


def _apply_play_goal_init_stage(env_cfg, stage: int | None) -> None:
    if stage is None:
        return
    stage_value = int(stage)
    source_idx = 0 if stage_value <= 0 else 1 if stage_value == 1 else 2
    motion_cfg = env_cfg.commands.motion
    for name in (
        "goal_aware_robot_x_ranges",
        "goal_aware_robot_y_ranges",
        "goal_aware_yaw_error_ranges",
        "goal_aware_ball_x_front_ranges",
        "goal_aware_ball_y_lat_ranges",
        "goal_aware_ball_y_lat_abs_ranges",
    ):
        if hasattr(motion_cfg, name):
            ranges = getattr(motion_cfg, name)
            if len(ranges) >= 3:
                setattr(motion_cfg, name, _repeat_goal_range(ranges[source_idx]))
    if stage_value >= 3:
        motion_cfg.goal_aware_curriculum_steps = (-3, -2, -1)
        motion_cfg.goal_gate_curriculum_steps = (-3, -2, -1)
    print(f"[INFO]: Play goal-aware init stage override: stage={stage_value}, source_range_index={source_idx}")


def _apply_play_midfield_kick(env_cfg, enabled: bool) -> None:
    if not enabled:
        return
    motion_cfg = env_cfg.commands.motion
    motion_cfg.goal_aware_robot_x_ranges = ((0.5, 3.2),) * 3
    motion_cfg.goal_aware_robot_y_ranges = ((-2.8, 2.8),) * 3
    motion_cfg.goal_aware_ball_x_front_ranges = ((0.55, 1.05),) * 3
    motion_cfg.goal_aware_ball_y_lat_ranges = ((-0.35, 0.35),) * 3
    motion_cfg.goal_aware_ball_y_lat_abs_ranges = ((0.06, 0.30),) * 3
    motion_cfg.goal_aware_curriculum_steps = (-3, -2, -1)
    motion_cfg.goal_gate_curriculum_steps = (-3, -2, -1)
    print("[INFO]: Play midfield kick override enabled: robot x=[0.5,3.2], real goal gate forced.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    apply_g1_robot_variant(env_cfg, args_cli.robot_variant)
    print(f"[INFO]: Using G1 robot variant: {args_cli.robot_variant}")
    _apply_play_robot_pose_range(env_cfg, args_cli.play_robot_pose_range)
    _apply_play_face_goal(env_cfg, args_cli.play_face_goal, args_cli.play_face_goal_yaw_offset_deg)
    _apply_play_goal_init_stage(env_cfg, args_cli.play_goal_init_stage)
    _apply_play_midfield_kick(env_cfg, args_cli.play_midfield_kick)

    if not args_cli.play_no_soccer_field:
        install_soccer_lab_field(env_cfg.scene, include_goal_assets=args_cli.play_goalpost_usd)

    if args_cli.play_follow_robot_camera:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (2.0, 2.0, 1.2)
    else:
        env_cfg.viewer.origin_type = None
        env_cfg.viewer.asset_name = None
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        _set_motion_from_cli_or_v4_default(env_cfg, args_cli)

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            _set_motion_file(env_cfg, str(pathlib.Path(art.download()) / "motion.npz"))

    else:
        _set_motion_from_cli_or_v4_default(env_cfg, args_cli)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # import ipdb; ipdb.set_trace()

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    ckpt = args_cli.checkpoint.split('_')[1].split('.')[0]
    task_name = str(args_cli.task)
    export_student = args_cli.export_student_policy or ("NearFieldGoalKickV4" in task_name and "Teacher" not in task_name)
    exported_filename = None
    if export_student:
        filename = f"policy_{ckpt}_student.onnx"
        export_student_policy_as_onnx(
            env.unwrapped,
            ppo_runner.alg.policy,
            normalizer=ppo_runner.obs_normalizer,
            path=export_model_dir,
            filename=filename,
        )
        exported_filename = filename
    else:
        export_motion_name = None
        if args_cli.motion_file is not None:
            export_motion_name = os.path.basename(args_cli.motion_file)
        elif args_cli.motion_path is not None and pathlib.Path(args_cli.motion_path).is_file():
            export_motion_name = os.path.basename(args_cli.motion_path)

        if export_motion_name is None and "Teacher" in task_name:
            print("[INFO]: Skipping teacher motion ONNX export for multi-motion play. "
                  "Use --motion_file <motion.npz> if you need a motion-embedded teacher export.")
        else:
            filename = f"policy_{ckpt}.onnx"
            export_motion_policy_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                normalizer=ppo_runner.obs_normalizer,
                path=export_model_dir,
                filename=filename,
                motion_name=export_motion_name,
            )
            exported_filename = filename
    if exported_filename is not None:
        attach_onnx_metadata(
            env.unwrapped,
            args_cli.wandb_path if args_cli.wandb_path else "none",
            export_model_dir,
            filename=exported_filename,
        )
    # reset environment
    # breakpoint()
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
