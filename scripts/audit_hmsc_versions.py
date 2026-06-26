#!/usr/bin/env python3
"""Audit HumanoidSoccer HMSC goal-kick training versions and deploy readiness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FOOTBALL_ROOT = REPO_ROOT.parent
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
DEPLOY_MODEL_DIR = FOOTBALL_ROOT / "football_action_go" / "actions" / "football" / "model"


REWARD_KEYS = (
    "goal_cross_speed_reward",
    "style_gated_side_foot_ball_speed",
    "side_foot_contact_leg_speed",
    "side_foot_ball_speed_lite",
    "autonomous_ball_speed",
    "ball_speed_reward",
    "ball_velocity_to_goal",
    "ball_forward_progress",
    "inside_foot_contact",
    "toe_contact_penalty",
    "instep_contact_penalty",
    "wrong_side_foot_contact_penalty",
    "pre_contact_motion_style_lite",
    "motion_foot_pos",
    "goal_aware_root_trajectory",
    "post_kick_stand_still",
    "post_kick_drift_penalty",
    "non_timeout_termination_penalty",
)

SCALAR_KEYS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/motion/kick_success_rate",
    "Metrics/motion/expected_kick_success_rate",
    "Metrics/motion/goal_success_rate",
    "Metrics/motion/goal_gate_miss_rate",
    "Metrics/motion/gate_lateral_error",
    "Metrics/motion/gate_cross_speed",
    "Metrics/motion/ball_forward_progress_mean",
    "Metrics/motion/ball_velocity_to_goal_mean",
    "Metrics/motion/side_foot_contact_rate",
    "Metrics/motion/inside_foot_contact_rate",
    "Metrics/motion/toe_contact_rate",
    "Metrics/motion/instep_contact_rate",
    "Metrics/motion/style_gated_ball_speed_raw",
    "Metrics/motion/style_gated_ball_speed",
    "Metrics/motion/side_foot_leg_speed",
    "Metrics/motion/side_foot_leg_speed_reward",
    "Metrics/motion/side_foot_ball_speed_lite",
    "Metrics/motion/side_foot_ball_speed_lite_forward_vel",
    "Episode_Termination/time_out",
    "Episode_Termination/anchor_pos_z",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/ee_body_pos",
)

SIM2REAL_EVENT_KEYS = (
    "physics_material",
    "base_com",
    "push_robot",
    "robot_body_mass",
    "actuator_gains",
    "joint_armature",
    "joint_friction",
    "ball_mass",
    "ball_material",
)


@dataclass(frozen=True)
class VersionSpec:
    name: str
    task: str
    run: str | None
    deploy_model: str | None
    notes: str


VERSIONS = (
    VersionSpec(
        "V1 nearfield kick",
        "Tracking-Flat-G1-NearFieldKick-RNN-v0",
        "2026-06-15_14-52-23_nearfield_kick_4096",
        "kick_right.onnx",
        "Legacy near-field kick baseline; not goal-gate or V4 student deploy native.",
    ),
    VersionSpec(
        "V2 goal gate stable",
        "Tracking-Flat-G1-NearFieldGoalKickV2-RNN-v0",
        "2026-06-16_23-29-46_nearfield_goal_gate_v2_stable_restart",
        None,
        "Goal-gate shaping matured but still motion/reference heavy.",
    ),
    VersionSpec(
        "V2 foot balance",
        "Tracking-Flat-G1-NearFieldGoalKickV2-RNN-v0",
        "2026-06-17_18-29-41_nearfield_goal_gate_v2_foot_balance",
        None,
        "Foot-balance branch; no deploy ONNX in football_action_go.",
    ),
    VersionSpec(
        "V3 goal kick",
        "Tracking-Flat-G1-NearFieldGoalKickV3-RNN-v0",
        None,
        "goalkick_v3.onnx",
        "Task is registered, but no dedicated V3 run was found in current logs.",
    ),
    VersionSpec(
        "V4 student",
        "Tracking-Flat-G1-NearFieldGoalKickV4Student-RNN-v0",
        "2026-06-19_00-08-36_nearfield_goalkick_v4_student_4096",
        "goalkick_v4_student.onnx",
        "Deploy-native 101-dim student observation; trained long but not current default.",
    ),
    VersionSpec(
        "V4.1 side-foot stable",
        "Tracking-Flat-G1-NearFieldGoalKickV4SideFootStable-RNN-v0",
        "2026-06-21_00-18-21_nearfield_goalkick_v4_1_sidefoot_stable_4096",
        "goalkick_v4_1_sidefoot_stable.onnx",
        "Best current deployed baseline; stable side-foot style, slower ball speed.",
    ),
    VersionSpec(
        "V4.1 power-stable finetune",
        "Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStable-RNN-v0",
        "nearfield_goalkick_v4_1_powerstable_ft10000_4096",
        None,
        "Planned v4.1 finetune from model_10000 with gated leg-speed and ball-speed rewards.",
    ),
    VersionSpec(
        "V4.1 power-stable boost from0",
        "Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0",
        "nearfield_goalkick_v4_1_powerstable_boost_from0_4096",
        None,
        "Boosted speed branch after play audit; new runs should train from scratch under the current reward set.",
    ),
    VersionSpec(
        "V4.1 power-stable from0",
        "Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStable-RNN-v0",
        "nearfield_goalkick_v4_1_powerstable_from0_4096",
        None,
        "Planned from-scratch control run for the same gated power-stable reward set.",
    ),
    VersionSpec(
        "V4.2 power mid-goal",
        "Tracking-Flat-G1-NearFieldGoalKickV4PowerMidGoal-RNN-v0",
        "2026-06-21_16-39-48_nearfield_goalkick_v4_2_power_mid_goal_4096",
        None,
        "High real-goal speed/reward branch; short run, no exported ONNX found.",
    ),
    VersionSpec(
        "V4.2 speedonly side-foot",
        "Tracking-Flat-G1-NearFieldGoalKickV4PowerMidGoal-RNN-v0",
        "2026-06-22_14-55-12_nearfield_goalkick_v4_2_speedonly_sidefoot_4096",
        "goalkick_v4_2_speedonly_sidefoot.onnx",
        "Short run to 7000 with export, but model is not synced to deploy dir.",
    ),
    VersionSpec(
        "V4.1 speedstyle",
        "Tracking-Flat-G1-NearFieldGoalKickV4SideFootSpeed-RNN-v0",
        "2026-06-22_21-28-49_nearfield_goalkick_v4_1_speedstyle_4096",
        "goalkick_v4_1_speedstyle.onnx",
        "Preferred speed fine-tune direction from V4.1 style; not current default.",
    ),
    VersionSpec(
        "V4 lite power",
        "Tracking-Flat-G1-NearFieldGoalKickV4LitePower-RNN-v0",
        "2026-06-23_22-46-58_nearfield_goalkick_v4_lite_power_from0_4096",
        None,
        "Current from-scratch experiment; treat as immature until checkpoints/export exist.",
    ),
)


def rel(path: Path | None) -> str:
    if path is None:
        return "-"
    try:
        return str(path.relative_to(FOOTBALL_ROOT))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.unsafe_load(f)
    return data or {}


def model_iterations(run_dir: Path) -> list[int]:
    out: list[int] = []
    for path in run_dir.glob("model_*.pt"):
        try:
            out.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def exported_onnx(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "exported").glob("policy*_student.onnx")) + sorted(
        (run_dir / "exported").glob("policy_*.onnx")
    )


def latest_export(run_dir: Path) -> Path | None:
    exports = exported_onnx(run_dir)
    return exports[-1] if exports else None


def deploy_state(model_name: str | None) -> str:
    if model_name is None:
        return "-"
    path = DEPLOY_MODEL_DIR / model_name
    return "present" if path.exists() else "missing"


def resolve_run_dir(run: str | None) -> Path | None:
    if run is None:
        return None
    exact = LOG_ROOT / run
    if exact.exists():
        return exact
    matches = sorted(path for path in LOG_ROOT.glob(f"*_{run}") if path.is_dir())
    return matches[-1] if matches else None


def reward_weights(env_yaml: dict[str, Any]) -> dict[str, Any]:
    rewards = env_yaml.get("rewards", {})
    out: dict[str, Any] = {}
    for key in REWARD_KEYS:
        if key in rewards:
            out[key] = rewards[key].get("weight")
    return out


def sim2real_events(env_yaml: dict[str, Any]) -> dict[str, Any]:
    events = env_yaml.get("events", {})
    return {key: events[key].get("params", {}) for key in SIM2REAL_EVENT_KEYS if key in events}


def command_summary(env_yaml: dict[str, Any]) -> dict[str, Any]:
    command = env_yaml.get("commands", {}).get("motion", {})
    keys = (
        "near_field_ball_visible_distance_range",
        "perception_ball_update_period_steps",
        "perception_ball_noise_std",
        "perception_ball_latency_range_s",
        "kick_latch_start_phase_range",
        "post_trigger_ball_dropout_prob_range",
        "kick_direction_yaw_noise_range",
        "goal_aware_ball_x_front_ranges",
        "goal_aware_ball_y_lat_ranges",
        "goal_aware_ball_y_lat_abs_ranges",
        "target_destination_center",
        "balance_motion_kick_leg_sampling",
    )
    return {key: command[key] for key in keys if key in command}


def latest_scalars(run_dir: Path) -> dict[str, tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    except Exception:
        return {}

    latest: dict[str, tuple[int, float]] = {}
    for event_file in sorted(run_dir.glob("events.out.tfevents*")):
        for event in EventFileLoader(str(event_file)).Load():
            if not event.summary:
                continue
            for value in event.summary.value:
                if value.tag not in SCALAR_KEYS:
                    continue
                if value.HasField("simple_value"):
                    latest[value.tag] = (event.step, float(value.simple_value))
                elif value.HasField("tensor") and value.tensor.float_val:
                    latest[value.tag] = (event.step, float(value.tensor.float_val[0]))
    return latest


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(fmt_value(v) for v in value) + ")"
    if isinstance(value, dict):
        items = ", ".join(f"{k}: {fmt_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    return str(value)


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def select_versions(names: list[str] | None) -> tuple[VersionSpec, ...]:
    if not names:
        return VERSIONS
    requested = {name.lower() for name in names}
    selected = tuple(
        spec
        for spec in VERSIONS
        if spec.name.lower() in requested
        or spec.name.lower().replace(" ", "-") in requested
        or spec.name.lower().replace(" ", "_") in requested
    )
    missing = sorted(requested - {spec.name.lower() for spec in selected}
                     - {spec.name.lower().replace(" ", "-") for spec in selected}
                     - {spec.name.lower().replace(" ", "_") for spec in selected})
    if missing:
        raise SystemExit(f"unknown version name(s): {', '.join(missing)}")
    return selected


def build_report(include_scalars: bool, versions: tuple[VersionSpec, ...] = VERSIONS) -> str:
    lines: list[str] = []
    lines.append("# HMSC Training and Sim2Real Audit")
    lines.append("")
    lines.append("Generated from local HumanoidSoccer logs and football_action_go deploy files.")
    lines.append("")

    rows = [["Version", "Task", "Run", "Checkpoint", "Export", "Deploy", "Notes"]]
    env_yamls: dict[str, dict[str, Any]] = {}
    scalar_data: dict[str, dict[str, tuple[int, float]]] = {}

    for spec in versions:
        run_dir = resolve_run_dir(spec.run)
        env_yaml = load_yaml(run_dir / "params" / "env.yaml") if run_dir else {}
        env_yamls[spec.name] = env_yaml
        iters = model_iterations(run_dir) if run_dir and run_dir.exists() else []
        checkpoint = str(iters[-1]) if iters else "-"
        export = rel(latest_export(run_dir)) if run_dir and run_dir.exists() else "-"
        deploy = deploy_state(spec.deploy_model)
        if include_scalars and run_dir and run_dir.exists():
            scalar_data[spec.name] = latest_scalars(run_dir)
        rows.append([spec.name, spec.task, run_dir.name if run_dir else spec.run or "-", checkpoint, export, deploy, spec.notes])

    lines.append("## Version Matrix")
    lines.append("")
    lines.append(markdown_table(rows))
    lines.append("")

    lines.append("## Reward Weights")
    lines.append("")
    reward_rows = [["Version", "Key weights"]]
    for spec in versions:
        weights = reward_weights(env_yamls.get(spec.name, {}))
        summary = ", ".join(f"{k}={fmt_value(v)}" for k, v in weights.items()) or "-"
        reward_rows.append([spec.name, summary])
    lines.append(markdown_table(reward_rows))
    lines.append("")

    if include_scalars:
        lines.append("## Latest Scalars")
        lines.append("")
        scalar_rows = [["Version", "Latest selected scalars"]]
        for spec in versions:
            latest = scalar_data.get(spec.name, {})
            summary = ", ".join(f"{k}={v:g}@{step}" for k, (step, v) in latest.items()) or "-"
            scalar_rows.append([spec.name, summary])
        lines.append(markdown_table(scalar_rows))
        lines.append("")

    lines.append("## Sim2Real Coverage")
    lines.append("")
    for spec in versions:
        env_yaml = env_yamls.get(spec.name, {})
        if not env_yaml:
            continue
        lines.append(f"### {spec.name}")
        lines.append("")
        lines.append("- Command/perception: " + (", ".join(f"{k}={fmt_value(v)}" for k, v in command_summary(env_yaml).items()) or "-"))
        lines.append("- Randomization events: " + (", ".join(sim2real_events(env_yaml).keys()) or "-"))
        lines.append("")

    lines.append("## Deploy Contract Checks")
    lines.append("")
    lines.append("- V4 student actor observation is 101 dims: projected gravity, gyro, 29 joint positions, 29 joint velocities, 29 previous actions, ball xyz/valid/age, kick direction xy, kick elapsed phase.")
    lines.append("- football_action_go currently reports these deploy model files:")
    for path in sorted(DEPLOY_MODEL_DIR.glob("*.onnx")):
        lines.append(f"  - {rel(path)} ({path.stat().st_size} bytes)")
    lines.append("")
    lines.append("## Recommended Direction")
    lines.append("")
    lines.append("Keep V4.1 side-foot stable as the deploy baseline. For new speed, lift, and post-still experiments, prefer from-scratch training under the current reward set; keep V4.2/lite-power experimental until they have mature checkpoints, exported ONNX, check-only, and fixed-ball MuJoCo validation.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-scalars", action="store_true", help="Read TensorBoard event files and include latest selected scalar metrics.")
    parser.add_argument(
        "--versions",
        nargs="+",
        help="Limit the audit to version names. Use quoted names, or replace spaces with '-' or '_'.",
    )
    args = parser.parse_args()
    print(build_report(include_scalars=args.include_scalars, versions=select_versions(args.versions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
