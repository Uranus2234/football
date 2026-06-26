#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/rsl_rl/train_student.py \
  --task Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0 \
  --num_envs "${NUM_ENVS:-4096}" \
  --motion_path "${MOTION_PATH:-motions/soccer-standard}" \
  --from_scratch \
  --run_name "${RUN_NAME:-nearfield_goalkick_v4_1_powerstable_boost_from0_4096}" \
  --max_iterations "${MAX_ITERATIONS:-45000}" \
  --headless
