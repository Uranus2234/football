import gymnasium as gym

from . import agents, flat_env_cfg
from . import soccer_flat_env_cfg

##
# Register Gym environments.
##

## Motion tracking environments
gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)


## Soccer environments
###  Stage 1
# Terrain
gym.register(
    id="Tracking-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Terrain-G1-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1TerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)
# Flat
gym.register(
    id="Tracking-Flat-G1-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


###  Stage 2
gym.register(
    id="Tracking-Flat-G1-SoccerDestination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-SoccerDestination-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerMoving-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatKickMovingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldKick-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKick-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV2-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV3-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4Student-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4StudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootStable-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootStableEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStable-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootPowerStableEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableBoost-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootPowerStableBoostEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableLift-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootPowerStableLiftEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootPowerStableLiftPostStill-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootPowerStableLiftPostStillEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4LitePower-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4LitePowerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4InsideStand-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4InsideStandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4RecoveryPrior-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4RecoveryPriorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4SideFootSpeed-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4SideFootSpeedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4BeyondMimicTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickBeyondMimicTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatBeyondMimicTeacherPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4BeyondMimicStudentDistill-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickBeyondMimicStudentDistillEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatBeyondMimicStudentTeacherRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKickV4PowerMidGoal-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickV4PowerMidGoalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentLargePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-NearFieldGoalKick-D455-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatNearFieldGoalKickD455EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)




## Advanced Soccer environments

# Only-vision
gym.register(
    id="Tracking-Flat-G1-SoccerBlind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SoccerBlind-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatRecurrentPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-SuperSoccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSuperSoccerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Soccer-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.G1FlatSoccerStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatStudentTeacherPPORunnerCfg",
    },
)
