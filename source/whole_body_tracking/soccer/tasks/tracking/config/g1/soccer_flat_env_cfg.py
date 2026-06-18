import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from soccer.assets.soccer_lab import (
    SOCCER_LAB_BALL_RADIUS,
    SOCCER_LAB_BALL_USD,
    SOCCER_LAB_FIELD,
    SOCCER_LAB_FOOTBALL_DETECTION_ONNX,
    SOCCER_LAB_GOALPOST_USD,
    build_field_line_specs,
    build_goal_asset_specs,
    build_goal_post_specs,
    require_soccer_lab_assets,
)
from soccer.robots.g1 import (
    G1_ACTION_SCALE,
    G1_BASE_MOTION_BODY_NAMES,
    G1_BODY_JOINT_NAMES,
    G1_CYLINDER_CFG,
    G1_CYLINDER_D455_CFG,
    G1_D455_SENSOR_HOLD_POS,
)
from soccer.robots.actuator import DelayedImplicitActuatorCfg
from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, MySceneCfg, CurriculumCfg
from .flat_env_cfg import G1FlatEnvCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains import TerrainGeneratorCfg

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab.managers import TerminationTermCfg as DoneTerm

SOCCER_BALL_RADIUS = SOCCER_LAB_BALL_RADIUS
SOCCER_FIELD_LENGTH = SOCCER_LAB_FIELD.field_length
SOCCER_FIELD_WIDTH = SOCCER_LAB_FIELD.field_width
SOCCER_GOAL_WIDTH = SOCCER_LAB_FIELD.goal_width
SOCCER_GOAL_HALF_WIDTH = SOCCER_GOAL_WIDTH * 0.5
SOCCER_LAB_BALL_RIGID_BODY_PRIM = "Ball_obj_cleaner_materialmerger_gles"


def _make_delayed_actuator_cfg(actuator_cfg, min_delay: int = 0, max_delay: int = 3) -> DelayedImplicitActuatorCfg:
    return DelayedImplicitActuatorCfg(
        joint_names_expr=actuator_cfg.joint_names_expr,
        effort_limit_sim=actuator_cfg.effort_limit_sim,
        velocity_limit_sim=actuator_cfg.velocity_limit_sim,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        armature=actuator_cfg.armature,
        min_delay=min_delay,
        max_delay=max_delay,
    )


def _soccer_lab_pitch_base_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PitchBase",
        spawn=sim_utils.CuboidCfg(
            size=(
                SOCCER_LAB_FIELD.field_length + 2.0 * SOCCER_LAB_FIELD.border_strip_width + 4.0,
                SOCCER_LAB_FIELD.field_width + 2.0 * SOCCER_LAB_FIELD.border_strip_width + 4.0,
                0.04,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.48, 0.24), roughness=0.9),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
    )


def _soccer_lab_field_line_cfg(line) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/FieldLine_{line.name}",
        spawn=sim_utils.CuboidCfg(
            size=line.size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=line.position, rot=line.orientation),
    )


def _soccer_lab_goal_post_cfg(post) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Goal_{post.name}",
        spawn=sim_utils.CuboidCfg(
            size=post.size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.92), roughness=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=post.position),
    )


def _soccer_lab_goal_asset_cfg(goal_asset) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{goal_asset.name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SOCCER_LAB_GOALPOST_USD),
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=goal_asset.position, rot=goal_asset.orientation),
    )


def _install_soccer_lab_field(scene_cfg) -> None:
    require_soccer_lab_assets()
    # TrackingEnvCfg expects a terrain object for the global physics material.
    # Keep the plane as the physical ground, but style it as Soccer_Lab grass
    # and add Soccer_Lab's pitch markings/goals as per-env assets.
    if scene_cfg.terrain is not None:
        scene_cfg.terrain.physics_material = scene_cfg.terrain.physics_material.replace(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        scene_cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.12, 0.48, 0.24),
            roughness=0.9,
        )
    scene_cfg.pitch_base = _soccer_lab_pitch_base_cfg()
    for line in build_field_line_specs(SOCCER_LAB_FIELD, line_height=0.01, z_offset=0.005):
        setattr(scene_cfg, f"field_line_{line.name}", _soccer_lab_field_line_cfg(line))
    for post in build_goal_post_specs(SOCCER_LAB_FIELD):
        setattr(scene_cfg, f"goal_post_{post.name}", _soccer_lab_goal_post_cfg(post))
    for goal_asset in build_goal_asset_specs(SOCCER_LAB_FIELD, z_offset=SOCCER_LAB_FIELD.goal_height * 0.5):
        setattr(scene_cfg, goal_asset.name, _soccer_lab_goal_asset_cfg(goal_asset))


def _apply_soccer_obs(cfg):
    cfg.observations.policy.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.policy.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )


def _apply_soccer_scene(cfg):
    cfg.scene.soccer_ball = cfg.scene.soccer_ball.replace(prim_path="{ENV_REGEX_NS}/SoccerBall")
    cfg.scene.soccer_ball.init_state.pos = (0.0, 0.0, SOCCER_BALL_RADIUS)

    cfg.commands.motion.target_point_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/TargetPoint",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    cfg.commands.motion.target_destination_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PostKickTarget",
        markers={
            "destination_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )


def _apply_g1_motion_body_indexes(cfg):
    cfg.commands.motion.motion_body_indexes = [
        G1_BASE_MOTION_BODY_NAMES.index(body_name) for body_name in cfg.commands.motion.body_names
    ]


## Scene configuration

@configclass
class G1FlatSoccerSceneCfg(MySceneCfg):
    def __post_init__(self):
        super().__post_init__()
        _install_soccer_lab_field(self)

    soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SOCCER_LAB_BALL_USD),
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                max_linear_velocity=80.0,
                max_angular_velocity=200.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, 0.0, SOCCER_BALL_RADIUS),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    soccer_ball_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/SoccerBall/{SOCCER_LAB_BALL_RIGID_BODY_PRIM}",
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )


@configclass
class G1FlatSoccerD455SceneCfg(G1FlatSoccerSceneCfg):
    """Soccer scene with the head-top D455 RGB-D camera enabled."""

    robot_camera = CameraCfg(
        # Match Soccer_Lab's camera topology. This IsaacLab version does not
        # expose Soccer_Lab's update_latest_camera_pose flag, so perception
        # scripts force camera buffer refreshes after head-joint updates.
        prim_path="{ENV_REGEX_NS}/Robot/d455_link/D455Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.661981459865443,
            horizontal_aperture=20.955,
            vertical_aperture=13.324690889790768,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.04061, 0.01000, -0.02207),
            rot=(0.939696, 0.0, 0.342002, 0.0),
            convention="world",
        ),
    )
    

## Environment configuration


def _apply_d455_policy_interfaces(cfg):
    def body_joint_cfg() -> SceneEntityCfg:
        return SceneEntityCfg("robot", joint_names=list(G1_BODY_JOINT_NAMES), preserve_order=True)

    cfg.scene.robot = G1_CYLINDER_D455_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.joint_names = list(G1_BODY_JOINT_NAMES)
    cfg.actions.joint_pos.preserve_order = True
    cfg.actions.joint_pos.scale = G1_ACTION_SCALE
    cfg.commands.motion.controlled_joint_names = list(G1_BODY_JOINT_NAMES)
    cfg.commands.motion.sensor_joint_hold_pos = dict(G1_D455_SENSOR_HOLD_POS)
    cfg.commands.motion.debug_vis = False
    _apply_g1_motion_body_indexes(cfg)

    for obs_group in (cfg.observations.policy, cfg.observations.critic):
        if getattr(obs_group, "joint_pos", None) is not None:
            obs_group.joint_pos.params = {"asset_cfg": body_joint_cfg()}
        if getattr(obs_group, "joint_vel", None) is not None:
            obs_group.joint_vel.params = {"asset_cfg": body_joint_cfg()}

    cfg.events.add_joint_default_pos.params["asset_cfg"] = body_joint_cfg()
    cfg.rewards.joint_limit.params["asset_cfg"] = body_joint_cfg()

@configclass
class G1TerrainEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        _apply_g1_motion_body_indexes(self)
        self.terminations.anchor_pos_z = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},  # Slightly larger threshold for robustness.
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25, # 0.75, # 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1., noise_range=(-0.02, 0.02), noise_step=0.02, border_width=0.0
                )
            },
        )

        # ground terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GRAVEL_TERRAINS_CFG
        )


@configclass
class G1TerrainMotionEnvCfg(G1TerrainEnvCfg):
    scene: G1FlatSoccerSceneCfg = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.sampling_strategy = "adaptive"
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@configclass
class G1FlatMotionEnvCfg(G1FlatEnvCfg):
    scene: G1FlatSoccerSceneCfg = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.commands.motion.sampling_strategy = "uniform"
        _apply_g1_motion_body_indexes(self)
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@configclass
class G1FlatProximityEnvCfg(G1FlatMotionEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.foot_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        )

        self.waist_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
        )

        self.commands.motion.curve_offset_range = {
            "radius": (-0.25, 0.25),
            "arc_angle": math.pi / 9,
            "height": SOCCER_BALL_RADIUS,
        }


        self.rewards.foot_distance = RewTerm(
            func=mdp.foot_distance,
            weight=0.2,
            params={
                "threshold": 0.24,
                "std": 0.5,
                "foot_cfg": self.foot_cfg,
            },
        )

        # self.rewards.feet_slip_penalty = RewTerm(
        #     func=mdp.feet_slip_penalty,
        #     weight=-1.0,
        #     params={
        #         "foot_cfg": self.foot_cfg,
        #         "slip_force_threshold": 5.0,
        #     },
        # )

        self.rewards.target_point_proximity = RewTerm(
            func=mdp.target_point_proximity,
            weight=1.0,
            params={
                "std": 4.0,
                "command_name": "motion",
            },
        )

        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            # weight=0.5,
            weight=0.0,
            params={"command_name": "motion", "std": 0.3},
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )

        self.rewards.waist_action_rate_l2 = RewTerm(
            func=mdp.waist_action_rate_l2_clip,
            weight=-2.5e-1,
            params={
                "waist_cfg": self.waist_cfg,
            },
        )

        self.rewards.pelvis_orientation = RewTerm(
            func=mdp.pelvis_orientation,
            weight=-1.0,
            params={"command_name": "motion",},
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names" : [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, 
                "body_names" : [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_foot_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3,
                    "foot_body_names" : [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            },
        )




@configclass
class G1FlatKickEnvCfg(G1FlatProximityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.target_point_contact = RewTerm(
            func=mdp.target_point_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.sideways_kick = RewTerm(
            func=mdp.sideways_kick,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        
        self.rewards.ball_velocity_direction_alignment = RewTerm(
            func=mdp.ball_velocity_direction_alignment,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_speed_reward = RewTerm(
            func=mdp.ball_speed_reward,
            weight=10.0,
            params={
                "command_name": "motion",
                # "target_speed": 4.0,
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_z_speed_penalty_reward = RewTerm(
            func=mdp.ball_z_speed_penalty_reward,
            weight=-0.0,
            params={
                "command_name": "motion",
                "std": 3,
                "velocity_threshold": 0.5,
            },
        )

@configclass
class G1FlatKickMovingEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Initial soccer-ball linear velocity configuration.
        self.commands.motion.enable_soccer_ball_init_vel = True  # Enable sampling of initial ball velocity.
        self.commands.motion.soccer_ball_init_lin_vel_range = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
        }


@configclass
class G1FlatNearFieldKickEnvCfg(G1FlatKickEnvCfg):
    """Static-ball near-field kicker with deployable perception observations.

    Actor observations follow the whole_body_tracking wo-state-estimation shape:
    no base linear velocity or global pose enters the low-level policy.  Global
    localization is represented only as a noisy desired kick direction.
    """

    def __post_init__(self):
        super().__post_init__()

        # Soccer_Lab convention: field center at origin, long axis x,
        # attacking the +x goal by default.
        self.commands.motion.target_destination_center = (SOCCER_FIELD_LENGTH * 0.5, 0.0, SOCCER_BALL_RADIUS)
        self.commands.motion.target_destination_length = 0.2
        self.commands.motion.target_destination_width = SOCCER_GOAL_WIDTH

        # Near-field perception model: YOLO/depth at about 10 Hz for a 50 Hz
        # policy, with mild metric noise before kick and heavy occlusion after
        # the sampled latch phase.
        self.commands.motion.near_field_ball_visible_distance_range = (0.15, 1.35)
        self.commands.motion.perception_ball_update_period_steps = 5
        self.commands.motion.perception_ball_noise_std = (0.03, 0.03, 0.02)
        self.commands.motion.blind_dropout_prob = 0.05
        self.commands.motion.kick_latch_start_phase_range = (0.15, 0.45)
        self.commands.motion.post_trigger_ball_dropout_prob_range = (0.0, 1.0)
        self.commands.motion.kick_direction_yaw_noise_range = (-math.radians(8.0), math.radians(8.0))

        # Actor gets local ball latch + validity/age, plus a goal direction
        # command.  The critic keeps the truth observations inherited from
        # G1FlatKickEnvCfg for asymmetric training.
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.near_field_latched_ball_observation,
            params={"command_name": "motion"},
        )
        self.observations.policy.target_destination_pos_local = None
        self.observations.policy.desired_kick_direction_base = ObsTerm(
            func=mdp.desired_kick_direction_base,
            params={"command_name": "motion"},
        )


@configclass
class G1FlatNearFieldGoalKickEnvCfg(G1FlatNearFieldKickEnvCfg):
    """Near-field kick task whose primary objective is crossing a goal confidence gate."""

    def __post_init__(self):
        super().__post_init__()

        # The destination point is the real Soccer_Lab +x goal center; the gate
        # half-width below represents the 2.4 m M-field goal mouth.
        self.commands.motion.target_destination_length = 0.0
        self.commands.motion.target_destination_width = 0.0
        self.commands.motion.goal_gate_curriculum_steps = (24000, 72000, 144000)
        self.commands.motion.goal_gate_local_distance = 1.5
        self.commands.motion.goal_gate_mid_distance = 3.0
        self.commands.motion.goal_gate_local_half_width = 0.45
        self.commands.motion.goal_gate_mid_half_width = 0.65
        self.commands.motion.goal_gate_real_half_width = SOCCER_GOAL_HALF_WIDTH

        self.rewards.target_point_contact.weight = 20.0
        self.rewards.sideways_kick.weight = 0.0
        self.rewards.ball_velocity_direction_alignment.weight = 15.0
        self.rewards.ball_speed_reward.weight = 15.0

        self.rewards.goal_gate_success = RewTerm(
            func=mdp.goal_gate_success,
            weight=200.0,
            params={"command_name": "motion"},
        )
        self.rewards.goal_gate_miss = RewTerm(
            func=mdp.goal_gate_miss,
            weight=-50.0,
            params={"command_name": "motion"},
        )


@configclass
class G1FlatNearFieldGoalKickV2EnvCfg(G1FlatNearFieldGoalKickEnvCfg):
    """Goal-aware near-field kick task with attack-half resets and trajectory shaping."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.enable_goal_aware_initialization = True
        self.commands.motion.align_motion_reference_to_initial_heading = True
        self.commands.motion.balance_motion_kick_leg_sampling = True
        self.commands.motion.goal_aware_ball_lateral_by_kick_leg = True
        self.commands.motion.goal_aware_curriculum_steps = (96000, 288000, 576000)
        self.commands.motion.goal_gate_curriculum_steps = (96000, 288000, 576000)
        self.commands.motion.goal_gate_real_half_width = SOCCER_GOAL_HALF_WIDTH

        self.rewards.motion_global_anchor_ori.weight = 1.5
        self.rewards.motion_body_pos.weight = 2.0
        self.rewards.motion_body_ori.weight = 2.0
        self.rewards.motion_body_lin_vel.weight = 1.5
        self.rewards.motion_body_ang_vel.weight = 1.5
        self.rewards.motion_foot_pos.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.waist_action_rate_l2.weight = -0.5
        self.rewards.pelvis_orientation.weight = -2.0
        self.rewards.joint_limit.weight = -15.0

        self.rewards.target_point_contact.weight = 18.0
        self.rewards.sideways_kick.weight = 8.0
        self.rewards.ball_velocity_direction_alignment.weight = 10.0
        self.rewards.ball_speed_reward.weight = 8.0
        self.rewards.goal_gate_success.weight = 300.0
        self.rewards.goal_gate_miss.weight = -75.0

        self.rewards.wrong_foot_contact_penalty = RewTerm(
            func=mdp.wrong_foot_contact_penalty,
            weight=-35.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )
        self.rewards.non_timeout_termination_penalty = RewTerm(
            func=mdp.non_timeout_termination_penalty,
            weight=-250.0,
        )
        self.rewards.post_kick_alive = RewTerm(
            func=mdp.post_kick_alive,
            weight=2.0,
            params={"command_name": "motion"},
        )
        self.rewards.post_goal_alive = RewTerm(
            func=mdp.post_goal_alive,
            weight=4.0,
            params={"command_name": "motion"},
        )

        self.rewards.ball_forward_progress = RewTerm(
            func=mdp.ball_forward_progress,
            weight=8.0,
            params={
                "command_name": "motion",
                "speed_scale": 3.0,
                "corridor_half_width": 0.5,
            },
        )
        self.rewards.ball_velocity_to_goal = RewTerm(
            func=mdp.ball_velocity_to_goal,
            weight=10.0,
            params={
                "command_name": "motion",
                "speed_scale": 3.0,
                "corridor_half_width": 0.5,
            },
        )
        self.rewards.ball_lateral_corridor_penalty = RewTerm(
            func=mdp.ball_lateral_corridor_penalty,
            weight=-12.0,
            params={
                "command_name": "motion",
                "speed_scale": 3.0,
                "corridor_half_width": 0.5,
            },
        )
        self.rewards.ball_wrong_way_penalty = RewTerm(
            func=mdp.ball_wrong_way_penalty,
            weight=-10.0,
            params={
                "command_name": "motion",
                "speed_scale": 3.0,
                "corridor_half_width": 0.5,
            },
        )


@configclass
class G1FlatNearFieldGoalKickV3EnvCfg(G1FlatNearFieldGoalKickV2EnvCfg):
    """Large-RNN, sim2real-randomized near-field goal kick task."""

    def __post_init__(self):
        super().__post_init__()

        for actuator_name in ("legs", "feet", "waist", "waist_yaw"):
            actuator_cfg = self.scene.robot.actuators.get(actuator_name)
            if actuator_cfg is not None:
                self.scene.robot.actuators[actuator_name] = _make_delayed_actuator_cfg(
                    actuator_cfg,
                    min_delay=0,
                    max_delay=3,
                )

        self.commands.motion.perception_ball_latency_range_s = (0.10, 0.20)
        self.commands.motion.goal_aware_ball_x_front_ranges = (
            (0.35, 0.65),
            (0.45, 0.85),
            (0.55, 0.95),
        )
        self.commands.motion.goal_aware_ball_y_lat_abs_ranges = (
            (0.10, 0.18),
            (0.08, 0.24),
            (0.06, 0.30),
        )

        self.events.robot_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.85, 1.15),
                "operation": "scale",
            },
        )
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.80, 1.20),
                "operation": "scale",
            },
        )
        self.events.joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.0, 0.08),
                "operation": "add",
            },
        )
        self.events.ball_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "mass_distribution_params": (0.38, 0.48),
                "operation": "abs",
                "recompute_inertia": True,
            },
        )
        self.events.ball_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("soccer_ball"),
                "static_friction_range": (0.45, 1.20),
                "dynamic_friction_range": (0.35, 1.00),
                "restitution_range": (0.0, 0.25),
                "num_buckets": 32,
                "make_consistent": True,
            },
        )

        self.rewards.motion_global_anchor_pos.weight = 0.0
        self.rewards.goal_aware_root_trajectory = RewTerm(
            func=mdp.goal_aware_root_trajectory_error_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "std": 0.35,
                "decay_after_contact": True,
            },
        )
        self.rewards.pre_contact_double_air_penalty = RewTerm(
            func=mdp.pre_contact_double_air_penalty,
            weight=-2.0,
            params={
                "command_name": "motion",
                "foot_cfg": self.foot_cfg,
                "contact_force_threshold": 5.0,
                "min_air_height": 0.04,
                "grace_steps": 5,
            },
        )

        self.rewards.target_point_contact.weight = 10.0
        self.rewards.sideways_kick.weight = 10.0
        self.rewards.inside_foot_contact = RewTerm(
            func=mdp.inside_foot_contact_reward,
            weight=25.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )
        self.rewards.toe_contact_penalty = RewTerm(
            func=mdp.toe_contact_penalty,
            weight=-25.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.goal_gate_success.weight = 0.0
        self.rewards.goal_gate_center_success = RewTerm(
            func=mdp.goal_gate_center_success,
            weight=300.0,
            params={"command_name": "motion"},
        )
        self.rewards.goal_cross_speed_reward = RewTerm(
            func=mdp.goal_cross_speed_reward,
            weight=60.0,
            params={
                "command_name": "motion",
                "speed_scale": 3.0,
            },
        )
        self.rewards.ball_velocity_to_goal.weight = 8.0
        self.rewards.ball_speed_reward.weight = 6.0
        self.rewards.ball_forward_progress.weight = 6.0


@configclass
class G1FlatNearFieldGoalKickD455EnvCfg(G1FlatNearFieldGoalKickEnvCfg):
    """Near-field goal kick with U2 body and the real head-top D455 camera asset."""

    scene: G1FlatSoccerD455SceneCfg = G1FlatSoccerD455SceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        _apply_d455_policy_interfaces(self)
        self.d455_detection_horizontal_fov = 89.0
        self.d455_detection_vertical_fov = 64.0
        self.d455_default_trt_engine_path = (
            "/home/lxj/文档/xwechat_files/wxid_jtl2a6uum2jk22_2c72/msg/file/2026-06/robocup-fine(2).engine"
        )
        self.d455_soccer_lab_detection_onnx_path = str(SOCCER_LAB_FOOTBALL_DETECTION_ONNX)


@configclass
class G1FlatSoccerBlindEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Custom blind-zone range: the ball is invisible when (x, y) distance is outside [min, max].
        self.commands.motion.blind_distance_min_range = (0.2, 0.8)  # Minimum distance sampling range.
        self.commands.motion.blind_distance_max_range = (1.8, 2.5)  # Maximum distance sampling range.
        self.commands.motion.blind_dropout_prob = 0.3  # 30% chance to miss the ball while otherwise visible.
        self.commands.motion.target_destination_dropout_prob = 0.3  # 30% chance to miss the goal/destination.
        
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )

        # Keep the policy input dimension unchanged.  The destination/goal is
        # updated throughout the episode in robot-base coordinates, with random
        # frame dropout; on dropped frames the policy receives the previous
        # destination input.  The critic keeps the truth observations inherited
        # from G1FlatKickEnvCfg.
        self.observations.policy.target_destination_pos_local = ObsTerm(
            func=mdp.dropout_target_destination_pos_local,
            params={"command_name": "motion"},
        )


@configclass
class G1FlatSuperSoccerEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        self.observations.policy.motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        self.observations.policy.body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        self.observations.policy.body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)


        self.observations.critic.projected_gravity = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.motion_ref_ang_vel = ObsTerm(func=mdp.motion_anchor_ang_vel, params={"command_name": "motion"})




@configclass
class G1FlatSoccerStudentEnvCfg(G1FlatKickEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        student_obs = self.observations.policy.copy()
        student_obs.target_point_pos = ObsTerm(
            func=mdp.target_point_pos_first_frame,
            params={"command_name": "motion"},
        )
        self.observations.StudentPolicyCfg = student_obs

        student_obs.target_destination_pos_local = ObsTerm(
            func=mdp.target_destination_pos_local_first_frame,
            params={"command_name": "motion"},
        )
