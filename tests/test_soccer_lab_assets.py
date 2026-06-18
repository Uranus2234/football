from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "source" / "whole_body_tracking"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def test_soccer_lab_assets_and_m_field_are_available():
    require_soccer_lab_assets()

    assert SOCCER_LAB_BALL_USD.name == "ball.usd"
    assert SOCCER_LAB_BALL_USD.is_file()
    assert SOCCER_LAB_GOALPOST_USD.name == "goalpost.usd"
    assert SOCCER_LAB_GOALPOST_USD.is_file()
    assert SOCCER_LAB_FOOTBALL_DETECTION_ONNX.is_file()

    assert SOCCER_LAB_FIELD.field_length == 14.0
    assert SOCCER_LAB_FIELD.field_width == 9.0
    assert SOCCER_LAB_FIELD.goal_width == 2.4
    assert SOCCER_LAB_FIELD.goal_height == 1.8
    assert SOCCER_LAB_BALL_RADIUS == 0.11


def test_soccer_lab_field_specs_build_pitch_lines_and_goals():
    lines = build_field_line_specs(SOCCER_LAB_FIELD)
    posts = build_goal_post_specs(SOCCER_LAB_FIELD)
    goal_assets = build_goal_asset_specs(SOCCER_LAB_FIELD, z_offset=SOCCER_LAB_FIELD.goal_height * 0.5)

    assert len(lines) >= 20
    assert {post.name for post in posts} == {
        "left_post_1",
        "left_post_2",
        "left_crossbar",
        "right_post_1",
        "right_post_2",
        "right_crossbar",
    }
    assert [asset.name for asset in goal_assets] == ["left_goal_asset", "right_goal_asset"]
