from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _default_soccer_lab_root() -> Path:
    # Keep the training package self-contained.  A full Soccer_Lab checkout can
    # still be selected explicitly with SOCCER_LAB_ROOT.
    return Path(__file__).resolve().parent / "soccer_lab_data"


SOCCER_LAB_ROOT = Path(os.environ.get("SOCCER_LAB_ROOT", _default_soccer_lab_root())).resolve()
SOCCER_LAB_ASSETS_DIR = SOCCER_LAB_ROOT / "assets"
SOCCER_LAB_BALL_USD = SOCCER_LAB_ASSETS_DIR / "ball_asset" / "ball.usd"
SOCCER_LAB_GOALPOST_USD = SOCCER_LAB_ASSETS_DIR / "goalpost.usd"
SOCCER_LAB_FOOTBALL_DETECTION_ONNX = SOCCER_LAB_ASSETS_DIR / "football_detection" / "weight.onnx"
SOCCER_LAB_BALL_RADIUS = 0.11

def _field_specs_candidates(root: Path) -> tuple[Path, ...]:
    return (
        root / "source" / "field_specs.py",
        root
        / "source"
        / "Soccer_Lab"
        / "Soccer_Lab"
        / "tasks"
        / "direct"
        / "soccer_lab_marl"
        / "field_specs.py",
    )


def _resolve_field_specs_path(root: Path) -> Path:
    for candidate in _field_specs_candidates(root):
        if candidate.is_file():
            return candidate
    return _field_specs_candidates(root)[0]


_FIELD_SPECS_PATH = _resolve_field_specs_path(SOCCER_LAB_ROOT)


def _load_field_specs() -> ModuleType:
    if not _FIELD_SPECS_PATH.is_file():
        candidates = "\n".join(f"- {path}" for path in _field_specs_candidates(SOCCER_LAB_ROOT))
        raise FileNotFoundError(
            "Soccer_Lab field_specs.py not found. Set SOCCER_LAB_ROOT or restore the embedded assets. "
            f"Checked:\n{candidates}"
        )

    module_name = "_humanoid_soccer_soccer_lab_field_specs"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, _FIELD_SPECS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Soccer_Lab field specs from {_FIELD_SPECS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_field_specs = _load_field_specs()

SoccerFieldConfig = _field_specs.SoccerFieldConfig
FieldLineSpec = _field_specs.FieldLineSpec
GoalPostSpec = _field_specs.GoalPostSpec
GoalAssetSpec = _field_specs.GoalAssetSpec
S_FIELD = _field_specs.S_FIELD
M_FIELD = _field_specs.M_FIELD
L_FIELD = _field_specs.L_FIELD
FIELD_PRESETS = _field_specs.FIELD_PRESETS
get_field_preset = _field_specs.get_field_preset
build_field_line_specs = _field_specs.build_field_line_specs
build_goal_post_specs = _field_specs.build_goal_post_specs
build_goal_asset_specs = _field_specs.build_goal_asset_specs
yaw_to_quat_wxyz = _field_specs.yaw_to_quat_wxyz

SOCCER_LAB_FIELD = get_field_preset("M")


def require_soccer_lab_assets() -> None:
    missing = [
        path
        for path in (
            SOCCER_LAB_BALL_USD,
            SOCCER_LAB_GOALPOST_USD,
            SOCCER_LAB_FOOTBALL_DETECTION_ONNX,
        )
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required Soccer_Lab asset(s):\n{formatted}")
