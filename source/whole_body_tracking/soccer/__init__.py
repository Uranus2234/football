"""
Python module serving as a project/extension template.
"""

# Register Gym environments when Isaac Lab/Isaac Sim is available.  Pure
# PyTorch utilities such as soccer.perception must stay importable without
# launching SimulationApp.
try:
    from .tasks import *  # noqa: F401, F403
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not (missing.startswith("omni") or missing.startswith("isaac")):
        raise
