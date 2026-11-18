"""Pytest setup shared by all test files.

The env package lives in src/so101_env but isn't pip-installed (plain venv,
no packaging step), so we add src/ to sys.path here once, before any test
file tries `from so101_env import ...`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from so101_env import SO101PickCubeEnv  # noqa: E402


@pytest.fixture
def make_env():
    """Factory fixture: env = make_env(reward_stage="lift"). Auto-closes
    every env it created at the end of the test, even on failure."""
    created = []

    def _make(**kwargs):
        env = SO101PickCubeEnv(**kwargs)
        created.append(env)
        return env

    yield _make

    for env in created:
        env.close()
