"""The env is importable via gymnasium.make and passes the official checker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import so101_env  # noqa: F401  (registers SO101PickCube-v0)


def test_gym_make_creates_env():
    env = gym.make("SO101PickCube-v0", reward_stage="reach")
    obs, info = env.reset(seed=0)
    assert "joint_pos" in obs
    env.close()


def test_check_env_passes_with_no_warnings():
    env = gym.make("SO101PickCube-v0", reward_stage="lift", render_mode="rgb_array")
    check_env(env.unwrapped, skip_render_check=False)
    env.close()
