"""Observation shapes and dtypes match what the Dict space declares."""

import numpy as np


def test_reset_observation_matches_space(make_env):
    env = make_env(reward_stage="reach")
    obs, info = env.reset(seed=0)

    assert set(obs.keys()) == {"wrist_image", "overhead_image", "joint_pos"}
    assert env.observation_space.contains(obs)

    assert obs["wrist_image"].shape == (64, 64, 3)
    assert obs["wrist_image"].dtype == np.uint8
    assert obs["overhead_image"].shape == (64, 64, 3)
    assert obs["overhead_image"].dtype == np.uint8
    assert obs["joint_pos"].shape == (6,)
    assert obs["joint_pos"].dtype == np.float32


def test_step_observation_matches_space(make_env):
    env = make_env(reward_stage="reach")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_info_dict_has_required_keys(make_env):
    env = make_env(reward_stage="lift")
    obs, info = env.reset(seed=0)
    for key in ("gripper_cube_distance", "is_grasped", "cube_height", "is_success"):
        assert key in info

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    for key in ("gripper_cube_distance", "is_grasped", "cube_height", "is_success"):
        assert key in info
