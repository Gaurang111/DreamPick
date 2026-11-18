"""reset(seed=...) must be reproducible."""

import numpy as np


def test_same_seed_gives_identical_reset_state(make_env):
    env = make_env(reward_stage="reach")

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    np.testing.assert_array_equal(obs1["joint_pos"], obs2["joint_pos"])
    np.testing.assert_array_equal(obs1["wrist_image"], obs2["wrist_image"])
    np.testing.assert_array_equal(obs1["overhead_image"], obs2["overhead_image"])
    assert info1 == info2


def test_different_seeds_give_different_reset_state(make_env):
    env = make_env(reward_stage="reach")

    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=2)

    assert not np.array_equal(obs1["joint_pos"], obs2["joint_pos"])


def test_same_seed_gives_identical_trajectory(make_env):
    """Same seed + same actions -> same states, not just the same reset."""
    env = make_env(reward_stage="reach")
    actions = [np.full(6, 0.3, dtype=np.float32), np.full(6, -0.2, dtype=np.float32)]

    env.reset(seed=42)
    for action in actions:
        obs1, reward1, *_ = env.step(action)

    env.reset(seed=42)
    for action in actions:
        obs2, reward2, *_ = env.step(action)

    np.testing.assert_allclose(obs1["joint_pos"], obs2["joint_pos"])
    assert reward1 == reward2
