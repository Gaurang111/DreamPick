"""Actions are deltas on a position target, which must stay within each
joint's real range from the MJCF no matter how extreme the action is."""

import numpy as np


def test_single_step_delta_matches_max_delta(make_env):
    env = make_env(reward_stage="reach", max_delta=0.05)
    env.reset(seed=0)
    start_target = env._ctrl_target.copy()

    action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
    env.step(action)

    expected = np.clip(start_target + action * 0.05, env._joint_low, env._joint_high)
    np.testing.assert_allclose(env._ctrl_target, expected, atol=1e-6)


def test_extreme_actions_stay_within_joint_limits(make_env):
    env = make_env(reward_stage="reach")
    env.reset(seed=0)

    for _ in range(50):
        env.step(np.ones(env.action_space.shape, dtype=np.float32))
    assert np.all(env._ctrl_target <= env._joint_high + 1e-6)
    assert np.all(env._ctrl_target >= env._joint_low - 1e-6)

    for _ in range(50):
        env.step(-np.ones(env.action_space.shape, dtype=np.float32))
    assert np.all(env._ctrl_target >= env._joint_low - 1e-6)
    assert np.all(env._ctrl_target <= env._joint_high + 1e-6)


def test_joint_limits_match_model_when_real_limits_are_off(make_env):
    env = make_env(reward_stage="reach", use_real_arm_limits=False)
    # Without the real-arm restriction, the clipping bounds are exactly the actuator's own
    # ctrlrange from the MJCF, not some hardcoded guess.
    np.testing.assert_array_equal(env._joint_low, env.model.actuator_ctrlrange[:, 0])
    np.testing.assert_array_equal(env._joint_high, env.model.actuator_ctrlrange[:, 1])


def test_default_limits_are_the_real_arms_usable_range(make_env):
    """Default: the sim only lets the policy use motions the real arm can follow.
    The real wrist and shoulder lose their top end to an encoder rollover (see
    scripts/sim_real_bridge.py), so these are tighter than the model's own limits."""
    env = make_env(reward_stage="reach")
    names = [env.model.actuator(i).name for i in range(env.n_joints)]
    high = dict(zip(names, np.degrees(env._joint_high)))
    low = dict(zip(names, np.degrees(env._joint_low)))
    assert high["wrist_flex"] == np.float64(65.9) or abs(high["wrist_flex"] - 65.9) < 1e-6
    assert abs(high["shoulder_lift"] - 79.1) < 1e-6
    assert abs(high["elbow_flex"] - 93.7) < 1e-6 and abs(low["elbow_flex"] + 93.7) < 1e-6
    # never wider than the model allows
    assert np.all(env._joint_low >= env.model.actuator_ctrlrange[:, 0] - 1e-9)
    assert np.all(env._joint_high <= env.model.actuator_ctrlrange[:, 1] + 1e-9)


def test_home_pose_is_inside_the_default_limits(make_env):
    from so101_env.pick_cube_env import _HOME_QPOS
    env = make_env(reward_stage="reach")
    assert np.all(_HOME_QPOS >= env._joint_low) and np.all(_HOME_QPOS <= env._joint_high)
