"""The cube starts at a random spot (and rotation) inside a zone the real arm
can actually lift from, instead of always in the same place."""

import numpy as np


def _cube_pose(env):
    adr = env._cube_qpos_adr
    q = env.data.qpos[adr:adr + 7]
    yaw = 2.0 * np.arctan2(q[6], q[3])
    return q[0], q[1], q[2], yaw


def test_cube_starts_inside_the_zone(make_env):
    env = make_env(reward_stage="reach")
    for seed in range(40):
        env.reset(seed=seed)
        x, y, z, _ = _cube_pose(env)
        assert env.cube_x_range[0] <= x <= env.cube_x_range[1]
        assert env.cube_y_range[0] <= y <= env.cube_y_range[1]
        assert z == env._cube_resting_z


def test_default_zone_is_where_the_real_arm_can_lift(make_env):
    # Measured in docs/training_run_1/README.md section 11: lifting 5cm needs the cube at 20cm+ out
    # (comfortably from 21cm), and the zone was sized to the arm's reach.
    env = make_env(reward_stage="reach")
    assert env.cube_x_range[0] >= 0.21
    assert env.cube_x_range[1] <= 0.30
    assert max(abs(v) for v in env.cube_y_range) <= 0.05


def test_cube_start_actually_varies(make_env):
    env = make_env(reward_stage="reach")
    poses = []
    for seed in range(30):
        env.reset(seed=seed)
        poses.append(_cube_pose(env))
    xs, ys, _, yaws = zip(*poses)
    assert np.std(xs) > 0.015 and np.std(ys) > 0.015
    yaws = np.mod(yaws, 2 * np.pi)
    assert np.std(yaws) > 0.8  # spread around the full circle, not a narrow band


def test_same_seed_gives_the_same_cube(make_env):
    env = make_env(reward_stage="reach")
    env.reset(seed=11)
    first = _cube_pose(env)
    env.reset(seed=11)
    assert _cube_pose(env) == first


def test_zone_can_be_pinned_to_one_spot(make_env):
    env = make_env(reward_stage="reach", cube_x_range=(0.22, 0.22), cube_y_range=(0.0, 0.0), randomize_cube_yaw=False)
    for seed in range(5):
        env.reset(seed=seed)
        x, y, z, yaw = _cube_pose(env)
        assert (x, y, z) == (0.22, 0.0, env._cube_resting_z)
        assert abs(yaw) < 1e-9


def test_a_fresh_episode_has_no_drag_and_no_held_time(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=1)
    for _ in range(5):
        env.step(np.zeros(6, dtype=np.float32))
    env.reset(seed=2)
    info = env._get_info()
    assert info["cube_drag"] == 0.0 and info["held_steps"] == 0
