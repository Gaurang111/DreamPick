"""A simple scripted policy that steers the gripper toward the cube should
earn more reach reward than doing nothing at all. This is the cheapest
possible sanity check that the reward signal actually rewards the right
behavior, without needing a trained agent."""

import numpy as np

# A joint configuration that puts the gripper tip right at the cube (found via
# a quick local kinematic search while building the scene - see assets/scene_pick_cube.xml
# for the cube's position). Used only as a waypoint to steer toward; the test
# only needs this to be meaningfully closer to the cube than the env's home pose.
_NEAR_CUBE_QPOS = np.array([0.0, -0.59337094, 1.21394148, 0.39050609, 0.0, 1.6])
# Steering all the way there drives the gripper into the cube and shoves it (which
# the drag penalty now charges for); halfway there is a clean approach.
_HOME_QPOS = np.array([0.0, -0.6, 1.0, 0.4, 0.0, 0.4])
_APPROACH_QPOS = _HOME_QPOS + 0.5 * (_NEAR_CUBE_QPOS - _HOME_QPOS)

STEPS = 20

# The waypoint above is tied to the original fixed cube spot, so pin the cube there.
FIXED_CUBE = dict(cube_x_range=(0.22, 0.22), cube_y_range=(0.0, 0.0), randomize_cube_yaw=False)


def test_scripted_reach_beats_staying_still(make_env):
    env = make_env(reward_stage="reach", reset_noise=0.0, **FIXED_CUBE)

    # Baseline: never move the position target.
    env.reset(seed=0)
    still_total_reward = 0.0
    for _ in range(STEPS):
        _, reward, *_ = env.step(np.zeros(6, dtype=np.float32))
        still_total_reward += reward

    # Scripted: each step, move the position target one max_delta closer to a
    # pose known to put the gripper at the cube - via the real action API.
    env.reset(seed=0)
    move_total_reward = 0.0
    for _ in range(STEPS):
        direction = np.clip((_APPROACH_QPOS - env._ctrl_target) / env.max_delta, -1.0, 1.0)
        _, reward, *_ = env.step(direction.astype(np.float32))
        move_total_reward += reward

    assert move_total_reward > still_total_reward


def test_scripted_reach_decreases_gripper_cube_distance(make_env):
    env = make_env(reward_stage="reach", reset_noise=0.0, **FIXED_CUBE)
    _, info_start = env.reset(seed=0)

    info_end = info_start
    for _ in range(STEPS):
        direction = np.clip((_APPROACH_QPOS - env._ctrl_target) / env.max_delta, -1.0, 1.0)
        _, _, _, _, info_end = env.step(direction.astype(np.float32))

    assert info_end["gripper_cube_distance"] < info_start["gripper_cube_distance"]


def test_shoving_the_cube_scores_worse_than_a_clean_approach(make_env):
    """Driving the gripper straight into the cube pushes it in toward the arm. That
    must cost more than the closer approach earns - the earlier policies all learned
    to drag the cube in rather than lift it."""
    def total(target):
        env = make_env(reward_stage="reach", reset_noise=0.0, **FIXED_CUBE)
        env.reset(seed=0)
        tot, worst_drag = 0.0, 0.0
        for _ in range(STEPS):
            direction = np.clip((target - env._ctrl_target) / env.max_delta, -1.0, 1.0)
            _, reward, *_, info = env.step(direction.astype(np.float32))
            tot += reward
            worst_drag = max(worst_drag, info["cube_drag"])
        return tot, worst_drag

    clean, clean_drag = total(_APPROACH_QPOS)
    shove, shove_drag = total(_NEAR_CUBE_QPOS)
    assert clean_drag < 0.02 < shove_drag
    assert shove < clean
