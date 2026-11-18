"""The grasp and lift rewards depend on detecting a real pinch: the cube only
touches the gripper through the two jaw pads (the jaw meshes' convex hulls are
excluded from cube contact), so detection must use the pads."""

import math

import mujoco
import numpy as np

# A straight-down grasp pose over the cube's spawn point (found by IK in
# scripts/sim_real_demo.py): pan, lift, elbow, wrist_flex, wrist_roll.
GRASP_ARM = np.radians([0.0, -4.9, 32.1, 62.9, 0.0])
OPEN, CLOSED = math.radians(40), math.radians(-10)


def _settle(env, secs):
    for _ in range(int(secs / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)


def _pose_with_cube_between_jaws(env):
    env.reset(seed=0)
    q = np.append(GRASP_ARM, OPEN)
    env.data.qpos[env._joint_qpos_adr] = q
    env.data.ctrl[:] = q
    mujoco.mj_forward(env.model, env.data)
    mid = 0.5 * (env.data.xpos[env._fixed_pad_body_id] + env.data.xpos[env._moving_pad_body_id])
    adr = env._cube_qpos_adr
    env.data.qpos[adr:adr + 3] = [mid[0], mid[1], env._cube_resting_z]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def test_open_jaws_around_cube_is_not_a_grasp(make_env):
    env = make_env(reward_stage="lift")
    _pose_with_cube_between_jaws(env)
    _settle(env, 0.5)
    assert not env._get_info()["is_grasped"]


def test_closing_on_cube_is_detected_as_grasp(make_env):
    env = make_env(reward_stage="lift")
    _pose_with_cube_between_jaws(env)
    env.data.ctrl[5] = CLOSED
    _settle(env, 1.0)
    info = env._get_info()
    assert info["is_grasped"]
    assert info["gripper_cube_distance"] < 0.015  # cube roughly centered between the pads


def test_grasped_cube_lifts_to_success(make_env):
    env = make_env(reward_stage="lift")
    _pose_with_cube_between_jaws(env)
    env.data.ctrl[5] = CLOSED
    _settle(env, 1.0)
    env.data.ctrl[1] = math.radians(-40)  # raise the shoulder with the cube held
    _settle(env, 1.5)
    info = env._get_info()
    assert info["is_grasped"]
    assert info["is_success"], f"cube only {info['cube_height'] * 100:.1f}cm up"
    assert env._compute_reward(info) > 3.0  # reach + grasp bonus + full lift bonus
