"""The fixed camera must never lose the cube behind the arm, and must keep every
joint in frame. The first camera sat behind the arm on its centre line: the arm
hid the cube completely once the gripper reached it (0 pixels), which nothing in
the training metrics would ever have revealed. Positions were chosen by
measurement (scripts/camera_search.py) - this test keeps them honest."""

import math

import mujoco
import numpy as np

GRASP_ARM = np.radians([0.0, -4.9, 32.1, 62.9, 0.0])
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MIN_CUBE_PIXELS = 10      # of 4096; the chosen camera shows 12-15 in every situation measured
MIN_JOINT_MARGIN_PX = 2   # every joint axis at least this far inside the 64px image


def _setup(env, arm_deg, gripper, cube="pads", cube_xy=None):
    q = np.append(arm_deg, gripper)
    env.data.qpos[env._joint_qpos_adr] = q
    env.data.ctrl[:] = q
    mujoco.mj_forward(env.model, env.data)
    adr = env._cube_qpos_adr
    if cube == "pads":
        mid = 0.5 * (env.data.xpos[env._fixed_pad_body_id] + env.data.xpos[env._moving_pad_body_id])
        env.data.qpos[adr:adr + 3] = mid
    else:
        env.data.qpos[adr:adr + 3] = [cube_xy[0], cube_xy[1], env._cube_resting_z]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def _cube_pixels(env):
    r = mujoco.Renderer(env.model, 64, 64)
    r.enable_segmentation_rendering()
    r.update_scene(env.data, camera="overhead_cam")
    seg = r.render()
    r.close()
    geom = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
    return int((env.model.geom_bodyid[geom] == env._cube_body_id).sum())


def _joint_margin_px(env):
    cam = env.model.camera("overhead_cam").id
    mujoco.mj_camlight(env.model, env.data)
    R = env.data.cam_xmat[cam].reshape(3, 3)
    f = 32.0 / math.tan(math.radians(env.model.cam_fovy[cam]) / 2)
    pts = [env.data.xanchor[env.model.joint(j).id] for j in J] + [env.data.site_xpos[env._gripperframe_site_id], np.zeros(3)]
    worst = 99.0
    for p in pts:
        rel = p - env.data.cam_xpos[cam]
        depth = -rel @ R[:, 2]
        assert depth > 0.01, "a joint is behind the camera"
        px, py = 32 + f * (rel @ R[:, 0]) / depth, 32 - f * (rel @ R[:, 1]) / depth
        worst = min(worst, px, 63 - px, py, 63 - py)
    return worst


def test_cube_is_visible_while_it_is_gripped(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=0)
    _setup(env, GRASP_ARM, math.radians(10), cube="pads")
    assert _cube_pixels(env) >= MIN_CUBE_PIXELS


def test_cube_is_visible_at_every_corner_of_the_start_zone_with_the_arm_at_home(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=0)
    for x in env.cube_x_range:
        for y in env.cube_y_range:
            _setup(env, np.array([0.0, -0.6, 1.0, 0.4, 0.0]), 0.4, cube="fixed", cube_xy=(x, y))
            assert _cube_pixels(env) >= MIN_CUBE_PIXELS, (x, y)


def test_every_joint_stays_in_frame_at_home_and_while_gripping(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=0)
    _setup(env, np.array([0.0, -0.6, 1.0, 0.4, 0.0]), 0.4, cube="fixed", cube_xy=(0.25, 0.0))
    assert _joint_margin_px(env) >= MIN_JOINT_MARGIN_PX
    _setup(env, GRASP_ARM, math.radians(10), cube="pads")
    assert _joint_margin_px(env) >= MIN_JOINT_MARGIN_PX
