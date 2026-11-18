"""The live controller must turn policy actions into joint targets exactly as the training environment does."""
import numpy as np

from so101_env.deploy import LiveController, observation
from so101_env.state_pick_env import SO101StatePickEnv


def test_controller_targets_match_the_environment_step_for_step():
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=21)
    ctl = LiveController.from_env(env)
    ctl.reset(env._target)
    rng = np.random.default_rng(3)
    for i in range(120):
        a = rng.uniform(-1, 1, 6) * (rng.random(6) < 0.7)              # includes many zeros and small values
        a[:2] *= 0.04 if i % 3 == 0 else 1.0                           # inside the dead band now and then
        env.step(a)
        assert np.allclose(ctl.step(a), env._target, atol=1e-12)


def test_dead_band_leaves_the_target_still():
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=1)
    ctl = LiveController.from_env(env)
    t0 = ctl.reset(env._target)
    for _ in range(40):
        t = ctl.step(np.full(6, 0.03))
    assert np.allclose(t, t0)


def test_observation_matches_the_environment_for_the_same_cube_pose():
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=5)
    cube = env.cube_pos()
    yaw = env.cube_yaw()
    mine = observation(env.data.qpos[env._adr], (cube[0] * 100, cube[1] * 100, np.degrees(yaw)))
    theirs = env._observe()
    assert np.allclose(mine["cube"], theirs["cube"], atol=1e-5)
    assert np.allclose(mine["proprio"], theirs["proprio"], atol=1e-6)
