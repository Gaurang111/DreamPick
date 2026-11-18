"""Run 3 environment, metrics and scripted solver (src/so101_env/state_pick_env.py, metrics.py, expert.py)."""
import mujoco
import numpy as np
import pytest

from so101_env.expert import ScriptedExpert
from so101_env.metrics import HOLD_STEPS, episode_metrics
from so101_env.state_pick_env import GRIPPER_FORCE_LIMIT, PRESETS, REST_QPOS, SO101StatePickEnv


def test_observation_shapes_and_no_images():
    env = SO101StatePickEnv("lift", "easy")
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"proprio", "cube"}
    assert obs["proprio"].shape == (6,) and obs["cube"].shape == (5,)
    assert obs["proprio"].dtype == np.float32 and obs["cube"].dtype == np.float32


def test_commanded_speed_and_acceleration_are_limited():
    env = SO101StatePickEnv("lift", "easy", v_max=0.9, a_max=6.0)
    env.reset(seed=1)
    prev, last_v = env._target.copy(), np.zeros(6)
    rng = np.random.default_rng(0)
    for _ in range(60):
        env.step(rng.choice([-1.0, 1.0], size=6))          # the harshest input: full-scale flips every step
        v = (env._target - prev) / env.dt
        assert np.all(np.abs(v) <= 0.9 + 1e-9)
        at_stop = (np.isclose(env._target, env._low) | np.isclose(env._target, env._high) |
                   np.isclose(prev, env._low) | np.isclose(prev, env._high))     # a joint on its hard stop just stops
        assert np.all((np.abs(v - last_v) / env.dt <= 6.0 + 1e-6) | at_stop)
        prev, last_v = env._target.copy(), v


def test_targets_stay_inside_the_real_arm_limits():
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=2)
    for _ in range(300):
        env.step(np.ones(6))
    assert np.all(env._target <= env._high + 1e-9)
    assert np.degrees(env._high[3]) <= 65.9 + 1e-6           # wrist_flex: the encoder-rollover limit
    for _ in range(300):
        env.step(-np.ones(6))
    assert np.all(env._target >= env._low - 1e-9)


def test_easy_has_exact_observations_and_nominal_physics():
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=3)
    assert env.params["kp"] == 1.0 and env.params["delay"] == 0
    assert np.allclose(env._observe()["proprio"], env.data.qpos[env._adr].astype(np.float32))
    assert env._observe()["cube"][4] == 1.0                   # always visible


def test_full_difficulty_randomises_physics_calibration_and_delay():
    env = SO101StatePickEnv("lift", "full")
    seen = {"kp": set(), "delay": set()}
    for s in range(40):
        env.reset(seed=s)
        seen["kp"].add(round(env.params["kp"], 3)); seen["delay"].add(env.params["delay"])
        assert 0.3 <= env.params["kp"] <= 1.0 and 0.6 <= env.params["friction"] <= 1.5
        assert env.params["joint_offset_deg"] <= 3.0 + 1e-6 and env.params["cam_offset_mm"] <= 10.0 + 1e-6
    assert len(seen["kp"]) > 30 and seen["delay"] == {0, 1, 2}


def test_physics_are_restored_when_the_next_episode_is_easy():
    env = SO101StatePickEnv("lift", "full")
    env.reset(seed=4)
    env.diff = PRESETS["easy"]
    env.reset(seed=5)
    assert env.params["kp"] == 1.0
    assert env.model.actuator_forcerange[5, 1] == pytest.approx(GRIPPER_FORCE_LIMIT)


def test_cube_zone_per_difficulty():
    for diff in ("easy", "wide", "full"):
        env = SO101StatePickEnv("lift", diff)
        d = env.diff
        for s in range(30):
            env.reset(seed=s)
            x, y = env.params["x"], env.params["y"]
            assert d.cube_x[0] <= x <= d.cube_x[1] and d.cube_y[0] <= y <= d.cube_y[1]
    assert PRESETS["wide"].cube_x[0] >= 0.21                  # the real arm cannot lift a cube closer than ~20 cm


def test_start_pose_covers_rest_to_home_at_wide_difficulty():
    env = SO101StatePickEnv("lift", "wide")
    lifts = []
    for s in range(60):
        env.reset(seed=s)
        lifts.append(env.data.qpos[env._adr][1])
    assert min(lifts) < -1.2 and max(lifts) > -0.8            # folded ... home (shoulder_lift)
    assert REST_QPOS[1] < -1.5


def test_cube_is_hidden_and_its_pose_frozen_while_the_gripper_is_over_it():
    env = SO101StatePickEnv("lift", "wide")
    env.reset(seed=6)
    env.data.qpos[env._adr] = np.array([0.0, -0.3, 0.6, 0.3, 0.0, 0.4])      # pads up in the air
    mujoco.mj_forward(env.model, env.data)
    seen = env._observe()["cube"].copy()
    assert np.linalg.norm(env.pad_mid()[:2] - env.cube_pos()[:2]) > 0.05      # far from the cube: visible, updating
    pm = env.pad_mid()
    env.data.qpos[env._cube_adr:env._cube_adr + 2] = pm[:2]                   # the cube is now right under the pads
    mujoco.mj_forward(env.model, env.data)
    assert np.linalg.norm(env.pad_mid()[:2] - env.cube_pos()[:2]) <= 0.05
    held = env._observe()["cube"]
    assert held[4] == 1.0                                                     # no flag: the policy is not told it is hidden
    assert np.array_equal(env._observe()["cube"], held)                       # pose frozen at the last visible value


def test_reward_stages_add_up_and_penalise_table_slams():
    r = {}
    for stage in ("reach", "grasp", "lift"):
        env = SO101StatePickEnv(stage, "easy")
        env.reset(seed=7)
        info = env._info()
        r[stage] = env._reward(info)
    assert r["reach"] == pytest.approx(r["grasp"]) and r["lift"] == pytest.approx(r["grasp"])   # nothing grasped or lifted yet
    env = SO101StatePickEnv("lift", "easy")
    env.reset(seed=7)
    info = env._info()
    base = env._reward(info)
    info["table_force"] = 60.0
    assert env._reward(info) == pytest.approx(base - 2.0)


def test_same_seed_same_episode():
    a, b = SO101StatePickEnv("lift", "full"), SO101StatePickEnv("lift", "full")
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.allclose(oa["proprio"], ob["proprio"]) and np.allclose(oa["cube"], ob["cube"])
    for _ in range(20):
        oa, ra, *_ = a.step(np.full(6, 0.3)); ob, rb, *_ = b.step(np.full(6, 0.3))
    assert np.allclose(oa["proprio"], ob["proprio"]) and ra == pytest.approx(rb)


def test_rest_pose_is_safe_on_the_table():
    env = SO101StatePickEnv("lift", "wide")
    env.reset(seed=0)
    q = np.clip(REST_QPOS, env._low, env._high)
    env.data.qpos[env._adr] = q
    env._target = q.copy()
    env.data.ctrl[:] = q
    worst = 0.0
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(6))
        worst = max(worst, info["table_force"])
    assert worst < 1.0


def test_curriculum_presets_add_one_difficulty_at_a_time():
    ex, nz, wd = PRESETS["wide_exact"], PRESETS["wide_noisy"], PRESETS["wide"]
    assert not ex.obs_noise and not ex.occlusion and ex.start == "family" and ex.cube_x == wd.cube_x
    assert nz.obs_noise and not nz.occlusion and wd.obs_noise and wd.occlusion
    env = SO101StatePickEnv("lift", "wide_noisy")
    env.reset(seed=3)
    env.data.qpos[env._adr] = np.array([0.0, -0.3, 0.6, 0.3, 0.0, 0.4])
    for _ in range(10):
        env.step(np.zeros(6))
    assert env._observe()["cube"][4] == 1.0                   # never hidden without occlusion


# ---- metrics ----------------------------------------------------------------

def test_success_needs_a_three_second_hold_and_no_sideways_carry():
    h = np.zeros(300); h[50:50 + HOLD_STEPS] = 0.06
    assert episode_metrics(h, np.zeros(300))["success"]
    h2 = np.zeros(300); h2[50:50 + HOLD_STEPS - 1] = 0.06
    assert not episode_metrics(h2, np.zeros(300))["success"]
    sh = np.zeros(300); sh[60:100] = 0.08                        # carried 8 cm while held
    assert not episode_metrics(h, sh)["success"]
    late_drop = episode_metrics(h, np.zeros(300))
    assert not late_drop["held_at_end"] and late_drop["success"]  # held 3 s, dropped later: still a success, not "held at the end"


# ---- scripted solver --------------------------------------------------------

def test_scripted_solver_picks_the_cube_up_at_easy_and_wide():
    for diff in ("easy", "wide"):
        env = SO101StatePickEnv("lift", diff)
        expert = ScriptedExpert(env)
        wins = 0
        for s in range(6):
            env.reset(seed=800 + s)
            expert.reset()
            hs, sh, done = [], [], False
            while not done:
                _, _, _, done, info = env.step(expert.act())
                hs.append(info["cube_height"]); sh.append(info["cube_shift"])
            wins += episode_metrics(hs, sh)["success"]
        assert wins >= 5, f"{diff}: {wins}/6"
