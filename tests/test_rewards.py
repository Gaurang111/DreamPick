"""Each reward stage produces sensible values: reach reward falls off with
distance, grasp adds a bonus on top of reach, lift adds a height bonus on
top of grasp."""

import numpy as np
import pytest


def info(distance=0.1, grasped=False, height=0.0, success=False, drag=0.0):
    return {
        "gripper_cube_distance": distance,
        "is_grasped": grasped,
        "cube_height": height,
        "is_success": success,
        "cube_drag": drag,
    }


def test_reach_reward_is_maximal_at_zero_distance(make_env):
    env = make_env(reward_stage="reach")
    assert env._compute_reward(info(distance=0.0)) == pytest.approx(1.0)


def test_reach_reward_decreases_with_distance(make_env):
    env = make_env(reward_stage="reach")
    near = env._compute_reward(info(distance=0.01))
    far = env._compute_reward(info(distance=0.3))
    assert near > far


def test_reach_reward_is_bounded(make_env):
    env = make_env(reward_stage="reach")
    for d in (0.0, 0.05, 0.5, 2.0):
        r = env._compute_reward(info(distance=d))
        assert 0.0 <= r <= 1.0


def test_grasp_stage_adds_bonus_over_reach_alone(make_env):
    env = make_env(reward_stage="grasp")
    not_grasped = env._compute_reward(info(distance=0.05, grasped=False))
    grasped = env._compute_reward(info(distance=0.05, grasped=True))
    assert grasped > not_grasped
    assert grasped == pytest.approx(not_grasped + 1.0)


def test_lift_stage_adds_height_bonus_on_top_of_grasp(make_env):
    env = make_env(reward_stage="lift", success_height=0.05)
    grasped_no_lift = env._compute_reward(info(distance=0.05, grasped=True, height=0.0))
    grasped_lifted = env._compute_reward(info(distance=0.05, grasped=True, height=0.05))
    assert grasped_lifted > grasped_no_lift


def test_lift_height_bonus_saturates_at_success_height(make_env):
    env = make_env(reward_stage="lift", success_height=0.05)
    at_target = env._compute_reward(info(distance=0.05, grasped=True, height=0.05))
    above_target = env._compute_reward(info(distance=0.05, grasped=True, height=0.20))
    # cube_height is clipped at success_height's worth of bonus - lifting
    # further shouldn't keep paying out indefinitely.
    assert at_target == pytest.approx(above_target)


def test_reward_stage_hierarchy_reach_leq_grasp_leq_lift(make_env):
    """At the same underlying state, a later stage's reward should never be
    less than an earlier stage's, since each stage adds a non-negative bonus
    on top of the one before it."""
    reach_env = make_env(reward_stage="reach")
    grasp_env = make_env(reward_stage="grasp")
    lift_env = make_env(reward_stage="lift")

    state = info(distance=0.05, grasped=True, height=0.03)
    reach_r = reach_env._compute_reward(state)
    grasp_r = grasp_env._compute_reward(state)
    lift_r = lift_env._compute_reward(state)

    assert reach_r <= grasp_r <= lift_r


def test_terminate_on_success_defaults_to_lift_stage_only(make_env):
    reach_env = make_env(reward_stage="reach")
    grasp_env = make_env(reward_stage="grasp")
    lift_env = make_env(reward_stage="lift")
    assert reach_env.terminate_on_success is False
    assert grasp_env.terminate_on_success is False
    assert lift_env.terminate_on_success is True


def test_terminate_on_success_can_be_overridden(make_env):
    env = make_env(reward_stage="reach", terminate_on_success=True)
    assert env.terminate_on_success is True


def test_lift_pays_2_per_cm_up_to_the_target(make_env):
    env = make_env(reward_stage="lift", success_height=0.05)
    base = env._compute_reward(info(distance=0.05, grasped=True, height=0.0))
    one_cm = env._compute_reward(info(distance=0.05, grasped=True, height=0.01))
    assert one_cm - base == pytest.approx(2.0)


def test_holding_at_the_target_height_earns_a_bonus_on_top(make_env):
    env = make_env(reward_stage="lift", success_height=0.05)
    just_below = env._compute_reward(info(distance=0.05, grasped=True, height=0.0499))
    at_target = env._compute_reward(info(distance=0.05, grasped=True, height=0.05))
    assert at_target - just_below > 3.9  # the 4.0 hold bonus (plus a hair of height)


def test_dragging_the_cube_in_is_penalised_in_every_stage(make_env):
    for stage in ("reach", "grasp", "lift"):
        env = make_env(reward_stage=stage)
        clean = env._compute_reward(info(distance=0.05, grasped=True, height=0.0))
        dragged = env._compute_reward(info(distance=0.05, grasped=True, height=0.0, drag=0.10))
        assert dragged < clean - 2.9, stage  # full 3.0 penalty at 8cm and beyond


def test_small_inward_movement_is_free(make_env):
    env = make_env(reward_stage="lift")
    clean = env._compute_reward(info(distance=0.05, grasped=True, height=0.02))
    nudged = env._compute_reward(info(distance=0.05, grasped=True, height=0.02, drag=0.02))
    assert nudged == pytest.approx(clean)


def test_dragging_costs_more_than_a_pinch_is_worth(make_env):
    """The earlier policies dragged the cube in and held it for the reach and grasp
    reward. Now that trade must lose: a fully dragged, pinched cube scores worse than
    an untouched cube a few centimetres from the jaws."""
    env = make_env(reward_stage="grasp")
    dragged_and_pinched = env._compute_reward(info(distance=0.0, grasped=True, drag=0.12))
    hovering_untouched = env._compute_reward(info(distance=0.03, grasped=False, drag=0.0))
    assert dragged_and_pinched < hovering_untouched


def test_drag_is_measured_from_where_the_cube_started(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=3)
    assert env._get_info()["cube_drag"] == pytest.approx(0.0)
    adr = env._cube_qpos_adr
    start_x, start_y = env.data.qpos[adr], env.data.qpos[adr + 1]
    r = float(np.hypot(start_x, start_y))
    env.data.qpos[adr] = start_x * (r - 0.05) / r      # pull it 5cm toward the base
    env.data.qpos[adr + 1] = start_y * (r - 0.05) / r
    mujoco_forward(env)
    assert env._get_info()["cube_drag"] == pytest.approx(0.05, abs=1e-6)
    env.data.qpos[adr] = start_x * (r + 0.05) / r      # pushing it OUT is not dragging
    env.data.qpos[adr + 1] = start_y * (r + 0.05) / r
    mujoco_forward(env)
    assert env._get_info()["cube_drag"] == 0.0


def mujoco_forward(env):
    import mujoco
    mujoco.mj_forward(env.model, env.data)


def shifted(shift, **kw):
    d = info(**kw)
    d["cube_shift"] = shift
    return d


def test_carrying_the_cube_sideways_is_penalised_in_the_lift_stage(make_env):
    """Run 2's first lift policy raised the cube 2.4cm and carried it ~20cm sideways. That must now
    score below simply pinching the cube where it is, and below the same lift done straight up."""
    env = make_env(reward_stage="lift")
    pinch_in_place = env._compute_reward(shifted(0.0, distance=0.0, grasped=True, height=0.0))
    straight_lift = env._compute_reward(shifted(0.0, distance=0.0, grasped=True, height=0.024))
    sideways_lift = env._compute_reward(shifted(0.20, distance=0.0, grasped=True, height=0.024))
    assert sideways_lift < pinch_in_place < straight_lift


def test_small_sideways_movement_is_free(make_env):
    env = make_env(reward_stage="lift")
    still = env._compute_reward(shifted(0.0, distance=0.0, grasped=True, height=0.03))
    nudged = env._compute_reward(shifted(0.03, distance=0.0, grasped=True, height=0.03))
    assert nudged == pytest.approx(still)


def test_sideways_penalty_is_only_for_the_lift_stage(make_env):
    # reach and grasp were trained and tested with the original reward; they must be unchanged
    for stage in ("reach", "grasp"):
        env = make_env(reward_stage=stage)
        a = env._compute_reward(shifted(0.0, distance=0.0, grasped=True, height=0.0))
        b = env._compute_reward(shifted(0.20, distance=0.0, grasped=True, height=0.0))
        assert a == pytest.approx(b), stage


def test_the_lift_penalty_is_the_larger_of_drag_and_shift_not_their_sum(make_env):
    env = make_env(reward_stage="lift")
    clean = env._compute_reward(shifted(0.0, distance=0.0, grasped=True, height=0.0))
    both = env._compute_reward(dict(shifted(0.20, distance=0.0, grasped=True, height=0.0), cube_drag=0.20))
    assert both == pytest.approx(clean - 5.0)  # the shift penalty (5.0) dominates the drag penalty (3.0)


def test_cube_shift_is_reported_in_info_and_measures_horizontal_movement(make_env):
    env = make_env(reward_stage="lift")
    env.reset(seed=2)
    assert env._get_info()["cube_shift"] == pytest.approx(0.0)
    adr = env._cube_qpos_adr
    env.data.qpos[adr] += 0.06
    env.data.qpos[adr + 1] += 0.08
    env.data.qpos[adr + 2] += 0.03  # going up is not a horizontal shift
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    assert env._get_info()["cube_shift"] == pytest.approx(0.10, abs=1e-6)


def smooth(change=0.0, open_speed=0.0, in_air=False, **kw):
    d = info(**kw)
    d.update(action_change=change, gripper_open_speed=open_speed, was_in_air=in_air)
    return d


@pytest.fixture
def penalties_on(monkeypatch):
    """The smoothness and slip terms are switched off in the env (the retrain that used them was abandoned);
    these tests turn them on to check the mechanism still works."""
    import so101_env.pick_cube_env as pce
    monkeypatch.setattr(pce, "_SMOOTH_PENALTY", 2.0)
    monkeypatch.setattr(pce, "_SLIP_PENALTY", 5.0)


def test_the_smoothness_and_slip_penalties_are_off_by_default(make_env):
    env = make_env(reward_stage="lift")
    a = env._compute_reward(smooth(distance=0.0, grasped=True, height=0.03))
    b = env._compute_reward(smooth(change=2.0, open_speed=0.05, in_air=True, distance=0.0, grasped=True, height=0.03))
    assert a == pytest.approx(b)


def test_jerky_actions_cost_reward_in_the_lift_stage(make_env, penalties_on):
    env = make_env(reward_stage="lift")
    calm = env._compute_reward(smooth(change=0.0, distance=0.0, grasped=True, height=0.03))
    jerky = env._compute_reward(smooth(change=0.8, distance=0.0, grasped=True, height=0.03))
    assert jerky == pytest.approx(calm - 1.6)  # weight 2.0 x mean |change in action|


def test_opening_the_gripper_in_the_air_is_penalised_but_not_on_the_table(make_env, penalties_on):
    env = make_env(reward_stage="lift")
    base = env._compute_reward(smooth(distance=0.0, grasped=True, height=0.03))
    slipping = env._compute_reward(smooth(open_speed=0.02, in_air=True, distance=0.0, grasped=True, height=0.03))
    on_table = env._compute_reward(smooth(open_speed=0.02, in_air=False, distance=0.0, grasped=True, height=0.03))
    assert slipping == pytest.approx(base - 5.0)
    assert on_table == pytest.approx(base)


def test_smoothness_and_slip_penalties_are_lift_stage_only(make_env, penalties_on):
    for stage in ("reach", "grasp"):
        env = make_env(reward_stage=stage)
        a = env._compute_reward(smooth(distance=0.0, grasped=True, height=0.0))
        b = env._compute_reward(smooth(change=2.0, open_speed=0.05, in_air=True, distance=0.0, grasped=True, height=0.0))
        assert a == pytest.approx(b), stage


def test_step_reports_action_change_and_gripper_opening(make_env):
    import numpy as np
    env = make_env(reward_stage="lift")
    env.reset(seed=3)
    zeros = np.zeros(6)
    _, _, _, _, i0 = env.step(zeros)
    assert i0["action_change"] == pytest.approx(0.0)
    flip = np.array([1, -1, 1, -1, 1, 1.0])
    _, _, _, _, i1 = env.step(flip)
    assert i1["action_change"] == pytest.approx(1.0)       # |1 - 0| averaged over 6 joints
    _, _, _, _, i2 = env.step(-flip)
    assert i2["action_change"] == pytest.approx(2.0)       # every joint flips sign
    env.reset(seed=3)
    for _ in range(10):
        env.step(np.array([0, 0, 0, 0, 0, 1.0]))
    _, _, _, _, opening = env.step(np.array([0, 0, 0, 0, 0, 1.0]))
    assert opening["gripper_open_speed"] > 0.0
    for _ in range(40):
        _, _, _, _, closing = env.step(np.array([0, 0, 0, 0, 0, -1.0]))
    assert closing["gripper_open_speed"] == 0.0
