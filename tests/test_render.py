"""render_mode="rgb_array" returns a real image; close() releases resources
cleanly without raising."""

import numpy as np


def test_rgb_array_render_returns_image(make_env):
    env = make_env(reward_stage="reach", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_close_does_not_raise(make_env):
    env = make_env(reward_stage="reach", render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    env.close()  # should not raise; conftest's fixture also calls this, so
    # calling it again here just confirms it's idempotent-safe for this test.
