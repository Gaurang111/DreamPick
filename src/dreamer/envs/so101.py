"""Adapts our Gymnasium SO101PickCubeEnv to the raw env interface
dreamerv3-torch's tools.simulate()/Damy/Parallel expect: a plain
reset() -> obs dict and step(action) -> (obs, reward, done, info) 4-tuple
(the pre-Gymnasium "done" convention, not separate terminated/truncated),
with "is_first"/"is_terminal" baked into the obs dict - mirrors the pattern
used by upstream's envs/dmc.py.

wrist_image and overhead_image pass straight through as two separate keys
(both naturally 64x64, square) rather than being combined into one wide
image - ConvDecoder's upsampling stage count is derived from height alone
(networks.py), so a non-square combined image silently produces the wrong
output shape. Two square keys avoid that entirely; models.py/tools.py were
adjusted in a few spots that hardcoded a single "image" key name to instead
handle any "*_image" key (see comments there).
"""

import datetime
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from so101_env import SO101PickCubeEnv  # noqa: E402


class SO101Dreamer:
    def __init__(self, reward_stage="reach", seed=0):
        # Never end an episode on success during training: every step pays
        # positive reward, so ending the episode at "cube > 5cm" would cut off
        # that reward and teach the arm to hover just below 5cm. The lift bonus
        # already caps at 5cm, so lifting to 5cm and holding scores the most.
        self._env = SO101PickCubeEnv(reward_stage=reward_stage, terminate_on_success=False)
        self._seed = seed
        self._seeded = False
        self._new_id()

    def _new_id(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{uuid.uuid4().hex}"

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._new_id()
        if not self._seeded:
            obs, info = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            # No explicit seed: Gymnasium's internal RNG keeps advancing, so
            # each episode gets a different (but overall-reproducible, since
            # it all derives from the one seed above) starting perturbation.
            obs, info = self._env.reset()
        obs = dict(obs)
        obs["is_first"] = True
        obs["is_terminal"] = False
        return obs

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = dict(obs)
        obs["is_first"] = False
        obs["is_terminal"] = terminated  # not truncated - a time limit isn't a true terminal state
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        self._env.close()
