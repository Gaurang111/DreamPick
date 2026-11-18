"""Dreamer-facing wrapper for SO101StatePickEnv (state observations, no images). Same interface as envs/so101.py:
reset() -> obs dict and step(action) -> (obs, reward, done, info) with is_first / is_terminal added."""
import datetime
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402


class SO101StateDreamer:
    def __init__(self, reward_stage="lift", difficulty="easy", seed=0, max_episode_steps=300):
        self._env = SO101StatePickEnv(reward_stage=reward_stage, difficulty=difficulty, max_episode_steps=max_episode_steps)
        self._seed = seed
        self._seeded = False
        self._new_id()

    def _new_id(self):
        self.id = f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}"

    @property
    def sim(self):
        """The underlying SO101StatePickEnv (the scripted solver needs the true state)."""
        return self._env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._new_id()
        if not self._seeded:
            obs, _ = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs, _ = self._env.reset()
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
        obs["is_terminal"] = terminated      # not truncated: a time limit is not a true terminal state
        return obs, reward, terminated or truncated, info

    def close(self):
        pass
