from gymnasium.envs.registration import register

from .pick_cube_env import SO101PickCubeEnv

register(
    id="SO101PickCube-v0",
    entry_point="so101_env.pick_cube_env:SO101PickCubeEnv",
    max_episode_steps=None,  # the env truncates itself in step(); avoid double-counting
)

__all__ = ["SO101PickCubeEnv"]
