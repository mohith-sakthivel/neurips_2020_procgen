from operator import imod
from gym.wrappers import FrameStack
import numpy
from ray.rllib.env.atari_wrappers import EpisodicLifeEnv
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import gym
from gym import spaces
import cv2
import numpy as np


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = env.observation_space.shape[0]
        self.height = env.observation_space.shape[1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :]


def wrap_deepmind(env, framestack=True):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """

    env = WarpFrame(env)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    if framestack:
        env = FrameStack(env, 4)
    return env


# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    # lambda config: FrameStack(ProcgenEnvWrapper(config), 4),
    lambda config: wrap_deepmind(ProcgenEnvWrapper(config), 4),
)
