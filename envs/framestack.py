from operator import imod
from gym.wrappers import FrameStack
from gym.wrappers import frame_stack
import numpy
from ray.rllib.env.atari_wrappers import EpisodicLifeEnv
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import gym
from gym import spaces
import cv2
import numpy as np


class CustomWarp(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to the specified size (dim x dim)."""
        gym.ObservationWrapper.__init__(self, env)
        # self.width = env.observation_space.shape[0]
        # self.height = env.observation_space.shape[1]
        self.width = env.observation_space.shape[1]
        self.height = env.observation_space.shape[2]
        self.channels = env.observation_space.shape[0] * env.observation_space.shape[3]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, self.channels),
            dtype=np.uint8)

    def observation(self, frame):
        frame = np.moveaxis(frame, [0, 1, 2, 3], [3, 0, 1, 2])
        frame = np.reshape(frame, (self.height, self.width, self.channels), order='F')

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


def CustomStacking(env):
    """Configure environment for DeepMind-style Atari with color

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        dim (int): Dimension to resize observations to (dim x dim).
        framestack (bool): Whether to framestack observations.
    """
    num_frames=3
    # env = CustomWarp(env, num_frames)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    env = FrameStack(env, num_frames)
    env = CustomWarp(env)
    return env


# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    # lambda config: FrameStack(ProcgenEnvWrapper(config), 4),
    lambda config: CustomStacking(ProcgenEnvWrapper(config)),
)
