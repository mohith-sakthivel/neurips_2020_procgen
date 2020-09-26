from envs.wrappers import CustomFrameStack
from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: CustomFrameStack(ProcgenEnvWrapper(config), 3),
)