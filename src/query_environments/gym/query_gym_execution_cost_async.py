import asyncio
from collections import OrderedDict
from copy import deepcopy
from typing import List, Any, Union

import numpy as np
import gymnasium as gym
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs
from stable_baselines3.common.vec_env.util import obs_space_info, dict_to_obs


class AsyncBatchQueryEnv(VecEnv):
    """
    Batches multiple QueryGymExecutionCost environments and steps them asynchronously.
    This allows multiple queries to be sent to the endpoint
    """

    def __init__(self, envs: List[gym.Env]):
        super().__init__(num_envs=len(envs),
                         observation_space=envs[0].observation_space,
                         action_space=envs[0].action_space)
        self.envs = envs
        obs_space = envs[0].observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                                    for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        self.actions = None

    async def _step_async(self, env, action):
        """Run a single step in a threadpool (so blocking HTTP doesn’t block event loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, env.step, action)

    def step_wait(self) -> VecEnvStepReturn:
        async def _step_all():
            tasks = [self._step_async(self.envs[i], self.actions[i])
                     for i in range(self.num_envs)]
            return await asyncio.gather(*tasks)
        # results = asyncio.run(_step_all())
        obs, self.buf_rewards, dones, truncs, self.buf_infos = zip(*asyncio.run(_step_all()))
        # obs, rewards, dones, truncs, infos = zip(*results)
        for env_idx in range(self.num_envs):
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = dones[env_idx] or truncs[env_idx]
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncs[env_idx] and not dones[env_idx]

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs[env_idx]
                reset_obs, reset_info = self.envs[env_idx].reset()
                self.reset_infos[env_idx] = reset_info
                self._save_obs(env_idx, reset_obs)
            else:
                self._save_obs(env_idx, obs[env_idx])

        return self._obs_from_buf(), np.copy(self.buf_rewards), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self, **kwargs):
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()


    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def close(self) -> None:
        for env in self.envs:
            env.close()

def _stack_obs(obs_list: Union[list[VecEnvObs], tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Stack observations (convert from a list of single env obs to a stack of obs),
    depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: Concatenated observations.
            A NumPy array or a dict or tuple of stacked numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([_ensure_batch_dim(single_obs[key]) for single_obs in obs_list]) for key in
                space.spaces.keys()}  # type: ignore[call-overload]
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(
            np.stack([_ensure_batch_dim(single_obs[i]) for single_obs in obs_list]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs_list)  # type: ignore[arg-type]


def _ensure_batch_dim(obs):
    obs = np.asarray(obs)
    if obs.ndim == 0:
        obs = np.expand_dims(obs, axis=-1)
    return obs

