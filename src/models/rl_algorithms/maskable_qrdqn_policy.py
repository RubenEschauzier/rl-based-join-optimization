from typing import Union, Optional

import numpy as np
import torch
from gym.vector.utils import spaces
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs

from src.models.rl_algorithms.maskable_quantile_network import MaskableQuantileNetwork


class MaskableQRDQNPolicy(QRDQNPolicy):
    quantile_net: MaskableQuantileNetwork
    quantile_net_target: MaskableQuantileNetwork

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete,
                 lr_schedule: Schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_quantile_net(self) -> MaskableQuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskableQuantileNetwork(**net_args).to(self.device)

    def forward(self, obs, deterministic=True, action_masks=None):
        """
        Q-learning policy forward pass that masks out invalid actions.
        :param obs:
        :param deterministic:
        :param action_masks:

        Returns:
            torch.Tensor: Masked Q-values.
        """
        # Get standard QR-DQN output
        q_values = self.q_net(obs)  # Shape: (batch_size, n_quantiles, action_dim)
        # Extract the mask from observation
        joined_mask = obs["joined"]  # Shape: (batch_size, max_triples)
        valid_mask = 1 - joined_mask  # Invert mask (1 for valid, 0 for invalid)

        # Compute mean Q-values across quantiles
        q_values_mean = q_values.mean(dim=1)  # Shape: (batch_size, action_dim)

        # Apply mask: Set Q-values for invalid actions to -inf
        q_values_mean[valid_mask == 0] = float("-inf")

        # Select action using masked Q-values
        action = torch.argmax(q_values_mean, dim=1)

        return action

    def _predict(self, obs: PyTorchObs, deterministic: bool = True, action_masks=None) -> torch.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic, action_masks=action_masks)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[tuple[np.ndarray, ...]] = None
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param action_masks:
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this corresponds to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic,
                                    action_masks=torch.tensor(action_masks, device=self.device).to(torch.bool))

        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]
