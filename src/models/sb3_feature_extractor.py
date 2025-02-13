from typing import Union, Optional

import torch
import torch.nn as nn
from gym.vector.utils import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.qrdqn.policies import QRDQNPolicy
import gymnasium as gym
from stable_baselines3.common.type_aliases import PyTorchObs
import numpy as np

class QRDQNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim=512):
        super().__init__(observation_space, feature_dim)

        self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
        self.feature_dim = feature_dim

        self.join_representation_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )
        self._internal_join_state = torch.zeros((feature_dim,))

        # Preprocess the entire join graph
        # AdaptiveMaxPool doesn't work
        self.result_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )
        self.test = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )
        # Preprocess the current join representation
        self.join_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        # Final feature output size (concatenated embeddings)
        self._features_dim = 2 * feature_dim

    def forward(self, observations):
        """
        observations:
            - result_embeddings: (batch_size, max_triples, feature_dim)
            - join_embedding: (batch_size, feature_dim)
            - joined: (batch_size, max_triples) - Binary mask
        """
        result_embeddings = observations["result_embeddings"]  # (B, max_triples, feature_dim)

        # TODO Hierarchical MLP application based on join order

        # Apply MLPs to preprocess features
        pooled = torch.mean(result_embeddings, dim=1)
        result_features = self.result_mlp(pooled)  # (B, feature_dim)


        return result_features

    def reset_join_state(self):
        self._internal_join_state = torch.zeros((self.feature_dim,))


class MaskableQRDQNPolicy(QRDQNPolicy):
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
        q_value_dist = self.quantile_net._predict(obs, deterministic=deterministic)
        print(q_value_dist.shape)

        return self.quantile_net._predict(obs, deterministic=deterministic)

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
            actions = self._predict(obs_tensor, deterministic=deterministic, action_masks=action_masks)
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


    class InternalStateResetCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self):
            done = self.locals["dones"]

            if done[0]:  # If episode ended
                print("End episode resetting state")
                # self.model.policy.features_extractor.reset_join_state()
                # print(f"Episode finished. Last reward: {rewards[0]}")
                # print(f"Final observation: {new_obs}")

            return True  # Continue training


if __name__ == "__main__":
    pass
