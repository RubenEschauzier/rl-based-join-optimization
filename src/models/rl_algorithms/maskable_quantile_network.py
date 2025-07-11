from typing import Union

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.qrdqn.policies import QuantileNetwork
from stable_baselines3.common.preprocessing import is_image_space
# from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class MaskableQuantileNetwork(QuantileNetwork):
    def _predict(self, observation: PyTorchObs, action_masks=None, deterministic: bool = True) -> torch.Tensor:
        masked_q_values = self(observation).mean(dim=1).masked_fill(action_masks, -torch.inf)
        # Greedy action
        action = masked_q_values.argmax(dim=1).reshape(-1)
        return action

    def forward(self, obs: PyTorchObs, action_masks=None) -> torch.Tensor:
        """
        Predict the quantiles.

        :param action_masks:
        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        # print("Obs in quantile network")
        # print(obs)
        # TODO: Translation from join order to feature extractor likely goes wrong in the preprocess of the
        #  feature extractor!!
        print(self.observation_space)
        quantiles = self.quantile_net(self.extract_features(obs, self.features_extractor))
        output = quantiles.view(-1, self.n_quantiles, int(self.action_space.n))
        return output

    def extract_features(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """
        preprocessed_obs = self.preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    @staticmethod
    def preprocess_obs(
        obs: Union[torch.Tensor, dict[str, torch.Tensor]],
        observation_space: spaces.Space,
        normalize_images: bool = True,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Preprocess observation to be to a neural network.
        For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
        For discrete observations, it creates a one hot vector.

        :param obs: Observation
        :param observation_space:
        :param normalize_images: Whether to normalize images or not
            (True by default)
        :return:
        """
        if isinstance(observation_space, spaces.Dict):
            # Do not modify by reference the original observation
            assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
            preprocessed_obs = {}
            for key, _obs in obs.items():
                preprocessed_obs[key] = MaskableQuantileNetwork.preprocess_obs(
                    _obs, observation_space[key], normalize_images=normalize_images)
            return preprocessed_obs  # type: ignore[return-value]

        assert isinstance(obs, torch.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

        if isinstance(observation_space, spaces.Box):
            if np.issubdtype(observation_space.dtype, np.integer):
                # Integer box space
                return obs.int()
            else:
                # Float box space
                return obs.float()

        elif isinstance(observation_space, spaces.Discrete):
            # One hot encoding and convert to float to avoid errors
            return torch.nn.functional.one_hot(obs.long(), num_classes=int(observation_space.n)).float()

        # Custom multi discrete logic as this captures previous join data and shouldn't be one-hot encoded
        elif isinstance(observation_space, spaces.MultiDiscrete):
            return obs.long()

        elif isinstance(observation_space, spaces.MultiBinary):
            return obs.float()
        else:
            raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
