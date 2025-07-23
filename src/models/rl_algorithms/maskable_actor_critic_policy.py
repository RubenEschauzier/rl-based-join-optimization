import warnings
import numpy as np
from typing import Optional, Union
import torch as th
import gymnasium as gym
from sb3_contrib.common.maskable.distributions import MaskableDistribution

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs


class MaskableActorCriticPolicyCustomPreprocessing(MaskableActorCriticPolicy):
    def extract_features(  # type: ignore[override]
            self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return self.extract_features_default(obs, features_extractor or self.features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self.extract_features_default(obs, self.pi_features_extractor)
            vf_features = self.extract_features_default(obs, self.vf_features_extractor)
            return pi_features, vf_features


    def get_distribution(self, obs: PyTorchObs, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = self.extract_features_default(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution


    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.extract_features_default(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)



    def extract_features_default(self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """
        preprocessed_obs = self.preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def preprocess_obs(
        self,
        obs: Union[th.Tensor, dict[str, th.Tensor]],
        observation_space: gym.spaces.Space,
        normalize_images: bool = True,
    ) -> Union[th.Tensor, dict[str, th.Tensor]]:
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
        if isinstance(observation_space, gym.spaces.Dict):
            # Do not modify by reference the original observation
            assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
            preprocessed_obs = {}
            for key, _obs in obs.items():
                preprocessed_obs[key] = self.preprocess_obs(
                    _obs, observation_space[key], normalize_images=normalize_images)
            return preprocessed_obs  # type: ignore[return-value]

        assert isinstance(obs, th.Tensor), f"Expecting a torch Tensor, but got {type(obs)}"

        if isinstance(observation_space, gym.spaces.Box):
            if np.issubdtype(observation_space.dtype, np.integer):
                # Integer box space
                return obs.int()
            else:
                # Float box space
                return obs.float()

        elif isinstance(observation_space, gym.spaces.Discrete):
            # One hot encoding and convert to float to avoid errors
            return th.nn.functional.one_hot(obs.long(), num_classes=int(observation_space.n)).float()

        # Custom multi discrete logic as this captures previous join data and shouldn't be one-hot encoded
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            return obs.long()

        elif isinstance(observation_space, gym.spaces.MultiBinary):
            return obs.float()
        else:
            raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")
