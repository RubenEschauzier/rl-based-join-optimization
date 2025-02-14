
from src.models.rl_algorithms.masked_qrdqn import MaskableQuantileNetwork
#
#
# class QRDQNFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, feature_dim=512):
#         super().__init__(observation_space, feature_dim)
#
#         self.max_triples = observation_space["result_embeddings"].shape[0]  # max_triples
#         self.feature_dim = feature_dim
#
#         # Preprocess the entire query graph
#         # AdaptiveMaxPool doesn't work
#         self.result_mlp = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, feature_dim),
#             nn.ReLU(),
#         )
#
#         # Embed the mask to allow model to learn what actions are invalid
#         self.mask_embedder = nn.Sequential(
#             nn.Linear(self.max_triples, self.max_triples),
#             nn.ReLU(),
#             nn.Linear(self.max_triples, self.max_triples),
#             nn.ReLU()
#         )
#
#         # Hierarchical embedding of join order
#         self.join_mlp = nn.Sequential(
#             nn.Linear(feature_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, feature_dim),
#             nn.ReLU(),
#         )
#
#         # Final feature output size
#         self._features_dim = feature_dim + self.max_triples
#
#     def forward(self, observations):
#         """
#         observations:
#             - result_embeddings: (batch_size, max_triples, feature_dim)
#             - join_embedding: (batch_size, feature_dim)
#             - joined: (batch_size, max_triples) - Binary mask
#         """
#         # (B, max_triples, feature_dim)
#         result_embeddings = observations["result_embeddings"]
#         joined = observations["joined"]
#         # Here we do -1 to obtain the actual join order as the environment increments all join orders by 1 for
#         # sb3 preprocessing purposes
#         join_order = observations["join_order"] - 1
#         # TODO Hierarchical MLP application based on join order
#
#         # Pool features through sum pooling, as the masked out features will be 0 biasing the representation
#         pooled = torch.sum(result_embeddings, dim=1)
#         # (B, feature_dim)
#         result_features = self.result_mlp(pooled)
#
#         mask_emb = self.mask_embedder(joined)
#         return torch.concat((result_features, mask_emb), dim=1)
#
#
# class MaskableQRDQNPolicy(QRDQNPolicy):
#     quantile_net: MaskableQuantileNetwork
#     quantile_net_target: MaskableQuantileNetwork
#
#     def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete,
#                  lr_schedule: Schedule, **kwargs):
#         super().__init__(observation_space, action_space, lr_schedule, **kwargs)
#
#     def make_quantile_net(self) -> MaskableQuantileNetwork:
#         # Make sure we always have separate networks for features extractors etc
#         net_args = self._update_features_extractor(self.net_args, features_extractor=None)
#         return MaskableQuantileNetwork(**net_args).to(self.device)
#
#     def forward(self, obs, deterministic=True, action_masks=None):
#         """
#         Q-learning policy forward pass that masks out invalid actions.
#         :param obs:
#         :param deterministic:
#         :param action_masks:
#
#         Returns:
#             torch.Tensor: Masked Q-values.
#         """
#         # Get standard QR-DQN output
#         q_values = self.q_net(obs)  # Shape: (batch_size, n_quantiles, action_dim)
#         # Extract the mask from observation
#         joined_mask = obs["joined"]  # Shape: (batch_size, max_triples)
#         valid_mask = 1 - joined_mask  # Invert mask (1 for valid, 0 for invalid)
#
#         # Compute mean Q-values across quantiles
#         q_values_mean = q_values.mean(dim=1)  # Shape: (batch_size, action_dim)
#
#         # Apply mask: Set Q-values for invalid actions to -inf
#         q_values_mean[valid_mask == 0] = float("-inf")
#
#         # Select action using masked Q-values
#         action = torch.argmax(q_values_mean, dim=1)
#
#         return action
#
#     def _predict(self, obs: PyTorchObs, deterministic: bool = True, action_masks=None) -> torch.Tensor:
#         return self.quantile_net._predict(obs, deterministic=deterministic)
#
#     def predict(
#         self,
#         observation: Union[np.ndarray, dict[str, np.ndarray]],
#         state: Optional[tuple[np.ndarray, ...]] = None,
#         episode_start: Optional[np.ndarray] = None,
#         deterministic: bool = False,
#         action_masks: Optional[tuple[np.ndarray, ...]] = None
#     ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
#         """
#         Get the policy action from an observation (and optional hidden state).
#         Includes sugar-coating to handle different observations (e.g. normalizing images).
#
#         :param action_masks:
#         :param observation: the input observation
#         :param state: The last hidden states (can be None, used in recurrent policies)
#         :param episode_start: The last masks (can be None, used in recurrent policies)
#             this corresponds to beginning of episodes,
#             where the hidden states of the RNN must be reset.
#         :param deterministic: Whether to return deterministic actions.
#         :return: the model's action and the next hidden state
#             (used in recurrent policies)
#         """
#         # Switch to eval mode (this affects batch norm / dropout)
#         self.set_training_mode(False)
#
#         # Check for common mistake that the user does not mix Gym/VecEnv API
#         # Tuple obs are not supported by SB3, so we can safely do that check
#         if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
#             raise ValueError(
#                 "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
#                 "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
#                 "vs `obs = vec_env.reset()` (SB3 VecEnv). "
#                 "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
#                 "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
#             )
#
#         obs_tensor, vectorized_env = self.obs_to_tensor(observation)
#
#         with torch.no_grad():
#             actions = self._predict(obs_tensor, deterministic=deterministic, action_masks=action_masks)
#             print("Actions in QRDQN Policy")
#             print(actions)
#         # Convert to numpy, and reshape to the original action shape
#         actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]
#
#         if isinstance(self.action_space, spaces.Box):
#             if self.squash_output:
#                 # Rescale to proper domain when using squashing
#                 actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
#             else:
#                 # Actions could be on arbitrary scale, so clip the actions to avoid
#                 # out of bound error (e.g. if sampling from a Gaussian distribution)
#                 actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]
#
#         # Remove batch dimension if needed
#         if not vectorized_env:
#             assert isinstance(actions, np.ndarray)
#             actions = actions.squeeze(axis=0)
#
#         return actions, state  # type: ignore[return-value]
#
#
#     class InternalStateResetCallback(BaseCallback):
#         def __init__(self, verbose=0):
#             super().__init__(verbose)
#
#         def _on_step(self):
#             done = self.locals["dones"]
#
#             if done[0]:  # If episode ended
#                 print("End episode resetting state")
#                 # self.model.policy.features_extractor.reset_join_state()
#                 # print(f"Episode finished. Last reward: {rewards[0]}")
#                 # print(f"Final observation: {new_obs}")
#
#             return True  # Continue training


if __name__ == "__main__":
    pass
