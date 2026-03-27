from dataclasses import dataclass

import torch
import numpy as np
from typing import Tuple, Union, NamedTuple, Any

from numpy.dtypes import StringDType

@dataclass(frozen=True)
class ExecutionBufferSamples:
    queries: np.ndarray
    join_plans: np.ndarray
    prepared_trees: np.ndarray
    prepared_idx: np.ndarray
    prepared_masks: np.ndarray
    unweighted_ensemble_priors: np.ndarray
    episode_zs: torch.Tensor
    intermediate_join_sizes: torch.Tensor
    is_valid_size: torch.Tensor
    is_valid_episode: torch.Tensor
    c_vectors: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor

@dataclass(frozen=True)
class ExecutionBufferSamplesWithTargets(ExecutionBufferSamples):
    total_cost: torch.Tensor
    latencies: torch.Tensor
    is_censored: torch.Tensor

class ExecutionReplayBuffer:

    def __init__(
            self,
            buffer_size: int,
            epi_index_dim: int,
            device: Union[torch.device, str] = "auto",
            use_per: bool = False,
            alpha: float = 0.6,
    ):
        """
        Query execution buffer that stores the required data to reconstruct the cost estimation.
        Optionally supports extensions to do Prioritized Experience Replay (PER). Either based on model error
        or estimated epistemic uncertainty.
        :param buffer_size: Max entries in buffer
        :param epi_index_dim: Dimension of the sampled epistemic index
        :param device: Device tensors should be stored on when returned by sample
        :param use_per: If PER should be used
        :param alpha: Hyperparameter for PER
        """
        self.buffer_size = buffer_size
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.use_per = use_per
        self.alpha = alpha
        self.max_priority = 1.0

        # Fixed-size NumPy arrays (Contiguous memory)
        self.query_strings = np.empty((self.buffer_size,), dtype=StringDType())
        self.episode_zs = np.zeros((self.buffer_size, epi_index_dim), dtype=np.float32)
        self.intermediate_join_sizes = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.is_valid_size = np.zeros((self.buffer_size, 1), dtype=bool)
        self.is_valid_episode = np.zeros((self.buffer_size, 1), dtype=bool)
        self.c_vectors = np.zeros((self.buffer_size, 2, 1, epi_index_dim), dtype=np.float32)
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

        # Variable-size NumPy arrays (Object pointers)
        self.join_plans = np.empty((self.buffer_size,), dtype=object)
        self.prepared_trees = np.empty((self.buffer_size,), dtype=object)
        self.prepared_idx = np.empty((self.buffer_size,), dtype=object)
        self.prepared_masks = np.empty((self.buffer_size,), dtype=object)
        self.unweighted_ensemble_priors = np.empty((self.buffer_size,), dtype=object)

        self.pos = 0
        self.full = False

    def add(
            self,
            query_string: str,
            join_plan: Any,
            prepared_trees: Union[np.ndarray, torch.Tensor],
            prepared_idx: Union[np.ndarray, torch.Tensor],
            prepared_masks: Union[np.ndarray, torch.Tensor],
            unweighted_ensemble_prior: np.ndarray,
            episode_z: np.ndarray,
            intermediate_join_size: float,
            is_valid_size: bool,
            is_valid_episode: bool,
            c_vectors_observation: np.ndarray,
    ) -> None:
        # Assign variable-length objects
        self.join_plans[self.pos] = join_plan
        self.prepared_trees[self.pos] = prepared_trees
        self.prepared_idx[self.pos] = prepared_idx
        self.prepared_masks[self.pos] = prepared_masks
        self.unweighted_ensemble_priors[self.pos] = unweighted_ensemble_prior

        # Assign fixed-length data
        self.query_strings[self.pos] = query_string
        self.episode_zs[self.pos] = episode_z
        self.intermediate_join_sizes[self.pos] = intermediate_join_size
        self.is_valid_size[self.pos] = is_valid_size
        self.is_valid_episode[self.pos] = is_valid_episode
        self.c_vectors[self.pos] = c_vectors_observation

        if self.use_per:
            self.priorities[self.pos] = self.max_priority
        else:
            self.priorities[self.pos] = 1.0

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, beta: float = 0.4) -> ExecutionBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos

        if self.use_per:
            priorities = self.priorities[:upper_bound] ** self.alpha
            probs = priorities / np.sum(priorities)
            indices = np.random.choice(upper_bound, batch_size, p=probs)
            weights = (upper_bound * probs[indices]) ** (-beta)
            weights = weights / weights.max()
        else:
            indices = np.random.randint(0, upper_bound, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        return ExecutionBufferSamples(
            queries=self.query_strings[indices],
            join_plans=self.join_plans[indices],
            prepared_trees=self.prepared_trees[indices],
            prepared_idx=self.prepared_idx[indices],
            prepared_masks=self.prepared_masks[indices],
            unweighted_ensemble_priors=self.unweighted_ensemble_priors[indices],
            episode_zs=self._to_torch(self.episode_zs[indices]),
            intermediate_join_sizes=self._to_torch(self.intermediate_join_sizes[indices]),
            is_valid_size=self._to_torch(self.is_valid_size[indices]),
            is_valid_episode=self._to_torch(self.is_valid_episode[indices]),
            c_vectors=self._to_torch(self.c_vectors[indices]),
            weights=self._to_torch(weights),
            indices=self._to_torch(indices)
        )

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """
        Updates the priorities of sampled transitions.
        Call this after computing the loss/error in the training loop.
        """
        if not self.use_per:
            return

        indices_np = indices.cpu().numpy()
        priorities_np = priorities.detach().cpu().numpy()

        # Add small epsilon to guarantee non-zero probability
        self.priorities[indices_np] = priorities_np + 1e-6

        # Track the maximum priority for new additions
        self.max_priority = max(self.max_priority, np.max(priorities_np))

        print(self.priorities)

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)