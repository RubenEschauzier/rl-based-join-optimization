from typing import List, Tuple, Any, TypedDict, NamedTuple, Union
import numpy as np

Plan = List[int]
EnvState = TypedDict('EnvState', {
    'unweighted_ensemble_prior': np.ndarray,
    "prepared_trees": np.ndarray,
    "prepared_idx": np.ndarray,
    "prepared_masks": np.ndarray,
    "z": np.ndarray
})
Cost = float
HistoryStep = Tuple[Plan, EnvState]
History = List[HistoryStep]
Candidate = Tuple[Plan, Cost, History, set[str]]
