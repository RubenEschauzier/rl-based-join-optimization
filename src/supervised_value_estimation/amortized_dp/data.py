"""Dataset loading, the held-out split, and cached labels / embeddings for amortised DP."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.supervised_value_estimation.amortized_dp.agents import embed_triple_patterns
from src.supervised_value_estimation.amortized_dp.labels import label_query
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset


def load_datasets(cfg):
    """Same featurisation as prepare_data, but with term mappings (the oracle's sub-query
    construction needs term_to_id) and WITHOUT shuffling the train file, so the sampled
    training subset is reproducible across runs."""
    return load_queries_into_dataset(
        cfg.dataset.queries_train, cfg.dataset.queries_val, cfg.dataset.endpoint_location,
        cfg.dataset.rdf2vec_vector_location, BlazeGraphQueryEnvironment(cfg.dataset.endpoint_location),
        "predicate_edge", load_mappings=True, to_load=None,
        occurrences_location=cfg.dataset.occurrences_location,
        tp_cardinality_location=cfg.dataset.tp_cardinality_location,
        multiplicity_location=cfg.dataset.multiplicity_location, hll_location=cfg.dataset.hll_location,
        shuffle_train=False, shuffle_val=False,
    )


def load_split(cfg):
    """The rerun's val/test halves of the original validation file (query strings)."""
    path = Path(cfg.rerun.output_directory) / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the trial-85 rerun (or its split) first.")
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    return split["val_queries"], split["test_queries"]


def query_index(dataset):
    return {dataset[i].query: i for i in range(len(dataset))}


def sample_indices(indices, n, seed):
    if n is None or n >= len(indices):
        return list(indices)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(list(indices), size=n, replace=False).tolist())


def load_or_build_labels(path, dataset, indices, oracle, device, estimator=None):
    """{query string: QueryLabels}, cached at `path`; extends the cache for new queries."""
    path = Path(path)
    labels = {}
    if path.exists():
        with open(path, "rb") as f:
            labels = pickle.load(f)
    missing = [i for i in indices if dataset[i].query not in labels
               or (estimator is not None and not labels[dataset[i].query].estimated_logcard)]
    if missing:
        for i in tqdm(missing, desc=f"Labelling {path.stem}", mininterval=30):
            query = dataset[i]
            labels[query.query] = label_query(query, oracle, device, estimator=estimator)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(temporary, "wb") as f:
            pickle.dump(labels, f)
        os.replace(temporary, path)
    wanted = {dataset[i].query for i in indices}
    return {query: value for query, value in labels.items() if query in wanted}


@torch.no_grad()
def compute_embeddings(gnn, dataset, indices, device):
    return {dataset[i].query: embed_triple_patterns(gnn, dataset[i], device).float().cpu()
            for i in tqdm(indices, desc="Embedding", mininterval=30)}
