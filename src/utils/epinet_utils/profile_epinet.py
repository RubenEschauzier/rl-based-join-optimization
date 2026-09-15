"""Where does epinet time actually go, in training and at planning time?

Runs the real modules on synthetic query graphs of realistic shape, so it needs no
endpoint and no dataset. Reports per-component milliseconds and, crucially, the implied
*planning latency* per query -- a query optimizer that plans slowly has no benefit however
good its plans are.

    python -m src.utils.epinet_utils.profile_epinet
    python -m src.utils.epinet_utils.profile_epinet --index-dim 64 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch, Data

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.epistemic_neural_network import prepare_epinet_model  # noqa: E402
from src.utils.tree_conv_utils import (  # noqa: E402
    precompute_left_deep_tree_conv_index,
    precompute_left_deep_tree_node_mask,
)

FULL_MODEL_CONFIG = (
    "experiments/model_configs/policy_networks/"
    "t_cv_repr_separate_head_own_embeddings_hll.yaml"
)
PRIOR_MODEL_CONFIG = "experiments/model_configs/prior_networks/prior_t_cv_smallest_hll.yaml"

NODE_FEATURE_DIM = 136
# TripleGineConv.message() consumes edge_attr[:, :-1] and reads edge_attr[:, -1] as a
# +1/-1 direction flag, so the stored tensor carries one column more than the config's
# edge_dim.
EDGE_FEATURE_DIM = 138


def make_query(n_triple_patterns: int, device: torch.device) -> Batch:
    """A star-shaped query graph: one subject joined to n objects.

    Edges are emitted in (forward, reverse) pairs because TriplePatternPooling takes
    every other column to recover triple patterns.
    """
    n_nodes = n_triple_patterns + 1
    forward = torch.stack([
        torch.zeros(n_triple_patterns, dtype=torch.long),
        torch.arange(1, n_nodes, dtype=torch.long),
    ])
    edge_index = torch.stack([forward, forward.flip(0)], dim=2).reshape(2, -1)
    edge_attr = torch.randn((edge_index.shape[1], EDGE_FEATURE_DIM))
    edge_attr[0::2, -1] = 1.0    # forward edges
    edge_attr[1::2, -1] = -1.0   # their reverses
    data = Data(
        x=torch.randn((n_nodes, NODE_FEATURE_DIM)),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.query = "SELECT * WHERE { synthetic }"
    data.triple_patterns = list(range(n_triple_patterns))
    return Batch.from_data_list([data]).to(device)


def timed(label: str, function, repeats: int, device: torch.device) -> tuple[str, float]:
    for _ in range(3):  # warm up allocator / autotuner
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return label, (time.perf_counter() - start) * 1000.0 / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dim", type=int, default=30)
    parser.add_argument("--triple-patterns", type=int, default=8)
    parser.add_argument("--plans", type=int, default=80)
    parser.add_argument("--epi-samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)

    torch.manual_seed(0)
    model = prepare_epinet_model(
        full_gnn_config=str(REPOSITORY_ROOT / FULL_MODEL_CONFIG),
        config_ensemble_prior=str(REPOSITORY_ROOT / PRIOR_MODEL_CONFIG),
        epinet_index_dim=arguments.index_dim,
        mlp_dimension=64,
        heads_config={"plan_cost": {"layer": nn.Linear(64, 1)}},
        heads_config_prior={"plan_cost": {"layer": nn.Linear(5, 1)}},
        device=device,
        epinet_feature_mode="mlp_plus_plan",
        epinet_hidden_dim=50,
    ).to(device)
    model.eval()

    query = make_query(arguments.triple_patterns, device)
    conv_index = precompute_left_deep_tree_conv_index(20)
    node_mask = precompute_left_deep_tree_node_mask(20)

    generator = torch.Generator().manual_seed(0)
    plans = [
        (torch.randperm(arguments.triple_patterns, generator=generator).tolist(), 0.0, i)
        for i in range(arguments.plans)
    ]

    print(f"device={device}  index_dim={arguments.index_dim}  "
          f"triple_patterns={arguments.triple_patterns}  plans={arguments.plans}  "
          f"epi_samples={arguments.epi_samples}\n")

    with torch.no_grad():
        embedded = model.embed_query_batched(query)
        embedded_prior = model.embed_query_batched_prior(query)
        _, last_feature = model.estimate_cost_full(plans, embedded[0], conv_index, node_mask)
        indexes = model.sample_epistemic_indexes_batched(arguments.epi_samples)

        measurements = [
            timed("base GNN embed (once per query)",
                  lambda: model.embed_query_batched(query), arguments.repeats, device),
            timed(f"prior ensemble GNN embed (x{arguments.index_dim}, once per query)",
                  lambda: model.embed_query_batched_prior(query), arguments.repeats, device),
            timed("tree-conv structure build (CPU, per plan set)",
                  lambda: model.prepare_cost_estimation_inputs(
                      plans, embedded[0], conv_index, node_mask, device),
                  arguments.repeats, device),
            timed("base cost estimate (per plan set)",
                  lambda: model.estimate_cost_full(plans, embedded[0], conv_index, node_mask),
                  arguments.repeats, device),
            timed(f"prior ensemble cost (x{arguments.index_dim}, per plan set)",
                  lambda: model.compute_ensemble_prior(
                      plans, embedded_prior, conv_index, node_mask, 0),
                  arguments.repeats, device),
            timed("learnable epinet (per plan set)",
                  lambda: model.compute_learnable_mlp_batched(last_feature, indexes),
                  arguments.repeats, device),
            timed("prior epinet MLP (per plan set)",
                  lambda: model.compute_mlp_prior_batched(last_feature, indexes),
                  arguments.repeats, device),
        ]

    width = max(len(label) for label, _ in measurements)
    total = sum(value for _, value in measurements)
    print(f"{'component':<{width}}   {'ms':>9}   {'share':>6}")
    print("-" * (width + 22))
    for label, value in measurements:
        print(f"{label:<{width}}   {value:9.3f}   {value / total:6.1%}")
    print("-" * (width + 22))
    print(f"{'TOTAL (one query, one plan set)':<{width}}   {total:9.3f}")

    per_query = dict(measurements)
    once = sum(v for k, v in per_query.items() if "once per query" in k)
    per_set = total - once
    print(f"\nPlanning-latency model, beam search with B steps per query:")
    print(f"   latency(B) ~= {once:.2f} ms + B x {per_set:.2f} ms")
    for beam_steps in (1, 8, 32):
        print(f"     B={beam_steps:<3} -> {once + beam_steps * per_set:8.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
