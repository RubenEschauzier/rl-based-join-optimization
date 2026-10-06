"""Train the contracted-join-graph value net on exact DP targets (offline, no epinet).

    python -m src.supervised_value_estimation.amortized_dp.train_amortized_dp

Per training query, every connected subset S is labelled with the oracle's log card(S)
and the exact log cost-to-go G(S). A training *state* is S together with its children
S u {a}; the root state's children are the connected pairs (the first join). Losses:

    card   MSE on standardised log card of every evaluated set
    G      MSE on standardised log G of every non-final child
    rank   listwise DP distillation: cross-entropy between the model's and the exact
           softmax(-log Q / T) over a state's children, where
               log Q(child) = log( card(child) + G(child) [+ min(card i, card j) at the root] )
           This is exactly the quantity the agent ranks by, so it trains the decision.

Model selection: after every epoch, greedy decoding (beam_search, beam 1, through the same
AmortizedDPAgent the comparison uses) on queries from the VALIDATION half, scored by the
cost ratio to the exact DP optimum. Outputs go to amortized_dp.output_directory/run-<time>/.
https://claude.ai/artifact/EbMQ24hHYmSnhRmK1nDGF9?sk=J6HCdLzaJr_BeUG6i1gQcQ
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from src.supervised_value_estimation.amortized_dp.agents import AmortizedDPAgent, load_cardinality_gnn
from src.supervised_value_estimation.amortized_dp.data import (
    compute_embeddings, load_datasets, load_or_build_labels, load_split, query_index, sample_indices,
)
from src.supervised_value_estimation.amortized_dp.labels import (
    JOIN_FEATURE_DIM, cost_to_go, join_position_features, logsumexp,
)
from src.supervised_value_estimation.amortized_dp.model import ContractedJoinGraphValueNet
from src.supervised_value_estimation.amortized_dp.simulated_execution import (
    SimulatedCostExecutionStrategy, summarize,
)
from src.supervised_value_estimation.optuna_epinet_sweep import _resolve_model_paths
from src.supervised_value_estimation.search_algorithms.beam_search_left_deep import beam_search
from src.utils.epinet_utils.epinet_loss import anchor_vectors, epinet_term_loss, stable_key


class StateTables:
    """Flat, GPU-resident training tables: one row per (state, child) pair.

    By default every connected subset is labelled (the full-oracle teacher). With
    `observed_logcards` only the subsets in each query's observed table are used, and the
    cost-to-go targets come from a DP over those entries alone ("stitching"): the cheapest
    completion assembled from observed pieces. Such a target is an upper bound on the true
    cost-to-go, which `cost_to_go_upper_bound` records for the one-sided loss.
    """

    def __init__(self, labels_list, embeddings_list, device, patterns_list=None, observed_logcards=None,
                 cost_to_go_upper_bound=False, latency_to_go=None, step_ms=None):
        """`latency_to_go`: optional per-query {set: log1p(remaining ms)} for the latency head.
        `step_ms`: optional per-query {(parent set, child set): ms} for the step head (parent 0 =
        a pair's first join). Every row also carries stable keys for the epinet anchors:
        key_state for (query, child set) and key_step for (query, parent set, child set)."""
        self.cost_to_go_upper_bound = cost_to_go_upper_bound
        self.has_latency = latency_to_go is not None
        self.has_step = step_ms is not None
        n_max = max(labels.n_tp for labels in labels_list)
        embedding_dim = embeddings_list[0].shape[1]
        n_queries = len(labels_list)
        self.embeddings = torch.zeros(n_queries, n_max, embedding_dim)
        self.pattern_mask = torch.zeros(n_queries, n_max, dtype=torch.bool)
        self.adjacency = torch.zeros(n_queries, n_max, n_max, dtype=torch.bool)
        # Join-position edge features, for the gine / gat message layers.
        self.edge_features = torch.zeros(n_queries, n_max, n_max, JOIN_FEATURE_DIM)
        if patterns_list is not None:
            for query_id, patterns in enumerate(patterns_list):
                n = len(patterns)
                self.edge_features[query_id, :n, :n] = torch.from_numpy(join_position_features(patterns))

        rows = {key: [] for key in ("query", "state", "mask", "card", "g", "full", "root",
                                    "single_a", "single_b", "card_a", "card_b", "true_logq",
                                    "latency", "latency_valid", "parent", "added", "step", "step_valid",
                                    "key_state", "key_step")}
        n_states = 0
        for query_id, (labels, embeddings) in enumerate(zip(labels_list, embeddings_list)):
            n = labels.n_tp
            self.embeddings[query_id, :n] = embeddings
            self.pattern_mask[query_id, :n] = True
            for i in range(n):
                for j in range(n):
                    self.adjacency[query_id, i, j] = bool(labels.neighbour_masks[i] >> j & 1)
            logcard = labels.logcard if observed_logcards is None else observed_logcards[query_id]
            log_g = cost_to_go(labels, logcard)
            latency = {} if latency_to_go is None else latency_to_go[query_id]
            steps = {} if step_ms is None else step_ms[query_id]
            query_key = stable_key(labels.query)

            def add_row(child, root=False, a=0, b=0, parent=0):
                full = child == labels.full_mask
                extra = min(logcard[1 << a], logcard[1 << b]) if root else -math.inf
                rows["query"].append(query_id)
                rows["state"].append(n_states)
                rows["mask"].append(child)
                rows["card"].append(logcard[child])
                rows["g"].append(0.0 if full else log_g[child])
                rows["full"].append(full)
                rows["root"].append(root)
                rows["single_a"].append(1 << a if root else 0)
                rows["single_b"].append(1 << b if root else 0)
                rows["card_a"].append(logcard[1 << a] if root else 0.0)
                rows["card_b"].append(logcard[1 << b] if root else 0.0)
                rows["true_logq"].append(logsumexp([logcard[child], -math.inf if full else log_g[child], extra]))
                rows["latency"].append(latency.get(child, 0.0))
                rows["latency_valid"].append(child in latency)
                rows["parent"].append(parent)
                rows["added"].append(-1 if root else (child ^ parent).bit_length() - 1)
                rows["step"].append(math.log1p(steps[(parent, child)]) if (parent, child) in steps else 0.0)
                rows["step_valid"].append((parent, child) in steps)
                rows["key_state"].append(stable_key(query_key, child))
                rows["key_step"].append(stable_key(query_key, parent, child))

            root_pairs = [(i, j) for i, j in labels.connected_pairs()
                          if {(1 << i) | (1 << j), 1 << i, 1 << j} <= logcard.keys()
                          # -inf (log 0) is a finished query, +inf an unknown completion.
                          and log_g.get((1 << i) | (1 << j), math.inf) < math.inf]
            for i, j in root_pairs:
                add_row((1 << i) | (1 << j), root=True, a=i, b=j)
            if root_pairs:
                n_states += 1
            for mask in logcard:
                if mask.bit_count() < 2 or mask == labels.full_mask:
                    continue
                children = [child for child in labels.children(mask)
                            if child in logcard and math.isfinite(log_g.get(child, math.inf) if child != labels.full_mask else 0.0)]
                if len(children) < 1:
                    continue
                for child in children:
                    add_row(child, parent=mask)
                n_states += 1

        self.rows = {key: torch.tensor(value) for key, value in rows.items()}
        self.n_states = n_states
        state = self.rows["state"]
        self.state_offsets = torch.zeros(n_states + 1, dtype=torch.long)
        self.state_offsets[1:] = torch.cumsum(torch.bincount(state, minlength=n_states), dim=0)
        self.device = device
        for name in ("embeddings", "pattern_mask", "adjacency", "edge_features", "state_offsets"):
            setattr(self, name, getattr(self, name).to(device))
        self.rows = {key: value.to(device) for key, value in self.rows.items()}
        self.n_max = n_max

    def latency_statistics(self):
        values = self.rows["latency"][self.rows["latency_valid"]]
        return (float(values.mean()), float(values.std())) if len(values) > 1 else (0.0, 1.0)

    def step_statistics(self):
        values = self.rows["step"][self.rows["step_valid"]]
        return (float(values.mean()), float(values.std())) if len(values) > 1 else (0.0, 1.0)

    def statistics(self):
        card = torch.cat([self.rows["card"], self.rows["card_a"][self.rows["root"]]])
        g = self.rows["g"][~self.rows["full"]]
        return float(card.mean()), float(card.std()), float(g.mean()), float(g.std())

    def batch_rows(self, state_ids):
        starts, ends = self.state_offsets[state_ids], self.state_offsets[state_ids + 1]
        counts = ends - starts
        local_state = torch.repeat_interleave(torch.arange(len(state_ids), device=self.device), counts)
        within = torch.arange(int(counts.sum()), device=self.device) - torch.repeat_interleave(
            torch.cumsum(counts, 0) - counts, counts)
        return torch.repeat_interleave(starts, counts) + within, local_state


def _bits(masks, n):
    return (masks.unsqueeze(1) >> torch.arange(n, device=masks.device)) & 1 == 1


def _evaluate_sets(model, tables, query_ids, masks, return_latency=False):
    return model(tables.embeddings[query_ids], tables.pattern_mask[query_ids],
                 tables.adjacency[query_ids], _bits(masks, tables.n_max),
                 tables.edge_features[query_ids] if model.edge_feature_dim else None,
                 return_latency=return_latency)


def _group_log_softmax(logits, groups, n_groups):
    peak = torch.full((n_groups,), -math.inf, device=logits.device).scatter_reduce(
        0, groups, logits, reduce="amax", include_self=True)
    shifted = torch.exp(logits - peak[groups])
    normaliser = torch.zeros(n_groups, device=logits.device).index_add(0, groups, shifted)
    return logits - peak[groups] - torch.log(normaliser[groups])


def _one_sided_mse(under_weight):
    """Squared error, weighted under_weight when the prediction is below the target: the
    target is an upper bound (the cheapest KNOWN completion), so the truth can only be lower."""
    def loss(prediction, target):
        error = prediction - target
        error = error * torch.where(error > 0, torch.ones_like(error), torch.full_like(error, under_weight ** 0.5))
        return torch.mean(error ** 2)
    return loss


def _mse(prediction, target):
    return torch.mean((prediction - target) ** 2)


# Distinct anchor draws per head (anchor_vectors' seed), so a set's card and G perturbations
# are independent.
EPINET_HEAD_SEEDS = {"card": 1, "cost_to_go": 2, "latency": 3, "step": 4}


def _epinet_losses(model, outputs, r, targets, epinet_settings):
    """The epinet term (src/utils/epinet_utils/epinet_loss.py) for every head with targets.
    targets: {head: (valid row mask, standardised target, loss_fn)}."""
    epinet_indexes = model.epinet_state.sample_epistemic_indexes_batched(int(epinet_settings.n_indexes_train))
    losses = {}
    for head, (valid, target, loss_fn) in targets.items():
        if not valid.any():
            continue
        keys = r["key_step" if head == "step" else "key_state"][valid].cpu().numpy()
        anchors = anchor_vectors(keys, model.epi_index_dim, EPINET_HEAD_SEEDS[head], target.device)
        features = outputs["features_step" if head == "step" else "features_state"][valid]
        losses[head], _, _ = epinet_term_loss(
            model.epinet_for(head), head, outputs[head][valid].unsqueeze(-1), features, target[valid],
            epinet_indexes, anchors, float(epinet_settings.sigma), model.epinet_alpha(head), loss_fn)
    return losses


@torch.no_grad()
def calibrate_epinet_prior(model, tables, heads, epinet_settings, max_rows=4096):
    """prior_scale_target (as in the plan-cost epinet): per head, set the prior weight so the
    prior's std over sampled indices, on (up to max_rows of) this table's rows, equals the
    target in standardised target units. Heads already calibrated keep their weight. Returns
    {head: (measured prior std, alpha)}; nothing if prior_scale_target is null."""
    target = epinet_settings.get("prior_scale_target")
    alphas = model.epinet_config.setdefault("alpha", {})
    heads = [head for head in heads if head not in alphas]
    if target is None or not heads:
        return {}
    rows = torch.randperm(len(tables.rows["query"]), device=tables.device)[:max_rows]
    r = {key: value[rows] for key, value in tables.rows.items()}
    was_training = model.training
    model.eval()
    outputs = model.forward_heads(tables.embeddings[r["query"]], tables.pattern_mask[r["query"]],
                                  tables.adjacency[r["query"]], _bits(r["mask"], tables.n_max),
                                  tables.edge_features[r["query"]] if model.edge_feature_dim else None,
                                  added=r["added"], heads=("card", "step") if "step" in heads else ("card",))
    model.train(was_training)
    calibrated = {}
    for head in heads:
        features = outputs["features_step" if head == "step" else "features_state"]
        scale = model.epinet_for(head).prior_scale(features, int(epinet_settings.get("prior_scale_indexes", 64)),
                                                    head_name=head)
        alphas[head] = float(target) / max(scale, 1e-8)
        calibrated[head] = (scale, alphas[head])
    return calibrated


def training_step(model, tables, state_ids, cfg_training, epinet_settings=None):
    """One gradient step's loss. `epinet_settings` ({sigma, n_indexes_train}) adds the epinet
    term for every head of a model built with an epinet."""
    rows, local_state = tables.batch_rows(state_ids)
    r = {key: value[rows] for key, value in tables.rows.items()}
    train_latency = tables.has_latency and getattr(model, "has_latency_head", False)
    train_step = tables.has_step and getattr(model, "has_step_head", False)
    heads = ("card", "cost_to_go") + (("latency",) if train_latency else ()) + (("step",) if train_step else ())
    outputs = model.forward_heads(tables.embeddings[r["query"]], tables.pattern_mask[r["query"]],
                                  tables.adjacency[r["query"]], _bits(r["mask"], tables.n_max),
                                  tables.edge_features[r["query"]] if model.edge_feature_dim else None,
                                  added=r["added"], heads=heads)
    card_std, g_std = outputs["card"], outputs["cost_to_go"]
    target_card_std, target_g_std = model.standardise(r["card"], r["g"])
    loss_card = torch.mean((card_std - target_card_std) ** 2)
    not_full = ~r["full"]
    g_error = g_std[not_full] - target_g_std[not_full]
    if tables.cost_to_go_upper_bound:
        # The target is the cheapest KNOWN completion, so the truth can only be lower:
        # predicting above it is always wrong, predicting below it is only weakly penalised.
        under_weight = cfg_training.get("upper_bound_under_weight", 0.1)
        g_error = g_error * torch.where(g_error > 0, torch.ones_like(g_error),
                                        torch.full_like(g_error, under_weight ** 0.5))
    loss_g = torch.mean(g_error ** 2) if not_full.any() else card_std.sum() * 0

    card, g = model.unstandardise(card_std, g_std)
    extra = torch.full_like(card, -math.inf)
    root = r["root"]
    if root.any():
        root_query = r["query"][root]
        single_masks = torch.cat([r["single_a"][root], r["single_b"][root]])
        single_std, _ = _evaluate_sets(model, tables, torch.cat([root_query, root_query]), single_masks)
        single_target, _ = model.standardise(torch.cat([r["card_a"][root], r["card_b"][root]]),
                                             torch.zeros_like(single_std))
        loss_card = loss_card + torch.mean((single_std - single_target) ** 2)
        single_card, _ = model.unstandardise(single_std, torch.zeros_like(single_std))
        a, b = single_card.chunk(2)
        extra[root] = torch.minimum(a, b)
    g_term = torch.where(r["full"], torch.full_like(g, -math.inf), g)
    predicted_logq = torch.logsumexp(torch.stack([card, g_term, extra]), dim=0)

    n_groups = len(state_ids)
    predicted = _group_log_softmax(-predicted_logq / cfg_training.rank_temperature, local_state, n_groups)
    target = torch.exp(_group_log_softmax(-r["true_logq"] / cfg_training.target_temperature, local_state, n_groups))
    loss_rank = -torch.zeros(n_groups, device=card.device).index_add(0, local_state, target * predicted).mean()
    total = loss_card + loss_g + cfg_training.rank_weight * loss_rank
    parts = {"loss_card": loss_card.item(), "loss_g": loss_g.item(), "loss_rank": loss_rank.item()}
    if train_latency and r["latency_valid"].any():
        valid = r["latency_valid"]
        latency_error = outputs["latency"][valid] - model.standardise_latency(r["latency"][valid])
        if tables.cost_to_go_upper_bound:
            # Stitched latency is the fastest KNOWN completion, so again an upper bound.
            under_weight = cfg_training.get("upper_bound_under_weight", 0.1)
            latency_error = latency_error * torch.where(latency_error > 0, torch.ones_like(latency_error),
                                                        torch.full_like(latency_error, under_weight ** 0.5))
        loss_latency = torch.mean(latency_error ** 2)
        total = total + cfg_training.get("latency_weight", 1.0) * loss_latency
        parts["loss_latency"] = loss_latency.item()
    if train_step and r["step_valid"].any():
        # Observed step times (the fastest of each (parent, child) step): plain squared error.
        valid = r["step_valid"]
        loss_step = torch.mean((outputs["step"][valid] - model.standardise_step(r["step"][valid])) ** 2)
        total = total + cfg_training.get("step_weight", 1.0) * loss_step
        parts["loss_step"] = loss_step.item()
    if epinet_settings is not None and getattr(model, "has_epinet", False):
        under_weight = cfg_training.get("upper_bound_under_weight", 0.1)
        bound_loss = _one_sided_mse(under_weight) if tables.cost_to_go_upper_bound else _mse
        targets = {"card": (torch.ones_like(r["full"]), target_card_std, _mse),
                   "cost_to_go": (not_full, target_g_std, bound_loss)}
        if train_latency:
            targets["latency"] = (r["latency_valid"], model.standardise_latency(r["latency"]), bound_loss)
        if train_step:
            targets["step"] = (r["step_valid"], model.standardise_step(r["step"]), _mse)
        for head, value in _epinet_losses(model, outputs, r, targets, epinet_settings).items():
            total = total + value
            parts[f"loss_epinet_{head}"] = value.item()
    parts["loss"] = total.item()
    return total, parts


@torch.no_grad()
def greedy_selection_metrics(model, dataset, indices, embeddings, labels, device, beam_width=1):
    model.eval()
    agent = AmortizedDPAgent(model, embed_fn=None, device=device, embedding_cache=embeddings)
    strategy = SimulatedCostExecutionStrategy(labels)
    results, planning = [], []
    for i in indices:
        query = Batch.from_data_list([dataset[i]])
        start = time.perf_counter()
        plan = beam_search(query, agent, beam_width)[0]["plan"]
        planning.append(time.perf_counter() - start)
        results.append(strategy.score(dataset[i].query, plan))
    model.train()
    summary = summarize(results)
    summary["planning_ms_mean"] = 1000 * float(np.mean(planning))
    return summary


@hydra.main(version_base=None,
            config_path="../../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
            config_name="amortized_dp_mixed_yago.yaml")
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    settings, cfg_training = cfg.amortized_dp, cfg.amortized_dp.training
    torch.manual_seed(cfg_training.seed)
    np.random.seed(cfg_training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_directory = Path(settings.output_directory) / (
        f"run-{settings.model.message_layer}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
    run_directory.mkdir(parents=True, exist_ok=False)
    with open(run_directory / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {run_directory} | device {device}")

    train_dataset, val_dataset = load_datasets(cfg)
    val_queries, _ = load_split(cfg)
    val_index = query_index(val_dataset)
    train_indices = sample_indices(range(len(train_dataset)), settings.n_train_queries, settings.train_query_seed)
    selection_indices = sample_indices([val_index[q] for q in val_queries if q in val_index],
                                       settings.n_selection_queries, settings.train_query_seed + 1)

    oracle = load_cardinality_gnn(cfg.models.oracle.config, cfg.models.oracle.dir, device)
    embedder = load_cardinality_gnn(cfg.models.embedder.config, cfg.models.embedder.dir, device)
    label_directory = Path(settings.label_directory)
    train_labels = load_or_build_labels(label_directory / "train.pkl", train_dataset, train_indices, oracle, device)
    selection_labels = load_or_build_labels(label_directory / "val.pkl", val_dataset, selection_indices,
                                            oracle, device)
    del oracle
    # Queries whose patterns do not form ONE variable-connected graph have no plan without
    # a cartesian product, so beam_search (and every agent) returns nothing for them.
    for name, indices, dataset, labels in (("train", train_indices, train_dataset, train_labels),
                                           ("selection", selection_indices, val_dataset, selection_labels)):
        kept = [i for i in indices if labels[dataset[i].query].is_connected]
        print(f"{name}: {len(indices) - len(kept)} of {len(indices)} queries are not variable-connected; dropped")
        indices[:] = kept
    train_embeddings = compute_embeddings(embedder, train_dataset, train_indices, device)
    selection_embeddings = compute_embeddings(embedder, val_dataset, selection_indices, device)
    del embedder

    queries = [train_dataset[i].query for i in train_indices]
    patterns = {train_dataset[i].query: list(train_dataset[i].triple_patterns) for i in train_indices}
    tables = StateTables([train_labels[q] for q in queries], [train_embeddings[q] for q in queries], device,
                         patterns_list=[patterns[q] for q in queries])
    print(f"Training tables: {len(queries)} queries, {tables.n_states:,} states, "
          f"{len(tables.rows['mask']):,} (state, child) rows, up to {tables.n_max} patterns")

    model_kwargs = {"embedding_dim": tables.embeddings.shape[-1], "hidden_dim": settings.model.hidden_dim,
                    "n_message_layers": settings.model.n_message_layers,
                    "message_layer": settings.model.message_layer,
                    "edge_feature_dim": JOIN_FEATURE_DIM if settings.model.edge_features else 0}
    # Optional epinet on the card and G heads (amortized_dp.epinet); online training adds the
    # latency and step heads to it.
    epinet_cfg = settings.get("epinet")
    epinet_settings = epinet_cfg if epinet_cfg is not None and epinet_cfg.enabled else None
    if epinet_settings is not None:
        model_kwargs["epinet"] = OmegaConf.to_container(epinet_cfg.model, resolve=True)
    model = ContractedJoinGraphValueNet(**model_kwargs).to(device)
    model.set_target_statistics(*tables.statistics())
    if epinet_settings is not None:
        for head, (scale, alpha) in calibrate_epinet_prior(model, tables, ["card", "cost_to_go"], epinet_settings).items():
            print(f"Epinet prior {head}: std {scale:.3f} -> alpha {alpha:.4f} (target {epinet_settings.prior_scale_target})")
        model_kwargs["epinet"] = model.epinet_config        # the calibrated weights go into the checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_training.lr, weight_decay=cfg_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_training.n_epochs)

    def save(path, epoch, metrics):
        torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs, "epoch": epoch,
                    "metrics": metrics}, path)

    # Model selection on one statistic of the greedy cost ratio to the optimum (lower is
    # better), ties broken by the geomean ratio. p90 by default: in online training a round
    # waits for its slowest plans, so the tail matters more than the average.
    selection_metric = cfg_training.get("selection_metric", "p90_ratio")
    best, best_epoch, patience = (math.inf, math.inf), 0, 0
    for epoch in range(1, cfg_training.n_epochs + 1):
        start = time.perf_counter()
        order = torch.randperm(tables.n_states, device=device)
        running = []
        for batch_start in range(0, tables.n_states, cfg_training.states_per_batch):
            loss, parts = training_step(model, tables, order[batch_start:batch_start + cfg_training.states_per_batch],
                                        cfg_training, epinet_settings=epinet_settings)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
            optimizer.step()
            running.append(parts)
        scheduler.step()
        train_seconds = time.perf_counter() - start
        selection = greedy_selection_metrics(model, val_dataset, selection_indices, selection_embeddings,
                                             selection_labels, device)
        metrics = {"epoch": epoch, "train_seconds": train_seconds, "lr": scheduler.get_last_lr()[0],
                   **{key: float(np.mean([r[key] for r in running])) for key in running[0]},
                   "val_greedy": selection}
        with open(run_directory / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        overall = selection["overall"]
        print(f"Epoch {epoch}: loss {metrics['loss']:.4f} (card {metrics['loss_card']:.4f}, "
              f"G {metrics['loss_g']:.4f}, rank {metrics['loss_rank']:.4f}"
              + "".join(f", epinet {key[12:]} {value:.4f}" for key, value in metrics.items()
                        if key.startswith("loss_epinet_"))
              + f") | val greedy geomean ratio "
              f"{overall['geomean_ratio']:.4f}, p90 {overall['p90_ratio']:.4f}, optimal {overall['frac_optimal']:.1%}, "
              f"p95 {overall['p95_ratio']:.3f} | {train_seconds:.0f}s")
        save(run_directory / "last_model.pt", epoch, metrics)
        score = (overall[selection_metric], overall["geomean_ratio"])
        if score[0] < best[0] - 1e-5 or (abs(score[0] - best[0]) <= 1e-5 and score[1] < best[1] - 1e-5):
            best, best_epoch, best_overall, patience = score, epoch, overall, 0
            save(run_directory / "best_model.pt", epoch, metrics)
        else:
            patience += 1
            if patience >= cfg_training.early_stopping_patience:
                print(f"Early stop: no improvement for {patience} epochs.")
                break

    with open(run_directory / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_epoch, "selection_metric": selection_metric,
                   "best_val_greedy_geomean_ratio": best_overall["geomean_ratio"],
                   "best_val_greedy": best_overall,
                   "best_model": str(run_directory / "best_model.pt")}, f, indent=2)
    print(f"Best epoch {best_epoch}: val greedy {selection_metric} {best[0]:.4f}, geomean ratio {best[1]:.4f} "
          f"-> {run_directory / 'best_model.pt'}")


if __name__ == "__main__":
    main()
