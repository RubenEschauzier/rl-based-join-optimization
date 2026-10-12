"""Pretraining corpus: query-shaped instances from the data graph and counterfactual negatives.

    python -m src.contextual_embeddings.corpus [hydra overrides]

An instance is a small query graph (one of the workload's shapes, `corpus.sizes` patterns)
embedded in the data graph by the workload generator's instantiator, with constants on some
nodes and variables elsewhere. For each positive (non-empty: the embedding is a witness) up to
two negatives are made by replacing ONE arm:
  near_miss  the generator's hard-zero constructions (hard_zero.HardZeroMaker): a constant
             swapped so that it still matches each other pattern at its variable but not all
             of them, or a predicate swapped to an anticorrelated one -- pairwise plausible,
             jointly empty;
  easy       a constant replaced by a random entity with the same predicate role.
Every negative is checked empty by exact counting (counting.ExactCounter); unverifiable ones
are dropped. Labels: `empty` per instance, `replaced` per pattern (the arms the replacement
touched). Shards of `corpus.shard_size` instances are written to `corpus.output_directory`;
a rerun continues after the last complete shard.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import time
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.random_query_generation.statistics_based_generation.counting import ExactCounter, PredicateIndex
from src.random_query_generation.statistics_based_generation.generate import _choke_points
from src.random_query_generation.statistics_based_generation.graph import load_graph
from src.random_query_generation.statistics_based_generation.hard_zero import HardZeroMaker
from src.random_query_generation.statistics_based_generation.instantiate import Instantiator, binding_plan
from src.random_query_generation.statistics_based_generation.shapes import build_shape
from src.random_query_generation.statistics_based_generation.statistics import load_statistics

_STATE = {}         # set before the worker pool forks (shared copy-on-write)


def encode(instance, empty, replaced, kind):
    """Compact instance: node entity ids (-1 for variables), oriented edges (u, v, predicate)."""
    constants = instance.plan.constants
    nodes = [int(instance.entities[n]) if n in constants else -1 for n in range(instance.shape.n_nodes)]
    edges = [(int(u), int(v), int(p)) for (u, v), p in zip(instance.oriented, instance.predicates)]
    return {"nodes": nodes, "edges": edges, "empty": int(empty), "replaced": [int(r) for r in replaced],
            "kind": kind, "shape": instance.shape.kind if hasattr(instance.shape, "kind") else None}


def _is_empty(instance):
    patterns = instance.structured_patterns()
    count = _STATE["counter"].count_exact(patterns, range(len(patterns)))
    return None if count is None else count == 0


def _replaced_arms(original, variant):
    changed_nodes = {n for n in range(original.shape.n_nodes) if original.entities[n] != variant.entities[n]}
    return [int(original.predicates[i] != variant.predicates[i] or bool({u, v} & changed_nodes))
            for i, (u, v) in enumerate(original.oriented)]


def _easy_negative(instance, rng):
    """A constant replaced by a random entity that has the predicate in the same role."""
    constants = list(instance.plan.constants)
    if not constants:
        return None
    node = constants[int(rng.integers(0, len(constants)))]
    incident = [i for i, e in enumerate(instance.oriented) if node in e]
    edge = incident[int(rng.integers(0, len(incident)))]
    p = int(instance.predicates[edge])
    hard = _STATE["hard_zero"]
    a, b = hard.predicate_ptr[p], hard.predicate_ptr[p + 1]
    pool = hard.by_predicate_subject if instance.oriented[edge][0] == node else hard.by_predicate_object
    variant = type(instance)(instance.shape, instance.entities.copy(), instance.predicates.copy(),
                             instance.plan, instance.oriented)
    variant.entities[node] = int(pool[int(rng.integers(a, b))])
    return variant


def make_instances(job):
    """Worker: [positive, negatives...] for one sampled shape, or []."""
    kind, n, choke_name, seed = job
    rng = np.random.default_rng(seed)
    choke_point = _STATE["choke_points"][choke_name][0]
    shape = build_shape(kind, n, rng)
    shape.kind = kind
    instance = _STATE["instantiators"][kind].embed(shape, binding_plan(shape, choke_point, rng), choke_point, rng)
    if instance is None:
        return []
    out = [encode(instance, False, [0] * len(instance.oriented), "positive")]
    settings = _STATE["settings"]
    if rng.random() < float(settings.near_miss_probability):
        variant = _STATE["hard_zero"].perturb(instance, rng)
        if variant is not None and _is_empty(variant) is True:
            out.append(encode(variant, True, _replaced_arms(instance, variant), f"near_miss_{variant.perturbation}"))
    if rng.random() < float(settings.easy_probability):
        variant = _easy_negative(instance, rng)
        if variant is not None and _is_empty(variant) is True:
            out.append(encode(variant, True, _replaced_arms(instance, variant), "easy"))
    return out


class _OverBudget(Exception):
    pass


def _on_alarm(signum, frame):
    raise _OverBudget()


def _init_worker():
    signal.signal(signal.SIGALRM, _on_alarm)


def make_instances_capped(job):
    """make_instances within corpus.job_time_budget seconds; None if over budget (a few cycle /
    flower instantiations and conditioned counts take minutes and would dominate the run)."""
    signal.setitimer(signal.ITIMER_REAL, float(_STATE["settings"].job_time_budget))
    try:
        return make_instances(job)
    except _OverBudget:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _setup(cfg):
    gen = OmegaConf.load(cfg.corpus.generation_config).generation
    graph = load_graph(gen.ntriples, gen.cache_directory)
    statistics = load_statistics(graph, gen.cache_directory)
    settings = gen.instantiation
    by_shape = dict(settings.get("max_restarts_by_shape") or {})
    counter = ExactCounter(graph, PredicateIndex(graph),
                           max_assignments=int(gen.certificate.get("max_conditioning_values", 2000)))
    # Fill the counter's per-predicate caches (degree messages, entity sets) once, before the
    # pool forks: otherwise every worker builds its own copy (hundreds of MB each) and the
    # workers run out of memory.
    for p in range(graph.n_predicates):
        for side in (True, False):
            counter._push(None, p, side)
            counter._entities_with(p, side)
    _STATE.update(
        graph=graph, counter=counter, settings=cfg.corpus,
        choke_points=_choke_points(gen.choke_points),
        instantiators={kind: Instantiator(graph, statistics, int(settings.max_pattern_matches),
                                          int(settings.max_candidates), int(by_shape.get(kind, settings.max_restarts)))
                       for kind in cfg.corpus.shapes},
        hard_zero=HardZeroMaker(graph, statistics, int(settings.max_pattern_matches),
                                float(gen.hard_zero.anticorrelation_gamma), counter=counter,
                                max_empty_pairs=int(gen.hard_zero.max_empty_pairs)),
    )


def build(cfg):
    out_dir = Path(cfg.corpus.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(out_dir.glob("shard_*.pt"))
    done = sum(len(torch.load(s, weights_only=False)) for s in shards)
    target = int(cfg.corpus.n_instances)
    if done >= target:
        print(f"[corpus] done: {done:,} instances in {out_dir}")
        return
    _setup(cfg)
    rng = np.random.default_rng(int(cfg.corpus.seed) + len(shards))
    seed_counter = iter(range((int(cfg.corpus.seed) + len(shards)) * 100_000_000, 10 ** 12))
    names = list(_STATE["choke_points"])
    weights = np.array([_STATE["choke_points"][c][1] for c in names], float)

    def jobs():
        while True:
            yield (str(rng.choice(list(cfg.corpus.shapes))), int(rng.choice(list(cfg.corpus.sizes))),
                   str(rng.choice(names, p=weights / weights.sum())), next(seed_counter))

    buffer, kinds, start = [], Counter(), time.perf_counter()
    shard_size = int(cfg.corpus.shard_size)
    with multiprocessing.get_context("fork").Pool(int(cfg.corpus.workers), initializer=_init_worker) as pool:
        for result in pool.imap_unordered(make_instances_capped, jobs(), chunksize=16):
            if result is None:
                kinds["over_budget"] += 1
                continue
            buffer.extend(result)
            kinds.update(r["kind"] for r in result)
            while len(buffer) >= shard_size:
                path = out_dir / f"shard_{len(shards):04d}.pt"
                temporary = path.with_suffix(".tmp")         # a shutdown mid-write leaves no broken shard
                torch.save(buffer[:shard_size], temporary)
                os.replace(temporary, path)
                shards.append(path)
                buffer = buffer[shard_size:]
                done += shard_size
                print(f"[corpus] {done:,}/{target:,} instances ({time.perf_counter() - start:.0f}s) {dict(kinds)}",
                      flush=True)
            if done >= target:
                pool.terminate()
                break
    print(f"[corpus] done: {done:,} instances in {out_dir}")


@hydra.main(version_base=None, config_path="../../experiments/experiment_configs/contextual_embeddings",
            config_name="contextualizer_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    build(cfg)


if __name__ == "__main__":
    main()
