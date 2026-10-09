"""All preprocessing of a generated query workload for cardinality pretraining, with the same
features as mixed_yago_hll (feature_type predicate_edge_hll):

    python prepare_pretraining_data.py --dataset mixed_yago_v2 \\
        --endpoints http://localhost:7000,http://localhost:7001 \\
        --reuse-stats data/term_occurrences/mixed_yago --reuse-walks data/rdf2vec_embeddings/mixed_yago

Steps, each skipping what is already done (rerun after an interruption to continue):
  1. split      data/generated_queries/<dataset>/*.json -> dataset_train/raw, dataset_val/raw
                (split_raw_queries, as split_input_queries_train_val.py)
  2. stats      data/term_occurrences/<dataset>/: occurrences.json, tp_cardinalities.json,
                multiplicities.json, hll_sketches.json (the queries and functions of
                precompute_term_triple_pattern_cardinalities.py). Only missing values are computed;
                values found in --reuse-stats directories are taken over after a sample of them
                is recounted and matches. Every count is strict: a failed or timed-out count is
                retried and never stored (QLeverOptimizerClient would return 0 for it), and
                zeros found in existing files are recounted.
  3. embed      data/rdf2vec_embeddings/<dataset>/model.json: streaming_pyrdf2vec.py in
                .venv_rdf2vec, walking only the entities not yet walked (walked_entities.txt) and
                not covered by a --reuse-walks corpus (whose walks join the Word2Vec corpus).
  4. featurise  dataset_{train,val}/processed with QueryCardinalityDataset, after checking that
                every term, pattern and predicate has its statistic (no endpoint fallback).
  5. check      feature layout against --compare (mixed_yago_hll), and a pretraining config.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import rdflib

from src.datastructures.prepare_feature_data import calculate_hll_sketches, extract_multiplicities, read_queries
from src.supervised_value_estimation.amortized_dp.online.true_cardinalities import count_subsets
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset, split_raw_queries

ROOT = Path(__file__).resolve().parent
BATCH = 20_000                  # counts between checkpoints of a statistics file


def _load(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save(path, data):
    temporary = Path(f"{path}.tmp")
    with open(temporary, "w") as f:
        json.dump(data, f)
    os.replace(temporary, path)


def term_count_query(term):
    # the query of QLeverOptimizerClient.cardinality_term
    return (f"SELECT (COUNT(*) AS ?count) WHERE {{ {{ {term} ?p ?o }} UNION {{ ?s {term} ?o }} "
            f"UNION {{ ?s ?p {term} }} }}")


def pattern_count_query(pattern):
    # the query of QLeverOptimizerClient.cardinality_triple_pattern
    return f"SELECT (COUNT(*) AS ?count) WHERE {{ {pattern} }}"


# --- 1. split ---------------------------------------------------------------------------------

def split(dataset_dir, val):
    train, validation = dataset_dir / "dataset_train" / "raw", dataset_dir / "dataset_val" / "raw"
    if train.is_dir() and validation.is_dir() and any(train.iterdir()) and any(validation.iterdir()):
        print(f"[split] done: {train.parent}, {validation.parent}")
        return
    train.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    split_raw_queries(str(dataset_dir), val, str(train), str(validation))


# --- 2. statistics ----------------------------------------------------------------------------

def _take_over(name, keys, path, reuse_dirs):
    """Values of `keys` in this dataset's file, and those only found in reuse directories."""
    wanted = set(keys)
    values = {k: v for k, v in _load(path).items() if k in wanted}
    reused = {}
    for directory in reuse_dirs:
        other = _load(Path(directory) / name)
        for k in keys:
            if k not in values and k not in reused and k in other:
                reused[k] = other[k]
    return values, reused


def _count_all(name, queries_by_key, endpoints, timeout_s, values, path):
    """Strict counts of queries_by_key into values (checkpointed to path); stops on failures."""
    keys = sorted(queries_by_key)
    for attempt in range(2):
        failed = []
        for start in range(0, len(keys), BATCH):
            batch = keys[start:start + BATCH]
            counted = count_subsets([(k, queries_by_key[k]) for k in batch], endpoints, timeout_s)
            for k in batch:
                if counted.get(k) is None:
                    failed.append(k)
                else:
                    values[k] = counted[k]
            _save(path, values)
            print(f"[stats] {name}: {min(start + BATCH, len(keys)):,}/{len(keys):,} counted, "
                  f"{len(failed)} failed", flush=True)
        keys = failed
        if not keys:
            return
        print(f"[stats] {name}: retrying {len(keys)} failed counts")
    raise SystemExit(f"[stats] {name}: {len(keys)} counts failed twice (e.g. {keys[:3]}); "
                     f"nothing was stored for them. Check the endpoints and rerun.")


def count_statistic(name, keys, query_of, stats_dir, reuse_dirs, endpoints, timeout_s, verify, rng):
    path = stats_dir / name
    values, reused = _take_over(name, keys, path, reuse_dirs)
    if reused:
        pool = sorted(k for k, v in reused.items() if v != 0)
        sample = rng.sample(pool, min(verify, len(pool)))
        fresh = count_subsets([(k, query_of(k)) for k in sample], endpoints, timeout_s)
        wrong = [(k, reused[k], fresh.get(k)) for k in sample if fresh.get(k) != reused[k]]
        if wrong:
            raise SystemExit(f"[stats] {name}: {len(wrong)}/{len(sample)} reused values differ from QLever, "
                             f"e.g. {wrong[:3]}; not reusing {reuse_dirs}.")
        print(f"[stats] {name}: reusing {len(reused):,} values ({len(sample)} recounted, all equal)")
        values.update(reused)
    todo = {k: query_of(k) for k in keys if k not in values or values[k] == 0}
    zeros = sum(1 for k in todo if k in values)
    print(f"[stats] {name}: {len(keys) - len(todo):,}/{len(keys):,} known, counting {len(todo):,}"
          + (f" (incl. {zeros:,} zeros recounted)" if zeros else ""), flush=True)
    if todo:
        _count_all(name, todo, endpoints, timeout_s, values, path)
    _save(path, values)
    return values


def multiplicity_statistic(predicates, stats_dir, reuse_dirs, endpoint, verify, rng):
    name, path = "multiplicities.json", stats_dir / "multiplicities.json"
    values, reused = _take_over(name, predicates, path, reuse_dirs)
    if reused:
        for p in rng.sample(sorted(reused), min(verify, len(reused))):
            fresh = list(extract_multiplicities(endpoint, p)[p])
            if any(abs(a - b) > 1e-9 * max(1.0, abs(b)) for a, b in zip(fresh, reused[p])):
                raise SystemExit(f"[stats] {name}: reused value of {p} differs from QLever "
                                 f"({reused[p]} vs {fresh}); not reusing {reuse_dirs}.")
        print(f"[stats] {name}: reusing {len(reused)} predicates (sample recomputed, all equal)")
        values.update(reused)
    missing = [p for p in predicates if p not in values]
    for p in missing:
        values[p] = list(extract_multiplicities(endpoint, p)[p])       # raises if QLever fails
    print(f"[stats] {name}: {len(predicates)} predicates ({len(missing)} computed)")
    _save(path, values)
    return values


def _decode(sketch):
    return pickle.loads(base64.b64decode(sketch))


def hll_statistic(predicates, stats_dir, reuse_dirs, endpoint, occurrences, verify):
    name, path = "hll_sketches.json", stats_dir / "hll_sketches.json"
    values, reused = _take_over(name, predicates, path, reuse_dirs)

    def compute(missing):
        sketches = calculate_hll_sketches([{"triple_patterns": [f"?s {p} ?o ."]} for p in missing], endpoint)
        for p, sketch in sketches.items():        # an empty sketch means a failed DISTINCT query
            if _decode(sketch["domain"]).count() == 0 or _decode(sketch["range"]).count() == 0:
                raise SystemExit(f"[stats] {name}: empty sketch for {p} (failed query?)")
        return sketches

    if reused:
        # recompute the smallest predicates (cheap) and require identical registers
        sample = sorted(reused, key=lambda p: occurrences.get(p, 0))[:verify]
        fresh = compute(sample)
        for p in sample:
            for side in ("domain", "range"):
                if list(_decode(fresh[p][side]).reg) != list(_decode(reused[p][side]).reg):
                    raise SystemExit(f"[stats] {name}: reused {side} sketch of {p} differs from QLever; "
                                     f"not reusing {reuse_dirs}.")
        print(f"[stats] {name}: reusing {len(reused)} predicates ({len(sample)} recomputed, identical)")
        values.update(reused)
    missing = [p for p in predicates if p not in values]
    if missing:
        values.update(compute(missing))
    print(f"[stats] {name}: {len(predicates)} predicates ({len(missing)} computed)")
    _save(path, values)
    return values


def statistics(queries, stats_dir, args, endpoints):
    stats_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    terms = sorted({t.n3() for q in queries for tp in q["rdflib_patterns"] for t in tp
                    if not isinstance(t, rdflib.term.Variable)})
    patterns = sorted({tp for q in queries for tp in q["triple_patterns"]})
    predicates = sorted({tp.split()[1] for q in queries for tp in q["triple_patterns"]})
    print(f"[stats] {len(queries):,} queries: {len(terms):,} terms, {len(patterns):,} patterns, "
          f"{len(predicates)} predicates")
    occurrences = count_statistic("occurrences.json", terms, term_count_query, stats_dir, args.reuse_stats,
                                  endpoints, args.timeout, args.verify_reused, rng)
    count_statistic("tp_cardinalities.json", patterns, pattern_count_query, stats_dir, args.reuse_stats,
                    endpoints, args.timeout, args.verify_reused, rng)
    multiplicity_statistic(predicates, stats_dir, args.reuse_stats, endpoints[0], 5, rng)
    hll_statistic(predicates, stats_dir, args.reuse_stats, endpoints[0], occurrences, 2)


# --- 3. embeddings ----------------------------------------------------------------------------

def embeddings(queries, embedding_dir, args, endpoints):
    entities = sorted({str(t) for q in queries for tp in q["rdflib_patterns"] for t in tp
                       if isinstance(t, rdflib.term.URIRef)})
    model, entity_file = embedding_dir / "model.json", embedding_dir / "entities.json"
    if model.exists() and entity_file.exists() and set(_load(entity_file)) >= set(entities):
        print(f"[embed] done: {model}")
        return
    embedding_dir.mkdir(parents=True, exist_ok=True)
    with open(entity_file, "w") as f:
        json.dump(entities, f)
    extra_walks, covered = [], set()
    for directory in args.reuse_walks:
        walks = Path(directory) / "walks.txt"
        extra_walks.append(str(walks))
        covered |= set(entities) & set(_load(Path(directory) / "model.json"))
    skip_file = embedding_dir / "covered_by_reused_walks.json"
    with open(skip_file, "w") as f:
        json.dump(sorted(covered), f)
    print(f"[embed] {len(entities):,} entities, {len(covered):,} covered by reused walks {args.reuse_walks}")
    command = [str(ROOT / args.rdf2vec_python), str(ROOT / "streaming_pyrdf2vec.py"),
               "--endpoint", endpoints[0], "--endpoints", ",".join(endpoints),
               "--entities_file", str(entity_file), "--skip_entities_file", str(skip_file), "--resume",
               "--output", str(embedding_dir), "--num_walks", str(args.num_walks), "--depth", str(args.depth),
               "--dimensions", str(args.dimensions), "--epochs", str(args.epochs), "--workers", str(args.w2v_workers)]
    if extra_walks:
        command += ["--extra_walks", *extra_walks]
    subprocess.run(command, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})


# --- 4. featurise -----------------------------------------------------------------------------

class _NoFallback:
    """Query environment for the featuriser: every statistic must be precomputed."""

    def __getattr__(self, name):
        raise RuntimeError(f"featuriser asked the endpoint for {name}: a statistic is missing")


def featurise(queries, dataset_dir, stats_dir, embedding_dir):
    splits = [dataset_dir / "dataset_train", dataset_dir / "dataset_val"]
    if all((s / "processed" / "processed_queries.pt").exists() for s in splits):
        print(f"[featurise] done: {[str(s / 'processed') for s in splits]}")
        return
    occurrences, patterns = _load(stats_dir / "occurrences.json"), _load(stats_dir / "tp_cardinalities.json")
    multiplicities, hll = _load(stats_dir / "multiplicities.json"), _load(stats_dir / "hll_sketches.json")
    missing = {"occurrences": set(), "tp_cardinalities": set(), "multiplicities": set(), "hll": set()}
    for q in queries:
        for s, p, o in q["rdflib_patterns"]:
            key = f"{s.n3()} {p.n3()} {o.n3()} ."           # the featuriser's pattern key
            if key not in patterns:
                missing["tp_cardinalities"].add(key)
            for t in (s, p, o):
                if not isinstance(t, rdflib.term.Variable) and t.n3() not in occurrences:
                    missing["occurrences"].add(t.n3())
            if p.n3() not in multiplicities:
                missing["multiplicities"].add(p.n3())
            if f"<{p}>" not in hll:
                missing["hll"].add(p.n3())
    if any(missing.values()):
        raise SystemExit(f"[featurise] statistics missing: { {k: len(v) for k, v in missing.items()} }")
    load_queries_into_dataset(str(splits[0]), str(splits[1]), None, str(embedding_dir / "model.json"),
                              _NoFallback(), "predicate_edge_hll", load_mappings=False,
                              occurrences_location=str(stats_dir / "occurrences.json"),
                              tp_cardinality_location=str(stats_dir / "tp_cardinalities.json"),
                              multiplicity_location=str(stats_dir / "multiplicities.json"),
                              hll_location=str(stats_dir / "hll_sketches.json"),
                              shuffle_train=False)


# --- 5. check and config ----------------------------------------------------------------------

def _processed(split_dir):
    from src.datastructures.query_cardinality_dataset import QueryCardinalityDataset
    return QueryCardinalityDataset(root=str(split_dir), featurizer=None, load_mappings=False)


def check(dataset_dir, compare_dir, embedding_flag_index):
    report = {}
    for name, directory in (("new", dataset_dir), ("reference", compare_dir)):
        dataset = _processed(directory / "dataset_train")
        x, edge = dataset._data.x, dataset._data.edge_attr
        terms = x[:, 1] == 1.0                                 # constants carry 1.0 at index 1
        report[name] = {"queries": len(dataset), "x_dim": int(x.shape[1]), "edge_dim": int(edge.shape[1]),
                        "keys": sorted(dataset[0].keys()),
                        "constants_with_embedding": float((x[terms, embedding_flag_index] == 1.0).float().mean()),
                        "zero_y": int((dataset._data.y == 0).sum())}
    print(json.dumps(report, indent=2))
    new, reference = report["new"], report["reference"]
    if (new["x_dim"], new["edge_dim"], new["keys"]) != (reference["x_dim"], reference["edge_dim"], reference["keys"]):
        raise SystemExit("[check] feature layout differs from the reference dataset")
    print("[check] same feature layout as the reference dataset")


def write_config(args, dataset_dir, stats_dir, embedding_dir):
    template = ROOT / args.config_template
    target = template.with_name(f"{template.stem}_{args.dataset}.yaml")
    if target.exists():
        print(f"[config] exists: {target}")
        return
    text = template.read_text()
    for old, new in [(f"{os.path.relpath(ROOT / args.compare, ROOT)}/", f"{os.path.relpath(dataset_dir, ROOT)}/"),
                     ("data/rdf2vec_embeddings/mixed_yago/model.json", os.path.relpath(embedding_dir / "model.json", ROOT)),
                     ("data/term_occurrences/mixed_yago/", f"{os.path.relpath(stats_dir, ROOT)}/"),
                     ("experiments/experiment_outputs/mixed_yago/", f"experiments/experiment_outputs/{args.dataset}/")]:
        if old not in text:
            raise SystemExit(f"[config] template {template} has no {old}")
        text = text.replace(old, new)
    target.write_text(text)
    print(f"[config] wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Directory name under data/generated_queries.")
    parser.add_argument("--endpoints", required=True, help="Comma-separated QLever endpoints of the same data.")
    parser.add_argument("--reuse-stats", nargs="*", default=[], help="Statistics directories to take values from.")
    parser.add_argument("--reuse-walks", nargs="*", default=[], help="Embedding directories whose walks to reuse.")
    parser.add_argument("--compare", default="data/generated_queries/mixed_yago_hll",
                        help="Reference dataset directory for the feature check.")
    parser.add_argument("--data-dir", default="data", help="Holds generated_queries, term_occurrences, rdf2vec_embeddings.")
    parser.add_argument("--config-template", default="experiments/experiment_configs/pretraining_experiments/"
                                                     "pretrain_experiments_yago_mixed/pretrain_experiment_huge_oracle_graph_norm_hll.yaml")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--verify-reused", type=int, default=500, help="Reused counts recounted per statistic.")
    parser.add_argument("--seed", type=int, default=0)
    # rdf2vec, as mixed_yago (streaming_pyrdf2vec.py example command)
    parser.add_argument("--rdf2vec-python", default=".venv_rdf2vec/bin/python")
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--w2v-workers", type=int, default=5)
    parser.add_argument("--steps", default="split,stats,embed,featurise,check,config",
                        help="Comma-separated subset of split,stats,embed,featurise,check,config.")
    args = parser.parse_args()

    endpoints = [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    steps = set(args.steps.split(","))
    data_dir = ROOT / args.data_dir
    dataset_dir = data_dir / "generated_queries" / args.dataset
    stats_dir = data_dir / "term_occurrences" / args.dataset
    embedding_dir = data_dir / "rdf2vec_embeddings" / args.dataset

    if "split" in steps:
        split(dataset_dir, args.val)
    queries = None
    if steps & {"stats", "embed", "featurise"}:
        queries = read_queries(str(dataset_dir))         # top-level files: all queries, train and val
    if "stats" in steps:
        statistics(queries, stats_dir, args, endpoints)
    if "embed" in steps:
        embeddings(queries, embedding_dir, args, endpoints)
    if "featurise" in steps:
        featurise(queries, dataset_dir, stats_dir, embedding_dir)
    if "check" in steps:
        n_multiplicities = len(next(iter(_load(stats_dir / "multiplicities.json").values())))
        check(dataset_dir, ROOT / args.compare, 2 + n_multiplicities + 1 + 2)
    if "config" in steps:
        write_config(args, dataset_dir, stats_dir, embedding_dir)


if __name__ == "__main__":
    sys.exit(main())
