"""A zero predictor on a frozen cardinality model: P(the join of a pattern subset is empty), used to STEER planning
(the planner's cost of U becomes (1 - P(U)) * card(U)); the cardinality estimate itself is never replaced.

    python -m src.supervised_value_estimation.amortized_dp.zero_predictor [overrides]

Structure (inductive bias: a join is empty when the values its patterns allow for some variable do not intersect):
    incidence   every pattern touching a variable v (in the subset's term graph) gives soft memberships
                m_e in (0, 1)^B over B learned "type bins", from the other endpoint (its raw features: the constant's
                query-conditioned profile vector and counts, or the neighbouring variable; plus its frozen 2-layer GINE
                embedding in the subset), the edge features (predicate vector, multiplicities, pattern count,
                direction) and v's own features; for a constant pattern (c p ?x / ?x p c, with `profiles`) AND-ed
                with the bins of the exact types of c's neighbours through (p, direction) (index data,
                src/contextual_embeddings/profiles.py): each type (characteristic set, multi-hot over the
                (predicate, direction) features) gets learned bin logits, and the pattern allows bin b as far as some
                neighbour type does (logsumexp over the types);
    propagate   (rounds > 0) learned arc consistency along the query's variable-variable edges: per round, each edge
                u -> v sends v soft memberships computed from u's current bin state WITHOUT v's own message to u
                (no echo, as in belief propagation) and the edge's features (predicate, direction, both endpoints);
                one message function shared by all rounds, so `rounds` = how far a restriction travels, e.g. the
                conflict between constants at the two ends of a chain c1 p ?x . ?x q ?y . ?y r c2;
    variable    AND over its incidences and incoming messages per bin (log q_vb = sum of log memberships), a soft
                OR over bins:  logit P(v non-empty) = a * logsumexp_b log q_vb + c,  a > 0;
    subset      AND over its variables:  P(U non-empty) = prod_v P(v non-empty).
This covers the hard-zero kinds at a variable: one constant + predicates (constant profile x predicate/direction),
two or more constants (two constant-made memberships must share a bin) and predicate-made zeros (predicate
memberships must share a bin). Zeros along edges need the propagation (without it, only the neighbour's 2-hop GINE
embedding reaches them).
Adding a pattern can only lower the bin products, but the GINE embeddings depend on the subset, so P is made
monotone over supersets at planning time (`inherit`: P(U) >= max over splits of its parts' P).

Data: the TRUE counts of the online plan-loss run (counts.pkl: the beam plans' subsets and their alternatives, i.e.
the subsets planning decides between) on the training queries; a fraction of those queries held out for model
selection; evaluation on the counts of the 2,000 validation queries. Timed-out counts are skipped. Loss: BCE on
"empty". Metrics: AUROC, average precision and recall at precision 0.9 / 0.95 / 0.99, on all counted join subsets
and on the MINIMAL empty ones (no counted empty proper subset: the zeros that must be predicted, supersets
inherit them), against the regression estimate as a score.

Outputs in output_directory/zero-predictor-<time>/: config.yaml, metrics.jsonl, zero_predictor.pt (best
held-out average precision on minimal empties).
"""
from __future__ import annotations

import json
import math
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from src.contextual_embeddings.profiles import ProfileTable
from src.supervised_value_estimation.amortized_dp import soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import load_backbone, load_split

VARIABLE = 5           # node feature column of the variable flag (one-hot variable / unknown / embedded constant)


def log1mexp(a):
    """log(1 - exp(a)) for a <= 0, stable at both ends."""
    a = a.clamp(max=-1e-7)
    return torch.where(a > -math.log(2), torch.log(-torch.expm1(a)), torch.log1p(-torch.exp(a)))


@torch.no_grad()
def backbone_pass(model, x, edge_index, edge_attr, batch, n_graphs):
    """(node representation [x, GINE embedding] per node, the model's log-card estimate per graph)."""
    convs = list(model.embedding_model.children())[:-1]
    h = x
    for conv in convs:
        h = conv(h, edge_index, edge_attr)
    phi = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, batch, h)
    head = model.heads[model.head_types.index("cardinality")]
    return torch.cat([x, h], 1), head(phi).reshape(-1)


class NeighbourTypes:
    """The types of a constant's neighbours through one (predicate, direction) feature (profile histograms), i.e. the
    types the variable at the other end of `c p ?x` (feature p) or `?x p c` (feature P + p) can have."""

    def __init__(self, directory, predicate_terms):
        self.directory, self.predicate_terms = str(directory), list(predicate_terms)
        self.table = ProfileTable(directory)
        self.n_predicates = len(predicate_terms)
        self.predicate = {f"<{t}>": i for i, t in enumerate(predicate_terms)}
        self._cache = {}

    def type_set(self, constant, feature):
        text = constant[1:-1] if constant.startswith("<") else ""
        suffix = text.rsplit("/", 1)[-1]
        if not suffix.isdigit():
            return None
        features, types, _ = self.table.rows(int(suffix))
        a, b = np.searchsorted(features, feature, "left"), np.searchsorted(features, feature, "right")
        return np.asarray(types[a:b], dtype=np.int64) if b > a else None

    def edge_sets(self, query):
        """Per edge of the query graph (2i: s -> o, 2i + 1: o -> s of pattern i): the type set when the source is a
        constant and the target a variable, else None."""
        key = query.query
        if key not in self._cache:
            out = []
            for pattern in query.triple_patterns:
                t = pattern.strip().rstrip(" .").split()
                s, p, o = t[0], t[1], " ".join(t[2:])
                pid = self.predicate.get(p)
                ok = pid is not None
                out.append(self.type_set(s, pid) if ok and not s.startswith("?") and o.startswith("?") else None)
                out.append(self.type_set(o, self.n_predicates + pid) if ok and not o.startswith("?") and s.startswith("?") else None)
            self._cache[key] = out
        return self._cache[key]


class ZeroPredictor(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, bins=64, rounds=0, n_types=0, n_type_features=0, init_membership=0.95):
        super().__init__()
        self.config = {"node_dim": node_dim, "edge_dim": edge_dim, "hidden": hidden, "bins": bins, "rounds": rounds,
                       "n_types": n_types, "n_type_features": n_type_features}
        self.rounds = rounds
        self.neighbour_types = None
        if n_types:
            self.register_buffer("type_features", torch.zeros(n_types, n_type_features, dtype=torch.uint8))
            self.type_bins = torch.nn.Sequential(
                torch.nn.Linear(n_type_features, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, bins))
            # starts uniform over bins (no restriction relative to other bins), out of sigmoid saturation
            torch.nn.init.zeros_(self.type_bins[-1].weight)
            torch.nn.init.zeros_(self.type_bins[-1].bias)
        if rounds:
            self.message = torch.nn.Sequential(
                torch.nn.Linear(2 * node_dim + edge_dim + bins, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, bins))
            torch.nn.init.zeros_(self.message[-1].weight)          # starts as "no restriction" (membership 0.99)
            torch.nn.init.constant_(self.message[-1].bias, math.log(99.0))
        self.incidence = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim + edge_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, bins))
        torch.nn.init.zeros_(self.incidence[-1].weight)
        torch.nn.init.constant_(self.incidence[-1].bias, math.log(init_membership / (1 - init_membership)))
        # soft OR over bins: logit P(v non-empty) = softplus(scale) * logsumexp_b log q_vb + offset (monotone in q)
        self.scale = torch.nn.Parameter(torch.tensor(math.log(math.e - 1)))
        self.offset = torch.nn.Parameter(torch.tensor(4.0 - math.log(bins)))
        self.register_buffer("mean", torch.zeros(2 * node_dim + edge_dim))
        self.register_buffer("std", torch.ones(2 * node_dim + edge_dim))

    def incidence_inputs(self, rep, edge_index, edge_attr):
        """Edge e = (u -> v) of pattern (s, p, o): v is the target (o for direction +1, s for -1)."""
        return torch.cat([rep[edge_index[0]], edge_attr, rep[edge_index[1]]], 1)

    def type_set_memberships(self, sets):
        """log memberships [n_sets, B] of the type sets: logsigmoid(logsumexp over the set's types of their bin logits)."""
        _, types, seg, n_sets = sets
        values = self.type_bins(self.type_features.float())[types]
        peak = torch.zeros(n_sets, values.shape[1], device=values.device).scatter_reduce(
            0, seg[:, None].expand_as(values), values, "amax", include_self=False).detach()
        total = torch.zeros_like(peak).index_add(0, seg, torch.exp(values - peak[seg]))
        return torch.nn.functional.logsigmoid(peak + torch.log(total))

    def forward(self, rep, edge_index, edge_attr, batch, n_graphs, sets=None):
        """log P(non-empty) per graph. sets: joint_batch's type sets of the constant patterns (None: not used)."""
        source, target = edge_index
        is_var = rep[:, VARIABLE] == 1
        on_variable = is_var[target]
        inputs = (self.incidence_inputs(rep, edge_index, edge_attr) - self.mean) / self.std
        log_m = torch.nn.functional.logsigmoid(self.incidence(inputs[on_variable]))
        if self.config["n_types"] and sets is not None and sets[3] > 0:
            edge_set = sets[0]
            has = edge_set >= 0
            extra = torch.zeros(len(target), log_m.shape[1], device=rep.device)
            extra[has] = self.type_set_memberships(sets)[edge_set[has]]
            log_m = log_m + extra[on_variable]
        log_q = torch.zeros(rep.shape[0], log_m.shape[1], device=rep.device).index_add(0, target[on_variable], log_m)
        if self.rounds:
            # edges 2i, 2i + 1 are pattern i's two directions (subset_batch), so the reverse of edge e is e ^ 1
            between = torch.nonzero(on_variable & is_var[source]).squeeze(1)
            message = torch.zeros(len(target), log_m.shape[1], device=rep.device)
            for _ in range(self.rounds):
                incoming = torch.zeros_like(log_q).index_add(0, target, message)
                state = log_q[source[between]] + incoming[source[between]] - message[between ^ 1]
                new = torch.nn.functional.logsigmoid(self.message(torch.cat([inputs[between], state.clamp(min=-20) / 5], 1)))
                message = message.index_copy(0, between, new)
            log_q = log_q + torch.zeros_like(log_q).index_add(0, target, message)
        logit = torch.nn.functional.softplus(self.scale) * torch.logsumexp(log_q, 1) + self.offset   # some bin survives at v
        log_full = torch.nn.functional.logsigmoid(logit)
        return torch.zeros(n_graphs, device=rep.device).index_add(0, batch[is_var], log_full[is_var])

    def save(self, path, extra=None):
        profiles = None if self.neighbour_types is None else {
            "directory": self.neighbour_types.directory, "predicate_terms": self.neighbour_types.predicate_terms}
        torch.save({"config": self.config, "state": self.state_dict(), "profiles": profiles, **(extra or {})}, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        net = cls(**checkpoint["config"])
        net.load_state_dict(checkpoint["state"])
        if checkpoint.get("profiles"):
            net.neighbour_types = NeighbourTypes(**checkpoint["profiles"])
        return net.to(device).eval()


def bce(log_full, empty):
    """-log P(label): empty -> log P(empty) = log1mexp(log P(non-empty))."""
    return -torch.where(empty, log1mexp(log_full), log_full).mean()


def joint_batch(items, device, neighbour_types=None):
    """items [(query Data, [masks])] -> the disjoint union of all their subset graphs, and with `neighbour_types` the
    type sets of the constant patterns: (set index per batched edge or -1, flat types, set of each type, number of
    sets), one set per (query, edge), shared by all subsets containing the edge."""
    xs, edges, attrs, batches, nodes, graphs = [], [], [], [], 0, 0
    edge_set, set_types, set_seg, n_sets = [], [], [], 0
    for data, masks in items:
        masks = torch.as_tensor(masks, device=device)
        x, ei, ea, b = soft_dp.subset_batch(data, masks, device)
        xs.append(x), edges.append(ei + nodes), attrs.append(ea), batches.append(b + graphs)
        nodes, graphs = nodes + x.shape[0], graphs + len(masks)
        if neighbour_types is not None:
            local = torch.full((2 * len(data.triple_patterns),), -1, dtype=torch.long)
            for e, types in enumerate(neighbour_types.edge_sets(data)):
                if types is not None:
                    local[e] = n_sets
                    set_types.append(torch.from_numpy(types))
                    set_seg.append(torch.full((len(types),), n_sets, dtype=torch.long))
                    n_sets += 1
            bits = ((masks[:, None] >> torch.arange(len(data.triple_patterns), device=device)) & 1).bool()
            _, original = torch.nonzero(bits.repeat_interleave(2, dim=1), as_tuple=True)     # as subset_batch
            edge_set.append(local.to(device)[original])
    sets = None
    if neighbour_types is not None:
        empty = torch.zeros(0, dtype=torch.long)
        sets = (torch.cat(edge_set), torch.cat(set_types or [empty]).to(device), torch.cat(set_seg or [empty]).to(device), n_sets)
    return torch.cat(xs), torch.cat(edges, 1), torch.cat(attrs), torch.cat(batches), graphs, sets


def predict(model, net, items, device):
    """(log-card estimate, log P(non-empty)) of every subset of items, in order."""
    x, ei, ea, b, n, sets = joint_batch(items, device, net.neighbour_types)
    rep, est = backbone_pass(model, x, ei, ea, b, n)
    return est, net(rep, ei, ea, b, n, sets)


@torch.no_grad()
def predict_subsets(model, net, query, masks, device, chunk=16384):
    """(log-card estimate, P(empty)) for all `masks` of one query (planning)."""
    est, p = [], []
    for start in range(0, len(masks), chunk):
        e, lf = predict(model, net, [(query, masks[start:start + chunk])], device)
        est.append(e), p.append(-torch.expm1(lf))
    return torch.cat(est), torch.cat(p)


def inherit(structure, p):
    """P of a subset = max of its own and of its parts' over all splits (bottom-up)."""
    p = p.clone()
    for subsets, u, s1, s2 in structure.levels:
        best = torch.zeros(len(structure), dtype=p.dtype, device=p.device).scatter_reduce(
            0, u, torch.maximum(p[s1], p[s2]), "amax")
        p[subsets] = torch.maximum(p[subsets], best[subsets])
    return p


def steered_logcard(structure, logcard, p_empty, cap=0.999):
    """The planner's log cost of each subset: log card + log(1 - P), P inherited over splits and capped, singletons
    unchanged. The estimate itself stays; only the planner's preference shifts."""
    p = inherit(structure, torch.where(structure.singleton, torch.zeros_like(p_empty), p_empty))
    return logcard + torch.log1p(-p.clamp(max=cap))


# --- data and evaluation -----------------------------------------------------------------------------------

def labelled_queries(counts, dataset, keys):
    """[(dataset index, masks, empty flags, minimal-empty flags)] of the counted join subsets of `keys`."""
    index = {" ".join(dataset[i].query.split()): i for i in range(len(dataset))}
    out = []
    for key in keys:
        c = {m: v for m, v in counts[key].items() if v is not None and m & (m - 1)}
        if not c:
            continue
        empties = [m for m, v in c.items() if v == 0]
        masks = list(c)
        minimal = [c[m] == 0 and not any(e != m and e & m == e for e in empties) for m in masks]
        out.append((index[key], masks, [c[m] == 0 for m in masks], minimal))
    return out


def recall_at_precision(y, score, precision):
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    prec = hits / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(prec >= precision)
    return float(hits[ok[-1]] / max(y.sum(), 1)) if len(ok) else 0.0


def metrics(y, score):
    return {"n": int(len(y)), "positives": int(y.sum()), "auroc": round(float(roc_auc_score(y, score)), 4),
            "ap": round(float(average_precision_score(y, score)), 4),
            **{f"recall@p{p}": round(recall_at_precision(y, score, p), 4) for p in (0.9, 0.95, 0.99)}}


@torch.no_grad()
def evaluate(model, net, dataset, queries, device, max_subsets):
    p, est, empty, minimal = [], [], [], []
    for group in batches(queries, max_subsets, shuffle=False):
        e, lf = predict(model, net, [(dataset[i], m) for i, m, _, _ in group], device)
        p.append(-torch.expm1(lf).cpu()), est.append(e.cpu())
        for _, _, z, mn in group:
            empty += z
            minimal += mn
    p, est = torch.cat(p).numpy(), torch.cat(est).numpy()
    empty, minimal = np.array(empty), np.array(minimal)
    rows = {"all": np.ones_like(empty), "minimal": minimal | ~empty}     # minimal empties vs all non-empty
    return {name: {"predictor": metrics(empty[r], p[r]), "estimate": metrics(empty[r], -est[r])}
            for name, r in rows.items()}


def batches(queries, max_subsets, shuffle=True, rng=None):
    order = list(range(len(queries)))
    if shuffle:
        rng.shuffle(order)
    group, size = [], 0
    for i in order:
        group.append(queries[i])
        size += len(queries[i][1])
        if size >= max_subsets:
            yield group
            group, size = [], 0
    if group:
        yield group


@hydra.main(version_base=None, config_path="../../../experiments/experiment_configs/amortized_dp",
            config_name="zero_predictor_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_directory) / f"zero-predictor-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {out} | device {device}", flush=True)
    model, epoch = load_backbone(cfg.estimator.config, cfg.estimator.run, cfg.estimator.select, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = pickle.load(open(cfg.data.counts, "rb"))
    val_keys = set(json.load(open(cfg.data.validation_keys)))
    train_set, val_set = load_split(cfg.data.train_dir), load_split(cfg.data.val_dir)
    train_keys = sorted(k for k in counts if k not in val_keys)
    rng.shuffle(train_keys)
    n_held = int(len(train_keys) * float(cfg.held_out_fraction))
    train = labelled_queries(counts, train_set, train_keys[n_held:])
    held = labelled_queries(counts, train_set, train_keys[:n_held])
    val = labelled_queries(counts, val_set, sorted(val_keys))
    for name, q in (("train", train), ("held out", held), ("validation", val)):
        n, e = sum(len(m) for _, m, _, _ in q), sum(sum(z) for _, _, z, _ in q)
        print(f"  {name}: {len(q):,} queries, {n:,} counted join subsets, {e:,} empty "
              f"({sum(sum(mn) for _, _, _, mn in q):,} minimal)", flush=True)

    with torch.no_grad():                                     # input standardisation from a sample
        sample = [(train_set[i], m) for i, m, _, _ in train[:200]]
        x, ei, ea, b, n, _ = joint_batch(sample, device)
        rep, _ = backbone_pass(model, x, ei, ea, b, n)
        neighbour_types, n_types, n_type_features = None, 0, 0
        if cfg.get("profiles"):
            terms = torch.load(cfg.profiles.contextualizer, map_location="cpu", weights_only=False)["predicate_terms"]
            neighbour_types = NeighbourTypes(cfg.profiles.directory, terms)
            n_types, n_type_features = neighbour_types.table.type_features.shape
        net = ZeroPredictor(rep.shape[1], ea.shape[1], int(cfg.hidden), int(cfg.bins), int(cfg.rounds),
                            n_types, n_type_features).to(device)
        if neighbour_types is not None:
            net.type_features.copy_(torch.from_numpy(neighbour_types.table.type_features))
            net.neighbour_types = neighbour_types
        if not bool((ea[0::2, -1] == 1).all() and (ea[1::2, -1] == -1).all()):
            raise ValueError("edges are not in (forward, reverse) pairs: the propagation's reverse index e ^ 1 is wrong")
        inputs = net.incidence_inputs(rep, ei, ea)
        net.mean.copy_(inputs.mean(0))
        net.std.copy_(inputs.std(0).clamp(min=1e-3))
    optimizer = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    record = {"epoch": 0, "held_out": evaluate(model, net, train_set, held, device, int(cfg.max_subsets))}
    print(f"[epoch 0] estimate as a score, held out: {record['held_out']['minimal']['estimate']}", flush=True)
    best = -1.0
    for epoch_index in range(1, int(cfg.epochs) + 1):
        start, losses = time.perf_counter(), []
        net.train()
        for group in batches(train, int(cfg.max_subsets), rng=rng):
            items = [(train_set[i], m) for i, m, _, _ in group]
            empty = torch.tensor([z for _, _, zs, _ in group for z in zs], device=device)
            minimal = torch.tensor([z for _, _, _, ms in group for z in ms], device=device)
            x, ei, ea, b, n, sets = joint_batch(items, device, net.neighbour_types)
            rep, _ = backbone_pass(model, x, ei, ea, b, n)
            log_full = net(rep, ei, ea, b, n, sets)
            weight = 1 + float(cfg.minimal_weight) * minimal.float()
            loss = (weight * -torch.where(empty, log1mexp(log_full), log_full)).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        net.eval()
        record = {"epoch": epoch_index, "loss": round(float(np.mean(losses)), 5), "seconds": round(time.perf_counter() - start, 1),
                  "held_out": evaluate(model, net, train_set, held, device, int(cfg.max_subsets)),
                  "validation": evaluate(model, net, val_set, val, device, int(cfg.max_subsets))}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        line = f"[epoch {epoch_index}] loss {record['loss']:.4f} ({record['seconds']:.0f}s)"
        for split in ("held_out", "validation"):
            for rows in ("all", "minimal"):
                m, e = record[split][rows]["predictor"], record[split][rows]["estimate"]
                line += (f"\n    {split} {rows}: AP {m['ap']:.3f} (estimate {e['ap']:.3f}), AUROC {m['auroc']:.3f} "
                         f"({e['auroc']:.3f}), recall@p0.95 {m['recall@p0.95']:.3f} ({e['recall@p0.95']:.3f}), "
                         f"recall@p0.99 {m['recall@p0.99']:.3f} ({e['recall@p0.99']:.3f})")
        print(line, flush=True)
        if record["held_out"]["minimal"]["predictor"]["ap"] > best:
            best = record["held_out"]["minimal"]["predictor"]["ap"]
            net.save(out / "zero_predictor.pt", {"epoch": epoch_index, "metrics": record})
    print(f"Done. Best held-out minimal-empty AP {best:.3f}; predictor in {out / 'zero_predictor.pt'}", flush=True)


if __name__ == "__main__":
    main()
