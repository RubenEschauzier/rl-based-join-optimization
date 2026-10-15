"""An epinet on a frozen cardinality model: epistemic uncertainty of log-card estimates, for Thompson
sampling of plans and for choosing what to count (train_plan_loss.py).

    python -m src.supervised_value_estimation.amortized_dp.cardinality_epinet [overrides]

    f(U, z) = base(U) + learnable(phi(U), z) + alpha * prior(phi(U), z),     z ~ N(0, I_D)
with base = the model's log1p-card estimate, phi = its pooled graph embedding (the head's input,
standardised with the training features' mean / std), MLPEpinet (src/models/mlp_epinet.py) and alpha
set so the prior's spread over z is `prior_scale_target` (log units). The base stays frozen: phi and
base are computed once per example and cached, so training touches only the epinet.

Data: the pretraining examples (src/datastructures/subquery_training.py), i.e. every full query and
its QLever-counted sub-queries, censored as in pretraining (empty: only predictions above 0 are wrong;
imputed timeout: only predictions below it are wrong). Loss: that censored L1 on the perturbed targets
log1p(y) + sigma * (z . c_U) (epinet_loss.epinet_term_loss), anchors c_U fixed per (query, subset).

Evaluation every epoch (the step-0 check: is the uncertainty where the errors are?), on
    val   the validation split's counted sub-queries (exact, non-censored ones);
    plans the TRUE counts of the plan subsets of the 500 online-validation queries (optional cache):
the Spearman correlation of std(U) with |base(U) - log1p(y)|, the error-retention curve (mean |error|
after discarding the most uncertain x%; its area relative to random = 1 and to the ideal ordering by
the error itself), and the coverage of the 5-95% interval of the samples.

Outputs in output_directory/cardinality-epinet-<time>/: config.yaml, cache_*.pt (features), metrics.jsonl,
epinet.pt (last) and epinet_best.pt (best val retention area).
"""
from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from src.models.mlp_epinet import MLPEpinet
from src.supervised_value_estimation.amortized_dp import soft_dp
from src.supervised_value_estimation.amortized_dp.bushy_train import load_backbone, load_split
from src.utils.epinet_utils.epinet_loss import anchor_vectors, epinet_term_loss, stable_key

EXACT, LEFT, RIGHT = 0, 1, 2          # as subquery_training.py


def _key(query):
    return " ".join((query if isinstance(query, str) else query[0]).split())


@torch.no_grad()
def features_and_base(model, x, edge_index, edge_attr, batch):
    """(phi (S, F): the cardinality head's input, base (S,): the model's log-card estimate)."""
    phi = model.embedding_model.forward(x, edge_index, edge_attr, batch)
    head = model.heads[model.head_types.index("cardinality")]
    return phi, head(phi).reshape(-1)


@torch.no_grad()
def subset_features(model, items, device, max_subsets=16384):
    """items: [(query Data, [masks])] -> (phi, base) for all their subsets in order, in batched forwards."""
    phis, bases, group, size = [], [], [], 0

    def flush():
        xs, edges, attrs, batches, nodes, graphs = [], [], [], [], 0, 0
        for data, masks in group:
            x, ei, ea, b = soft_dp.subset_batch(data, torch.tensor(masks, device=device), device)
            xs.append(x), edges.append(ei + nodes), attrs.append(ea), batches.append(b + graphs)
            nodes, graphs = nodes + x.shape[0], graphs + len(masks)
        phi, base = features_and_base(model, torch.cat(xs), torch.cat(edges, 1), torch.cat(attrs), torch.cat(batches))
        phis.append(phi.half().cpu()), bases.append(base.float().cpu())
        group.clear()
    for data, masks in items:
        group.append((data, masks))
        size += len(masks)
        if size >= max_subsets:
            flush()
            size = 0
    if group:
        flush()
    return torch.cat(phis), torch.cat(bases)


def build_cache(model, dataset, labels_path, device, include_full=True):
    """The pretraining examples of a split with the frozen model's phi and base: a dict of tensors."""
    with open(labels_path) as f:
        labels = json.load(f)
    rows, items = [], []                                # rows: (key, count, censor, n, size)
    for i in range(len(dataset)):
        data = dataset[i]
        q, n = _key(data.query), len(data.triple_patterns)
        masks, entry = [], labels.get(q)
        if include_full:
            y = float(data.y.view(-1)[0])
            masks.append((1 << n) - 1)
            rows.append((stable_key(q, (1 << n) - 1), y, LEFT if y == 0 else EXACT, n, n))
        if entry is not None and list(entry["triple_patterns"]) == list(data.triple_patterns):
            imputed = set(entry.get("imputed", []))
            for mask, count in entry["counts"].items():
                m = int(mask)
                if include_full and m == (1 << n) - 1:
                    continue
                masks.append(m)
                rows.append((stable_key(q, m), float(count), RIGHT if mask in imputed else LEFT if count == 0 else EXACT,
                             n, m.bit_count()))
        if masks:
            items.append((data, masks))
        if (i + 1) % 20000 == 0:
            print(f"  listed {i + 1:,} queries, {len(rows):,} examples", flush=True)
    start = time.perf_counter()
    phi, base = subset_features(model, items, device)
    print(f"  features of {len(rows):,} examples in {time.perf_counter() - start:.0f}s", flush=True)
    key, y, censor, n, size = (np.array(c) for c in zip(*rows))
    return {"phi": phi, "base": base, "key": torch.from_numpy(key.astype(np.int64)), "y": torch.from_numpy(y).float(),
            "censor": torch.from_numpy(censor.astype(np.int8)), "n": torch.from_numpy(n.astype(np.int8)),
            "size": torch.from_numpy(size.astype(np.int8))}


def plan_count_cache(model, dataset, counts_path, device):
    """The true counts of the plan subsets of the online-validation queries ({query: {mask: count}})."""
    with open(counts_path, "rb") as f:
        counts = pickle.load(f)
    index = {_key(dataset[i].query): i for i in range(len(dataset))}
    rows, items = [], []
    for q, table in counts.items():
        if q not in index:
            continue
        data = dataset[index[q]]
        masks = [m for m, c in table.items() if c is not None and m & (m - 1)]
        if masks:
            items.append((data, masks))
            rows += [(stable_key(q, m), float(table[m]), LEFT if table[m] == 0 else EXACT, len(data.triple_patterns),
                      m.bit_count()) for m in masks]
    phi, base = subset_features(model, items, device)
    key, y, censor, n, size = (np.array(c) for c in zip(*rows))
    return {"phi": phi, "base": base, "key": torch.from_numpy(key.astype(np.int64)), "y": torch.from_numpy(y).float(),
            "censor": torch.from_numpy(censor.astype(np.int8)), "n": torch.from_numpy(n.astype(np.int8)),
            "size": torch.from_numpy(size.astype(np.int8))}


class CardinalityEpinet(torch.nn.Module):
    """The epinet with its input standardisation and prior weight; sample(phi, base, z) -> (K, S)."""

    def __init__(self, feature_dim, index_dim, hidden_dim, prior_hidden_dim, feature_mean, feature_std, alpha=1.0):
        super().__init__()
        self.epinet = MLPEpinet(feature_dim, index_dim, ["card"], hidden_dim, prior_hidden_dim)
        self.register_buffer("mean", feature_mean.float())
        self.register_buffer("std", feature_std.float())
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.kwargs = dict(feature_dim=feature_dim, index_dim=index_dim, hidden_dim=hidden_dim, prior_hidden_dim=prior_hidden_dim)

    @property
    def index_dim(self):
        return self.epinet.epi_index_dim

    def inputs(self, phi):
        return (phi.float() - self.mean) / self.std

    def sample(self, phi, base, indexes):
        return self.epinet.sample(base.float(), self.inputs(phi), indexes, "card", float(self.alpha))

    def save(self, path, extra=None):
        torch.save({"kwargs": self.kwargs, "state_dict": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device):
        saved = torch.load(path, map_location=device, weights_only=False)
        state = saved["state_dict"]
        net = cls(**saved["kwargs"], feature_mean=state["mean"], feature_std=state["std"])
        net.load_state_dict(state)
        return net.to(device)


def censored_l1(prediction, target, censor):
    return torch.where(censor == LEFT, torch.relu(prediction),
                       torch.where(censor == RIGHT, torch.relu(target - prediction), (prediction - target).abs())).mean()


@torch.no_grad()
def evaluate(net, cache, device, n_indexes=256, seed=1, chunk=4096):
    """Step-0 metrics on the exact examples of a cache."""
    keep = cache["censor"] == EXACT
    phi, base, y = cache["phi"][keep], cache["base"][keep], cache["y"][keep]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n_indexes, net.index_dim, device=device, generator=g)
    stds, lo, hi = [], [], []
    for s in range(0, len(base), chunk):
        samples = net.sample(phi[s:s + chunk].to(device), base[s:s + chunk].to(device), z)
        stds.append(samples.std(0).cpu())
        q = torch.quantile(samples, torch.tensor([0.05, 0.95], device=device), dim=0)
        lo.append(q[0].cpu()), hi.append(q[1].cpu())
    std, lo, hi = torch.cat(stds).numpy(), torch.cat(lo).numpy(), torch.cat(hi).numpy()
    target = torch.log1p(y).numpy()
    err = np.abs(base.numpy() - target)
    fractions = np.linspace(0.1, 1.0, 10)

    def retention(order):        # mean |error| of the most certain fraction, relative to all
        sorted_err = err[order]
        return np.array([sorted_err[:max(1, int(round(f * len(err))))].mean() for f in fractions]) / err.mean()
    curve = retention(np.argsort(std, kind="stable"))
    ideal = retention(np.argsort(err, kind="stable"))
    area, ideal_area = float(curve.mean()), float(ideal.mean())
    return {"n": int(len(err)), "spearman": float(spearmanr(std, err).correlation), "mean_std": float(std.mean()),
            "mean_abs_error": float(err.mean()), "coverage_90": float(np.mean((target >= lo) & (target <= hi))),
            "retention_area": round(area, 4), "retention_area_ideal": round(ideal_area, 4),
            # 0 = no better than random (area 1), 1 = the ideal ordering
            "retention_skill": round((1 - area) / (1 - ideal_area), 4) if ideal_area < 1 else float("nan"),
            "retention_curve": [round(float(c), 3) for c in curve]}


@hydra.main(version_base=None, config_path="../../../experiments/experiment_configs/amortized_dp",
            config_name="cardinality_epinet_mixed_yago_v2.yaml")
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.output_directory) / f"cardinality-epinet-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Run directory: {out} | device {device}", flush=True)
    model, epoch = load_backbone(cfg.estimator.config, cfg.estimator.run, cfg.estimator.select, device)

    caches = {}
    for split in ("train", "val"):
        path = Path(cfg.cache_directory) / f"cache_{split}.pt" if cfg.get("cache_directory") else None
        if path is not None and path.exists():
            caches[split] = torch.load(path)
            print(f"Loaded {split} cache {path}: {len(caches[split]['y']):,} examples", flush=True)
            continue
        print(f"Building the {split} cache (frozen model, epoch {epoch})...", flush=True)
        dataset = load_split(cfg.data[f"{split}_dir"])
        caches[split] = build_cache(model, dataset, cfg.data[f"labels_{split}"], device)
        torch.save(caches[split], out / f"cache_{split}.pt")
        if split == "val" and cfg.data.get("plan_counts") and Path(cfg.data.plan_counts).exists():
            caches["plans"] = plan_count_cache(model, dataset, cfg.data.plan_counts, device)
            torch.save(caches["plans"], out / "cache_plans.pt")
        del dataset
    if "plans" not in caches and cfg.get("cache_directory") and (Path(cfg.cache_directory) / "cache_plans.pt").exists():
        caches["plans"] = torch.load(Path(cfg.cache_directory) / "cache_plans.pt")
    for split, c in caches.items():
        print(f"  {split}: {len(c['y']):,} examples ({int((c['censor'] == EXACT).sum()):,} exact, "
              f"{int((c['censor'] == LEFT).sum()):,} empty, {int((c['censor'] == RIGHT).sum()):,} timed out)", flush=True)

    train = caches["train"]
    phi = train["phi"].float()
    net = CardinalityEpinet(phi.shape[1], int(cfg.epinet.index_dim), int(cfg.epinet.hidden_dim), int(cfg.epinet.prior_hidden_dim),
                            phi.mean(0), phi.std(0).clamp(min=1e-3)).to(device)
    sample = net.inputs(phi[torch.randperm(len(phi))[:20000]].to(device))
    width = net.epinet.prior_scale(sample, n_indexes=64, head_name="card")
    net.alpha.fill_(float(cfg.epinet.prior_scale_target) / width)
    print(f"Prior width {width:.3f} -> alpha {float(net.alpha):.3f} (prior spread {cfg.epinet.prior_scale_target} log units)", flush=True)
    del phi, sample

    optimizer = torch.optim.AdamW(net.epinet.get_learnable_epinet_params(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    n, batch, k = len(train["y"]), int(cfg.batch_size), int(cfg.epinet.n_indexes_train)
    anchors = anchor_vectors(train["key"].numpy(), net.index_dim, int(cfg.seed))          # (n, D), fixed per example
    data = {name: train[name].to(device) for name in ("phi", "base", "y", "censor")}
    anchors = anchors.to(device)
    best = -math.inf
    for epoch_index in range(1, int(cfg.epochs) + 1):
        start, losses = time.perf_counter(), []
        net.train()
        for s in range(0, n, batch):
            idx = torch.randint(0, n, (batch,), device=device)
            z = torch.randn(k, net.index_dim, device=device)
            censor = data["censor"][idx].long().repeat(k)
            loss, _, _ = epinet_term_loss(net.epinet, "card", data["base"][idx].unsqueeze(1), net.inputs(data["phi"][idx]),
                                          torch.log1p(data["y"][idx]), z, anchors[idx], float(cfg.epinet.sigma),
                                          float(net.alpha), lambda p, t: censored_l1(p, t, censor))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        net.eval()
        record = {"epoch": epoch_index, "loss": round(float(np.mean(losses)), 5), "seconds": round(time.perf_counter() - start, 1)}
        for split in ("val", "plans"):
            if split in caches:
                record[split] = evaluate(net, caches[split], device)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        line = f"[epoch {epoch_index}] loss {record['loss']:.4f} ({record['seconds']:.0f}s)"
        for split in ("val", "plans"):
            if split in record:
                m = record[split]
                line += (f" | {split}: spearman(std,|err|) {m['spearman']:.3f}, retention skill {m['retention_skill']:.3f}, "
                         f"coverage90 {m['coverage_90']:.2f}, mean std {m['mean_std']:.3f}")
        print(line, flush=True)
        net.save(out / "epinet.pt", {"epoch": epoch_index, "metrics": record})
        if record["val"]["retention_skill"] > best:
            best = record["val"]["retention_skill"]
            net.save(out / "epinet_best.pt", {"epoch": epoch_index, "metrics": record})
    print(f"Done. Best val retention skill {best:.3f}; epinet in {out / 'epinet_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
