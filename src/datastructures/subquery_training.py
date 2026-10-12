"""Training an estimator on sub-queries as the DP asks about them, with a censored loss.

Examples are full queries plus their counted sub-queries (src/datastructures/subquery_labels.py,
`merge` output). A sub-query is sliced from its featurised full query exactly as the DP does
(QueryGymEstimatedCost.reduced_form_query), on the fly in the DataLoader workers, so it carries the
same full-query context the DP's estimates see.

Censoring (log1p scale, L1 like the standard pretraining loss), per example:
    EXACT  |prediction - log1p(y)|
    LEFT   y = 0: the result is empty, any prediction <= 0 is right   -> max(0, prediction)
    RIGHT  an imputed count (the sub-query timed out): the true count is at least the imputed
           value, any prediction above it is right                     -> max(0, log1p(y) - prediction)
Without censoring every example is EXACT (the standard loss).
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch_geometric.data import Data


EXACT, LEFT, RIGHT = 0, 1, 2


def _key(query):
    return " ".join((query if isinstance(query, str) else query[0]).split())


class CardinalityExamples(torch.utils.data.Dataset):
    """Full queries (mask None) and sub-queries (mask = bitmask over the query's patterns) of a
    processed dataset, as minimal Data objects with y and a censoring flag."""

    def __init__(self, dataset, labels_path=None, include_full=True, censored=False):
        """`dataset` must be loaded with its node mappings (load_mappings=True); they are turned
        into compact per-pattern node-id arrays here and then dropped from the dataset, so forked
        DataLoader workers do not copy 100k+ Python dicts page by page."""
        self.dataset = dataset
        items = []                                         # (query index, mask or -1, count, censor)
        if include_full:
            for i in range(len(dataset)):
                y = float(dataset[i].y.view(-1)[0])
                items.append((i, -1, y, LEFT if censored and y == 0 else EXACT))
        self.n_full = len(items)
        index = {}
        pattern_nodes, self.offsets = [], np.zeros(len(dataset) + 1, dtype=np.int64)
        for i in range(len(dataset)):
            query = dataset[i]
            index[_key(query.query)] = i
            nodes = [(int(query.term_to_id[tp.split(" ")[0]]), int(query.term_to_id[tp.split(" ")[2]]))
                     for tp in query.triple_patterns]
            pattern_nodes.extend(nodes)
            self.offsets[i + 1] = self.offsets[i] + len(nodes)
        self.pattern_nodes = np.array(pattern_nodes, dtype=np.int64).reshape(-1, 2)
        if labels_path:
            with open(labels_path) as f:
                labels = json.load(f)
            missing = 0
            for query, entry in labels.items():
                i = index.get(_key(query))
                if i is None:
                    missing += 1
                    continue
                if list(dataset[i].triple_patterns) != list(entry["triple_patterns"]):
                    raise ValueError(f"pattern order of the labels differs from the dataset for {query[:80]}")
                imputed = set(entry.get("imputed", []))
                for mask, count in entry["counts"].items():
                    censor = EXACT
                    if censored:
                        censor = RIGHT if mask in imputed else LEFT if count == 0 else EXACT
                    items.append((i, int(mask), float(count), censor))
            if missing:
                print(f"[subqueries] {missing:,} labelled queries are not in the dataset (filtered out); skipped")
        self.n_sub = len(items) - self.n_full
        # numpy columns, not a list of Python tuples (forked workers would copy the tuples page by page)
        self.query_index = np.array([t[0] for t in items], dtype=np.int64)
        self.mask = np.array([t[1] for t in items], dtype=np.int64)
        self.count = np.array([t[2] for t in items], dtype=np.float64)
        self.censor = np.array([t[3] for t in items], dtype=np.int8)
        del items, index
        dataset.load_mappings, dataset.data_mappings = False, None     # the arrays replace the dicts

    def __len__(self):
        return len(self.mask)

    def slice(self, i, members):
        """The sub-query of query i on `members`, exactly as QueryGymEstimatedCost.reduced_form_query
        builds it (every node of the selected patterns once, in order of first occurrence), from the
        per-pattern node-id arrays instead of the term_to_id dict."""
        query = self.dataset[i]
        nodes = self.pattern_nodes[self.offsets[i]:self.offsets[i + 1]]
        x_index = list(dict.fromkeys(int(x) for x in nodes[members].reshape(-1)))
        new_id = {x: k for k, x in enumerate(x_index)}
        edges = torch.tensor(members).repeat_interleave(2) * 2 + torch.tensor([0, 1] * len(members))
        edge_index = query.edge_index[:, edges]
        mapped = torch.tensor([[new_id[int(v)] for v in edge_index[r]] for r in range(2)], dtype=torch.long)
        return query.x[torch.as_tensor(x_index)], mapped, query.edge_attr[edges]

    def __getitem__(self, index):
        i, mask = int(self.query_index[index]), int(self.mask[index])
        y, censor = float(self.count[index]), int(self.censor[index])
        if mask >= 0:
            x, edge_index, edge_attr = self.slice(i, [j for j in range(mask.bit_length()) if mask >> j & 1])
        else:
            query = self.dataset[i]
            x, edge_index, edge_attr = query.x, query.edge_index, query.edge_attr
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([y], dtype=torch.float32), censor=torch.tensor([censor], dtype=torch.long))


class EpochSampler(torch.utils.data.Sampler):
    """Per epoch: every full query and a fresh random `fraction` of the sub-queries, shuffled."""

    def __init__(self, examples, fraction=1.0, seed=0):
        self.n_full, self.n_sub, self.fraction = examples.n_full, examples.n_sub, float(fraction)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_full + int(round(self.n_sub * self.fraction))

    def __iter__(self):
        n_take = int(round(self.n_sub * self.fraction))
        sub = self.n_full + (self.rng.permutation(self.n_sub)[:n_take] if n_take < self.n_sub else np.arange(self.n_sub))
        order = self.rng.permutation(np.concatenate([np.arange(self.n_full), sub]))
        return iter(order.tolist())


def censored_l1(prediction, y, censor):
    target = torch.log1p(y)
    loss = torch.where(censor == LEFT, torch.relu(prediction),
                       torch.where(censor == RIGHT, torch.relu(target - prediction), (prediction - target).abs()))
    return loss.mean()


@torch.no_grad()
def subquery_metrics(model, loader, device, predict):
    """On sub-query examples: censored loss, median log10(estimate / true) on exact non-empty ones
    (negative = underestimation, the DP's 'falsely cheap' subsets), share of empty ones predicted < 1."""
    model.eval()
    predictions, ys, censors = [], [], []
    for batch in loader:
        predictions.append(predict(model, batch).cpu())
        ys.append(batch.y.view(-1))
        censors.append(batch.censor.view(-1))
    p, y, c = torch.cat(predictions), torch.cat(ys), torch.cat(censors)
    exact_nonempty = (c == EXACT) & (y > 0)
    empty = y == 0
    log10_ratio = (torch.log10(torch.expm1(p[exact_nonempty]).clamp(min=1.0)) - torch.log10(y[exact_nonempty]))
    return {
        "val_sub_loss": float(censored_l1(p, y, torch.where(empty, torch.full_like(c, LEFT), c))),
        "val_sub_median_log10_ratio": float(log10_ratio.median()) if exact_nonempty.any() else float("nan"),
        "val_sub_abs_median_log10_ratio": float(log10_ratio.median().abs()) if exact_nonempty.any() else float("nan"),
        "val_sub_share_100x_too_low": float((log10_ratio < -2).float().mean()) if exact_nonempty.any() else float("nan"),
        "val_sub_empty_below_one": float((torch.expm1(p[empty]) < 1.0).float().mean()) if empty.any() else float("nan"),
    }


SUBQUERY_FIELDS = [("val_sub_loss", "min"), ("val_sub_abs_median_log10_ratio", "min"),
                   ("val_sub_median_log10_ratio", "max"),          # signed, for the record (< 0: underestimation)
                   ("val_sub_share_100x_too_low", "min"), ("val_sub_empty_below_one", "max")]
