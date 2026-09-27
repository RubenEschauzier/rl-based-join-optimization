"""Does a log-compositional cardinality model beat a flat regressor on extrapolation?

The bias under test: cardinality is multiplicative, so in log space a join is

    logcard(L u R) = logcard(L) + logcard(R) - logsel(L, R),    logsel >= 0

A COMPOSITIONAL model takes the leaf cardinalities as given (you know them exactly) and
learns only the selectivity corrections. A FLAT model regresses logcard(S) directly and
must learn the magnitude too -- including for subset sizes it never saw in training.

Ground truth is the standard independence-plus-pairwise-correlation model, defined on
SETS so it is order-invariant by construction:

    logcard(S)    = sum_{i in S} c_i  -  sum_{i<j in S} s_ij
    logsel(L, R)  = sum_{i in L, j in R} s_ij            (the cross pairs)
    s_ij          = softplus(alpha * <k_i, k_j> + beta)  >= 0

Both models get the same inputs and comparable capacity. Train on small joins, test on
large ones. Run:

    python -m src.supervised_value_estimation.compositional_cardinality_prototype
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

KEY_DIM = 4


class SyntheticJoinWorld:
    """Relations with known cardinalities and a known pairwise-correlation structure."""

    def __init__(self, n_relations: int = 14, seed: int = 0, alpha: float = 2.0,
                 beta: float = -0.5):
        generator = torch.Generator().manual_seed(seed)
        # Base cardinalities spanning several orders of magnitude, as real tables do.
        self.log_cardinality = torch.rand(n_relations, generator=generator) * 10.0 + 2.0
        self.keys = nn.functional.normalize(
            torch.randn((n_relations, KEY_DIM), generator=generator), dim=-1
        )
        self.n_relations = n_relations
        # s_ij: how much joining i and j shrinks the cross product, in log space.
        self.pairwise = nn.functional.softplus(
            alpha * (self.keys @ self.keys.T) + beta
        )
        self.pairwise.fill_diagonal_(0.0)

    def true_log_cardinality(self, subset: tuple[int, ...]) -> float:
        index = torch.tensor(subset, dtype=torch.long)
        independent = self.log_cardinality[index].sum()
        correction = self.pairwise[index][:, index].sum() / 2.0   # each pair counted twice
        return float(independent - correction)

    def true_log_selectivity(self, left: tuple[int, ...], right: tuple[int, ...]) -> float:
        li = torch.tensor(left, dtype=torch.long)
        ri = torch.tensor(right, dtype=torch.long)
        return float(self.pairwise[li][:, ri].sum())

    def features(self, subset: tuple[int, ...]) -> torch.Tensor:
        """Sufficient-ish summary of a subplan: summed keys, count, and its own logcard."""
        index = torch.tensor(subset, dtype=torch.long)
        return torch.cat([
            self.keys[index].sum(dim=0),
            torch.tensor([float(len(subset))]),
        ])

    def sample_subsets(self, sizes, n_per_size, generator) -> list[tuple[int, ...]]:
        subsets = []
        for size in sizes:
            for _ in range(n_per_size):
                perm = torch.randperm(self.n_relations, generator=generator)[:size]
                subsets.append(tuple(sorted(perm.tolist())))
        return subsets


class FlatRegressor(nn.Module):
    """DeepSets over the members, regressing logcard(S) directly.

    Deliberately given each member's log-cardinality, so summing them is available to it;
    the question is whether it learns that, and whether it holds up at set sizes it never
    saw during training.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.member = nn.Sequential(
            nn.Linear(KEY_DIM + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, member_features, mask):
        # member_features: (batch, max_members, KEY_DIM + 1); mask: (batch, max_members)
        encoded = self.member(member_features) * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True)
        return self.readout(torch.cat([pooled, counts], dim=1)).squeeze(-1)


class CompositionalSelectivity(nn.Module):
    """Predicts only logsel(L, R) >= 0; magnitudes come from the leaves.

    Symmetric in its two arguments, because |L join R| = |R join L|. The elementwise
    product of the summed keys gives the bilinear term the true cross-pair sum needs.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * KEY_DIM + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, left_keys, left_count, right_keys, right_count):
        symmetric = torch.cat([
            left_keys + right_keys,
            left_keys * right_keys,
            (left_keys - right_keys).abs(),
            (left_count + right_count).unsqueeze(-1),
            (left_count * right_count).unsqueeze(-1),
        ], dim=-1)
        return nn.functional.softplus(self.net(symmetric).squeeze(-1))


def build_flat_batch(world, subsets, max_members):
    features = torch.zeros((len(subsets), max_members, KEY_DIM + 1))
    mask = torch.zeros((len(subsets), max_members))
    for row, subset in enumerate(subsets):
        index = torch.tensor(subset, dtype=torch.long)
        features[row, :len(subset), :KEY_DIM] = world.keys[index]
        features[row, :len(subset), KEY_DIM] = world.log_cardinality[index]
        mask[row, :len(subset)] = 1.0
    return features, mask


def build_split_batch(world, splits):
    """splits: list of (left, right) subsets -> the tensors CompositionalSelectivity wants."""
    left_keys = torch.stack([world.keys[torch.tensor(l, dtype=torch.long)].sum(0) for l, _ in splits])
    right_keys = torch.stack([world.keys[torch.tensor(r, dtype=torch.long)].sum(0) for _, r in splits])
    left_count = torch.tensor([float(len(l)) for l, _ in splits])
    right_count = torch.tensor([float(len(r)) for _, r in splits])
    return left_keys, left_count, right_keys, right_count


def random_split(subset, generator):
    """Split a subset into two non-empty halves, the way a join tree would."""
    members = list(subset)
    while True:
        assignment = torch.randint(0, 2, (len(members),), generator=generator)
        if 0 < int(assignment.sum()) < len(members):
            break
    left = tuple(sorted(m for m, a in zip(members, assignment) if a == 0))
    right = tuple(sorted(m for m, a in zip(members, assignment) if a == 1))
    return left, right


@torch.no_grad()
def compositional_predict(world, model, subset, generator):
    """Recursively compose: logcard(S) = logcard(L) + logcard(R) - logsel(L, R)."""
    if len(subset) == 1:
        return float(world.log_cardinality[subset[0]])
    left, right = random_split(subset, generator)
    lk, lc, rk, rc = build_split_batch(world, [(left, right)])
    correction = float(model(lk, lc, rk, rc)[0])
    return (compositional_predict(world, model, left, generator)
            + compositional_predict(world, model, right, generator)
            - correction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[2, 3, 4, 6, 8, 10, 12])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    world = SyntheticJoinWorld(seed=arguments.seed)
    generator = torch.Generator().manual_seed(arguments.seed + 1)
    max_members = max(arguments.test_sizes)

    train_subsets = world.sample_subsets(arguments.train_sizes, 400, generator)
    train_targets = torch.tensor([world.true_log_cardinality(s) for s in train_subsets])

    print(f"training both models on join sizes {arguments.train_sizes}, "
          f"{len(train_subsets)} subsets\n")

    # ---- flat regressor: predict logcard(S) directly ----
    flat = FlatRegressor()
    flat_optimizer = torch.optim.Adam(flat.parameters(), lr=1e-3)
    features, mask = build_flat_batch(world, train_subsets, max_members)
    for step in range(arguments.steps):
        flat_optimizer.zero_grad()
        loss = nn.functional.mse_loss(flat(features, mask), train_targets)
        loss.backward()
        flat_optimizer.step()
    flat_train_loss = float(loss)

    # ---- compositional: predict only the selectivity correction ----
    compositional = CompositionalSelectivity()
    comp_optimizer = torch.optim.Adam(compositional.parameters(), lr=1e-3)
    splits = [random_split(s, generator) for s in train_subsets]
    split_targets = torch.tensor([world.true_log_selectivity(l, r) for l, r in splits])
    lk, lc, rk, rc = build_split_batch(world, splits)
    for step in range(arguments.steps):
        comp_optimizer.zero_grad()
        loss = nn.functional.mse_loss(compositional(lk, lc, rk, rc), split_targets)
        loss.backward()
        comp_optimizer.step()
    comp_train_loss = float(loss)

    print(f"final train MSE   flat={flat_train_loss:.4f}   "
          f"compositional (on selectivities)={comp_train_loss:.4f}\n")

    header = f"{'join size':>10} {'trained?':>9} {'true logcard':>13} {'flat MAE':>10} {'comp MAE':>10}"
    print(header)
    print("-" * len(header))
    for size in arguments.test_sizes:
        subsets = world.sample_subsets([size], 120, generator)
        truth = torch.tensor([world.true_log_cardinality(s) for s in subsets])

        with torch.no_grad():
            features, mask = build_flat_batch(world, subsets, max_members)
            flat_error = (flat(features, mask) - truth).abs().mean()

        comp_predictions = torch.tensor([
            compositional_predict(world, compositional, s, generator) for s in subsets
        ])
        comp_error = (comp_predictions - truth).abs().mean()

        seen = "train" if size in arguments.train_sizes else "EXTRAP"
        print(f"{size:>10} {seen:>9} {truth.mean():>13.2f} "
              f"{flat_error:>10.3f} {comp_error:>10.3f}")

    # ---- does sharing a subexpression correlate the errors? ----
    print("\nerror correlation between plans that SHARE a subexpression:")
    shared, disjoint = [], []
    base = world.sample_subsets([4], 60, generator)
    for anchor in base:
        extension_a, extension_b = random_split(tuple(range(world.n_relations)), generator)
        left = tuple(sorted(set(anchor) | {extension_a[0]}))
        right = tuple(sorted(set(anchor) | {extension_b[0]}))
        if len(left) == len(anchor) or len(right) == len(anchor) or left == right:
            continue
        shared.append((left, right))
    # A genuinely disjoint control. Sampling a random subset is not good enough: two
    # size-5 subsets drawn from 14 relations overlap by ~1.8 members on average, so the
    # "unrelated" baseline would quietly share subexpressions too.
    for pair in shared[:40]:
        available = [i for i in range(world.n_relations) if i not in set(pair[0])]
        if len(available) < len(pair[0]):
            continue
        picked = torch.randperm(len(available), generator=generator)[:len(pair[0])]
        disjoint.append((pair[0], tuple(sorted(available[i] for i in picked.tolist()))))

    def error_pairs(pairs, predict):
        first = np.array([predict(a) - world.true_log_cardinality(a) for a, _ in pairs])
        second = np.array([predict(b) - world.true_log_cardinality(b) for _, b in pairs])
        if len(first) < 3 or first.std() < 1e-9 or second.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(first, second)[0, 1])

    def flat_predict(subset):
        with torch.no_grad():
            f, m = build_flat_batch(world, [subset], max_members)
            return float(flat(f, m)[0])

    def comp_predict(subset):
        return compositional_predict(world, compositional, subset, generator)

    print(f"   flat regressor : {error_pairs(shared, flat_predict):+.3f} (sharing) vs "
          f"{error_pairs(disjoint, flat_predict):+.3f} (unrelated)")
    print(f"   compositional  : {error_pairs(shared, comp_predict):+.3f} (sharing) vs "
          f"{error_pairs(disjoint, comp_predict):+.3f} (unrelated)")
    print("\nNote: this compares POINT-ERROR correlation, which is only a weak proxy for")
    print("the joint-predictive claim. That claim is about correlation between sampled")
    print("predictions under a shared epistemic index, and needs the epinet attached to")
    print("the corrections to test properly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
