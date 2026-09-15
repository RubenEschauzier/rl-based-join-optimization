"""Fast failing-first diagnostics for the epinet training stack.

Seven independent checks, each with a PASS/FAIL verdict, the whole suite running
in a few minutes on CPU. They exercise the *production* objects
(``MultiHeadEpistemicNetwork``, ``loss_epinet``, ``GaussianJointLogLoss``,
``compute_calibration_measures``, ``DualMetricScheduler``) rather than
re-implementations, so a fix in ``src/`` is immediately visible here.

Run from the repository root::

    python -m src.supervised_value_estimation.epinet_diagnostics_testbed
    python -m src.supervised_value_estimation.epinet_diagnostics_testbed --check gap

The checks, and what each one is a proxy for in the real YAGO run:

1. ``prior``       The randomized-prior members must be that many *different*
                   functions. If they collapse, ``alpha_ensemble * z^T P(x)``
                   has almost no input-dependent structure and the epinet has
                   no epistemic signal to keep or cancel.

2. ``schedule``    ``DualMetricScheduler`` must not annihilate the learning
                   rate while the loss is still improving.

3. ``gap``         The one that matters. Trains the real learnable epinet with
                   the real ``loss_epinet`` on 1-D regression with a held-out
                   input gap. A working ENN is *sharp where it has data and
                   wide where it does not*, and its spread must shrink as data
                   accumulates. Decomposes the prediction into the learnable and
                   the two prior terms, which is the only way to tell "learned
                   nothing" from "cancelled the prior" -- they are identical in
                   ``val_sharpness``.

4. ``monotonic``   Uncertainty must degrade *gracefully* with distance from the
                   training manifold, not erratically. Sweeps continuously
                   outward and checks the binned profile never drops materially.
                   A drop means some region further from the data is predicted
                   MORE confidently than a nearer one -- confident-and-wrong,
                   the failure mode that costs a planner a timeout. Distance is
                   measured in FEATURE space, since that is all the epinet sees.

5. ``selective``   The thesis metric. Sorts points by predicted std, discards
                   the most uncertain, and reports the MSE of what remains --
                   the risk-coverage curve -- scored against random and oracle
                   orderings. Answers "does this model know when it is wrong" on
                   the distribution you care about, with no held-out split
                   needed. Reported twice: on a mixed set (easy -- a few far
                   points dominate) and on the in-distribution set alone, which
                   is the honest number and the one matching an i.i.d. split.

6. ``epistemic``   Splits in-distribution error into the part the hypothesis
                   class can never represent (approximation, irreducible at any
                   sample size) and the part more data would remove
                   (estimation). Scores the epinet only against the second --
                   the fair test, since an epistemic measure is not supposed to
                   flag irreducible error. The base predictor is linear in fixed
                   features, so the converged fit has a closed form.

7. ``metrics``     Feeds an analytically perfect predictive distribution into
                   the validation metrics. Calibration error must be ~0,
                   sharpness must equal the noise variance, and the "excess"
                   NLL (the KL-to-optimal proxy) must be ~0 per target.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.epistemic_neural_network import prepare_epinet_model  # noqa: E402
from src.pretrain_procedure import DualMetricScheduler  # noqa: E402
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import (  # noqa: E402
    loss_epinet,
)
from src.utils.epinet_utils.calibration_plot import (  # noqa: E402
    calculate_calibration_metrics,
    compute_calibration_measures,
)
from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss  # noqa: E402
from src.utils.tree_conv_utils import (  # noqa: E402
    apply_features_to_structure,
    get_shared_structure,
    precompute_left_deep_tree_conv_index,
    precompute_left_deep_tree_node_mask,
)

FULL_MODEL_CONFIG = (
    "experiments/model_configs/policy_networks/"
    "t_cv_repr_separate_head_own_embeddings_hll.yaml"
)
PRIOR_MODEL_CONFIG = "experiments/model_configs/prior_networks/prior_t_cv_smallest_hll.yaml"

# Mirrors experiments/experiment_configs/.../..._train_epinet.yaml
INDEX_DIM = 30
MLP_DIMENSION = 64
EPINET_HIDDEN_DIM = 50
EPINET_FEATURE_MODE = "mlp_plus_plan"
SIGMA = 0.1
ALPHA_MLP = 1.0
ALPHA_ENSEMBLE = 1.0
PRIOR_SCALE_TARGET = 1.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
N_EPI_INDEXES_TRAIN = 32
N_EPI_INDEXES_EVAL = 256

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def build_epinet(device: torch.device, index_dim: int = INDEX_DIM,
                 feature_mode: str = EPINET_FEATURE_MODE):
    """The exact construction used by main_supervised_value_estimation."""
    heads_config = {"plan_cost": {"layer": nn.Linear(MLP_DIMENSION, 1)}}
    heads_config_prior = {"plan_cost": {"layer": nn.Linear(5, 1)}}
    return prepare_epinet_model(
        full_gnn_config=str(REPOSITORY_ROOT / FULL_MODEL_CONFIG),
        config_ensemble_prior=str(REPOSITORY_ROOT / PRIOR_MODEL_CONFIG),
        epinet_index_dim=index_dim,
        mlp_dimension=MLP_DIMENSION,
        heads_config=heads_config,
        heads_config_prior=heads_config_prior,
        device=device,
        model_weights=None,
        cost_only=True,
        freeze_embedding=True,
        epinet_feature_mode=feature_mode,
        epinet_hidden_dim=EPINET_HIDDEN_DIM,
    )


# --------------------------------------------------------------------------- #
# 1. Randomized-prior ensemble diversity
# --------------------------------------------------------------------------- #
def check_prior_diversity(device: torch.device) -> None:
    print("\n[1/7] randomized-prior ensemble diversity")
    torch.manual_seed(0)
    model = build_epinet(device)
    members = list(model.ensemble_combined_prior_models)

    heads = [m.query_plan_model.heads["plan_cost"] for m in members]
    n_distinct_heads = len({id(h.weight) for h in heads})
    record(
        "prior heads are distinct modules",
        n_distinct_heads == len(heads),
        f"{n_distinct_heads}/{len(heads)} distinct output-head weight tensors",
    )

    # Drive the real PlanCostEstimatorTiny stack on synthetic triple embeddings.
    torch.manual_seed(0)
    in_channels = members[0].query_plan_model.plan_embedding_nn[0].weights.in_channels
    # Needs comfortably more plans than ensemble members, or the rank is capped by the
    # number of columns rather than by the diversity we are trying to measure.
    n_triples, n_plans = 8, 4 * len(members)
    node_features = torch.randn((n_triples, in_channels), device=device)
    generator = torch.Generator().manual_seed(0)
    join_orders = [
        torch.randperm(n_triples, generator=generator).tolist() for _ in range(n_plans)
    ]

    gather_indices, indexes, masks = get_shared_structure(
        join_orders,
        n_triples,
        precompute_left_deep_tree_conv_index(20),
        precompute_left_deep_tree_node_mask(20),
        device,
    )
    trees = apply_features_to_structure(node_features, gather_indices)

    with torch.no_grad():
        prior_values = torch.stack(
            [
                m.query_plan_model(trees, indexes, masks)[0]["plan_cost"].view(-1)
                for m in members
            ]
        )  # (index_dim, n_plans) -- the exact interface loss_epinet consumes

    centred = prior_values - prior_values.mean(dim=1, keepdim=True)
    scale = centred.norm(dim=1, keepdim=True).clamp_min(1e-12)
    correlation = (centred / scale) @ (centred / scale).T
    off_diagonal = correlation[~torch.eye(len(members), dtype=torch.bool, device=device)]
    mean_abs_correlation = off_diagonal.abs().mean().item()

    singular_values = torch.linalg.svdvals(centred)
    effective_rank = (singular_values.sum() ** 2 / (singular_values**2).sum()).item()

    record(
        "prior members are decorrelated",
        mean_abs_correlation < 0.5,
        f"mean |corr| between members = {mean_abs_correlation:.3f} (want < 0.5)",
    )
    record(
        "prior ensemble spans many directions",
        effective_rank > 0.5 * len(members),
        f"effective rank = {effective_rank:.2f} / {len(members)}",
    )
    print(
        f"       per-plan prior std over z: "
        f"{prior_values.std(dim=0).mean().item():.4f} "
        f"(x alpha_ensemble={ALPHA_ENSEMBLE} -> "
        f"{ALPHA_ENSEMBLE * prior_values.std(dim=0).mean().item():.4f} "
        f"vs target noise sigma={SIGMA})"
    )


# --------------------------------------------------------------------------- #
# 2. Learning-rate schedule
# --------------------------------------------------------------------------- #
def _recorded_loss_series() -> tuple[list[float], list[float], str]:
    """A still-improving loss curve at the scale this project actually trains at.

    Deliberately synthetic rather than read from experiment_outputs: a recorded run
    reflects whatever config produced it, so asserting against one turns this into a test
    of a historical experiment instead of a test of the scheduler.
    """
    epochs = 60
    train = [0.20 * math.exp(-0.02 * e) for e in range(epochs)]
    val = [0.38 * math.exp(-0.02 * e) for e in range(epochs)]
    return train, val, "synthetic curve, still improving at every epoch"


def check_learning_rate_schedule() -> None:
    print("\n[2/7] learning-rate schedule")
    train_losses, val_losses, source = _recorded_loss_series()
    print(f"       replaying {len(train_losses)} epochs from {source}")

    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=LEARNING_RATE)
    # Keep in step with the construction in train_simulated_epinet_cached.
    scheduler = DualMetricScheduler(
        optimizer,
        patience=5,
        threshold=1e-3,
        threshold_mode="rel",
        factor=0.3,
        min_lr=LEARNING_RATE * 1e-3,
    )

    first_collapse_epoch = None
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        scheduler.step(train_loss=train_loss, val_metric=val_loss)
        if first_collapse_epoch is None and scheduler.get_last_lr() < LEARNING_RATE / 100:
            first_collapse_epoch = epoch

    final_lr = scheduler.get_last_lr()
    loss_scale = float(np.median(train_losses))
    effective_threshold = (
        scheduler.threshold * loss_scale
        if scheduler.threshold_mode == "rel"
        else scheduler.threshold
    )
    record(
        "plateau threshold is on the scale of the loss",
        effective_threshold < 0.2 * loss_scale,
        f"threshold={scheduler.threshold:g} ({scheduler.threshold_mode}) -> requires a "
        f"{effective_threshold:.2e} drop against a median train loss of {loss_scale:.4f} "
        f"(an absolute threshold near the loss itself makes every epoch a 'plateau')",
    )
    record(
        "learning rate survives the run",
        final_lr >= LEARNING_RATE / 100,
        f"lr {LEARNING_RATE:g} -> {final_lr:.3e} after {len(train_losses)} epochs"
        + (
            f"; fell below lr/100 at epoch {first_collapse_epoch}"
            if first_collapse_epoch
            else ""
        ),
    )


# --------------------------------------------------------------------------- #
# 3. In-distribution vs out-of-distribution uncertainty
# --------------------------------------------------------------------------- #
def _random_feature_map(inputs: torch.Tensor, device: torch.device,
                        width: int = MLP_DIMENSION) -> torch.Tensor:
    """Stand-in for the frozen tree-conv representation handed to the epinet.

    The first MLP_DIMENSION columns play the role of the cost head's input; any further
    columns play the role of the wider pre-MLP plan representation, matching the layout
    of `epinet_features = cat([mlp_out, combined])`.
    """
    generator = torch.Generator().manual_seed(11)
    frequencies = torch.randn((width // 2, 1), generator=generator) * 1.2
    phases = torch.rand((width // 2, 1), generator=generator) * 2 * math.pi
    projected = inputs.view(1, -1) * frequencies + phases
    features = torch.cat([torch.cos(projected), torch.sin(projected)], dim=0).T
    return features.to(device=device, dtype=torch.float32).contiguous()


# Amplitude of alpha_ensemble * z^T P(x) on the real prior ensemble, as measured by
# check_prior_diversity, with the reference alpha_ensemble = 1.0. Only the *ratio*
# against the MLP prior matters once prior_scale_target rescales both.
PRODUCTION_PRIOR_AMPLITUDE = 0.45


@dataclasses.dataclass
class FittedEpinet:
    """A trained 1-D epinet plus everything needed to interrogate it."""
    model: object
    features_of: object      # Tensor[x] -> Tensor[n, feature_dim]
    decompose: object        # Tensor[features] -> dict of per-term std + samples
    latent: object           # Tensor[x] -> noiseless target
    train_inputs: torch.Tensor
    train_features: torch.Tensor
    alpha_mlp: float
    alpha_ensemble: float
    sigma: float
    final_loss: float

    def samples(self, inputs: torch.Tensor) -> np.ndarray:
        """(K, n) predictive samples at the given inputs."""
        return self.decompose(self.features_of(inputs))["samples"]

    def std(self, inputs: torch.Tensor) -> np.ndarray:
        return self.samples(inputs).std(axis=0)

    def feature_distance(self, inputs: torch.Tensor) -> np.ndarray:
        """Distance to the nearest training point, measured in FEATURE space.

        Input-space distance is the wrong ruler here: the random Fourier map is periodic,
        so a large |x - x_train| does not imply the epinet sees anything unfamiliar. The
        epinet only ever sees phi(x), so that is where "distance from the training
        manifold" has to be measured.
        """
        with torch.no_grad():
            query = self.features_of(inputs)
            return torch.cdist(query, self.train_features).min(dim=1).values.cpu().numpy()


def _fit_1d_epinet(
    device: torch.device,
    prior_amplitude: float,
    steps: int,
    n_train: int = 80,
    sigma: float = SIGMA,
    prior_scale_target: float | None = PRIOR_SCALE_TARGET,
    density_ratio: float = 1.0,
) -> FittedEpinet:
    """Fit the real learnable epinet with the real loss on gapped 1-D regression.

    ``prior_amplitude`` fixes the ensemble prior's scale *relative* to the MLP prior, at
    the ratio measured on the real model. ``prior_scale_target`` then rescales both alphas
    by a common factor so the *combined* prior standard deviation lands on that value --
    the same calibration train_simulated_epinet_cached performs, so this experiment tests
    the configuration production actually runs.
    """
    torch.manual_seed(3)
    # `density_ratio` > 1 makes the left island denser than the right at equal width, so
    # in-distribution difficulty VARIES. Without it both islands are fit essentially
    # perfectly and there is no in-distribution error left for uncertainty to rank --
    # which makes any selective-prediction score on that region pure noise.
    n_left = max(int(round(n_train * density_ratio / (1.0 + density_ratio))), 2)
    n_right = max(n_train - n_left, 2)

    # Training support is two islands; the gap and the tails are never observed.
    left = torch.linspace(-3.0, -1.2, n_left)
    right = torch.linspace(1.2, 3.0, n_right)
    train_inputs = torch.cat([left, right])
    gap_inputs = torch.linspace(-0.8, 0.8, 25)
    tail_inputs = torch.cat([torch.linspace(-6.0, -4.0, 12), torch.linspace(4.0, 6.0, 12)])

    def latent(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(1.3 * x) + 0.25 * x

    model = build_epinet(device)
    width = model.epinet_feature_dim

    train_targets = (latent(train_inputs) + sigma * torch.randn(train_inputs.shape)).to(device)
    train_features = _random_feature_map(train_inputs, device, width)
    gap_features = _random_feature_map(gap_inputs, device, width)
    tail_features = _random_feature_map(tail_inputs, device, width)

    for parameter in model.parameters():
        parameter.requires_grad = False
    base_head = model.cost_estimation_model.query_plan_model.heads["plan_cost"]
    base_head.weight.requires_grad = True
    base_head.bias.requires_grad = True
    for parameter in model.get_learnable_epinet_params():
        parameter.requires_grad = True

    # The cost head reads only the first MLP_DIMENSION columns, as in the real model.
    def base_estimate(features: torch.Tensor) -> torch.Tensor:
        return base_head(features[:, :MLP_DIMENSION])

    # Linear randomized prior in the (index_dim, n) interface loss_epinet expects,
    # scaled so its pre-calibration amplitude matches the real ensemble's.
    prior_generator = torch.Generator().manual_seed(5)
    prior_weights = torch.randn((INDEX_DIM, width), generator=prior_generator).to(device)
    raw_amplitude = (prior_weights @ train_features.T).norm(dim=0).mean().item()
    prior_weights *= prior_amplitude / (ALPHA_ENSEMBLE * raw_amplitude)

    def prior_values(features: torch.Tensor) -> torch.Tensor:
        return prior_weights @ features.T  # (index_dim, n)

    alpha_mlp, alpha_ensemble = ALPHA_MLP, ALPHA_ENSEMBLE
    if prior_scale_target:
        with torch.no_grad():
            indexes = model.sample_epistemic_indexes_batched(N_EPI_INDEXES_EVAL)
            n = train_features.shape[0]
            combined_prior = (
                alpha_mlp * model.compute_mlp_prior_batched(train_features, indexes)["plan_cost"]
                + alpha_ensemble * (indexes @ prior_values(train_features)).reshape(-1, 1)
            ).reshape(N_EPI_INDEXES_EVAL, n)
            measured = combined_prior.std(dim=0).mean().item()
        rescale = prior_scale_target / measured
        alpha_mlp *= rescale
        alpha_ensemble *= rescale

    trainable = [base_head.weight, base_head.bias] + list(model.get_learnable_epinet_params())
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE * 3, weight_decay=WEIGHT_DECAY)
    mse = nn.MSELoss(reduction="mean")
    generator = torch.Generator(device=device)

    # loss_epinet only reads plan[1] (target) and plans[0][2] (anchor seed).
    plans = [(None, float(target), 12345) for target in train_targets.tolist()]
    train_priors = prior_values(train_features)

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss, _, _ = loss_epinet(
            train_priors,
            model,
            mse,
            base_estimate(train_features),
            train_features,
            plans,
            N_EPI_INDEXES_TRAIN,
            sigma,
            alpha_mlp,
            alpha_ensemble,
            generator,
            device,
        )
        total_loss.backward()
        optimizer.step()

    @torch.no_grad()
    def decompose(features: torch.Tensor) -> dict[str, object]:
        """std over z of each additive term, and of their sum.

        The learnable term is supposed to *cancel* the prior where there is data and fail
        to cancel it where there is not. Reporting the terms separately is the only way to
        tell "the epinet learned nothing" (learnable ~ 0) from "the epinet cancelled the
        prior" (learnable ~ prior, sum << either) -- they look identical in val_sharpness.
        """
        n = features.shape[0]
        indexes = model.sample_epistemic_indexes_batched(N_EPI_INDEXES_EVAL)
        learnable = model.compute_learnable_mlp_batched(features, indexes)["plan_cost"]
        mlp_prior = alpha_mlp * model.compute_mlp_prior_batched(features, indexes)["plan_cost"]
        ensemble = alpha_ensemble * (indexes @ prior_values(features)).reshape(-1, 1)

        reshape = lambda t: t.reshape(N_EPI_INDEXES_EVAL, n)
        epistemic = reshape(learnable + mlp_prior + ensemble)
        return {
            "learnable": float(reshape(learnable).std(dim=0).mean()),
            "mlp_prior": float(reshape(mlp_prior).std(dim=0).mean()),
            "ens_prior": float(reshape(ensemble).std(dim=0).mean()),
            "std": float(epistemic.std(dim=0).mean()),
            "samples": (base_estimate(features).view(1, n) + epistemic).cpu().numpy(),
        }

    model.eval()
    return FittedEpinet(
        model=model,
        features_of=lambda x: _random_feature_map(x, device, width),
        decompose=decompose,
        latent=latent,
        train_inputs=train_inputs,
        train_features=train_features,
        alpha_mlp=alpha_mlp,
        alpha_ensemble=alpha_ensemble,
        sigma=sigma,
        final_loss=total_loss.item(),
    )


def _run_gap_experiment(device, prior_amplitude, steps, n_train=80,
                        sigma=SIGMA, prior_scale_target=PRIOR_SCALE_TARGET):
    """Backwards-compatible wrapper: fit, then report the support/gap/tail decomposition."""
    fit = _fit_1d_epinet(device, prior_amplitude, steps, n_train, sigma, prior_scale_target)
    gap_inputs = torch.linspace(-0.8, 0.8, 25)
    tail_inputs = torch.cat([torch.linspace(-6.0, -4.0, 12), torch.linspace(4.0, 6.0, 12)])
    parts = {
        "support": fit.decompose(fit.train_features),
        "gap": fit.decompose(fit.features_of(gap_inputs)),
        "tail": fit.decompose(fit.features_of(tail_inputs)),
    }
    alpha_mlp, alpha_ensemble, total_loss = fit.alpha_mlp, fit.alpha_ensemble, fit

    # Same quantity validate_cached reports as val_sharpness.
    from src.utils.epinet_utils.calibration_plot import (
        calculate_predicted_distribution_variance,
        calculate_sharpness,
    )

    result = {"final_loss": fit.final_loss, "alpha_mlp": alpha_mlp,
              "alpha_ensemble": alpha_ensemble, "sigma": fit.sigma}
    for name, part in parts.items():
        for key in ("std", "learnable", "mlp_prior", "ens_prior"):
            result[f"{name}_{key}"] = part[key]
        result[f"{name}_sharpness"] = float(
            calculate_sharpness(
                calculate_predicted_distribution_variance(part["samples"].T, 0.2)
            )
        )
    return result


def check_gap_uncertainty(device: torch.device, steps: int = 1500) -> None:
    print("\n[3/7] uncertainty must grow away from the training data")
    print(
        "       fitting the real learnable epinet with the real loss_epinet on "
        "gapped 1-D regression"
    )

    outcomes = {}
    for n_train in (80, 800):
        outcome = _run_gap_experiment(device, PRODUCTION_PRIOR_AMPLITUDE, steps, n_train)
        outcomes[n_train] = outcome
        print(
            f"\n       N_train = {n_train}   sigma = {outcome['sigma']}   "
            f"prior calibrated to {PRIOR_SCALE_TARGET} "
            f"(alpha_mlp={outcome['alpha_mlp']:.4f}, "
            f"alpha_ensemble={outcome['alpha_ensemble']:.4f})"
        )
        header = f"{'where':>9} {'learnable':>10} {'mlp_prior':>10} {'ens_prior':>10} {'total':>8} {'sharpness':>10}"
        print("        " + header)
        for where in ("support", "gap", "tail"):
            print(
                f"        {where:>9} {outcome[f'{where}_learnable']:>10.4f} "
                f"{outcome[f'{where}_mlp_prior']:>10.4f} {outcome[f'{where}_ens_prior']:>10.4f} "
                f"{outcome[f'{where}_std']:>8.4f} {outcome[f'{where}_sharpness']:>10.4f}"
            )

    outcome = outcomes[800]
    ratio = outcome["gap_std"] / max(outcome["support_std"], 1e-12)
    prior_on_support = outcome["support_mlp_prior"] + outcome["support_ens_prior"]

    record(
        "uncertainty widens inside the input gap",
        ratio > 2.0,
        f"gap/support std ratio = {ratio:.2f} (want > 2)",
    )
    record(
        "the learnable epinet cancels the prior on support",
        outcome["support_std"] < 0.5 * prior_on_support,
        f"total std {outcome['support_std']:.4f} vs prior std {prior_on_support:.4f} on "
        f"support (learnable={outcome['support_learnable']:.4f}); no cancellation means "
        f"the epinet is not learning, and the 'uncertainty' is just the frozen prior",
    )
    record(
        "the two prior terms are within an order of magnitude",
        0.1 <= outcome["support_mlp_prior"] / max(outcome["support_ens_prior"], 1e-12) <= 10.0,
        f"alpha_mlp*mlp_prior={outcome['support_mlp_prior']:.4f} vs "
        f"alpha_ensemble*ens_prior={outcome['support_ens_prior']:.4f}; a term 10x smaller "
        f"contributes almost nothing however expensive it was to compute",
    )
    record(
        "uncertainty contracts as data grows",
        outcomes[800]["support_std"] < 0.9 * outcomes[80]["support_std"],
        f"support std {outcomes[80]['support_std']:.4f} (N=80) -> "
        f"{outcomes[800]['support_std']:.4f} (N=800); a flat value means the posterior "
        f"never concentrates, so 'knows what it knows' fails even in distribution",
    )
    record(
        "sharpness carries usable signal",
        (outcome["gap_sharpness"] - outcome["support_sharpness"])
        > 0.25 * outcome["support_sharpness"],
        f"sharpness moves {outcome['support_sharpness']:.4f} -> "
        f"{outcome['gap_sharpness']:.4f} between support and gap "
        f"(a flat val_sharpness curve is the symptom)",
    )


# --------------------------------------------------------------------------- #
# 4. Uncertainty must grow monotonically with distance from the training manifold
# --------------------------------------------------------------------------- #
def check_uncertainty_monotonicity(device: torch.device, steps: int = 1500) -> None:
    print("\n[4/7] uncertainty must degrade gracefully, not erratically")
    print("       sweeping continuously outward from the training islands")

    fit = _fit_1d_epinet(device, PRODUCTION_PRIOR_AMPLITUDE, steps, n_train=800)

    # Dense sweep spanning deep inside the islands out to well beyond them.
    grid = torch.linspace(-8.0, 8.0, 400)
    distance = fit.feature_distance(grid)
    std = fit.std(grid)

    # Spearman: monotone *association*, robust to the nonlinear distance/std relation.
    distance_rank = np.argsort(np.argsort(distance))
    std_rank = np.argsort(np.argsort(std))
    spearman = float(np.corrcoef(distance_rank, std_rank)[0, 1])

    # Binned profile: the assertion the planner actually cares about. A dip means some
    # region further from the data is predicted MORE confidently than a nearer one.
    n_bins = 8
    edges = np.quantile(distance, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_index = np.clip(np.digitize(distance, edges) - 1, 0, n_bins - 1)
    profile = np.array([std[bin_index == b].mean() for b in range(n_bins)])

    print(f"       {'bin':>4} {'mean dist':>10} {'mean std':>9}")
    for b in range(n_bins):
        print(f"       {b:>4} {distance[bin_index == b].mean():>10.3f} {profile[b]:>9.4f}")

    # Feature-space distance SATURATES: the random Fourier map puts every phi(x) on a
    # sphere of fixed radius, so once inputs are far enough apart they are all equally
    # unfamiliar and uncertainty should plateau rather than keep climbing. Tiny wobbles
    # on that plateau are sampling noise, not the failure mode. Only a drop that is large
    # relative to the overall spread indicates a region predicted MORE confidently than a
    # genuinely nearer one.
    drops = np.diff(profile)
    tolerance = 0.10 * (profile.max() - profile.min())
    worst_drop = float(drops.min()) if len(drops) else 0.0
    n_material = int((drops < -tolerance).sum())

    record(
        "uncertainty is monotone in distance from the data",
        n_material == 0,
        f"{n_material}/{len(drops)} bins where uncertainty drops by more than "
        f"{tolerance:.4f} (10% of its range) as you move further from the data; "
        f"worst drop {worst_drop:+.4f}. A material drop is the confident-and-wrong mode",
    )
    record(
        "uncertainty tracks distance from the data",
        spearman > 0.8,
        f"Spearman(distance, std) = {spearman:.3f} (want > 0.8)",
    )


# --------------------------------------------------------------------------- #
# 5. Selective prediction: does the model know when it is wrong?
# --------------------------------------------------------------------------- #
def _risk_coverage(errors: np.ndarray, ordering: np.ndarray) -> tuple[np.ndarray, float]:
    """MSE of the retained points as the most-uncertain ones are discarded first.

    `ordering` ranks points most-suspect-first. Returns the curve over retention
    fractions and its area (lower is better).
    """
    ranked_errors = errors[ordering]
    coverages = np.linspace(0.1, 1.0, 10)
    risks = np.array([
        ranked_errors[-max(1, int(round(c * len(errors)))):].mean() for c in coverages
    ])
    return risks, float(np.trapz(risks, coverages) / (coverages[-1] - coverages[0]))


def check_selective_prediction(device: torch.device, steps: int = 1500) -> None:
    print("\n[5/7] selective prediction: does predicted std rank the model's own errors?")

    # Uneven density across the two islands, so in-distribution error genuinely varies
    # and there is something for the uncertainty to rank.
    fit = _fit_1d_epinet(device, PRODUCTION_PRIOR_AMPLITUDE, steps, n_train=800,
                         density_ratio=12.0)

    # A realistic mixed evaluation set: mostly in-distribution, with the gap and tails
    # mixed in, so the retention curve has something to separate.
    generator = torch.Generator().manual_seed(21)
    in_distribution = torch.cat([
        torch.rand(150, generator=generator) * 1.8 - 3.0,
        torch.rand(150, generator=generator) * 1.8 + 1.2,
    ])
    harder = torch.cat([
        torch.rand(60, generator=generator) * 1.6 - 0.8,
        torch.rand(40, generator=generator) * 2.0 + 4.0,
    ])
    def evaluate(inputs: torch.Tensor, n_neighbours: int = 15):
        samples = fit.samples(inputs)
        predicted_mean = samples.mean(axis=0)
        predicted_std = samples.std(axis=0)
        truth = fit.latent(inputs).cpu().numpy()
        squared_error = (predicted_mean - truth) ** 2

        # Attainable ceiling. Sorting by *realized* squared error is not a fair oracle:
        # even with s(x) known exactly, the error at a point is a random draw of that
        # scale, so no model can reproduce the realized ordering. The best any model can
        # do is rank by E[e^2 | x], estimated here by averaging squared error over each
        # point's nearest neighbours in feature space, which averages the draw away.
        with torch.no_grad():
            features = fit.features_of(inputs)
            k = min(n_neighbours, len(inputs))
            neighbours = torch.cdist(features, features).topk(k, largest=False).indices
        expected_squared_error = squared_error[neighbours.cpu().numpy()].mean(axis=1)

        curve, area = _risk_coverage(squared_error, np.argsort(-predicted_std))
        _, ceiling_area = _risk_coverage(squared_error, np.argsort(-expected_squared_error))
        _, realized_area = _risk_coverage(squared_error, np.argsort(-squared_error))
        random_generator = np.random.default_rng(0)
        random_area = float(np.mean([
            _risk_coverage(squared_error, random_generator.permutation(len(squared_error)))[1]
            for _ in range(20)
        ]))

        def skill_against(reference_area):
            spread = random_area - reference_area
            return (random_area - area) / spread if spread > 1e-12 else 0.0

        return curve, skill_against(ceiling_area), skill_against(realized_area)

    mixed_curve, mixed_skill, mixed_naive = evaluate(torch.cat([in_distribution, harder]))
    # The in-distribution set on its own is the honest test. A mixed set is dominated by
    # a handful of far-tail points whose error and uncertainty are both enormous, so any
    # ordering separates them and the skill saturates near 1. Production validation data
    # is i.i.d. with training, so this second number is the one that transfers.
    inlier_curve, inlier_skill, inlier_naive = evaluate(in_distribution)

    print(f"       {'retention':>10} {'MSE mixed':>11} {'MSE in-dist':>12}")
    for coverage, mixed_risk, inlier_risk in zip(
        np.linspace(0.1, 1.0, 10), mixed_curve, inlier_curve
    ):
        print(f"       {coverage:>10.0%} {mixed_risk:>11.4f} {inlier_risk:>12.5f}")

    record(
        "discarding uncertain predictions reduces error",
        mixed_curve[0] < mixed_curve[-1],
        f"MSE at 100% retention {mixed_curve[-1]:.4f} -> {mixed_curve[0]:.4f} at 10%; "
        f"a flat curve means the uncertainty carries no information about the error",
    )
    print(f"       skill vs attainable ceiling: mixed={mixed_skill:.3f}  "
          f"in-distribution={inlier_skill:.3f}")
    print(f"       (same, scored against the unattainable realized-error oracle: "
          f"{mixed_naive:.3f} / {inlier_naive:.3f})")
    record(
        "uncertainty ranks errors better than chance (mixed)",
        mixed_skill > 0.3,
        f"skill = {mixed_skill:.3f} (1.0 = the best any model could do, 0.0 = random). "
        f"Easy: a few far-tail points dominate the error and any ordering finds them",
    )
    record(
        "uncertainty ranks errors better than chance (in-distribution only)",
        inlier_skill > 0.3,
        f"skill = {inlier_skill:.3f} vs an ATTAINABLE ceiling. Note in-distribution error "
        f"here is partly capacity-bound approximation error, which is irreducible and "
        f"which an epistemic measure is not supposed to flag",
    )


# --------------------------------------------------------------------------- #
# 6. Split the error into what the epinet CAN and CANNOT be expected to rank
# --------------------------------------------------------------------------- #
def _skill(squared_error: np.ndarray, ordering_score: np.ndarray,
           reference_score: np.ndarray) -> tuple[np.ndarray, float]:
    """Risk-coverage curve for `ordering_score`, scored between random and `reference`."""
    curve, area = _risk_coverage(squared_error, np.argsort(-ordering_score))
    _, reference_area = _risk_coverage(squared_error, np.argsort(-reference_score))
    random_generator = np.random.default_rng(0)
    random_area = float(np.mean([
        _risk_coverage(squared_error, random_generator.permutation(len(squared_error)))[1]
        for _ in range(20)
    ]))
    spread = random_area - reference_area
    return curve, ((random_area - area) / spread if spread > 1e-12 else 0.0)


def check_epistemic_ranking(device: torch.device, steps: int = 1500,
                            repeats: int = 15) -> None:
    print("\n[6/7] does the epinet rank the REDUCIBLE part of the error?")

    fit = _fit_1d_epinet(device, PRODUCTION_PRIOR_AMPLITUDE, steps, n_train=800,
                         density_ratio=12.0)

    # The base predictor is Linear(MLP_DIMENSION, 1) on fixed features, so "what this
    # model class would converge to given unlimited noiseless data" has a closed form:
    # least squares against the latent over a dense grid on the training support.
    dense = torch.cat([torch.linspace(-3.0, -1.2, 2000), torch.linspace(1.2, 3.0, 2000)])

    def design(inputs: torch.Tensor) -> torch.Tensor:
        columns = fit.features_of(inputs)[:, :MLP_DIMENSION]
        return torch.cat([columns, torch.ones((len(inputs), 1), device=columns.device)], dim=1)

    with torch.no_grad():
        weights = torch.linalg.pinv(design(dense)) @ fit.latent(dense).to(device).unsqueeze(1)

        def converged_fit(inputs: torch.Tensor) -> np.ndarray:
            return (design(inputs) @ weights).squeeze(-1).cpu().numpy()

    generator = torch.Generator().manual_seed(21)
    evaluation_inputs = torch.cat([
        torch.rand(150, generator=generator) * 1.8 - 3.0,
        torch.rand(150, generator=generator) * 1.8 + 1.2,
    ])

    latent = fit.latent(evaluation_inputs).cpu().numpy()
    converged = converged_fit(evaluation_inputs)

    # A single evaluation draw is NOT enough here. In-distribution the predicted std is
    # nearly uniform across points (it is k * prior, roughly constant), so the ranking is
    # driven as much by the sampling error in each point's std as by real variation:
    # repeating this with only the index draw changed swings the skill from -1.0 to +0.54.
    # Average over independent draws and report the spread, or the number is not a result.
    totals, estimations = [], []
    for repeat in range(repeats):
        torch.manual_seed(1000 + repeat)
        samples = fit.samples(evaluation_inputs)
        predicted = samples.mean(axis=0)
        predicted_std = samples.std(axis=0)
        total_error = (predicted - latent) ** 2
        estimation_error = (predicted - converged) ** 2
        totals.append(_skill(total_error, predicted_std, total_error)[1])
        estimations.append(_skill(estimation_error, predicted_std, estimation_error)[1])

    samples = fit.samples(evaluation_inputs)
    predicted = samples.mean(axis=0)
    predicted_std = samples.std(axis=0)

    # approximation: what the hypothesis class can never represent, no matter the data.
    # estimation:    how far this fit sits from that ceiling -- the part more data fixes,
    #                and the only part an epistemic measure is meant to track.
    approximation = converged - latent
    estimation = predicted - converged
    total = predicted - latent

    share = float(np.mean(approximation**2) / max(np.mean(total**2), 1e-12))
    print(f"       mean squared error   total={np.mean(total**2):.6f}  "
          f"approximation={np.mean(approximation**2):.6f}  "
          f"estimation={np.mean(estimation**2):.6f}")
    print(f"       irreducible share of in-distribution error: {share:.1%}")

    skill_total, skill_estimation = float(np.mean(totals)), float(np.mean(estimations))
    spread = float(np.std(estimations)) / max(math.sqrt(repeats), 1.0)

    print(f"       over {repeats} index draws:")
    print(f"         selective skill vs TOTAL error      : {skill_total:+.3f}")
    print(f"         selective skill vs ESTIMATION error : {skill_estimation:+.3f} "
          f"+/- {spread:.3f} (standard error)")

    record(
        "in-distribution error is not overwhelmingly irreducible",
        share < 0.9,
        f"{share:.1%} of in-distribution squared error is approximation error the model "
        f"class cannot represent at any sample size; an epistemic measure should not, and "
        f"cannot, rank it",
    )
    record(
        "epinet ranks the reducible part of the error",
        skill_estimation - 2 * spread > 0.0,
        f"skill against estimation error = {skill_estimation:+.3f} +/- {spread:.3f} "
        f"(vs {skill_total:+.3f} against total). Requiring it to clear zero by two "
        f"standard errors, since a single draw of this statistic is worthless",
    )


# --------------------------------------------------------------------------- #
# 7. Validation metrics against an analytically perfect model
# --------------------------------------------------------------------------- #
def check_metric_calibration() -> None:
    print("\n[7/7] validation metrics on an analytically perfect predictive law")
    noise_std, n_points, n_samples, tau = 0.2, 4000, 200, 8
    rng = np.random.default_rng(0)

    latent_means = rng.normal(0.0, 1.0, n_points)
    observations = latent_means + rng.normal(0.0, noise_std, n_points)
    # Zero epistemic spread + the correct aleatoric noise_std == the true law.
    predictions = np.repeat(latent_means[:, None], n_samples, axis=1)

    p_values, variances = compute_calibration_measures(observations, predictions, noise_std)
    calibration_error, sharpness = calculate_calibration_metrics(p_values, variances, 100, None)

    record(
        "calibration error ~ 0 for the true predictive law",
        calibration_error < 0.02,
        f"calibration_error = {calibration_error:.4f}",
    )
    record(
        "sharpness equals the predictive variance",
        abs(sharpness - noise_std**2) < 1e-3,
        f"sharpness = {sharpness:.4f} vs noise_std^2 = {noise_std ** 2:.4f} "
        f"(a fixed noise_std^2 floor is {noise_std ** 2 / max(sharpness, 1e-12):.0%} of it)",
    )

    joint_loss = GaussianJointLogLoss(noise_std=noise_std, tau=tau)
    nll = joint_loss(
        torch.tensor(predictions.T, dtype=torch.float32),
        torch.tensor(observations, dtype=torch.float32),
    ).item()
    gaussian_constant = tau * (np.log(np.sqrt(2 * np.pi) * noise_std) + 0.5)
    excess_per_target = (nll - gaussian_constant) / tau
    record(
        "excess joint NLL ~ 0 for the true predictive law",
        abs(excess_per_target) < 0.05,
        f"excess_per_target = {excess_per_target:.4f} (must match the constant used by "
        f"_nll_metrics, i.e. include the +0.5/target differential-entropy term)",
    )


CHECKS = {
    "prior": lambda device: check_prior_diversity(device),
    "schedule": lambda device: check_learning_rate_schedule(),
    "gap": lambda device: check_gap_uncertainty(device),
    "monotonic": lambda device: check_uncertainty_monotonicity(device),
    "selective": lambda device: check_selective_prediction(device),
    "epistemic": lambda device: check_epistemic_ranking(device),
    "metrics": lambda device: check_metric_calibration(),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=sorted(CHECKS), action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1500, help="training steps for --check gap")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    selected = arguments.check or list(CHECKS)

    stepped = {"gap": check_gap_uncertainty,
               "monotonic": check_uncertainty_monotonicity,
               "selective": check_selective_prediction,
               "epistemic": check_epistemic_ranking}
    for name in selected:
        if name in stepped:
            stepped[name](device, steps=arguments.steps)
        else:
            CHECKS[name](device)

    failures = [name for name, passed, _ in RESULTS if not passed]
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - len(failures)}/{len(RESULTS)} assertions passed")
    for name, passed, detail in RESULTS:
        if not passed:
            print(f"  FAIL  {name}: {detail}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
