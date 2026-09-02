"""Synthetic Bayesian linear-regression diagnostic for the project's epinet.

This test compares function samples from the real ``MultiHeadEpistemicNetwork``
head against the exact posterior of a small Bayesian linear-regression problem.
It deliberately bypasses query embedding and tree convolution: synthetic design
vectors are supplied at the point where the real plan estimator would supply its
last hidden features. The real base cost head, learnable epinet modules,
epistemic-index sampler, and production loss composition are retained.

The fixed ensemble prior is replaced by the linear random prior required by
Theorem 4 of "Epistemic Neural Networks". It is passed through the same
``[index_dim, n_examples]`` interface as production ensemble priors.

Examples, run from the repository root:

    python -m src.supervised_value_estimation.synthetic_bayesian_linear_regression_test
    python -m src.supervised_value_estimation.synthetic_bayesian_linear_regression_test --objective paper

The default ``implementation`` objective calls ``loss_epinet`` from
``supervised_value_estimation_cached_prior.py``. The ``paper`` objective is an
ablation which uses Equation 9's Gaussian target perturbation explicitly. A
failure of the former but not the latter localizes the problem to the training
objective rather than the epinet architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.models.epistemic_neural_network import (  # noqa: E402
    MultiHeadEpistemicNetwork,
    prepare_epinet_model,
)
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import (  # noqa: E402
    loss_epinet,
)


Objective = Literal["implementation", "paper"]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    n_train: int = 64
    n_test: int = 16
    input_dim: int = 3
    index_dim: int = 32
    train_index_samples: int = 16
    posterior_samples: int = 4096
    steps: int = 2500
    learning_rate: float = 3e-3
    observation_noise_std: float = 0.30
    prior_std: float = 1.0
    objective: Objective = "implementation"
    device: str = "cpu"
    log_every: int = 250
    mean_nrmse_threshold: float = 0.25
    covariance_relative_error_threshold: float = 0.50
    std_relative_error_threshold: float = 0.35
    covariance_correlation_threshold: float = 0.65


@dataclass(frozen=True)
class PosteriorMetrics:
    objective: str
    passed: bool
    final_data_loss: float
    final_regularized_loss: float
    initial_epinet_gradient_norm: float
    initial_base_gradient_norm: float
    mean_nrmse: float
    covariance_relative_frobenius_error: float
    covariance_relative_error_to_finite_index_optimum: float
    finite_index_relative_error_to_bayes: float
    std_relative_error: float
    covariance_upper_triangle_correlation: float
    analytic_posterior_trace: float
    empirical_posterior_trace: float
    prior_trace: float
    analytic_contraction_ratio: float
    empirical_contraction_ratio: float


@dataclass(frozen=True)
class SyntheticProblem:
    train_design: torch.Tensor
    train_targets: torch.Tensor
    test_design: torch.Tensor
    train_features: torch.Tensor
    test_features: torch.Tensor
    context_vectors: torch.Tensor
    fixed_prior_weights: torch.Tensor
    fixed_prior_values_train: torch.Tensor
    fixed_prior_values_test: torch.Tensor
    analytic_weight_mean: torch.Tensor
    analytic_weight_covariance: torch.Tensor
    analytic_test_mean: torch.Tensor
    analytic_test_covariance: torch.Tensor
    finite_index_test_covariance: torch.Tensor
    analytic_test_prior_covariance: torch.Tensor


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _unit_vectors(rows: int, columns: int, generator: torch.Generator) -> torch.Tensor:
    vectors = torch.randn((rows, columns), generator=generator, dtype=torch.float64)
    return nn.functional.normalize(vectors, dim=-1)


def _make_problem(
    config: ExperimentConfig,
    feature_dim: int,
    device: torch.device,
) -> SyntheticProblem:
    """Generate data and its exact latent-function posterior in float64."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)

    train_design = torch.randn(
        (config.n_train, config.input_dim), generator=generator, dtype=torch.float64
    )
    test_design = torch.randn(
        (config.n_test, config.input_dim), generator=generator, dtype=torch.float64
    )
    true_weights = config.prior_std * torch.randn(
        config.input_dim, generator=generator, dtype=torch.float64
    )
    observation_noise = config.observation_noise_std * torch.randn(
        config.n_train, generator=generator, dtype=torch.float64
    )
    train_targets = train_design @ true_weights + observation_noise

    noise_variance = config.observation_noise_std**2
    prior_variance = config.prior_std**2
    precision = (
        train_design.T @ train_design / noise_variance
        + torch.eye(config.input_dim, dtype=torch.float64) / prior_variance
    )
    analytic_weight_covariance = torch.linalg.inv(precision)
    analytic_weight_mean = torch.linalg.solve(
        precision, train_design.T @ train_targets / noise_variance
    )
    analytic_test_mean = test_design @ analytic_weight_mean
    analytic_test_covariance = (
        test_design @ analytic_weight_covariance @ test_design.T
    )
    analytic_test_prior_covariance = prior_variance * test_design @ test_design.T

    # Theorem 4 samples every c_i from the unit sphere. The production loss
    # regenerates these from this seed on every call.
    context_generator = torch.Generator(device=device).manual_seed(config.seed)
    context_vectors = nn.functional.normalize(
        torch.randn(
            (config.n_train, config.index_dim),
            generator=context_generator,
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    # P0 has one unit-sphere column per linear-regression coefficient. Scaling
    # columns by prior_std gives P0^T P0 -> prior_std^2 I as index_dim grows.
    prior_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    fixed_prior_weights = config.prior_std * _unit_vectors(
        config.input_dim, config.index_dim, prior_generator
    ).T
    fixed_prior_values_train = fixed_prior_weights @ train_design.T
    fixed_prior_values_test = fixed_prior_weights @ test_design.T

    # At finite index dimension, random sphere vectors are only approximately
    # orthogonal. This is the exact optimum of the linear randomized-prior
    # objective for the sampled C and P0. Reporting it separately distinguishes
    # optimization/architecture error from finite-index approximation error.
    ridge_precision = noise_variance / prior_variance
    regularized_design_inverse = torch.linalg.inv(
        train_design.T @ train_design
        + ridge_precision * torch.eye(config.input_dim, dtype=torch.float64)
    )
    finite_index_weights = (
        config.observation_noise_std
        * context_vectors.to(device="cpu", dtype=torch.float64).T
        @ train_design
        + ridge_precision * fixed_prior_weights
    ) @ regularized_design_inverse
    finite_index_test_covariance = (
        test_design @ finite_index_weights.T @ finite_index_weights @ test_design.T
    )

    def pad_features(design: torch.Tensor) -> torch.Tensor:
        if design.shape[1] > feature_dim:
            raise ValueError(
                f"Synthetic input dimension {design.shape[1]} exceeds the real "
                f"epinet feature dimension {feature_dim}."
            )
        features = torch.zeros(
            (design.shape[0], feature_dim), dtype=torch.float32, device=device
        )
        features[:, : design.shape[1]] = design.to(device=device, dtype=torch.float32)
        return features

    return SyntheticProblem(
        train_design=train_design,
        train_targets=train_targets,
        test_design=test_design,
        train_features=pad_features(train_design),
        test_features=pad_features(test_design),
        context_vectors=context_vectors,
        fixed_prior_weights=fixed_prior_weights,
        fixed_prior_values_train=fixed_prior_values_train.to(
            device=device, dtype=torch.float32
        ),
        fixed_prior_values_test=fixed_prior_values_test.to(
            device=device, dtype=torch.float32
        ),
        analytic_weight_mean=analytic_weight_mean,
        analytic_weight_covariance=analytic_weight_covariance,
        analytic_test_mean=analytic_test_mean,
        analytic_test_covariance=analytic_test_covariance,
        finite_index_test_covariance=finite_index_test_covariance,
        analytic_test_prior_covariance=analytic_test_prior_covariance,
    )


def _build_real_epinet(
    config: ExperimentConfig, device: torch.device
) -> MultiHeadEpistemicNetwork:
    """Construct the same model class and heads used by YAGO training."""
    full_model_config = REPOSITORY_ROOT / (
        "experiments/model_configs/policy_networks/"
        "t_cv_repr_separate_head_own_embeddings_hll.yaml"
    )
    prior_model_config = REPOSITORY_ROOT / (
        "experiments/model_configs/prior_networks/prior_t_cv_smallest_hll.yaml"
    )

    heads_config = {
        "plan_cost": {"layer": nn.Linear(64, 1)},
    }
    heads_config_prior = {
        "plan_cost": {"layer": nn.Linear(5, 1)},
    }
    model = prepare_epinet_model(
        full_gnn_config=str(full_model_config),
        config_ensemble_prior=str(prior_model_config),
        epinet_index_dim=config.index_dim,
        mlp_dimension=64,
        heads_config=heads_config,
        heads_config_prior=heads_config_prior,
        device=device,
        model_weights=None,
        cost_only=True,
        freeze_embedding=True,
    )

    # The graph and tree modules are intentionally not part of this controlled
    # linear test. Freeze everything, then enable only the real deterministic
    # cost head and real learnable epinet modules used by production.
    for parameter in model.parameters():
        parameter.requires_grad = False

    base_head = model.cost_estimation_model.query_plan_model.heads["plan_cost"]
    nn.init.zeros_(base_head.weight)
    if base_head.bias is not None:
        nn.init.zeros_(base_head.bias)
        base_head.bias.requires_grad = False
    base_head.weight.requires_grad = True

    for parameter in model.get_learnable_epinet_params():
        parameter.requires_grad = True

    return model


def _trainable_parameters(
    model: MultiHeadEpistemicNetwork,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    base_head = model.cost_estimation_model.query_plan_model.heads["plan_cost"]
    base_parameters = [base_head.weight]
    epinet_parameters = model.get_learnable_epinet_params()
    return base_parameters + epinet_parameters, base_parameters, epinet_parameters


def _squared_parameter_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    terms = [parameter.square().sum() for parameter in parameters]
    if not terms:
        raise ValueError("Expected at least one trainable parameter.")
    return torch.stack(terms).sum()


def _paper_loss(
    model: MultiHeadEpistemicNetwork,
    base_estimate: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    fixed_prior_values: torch.Tensor,
    context_vectors: torch.Tensor,
    n_index_samples: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 9 using the same block layout and model methods as production."""
    n_examples = targets.shape[0]
    epistemic_indexes = model.sample_epistemic_indexes_batched(n_index_samples)

    ensemble_prior = epistemic_indexes @ fixed_prior_values
    learnable = model.compute_learnable_mlp_batched(
        features.detach(), epistemic_indexes
    )["plan_cost"].view(n_index_samples, n_examples)

    predictions = (
        base_estimate.view(1, n_examples).detach()
        + learnable
        + ensemble_prior
    )
    perturbations = noise_std * (epistemic_indexes @ context_vectors.T)
    perturbed_targets = targets.view(1, n_examples) + perturbations

    base_loss = nn.functional.mse_loss(base_estimate.view(-1), targets)
    epinet_loss = nn.functional.mse_loss(predictions, perturbed_targets)
    return base_loss + epinet_loss, predictions


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().square().sum().item()
    return math.sqrt(total)


def _fit_model(
    model: MultiHeadEpistemicNetwork,
    problem: SyntheticProblem,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, float, float, float]:
    all_parameters, base_parameters, epinet_parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        all_parameters, lr=config.learning_rate, weight_decay=0.0
    )
    mse = nn.MSELoss(reduction="mean")
    base_head = model.cost_estimation_model.query_plan_model.heads["plan_cost"]
    targets = problem.train_targets.to(device=device, dtype=torch.float32)

    # Equation 9 states lambda per example. Since MSE averages over N examples,
    # this is sigma^2 / (N * prior_std^2), as in Theorem 4.
    ridge_coefficient = config.observation_noise_std**2 / (
        config.n_train * config.prior_std**2
    )
    production_plans = [
        ((), float(target), config.seed + index)
        for index, target in enumerate(problem.train_targets.tolist())
    ]
    production_generator = torch.Generator(device=device)

    initial_base_gradient_norm = float("nan")
    initial_epinet_gradient_norm = float("nan")
    final_data_loss = float("nan")
    final_regularized_loss = float("nan")

    model.train()
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        base_estimate = base_head(problem.train_features)

        if config.objective == "implementation":
            data_loss, _, _ = loss_epinet(
                problem.fixed_prior_values_train,
                model,
                mse,
                base_estimate,
                problem.train_features,
                production_plans,
                config.train_index_samples,
                config.observation_noise_std,
                0.0,  # Isolate the exact linear ensemble prior.
                1.0,
                production_generator,
                device,
                head_name="plan_cost",
            )
        else:
            data_loss, _ = _paper_loss(
                model=model,
                base_estimate=base_estimate,
                features=problem.train_features,
                targets=targets,
                fixed_prior_values=problem.fixed_prior_values_train,
                context_vectors=problem.context_vectors,
                n_index_samples=config.train_index_samples,
                noise_std=config.observation_noise_std,
            )

        regularizer = ridge_coefficient * _squared_parameter_norm(all_parameters)
        regularized_loss = data_loss + regularizer
        regularized_loss.backward()

        if step == 1:
            initial_base_gradient_norm = _gradient_norm(base_parameters)
            initial_epinet_gradient_norm = _gradient_norm(epinet_parameters)
            if not math.isfinite(initial_epinet_gradient_norm) or initial_epinet_gradient_norm <= 0:
                raise AssertionError("No finite, nonzero gradient reached the learnable epinet.")
            if not math.isfinite(initial_base_gradient_norm) or initial_base_gradient_norm <= 0:
                raise AssertionError("No finite, nonzero gradient reached the base cost head.")

        optimizer.step()
        final_data_loss = data_loss.detach().item()
        final_regularized_loss = regularized_loss.detach().item()

        if config.log_every > 0 and (
            step == 1 or step % config.log_every == 0 or step == config.steps
        ):
            print(
                f"step={step:5d} data_loss={final_data_loss:.6f} "
                f"regularized_loss={final_regularized_loss:.6f}"
            )

    return (
        final_data_loss,
        final_regularized_loss,
        initial_base_gradient_norm,
        initial_epinet_gradient_norm,
    )


def _empirical_predictions(
    model: MultiHeadEpistemicNetwork,
    features: torch.Tensor,
    fixed_prior_values: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Return coherent function samples with shape [n_samples, n_test]."""
    base_head = model.cost_estimation_model.query_plan_model.heads["plan_cost"]
    model.eval()
    with torch.no_grad():
        indexes = model.sample_epistemic_indexes_batched(n_samples)
        base = base_head(features).view(1, -1)
        learnable = model.compute_learnable_mlp_batched(
            features, indexes
        )["plan_cost"].view(n_samples, features.shape[0])
        prior = indexes @ fixed_prior_values
        return (base + learnable + prior).to(dtype=torch.float64, device="cpu")


def _upper_triangle_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    indices = torch.triu_indices(left.shape[0], left.shape[1], offset=1)
    left_values = left[indices[0], indices[1]].numpy()
    right_values = right[indices[0], indices[1]].numpy()
    if np.std(left_values) == 0 or np.std(right_values) == 0:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _evaluate(
    samples: torch.Tensor,
    problem: SyntheticProblem,
    config: ExperimentConfig,
    training_stats: tuple[float, float, float, float],
) -> PosteriorMetrics:
    empirical_mean = samples.mean(dim=0)
    empirical_covariance = torch.cov(samples.T)
    empirical_std = torch.sqrt(torch.diag(empirical_covariance).clamp_min(0))

    analytic_mean = problem.analytic_test_mean.cpu()
    analytic_covariance = problem.analytic_test_covariance.cpu()
    finite_index_covariance = problem.finite_index_test_covariance.cpu()
    analytic_std = torch.sqrt(torch.diag(analytic_covariance).clamp_min(0))
    prior_covariance = problem.analytic_test_prior_covariance.cpu()

    mean_scale = torch.sqrt(torch.mean(analytic_mean.square())).clamp_min(1e-12)
    mean_nrmse = (
        torch.sqrt(torch.mean((empirical_mean - analytic_mean).square())) / mean_scale
    ).item()
    covariance_relative_error = (
        torch.linalg.matrix_norm(empirical_covariance - analytic_covariance)
        / torch.linalg.matrix_norm(analytic_covariance).clamp_min(1e-12)
    ).item()
    covariance_error_to_finite_index = (
        torch.linalg.matrix_norm(empirical_covariance - finite_index_covariance)
        / torch.linalg.matrix_norm(finite_index_covariance).clamp_min(1e-12)
    ).item()
    finite_index_error_to_bayes = (
        torch.linalg.matrix_norm(finite_index_covariance - analytic_covariance)
        / torch.linalg.matrix_norm(analytic_covariance).clamp_min(1e-12)
    ).item()
    std_relative_error = torch.mean(
        torch.abs(empirical_std - analytic_std) / analytic_std.clamp_min(1e-12)
    ).item()
    covariance_correlation = _upper_triangle_correlation(
        empirical_covariance, analytic_covariance
    )

    analytic_trace = torch.trace(analytic_covariance).item()
    empirical_trace = torch.trace(empirical_covariance).item()
    prior_trace = torch.trace(prior_covariance).item()
    analytic_contraction = analytic_trace / prior_trace
    empirical_contraction = empirical_trace / prior_trace

    passed = bool(
        math.isfinite(mean_nrmse)
        and math.isfinite(covariance_relative_error)
        and math.isfinite(std_relative_error)
        and math.isfinite(covariance_correlation)
        and mean_nrmse <= config.mean_nrmse_threshold
        and covariance_error_to_finite_index
        <= config.covariance_relative_error_threshold
        and finite_index_error_to_bayes
        <= config.covariance_relative_error_threshold
        and std_relative_error <= config.std_relative_error_threshold
        and covariance_correlation >= config.covariance_correlation_threshold
    )

    return PosteriorMetrics(
        objective=config.objective,
        passed=passed,
        final_data_loss=training_stats[0],
        final_regularized_loss=training_stats[1],
        initial_base_gradient_norm=training_stats[2],
        initial_epinet_gradient_norm=training_stats[3],
        mean_nrmse=mean_nrmse,
        covariance_relative_frobenius_error=covariance_relative_error,
        covariance_relative_error_to_finite_index_optimum=covariance_error_to_finite_index,
        finite_index_relative_error_to_bayes=finite_index_error_to_bayes,
        std_relative_error=std_relative_error,
        covariance_upper_triangle_correlation=covariance_correlation,
        analytic_posterior_trace=analytic_trace,
        empirical_posterior_trace=empirical_trace,
        prior_trace=prior_trace,
        analytic_contraction_ratio=analytic_contraction,
        empirical_contraction_ratio=empirical_contraction,
    )


def run_experiment(config: ExperimentConfig) -> PosteriorMetrics:
    """Train the real epinet head and compare it with the exact BLR posterior."""
    if config.input_dim <= 0 or config.input_dim > 64:
        raise ValueError("input_dim must be in [1, 64].")
    if config.index_dim < config.input_dim:
        raise ValueError("index_dim must be at least input_dim.")
    if config.n_train <= config.input_dim:
        raise ValueError("n_train must be greater than input_dim.")
    if config.posterior_samples < 2:
        raise ValueError("posterior_samples must be at least 2.")

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    print(f"Building real MultiHeadEpistemicNetwork on {device}...")
    model = _build_real_epinet(config, device)
    feature_dim = model.mlp_output_dim_cost_model
    problem = _make_problem(config, feature_dim, device)

    training_stats = _fit_model(model, problem, config, device)
    samples = _empirical_predictions(
        model,
        problem.test_features,
        problem.fixed_prior_values_test,
        config.posterior_samples,
    )
    return _evaluate(samples, problem, config, training_stats)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=("implementation", "paper"),
        default="implementation",
        help="Use the production loss or an explicit Equation 9 ablation.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--index-dim", type=int, default=32)
    parser.add_argument("--train-index-samples", type=int, default=16)
    parser.add_argument("--posterior-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--observation-noise-std", type=float, default=0.30)
    parser.add_argument("--prior-std", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Return exit code zero even when posterior thresholds are missed.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        index_dim=args.index_dim,
        train_index_samples=args.train_index_samples,
        posterior_samples=args.posterior_samples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        observation_noise_std=args.observation_noise_std,
        prior_std=args.prior_std,
        objective=args.objective,
        device=args.device,
        log_every=args.log_every,
    )
    metrics = run_experiment(config)
    result = {
        "config": asdict(config),
        "metrics": asdict(metrics),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print("\nBayesian linear-regression posterior comparison:\n" + rendered)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")

    if metrics.passed:
        print("\nPASS: empirical epinet moments match the analytic posterior thresholds.")
        return 0

    print(
        "\nFAIL: empirical epinet moments do not match the analytic posterior. "
        "Compare --objective implementation with --objective paper."
    )
    return 0 if args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
