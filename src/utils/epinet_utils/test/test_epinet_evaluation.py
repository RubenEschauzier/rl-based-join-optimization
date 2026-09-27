import math

import numpy as np
import pytest
import torch

from src.utils.epinet_utils.epinet_evaluation import (
    SplitAccumulator,
    excess_constant,
    metric_mode,
    metric_names,
    query_statistics,
    selective_risk_curve,
    weighted_quantile,
)
from src.utils.training_utils.training_tracking import TrainSummary

NOISE = 0.2
TAUS = (1, 2, 4, 8)


def _random_query(n_plans, n_samples=64, seed=0, shared_error=0.0):
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randn(n_plans, generator=generator)
    base = targets + 0.3 * torch.randn(n_plans, generator=generator)
    # A query-level offset shared by every plan: the dependence an epinet should capture.
    offsets = shared_error * torch.randn(n_samples, 1, generator=generator)
    samples = base + offsets + 0.05 * torch.randn(n_samples, n_plans, generator=generator)
    return samples, base, targets


def _summarize(queries, split="val", std_cost=2.0):
    accumulator = SplitAccumulator(split, TAUS, NOISE, std_cost)
    for index, (samples, base, targets) in enumerate(queries):
        accumulator.add(query_statistics(samples, base, targets, NOISE, TAUS, seed=index, std_cost=std_cost),
                        loss_epinet=0.0)
    return accumulator.summarize(accumulator.fitted_base_noise_std())


def test_summary_produces_exactly_the_registered_metric_names():
    metrics, _ = _summarize([_random_query(12, seed=s) for s in range(3)])
    registered = {name for name, _ in metric_names(("val",), TAUS)} - {"train_loss", "train_epoch_seconds"}
    assert set(metrics) == registered


def test_shuffling_samples_cannot_matter_for_single_plan_groups():
    metrics, _ = _summarize([_random_query(10, seed=s, shared_error=0.5) for s in range(4)])
    assert metrics["val_jnll_tau1_dependence_gain"] == pytest.approx(0.0, abs=1e-6)


def test_identical_samples_reduce_the_epinet_to_the_fixed_noise_base():
    targets = torch.tensor([0.0, 0.5, -0.5, 1.0, 0.2, -0.1, 0.3, 0.9])
    base = targets + 0.1
    samples = base.repeat(16, 1)
    metrics, _ = _summarize([(samples, base, targets)])
    for tau in TAUS:
        prefix = f"val_jnll_tau{tau}"
        assert metrics[f"{prefix}_epinet_excess_per_target"] == pytest.approx(
            metrics[f"{prefix}_base_fixed_excess_per_target"], abs=1e-5)
        assert metrics[f"{prefix}_dependence_gain"] == pytest.approx(0.0, abs=1e-5)


def test_queries_smaller_than_tau_are_excluded_not_padded():
    metrics, curves = _summarize([_random_query(3, seed=0), _random_query(9, seed=1)])
    assert metrics["val_jnll_tau2_n_queries"] == 2
    assert metrics["val_jnll_tau4_n_queries"] == 1
    assert metrics["val_jnll_tau8_n_queries"] == 1
    assert curves["jnll_n_queries"] == [2, 2, 1, 1]


def test_tau_without_any_eligible_query_is_nan_not_an_error():
    metrics, _ = _summarize([_random_query(3, seed=0)])
    assert metrics["val_jnll_tau8_n_queries"] == 0
    assert math.isnan(metrics["val_jnll_tau8_epinet_excess_per_target"])


def test_shared_query_error_shows_up_as_positive_dependence_gain():
    queries = []
    for seed in range(6):
        generator = torch.Generator().manual_seed(seed)
        targets = torch.randn(16, generator=generator)
        # The base is off by one shared amount; epinet samples span that offset coherently.
        base = targets + 0.6
        offsets = torch.linspace(-1.2, 0.0, 40).unsqueeze(1)
        queries.append((base + offsets, base, targets))
    metrics, _ = _summarize(queries)
    assert metrics["val_jnll_tau8_dependence_gain"] > 0.1
    assert metrics["val_jnll_tau8_gain_vs_base_fixed"] > 0


def test_fitted_noise_base_matches_its_closed_form():
    targets = torch.zeros(8)
    base = torch.tensor([0.5, -0.5] * 4)
    samples = base.repeat(4, 1)
    accumulator = SplitAccumulator("val", (8,), NOISE, 1.0)
    accumulator.add(query_statistics(samples, base, targets, NOISE, (8,), seed=0, std_cost=1.0))
    fitted = accumulator.fitted_base_noise_std()
    assert fitted == pytest.approx(0.5)
    metrics, _ = accumulator.summarize(fitted)
    expected = math.log(math.sqrt(2 * math.pi) * 0.5) + 0.5 - excess_constant(NOISE)
    assert metrics["val_jnll_tau8_base_fitted_excess_per_target"] == pytest.approx(expected, abs=1e-6)


def test_uncertainty_equal_to_error_is_oracle_selective_prediction():
    rng = np.random.default_rng(0)
    error = rng.random(200)
    weights = np.ones(200)
    risk_informed = selective_risk_curve(error, error, weights)
    risk_uninformed = selective_risk_curve(-error, error, weights)
    assert risk_informed.mean() < error.mean() < risk_uninformed.mean()
    assert risk_informed[-1] == pytest.approx(error.mean())


def test_selective_skill_is_one_when_std_tracks_the_error():
    generator = torch.Generator().manual_seed(0)
    targets = torch.randn(50, generator=generator)
    base = targets.clone()
    scale = torch.linspace(0.01, 1.0, 50)
    # Samples spread exactly as far as the mean is off: std ranks plans by their error.
    samples = targets + scale + scale * torch.tensor([-1.0, 1.0]).repeat(32).unsqueeze(1)
    metrics, _ = _summarize([(samples, base, targets)])
    assert metrics["val_selective_skill"] == pytest.approx(1.0, abs=1e-6)


def test_every_query_weighs_equally_regardless_of_plan_count():
    values = np.array([10.0] * 1000 + [1.0] * 2)
    weights = np.array([1 / 1000] * 1000 + [1 / 2] * 2)
    assert weighted_quantile(values, weights, 0.25) == 1.0
    assert weighted_quantile(values, weights, 0.75) == 10.0


def test_thompson_regret_is_zero_when_every_sample_picks_the_best_plan():
    targets = torch.tensor([0.0, 1.0, 2.0])
    samples = torch.tensor([[0.0, 1.0, 2.0], [0.1, 1.5, 2.2]])
    statistics = query_statistics(samples, targets, targets, NOISE, (1,), seed=0, std_cost=2.0)
    assert statistics["regret_epinet_thompson"] == 0.0
    assert statistics["optimal_epinet_thompson"] == 1.0


def test_thompson_regret_is_in_log_cost_units():
    targets = torch.tensor([0.0, 1.0])
    samples = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    statistics = query_statistics(samples, targets, targets, NOISE, (1,), seed=0, std_cost=2.0)
    # Half the samples pick the plan one standardized unit (= 2 log-cost units) worse.
    assert statistics["regret_epinet_thompson"] == pytest.approx(1.0)


@pytest.mark.parametrize(("name", "mode"), [
    ("val_jnll_tau8_epinet_excess_per_target", "min"),
    ("val_jnll_tau8_dependence_gain", "max"),
    ("val_jnll_tau8_n_queries", "none"),
    ("test_coverage95_epinet", "none"),
    ("test_coverage_error_epinet", "min"),
    ("val_mse_at_coverage80_epinet", "min"),
    ("val_selective_skill", "max"),
    ("val_aurc_oracle", "none"),
    ("val_regret_epinet_thompson", "min"),
    ("val_optimal_rate_base", "max"),
    ("val_ms_per_query_epinet", "none"),
])
def test_metric_directions(name, mode):
    assert metric_mode(name) == mode


def test_train_summary_logs_directionless_metrics_without_ranking_them():
    summary = TrainSummary([("val_jnll_tau8_n_queries", "none"), ("train_loss", "min")])
    summary.update({"val_jnll_tau8_n_queries": 5, "train_loss": 1.0}, 1)
    summary.update({"val_jnll_tau8_n_queries": 7, "train_loss": 0.5}, 2)
    best, per_epoch = summary.summary()
    assert per_epoch["val_jnll_tau8_n_queries"] == [5, 7]
    assert best["train_loss"]["epoch"] == 2
