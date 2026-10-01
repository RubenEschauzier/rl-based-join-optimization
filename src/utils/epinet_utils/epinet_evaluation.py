"""Evaluation metrics for the epinet cost model, computed per query and pooled per split.

Everything here works in *standardized* target space (log-cost minus mean, over std)
unless a name says otherwise. Two conventions hold throughout:

- Every query weighs equally. Queries range from 2 to thousands of plans, and pooling
  plans would let a handful of large queries decide every number. Per-query quantities
  are averaged over queries; per-plan quantities carry a weight of 1 / n_plans.
- Joint NLL at group size tau is only computed on queries with at least tau distinct
  plans. Padding a small query with repeated plans (sampling with replacement) rewards
  a model for being self-consistent on the same plan, which is not the cross-plan
  dependence the metric is meant to measure. `*_n_queries` reports the coverage.

Baselines for the joint NLL:
- epinet:       the epinet's K samples, each with Gaussian observation noise noise_std.
- independent:  the same samples, permuted independently per plan. Marginals are
                identical to the epinet's; only the dependence across plans is destroyed.
- base_fixed:   the base network with the same fixed noise_std as the epinet.
- base_fitted:  the base network with its noise std fitted to its own residuals, so it
                is not handicapped by an overconfident fixed noise.
- epinet_fitted / independent_fitted:
                the epinet (and its shuffled control) with the observation-noise std
                fitted on validation, by per-plan (tau = 1) NLL over NOISE_GRID. The fixed
                noise_std is an evaluation assumption: the simulated targets carry no real
                noise, and when the epistemic spread already covers the error the fixed
                noise double counts it. Fitting on tau = 1 keeps the choice neutral with
                respect to the joint (tau > 1) claims.
"""
import math

import numpy as np
import torch

from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss

JOINT_MODELS = ("epinet", "independent", "base_fixed", "base_fitted", "epinet_fitted", "independent_fitted")
COVERAGE_LEVELS = (0.5, 0.8, 0.95)
SELECTIVE_COVERAGE_GRID = np.linspace(0.01, 1.0, 100)
CALIBRATION_GRID = np.linspace(0.0, 1.0, 101)
# Candidate observation-noise stds for the epinet, fitted on validation. Log-spaced; the
# fixed evaluation value 0.2 is included so "fitted" can never do worse than "fixed".
NOISE_GRID = np.unique(np.concatenate([np.geomspace(0.005, 0.4, 48), [0.2]]))


def excess_constant(noise_std):
    """Expected per-target NLL of the true law under N(0, noise_std^2) noise (its entropy).

    Subtracting it makes a perfect model score 0, so "excess" reads as KL-to-optimal.
    """
    return math.log(math.sqrt(2 * math.pi) * noise_std) + 0.5


def _joint_nll_over_noise_grid(pred_matrix, targets, plan_indices, noise_grid):
    """Per-target joint NLL of an epistemic mixture, for every candidate noise std.

    Same quantity as GaussianJointLogLoss / tau, but the squared errors per sample and
    group are computed once and reused for every noise value.
    """
    n_samples = pred_matrix.shape[0]
    tau = plan_indices.shape[1]
    squared = (pred_matrix[:, plan_indices] - targets[plan_indices]).square().sum(dim=2)  # (K, G)
    values = np.empty(len(noise_grid))
    for index, noise in enumerate(noise_grid):
        log_likelihood = -tau * math.log(math.sqrt(2 * math.pi) * noise) - squared / (2 * noise ** 2)
        per_group = torch.logsumexp(log_likelihood, dim=0) - math.log(n_samples)
        values[index] = -per_group.mean().item() / tau
    return values


def query_statistics(pred_matrix, base_predictions, targets, noise_std, taus, seed, std_cost,
                     noise_grid=NOISE_GRID):
    """All per-query quantities needed for the split-level metrics.

    pred_matrix:      (K, n_plans) epinet samples, standardized.
    base_predictions: (n_plans,) base network predictions, standardized.
    targets:          (n_plans,) standardized targets.
    seed:             fixes the plan grouping and the independence permutation, so every
                      model is scored on the same groups and runs are comparable.
    """
    pred_matrix = pred_matrix.detach().float().cpu()
    base_predictions = base_predictions.detach().float().cpu().reshape(-1)
    targets = targets.detach().float().cpu().reshape(-1)
    n_samples, n_plans = pred_matrix.shape

    permutation_generator = torch.Generator().manual_seed(seed + 1_000_003)
    permuted = torch.gather(
        pred_matrix, 0,
        torch.argsort(torch.rand(pred_matrix.shape, generator=permutation_generator), dim=0),
    )

    joint = {}
    for tau in taus:
        if n_plans < tau:
            continue
        joint_loss = GaussianJointLogLoss(noise_std=noise_std, tau=tau)
        plan_indices = joint_loss.sample_plan_indices(
            n_plans, pred_matrix.device, torch.Generator().manual_seed(seed + tau)
        )
        grouped_residuals = (base_predictions - targets)[plan_indices]
        joint[tau] = {
            "epinet": joint_loss(pred_matrix, targets, plan_indices).item() / tau,
            "independent": joint_loss(permuted, targets, plan_indices).item() / tau,
            "base_fixed": joint_loss(base_predictions.view(1, -1), targets, plan_indices).item() / tau,
            # The fitted-noise NLL is closed form in the mean squared residual, so the
            # noise std can be fitted after the pass instead of needing a second one.
            "base_mean_squared_residual": grouped_residuals.square().mean().item(),
            "epinet_grid": _joint_nll_over_noise_grid(pred_matrix, targets, plan_indices, noise_grid),
            "independent_grid": _joint_nll_over_noise_grid(permuted, targets, plan_indices, noise_grid),
        }

    epinet_mean = pred_matrix.mean(dim=0)
    epinet_std = pred_matrix.std(dim=0, unbiased=n_samples > 1)
    residuals = targets.unsqueeze(0) - pred_matrix
    pit_epinet = torch.special.ndtr(residuals / noise_std).mean(dim=0)
    # (n_plans, len(noise_grid)); float16 is ample for a value in [0, 1] and keeps a full
    # split's worth of plans small enough to hold until the noise is fitted.
    pit_epinet_grid = torch.stack(
        [torch.special.ndtr(residuals / noise).mean(dim=0) for noise in noise_grid], dim=1
    ).to(torch.float16)

    best_cost = targets.min()
    thompson_choice = pred_matrix.argmin(dim=1)
    thompson_regret = targets[thompson_choice] - best_cost
    mean_regret = targets[epinet_mean.argmin()] - best_cost
    base_regret = targets[base_predictions.argmin()] - best_cost

    return {
        "n_plans": n_plans,
        "joint": joint,
        "targets": targets.numpy(),
        "epinet_mean": epinet_mean.numpy(),
        "epinet_std": epinet_std.numpy(),
        "base": base_predictions.numpy(),
        "pit_epinet": pit_epinet.numpy(),
        "pit_epinet_grid": pit_epinet_grid.numpy(),
        # Regret in natural-log cost units: exp(regret) is the cost ratio to the best plan.
        "regret_base": base_regret.item() * std_cost,
        "regret_epinet_mean": mean_regret.item() * std_cost,
        "regret_epinet_thompson": thompson_regret.mean().item() * std_cost,
        "optimal_base": float(base_regret.item() <= 0.0),
        "optimal_epinet_mean": float(mean_regret.item() <= 0.0),
        "optimal_epinet_thompson": (thompson_regret <= 0.0).float().mean().item(),
    }


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, quantile), len(values) - 1)])


def selective_risk_curve(uncertainty, error, weights, coverage_grid=SELECTIVE_COVERAGE_GRID):
    """Weighted mean error of the `c` most-certain fraction of plans, for each coverage c.

    Sorting by the error itself gives the oracle curve; a constant uncertainty gives the
    random curve, which is flat at the overall error.
    """
    order = np.argsort(uncertainty, kind="stable")
    cumulative_weight = np.cumsum(weights[order])
    cumulative_error = np.cumsum((weights * error)[order])
    total = cumulative_weight[-1]
    cut = np.minimum(np.searchsorted(cumulative_weight, coverage_grid * total - 1e-12),
                     len(order) - 1)
    return cumulative_error[cut] / cumulative_weight[cut]


def weighted_calibration_curve(pit, weights, grid=CALIBRATION_GRID):
    """Observed fraction of targets at or below each predicted quantile (Kuleshov et al.)."""
    total = weights.sum()
    return np.array([weights[pit <= level].sum() / total for level in grid])


def interval_coverage(pit, weights, level):
    """Fraction of targets inside the central `level` predictive interval."""
    inside = np.abs(pit - 0.5) <= level / 2
    return float((weights * inside).sum() / weights.sum())


def _metric_names_for_split(split, taus):
    names = [
        f"{split}_mse_scaled_epinet", f"{split}_mse_scaled_base", f"{split}_loss_plan_cost_unscaled",
        f"{split}_qerror_median_epinet", f"{split}_qerror_median_base",
        f"{split}_qerror_p95_epinet", f"{split}_qerror_p95_base",
        f"{split}_base_noise_std", f"{split}_epinet_noise_std",
        f"{split}_calibration_error_epinet", f"{split}_calibration_error_base_fitted",
        f"{split}_calibration_error_epinet_fitted",
        f"{split}_coverage_error_epinet", f"{split}_coverage_error_base_fitted",
        f"{split}_coverage_error_epinet_fitted",
        f"{split}_sharpness_epinet", f"{split}_sharpness_epinet_fitted", f"{split}_epistemic_std_mean",
        f"{split}_aurc_epinet", f"{split}_aurc_oracle", f"{split}_aurc_random",
        f"{split}_selective_skill", f"{split}_mse_at_coverage80_epinet",
        f"{split}_loss_epinet",
        f"{split}_ms_per_query_base", f"{split}_ms_per_query_epinet", f"{split}_eval_seconds",
    ]
    for level in COVERAGE_LEVELS:
        percent = int(round(level * 100))
        names += [f"{split}_coverage{percent}_epinet", f"{split}_coverage{percent}_base_fitted",
                  f"{split}_coverage{percent}_epinet_fitted"]
    for model in ("base", "epinet_mean", "epinet_thompson"):
        names += [f"{split}_regret_{model}", f"{split}_optimal_rate_{model}"]
    for tau in taus:
        prefix = f"{split}_jnll_tau{tau}"
        names += [f"{prefix}_{model}_excess_per_target" for model in JOINT_MODELS]
        names += [f"{prefix}_dependence_gain", f"{prefix}_gain_vs_base_fitted",
                  f"{prefix}_gain_vs_base_fixed", f"{prefix}_n_queries",
                  f"{prefix}_dependence_gain_fitted", f"{prefix}_fitted_gain_vs_base_fitted"]
    return names


def metric_mode(name):
    """'min' / 'max' for metrics with a best direction, 'none' for references and counts."""
    if (name.endswith(("_n_queries", "_seconds", "_base_noise_std", "_epinet_noise_std",
                       "_aurc_oracle", "_aurc_random", "_epistemic_std_mean"))
            or "_ms_per_query_" in name
            or ("_coverage" in name and "_coverage_error_" not in name and "_at_coverage" not in name)):
        return "none"
    if "gain" in name or "skill" in name or "optimal_rate" in name:
        return "max"
    return "min"


def metric_names(splits, taus):
    """Every scalar evaluate/summarize produces, with its best direction."""
    names = ["train_loss", "train_epoch_seconds"]
    for split in splits:
        names += _metric_names_for_split(split, taus)
    return [(name, "none" if name == "train_epoch_seconds" else metric_mode(name)) for name in names]


class SplitAccumulator:
    """Collects per-query statistics for one split and pools them into metrics and curves."""

    def __init__(self, split, taus, noise_std, std_cost, noise_grid=NOISE_GRID):
        self.split = split
        self.taus = tuple(taus)
        self.noise_std = noise_std
        self.noise_grid = np.asarray(noise_grid)
        self.std_cost = std_cost
        self.queries = []
        self.loss_epinet = []
        self.seconds_base = 0.0
        self.seconds_epinet = 0.0
        self.eval_seconds = 0.0

    def add(self, statistics, loss_epinet=None):
        self.queries.append(statistics)
        if loss_epinet is not None:
            self.loss_epinet.append(loss_epinet)

    def fitted_base_noise_std(self):
        """MLE noise std for the base network under per-query averaging, floored for safety."""
        per_query = [np.mean((q["base"] - q["targets"]) ** 2) for q in self.queries]
        return max(float(np.sqrt(np.mean(per_query))), 1e-6)

    def fitted_epinet_noise_std(self):
        """Grid value minimising the epinet's per-plan NLL (the smallest tau, normally 1).

        Fitted on per-plan likelihood rather than by matching the average variance: with a
        noise near zero the predictive is K point masses, and the likelihood -- not the
        variance -- is what penalises the gaps between them.
        """
        tau = min(self.taus)
        nll = np.mean([q["joint"][tau]["epinet_grid"] for q in self.queries if tau in q["joint"]], axis=0)
        index = int(np.argmin(nll))
        if index in (0, len(self.noise_grid) - 1):
            print(f"WARNING: fitted epinet noise {self.noise_grid[index]:.4g} is at the edge of "
                  f"NOISE_GRID; widen the grid.")
        return float(self.noise_grid[index])

    def summarize(self, base_noise_std, epinet_noise_std=None):
        """Pool into (scalars, curves). Both noise stds are fitted on validation and reused
        for test, so neither fitted baseline ever sees the test residuals. When
        `epinet_noise_std` is omitted it is fitted on this split itself (in-sample)."""
        if not self.queries:
            raise ValueError(f"No queries were evaluated for split '{self.split}'.")
        s = self.split
        queries = self.queries
        weights = np.concatenate([np.full(q["n_plans"], 1.0 / q["n_plans"]) for q in queries])
        targets = np.concatenate([q["targets"] for q in queries])
        epinet_mean = np.concatenate([q["epinet_mean"] for q in queries])
        epinet_std = np.concatenate([q["epinet_std"] for q in queries])
        base = np.concatenate([q["base"] for q in queries])
        pit_epinet = np.concatenate([q["pit_epinet"] for q in queries])
        if epinet_noise_std is None:
            epinet_noise_std = self.fitted_epinet_noise_std()
        noise_index = int(np.argmin(np.abs(self.noise_grid - epinet_noise_std)))
        if not np.isclose(self.noise_grid[noise_index], epinet_noise_std):
            raise ValueError(f"epinet_noise_std={epinet_noise_std} is not a NOISE_GRID value.")
        pit_epinet_fitted = np.concatenate(
            [q["pit_epinet_grid"][:, noise_index] for q in queries]
        ).astype(np.float64)

        from scipy.stats import norm
        pit_base = norm.cdf((targets - base) / base_noise_std)

        squared_error_epinet = (epinet_mean - targets) ** 2
        squared_error_base = (base - targets) ** 2
        total_weight = weights.sum()
        mse_epinet = float((weights * squared_error_epinet).sum() / total_weight)
        mse_base = float((weights * squared_error_base).sum() / total_weight)
        qerror_epinet = np.exp(np.abs(epinet_mean - targets) * self.std_cost)
        qerror_base = np.exp(np.abs(base - targets) * self.std_cost)

        risk_epinet = selective_risk_curve(epinet_std, squared_error_epinet, weights)
        risk_oracle = selective_risk_curve(squared_error_epinet, squared_error_epinet, weights)
        aurc_epinet, aurc_oracle = float(risk_epinet.mean()), float(risk_oracle.mean())
        headroom = mse_epinet - aurc_oracle

        calibration_epinet = weighted_calibration_curve(pit_epinet, weights)
        calibration_base = weighted_calibration_curve(pit_base, weights)
        coverage_epinet = [interval_coverage(pit_epinet, weights, level) for level in COVERAGE_LEVELS]
        coverage_base = [interval_coverage(pit_base, weights, level) for level in COVERAGE_LEVELS]
        calibration_epinet_fitted = weighted_calibration_curve(pit_epinet_fitted, weights)
        coverage_epinet_fitted = [interval_coverage(pit_epinet_fitted, weights, level)
                                  for level in COVERAGE_LEVELS]

        metrics = {
            f"{s}_mse_scaled_epinet": mse_epinet,
            f"{s}_mse_scaled_base": mse_base,
            f"{s}_loss_plan_cost_unscaled": mse_base * self.std_cost ** 2,
            f"{s}_qerror_median_epinet": weighted_quantile(qerror_epinet, weights, 0.5),
            f"{s}_qerror_median_base": weighted_quantile(qerror_base, weights, 0.5),
            f"{s}_qerror_p95_epinet": weighted_quantile(qerror_epinet, weights, 0.95),
            f"{s}_qerror_p95_base": weighted_quantile(qerror_base, weights, 0.95),
            f"{s}_base_noise_std": base_noise_std,
            f"{s}_epinet_noise_std": epinet_noise_std,
            f"{s}_calibration_error_epinet_fitted": float(np.mean(np.abs(calibration_epinet_fitted - CALIBRATION_GRID))),
            f"{s}_coverage_error_epinet_fitted": float(np.mean(np.abs(np.array(coverage_epinet_fitted) - COVERAGE_LEVELS))),
            f"{s}_sharpness_epinet_fitted": float((weights * (epinet_std ** 2 + epinet_noise_std ** 2)).sum() / total_weight),
            f"{s}_calibration_error_epinet": float(np.mean(np.abs(calibration_epinet - CALIBRATION_GRID))),
            f"{s}_calibration_error_base_fitted": float(np.mean(np.abs(calibration_base - CALIBRATION_GRID))),
            f"{s}_coverage_error_epinet": float(np.mean(np.abs(np.array(coverage_epinet) - COVERAGE_LEVELS))),
            f"{s}_coverage_error_base_fitted": float(np.mean(np.abs(np.array(coverage_base) - COVERAGE_LEVELS))),
            f"{s}_sharpness_epinet": float((weights * (epinet_std ** 2 + self.noise_std ** 2)).sum() / total_weight),
            f"{s}_epistemic_std_mean": float((weights * epinet_std).sum() / total_weight),
            f"{s}_aurc_epinet": aurc_epinet,
            f"{s}_aurc_oracle": aurc_oracle,
            f"{s}_aurc_random": mse_epinet,
            # 1 = ranks plans by error as well as the oracle, 0 = no better than random.
            f"{s}_selective_skill": float((mse_epinet - aurc_epinet) / headroom) if headroom > 0 else float("nan"),
            f"{s}_mse_at_coverage80_epinet": float(np.interp(0.8, SELECTIVE_COVERAGE_GRID, risk_epinet)),
            f"{s}_loss_epinet": float(np.mean(self.loss_epinet)) if self.loss_epinet else float("nan"),
            f"{s}_ms_per_query_base": 1000 * self.seconds_base / len(queries),
            f"{s}_ms_per_query_epinet": 1000 * self.seconds_epinet / len(queries),
            f"{s}_eval_seconds": self.eval_seconds,
        }
        for level, observed_epinet, observed_base, observed_fitted in zip(
                COVERAGE_LEVELS, coverage_epinet, coverage_base, coverage_epinet_fitted):
            percent = int(round(level * 100))
            metrics[f"{s}_coverage{percent}_epinet"] = observed_epinet
            metrics[f"{s}_coverage{percent}_base_fitted"] = observed_base
            metrics[f"{s}_coverage{percent}_epinet_fitted"] = observed_fitted
        for model in ("base", "epinet_mean", "epinet_thompson"):
            metrics[f"{s}_regret_{model}"] = float(np.mean([q[f"regret_{model}"] for q in queries]))
            metrics[f"{s}_optimal_rate_{model}"] = float(np.mean([q[f"optimal_{model}"] for q in queries]))

        # One shared constant for every model (the entropy of the fixed evaluation noise), so
        # differences between models are unaffected by which noise each one is scored with.
        constant = excess_constant(self.noise_std)
        joint_curves = {model: [] for model in JOINT_MODELS}
        n_queries_per_tau = []
        for tau in self.taus:
            prefix = f"{s}_jnll_tau{tau}"
            covered = [q["joint"][tau] for q in queries if tau in q["joint"]]
            n_queries_per_tau.append(len(covered))
            metrics[f"{prefix}_n_queries"] = len(covered)
            if covered:
                excess = {model: float(np.mean([c[model] for c in covered])) - constant
                          for model in ("epinet", "independent", "base_fixed")}
                mean_squared_residual = float(np.mean([c["base_mean_squared_residual"] for c in covered]))
                excess["base_fitted"] = (math.log(math.sqrt(2 * math.pi) * base_noise_std)
                                         + mean_squared_residual / (2 * base_noise_std ** 2)
                                         - constant)
                for model in ("epinet", "independent"):
                    excess[f"{model}_fitted"] = float(np.mean(
                        [c[f"{model}_grid"][noise_index] for c in covered])) - constant
            else:
                excess = {model: float("nan") for model in JOINT_MODELS}
            for model in JOINT_MODELS:
                metrics[f"{prefix}_{model}_excess_per_target"] = excess[model]
                joint_curves[model].append(excess[model])
            metrics[f"{prefix}_dependence_gain"] = excess["independent"] - excess["epinet"]
            metrics[f"{prefix}_gain_vs_base_fitted"] = excess["base_fitted"] - excess["epinet"]
            metrics[f"{prefix}_gain_vs_base_fixed"] = excess["base_fixed"] - excess["epinet"]
            metrics[f"{prefix}_dependence_gain_fitted"] = excess["independent_fitted"] - excess["epinet_fitted"]
            metrics[f"{prefix}_fitted_gain_vs_base_fitted"] = excess["base_fitted"] - excess["epinet_fitted"]

        curves = {
            "taus": list(self.taus),
            "jnll_excess_per_target": joint_curves,
            "jnll_n_queries": n_queries_per_tau,
            "selective": {
                "coverage": SELECTIVE_COVERAGE_GRID.tolist(),
                "epinet": risk_epinet.tolist(),
                "oracle": risk_oracle.tolist(),
                "random": mse_epinet,
            },
            "calibration": {
                "expected": CALIBRATION_GRID.tolist(),
                "epinet": calibration_epinet.tolist(),
                "base_fitted": calibration_base.tolist(),
                "epinet_fitted": calibration_epinet_fitted.tolist(),
            },
            "coverage": {
                "levels": list(COVERAGE_LEVELS),
                "epinet": coverage_epinet,
                "base_fitted": coverage_base,
                "epinet_fitted": coverage_epinet_fitted,
            },
            "noise": {"base_fitted": base_noise_std, "epinet_fitted": epinet_noise_std,
                      "epinet_fixed": self.noise_std},
            "regret": {model: metrics[f"{s}_regret_{model}"]
                       for model in ("base", "epinet_mean", "epinet_thompson")},
        }
        return metrics, curves
