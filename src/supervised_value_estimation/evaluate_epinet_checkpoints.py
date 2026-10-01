"""Re-score saved rerun checkpoints with the epinet's observation noise fitted on validation.

    python -m src.supervised_value_estimation.evaluate_epinet_checkpoints            # every finished seed
    python -m src.supervised_value_estimation.evaluate_epinet_checkpoints 'evaluate.seeds=[0,3]'

No retraining: each seed's best checkpoint (the epoch final_summary.json selected on
validation) is loaded and evaluated on the same val/test split, with the same evaluation
seed, as during training. The base network's and the epinet's noise stds are both fitted
on validation and reused for test. Results go to <seed dir>/noise_fitted_eval/:
    metrics.json   every scalar (val_* and test_*), including the *_fitted variants
    curves.json    curve data per split, as epoch-N/curves.json
    report_<split>.png
The cross-seed summary (summary/) is rebuilt afterwards and prefers these results.

Sanity check: the fixed-noise objective is recomputed and compared with the value logged
during training. A mismatch means the checkpoint or the split did not load as trained.
"""
import gc
import json
import math
import os
from pathlib import Path

import diskcache
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from src.supervised_value_estimation.optuna_epinet_sweep import (
    _build_epinet,
    _prepare_datasets_and_plans,
    _resolve_model_paths,
)
from src.supervised_value_estimation.rerun_epinet_seeds import (
    _check_or_save_split,
    split_validation_into_val_and_test,
)
from src.supervised_value_estimation.summarize_epinet_rerun import (
    NOISE_FITTED_DIRECTORY,
    _find_runs,
    summarize_rerun,
)
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import validate_cached
from src.utils.epinet_utils.epinet_report import plot_split_report, write_json
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask

import matplotlib.pyplot as plt  # noqa: E402  (backend is set by epinet_report)

SANITY_TOLERANCE = 1e-4


def _evaluate_seed(cfg, seed, run_directory, data, device, overwrite):
    val_dataset, val_plans, test_dataset, test_plans, mean_cost, std_cost = data
    with open(os.path.join(run_directory, "final_summary.json"), encoding="utf-8") as f:
        final_summary = json.load(f)
    epoch = cfg.evaluate.epoch if cfg.evaluate.epoch is not None else final_summary["best_epoch"]
    output_directory = os.path.join(run_directory, NOISE_FITTED_DIRECTORY)
    metrics_path = os.path.join(output_directory, "metrics.json")
    if os.path.exists(metrics_path) and not overwrite:
        with open(metrics_path, encoding="utf-8") as f:
            if json.load(f).get("epoch") == epoch:
                print(f"Seed {seed}: already evaluated at epoch {epoch}, skipping "
                      f"(evaluate.overwrite=true to redo).")
                return
    os.makedirs(output_directory, exist_ok=True)

    architecture = {
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "epinet_hidden_dim": cfg.hyperparameters.epinet_hidden_dim,
        "prior_epinet_hidden_dim": cfg.hyperparameters.prior_epinet_hidden_dim,
    }
    model, _ = _build_epinet(cfg, device, seed, architecture)
    checkpoint = os.path.join(run_directory, f"epoch-{epoch}", "model", "epinet_model.pt")
    model.load_epinet(checkpoint, load_only_cost_model=False, strict=True)
    model.to(device)
    print(f"\n=== Seed {seed}: {checkpoint} ===")

    alpha_ensemble = cfg.hyperparameters.alpha_ensemble
    cache = diskcache.Cache(os.path.join(cfg.rerun.cache_root, f"seed-{seed}", "val_cache"),
                            size_limit=50 * 1024 ** 3)
    common = dict(
        epinet_cost_estimation=model, val_cache=cache,
        mean_vals={"plan_cost": mean_cost}, std_vals={"plan_cost": std_cost},
        train_loss=torch.nn.MSELoss(reduction="mean"), device=device,
        n_val_epi_indexes=cfg.hyperparameters.n_epi_indexes_val,
        sigma=cfg.hyperparameters.sigma, alpha_mlp=cfg.hyperparameters.alpha_mlp,
        alpha_ensemble=alpha_ensemble,
        precomputed_indexes=precompute_left_deep_tree_conv_index(20),
        precomputed_masks=precompute_left_deep_tree_node_mask(20),
        evaluation_noise_std=cfg.evaluation.noise_std,
        joint_taus=[int(tau) for tau in cfg.evaluation.joint_taus],
        evaluation_seed=cfg.evaluation.seed,
        zero_ensemble_prior=alpha_ensemble == 0.0,
    )
    try:
        val_accumulator = validate_cached(DataLoader(val_dataset, batch_size=1, shuffle=False),
                                          val_plans, split="val", **common)
        base_noise_std = val_accumulator.fitted_base_noise_std()
        epinet_noise_std = val_accumulator.fitted_epinet_noise_std()
        metrics, val_curves = val_accumulator.summarize(base_noise_std, epinet_noise_std)
        del val_accumulator
        test_metrics, test_curves = validate_cached(
            DataLoader(test_dataset, batch_size=1, shuffle=False), test_plans, split="test", **common
        ).summarize(base_noise_std, epinet_noise_std)
        metrics.update(test_metrics)
    finally:
        cache.close()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    objective = final_summary["objective_metric"]
    logged = final_summary["metrics_at_best_epoch"].get(objective) if epoch == final_summary["best_epoch"] else None
    if logged is not None and not math.isclose(metrics[objective], logged, abs_tol=SANITY_TOLERANCE):
        raise RuntimeError(
            f"Seed {seed}: recomputed {objective}={metrics[objective]:.6f} but training logged "
            f"{logged:.6f}. The checkpoint, split or evaluation settings differ from training; "
            f"not writing results."
        )

    curves = {"val": val_curves, "test": test_curves}
    write_json(metrics_path, {"epoch": epoch, **metrics})
    write_json(os.path.join(output_directory, "curves.json"), curves)
    for split, split_curves in curves.items():
        figure = plot_split_report(split_curves, f"{split} · seed {seed} · epoch {epoch} · fitted noise",
                                   os.path.join(output_directory, f"report_{split}.png"))
        plt.close(figure)

    print(f"Seed {seed} epoch {epoch}: noise std fitted on val -> epinet {epinet_noise_std:.4g}, "
          f"base {base_noise_std:.4g} | sanity {objective} recomputed "
          f"{metrics[objective]:.5f} vs logged {logged if logged is None else round(logged, 5)}")
    for split in ("val", "test"):
        print(f"  {split}: tau8 dependence gain fixed {metrics[f'{split}_jnll_tau8_dependence_gain']:.4f} "
              f"-> fitted {metrics[f'{split}_jnll_tau8_dependence_gain_fitted']:.4f} | "
              f"tau1 gain vs base {metrics[f'{split}_jnll_tau1_gain_vs_base_fitted']:.4f} "
              f"-> {metrics[f'{split}_jnll_tau1_fitted_gain_vs_base_fitted']:.4f} | "
              f"coverage 50/80/95 fixed "
              f"{metrics[f'{split}_coverage50_epinet']:.2f}/{metrics[f'{split}_coverage80_epinet']:.2f}/"
              f"{metrics[f'{split}_coverage95_epinet']:.3f} -> fitted "
              f"{metrics[f'{split}_coverage50_epinet_fitted']:.2f}/{metrics[f'{split}_coverage80_epinet_fitted']:.2f}/"
              f"{metrics[f'{split}_coverage95_epinet_fitted']:.3f}")


@hydra.main(
    version_base=None,
    config_path="../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
    config_name="simulated_supervised_cost_estimation_mixed_yago_rerun_trial85.yaml",
)
def main(cfg: DictConfig):
    if cfg.hyperparameters.prior_scale_target:
        raise NotImplementedError(
            "prior_scale_target rescales the alphas from a prior scale measured on shuffled "
            "training batches, which cannot be reproduced exactly here. This config has it null."
        )
    _resolve_model_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerun_directory = Path(cfg.rerun.output_directory)

    runs = {seed: directory for seed, (directory, finished) in _find_runs(str(rerun_directory)).items()
            if finished}
    seeds = sorted(runs) if cfg.evaluate.seeds is None else [s for s in cfg.evaluate.seeds if s in runs]
    missing = sorted(set(cfg.evaluate.seeds or []) - set(runs))
    if missing:
        print(f"No finished run for seeds {missing}; skipping them.")
    if not seeds:
        print(f"No finished seeds to evaluate under {rerun_directory}.")
        return
    print(f"Evaluating seeds {seeds} on {device}")

    train_dataset, val_dataset, train_plans, val_plans, mean_cost, std_cost = \
        _prepare_datasets_and_plans(cfg, device)
    del train_dataset, train_plans
    gc.collect()
    val_dataset, val_plans, test_dataset, test_plans, record = \
        split_validation_into_val_and_test(val_dataset, val_plans, cfg)
    # Refuses to continue if this is not the split the seeds were trained and scored on.
    _check_or_save_split(record, rerun_directory)

    data = (val_dataset, val_plans, test_dataset, test_plans, mean_cost, std_cost)
    for seed in seeds:
        _evaluate_seed(cfg, seed, runs[seed], data, device, cfg.evaluate.overwrite)

    summarize_rerun(rerun_directory, cfg.rerun.objective_metric)


if __name__ == "__main__":
    main()
