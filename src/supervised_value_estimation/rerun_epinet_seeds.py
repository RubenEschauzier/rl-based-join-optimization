"""Full-data, multi-seed rerun of one epinet configuration, with a held-out test split.

    python -m src.supervised_value_estimation.rerun_epinet_seeds              # all seeds
    python -m src.supervised_value_estimation.rerun_epinet_seeds rerun.seed=3 # one seed

Each seed writes <output_directory>/seed-<s>-<timestamp>/ (see epinet_report for the
layout). When every configured seed has finished, the cross-seed summary is written to
<output_directory>/summary/; it can also be rebuilt at any time with
`python -m src.supervised_value_estimation.summarize_epinet_rerun <output_directory>`.
"""
import gc
import glob
import json
import os
import random
import signal
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.supervised_value_estimation.optuna_epinet_sweep import (
    _build_epinet,
    _prepare_datasets_and_plans,
    _resolve_model_paths,
    _subsample,
)
from src.supervised_value_estimation.summarize_epinet_rerun import summarize_rerun
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import train_simulated_epinet_cached
from src.utils.training_utils.training_tracking import ExperimentWriter


def split_validation_into_val_and_test(val_dataset, val_plans, cfg):
    """Carve a test split out of the validation file, deterministically.

    Queries the sweep validated on (its subsample of the validation file) always stay in
    validation: trial 85 was chosen on them, so a test set containing them would not be
    held out. Returns (val_dataset, val_plans, test_dataset, test_plans, split_record).
    """
    queries = [val_dataset[i].query for i in range(len(val_dataset))]
    sweep_subset, _ = _subsample(
        val_dataset, val_plans, cfg.rerun.sweep_validation_fraction, cfg.rerun.sweep_validation_seed
    )
    sweep_queries = {sweep_subset[i].query for i in range(len(sweep_subset))}

    candidates = [i for i, query in enumerate(queries) if query not in sweep_queries]
    n_test = min(int(round(cfg.rerun.test_fraction * len(queries))), len(candidates))
    generator = torch.Generator().manual_seed(cfg.rerun.split_seed)
    test_indices = sorted(candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:n_test].tolist())
    test_set = set(test_indices)
    val_indices = [i for i in range(len(queries)) if i not in test_set]

    def take(indices):
        kept = {queries[i] for i in indices}
        return val_dataset[indices], {q: plans for q, plans in val_plans.items() if q in kept}

    val_split, val_split_plans = take(val_indices)
    test_split, test_split_plans = take(test_indices)
    record = {
        "val_queries": [queries[i] for i in val_indices],
        "test_queries": [queries[i] for i in test_indices],
        "n_sweep_validation_queries_forced_into_val": len(sweep_queries),
        "test_fraction": cfg.rerun.test_fraction,
        "split_seed": cfg.rerun.split_seed,
    }
    print(f"Split -> val: {len(val_indices)} queries "
          f"({sum(len(p) for p in val_split_plans.values()):,} plans, incl. {len(sweep_queries)} "
          f"the sweep selected on) | test: {len(test_indices)} queries "
          f"({sum(len(p) for p in test_split_plans.values()):,} plans)")
    return val_split, val_split_plans, test_split, test_split_plans, record


def _check_or_save_split(record, output_directory):
    """Every seed must see the same split; refuse to continue if the saved one differs."""
    path = output_directory / "split.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved["val_queries"] != record["val_queries"] or saved["test_queries"] != record["test_queries"]:
            raise RuntimeError(
                f"{path} holds a different val/test split than the one just built. Seeds would "
                f"not be comparable; move the old outputs away or restore the old split settings."
            )
        return
    # Seeds usually start as parallel jobs; write via a private temp file and an atomic
    # rename so no job can ever read a half-written split.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    os.replace(temporary, path)


def _exit_for_restart(signum, frame):
    """GPULab halts a job with SIGUSR1 and expects exit status 123 within 15 seconds; with
    `restartable: true` the job is then re-queued. The interrupted seed has no
    final_summary.json, so the restarted job simply trains it again from scratch."""
    print("Received SIGUSR1 (GPULab halt): exiting with 123 so the job is re-queued.", flush=True)
    sys.stdout.flush()
    os._exit(123)


def _seed_is_finished(output_directory, seed):
    return bool(glob.glob(str(output_directory / f"seed-{seed}-*" / "final_summary.json")))


def _run_seed(cfg, seed, device, data, output_directory):
    train_dataset, train_plans, val_dataset, val_plans, test_dataset, test_plans, mean_cost, std_cost = data
    random.seed(seed)
    np.random.seed(seed)
    architecture = {
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "epinet_hidden_dim": cfg.hyperparameters.epinet_hidden_dim,
        "prior_epinet_hidden_dim": cfg.hyperparameters.prior_epinet_hidden_dim,
    }
    # Seeds torch (CPU and CUDA) before building: prior and epinet initialisation, then
    # training order and index sampling, all follow from `seed`.
    model, model_kwargs = _build_epinet(cfg, device, seed, architecture)

    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config["seed"] = seed
    writer = ExperimentWriter(
        str(output_directory), f"seed-{seed}", run_config,
        {"seed": seed, **architecture, "n_epi_indexes_train": cfg.hyperparameters.n_epi_indexes_train},
    )
    writer.create_experiment_directory()
    print(f"\n=== Seed {seed} -> {writer.experiment_directory} ===")

    try:
        return train_simulated_epinet_cached(
            queries_train=train_dataset,
            query_plans_train=train_plans,
            mean_train={"plan_cost": mean_cost},
            std_train={"plan_cost": std_cost},
            queries_val=val_dataset,
            query_plans_val=val_plans,
            queries_test=test_dataset,
            query_plans_test=test_plans,
            model_builder_fn=None,
            model_kwargs=model_kwargs,
            model_state_dict=model.state_dict(),
            epinet_cost_estimation=model,
            device=device,
            query_batch_size=cfg.hyperparameters.query_batch_size,
            n_epi_indexes_train=cfg.hyperparameters.n_epi_indexes_train,
            sigma=cfg.hyperparameters.sigma,
            alpha_mlp=cfg.hyperparameters.alpha_mlp,
            alpha_ensemble=cfg.hyperparameters.alpha_ensemble,
            lr=cfg.hyperparameters.lr,
            weight_decay=cfg.hyperparameters.weight_decay,
            n_epochs=cfg.rerun.n_epochs,
            n_epi_indexes_val=cfg.hyperparameters.n_epi_indexes_val,
            writer=writer,
            cache_directory=str(Path(cfg.rerun.cache_root) / f"seed-{seed}"),
            clear_prior_cache=True,
            validate_every=cfg.rerun.validate_every,
            plot_calibration=cfg.rerun.write_figures,
            use_tensorboard=cfg.rerun.tensorboard,
            evaluation_noise_std=cfg.evaluation.noise_std,
            joint_taus=list(cfg.evaluation.joint_taus),
            evaluation_seed=cfg.evaluation.seed,
            anchor_seed=seed,
            objective_metric=cfg.rerun.objective_metric,
            early_stopping_patience=cfg.rerun.early_stopping_patience,
            early_stopping_min_delta=cfg.rerun.early_stopping_min_delta,
            early_stopping_min_epochs=cfg.rerun.early_stopping_min_epochs,
            l2_mode=cfg.hyperparameters.l2_mode,
            prior_scale_target=cfg.hyperparameters.prior_scale_target,
            debug_single_batch=cfg.debug.debug_single_batch,
        )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@hydra.main(
    version_base=None,
    config_path="../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
    config_name="simulated_supervised_cost_estimation_mixed_yago_rerun_trial85.yaml",
)
def main(cfg: DictConfig):
    signal.signal(signal.SIGUSR1, _exit_for_restart)
    _resolve_model_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_directory = Path(cfg.rerun.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # What THIS invocation trains: one seed, a subset run back to back, or all of them.
    # `rerun.seeds` stays the full experiment, so the summary waits for every seed.
    if cfg.rerun.seed is not None:
        seeds = [cfg.rerun.seed]
    elif cfg.rerun.run_seeds is not None:
        seeds = list(cfg.rerun.run_seeds)
    else:
        seeds = list(cfg.rerun.seeds)
    unknown = sorted(set(seeds) - set(cfg.rerun.seeds))
    if unknown:
        print(f"WARNING: seeds {unknown} are not in rerun.seeds; the cross-seed summary "
              f"will not wait for them.")
    pending = [seed for seed in seeds if not _seed_is_finished(output_directory, seed)]
    for seed in sorted(set(seeds) - set(pending)):
        print(f"Seed {seed} already finished, skipping.")

    if pending:
        # Full data: no sweep.data_fraction in this config, so nothing is subsampled.
        train_dataset, val_dataset, train_plans, val_plans, mean_cost, std_cost = \
            _prepare_datasets_and_plans(cfg, device)
        val_dataset, val_plans, test_dataset, test_plans, record = \
            split_validation_into_val_and_test(val_dataset, val_plans, cfg)
        _check_or_save_split(record, output_directory)
        data = (train_dataset, train_plans, val_dataset, val_plans, test_dataset, test_plans,
                mean_cost, std_cost)
        for seed in pending:
            _run_seed(cfg, seed, device, data, output_directory)

    if all(_seed_is_finished(output_directory, seed) for seed in cfg.rerun.seeds):
        summarize_rerun(output_directory, cfg.rerun.objective_metric)


if __name__ == "__main__":
    main()
