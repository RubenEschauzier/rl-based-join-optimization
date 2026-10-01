import gc
import hashlib
import json
import os
import shutil
from functools import partial
from pathlib import Path

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from main import find_best_epoch_directory
from src.models.epistemic_neural_network import prepare_epinet_model
from src.supervised_value_estimation.supervised_value_estimation_cached_prior import (
    prepare_cardinality_estimator,
    train_simulated_epinet_cached,
)
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.query_loading_utils import prepare_data
from src.utils.training_utils.training_tracking import ExperimentWriter


def _study_name_for_search_space(cfg) -> str:
    """Append a fingerprint of the search space to the configured study name.

    Optuna stores a distribution per parameter and refuses to change it
    ("CategoricalDistribution does not support dynamic value space"), so editing any value
    list invalidates the study -- and resuming into it fails at the first trial that
    samples the changed parameter. Fingerprinting means an edited space transparently
    starts a new study while an unchanged one still resumes.
    """
    select = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    space = {
        "sigma": [select("sweep.sigma_min"), select("sweep.sigma_max")],
        "lr": [select("sweep.tune_lr"), select("sweep.lr_min"), select("sweep.lr_max")],
        "prior_mix": [select("sweep.tune_prior_mix", False),
                      list(select("sweep.prior_mix_values", []) or []),
                      list(select("sweep.alpha_values", []) or [])],
        "widths": [select("sweep.tune_widths", False),
                   list(select("sweep.epinet_hidden_values", []) or []),
                   list(select("sweep.prior_epinet_hidden_values", []) or [])],
        "index_dim": [select("sweep.tune_index_dim", False),
                      list(select("sweep.epinet_index_dim_values", []) or [])],
        "n_epi_indexes": [select("sweep.tune_n_epi_indexes", False),
                          list(select("sweep.n_epi_indexes_values", []) or [])],
    }
    digest = hashlib.sha1(json.dumps(space, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg.sweep.study_name}-{digest}"


def _resolve_model_paths(cfg):
    OmegaConf.set_struct(cfg, False)
    best_cost_model_dir = find_best_epoch_directory(
        cfg.models.epinet.experiment_dir,
        "val_loss_cost_unscaled",
    )
    cfg.models.embedder.dir = str(find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error"))
    cfg.models.oracle.dir = str(find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_p99_q_error"))
    cfg.models.epinet.dir = str(best_cost_model_dir)
    cfg.models.epinet.model_file = str(os.path.join(best_cost_model_dir, "epinet_model.pt"))
    OmegaConf.set_struct(cfg, True)


def _prepare_datasets_and_plans(cfg, device):
    train_dataset, val_dataset = prepare_data(
        cfg.dataset.endpoint_location,
        cfg.dataset.queries_train,
        cfg.dataset.queries_val,
        cfg.dataset.rdf2vec_vector_location,
        cfg.dataset.occurrences_location,
        cfg.dataset.tp_cardinality_location,
        hll_location=cfg.dataset.hll_location,
        multiplicity_location=cfg.dataset.multiplicity_location,
    )

    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config,
        model_directory=cfg.models.oracle.dir,
    ).to(device)
    train_data = prepare_simulated_dataset(
        train_dataset,
        oracle_model,
        device,
        cfg.dataset.save_loc_simulated,
    )
    val_data = prepare_simulated_dataset(
        val_dataset,
        oracle_model,
        device,
        cfg.dataset.save_loc_simulated_val,
    )
    del oracle_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_plan_dict = {key: value for batch in train_data for key, value in batch.items()}
    val_plan_dict = {key: value for batch in val_data for key, value in batch.items()}
    # Normalisation statistics are taken from the FULL training set before subsampling,
    # so mean/stdand, therefore the target scale every metric is reported in, do not
    # drift with the sweep fraction.
    train_plans, mean_cost, std_cost = preprocess_plans(train_plan_dict)
    val_plans, _, _ = preprocess_plans(val_plan_dict, mean_cost, std_cost)

    fraction = OmegaConf.select(cfg, "sweep.data_fraction", default=None)
    seed = OmegaConf.select(cfg, "sweep.data_subset_seed", default=0)
    if fraction is not None and fraction < 1.0:
        print(f"--- Sweep running on {fraction:.0%} of the data ---")
        print("train:", end=" ")
        train_dataset, train_plans = _subsample(train_dataset, train_plans, fraction, seed)
        if OmegaConf.select(cfg, "sweep.subsample_validation", default=True):
            print("val:  ", end=" ")
            val_dataset, val_plans = _subsample(val_dataset, val_plans, fraction, seed + 1)

    return train_dataset, val_dataset, train_plans, val_plans, mean_cost, std_cost


def _subsample(dataset, plans, fraction, seed):
    """Take a deterministic random subset of *queries*, keeping plans consistent.

    Sweeps only need the ranking between configurations, not the final numbers, so
    trading absolute quality for trial throughput is usually the right call. Note the
    subset changes what the trial is optimising: fewer queries means a smaller N, and
    both the plan-count-dependent quantities and the achievable loss shift with it. Use
    the same fraction for every trial, and re-validate the winner on the full dataset.
    """
    if fraction is None or fraction >= 1.0:
        return dataset, plans

    n_queries = len(dataset)
    n_keep = max(1, int(round(n_queries * fraction)))
    generator = torch.Generator().manual_seed(seed)
    keep_indices = torch.randperm(n_queries, generator=generator)[:n_keep].tolist()

    subset = dataset[keep_indices]
    kept_queries = {dataset[i].query for i in keep_indices}
    kept_plans = {query: plan for query, plan in plans.items() if query in kept_queries}
    print(f"Subsampled {n_keep}/{n_queries} queries ({fraction:.0%}), "
          f"{sum(len(p) for p in kept_plans.values()):,} plans")
    return subset, kept_plans


def _build_epinet(cfg, device, model_seed, architecture):
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)

    model_kwargs = {
        "full_gnn_config": cfg.models.embedder.config,
        "config_ensemble_prior": cfg.models.epinet.prior_config,
        "epinet_index_dim": architecture["epinet_index_dim"],
        "mlp_dimension": cfg.hyperparameters.mlp_dimension,
        "model_weights": cfg.models.epinet.model_file,
        "cost_only": True,
        "epinet_feature_mode": OmegaConf.select(
            cfg, "hyperparameters.epinet_feature_mode", default="mlp"
        ),
        "epinet_hidden_dim": architecture["epinet_hidden_dim"],
        "prior_epinet_hidden_dim": architecture["prior_epinet_hidden_dim"],
    }
    heads_config = {
        "plan_cost": {
            "layer": torch.nn.Linear(cfg.hyperparameters.mlp_dimension, 1),
        }
    }
    heads_config_prior = {
        "plan_cost": {
            "layer": torch.nn.Linear(5, 1),
        }
    }
    model = prepare_epinet_model(
        **model_kwargs,
        device=device,
        heads_config=heads_config,
        heads_config_prior=heads_config_prior,
    )
    return model, model_kwargs


def _objective(trial, cfg, prepared_data, device, output_directory, cache_root):
    train_dataset, val_dataset, train_plans, val_plans, mean_cost, std_cost = prepared_data
    select = partial(OmegaConf.select, cfg)

    sigma = trial.suggest_float("sigma", cfg.sweep.sigma_min, cfg.sweep.sigma_max, log=True)

    # prior_scale_target rescales both alphas by a common factor so the *combined* prior
    # standard deviation hits its target, which means only their RATIO is a real degree of
    # freedom. Sampling them independently would spend trials on duplicates -- (0.03,0.03),
    # (0.05,0.05) and (0.1,0.1) are the same model -- and would eventually sample (0,0),
    # where the calibration divides by a zero prior scale. One mix parameter instead:
    #   0.0 -> GNN-ensemble prior only, 1.0 -> MLP prior only.
    # The endpoints are the ablation that tests whether the GNN prior, which is computed
    # from an embedding the learnable net never sees, is cancellable at all.
    if select("sweep.tune_prior_mix", default=False):
        prior_mix = trial.suggest_categorical("prior_mix", list(cfg.sweep.prior_mix_values))
        alpha_mlp, alpha_ensemble = prior_mix, 1.0 - prior_mix
    else:
        alpha_mlp = trial.suggest_categorical("alpha_mlp", list(cfg.sweep.alpha_values))
        alpha_ensemble = trial.suggest_categorical("alpha_ensemble", list(cfg.sweep.alpha_values))
        if alpha_mlp == 0.0 and alpha_ensemble == 0.0:
            raise optuna.TrialPruned("Both prior weights are zero: no prior to calibrate.")
    lr = cfg.hyperparameters.lr
    if cfg.sweep.tune_lr:
        lr = trial.suggest_float("lr", cfg.sweep.lr_min, cfg.sweep.lr_max, log=True)


    architecture = {
        "epinet_index_dim": cfg.hyperparameters.epinet_index_dim,
        "epinet_hidden_dim": select("hyperparameters.epinet_hidden_dim", default=None),
        "prior_epinet_hidden_dim": select("hyperparameters.prior_epinet_hidden_dim", default=None),
    }
    if select("sweep.tune_widths", default=False):
        architecture["epinet_hidden_dim"] = trial.suggest_categorical(
            "epinet_hidden_dim", list(cfg.sweep.epinet_hidden_values)
        )
        architecture["prior_epinet_hidden_dim"] = trial.suggest_categorical(
            "prior_epinet_hidden_dim", list(cfg.sweep.prior_epinet_hidden_values)
        )
    if select("sweep.tune_index_dim", default=False):
        architecture["epinet_index_dim"] = trial.suggest_categorical(
            "epinet_index_dim", list(cfg.sweep.epinet_index_dim_values)
        )

    n_epi_indexes_train = cfg.hyperparameters.n_epi_indexes_train
    if select("sweep.tune_n_epi_indexes", default=False):
        n_epi_indexes_train = trial.suggest_categorical(
            "n_epi_indexes_train", list(cfg.sweep.n_epi_indexes_values)
        )

    model, model_kwargs = _build_epinet(cfg, device, cfg.sweep.model_seed, architecture)

    # The frozen priors depend ONLY on (model_seed, epinet_index_dim, prior config): the
    # ensemble is built before any of the tuned epinet layers, from a fixed seed, so two
    # trials sharing an index dimension have bit-identical priors. Keying the cache by
    # trial number therefore threw away every entry and recomputed the whole ensemble --
    # a sequential pass over `epinet_index_dim` GNNs per query, plus a tree-conv pass per
    # plan -- once per trial. With index_dim searched over four values that is four cache
    # builds instead of `n_trials`.
    prior_signature = "-".join([
        f"seed{cfg.sweep.model_seed}",
        f"idx{architecture['epinet_index_dim']}",
        Path(str(cfg.models.epinet.prior_config)).stem,
    ])
    trial_cache = cache_root / prior_signature

    trial_config = OmegaConf.to_container(cfg, resolve=True)
    trial_config["sampled_hyperparameters"] = dict(trial.params)
    writer = ExperimentWriter(
        str(output_directory),
        f"trial-{trial.number}",
        trial_config,
        trial.params,
    )
    writer.create_experiment_directory()

    trial.set_user_attr("experiment_directory", writer.experiment_directory)
    trial.set_user_attr("cache_directory", str(trial_cache))
    print(f"\n--- Trial {trial.number}: {trial.params} ---")

    try:
        best_result = train_simulated_epinet_cached(
            queries_train=train_dataset,
            query_plans_train=train_plans,
            mean_train={"plan_cost": mean_cost},
            std_train={"plan_cost": std_cost},
            queries_val=val_dataset,
            query_plans_val=val_plans,
            model_builder_fn=prepare_epinet_model,
            model_kwargs=model_kwargs,
            model_state_dict=model.state_dict(),
            epinet_cost_estimation=model,
            device=device,
            query_batch_size=cfg.hyperparameters.query_batch_size,
            n_epi_indexes_train=n_epi_indexes_train,
            sigma=sigma,
            alpha_mlp=alpha_mlp,
            alpha_ensemble=alpha_ensemble,
            lr=lr,
            weight_decay=cfg.hyperparameters.weight_decay,
            n_epochs=cfg.sweep.max_epochs,
            n_epi_indexes_val=cfg.hyperparameters.n_epi_indexes_val,
            writer=writer,
            cache_directory=str(trial_cache),
            clear_prior_cache=False,   # shared across trials; see prior_signature above
            validate_every=select("sweep.validate_every", default=1),
            plot_calibration=select("sweep.plot_calibration", default=False),
            evaluation_noise_std=cfg.evaluation.noise_std,
            joint_taus=cfg.evaluation.joint_taus,
            evaluation_seed=cfg.evaluation.seed,
            objective_metric=cfg.sweep.objective_metric,
            early_stopping_patience=cfg.sweep.early_stopping_patience,
            early_stopping_min_delta=cfg.sweep.early_stopping_min_delta,
            early_stopping_min_epochs=cfg.sweep.early_stopping_min_epochs,
            l2_mode=select("hyperparameters.l2_mode", default="manual"),
            prior_scale_target=select("hyperparameters.prior_scale_target", default=None),
            debug_single_batch=cfg.debug.debug_single_batch,
            trial=trial,
        )
        trial.set_user_attr("best_epoch", best_result["epoch"])
        return best_result["value"]
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Deliberately NOT removing trial_cache: it is shared by every trial with the
        # same prior signature and is the single most expensive thing to rebuild.


@hydra.main(
    version_base=None,
    config_path="../../experiments/experiment_configs/epinet_cost_estimation/cost_estimation_yago_mixed",
    config_name="simulated_supervised_cost_estimation_mixed_yago_optuna.yaml",
)
def main(cfg: DictConfig):
    _resolve_model_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prepared_data = _prepare_datasets_and_plans(cfg, device)
    output_directory = Path(cfg.sweep.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    cache_root = Path(cfg.sweep.cache_root)
    # Kept across runs so a resumed sweep reuses the priors it already computed. Delete it
    # by hand if the prior config or model seed changes in a way the signature misses.
    cache_root.mkdir(parents=True, exist_ok=True)

    storage_path = Path(cfg.sweep.storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=cfg.sweep.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.sweep.pruner_startup_trials,
        n_warmup_steps=cfg.sweep.pruner_warmup_epochs,
        interval_steps=1,
    )
    study_name = _study_name_for_search_space(cfg)
    existing = optuna.study.get_all_study_names(f"sqlite:///{storage_path}") \
        if storage_path.exists() else []
    print(f"Study: {study_name}"
          f"{'  (resuming)' if study_name in existing else '  (new -- search space changed)'}")
    for other in existing:
        if other != study_name:
            print(f"  sibling study in this storage, left untouched: {other}")

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = partial(
        _objective,
        cfg=cfg,
        prepared_data=prepared_data,
        device=device,
        output_directory=output_directory,
        cache_root=cache_root,
    )
    remaining_trials = max(0, cfg.sweep.n_trials - len(study.trials))
    study.optimize(objective, n_trials=remaining_trials, n_jobs=1, gc_after_trial=True)

    study.trials_dataframe().to_csv(output_directory / "trials.csv", index=False)
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        print(f"Best objective: {study.best_value:.6f}")
        print(f"Best parameters: {study.best_params}")
        print(f"Best run: {study.best_trial.user_attrs.get('experiment_directory')}")
    else:
        print("No trials completed successfully.")


if __name__ == "__main__":
    main()