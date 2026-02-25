import os

import optuna
import torch
from datetime import datetime
from functools import partial

from main import find_best_epoch_directory
from src.models.epistemic_neural_network import EpistemicNetwork
from src.models.query_plan_prediction_model import PlanCostEstimatorFull, QueryPlansPredictionModel
from src.supervised_value_estimation import train_simulated_cost_model, prepare_data, prepare_cardinality_estimator
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.training_tracking import ExperimentWriter


def objective(trial, train_dataset, train_plans, mean_train, std_train,
              val_dataset, val_plans, model_config_epistemic_prior,
              combined_model, trained_cost_model_loc, device, base_dir):
    alpha_mlp = trial.suggest_float('alpha_mlp', 0.05, 1)
    alpha_ensemble = trial.suggest_float('alpha_ensemble', 0.05, 1)
    sigma = trial.suggest_float('sigma', 0.05, 1.0)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    epi_index_dim = trial.suggest_int('epi_index_dim', 16, 64)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Testing params: {trial.params}")

    # 2. Instantiate a fresh epinet for this specific run, wrapping the shared frozen base model
    epinet_cost_estimation = EpistemicNetwork(
        epi_index_dim=epi_index_dim,
        prior_config=model_config_epistemic_prior,
        cost_estimation_model=combined_model,
        device=device
    )
    epinet_cost_estimation.load_epinet(trained_cost_model_loc, load_only_cost_model=True)
    epinet_cost_estimation.to(device)

    # 3. Clean string formatting for the run name
    run_name = (
        f"trial_{trial.number}_"
        f"am_{alpha_mlp:.2f}_"
        f"ae_{alpha_ensemble:.2f}_"
        f"sig_{sigma:.2f}_"
        f"wd_{weight_decay:.4f}_"
        f"dim_{epi_index_dim}"
    )

    writer = ExperimentWriter(base_dir, run_name, trial.params, trial.params)
    writer.create_experiment_directory()

    # 4. Execute the training loop
    best_joint_nll_run = train_simulated_cost_model(
        queries_train=train_dataset, query_plans_train=train_plans,
        mean_train=mean_train, std_train=std_train,
        queries_val=val_dataset, query_plans_val=val_plans,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        query_batch_size=8,
        n_epi_indexes_train=16, n_epi_indexes_val=100,
        train_epi_network=True,
        writer=writer,
        sigma=sigma,
        alpha_mlp=alpha_mlp,
        alpha_ensemble=alpha_ensemble,
        lr=lr,
        weight_decay=weight_decay,
        n_epochs=15
    )

    print(f"Trial {trial.number} finished. Best Val Joint NLL: {best_joint_nll_run['value']:.4f}")

    return best_joint_nll_run['value'], epi_index_dim


def main_optuna_search(endpoint_location, queries_location_train, queries_location_val,
                       rdf2vec_vector_location, save_loc_simulated, save_loc_simulated_val,
                       occurrences_location, tp_cardinality_location,
                       model_config_oracle, model_dir_oracle,
                       model_config_emb, model_dir_emb,
                       model_config_epistemic_prior, trained_cost_model_loc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_train_queries = 2500
    max_val_queries = 250

    # Data Preparation (Happens once)
    print("Preparing datasets and base models...")
    train_dataset, val_dataset = prepare_data(
        endpoint_location, queries_location_train, queries_location_val,
        rdf2vec_vector_location, occurrences_location, tp_cardinality_location
    )

    # PyTorch Geometric dataset subsetting and shuffling
    train_dataset = train_dataset.shuffle()[:max_train_queries]
    val_dataset = val_dataset.shuffle()[:max_val_queries]

    # oracle_model = prepare_cardinality_estimator(model_config=model_config_oracle, model_directory=model_dir_oracle)
    cost_net_attention_pooling = PlanCostEstimatorFull(device, 100)
    embedding_model = prepare_cardinality_estimator(model_config=model_config_emb, model_directory=model_dir_emb)
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, device)

    # Lock the shared base model in evaluation mode to prevent state leakage across threads
    combined_model.eval()

    # Pre-process datasets
    data = prepare_simulated_dataset(train_dataset, "should-be-oracle", device, save_loc_simulated)
    query_plans_dict = {k: v for d in data for k, v in d.items()}
    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)

    val_data = prepare_simulated_dataset(val_dataset, "should-be-oracle", device, save_loc_simulated_val)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}
    val_plans, _, _ = preprocess_plans(query_plans_dict_val, mean_train, std_train)

    # Generate a single timestamp directory for all Optuna trials
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    base_dir = f"experiments/experiment_outputs/yago_gnce/optuna_search-{timestamp}"

    # 4. Bind the shared objects to the objective function using functools.partial
    bound_objective = partial(
        objective,
        train_dataset=train_dataset,
        train_plans=train_plans,
        mean_train=mean_train,
        std_train=std_train,
        val_dataset=val_dataset,
        val_plans=val_plans,
        model_config_epistemic_prior=model_config_epistemic_prior,
        combined_model=combined_model,
        trained_cost_model_loc=trained_cost_model_loc,
        device=device,
        base_dir=base_dir
    )
    # Only prune the worst 10% of trials
    conservative_pruner = optuna.pruners.PercentilePruner(
        percentile=60,  # 90th percentile of 'badness' (if minimizing)
        n_startup_trials=5,
        n_warmup_steps=4
    )

    print("Starting Optuna hyperparameter search...")
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="epinet_join_cost_estimation",
        pruner=conservative_pruner,
        storage="sqlite:///epinet_optuna.db",
        load_if_exists=True
    )

    study.optimize(bound_objective, n_trials=50, n_jobs=1)

    # 6. Results
    print("\n=================================================")
    print(f"Optuna Search Complete.")
    print(f"Best Joint Gaussian NLL: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=================================================")


if __name__ == '__main__':
    endpoint_location = "http://localhost:9999/blazegraph/namespace/yago/sparql"
    queries_location_train = "data/generated_queries/star_yago_gnce/dataset_train"
    queries_location_val = "data/generated_queries/star_yago_gnce/dataset_val"
    rdf2vec_vector_location = "data/rdf2vec_embeddings/yago_gnce/model.json"
    occurrences_location = "data/term_occurrences/yago_gnce/occurrences.json"
    tp_cardinality_location = "data/term_occurrences/yago_gnce/tp_cardinalities.json"

    model_config_oracle = "experiments/model_configs/policy_networks/t_cv_repr_huge.yaml"
    model_config_emb = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head_own_embeddings.yaml"
    model_config_emb_pair_norm = "experiments/model_configs/policy_networks/t_cv_repr_pair_norm_cardinality_head_own_embeddings.yaml"
    model_config_emb_graph_norm = "experiments/model_configs/policy_networks/t_cv_repr_graph_norm_cardinality_head_own_embeddings.yaml"

    model_config_prior = "experiments/model_configs/prior_networks/prior_t_cv_tiny.yaml"

    oracle_experiment_dir = "experiments/experiment_outputs/yago_gnce/pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars_gnce_large_pretrain-05-10-2025-18-13-40"
    trained_cost_model_file = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training/simulated_cost-12-02-2026-17-17-13/epoch-25/model/epinet_model.pt"

    emb_experiment_dir = ("experiments/experiment_outputs/yago_gnce/pretrained_models/"
                      "pretrain_experiment_triple_conv-15-12-2025-11-10-45")
    emb_experiment_dir_pair_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_pair_norm-15-12-2025-10-00-26")

    emb_experiment_dir_graph_norm = ("experiments/experiment_outputs/yago_gnce/pretrained_models"
                                "/pretrain_experiment_triple_conv_graph_norm-15-12-2025-09-12-57")

    save_loc_simulated = "data/simulated_query_plan_data/star_yago_gnce/data"
    save_loc_simulated_val = "data/simulated_query_plan_data/star_yago_gnce/val_data"

    # model_dir_oracle = find_best_epoch_directory(oracle_experiment_dir, "val_q_error")
    model_dir_oracle = os.path.join(oracle_experiment_dir, "epoch-39/model")
    model_dir_embedder = find_best_epoch_directory(emb_experiment_dir, "val_q_error")

    main_optuna_search(endpoint_location, queries_location_train, queries_location_val,
                       rdf2vec_vector_location, save_loc_simulated, save_loc_simulated_val,
                       occurrences_location, tp_cardinality_location,
                       model_config_oracle, model_dir_oracle,
                       model_config_emb, model_dir_embedder,
                       model_config_prior, trained_cost_model_file)