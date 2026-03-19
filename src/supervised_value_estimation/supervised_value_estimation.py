import os
import sys

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from torchmetrics.regression import MeanAbsolutePercentageError
from torch_geometric.loader import DataLoader

from src.models.epistemic_neural_network import MultiHeadEpistemicNetwork
from src.models.query_plan_prediction_model import PlanCostEstimatorFull, QueryPlansPredictionModel
from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss
from src.utils.epinet_utils.simulated_plan_cost_dataset import prepare_simulated_dataset, preprocess_plans
from src.utils.training_utils.training_tracking import TrainSummary, ExperimentWriter

# Get the path of the parent directory (the root of the project)
# This finds the directory of the current script (__file__), goes up one level ('...'),
# and then converts it to an absolute path for reliability.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert the project root path at the beginning of the search path (sys.path)
# This forces Python to look in the parent directory first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import find_best_epoch_directory
from src.models.model_instantiator import ModelFactory
from src.rl_fine_tuning_qr_dqn_learning import load_weights_from_pretraining
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset, prepare_data
from src.utils.tree_conv_utils import precompute_left_deep_tree_conv_index, precompute_left_deep_tree_node_mask
import torch


def prepare_cardinality_estimator(model_config, model_directory=None):
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    if model_directory:
        load_weights_from_pretraining(gine_conv_model, model_directory,
                                      "embedding_model.pt",
                                      ["head_cardinality.pt"],
                                      float_weights=True)
    return gine_conv_model


def validate(queries_val, query_plans_val,
             precomputed_indexes, precomputed_masks,
             mean_cost, std_cost,
             train_loss,
             epinet_cost_estimation,
             device,
             ):
    mape = MeanAbsolutePercentageError()
    joint_loss = GaussianJointLogLoss()
    mape.to(device)

    val_loader = DataLoader(queries_val, batch_size=1, shuffle=False)

    query_to_val_metrics = {}
    for queries in tqdm(val_loader, total=len(val_loader)):
        with torch.no_grad():
            embedded = epinet_cost_estimation.embed_query_batched(queries)
            plans = query_plans_val[queries.query[0]]

            estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                plans, embedded[0], precomputed_indexes, precomputed_masks
            )

            query_metrics = {}

            estimated_cost = estimated_cost["plan_cost"].squeeze()
            original_cost = (estimated_cost * std_cost) + mean_cost

            target = torch.tensor([plan[1] for plan in plans], device=device)
            original_target = (target * std_cost) + mean_cost

            mape_val_scaled = mape(estimated_cost, target)
            mape_val_unscaled = mape(original_cost, original_target)

            val_loss_scaled = train_loss(estimated_cost, target)
            val_loss_unscaled = train_loss(original_cost, original_target)

            joint_nll_no_epinet = joint_loss(estimated_cost.unsqueeze(0), target)

            query_metrics["val_loss_cost_scaled"] = val_loss_scaled.cpu().item()
            query_metrics["val_loss_cost_unscaled"] = val_loss_unscaled.cpu().item()

            query_metrics["val_mape_cost_scaled"] = mape_val_scaled.cpu().item()
            query_metrics["val_mape_cost_unscaled"] = mape_val_unscaled.cpu().item()
            query_metrics["val_joint_nll_no_epinet"] = joint_nll_no_epinet.cpu().item()

            query_to_val_metrics[queries.query[0]] = query_metrics

    return query_to_val_metrics

def summarize_metrics(metrics_dict, train_loss):
    keys = list(metrics_dict.values())[0].keys()

    summary = {
        k: np.mean([q_metrics[k] for q_metrics in metrics_dict.values()]).item()
        for k in keys
    }
    print(f"Cost estimation train loss: {train_loss:.4f}\n")
    print(
        "Cost estimation metrics\n"
        f"  Validation Loss Scaled : {summary['val_loss_cost_scaled']:.4f}\n"
        f"  Validation MAPE Scaled : {summary['val_mape_cost_scaled']:.4f}\n"
        f"  Validation Loss True : {summary['val_loss_cost_unscaled']:.4f}\n"
        f"  Validation MAPE True : {summary['val_mape_cost_unscaled']:.4f}\n"
        f"  Validation joint NLL base: {summary['val_joint_nll_no_epinet']:.4f}"

    )
    return summary


def validation_step(epoch_train_loss, epoch,
                    train_summary, writer,
                    queries_val, query_plans_val,
                    precomputed_indexes, precomputed_masks,
                    mean_train, std_train,
                    loss,
                    epinet_cost_estimation,
                    device):
    query_to_val_cost = validate(queries_val, query_plans_val,
                                 precomputed_indexes, precomputed_masks,
                                 mean_train, std_train,
                                 loss,
                                 epinet_cost_estimation,
                                 device,
                                 )

    mean_metrics_val = summarize_metrics(query_to_val_cost, epoch_train_loss.item())
    mean_metrics_val["train_loss"] = epoch_train_loss.item()
    train_summary.update(mean_metrics_val, epoch)

    best, per_epoch = train_summary.summary()
    writer.write_epoch_to_file([], best, per_epoch, epinet_cost_estimation, epoch)
    return mean_metrics_val['val_loss_cost_scaled'], None


def train_simulated_cost_model(queries_train, query_plans_train,
                               mean_train, std_train,
                               queries_val, query_plans_val,
                               epinet_cost_estimation: MultiHeadEpistemicNetwork,
                               device,
                               query_batch_size,
                               # Hyperparameters
                               lr, weight_decay, n_epochs,
                               writer,
                               first_validate=False,
                               trial: optuna.Trial = None):
    train_summary = TrainSummary([("train_loss", "min"), ("val_loss_cost_scaled", "min"),
                                  ("val_loss_cost_unscaled", "min"), ("val_mape_cost_scaled", "min"),
                                  ("val_mape_cost_unscaled", "min"),
                                  ("val_joint_nll_no_epinet", "min")])
    epinet_cost_estimation.to(device)

    precomputed_indexes = precompute_left_deep_tree_conv_index(20)
    precomputed_masks = precompute_left_deep_tree_node_mask(20)
    loader = DataLoader(queries_train, batch_size=query_batch_size, shuffle=True)

    params = list(epinet_cost_estimation.parameters())
    params_cost_estimate = list(epinet_cost_estimation.cost_estimation_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2
                                  )
    previous_lr = scheduler.get_last_lr()

    total_params_cost_estimation = 0
    for param in params_cost_estimate:
        total_params_cost_estimation += param.numel()
    print(f"Cost estimation model has {total_params_cost_estimation} parameters")

    loss = torch.nn.MSELoss(reduction='mean')

    # First validation run to see performance of randomly initialized model
    if first_validate:
        validation_step(np.array(-1), 0, train_summary, writer,
                        queries_val, query_plans_val,
                        precomputed_indexes, precomputed_masks,
                        mean_train, std_train,
                        loss,
                        epinet_cost_estimation,
                        device)

    for epoch in range(1, n_epochs + 1):
        query_loss_epoch = []
        for k, queries in tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()

            embedded = epinet_cost_estimation.embed_query_batched(queries)

            total_loss_tensor = torch.tensor(0.0, device=device)
            for i in range(len(queries.query)):
                if queries.query[i] not in query_plans_train:
                    continue
                plans = query_plans_train[queries.query[i]]
                estimated_cost, last_feature = epinet_cost_estimation.estimate_cost_full(
                    plans, embedded[i], precomputed_indexes, precomputed_masks
                )

                target = torch.tensor([plan[1] for plan in plans], device=device).squeeze()
                total_loss_tensor += loss(estimated_cost["plan_cost"].squeeze(), target)

            total_loss_tensor.backward()
            optimizer.step()

            query_loss_epoch.append(total_loss_tensor.detach().cpu().item() / query_batch_size)

        epoch_train_loss = np.mean(query_loss_epoch)
        val_loss, joint_nll = validation_step(epoch_train_loss, epoch, train_summary, writer,
                                              queries_val, query_plans_val,
                                              precomputed_indexes, precomputed_masks,
                                              mean_train, std_train,
                                              loss,
                                              epinet_cost_estimation,
                                              device)

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(val_loss)

    return train_summary.best_values["val_loss_cost_unscaled"]


def main_simulated_training(cfg: DictConfig,
                            train_dataset,
                            val_dataset,
                            oracle_model,
                            epinet_cost_estimation,
                            device,
                            writer):
    writer.create_experiment_directory()

    # Prepare datasets
    data = prepare_simulated_dataset(train_dataset, oracle_model, device, cfg.dataset.save_loc_simulated,
                                     max_plans_per_relation=50)
    query_plans_dict = {k: v for d in data for k, v in d.items()}

    val_data = prepare_simulated_dataset(val_dataset, oracle_model, device, cfg.dataset.save_loc_simulated_val,
                                         max_plans_per_relation=50)
    query_plans_dict_val = {k: v for d in val_data for k, v in d.items()}

    train_plans, mean_train, std_train = preprocess_plans(query_plans_dict)
    val_plans, _, _ = preprocess_plans(query_plans_dict_val)

    # Execute training
    train_simulated_cost_model(
        queries_train=train_dataset,
        query_plans_train=train_plans,
        mean_train=mean_train,
        std_train=std_train,
        queries_val=val_dataset,
        query_plans_val=val_plans,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer,

        # Hyperparameters
        query_batch_size=cfg.hyperparameters.query_batch_size,
        lr=cfg.hyperparameters.lr,
        weight_decay=cfg.hyperparameters.weight_decay,
        n_epochs=cfg.hyperparameters.n_epochs
    )


def main_supervised_value_estimation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = prepare_data(
        cfg.dataset.endpoint_location,
        cfg.dataset.queries_train,
        cfg.dataset.queries_val,
        cfg.dataset.rdf2vec_vector_location,
        cfg.dataset.occurrences_location,
        cfg.dataset.tp_cardinality_location
    )

    embedding_model = prepare_cardinality_estimator(
        model_config=cfg.models.embedder.config,
        model_directory=cfg.models.embedder.dir
    )

    heads_config = {
        'plan_cost': {
            'layer': torch.nn.Linear(cfg.hyperparameters.mlp_dimension, 1),
        }
    }

    # Prepare large (20 million parameters) oracle model to estimate cardinality of join plans
    oracle_model = prepare_cardinality_estimator(
        model_config=cfg.models.oracle.config, model_directory=cfg.models.oracle.dir
    ).to(device)

    # Prepare plan cost estimation models and epinet
    cost_net_attention_pooling = PlanCostEstimatorFull(
        heads_config, device, mlp_output_dim=cfg.hyperparameters.mlp_dimension
    )
    combined_model = QueryPlansPredictionModel(embedding_model, cost_net_attention_pooling, device)
    epinet_cost_estimation = MultiHeadEpistemicNetwork(
        2, cfg.models.epinet.prior_config, combined_model, device=device
    )

    experiment_base_dir = "experiments/experiment_outputs/yago_gnce/supervised_epinet_training"
    experiment_name = "simulated_cost"

    writer = ExperimentWriter(experiment_base_dir, experiment_name, {}, {})

    main_simulated_training(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        oracle_model=oracle_model,
        epinet_cost_estimation=epinet_cost_estimation,
        device=device,
        writer=writer
    )


@hydra.main(version_base=None,
            config_path="../../experiments/experiment_configs/epinet_cost_estimation",
            config_name="simulated_supervised_cost_estimation_v2.yaml")
def main(cfg: DictConfig):
    # Temporarily unlock the config to allow dynamic updates
    OmegaConf.set_struct(cfg, False)

    # Locate the best directories of pretrained models dynamically
    best_embedder_dir = find_best_epoch_directory(cfg.models.embedder.experiment_dir, "val_q_error")
    best_oracle_dir = find_best_epoch_directory(cfg.models.oracle.experiment_dir, "val_q_error")

    # Inject the resolved path directly into the config state
    cfg.models.embedder.dir = best_embedder_dir
    cfg.models.oracle.dir = best_oracle_dir

    # Relock the config to prevent accidental downstream modifications
    OmegaConf.set_struct(cfg, True)

    # Pass the unified config to the main setup function
    main_supervised_value_estimation(cfg)


if __name__ == "__main__":
    main()
