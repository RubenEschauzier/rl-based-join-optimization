import faulthandler

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.training_utils.training_tracking import TrainSummary
from src.utils.training_utils.utils import q_error_fn


class DualMetricScheduler:

    def __init__(self, optimizer, patience=3, threshold=1e-2, mode='min'):
        self.optimizer = optimizer
        self.val_tracker = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, patience=patience, threshold=threshold
        )
        self.train_tracker = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode, patience=patience, threshold=threshold
        )

    def step(self, train_loss, val_metric):
        # We manually check if BOTH have reached their plateau
        # Logic: Only reduce if the "patience" has run out for both trajectories

        # Step both trackers internally
        self.val_tracker.step(val_metric)
        self.train_tracker.step(train_loss)

        # Access the underlying counters
        # num_bad_epochs tracks how many epochs the metric hasn't improved
        if (self.val_tracker.num_bad_epochs > self.val_tracker.patience and
                self.train_tracker.num_bad_epochs > self.train_tracker.patience):

            # Manually trigger the LR reduction across the optimizer
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = old_lr * self.val_tracker.factor

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # Reset the internal trackers so they don't trigger again immediately
            self.val_tracker.num_bad_epochs = 0
            self.train_tracker.num_bad_epochs = 0

            return True
        return False

    def get_last_lr(self):
        """Returns the single current learning rate as a float."""
        return self.optimizer.param_groups[0]['lr']


def validate_model_dataset(model, val_dataset_loader, loss_fn, device, print_top_k=5):
    model.eval()
    losses, maes = [], []
    all_q_errors = []
    val_predictions = []

    for val_batch in val_dataset_loader:
        x = val_batch.x.to(device)
        edge_index = val_batch.edge_index.to(device)
        edge_attr = val_batch.edge_attr.to(device)
        batch = val_batch.batch.to(device)

        y_true = val_batch.y.to(device).view(-1)

        with torch.no_grad():
            pred = model.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            pred = pred[0]['output'].view(-1)

        # Use expm1 to correctly invert log1p
        pred_linear = torch.expm1(pred)

        loss = loss_fn(pred, torch.log1p(y_true))
        mae = torch.mean(torch.abs(pred_linear - y_true))

        q_errors_raw = q_error_fn(pred_linear, y_true)

        losses.append(loss.item())
        maes.append(mae.item())

        # Store the linear-scaled predictions, not the log-scaled ones
        preds_list = pred_linear.detach().cpu().numpy().tolist()
        actuals_list = y_true.detach().cpu().numpy().tolist()
        q_errors_list = q_errors_raw.view(-1).detach().cpu().numpy().tolist()

        all_q_errors.extend(q_errors_list)

        for i in range(len(preds_list)):
            val_predictions.append({
                "query": val_batch.query[i] if isinstance(val_batch.query, (list, tuple)) else val_batch.query,
                "prediction": preds_list[i],
                "actual": actuals_list[i],
                "q_error": q_errors_list[i],
                "type": val_batch.type[i] if hasattr(val_batch, 'type') else None
            })

    mean_loss = np.mean(losses)
    mean_mae = np.mean(maes)

    q_error_array = np.array(all_q_errors)
    percentiles = np.percentile(q_error_array, [50, 90, 95, 99])

    global_q_error_stats = {
        "val_mean_q_error": np.mean(q_error_array),
        "val_median_q_error": percentiles[0],
        "val_p90_q_error": percentiles[1],
        "val_p95_q_error": percentiles[2],
        "val_p99_q_error": percentiles[3]
    }

    val_predictions.sort(key=lambda x: x["q_error"], reverse=True)

    print(f"\n--- Top {print_top_k} Highest Q-Errors ---")
    for i in range(min(print_top_k, len(val_predictions))):
        entry = val_predictions[i]
        print(
            f"Rank {i + 1}: Q-Error = {entry['q_error']:.4f} | Pred = {entry['prediction']:.4f} | Actual = {entry['actual']:.4f}")
        print(f"Type: {entry['type']}")
        print(f"Query: {entry['query']}\n")

    return mean_loss, mean_mae, global_q_error_stats['val_mean_q_error'], val_predictions, global_q_error_stats


def run_pretraining_dataset(train_dataset, validation_dataset, writer, model_config_location,
                            device, n_epoch, batch_size, lr,):
    writer_tb = SummaryWriter(log_dir='runs/moe_experiment_1')
    lambda_aux = 0.01
    print("Training on {} queries, device: {}".format(len(train_dataset), device))

    model_factory_gine_conv= ModelFactory(model_config_location)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    gine_conv_model.to(device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    train_summary = TrainSummary([
        ("train_loss", "min"),
        ("val_loss", "min"),
        ("val_mae", "min"),
        ("val_q_error", "min"),
        ("val_mean_q_error", "min"),
        ("val_median_q_error", "min"),
        ("val_p90_q_error", "min"),
        ("val_p95_q_error", "min"),
        ("val_p99_q_error", "min")
    ])
    total_params = sum(p.numel() for p in gine_conv_model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(gine_conv_model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',
    #                               patience=3,
    #                               threshold=1e-2
    #                               )
    scheduler = DualMetricScheduler(optimizer,
                                    patience=3,
                                    threshold=1e-2
                                    )

    previous_lr = scheduler.get_last_lr()
    loss_fn = torch.nn.L1Loss(reduction="mean")
    # loss_fn = torch.nn.MSELoss(reduction="mean")

    global_step = 0

    gine_conv_model.register_moe_gradient_hooks(writer_tb, lambda: global_step)

    for i in range(n_epoch):
        gine_conv_model.train()

        train_losses = []
        # noinspection PyTypeChecker
        for batch in train_data_loader:
            optimizer.zero_grad()
            pred = gine_conv_model.forward(x=batch.x.to(device),
                                           edge_index=batch.edge_index.to(device),
                                           edge_attr=batch.edge_attr.to(device),
                                           batch = batch.batch.to(device))
            # Assume only one cardinality estimation head
            pred = pred[0]['output'].to(device)
            y = torch.log1p(batch.y.to(device))

            loss = loss_fn(pred.squeeze(), y)
            aux_load_balance_loss = gine_conv_model.get_load_balancing_loss(device=device,
                                                                            writer=writer_tb,
                                                                            epoch=i)
            if aux_load_balance_loss:
                loss += lambda_aux * aux_load_balance_loss
            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())
            global_step+=1

        # noinspection PyTypeChecker
        val_loss, val_mae, val_q_error, val_predictions, stats_q_error = (
            validate_model_dataset(gine_conv_model, val_data_loader, loss_fn, device=device))
        print('Epoch {}/{}, average train loss: {}, val_loss: {}, val_mae: {}, val_q_error: {}'.
            format(i+1, n_epoch, np.mean(train_losses), val_loss, val_mae, val_q_error))
        print(stats_q_error)

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(np.mean(train_losses), val_loss)
        val_summary_epoch =  {
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "val_q_error": float(val_q_error),
        }
        val_summary_epoch.update(stats_q_error)

        train_summary.update(val_summary_epoch, i)
        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file(val_predictions, best, per_epoch, gine_conv_model, i)


def main_pretraining_dataset(queries_location_train, queries_location_val,
                             endpoint_location, rdf2vec_vector_location, writer,
                             feature_type, model_config_location, n_epoch, batch_size, lr, seed,
                             occurrences_location = None, tp_cardinality_location = None, multiplicity_location=None,
                             hll_location = None,
                             test_queries=None, test_cardinalities=None,
                             to_load=None, device='cpu'
                             ):
    faulthandler.enable()
    writer.create_experiment_directory()
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location_train, queries_location_val,
                                                           endpoint_location,
                                                           rdf2vec_vector_location, query_env, feature_type,
                                                           to_load=to_load,
                                                           load_mappings=False,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           multiplicity_location=multiplicity_location,
                                                           hll_location=hll_location,
                                                           )
    run_pretraining_dataset(train_dataset, val_dataset, writer, model_config_location, device, n_epoch, batch_size, lr)
    return train_dataset, val_dataset

