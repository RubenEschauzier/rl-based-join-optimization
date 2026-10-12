import faulthandler
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from src.datastructures.subquery_training import (CardinalityExamples, EpochSampler, SUBQUERY_FIELDS,
                                                  censored_l1, subquery_metrics)
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.training_utils.training_tracking import TrainSummary
from src.utils.training_utils.utils import q_error_fn

# A model with an "emptiness" head is a hurdle model: that head gives the logit of P(result is
# empty), the cardinality head (log1p rows) is trained on non-empty queries only, and the
# prediction is 0 rows when P(empty) exceeds this threshold.
EMPTINESS_THRESHOLD = 0.5


def _head(outputs, head_type):
    return next((h["output"] for h in outputs if h["output_type"] == head_type), None)


def cardinality_and_emptiness(outputs):
    """The cardinality head's log1p prediction and the emptiness logit (None without that head)."""
    cardinality = _head(outputs, "cardinality")
    if cardinality is None:                       # single-head models: the first head
        cardinality = outputs[0]["output"]
    emptiness = _head(outputs, "emptiness")
    return cardinality.reshape(-1), None if emptiness is None else emptiness.reshape(-1)


def pretraining_loss(loss_fn, cardinality, emptiness, y):
    target = torch.log1p(y)
    if emptiness is None:
        return loss_fn(cardinality, target)
    empty = y == 0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(emptiness, empty.float())
    if (~empty).any():
        loss = loss + loss_fn(cardinality[~empty], target[~empty])
    return loss


def combined_log_prediction(cardinality, emptiness):
    """log1p rows predicted by the model: the cardinality head, or 0 where the emptiness head
    says the result is empty."""
    if emptiness is None:
        return cardinality
    return torch.where(torch.sigmoid(emptiness) > EMPTINESS_THRESHOLD, torch.zeros_like(cardinality), cardinality)


def emptiness_statistics(p_empty, actual):
    from sklearn.metrics import roc_auc_score
    p_empty, empty = np.asarray(p_empty), np.asarray(actual) == 0
    said_empty = p_empty > EMPTINESS_THRESHOLD
    return {
        "val_empty_accuracy": float(np.mean(said_empty == empty)),
        "val_empty_precision": float(np.sum(said_empty & empty) / max(np.sum(said_empty), 1)),
        "val_empty_recall": float(np.sum(said_empty & empty) / max(np.sum(empty), 1)),
        "val_empty_auroc": float(roc_auc_score(empty, p_empty)) if 0 < empty.sum() < len(empty) else float("nan"),
    }

EMPTINESS_FIELDS = [("val_empty_accuracy", "max"), ("val_empty_precision", "max"),
                    ("val_empty_recall", "max"), ("val_empty_auroc", "max")]


class DualMetricScheduler:

    def __init__(self, optimizer, patience=3, threshold=1e-2, mode='min', factor=0.1,
                 threshold_mode='rel', min_lr=1e-6):
        """
        :param threshold: Minimum improvement that counts as progress.
        :param threshold_mode: 'rel' interprets `threshold` as a fraction of the current
            best value, 'abs' as a raw offset. Prefer 'rel': an absolute threshold that is
            large relative to the loss makes every epoch look like a plateau and decays the
            learning rate into oblivion (e.g. threshold=1e-2 on a loss of ~0.07).
        :param min_lr: Floor on the learning rate, so a run cannot silently become a no-op.
        """
        if threshold_mode not in ('rel', 'abs'):
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode!r}")
        self.optimizer = optimizer
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.mode = mode
        self.factor = factor
        self.min_lr = min_lr
        initial_best = float('inf') if mode == 'min' else float('-inf')
        self.best_train = initial_best
        self.best_val = initial_best
        self.bad_train_epochs = 0
        self.bad_val_epochs = 0

    def _margin(self, best):
        """Improvement required over `best` to count as progress."""
        if self.threshold_mode == 'abs' or not math.isfinite(best):
            return self.threshold
        # abs() keeps the margin a genuine tightening for negative metrics too
        # (e.g. joint NLL), where scaling by (1 - threshold) would flip its sign.
        return self.threshold * abs(best)

    def _is_better(self, value, best):
        if not math.isfinite(best):
            return True
        margin = self._margin(best)
        if self.mode == 'min':
            return value < best - margin
        return value > best + margin

    def _update_tracker(self, value, best, bad_epochs):
        if self._is_better(value, best):
            return value, 0
        return best, bad_epochs + 1

    def step(self, train_loss, val_metric):
        self.best_train, self.bad_train_epochs = self._update_tracker(
            train_loss, self.best_train, self.bad_train_epochs
        )
        self.best_val, self.bad_val_epochs = self._update_tracker(
            val_metric, self.best_val, self.bad_val_epochs
        )

        if self.bad_train_epochs > self.patience and self.bad_val_epochs > self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.factor, self.min_lr)

            self.bad_train_epochs = 0
            self.bad_val_epochs = 0

            if new_lr == current_lr:
                return False

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
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
    all_p_empty, all_actual = [], []

    for val_batch in val_dataset_loader:
        x = val_batch.x.to(device)
        edge_index = val_batch.edge_index.to(device)
        edge_attr = val_batch.edge_attr.to(device)
        batch = val_batch.batch.to(device)

        y_true = val_batch.y.to(device).view(-1)

        with torch.no_grad():
            outputs = model.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            cardinality, emptiness = cardinality_and_emptiness(outputs)
            pred = combined_log_prediction(cardinality, emptiness)
            p_empty = None if emptiness is None else torch.sigmoid(emptiness).cpu().numpy().tolist()

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
            if p_empty is not None:
                val_predictions[-1]["p_empty"] = p_empty[i]
        if p_empty is not None:
            all_p_empty.extend(p_empty)
            all_actual.extend(actuals_list)

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
    if all_p_empty:
        global_q_error_stats.update(emptiness_statistics(all_p_empty, all_actual))

    val_predictions.sort(key=lambda x: x["q_error"], reverse=True)

    print(f"\n--- Top {print_top_k} Highest Q-Errors ---")
    for i in range(min(print_top_k, len(val_predictions))):
        entry = val_predictions[i]
        print(
            f"Rank {i + 1}: Q-Error = {entry['q_error']:.4f} | Pred = {entry['prediction']:.4f} | Actual = {entry['actual']:.4f}")
        print(f"Type: {entry['type']}")
        print(f"Query: {entry['query']}\n")

    return mean_loss, mean_mae, global_q_error_stats['val_mean_q_error'], val_predictions, global_q_error_stats


def _predict(model, batch, device):
    outputs = model.forward(x=batch.x.to(device), edge_index=batch.edge_index.to(device),
                            edge_attr=batch.edge_attr.to(device), batch=batch.batch.to(device))
    cardinality, emptiness = cardinality_and_emptiness(outputs)
    return combined_log_prediction(cardinality, emptiness)


def run_pretraining_dataset(train_dataset, validation_dataset, writer, model_config_location,
                            device, n_epoch, batch_size, lr, weight_decay=0.0,
                            subquery_labels_train=None, subquery_labels_val=None, censored_loss=False,
                            subquery_fraction=1.0, subquery_val_examples=20000, num_workers=2, resume=False):
    """With subquery_labels_train (src/datastructures/subquery_training.py): train on the full
    queries plus their counted sub-queries (a fresh `subquery_fraction` of them per epoch), with
    the censored loss when censored_loss; validation adds sub-query metrics (val_sub_*) on a fixed
    sample of `subquery_val_examples` validation sub-queries.
    With resume, `writer` points at an existing run: training continues after its last saved epoch
    from that epoch's weights (a fresh optimizer), and its recorded epochs are replayed into the
    summary so train_progress.json stays one continuous history."""
    writer_tb = SummaryWriter(log_dir='runs/moe_experiment_1')
    lambda_aux = 0.01
    print("Training on {} queries, device: {}".format(len(train_dataset), device))

    model_factory_gine_conv= ModelFactory(model_config_location)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    gine_conv_model.to(device)

    start_epoch = 0
    previous = None
    if resume:
        import json as _json
        import os as _os
        with open(_os.path.join(str(writer.experiment_directory), "train_progress.json")) as f:
            previous = _json.load(f)
        start_epoch = len(previous["val_loss"])
        last = _os.path.join(str(writer.experiment_directory), f"epoch-{start_epoch - 1}", "model")
        gine_conv_model.embedding_model.load_state_dict(
            torch.load(_os.path.join(last, "embedding_model.pt"), map_location=device))
        for index, head in enumerate(gine_conv_model.heads):
            head_type = gine_conv_model.head_types[index]
            head.load_state_dict(torch.load(_os.path.join(last, f"head_{head_type}.pt"), map_location=device))
        print(f"Resuming {writer.experiment_directory} at epoch {start_epoch} from {last}")
    use_examples = bool(subquery_labels_train) or censored_loss
    sub_val_loader = None
    if use_examples:
        if "emptiness" in gine_conv_model.head_types and censored_loss:
            raise ValueError("censored_loss replaces the emptiness head; use a model without it")
        examples = CardinalityExamples(train_dataset, subquery_labels_train, censored=censored_loss)
        print(f"Training examples: {examples.n_full:,} full queries + {examples.n_sub:,} sub-queries "
              f"({subquery_fraction:.0%} of them per epoch), censored loss: {censored_loss}")
        train_data_loader = DataLoader(examples, batch_size=batch_size,
                                       sampler=EpochSampler(examples, subquery_fraction, seed=start_epoch),
                                       num_workers=num_workers, persistent_workers=num_workers > 0)
        if subquery_labels_val:
            val_examples = CardinalityExamples(validation_dataset, subquery_labels_val, include_full=False,
                                               censored=True)
            keep = np.random.default_rng(0).permutation(len(val_examples))[:subquery_val_examples]
            sub_val_loader = DataLoader(torch.utils.data.Subset(val_examples, keep.tolist()), batch_size=512,
                                        num_workers=0)       # no extra forks next to the training workers
    else:
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
    ] + (EMPTINESS_FIELDS if "emptiness" in gine_conv_model.head_types else [])
      + (SUBQUERY_FIELDS if sub_val_loader is not None else []))
    if previous is not None:                       # replay the recorded epochs into the summary
        for epoch in range(start_epoch):
            train_summary.update({k: v[epoch] for k, v in previous.items()}, epoch)
    total_params = sum(p.numel() for p in gine_conv_model.parameters())
    print(f"Total number of parameters: {total_params}")

    # weight_decay > 0: AdamW (decoupled weight decay); otherwise plain Adam as before
    optimizer = (torch.optim.AdamW(gine_conv_model.parameters(), lr=lr, weight_decay=weight_decay)
                 if weight_decay else torch.optim.Adam(gine_conv_model.parameters(), lr=lr))
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

    for i in range(start_epoch, n_epoch):
        gine_conv_model.train()

        train_losses = []
        # noinspection PyTypeChecker
        for batch in train_data_loader:
            optimizer.zero_grad()
            pred = gine_conv_model.forward(x=batch.x.to(device),
                                           edge_index=batch.edge_index.to(device),
                                           edge_attr=batch.edge_attr.to(device),
                                           batch = batch.batch.to(device))
            cardinality, emptiness = cardinality_and_emptiness(pred)
            if censored_loss:
                loss = censored_l1(cardinality, batch.y.to(device).view(-1), batch.censor.to(device).view(-1))
            else:
                loss = pretraining_loss(loss_fn, cardinality, emptiness, batch.y.to(device).view(-1))
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
        if sub_val_loader is not None:
            stats_q_error.update(subquery_metrics(gine_conv_model, sub_val_loader, device,
                                                  lambda model, batch: _predict(model, batch, device)))
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
                             hll_location = None, minhash_location=None, contextualizer_location=None,
                             weight_decay=0.0, subquery_labels_train=None, subquery_labels_val=None,
                             censored_loss=False, subquery_fraction=1.0, resume_from=None,
                             test_queries=None, test_cardinalities=None,
                             to_load=None, device='cpu'
                             ):
    faulthandler.enable()
    if resume_from:                                # continue an existing run in its own directory
        writer.experiment_directory = resume_from
    else:
        writer.create_experiment_directory()
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location_train, queries_location_val,
                                                           endpoint_location,
                                                           rdf2vec_vector_location, query_env, feature_type,
                                                           to_load=to_load,
                                                           load_mappings=bool(subquery_labels_train),
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           multiplicity_location=multiplicity_location,
                                                           hll_location=hll_location,
                                                           minhash_location=minhash_location,
                                                           contextualizer_location=contextualizer_location,
                                                           )
    run_pretraining_dataset(train_dataset, val_dataset, writer, model_config_location, device, n_epoch, batch_size, lr,
                            weight_decay=weight_decay, subquery_labels_train=subquery_labels_train,
                            subquery_labels_val=subquery_labels_val, censored_loss=censored_loss,
                            subquery_fraction=subquery_fraction, resume=bool(resume_from))
    return train_dataset, val_dataset

