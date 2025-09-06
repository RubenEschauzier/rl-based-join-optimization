import faulthandler
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.utils.training_utils.training_tracking import TrainSummary
from src.utils.training_utils.utils import q_error_fn, mixed_mae_q_error_loss


def validate_model_dataset(model, val_dataset_loader, loss_fn):
    model.eval()
    losses, maes, q_errors = [], [], []
    val_predictions = []
    for val_batch in val_dataset_loader:
        with torch.no_grad():
            # pred = model.forward(x=val_batch.x.double(),
            #                      edge_index=val_batch.edge_index,
            #                      edge_attr=val_batch.edge_attr.double(),
            #                      batch=val_batch.batch)
            pred = model.forward(x=val_batch.x,
                                 edge_index=val_batch.edge_index,
                                 edge_attr=val_batch.edge_attr,
                                 batch=val_batch.batch)
            pred = pred[0]['output']

        loss = loss_fn(pred.squeeze(), torch.log(val_batch.y).squeeze())
        mae = torch.mean(torch.abs(torch.exp(pred.squeeze()) - val_batch.y.squeeze()))
        q_error = torch.mean(q_error_fn(torch.exp(pred.squeeze()), val_batch.y.squeeze()))

        losses.append(loss.item())
        maes.append(mae.item())
        q_errors.append(q_error)
        val_predictions.append( {
            "query": val_batch.query,
            "prediction": float(pred.item()),
            "actual": float(np.log(val_batch.y.item())),
            "loss": float(loss.item()),
            "mae": float(mae.item()),
            "q_error": float(q_error.item()),
            "type": val_batch.type
        })
    return np.mean(losses), np.mean(maes), np.mean(q_errors), val_predictions

def run_pretraining_dataset(train_dataset, validation_dataset, writer, model_config_location, device, n_epoch, batch_size, lr,
                            seed, test_queries=None, test_cardinalities=None):
    print("Training on {} queries".format(len(train_dataset)))
    n_annealing_steps_alpha = 20

    model_factory_gine_conv= ModelFactory(model_config_location)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    # gine_conv_model.double()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    train_summary = TrainSummary([("train_loss", "min"), ("val_loss", "min"),
                                  ("val_mae", "min"), ("val_q_error", "min")])
    total_params = sum(p.numel() for p in gine_conv_model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(gine_conv_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2
                                  )
    previous_lr = scheduler.get_last_lr()
    loss_fn = torch.nn.L1Loss(reduction="mean")
    # loss_fn = torch.nn.MSELoss(reduction="mean")


    for i in range(n_epoch):
        gine_conv_model.train()

        train_losses = []
        # noinspection PyTypeChecker
        for batch in train_data_loader:
            optimizer.zero_grad()
            pred = gine_conv_model.forward(x=batch.x,
                                           edge_index=batch.edge_index,
                                           edge_attr=batch.edge_attr,
                                           batch = batch.batch)

            # pred = gine_conv_model.forward(x=batch.x.double(),
            #                                edge_index=batch.edge_index,
            #                                edge_attr=batch.edge_attr.double(),
            #                                batch = batch.batch)
            # Assume only one cardinality estimation head
            pred = pred[0]['output']
            y = torch.log(batch.y + 1)

            loss = loss_fn(pred.squeeze(), y)
            loss.backward()

            optimizer.step()
            # if math.isinf(loss.item()):
            #     print(pred.squeeze())
            #     print(y)
            train_losses.append(loss.item())

        # noinspection PyTypeChecker
        val_loss, val_mae, val_q_error, val_predictions = validate_model_dataset(gine_conv_model, val_data_loader, loss_fn)
        print('Epoch {}/{}, average train loss: {}, val_loss: {}, val_mae: {}, val_q_error: {}'.
            format(i+1, n_epoch, np.mean(train_losses), val_loss, val_mae, val_q_error))

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(val_loss)

        train_summary.update({
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "val_q_error": float(val_q_error),
        }, i)
        best, per_epoch = train_summary.summary()
        writer.write_epoch_to_file(val_predictions, best, per_epoch, gine_conv_model, i)


def main_pretraining_dataset(queries_location_train, queries_location_val,
                             endpoint_location, rdf2vec_vector_location, writer,
                             feature_type, model_config_location, n_epoch, batch_size, lr, seed,
                             occurrences_location = None, tp_cardinality_location = None,
                             test_queries=None, test_cardinalities=None,
                             to_load=None
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
                                                           tp_cardinality_location=tp_cardinality_location)
    run_pretraining_dataset(train_dataset, val_dataset, writer, model_config_location,'cpu', n_epoch, batch_size, lr,
                            seed, test_queries=test_queries, test_cardinalities=test_cardinalities)
    return train_dataset, val_dataset

