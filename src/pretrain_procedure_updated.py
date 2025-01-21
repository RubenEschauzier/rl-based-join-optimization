import functools
import math

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.datastructures.query_pytorch_dataset import QueryCardinalityDataset
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_edge_labeled_graph import QueryToEdgeLabeledGraph
from src.query_featurizers.featurize_predicate_edges import QueryToEdgePredicateGraph
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.utils.training_utils.utils import q_error_fn


#TODO: Load test data into a dataset too
def load_queries_into_dataset(queries_location, endpoint_location, rdf2vec_vector_location, validation_size=.2,
                              to_load=None):
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    query_env = BlazeGraphQueryEnvironment(endpoint_location)

    query_to_graph = QueryToEdgePredicateGraph(vectors, query_env)
    featurizer_edge_labeled_graph = functools.partial(query_to_graph.transform_undirected)
    dataset = QueryCardinalityDataset(root=queries_location,
                                      featurizer=featurizer_edge_labeled_graph,
                                      to_load=to_load
                                      )
    dataset = dataset.shuffle()
    train_dataset = dataset[math.floor(len(dataset)*validation_size):]
    validation_dataset = dataset[:math.floor(len(dataset)*validation_size)]

    return train_dataset, validation_dataset

def validate_model_dataset(model, val_dataset_loader, loss_fn):
    model.eval()
    losses, maes, q_errors = [], [], []
    for val_batch in val_dataset_loader:
        with torch.no_grad():
            pred = model.forward(x=val_batch.x,
                                 edge_index=val_batch.edge_index,
                                 edge_attr=val_batch.edge_attr,
                                 batch=val_batch.batch)

        loss = loss_fn(pred.squeeze(), torch.log(val_batch.y))
        mae = torch.mean(torch.abs(torch.exp(pred.squeeze()) - val_batch.y))
        q_error = torch.mean(q_error_fn(torch.exp(pred.squeeze()), val_batch.y))

        losses.append(loss.item())
        maes.append(mae.item())
        q_errors.append(q_error)
    return np.mean(losses), np.mean(maes), np.mean(q_errors)

def run_pretraining_dataset(train_dataset, validation_dataset, device, n_epoch, batch_size, lr, seed,
                    ckp_dir=None, test_queries=None, test_cardinalities=None):
    print("Training on {} queries".format(len(train_dataset)))
    model_factory_gine_conv= ModelFactory("experiments/model_configs/triple_gine_conv_model.yaml")
    gine_conv_model = model_factory_gine_conv.load_gine_conv()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    total_params = sum(p.numel() for p in gine_conv_model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(gine_conv_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=3,
                                  threshold=1e-2
                                  )
    previous_lr = scheduler.get_last_lr()
    loss_fn = torch.nn.L1Loss(reduction="mean")

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
            y = torch.log(batch.y)

            loss = loss_fn(pred.squeeze(), y)
            loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

        # noinspection PyTypeChecker
        val_loss, val_mae, val_q_error = validate_model_dataset(gine_conv_model, val_data_loader, loss_fn)
        print('Epoch {}/{}, average train loss: {}, val_loss: {}, val_mae: {}, val_q_error: {}'.
            format(i+1, n_epoch, np.mean(train_losses), val_loss, val_mae, val_q_error))

        if scheduler.get_last_lr() != previous_lr:
            print("INFO: Lr Updated from {} to {}".format(previous_lr, scheduler.get_last_lr()))
            previous_lr = scheduler.get_last_lr()

        scheduler.step(val_loss)


def main_pretraining_dataset(queries_location, endpoint_location, rdf2vec_vector_location,
                             n_epoch, batch_size, lr, seed,
                             ckp_dir=None, test_queries=None, test_cardinalities=None,
                             validation_size=.2, to_load=None
                             ):
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location,
                                                           validation_size=validation_size, to_load=to_load)
    run_pretraining_dataset(train_dataset, val_dataset, 'cpu', n_epoch, batch_size, lr, seed,
                            ckp_dir=ckp_dir, test_queries=test_queries, test_cardinalities=test_cardinalities)