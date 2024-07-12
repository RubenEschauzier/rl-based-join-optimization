import math
import cPickle as pickle

import numpy as np
from sklearn.utils import shuffle

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datastructures.query import Query
from src.models.model_instantiator import ModelFactory
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_graph_featurizers.quad_views import FeaturizeQueryGraphQuadViews
from src.utils.training_utils.utils import initialize_graph_models, get_parameters_model, embed_query_graphs


def load_data_txt_file(location):
    data = []
    num_processed = 0
    with open(location, 'r') as f:
        for line in f.readlines():
            # Skip white spaces
            if len(line.strip()) > 0:
                data.append(line.strip())
                num_processed += 1
    return data


def prepare_pretraining_queries(env, queries, rdf2vec_vector_location, save_prepared_queries=False, save_location=None):
    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    rdf2vec_featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    view_creator = FeaturizeQueryGraphQuadViews()

    queries = rdf2vec_featurizer.run(queries)
    queries = view_creator.run(queries, "edge_index")

    # When preparing many queries this can be used to reuse the computations
    if save_prepared_queries:
        with open(save_location, 'wb') as f:
            pickle.dump(queries, f)
    return queries


def compute_q_error(predictions, true_cardinalities):
    total_q_error = 0
    for pred, actual in zip(predictions.cpu(), true_cardinalities):
        pred = pred.detach()
        ratios = np.array([actual/ pred, pred / actual])
        total_q_error += ratios[np.isfinite(ratios)].max()
    return total_q_error


def validate_model(graph_embedding_models, cardinality_estimation_head,
                   val_queries, val_cardinalities, batch_size,
                   loss_fn, device):
    total_processed = 0
    total_q_error = 0
    total_loss = 0
    # TODO: PROPER BATCHED ITERATION WE'RE MISSING OUT ON ELEMENTS
    for b in range(0, len(val_queries), batch_size):
        queries_batch = val_queries[b:b + batch_size]
        cardinalities_batch = val_cardinalities[b:b + batch_size]

        embedded_features = embed_query_graphs(queries_batch, graph_embedding_models)
        embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

        pred = cardinality_estimation_head.forward(embedded_features_batched)

        total_processed += len(queries_batch)

        q_error = compute_q_error(pred, cardinalities_batch)
        total_q_error += q_error

        loss = loss_fn(pred.squeeze(), torch.tensor(cardinalities_batch, device=device))
        total_loss += loss.item()

    return total_loss / total_processed, total_q_error / total_processed


def run_pretraining(queries_location, cardinalities_location, prepared_queries_location, rdf2vec_vector_location,
                    endpoint_uri, n_epoch, batch_size, lr, seed, save_prepared_queries=False):
    # Initialize the query environment
    env = BlazeGraphQueryEnvironment(endpoint_uri)

    raw_queries = load_data_txt_file(queries_location)

    queries = []
    for query in tqdm(raw_queries):
        queries.append(Query(query))

    cardinalities = [math.log(int(card)+1) for card in load_data_txt_file(cardinalities_location)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build graph embedding models from config
    model_factory_graph_emb = ModelFactory("experiments/configs/graph_embedding_model.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory_graph_emb, 4)])

    # Build cardinality estimator from config
    model_factory_card_est_head = ModelFactory("experiments/configs/cardinality_estimation_head.yaml")
    cardinality_estimation_head = model_factory_card_est_head.load_mlp()

    # Move models to cuda
    [graph_model.to(device) for graph_model in graph_embedding_models]
    cardinality_estimation_head.to(device)

    # Get parameters and set optimizer
    parameters = get_parameters_model(graph_embedding_models)
    parameters.extend(cardinality_estimation_head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # Prepare features of queries
    queries = prepare_pretraining_queries(env, queries, rdf2vec_vector_location,
                                          save_prepared_queries=save_prepared_queries,
                                          save_location=prepared_queries_location
                                          )

    # Move query features to device
    for query in queries:
        query.features = query.features.to(device)
        query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]

    train_queries, val_queries, train_card, val_card = train_test_split(queries, cardinalities,
                                                                        test_size=.2, random_state=seed)

    # Use L1 loss
    loss_fn = torch.nn.L1Loss()

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    val_q_error_per_epoch = []
    for i in range(n_epoch):
        n_batches_processed = 0
        loss_epoch = 0
        train_queries, train_card = shuffle(train_queries, train_card, random_state=seed)

        for b in range(0, len(train_queries), batch_size):

            queries_batch = train_queries[b:b + batch_size]
            cardinalities_batch = train_card[b:b + batch_size]

            optimizer.zero_grad()

            embedded_features = embed_query_graphs(queries_batch, graph_embedding_models)
            embedded_features_batched = torch.stack([feature.sum(dim=0) for feature in embedded_features])

            pred = cardinality_estimation_head.forward(embedded_features_batched)
            loss = loss_fn(pred.squeeze(), torch.tensor(cardinalities_batch, device=device))

            loss_epoch += loss.item()
            n_batches_processed += 1

            loss.backward()

            optimizer.step()

        val_loss, val_q_error = validate_model(
            graph_embedding_models=graph_embedding_models,
            cardinality_estimation_head=cardinality_estimation_head,
            val_queries=val_queries,
            val_cardinalities=val_card,
            batch_size=batch_size,
            loss_fn=loss_fn,
            device=device
        )
        train_loss_per_epoch.append(loss_epoch/n_batches_processed)
        val_loss_per_epoch.append(val_loss)
        val_q_error_per_epoch.append(val_q_error)

        print("Train loss: {}, Validation loss: {}, q-error: {}".format(loss_epoch/n_batches_processed,
                                                                        val_loss,
                                                                        val_q_error))

    #
    # # Move query features to device
    # for query in queries:
    #     query.features = query.features.to(device)
    #     query.query_graph_representations = [graph.to(device) for graph in query.query_graph_representations]
    #
    # # Split queries
    # train_queries, test_queries = train_test_split(queries, test_size=.2, random_state=seed)
    #
    # # Get average execution time to compare against
    # average_exec_default_opt, total_average = get_average_execution_time_queries(env, train_queries, 60, 20)
    # print(average_exec_default_opt)
    # print(total_average)
    #
    # # Train loop
    # for i in range(n_epoch):
    #     print("Epoch {}/{} ".format(i + 1, n_epoch))
    #     rewards_epoch = []
    #     execution_times_epoch = []
    #     train_queries = shuffle(train_queries, random_state=seed)
    #     for b in range(0, len(train_queries), batch_size):
    #         log_probs, rewards, actions = [], [], []
    #         queries_batch = train_queries[b:b + batch_size]
    #         for n in range(n_episodes_query):
    #             # Embed the query graphs + features using graph convolution
    #             features, un_padded_features, sequence_lengths = preprocess(queries_batch, graph_embedding_models)
    #
    #             # Get the pointer network-based agent's output
    #             join_orders, log_probs_batch = policy(ptr_net, features, sequence_lengths)
    #
    #             # For each join order
    #             for k, join_order in enumerate(join_orders):
    #                 # Execute the query and record join ratio + execution time
    #                 penalty, exec_time = env_step(env,
    #                                               queries_batch[k],
    #                                               join_order[:un_padded_features[k].shape[0]],
    #                                               60)
    #                 # If the query timed out this function returns an integer, so we turn it into penalty sequence
    #                 if isinstance(penalty, int):
    #                     # Set penalty to the high number returned in case of a timeout
    #                     penalty = [penalty] * len(join_order)
    #
    #                 # Sometimes blazegraph fails, we record these fails to ensure the failing query is not systemic
    #                 if isinstance(penalty, str) and penalty == "FAIL":
    #                     # TODO Record fails and their queries to check if it is failing systemic
    #                     continue
    #
    #                 # For join ratio, lower is better, thus we take the negative of the join ratio to get reward
    #                 reward = -np.array(penalty)
    #
    #                 # For each decision in join order creation we record discounted rewards and the log probability
    #                 # of selecting that triple pattern for calculating the policy gradient using REINFORCE
    #                 for j in range(reward.shape[0]):
    #                     # Query k, at join order timestep j, chose pointer join_order[j]
    #                     log_prob = log_probs_batch[k][j][join_order[j]].to(device)
    #
    #                     # Discounted reward do NOT take into account previous rewards
    #                     discounted_reward = sum([r * discount_factor ** i for i, r in enumerate(reward[j:])])
    #
    #                     log_probs.append(log_prob)
    #                     rewards.append(discounted_reward)
    #                     rewards_epoch.append(discounted_reward)
    #
    #                 # Record execution time for statistics tracking
    #                 execution_times_epoch.append(exec_time)
    #
    #         # Prepare data policy gradient
    #         log_prob_tensor = torch.stack(log_probs).to(device)
    #         reward_tensor = torch.from_numpy(np.array(rewards)).to(device)
    #
    #         # Policy gradient term
    #         performance = (-(log_prob_tensor * reward_tensor)).sum()
    #
    #         # Backprop on policy gradient
    #         optimizer.zero_grad()
    #         performance.backward()
    #         optimizer.step()
    #     print("Mean reward: {}, Mean exec time: {}".format(sum(rewards_epoch) / len(rewards_epoch),
    #                                                        sum(execution_times_epoch) /
    #                                                        len(execution_times_epoch)
    #                                                        ))
