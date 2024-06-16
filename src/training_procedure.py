from os import listdir
from os.path import isfile, join
from time import sleep

from SPARQLWrapper import JSON
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import numpy as np

from src.datastructures.query import Query
from src.models.model_instantiator import ModelFactory
from src.models.pointer_network import PointerNet
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_featurizers.featurize_rdf2vec import FeaturizeQueriesRdf2Vec
from src.query_graph_featurizers.quad_views import FeaturizeQueryGraphQuadViews


def load_queries(location):
    raw_queries = []
    files = [f for f in listdir(location) if isfile(join(location, f))]
    for file in files:
        with open(join(location, file), 'r') as f:
            raw_queries.extend(f.read().strip().split('\n\n'))

    queries = [Query(query) for query in raw_queries]
    return queries


def get_parameters_model(all_models):
    parameters = []
    for model in all_models:
        parameters.extend(list(model.parameters()))
    return parameters


# Not sure how to batch this yet, might not need it due to small model size
def embed_query_graphs(queries, embedding_models):
    query_graph_embeddings = []
    for query in queries:
        query_emb = run_models(query, embedding_models)
        query_graph_embeddings.append(query_emb)
    return query_graph_embeddings


def run_models(query, embedding_models):
    embeddings_list = []
    for graph, model in zip(query.query_graph_representations, embedding_models):
        embedding = model.run(query.features, graph)
        embeddings_list.append(embedding)

    return torch.cat(embeddings_list, dim=1)


def prepare(env, queries_location, rdf2vec_vector_location):
    queries = load_queries(queries_location)[0::20]

    vectors = FeaturizeQueriesRdf2Vec.load_vectors(rdf2vec_vector_location)
    rdf2vec_featurizer = FeaturizeQueriesRdf2Vec(env, vectors)
    view_creator = FeaturizeQueryGraphQuadViews()

    queries = rdf2vec_featurizer.run(queries)
    queries = view_creator.run(queries, "edge_index")
    return queries


def initialize_graph_models(factories: [tuple]):
    """
    Initializes graph models in order specified in factories. This order should align with the order of generated graphs
    to run over
    :param factories: list of tuples of model factories and number of model should be created from this factory
    :return: list with created models
    """
    models = []
    for (factory, n_instances) in factories:
        for i in range(n_instances):
            models.append(factory.build_model_from_config())
    return models


def policy(network, features, sequence_lengths):
    log_probs_batch, sampled_pointer, mask = network.forward(features, sequence_lengths, greedy=False)
    join_orders = np.array(sampled_pointer)
    return join_orders, log_probs_batch


def env_step(env, batch_queries, join_order):
    env_result, exec_time = env.run(batch_queries, join_order, 60, JSON, {"explain": "True"})
    penalty = env.reward(env_result, "intermediate-results")
    return penalty, exec_time


def preprocess(queries, graph_embedding_models):
    embedded_features = embed_query_graphs(queries, graph_embedding_models)
    sequence_lengths = torch.Tensor([seq.shape[0] for seq in embedded_features])
    padded_features = nn.utils.rnn.pad_sequence(embedded_features, batch_first=True)
    return padded_features, embedded_features, sequence_lengths


def get_future_rewards(reward_sequence, discounted_rewards, k):
    return reward_sequence + np.sum(discounted_rewards[k:])


def run_training(endpoint, queries_location, rdf2vec_vector_location,
                 n_epoch, batch_size, lr, n_episodes_query, discount_factor, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on {}".format(device))
    torch.set_default_device(device)

    # Build graph embedding models from config
    model_factory = ModelFactory("experiments/configs/test_config.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory, 4)])

    # Pointer network for join order sequence generation
    ptr_net = PointerNet(8192, 8192, 8192)

    # Define optimizer over all model parameters
    all_models = [ptr_net] + graph_embedding_models
    parameters = get_parameters_model(all_models)
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # Initialize the query environment
    env = BlazeGraphQueryEnvironment(endpoint)

    # Prepare features of queries
    queries = prepare(env, queries_location, rdf2vec_vector_location)

    # Split queries
    train_queries, test_queries = train_test_split(queries, test_size=.2, random_state=seed)

    # Train loop
    for i in range(n_epoch):
        rewards_epoch = []
        execution_times_epoch = []
        train_queries = shuffle(train_queries, random_state=seed)
        for b in range(0, len(train_queries), batch_size):
            log_probs, rewards, actions = [], [], []
            queries_batch = train_queries[b:b + batch_size]
            for n in range(n_episodes_query):
                # Embed the query graphs + features using graph convolution
                features, un_padded_features, sequence_lengths = preprocess(queries_batch, graph_embedding_models)

                # Get the pointer network-based agent's output
                join_orders, log_probs_batch = policy(ptr_net, features, sequence_lengths)

                # For each join order
                for k, join_order in enumerate(join_orders):
                    # Execute the query and record join ratio + execution time
                    penalty, exec_time = env_step(env, queries_batch[k], join_order[:un_padded_features[k].shape[0]])

                    # If the query timed out this function returns an integer, so we turn it into penalty sequence
                    if isinstance(penalty, int):
                        # TODO: Deal with case where timeout occurs (make array of discounted rewards with large penalty
                        continue
                    # Sometimes blazegraph fails, we record these fails to ensure the failing query is not systemic
                    if isinstance(penalty, str) and penalty == "FAIL":
                        # TODO Record fails and their queries to check if it is failing systemic
                        continue
                    # For join ratio, lower is better, thus we take the negative of the join ratio to get reward
                    reward = -np.array(penalty)

                    # For each decision in join order creation we record discounted rewards and the log probability
                    # of selecting that triple pattern for calculating the policy gradient using REINFORCE
                    for j in range(reward.shape[0]):
                        # Query k, at timestep in optimization j, chose pointer join_order[j]
                        log_prob = log_probs_batch[k][j][join_order[j]]

                        # Discounted reward do NOT take into account previous rewards
                        discounted_reward = sum([r * discount_factor ** i for i, r in enumerate(reward[j:])])

                        log_probs.append(log_prob)
                        rewards.append(discounted_reward)
                        rewards_epoch.append(discounted_reward)
                        # TODO CHECK IF LOG PROB IS CORRECT

                    # Record execution time for statistics tracking
                    execution_times_epoch.append(exec_time)

            # Prepare data policy gradient
            log_prob_tensor = torch.stack(log_probs)
            reward_tensor = torch.from_numpy(np.array(rewards))

            # Policy gradient term
            performance = (-(log_prob_tensor * reward_tensor)).sum()

            # Backprop on policy gradient
            optimizer.zero_grad()
            performance.backward()
            optimizer.step()
        print("Epoch {}/{} (Mean reward: {}, Mean exec time: {}".format(i + 1, n_epoch,
                                                                        sum(rewards_epoch) / len(rewards_epoch),
                                                                        sum(execution_times_epoch) /
                                                                        len(execution_times_epoch)
                                                                        ))
