import torch
import torch.nn as nn

from src.models.model_instantiator import ModelFactory
from src.utils.training_utils.utils import initialize_graph_models, get_parameters_model


def load_queries_txt_file(location):
    queries = []
    with open(location, 'r') as f:
        for line in f.readlines():
            queries.append(line)
            print(line)
    pass


def load_cardinalities_txt_file():
    pass


def run_training(queries_location, rdf2vec_vector_location, endpoint_uri,
                 n_epoch, batch_size, lr, n_episodes_query, discount_factor, seed):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on {}".format(device))

    # Build graph embedding models from config
    model_factory = ModelFactory("experiments/configs/test_config.yaml")
    graph_embedding_models = initialize_graph_models([(model_factory, 4)])

    # Build cardinality estimator from config
    # TODO

    parameters = get_parameters_model(graph_embedding_models)

    # Move models to cuda
    [graph_model.to(device) for graph_model in graph_embedding_models]

    # Define optimizer over parameters
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # # Prepare features of queries
    # queries = prepare(env, queries_location, rdf2vec_vector_location)
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
