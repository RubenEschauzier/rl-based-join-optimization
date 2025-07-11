import os
import warnings

import torch.nn
import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from scipy.stats import linregress, stats
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from src.models.model_instantiator import ModelFactory
from src.models.rl_algorithms.maskable_qrdqn_policy import MaskableQRDQNPolicy
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.models.rl_algorithms.qrdqn_feature_extractors import QRDQNFeatureExtractor, QRDQNFeatureExtractorTreeLSTM
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_cardinality_estimation_feedback import QueryGymCardinalityEstimationFeedback
from src.query_environments.gym.query_gym_execution_feedback import QueryExecutionGymExecutionFeedback
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN


def load_weights_from_pretraining(model_to_init, model_dir: str,
                                  embedding_file: str, heads_files,
                                  float_weights=False):
    """
    Load the weights from pretraining into the model. It assumes the model has a separate embedding component and
    separate estimation heads. Furthermore, it assumes the model heads appear in the same order for both the model
    and the file names list :param heads_files. It requires an id for each weight to be given in the config to
    match the uninitialized weights with the pretrained weights.
    :param model_to_init: Gine_conv_model to init. Currently, it does not work for other model types
    :param model_dir: Directory where different model parts are saved (like embedding.pt or cardinality_head.pt)
    :param embedding_file: Filename of the embedding part of model
    :param heads_files: Filename of the estimation heads of the model
    :param float_weights: Whether the weights should forcefully be converted to float
    :return: void
    """
    un_init_emb = model_to_init.embedding_model.state_dict()
    un_init_heads = [head.state_dict() for head in model_to_init.heads]

    weights_embedding = torch.load(os.path.join(model_dir, embedding_file), weights_only=True)
    weights_heads = [torch.load(os.path.join(model_dir, head_file)) for head_file in heads_files]

    filtered_emb_weights = {k: v for k, v in weights_embedding.items() if k in un_init_emb}
    filtered_heads_weights = [{k: v for k, v in weights_head.items() if k in un_init_head}
                                for un_init_head, weights_head in zip(un_init_heads, weights_heads)]
    n_filtered_emb = len(list(weights_embedding.keys())) - len(list(filtered_emb_weights.keys()))
    n_filtered_heads = [len(list(weights_head.keys())) - len(list(filtered_head_weights.keys()))
        for weights_head, filtered_head_weights in zip(weights_heads, filtered_heads_weights)]
    if n_filtered_emb > 0:
        warnings.warn(
            "Filtered {} weights from the embedding layer".format(n_filtered_emb),
            stacklevel=2
        )
    for i, n_filtered_head in enumerate(n_filtered_heads):
        if n_filtered_head > 0:
            warnings.warn(
                "Filtered {} weights from the head {}".format(n_filtered_head, i),
                stacklevel=2
            )
    model_to_init.embedding_model.load_state_dict(filtered_emb_weights)
    for model_head, filtered_head_weights in zip(model_to_init.heads, filtered_heads_weights):
        model_head.load_state_dict(filtered_head_weights)

    if float_weights:
        model_to_init.float()


def freeze_weights(model_to_freeze):
    for param in model_to_freeze.parameters():
        param.requires_grad = False


def scatter_plot_reward_execution_time(reward, execution_time):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(reward, execution_time)
    line = slope * reward + intercept

    # Plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    plt.scatter(reward, execution_time, color='dodgerblue', s=5, edgecolor='k', label='Query Executions')
    plt.plot(reward, line, color='crimson', linewidth=1, label='Linear OLS fit')

    # Annotate regression info
    plt.text(min(reward), max(execution_time), f'$R^2$ = {r_value ** 2:.3f}',
             fontsize=12, color='crimson', bbox=dict(facecolor='white', alpha=0.8))

    plt.title('Query execution time as a function of query reward signal', fontsize=16)
    plt.xlabel('Query Reward Signal', fontsize=20)
    plt.ylabel('Log(Query Execution Time)', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_env(embedding_model, dataset, train_mode):
    #TODO For default without cardinality estimation we can only use the embedding head. By filtering on
    # the head output type
    return QueryExecutionGymExecutionFeedback(dataset, embedding_model, query_env, reward_type='cost_ratio',
                                              train_mode=train_mode)

def make_env_cardinality_estimation(embedding_cardinality_model, dataset, train_mode):
    # TODO in this one, just pass an argument that gives the type associated with the cardinality estimation head.
    return QueryGymCardinalityEstimationFeedback(query_dataset=dataset,
                                              query_embedder=embedding_cardinality_model, env=query_env,
                                              reward_type='cost_ratio', train_mode=train_mode)


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_full_101"
    rdf2vec_vector_location = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/full/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/full/tp_cardinalities.json"
    model_config = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head.yaml"
    model_directory = (r"experiments/experiment_outputs/"
                       r"pretrain_experiment_triple_conv_l1loss_full_run-05-07-2025"
                       r"/epoch-49/model")

    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.2, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location)
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    load_weights_from_pretraining(gine_conv_model, model_directory,
                                  "embedding_model.pt",
                                  ["head_cardinality.pt"],
                                  float_weights=True)

    gine_conv_model.embedding_model.eval()

    # Set the embedding head to eval too as these layers will be frozen
    embedding_head_name = "triple_embedding"
    if embedding_head_name in gine_conv_model.head_types:
        idx = gine_conv_model.head_types.index(embedding_head_name)
        gine_conv_model.heads[idx].eval()

    # Freeze the weights
    freeze_weights(gine_conv_model)

    policy_kwargs = dict(
        features_extractor_class=QRDQNFeatureExtractorTreeLSTM,
        features_extractor_kwargs=dict(feature_dim=200),
    )

    train_env = make_env_cardinality_estimation(gine_conv_model, train_dataset, True)
    val_env = Monitor(make_env_cardinality_estimation(gine_conv_model, val_dataset.shuffle()[:30], False))
    # train_env = make_env(gine_conv_model, train_dataset, True)
    # val_env = Monitor(make_env(gine_conv_model, val_dataset.shuffle()[:30], False))

    # train_env = make_env_cardinality_estimation(train_dataset, True)
    # data_execution_time, data_reward = train_env.validate_cost_function(
    #     train_dataset, 400, 3, "intermediate_results"
    # )
    # scatter_plot_reward_execution_time(np.array(data_reward), np.array(data_execution_time))
    # scatter_plot_reward_execution_time(np.array(data_reward), np.log(data_execution_time))
    model = MaskableQRDQN(MaskableQRDQNPolicy,
                          train_env,
                          policy_kwargs=policy_kwargs,
                          exploration_fraction=0.2,
                          exploration_initial_eps=1,
                          exploration_final_eps=0.05,
                          learning_starts=100,
                          verbose=1,
                          buffer_size=200000,
                          replay_buffer_class=MaskedDictReplayBuffer,
                          tensorboard_log="./tensorboard_logs/",
                          device='cuda',
                          train_freq=(15, "episode"),
                          delayed_rewards=False
                          )
    eval_dataset = val_dataset.shuffle()
    eval_callback = MaskableEvalCallback(
        eval_env=val_env,
        use_masking=True,
        n_eval_episodes=30,
        eval_freq=300,
        deterministic=True,
        render=False,
    )

    # model = MaskableQRDQN(MaskableQRDQNPolicy,
    #                       DummyVecEnv([lambda: make_env() for _ in range(n_envs)]),
    #                       policy_kwargs=policy_kwargs, verbose=2, buffer_size=10000,
    #                       replay_buffer_class=MaskedDictReplayBuffer)

    # Train the model
    model.learn(total_timesteps=1000000, callback=eval_callback)
    model.save("TempModelNoRnn")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

    # I have not yet disabled gradients for other Q-values, how do I know my Q-values are being masked properly in the
    # training part of my model. Also, still have no statefulness
    # https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/qrdqn/qrdqn.html#QRDQN
    # Note that this DOES include a state variable, how can we use it? Look at how recurrent PPO works?
    # Overwrite _sample_action to use the mask and state, collect_rollout, use a recurrent buffer (?)
    # I will have to write my own QRDQN class that inherits from OffPolicyAlgorithm in it I need to add action_mask
    # and a recurrent maskable roll_out buffer.
    # For the policy, I'll use an LSTM-based policy
