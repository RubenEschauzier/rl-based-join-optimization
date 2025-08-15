import os
import warnings
from typing import Literal

import gymnasium as gym
import torch.nn
from matplotlib import pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from scipy.stats import linregress, stats
from stable_baselines3.common.monitor import Monitor

from src.models.model_instantiator import ModelFactory
from src.models.rl_algorithms.custom_callbacks import EvalWithOptimalLoggingCallback
from src.models.rl_algorithms.maskable_actor_critic_policy import MaskableActorCriticPolicyCustomPreprocessing
from src.models.rl_algorithms.maskable_qrdqn_policy import MaskableQRDQNPolicy
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.models.rl_algorithms.qrdqn_feature_extractors import QRDQNFeatureExtractorTreeLSTM, QRDQNFeatureExtractor
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_estimated_cost import QueryGymEstimatedCost
from src.query_environments.gym.query_gym_execution_cost import QueryGymExecutionCost
from src.query_environments.gym.query_gym_execution_latency import QueryGymExecutionLatency
from src.query_environments.gym.query_gym_wrapper_dp_baseline import OrderDynamicProgramming
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN
from src.utils.training_utils.utils import reset_value_head_only


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


def make_env_estimated_cost(embedding_cardinality_model, dataset, train_mode, env):
    return QueryGymEstimatedCost(query_dataset=dataset,
                                 query_embedder=embedding_cardinality_model,
                                 env=env,
                                 train_mode=train_mode)

def make_env_execution_cost(embedding_cardinality_model, dataset, train_mode, env):
    return QueryGymExecutionCost(query_timeout=30000,
                                 query_dataset=dataset,
                                 query_embedder=embedding_cardinality_model,
                                 env=env,
                                 train_mode=train_mode)

def make_env_execution_latency(embedding_cardinality_model, dataset, train_mode, env, curriculum):
    return QueryGymExecutionLatency(query_timeout=30000,
                                    query_dataset=dataset,
                                    query_embedder=embedding_cardinality_model,
                                    env=env,
                                    train_mode=train_mode,
                                    n_train_episodes=50000,
                                    cost_frac=.2,
                                    curriculum=curriculum)

def wrap_validation_environment_with_baseline(val_env, cache_optimal_cost):
    return OrderDynamicProgramming(val_env, cache_optimal_cost)

def prepare_queries(query_env,
                    queries_location, endpoint_location, rdf2vec_vector_location, occurrences_location,
                    tp_cardinality_location,
                    validation_size = .1):
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=validation_size, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           shuffle=True)
    return train_dataset, val_dataset

def prepare_embedding_model(model_config, model_directory)
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
    return gine_conv_model


def prepare_experiment(endpoint_location, queries_location, rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location,
                 model_config, model_directory):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.03, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location,
                                                           shuffle=True)
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
    return gine_conv_model, train_dataset, val_dataset, query_env

def prepare_cardinality_envs(emb_model, train_dataset, val_dataset, query_env, cache_optimal_cost):
    train_env = make_env_estimated_cost(emb_model, train_dataset, True, query_env)
    val_env = make_env_estimated_cost(emb_model, val_dataset.shuffle(),
                                      False, query_env)
    val_env = wrap_validation_environment_with_baseline(val_env, cache_optimal_cost)

    return train_env, val_env

def prepare_execution_cost_envs(emb_model, train_dataset, val_dataset, query_env, cache_optimal_cost):
    train_env = make_env_execution_cost(emb_model, train_dataset, True, query_env)
    val_env = make_env_execution_cost(emb_model, val_dataset.shuffle(),
                                      False, query_env)
    val_env = wrap_validation_environment_with_baseline(val_env, cache_optimal_cost)

    return train_env, val_env

def prepare_execution_latency_envs(emb_model, train_dataset, val_dataset, query_env, cache_optimal_cost, curriculum):
    train_env = make_env_execution_latency(emb_model, train_dataset, True, query_env, curriculum)
    val_env = make_env_execution_latency(emb_model, val_dataset.shuffle(),
                                      False, query_env, curriculum)
    val_env = wrap_validation_environment_with_baseline(val_env, cache_optimal_cost)
    return train_env, val_env

def run_qr_dqn_estimated_cardinality(n_steps, n_steps_fine_tune, n_eval_queries,
                                     model_save_loc_estimated, model_save_loc_fine_tuned,
                                     endpoint_location,
                                     queries_location,
                                     rdf2vec_vector_location,
                                     occurrences_location,
                                     tp_cardinality_location,
                                     model_config, model_directory,
                                     extractor_class, extractor_kwargs, net_arch
                                     ):
    emb_model, train_dataset, val_dataset, query_env = (
        prepare_experiment(endpoint_location, queries_location, rdf2vec_vector_location,
                     occurrences_location, tp_cardinality_location,
                     model_config, model_directory))
    train_env, val_env = prepare_cardinality_envs(emb_model, train_dataset, val_dataset, query_env, True)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=net_arch,
    )

    model = MaskableQRDQN(MaskableQRDQNPolicy,
                          train_env,
                          policy_kwargs=policy_kwargs,
                          exploration_fraction=0.2,
                          exploration_initial_eps=1,
                          exploration_final_eps=0.05,
                          learning_starts=1000,
                          verbose=0,
                          buffer_size=100000,
                          replay_buffer_class=MaskedDictReplayBuffer,
                          tensorboard_log="./tensorboard_logs/",
                          device='cpu',
                          train_freq=(15, "episode"),
                          )
    print("Validation set contains {} queries".format(len(val_dataset)))
    eval_callback = EvalWithOptimalLoggingCallback(
        eval_env=Monitor(val_env),
        n_eval_episodes=len(val_dataset),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=n_steps, callback=eval_callback)
    model.save(model_save_loc_estimated)

    # Finetune based on query execution. This is with fewer steps due to cost of executing queries
    exec_train_env, exec_val_env = prepare_execution_cost_envs(
        emb_model, train_dataset, val_dataset[:n_eval_queries], query_env, False)
    # exec_train_env, exec_val_env = prepare_execution_latency_envs(
    #     emb_model, train_dataset, val_dataset[:n_eval_queries], query_env, False, False)

    model.set_env(exec_train_env)
    model.learning_rate = .1 * model.learning_rate
    model.exploration_rate = .5

    eval_callback_fine_tuned = EvalWithOptimalLoggingCallback(
        eval_env=Monitor(exec_val_env),
        use_masking=True,
        n_eval_episodes=len(val_dataset[:n_eval_queries]),
        eval_freq=500,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=n_steps_fine_tune, callback=eval_callback_fine_tuned)
    model.save(model_save_loc_fine_tuned)

    return model


def run_ppo_estimated_cardinality(n_steps, n_steps_fine_tune, n_eval_queries,
                                  model_save_loc_estimated, model_save_loc_fine_tuned,
                                  endpoint_location, queries_location, rdf2vec_vector_location,
                                  occurrences_location, tp_cardinality_location,
                                  model_config, model_directory,
                                  extractor_class, extractor_kwargs,
                                  net_arch
                                  ):
    emb_model, train_dataset, val_dataset, query_env = (
        prepare_experiment(endpoint_location, queries_location, rdf2vec_vector_location,
                     occurrences_location, tp_cardinality_location,
                     model_config, model_directory))
    train_env, val_env = prepare_cardinality_envs(emb_model, train_dataset, val_dataset, query_env, True)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=net_arch,
    )

    def mask_fn(env):
        return env.action_masks_ppo()

    train_env = ActionMasker(train_env, mask_fn)
    val_env = ActionMasker(val_env, mask_fn)


    model = MaskablePPO(MaskableActorCriticPolicyCustomPreprocessing,
                        train_env,
                        policy_kwargs=policy_kwargs,
                        device='cpu',
                        tensorboard_log="./tensorboard_logs/",
                        verbose=0
                        )
    print("Validation environment contains {} queries".format(len(val_dataset)))
    eval_callback = EvalWithOptimalLoggingCallback(
        eval_env=Monitor(val_env),
        use_masking=True,
        n_eval_episodes=len(val_dataset),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=n_steps, callback=eval_callback)
    model.save(model_save_loc_estimated)

    # Finetune based on query execution. This is with fewer steps due to cost of executing queries
    exec_train_env, exec_val_env = prepare_execution_cost_envs(
        emb_model, train_dataset, val_dataset[:n_eval_queries], query_env, False)

    exec_train_env = ActionMasker(exec_train_env, mask_fn)
    exec_val_env = ActionMasker(exec_val_env, mask_fn)
    print("Fine tune environment contains {} queries".format(len(val_dataset[:n_eval_queries])))
    #TODO: Finetuning collapses policy behavior as explained variance goes below zero.
    # # Reinitialize only the critic
    # model.policy.value_net.apply(model.policy._init_weights)
    # Reinitialize value network. Also look into how QR-DQN behaves.
    # Problem: in execution environment intermediate reward is 0, this leads to collapse of value function and thus
    # policy updates.
    # Solution? Use estimated reward as intermediate step, which introduces bias but smooths the reward compared to
    # pretraining.
    # Another: decay the estimated reward slowly
    # Another (not sure yet look-up paper) http://chatgpt.com/c/6887ed05-7fec-832a-8af1-8f93783248a9
    #TODO: Queries with multiple predicates are impossible to properly optimize and will cause a lot of timeouts due
    # to cartesian products. REMOVE THEM / PREVENT THEM.
    model.set_env(exec_train_env)
    # Ensure critic is reset due to change in reward function
    reset_value_head_only(model)
    model.learning_rate = .1 * model.learning_rate
    eval_callback_fine_tuned = EvalWithOptimalLoggingCallback(
        eval_env=Monitor(exec_val_env),
        use_masking=True,
        n_eval_episodes=len(val_dataset[:n_eval_queries]),
        eval_freq=500,
        deterministic=True,
        render=False,
    )
    model.save(model_save_loc_fine_tuned)
    model.learn(total_timesteps=n_steps_fine_tune, callback=eval_callback_fine_tuned)

def main_qr_dqn():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_size_3_5_101"
    rdf2vec_vector_location = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/full/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/full/tp_cardinalities.json"
    model_config = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head.yaml"
    model_directory = (r"experiments/experiment_outputs/"
                       r"pretrain_experiment_triple_conv_l1loss_full_run-05-07-2025"
                       r"/epoch-49/model")

    extractor_type: Literal["tree_lstm", "naive"] = "naive"
    if extractor_type == "tree_lstm":
        model = run_qr_dqn_estimated_cardinality(500000, 20000, 100,
                    "experiments/experiment_outputs/qr_dqn_tree_lstm_3_5",
                    "experiments/experiment_outputs/qr_dqn_tree_lstm_3_5_fine_tuned",
                                         endpoint_location, queries_location, rdf2vec_vector_location,
                                         occurrences_location, tp_cardinality_location,
                                         model_config, model_directory,
                                         extractor_class=QRDQNFeatureExtractorTreeLSTM,
                                         extractor_kwargs=dict(feature_dim=200),
                                         net_arch=[256, 256])
    elif extractor_type == "naive":
        model = run_qr_dqn_estimated_cardinality(400000, 50000, 100,
                    "experiments/experiment_outputs/qr_dqn_tree_naive_3_5",
                    "experiments/experiment_outputs/qr_dqn_tree_naive_3_5_fine_tuned",
                                         endpoint_location, queries_location, rdf2vec_vector_location,
                                         occurrences_location, tp_cardinality_location,
                                         model_config, model_directory,
                                         extractor_class=QRDQNFeatureExtractor,
                                         extractor_kwargs=dict(feature_dim=200),
                                         net_arch=[256, 256])
    else:
        raise ValueError("Invalid extractor type: {}".format(extractor_type))

def main_ppo():
    # TODO: Use same train-val-test split as pretraining?
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_size_3_5_101"

    rdf2vec_vector_location = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/full/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/full/tp_cardinalities.json"
    model_config = "experiments/model_configs/policy_networks/t_cv_repr_exact_cardinality_head.yaml"
    model_directory = (r"experiments/experiment_outputs/"
                       r"pretrain_experiment_triple_conv_l1loss_full_run-05-07-2025"
                       r"/epoch-49/model")
    extractor_type: Literal["tree_lstm", "naive"] = "tree_lstm"
    if extractor_type == "tree_lstm":
        run_ppo_estimated_cardinality(100000, 40000, 100,
                 "experiments/experiment_outputs/ppo-lstm-3-5-cost-est",
                 "experiments/experiment_outputs/ppo-lstm-3-5-fine-tuned",
                                      endpoint_location, queries_location, rdf2vec_vector_location,
                                      occurrences_location, tp_cardinality_location,
                                      model_config, model_directory,
                                      extractor_class=QRDQNFeatureExtractorTreeLSTM,
                                      extractor_kwargs=dict(feature_dim=200),
                                      net_arch=[256, 256])
    elif extractor_type == "naive":
        run_ppo_estimated_cardinality(100000, 40000, 100,
                 "experiments/experiment_outputs/ppo-naive-3-5-cost-est",
                 "experiments/experiment_outputs/ppo-naive-3-5-fine-tuned",
                                      endpoint_location, queries_location, rdf2vec_vector_location,
                                      occurrences_location, tp_cardinality_location,
                                      model_config, model_directory,
                                      extractor_class=QRDQNFeatureExtractor,
                                      extractor_kwargs=dict(feature_dim=200),
                                      net_arch=[256, 256])
    else:
        raise ValueError("Invalid extractor type: {}".format(extractor_type))

def main_rl_tuning(endpoint_location, model_config, model_directory, train_dataset=None, val_dataset=None,
                   query_location_dict = None):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    if query_location_dict:
        train_dataset, val_dataset = prepare_queries(query_env,
                                                     endpoint_location,
                                                     query_location_dict['queries'],
                                                     query_location_dict['rdf2vec_vectors'],
                                                     query_location_dict['occurrences'],
                                                     query_location_dict['tp_cardinalities'],
                                                     )
    if train_dataset is None or val_dataset is None:
        raise ValueError("Either train or validation dataset was None and there is no query_location_directory given in"
                         "config")
    emb_model = prepare_embedding_model(model_config, model_directory)

if __name__ == "__main__":
    #TODO: Fix Tree-LSTM
    #TODO: Make config based experiment for RL
    #TODO: Train on full queries, train on full query sets with all different shapes
    #TODO: Make functions that given trained RL model and trained pretrained model
    # - tests on full validation set.
    # - test on watdiv queries
    # - test the variability in performance
    #TODO: Ensure that pretraining train and validation sets are separated and are the same separation for RL training
    #TODO: Check validity baseline
    torch.manual_seed(0)
    main_ppo()
    # main_qr_dqn()