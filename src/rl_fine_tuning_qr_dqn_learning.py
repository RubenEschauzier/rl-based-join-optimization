import os
import warnings
from typing import Literal

import gymnasium as gym
import torch.nn
from matplotlib import pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from scipy.stats import linregress, stats
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.models.model_instantiator import ModelFactory
from src.models.rl_algorithms.custom_callbacks import EvalWithOptimalLoggingCallback
from src.models.rl_algorithms.maskable_actor_critic_policy import MaskableActorCriticPolicyCustomPreprocessing
from src.models.rl_algorithms.maskable_qrdqn_policy import MaskableQRDQNPolicy
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.models.rl_algorithms.qrdqn_feature_extractors import QRDQNFeatureExtractorTreeLSTM, QRDQNFeatureExtractor
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


def make_env(embedding_model, dataset, train_mode, env):
    #TODO For default without cardinality estimation we can only use the embedding head. By filtering on
    # the head output type
    return QueryExecutionGymExecutionFeedback(dataset, embedding_model, env, reward_type='cost_ratio',
                                              train_mode=train_mode)

def make_env_cardinality_estimation(embedding_cardinality_model, dataset, train_mode, env, enable_optimal_eval=False):
    return QueryGymCardinalityEstimationFeedback(query_dataset=dataset,
                                                 query_embedder=embedding_cardinality_model, env=env,
                                                 reward_type='cost_ratio', train_mode=train_mode,
                                                 enable_optimal_eval=enable_optimal_eval)

def prepare_envs(endpoint_location, queries_location, rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location,
                 model_config, model_directory):
    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.02, to_load=None,
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

    train_env = make_env_cardinality_estimation(gine_conv_model, train_dataset, True, query_env)
    val_env = make_env_cardinality_estimation(gine_conv_model,
                                              val_dataset.shuffle(),
                                              False,
                                              query_env,
                                              enable_optimal_eval=True)

    return train_env, val_env, train_dataset, val_dataset


def run_qr_dqn_estimated_cardinality(n_steps, model_save_loc,
                                     endpoint_location,
                                     queries_location,
                                     rdf2vec_vector_location,
                                     occurrences_location,
                                     tp_cardinality_location,
                                     model_config, model_directory,
                                     extractor_class, extractor_kwargs, net_arch
                                     ):
    train_env, val_env, train_dataset, val_dataset = prepare_envs(endpoint_location, queries_location, rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location,
                 model_config, model_directory)
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
    model.save(model_save_loc)

def run_ppo_estimated_cardinality(n_steps, model_save_loc,
                                  endpoint_location, queries_location, rdf2vec_vector_location,
                                  occurrences_location, tp_cardinality_location,
                                  model_config, model_directory,
                                  extractor_class, extractor_kwargs
                                  ):
    train_env, val_env, train_dataset, val_dataset = prepare_envs(endpoint_location, queries_location, rdf2vec_vector_location,
                 occurrences_location, tp_cardinality_location,
                 model_config, model_directory)
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
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
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=500000, callback=eval_callback)


def main_qr_dqn():
    torch.manual_seed(0)
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
        run_qr_dqn_estimated_cardinality(500000, "experiments/experiment_outputs/tree-lstm-3-only",
                                         endpoint_location, queries_location, rdf2vec_vector_location,
                                         occurrences_location, tp_cardinality_location,
                                         model_config, model_directory,
                                         extractor_class=QRDQNFeatureExtractorTreeLSTM,
                                         extractor_kwargs=dict(feature_dim=200),
                                         net_arch=[256, 256])
    elif extractor_type == "naive":
        run_qr_dqn_estimated_cardinality(500000, "experiments/experiment_outputs/naive-3-only",
                                         endpoint_location, queries_location, rdf2vec_vector_location,
                                         occurrences_location, tp_cardinality_location,
                                         model_config, model_directory,
                                         extractor_class=QRDQNFeatureExtractor,
                                         extractor_kwargs=dict(feature_dim=200),
                                         net_arch=[256, 256])
    else:
        raise ValueError("Invalid extractor type: {}".format(extractor_type))

def main_ppo():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    # queries_location = "data/pretrain_data/datasets/p_e_full_101"
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
        run_ppo_estimated_cardinality(500000, "experiments/experiment_outputs/ppo-tree-lstm-3-5",
                                      endpoint_location, queries_location, rdf2vec_vector_location,
                                      occurrences_location, tp_cardinality_location,
                                      model_config, model_directory,
                                      extractor_class=QRDQNFeatureExtractorTreeLSTM,
                                      extractor_kwargs=dict(feature_dim=200))
    elif extractor_type == "naive":
        run_ppo_estimated_cardinality(500000, "experiments/experiment_outputs/ppo-naive-3-5-only",
                                         endpoint_location, queries_location, rdf2vec_vector_location,
                                         occurrences_location, tp_cardinality_location,
                                         model_config, model_directory,
                                         extractor_class=QRDQNFeatureExtractor,
                                         extractor_kwargs=dict(feature_dim=200))
    else:
        raise ValueError("Invalid extractor type: {}".format(extractor_type))


if __name__ == "__main__":
    main_qr_dqn()