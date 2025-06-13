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
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym_cardinality_estimation_feedback import QueryGymCardinalityEstimationFeedback
from src.query_environments.gym.query_gym_execution_feedback import QueryExecutionGymExecutionFeedback
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN, QRDQNFeatureExtractor, MaskableQRDQNPolicy


def load_weights_from_pretraining(model_to_init, state_dict_location, float_weights=False):
    partial = model_to_init.state_dict()
    print(partial)
    weights = torch.load(state_dict_location, weights_only=True)
    print(weights)
    filtered = {k: v for k, v in weights.items() if k in partial}
    model_to_init.load_state_dict(filtered)
    if float_weights:
        model_to_init.float()

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


def make_env(dataset, train_mode):
    return QueryExecutionGymExecutionFeedback(dataset, 512, gine_conv_model, query_env, reward_type='cost_ratio',
                                              train_mode=train_mode)

def make_env_cardinality_estimation(dataset, train_mode):
    return QueryGymCardinalityEstimationFeedback(query_dataset=dataset, feature_dim=512,
                                              query_embedder=gine_conv_model, env=query_env,
                                              reward_type='cost_ratio', train_mode=train_mode)


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_size_3_only_101"
    rdf2vec_vector_location = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/full/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/full/tp_cardinalities.json"
    model_config = "experiments/model_configs/policy_networks/t_cv_repr_large.yaml"
    weights_path = (r"experiments/experiment_outputs/"
                    r"pretrain_experiment_triple_conv-15-04-2025-19-54-41/epoch-40/"
                    r"model.pt")

    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.2, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location)
    model_factory_gine_conv = ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()
    load_weights_from_pretraining(gine_conv_model, weights_path, float_weights=True)
    gine_conv_model.eval()


    policy_kwargs = dict(
        features_extractor_class=QRDQNFeatureExtractor,
        features_extractor_kwargs=dict(feature_dim=512),
    )
    train_env = make_env(train_dataset, True)
    val_env = Monitor(make_env(val_dataset.shuffle()[:30], False)),
    data_execution_time, data_reward = train_env.validate_cost_function(
        train_dataset, 400, 3, "intermediate_results"
    )
    scatter_plot_reward_execution_time(np.array(data_reward), np.array(data_execution_time))
    scatter_plot_reward_execution_time(np.array(data_reward), np.log(data_execution_time))
    model = MaskableQRDQN(MaskableQRDQNPolicy,
                          train_env,
                          policy_kwargs=policy_kwargs,
                          exploration_fraction=0.2,
                          exploration_initial_eps=1,
                          exploration_final_eps=0.05,
                          learning_starts=100,
                          verbose=1,
                          buffer_size=10000,
                          replay_buffer_class=MaskedDictReplayBuffer,
                          tensorboard_log="./tensorboard_logs/",
                          device='cpu',
                          train_freq=(15, "episode"),
                          delayed_rewards=False
                          )
    eval_dataset = val_dataset.shuffle()[:30]
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
    model.learn(total_timesteps=20000, callback=eval_callback)
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
