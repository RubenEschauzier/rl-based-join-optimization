import torch.nn
from numpy.ma.core import MaskError
from sb3_contrib import QRDQN
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from src.models.model_instantiator import ModelFactory
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym import QueryExecutionGym
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN, QRDQNFeatureExtractor, MaskableQRDQNPolicy

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None

    def _on_training_start(self) -> None:
        # Get the writer from the logger (SB3 uses TensorBoard by default if specified)
        self.writer = SummaryWriter(log_dir=self.logger.dir)

    def _on_step(self) -> bool:
        # Get the reward of the current step
        reward = self.locals["rewards"][0]  # assuming a single environment
        step = self.num_timesteps

        if self.writer:
            self.writer.add_scalar("raw_reward", reward, step)
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()


def load_weights_from_pretraining(model_to_init, state_dict_location, float_weights=False):
    partial = model_to_init.state_dict()
    weights = torch.load(state_dict_location, weights_only=True)
    filtered = {k: v for k, v in weights.items() if k in partial}
    model_to_init.load_state_dict(filtered)
    if float_weights:
        model_to_init.float()


if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_full_101"
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

    def make_env():
        return QueryExecutionGym(train_dataset, 512, gine_conv_model, query_env, reward_type='intermediate_results')

    n_envs = 4
    policy_kwargs = dict(
        features_extractor_class=QRDQNFeatureExtractor,
        features_extractor_kwargs=dict(feature_dim=512),
    )
    model = MaskableQRDQN(MaskableQRDQNPolicy,
                          make_env(),
                          policy_kwargs=policy_kwargs,
                          exploration_fraction=0.5,
                          verbose=1,
                          buffer_size=10000,
                          replay_buffer_class=MaskedDictReplayBuffer,
                          tensorboard_log="./tensorboard_logs/",
                          device='cpu')

    # model = MaskableQRDQN(MaskableQRDQNPolicy,
    #                       DummyVecEnv([lambda: make_env() for _ in range(n_envs)]),
    #                       policy_kwargs=policy_kwargs, verbose=2, buffer_size=10000,
    #                       replay_buffer_class=MaskedDictReplayBuffer)

    # Train the model
    model.learn(total_timesteps=20000, callback = RewardLoggerCallback(verbose=0))
    model.save("TempModelNoRnn")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

    # I have not yet disabled gradients for other Q-values, how do I know my Q-values are being masked properly in the
    # training part of my model. Also still have no statefulness
    # https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/qrdqn/qrdqn.html#QRDQN
    # Note that this DOES include a state variable, how can we use it? Look at how recurrent PPO works?
    # Overwrite _sample_action to use the mask and state, collect_rollout, use a recurrent buffer (?)
    # I will have to write my own QRDQN class that inherits from OffPolicyAlgorithm in it I need to add action_mask
    # and a recurrent maskable roll_out buffer.
    # For the policy, I'll use an LSTM-based policy
