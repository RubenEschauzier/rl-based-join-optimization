import torch.nn
from numpy.ma.core import MaskError
from sb3_contrib import QRDQN
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.models.model_instantiator import ModelFactory
from src.models.rl_algorithms.masked_replay_buffer import MaskedDictReplayBuffer
from src.models.sb3_feature_extractor import MaskableQRDQNPolicy, QRDQNFeatureExtractor
from src.query_environments.blazegraph.query_environment_blazegraph import BlazeGraphQueryEnvironment
from src.query_environments.gym.query_gym import QueryExecutionGym
from src.utils.training_utils.query_loading_utils import load_queries_into_dataset
from src.models.rl_algorithms.masked_qrdqn import MaskableQRDQN



if __name__ == "__main__":
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    queries_location = "data/pretrain_data/datasets/p_e_full_128"
    rdf2vec_vector_location = "data/input/rdf2vec_embeddings/rdf2vec_vectors_depth_2_quick.json"
    occurrences_location = "data/pretrain_data/pattern_term_cardinalities/partial/occurrences.json"
    tp_cardinality_location = "data/pretrain_data/pattern_term_cardinalities/partial/tp_cardinalities.json"
    model_config = r"experiments/model_configs/policy_networks/t_cv_repr_large.yaml"

    query_env = BlazeGraphQueryEnvironment(endpoint_location)
    train_dataset, val_dataset = load_queries_into_dataset(queries_location, endpoint_location,
                                                           rdf2vec_vector_location, query_env,
                                                           "predicate_edge",
                                                           validation_size=.2, to_load=None,
                                                           occurrences_location=occurrences_location,
                                                           tp_cardinality_location=tp_cardinality_location)
    model_factory_gine_conv= ModelFactory(model_config)
    gine_conv_model = model_factory_gine_conv.load_gine_conv()

    temp_join_emb_model = torch.nn.Linear(512, 512)

    def make_env():
        return QueryExecutionGym(train_dataset, 512, gine_conv_model, temp_join_emb_model, query_env)
    # gym_env = QueryExecutionGym(train_dataset, 512, gine_conv_model, temp_join_emb_model, query_env)
    n_envs = 4
    policy_kwargs = dict(
        features_extractor_class=QRDQNFeatureExtractor,
        features_extractor_kwargs=dict(feature_dim=512),
    )
    model = MaskableQRDQN(MaskableQRDQNPolicy,
                          make_env(),
                          policy_kwargs=policy_kwargs, verbose=2, buffer_size=10000,
                          replay_buffer_class=MaskedDictReplayBuffer)

    # model = MaskableQRDQN(MaskableQRDQNPolicy,
    #                       DummyVecEnv([lambda: make_env() for _ in range(n_envs)]),
    #                       policy_kwargs=policy_kwargs, verbose=2, buffer_size=10000,
    #                       replay_buffer_class=MaskedDictReplayBuffer)

    # Train the model
    model.learn(total_timesteps=100000)
    # My method does not work as simply giving a large negative reward does not work as the model can never know
    # Why this reward is obtained. Instead, we should overwrite this OR just use the mask data as input to the network
    # (The easy way)
    # https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/qrdqn/qrdqn.html#QRDQN
    # Note that this DOES include a state variable, how can we use it? Look at how recurrent PPO works?
    # Overwrite _sample_action to use the mask and state, collect_rollout, use a recurrent buffer (?)
    # I will have to write my own QRDQN class that inherits from OffPolicyAlgorithm in it I need to add action_mask
    # and a recurrent maskable roll_out buffer.
    # For the policy, I'll use an LSTM-based policy
