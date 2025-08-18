import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from src.policy_gradient_rl_procedure import run_training_policy_gradient
from src.pretrain_procedure import main_pretraining_dataset
from src.rl_fine_tuning_qr_dqn_learning import main_rl_tuning
from src.utils.training_utils.training_tracking import ExperimentWriter

# Root dir global for file loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set experiment config path
os.environ["HYDRA_CONFIG_PATH"] = os.path.join(ROOT_DIR,
                                               "experiments", "experiment_configs", "rl_fine_tuning_experiments")


def main_policy_rl():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv-default-instantiation/sparql"
    query_location = "data/input/queries"
    rdf2vec_vector_location = "data/rdf2vec_vectors/vectors_depth_1_full_entities.json"
    n_epoch = 50
    batch_size = 12
    seed = 0
    run_training_policy_gradient(query_location, rdf2vec_vector_location, endpoint_location, n_epoch, batch_size, 1e-6,
                                 4, .99, seed)


def main_q_learning_rl():
    pass

@hydra.main(version_base=None, config_path=os.getenv("HYDRA_CONFIG_PATH"),
            config_name="fine_tune_3_5_stars_ppo_lstm")
def main(cfg: DictConfig):
    train_set, val_set = None, None
    if "pretraining" in cfg:
        c1 = cfg.pretraining

        with open(os.path.join(ROOT_DIR,c1.model_config), "r") as f:
            config = yaml.safe_load(f)

        writer = ExperimentWriter(c1.experiment_root_directory, "pretrain_experiment_triple_conv",
                                  dict(c1), dict(config['model']))
        train_set, val_set = main_pretraining_dataset(
            queries_location=c1.dataset,
            endpoint_location=c1.endpoint,
            rdf2vec_vector_location=c1.embeddings,
            feature_type=c1.feature_type,
            occurrences_location=c1.occurrences_location,
            tp_cardinality_location=c1.tp_cardinality_location,
            writer=writer,
            model_config_location=c1.model_config,
            n_epoch=c1.n_epoch,
            batch_size=c1.batch_size,
            seed=c1.seed,
            lr=c1.lr
        )
    # One cardinality estimation model can spawn multiple RL-based fine-tuning experiments that use the same train/val
    # dataset and estimated cardinality model.
    if "rl_training" in cfg:
        c2 = cfg.rl_training
        if (not train_set or not val_set) and not c2.query_location_dict:
            raise ValueError("Either train_set and val_set or query_location_dict must be specified")
        main_rl_tuning(
            c2.algorithm,
            c2.extractor_type,
            c2.n_steps,
            c2.n_steps_fine_tune,
            c2.n_eval_episodes,
            c2.model_save_loc_estimated,
            c2.model_save_loc_fine_tuned,
            c2.net_arch,
            c2.feature_dim,
            c2.endpoint_location,
            c2.model_config_location,
            c2.model_directory,
            train_set,
            val_set,
            c2.query_location_dict,
            c2.seed
        )

if __name__ == "__main__":
    main()
