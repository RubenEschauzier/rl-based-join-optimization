import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from src.policy_gradient_rl_procedure import run_training_policy_gradient
from src.pretrain_procedure import main_pretraining_dataset
from src.utils.training_utils.training_tracking import ExperimentWriter

# Root dir global for file loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set experiment config path
os.environ["HYDRA_CONFIG_PATH"] = os.path.join(ROOT_DIR, "experiments", "experiment_configs")


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
            config_name="pretrain_experiment_triple_conv")
def main(cfg: DictConfig):
    if cfg.pretraining:
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
    if cfg.rltraining:
        pass



if __name__ == "__main__":
    main()
