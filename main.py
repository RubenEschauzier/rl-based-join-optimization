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


def pretraining():
    endpoint_location = "http://localhost:9999/blazegraph/namespace/watdiv/sparql"

    train_query_location = "data/pretrain_data/generated_queries/sub_sampled/star_queries_2_3_5.json"
    dataset_root_location = "data/pretrain_data/generated_queries/sub_sampled_predicate_edge_undirected"

    # test_watdiv_query_location = "data/pretrain_data/test_queries/watdiv/template_queries"
    # test_watdiv_cardinality_location = "data/pretrain_data/test_queries/watdiv/template_cardinalities"

    rdf2vec_vector_location = "data/output/entity_embeddings/rdf2vec_vectors_depth_2_quick.json"
    rdf2vec_vector_location_dataset = "data/input/rdf2vec_vectors_gnce/vectors_gnce.json"

    model_config_location = "experiments/model_configs/triple_gine_conv_model.yaml"
    ckp_directory = "data/output/cardinality_estimation/cardinality_estimation_reproduction"

    n_epoch = 50
    batch_size = 32
    seed = 0
    lr = 1e-4
    main_pretraining_dataset(dataset_root_location, endpoint_location, rdf2vec_vector_location_dataset,
                             model_config_location,
                             n_epoch, batch_size, lr, seed)

def main_q_learning_rl():
    pass

@hydra.main(version_base=None, config_path=os.getenv("HYDRA_CONFIG_PATH"),
            config_name="pretrain_directional_gine_conv_large")
def main(cfg: DictConfig):
    if cfg.pretraining:
        c1 = cfg.pretraining

        with open(os.path.join(ROOT_DIR,c1.model_config), "r") as f:
            config = yaml.safe_load(f)
        writer = ExperimentWriter(c1.experiment_root_directory, "pretrain_directional_gine_conv_large",
                                  dict(c1), dict(config['model']))
        main_pretraining_dataset(
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
    if cfg.rltraining:
        pass



if __name__ == "__main__":
    main()
