import json
import os
import re
import argparse
import sys
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.pretrain_procedure import main_pretraining_dataset
from src.rl_fine_tuning_qr_dqn_learning import main_rl_tuning
from src.utils.training_utils.training_tracking import ExperimentWriter

# Root dir global for file loading
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set experiment config path
os.environ["HYDRA_CONFIG_PATH"] = os.path.join(ROOT_DIR,
                                               "experiments", "experiment_configs", "pretraining_experiments")


def get_config_name():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="pretrain_ppo_qr_dqn_naive_tree_lstm_yago_stars",
        help="Optional Hydra config name"
    )
    args, remaining = parser.parse_known_args()
    # Put back remaining args so Hydra still sees them
    sys.argv = [sys.argv[0]] + remaining
    return args.config


# Get config_name before Hydra runs
config_name = get_config_name()

@hydra.main(version_base=None, config_path=os.getenv("HYDRA_CONFIG_PATH"),
            config_name=config_name)
def main(cfg: DictConfig):
    print(f"Loaded config: {config_name}")
    train_set, val_set = None, None
    writer = None
    rl_configs = OmegaConf.select(cfg, "rl_training", default=None)

    skip_pretraining = False
    if rl_configs:
        skip_pretraining = OmegaConf.select(list(rl_configs.values())[0], "model_directory", default=None)

    if "pretraining" in cfg and not skip_pretraining:
        c1 = cfg.pretraining
        with open(os.path.join(ROOT_DIR, c1.model_config), "r") as f:
            config = yaml.safe_load(f)

        writer = ExperimentWriter(c1.experiment_root_directory, config_name,
                                  dict(c1), dict(config['model']))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_set, val_set = main_pretraining_dataset(
            queries_location_train=c1.dataset_train,
            queries_location_val=c1.dataset_val,
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
            lr=c1.lr,
            device=device
        )
    # One cardinality estimation model can spawn multiple RL-based fine-tuning experiments that use the same train/val
    # dataset and estimated cardinality model.
    if "rl_training" in cfg:
        for name, c2 in cfg["rl_training"].items():
            print("Experiment: {}-{}".format(c2.algorithm, c2.extractor_type))

            if OmegaConf.select(c2, "model_directory", default=None):
                model_dir = c2.model_directory
            elif writer:
                model_dir = find_best_epoch_directory(writer.experiment_directory, "val_q_error")
            else:
                raise ValueError("Either a pretraining experiment or a model directory must be specified")
            print(f"Determined model directory: {model_dir}")

            if not OmegaConf.select(c2, "query_location_dict", default=None):
                if "pretraining" not in cfg:
                    raise ValueError("Either a pretraining experiment or a model directory must be specified")
                print("INFO: No query_location_dict provided, using train and validation datasets from pretraining")
                c1 = cfg.pretraining
                query_location_dict = {
                    "queries_train": c1.dataset_train,
                    "queries_val": c1.dataset_val,
                    "endpoint_location": c1.endpoint,
                    "rdf2vec_vectors": c1.embeddings,
                    "occurrences": c1.occurrences_location,
                    "tp_cardinalities": c1.tp_cardinality_location,
                }
            else:
                query_location_dict = c2.query_location_dict

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
                model_dir,
                train_set,
                val_set,
                query_location_dict,
                model_ckp_fine_tune=OmegaConf.select(c2, "model_ckp_fine_tune", default=None),
                seed=c2.seed
            )


def find_last_epoch_directory(base_model_dir):
    epoch_dirs = [
        d for d in os.listdir(base_model_dir)
        if os.path.isdir(os.path.join(base_model_dir, d)) and d.startswith("epoch-")
    ]

    # Extract the numbers
    epochs = []
    for d in epoch_dirs:
        m = re.match(r"epoch-(\d+)", d)
        if m:
            epochs.append((int(m.group(1)), d))

    # Pick the max
    if epochs:
        last_epoch_num, last_epoch_dir = max(epochs, key=lambda x: x[0])
        final_path = os.path.join(base_model_dir, last_epoch_dir, "model")
        print(f"Last epoch path: {final_path}")
    else:
        raise ValueError("No epoch directories found.")
    return final_path


def find_best_epoch_directory(base_model_dir, values_key):
    last_epoch_model_dir = find_last_epoch_directory(base_model_dir)
    last_epoch_dir = os.path.dirname(last_epoch_model_dir)
    with open(os.path.join(last_epoch_dir, "best_values.json"), "r") as f:
        best_values_dict = json.load(f)
        best_values = best_values_dict[values_key]
        best_epoch = best_values["epoch"]
    path = Path(last_epoch_model_dir)

    # Replace last epoch with best epoch
    parts = list(path.parts)
    new_parts = [p if not p.startswith("epoch-") else f"epoch-{best_epoch}" for p in parts]
    best_epoch_model_dir = Path(*new_parts)

    return best_epoch_model_dir


if __name__ == "__main__":
    main()
