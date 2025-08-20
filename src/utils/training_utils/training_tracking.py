import builtins
import os
import json
import warnings
from datetime import datetime

import torch


class TrainSummary:
    # Fields to track is list of tuples with (name, max or min) depending on whether max or min is best
    def __init__(self, fields_to_track):
        # Note that these dictionary values can be anything
        self.best_values = {}
        self.per_epoch_values = {}
        for field in fields_to_track:
            self.best_values[field[0]] = {
                "epoch": 0,
                "value": float("-inf") if field[1] == 'max' else float("inf") ,
                "type": field[1]
            }
            self.per_epoch_values[field[0]] = []

    def update(self, values, epoch):
        self.update_best(values, epoch)
        self.update_epoch(values)
        pass

    def update_epoch(self, values):
        for key, value in values.items():
            self.per_epoch_values[key].append(value)

    def update_best(self, values, epoch):
        for key, value in values.items():
            best_value = getattr(builtins, self.best_values[key]["type"])(value, self.best_values[key]["value"])
            # We found a better value
            if best_value == value:
                self.best_values[key]["value"] = value
                self.best_values[key]["epoch"] = epoch

    def summary(self):
        return self.best_values, self.per_epoch_values

class ExperimentWriter:
    def __init__(self, experiment_root_directory, experiment_name, experiment_config, model_config):
        self.experiment_directory = os.path.join(experiment_root_directory,
                                                 experiment_name + "-" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        self.experiment_config = experiment_config
        self.model_config = model_config

        pass
    def create_experiment_directory(self):
        if os.path.exists(self.experiment_directory):
            raise FileExistsError(f"Experiment directory already exists: {self.experiment_directory}")
        os.mkdir(self.experiment_directory)
        experiment_details = {
            "experiment_config": self.experiment_config,
            "model_config": self.model_config,
        }
        experiment_details_file_location = os.path.join(str(self.experiment_directory), "experiment_details.json")
        with open(experiment_details_file_location, "w", encoding="utf-8") as f:
            json.dump(experiment_details, f, indent=2, sort_keys=True)



    def write_epoch_to_file(self, val_predictions, best_values, per_epoch_values, model, epoch):
        epoch_progress_file = os.path.join(str(self.experiment_directory), "train_progress.json")
        epoch_dir = os.path.join(str(self.experiment_directory), "epoch-{}".format(epoch))
        os.mkdir(epoch_dir)
        with open(epoch_progress_file, "w", encoding="utf-8") as f:
            json.dump(per_epoch_values, f, indent=2, sort_keys=True)
        with open(os.path.join(epoch_dir, "val_predictions.json"), "w", encoding="utf-8") as f1:
            json.dump(val_predictions, f1, indent=2, sort_keys=True)
        with open(os.path.join(epoch_dir, "best_values.json"), "w", encoding="utf-8") as f2:
            json.dump(best_values, f2, indent=2, sort_keys=True)

        # Serialize model in separate directory
        os.makedirs(os.path.join(epoch_dir, "model"), exist_ok=True)
        try:
            # If model has its own serialize_model() function we use that
            model.serialize_model(os.path.join(epoch_dir, "model"))
        except AttributeError as err:
            warnings.warn("{} \n Falling back to default torch.save() call".format(err),
                stacklevel=2
            )
            torch.save(model.state_dict(), os.path.join(epoch_dir, "model.pt"))