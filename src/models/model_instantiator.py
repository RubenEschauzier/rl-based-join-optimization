import torch.nn as nn
import yaml

from src.models.graph_convolution_query_embedder import GCNConvQueryEmbeddingModel


# Adapted from https://siddhibajracharya.hashnode.dev/using-yaml-file-to-create-pytorch-model
class ModelFactory:
    """
        Create and load torch model using YAML files.

        params:
            config_file (str): yaml file path
    """

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.SUPPORTED = {"GCNConv": self.load_gcn_conv, "MLP": self.load_mlp}

    def build_model_from_config(self):
        """
            builds model from config file

            returns:
                torch model
        """
        try:
            return self.SUPPORTED[self.config["type"]]()
        except KeyError:
            ValueError(f"Unsupported model type: {self.config['type']}")

    def load_sequential(self):
        """
            Loads sequential model from config file.

            returns:
                sequential torch model
        """
        layers = []
        for layer_config in self.config["layers"]:
            layer_type = getattr(nn, layer_config["type"])
            parameters = {key: value for key, value in list(layer_config.items())[1:]}
            layer = layer_type(**parameters)
            layers.append(layer)
        return nn.Sequential(*layers)

    def load_gcn_conv(self):
        gcn_conv_model = GCNConvQueryEmbeddingModel()
        gcn_conv_model.init_model(self.config)
        return gcn_conv_model

    def load_mlp(self):
        return self.load_sequential()

    def load_config(self):
        """
            load config file

            returns:
                dictionary for YAML file.
        """
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        return config['model']
