import numpy as np
import torch

from src.models.model_layers.graph_convolution import GCNConv


class GCNConvQueryEmbeddingModel:
    def __init__(self):
        self.layers = []
        self.layer_types = []
        pass

    def init_model(self, model_architecture_config):
        """
            Loads model from config file.
        """
        layers = []
        for layer_config in model_architecture_config["layers"]:
            parameters = {key: value for key, value in list(layer_config.items())[1:]}
            if layer_config["type"] == "gcn_conv":
                gcn_conv_layer = GCNConv(**parameters)
                layers.append(gcn_conv_layer)
                pass
            else:
                layer_type = getattr(torch.nn, layer_config["type"])
                layer = layer_type(**parameters)
                layers.append(layer)
            self.layer_types.append(layer_config["type"])
        self.layers = layers

    def run(self, features, graph):
        hs = features
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'gcn_conv':
                hs = layer(hs, graph)
            else:
                hs = layer(hs)
        return hs

