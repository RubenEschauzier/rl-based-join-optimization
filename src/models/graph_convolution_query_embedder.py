import torch

from src.models.model_layers.graph_convolution import GCNConv


class GCNConvQueryEmbeddingModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        self.layer_types = []

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

    def forward(self, features, graph, **kwargs):
        hs = features
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'gcn_conv':
                hs = layer(hs, graph)
            # Do no apply dropout during validation
            elif layer_type == 'Dropout' and kwargs['training'] == False:
                continue
            else:
                hs = layer(hs)
        return hs

    def parameters(self):
        return [param for layer in self.layers for param in list(layer.parameters())]

    def devices(self):
        params = self.parameters()
        return [param.device for param in params]

    def to(self, device):
        self.layers = [layer.to(device) for layer in self.layers]
