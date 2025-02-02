import torch
import torch_geometric
from torch_geometric.nn import GINEConv

from src.models.model_layers.directional_gine_conv import DirectionalGINEConv
from src.models.model_layers.triple_gine_conv import TripleGineConv
from src.models.model_layers.triple_pattern_pool import TriplePatternPooling
from src.utils.training_utils.utils import register_debugging_hooks


class GINEConvModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.supported_mlp_layers = ['Linear', 'Dropout', 'ReLU', 'Softplus']
        self.supported_pooling = ['SumAggregation', 'MeanAggregation', 'MaxAggregation', 'TriplePatternPooling']
        self.supported_gnn_layers = ['TripleGINEConv', 'GINEConv', 'DirectionalGINEConv']

    def init_model(self, model_architecture_config):
        """
            Loads model from config file.
        """
        layers = []
        for layer_config in model_architecture_config['layers']:
            if layer_config['type'] in self.supported_gnn_layers:
                # Build nn used in gine_conv from config
                nn_config = layer_config['nn']
                nn_layers = []
                for layer in nn_config:
                    parameters = {key: value for key, value in list(layer.items())[1:]}
                    layer_type = getattr(torch.nn, layer['type'])
                    nn_layers.append(layer_type(**parameters))
                nn = torch.nn.Sequential(*nn_layers)
                gine_parameters = {key: value for key, value in list(layer_config.items())[2:]}
                if layer_config['type'] == 'GINEConv':
                    gine_conv_layer = GINEConv(nn, **gine_parameters)
                    layers.append((gine_conv_layer, 'x, edge_index, edge_attr -> x'))
                elif layer_config['type'] == 'TripleGINEConv':
                    triple_gine_conv_layer = TripleGineConv(nn, **gine_parameters)
                    layers.append((triple_gine_conv_layer, 'x, edge_index, edge_attr -> x'))
                elif layer_config['type'] == 'DirectionalGINEConv':
                    directional_gine_conv_layer = DirectionalGINEConv(nn, **gine_parameters)
                    layers.append((directional_gine_conv_layer, 'x, edge_index, edge_attr -> x'))
                else:
                    raise ValueError('Unknown layer type')
            elif layer_config['type'] == 'TripleGINEConv':
                pass

            elif layer_config['type'] in self.supported_pooling:
                if layer_config['type'] == 'TriplePatternPooling':
                    layers.append(
                        (TriplePatternPooling(512, 512), 'x, edge_index, edge_attr, batch -> x')
                    )
                    continue
                layer_type = getattr(torch_geometric.nn.aggr.basic, layer_config['type'])
                pooling_parameters = {key: value for key, value in list(layer_config.items())[1:]}
                layers.append((layer_type(**pooling_parameters), 'x, batch -> x'))
                pass

            elif layer_config['type'] in self.supported_mlp_layers:
                layer_type = getattr(torch.nn, layer_config['type'])
                mlp_parameters = {key: value for key, value in list(layer_config.items())[1:]}
                layers.append(layer_type(**mlp_parameters))
                pass
            else:
                raise NotImplementedError
        self.model = torch_geometric.nn.Sequential('x, edge_index, edge_attr, batch', layers)
        print("Model:")
        print(self.model)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.model.forward(x, edge_index, edge_attr, batch)
