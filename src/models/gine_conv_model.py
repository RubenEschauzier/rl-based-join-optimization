import inspect
import os
import warnings
from collections import OrderedDict

import torch
import torch_geometric
from torch.nn import ReLU, Linear
from torch_geometric.nn import GINEConv

from src.models.model_layers.directional_gine_conv import DirectionalGINEConv
from src.models.model_layers.triple_gine_conv import TripleGineConv
from src.models.model_layers.triple_pattern_pool import TriplePatternPooling


class GINEConvModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model = None
        self.heads = torch.nn.ModuleList()
        self.head_types = []
        self.supported_mlp_layers = ['Linear', 'Dropout', 'ReLU', 'Softplus', 'LayerNorm']
        self.supported_pooling = ['SumAggregation', 'MeanAggregation', 'MaxAggregation', 'TriplePatternPooling']
        self.supported_gnn_layers = ['TripleGINEConv', 'GINEConv', 'DirectionalGINEConv']
        self.supported_normalization = ['PairNorm', 'GraphNorm']
        self.verbose = 0

    def init_model(self, model_architecture_config):
        """
            Loads model from config file.
        """
        embedding_config = model_architecture_config['embedding']
        embedding_layers = self.init_layers(embedding_config)
        self.embedding_model = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr, batch',
            embedding_layers
        )
        head_configs = model_architecture_config['heads']
        for head_config in head_configs:
            estimation_type = head_config['estimation_type']
            head_config = self.__filter_parameters(head_config, ['estimation_type'])
            # head_config = {key: value for key, value in head_config.items() if key != 'estimation_type'}
            head_layers = self.init_layers(head_config)

            # Convert to a type of layer that includes input and output arguments to allow for use in a
            # pytorch geometric sequential model.
            head_layers_geometric = OrderedDict()
            last_output = 'x'
            for name, head_layer in head_layers.items():
                if isinstance(head_layer, tuple):
                    last_output = head_layer[1].split('->')[1].strip()
                    head_layers_geometric[name] = head_layer
                else:
                    geometric_head_layer = (head_layer, '{} -> {}'.format(last_output, last_output))
                    head_layers_geometric[name] = geometric_head_layer
                    last_output = 'x'

            # First layer defines the input args
            input_args = list(head_layers_geometric.values())[0][1].split('->')[0].strip()
            self.heads.append(torch_geometric.nn.Sequential(input_args, head_layers_geometric))
            self.head_types.append(estimation_type)

        if self.verbose > 0:
            print("Embedding layers:")
            print(self.embedding_model)
            print("Estimation heads:")
            print(self.heads)
            print("Head types:")
            print(self.head_types)

    def init_layers(self, config):
        embedding_layers = OrderedDict()
        for layer_config in config['layers']:
            self.__build_layer(embedding_layers, layer_config)
        return embedding_layers

    def __build_layer(self, embedding_layers, layer_config):
        layer_type = layer_config['type']
        layer_id = layer_config['id']

        if layer_id in embedding_layers:
            warnings.warn(
                f"Layer id '{layer_id}' already exists in embedding_layers — it will be overwritten.",
                stacklevel=2
            )

        if layer_type in self.supported_gnn_layers:
            nn_layers = OrderedDict()
            for layer in layer_config['nn']:
                layer_id = layer['id']
                nn_class = getattr(torch.nn, layer['type'])
                layer_params = self.__filter_parameters(layer, ['type', 'id'])
                nn_layers[layer_id] = nn_class(**layer_params)

            nn = torch.nn.Sequential(nn_layers)
            gine_params = self.__filter_parameters(layer_config, ['type', 'nn', 'id'])
            layer_class_map = {
                'GINEConv': GINEConv,
                'TripleGINEConv': TripleGineConv,
                'DirectionalGINEConv': DirectionalGINEConv
            }
            conv_class = layer_class_map.get(layer_type)
            if conv_class is None:
                raise ValueError(f'Unknown GNN layer type: {layer_type}')
            embedding_layers[layer_id] = (conv_class(nn, **gine_params), 'x, edge_index, edge_attr -> x')

        elif layer_type in self.supported_pooling:
            pool_params = self.__filter_parameters(layer_config, ['type', 'id'])
            if layer_type == 'TriplePatternPooling':
                embedding_layers[layer_id] = (TriplePatternPooling(), 'x, edge_index, batch -> x, edge_batch')
            else:
                pool_class = getattr(torch_geometric.nn.aggr.basic, layer_type)
                embedding_layers[layer_id] = (pool_class(**pool_params), 'x, batch -> x')

        elif layer_type in self.supported_mlp_layers:
            mlp_class = getattr(torch.nn, layer_type)
            mlp_params = self.__filter_parameters(layer_config, ['type', 'id'])
            embedding_layers[layer_id] = mlp_class(**mlp_params)
        elif layer_type in self.supported_normalization:
            norm_class = getattr(torch_geometric.nn, layer_type)
            norm_params = self.__filter_parameters(layer_config, ['type', 'id'])
            embedding_layers[layer_id] = (norm_class(**norm_params), 'x, batch -> x')
        else:
            raise NotImplementedError(f'Unsupported layer type: {layer_type}')

    def forward(self, x, edge_index, edge_attr, batch):
        embedded = self.embedding_model.forward(x, edge_index, edge_attr, batch)
        outputs = []
        head_input_map = {
            'x': embedded,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'batch': batch
        }
        for head_type, head_model in zip(self.head_types, self.heads):
            head_args = inspect.signature(head_model.forward).parameters
            filtered_input = {arg: head_input_map[arg] for arg in head_args}
            output = head_model(**filtered_input)
            outputs.append({'output_type': head_type, 'output': output})
        return outputs

    def serialize_model(self, model_dir):
        embedding_state_dict = self.embedding_model.state_dict()
        torch.save(embedding_state_dict, os.path.join(model_dir, "embedding_model.pt"))
        heads_state_dicts = [head.state_dict() for head in self.heads]
        for state_dict, head_type in zip(heads_state_dicts, self.head_types):
            torch.save(state_dict, os.path.join(model_dir, "head_{}.pt".format(head_type)))

    @staticmethod
    def __filter_parameters(params, params_to_exclude):
        return {k: v for k, v in params.items() if k not in params_to_exclude}

if __name__ == "__main__":
    from collections import OrderedDict
    from torch_geometric.nn import Sequential, GCNConv, global_mean_pool

    layers = OrderedDict([
        ('conv1_tests', (GCNConv(5, 64), 'x, edge_index -> x')),
        ('relu1', ReLU()),
        ('conv2', (GCNConv(64, 64), 'x, edge_index -> x')),
        ('pool', (global_mean_pool, 'x, batch -> x')),
        ('lin', Linear(64, 4)),
    ])

    model = Sequential('x, edge_index, batch', layers)
    print(model.state_dict())
