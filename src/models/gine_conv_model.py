import inspect
import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch
import torch_geometric
from torch_geometric.nn import GINEConv, Sequential

from src.models.model_layers.directional_gine_conv import DirectionalGINEConv
from src.models.model_layers.triple_gine_conv import TripleGineConv
from src.models.model_layers.triple_gine_conv_moe import TripleGineConvMoE
from src.models.model_layers.triple_pattern_pool import TriplePatternPooling


class GINEConvModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model: Optional[Sequential] = None
        self.heads = torch.nn.ModuleList()
        self.head_types = []
        self.supported_mlp_layers = ['Linear', 'Dropout', 'ReLU', 'Softplus', 'LayerNorm']
        self.supported_pooling = ['SumAggregation', 'MeanAggregation', 'MaxAggregation', 'TriplePatternPooling']
        self.supported_gnn_layers = ['TripleGINEConv', 'GINEConv', 'TripleGINEConvMoE', 'DirectionalGINEConv']
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
                'TripleGINEConvMoE': TripleGineConvMoE,
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
            embedding_layers[layer_id] = (mlp_class(**mlp_params), 'x -> x')
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

    def get_load_balancing_loss(self, device, writer=None, epoch=None) -> torch.Tensor | None:
        """Computes the auxiliary loss across all MoE layers."""
        aux_loss = torch.tensor(0, device=device, dtype=torch.float32)
        moe_layers = 0

        has_moe_layer = False
        for module in self.modules():
            if isinstance(module, TripleGineConvMoE):
                has_moe_layer = True
                routing_prob = module.get_current_routing_probs()

                # if writer and epoch and hasattr(writer, 'add_histogram'):
                #     # Log the full distribution of routing scores before softmax to see raw router confidence
                #     writer.add_histogram('Routing/Expert_Score_Distribution', routing_prob, epoch)

                if writer and epoch is not None:
                    # Calculate the mean probability for each expert across the batch
                    mean_probs = torch.mean(routing_prob, dim=0)

                    # Create a dictionary mapping expert IDs to their mean probability
                    prob_dict = {f'Expert_{i}': p.item() for i, p in enumerate(mean_probs)}

                    # Use add_scalars (plural) to group them in one chart
                    if hasattr(writer, 'add_scalars'):
                        writer.add_scalars('Routing/Expert_Distribution', prob_dict, epoch)
                    else:
                        # Fallback: Many loggers group scalars automatically if they share the same folder name
                        for label, val in prob_dict.items():
                            writer.log({f'Routing_Distribution/{label}': val, 'step': epoch})

                num_nodes, num_experts = routing_prob.shape
                # Sum probabilities over all nodes for each expert
                # sum_P shape: (num_experts,)
                sum_probability_nodes = routing_prob.sum(dim=0)

                # Compute routing loss https://arxiv.org/abs/2511.04008
                layer_loss = (num_experts / (num_nodes ** 2)) * torch.sum(sum_probability_nodes ** 2)
                aux_loss += layer_loss
                moe_layers += 1

        if not has_moe_layer:
            return None
        if writer and epoch:
            writer.add_scalar(f'Routing/Auxiliary loss', (aux_loss / max(1, moe_layers)).detach().item(), epoch)
        # Average the loss across all MoE layers in the network
        return aux_loss / max(1, moe_layers)

    def register_moe_gradient_hooks(self, writer, get_global_step_fn):
        """
        Registers backward hooks on MoE layers to log gradient norms to TensorBoard.

        Args:
            writer: SummaryWriter instance.
            get_global_step_fn: A callable returning the current global training step.
        """
        moe_count = 0
        for name, module in self.named_modules():
            if isinstance(module, TripleGineConvMoE):
                moe_layer_id = f"MoE_Layer_{moe_count}"

                # Closure to capture the correct parameter name and layer ID
                def make_hook(param_name):
                    def hook(grad):
                        if grad is not None and writer is not None:
                            norm = torch.norm(grad, p=2).item()
                            step = get_global_step_fn()
                            writer.add_scalar(f'Gradients/{moe_layer_id}_{param_name}_L2', norm, step)

                    return hook

                # Register hooks on the specific parameters
                if hasattr(module, 'router') and hasattr(module.router, 'weight'):
                    module.router.weight.register_hook(make_hook('Router_Weight'))

                if hasattr(module, 'experts') and hasattr(module.experts, 'expert_L'):
                    module.experts.expert_L.register_hook(make_hook('Expert_L_Factor'))

                moe_count += 1

        if self.verbose > 0:
            print(f"Registered gradient hooks for {moe_count} MoE layers.")

    def serialize_model(self, model_dir):
        embedding_state_dict = self.embedding_model.state_dict()
        torch.save(embedding_state_dict, os.path.join(model_dir, "embedding_model.pt"))
        heads_state_dicts = [head.state_dict() for head in self.heads]
        for state_dict, head_type in zip(heads_state_dicts, self.head_types):
            torch.save(state_dict, os.path.join(model_dir, "head_{}.pt".format(head_type)))

    def freeze_model(self):
        """
        Freezes the entire model (embedding backbone and all heads)
        and sets it to evaluation mode.
        """
        # 1. Turn off gradients for all parameters recursively
        for param in self.parameters():
            param.requires_grad = False

        # 2. Set the entire model to evaluation mode recursively
        self.eval()

        if self.verbose > 0:
            print("Entire GINEConvModel successfully frozen and set to eval mode.")

    @staticmethod
    def __filter_parameters(params, params_to_exclude):
        return {k: v for k, v in params.items() if k not in params_to_exclude}

