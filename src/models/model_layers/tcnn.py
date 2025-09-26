# < begin copyright >
# Copyright Ryan Marcus 2019
#
# This file is part of TreeConvolution.
#
# TreeConvolution is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TreeConvolution is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TreeConvolution.  If not, see <http://www.gnu.org/licenses/>.
#
# < end copyright >
import numpy as np
import torch
import torch.nn as nn

from src.utils.tree_conv_utils import prepare_trees


class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data):
        trees, idxes = flat_data
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes)


class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]), x[1])


class TreeLayerNorm(nn.Module):
    def forward(self, x):
        data, idxes = x
        mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        normd = (data - mean) / (std + 0.00001)
        return (normd, idxes)


class DynamicPooling(nn.Module):
    def forward(self, x):
        return torch.max(x[0], dim=2).values


def build_t_cnn_tree_from_order(join_order: list[int], features: torch.Tensor):
    if len(join_order) < 2:
        raise ValueError("Insufficient joins selected, can't apply tree convolution")
    start_tuple = (
        torch.mean(features[join_order[:2]], dim=0),
        (features[join_order[0]],),
        (features[join_order[1]],),
    )
    tuples_in_order = [start_tuple]
    for i, additional_relation in enumerate(join_order[2:]):
        new_tuple = (
            # Average the previous 'root' node representation and new relation
            torch.mean(torch.stack([tuples_in_order[i][0], features[additional_relation]],dim=0),dim=0),
            # Previous tuple is the left node of new 'root'
            tuples_in_order[i],
            # Right tuple is the feature of additional relation joined
            (features[additional_relation],)
        )
        tuples_in_order.append(new_tuple)
    return tuples_in_order[-1]

# function to extract the left child of a node
def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[1]

# function to extract the right child of node
def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]

# function to transform a node into a (feature) vector,
# should be a numpy array.
def transformer(x):
    return np.array(x[0])

if __name__ == "__main__":
    # Example: left-deep join order for your tree structure
    # Tree:   8
    #       7   4
    #     6   3
    #   5   2
    # 0   1

    join_order = [0, 1, 2, 3, 4]  # Bottom-up, left-to-right traversal

    # Create example features tensor (7 relations, 2 features each)
    features = torch.tensor([
        [1.0, 2.0],  # relation 0
        [3.0, 4.0],  # relation 1
        [5.0, 6.0],  # relation 2
        [7.0, 8.0],  # relation 3
        [9.0, 10.0],  # relation 4
    ])

    # Convert to nested tuple format
    tree_structure_1 = build_t_cnn_tree_from_order(join_order, features)

    features = torch.tensor([
        [5.0,3.0],
        [2.0,6.0],
        [2.0,9.0]
    ])
    tree_structure_2 = build_t_cnn_tree_from_order([0,1,2], features)
    trees = [tree_structure_2]
    # this call to `prepare_trees` will create the correct input for
    # a `tcnn.BinaryTreeConv` operator.
    prepared_trees = prepare_trees(trees, transformer, left_child, right_child)
    print(prepared_trees)
