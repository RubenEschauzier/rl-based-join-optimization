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


#TODO: Change this binary tree conv to take a batch of features (b, n, feature_dim) and batch of orders (b, 1(list))
# Batchwise converts this to a convable index and does batchwise conv
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
        device = self.weights.weight.device

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels), device=device).unsqueeze(2)
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

def build_t_cnn_trees(join_orders: list[list[int]], features: torch.Tensor):
    # Input is [full_plan_order, smallest_sub_order, larger.. , ]
    if len(join_orders) == 1:
        smallest_sub_order = len(join_orders[0])
    else:
        smallest_sub_order = len(join_orders[1])
    full_order = join_orders[0]
    start_tuple = (
        torch.zeros_like(features[0]),
        (features[full_order[0]],),
        (features[full_order[1]],),
    )
    sub_plan_trees = []
    tuples_in_order = [start_tuple]
    # First tuple encodes two triple patterns joined together, every new tuple encodes a single triple pattern
    if smallest_sub_order <= 2:
        sub_plan_trees.append(tuples_in_order[-1])
    for i, additional_relation in enumerate(full_order[2:]):
        # added to the plan
        new_tuple = (
            # Zero vector to represent join relation
            torch.zeros_like(features[0]),
            # Previous tuple is the left node of new 'root'
            tuples_in_order[i],
            # Right tuple is the feature of additional relation joined
            (features[additional_relation],)
        )
        tuples_in_order.append(new_tuple)
        if smallest_sub_order <= i + 3:
            sub_plan_trees.append(tuples_in_order[-1])

    return sub_plan_trees

def build_t_cnn_tree_from_order(join_order: list[int], features: torch.Tensor):
    #TODO: This can and needs to be sped up
    if len(join_order) < 2:
        raise ValueError("Insufficient joins selected, can't apply tree convolution")
    start_tuple = (
        torch.zeros_like(features[0]),
        (features[join_order[0]],),
        (features[join_order[1]],),
    )
    tuples_in_order = [start_tuple]
    for i, additional_relation in enumerate(join_order[2:]):
        if len(join_order) == 2:
            raise ValueError("THIS SHOULDN'T BE ITERATING")
        new_tuple = (
            # Zero vector to represent join relation
            torch.zeros_like(features[0]),
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
