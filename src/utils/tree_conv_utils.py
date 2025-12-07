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
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class TreeConvolutionError(Exception):
    pass


def _is_leaf(x, left_child, right_child):
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None

    if has_left != has_right:
        raise TreeConvolutionError(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left


def _flatten(root, transformer, left_child, right_child):
    """ turns a tree into a flattened vector, preorder """

    if not callable(transformer):
        raise TreeConvolutionError(
            "Transformer must be a function mapping a tree node to a vector"
        )

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    accum = []

    def recurse(x):
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return

        accum.append(transformer(x))
        recurse(left_child(x))
        recurse(right_child(x))

    recurse(root)

    try:
        accum = [np.zeros(accum[0].shape)] + accum
    except:
        raise TreeConvolutionError(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )

    return np.array(accum)


def _preorder_indexes(root, left_child, right_child, idx=1):
    """ transforms a tree into a tree of preorder indexes """

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )

    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree):
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree

    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx + 1)

    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)


def _tree_conv_indexes(root, left_child, right_child):
    """
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root):
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            yield [my_id, left_id, right_id]

            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]

    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)


def _pad_and_combine(x):
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise TreeConvolutionError(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )

    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim

    max_first_dim = max(arr.shape[0] for arr in x)

    vecs = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)


def precompute_left_deep_tree_conv_index(max_leaves: int, device):
    #TODO: These still need padding, use max_size in some way to determine max size of the indexes in batch.
    # Then just fill in the tensor from lookup in the pre-allocated pytorch tensor
    # Current idea is good but doesn't work as indexes are precomputed, so instead
    # Pre-alloc should be size n_plans, max_size, 1
    """
    In left-deep trees its trivial precompute the indexes based on the number of leaves
    :param max_leaves:
    :param device:
    :return:
    """
    leaves_to_idx = {}
    for i in range(2, max_leaves):
        tree_conv_idx = left_deep_tree_conv_index(i)
        leaves_to_idx[i] = torch.tensor(tree_conv_idx.flatten().reshape(-1, 1), device=device)
    return leaves_to_idx


def left_deep_tree_conv_index(n_leaves: int):
    n_internal = n_leaves - 1
    total_nodes = n_internal + n_leaves
    triples = []

    # Internal nodes
    for i in range(1, n_internal + 1):
        triples.append([i, i + 1, total_nodes - (i - 1)])

    # Leaves
    for i in range(n_internal + 1, total_nodes + 1):
        triples.append([i, 0, 0])

    return np.array(triples)


def prepare_trees(trees, transformer, left_child, right_child):
    flat_trees = [_flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = _pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # flat trees is now batch x max tree nodes x channels
    flat_trees = flat_trees.transpose(1, 2)

    indexes = [_tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = _pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    return (flat_trees, indexes)

def build_batched_flat_trees_and_indexes(
    join_orders: List[List[int]],
    features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """
    Build the same outputs as `prepare_trees(trees, transformer, left_child, right_child)`
    but directly, for left-deep trees generated from `join_orders` and `features`.

    Args:
        join_orders: list of join orders (each a list of relation indices into `features`)
        features: tensor of shape [n_rels, channels] (requires_grad if backbone is trainable)

    Returns:
        flat_trees: Tensor (batch, channels, max_nodes_plus_one)
        indexes: LongTensor (batch, max_triplets, 1)
    """

    device = features.device
    dtype = features.dtype
    batch_node_feats = []   # list of tensors (n_nodes_plus1, channels)
    batch_index_vecs = []   # list of tensors (n_nodes*3, 1) -- flattened triplets

    # Helper: build a lightweight index-only tree structure for a left-deep join order
    # We'll represent nodes by dicts with integers, but the features we append are torch tensors.
    for join_order in join_orders:
        if len(join_order) < 2:
            raise ValueError("Each join_order must have at least two relations.")

        # Build a left-deep tree structure using indices (integers):
        # Leaves are stored as ('leaf', rel_idx)
        # Internals as ('node', left_node, right_node)
        # We'll create the tree iteratively to avoid recursion on Python objects with tensors.
        # Start with base tree combining first two leaves
        # Represent nodes as tuples: ('leaf', rel_idx) or ('node', left_obj, right_obj)
        # These are small Python objects used only to compute traversal order and indexes.
        left_obj = ('leaf', join_order[0])
        right_obj = ('leaf', join_order[1])
        root = ('node', left_obj, right_obj)

        for rel in join_order[2:]:
            root = ('node', root, ('leaf', rel))

        # Now compute preorder traversal and map nodes to consecutive preorder indices starting at 1
        preorder_nodes = []  # list of node objects in preorder
        node_to_idx = {}     # id(node) -> preorder index (1-based)

        def preorder_traverse(node):
            preorder_nodes.append(node)
            if node[0] == 'node':
                preorder_traverse(node[1])
                preorder_traverse(node[2])

        preorder_traverse(root)

        # assign indices
        for i, node in enumerate(preorder_nodes, start=1):
            node_to_idx[id(node)] = i

        # Build the feature list in the same preorder used in `_flatten`,
        # then prepend the zero vector as prepare_trees does.
        node_feats = []
        for node in preorder_nodes:
            if node[0] == 'leaf':
                rel_idx = node[1]
                node_feats.append(features[rel_idx])   # tensor view, keeps grad
            else:  # internal node
                node_feats.append(torch.zeros_like(features[0]))

        # prepend the zero-vector as in _flatten
        zero_vec = torch.zeros_like(features[0])
        node_feats = [zero_vec] + node_feats  # list of tensors shape (channels,)
        # stack to (nodes_plus_one, channels)
        node_feats_tensor = torch.stack(node_feats, dim=0).to(device=device, dtype=dtype)
        batch_node_feats.append(node_feats_tensor)

        # Now build the index triplets (my_id, left_id, right_id) for each preorder node
        # Note: the indices used in prepare_trees's _tree_conv_indexes start at 1 (because of the prepended zero)
        # Our node_to_idx maps preorder indices starting at 1 for the actual nodes; since we prepended a zero at index 0,
        # the mapping into flattened array is already compatible: the first element in flattened array is the zero vector (index 0),
        # and our nodes occupy indices 1..n_nodes as expected.
        triplet_list = []
        for node in preorder_nodes:
            if node[0] == 'leaf':
                my_id = node_to_idx[id(node)]
                triplet_list.extend([my_id, 0, 0])
            else:
                my_id = node_to_idx[id(node)]
                left_node = node[1]
                right_node = node[2]
                left_id = node_to_idx[id(left_node)]
                right_id = node_to_idx[id(right_node)]
                triplet_list.extend([my_id, left_id, right_id])

        # convert to tensor shaped (n_nodes*3, 1)
        triplets_tensor = torch.tensor(triplet_list, dtype=torch.long, device=device).view(-1, 1)
        batch_index_vecs.append(triplets_tensor)

    # Pad node feature tensors to same first-dim length
    max_nodes = max(t.shape[0] for t in batch_node_feats)
    channels = features.shape[1]
    padded_feats = []
    for t in batch_node_feats:
        if t.shape[0] < max_nodes:
            pad = torch.zeros((max_nodes - t.shape[0], channels), dtype=dtype, device=device)
            padded = torch.cat([t, pad], dim=0)
        else:
            padded = t
        padded_feats.append(padded.unsqueeze(0))  # (1, max_nodes, channels)

    # stacked -> (batch, max_nodes, channels)
    stacked_feats = torch.cat(padded_feats, dim=0)
    # transpose to (batch, channels, max_nodes) to match prepare_trees output after transpose
    flat_trees = stacked_feats.transpose(1, 2).contiguous()

    # Pad index vectors to same length
    max_triplets = max(t.shape[0] for t in batch_index_vecs)
    padded_idxs = []
    for idxv in batch_index_vecs:
        if idxv.shape[0] < max_triplets:
            pad = torch.zeros((max_triplets - idxv.shape[0], 1), dtype=torch.long, device=device)
            padded = torch.cat([idxv, pad], dim=0)
        else:
            padded = idxv
        padded_idxs.append(padded.unsqueeze(0))  # (1, max_triplets, 1)

    indexes = torch.cat(padded_idxs, dim=0)  # (batch, max_triplets, 1)

    return flat_trees, indexes

def expand_join_orders(join_orders_batched, n_nodes_batched, device):
    total_join_orders = 0
    max_nodes_in_batch = 0
    max_triplets_in_batch = 0

    expanded_join_orders = []
    for join_orders, n_nodes_join_order in zip(join_orders_batched, n_nodes_batched):
        total_join_orders += len(join_orders)
        node_counts = [2 * len(jo) - 1 for jo in join_orders]
        # We need to prepend zero
        max_nodes = max(node_counts) + 1
        max_triplets = max(node_counts) * 3
        if max_nodes > max_nodes_in_batch:
            max_nodes_in_batch = max_nodes
            max_triplets_in_batch = max_triplets

        # Join order prepended with n_nodes - 1 intermediate result indices (zero tensor) and 1 zero tensor
        # Then padded with zero tensor until max_nodes * 2 size
        join_order_tensors = []
        for join_order in join_orders:
            padded_order = ([n_nodes_join_order] * len(join_order)) + join_order
            padded_order += [n_nodes_join_order] * (max_nodes - len(padded_order))
            join_order_tensors.append(
                torch.tensor(padded_order, device=device)
            )
        expanded_join_orders.append(join_order_tensors)
    return expanded_join_orders, total_join_orders, max_nodes_in_batch, max_triplets_in_batch

def build_trees_and_indexes(join_orders_batched, features_batch, precomputed_indexes):
    device = features_batch[0].device
    n_nodes_batched = [feature.shape[0] for feature in features_batch]

    # Pre-compute and expand join orders
    expanded_join_orders, total_join_orders, max_nodes_in_batch, max_triplets_in_batch\
        = expand_join_orders(join_orders_batched, n_nodes_batched, device)
    trees = build_trees(expanded_join_orders, total_join_orders, features_batch, max_nodes_in_batch)
    indexes = get_tree_conv_indexes(join_orders_batched, precomputed_indexes, device)
    return trees, indexes

def build_trees(expanded_join_orders, total_join_orders, features_batch, max_nodes_in_batch):
    device = features_batch[0].device
    channels = features_batch[0].shape[1]
    dtype = features_batch[0].dtype
    # Pre-compute zero vector once
    zero_vec = torch.zeros(channels, dtype=dtype, device=device)
    feature_batch_with_zero_tensor = []

    for features in features_batch:
        feature_batch_with_zero_tensor.append(torch.cat([features, zero_vec.unsqueeze(0)], dim=0))

    # Pre-allocate output tensors
    flat_trees = torch.zeros((total_join_orders, channels, max_nodes_in_batch), dtype=dtype, device=device)
    # indexes = torch.zeros((total_join_orders, max_triplets_in_batch, 1), dtype=torch.long, device=device)

    current_join_order_index = 0
    for i, join_orders in enumerate(expanded_join_orders):
        batch_n_join_orders = len(join_orders)
        idx = torch.stack(join_orders)
        # Expand x for batch dimension to match number of join orders
        # (batch, n_nodes, channels)
        x_expanded = feature_batch_with_zero_tensor[i].unsqueeze(0).expand(batch_n_join_orders, -1, -1)
        # Expand idx to match channel dimension
        # (batch, n_nodes, channels)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, channels)
        # Gather along the node dimension (dim=1)
        # (batch, n_nodes, n_channels)
        flattened = torch.gather(x_expanded, 1, idx_expanded)
        # (batch, channels, n_nodes)
        flattened_transposed = flattened.transpose(1, 2)
        # Assign to output tensor with join order batch indexes
        flat_trees[current_join_order_index:current_join_order_index+batch_n_join_orders] = flattened_transposed
        current_join_order_index += batch_n_join_orders
    return flat_trees

def get_tree_conv_indexes(join_orders_batched, precomputed_indexes, device):
    max_sizes_join_order = [max([precomputed_indexes[len(join_order)].shape[0] for join_order in join_orders]) for join_orders in join_orders_batched]
    batched_indexes = []
    for j in range(len(join_orders_batched)):
        indexes = torch.zeros((len(join_orders_batched[j]), max_sizes_join_order[j], 1), dtype=torch.long, device=device)

        for i, join_orders in enumerate(join_orders_batched[j]):
            t = precomputed_indexes[len(join_orders)]
            actual_size = t.shape[0]
            indexes[i, :actual_size] = t
        batched_indexes.append(indexes)
    return batched_indexes