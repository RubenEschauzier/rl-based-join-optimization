import numpy as np
import torch


def precompute_left_deep_tree_conv_index(max_leaves: int, device):
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

def precompute_left_deep_trees_placeholders(max_leaves: int, emb_dim, precomputed_indexes, device):
    """
    Precomputes tree structures used for 1d convolution in tree cnn. These tree structures must be right-padded with
    zero tensors of emb_dim size tto batch multiple plan sizes together
    :param max_leaves: Max join entries to precompute trees for
    :param emb_dim: The dimension of the placeholder embeddings
    :param precomputed_indexes: Precomputed index tensors (needed for tcnn)
    :param device:
    :return: Dict mapping number of join entries to the tree structure dict. With elements 'trees',
    'node_mapping'. node_mapping maps the index in the join order to the index of the placeholder tensor.
    """
    leaves_to_tree = {}
    for i in range(2, max_leaves):
        node_ids = torch.arange(i, device=device).unsqueeze(1)
        embedded = node_ids.repeat(1, emb_dim)
        embedded = embedded.float()

        # Placeholder join order
        join_order = [j for j in range(i)]
        node_count = 2 * len(join_order)


        prepared_trees, prepared_indexes = build_trees_and_indexes(
            [[[j for j in range(i)]]], [embedded],
            precomputed_indexes
        )
        prepared_trees = prepared_trees.transpose(1, 2)

        jo_idx_to_tree_idx = { j: node_count - len(join_order) + j for j in range(len(join_order))}

        for key, value in jo_idx_to_tree_idx.items():
            assert join_order[key] == prepared_trees[0][value][0]


        leaves_to_tree[i] = {
            'trees': prepared_trees,
            'node_mapping': jo_idx_to_tree_idx
        }

    return leaves_to_tree


def fill_placeholder_trees(plans, embedded, precomputed_trees, precomputed_indexes, device):
    indexes = get_tree_conv_indexes([[plan[0] for plan in plans]], precomputed_indexes, device)

    max_plan_size = max([len(plan[0]) for plan in plans])
    max_size_placeholder_tree = precomputed_trees[max_plan_size]['trees']

    filled_trees = []
    for plan, cost in plans:
        precomputed_tree_dict = precomputed_trees[len(plan)]
        placeholder_tree, mapping = precomputed_tree_dict['trees'], precomputed_tree_dict['node_mapping']
        filled_tree = torch.zeros_like(max_size_placeholder_tree)
        for i in range(len(plan)):
            emb_idx = plan[i]
            tree_idx = mapping[i]
            filled_tree[0][tree_idx] = embedded[emb_idx]
        filled_trees.append(filled_tree)
    stacked_trees = torch.stack(filled_trees).squeeze()
    stacked_trees = stacked_trees.transpose(2, 1)
    return stacked_trees, indexes

    pass

def build_trees_and_indexes(join_orders_batched, features_batch, precomputed_indexes):
    device = features_batch[0].device
    n_nodes_batched = [feature.shape[0] for feature in features_batch]

    # Pre-compute and expand join orders
    expanded_join_orders, total_join_orders, max_nodes_in_batch, max_triplets_in_batch\
        = expand_join_orders(join_orders_batched, n_nodes_batched, device)
    trees = build_trees(expanded_join_orders, total_join_orders, features_batch, max_nodes_in_batch)
    indexes = get_tree_conv_indexes(join_orders_batched, precomputed_indexes, device)
    return trees, indexes

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