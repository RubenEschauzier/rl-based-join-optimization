import numpy as np
import torch
from typing_extensions import deprecated


def precompute_left_deep_tree_conv_index(max_leaves: int):
    """
    In left-deep trees its trivial precompute the indexes based on the number of leaves
    :param max_leaves:
    :param device:
    :return:
    """
    leaves_to_idx = {}
    for i in range(2, max_leaves):
        tree_conv_idx = left_deep_tree_conv_index(i)
        leaves_to_idx[i] = tree_conv_idx.flatten().reshape(-1, 1)
    return leaves_to_idx


def precompute_left_deep_tree_node_mask(max_leaves: int):
    n_leaves_to_mask = {}
    for i in range(2, max_leaves):
        # i - 1 intermediate results + zero_vector in plan
        n_nodes = 2 * i
        n_leaves_to_mask[i] = np.zeros(n_nodes, dtype="bool")
        # Mask out the zero vector
        n_leaves_to_mask[i][0] = True
    return n_leaves_to_mask


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

        jo_idx_to_tree_idx = {j: node_count - len(join_order) + j for j in range(len(join_order))}

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


@deprecated("Use build_t_cnn_input instead, which is much faster")
def build_trees_and_indexes(join_orders_batched, features_batch, precomputed_indexes):
    device = features_batch[0].device
    n_nodes_batched = [feature.shape[0] for feature in features_batch]

    # Pre-compute and expand join orders
    expanded_join_orders, total_join_orders, max_nodes_in_batch, _ \
        = expand_join_orders(join_orders_batched, n_nodes_batched, device)
    trees = build_trees(expanded_join_orders, total_join_orders, features_batch, max_nodes_in_batch)
    indexes = get_tree_conv_indexes(join_orders_batched, precomputed_indexes, device)
    return trees, indexes


@deprecated("Use build_t_cnn_input instead, which is much faster")
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


@deprecated("Use build_t_cnn_input instead, which is much faster")
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
        flat_trees[current_join_order_index:current_join_order_index + batch_n_join_orders] = flattened_transposed
        current_join_order_index += batch_n_join_orders
    return flat_trees


@deprecated("Use build_t_cnn_input instead, which is much faster")
def get_tree_conv_indexes(join_orders_batched, precomputed_indexes, device):
    max_sizes_join_order = [max([precomputed_indexes[len(join_order)].shape[0] for join_order in join_orders]) for
                            join_orders in join_orders_batched]
    batched_indexes = []
    for j in range(len(join_orders_batched)):
        indexes = torch.zeros((len(join_orders_batched[j]), max_sizes_join_order[j], 1), dtype=torch.long,
                              device=device)

        for i, join_orders in enumerate(join_orders_batched[j]):
            t = precomputed_indexes[len(join_orders)]
            actual_size = t.shape[0]
            indexes[i, :actual_size] = t
        batched_indexes.append(indexes)
    return batched_indexes


def get_shared_structure(join_orders, n_nodes, precomputed_indexes,
                         precomputed_masks, device):
    """
    Computes or retrieves the structural indices and masks that are
    identical across all ensemble models.
    """
    # Unique key for the topology of the batch
    structure_key = (tuple(tuple(jo) for jo in join_orders), n_nodes)

    # if structure_key in cache:
    #     gather_indices, conv_indexes, masks =  cache[structure_key]
    # else:
    # Compute on MISS
    gather_indices, conv_indexes, masks = _compute_structural_indices(
        join_orders, n_nodes, precomputed_indexes, precomputed_masks
    )
    # cache[structure_key] = (gather_indices, conv_indexes, masks)

    gather_indices = torch.from_numpy(gather_indices).to(device)
    conv_indexes = torch.from_numpy(conv_indexes).to(device)
    masks = torch.from_numpy(masks).to(device)

    return gather_indices, conv_indexes, masks


def apply_features_to_structure_batched(stacked_features, gather_indices):
    """
    Batched hot-path function: maps specific query embeddings onto the shared structure
    for ALL ensembles at once.

    Args:
        stacked_features: (E, n_nodes, channels)
        gather_indices: (num_plans, tree_width)
    Returns:
        (E, num_plans, channels, tree_width)
    """
    batch, _, feature_dim = stacked_features.shape
    n_plans, _ = gather_indices.shape
    device = stacked_features.device
    dtype = stacked_features.dtype

    # 1. Add zero vector for padding -> (batch, n_nodes+1, feature_dim)
    zero_vec = torch.zeros((batch, 1, feature_dim), dtype=dtype, device=device)
    features_padded = torch.cat([stacked_features, zero_vec], dim=1)

    # features_expanded: (batch, n_plans, n_nodes+1, feature_dim)
    features_expanded = features_padded.unsqueeze(1).expand(-1, n_plans, -1, -1)
    # idx_expanded: (batch, n_plans, tree_width, feature_dim)
    idx_expanded = gather_indices.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, feature_dim)

    # flattened_trees: (batch, n_plans, tree_width, feature_dim)
    flattened_trees = torch.gather(features_expanded, 2, idx_expanded)

    # Final shape: (batch, n_plans, feature_dim, tree_width)
    return flattened_trees.transpose(2, 3)

def apply_features_to_structure(features, gather_indices):
    """
    The hot-path function: maps specific query embeddings onto the shared structure.
    """
    device = features.device
    dtype = features.dtype
    num_plans = gather_indices.shape[0]
    channels = features.shape[1]

    # 1. Add zero vector for padding (n_nodes + 1)
    zero_vec = torch.zeros((1, channels), dtype=dtype, device=device)
    features_padded = torch.cat([features, zero_vec], dim=0)

    # 2. Vectorized Gather
    # features_expanded: (num_plans, n_nodes+1, channels)
    features_expanded = features_padded.unsqueeze(0).expand(num_plans, -1, -1)
    # idx_expanded: (num_plans, tree_width, channels)
    idx_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, channels)

    # flattened_trees: (num_plans, tree_width, channels)
    flattened_trees = torch.gather(features_expanded, 1, idx_expanded)

    # Final shape: (num_plans, channels, tree_width)
    return flattened_trees.transpose(1, 2)

def _compute_structural_indices(join_orders, n_nodes, precomputed_indexes, precomputed_masks):
    """
    Internal helper to build indices using fast NumPy vectorization and avoiding GPU communication costs
    """
    lengths = [len(jo) for jo in join_orders]
    max_len = max(lengths)
    max_nodes_in_batch = 2 * max_len

    # Create the base array filled with 'n_nodes' (which points to the zero-vector)
    # Shape: (num_plans, max_nodes_in_batch)
    # Using int64 for indices
    batch_indices = np.full((len(join_orders), max_nodes_in_batch), n_nodes, dtype=np.int64)

    # Fill in the actual join orders
    # Original logic: padded_order = [n_nodes]*len + join_order + [n_nodes]*remainder
    # This means the join_order data starts at index `len(jo)`
    for i, jo in enumerate(join_orders):
        l = len(jo)
        # We copy the join order into the middle of the array
        # The left side (0 to l) is already n_nodes
        # The right side (2*l to end) is already n_nodes
        batch_indices[i, l: l + l] = jo

    # --- 2. Build Conv Indexes ---
    # Logic from original: max size based on precomputed_indexes shapes
    max_conv_size = max([precomputed_indexes[l].shape[0] for l in lengths])

    conv_indexes = np.zeros((len(join_orders), max_conv_size, 1), dtype=np.int64)

    for i, l in enumerate(lengths):
        t = precomputed_indexes[l]
        actual_size = t.shape[0]
        conv_indexes[i, :actual_size] = t
    #TODO: Applied fix here where when the number of max number of joins wasn't aligning with the number of nodes
    # this broke. I think mask should always be size of max size of joins as join number decides
    # representation size

    # target_width = n_nodes * 2
    target_width = max_nodes_in_batch

    # Create batch mask on CPU using Numpy
    batch_mask_np = np.ones((len(join_orders), target_width), dtype=bool)

    # We need the raw mask data available here.
    # Assumption: precomputed_masks is a dict of Tensors.
    # It is faster to have precomputed_masks_np (dict of numpy arrays) for this step.

    for i, jo in enumerate(join_orders):
        l = len(jo)
        # Convert to numpy if it isn't already (ideally convert dict to numpy once in init)
        mask_data = precomputed_masks[l]
        batch_mask_np[i, :mask_data.shape[0]] = mask_data

    return batch_indices, conv_indexes, batch_mask_np
