from functools import lru_cache

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

def get_shared_structure(join_orders, n_nodes, precomputed_indexes,
                         precomputed_masks, device):
    """
    Computes or retrieves the structural indices and masks that are
    identical across all ensemble models.
    """
    gather_indices, conv_indexes, masks = _compute_structural_indices(
        join_orders, n_nodes, precomputed_indexes, precomputed_masks
    )

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

    # Logic from original: max size based on precomputed_indexes shapes
    max_conv_size = max([precomputed_indexes[l].shape[0] for l in lengths])

    conv_indexes = np.zeros((len(join_orders), max_conv_size, 1), dtype=np.int64)

    for i, l in enumerate(lengths):
        t = precomputed_indexes[l]
        actual_size = t.shape[0]
        conv_indexes[i, :actual_size] = t

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

# From here is code for bushy trees
def extract_topology_and_leaves(plan):
    """
    Separates a bushy plan (nested tuples) into a generic topology and a list of leaves.
    E.g., ((0, 1), 2) -> (((None, None), None), [0, 1, 2])
    """
    if isinstance(plan, int):  # It's a leaf
        return None, [plan]

    left_top, left_leaves = extract_topology_and_leaves(plan[0])
    right_top, right_leaves = extract_topology_and_leaves(plan[1])

    return (left_top, right_top), left_leaves + right_leaves


@lru_cache(maxsize=10000)
def get_cached_topology_structure(topology):
    """
    Computes the Tree-CNN indices for a unique topology shape.
    Returns:
        gather_bools: boolean array where True indicates a leaf position,
                      and False indicates an internal node.
        conv_triplets: flattened array of [parent_id, left_id, right_id]
    """

    def _traverse(top, current_id=1):
        if top is None:  # Leaf
            return [True], [[current_id, 0, 0]], current_id

        left_gather, left_trip, max_left = _traverse(top[0], current_id + 1)
        right_gather, right_trip, max_right = _traverse(top[1], max_left + 1)

        # Preorder: Root (False), Left, Right
        gather = [False] + left_gather + right_gather

        # Stride-3 convolution triplets
        root_trip = [[current_id, current_id + 1, max_left + 1]]
        trip = root_trip + left_trip + right_trip

        return gather, trip, max_right

    gather_bools, triplets, _ = _traverse(topology)

    # Flatten triplets for 1D tree convolution
    conv_triplets = np.array(triplets).flatten().reshape(-1, 1)

    return np.array(gather_bools), conv_triplets


def compute_bushy_structural_indices(bushy_plans, n_nodes):
    """
    Computes the gather indices and conv indexes for a batch of bushy plans.

    Args:
        bushy_plans: List of nested tuples, e.g., [((0, 1), 2), ((0, 1), (2, 3))]
        n_nodes: The index pointing to the zero-tensor (padding) in the feature matrix.
    Returns:
        batch_indices: (num_plans, max_nodes)
        conv_indexes: (num_plans, max_conv_size, 1)
        batch_mask: (num_plans, max_nodes)
    """
    num_plans = len(bushy_plans)

    gather_lists = []
    conv_lists = []

    for plan in bushy_plans:
        topology, leaves = extract_topology_and_leaves(plan)
        gather_bools, conv_triplets = get_cached_topology_structure(topology)

        # Create a gather array filled with n_nodes (pointing to the zero-vector)
        # and plug the actual leaf indices into the True positions.
        gather_array = np.full(len(gather_bools), n_nodes, dtype=np.int64)
        gather_array[gather_bools] = leaves

        gather_lists.append(gather_array)
        conv_lists.append(conv_triplets)

    # 2. Pad to max size in batch for GPU transfer
    max_nodes = max(len(g) for g in gather_lists)
    max_conv = max(len(c) for c in conv_lists)

    batch_indices = np.full((num_plans, max_nodes), n_nodes, dtype=np.int64)
    conv_indexes = np.zeros((num_plans, max_conv, 1), dtype=np.int64)

    # The mask identifies the padding node (0th element) based on your original logic
    # In tree convolution, index 0 is typically the dummy padding node.
    batch_mask = np.zeros((num_plans, max_nodes), dtype=bool)

    for i, (g, c) in enumerate(zip(gather_lists, conv_lists)):
        batch_indices[i, :len(g)] = g
        conv_indexes[i, :len(c)] = c
        batch_mask[i, 0] = True  # Marking the zero-vector root padding

    return batch_indices, conv_indexes, batch_mask


def get_bushy_shared_structure(bushy_plans, n_nodes, device):
    """
    Generates the structural tensors needed for `apply_features_to_structure_batched`.
    """
    gather_indices, conv_indexes, masks = compute_bushy_structural_indices(
        bushy_plans, n_nodes
    )

    gather_indices = torch.from_numpy(gather_indices).to(device)
    conv_indexes = torch.from_numpy(conv_indexes).to(device)
    masks = torch.from_numpy(masks).to(device)

    return gather_indices, conv_indexes, masks


def run_validation():
    device = torch.device("cpu")
    n_leaves = 4
    feature_dim = 16
    batch_size = 1

    # Generate dummy features for 4 relations.
    # The valid nodes are at indices 0, 1, 2, 3.
    # n_nodes is 4 (the index used to pull the zero-padding vector).
    features = torch.randn((batch_size, n_leaves, feature_dim), device=device)
    n_nodes = n_leaves

    print("--- Setup ---")
    print(f"Features Shape: {features.shape} (Batch, Nodes, Dim)")
    print(f"Padding Index (n_nodes): {n_nodes}\n")

    # 1. Run Old Method
    old_plan = [[0, 1, 2, 3]]
    precomputed_indexes = precompute_left_deep_tree_conv_index(max_leaves=8)
    precomputed_masks = precompute_left_deep_tree_node_mask(max_leaves=8)

    old_gather, old_conv, _ = _compute_structural_indices_old(
        old_plan, n_nodes, precomputed_indexes, precomputed_masks
    )
    old_gather_tensor = torch.from_numpy(old_gather).to(device)
    old_output = apply_features_to_structure_batched(features, old_gather_tensor)

    # 2. Run New Method
    # A left deep plan [0,1,2,3] translates to nested tuples: (((0, 1), 2), 3)
    new_plan = [(((0, 1), 2), 3)]

    new_gather, new_conv, _ = compute_bushy_structural_indices(new_plan, n_nodes)
    new_gather_tensor = torch.from_numpy(new_gather).to(device)
    new_output = apply_features_to_structure_batched(features, new_gather_tensor)

    # 3. Compare Results
    print("--- Comparison ---")
    print("Old Gather Indices:\n", old_gather)
    print("New Gather Indices:\n", new_gather)
    print("\nOld Conv Indices Flattened:\n", old_conv.flatten())
    print("New Conv Indices Flattened:\n", new_conv.flatten())

    # Assertions
    np.testing.assert_array_equal(old_gather, new_gather, err_msg="Gather indices do not match!")
    np.testing.assert_array_equal(old_conv, new_conv, err_msg="Convolution indices do not match!")

    assert torch.allclose(old_output, new_output, atol=1e-6), "Final feature maps do not match!"
    print("\n✅ SUCCESS: The dynamic bushy tree implementation "
          "produces identical structural matrices and feature maps as the original precomputed implementation.")


if __name__ == "__main__":
    run_validation()