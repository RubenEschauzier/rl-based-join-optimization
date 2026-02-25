import h5py
import numpy as np
import torch
import os

class DiskCacheFrozenRepresentations:
    def __init__(self, filepath="frozen_embeddings.h5", force_overwrite=False):
        self.filepath = filepath

        # Only initialize if explicitly forced, or if the file does not exist yet
        if force_overwrite or not os.path.exists(self.filepath):
            # 'w' mode creates a new file, overwriting any existing file with the same name
            with h5py.File(self.filepath, 'w') as f:
                pass

    def has(self, state_key: str) -> bool:
        """Checks if the embedding group for the given state_key exists."""
        with h5py.File(self.filepath, 'r') as f:
            return state_key in f

    def __contains__(self, state_key: str) -> bool:
        return self.has(state_key)

    def _store_recursive(self, base_group, name, item):
        """Recursively stores lists, tuples, or tensors into HDF5 groups/datasets."""
        if isinstance(item, (list, tuple)):
            # Create a subgroup for the list/tuple
            subgroup = base_group.create_group(name)
            # Remember the original python type
            subgroup.attrs['type'] = 'list' if isinstance(item, list) else 'tuple'

            for i, v in enumerate(item):
                self._store_recursive(subgroup, str(i), v)
        else:
            # Base case: it's a tensor or numpy array
            if torch.is_tensor(item):
                np_item = item.detach().cpu().numpy()
            else:
                np_item = np.asarray(item)

            base_group.create_dataset(name, data=np_item, compression="lzf")

    def store_embeddings(self, state_key: str, embeddings):
        """Saves arbitrarily nested lists/tuples of PyTorch tensors to disk."""
        with h5py.File(self.filepath, 'a') as f:
            if state_key in f:
                del f[state_key]

            self._store_recursive(f, state_key, embeddings)

    def _load_recursive(self, node, device):
        """Recursively reconstructs lists, tuples, or tensors from HDF5."""
        if isinstance(node, h5py.Dataset):
            # Base case: Read dataset and convert to tensor
            return torch.from_numpy(node[:]).to(device)
        elif isinstance(node, h5py.Group):
            # Sort keys as integers to preserve original order
            # (otherwise '10' comes before '2' in string sorting)
            keys = sorted(node.keys(), key=int)
            loaded_items = [self._load_recursive(node[k], device) for k in keys]

            # Reconstruct the original Python type
            item_type = node.attrs.get('type', 'tuple')
            if item_type == 'list':
                return list(loaded_items)
            return tuple(loaded_items)
        return None

    def get_embeddings(self, state_key: str, device="cpu"):
        """Retrieves the nested embeddings from disk and converts them to Tensors."""
        with h5py.File(self.filepath, 'r') as f:
            if state_key not in f:
                raise KeyError(f"Embeddings for {state_key} not found on disk.")

            return self._load_recursive(f[state_key], device)