import numpy as np
import torch


def tensors_to_numpy(obj):
    """
    Recursively converts all PyTorch tensors in a dictionary, list, or tuple
    to NumPy arrays to prevent file descriptor leaks in multiprocessing queues.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: tensors_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensors_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensors_to_numpy(v) for v in obj)
    else:
        return obj

def numpy_to_tensors(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_tensors(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_tensors(v) for v in obj)
    else:
        return obj
