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
