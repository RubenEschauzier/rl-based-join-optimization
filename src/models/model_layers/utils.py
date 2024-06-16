import torch
import torch.nn as nn


# Written by: ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py)
# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
	``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
	``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
	broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	In the case that the input vector is completely masked, the return value of this function is
	arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
	of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
	that we deal with this case relies on having single-precision floats; mixing half-precision
	floats with fully-masked vectors will likely give you ``nans``.
	If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
	lower), the way we handle masking here could mess you up.  But if you've got logit values that
	extreme, you've got bigger problems than this.
	"""
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor,
               mask: torch.Tensor,
               dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    """
	To calculate max along certain dimensions on masked values
	Parameters
	----------
	vector : ``torch.Tensor``
		The vector to calculate max, assume unmasked parts are already zeros
	mask : ``torch.Tensor``
		The mask of the vector. It must be broadcastable with vector.
	dim : ``int``
		The dimension to calculate max
	keepdim : ``bool``
		Whether to keep dimension
	min_val : ``float``
		The minimal value for paddings
	Returns
	-------
	A ``torch.Tensor`` of including the maximum values.
	"""
    one_minus_mask = (1.0 - mask).bool()
    print(one_minus_mask)
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index


# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_sample(vector: torch.Tensor,
                  mask: torch.Tensor,
                  sequence_mask: torch.Tensor,
                  dim: int,
                  keepdim: bool = False,
                  min_val: float = -1e10) -> (torch.Tensor, torch.Tensor):
    """
	To calculate max along certain dimensions on masked values
	Parameters
	----------
	vector : ``torch.Tensor`` The vector to calculate max, assume unmasked parts are already zeros
	mask : ``torch.Tensor``
		The mask of the vector. It must be broadcastable with vector.
	dim : ``int``
		The dimension to calculate max
	keepdim : ``bool``
		Whether to keep dimension
	min_val : ``float``
		The minimal value for paddings
	Returns
	-------
	A ``torch.Tensor`` of including the maximum values.
	"""
    one_minus_mask = (1.0 - mask).bool()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    # Ensure mask is applied absolutely with no residual probability (very likely not needed)
    probabilities = torch.exp(replaced_vector) * mask

    no_generation = torch.where(torch.sum(mask, 1) == 0, 1, 0).unsqueeze(1).expand(-1, probabilities.shape[1])

    # Ensure that get no zero probabilities for all tokens if no token should be generated to prevent nans in sampling
    # The generated token will be discarded by applying the mask, so the actual probability does not matter
    probabilities = probabilities + (no_generation * (1/probabilities.shape[1]))

    m = torch.distributions.Categorical(probs=probabilities)
    mask_sampled = (m.sample() * sequence_mask).unsqueeze(1)

    return probabilities, mask_sampled
