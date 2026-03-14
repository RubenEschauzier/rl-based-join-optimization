import torch

def right_censored_hinge_loss(predictions, targets, is_censored, timeout_threshold):
    """
    Computes a combination of MSE for valid targets and Hinge Loss for censored targets.

    Args:
        predictions: Tensor of shape (batch_size,)
        targets: Tensor of shape (batch_size,). Contains actual latency or timeout value.
        is_censored: Boolean tensor of shape (batch_size,). True if query timed out/OOMed.
        timeout_threshold: Float representing the timeout limit.
    """

    # Calculate standard MSE for successful queries
    mse_loss = torch.nn.functional.mse_loss(predictions, targets, reduction='none')

    # Calculate one-sided hinge loss for timeouts/OOMs
    # Clamp negative values to 0 (i.e., prediction is greater than timeout)
    censored_error = torch.clamp(timeout_threshold - predictions, min=0.0)
    censored_loss = censored_error ** 2

    # Combine losses using the boolean mask
    # Use torch.where to select mse_loss where ~is_censored, else censored_loss
    total_loss = torch.where(is_censored, censored_loss, mse_loss)

    return total_loss.mean()