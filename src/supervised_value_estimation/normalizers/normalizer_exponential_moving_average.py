import torch

class NormalizerExponentialMovingAverage:
    def __init__(self, device, ema_momentum=0.05):
        self.mean = torch.tensor(0.0, device=device)
        self.std = torch.tensor(1.0, device=device)
        self.ema_momentum = ema_momentum
        self.is_initialized = False

    def update(self, data: torch.Tensor, valid_mask: torch.Tensor = None):
        """Updates mean and standard deviation. Applies EMA if already initialized."""
        if valid_mask is not None:
            data = data[valid_mask]

        if data.numel() == 0:
            return

        batch_mean = data.mean()
        batch_std = data.std().clamp(min=1e-8)  # Prevent division by zero

        if not self.is_initialized:
            self.mean = batch_mean
            self.std = batch_std
            self.is_initialized = True
        else:
            self.mean = (1 - self.ema_momentum) * self.mean + self.ema_momentum * batch_mean
            self.std = (1 - self.ema_momentum) * self.std + self.ema_momentum * batch_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean