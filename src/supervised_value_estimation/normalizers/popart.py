import torch
import torch.nn as nn

class PopArtLinear(nn.Module):
    def __init__(self, in_features, out_features, ema_momentum=0.01, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ema_momentum = ema_momentum

        # The learnable normalized weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device))

        # Non-learnable statistics buffers
        self.register_buffer('mu', torch.zeros(out_features, requires_grad=False, device=device))
        self.register_buffer('sigma', torch.ones(out_features, requires_grad=False, device=device))

        self.is_initialized = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @torch.no_grad()
    def update_statistics(self, targets, valid_mask=None):
        """Updates statistics and precisely rescales weights/biases to preserve outputs."""
        if valid_mask is not None:
            targets = targets[valid_mask]

        if targets.numel() == 0:
            return

        batch_mu = targets.mean(dim=0)
        batch_var = targets.var(dim=0, unbiased=False)
        batch_sigma = torch.sqrt(batch_var).clamp(min=1e-4)

        if not self.is_initialized:
            new_mu = batch_mu
            new_sigma = batch_sigma
            self.is_initialized = True
        else:
            new_mu = (1 - self.ema_momentum) * self.mu + self.ema_momentum * batch_mu
            new_var = (1 - self.ema_momentum) * (self.sigma ** 2) + self.ema_momentum * batch_var
            new_sigma = torch.sqrt(new_var).clamp(min=1e-4)

        # PopArt exactly preserves the unnormalized output: W_new = W_old * (sigma_old / sigma_new)
        self.weight.data.mul_(self.sigma.unsqueeze(1) / new_sigma.unsqueeze(1))

        # b_new = (b_old * sigma_old + mu_old - mu_new) / sigma_new
        self.bias.data.mul_(self.sigma).add_(self.mu).sub_(new_mu).div_(new_sigma)

        self.mu.copy_(new_mu)
        self.sigma.copy_(new_sigma)

    def forward(self, x):
        """Returns the NORMALIZED prediction."""
        return nn.functional.linear(x, self.weight, self.bias)

    def unnormalize(self, normalized_prediction):
        """Converts the normalized output back to the true scale for inference/search."""
        return (normalized_prediction * self.sigma) + self.mu

    def normalize_target(self, target):
        """Normalizes external targets using current statistics for loss computation."""
        return (target - self.mu) / self.sigma