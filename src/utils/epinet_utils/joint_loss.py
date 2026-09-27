import torch
import torch.nn as nn
import numpy as np

# #TODO: ChatGPT code evaluate it
# class GaussianJointLogLoss(nn.Module):
#     def __init__(self, noise_std=1.0):
#         """
#         Args:
#             noise_std: The assumed aleatoric noise (sigma) for the data.
#                        This scales the penalty. 1.0 is a standard default if data is normalized.
#         """
#         super().__init__()
#         self.noise_std = noise_std
#
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
#         """
#         Computes Joint Log-Loss for a group of correlated plans (e.g., one query).
#
#         Args:
#             predictions: Tensor of shape [K, tau]
#                          K   = number of epistemic index samples (z)
#                          tau = number of plans in this group/query
#             targets:     Tensor of shape [tau] (True costs)
#
#         Returns:
#             Scalar: The Joint Negative Log-Likelihood (to be minimized).
#         """
#         # predictions: [, tau]
#         n_z_sampled, n_plans = predictions.shape
#
#         # 1. Expand targets to compare against every z sample
#         # targets_exp: [K, tau]
#         targets_exp = targets.unsqueeze(0).expand(n_z_sampled, -1)
#
#         # 2. Compute Gaussian Log-Likelihood for EACH plan under EACH z
#         # Formula: -0.5 * log(2*pi*var) - (y - pred)^2 / (2*var)
#         var = self.noise_std ** 2
#         log_scale = np.log(np.sqrt(2 * np.pi * var))
#         squared_error = (predictions - targets_exp) ** 2
#
#         # pointwise_ll: [K, tau]
#         pointwise_ll = -log_scale - (squared_error / (2 * var))
#
#         # 3. Sum over the group (tau)
#         # This represents the Joint Probability: P(y_1...y_tau | z) = Prod P(y_i|z)
#         # In log-space: Sum(log P)
#         # joint_ll_per_z: [K]
#         joint_ll_per_z = torch.sum(pointwise_ll, dim=1)
#
#         # 4. Average over K samples (z)
#         # We need log( Mean( Probability ) )
#         # Using LogSumExp trick: log( 1/K * sum(exp(ll)) )
#         #                      = log(sum(exp(ll))) - log(K)
#         # log_joint_prob: Scalar
#         log_joint_prob = torch.logsumexp(joint_ll_per_z, dim=0) - np.log(n_z_sampled)
#
#         # 5. Return Negative Log Likelihood (Minimize this)
#         return -log_joint_prob


class GaussianJointLogLoss(nn.Module):
    def __init__(self, noise_std=1.0, tau=10):
        """
        Args:
            noise_std: The assumed aleatoric noise (sigma) for the data.
            tau:       The fixed evaluation size for the joint probability.
        """
        super().__init__()
        if noise_std <= 0:
            raise ValueError("noise_std must be greater than zero")
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        self.noise_std = noise_std
        self.tau = tau

    def sample_plan_indices(self, n_plans, device, generator=None):
        if n_plans <= 0:
            raise ValueError("n_plans must be greater than zero")
        if n_plans >= self.tau:
            indices = torch.randperm(n_plans, device=device, generator=generator)
            n_keep = (n_plans // self.tau) * self.tau
            return indices[:n_keep].view(-1, self.tau)

        return torch.randint(
            0,
            n_plans,
            (1, self.tau),
            device=device,
            generator=generator,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, plan_indices=None):
        """
        predictions: Tensor of shape [K, n_plans]
        targets:     Tensor of shape [n_plans]
        """
        n_z_sampled, n_plans = predictions.shape
        var = self.noise_std ** 2
        log_scale = np.log(np.sqrt(2 * np.pi * var))

        if plan_indices is None:
            plan_indices = self.sample_plan_indices(n_plans, predictions.device)

        chunked_preds = predictions[:, plan_indices]
        chunked_targets = targets[plan_indices]
        targets_exp = chunked_targets.unsqueeze(0).expand(n_z_sampled, -1, -1)

        squared_error = (chunked_preds - targets_exp) ** 2
        pointwise_ll = -log_scale - (squared_error / (2 * var))
        joint_ll_per_z = torch.sum(pointwise_ll, dim=2)
        log_joint_prob_per_chunk = torch.logsumexp(joint_ll_per_z, dim=0) - np.log(n_z_sampled)

        # Average per query so queries with more enumerated plans do not dominate.
        return -torch.mean(log_joint_prob_per_chunk)