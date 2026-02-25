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
        self.noise_std = noise_std
        self.tau = tau

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        predictions: Tensor of shape [K, n_plans]
        targets:     Tensor of shape [n_plans]
        """
        n_z_sampled, n_plans = predictions.shape
        var = self.noise_std ** 2
        log_scale = np.log(np.sqrt(2 * np.pi * var))

        # Chunk plans for calculation
        if n_plans >= self.tau:
            indices = torch.randperm(n_plans, device=predictions.device)

            n_chunks = n_plans // self.tau
            n_keep = n_chunks * self.tau
            indices = indices[:n_keep]

            # 3. Reshape into parallel chunks
            chunked_preds = predictions[:, indices].view(n_z_sampled, n_chunks, self.tau)
            chunked_targets = targets[indices].view(n_chunks, self.tau)

            # 4. Expand targets across K samples
            # targets_exp: [K, n_chunks, tau]
            targets_exp = chunked_targets.unsqueeze(0).expand(n_z_sampled, -1, -1)

            # 5. Compute pointwise Log-Likelihoods
            squared_error = (chunked_preds - targets_exp) ** 2
            pointwise_ll = -log_scale - (squared_error / (2 * var))  # [K, n_chunks, tau]

            # 6. Sum over the tau dimension to get Joint LL per chunk
            joint_ll_per_z = torch.sum(pointwise_ll, dim=2)  # [K, n_chunks]

            # 7. LogSumExp over K samples
            log_joint_prob_per_chunk = torch.logsumexp(joint_ll_per_z, dim=0) - np.log(n_z_sampled)  # [n_chunks]

            # We calculate the mean on a per-query basis, otherwise the joint loss is dominated by larger queries
            # with many more plans
            return -torch.mean(log_joint_prob_per_chunk)

        # When we have too little observations for a single tau sized batch we resample
        else:
            # Sample WITH replacement to pad it up to exactly 'tau' elements
            indices = torch.randint(0, n_plans, (self.tau,), device=predictions.device)

            sampled_preds = predictions[:, indices]  # [K, tau]
            sampled_targets = targets[indices]  # [tau]

            targets_exp = sampled_targets.unsqueeze(0).expand(n_z_sampled, -1)  # [K, tau]

            squared_error = (sampled_preds - targets_exp) ** 2
            pointwise_ll = -log_scale - (squared_error / (2 * var))

            joint_ll_per_z = torch.sum(pointwise_ll, dim=1)  # [K]
            log_joint_prob = torch.logsumexp(joint_ll_per_z, dim=0) - np.log(n_z_sampled)  # Scalar

            return -log_joint_prob