import numpy as np
import pytest
import torch

from src.utils.epinet_utils.calibration_plot import compute_calibration_measures
from src.utils.epinet_utils.joint_loss import GaussianJointLogLoss


def test_shared_plan_indices_make_identical_distributions_score_equally():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=4)
    targets = torch.tensor([0.0, 0.5, -0.5, 1.0])
    base_predictions = torch.tensor([[0.1, 0.4, -0.4, 0.8]])
    epinet_predictions = base_predictions.repeat(8, 1)
    plan_indices = torch.arange(4).view(1, 4)

    base_nll = loss(base_predictions, targets, plan_indices)
    epinet_nll = loss(epinet_predictions, targets, plan_indices)

    assert torch.allclose(base_nll, epinet_nll)


def test_joint_nll_rewards_coherent_epistemic_samples():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=2)
    targets = torch.tensor([0.0, 0.0])
    coherent_predictions = torch.tensor([
        [0.0, 0.0],
        [2.0, 2.0],
    ])
    dependence_destroyed_predictions = torch.tensor([
        [0.0, 2.0],
        [2.0, 0.0],
    ])
    plan_indices = torch.arange(2).view(1, 2)

    coherent_nll = loss(coherent_predictions, targets, plan_indices)
    independent_nll = loss(dependence_destroyed_predictions, targets, plan_indices)

    assert coherent_nll < independent_nll


def test_plan_sampling_is_reproducible_and_shared_across_models():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=3)
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first_indices = loss.sample_plan_indices(8, torch.device("cpu"), first_generator)
    second_indices = loss.sample_plan_indices(8, torch.device("cpu"), second_generator)

    assert torch.equal(first_indices, second_indices)
    assert first_indices.shape == (2, 3)
    assert torch.unique(first_indices).numel() == 6


def test_dyadic_sampling_is_reproducible_and_uses_two_anchors_per_group():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=10)
    first_generator = torch.Generator().manual_seed(23)
    second_generator = torch.Generator().manual_seed(23)

    first_indices = loss.sample_dyadic_plan_indices(
        20,
        8,
        torch.device("cpu"),
        first_generator,
    )
    second_indices = loss.sample_dyadic_plan_indices(
        20,
        8,
        torch.device("cpu"),
        second_generator,
    )

    assert torch.equal(first_indices, second_indices)
    assert first_indices.shape == (8, 10)
    assert all(torch.unique(group).numel() == 2 for group in first_indices)


def test_dyadic_sampling_repeats_the_only_available_plan():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=4)

    indices = loss.sample_dyadic_plan_indices(1, 3, torch.device("cpu"))

    assert torch.equal(indices, torch.zeros((3, 4), dtype=torch.long))


def test_predictive_calibration_includes_gaussian_observation_noise():
    targets = np.array([0.0, 1.0])
    epistemic_means = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    p_values, predictive_variances = compute_calibration_measures(
        targets,
        epistemic_means,
        noise_std=0.2,
    )

    np.testing.assert_allclose(p_values, np.array([0.5, 0.5]))
    np.testing.assert_allclose(predictive_variances, np.array([0.04, 0.04]))


@pytest.mark.parametrize(
    ("noise_std", "tau"),
    [(0.0, 10), (-0.1, 10), (0.2, 0)],
)
def test_joint_loss_rejects_invalid_evaluation_parameters(noise_std, tau):
    with pytest.raises(ValueError):
        GaussianJointLogLoss(noise_std=noise_std, tau=tau)


def test_dyadic_sampling_rejects_invalid_pair_count():
    loss = GaussianJointLogLoss(noise_std=0.2, tau=4)

    with pytest.raises(ValueError, match="n_pairs"):
        loss.sample_dyadic_plan_indices(4, 0, torch.device("cpu"))