import pytest
import torch

from src.pretrain_procedure import DualMetricScheduler


def test_scheduler_waits_for_both_metrics_to_plateau():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = DualMetricScheduler(optimizer, patience=1, threshold=0.0, factor=0.1)

    scheduler.step(2.0, 2.0)
    scheduler.step(1.0, 2.0)
    scheduler.step(0.5, 2.0)
    assert optimizer.param_groups[0]['lr'] == 0.1

    scheduler.step(0.5, 2.0)
    scheduler.step(0.5, 2.0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.01)


def test_scheduler_respects_min_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = DualMetricScheduler(optimizer, patience=0, threshold=0.0, factor=0.1,
                                    min_lr=0.05)

    for _ in range(20):
        scheduler.step(1.0, 1.0)

    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.05)


def test_relative_threshold_scales_with_the_metric():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    # 1% relative: on a loss of ~0.07 this asks for a 7e-4 drop, not 1e-2.
    scheduler = DualMetricScheduler(optimizer, patience=1, threshold=1e-2,
                                    threshold_mode='rel', factor=0.1)

    scheduler.step(0.0700, 0.0700)
    # A 1.4% drop is real progress under a relative threshold; an absolute
    # threshold of 1e-2 would have dismissed it as a plateau.
    scheduler.step(0.0690, 0.0690)
    assert scheduler.bad_train_epochs == 0
    assert scheduler.bad_val_epochs == 0