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
    assert optimizer.param_groups[0]['lr'] == 0.01