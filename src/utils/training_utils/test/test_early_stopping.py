import pytest

from src.utils.training_utils.early_stopping import ConvergenceEarlyStopping


def test_minimization_stops_after_patience_beyond_minimum_epochs():
    stopper = ConvergenceEarlyStopping(
        patience=3,
        min_delta=0.01,
        min_epochs=4,
        mode="min",
    )

    assert not stopper.step(1.0, 1)
    assert not stopper.step(0.8, 2)
    assert not stopper.step(0.795, 3)
    assert not stopper.step(0.794, 4)
    assert not stopper.step(0.793, 5)
    assert stopper.step(0.792, 6)
    assert stopper.best_value == 0.8
    assert stopper.best_epoch == 2


def test_maximization_resets_patience_after_meaningful_improvement():
    stopper = ConvergenceEarlyStopping(patience=2, min_delta=0.1, mode="max")

    assert not stopper.step(1.0, 1)
    assert not stopper.step(1.05, 2)
    assert not stopper.step(1.2, 3)
    assert not stopper.step(1.25, 4)
    assert stopper.step(1.26, 5)
    assert stopper.best_value == 1.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patience": 0},
        {"patience": 1, "min_delta": -0.1},
        {"patience": 1, "min_epochs": -1},
        {"patience": 1, "mode": "invalid"},
    ],
)
def test_invalid_early_stopping_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ConvergenceEarlyStopping(**kwargs)


def test_non_finite_metric_is_rejected():
    stopper = ConvergenceEarlyStopping(patience=2)

    with pytest.raises(ValueError, match="finite"):
        stopper.step(float("nan"), 1)