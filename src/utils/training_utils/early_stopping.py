import math


class ConvergenceEarlyStopping:
    def __init__(self, patience, min_delta=0.0, min_epochs=0, mode="min"):
        if patience <= 0:
            raise ValueError("patience must be greater than zero")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if min_epochs < 0:
            raise ValueError("min_epochs must be non-negative")
        if mode not in ("min", "max"):
            raise ValueError("mode must be either 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.mode = mode
        self.best_value = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.bad_epochs = 0

    def _is_better(self, value):
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def step(self, value, epoch):
        if not math.isfinite(value):
            raise ValueError("early-stopping metric must be finite")

        if self._is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        elif epoch >= self.min_epochs:
            self.bad_epochs += 1

        return epoch >= self.min_epochs and self.bad_epochs >= self.patience