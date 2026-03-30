class UncertaintyAnnealingScheduler:
    """Tracks uncertainty EMA and computes the blending weight centrally."""

    def __init__(self, ema_alpha=0.05, method="epistemic_uncertainty"):
        self.ema_alpha = ema_alpha
        self.method = method
        self.head_uncertainties = {}
        self.current_weight = 0.0

    def update_and_get_weight(self, estimate_variances):
        # 1. Update EMA
        for key, variance in estimate_variances.items():
            if key not in self.head_uncertainties:
                self.head_uncertainties[key] = {"average": variance, "min": variance, "max": variance}
            else:
                current_avg = self.head_uncertainties[key]["average"]
                new_avg = (1 - self.ema_alpha) * current_avg + self.ema_alpha * variance
                self.head_uncertainties[key]["average"] = new_avg
                self.head_uncertainties[key]["min"] = min(self.head_uncertainties[key]["min"], new_avg)
                self.head_uncertainties[key]["max"] = max(self.head_uncertainties[key]["max"], new_avg)

        # 2. Calculate Coefficient
        if self.method == "epistemic_uncertainty":
            # Guard against division by zero if latency variance is 0 initially
            lat_var = self.head_uncertainties.get("latency", {}).get("average", 1e-6)
            cost_var = self.head_uncertainties.get("plan_cost", {}).get("average", 0.0)

            estimated_variance_ratio = cost_var / max(lat_var, 1e-6)
            self.current_weight = estimated_variance_ratio
        else:
            raise NotImplementedError("Annealing Method Not Implemented")

        return self.current_weight