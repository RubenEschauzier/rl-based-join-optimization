import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def compute_calibration_measures(y_true, epinet_predictions):
    p_values = compute_p_values(y_true, epinet_predictions)
    epinet_distribution_variance = calculate_predicted_distribution_variance(epinet_predictions)
    return p_values, epinet_distribution_variance

def calculate_predicted_distribution_variance(epinet_predictions):
    return np.var(epinet_predictions, axis=1, ddof=1)

def calculate_sharpness(predicted_distribution_variance):
    return np.mean(predicted_distribution_variance)

def calculate_calibration_error(expected, observed):
    return np.mean(np.abs(expected - observed))

def compute_p_values(y_true, epinet_predictions):
    n, m = epinet_predictions.shape

    p_values = []
    for i in range(n):
        pred_dist = epinet_predictions[i, :]
        cdf_value = np.sum(pred_dist <= y_true[i]) / m
        p_values.append(cdf_value)

    return np.array(p_values)

def compute_calibration_curve(p_values, n_confidences):
    expected_p = np.linspace(0, 1, n_confidences)

    observed_p = []
    for p in expected_p:
        # Count how many data points from empirical CDF fall inside this quantile
        fraction = np.mean(p_values <= p)
        observed_p.append(fraction)
    return observed_p, expected_p

def calculate_calibration_metrics(p_values, predicted_distribution_variance, n_confidences, save_location):
    observed_p, expected_p = compute_calibration_curve(p_values, n_confidences)
    calibration_error = calculate_calibration_error(expected_p, observed_p)
    sharpness = calculate_sharpness(predicted_distribution_variance)
    plot_calibration(observed_p, expected_p, calibration_error, save_location)
    return calibration_error, sharpness

def plot_calibration(observed, expected, error, save_location=None):
    plt.figure(figsize=(6, 6))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Model calibration curve
    plt.plot(expected, observed, "r-", label=f"Model (Error: {error:.4f})")

    plt.xlabel("Expected Confidence Level")
    plt.ylabel("Observed Confidence Level")
    plt.title("Regression Calibration Plot (Kuleshov et al.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()
    plt.close()