import numpy as np

# Reduce the sample size to 50 and keep the large mean difference
from scipy.stats import ttest_ind

n_small = 50  # Smaller sample size
mean1, mean2 = 0, 2  # Large mean difference

# Generate smaller random samples with the updated sample size and means
sample1_small = np.random.normal(mean1, .5, n_small)
sample2_small = np.random.normal(mean2, .5, n_small)

# Perform two-sample t-test again
stat, p_value = ttest_ind(sample1_small, sample2_small)

print(stat, p_value)