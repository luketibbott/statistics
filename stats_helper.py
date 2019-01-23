from scipy.stats import norm
from scipy.stats import t
import numpy as np
import math

def get_confidence_from_z_star(z_star):
     one_tail = norm.cdf(z_star)
     total_confidence = abs(1 - 2*one_tail)
     return total_confidence

def get_z_star_from_confidence(confidence):
    one_tail = (1 - confidence) / 2
    return abs(norm.ppf(one_tail))

def get_mean_diff_ci(confidence, obs_diff, standard_dev, n):
    one_tail = (1 - confidence) / 2
    t_star = abs(t.ppf(one_tail, df=n-1))
    margin = t_star * (standard_dev / np.sqrt(n))
    ci = (obs_diff - margin, obs_diff + margin)

    return ci

def get_t_star_from_confidence(confidence, n):
    one_tail = (1 - confidence) / 2
    return abs(t.ppf(one_tail, df=n - 1))

def get_smallest_sample_size(confidence, std, margin):
    one_tail = (1 - confidence) / 2
    z_star = abs(norm.ppf(one_tail))
    
    return math.ceil((z_star * std / margin)**2)

def nCr(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def std_of_discrete(observed_probs):
    expected_val = 0
    variance = 0

    for key in observed_probs:
        expected_val += key * observed_probs[key]

    for key in observed_probs:
        variance += (key - expected_val)**2 * observed_probs[key]

    std = np.sqrt(variance)

    return std

def binomial(n_trials, n_success, prob_success):
    return nCr(n_trials, n_success) * prob_success**n_success * (1 - prob_success)**(n_trials - n_success)

def r(xs, ys):
    xs = np.array(xs)
    ys = np.array(ys)
    
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)

    x_std = xs.std(ddof=1)
    y_std = ys.std(ddof=1)

    r = 0

    for x, y in zip(xs, ys):
        x_z_score = (x - x_mean)/x_std
        y_z_score = (y - y_mean)/y_std

        r += x_z_score * y_z_score

    return r / (len(xs) - 1)