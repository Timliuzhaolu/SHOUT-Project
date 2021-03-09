import mpmath as mp
import numpy as np
from scipy.optimize import fsolve
import scipy
import scipy.stats
from typing import Callable, Tuple


def get_twosided_pval(mcc, p_train, sample_n, pred_mean):
    """
    Obtains two-sided P-value for the hypothesis that the evaluation set and training set are random samples from
    the same population, i.e. that there is no bias in the evaluation set.

    :param mcc: Matthews Correlation Coefficient
    :param p_train: Mean of binary variable in the training set
    :param sample_n: Sample size of evaluation set
    :param pred_mean: Predicted mean on the evaluation sample
    :return: P value
    """
    # Assume pe=pt
    e1 = mcc + (1 - mcc) * p_train
    e0 = (1 - mcc) * p_train
    v = e1 * (1 - e1) * p_train + e0 * (1 - e0) * (1 - p_train)
    std_err = np.sqrt(v / sample_n)
    e = p_train
    z = abs(pred_mean - e) / std_err
    return scipy.stats.norm.cdf(-z) * 2


def get_posterior_confidence_interval(mcc, p_train, pred_mean, sample_n, confidence=0.99, integral_max_n=1000) -> Tuple[
    Tuple[float, float], float]:
    """
    Obtain a confidence interval for a posterior distribution
    :param mcc: Matthews Correlation Coefficient
    :param p_train: Mean of binary variable in the training set
    :param pred_mean: Predicted mean on the evaluation sample
    :param sample_n: Sample size of evaluation set
    :param confidence: Desire confidence level of confidence interval (default: 0.99)
    :param integral_max_n: Maximum sample_n to numerically obtain the integral, above which
    approximate methods derived from the Central Limit Theorem are used.
    :return: (CI_lower, CI_upper), confidence of this interval (not always identical to the desired confidence)
    """
    likelihood = get_likelihood_lambda(mcc, p_train, pred_mean,
                                       sample_n if integral_max_n is None or sample_n < integral_max_n else integral_max_n)
    max_likelihood = get_max_likelihood(mcc, p_train, pred_mean)
    posterior = get_posterior(likelihood, max_likelihood)
    mean, std = get_expectation_std(posterior)
    if integral_max_n is None or sample_n < integral_max_n:
        ci, s = solve_for_ci(posterior, mean, std * 2.5, confidence=confidence)
        if (s - confidence) > 1e-5:
            raise RuntimeWarning("Confidence is not correct")
        return ci
    else:
        mean = max_likelihood
        std = float(std * mp.sqrt(integral_max_n / sample_n))
        return get_confidence_interval(mean, std * 2.576)


def get_max_likelihood(mcc: float, p_train: float, pred_mean: float) -> float:
    """
    Returns the maximum likelihood estimate for a binary variable and a trained model,
    given the Matthews Correlation Coefficient, the training set mean, and the predicted mean
    """
    max_likelihood: float = (pred_mean - (1 - mcc) * p_train) / mcc
    return max(min(max_likelihood, 1), 0)


def get_likelihood_lambda(mcc: float, p_train: float, pred_mean: float, n: int) -> Callable[[float], float]:
    """
    Returns a callable for the likelihood function
    """
    h, t = pred_mean * n, (1 - pred_mean) * n
    return lambda mu: ((mcc * mu + (1 - mcc) * p_train) ** h) * ((mcc * (1 - mu) + (1 - p_train) * (1 - mcc)) ** t)


def get_posterior(likelihood_lambda: Callable[[float], float], max_likelihood: float) -> Callable[[float], float]:
    """
    Returns a numerical estimate of the posterior distribution over mu, as a callable
    """
    # Use 100 intermediate points around the maximum likelihood in the numerical integral to improve stability
    bounds = np.linspace(max_likelihood - 0.01, max_likelihood + 0.01, 100).tolist() + [0, 1, max_likelihood]
    bounds = [i for i in sorted(set(bounds)) if 0 <= i <= 1]

    marginal = mp.quad(likelihood_lambda, bounds)
    posterior = lambda mu: likelihood_lambda(mu) / marginal

    # Repeat to fix errors from the first time
    marginal2 = mp.quad(posterior, bounds)
    posterior = lambda mu: likelihood_lambda(mu) / marginal / marginal2

    # Now just verify that it's all chill.
    marginal3 = mp.quad(posterior, bounds)
    assert (marginal3 - 1) < 1e-10
    return posterior


def get_expectation_std(distribution: Callable[[float], float]) -> Tuple[float, float]:
    """
    Given a distribution, numerically obtains the expectation and standard deviation
    """
    expectation = mp.quad(lambda x: x * distribution(x), [0, 1])
    expectation2 = mp.quad(lambda x: (x ** 2) * distribution(x), [0, 1])
    var = expectation2 - expectation ** 2
    std = mp.sqrt(var)
    return expectation, std


def get_confidence_interval(centre: float, radius: float) -> Tuple[float, float]:
    """
    Obtains the bounds for an interval, given the centre and radius of the interval
    """
    radius = float(radius)
    ciupper = min(centre + radius, 1)
    cilower = max(centre - radius, 0)
    return float(cilower), float(ciupper)


def solve_for_ci(distribution: Callable[[float], float],
                 centre: float,
                 guess_radius: float, confidence: float = 0.99) -> Tuple[Tuple[float, float], float]:
    """
    Numerically obtains a confidence interval over a distribution, given the desired centre of the confidence interval
    and a guess of the confidence interval's radius. More precise than relying on the Central Limit Theorem.
    :return: (CI_lower, CI_upper), empirical confidence level (not always exactly the same as the desired confidence)
    """

    def try_ci(s):
        cilower, ciupper = get_confidence_interval(centre, s)
        integral = mp.quad(distribution, [cilower, ciupper])
        return integral - confidence

    s = fsolve(try_ci, float(guess_radius))
    return get_confidence_interval(centre, s), try_ci(s) + confidence
