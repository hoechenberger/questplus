from typing import Union, Iterable
import numpy as np
import scipy.stats


def weibull_log10(*,
                  intensity: Union[float, Iterable[float]],
                  threshold: Union[float, Iterable[float]],
                  slope: Union[float, Iterable[float]] = 3.5,
                  lower_asymptote: Union[float, Iterable[float]] = 0.01,
                  lapse_rate: Union[float, Iterable[float]] = 0.01) -> np.ndarray:
    intensity = np.array(intensity, dtype='float64')
    threshold = np.array(threshold, dtype='float64')
    slope = np.array(slope, dtype='float64')
    lower_asymptote = np.array(lower_asymptote, dtype='float64')
    lapse_rate = np.array(lapse_rate, dtype='float64')

    x, t, beta, gamma, delta = np.meshgrid(intensity,
                                           threshold,
                                           slope,
                                           lower_asymptote,
                                           lapse_rate,
                                           indexing='ij', sparse=True)

    assert np.atleast_1d(intensity.squeeze()).shape == np.atleast_1d(intensity).shape
    assert np.atleast_1d(t.squeeze()).shape == np.atleast_1d(threshold).shape
    assert np.atleast_1d(beta.squeeze()).shape == np.atleast_1d(slope).shape
    assert np.atleast_1d(gamma.squeeze()).shape == np.atleast_1d(lower_asymptote).shape
    assert np.atleast_1d(delta.squeeze()).shape == np.atleast_1d(lapse_rate).shape

    p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t)))
    return p


def normpdf(x, mu, sigma, gamma, lambda_):
    x = np.array(x, dtype='float64')
    sigma = np.array(sigma, dtype='float64')
    gamma = np.array(gamma, dtype='float64')
    lambda_ = np.array(lambda_, dtype='float64')

    p = gamma+(1 - gamma - lambda_) * scipy.stats.norm.cdf(x, mu, sigma)
    return p
