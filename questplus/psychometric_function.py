from typing import Union, Iterable
import numpy as np


# import scipy.stats


def weibull(*,
            intensity: Union[float, Iterable[float]],
            threshold: Union[float, Iterable[float]],
            slope: Union[float, Iterable[float]] = 3.5,
            lower_asymptote: Union[float, Iterable[float]] = 0.01,
            lapse_rate: Union[float, Iterable[float]] = 0.01,
            scale: str = 'log10') -> np.ndarray:
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

    s = 1 if scale == 'log10' else 20
    p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t) / s))
    return p


def csf(*,
        contrast: Union[float, Iterable[float]],
        spatial_freq: Union[float, Iterable[float]],
        temporal_freq: Union[float, Iterable[float]] = 0,
        c0: Union[float, Iterable[float]],
        cf: Union[float, Iterable[float]],
        cw: Union[float, Iterable[float]] = 0,
        min_thresh: Union[float, Iterable[float]],
        slope: Union[float, Iterable[float]] = 3.5,
        lower_asymptote: Union[float, Iterable[float]] = 0.01,
        lapse_rate: Union[float, Iterable[float]] = 0.01,
        scale: str = 'log10') -> np.ndarray:
    contrast = np.array(contrast, dtype='float64')
    spatial_freq = np.array(spatial_freq, dtype='float64')
    temporal_freq = np.array(temporal_freq, dtype='float64')
    c0 = np.array(c0, dtype='float64')
    cf = np.array(cf, dtype='float64')
    cw = np.array(cw, dtype='float64')
    min_thresh = np.array(min_thresh, dtype='float64')
    slope = np.array(slope, dtype='float64')
    lower_asymptote = np.array(lower_asymptote, dtype='float64')
    lapse_rate = np.array(lapse_rate, dtype='float64')

    t, c0_, cf_, cw_, f, w = np.meshgrid(min_thresh, c0, cf, cw,
                                         spatial_freq, temporal_freq)
    threshold = np.max([t,
                        c0_ +
                        cf_ * f +
                        cw_ * w])

    p = weibull(intensity=contrast,
                threshold=threshold,
                slope=slope,
                lower_asymptote=lower_asymptote,
                lapse_rate=lapse_rate,
                scale=scale)

    return p


# def norm_cdf(*,
#              intensity: Union[float, Iterable[float]],
#              mean: Union[float, Iterable[float]],
#              sd: Union[float, Iterable[float]],
#              lapse_rate: Union[float, Iterable[float]], ):
#     intensity = np.array(intensity, dtype='float64')
#     sd = np.array(sd, dtype='float64')
#     lapse_rate = np.array(lapse_rate, dtype='float64')
#
#     x, mu, sd_, delta = np.meshgrid(intensity,
#                                     mean,
#                                     sd,
#                                     lapse_rate,
#                                     indexing='ij', sparse=True)
#
#     assert np.atleast_1d(intensity.squeeze()).shape == np.atleast_1d(intensity).shape
#     assert np.atleast_1d(x.squeeze()).shape == np.atleast_1d(intensity).shape
#     assert np.atleast_1d(sd_.squeeze()).shape == np.atleast_1d(sd).shape
#     assert np.atleast_1d(delta.squeeze()).shape == np.atleast_1d(lapse_rate).shape
#
#     p = delta + (1 - 2*delta) * scipy.stats.norm.cdf(x, mu, sd)
#     return p
