from typing import Union, Iterable
import numpy as np
import xarray as xr


# import scipy.stats


def weibull(*,
            intensity: Union[float, Iterable[float]],
            threshold: Union[float, Iterable[float]],
            slope: Union[float, Iterable[float]] = 3.5,
            lower_asymptote: Union[float, Iterable[float]] = 0.01,
            lapse_rate: Union[float, Iterable[float]] = 0.01,
            scale: str = 'log10') -> xr.DataArray:
    intensity = np.atleast_1d(intensity)
    threshold = np.atleast_1d(threshold)
    slope = np.atleast_1d(slope)
    lower_asymptote = np.atleast_1d(lower_asymptote)
    lapse_rate = np.atleast_1d(lapse_rate)

    # Implementation using NumPy. Leave it here for reference.
    #
    # x, t, beta, gamma, delta = np.meshgrid(intensity,
    #                                        threshold,
    #                                        slope,
    #                                        lower_asymptote,
    #                                        lapse_rate,
    #                                        indexing='ij', sparse=True)

    x = xr.DataArray(data=intensity, dims=['intensity'],
                     coords=dict(intensity=intensity))
    t = xr.DataArray(data=threshold, dims=['threshold'],
                     coords=dict(threshold=threshold))
    beta = xr.DataArray(data=slope, dims=['slope'],
                        coords=dict(slope=slope))
    gamma = xr.DataArray(data=lower_asymptote, dims=['lower_asymptote'],
                         coords=dict(lower_asymptote=lower_asymptote))
    delta = xr.DataArray(data=lapse_rate, dims=['lapse_rate'],
                         coords=dict(lapse_rate=lapse_rate))
    assert np.atleast_1d(x.squeeze()).shape == np.atleast_1d(intensity).shape
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
        temporal_freq: Union[float, Iterable[float]],
        c0: Union[float, Iterable[float]],
        cf: Union[float, Iterable[float]],
        cw: Union[float, Iterable[float]],
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

    # Implementation using NumPy. Leave it here for reference.
    #
    # c, f, w, c0_, cf_, cw_, t, beta, gamma, delta = np.meshgrid(
    #     contrast, spatial_freq, temporal_freq, c0, cf, cw, min_thresh,
    #     slope, lower_asymptote, lapse_rate,
    #     indexing='ij', sparse=True)

    x = xr.DataArray(data=contrast, dims=['contrast'],
                     coords=dict(contrast=contrast))
    f = xr.DataArray(data=spatial_freq, dims=['spatial_freq'],
                     coords=dict(spatial_freq=spatial_freq))
    w = xr.DataArray(data=temporal_freq, dims=['temporal_freq'],
                     coords=dict(temporal_freq=temporal_freq))
    c0_ = xr.DataArray(data=c0, dims=['c0'],
                       coords=dict(c0=c0))
    cf_ = xr.DataArray(data=cf, dims=['cf'],
                       coords=dict(cf=cf))
    cw_ = xr.DataArray(data=cw, dims=['cw'],
                       coords=dict(cw=cw))
    t = xr.DataArray(data=min_thresh, dims=['min_thresh'],
                     coords=dict(min_thresh=min_thresh))
    beta = xr.DataArray(data=slope, dims=['slope'],
                        coords=dict(slope=slope))
    gamma = xr.DataArray(data=lower_asymptote, dims=['lower_asymptote'],
                         coords=dict(lower_asymptote=lower_asymptote))
    delta = xr.DataArray(data=lapse_rate, dims=['lapse_rate'],
                         coords=dict(lapse_rate=lapse_rate))

    threshold = np.maximum(t, c0_ + cf_ * f + cw_ * w)
    s = 1 if scale == 'log10' else 20

    # p = weibull(intensity=contrast,
    #             threshold=threshold,
    #             slope=slope,
    #             lower_asymptote=lower_asymptote,
    #             lapse_rate=lapse_rate,
    #             scale=scale)

    p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - threshold) / s))

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
