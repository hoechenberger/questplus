from typing import Union, Iterable
import numpy as np
import scipy.stats
import xarray as xr


def weibull(*,
            intensity: Union[float, Iterable[float]],
            threshold: Union[float, Iterable[float]],
            slope: Union[float, Iterable[float]] = 3.5,
            lower_asymptote: Union[float, Iterable[float]] = 0.01,
            lapse_rate: Union[float, Iterable[float]] = 0.01,
            scale: str = 'log10') -> xr.DataArray:
    """
    A Weibull psychometric function.

    Parameters
    ----------
    intensity
        Stimulus values on the abscissa, :math:`x`.

    threshold
        The threshold parameter, :math:`\\alpha`.

    slope
        The slope parameter, :math:`\\beta`.

    lower_asymptote
        The lower asymptote, :math:`\\gamma`, which is equivalent to the
        false-alarm rate in a yes-no task, or :math:`\\frac{1}{n}` in an
        :math:`n`-AFC task.

    lapse_rate
        The lapse rate, :math:`\\delta`. The upper asymptote of the psychometric
        function will be :math:`1-\\delta`.

    scale
        The scale of the stimulus parameters. Possible values are ``log10``,
        ``dB``, and ``linear``.

    Returns
    -------
    p
        The psychometric function evaluated at the specified intensities for
        all parameters combinations.

    Notes
    -----
    An appropriate parametrization of the function is chosen based on the
    `scale` keyword argument. Specifically, the following parametrizations
    are used:

        scale='linear'
            :math:`p = 1 - \delta - (1 - \gamma - \delta)\\, e^{-\\left (\\frac{x}{t} \\right )^\\beta}`

        scale='log10'
            :math:`p = 1 - \delta - (1 - \gamma - \delta)\\, e^{-10^{\\beta (x - t)}}`

        scale='dB':
            :math:`p = 1 - \delta - (1 - \gamma - \delta)\\, e^{-10^{\\frac{\\beta}{20} (x - t)}}`

    """
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

    if scale == 'linear':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(x / t)**beta)
    elif scale == 'log10':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t)))
    elif scale == 'dB':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t) / 20))
    else:
        raise ValueError('Invalid scale specified.')

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
    """
    The spatio-temporal contrast sensitivity function.

    Parameters
    ----------
    contrast
    spatial_freq
    temporal_freq
    c0
    cf
    cw
    min_thresh
    slope
    lower_asymptote
    lapse_rate
    scale

    Returns
    -------

    """
    contrast = np.atleast_1d(contrast)
    spatial_freq = np.atleast_1d(spatial_freq)
    temporal_freq = np.atleast_1d(temporal_freq)
    c0 = np.atleast_1d(c0)
    cf = np.atleast_1d(cf)
    cw = np.atleast_1d(cw)
    min_thresh = np.atleast_1d(min_thresh)
    slope = np.atleast_1d(slope)
    lower_asymptote = np.atleast_1d(lower_asymptote)
    lapse_rate = np.atleast_1d(lapse_rate)

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
    min_t = xr.DataArray(data=min_thresh, dims=['min_thresh'],
                         coords=dict(min_thresh=min_thresh))
    beta = xr.DataArray(data=slope, dims=['slope'],
                        coords=dict(slope=slope))
    gamma = xr.DataArray(data=lower_asymptote, dims=['lower_asymptote'],
                         coords=dict(lower_asymptote=lower_asymptote))
    delta = xr.DataArray(data=lapse_rate, dims=['lapse_rate'],
                         coords=dict(lapse_rate=lapse_rate))

    t = np.maximum(min_t, c0_ + cf_ * f + cw_ * w)

    # p = weibull(intensity=contrast,
    #             threshold=threshold,
    #             slope=slope,
    #             lower_asymptote=lower_asymptote,
    #             lapse_rate=lapse_rate,
    #             scale=scale)

    if scale == 'linear':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(x / t)**beta)
    elif scale == 'log10':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t)))
    elif scale == 'dB':
        p = 1 - delta - (1 - gamma - delta) * np.exp(-10 ** (beta * (x - t) / 20))
    else:
        raise ValueError('Invalid scale specified.')

    return p


def norm_cdf(*,
             intensity: Union[float, Iterable[float]],
             mean: Union[float, Iterable[float]],
             sd: Union[float, Iterable[float]],
             lower_asymptote: Union[float, Iterable[float]] = 0.01,
             lapse_rate: Union[float, Iterable[float]] = 0.01,
             scale: str = 'linear'):
    """
    The cumulate normal distribution.

    Parameters
    ----------
    intensity
    mean
    sd
    lower_asymptote
    lapse_rate
    scale

    Returns
    -------

    """
    if scale != 'linear':
        msg = ('Currently, only linear stimulus scaling is supported for this '
               'psychometric function.')
        raise ValueError(msg)

    intensity = np.atleast_1d(intensity)
    mean = np.atleast_1d(mean)
    sd = np.atleast_1d(sd)
    lower_asymptote = np.atleast_1d(lower_asymptote)
    lapse_rate = np.atleast_1d(lapse_rate)

    x = xr.DataArray(data=intensity, dims=['intensity'],
                     coords=dict(intensity=intensity))
    mu = xr.DataArray(data=mean, dims=['mean'],
                      coords=dict(mean=mean))
    sd_ = xr.DataArray(data=sd, dims=['sd'],
                       coords=dict(sd=sd))
    gamma = xr.DataArray(data=lower_asymptote, dims=['lower_asymptote'],
                         coords=dict(lower_asymptote=lower_asymptote))
    delta = xr.DataArray(data=lapse_rate, dims=['lapse_rate'],
                         coords=dict(lapse_rate=lapse_rate))

    # x, mu, sd_, delta = np.meshgrid(intensity,
    #                                 mean,
    #                                 sd,
    #                                 lapse_rate,
    #                                 indexing='ij', sparse=True)
    #
    # assert np.atleast_1d(intensity.squeeze()).shape == np.atleast_1d(intensity).shape
    # assert np.atleast_1d(x.squeeze()).shape == np.atleast_1d(intensity).shape
    # assert np.atleast_1d(sd_.squeeze()).shape == np.atleast_1d(sd).shape
    # assert np.atleast_1d(delta.squeeze()).shape == np.atleast_1d(lapse_rate).shape

    # p = delta + (1 - 2*delta) * scipy.stats.norm.cdf(x, mu, sd_)

    def _mu_func(x, mu, sd_, gamma, delta):
        norm = scipy.stats.norm(loc=mu, scale=sd_)
        return delta + (1 - gamma - delta) * norm.cdf(x)

    p = xr.apply_ufunc(_mu_func, x, mu, sd_, gamma, delta)
    return p


def norm_cdf_2(*,
               intensity: Union[float, Iterable[float]],
               mean: Union[float, Iterable[float]],
               sd: Union[float, Iterable[float]],
               lapse_rate: Union[float, Iterable[float]] = 0.01,
               scale: str = 'linear'):
    """
    The cumulative normal distribution with lapse rate equal to lower
    asymptote.

    Parameters
    ----------
    intensity
    mean
    sd
    lapse_rate
    scale

    Returns
    -------

    """
    if scale != 'linear':
        msg = ('Currently, only linear stimulus scaling is supported for this '
               'psychometric function.')
        raise ValueError(msg)

    intensity = np.atleast_1d(intensity)
    mean = np.atleast_1d(mean)
    sd = np.atleast_1d(sd)
    lapse_rate = np.atleast_1d(lapse_rate)

    x = xr.DataArray(data=intensity, dims=['intensity'],
                     coords=dict(intensity=intensity))
    mu = xr.DataArray(data=mean, dims=['mean'],
                      coords=dict(mean=mean))
    sd_ = xr.DataArray(data=sd, dims=['sd'],
                       coords=dict(sd=sd))
    delta = xr.DataArray(data=lapse_rate, dims=['lapse_rate'],
                         coords=dict(lapse_rate=lapse_rate))

    def _mu_func(x, mu, sd_, delta):
        norm = scipy.stats.norm(loc=mu, scale=sd_)
        return delta + (1 - 2*delta) * norm.cdf(x)

    p = xr.apply_ufunc(_mu_func, x, mu, sd_, delta)
    return p
