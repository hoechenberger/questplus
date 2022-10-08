from typing import Union, Iterable
import numpy as np
from numpy.typing import ArrayLike
import scipy.stats
import xarray as xr


def weibull(
    *,
    intensity: Union[float, Iterable[float]],
    threshold: Union[float, Iterable[float]],
    slope: Union[float, Iterable[float]] = 3.5,
    lower_asymptote: Union[float, Iterable[float]] = 0.01,
    lapse_rate: Union[float, Iterable[float]] = 0.01,
    scale: str = "log10",
) -> xr.DataArray:
    r"""
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

    x = xr.DataArray(
        data=intensity, dims=["intensity"], coords=dict(intensity=intensity)
    )
    t = xr.DataArray(
        data=threshold, dims=["threshold"], coords=dict(threshold=threshold)
    )
    beta = xr.DataArray(data=slope, dims=["slope"], coords=dict(slope=slope))
    gamma = xr.DataArray(
        data=lower_asymptote,
        dims=["lower_asymptote"],
        coords=dict(lower_asymptote=lower_asymptote),
    )
    delta = xr.DataArray(
        data=lapse_rate, dims=["lapse_rate"], coords=dict(lapse_rate=lapse_rate)
    )
    assert np.atleast_1d(x.squeeze()).shape == np.atleast_1d(intensity).shape
    assert np.atleast_1d(t.squeeze()).shape == np.atleast_1d(threshold).shape
    assert np.atleast_1d(beta.squeeze()).shape == np.atleast_1d(slope).shape
    assert np.atleast_1d(gamma.squeeze()).shape == np.atleast_1d(lower_asymptote).shape
    assert np.atleast_1d(delta.squeeze()).shape == np.atleast_1d(lapse_rate).shape

    if scale == "linear":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-((x / t) ** beta))
    elif scale == "log10":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(10 ** (beta * (x - t))))
    elif scale == "dB":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(10 ** (beta * (x - t) / 20)))
    else:
        raise ValueError("Invalid scale specified.")

    return p


def csf(
    *,
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
    scale: str = "log10",
) -> np.ndarray:
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

    x = xr.DataArray(data=contrast, dims=["contrast"], coords=dict(contrast=contrast))
    f = xr.DataArray(
        data=spatial_freq, dims=["spatial_freq"], coords=dict(spatial_freq=spatial_freq)
    )
    w = xr.DataArray(
        data=temporal_freq,
        dims=["temporal_freq"],
        coords=dict(temporal_freq=temporal_freq),
    )
    c0_ = xr.DataArray(data=c0, dims=["c0"], coords=dict(c0=c0))
    cf_ = xr.DataArray(data=cf, dims=["cf"], coords=dict(cf=cf))
    cw_ = xr.DataArray(data=cw, dims=["cw"], coords=dict(cw=cw))
    min_t = xr.DataArray(
        data=min_thresh, dims=["min_thresh"], coords=dict(min_thresh=min_thresh)
    )
    beta = xr.DataArray(data=slope, dims=["slope"], coords=dict(slope=slope))
    gamma = xr.DataArray(
        data=lower_asymptote,
        dims=["lower_asymptote"],
        coords=dict(lower_asymptote=lower_asymptote),
    )
    delta = xr.DataArray(
        data=lapse_rate, dims=["lapse_rate"], coords=dict(lapse_rate=lapse_rate)
    )

    t = np.maximum(min_t, c0_ + cf_ * f + cw_ * w)

    # p = weibull(intensity=contrast,
    #             threshold=threshold,
    #             slope=slope,
    #             lower_asymptote=lower_asymptote,
    #             lapse_rate=lapse_rate,
    #             scale=scale)

    if scale == "linear":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-((x / t) ** beta))
    elif scale == "log10":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(10 ** (beta * (x - t))))
    elif scale == "dB":
        p = 1 - delta - (1 - gamma - delta) * np.exp(-(10 ** (beta * (x - t) / 20)))
    else:
        raise ValueError("Invalid scale specified.")

    return p


def norm_cdf(
    *,
    intensity: Union[float, Iterable[float]],
    mean: Union[float, Iterable[float]],
    sd: Union[float, Iterable[float]],
    lower_asymptote: Union[float, Iterable[float]] = 0.01,
    lapse_rate: Union[float, Iterable[float]] = 0.01,
    scale: str = "linear",
):
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
    if scale != "linear":
        msg = (
            "Currently, only linear stimulus scaling is supported for this "
            "psychometric function."
        )
        raise ValueError(msg)

    intensity = np.atleast_1d(intensity)
    mean = np.atleast_1d(mean)
    sd = np.atleast_1d(sd)
    lower_asymptote = np.atleast_1d(lower_asymptote)
    lapse_rate = np.atleast_1d(lapse_rate)

    x = xr.DataArray(
        data=intensity, dims=["intensity"], coords=dict(intensity=intensity)
    )
    mu = xr.DataArray(data=mean, dims=["mean"], coords=dict(mean=mean))
    sd_ = xr.DataArray(data=sd, dims=["sd"], coords=dict(sd=sd))
    gamma = xr.DataArray(
        data=lower_asymptote,
        dims=["lower_asymptote"],
        coords=dict(lower_asymptote=lower_asymptote),
    )
    delta = xr.DataArray(
        data=lapse_rate, dims=["lapse_rate"], coords=dict(lapse_rate=lapse_rate)
    )

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
        return gamma + (1 - gamma - delta) * norm.cdf(x)

    p = xr.apply_ufunc(_mu_func, x, mu, sd_, gamma, delta)
    return p


def norm_cdf_2(
    *,
    intensity: Union[float, Iterable[float]],
    mean: Union[float, Iterable[float]],
    sd: Union[float, Iterable[float]],
    lapse_rate: Union[float, Iterable[float]] = 0.01,
    scale: str = "linear",
):
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
    if scale != "linear":
        msg = (
            "Currently, only linear stimulus scaling is supported for this "
            "psychometric function."
        )
        raise ValueError(msg)

    intensity = np.atleast_1d(intensity)
    mean = np.atleast_1d(mean)
    sd = np.atleast_1d(sd)
    lapse_rate = np.atleast_1d(lapse_rate)

    x = xr.DataArray(
        data=intensity, dims=["intensity"], coords=dict(intensity=intensity)
    )
    mu = xr.DataArray(data=mean, dims=["mean"], coords=dict(mean=mean))
    sd_ = xr.DataArray(data=sd, dims=["sd"], coords=dict(sd=sd))
    delta = xr.DataArray(
        data=lapse_rate, dims=["lapse_rate"], coords=dict(lapse_rate=lapse_rate)
    )

    def _mu_func(x, mu, sd_, delta):
        norm = scipy.stats.norm(loc=mu, scale=sd_)
        return delta + (1 - 2 * delta) * norm.cdf(x)

    p = xr.apply_ufunc(_mu_func, x, mu, sd_, delta)
    return p


def scaling_function(
    *,
    x: Union[ArrayLike, float],
    m: float,
    mag_min: float = 0,
    mag_max: float = 1,
    t: float,
    q: float,
) -> ArrayLike:
    """
    The scaling function.

    Parameters
    ----------
    x
        This pyhysical stimulus magnitude(s).
    m
        The maximum value of the subjective scale.
    mag_min
        The minimum value of the physical stimulus magnitude.
    mag_max
        The maximum value of the physical stimulus magnitude.
    t
        The threshold value (physical stimulus magnitude at which the
        participant starts to perceive the stimulus).
    q
        The power exponent.

    Returns
    -------
    result
        The subjectively perceived intensities corresponding to the physical
        stimulus magnitudes.
    """
    # x = np.atleast_1d(x)
    # m = np.atleast_1d(m)
    # mag_min = np.atleast_1d(mag_min)
    # mag_max = np.atleast_1d(mag_max)
    # t = np.atleast_1d(t)
    # q = np.atleast_1d(q)
    #
    # assert len(mag_min) == len(mag_max) == 1

    nom = np.maximum(mag_min, x - t)
    denom = mag_max - t

    result = m * (nom / denom) ** q
    return result


def thurstone_scaling_function(
    *,
    physical_magnitudes_stim_1: Union[ArrayLike, float],
    physical_magnitudes_stim_2: Union[ArrayLike, float],
    threshold: Union[ArrayLike, float],
    power: Union[ArrayLike, float],
    perceptual_scale_max: Union[ArrayLike, float],
) -> ArrayLike:
    """
    The Thurstone scaling function.

    Parameters
    ----------
    physical_magnitudes_stim_1, physical_magnitudes_stim_2
        This pyhysical stimulus magnitudes the participant is asked to
        compare. All possible pairings will be generated automatically.
        The values in each array must be unique.
    threshold
        The threshold value (physical stimulus magnitude at which the
        participant starts to perceive the stimulus).
    power
        The power exponent.
    perceptual_scale_max
        The maximum value of the subjective perceptual scale (in JND / S.D.).

    Returns
    -------
    """
    physical_magnitudes_stim_1 = np.atleast_1d(physical_magnitudes_stim_1)
    physical_magnitudes_stim_2 = np.atleast_1d(physical_magnitudes_stim_2)
    threshold = np.atleast_1d(threshold)
    power = np.atleast_1d(power)
    perceptual_scale_max = np.atleast_1d(perceptual_scale_max)

    # assert np.allclose(physical_magnitudes_stim_1, physical_magnitudes_stim_2)
    # mag_min = x1.min()
    # mag_max = x2.max()

    # assert len(physical_magnitudes_stim_1) == len(physical_magnitudes_stim_2)
    # assert np.allclose(physical_magnitudes_stim_1.min(), physical_magnitudes_stim_2.min())
    # assert np.allclose(physical_magnitudes_stim_1.max(), physical_magnitudes_stim_2.max())

    if not np.array_equal(
        np.unique(physical_magnitudes_stim_1), np.sort(physical_magnitudes_stim_1)
    ):
        raise ValueError(f"Values in physical_magnitudes_stim_1 must be unique.")
    if not np.array_equal(
        np.unique(physical_magnitudes_stim_2), np.sort(physical_magnitudes_stim_2)
    ):
        raise ValueError(f"Values in physical_magnitudes_stim_2 must be unique.")

    # mag_min = np.min([physical_magnitudes_stim_1, physical_magnitudes_stim_2])
    mag_min = 0
    mag_max = np.hstack([
        physical_magnitudes_stim_1,
        physical_magnitudes_stim_2
    ]).max()

    physical_magnitudes_stim_1 = xr.DataArray(
        data=physical_magnitudes_stim_1,
        dims=["physical_magnitude_stim_1"],
        coords={"physical_magnitude_stim_1": physical_magnitudes_stim_1},
    )
    physical_magnitudes_stim_2 = xr.DataArray(
        data=physical_magnitudes_stim_2,
        dims=["physical_magnitude_stim_2"],
        coords={"physical_magnitude_stim_2": physical_magnitudes_stim_2},
    )
    threshold = xr.DataArray(
        data=threshold, dims=["threshold"], coords={"threshold": threshold}
    )
    power = xr.DataArray(data=power, dims=["power"], coords={"power": power})
    perceptual_scale_max = xr.DataArray(
        data=perceptual_scale_max,
        dims=["perceptual_scale_max"],
        coords={"perceptual_scale_max": perceptual_scale_max},
    )

    scale_x1 = scaling_function(
        x=physical_magnitudes_stim_1,
        m=perceptual_scale_max,
        mag_min=mag_min,
        mag_max=mag_max,
        t=threshold,
        q=power,
    )
    scale_x2 = scaling_function(
        x=physical_magnitudes_stim_2,
        m=perceptual_scale_max,
        mag_min=mag_min,
        mag_max=mag_max,
        t=threshold,
        q=power,
    )

    def _mu_func(scale_x1, scale_x2):
        return scipy.stats.norm.cdf((scale_x1 - scale_x2) / np.sqrt(2))

    result = xr.apply_ufunc(_mu_func, scale_x1, scale_x2)
    return result
