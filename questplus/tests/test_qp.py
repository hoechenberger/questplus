import pytest
import scipy.stats
import numpy as np
from questplus.qp import QuestPlus, QuestPlusWeibull, QuestPlusThurstone
from questplus import _constants


def test_threshold():
    """
    Watson 2017, Example 1:
    "Estimation of contrast threshold {1, 1, 2}"

    """
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    expected_contrasts = [-18, -22, -25, -28, -30, -22, -13, -15, -16, -18,
                          -19, -20, -21, -22, -23, -19, -20, -20, -18, -18,
                          -19, -17, -17, -18, -18, -18, -19, -19, -19, -19,
                          -19, -19]

    responses = ['Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct']

    expected_mode_threshold = -20

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    for expected_contrast, response in zip(expected_contrasts, responses):
        assert q.next_stim == dict(intensity=expected_contrast)
        q.update(stim=q.next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.param_estimate['threshold'],
                       expected_mode_threshold)


def test_threshold_slope():
    """
    Watson 2017, "Implementation"

    """
    threshold = np.arange(-40, 0 + 1)
    slope = np.arange(2, 5 + 1)
    guess = 0.5
    lapse = 0.02
    contrasts = threshold.copy()

    expected_mode_threshold = -20
    expected_mode_slope = 3

    expected_contrasts = [-18, -22, -12, -13, -15, -16, -17, -18, -19, -21,
                          -24, -25, -26, -27, -21, -22, -23, -20, -20, -20,
                          -21, -19, -19, -20, -20, -20, -19, -19, -19, -19,
                          -19, -20, -20, -20, -20, -20, -21, -21, -20, -20,
                          -20, -19, -18, -18, -19, -19, -19, -18, -18, -17,
                          -18, -18, -18, -17, -17, -17, -17, -17, -17, -17,
                          -18, -18, -18, -18]

    responses = ['Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Incorrect', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct']

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    for expected_contrast, response in zip(expected_contrasts, responses):
        assert q.next_stim == dict(intensity=expected_contrast)
        q.update(stim=q.next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.param_estimate['threshold'],
                       expected_mode_threshold)
    assert np.allclose(q.param_estimate['slope'],
                       expected_mode_slope)


def test_threshold_slope_lapse():
    """
    Watson 2017, Example 2:
    "Estimation of contrast threshold, slope, and lapse {1, 3, 2}"

    """
    expected_mode_threshold = -20
    expected_mode_slope = 5
    expected_mode_lapse = 0.04

    contrasts = np.arange(-40, 0 + 1)
    slope = np.arange(2, 5 + 1)
    lower_asymptote = 0.5
    lapse_rate = np.arange(0, 0.04 + 0.01, 0.01)

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=contrasts, slope=slope,
                        lower_asymptote=lower_asymptote, lapse_rate=lapse_rate)
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    expected_contrasts = [-18, -22, -25, -28, -30, -21, -12, -14, -15, -16,
                          -18, -19, -20, -22, -23, -18, -19, -16, -17, -17,
                          -18, -18, -18, -19, -19, -17, -17, -18, -18, -18,
                          -18, -19, -19, -20, -18, -18, -18, -18, -18, -19,
                          -19, -19, -19, -20, -18, -18, -18, -18, -18, -18,
                          -18, -18, -18, -19, -19, -19, -20, -18, -18, -18,
                          -18, -19, -19, -19, -18, -18, -18, -18, -19, -19,
                          -19, -19, -19, -19, -19, -19, -20, -21, -19, -19,
                          -19, -19, -19, -19, -19, -18, -19, -19, -16, -16,
                          -16, -16, -16, -16, 0, 0, -19, -19, -19, -19,
                          -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
                          -21, -21, -19, -19, -19, -19, -19, -19, -19, -19,
                          -19, -19, -19, -19, -19, -19, -19, -19]

    responses = ['Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Correct']

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    for expected_contrast, response in zip(expected_contrasts, responses):
        assert q.next_stim == dict(intensity=expected_contrast)
        q.update(stim=q.next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.param_estimate['threshold'],
                       expected_mode_threshold)
    assert np.allclose(q.param_estimate['slope'],
                       expected_mode_slope)
    assert np.allclose(q.param_estimate['lapse_rate'],
                       expected_mode_lapse)


def test_mean_sd_lapse():
    """
    Watson 2017, Example 3:
    "Estimation of mean, standard deviation, and lapse {1, 3, 2}"

    """
    expected_mode_mean = 0
    expected_mode_sd = 4
    expected_mode_lapse_rate = 0.01

    # Stimulus domain.
    orientation = np.arange(-10, 10+1)
    stim_domain = dict(intensity=orientation)

    # Parameter domain.
    orientation = np.arange(-5, 5+1)
    sd = np.arange(1, 10+1)
    lapse_rate = np.arange(0, 0.04 + 0.01, 0.01)
    param_domain = dict(mean=orientation, sd=sd,
                        lapse_rate=lapse_rate)

    # Outcome domain.
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'norm_cdf_2'
    scale = 'linear'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    expected_orientations = [ 0, -1,  2, -2,  2,  3,  3, -5,  3,  3,  3,  3,
                             -7, -7, -7, -7,  4, -7, 4,  4,  4, -8, -8, -8,
                              4, -8,  4, -8, -7,  4, -7, -7,  4, -7,  4, -7,
                             -6,  3, -4, -5,  3,  3, -7,  2,  2,  1,  0, -7,
                             -6, -6, -6,  2, -5,  2, -6, -5,  3, -5, -5,  3,
                              4, -5, -5,  4, -4, -4, -4, -4,  4,  3,  3,  4,
                              5,  5, -5,  5, -5, -5,  4,  5,  5, -5, -4, -4,
                             -4, -4,  5, -4, -4, -3,  4,  4,  5,  5, -4,  5,
                             -4, -4,  5,  4,  5,  5, -5,  5, -4, -4, -4, -4,
                              5, -4,  5,  4, -4,  4, -4, -4, -4,  4, -4, -4,
                             -4,  4,  4, -4,  4,  4,  4, -4]

    responses = ['Correct', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Correct', 'Incorrect', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Incorrect', 'Incorrect',
                 'Correct', 'Incorrect', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Incorrect',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Correct',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Correct', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Correct', 'Incorrect', 'Incorrect', 'Correct',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Incorrect', 'Incorrect', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct',
                 'Incorrect', 'Incorrect', 'Incorrect', 'Incorrect',
                 'Correct', 'Correct', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Incorrect', 'Incorrect',
                 'Incorrect', 'Correct', 'Incorrect', 'Incorrect',
                 'Incorrect', 'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect']

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    for expected_orientation, response in zip(expected_orientations, responses):
        next_stim = q.next_stim
        assert next_stim == dict(intensity=expected_orientation)
        q.update(stim=next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.param_estimate['mean'],
                       expected_mode_mean)
    assert np.allclose(q.param_estimate['sd'],
                       expected_mode_sd)
    assert np.allclose(q.param_estimate['lapse_rate'],
                       expected_mode_lapse_rate)


def test_spatial_contrast_sensitivity():
    """
    Watson 2017, Example 4:
    "Contrast sensitivity function {2, 3, 2}"

    """
    spatial_freqs = np.arange(0, 40 + 2, 2)
    contrasts = np.arange(-50, 0 + 2, 2)
    temporal_freq = 0

    # Stimulus domain.
    stim_domain = dict(contrast=contrasts,
                       spatial_freq=spatial_freqs,
                       temporal_freq=temporal_freq)

    # Parameter domain.
    min_threshs = np.arange(-50, -30 + 2, 2)
    c0s = np.arange(-60, -40 + 2, 2)
    cfs = np.arange(0.8, 1.6 + 0.2, 0.2)
    cw = 0
    slope = 3
    lower_asymptote = 0.5
    lapse_rate = 0.01

    param_domain = dict(min_thresh=min_threshs,
                        c0=c0s,
                        cf=cfs,
                        cw=cw,
                        slope=slope,
                        lower_asymptote=lower_asymptote,
                        lapse_rate=lapse_rate)

    # Outcome domain.
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'csf'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    expected_mode_min_thresh = -32
    expected_mode_c0 = -56
    expected_mode_cf = 1.4

    expected_contrasts = [0, -4, 0, 0, -38, 0, -40, 0, -26, -26,
                          0, -36, -36, 0, -26, -26, -2, -26, -6, -26,
                          0, -26, 0, -26, -32, -32, -34, -34, 0, -26,
                          0, -26]

    expected_spatial_freqs = [40, 40, 34, 36, 0, 38, 0, 38, 18, 18, 40, 0,
                              0, 40, 20, 20, 40, 22, 40, 20, 40, 18, 40, 18,
                              0, 0, 0, 0, 40, 18, 38, 18]

    responses = ['Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Incorrect',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Correct']

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    for x in zip(expected_contrasts, expected_spatial_freqs, responses):
        expected_contrast, expected_spatial_freq, response = x

        assert q.next_stim == dict(contrast=expected_contrast,
                                   spatial_freq=expected_spatial_freq,
                                   temporal_freq=0)
        q.update(stim=q.next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.param_estimate['min_thresh'],
                       expected_mode_min_thresh)
    assert np.allclose(q.param_estimate['c0'],
                       expected_mode_c0)
    assert np.allclose(q.param_estimate['cf'],
                       expected_mode_cf)


def test_thurstone_scaling():
    """
    Watson 2017, Example 6:
    "Thurstone scaling {2, 3, 2}"
    """
    stim_magnitudes = np.arange(0, 1+0.1, 0.1)
    perceptual_scale_maxs = np.arange(1, 10+1)
    thresholds = np.arange(0, 0.9+0.1, 0.1)
    powers = np.arange(0.1, 1+0.1, 0.1)

    # Due to differences in rounding, the order of stimuli (1 or 2) is swapped on some trials
    # compared to the paper. We therefore have to swap the example response as well.
    #
    # We're only testing the first 22 trials here.
    responses = ['Second'] * 6
    responses.extend(['Second'])       # rounding difference
    responses.extend(['Second'] * 13)
    responses.extend(['Second'])       # rounding difference
    responses.extend(['First'])

    expected_stims = [
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.7},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.6},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.5},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.4},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.0},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.4},
        {'physical_magnitude_stim_1': 0.0, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.5, 'physical_magnitude_stim_2': 1.0},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.5, 'physical_magnitude_stim_2': 1.0},
        {'physical_magnitude_stim_1': 0.5, 'physical_magnitude_stim_2': 1.0},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.5, 'physical_magnitude_stim_2': 1.0},
        {'physical_magnitude_stim_1': 0.6, 'physical_magnitude_stim_2': 1.0},
        {'physical_magnitude_stim_1': 0.2, 'physical_magnitude_stim_2': 0.3},
        {'physical_magnitude_stim_1': 0.6, 'physical_magnitude_stim_2': 1.0},
    ]

    qp = QuestPlusThurstone(
        physical_magnitudes_stim_1=stim_magnitudes,
        physical_magnitudes_stim_2=stim_magnitudes,
        thresholds=thresholds,
        powers=powers,
        perceptual_scale_maxs=perceptual_scale_maxs
    )

    for trial_idx, x in enumerate(zip(expected_stims, responses)):
        expected_stim, response = x
        
        expected_stim_1 = expected_stim['physical_magnitude_stim_1']
        expected_stim_2 = expected_stim['physical_magnitude_stim_2']
        
        next_stim_1 =  qp.next_stim['physical_magnitude_stim_1']
        next_stim_2 =  qp.next_stim['physical_magnitude_stim_2']

        if trial_idx in (6, 20):
            # Rounding errors make the algorithm behave differently on different platforms.
            if (
                expected_stim_1 == next_stim_2 and
                expected_stim_2 == next_stim_1
            ):
                expected_stim_1, expected_stim_2 = expected_stim_2, expected_stim_1
                response = 'First' if response == 'Second' else 'Second'

        assert np.isclose(next_stim_1, expected_stim_1)
        assert np.isclose(next_stim_2, expected_stim_2)
        qp.update(stim=qp.next_stim, response=response)


def test_weibull():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    expected_contrasts = [-18, -22, -25, -28, -30, -22, -13, -15, -16, -18,
                          -19, -20, -21, -22, -23, -19, -20, -20, -18, -18,
                          -19, -17, -17, -18, -18, -18, -19, -19, -19, -19,
                          -19, -19]

    responses = ['Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Incorrect',
                 'Correct', 'Correct', 'Incorrect', 'Correct', 'Correct',
                 'Incorrect', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct']

    expected_mode_threshold = -20

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    q_weibull = QuestPlusWeibull(intensities=stim_domain['intensity'],
                                 thresholds=param_domain['threshold'],
                                 slopes=param_domain['slope'],
                                 lower_asymptotes=param_domain['lower_asymptote'],
                                 lapse_rates=param_domain['lapse_rate'],
                                 responses=outcome_domain['response'],
                                 stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        assert q.next_stim['intensity'] == q_weibull.next_intensity
        assert q_weibull.next_intensity == expected_contrast
        q.update(stim=q.next_stim,
                 outcome=dict(response=response))
        q_weibull.update(intensity=q_weibull.next_intensity,
                         response=response)

    assert np.allclose(q.param_estimate['threshold'],
                       expected_mode_threshold)


def test_eq():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    q1 = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                   outcome_domain=outcome_domain, func=f, stim_scale=scale,
                   stim_selection_method=stim_selection_method,
                   param_estimation_method=param_estimation_method)

    q2 = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                   outcome_domain=outcome_domain, func=f, stim_scale=scale,
                   stim_selection_method=stim_selection_method,
                   param_estimation_method=param_estimation_method)

    # Add some random responses.
    q1.update(stim=q1.next_stim, outcome=dict(response='Correct'))
    q1.update(stim=q1.next_stim, outcome=dict(response='Incorrect'))
    q2.update(stim=q2.next_stim, outcome=dict(response='Correct'))
    q2.update(stim=q2.next_stim, outcome=dict(response='Incorrect'))

    assert q1 == q2


def test_json():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method)

    # Add some random responses.
    q.update(stim=q.next_stim, outcome=dict(response='Correct'))
    q.update(stim=q.next_stim, outcome=dict(response='Incorrect'))

    q_dumped = q.to_json()
    q_loaded = QuestPlus.from_json(q_dumped)

    assert q_loaded == q

    q_loaded.update(stim=q_loaded.next_stim, outcome=dict(response='Correct'))


def test_json_rng():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_n_entropy'
    param_estimation_method = 'mode'
    random_seed = 5
    stim_selection_options = dict(n=3, random_seed=random_seed)

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method,
                  stim_selection_options=stim_selection_options)

    q2 = QuestPlus.from_json(q.to_json())

    rand = q._rng.random_sample(10)
    rand2 = q2._rng.random_sample(10)

    assert np.allclose(rand, rand2)


def test_marginal_posterior():
    contrasts = np.arange(-40, 0 + 1)
    slope = np.arange(2, 5 + 1)
    lower_asymptote = (0.5,)
    lapse_rate = np.arange(0, 0.04 + 0.01, 0.01)

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=contrasts, slope=slope,
                        lower_asymptote=lower_asymptote, lapse_rate=lapse_rate)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    func = 'weibull'
    stim_scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain,
                  param_domain=param_domain,
                  outcome_domain=outcome_domain,
                  func=func, stim_scale=stim_scale)

    marginal_posterior = q.marginal_posterior

    assert np.allclose(marginal_posterior['threshold'],
                       np.ones(len(contrasts)) / len(contrasts))
    assert np.allclose(marginal_posterior['slope'],
                       np.ones(len(slope)) / len(slope))
    assert np.allclose(marginal_posterior['lower_asymptote'],
                       np.ones(len(lower_asymptote)) / len(lower_asymptote))
    assert np.allclose(marginal_posterior['lapse_rate'],
                       np.ones(len(lapse_rate)) / len(lapse_rate))


def test_prior_for_unknown_parameter():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    prior = dict(Foo=[1, 2, 3])

    with pytest.raises(ValueError):
        q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                      outcome_domain=outcome_domain, func=f, stim_scale=scale,
                      stim_selection_method=stim_selection_method,
                      param_estimation_method=param_estimation_method,
                      prior=prior)


def test_prior_for_parameter_subset():
    threshold = np.arange(-40, 0 + 1)
    slopes = np.linspace(start=2, stop=4, num=11)
    guess, lapse = 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slopes,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_entropy'
    param_estimation_method = 'mode'

    threshold_prior = scipy.stats.norm.pdf(contrasts, loc=-20, scale=10)
    threshold_prior /= threshold_prior.sum()
    
    slope_prior = scipy.stats.norm.pdf(slopes, loc=3, scale=0.5)
    slope_prior /= slope_prior.sum()

    prior = dict(threshold=threshold_prior,
                 slope=slope_prior)

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale,
                  stim_selection_method=stim_selection_method,
                  param_estimation_method=param_estimation_method,
                  prior=prior)
    
    assert np.allclose(threshold_prior,
                       q.prior.sum(dim=['slope', 'lower_asymptote',
                                        'lapse_rate']).values)

    assert np.allclose(slope_prior,
                       q.prior.sum(dim=['threshold', 'lower_asymptote',
                                        'lapse_rate']).values)

    assert np.isclose(1,
                      q.prior.sum(dim=['threshold', 'slope',
                                       'lapse_rate']).sum())

    assert np.isclose(1,
                      q.prior.sum(dim=['threshold', 'slope',
                                       'lower_asymptote']).sum())


def test_stim_selection_options():
    threshold = np.arange(-40, 0 + 1)
    slope, guess, lapse = 3.5, 0.5, 0.02
    contrasts = threshold.copy()

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])

    f = 'weibull'
    scale = 'dB'
    stim_selection_method = 'min_n_entropy'
    param_estimation_method = 'mode'

    common_params = dict(stim_domain=stim_domain, param_domain=param_domain,
                         outcome_domain=outcome_domain, func=f,
                         stim_scale=scale,
                         stim_selection_method=stim_selection_method,
                         param_estimation_method=param_estimation_method)

    stim_selection_options = None
    q = QuestPlus(**common_params,
                  stim_selection_options=stim_selection_options)
    expected = dict(n=_constants.DEFAULT_N,
                    max_consecutive_reps=_constants.DEFAULT_MAX_CONSECUTIVE_REPS,
                    random_seed=_constants.DEFAULT_RANDOM_SEED)
    assert expected == q.stim_selection_options

    stim_selection_options = dict(n=5)
    q = QuestPlus(**common_params,
                  stim_selection_options=stim_selection_options)
    expected = dict(n=5,
                    max_consecutive_reps=_constants.DEFAULT_MAX_CONSECUTIVE_REPS,
                    random_seed=_constants.DEFAULT_RANDOM_SEED)
    assert expected == q.stim_selection_options

    stim_selection_options = dict(max_consecutive_reps=4)
    q = QuestPlus(**common_params,
                  stim_selection_options=stim_selection_options)
    expected = dict(n=_constants.DEFAULT_N,
                    max_consecutive_reps=4,
                    random_seed=_constants.DEFAULT_RANDOM_SEED)
    assert expected == q.stim_selection_options

    stim_selection_options = dict(random_seed=999)
    q = QuestPlus(**common_params,
                  stim_selection_options=stim_selection_options)
    expected = dict(n=_constants.DEFAULT_N,
                    max_consecutive_reps=_constants.DEFAULT_MAX_CONSECUTIVE_REPS,
                    random_seed=999)
    assert expected == q.stim_selection_options

    stim_selection_options = dict(n=5, max_consecutive_reps=4, random_seed=999)
    q = QuestPlus(**common_params,
                  stim_selection_options=stim_selection_options)
    expected = stim_selection_options.copy()
    assert expected == q.stim_selection_options


def test_weibull_prior():
    intensities = np.linspace(-10, 0)
    thresholds = intensities.copy()
    slopes = [3.5]
    lower_asymptotes = [0.01]
    lapse_rates = [0.01]

    prior_val = scipy.stats.norm.pdf(intensities, loc=-5, scale=0.2)
    prior_val /= prior_val.sum()
    prior = dict(threshold=prior_val)

    q = QuestPlusWeibull(intensities=intensities,
                         thresholds=thresholds,
                         slopes=slopes,
                         lower_asymptotes=lower_asymptotes,
                         lapse_rates=lapse_rates,
                         prior=prior)

    assert np.allclose(q.prior.squeeze().values, prior_val)


if __name__ == '__main__':
    test_threshold()
    test_threshold_slope()
    test_threshold_slope_lapse()
    test_mean_sd_lapse()
    test_spatial_contrast_sensitivity()
    test_weibull()
    test_eq()
    test_json()
    test_json_rng()
    test_marginal_posterior()
    test_prior_for_unknown_parameter()
    test_prior_for_parameter_subset()
    test_stim_selection_options()
    test_weibull_prior()
