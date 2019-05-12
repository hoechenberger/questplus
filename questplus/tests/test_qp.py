import numpy as np
from questplus.qp import QuestPlus


def test_threshold():
    # Watson 2017, Example 1:
    # "Estimation of contrast threshold {1, 1, 2}"
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
    expected_mean_threshold = -20.39360311638961

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=threshold, slope=slope,
                        lower_asymptote=guess, lapse_rate=lapse)
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'weibull'
    scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        next_stim = q.next_stim(stim_selection='min_entropy')
        assert next_stim == dict(intensity=expected_contrast)
        q.update(stim=next_stim,
                 outcome=dict(response=response))

    assert np.allclose(q.get_param_estimates(method='mode')['threshold'],
                       expected_mode_threshold)
    assert np.allclose(q.get_param_estimates(method='mean')['threshold'],
                       expected_mean_threshold)


def test_threshold_slope():
    # Watson 2017, "Implementation"
    threshold = np.arange(-40, 0+1)
    slope = np.arange(2, 5+1)
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

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        next_stim = q.next_stim(stim_selection='min_entropy')
        assert next_stim == dict(intensity=expected_contrast)
        q.update(stim=next_stim,
                 outcome=dict(response=response))

    fitted_mode_params = q.get_param_estimates(method='mode')

    assert np.allclose(fitted_mode_params['threshold'],
                       expected_mode_threshold)
    assert np.allclose(fitted_mode_params['slope'],
                       expected_mode_slope)


def test_threshold_slope_lapse():
    # Watson 2017, Example 2:
    # "Estimation of contrast threshold, slope, and lapse {1, 3, 2}"
    expected_mode_threshold = -20
    expected_mode_slope = 5
    expected_mode_lapse = 0.04

    contrasts = np.arange(-40, 0 + 1)
    slope = np.arange(2, 5 + 1)
    lower_asymptote = 0.5
    lapse_rate = np.arange(0, 0.04+0.01, 0.01)

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=contrasts, slope=slope,
                        lower_asymptote=lower_asymptote, lapse_rate=lapse_rate)
    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'weibull'
    scale = 'dB'

    expected_contrasts = [-18, -22, -25, -28, -30, -21, -12, -14, -15, -16,
                          -18, -19, -20, -22, -23, -18, -19, -16, -17, -17,
                          -18, -18, -18, -19, -19, -17, -17, -18, -18, -18,
                          -18, -19, -19, -20, -18, -18, -18, -18, -18, -19,
                          -19, -19, -19, -20, -18, -18, -18, -18, -18, -18,
                          -18, -18, -18, -19, -19, -19, -20, -18, -18, -18,
                          -18, -19, -19, -19, -18, -18, -18, -18, -19, -19,
                          -19, -19, -19, -19, -19, -19, -20, -21, -19, -19,
                          -19, -19, -19, -19, -19, -18, -19, -19, -16, -16,
                          -16, -16, -16, -16,   0,   0, -19, -19, -19, -19,
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
                  outcome_domain=outcome_domain, func=f, stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        next_stim = q.next_stim(stim_selection='min_entropy')
        assert next_stim == dict(intensity=expected_contrast)
        q.update(stim=next_stim,
                 outcome=dict(response=response))

    fitted_mode_params = q.get_param_estimates(method='mode')

    assert np.allclose(fitted_mode_params['threshold'],
                       expected_mode_threshold)
    assert np.allclose(fitted_mode_params['slope'],
                       expected_mode_slope)
    assert np.allclose(fitted_mode_params['lapse_rate'],
                       expected_mode_lapse)


# def test_mean_sd_lapse():
#     # Watson 2017, Example 3:
#     # "Estimation of mean, standard deviation, and lapse {1, 3, 2}"
#     true_params = dict(mean=1,
#                        sd=3,
#                        lapse_rate=0.02)
#
#     orientation = np.arange(-10, 10+1)
#     stim_domain = dict(intensity=orientation)
#
#     orientation = np.arange(-5, 5+1)
#     sd = np.arange(1, 10+1)
#     lapse_rate = np.arange(0, 0.04 + 0.01, 0.01)
#     param_domain = dict(mean=orientation, sd=sd, lapse_rate=lapse_rate)
#
#     outcome_domain = ['Correct', 'Incorrect']
#     f = 'norm_cdf'
#     scale = 'dB'
#
#     q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
#                   outcome_domain=outcome_domain, func=f, stim_scale=scale)


def test_spatial_contrast_sensitivity():
    # Watson 2017, Example 4:
    # "Contrast sensitivity function {2, 3, 2}"

    spatial_freqs = np.arange(0, 40+2, 2)
    contrasts = np.arange(-50, 0+2, 2)
    temporal_freq = 0

    stim_domain = dict(contrast=contrasts,
                       spatial_freq=spatial_freqs,
                       temporal_freq=temporal_freq)

    min_threshs = np.arange(-50, -30+2, 2)
    c0s = np.arange(-60, -40+2, 2)
    cfs = np.arange(0.8, 1.6+0.2, 0.2)
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

    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'csf'
    scale = 'dB'

    expected_mode_min_thresh = -32
    expected_mode_c0 = -56
    expected_mode_cf = 1.4

    expected_contrasts = [0,  -4,   0,   0, -38,   0, -40,   0, -26, -26,
                          0, -36, -36,   0, -26, -26,  -2, -26,  -6, -26,
                          0, -26,   0, -26, -32, -32, -34, -34,   0, -26,
                          0, -26]

    expected_spatial_freqs = [40, 40, 34, 36,  0, 38,  0, 38, 18, 18, 40,  0,
                               0, 40, 20, 20, 40,  22, 40, 20, 40, 18, 40, 18,
                               0,  0,  0,  0, 40, 18, 38, 18]

    responses = ['Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Incorrect', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct',  'Incorrect', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Correct', 'Incorrect', 'Incorrect',
                 'Correct', 'Correct', 'Correct', 'Correct', 'Correct',
                 'Correct', 'Correct', 'Incorrect', 'Incorrect', 'Correct',
                 'Correct', 'Correct']

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale)

    for x in zip(expected_contrasts, expected_spatial_freqs, responses):
        expected_contrast, expected_spatial_freq, response = x

        next_stim = q.next_stim(stim_selection='min_entropy')
        assert next_stim == dict(contrast=expected_contrast,
                                 spatial_freq=expected_spatial_freq,
                                 temporal_freq=0)
        q.update(stim=next_stim,
                 outcome=dict(response=response))

    fitted_mode_params = q.get_param_estimates(method='mode')


    assert np.allclose(fitted_mode_params['min_thresh'],
                       expected_mode_min_thresh)
    assert np.allclose(fitted_mode_params['c0'],
                       expected_mode_c0)
    assert np.allclose(fitted_mode_params['cf'],
                       expected_mode_cf)


if __name__ == '__main__':
    # test_threshold()
    test_spatial_contrast_sensitivity()
