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

    responses = ["Correct", "Correct", "Correct", "Correct", "Incorrect",
                 "Incorrect", "Correct", "Correct", "Correct", "Correct",
                 "Correct", "Correct", "Correct", "Correct", "Incorrect",
                 "Correct", "Correct", "Incorrect", "Correct", "Correct",
                 "Incorrect", "Correct", "Correct", "Correct", "Correct",
                 "Correct", "Correct", "Correct", "Correct", "Correct",
                 "Correct", "Correct"]

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
        q.update(stimulus=next_stim,
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
        q.update(stimulus=next_stim,
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
        q.update(stimulus=next_stim,
                 outcome=dict(response=response))

    fitted_mode_params = q.get_param_estimates(method='mode')

    assert np.allclose(fitted_mode_params['threshold'],
                       expected_mode_threshold)
    assert np.allclose(fitted_mode_params['slope'],
                       expected_mode_slope)
    assert np.allclose(fitted_mode_params['lapse_rate'],
                       expected_mode_lapse)

    print(q.get_param_estimates(method='mean'))




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


def test_contrast_sensitivity():
    # Watson 2017, Example 4:
    # "Contrast sensitivity function {2, 3, 2}"

    true_params = dict(min_thresh=-35,
                       c0=-50,
                       cf=1.2,
                       slope=3,
                       lower_asymptote=0.5,
                       lapse_rate=0.01)

    spatial_freqs = np.arange(0, 40+2, 2)
    contrasts = np.arange(-50, 0+2, 2)

    stim_domain = dict(contrast=contrasts,
                       spatial_freq=spatial_freqs)

    min_threshs = np.arange(-50, -30+2, 2)
    c0s = np.arange(-60, -40+2, 2)
    cfs = np.arange(0.8, 1.6+0.2, 0.2)

    param_domain = dict(min_thresh=min_threshs,
                        c0=c0s,
                        cf=cfs)

    outcome_domain = dict(response=['Correct', 'Incorrect'])
    f = 'csf'
    scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  outcome_domain=outcome_domain, func=f, stim_scale=scale)
    q.next_stim()

if __name__ == '__main__':
    test_threshold_slope_lapse()
