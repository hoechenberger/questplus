import numpy as np
from questplus.qp import QuestPlus
from questplus.utils import simulate_response


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
    resp_domain = ['Correct', 'Incorrect']
    f = 'weibull'
    scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  resp_domain=resp_domain, func=f, stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        next_contrast = q.next_stim(method='min_entropy')
        assert next_contrast == expected_contrast
        q.update(stimulus=dict(intensity=next_contrast), response=response)

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
    expected_mean_threshold = -19.582425723172893
    expected_mean_slope = 3.3075361994603325

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
    resp_domain = ['Correct', 'Incorrect']
    f = 'weibull'
    scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  resp_domain=resp_domain, func=f, stim_scale=scale)

    for expected_contrast, response in zip(expected_contrasts, responses):
        next_contrast = q.next_stim(method='min_entropy')
        assert next_contrast == expected_contrast
        q.update(stimulus=dict(intensity=next_contrast), response=response)
        # print(q.get_param_estimates(method='mode')['threshold'])

    fitted_mode_params = q.get_param_estimates(method='mode')
    fitted_mean_params = q.get_param_estimates(method='mean')

    assert np.allclose(fitted_mode_params['threshold'],
                       expected_mode_threshold)
    assert np.allclose(fitted_mode_params['slope'],
                       expected_mode_slope)



    # assert np.allclose(fitted_mean_params['threshold'],
    #                    expected_mean_threshold)
    # assert np.allclose(fitted_mean_params['slope'],
    #                    expected_mean_slope)




def test_threshold_slope_lapse():
    # Watson 2017, Example 2:
    # "Estimation of contrast threshold, slope, and lapse {1, 3, 2}"
    true_params = dict(threshold=-20,
                       slope=3,
                       lower_asymptote=0.5,
                       lapse_rate=0.02)

    contrasts = np.arange(-40, 0 + 1)
    slope = np.arange(2, 5 + 1)
    lower_asymptote = 0.5
    lapse_rate = np.arange(0, 0.04 + 0.01, 0.01)

    stim_domain = dict(intensity=contrasts)
    param_domain = dict(threshold=contrasts, slope=slope,
                        lower_asymptote=lower_asymptote, lapse_rate=lapse_rate)
    resp_domain = ['Correct', 'Incorrect']
    f = 'weibull'
    scale = 'dB'

    q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
                  resp_domain=resp_domain, func=f, stim_scale=scale)

    for _ in range(64):
        next_contrast = q.next_stim(method='min_entropy')
        response = simulate_response(func=f,
                                     stimulus=dict(intensity=next_contrast),
                                     params=true_params,
                                     stim_scale=scale)
        q.update(stimulus=dict(intensity=next_contrast), response=response)

    estimated_params = q.get_param_estimates(method='mean')

    assert -22 <= estimated_params['threshold'] <= -18
    assert 2.5 <= estimated_params['slope'] <= 5
    assert 0 <= estimated_params['lapse_rate'] <= 0.04


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
#     resp_domain = ['Correct', 'Incorrect']
#     f = 'norm_cdf'
#     scale = 'dB'
#
#     q = QuestPlus(stim_domain=stim_domain, param_domain=param_domain,
#                   resp_domain=resp_domain, func=f, stim_scale=scale)


if __name__ == '__main__':
    test_threshold_slope()
