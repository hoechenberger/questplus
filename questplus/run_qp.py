import numpy as np
import scipy.stats
from questplus.qp import QuestPlus
from questplus.psychometric_function import weibull
import matplotlib.pyplot as plt


def plot(q):
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(q.param_domain['threshold'],
                  q.posterior.sum(dim=['lower_asymptote', 'slope', 'lapse_rate']),
                  'o-')
    # plt.semilogx()
    ax[0, 0].set_title('Threshold')

    ax[0, 1].plot(q.param_domain['slope'],
                  q.posterior.sum(dim=['lower_asymptote', 'threshold', 'lapse_rate']),
                  'o-')
    ax[0, 1].set_title('Slope')

    ax[1, 0].plot(q.param_domain['lower_asymptote'],
                  q.posterior.sum(dim=['slope', 'threshold', 'lapse_rate']),
                  'o-')
    ax[1, 0].set_title('fa_rate')

    param_estimates_mean = q.get_param_estimates(method='mean')
    param_estimates_mode = q.get_param_estimates(method='mode')

    y_mean = weibull(intensity=intensities, threshold=param_estimates_mean['threshold'],
                     # x=np.linspace(stim_domain[0], stim_domain[-1], 500), t=param_estimates_mean['threshold'],
                     slope=param_estimates_mean['slope'],
                     lower_asymptote=param_estimates_mean['lower_asymptote'],
                     lapse_rate=param_estimates_mean['lapse_rate'],
                     scale='log10')[:, 0, 0, 0]

    y_mode = weibull(intensity=intensities, threshold=param_estimates_mode['threshold'],
                     slope=param_estimates_mode['slope'],
                     lower_asymptote=param_estimates_mode['lower_asymptote'],
                     lapse_rate=param_estimates_mode['lapse_rate'],
                     scale='log10')[:, 0, 0, 0]

    ax[1, 1].plot(intensities, y_mean, 'o-', lw=2, label='mean')
    # ax[1, 1].plot(stim_domain, y_mode, 'o-', lw=2, label='mode')
    # ax[1, 1].plot(np.linspace(stim_domain[0], stim_domain[-1], 500), y_mean, '-', lw=2, label='mean')


    ax[1, 1].legend(loc='best')
    # plt.semilogx()
    ax[1, 1].set_ylim((-0.05, 1.05))
    plt.show()

#Parameters.
# param = dict(threshold=np.linspace(start=-2, stop=4, num=29),
#              slope=np.linspace(start=0.1, stop=5, num=5),
#              lower_asymptote=np.linspace(start=0, stop=0.5, num=10),
#              lapse_rate=np.array([0.01, 0.1]))

# Intensities.
# stim_domain = param['threshold'].copy()
# stim_domain = np.linspace(start=-2, stop=4, num=30, dtype='float64')
# intensities_salty = np.logspace(np.log10(2), np.log10(0.002), num=12, base=10)
# intensities_sweet = np.logspace(np.log10(20), np.log10(0.01), num=12, base=10)


intensities_sweet = np.arange(-4.25, -0.25+0.25, step=0.25)
intensities_salty = np.arange(start=-3.5, stop=-0.5+0.25, step=0.25)
intensities_bitter = np.arange(start=-6.75, stop=-2.5+0.25, step=0.25)
intensities_sour = np.arange(start=-4.8, stop=-1.3+0.25, step=0.25)
# intensities_salty = np.linspace(-4.25, -0.25, num=15)

intensities = np.flipud(intensities_sour)

param = dict(#threshold=np.array([0, 1, 2], dtype='float64'),
             threshold=intensities,
             # slope=np.array([0.5, 0.75, 1], dtype='float64'),
             # lower_asymptote=np.array([0.1, 0.15], dtype='float64'),
             # slope=np.array([3.5], dtype='float64'),
             slope=np.linspace(0.5, 15, 5),
             # threshold=np.array([stim_domain[6]]),
             # lower_asymptote=np.array([0.05, 0.1, 0.15], dtype='float64'),
             lower_asymptote=np.linspace(0.0001, 0.5, 5),
             # lower_asymptote=np.linspace(0.1, 0.5, 3),
             # lower_asymptote=np.array([0.01], dtype='float64'),
             # lapse_rate=np.linspace(0.01, 0.1, 4))
             #
             lapse_rate=np.array([0.01], dtype='float64'))


# Intensities.
# stim_domain = param['threshold'].copy()
# stim_domain = np.linspace(start=-2, stop=4, num=30, dtype='float64')

# Response outcomes.
response_outcomes = np.array(['Yes', 'No'])


# def gen_prior():
#     threshold_prior = scipy.stats.norm.pdf(param['threshold'], loc=0, scale=5)
#     # threshold_prior /= threshold_prior.sum()
#
#     slope_prior = scipy.stats.norm.pdf(param['slope'], loc=2.5, scale=1.25)
#     # slope_prior /= slope_prior.sum()
#
#     lower_asymptote_prior = scipy.stats.norm.pdf(param['lower_asymptote'], loc=0.25, scale=0.125)
#     # lower_asymptote_prior /= lower_asymptote_prior.sum()
#
#     lapse_rate_prior = np.array([0.5, 0.5], dtype='float64')
#
#     prior = dict(threshold=threshold_prior,
#                  slope=slope_prior,
#                  lower_asymptote=lower_asymptote_prior,
#                  lapse_rate=lapse_rate_prior)
#
#     return prior
#
# prior = gen_prior()

prior = dict(threshold=np.ones(len(param['threshold'])),
             slope=np.ones(len(param['slope'])),
             lower_asymptote=np.ones(len(param['lower_asymptote'])),
             # lower_asymptote=scipy.stats.norm.pdf(
             #     param['lower_asymptote'],
             #     loc=param['lower_asymptote'][0],
             #     scale=param['lower_asymptote'].std(ddof=1)),
             lapse_rate=np.ones(len(param['lapse_rate'])))

stim_domain = dict(intensity=intensities)
q = QuestPlus(stim_domain=stim_domain, func='weibull',
              stim_scale='log10',
              param_domain=param, prior=None,
              resp_domain=response_outcomes)

with np.printoptions(precision=3, suppress=True):
    print(q.stim_domain)

plot(q)
print(q.next_stim(method='min_entropy'))
for trial_no in range(1, 20+1):
    if trial_no == 1:
        intensity = intensities[3]  # start with a relatively high concentration
    else:
        intensity = q.next_stim(method='min_n_entropy')
    # intensity = np.random.choice(stim_domain, 1)

    print(f'\n ==> Trial {trial_no}, intensity: {intensity}')

    # response = input(f'{intensity}: Y/N?')
    # response_ = 'Yes' if response == 'y' else 'No'
    # if trial_no >= 15:
    #     intensity = stim_domain[8]

    if intensity >= -2.3:
        response_ = 'Yes'
    elif -2.8 <= intensity < -2.3:
        p = [0.9, 0.1]
        response_ = np.random.choice(['Yes', 'No'], p=p)
        if response_ == 'No':
            print('   --> Inserting MISS...')
    else:
        p = [0.1, 0.9]
        response_ = np.random.choice(['Yes', 'No'], p=p)
        if response_ == 'Yes':
            print('   --> Inserting FALSE-ALARM...')
    # response_ = 'No'
    q.update(stimulus=dict(intensity=intensity), response=response_)
    print(f'   Response: {response_}, entropy: {q.entropy}')

    # p_thresh_cdf = q.posterior.sum(['slope', 'lower_asymptote', 'lapse_rate']).cumsum()
    # t_var = (q.posterior.sum(['slope', 'lower_asymptote', 'lapse_rate']) * np.flipud(q.stim_domain)).var(ddof=1).values
    # print(f'T var: {t_var}')

    # plot(q)
    # plt.title(f'Trial {trial_no}')
    # input('\nPress return to continue...')
plot(q)


# Add a false alarm
# q.update(intensity=stim_domain[-1], response='Yes')
# q.update(intensity=stim_domain[-3], response='Yes')

print('\nParameter estimates:')
param_estimates_mean = q.get_param_estimates(method='mean')
print(param_estimates_mean)
param_estimates_mode = q.get_param_estimates(method='mode')
print(param_estimates_mode)

with np.printoptions(precision=3, suppress=True):
    print(q.stim_domain)

# print('Mean: ', y_mean.min(), y_mean.max())
# print('Mode: ', y_mode.min(), y_mode.max())


# d-prime
print('\nd-prime:')
import scipy.stats
print(scipy.stats.norm.ppf(1-param_estimates_mean['lapse_rate']) -
      scipy.stats.norm.ppf(param_estimates_mean['lower_asymptote']))

# print(scipy.stats.norm.ppf(1-param_estimates_mode['lapse_rate']) -
#       scipy.stats.norm.ppf(param_estimates_mode['lower_asymptote']))


#%% d-prime
param_estimates_mean = q.get_param_estimates(method='mean')
x = np.linspace(intensities[0], intensities[-1], 10000)
y_mean = weibull(intensity=x,
                 threshold=param_estimates_mean['threshold'],
                 slope=param_estimates_mean['slope'],
                 lower_asymptote=param_estimates_mean['lower_asymptote'],
                 lapse_rate=param_estimates_mean['lapse_rate'],
                 scale='log10')[:, 0, 0, 0]

dp = scipy.stats.norm.ppf(y_mean) - scipy.stats.norm.ppf(param_estimates_mean['lower_asymptote'])
dp_minus_1 = np.abs(dp-1)

print(f'd-prime == 1 at: T={x[dp_minus_1.argmin()]}, '
      f'Psi(T)={y_mean[dp_minus_1.argmin()]}')


print('bias: ')
print(-(scipy.stats.norm.ppf(1-param_estimates_mean['lapse_rate']) + scipy.stats.norm.ppf(param_estimates_mean['lower_asymptote']))/2)

print(q.stim_history)

# print('Probability of guessing:')
# print(q.posterior.sum(['threshold', 'slope', 'lapse_rate']).sel(lower_asymptote=0.5).item())

# intensities_ = np.logspace(np.log10(2), np.log10(0.002), num=12, base=10)
# y = weibull_log10(x=intensities_, t=param_estimates['threshold'] + 0.5,
#             beta=param_estimates['slope'],
#             gamma=param_estimates['lower_asymptote'],
#             delta=param_estimates['lapse_rate'])[:,0,0,0]
#
# print(y.min(), y.max())
#
#
# plt.plot(intensities_, y, 'o-', lw=2)
# # plt.semilogx()
# plt.show()
#
