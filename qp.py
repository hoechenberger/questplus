import psychometric_function

import xarray as xr
import numpy as np
import warnings
from copy import deepcopy


class QuestPlus:
    def __init__(self, *,
                 stim_domain, param_domain, resp_domain,
                 prior=None,
                 func='weibull_log10'):
        self.func = func
        self.stim_domain = stim_domain
        self.param_domain = param_domain
        self.resp_domain = np.asarray(resp_domain)

        self.prior = self.gen_prior(prior=prior)
        self.posterior = deepcopy(self.prior)
        self.likelihoods = self._gen_likelihoods()
        self.resp_history = list()
        self.stim_history = {p: [] for p in self.stim_domain.keys()}
        self.entropy = np.nan

    @staticmethod
    def gen_prior(*, prior):
        prior_orig = deepcopy(prior)
        # Normalize prior.
        # prior = deepcopy(prior)
        # for k, v in prior.items():
        #     prior[k] = np.asarray(v)
        #     if not np.isclose(prior[k].sum(), 1):
        #         msg = f'Prior {k} was not normalized. Normalizing now...'
        #         warnings.warn(msg)
        #         prior[k] /= prior[k].sum()
        #
        # prior = (xr.DataArray(prior['threshold'], dims=('threshold',)) *
        #           xr.DataArray(prior['slope'], dims=('slope',)) *
        #           xr.DataArray(prior['lower_asymptote'], dims=('lower_asymptote',)) *
        #           xr.DataArray(prior['lapse_rate'], dims=('lapse_rate',)))

        prior_2 = (xr.DataArray(prior_orig['threshold'], dims=('threshold',)) *
                   xr.DataArray(prior_orig['slope'], dims=('slope',)) *
                   xr.DataArray(prior_orig['lower_asymptote'], dims=('lower_asymptote',)) *
                   xr.DataArray(prior_orig['lapse_rate'], dims=('lapse_rate',)))

        prior_2 /= prior_2.sum()
        return prior_2

        # assert prior == prior_2
        # t, s, f, l = np.meshgrid(prior['threshold'], prior['slope'],
        #                          prior['lower_asymptote'], prior['lapse_rate'],
        #                          indexing='ij',
        #                          sparse=True)
        # prior_grid = dict(threshold=t, slope=s, lower_asymptote=f, lapse_rate=l)
        # return prior_grid
        # return prior

    def _gen_likelihoods(self):
        if self.func == 'weibull_log10':
            f = psychometric_function.weibull_log10
            pf_resp_corr = f(intensity=self.stim_domain['intensity'],
                             **self.param_domain)
            pf_resp_incorr = 1 - pf_resp_corr

            likelihoods = np.empty((len(self.resp_domain),
                                    len(self.stim_domain['intensity']),
                                    len(self.param_domain['threshold']),
                                    len(self.param_domain['slope']),
                                    len(self.param_domain['lower_asymptote']),
                                    len(self.param_domain['lapse_rate'])))

            likelihoods[0, :] = pf_resp_corr
            likelihoods[1, :] = pf_resp_incorr

            dims = ('response',
                    *self.stim_domain.keys(),
                    *self.param_domain.keys())
            coords = dict(response=self.resp_domain,
                          **self.stim_domain,
                          **self.param_domain)

            pf_values = xr.DataArray(data=likelihoods, dims=dims,
                                     coords=coords)
        else:
            raise ValueError('Unknown psychometric function name specified.')

        return pf_values

    def update(self, *,
               stimulus, response):
        likelihood = (self.likelihoods
                      .sel(**stimulus, response=response))

        self.posterior = self.posterior * likelihood
        self.posterior /= self.posterior.sum()

        # Log the results, too.
        for stim_property, stim_val in stimulus.items():
            self.stim_history[stim_property].append(stim_val)
        self.resp_history.append(response)

    def next_stim(self, *,
                  method='min_entropy',
                  sample_size=None):
        new_posterior = self.posterior * self.likelihoods

        # https://github.com/petejonze/QuestPlus/blob/master/QuestPlus.m,
        # "probability" in Watson 2017
        pk = new_posterior.sum(dim=self.param_domain.keys())
        new_posterior /= pk

        H = -(new_posterior * np.log(new_posterior)).sum(dim=self.param_domain.keys())
        EH = (pk * H).sum(dim=['response'])

        if method == 'min_entropy':
            # stim = EH.isel(intensity=EH.argmin()).coords['intensity'].values
            stim = self.stim_domain['intensity'][EH.argmin()]
            self.entropy = EH.min().item()
        elif method == 'min_n_entropy':
            index = np.argsort(EH) <= 4

            while True:
                stim = np.random.choice(self.stim_domain['intensity'][index.values])
                if len(self.stim_history['intensity']) < 2:
                    break
                elif (np.isclose(stim, self.stim_history['intensity'][-1]) and
                      np.isclose(stim, self.stim_history['intensity'][-2])):
                    print('\n  ==> shuffling again... <==\n')
                    continue
                else:
                    break

            print(f'options: {self.stim_domain["intensity"][index.values]} -> {stim}')
        else:
            raise ValueError('Unknown method supplied.')

        return stim

    def get_param_estimates(self, *, method='mean'):
        param_estimates = dict()
        for param_name in self.param_domain.keys():
            params = list(self.param_domain.keys())
            params.remove(param_name)

            if method == 'mean':
                param_estimates[param_name] = ((self.posterior.sum(dim=params) *
                                                self.param_domain[param_name])
                                               .sum()
                                               .item())
            elif method == 'mode':
                index = self.posterior.sum(dim=params).argmax()
                param_estimates[param_name] = self.param_domain[param_name][index]
            else:
                raise ValueError('Unknown method parameter.')

        return param_estimates
