from typing import Dict, Iterable, Optional
from questplus import psychometric_function

import xarray as xr
import numpy as np
from copy import deepcopy


class QuestPlus:
    def __init__(self, *,
                 stim_domain: dict,
                 param_domain: Dict[str, float],
                 resp_domain: Iterable,
                 prior: Optional[Dict[str, float]] = None,
                 func: str = 'weibull',
                 stim_scale: str = 'log10'):
        self.func = func
        self.stim_scale = stim_scale
        self.stim_domain = self._ensure_ndarray(stim_domain)
        self.param_domain = self._ensure_ndarray(param_domain)
        self.resp_domain = np.array(resp_domain)

        self.prior = self.gen_prior(prior=prior)
        self.posterior = deepcopy(self.prior)
        self.likelihoods = self._gen_likelihoods()
        self.resp_history = list()
        self.stim_history = {p: [] for p in self.stim_domain.keys()}
        self.entropy = np.nan

    def _ensure_ndarray(self, x: dict) -> dict:
        x = deepcopy(x)
        for k, v in x.items():
            x[k] = np.atleast_1d(v)

        return x

    def gen_prior(self, *,
                  prior: dict) -> xr.DataArray:
        prior_orig = deepcopy(prior)

        if prior_orig is None:
            prior = np.ones([len(x) for x in self.param_domain.values()])
        else:
            prior_grid = np.meshgrid(*list(prior_orig.values()),
                                     sparse=True, indexing='ij')
            prior = np.prod(prior_grid)

        # Normalize.
        prior /= prior.sum()

        dims = *self.param_domain.keys(),
        coords = dict(**self.param_domain)
        prior_ = xr.DataArray(data=prior,
                              dims=dims,
                              coords=coords)

        return prior_

    def _gen_likelihoods(self) -> xr.DataArray:
        if self.func == 'weibull':
            f = psychometric_function.weibull
            pf_resp_corr = f(intensity=self.stim_domain['intensity'],
                             **self.param_domain,  scale=self.stim_scale)
            pf_resp_incorr = 1 - pf_resp_corr

            likelihood_dim = (len(self.resp_domain),
                              len(self.stim_domain['intensity']),
                              *[len(x) for x in self.param_domain.values()])
            likelihoods = np.empty(likelihood_dim)
            likelihoods[0, :] = pf_resp_corr
            likelihoods[1, :] = pf_resp_incorr

            dims = ('response',
                    *self.stim_domain.keys(),
                    *self.param_domain.keys())
            coords = dict(response=self.resp_domain,
                          **self.stim_domain,
                          **self.param_domain)
        else:
            raise ValueError('Unknown psychometric function name specified.')

        pf_values = xr.DataArray(data=likelihoods,
                                 dims=dims,
                                 coords=coords)

        return pf_values

    def update(self, *,
               stimulus: dict,
               response: str):
        likelihood = (self.likelihoods
                      .sel(**stimulus, response=response))

        self.posterior = self.posterior * likelihood
        self.posterior /= self.posterior.sum()

        # Log the results, too.
        for stim_property, stim_val in stimulus.items():
            self.stim_history[stim_property].append(stim_val)
        self.resp_history.append(response)

    def next_stim(self, *,
                  method: str = 'min_entropy',
                  sample_size: Optional[int] = None) -> float:
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

    def get_param_estimates(self, *,
                            method: str = 'mean') -> Dict[str, float]:
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

    def fit(self, *,
            stimuli: Iterable[dict],
            responses: Iterable[str]):
        for stimulus, response in zip(stimuli, responses):
            self.update(stimulus=stimulus, response=response)
