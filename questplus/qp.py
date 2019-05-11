from typing import Optional, Union
from questplus import psychometric_function

import xarray as xr
import numpy as np
from copy import deepcopy


class QuestPlus:
    def __init__(self, *,
                 stim_domain: dict,
                 param_domain: dict,
                 outcome_domain: dict,
                 prior: Optional[dict] = None,
                 func: str = 'weibull',
                 stim_scale: str = 'log10',
                 stim_selection: str = 'min_entropy',
                 stim_selection_options: Optional = None):
        self.func = func
        self.stim_scale = stim_scale
        self.stim_domain = self._ensure_ndarray(stim_domain)
        self.param_domain = self._ensure_ndarray(param_domain)
        self.outcome_domain = self._ensure_ndarray(outcome_domain)

        self.prior = self.gen_prior(prior=prior)
        self.posterior = deepcopy(self.prior)
        self.likelihoods = self._gen_likelihoods()

        self.stim_selection = stim_selection
        self.stim_selection_options = stim_selection_options

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
        if self.func in ['weibull', 'csf']:
            if self.func == 'weibull':
                f = psychometric_function.weibull
            else:
                f = psychometric_function.csf

            pf_resp_corr = f(**self.stim_domain,
                             **self.param_domain,
                             scale=self.stim_scale)

            pf_resp_incorr = 1 - pf_resp_corr

            likelihood_dim = (len(self.outcome_domain['response']),
                              *[len(x) for x in self.stim_domain.values()],
                              *[len(x) for x in self.param_domain.values()])
            likelihoods = np.empty(likelihood_dim)
            likelihoods[0, :] = pf_resp_corr
            likelihoods[1, :] = pf_resp_incorr

            dims = (*self.outcome_domain.keys(),
                    *self.stim_domain.keys(),
                    *self.param_domain.keys())
            coords = dict(**self.outcome_domain,
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
               outcome: dict):
        likelihood = (self.likelihoods
                      .sel(**stimulus, **outcome))

        self.posterior = self.posterior * likelihood
        self.posterior /= self.posterior.sum()

        # Log the results, too.
        for stim_property, stim_val in stimulus.items():
            self.stim_history[stim_property].append(stim_val)
        self.resp_history.append(outcome)

    def next_stim(self, *,
                  stim_selection: Optional[str] = None) -> dict:
        if stim_selection is None:
            stim_selection = self.stim_selection

        new_posterior = self.posterior * self.likelihoods

        # https://github.com/petejonze/QuestPlus/blob/master/QuestPlus.m,
        # "probability" in Watson 2017
        pk = new_posterior.sum(dim=self.param_domain.keys())
        new_posterior /= pk

        H = -(new_posterior * np.log(new_posterior)).sum(dim=self.param_domain.keys())
        EH = (pk * H).sum(dim=list(self.outcome_domain.keys()))

        if stim_selection == 'min_entropy':
            # Get coordinates of stimulus properties that minimize entropy.
            index = np.unravel_index(EH.argmin(), EH.shape)
            coords = EH[index].coords
            stim = {k: v.item() for k, v in coords.items()}
            self.entropy = EH.min().item()
        elif stim_selection == 'min_n_entropy':
            index = np.argsort(EH)[:4]
            while True:
                stim_candidates = self.stim_domain['intensity'][index.values]
                stim = np.random.choice(stim_candidates)

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
            raise ValueError('Unknown stim_selection supplied.')

        return stim

    def get_param_estimates(self, *,
                            method: str = 'mean') -> dict:
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
                index = np.unravel_index(self.posterior.argmax(),
                                         self.posterior.shape)
                coords = self.posterior[index].coords
                param_estimates[param_name] = coords[param_name].item()
            else:
                raise ValueError('Unknown method parameter.')

        return param_estimates
