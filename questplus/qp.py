from typing import Optional, Sequence, Literal
from copy import deepcopy

import numpy as np
import xarray as xr
import json_tricks

from questplus import psychometric_function


class QuestPlus:
    def __init__(
        self,
        *,
        stim_domain: dict,
        param_domain: dict,
        outcome_domain: dict,
        prior: Optional[dict] = None,
        func: Literal["weibull", "csf", "norm_cdf", "norm_cdf_2", "thurstone_scaling"],
        stim_scale: Optional[Literal["log10", "dB", "linear"]],
        stim_selection_method: str = "min_entropy",
        stim_selection_options: Optional[dict] = None,
        param_estimation_method: str = "mean",
    ):
        """
        A QUEST+ staircase procedure.

        Parameters
        ----------
        stim_domain
            Specification of the stimulus domain: dictionary keys correspond to
            the names of the stimulus dimensions, and  values describe the
            respective possible stimulus values (e.g., intensities, contrasts,
            or orientations).

        param_domain
            Specification of the parameter domain: dictionary keys correspond
            to the names of the parameter dimensions, and  values describe the
            respective possible parameter values (e.g., threshold, slope,
            lapse rate.

        outcome_domain
            Specification of the outcome domain: dictionary keys correspond
            to the names of the outcome dimensions, and  values describe the
            respective possible outcome values (e.g., "Yes", "No", "Correct",
            "Incorrect"). This argument typically describes the responses a
            participant can provide.

        prior
            A-priori probabilities of parameter values.

        func
            The psychometric function whose parameters to estimate.

        stim_scale
            The scale on which the stimuli are provided. Has no effect for the
            Thurstonian scaling function.

        stim_selection_method
            How to select the next stimulus. `min_entropy` picks the stimulus
            that will minimize the expected entropy. `min_n_entropy` randomly
            selects a stimulus from the set of stimuli that will yield the `n`
            smallest entropies. `n` has to be specified via the
            `stim_selection_options` keyword argument.

        stim_selection_options
            Use this argument to specify options for the stimulus selection
            method specified via `stim_selection_method`. Currently, this can
            be used to specify the number of `n` stimuli that will yield the
            `n` smallest entropies if `stim_selection_method=min_n_entropy`,
            and `max_consecutive_reps`, the number of times the same stimulus
            can be presented consecutively. A random number generator seed
            may be passed via `random_seed=12345`.

        param_estimation_method
            The method to use when deriving the final parameter estimate.
            Possible values are `mean` (mean of each parameter, weighted by the
            posterior probabilities) and `mode` (the parameters at the peak of
            the posterior distribution).

        """
        if func == "thurstone_scaling" and stim_scale is not None:
            raise ValueError(
                "The Thurstonian scaling function cannot be used with "
                "a stim_scale parameter."
            )

        self.func = func
        self.stim_scale = stim_scale
        self.stim_domain = self._ensure_ndarray(stim_domain)
        self.param_domain = self._ensure_ndarray(param_domain)
        self.outcome_domain = self._ensure_ndarray(outcome_domain)

        self.prior = self._gen_prior(prior=prior)
        self.posterior = deepcopy(self.prior)
        self.likelihoods = self._gen_likelihoods()

        self.stim_selection = stim_selection_method

        if self.stim_selection == "min_n_entropy":
            from ._constants import (
                DEFAULT_N,
                DEFAULT_RANDOM_SEED,
                DEFAULT_MAX_CONSECUTIVE_REPS,
            )

            if stim_selection_options is None:
                self.stim_selection_options = dict(
                    n=DEFAULT_N,
                    max_consecutive_reps=DEFAULT_MAX_CONSECUTIVE_REPS,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
            else:
                self.stim_selection_options = stim_selection_options.copy()

                if "n" not in stim_selection_options:
                    self.stim_selection_options["n"] = DEFAULT_N
                if "max_consecutive_reps" not in stim_selection_options:
                    self.stim_selection_options[
                        "max_consecutive_reps"
                    ] = DEFAULT_MAX_CONSECUTIVE_REPS
                if "random_seed" not in stim_selection_options:
                    self.stim_selection_options["random_seed"] = DEFAULT_RANDOM_SEED

            del DEFAULT_N, DEFAULT_MAX_CONSECUTIVE_REPS, DEFAULT_RANDOM_SEED

            seed = self.stim_selection_options["random_seed"]
            self._rng = np.random.RandomState(seed=seed)
            del seed
        else:
            self.stim_selection_options = stim_selection_options
            self._rng = None

        self.param_estimation_method = param_estimation_method

        self.resp_history = list()
        self.stim_history = list()
        self.entropy = np.nan

    @staticmethod
    def _ensure_ndarray(x: dict) -> dict:
        x = deepcopy(x)
        for k, v in x.items():
            x[k] = np.atleast_1d(v)

        return x

    def _gen_prior(self, *, prior: dict) -> xr.DataArray:
        """
        Raises
        ------
        ValueError
            If the user specifies priors for parameters that do not appear in
            the parameter domain.

        """
        prior_orig = deepcopy(prior)

        if prior_orig is None:
            # Uninformative prior.
            prior = np.ones([len(x) for x in self.param_domain.values()])
        elif set(prior_orig.keys()) - set(self.param_domain.keys()):
            msg = (
                f"Mismatch between specified parameter domain and supplied "
                f"prior.\n"
                f"You specified priors for the following parameters that "
                f"do not appear in the parameter domain: "
                f"{set(prior_orig.keys()) - set(self.param_domain.keys())}"
            )
            raise ValueError(msg)
        elif set(self.param_domain.keys()) - set(prior_orig.keys()):
            # The user specified prior probabilities for only a subset
            # of the parameters. We use those, obviously; and fill the
            # remaining prior distributions with uninformative priors.
            grid_dims = []
            for param_name, param_vals in self.param_domain.items():
                if param_name in prior_orig.keys():
                    prior_vals = np.atleast_1d(prior_orig[param_name])
                else:
                    prior_vals = np.ones(len(param_vals))

                grid_dims.append(prior_vals)

            prior_grid = np.meshgrid(*grid_dims, sparse=True, indexing="ij")
            prior = np.prod(
                np.array(prior_grid, dtype="object")  # avoid warning re "ragged" array
            )
        else:
            # A "proper" prior was specified (i.e., prior probabilities for
            # all parameters.)
            prior_grid = np.meshgrid(
                *list(prior_orig.values()), sparse=True, indexing="ij"
            )
            prior = np.prod(
                np.array(prior_grid, dtype="object")  # avoid warning re "ragged" array
            )

        # Normalize.
        prior /= prior.sum()

        # Create the prior object we are actually going to use.
        dims = (*self.param_domain.keys(),)
        coords = dict(**self.param_domain)
        prior_ = xr.DataArray(data=prior, dims=dims, coords=coords)

        return prior_

    def _gen_likelihoods(self) -> xr.DataArray:
        outcome_dim_name = list(self.outcome_domain.keys())[0]
        outcome_values = list(self.outcome_domain.values())[0]

        if self.func not in [
            "weibull",
            "csf",
            "norm_cdf",
            "norm_cdf_2",
            "thurstone_scaling",
        ]:
            raise ValueError(
                f"Unknown psychometric function name specified: {self.func}"
            )

        if self.func == "weibull":
            f = psychometric_function.weibull
        elif self.func == "csf":
            f = psychometric_function.csf
        elif self.func == "norm_cdf":
            f = psychometric_function.norm_cdf
        elif self.func == "norm_cdf_2":
            f = psychometric_function.norm_cdf_2
        elif self.func == "thurstone_scaling":
            f = psychometric_function.thurstone_scaling_function

        if self.func == "thurstone_scaling":
            prop_correct = f(**self.stim_domain, **self.param_domain)
        else:
            prop_correct = f(
                **self.stim_domain, **self.param_domain, scale=self.stim_scale
            )
        prop_incorrect = 1 - prop_correct

        # Now this is a bit awkward. We concatenate the psychometric
        # functions for the different responses. To do that, we first have
        # to add an additional dimension.
        # TODO: There's got to be a neater way to do this?!
        corr_resp_dim = {outcome_dim_name: [outcome_values[0]]}
        inccorr_resp_dim = {outcome_dim_name: [outcome_values[1]]}

        prop_correct = prop_correct.expand_dims(corr_resp_dim)
        prop_incorrect = prop_incorrect.expand_dims(inccorr_resp_dim)

        pf_values = xr.concat(
            [prop_correct, prop_incorrect],
            dim=outcome_dim_name,
            coords=self.outcome_domain,
        )
        return pf_values

    def update(self, *, stim: dict, outcome: dict) -> None:
        """
        Inform QUEST+ about a newly gathered measurement outcome for a given
        stimulus parameter set, and update the posterior accordingly.

        Parameters
        ----------
        stim
            The stimulus that was used to generate the given outcome.

        outcome
            The observed outcome.

        """
        likelihood = self.likelihoods.sel(**stim, **outcome)

        self.posterior = self.posterior * likelihood
        self.posterior /= self.posterior.sum()

        # Log the results, too.
        self.stim_history.append(stim)
        self.resp_history.append(outcome)

    @property
    def next_stim(self) -> dict:
        """
        Retrieve the stimulus to present next.

        The stimulus will be selected based on the method in
        ``self.stim_selection``.

        """
        new_posterior = self.posterior * self.likelihoods

        # Probability.
        pk = new_posterior.sum(dim=self.param_domain.keys())
        new_posterior /= pk

        # Entropies.
        #
        # Note:
        #   - np.log(0) returns -inf (division by zero)
        #   - the multiplcation of new_posterior with -inf values generates
        #     NaN's
        #   - xr.DataArray.sum() has special handling for NaN's.
        #
        # NumPy also emits a warning, which we suppress here.
        with np.errstate(divide="ignore"):
            H = -(
                (new_posterior * np.log(new_posterior)).sum(
                    dim=self.param_domain.keys()
                )
            )

        # Expected entropies for all possible stimulus parameters.
        EH = (pk * H).sum(dim=list(self.outcome_domain.keys()))

        if self.stim_selection == "min_entropy":
            # Get the stimulus properties that minimize entropy.
            indices = EH.argmin(dim=...)
            stim = dict()
            for stim_property, index in indices.items():
                stim_val = EH[stim_property][index].item()
                stim[stim_property] = stim_val

            self.entropy = EH.min().item()
        elif self.stim_selection == "min_n_entropy":
            # Number of stimuli to include (the n stimuli that yield the lowest
            # entropies)
            n_stim = self.stim_selection_options["n"]

            indices = np.unravel_index(EH.argsort(), EH.shape)[0]
            indices = indices[:n_stim]

            while True:
                # Randomly pick one index and retrieve its coordinates
                # (stimulus parameters).
                candidate_index = self._rng.choice(indices)
                coords = EH[candidate_index].coords
                stim = {
                    stim_property: stim_val.item()
                    for stim_property, stim_val in coords.items()
                }

                max_reps = self.stim_selection_options["max_consecutive_reps"]

                if len(self.stim_history) < 2:
                    break
                elif all(
                    [stim == prev_stim for prev_stim in self.stim_history[-max_reps:]]
                ):
                    # Shuffle again.
                    continue
                else:
                    break
        else:
            raise ValueError("Unknown stim_selection supplied.")

        return stim

    @property
    def param_estimate(self) -> dict:
        """
        Retrieve the final parameter estimates after the QUEST+  run.

        The parameters will be calculated according to
        ``self.param_estimation_method``.

        This returns a dictionary of parameter estimates, where the dictionary
        keys correspond to the parameter names.

        """
        method = self.param_estimation_method
        param_estimates = dict()
        for param_name in self.param_domain.keys():
            params = list(self.param_domain.keys())
            params.remove(param_name)

            if method == "mean":
                param_estimates[param_name] = (
                    (self.posterior.sum(dim=params) * self.param_domain[param_name])
                    .sum()
                    .item()
                )
            elif method == "mode":
                indices = self.posterior.argmax(dim=...)
                coords = self.posterior[indices]
                param_estimates[param_name] = coords[param_name].item()
            else:
                raise ValueError("Unknown method parameter.")

        return param_estimates

    @property
    def marginal_posterior(self) -> dict:
        """
        Retrieve the a dictionary of marginal posterior probability
        density functions (PDFs).

        This returns a  dictionary of marginal PDFs, where the dictionary keys
        correspond to the parameter names.

        """
        marginal_posterior = dict()
        for param_name in self.param_domain.keys():
            marginalized_out_params = list(self.param_domain.keys())
            marginalized_out_params.remove(param_name)
            marginal_posterior[param_name] = self.posterior.sum(
                dim=marginalized_out_params
            ).values

        return marginal_posterior

    def to_json(self) -> str:
        """
        Dump this `QuestPlus` instance as a JSON string which can be loaded
        again later.

        Returns
        -------
        str
            A JSON dump of the current `QuestPlus` instance.

        See Also
        --------
        from_json

        """
        self_copy = deepcopy(self)
        self_copy.prior = self_copy.prior.to_dict()
        self_copy.posterior = self_copy.posterior.to_dict()
        self_copy.likelihoods = self_copy.likelihoods.to_dict()

        if self_copy._rng is not None:  # NumPy RandomState cannot be serialized.
            self_copy._rng = self_copy._rng.get_state()

        return json_tricks.dumps(self_copy, allow_nan=True)

    @staticmethod
    def from_json(data: str):
        """
        Load and recreate a ``QuestPlus`` instance from a JSON string.

        Parameters
        ----------
        data
            The JSON string, generated via :meth:`to_json`.

        Returns
        -------
        QuestPlus
            A ``QuestPlus`` instance, generated from the JSON string.

        See Also
        --------
        to_json

        """
        loaded = json_tricks.loads(data)
        loaded.prior = xr.DataArray.from_dict(loaded.prior)
        loaded.posterior = xr.DataArray.from_dict(loaded.posterior)
        loaded.likelihoods = xr.DataArray.from_dict(loaded.likelihoods)

        if loaded._rng is not None:
            state = deepcopy(loaded._rng)
            loaded._rng = np.random.RandomState()
            loaded._rng.set_state(state)

        return loaded

    def __eq__(self, other):
        if not self.likelihoods.equals(other.likelihoods):
            return False

        if not self.prior.equals(other.prior):
            return False

        if not self.posterior.equals(other.posterior):
            return False

        for param_name in self.param_domain.keys():
            if not np.array_equal(
                self.param_domain[param_name], other.param_domain[param_name]
            ):
                return False

        for stim_property in self.stim_domain.keys():
            if not np.array_equal(
                self.stim_domain[stim_property], other.stim_domain[stim_property]
            ):
                return False

        for outcome_name in self.outcome_domain.keys():
            if not np.array_equal(
                self.outcome_domain[outcome_name], other.outcome_domain[outcome_name]
            ):
                return False

        if self.stim_selection != other.stim_selection:
            return False

        if self.stim_selection_options != other.stim_selection_options:
            return False

        if self.stim_scale != other.stim_scale:
            return False

        if self.stim_history != other.stim_history:
            return False

        if self.resp_history != other.resp_history:
            return False

        if self.param_estimation_method != other.param_estimation_method:
            return False

        if self.func != other.func:
            return False

        return True


class QuestPlusWeibull(QuestPlus):
    def __init__(
        self,
        *,
        intensities: Sequence,
        thresholds: Sequence,
        slopes: Sequence,
        lower_asymptotes: Sequence,
        lapse_rates: Sequence,
        prior: Optional[dict] = None,
        responses: Sequence = ("Yes", "No"),
        stim_scale: str = "log10",
        stim_selection_method: str = "min_entropy",
        stim_selection_options: Optional[dict] = None,
        param_estimation_method: str = "mean",
    ):
        """QUEST+ using the Weibull distribution function.

        This is a convenience class that wraps `QuestPlus`.
        """
        super().__init__(
            stim_domain=dict(intensity=intensities),
            param_domain=dict(
                threshold=thresholds,
                slope=slopes,
                lower_asymptote=lower_asymptotes,
                lapse_rate=lapse_rates,
            ),
            outcome_domain=dict(response=responses),
            prior=prior,
            stim_scale=stim_scale,
            stim_selection_method=stim_selection_method,
            stim_selection_options=stim_selection_options,
            param_estimation_method=param_estimation_method,
            func="weibull",
        )

    @property
    def intensities(self) -> np.ndarray:
        """
        Stimulus intensity or contrast domain.
        """
        return self.stim_domain["intensity"]

    @property
    def thresholds(self) -> np.ndarray:
        """
        The threshold parameter domain.
        """
        return self.param_domain["threshold"]

    @property
    def slopes(self) -> np.ndarray:
        """
        The slope parameter domain.
        """
        return self.param_domain["slope"]

    @property
    def lower_asymptotes(self) -> np.ndarray:
        """
        The lower asymptote parameter domain.
        """
        return self.param_domain["lower_asymptote"]

    @property
    def lapse_rates(self) -> np.ndarray:
        """
        The lapse rate parameter domain.
        """
        return self.param_domain["lapse_rate"]

    @property
    def responses(self) -> np.ndarray:
        """
        The response (outcome) domain.
        """
        return self.outcome_domain["response"]

    @property
    def next_intensity(self) -> float:
        """
        The intensity or contrast to present next.
        """
        return super().next_stim["intensity"]

    def update(self, *, intensity: float, response: str) -> None:
        """
        Inform QUEST+ about a newly gathered measurement outcome for a given
        stimulus intensity or contrast, and update the posterior accordingly.

        Parameters
        ----------
        intensity
            The intensity or contrast of the presented stimulus.

        response
            The observed response.

        """
        super().update(stim=dict(intensity=intensity), outcome=dict(response=response))


class QuestPlusThurstone(QuestPlus):
    def __init__(
        self,
        *,
        physical_magnitudes_stim_1: Sequence,
        physical_magnitudes_stim_2: Sequence,
        thresholds: Sequence,
        powers: Sequence,
        perceptual_scale_maxs: Sequence,
        prior: Optional[dict] = None,
        responses: Sequence = ("First", "Second"),
        stim_selection_method: str = "min_entropy",
        stim_selection_options: Optional[dict] = None,
        param_estimation_method: str = "mean",
    ):
        """QUEST+ for Thurstonian scaling.

        This is a convenience class that wraps `QuestPlus`.
        """
        super().__init__(
            stim_domain={
                "physical_magnitudes_stim_1": physical_magnitudes_stim_1,
                "physical_magnitudes_stim_2": physical_magnitudes_stim_2,
            },
            param_domain={
                "threshold": thresholds,
                "power": powers,
                "perceptual_scale_max": perceptual_scale_maxs,
            },
            outcome_domain={"response": responses},
            prior=prior,
            stim_scale=None,
            stim_selection_method=stim_selection_method,
            stim_selection_options=stim_selection_options,
            param_estimation_method=param_estimation_method,
            func="thurstone_scaling",
        )

    @property
    def physical_magnitudes_stim_1(self) -> np.ndarray:
        """
        Physical magnitudes of the first stimulus.
        """
        return self.stim_domain["physical_magnitudes_stim_1"]

    @property
    def physical_magnitudes_stim_2(self) -> np.ndarray:
        """
        Physical magnitudes of the second stimulus.
        """
        return self.stim_domain["physical_magnitudes_stim_2"]

    @property
    def thresholds(self) -> np.ndarray:
        """
        The threshold parameter domain.
        """
        return self.param_domain["threshold"]

    @property
    def powers(self) -> np.ndarray:
        """
        The power parameter domain.
        """
        return self.param_domain["power"]

    @property
    def perceptual_scale_maxss(self) -> np.ndarray:
        """
        The "maximum value of the subjective perceptual scale" parameter domain.
        """
        return self.param_domain["perceptual_scale_max"]

    @property
    def responses(self) -> np.ndarray:
        """
        The response (outcome) domain.
        """
        return self.outcome_domain["response"]

    def update(self, *, stim: dict, response: str) -> None:
        """
        Inform QUEST+ about a newly gathered measurement outcome for a given
        stimulus parameter set, and update the posterior accordingly.

        Parameters
        ----------
        stim
            The stimulus that was used to generate the given outcome.

        outcome
            The observed outcome.

        """
        super().update(stim=stim, outcome=dict(response=response))
