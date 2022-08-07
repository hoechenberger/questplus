from typing import Sequence, Union
import numpy as np
from questplus import psychometric_function


def simulate_response(
    *,
    func: str = "weibull",
    stimulus: dict,
    params: dict,
    response_domain: Sequence = ("Correct", "Incorrect"),
    stim_scale: str = "log10",
) -> Union[float, str]:
    """
    Simulate an observer with the given psychometric function parameters.

    Parameters
    ----------
    func
        The psychometric function. Currently, only ``weibull`` is supported.

    stimulus
        The stimulus domain.

    params
        The psychometric function parameters.

    response_domain
        The response domain.

    stim_scale
        The scale on which the stimulus values are provided. Can be either
        ``linear``, ``log10``, or ``dB``.

    Returns
    -------
    float or str
        A simulated response for the given stimulus.

    """
    if func == "weibull":
        f = psychometric_function.weibull
        p_correct = f(
            intensity=stimulus["intensity"], **params, scale=stim_scale
        ).squeeze()

        response = np.random.choice(response_domain, p=[p_correct, 1 - p_correct])
    else:
        raise ValueError("Invalid function specified.")

    return response
