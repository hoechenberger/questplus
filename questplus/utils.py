from typing import Dict, Sequence
import numpy as np
from questplus import psychometric_function


def simulate_response(*,
                      func: str = 'weibull',
                      stimulus: Dict,
                      params: Dict[str, float],
                      response_domain: Sequence = ('Correct', 'Incorrect'),
                      stim_scale: str = 'log10'):
    if func == 'weibull':
        f = psychometric_function.weibull
        p_correct = f(intensity=stimulus['intensity'],
                      **params, scale=stim_scale).squeeze()

        response = np.random.choice(response_domain,
                                    p=[p_correct, 1-p_correct])
    else:
        raise ValueError('Invalid function specified.')

    return response
