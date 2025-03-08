"""
Tools for manipulating classical probability distributions.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as _np


def tvd(p, q):
    """
    Calculates the total variational distance between two probability distributions.

    The distributions must be dictionaries, where keys are events (e.g., bit strings) and values are the
    probabilities. If an event in the keys of one dictionary isn't in the keys of the other
    then that probability is assumed to be zero. There are no checks that the input probability
    distributions are valid (i.e., that the probabilities sum up to one and are postiive).

    Parameters
    ----------
    p, q : dicts
        The distributions to calculate the TVD between.

    Returns
    -------
    float
    """
    q_temp = q.copy()

    tvd = 0
    for (event, x) in p.items():
        try:
            y = q_temp[event]
            del q_temp[event]
        except:
            y = 0.
        tvd += 0.5 * _np.abs(x - y)

    for (event, y) in q_temp.items():
        tvd += 0.5 * abs(y)

    return tvd


def classical_fidelity(p, q):
    """
    Calculates the (classical) fidelity between two probability distributions.

    The distributions must be dictionaries, where keys are events (e.g., bit strings) and values are the
    probabilities. If an event in the keys of one dictionary isn't in the keys of the other
    then that probability is assumed to be zero. There are no checks that the input probability
    distributions are valid (i.e., that the probabilities sum up to one and are postiive).

    Parameters
    ----------
    p, q : dicts
        The distributions to calculate the TVD between.

    Returns
    -------
    float
    """
    #sqrt_fidelity = 0
    #for (event, x) in x.items():
    #    y = q.get(event, 0.)
    #    sqrt_fidelity += _np.sqrt(x * y)

    return _np.sum([_np.sqrt(x * q.get(event, 0.)) for (event, x) in p.items()]) ** 2

    #return root_fidelity ** 2


# def Hoyer_sparsity_measure(p, n):
#     """
#     Computes a measure of the sparsity ("spikyness") of a probability distribution (or a
#     general real vector).

#     Parameters
#     ----------
#     p : dict
#         The distribution

#     n : the number of possible events (zero probability events do not need to be included in `p`)

#     Returns
#     -------
#     float
#     """
#     plist = _np.array(list(p.values()))
#     twonorm = _np.sqrt(_np.sum(plist**2))
#     onenorm = _np.sum(_np.abs(plist))
#     max_onenorm_over_twonorm = _np.sqrt(n)
#     return (max_onenorm_over_twonorm - onenorm/twonorm) / (max_onenorm_over_twonorm - 1)
