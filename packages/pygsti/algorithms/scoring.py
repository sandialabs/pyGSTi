from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Common functions used in scoring germ and fiducial sets."""

from functools import total_ordering

import numpy as _np


def list_score(input_array, scoreFunc='all'):
    """Score an array of eigenvalues. Smaller scores are better.

    Parameters
    ----------
    input_array : numpy array
        The eigenvalues to be scored.

    scoreFunc : {'all', 'worst'}, optional
        Sets the objective function for scoring the eigenvalues. If 'all',
        score is ``sum(1/input_array)``. If 'worst', score is
        ``1/min(input_array)``.

        Note: we use this function in various optimization routines, and
        sometimes choosing one or the other objective function can help avoid
        suboptimal local minima.

    Returns
    -------
    float
        Score for the eigenvalues.

    """
    if scoreFunc == 'all':
        score = sum(1. / _np.abs(input_array))
    elif scoreFunc == 'worst':
        score = 1. / min(_np.abs(input_array))
    else:
        raise ValueError("'%s' is not a valid value for scoreFunc.  "
                         "Either 'all' or 'worst' must be specified!"
                         % scoreFunc)

    return score


@total_ordering
class CompositeScore():
    """Class for storing and comparing scores calculated from eigenvalues.

    The comparison functions operate according to the logic that a lower score
    is better. Therefore, a score that has more non-zero eigenvalues (higher
    `N`) will always compare as less than a score that has fewer non-zero
    eigenvalues (lower `N`), with ties for `N` being resolved by comparing
    `score` in the straightforward manner (since `score` is assumed to be
    better for lower values).

    Parameters
    ----------
    N : int
        The number of non-zero eigenvalues.
    score : float
        The score computed considering only the non-zero eigenvalues (lower
        values are better).

    """
    def __init__(self, score, N):
        self.score = score
        self.N = N

    def __lt__(self, other):
        if self.N > other.N:
            return True
        elif self.N < other.N:
            return False
        else:
            return self.score < other.score

    def __eq__(self, other):
        return self.N == other.N and self.score == other.score

    def __repr__(self):
        return 'Score: {}, N: {}'.format(self.score, self.N)
