from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Common functions used in scoring germ and fiducial sets."""

from functools import total_ordering


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
