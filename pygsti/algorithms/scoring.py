"""Common functions used in scoring germ and fiducial sets."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from functools import total_ordering

import numpy as _np


def list_score(input_array, score_func='all'):
    """Score an array of eigenvalues. Smaller scores are better.

    Parameters
    ----------
    input_array : numpy array
        The eigenvalues to be scored.

    score_func : {'all', 'worst'}, optional
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
    # We're expecting division by zero in many instances when we call this
    # function, and the inf can be handled appropriately, so we suppress
    # division warnings printed to stderr.
    with _np.errstate(divide='ignore'):
        if score_func == 'all':
            score = sum(1. / _np.abs(input_array))
        elif score_func == 'worst':
            score = 1. / min(_np.abs(input_array))
        else:
            raise ValueError("'%s' is not a valid value for score_func.  "
                             "Either 'all' or 'worst' must be specified!"
                             % score_func)

    return score


@total_ordering
class CompositeScore():
    """Class for storing and comparing scores calculated from eigenvalues.

    The comparison functions operate according to the logic that a lower score
    is better. The score value is broken into two parts: 'major' and 'minor'.
    A CompositeScore with a smaller 'major' part is always smaller than one
    with a larger 'major' part.  The 'minor' parts are only compared when the
    major parts are equal.  Typically, the negative of the number of non-zero
    eigenvalues is used to as the major part so that a score that has more non-zero
    eigenvalues (higher `N`) will always compare as less than a score that has
    fewer non-zero eigenvalues (lower `N`), with ties for `N` being resolved by
    comparing the minor score in the straightforward manner (since the non-AC
    `score` is assumed to be better for lower values).  For bookeeping, the
    CompositeScore object also separately holds the  number of non-zero eigenvalues,
    as this may not always be recovered from the major part of the score.

    Parameters
    ----------
    major, minor : float
        The major and minor parts of the score.
    N : int
        The number of non-zero eigenvalues.
    """

    def __init__(self, major, minor, n):
        self.major = major
        self.minor = minor
        self.N = n

    def __lt__(self, other):
        #Just base on *scores*
        if self.major < other.major:
            return True
        elif self.major > other.major:
            return False
        else:
            return self.minor < other.minor

    def __eq__(self, other):
        return self.major == other.major and \
            self.minor == other.minor

    def __repr__(self):
        return 'Score: major={} minor={}, N: {}'.format(
            self.major, self.minor, self.N)


def composite_rcl_fn(candidate_scores, alpha):
    """Create a restricted candidate list (RCL) based on CompositeScore objects.

    Parameters
    ----------
    candidate_scores : list of CompositScore
        List of scores to be sorted in RCL and not RCL.

    alpha : float
        A number between 0 and 1 that roughly specifies a score theshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring element is
        included in the RCL), while a value of 1 for `alpha` will include all
        elements in the RCL.

        Intermediate values of alpha attempt to mimic the behavior of alpha for
        simple float scores. For those scores, the score that all elements must
        beat is ``(1 - alpha)*best + alpha*worst``. For CompositeScore objects,
        thresholding is done on the major part of the score unless all the
        candidates have the same major score, in which case thresholding is
        performed using only the minor score.

    Returns
    -------
    numpy.array
        The indices of the scores sufficiently good to be in the RCL.

    """
    maxScore = max(candidate_scores)
    minScore = min(candidate_scores)
    if maxScore.major == minScore.major:
        threshold = CompositeScore(maxScore.major,
                                   ((1 - alpha) * minScore.minor
                                    + alpha * maxScore.minor), None)
    else:
        maxMinorScore = max([s.minor for s in candidate_scores])
        threshold = CompositeScore(((1 - alpha) * minScore.major
                                    + alpha * maxScore.major),
                                   maxMinorScore, None)
        # take *all* candidates with computed major score, so use
        # maximal minor score
    return _np.where(_np.array(candidate_scores) <= threshold)[0]
