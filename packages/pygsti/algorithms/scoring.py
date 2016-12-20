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
    # We're expecting division by zero in many instances when we call this
    # function, and the inf can be handled appropriately, so we suppress
    # division warnings printed to stderr.
    with _np.errstate(divide='ignore'):
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


def composite_rcl_fn(candidateScores, alpha):
    """Create a restricted candidate list (RCL) based on CompositeScore objects.

    Parameters
    ----------
    candidateScores : list of CompositScore
        List of scores to be sorted in RCL and not RCL.

    alpha : float
        A number between 0 and 1 that roughly specifies a score theshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring elementt is
        included in the RCL), while a value of 1 for `alpha` will include all
        elements in the RCL.

        Intermediate values of alpha attempt to mimic the behavior of alpha for
        simple float scores. For those scores, the score that all elements must
        beat is ``(1 - alpha)*best + alpha*worst``. For CompositeScore objects,
        the most important part of the score is the integer `N`. The
        appropriate threshold for `N` is ``(1 - alpha)*N_max + alpha*(N_min -
        1)``. If ``N <= floor(N_thresh)``, it is automatically rejected from
        the RCL, and if ``N >= ceil(N_thresh + 1)``, it is automatically
        included in the RCL.  If neither of those initial conditions is the
        case (i.e.  `N_thresh` is not an integer and ``N == ceil(N_thresh)``),
        then the real-valued sub score is compared to the range of real valued
        sub scores for the value of `N` in question as described for simple
        float scores, using ``ceil(N_thresh) - N_thresh`` as the value for
        alpha in that case. It may be that there are no scores at the value of
        `N` such that ``N == ceil(N_thresh)``. In this case it doesn't matter
        what value of real sub score the threshold score is given, since no
        elements will need to compare against it.

    Returns
    -------
    numpy.array
        The indices of the scores sufficiently good to be in the RCL.

    """
    maxScore = max(candidateScores)
    minScore = min(candidateScores)
    NThreshold = (alpha * (minScore.N - 1)
                  + (1 - alpha) * maxScore.N)
    score_alpha = _np.ceil(NThreshold) - NThreshold
    thresholdScores = [candidateScore.score
                       for candidateScore in candidateScores
                       if candidateScore.N == int(_np.ceil(NThreshold))]
    if len(thresholdScores) == 0:
        # Don't care about sorting out scores with threshold N since there are
        # no scores with this N.
        compositeScoreThreshold = CompositeScore(0, int(_np.ceil(NThreshold)))
    else:
        # If there are N that aren't clearly in or out of the RCL, we have to
        # be a bit more careful.
        scoreThreshold = ((1 - score_alpha) * min(thresholdScores)
                          + score_alpha * max(thresholdScores))
        compositeScoreThreshold = CompositeScore(scoreThreshold, NThreshold)
    # Now that we've build a sensible threshold, compare all scores against
    # this.
    return _np.where(_np.array(candidateScores) <= compositeScoreThreshold)[0]
