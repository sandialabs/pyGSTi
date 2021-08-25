"""
Functions to facilitate using GRASP.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools
import random

import numpy as _np

from pygsti import baseobjs as _baseobjs
from pygsti import circuits as _circuits


def get_swap_neighbors(weights, forced_weights=None, shuffle=False):
    """
    Return the list of weights in the neighborhood of a given weight vector.

    A weight vector is in the neighborhood of a given weight vector if it is
    only a single swap away from the given weight vector. There is an option to
    use `forced_weights` to indicate elements you don't want to swap out.

    Parameters
    ----------
    weights : numpy.array
        Binary vector to find the neighborhood of.

    forced_weights : numpy.array, optional
        Binary vector indicating elements that must be included in all
        neighboring vectors (these elements are assumed to already be present
        in `weights`.

    shuffle : bool, optional
        Whether the neighborhood should be presented to the optimizer in a
        random order (important if the local optimizer updates the solution to
        the first better solution it finds in the neighborhood instead of
        exhaustively searching the neighborhood for the best solution).

    Returns
    -------
    list of numpy.array
        List of binary vectors corresponding to all the neighbors of `weights`.
    """
    if forced_weights is None:
        forced_weights = _np.zeros(len(weights))

    swap_out_idxs = _np.where(_np.logical_and(weights == 1,
                                              forced_weights == 0))[0]
    swap_in_idxs = _np.where(weights == 0)[0]
    neighbors = []
    for swap_out, swap_in in itertools.product(swap_out_idxs, swap_in_idxs):
        neighbor = weights.copy()
        neighbor[swap_out] = 0
        neighbor[swap_in] = 1
        neighbors.append(neighbor)

    if shuffle:
        random.shuffle(neighbors)

    return neighbors


def _grasp_construct_feasible_solution(elements, score_fn, rcl_fn, feasible_threshold=None,
                                       feasible_fn=None, initial_elements=None):
    """
    Constructs a subset of `elements` that represents a feasible solution.

    This function performs the "greedy-construction" part of a grasp
    iteration (see :func:`run_grasp_iteration`). The returned solution
    subset is built up by repeating the following step until a feasible
    solution (using `feasible_threshold` OR `feasible_fn`):

    1. Build a candidate list from elements that haven't been chosen.
    2. Based on the scores of the candidates (using `score_fn`), construct a
       "reduced candidate list" (using `rcl_fn`) that need not (but could) be
       just the single best-scoring element.
    3. Choose a random element from the reduced candidate list and add it
       to the solution subset.

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores.

    rcl_fn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    feasible_threshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasible_threshold``). Overrides `feasible_fn` if set to a
        value other than ``None``.

    feasible_fn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasible_threshold` is not
        ``None``.

    initial_elements : numpy.array
        Binary vector indicating whether the corresponding elements in
        `elements` should be automatically included at the start of this
        construction.

    Returns
    -------
    list
        A sub-list of `elements`.
    """

    if initial_elements is None:
        weights = _np.zeros(len(elements))
    else:
        if len(initial_elements) != len(elements):
            raise ValueError('initial_elements must have the same length as '
                             'elements ({}), not {}!'.format(len(elements),
                                                             len(initial_elements)))
        weights = _np.array(initial_elements)

    soln = [elements[idx] for idx in _np.nonzero(weights)[0]]

    if feasible_threshold is not None:
        feasibleTest = 'threshold'
    elif feasible_fn is not None:
        feasibleTest = 'function'
    else:
        raise ValueError('Must provide either feasible_fn or '
                         'feasible_threshold!')

    feasible = feasible_fn(soln) if feasibleTest == 'function' else score_fn(soln) <= feasible_threshold

    while _np.any(weights == 0) and not feasible:
        candidateIdxs = _np.where(weights == 0)[0]
        candidateSolns = [soln + [elements[idx]] for idx in candidateIdxs]
        candidateScores = _np.array([score_fn(candidateSoln)
                                     for candidateSoln in candidateSolns])
        rclIdxs = rcl_fn(candidateScores)
        assert(len(rclIdxs) > 0), "Empty reduced candidate list!"
        chosenIdx = _np.random.choice(rclIdxs)
        soln = candidateSolns[chosenIdx]
        weights[candidateIdxs[chosenIdx]] = 1
        if feasibleTest == 'threshold':
            feasible = candidateScores[chosenIdx] <= feasible_threshold
        elif feasibleTest == 'function':
            feasible = feasible_fn(soln)

    if not feasible:
        raise ValueError('No feasible solution found!')

    return soln


def _grasp_local_search(initial_solution, score_fn, elements, get_neighbors_fn,
                        feasible_threshold=None, feasible_fn=None):
    """
    Perfom the local-search part of a grasp iteration.

    Attempts to find a better (lower-scoring) solution based on successive
    "local" (as determined by `get_neighbors_fn`) steps from `initialSolution`.

    Parameters
    ----------
    initial_solution : list
        A list of some (or all) of the items in `elements`, representing an
        initial solution.  This solution must be "feasbile" as determined by
        `feasible_threshold` or `feasible_fn`.

    score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    get_neighbors_fn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    feasible_threshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasible_threshold``). Overrides `feasible_fn` if set to a
        value other than ``None``.

    feasible_fn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasible_threshold` is not
        ``None``.

    Returns
    -------
    list
        A sub-list of `elements`, representing the locally-improved solution.
    """
    if feasible_threshold is not None:
        feasibleTest = 'threshold'
    elif feasible_fn is not None:
        feasibleTest = 'function'
    else:
        raise ValueError('Must provide either feasible_fn or '
                         'feasible_threshold!')

    currentSoln = initial_solution
    currentWeights = _np.zeros(len(elements))
    for element in initial_solution:
        currentWeights[elements.index(element)] = 1
    currentScore = score_fn(currentSoln)

    betterSolnFound = True

    while betterSolnFound:
        betterSolnFound = False
        weightsNeighbors = get_neighbors_fn(currentWeights)
        neighborSolns = [[element for element
                          in _np.array(elements)[_np.nonzero(weightsNeighbor)]]
                         for weightsNeighbor in weightsNeighbors]
        if feasibleTest == 'function':
            feasibleNeighborSolns = [(idx, soln) for idx, soln
                                     in enumerate(neighborSolns)
                                     if feasible_fn(soln)]
            for idx, soln in feasibleNeighborSolns:
                solnScore = score_fn(soln)
                if solnScore < currentScore:
                    betterSolnFound = True
                    currentScore = solnScore
                    currentSoln = soln
                    currentWeights = weightsNeighbors[idx]
                    break

        elif feasibleTest == 'threshold':
            for idx, soln in enumerate(neighborSolns):
                solnScore = score_fn(soln)
                # The current score is by construction below the threshold,
                # so we don't need to check that.
                if solnScore < currentScore:
                    betterSolnFound = True
                    currentScore = solnScore
                    currentSoln = soln
                    currentWeights = weightsNeighbors[idx]
                    break

    return currentSoln


def run_grasp_iteration(elements, greedy_score_fn, rcl_fn, local_score_fn,
                        get_neighbors_fn, feasible_threshold=None, feasible_fn=None,
                        initial_elements=None, seed=None, verbosity=0):
    """
    Perform one iteration of GRASP (greedy construction and local search).

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    greedy_score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the greedy
        construction to construct the initial feasible subset.

    rcl_fn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    local_score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    get_neighbors_fn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    feasible_threshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasible_threshold``). Overrides `feasible_fn` if set to a
        value other than ``None``.

    feasible_fn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasible_threshold` is not
        ``None``.

    initial_elements : numpy.array
        Binary vector indicating whether the corresponding elements in
        `elements` should be automatically included by the greedy construction
        routine at the start of its construction.

    seed : int
        Seed for the random number generator.

    verbosity : int
        Sets the level of logging messages the printer will display.

    Returns
    -------
    initialSoln : list
        The sublist of `elements` given by the greedy construction.
    localSoln : list
        The sublist of `elements` given by the local search.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    initialSoln = _grasp_construct_feasible_solution(elements, greedy_score_fn, rcl_fn,
                                                     feasible_threshold, feasible_fn,
                                                     initial_elements)
    printer.log('Initial construction:', 1)
    def to_str(x): return x.str if isinstance(x, _circuits.Circuit) else str(x)
    printer.log(str([to_str(element) for element in initialSoln]), 1)

    localSoln = _grasp_local_search(initialSoln, local_score_fn, elements,
                                    get_neighbors_fn, feasible_threshold,
                                    feasible_fn)
    printer.log('Local optimum:', 1)
    printer.log(str([to_str(element) for element in localSoln]), 1)

    return initialSoln, localSoln


def run_grasp(elements, greedy_score_fn, rcl_fn, local_score_fn, get_neighbors_fn,
              final_score_fn, iterations, feasible_threshold=None, feasible_fn=None,
              initial_elements=None, seed=None, verbosity=0):
    """
    Perform GRASP to come up with an optimal feasible set of elements.

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    greedy_score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the greedy
        construction to construct the initial feasible subset.

    rcl_fn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    local_score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    get_neighbors_fn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    final_score_fn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used to compare the
        solutions from various iterations in order to choose an optimum.

    iterations : int
        How many iterations of greedy construction followed by local search to
        perform.

    feasible_threshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasible_threshold``). Overrides `feasible_fn` if set to a
        value other than ``None``.

    feasible_fn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasible_threshold` is not
        ``None``.

    initial_elements : numpy.array
        Binary vector with 1s at indices corresponding to elements in
        `elements` that the greedy construction routine will include at the
        start of its construction.

    seed : int
        Seed for the random number generator.

    verbosity : int
        Sets the level of logging messages the printer will display.

    Returns
    -------
    list of Circuits
        The best germ set from all locally-optimal germ sets constructed.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    bestSoln = None
    for iteration in range(iterations):
        printer.log('Iteration {}'.format(iteration), 1)
        _, localSoln = run_grasp_iteration(elements, greedy_score_fn,
                                           rcl_fn, local_score_fn,
                                           get_neighbors_fn,
                                           feasible_threshold,
                                           feasible_fn,
                                           initial_elements, seed,
                                           verbosity)
        if bestSoln is None:
            bestSoln = localSoln
        elif final_score_fn(localSoln) < final_score_fn(bestSoln):
            bestSoln = localSoln

    return bestSoln
