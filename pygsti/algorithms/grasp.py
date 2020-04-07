"""Functions to facilitate using GRASP."""
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

from .. import objects as _objs


def get_swap_neighbors(weights, forcedWeights=None, shuffle=False):
    """Return the list of weights in the neighborhood of a given weight vector.

    A weight vector is in the neighborhood of a given weight vector if it is
    only a single swap away from the given weight vector. There is an option to
    use `forcedWeights` to indicate elements you don't want to swap out.

    Parameters
    ----------
    weights : numpy.array
        Binary vector to find the neighborhood of.

    forcedWeights : numpy.array, optional
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
    if forcedWeights is None:
        forcedWeights = _np.zeros(len(weights))

    swap_out_idxs = _np.where(_np.logical_and(weights == 1,
                                              forcedWeights == 0))[0]
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


def grasp_greedy_construction(elements, scoreFn, rclFn, feasibleThreshold=None,
                              feasibleFn=None, initialElements=None):
    """
    Constructs a subset of `elements` that represents a feasible solution.

    This function performs the "greedy-construction" part of a grasp
    iteration (see :func:`do_grasp_iteration`). The returned solution
    subset is built up by repeating the following step until a feasible
    solution (using `feasibleThreshold` OR `feasibleFn`):

    1. Build a candidate list from elements that haven't been chosen.
    2. Based on the scores of the candidates (using `scoreFn`), construct a
       "reduced candidate list" (using `rclFn`) that need not (but could) be
       just the single best-scoring element.
    3. Choose a random element from the reduced candidate list and add it
       to the solution subset.

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    scoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores.

    rclFn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    feasibleThreshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasibleThreshold``). Overrides `feasibleFn` if set to a
        value other than ``None``.

    feasibleFn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasibleThreshold` is not
        ``None``.

    initialElements : numpy.array
        Binary vector indicating whether the corresponding elements in
        `elements` should be automatically included at the start of this
        construction.

    Returns
    -------
    list
        A sub-list of `elements`.
    """

    if initialElements is None:
        weights = _np.zeros(len(elements))
    else:
        if len(initialElements) != len(elements):
            raise ValueError('initialElements must have the same length as '
                             'elements ({}), not {}!'.format(len(elements),
                                                             len(initialElements)))
        weights = _np.array(initialElements)

    soln = [elements[idx] for idx in _np.nonzero(weights)[0]]

    if feasibleThreshold is not None:
        feasibleTest = 'threshold'
    elif feasibleFn is not None:
        feasibleTest = 'function'
    else:
        raise ValueError('Must provide either feasibleFn or '
                         'feasibleThreshold!')

    feasible = False

    while _np.any(weights == 0) and not feasible:
        candidateIdxs = _np.where(weights == 0)[0]
        candidateSolns = [soln + [elements[idx]] for idx in candidateIdxs]
        candidateScores = _np.array([scoreFn(candidateSoln)
                                     for candidateSoln in candidateSolns])
        rclIdxs = rclFn(candidateScores)
        assert(len(rclIdxs) > 0), "Empty reduced candidate list!"
        chosenIdx = _np.random.choice(rclIdxs)
        soln = candidateSolns[chosenIdx]
        weights[candidateIdxs[chosenIdx]] = 1
        if feasibleTest == 'threshold':
            feasible = candidateScores[chosenIdx] <= feasibleThreshold
        elif feasibleTest == 'function':
            feasible = feasibleFn(soln)

    if not feasible:
        raise ValueError('No feasible solution found!')

    return soln


def grasp_local_search(initialSoln, scoreFn, elements, getNeighborsFn,
                       feasibleThreshold=None, feasibleFn=None):
    """
    Perfom the local-search part of a grasp iteration.

    Attempts to find a better (lower-scoring) solution based on successive
    "local" (as determined by `getNeighborsFn`) steps from `initialSolution`.

    Parameters
    ----------
    initialSoln : list
        A list of some (or all) of the items in `elements`, representing an
        initial solution.  This solution must be "feasbile" as determined by
        `feasibleThreshold` or `feasibleFn`.

    scoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    getNeighborsFn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    feasibleThreshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasibleThreshold``). Overrides `feasibleFn` if set to a
        value other than ``None``.

    feasibleFn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasibleThreshold` is not
        ``None``.

    Returns
    -------
    list
        A sub-list of `elements`, representing the locally-improved solution.
    """
    if feasibleThreshold is not None:
        feasibleTest = 'threshold'
    elif feasibleFn is not None:
        feasibleTest = 'function'
    else:
        raise ValueError('Must provide either feasibleFn or '
                         'feasibleThreshold!')

    currentSoln = initialSoln
    currentWeights = _np.zeros(len(elements))
    for element in initialSoln:
        currentWeights[elements.index(element)] = 1
    currentScore = scoreFn(currentSoln)

    betterSolnFound = True

    while betterSolnFound:
        betterSolnFound = False
        weightsNeighbors = getNeighborsFn(currentWeights)
        neighborSolns = [[element for element
                          in _np.array(elements)[_np.nonzero(weightsNeighbor)]]
                         for weightsNeighbor in weightsNeighbors]
        if feasibleTest == 'function':
            feasibleNeighborSolns = [(idx, soln) for idx, soln
                                     in enumerate(neighborSolns)
                                     if feasibleFn(soln)]
            for idx, soln in feasibleNeighborSolns:
                solnScore = scoreFn(soln)
                if solnScore < currentScore:
                    betterSolnFound = True
                    currentScore = solnScore
                    currentSoln = soln
                    currentWeights = weightsNeighbors[idx]
                    break

        elif feasibleTest == 'threshold':
            for idx, soln in enumerate(neighborSolns):
                solnScore = scoreFn(soln)
                # The current score is by construction below the threshold,
                # so we don't need to check that.
                if solnScore < currentScore:
                    betterSolnFound = True
                    currentScore = solnScore
                    currentSoln = soln
                    currentWeights = weightsNeighbors[idx]
                    break

    return currentSoln


def do_grasp_iteration(elements, greedyScoreFn, rclFn, localScoreFn,
                       getNeighborsFn, feasibleThreshold=None, feasibleFn=None,
                       initialElements=None, seed=None, verbosity=0):
    """Perform one iteration of GRASP (greedy construction and local search).

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    greedyScoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the greedy
        construction to construct the initial feasible subset.

    rclFn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    localScoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    getNeighborsFn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    feasibleThreshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasibleThreshold``). Overrides `feasibleFn` if set to a
        value other than ``None``.

    feasibleFn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasibleThreshold` is not
        ``None``.

    initialElements : numpy.array
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
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    initialSoln = grasp_greedy_construction(elements, greedyScoreFn, rclFn,
                                            feasibleThreshold, feasibleFn,
                                            initialElements)
    printer.log('Initial construction:', 1)
    def to_str(x): return x.str if isinstance(x, _objs.Circuit) else str(x)
    printer.log(str([to_str(element) for element in initialSoln]), 1)

    localSoln = grasp_local_search(initialSoln, localScoreFn, elements,
                                   getNeighborsFn, feasibleThreshold,
                                   feasibleFn)
    printer.log('Local optimum:', 1)
    printer.log(str([to_str(element) for element in localSoln]), 1)

    return initialSoln, localSoln


def do_grasp(elements, greedyScoreFn, rclFn, localScoreFn, getNeighborsFn,
             finalScoreFn, iterations, feasibleThreshold=None, feasibleFn=None,
             initialElements=None, seed=None, verbosity=0):
    """Perform GRASP to come up with an optimal feasible set of elements.

    Parameters
    ----------
    elements : list
        A list containing some representation of the elements that can be used
        by the verious score functions.

    greedyScoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the greedy
        construction to construct the initial feasible subset.

    rclFn : callable
        Function that takes a list of sublists of `elements` (that is, a list
        of candidate partial solutions) and returns the indices within that
        list of partial solutions to be included in the restricted candidate
        list.

    localScoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used by the local
        search to find a locally optimal feasible subset.

    getNeighborsFn : callable
        Function that takes a binary vector indicating which members of
        `elements` are included in the current solution and returns a list
        of binary vectors indicating which potential solutions are in the
        neighborhood of the current solution for the purposes of local
        optimization.

    finalScoreFn : callable
        Function that takes a sublist of `elements` and returns a score to
        minimize that is comparable with other scores. Used to compare the
        solutions from various iterations in order to choose an optimum.

    iterations : int
        How many iterations of greedy construction followed by local search to
        perform.

    feasibleThreshold : score
        A value comparable with the return value of the various score functions
        against which a score may be compared to test if the solution is
        feasible (the solution is feasible iff
        ``solnScore < feasibleThreshold``). Overrides `feasibleFn` if set to a
        value other than ``None``.

    feasibleFn : callable
        Function that takes a sublist of `elements` defining a potential
        solution and returns ``True`` if that solution is feasible (otherwise
        should return ``False``). Not used if `feasibleThreshold` is not
        ``None``.

    initialElements : numpy.array
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
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    bestSoln = None
    for iteration in range(iterations):
        printer.log('Iteration {}'.format(iteration), 1)
        _, localSoln = do_grasp_iteration(elements, greedyScoreFn,
                                          rclFn, localScoreFn,
                                          getNeighborsFn,
                                          feasibleThreshold,
                                          feasibleFn,
                                          initialElements, seed,
                                          verbosity)
        if bestSoln is None:
            bestSoln = localSoln
        elif finalScoreFn(localSoln) < finalScoreFn(bestSoln):
            bestSoln = localSoln

    return bestSoln
