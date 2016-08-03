from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions to facilitate using GRASP."""

import itertools

import numpy as _np

from .. import objects as _objs


def get_swap_neighbors(weights):
    """Return the list of weights in the neighborhood of a given weight vector.

    A weight vector is in the neighborhood of a given weight vector if it is
    only a single swap away from the given weight vector.

    """
    included_idxs = _np.where(weights==1)[0]
    excluded_idxs = _np.where(weights==0)[0]
    neighbors = []
    for swap_out, swap_in in itertools.product(included_idxs, excluded_idxs):
        neighbor = weights.copy()
        neighbor[swap_out] = 0
        neighbor[swap_in] = 1
        neighbors.append(neighbor)

    return neighbors


def grasp_greedy_construction(elements, scoreFn, rclFn, feasibleThreshold=None,
                              feasibleFn=None, initialElements=None,
                              seed=None):
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

    while _np.any(weights==0) and not feasible:
        candidateIdxs = _np.where(weights==0)[0]
        candidateSolns = [soln + [elements[idx]] for idx in candidateIdxs]
        candidateScores = _np.array([scoreFn(candidateSoln)
                                     for candidateSoln in candidateSolns])
        rclIdxs = rclFn(candidateScores)
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


def grasp(elements, greedyScoreFn, rclFn, localScoreFn, getNeighborsFn,
          finalScoreFn, iterations, feasibleThreshold=None, feasibleFn=None,
          initialElements=None, seed=None, verbosity=0):

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    bestSoln = None
    for iteration in range(iterations):
        printer.log('Iteration {}'.format(iteration), 1)
        initialSoln = grasp_greedy_construction(elements, greedyScoreFn, rclFn,
                                                feasibleThreshold, feasibleFn,
                                                initialElements, seed)
        printer.log('Initial construction:')
        printer.log(str([str(element) for element in initialSoln]), 1)

        localSoln = grasp_local_search(initialSoln, localScoreFn, elements,
                                       getNeighborsFn, feasibleThreshold,
                                       feasibleFn)
        printer.log('Local optimum:')
        printer.log(str([str(element) for element in localSoln]), 1)

        if bestSoln is None:
            bestSoln = localSoln
        elif finalScoreFn(localSoln) < finalScoreFn(bestSoln):
            bestSoln = localSoln

    return bestSoln
