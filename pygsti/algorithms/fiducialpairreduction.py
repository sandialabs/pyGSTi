"""
Functions for reducing the number of required fiducial pairs for analysis.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import itertools as _itertools
import random as _random

import numpy as _np
import scipy.special as _spspecial

from pygsti import baseobjs as _baseobjs
from pygsti import circuits as _circuits

from pygsti.construction import circuitconstruction as _gsc
from pygsti.modelmembers.operations import EigenvalueParamDenseOp as _EigenvalueParamDenseOp
from pygsti.tools import remove_duplicates as _remove_duplicates
from pygsti.tools import slicetools as _slct


def _nCr(n, r):                                                                           # noqa
    """Number of combinations of r items out of a set of n.  Equals n!/(r!(n-r)!)"""
    #f = _math.factorial; return f(n) / f(r) / f(n-r)
    return _spspecial.comb(n, r)


def _random_combination(indices_tuple, r):
    """
    Random selection from itertools.combinations(indices_tuple, r)
      from http://docs.python.org/2/library/itertools.html#recipes
    """
    n = len(indices_tuple)
    iis = sorted(_random.sample(range(n), r))
    return tuple(indices_tuple[i] for i in iis)


def find_sufficient_fiducial_pairs(target_model, prep_fiducials, meas_fiducials, germs,
                                   test_lengths=(256, 2048), prep_povm_tuples="first", tol=0.75,
                                   search_mode="sequential", n_random=100, seed=None,
                                   verbosity=0, test_pair_list=None, mem_limit=None,
                                   minimum_pairs=1):
    """
    Finds a (global) set of fiducial pairs that are amplificationally complete.

    A "standard" set of GST circuits consists of all circuits of the form:

    statePrep + prepFiducial + germPower + measureFiducial + measurement

    This set is typically over-complete, and it is possible to restrict the
    (prepFiducial, measureFiducial) pairs to a subset of all the possible
    pairs given the separate `prep_fiducials` and `meas_fiducials` lists.  This function
    attempts to find a set of fiducial pairs that still amplify all of the
    model's parameters (i.e. is "amplificationally complete").  The test
    for amplification is performed using the two germ-power lengths given by
    `test_lengths`, and tests whether the magnitudes of the Jacobian's singular
    values scale linearly with the germ-power length.

    In the special case when `test_pair_list` is not None, the function *tests*
    the given set of fiducial pairs for amplificational completeness, and
    does not perform any search.

    Parameters
    ----------
    target_model : Model
        The target model used to determine amplificational completeness.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    germs : list of Circuits
        The germ circuits that are repeated to amplify errors.

    test_lengths : (L1,L2) tuple of ints, optional
        A tuple of integers specifying the germ-power lengths to use when
        checking for amplificational completeness.

    prep_povm_tuples : list or "first", optional
        A list of `(prepLabel, povmLabel)` tuples to consider when
        checking for completeness.  Usually this should be left as the special
        (and default) value "first", which considers the first prep and POVM
        contained in `target_model`.

    tol : float, optional
        The tolerance for the fraction of the expected amplification that must
        be observed to call a parameter "amplified".

    search_mode : {"sequential","random"}, optional
        If "sequential", then all potential fiducial pair sets of a given length
        are considered in sequence before moving to sets of a larger size.  This
        can take a long time when there are many possible fiducial pairs.
        If "random", then only `n_random` randomly chosen fiducial pair sets are
        considered for each set size before the set is enlarged.

    n_random : int, optional
        The number of random-pair-sets to consider for a given set size.

    seed : int, optional
        The seed to use for generating random-pair-sets.

    verbosity : int, optional
        How much detail to print to stdout.

    test_pair_list : list or None, optional
        If not None, a list of (prepfid_index,measfid_index) tuples of integers,
        specifying a list of fiducial pairs (indices are into `prep_fiducials` and
        `meas_fiducials`, respectively).  These pairs are then tested for
        amplificational completeness and the number of amplified parameters
        is printed to stdout.  (This is a special debugging functionality.)

    mem_limit : int, optional
        A memory limit in bytes.

    minimum_pairs : int, optional
        The minimium number of fiducial pairs to try (default == 1).  Set this
        to integers larger than 1 to avoid trying pair sets that are known to
        be too small.

    Returns
    -------
    list
        A list of (prepfid_index,measfid_index) tuples of integers, specifying a list
        of fiducial pairs (indices are into `prep_fiducials` and `meas_fiducials`).
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    #trim LSGST list of all f1+germ^exp+f2 strings to just those needed to get full rank jacobian. (compressed sensing
    #like)

    #tol = 0.5 #fraction of expected amplification that must be observed to call a parameter "amplified"
    if prep_povm_tuples == "first":
        firstRho = list(target_model.preps.keys())[0]
        firstPOVM = list(target_model.povms.keys())[0]
        prep_povm_tuples = [(firstRho, firstPOVM)]
    prep_povm_tuples = [(_circuits.Circuit((prepLbl,)), _circuits.Circuit((povmLbl,)))
                        for prepLbl, povmLbl in prep_povm_tuples]

    def get_derivs(length):
        """ Compute all derivative info: get derivative of each <E_i|germ^exp|rho_j>
            where i = composite EVec & fiducial index and j similar """

        st = 0  # running row count over to-be-concatenated dPall matrix
        elIndicesForPair = [[] for i in range(len(prep_fiducials) * len(meas_fiducials))]
        #contains lists of final leading-dim indices corresponding to each fiducial pair

        dPall = []  # one element per germ, concatenate later

        for iGerm, germ in enumerate(germs):
            expGerm = _gsc.repeat_with_max_length(germ, length)  # could pass exponent and set to germ**exp here
            lst = _gsc.create_circuits(
                "pp[0]+f0+expGerm+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
                expGerm=expGerm, pp=prep_povm_tuples, order=('f0', 'f1', 'pp'))

            resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
            layout = target_model.sim.create_layout(lst, None, resource_alloc, array_types=('ep',), verbosity=0)
            #FUTURE: assert that no instruments are allowed?

            local_dP = layout.allocate_local_array('ep', 'd')
            target_model.sim.bulk_fill_dprobs(local_dP, layout, None)  # num_els x num_params
            dP = local_dP.copy()  # local == global (no layout.gather required) b/c we used comm=None above
            layout.free_local_array(local_dP)  # not needed - local_dP isn't shared (comm=None)

            dPall.append(dP)

            #Add this germ's element indices for each fiducial pair (final circuit of evTree)
            nPrepPOVM = len(prep_povm_tuples)
            for k in range(len(prep_fiducials) * len(meas_fiducials)):
                for o in range(k * nPrepPOVM, (k + 1) * nPrepPOVM):
                    # "original" indices into lst for k-th fiducial pair
                    elArray = _slct.to_array(layout.indices_for_index(o)) + st
                    elIndicesForPair[k].extend(list(elArray))
            st += layout.num_elements  # b/c we'll concatenate tree's elements later

        return _np.concatenate(dPall, axis=0), elIndicesForPair
        #indexed by [iElement, iGatesetParam] : gives d(<SP|f0+exp_iGerm+f1|AM>)/d(iGatesetParam)
        # where iGerm, f0, f1, and SPAM are all bundled into iElement (but elIndicesForPair
        # provides the necessary indexing for picking out certain pairs)

    def get_number_amplified(m0, m1, len0, len1, verb):
        """ Return the number of amplified parameters """
        printer = _baseobjs.VerbosityPrinter.create_printer(verb)
        L_ratio = float(len1) / float(len0)
        try:
            s0 = _np.linalg.svd(m0, compute_uv=False)
            s1 = _np.linalg.svd(m1, compute_uv=False)
        except:                                       # pragma: no cover
            printer.warning("SVD error!!"); return 0  # pragma: no cover
            #SVD did not converge -> just say no amplified params...

        numAmplified = 0
        printer.log("Amplified parameter test: matrices are %s and %s." % (m0.shape, m1.shape), 4)
        printer.log("Index : SV(L=%d)  SV(L=%d)  AmpTest ( > %g ?)" % (len0, len1, tol), 4)
        for i, (v0, v1) in enumerate(zip(sorted(s0, reverse=True), sorted(s1, reverse=True))):
            if abs(v0) > 0.1 and (v1 / v0) / L_ratio > tol:
                numAmplified += 1
                printer.log("%d: %g  %g  %g YES" % (i, v0, v1, (v1 / v0) / L_ratio), 4)
            printer.log("%d: %g  %g  %g NO" % (i, v0, v1, (v1 / v0) / L_ratio), 4)
        return numAmplified

    #rank = len( [v for v in s if v > 0.001] )

    printer.log("------  Fiducial Pair Reduction --------")

    L0 = test_lengths[0]; dP0, elIndices0 = get_derivs(L0)
    L1 = test_lengths[1]; dP1, elIndices1 = get_derivs(L1)
    fullTestMx0 = dP0
    fullTestMx1 = dP1

    #Get number of amplified parameters in the "full" test matrix: the one we get when we use all possible fiducial
    #pairs
    if test_pair_list is None:
        maxAmplified = get_number_amplified(fullTestMx0, fullTestMx1, L0, L1, verbosity + 1)
        printer.log("maximum number of amplified parameters = %s" % maxAmplified)

    #Loop through fiducial pairs and add all derivative rows (1 x nModelParams) to test matrix
    # then check if testMatrix has full rank ( == nModelParams)

    nPossiblePairs = len(prep_fiducials) * len(meas_fiducials)
    allPairIndices = list(range(nPossiblePairs))
    nRhoStrs, nEStrs = len(prep_fiducials), len(meas_fiducials)

    if test_pair_list is not None:  # special mode for testing/debugging single pairlist
        pairIndices0 = _np.concatenate([elIndices0[prepfid_index * nEStrs + iEStr]
                                        for prepfid_index, iEStr in test_pair_list])
        pairIndices1 = _np.concatenate([elIndices1[prepfid_index * nEStrs + iEStr]
                                        for prepfid_index, iEStr in test_pair_list])
        testMx0 = _np.take(fullTestMx0, pairIndices0, axis=0)
        testMx1 = _np.take(fullTestMx1, pairIndices1, axis=0)
        nAmplified = get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
        printer.log("Number of amplified parameters = %s" % nAmplified)
        return None

    bestAmplified = 0
    for nNeededPairs in range(minimum_pairs, nPossiblePairs):
        printer.log("Beginning search for a good set of %d pairs (%d pair lists to test)" %
                    (nNeededPairs, _nCr(nPossiblePairs, nNeededPairs)))

        bestAmplified = 0
        if search_mode == "sequential":
            pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

        elif search_mode == "random":
            _random.seed(seed)  # ok if seed is None
            nTotalPairCombos = _nCr(len(allPairIndices), nNeededPairs)
            if n_random < nTotalPairCombos:
                pairIndicesToIterateOver = [_random_combination(allPairIndices, nNeededPairs) for i in range(n_random)]
            else:
                pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

        for pairIndicesToTest in pairIndicesToIterateOver:
            pairIndices0 = _np.concatenate([elIndices0[i] for i in pairIndicesToTest])
            pairIndices1 = _np.concatenate([elIndices1[i] for i in pairIndicesToTest])
            testMx0 = _np.take(fullTestMx0, pairIndices0, axis=0)
            testMx1 = _np.take(fullTestMx1, pairIndices1, axis=0)
            nAmplified = get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
            bestAmplified = max(bestAmplified, nAmplified)
            if printer.verbosity > 1:
                ret = []
                for i in pairIndicesToTest:
                    prepfid_index = i // nEStrs
                    iEStr = i - prepfid_index * nEStrs
                    ret.append((prepfid_index, iEStr))
                printer.log("Pair list %s ==> %d amplified parameters" % (" ".join(map(str, ret)), nAmplified))

            if nAmplified == maxAmplified:
                ret = []
                for i in pairIndicesToTest:
                    prepfid_index = i // nEStrs
                    iEStr = i - prepfid_index * nEStrs
                    ret.append((prepfid_index, iEStr))
                return ret

    printer.log(" --> Highest number of amplified parameters was %d" % bestAmplified)

    #if we tried all the way to nPossiblePairs-1 and no success, just return all the pairs, which by definition will hit
    #the "max-amplified" target
    listOfAllPairs = [(prepfid_index, iEStr)
                      for prepfid_index in range(nRhoStrs)
                      for iEStr in range(nEStrs)]
    return listOfAllPairs


def find_sufficient_fiducial_pairs_per_germ(target_model, prep_fiducials, meas_fiducials,
                                            germs, pre_povm_tuples="first",
                                            search_mode="sequential", constrain_to_tp=True,
                                            n_random=100, seed=None, verbosity=0,
                                            mem_limit=None):
    """
    Finds a per-germ set of fiducial pairs that are amplificationally complete.

    A "standard" set of GST circuits consists of all circuits of the form:

    statePrep + prepFiducial + germPower + measureFiducial + measurement

    This set is typically over-complete, and it is possible to restrict the
    (prepFiducial, measureFiducial) pairs to a subset of all the possible
    pairs given the separate `prep_fiducials` and `meas_fiducials` lists.  This function
    attempts to find sets of fiducial pairs, one set per germ, that still
    amplify all of the model's parameters (i.e. is "amplificationally
    complete").  For each germ, a fiducial pair set is found that amplifies
    all of the "parameters" (really linear combinations of them) that the
    particular germ amplifies.

    To test whether a set of fiducial pairs satisfies this condition, the
    sum of projectors `P_i = dot(J_i,J_i^T)`, where `J_i` is a matrix of the
    derivatives of each of the selected (prepFiducial+germ+effectFiducial)
    sequence probabilities with respect to the i-th germ eigenvalue (or
    more generally, amplified parameter), is computed.  If the fiducial-pair
    set is sufficient, the rank of the resulting sum (an operator) will be
    equal to the total (maximal) number of parameters the germ can amplify.

    Parameters
    ----------
    target_model : Model
        The target model used to determine amplificational completeness.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    germs : list of Circuits
        The germ circuits that are repeated to amplify errors.

    pre_povm_tuples : list or "first", optional
        A list of `(prepLabel, povmLabel)` tuples to consider when
        checking for completeness.  Usually this should be left as the special
        (and default) value "first", which considers the first prep and POVM
        contained in `target_model`.

    search_mode : {"sequential","random"}, optional
        If "sequential", then all potential fiducial pair sets of a given length
        are considered in sequence (per germ) before moving to sets of a larger
        size.  This can take a long time when there are many possible fiducial
        pairs.  If "random", then only `n_random` randomly chosen fiducial pair
        sets are considered for each set size before the set is enlarged.

    constrain_to_tp : bool, optional
        Whether or not to consider non-TP parameters the the germs amplify.  If
        the fiducal pairs will be used in a GST estimation where the model is
        constrained to being trace-preserving (TP), this should be set to True.

    n_random : int, optional
        The number of random-pair-sets to consider for a given set size.

    seed : int, optional
        The seed to use for generating random-pair-sets.

    verbosity : int, optional
        How much detail to print to stdout.

    mem_limit : int, optional
        A memory limit in bytes.

    Returns
    -------
    dict
        A dictionary whose keys are the germ circuits and whose values are
        lists of (iRhoFid,iMeasFid) tuples of integers, each specifying the
        list of fiducial pairs for a particular germ (indices are into
        `prep_fiducials` and `meas_fiducials`).
    """

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    if pre_povm_tuples == "first":
        firstRho = list(target_model.preps.keys())[0]
        firstPOVM = list(target_model.povms.keys())[0]
        pre_povm_tuples = [(firstRho, firstPOVM)]
    pre_povm_tuples = [(_circuits.Circuit((prepLbl,)), _circuits.Circuit((povmLbl,)))
                       for prepLbl, povmLbl in pre_povm_tuples]

    pairListDict = {}  # dict of lists of 2-tuples: one pair list per germ

    printer.log("------  Individual Fiducial Pair Reduction --------")
    with printer.progress_logging(1):
        for i, germ in enumerate(germs):

            #Create a new model containing static target gates and a
            # special "germ" gate that is parameterized only by it's
            # eigenvalues (and relevant off-diagonal elements)
            gsGerm = target_model.copy()
            gsGerm.set_all_parameterizations("static")
            germMx = gsGerm.sim.product(germ)
            gsGerm.operations["Ggerm"] = _EigenvalueParamDenseOp(
                germMx, True, constrain_to_tp)

            printer.show_progress(i, len(germs),
                                  suffix='-- %s germ (%d params)' %
                                  (germ, gsGerm.num_params))
            #Debugging
            #print(gsGerm.operations["Ggerm"].evals)
            #print(gsGerm.operations["Ggerm"].params)

            #Get dP-matrix for full set of fiducials, where
            # P_ij = <E_i|germ^exp|rho_j>, i = composite EVec & fiducial index,
            #   j is similar, and derivs are wrt the "eigenvalues" of the germ
            #  (i.e. the parameters of the gsGerm model).
            lst = _gsc.create_circuits(
                "pp[0]+f0+germ+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
                germ=_circuits.Circuit(("Ggerm",)), pp=pre_povm_tuples,
                order=('f0', 'f1', 'pp'))

            resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
            layout = gsGerm.sim.create_layout(lst, None, resource_alloc, array_types=('ep',), verbosity=0)

            elIndicesForPair = [[] for i in range(len(prep_fiducials) * len(meas_fiducials))]
            nPrepPOVM = len(pre_povm_tuples)
            for k in range(len(prep_fiducials) * len(meas_fiducials)):
                for o in range(k * nPrepPOVM, (k + 1) * nPrepPOVM):
                    # "original" indices into lst for k-th fiducial pair
                    elIndicesForPair[k].extend(_slct.to_array(layout.indices_for_index(o)))

            local_dPall = layout.allocate_local_array('ep', 'd')
            gsGerm.sim.bulk_fill_dprobs(local_dPall, layout, None)  # num_els x num_params
            dPall = local_dPall.copy()  # local == global (no layout.gather required) b/c we used comm=None above
            layout.free_local_array(local_dPall)  # not needed - local_dPall isn't shared (comm=None)

            # Construct sum of projectors onto the directions (1D spaces)
            # corresponding to varying each parameter (~eigenvalue) of the
            # germ.  If the set of fiducials is sufficient, then the rank of
            # the resulting operator will equal the number of parameters,
            # indicating that the P matrix is (independently) sensitive to
            # each of the germ parameters (~eigenvalues), which is *all* we
            # want sensitivity to.
            RANK_TOL = 1e-7
            rank = _np.linalg.matrix_rank(_np.dot(dPall, dPall.T), RANK_TOL)
            if rank < gsGerm.num_params:  # full fiducial set should work!
                raise ValueError("Incomplete fiducial-pair set!")

            #Below will take a *subset* of the rows in dPall
            # depending on which (of all possible) fiducial pairs
            # are being considered.

            # nRhoStrs, nEStrs = len(prep_fiducials), len(meas_fiducials)
            nEStrs = len(meas_fiducials)
            nPossiblePairs = len(prep_fiducials) * len(meas_fiducials)
            allPairIndices = list(range(nPossiblePairs))

            #Determine which fiducial-pair indices to iterate over
            goodPairList = None; maxRank = 0
            for nNeededPairs in range(gsGerm.num_params, nPossiblePairs):
                printer.log("Beginning search for a good set of %d pairs (%d pair lists to test)" %
                            (nNeededPairs, _nCr(nPossiblePairs, nNeededPairs)), 2)

                if search_mode == "sequential":
                    pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

                elif search_mode == "random":
                    _random.seed(seed)  # ok if seed is None
                    nTotalPairCombos = _nCr(len(allPairIndices), nNeededPairs)
                    if n_random < nTotalPairCombos:
                        pairIndicesToIterateOver = [_random_combination(
                            allPairIndices, nNeededPairs) for i in range(n_random)]
                    else:
                        pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

                for pairIndicesToTest in pairIndicesToIterateOver:

                    #Get list of pairs as tuples for printing & returning
                    pairList = []
                    for i in pairIndicesToTest:
                        prepfid_index = i // nEStrs; iEStr = i - prepfid_index * nEStrs
                        pairList.append((prepfid_index, iEStr))

                    # Same computation of rank as above, but with only a
                    # subset of the total fiducial pairs.
                    elementIndicesToTest = _np.concatenate([elIndicesForPair[i] for i in pairIndicesToTest])
                    dP = _np.take(dPall, elementIndicesToTest, axis=0)  # subset_of_num_elements x num_params
                    rank = _np.linalg.matrix_rank(_np.dot(dP, dP.T), RANK_TOL)
                    maxRank = max(maxRank, rank)

                    printer.log("Pair list %s ==> %d of %d amplified parameters"
                                % (" ".join(map(str, pairList)), rank,
                                   gsGerm.num_params), 3)

                    if rank == gsGerm.num_params:
                        printer.log("Found a good set of %d pairs: %s" %
                                    (nNeededPairs, " ".join(map(str, pairList))), 2)
                        goodPairList = pairList
                        break

                if goodPairList is not None:
                    break  # exit another loop level if a solution was found

            assert(goodPairList is not None)
            pairListDict[germ] = goodPairList  # add to final list of per-germ pairs

            #TODO REMOVE: should never be called b/c of ValueError catch above for insufficient fidicials
            #else:
            #    #we tried all the way to nPossiblePairs-1 and no success,
            #    # just return all the pairs
            #    printer.log(" --> Highest number amplified = %d of %d" %
            #                (maxRank, gsGerms.num_params))
            #    listOfAllPairs = [ (prepfid_index,iEStr)
            #                       for prepfid_index in range(nRhoStrs)
            #                       for iEStr in range(nEStrs) ]
            #    pairListDict[germ] = listOfAllPairs

    return pairListDict


def test_fiducial_pairs(fid_pairs, target_model, prep_fiducials, meas_fiducials, germs,
                        test_lengths=(256, 2048), pre_povm_tuples="first", tol=0.75,
                        verbosity=0, mem_limit=None):
    """
    Tests a set of global or per-germ fiducial pairs.

    Determines how many model parameters (of `target_model`) are
    amplified by the fiducial pairs given by `fid_pairs`, which can be
    either a list of 2-tuples (for global-FPR) or a dictionary (for
    per-germ FPR).

    Parameters
    ----------
    fid_pairs : list or dict
        Either a single list of fiducial-index pairs (2-tuples) that is applied
        to every germ (global FPR) OR a per-germ dictionary of lists, each
        containing the fiducial-index pairs (2-tuples) for that germ (for
        per-germ FPR).

    target_model : Model
        The target model used to determine amplificational completeness.

    prep_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective preparation.

    meas_fiducials : list of Circuits
        Fiducial circuits used to construct an informationally complete
        effective measurement.

    germs : list of Circuits
        The germ circuits that are repeated to amplify errors.

    test_lengths : (L1,L2) tuple of ints, optional
        A tuple of integers specifying the germ-power lengths to use when
        checking for amplificational completeness.

    pre_povm_tuples : list or "first", optional
        A list of `(prepLabel, povmLabel)` tuples to consider when
        checking for completeness.  Usually this should be left as the special
        (and default) value "first", which considers the first prep and POVM
        contained in `target_model`.

    tol : float, optional
        The tolerance for the fraction of the expected amplification that must
        be observed to call a parameter "amplified".

    verbosity : int, optional
        How much detail to print to stdout.

    mem_limit : int, optional
        A memory limit in bytes.

    Returns
    -------
    numAmplified : int
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    if pre_povm_tuples == "first":
        firstRho = list(target_model.preps.keys())[0]
        firstPOVM = list(target_model.povms.keys())[0]
        pre_povm_tuples = [(firstRho, firstPOVM)]
    pre_povm_tuples = [(_circuits.Circuit((prepLbl,)), _circuits.Circuit((povmLbl,)))
                       for prepLbl, povmLbl in pre_povm_tuples]

    def get_derivs(length):
        """ Compute all derivative info: get derivative of each <E_i|germ^exp|rho_j>
            where i = composite EVec & fiducial index and j similar """

        circuits = []
        for germ in germs:
            expGerm = _gsc.repeat_with_max_length(germ, length)  # could pass exponent and set to germ**exp here
            pairList = fid_pairs[germ] if isinstance(fid_pairs, dict) else fid_pairs
            circuits += _gsc.create_circuits("pp[0]+p[0]+expGerm+p[1]+pp[1]",
                                             p=[(prep_fiducials[i], meas_fiducials[j]) for i, j in pairList],
                                             pp=pre_povm_tuples, expGerm=expGerm, order=['p', 'pp'])
        circuits = _remove_duplicates(circuits)

        resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
        layout = target_model.sim.create_layout(circuits, None, resource_alloc, array_types=('ep',), verbosity=0)

        local_dP = layout.allocate_local_array('ep', 'd')
        target_model.sim.bulk_fill_dprobs(local_dP, layout, None)
        dP = local_dP.copy()  # local == global (no layout.gather required) b/c we used comm=None above
        layout.free_local_array(local_dP)  # not needed - local_dP isn't shared (comm=None)

        return dP

    def get_number_amplified(m0, m1, len0, len1):
        """ Return the number of amplified parameters """
        L_ratio = float(len1) / float(len0)
        try:
            s0 = _np.linalg.svd(m0, compute_uv=False)
            s1 = _np.linalg.svd(m1, compute_uv=False)
        except:                                       # pragma: no cover
            printer.warning("SVD error!!"); return 0  # pragma: no cover
            #SVD did not converge -> just say no amplified params...

        numAmplified = 0
        printer.log("Amplified parameter test: matrices are %s and %s." % (m0.shape, m1.shape), 4)
        printer.log("Index : SV(L=%d)  SV(L=%d)  AmpTest ( > %g ?)" % (len0, len1, tol), 4)
        for i, (v0, v1) in enumerate(zip(sorted(s0, reverse=True), sorted(s1, reverse=True))):
            if abs(v0) > 0.1 and (v1 / v0) / L_ratio > tol:
                numAmplified += 1
                printer.log("%d: %g  %g  %g YES" % (i, v0, v1, (v1 / v0) / L_ratio), 4)
            printer.log("%d: %g  %g  %g NO" % (i, v0, v1, (v1 / v0) / L_ratio), 4)
        return numAmplified

    L0, L1 = test_lengths

    printer.log("----------  Testing Fiducial Pairs ----------")
    printer.log("Getting jacobian at L=%d" % L0, 2)
    dP0 = get_derivs(L0)
    printer.log("Getting jacobian at L=%d" % L1, 2)
    dP1 = get_derivs(L1)
    printer.log("Computing number amplified", 2)
    nAmplified = get_number_amplified(dP0, dP1, L0, L1)
    printer.log("Number of amplified parameters = %s" % nAmplified)

    return nAmplified
