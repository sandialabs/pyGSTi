"""
Functions for reducing the number of required fiducial pairs for analysis.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
import scipy.linalg as _sla

from math import ceil

from pygsti import baseobjs as _baseobjs
from pygsti import circuits as _circuits

from pygsti.circuits import circuitconstruction as _gsc
from pygsti.modelmembers.operations import EigenvalueParamDenseOp as _EigenvalueParamDenseOp
from pygsti.tools import remove_duplicates as _remove_duplicates
from pygsti.tools import slicetools as _slct
from pygsti.tools.legacytools import deprecate as _deprecated_fn
from pygsti.forwardsims import MatrixForwardSimulator as _MatrixForwardSimulator

from pygsti.algorithms.germselection import construct_update_cache, minamide_style_inverse_trace, compact_EVD, germ_set_spanning_vectors
from pygsti.algorithms import scoring as _scoring


import warnings


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


@_deprecated_fn('find_sufficient_fiducial_pairs_per_germ')
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
    prep_povm_tuples = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs = False)

    def _get_derivs(length):
        """ Compute all derivative info: get derivative of each `<E_i|germ^exp|rho_j>`
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

    def _get_number_amplified(m0, m1, len0, len1, verb):
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

    L0 = test_lengths[0]; dP0, elIndices0 = _get_derivs(L0)
    L1 = test_lengths[1]; dP1, elIndices1 = _get_derivs(L1)
    fullTestMx0 = dP0
    fullTestMx1 = dP1

    #Get number of amplified parameters in the "full" test matrix: the one we get when we use all possible fiducial
    #pairs
    if test_pair_list is None:
        maxAmplified = _get_number_amplified(fullTestMx0, fullTestMx1, L0, L1, verbosity + 1)
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
        nAmplified = _get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
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
            nAmplified = _get_number_amplified(testMx0, testMx1, L0, L1, verbosity)
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
                                            germs, prep_povm_tuples="first",
                                            search_mode="random", constrain_to_tp=True,
                                            n_random=100, min_iterations=None, base_loweig_tol= 1e-1,
                                            seed=None ,verbosity=0, num_soln_returned=1, type_soln_returned='best', retry_for_smaller=True,
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

    prep_povm_tuples : list or "first", optional
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
        
    min_iterations : int, optional
        A minimum number of candidate fiducial sets to try for a given
        set size before allowing the search to exit early in the event
        an acceptable candidate solution has already been found.
        
    base_loweig_tol : float, optional (default 1e-1)
        A relative threshold value for determining if a fiducial set
        is an acceptable candidate solution. The tolerance value indicates
        the decrease in the magnitude of the smallest eigenvalue of
        the Jacobian we're will to accept relative to that of the full
        fiducial set.

    seed : int, optional
        The seed to use for generating random-pair-sets.

    verbosity : int, optional
        How much detail to print to stdout.
       
    num_soln_returned : int, optional
        The number of candidate solutions to return for each run of the fiducial pair search.
        
    type_soln_returned : str, optional
        Which type of criteria to use when selecting which of potentially many candidate fiducial pairs to search through.
        Currently only "best" supported which returns the num_soln_returned best candidates as measured by minimum eigenvalue.
        
    retry_for_smaller : bool, optional
        If true then a second search is performed seeded by the candidate solution sets found in the first pass.
        The search routine then randomly subsamples sets of fiducial pairs from these candidate solutions to see if
        a smaller subset will suffice.

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
    
    prep_povm_tuples, dof_per_povm = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs = True)
           
    pairListDict = {}  # dict of lists of 2-tuples: one pair list per germ
    
    if min_iterations is None:
        min_iterations = min(n_random // 2, 1000) if search_mode == 'random' else 10  # HARDCODED
        #also assert that the number of iterations is less than the number of random samples
        if search_mode=='random':
            assert(min_iterations<=n_random)
            
            
    if (not retry_for_smaller) and (num_soln_returned>1):
        warnings.warn('You are not retrying for smaller solutions, so returning more than 1 candidate solution is not useful.')

    printer.log("------  Per Germ (L=1) Fiducial Pair Reduction --------")
    with printer.progress_logging(1):
        for i, germ in enumerate(germs):
            #Create a new model containing static target gates and a
            # special "germ" gate that is parameterized only by it's
            # eigenvalues (and relevant off-diagonal elements)
            gsGerm = target_model.copy()
            gsGerm.set_all_parameterizations("static")
            germMx = gsGerm.sim.product(germ)
            #give this state space labels equal to the line_labels of 
            gsGerm.operations['Ggerm'] = _EigenvalueParamDenseOp(
                germMx, True, constrain_to_tp)

            printer.show_progress(i, len(germs),
                                  suffix='-- %s germ (%d params)' %
                                  (repr(germ), gsGerm.num_params))

            #Determine which fiducial-pair indices to iterate over
            #initial run
            candidate_solution_list, bestFirstEval = _get_per_germ_power_fidpairs(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                                                    gsGerm, 1, mem_limit,
                                                                    printer, search_mode, seed, n_random, dof_per_povm,
                                                                    min_iterations, base_loweig_tol, candidate_set_seed=None,
                                                                    num_soln_returned=num_soln_returned, type_soln_returned=type_soln_returned,
                                                                    germ_circuit=germ)
                                                                   
            #the algorithm isn't guaranteed to actually find the requested number of solutions, so check how many there actually are
            #by checking the length of the list of returned eigenvalues. 
            actual_num_soln_returned= len(bestFirstEval)
            
            printer.log('Found %d solutions out of %d requested.'%(actual_num_soln_returned, num_soln_returned),2)
            
            #The goodPairList is now a dictionary with the keys corresponding to the minimum eigenvalue of the candidate solution.
            #iterate through the values of the dictionary.
            if retry_for_smaller:
            
                printer.log('Resampling from returned solutions to search for smaller sets.',2)
                
                assert(actual_num_soln_returned!=0)
                
                updated_solns=[]
                for candidate_solution in candidate_solution_list.values():
                    #now do a seeded run for each of the candidate solutions returned in the initial run:
                    #for these internal runs just return a single solution. 
                    reducedPairlist, bestFirstEval = _get_per_germ_power_fidpairs(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                                                            gsGerm, 1, mem_limit,
                                                                            printer, search_mode, seed, n_random, dof_per_povm,
                                                                            min_iterations, base_loweig_tol, candidate_set_seed=candidate_solution,
                                                                            num_soln_returned= 1, type_soln_returned='best',
                                                                            germ_circuit=germ)
                    #This should now return a dictionary with a single entry. Append that entry to a running list which we'll process at the end.
                    updated_solns.append(list(reducedPairlist.values())[0])

                printer.log('Finished resampling from returned solutions to search for smaller sets.',2)

                #At the very worst we should find that the updated solutions are the same length as the original candidate we seeded with
                #(in fact, it would just return the seed in that case). So we should be able to just check for which of the lists of fiducial pairs is shortest.
                solution_lengths= [len(fid_pair_list) for fid_pair_list in updated_solns]
                #get the index of the minimum length set.
                min_length_idx= _np.argmin(solution_lengths)
                
                #set the value of goodPairList to be this value.
                goodPairList= updated_solns[min_length_idx]
                    
                #print some output about the minimum eigenvalue acheived.
                printer.log('Minimum Eigenvalue Achieved: %f' %(bestFirstEval[0]), 3)

            else:
                #take the first entry of the candidate solution list if there is more than one.
                goodPairList= list(candidate_solution_list.values())[0]
                bestFirstEval=bestFirstEval[0]
            
                #print some output about the minimum eigenvalue acheived.
                printer.log('Minimum Eigenvalue Achieved: %f' %(bestFirstEval), 3)
            
            try:
                assert(goodPairList is not None)
            except AssertionError as err:
                print('Failed to find an acceptable fiducial set for germ power pair: ', germ)
                print(err)

            pairListDict[germ] = goodPairList  # add to final list of per-germ pairs

    return pairListDict
    
def find_sufficient_fiducial_pairs_per_germ_greedy(target_model, prep_fiducials, meas_fiducials,
                                                germs, prep_povm_tuples="first", constrain_to_tp=True,
                                                inv_trace_tol= 10, initial_seed_mode='random',
                                                evd_tol=1e-10, sensitivity_threshold=1e-10, seed=None ,verbosity=0, check_complete_fid_set=True,
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

    prep_povm_tuples : list or "first", optional
        A list of `(prepLabel, povmLabel)` tuples to consider when
        checking for completeness.  Usually this should be left as the special
        (and default) value "first", which considers the first prep and POVM
        contained in `target_model`.

    constrain_to_tp : bool, optional (default True)
        Whether or not to consider non-TP parameters the the germs amplify.  If
        the fiducal pairs will be used in a GST estimation where the model is
        constrained to being trace-preserving (TP), this should be set to True.

    inv_trace_tol : float, optional (default 10)
        Tolerance used to identify whether a candidate fiducial set has a feasible
        cost function value and for comparing among both complete and incomplete candidate
        sets. The cost function corresonds to trace(pinv(sum_i(dot(J_i,J_i^T)))), where
        the J_i's are described above, which is related to a condition in optimal 
        design theory called a-optimality. The tolerance is a relative tolerance 
        compared to the complete fiducial set. This essentially measures the relative
        sensitivity loss we're willing to accept as part of applying FPR. 

    initial_seed_mode : str, optional (default 'random')
        A string corresponding to the method used to seed an initial set of fiducial
        pairs in starting the greedy search routine. The default 'random' identifies
        the minimum number of fiducial pairs required for sensitivity to every parameter
        of a given germ and then randomly samples that number of fiducial pairs to use
        as an initial seed. Also supports the option 'greedy' in which case we start the
        greedy search from scratch using an empty seed set and select every fiducial pair
        greedily. Random is generally faster computationally, but greedy has been fount to
        often (but not always) find a smaller solution.

    evd_tol : float, optional (default 1e-10)
        Threshold value for eigenvalues below which they are treated as zero.
        Used in the construction of compact eigenvalue decompositions
        (technically rank-decompositions) of jacobians. If encountering an error
        related to failing Cholesky decompositions consider increasing the value
        of this tolerance.

    sensitivity_threshold : float, optional (default 1e-10)
        Threshold used for determing is a fiducial pair is useless 
        for measuring a given germ's amplified parameters due to 
        trivial sensitivity to the germ kite parameters 
        (norm of jacobian w.r.t. the kite parameters is <sensitivity_threshold).
    
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

    prep_povm_tuples, dof_per_povm = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs=True)

    pairListDict = {}  # dict of lists of 2-tuples: one pair list per germ
    
    #log the total number of parameters amplified overall.
    total_num_amplified_parameters=0
    
    printer.log("------  Per Germ (L=1) Fiducial Pair Reduction --------")
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
            total_num_amplified_parameters += gsGerm.num_params
            

            printer.show_progress(i, len(germs),
                                  suffix='-- %s germ (%d params)' %
                                  (repr(germ), gsGerm.num_params))

            #Determine which fiducial-pair indices to iterate over
            #initial run
            candidate_solution_list, best_score = _get_per_germ_power_fidpairs_greedy(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                                                    gsGerm, 1, mem_limit,
                                                                    printer, seed, dof_per_povm,
                                                                    inv_trace_tol, initial_seed_mode=initial_seed_mode,
                                                                    check_complete_fid_set=check_complete_fid_set, evd_tol=evd_tol,
                                                                    sensitivity_threshold=sensitivity_threshold,
                                                                    germ_circuit= germ)
            
            #print some output about the minimum eigenvalue acheived.
            printer.log('Score Achieved: ' + str(best_score), 2)
            
            try:
                assert(candidate_solution_list is not None)
            except AssertionError as err:
                print('Failed to find an acceptable fiducial set for germ power pair: ', germ)
                print(err)

            pairListDict[germ] = candidate_solution_list  # add to final list of per-germ pairs
            
    printer.log('Total number of amplified parameters = %d'%(total_num_amplified_parameters),2)

    return pairListDict

def find_sufficient_fiducial_pairs_per_germ_power(target_model, prep_fiducials, meas_fiducials,
                                                  germs, max_lengths,
                                                  prep_povm_tuples="first",
                                                  search_mode="random", constrain_to_tp=True,
                                                  trunc_scheme="whole germ powers",
                                                  n_random=100, min_iterations=None, base_loweig_tol= 1e-1, seed=None,
                                                  verbosity=0, mem_limit=None, per_germ_candidate_set=None):
    """
    Finds a per-germ set of fiducial pairs that are amplificationally complete.

    A "standard" set of GST circuits consists of all circuits of the form:

    Case: trunc_scheme == 'whole germ powers':
      state_prep + prep_fiducial + pygsti.circuits.repeat_with_max_length(germ,L) + meas_fiducial + meas

    Case: trunc_scheme == 'truncated germ powers':
      state_prep + prep_fiducial + pygsti.circuits.repeat_and_truncate(germ,L) + meas_fiducial + meas

    Case: trunc_scheme == 'length as exponent':
      state_prep + prep_fiducial + germ^L + meas_fiducial + meas

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

    max_lengths: list of int
        The germ powers (number of repetitions) to be used to amplify errors.

    prep_povm_tuples : list or "first", optional
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
    
    min_iterations: int, optional
        The number of random-pair-sets to consider before we allow the algorithm
        to terminate early if it has found an acceptable candidate fiducial set.
        If left with the default value of None then this is 1/2 the value of
        n_random.
        
    base_loweig_tol: float, optional
        A relative tolerance to apply to candidate fiducial pair sets. 
        Gives the multiplicative reduction in the magnitude of the minimum
        eigenvalue relative to the value for the full fiducial set
        the user is willing to tolerate.

    seed : int, optional
        The seed to use for generating random-pair-sets.

    verbosity : int, optional
        How much detail to print to stdout.

    mem_limit : int, optional
        A memory limit in bytes.
    
    per_germ_candidate_set : dict, optional
        If specified, this is a dictionary with keys given by the germ set. This dictionary
        is a previously found candidate set of fiducials output from the per-germ FPR function
        find_sufficient_fiducial_pairs_per_germ.
        

    Returns
    -------
    dict
        A dictionary whose keys are the germ circuits and whose values are
        lists of (iRhoFid,iMeasFid) tuples of integers, each specifying the
        list of fiducial pairs for a particular germ (indices are into
        `prep_fiducials` and `meas_fiducials`).
    """
    
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    prep_povm_tuples = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs=False)
    
    pairListDict = {}  # dict of lists of 2-tuples: one pair list per germ
    
    #Check whether the user has passed in a candidate set as a seed from a previous run of
    #per-germ FPR.
    if per_germ_candidate_set is not None:
        #in that case check that all of the germs are accounted for.
        try:
            assert(set(per_germ_candidate_set.keys()) == set(germs))
        except AssertionError as err:
            print('Candidate germs in seed set not equal to germs passed into this function.')
            print(err)    

    printer.log("------  Per Germ-Power Fiducial Pair Reduction --------")
    printer.log("  Using %s germ power truncation scheme" % trunc_scheme)
    with printer.progress_logging(1):
        for i, germ_power in enumerate(_itertools.product(germs, max_lengths)):
            germ, L = germ_power
            # TODO: Could check for when we become identity and skip rest
            #Create a new model containing static target gates and a
            # special "germ" gate that is parameterized only by it's
            # eigenvalues (and relevant off-diagonal elements)
            gsGerm = target_model.copy()
            gsGerm.set_all_parameterizations("static")
            germMx = gsGerm.sim.product(germ)
            
            gsGerm.operations["Ggerm"] = _EigenvalueParamDenseOp(
                germMx, True, constrain_to_tp)

            # SS: Difference from _per_germ version
            if trunc_scheme == "whole germ powers":
                power = _gsc.repeat_count_with_max_length(germ, L)
 
            # TODO: Truncation doesn't work nicely with a single "germ"
            #elif trunc_scheme == "truncated germ powers":
            #    germPowerCirc = _gsc.repeat_and_truncate(germ, L)
            elif trunc_scheme == "length as exponent":
                power = L

            else:
                raise ValueError("Truncation scheme %s not allowed" % trunc_scheme)

            if power == 0:
                # Skip empty circuits (i.e. germ^power > max_length)
                printer.show_progress(i, len(germs) * len(max_lengths),
                                      suffix='-- %s germ skipped since longer than max length %d' %
                                      (repr(germ), L))
                continue

            printer.show_progress(i, len(germs) * len(max_lengths),
                                  suffix='-- %s germ, %d L (%d params)' %
                                  (repr(germ), L, gsGerm.num_params))

            #Determine which fiducial-pair indices to iterate over
            #TODO: Evaluate the value of the minimum number of iterations before the algorithm for
            #getting candidate fiducial pairs is allowed to exit early.
            if min_iterations is None:
                min_iterations = min(n_random // 2, 1000) if search_mode == 'random' else 10  # HARDCODED
            #also assert that the number of iterations is less than the number of random samples
            if search_mode=='random':
                assert(min_iterations<=n_random)
                
            #if there is a candidate fiducial seed set pass that in, otherwise pass in None.
            if per_germ_candidate_set is not None:
                candidate_set_seed= per_germ_candidate_set[germ]
            else:
                candidate_set_seed= None
            
            goodPairList, _ = _get_per_germ_power_fidpairs(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                                                gsGerm, power, mem_limit,
                                                                printer, search_mode, seed, n_random,
                                                                min_iterations, base_loweig_tol, candidate_set_seed,
                                                                num_soln_returned=1, type_soln_returned='best', germ_circuit = germ)
                                                                
            #This should now return a dictionary with a single entry. pull that entry out.
            goodPairList= list(goodPairList.values())[0]
            
            try:
                assert(goodPairList is not None)
            except AssertionError as err:
                print('Failed to find an acceptable fiducial set for germ power pair: ', germ_power)
                print(err)
            pairListDict[germ_power] = goodPairList  # add to final list of per-germ-power pairs

    return pairListDict

def test_fiducial_pairs(fid_pairs, target_model, prep_fiducials, meas_fiducials, germs,
                        test_lengths=(256, 2048), prep_povm_tuples="first", tol=0.75,
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

    prep_povm_tuples : list or "first", optional
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

    prep_povm_tuples = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs=False)

    def _get_derivs(length):
        """ Compute all derivative info: get derivative of each `<E_i|germ^exp|rho_j>`
            where i = composite EVec & fiducial index and j similar """

        circuits = []
        for germ in germs:
            expGerm = _gsc.repeat_with_max_length(germ, length)  # could pass exponent and set to germ**exp here
            pairList = fid_pairs[germ] if isinstance(fid_pairs, dict) else fid_pairs
            circuits += _gsc.create_circuits("pp[0]+p[0]+expGerm+p[1]+pp[1]",
                                             p=[(prep_fiducials[i], meas_fiducials[j]) for i, j in pairList],
                                             pp=prep_povm_tuples, expGerm=expGerm, order=['p', 'pp'])
        circuits = _remove_duplicates(circuits)

        resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
        layout = target_model.sim.create_layout(circuits, None, resource_alloc, array_types=('ep',), verbosity=0)

        local_dP = layout.allocate_local_array('ep', 'd')
        target_model.sim.bulk_fill_dprobs(local_dP, layout, None)
        dP = local_dP.copy()  # local == global (no layout.gather required) b/c we used comm=None above
        layout.free_local_array(local_dP)  # not needed - local_dP isn't shared (comm=None)

        return dP

    def _get_number_amplified(m0, m1, len0, len1):
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
    dP0 = _get_derivs(L0)
    printer.log("Getting jacobian at L=%d" % L1, 2)
    dP1 = _get_derivs(L1)
    printer.log("Computing number amplified", 2)
    nAmplified = _get_number_amplified(dP0, dP1, L0, L1)
    printer.log("Number of amplified parameters = %s" % nAmplified)

    return nAmplified


# Helper function for per_germ and per_germ_power FPR
def _get_per_germ_power_fidpairs(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                 gsGerm, power, mem_limit, printer, search_mode, seed, n_random, dof_per_povm,
                                 min_iterations=1, lowest_eigenval_tol=1e-1,
                                 candidate_set_seed=None, num_soln_returned=1, type_soln_returned='best',
                                 germ_circuit = None):
    #Get dP-matrix for full set of fiducials, where
    # P_ij = <E_i|germ^exp|rho_j>, i = composite EVec & fiducial index,
    #   j is similar, and derivs are wrt the "eigenvalues" of the germ
    #  (i.e. the parameters of the gsGerm model).
    
    printer.log('Entered helper function _get_per_germ_power_fidpairs', 3)

    # nRhoStrs, nEStrs = len(prep_fiducials), len(meas_fiducials)
    nEStrs = len(meas_fiducials)
    nPossiblePairs = len(prep_fiducials) * len(meas_fiducials)
    
    allPairIndices = list(range(nPossiblePairs))
    
    printer.log('Number of possible pairs: %d'%(nPossiblePairs), 4)

    #Determine which fiducial-pair indices to iterate over
    goodPairList = None; bestFirstEval = []; bestPairs = {}
    #loops over a number of pairs between min_pairs_needed and up to and not including the number of possible pairs
    
    min_pairs_needed= ceil((gsGerm.num_params/(nPossiblePairs*dof_per_povm))*nPossiblePairs)
    printer.log('Minimum Number of Pairs Needed for this Germ: %d'%(min_pairs_needed), 2)    
    line_labels = germ_circuit.line_labels if germ_circuit is not None else 'auto'

    lst = _gsc.create_circuits(
        "pp[0]+f0+germ*power+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
        germ=_circuits.Circuit(['Ggerm'], line_labels=line_labels), pp=prep_povm_tuples, power=power,
        order=('f0', 'f1', 'pp'))

    resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
    layout = gsGerm.sim.create_layout(lst, None, resource_alloc, array_types=('ep',), verbosity=0)

    elIndicesForPair = [[] for i in range(len(prep_fiducials) * len(meas_fiducials))]
    nPrepPOVM = len(prep_povm_tuples)
    for k in range(len(prep_fiducials) * len(meas_fiducials)):
        for o in range(k * nPrepPOVM, (k + 1) * nPrepPOVM):
            # "original" indices into lst for k-th fiducial pair
            elIndicesForPair[k].extend(_slct.to_array(layout.indices_for_index(o)))
    
    printer.log('Constructing Jacobian for Full Fiducial Set' , 3)
    local_dPall = layout.allocate_local_array('ep', 'd')
    gsGerm.sim.bulk_fill_dprobs(local_dPall, layout, None)  # num_els x num_params
    dPall = local_dPall.copy()  # local == global (no layout.gather required) b/c we used comm=None above
    layout.free_local_array(local_dPall)  # not needed - local_dPall isn't shared (comm=None)
    
    printer.log('Calculating Spectrum of Full Fiducial Set', 3)
    
    # Construct sum of projectors onto the directions (1D spaces)
    # corresponding to varying each parameter (~eigenvalue) of the
    # germ.  If the set of fiducials is sufficient, then the rank of
    # the resulting operator will equal the number of parameters,
    # indicating that the P matrix is (independently) sensitive to
    # each of the germ parameters (~eigenvalues), which is *all* we
    # want sensitivity to.
    RANK_TOL = 1e-7 #HARDCODED
    #rank = _np.linalg.matrix_rank(_np.dot(dPall, dPall.T), RANK_TOL)
    
    spectrum_full_fid_set= _np.abs(_np.linalg.eigvalsh(_np.dot(dPall, dPall.T)))
    
    #use the spectrum to calculate the rank instead.
    rank= _np.count_nonzero(spectrum_full_fid_set>RANK_TOL)
    
    if rank < gsGerm.num_params:  # full fiducial set should work!
        raise ValueError("Incomplete fiducial-pair set!")

    spectrum_full_fid_set= list(sorted(_np.abs(_np.linalg.eigvalsh(_np.dot(dPall, dPall.T)))))
    
    imin_full_fid_set = len(spectrum_full_fid_set) - gsGerm.num_params
    condition_full_fid_set = spectrum_full_fid_set[-1] / spectrum_full_fid_set[imin_full_fid_set] if (spectrum_full_fid_set[imin_full_fid_set] > 0) else _np.inf
    
    printer.log('J J^T Rank Full Fiducial Set: %d' % rank, 2)
    printer.log('Num Parameters (should equal above rank): %d' % (gsGerm.num_params), 2)
    printer.log('Full Fiducial Set Min Eigenvalue: %f' % (spectrum_full_fid_set[imin_full_fid_set]), 2)
    printer.log('Full Fiducial Set Max Eigenvalue: %f' % (spectrum_full_fid_set[-1]), 2)
    printer.log('Full Fiducial Set Condition Number: %f' % (condition_full_fid_set), 2)

    #Below will take a *subset* of the rows in dPall
    # depending on which (of all possible) fiducial pairs
    # are being considered.
    
    #if we have a candidate seed set from per-germ FPR then we'll search for candidate sets which are subsets of the
    #seed set up to the size of candidate set. If we fail to find an appropriate set from among those subsets then we'll
    #go through the standard search algorithm. 
    
    rng= _np.random.default_rng(seed=seed)
    found_from_seed_set=False
    
    if candidate_set_seed is not None:
        printer.log('Searching from among subsets of the candidate seed set.', 3)
        size_candidate_set= len(candidate_set_seed)
        for nNeededPairs in range(min_pairs_needed, size_candidate_set+1):
            printer.log("Beginning search for a good set of %d pairs (%.1e pair lists to test)" %
                        (nNeededPairs, _nCr(nPossiblePairs, nNeededPairs)), 3)
            printer.log("  Low eigenvalue must be >= %g relative to the values of the full fiducial set" %
                        (lowest_eigenval_tol),3)
        
            printer.log('Searching for a good set with this many pairs: %d' % (nNeededPairs), 4)
            
            #We'll ignore the search mode argument and just focus on sampling random subsets of the candidate seed set.
            nTotalPairCombos = _nCr(size_candidate_set, nNeededPairs)
            
            #convert the candidate_set_seed which is a list of pairs of indices to an equivalent linear index.
            #Should be lin_idx= prep_idx+num_meas+meas_idx
            linearized_candidate_seed_set=[]
            for pair in candidate_set_seed:
                linearized_candidate_seed_set.append(pair[0]*nEStrs+pair[1])
            
            if n_random < nTotalPairCombos:
                pairIndicesToIterateOver = [rng.choice(linearized_candidate_seed_set, size=nNeededPairs, replace=False) for i in range(n_random)]
                #this will return numpy int64 values. we want standard python integers for the sake of serialization.
                #cast these values back.
                pairIndicesToIterateOver= [ [int(value) for value in nppairindexlist] for nppairindexlist in pairIndicesToIterateOver ]
            else:
                pairIndicesToIterateOver = _itertools.combinations(linearized_candidate_seed_set, nNeededPairs)
                
            max_rank_seen= 0    
                
            for i, pairIndicesToTest in enumerate(pairIndicesToIterateOver, start=1):
                
                #Get list of pairs as tuples for printing & returning
                pairList = []
                for i in pairIndicesToTest:
                    prepfid_index = i // nEStrs; iEStr = i - prepfid_index * nEStrs
                    pairList.append((prepfid_index, iEStr))
                
                # Same computation of rank as above, but with only a
                # subset of the total fiducial pairs.
                elementIndicesToTest = _np.concatenate([elIndicesForPair[i] for i in pairIndicesToTest])
                dP = _np.take(dPall, elementIndicesToTest, axis=0)  # subset_of_num_elements x num_params
                spectrum = _np.abs(_np.linalg.eigvalsh(_np.dot(dP, dP.T)))
                current_rank= _np.count_nonzero(spectrum>1e-10) #HARDCODED
                spectrum= list(sorted(spectrum))
                
                imin = len(spectrum) - gsGerm.num_params
                if current_rank > max_rank_seen:
                    max_rank_seen=current_rank
                
                if (spectrum[imin] >= (lowest_eigenval_tol*spectrum_full_fid_set[imin_full_fid_set])):
                    #if the list for bestFirstEval is empty or else we haven't hit the number of solutions to return
                    #yet then we'll append the value to the list and sort it regardless of the value. Otherwise
                    #we need to evaluate whether to swap out one of the elements of the list or not.
                    if len(bestFirstEval) < num_soln_returned:
                        bestFirstEval.append(spectrum[imin])
                        #keep the list sorted in descending order
                        bestFirstEval.sort(reverse=True)
                        #also add a corresponding entry to a dictionary for the best fiducial pairs we've seen thus far
                        #with the key given by the corresponding eigenvalue.
                        bestPairs[spectrum[imin]]= pairList
                    else:
                        if type_soln_returned=='best':
                            #if any of the eigenvalue are less than the one we found
                            #then we'll drop the last element of the bestFirstEval list
                            #append the new element to the list and re-sort the values.
                            if any([eigval<spectrum[imin] for eigval in bestFirstEval]):
                                #need to remove the entry corresponding to the smallest eigenvalue from the dictionary
                                #of fiducial pair sets and from the list of eigenvalues.
                                bestPairs.pop(bestFirstEval[-1])
                                bestFirstEval.pop()
                                
                                #add the new eigenvalue to the list and re-sort it.
                                bestFirstEval.append(spectrum[imin])
                                bestFirstEval.sort(reverse=True)
                                #also add a corresponding entry to a dictionary for the best fiducial pairs we've seen thus far
                                #with the key given by the corresponding eigenvalue.
                                bestPairs[spectrum[imin]]= pairList
                                
                        #for any other value in type_soln_returned raise a NotImplementedError.
                        #may implement other things later on.
                        else:
                            raise NotImplementedError('Only option currently implemented for type_soln_returned is \"best\".')

                if i >= min_iterations and len(bestFirstEval)>=num_soln_returned:
                    printer.log('we have looked long enough and have found the requested number of acceptable solutions from among the candidate seed set.', 3)
                    found_from_seed_set=True
                    break  # we've looked long enough and have found an acceptable solution

            printer.log('Maximum Rank Seen This Iteration: ' + str(max_rank_seen), 3)
            
            if any([eigval >= (lowest_eigenval_tol*spectrum_full_fid_set[imin_full_fid_set]) for eigval in bestFirstEval]):
                printer.log('Found at least one good set of pairs from within the seed set with length %d:' % (nNeededPairs), 3)
                found_from_seed_set=True
                goodPairList = bestPairs
                break
        if found_from_seed_set==False:
            printer.log('Failed to find acceptable candidate from among the seed set.', 3)
                
    #if we've found a good candidate set from the user specified seed then this will be skipped, else if we haven't
    #or there wasn't a candidate_set_seed passed in we'll default to the standard algorithm.
    if (candidate_set_seed is None) or (found_from_seed_set==False):
        for nNeededPairs in range(min_pairs_needed, nPossiblePairs):
            printer.log("Beginning search for a good set of %d pairs (%d pair lists to test)" %
                        (nNeededPairs, _nCr(nPossiblePairs, nNeededPairs)), 2)
            printer.log("  Low eigenvalue must be >= %g relative to the values of the full fiducial set" %
                        (lowest_eigenval_tol),2)
            
            printer.log('Searching for a good set with this many pairs: %d' % (nNeededPairs), 4)

            if search_mode == "sequential":
                printer.log("Performing sequential search", 3)
                pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs)

            elif search_mode == "random":
                _random.seed(seed)  # ok if seed is None
                nTotalPairCombos = _nCr(len(allPairIndices), nNeededPairs)

                if n_random < nTotalPairCombos:
                    pairIndicesToIterateOver = [_random_combination(
                        allPairIndices, nNeededPairs) for i in range(n_random)]
                else:
                    pairIndicesToIterateOver = _itertools.combinations(allPairIndices, nNeededPairs) 
            
            max_rank_seen= 0
            
            for i, pairIndicesToTest in enumerate(pairIndicesToIterateOver, start=1):                
                #Get list of pairs as tuples for printing & returning
                pairList = []
                for i in pairIndicesToTest:
                    prepfid_index = i // nEStrs; iEStr = i - prepfid_index * nEStrs
                    pairList.append((prepfid_index, iEStr))
                
                # Same computation of rank as above, but with only a
                # subset of the total fiducial pairs.
                elementIndicesToTest = _np.concatenate([elIndicesForPair[i] for i in pairIndicesToTest])
                dP = _np.take(dPall, elementIndicesToTest, axis=0)  # subset_of_num_elements x num_params
                
                spectrum = _np.abs(_np.linalg.eigvalsh(_np.dot(dP, dP.T)))
                current_rank= _np.count_nonzero(spectrum>1e-10)
                spectrum= list(sorted(spectrum))
                
                imin = len(spectrum) - gsGerm.num_params
                if current_rank > max_rank_seen:
                    max_rank_seen=current_rank
                
                if (spectrum[imin] >= (lowest_eigenval_tol*spectrum_full_fid_set[imin_full_fid_set])):# and condition <= (condition_number_tol*condition_full_fid_set)):
                    
                    #if the list for bestFirstEval is empty or else we haven't hit the number of solutions to return
                    #yet then we'll append the value to the list and sort it regardless of the value. Otherwise
                    #we need to evaluate whether to swap out one of the elements of the list or not.
                    if len(bestFirstEval)<num_soln_returned:
                        bestFirstEval.append(spectrum[imin])
                        #keep the list sorted in descending order
                        bestFirstEval.sort(reverse=True)
                        #also add a corresponding entry to a dictionary for the best fiducial pairs we've seen thus far
                        #with the key given by the corresponding eigenvalue.
                        bestPairs[spectrum[imin]]= pairList
                    else:
                        if type_soln_returned=='best':
                            #if the smallest eigenvalue is less than the one we found
                            #then we'll drop the last element of the bestFirstEval list
                            #append the new element to the list and re-sort the values.
                            if  bestFirstEval[-1] < spectrum[imin]:
                                #need to remove the entry corresponding to the smallest eigenvalue from the dictionary
                                #of fiducial pair sets and from the list of eigenvalues.
                                try:
                                    bestPairs.pop(bestFirstEval[-1])
                                except KeyError as err:
                                    printer.log(f"Trying to drop the element from bestPairs with key: {bestFirstEval[-1]}", 3)
                                    printer.log(f"Current keys in this dictionary: {bestPairs.keys()}", 3)
                                    
                                    #This seems to be happening when there are multiple entries with virtually
                                    #identical values for the keys. 
                                    
                                    #HACK
                                    #get the key that is closest to bestFirstEval[-1] and pop that, no idea why 
                                    #we're getting this tiny change in the floating point value when making it a key.
                                    closest_index = _np.argmin(_np.fromiter(bestPairs.keys(), dtype=_np.double)-bestFirstEval[-1])
                                    bestPairs.pop(bestFirstEval[closest_index])
                                    #raise err
                                bestFirstEval.pop()
                                
                                #add the new eigenvalue to the list and re-sort it.
                                bestFirstEval.append(spectrum[imin])
                                bestFirstEval.sort(reverse=True)
                                #also add a corresponding entry to a dictionary for the best fiducial pairs we've seen thus far
                                #with the key given by the corresponding eigenvalue.
                                bestPairs[spectrum[imin]]= pairList
                                
                        #for any other value in type_soln_returned raise a NotImplementedError.
                        #may implement other things later on.
                        else:
                            raise NotImplementedError('Only option currently implemented for type_soln_returned is \"best\".')

                if i >= min_iterations and len(bestFirstEval)>=num_soln_returned:
                    printer.log('we have looked long enough and have found the requested number of acceptable solutions.', 3)
                    break  # we've looked long enough and have found an acceptable solution
            
            printer.log('Maximum Rank Seen This Iteration: ' + str(max_rank_seen), 3)
            
            if any([eigval >= (lowest_eigenval_tol*spectrum_full_fid_set[imin_full_fid_set]) for eigval in bestFirstEval]):
                goodPairList = bestPairs
                break
    if goodPairList is None:
        printer.log('Failed to find a sufficient fiducial set.',1)
    printer.log('Exiting _get_per_germ_power_fidpairs', 4)
    return goodPairList, bestFirstEval
    

#New greedy-style FPR algorithm:
# Helper function for per_germ and per_germ_power FPR
# This version uses a greedy style algorithm and an
# alternative objective function which leverages the performance enhancements
# utilized for the germ selection algorithm using low-rank updates.
def _get_per_germ_power_fidpairs_greedy(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                 gsGerm, power, mem_limit, printer, seed, dof_per_povm,
                                 inv_trace_tol=10, initial_seed_mode= 'random',
                                 check_complete_fid_set= True, evd_tol=1e-10, sensitivity_threshold= 1e-10,
                                 germ_circuit = None):
    #Get dP-matrix for full set of fiducials, where
    # P_ij = <E_i|germ^exp|rho_j>, i = composite EVec & fiducial index,
    #   j is similar, and derivs are wrt the "eigenvalues" of the germ
    #  (i.e. the parameters of the gsGerm model).
    
    printer.log('Entered helper function _get_per_germ_power_fidpairs_greedy', 3)

    nEStrs = len(meas_fiducials)
    nPossiblePairs = len(prep_fiducials) * len(meas_fiducials)
    
    allPairIndices = list(range(nPossiblePairs))

    printer.log('Number of possible pairs: %d'%(nPossiblePairs), 4)

    #Determine which fiducial-pair indices to iterate over
    goodPairList = None; bestFirstEval = []; bestPairs = {}
    #loops over a number of pairs between min_pairs_needed and up to and not including the number of possible pairs
    
    min_pairs_needed= ceil((gsGerm.num_params/(nPossiblePairs*dof_per_povm))*nPossiblePairs)
    printer.log('Minimum Number of Pairs Needed for this Germ: %d'%(min_pairs_needed), 2)
    line_labels = germ_circuit.line_labels if germ_circuit is not None else 'auto'

    lst = _gsc.create_circuits(
        "pp[0]+f0+germ*power+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
        germ=_circuits.Circuit(['Ggerm'], line_labels=line_labels), pp=prep_povm_tuples, power=power,
        order=('f0', 'f1', 'pp'))

    resource_alloc = _baseobjs.ResourceAllocation(comm=None, mem_limit=mem_limit)
    layout = gsGerm.sim.create_layout(lst, None, resource_alloc, array_types=('ep',), verbosity=0)

    elIndicesForPair = [[] for i in range(len(prep_fiducials) * len(meas_fiducials))]
    nPrepPOVM = len(prep_povm_tuples)
    for k in range(len(prep_fiducials) * len(meas_fiducials)):
        for o in range(k * nPrepPOVM, (k + 1) * nPrepPOVM):
            # "original" indices into lst for k-th fiducial pair
            elIndicesForPair[k].extend(_slct.to_array(layout.indices_for_index(o)))
    
    printer.log('Constructing Jacobian for Full Fiducial Set' , 3)
    
    local_dPall = layout.allocate_local_array('ep', 'd')
    gsGerm.sim.bulk_fill_dprobs(local_dPall, layout, None)  # num_els x num_params
    dPall = local_dPall.copy()  # local == global (no layout.gather required) b/c we used comm=None above
    layout.free_local_array(local_dPall)  # not needed - local_dPall isn't shared (comm=None)
    
    printer.log('Calculating Inverse Trace of Full Fiducial Set', 3)
    
    # Construct sum of projectors onto the directions (1D spaces)
    # corresponding to varying each parameter (~eigenvalue) of the
    # germ.  If the set of fiducials is sufficient, then the rank of
    # the resulting operator will equal the number of parameters,
    # indicating that the P matrix is (independently) sensitive to
    # each of the germ parameters (~eigenvalues), which is *all* we
    # want sensitivity to.
    RANK_TOL = 1e-7 #HARDCODED
    #rank = _np.linalg.matrix_rank(_np.dot(dPall, dPall.T), RANK_TOL)
    
    if check_complete_fid_set:
        spectrum_full_fid_set= _np.abs(_np.linalg.eigvalsh(_np.dot(dPall, dPall.T)))
        
        #use the spectrum to calculate the rank instead.
        rank= _np.count_nonzero(spectrum_full_fid_set>RANK_TOL)
    
        if rank < gsGerm.num_params:  # full fiducial set should work!
            raise ValueError("Incomplete fiducial-pair set!")
    
        spectrum_full_fid_set= list(sorted(_np.abs(_np.linalg.eigvalsh(_np.dot(dPall, dPall.T)))))
        
        imin_full_fid_set = len(spectrum_full_fid_set) - gsGerm.num_params
        condition_full_fid_set = spectrum_full_fid_set[-1] / spectrum_full_fid_set[imin_full_fid_set] if (spectrum_full_fid_set[imin_full_fid_set] > 0) else _np.inf
        
        printer.log('J J^T Rank Full Fiducial Set: %d' % rank, 2)
        printer.log('Num Parameters (should equal above rank): %d' % (gsGerm.num_params), 2)
        printer.log('Full Fiducial Set Min Eigenvalue: %f' % (spectrum_full_fid_set[imin_full_fid_set]), 2)
        printer.log('Full Fiducial Set Max Eigenvalue: %f' % (spectrum_full_fid_set[-1]), 2)
        printer.log('Full Fiducial Set Condition Number: %f' % (condition_full_fid_set), 2)
    
    inv_trace_complete= _np.trace(_np.linalg.pinv(dPall.T@dPall))
    
    printer.log('Full Fiducial Set Inverse Trace: %f' % (inv_trace_complete), 2)
    
    #It can happen that there are fiducial pairs which are entirely insensitive to any
    #of the kite parameters, which show up as sets of empty rows in the jacobian.
    #We should identify which fiducial pairs those are and remove them from the 
    #search space.
    printer.log('Removing useless fiducial pairs with trivial sensitivity to the germ kite parameters. Sensitivity threshold is %.3f'%(sensitivity_threshold), 3)
    printer.log('Number of fiducial pairs before filtering trivial ones: %d' %(len(allPairIndices)), 3)
    cleaned_pair_indices_list= filter_useless_fid_pairs(allPairIndices, elIndicesForPair, dPall, sensitivity_threshold)
    printer.log('Number of fiducial pairs after filtering trivial ones: %d' %(len(cleaned_pair_indices_list)), 3)
    #Change the value of nPossiblePairs to reflect the filtered list
    nPossiblePairs= len(cleaned_pair_indices_list)
    
    #Need to choose how to initially seed a set of candidate fiducials.
    #I can think of two ways atm, one is to seed it with a random set of
    #fiducial pairs of size min_pairs_needed, the other is to run
    #the greedy search all the way from zero.
    
    #If random, pick a random set of fiducial pairs to start the search with:
    if initial_seed_mode=='random':
        #set the seed for the prng:
        _random.seed(seed)
        initial_fiducial_set= list(_random_combination(cleaned_pair_indices_list, min_pairs_needed))
        initial_pair_count= min_pairs_needed
    #if starting from zero then we'll change min_pairs_needed to zero and set the initial_fiducial_set
    #to None (may not be needed, will remove the placeholder value later if that is the case). May also
    #need some special logic to handle the very first step of the search.
    elif initial_seed_mode=='greedy':
        initial_fiducial_set=None
        initial_pair_count=0
    
    printer.log('Initial fiducial set: ' + str(initial_fiducial_set), 3)
    
    current_best_fiducial_list= initial_fiducial_set
    
    printer.log('Building Compact EVD Cache For Fiducial Updates', 3)
    #build a compact evd cache for the pairs we are iterating over too.
    fiducial_update_EVD_cache= construct_compact_evd_cache(cleaned_pair_indices_list, dPall,
                                                            elIndicesForPair, 
                                                            eigenvalue_tolerance=evd_tol)
    
    for nNeededPairs in range(initial_pair_count, nPossiblePairs):
        if nNeededPairs < min_pairs_needed:
            printer.log("Searching for the best initial set of %d pairs. Need %d pairs for the initial set." %
                    (nNeededPairs+1, min_pairs_needed), 2)
        else:
            printer.log("Beginning search for a good set of %d pairs" %
                        (nNeededPairs), 2)
            printer.log("    Inverse trace must be <= %g times more than the value for the full fiducial set" %
                        (inv_trace_tol),2)
            
        #The pairs we want to iterate over in the upcoming greedy search are those
        #in the set difference of allPairIndices-current_best_fiducial_list
        if current_best_fiducial_list is None:
            pairIndicesToIterateOver = cleaned_pair_indices_list
            current_best_jacobian= None
        else:
            pairIndicesToIterateOver = list(set(cleaned_pair_indices_list)-set(current_best_fiducial_list))
            current_best_jacobian= _np.take(dPall, _np.concatenate([elIndicesForPair[i] for i in current_best_fiducial_list]), axis=0)
            #take the gramian of this jacobian.
            current_best_jacobian= current_best_jacobian.T@current_best_jacobian
            current_best_inv_trace = _np.trace(_np.linalg.pinv(current_best_jacobian))
         
        #Construct the update cache for the current_best_fiducial_list's jacobian.
        if current_best_jacobian is not None:
            update_cache= construct_update_cache(current_best_jacobian, evd_tol=evd_tol)
            current_best_rank= len(update_cache[0])

            #make a CompositeScore object for the current best fiducial set.
            current_best_score= _scoring.CompositeScore(-current_best_rank, current_best_inv_trace, current_best_rank)
            
        else:
            update_cache= None  
            current_best_score= None

        printer.log('Initial Score: ' + str(current_best_score), 3)
        
        for pairIndexToTest in pairIndicesToIterateOver:    
            printer.log('Tested Index: %d' %(pairIndexToTest), 4)
            
            # Same computation of rank as above, but with only a
            # subset of the total fiducial pairs.
            elementIndicesToTest = elIndicesForPair[pairIndexToTest]
            update_to_test = _np.take(dPall, elementIndicesToTest, axis=0)  # subset_of_num_elements x num_params
            
            #We need some logic to handle the case where we are starting from scratch
            #and don't have an initial current_best_jacobian.
            if (initial_pair_count==0) and (nNeededPairs==0) and (current_best_jacobian is None):
                updated_pinv, updated_rank= _sla.pinv(update_to_test.T@update_to_test, return_rank=True)
                updated_inv_trace= _np.trace(updated_pinv)
                current_best_score =  _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                idx_current_best_update = pairIndexToTest
            elif (initial_pair_count==0) and (nNeededPairs==0) and (current_best_jacobian is not None):
                updated_pinv, updated_rank= _sla.pinv(update_to_test.T@update_to_test, return_rank=True)
                updated_inv_trace= _np.trace(updated_pinv)
                updated_score =  _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                if updated_score < current_best_score:
                    current_best_score= updated_score
                    idx_current_best_update = pairIndexToTest
            #otherwise we already have an initial Jacobian and should use the standard update logic.
            else:
                updated_inv_trace, updated_rank, _ = minamide_style_inverse_trace(fiducial_update_EVD_cache[pairIndexToTest], 
                                                                 update_cache[0], update_cache[1], 
                                                                 update_cache[2], False)
                #Construct a composite score object where the major score is
                #the rank of the updated jacobian and the minor score is
                #the inverse trace.
                updated_score= _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                
                printer.log('Updated Score: '+ str(updated_score), 3)
                
                if updated_score < current_best_score:
                    current_best_score=updated_score
                    idx_current_best_update= pairIndexToTest
        
        #Add the best index found to the list for current_best_fiducial_list (or initialize it if this list isn't set yet).
        if current_best_fiducial_list is None:
            current_best_fiducial_list= [idx_current_best_update]
        else:
            current_best_fiducial_list.append(idx_current_best_update)
            
        printer.log('Index of best update fiducial : %d'%(idx_current_best_update), 3)
        
        #check whether we have found an acceptable reduced set of fiducials.
        if (nNeededPairs>=min_pairs_needed) and (current_best_score.minor < inv_trace_tol*inv_trace_complete)\
            and (current_best_score.major <= -gsGerm.num_params):
            break

    #Get list of pairs as tuples for printing & returning
    goodPairList = []
    for i in current_best_fiducial_list:
        prepfid_index = i // nEStrs; iEStr = i - prepfid_index * nEStrs
        goodPairList.append((prepfid_index, iEStr))        
                    
    if goodPairList is None:
        printer.log('Failed to find a sufficient fiducial set.',1)
    printer.log('Exiting _get_per_germ_power_fidpairs_greedy', 4)
    return goodPairList, current_best_score

    
#helper function for building a compact evd cache:
def construct_compact_evd_cache(fiducial_indices, complete_jacobian, element_map, eigenvalue_tolerance=1e-10):
    sqrteU_dict = {}
    
    for fid_index in fiducial_indices:
        fid_element_indices = element_map[fid_index]
        fid_jacobian_components = _np.take(complete_jacobian, fid_element_indices, axis=0)
        e, U = compact_EVD(fid_jacobian_components.T@fid_jacobian_components, eigenvalue_tolerance)
        sqrteU_dict[fid_index]= U@_np.diag(_np.sqrt(e))  
    return sqrteU_dict    
    
    
#helper function for removing fiducial pairs which are entirely insensitive to the kite
#parameters from the search space:
def filter_useless_fid_pairs(fiducial_indices, element_map, complete_jacobian, sensitivity_threshold= 1e-10):
    
    #loop through the list of fiducial pair indices and extract the components of the jacobian for
    #that fiducial pair. The take the norm of that matrix and check if it is sufficiently larger 
    #than the sensitivity threshold to determine if it is zero (numpy used Frobenius by default).
    useful_fiducial_indices=[]
    for fid_index in fiducial_indices:
        fid_element_indices = element_map[fid_index]
        fid_jacobian_components = _np.take(complete_jacobian, fid_element_indices, axis=0)
        if _np.linalg.norm(fid_jacobian_components)>sensitivity_threshold:
            useful_fiducial_indices.append(fid_index)
        else:
            continue
    #return the list of fiducial pair indices with non-trivial frobenius norm.
    return useful_fiducial_indices
    
    
#-------------- Globally Aware Per-Germ FPR------------------------#
    
#Define a function that performs a modified version of per-germ FPR leveraging global informationally
#about the amplificational properties of the germ set as a whole.

def find_sufficient_fiducial_pairs_per_germ_global(target_model, prep_fiducials, meas_fiducials,
                                                   germ_vector_spanning_set=None, germs=None, prep_povm_tuples="first",
                                                   mem_limit=None, inv_trace_tol= 10, initial_seed_mode='greedy',
                                                   evd_tol=1e-10, seed=None ,verbosity=0, float_type = _np.cdouble,
                                                   germ_set_spanning_kwargs = None, precomputed_jacobians = None):
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
    sequence probabilities with respect to the i-th amplified parameter.

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

    germ_vector_spanning_set : dictionary of lists
        A dictionary with keys corresponding to germs and values
        corresponding to lists of parameter vectors (less spam parameters)
        for which we need sensitivity.
    
    germs : list of circuits
        If passed in and germ_vector_spanning_set is None then we'll use this
        in the calculation of the germ vector spanning set.

    prep_povm_tuples : list or "first", optional
        A list of `(prepLabel, povmLabel)` tuples to consider when
        checking for completeness.  Usually this should be left as the special
        (and default) value "first", which considers the first prep and POVM
        contained in `target_model`.

    verbosity : int, optional
        How much detail to print to stdout.

    mem_limit : int, optional
        A memory limit in bytes.

    inv_trace_tol : float, optional (default 10)
        A threshold tolerance value corresponding to the maximum reduction
        in sensitivity we are willing to tolerate relative to the complete
        set of fiducial pairs. Specifically the increase in the trace of the 
        inverse/psuedoinverse of the directional derivative matrix for a germ's 
        specified amplified parameters that we are willing to tolerate. 

    initial_seed_mode : str, optional (default 'greedy')
        Specifies the manner in which an initial set of fiducial pairs is
        selected for a germ to seed the greedy search routine. Currently only
        supports 'greedy' wherein we start from scratch from an empty set
        of fiducial pairs and select them entirely greedily.
    
    evd_tol : float, optional (default 1e-10)
        Threshold value for eigenvalues below which they are treated as zero.
        Used in the construction of compact eigenvalue decompositions
        (technically rank-decompositions) for directional derivative matrices.

    seed : int, optional (default None)
        Seed for PRNGs. Not currently used.

    float_type : numpy dtype, optional (default numpy.cdouble)
        Numpy data type to use for floating point arrays.

    germ_set_spanning_kwargs : dict, optional (default None)
        A dictionary of optional kwargs for the function
        `pygsti.algorithms.germselection.germ_set_spanning_vectors`.
        See doctring for that function for more details. Only utilized
        if the germ_vector_spanning_set argument of this function is None.
    
    precomputed_jacobians : dict, optional (default None)
        An optional dictionary of precomputed jacobian dictionaries for the
        germ fiducial pair set. The keys are germs and the values are the dictionaries
        corresponding that that germ-fiducial-pair set's jacobian.

    Returns
    -------
    dict
        A dictionary whose keys are the germ circuits and whose values are
        lists of (iRhoFid,iMeasFid) tuples of integers, each specifying the
        list of fiducial pairs for a particular germ (indices are into
        `prep_fiducials` and `meas_fiducials`).
    """

    if not isinstance(target_model.sim, _MatrixForwardSimulator):
        target_model = target_model.copy()
        target_model.sim = 'matrix'

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    #if no germ_vector_spanning_set is passed in compute it here.
    if germ_vector_spanning_set is None:
        if germs is None:
            raise ValueError('Must specify either germ_vector_spanning_set or germs.')
        else:
            default_germ_set_kwargs = {'assume_real':False, 
                                       'num_nongauge_params': None,
                                       'tol':1e-6,
                                       'pretest': False,
                                       'threshold':1e6}
            used_kwargs = default_germ_set_kwargs.copy()
            if germ_set_spanning_kwargs is not None:
                used_kwargs.update(germ_set_spanning_kwargs)
                                       
            germ_vector_spanning_set, _ = germ_set_spanning_vectors(target_model, germs, 
                                                                 float_type=float_type, 
                                                                 evd_tol = evd_tol,
                                                                 verbosity=verbosity,
                                                                 **used_kwargs)

    prep_povm_tuples, dof_per_povm = _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs=True)

    pairListDict = {}  # dict of lists of 2-tuples: one pair list per germ
    
    #if precomputed_jacobians is None then make sure we pass in None for each germ
    #hack this in without branching by constructing a dictionary of Nones.
    if precomputed_jacobians is None:
        precomputed_jacobians = {germ:None for germ in germ_vector_spanning_set.keys()}
    
    printer.log("------  Per Germ Global Fiducial Pair Reduction --------")
    with printer.progress_logging(1):
        for i, (germ, germ_vector_list) in enumerate(germ_vector_spanning_set.items()):
            candidate_solution_list, best_score = get_per_germ_fid_pairs_global(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                                                    target_model, germ, germ_vector_list, mem_limit,
                                                                    printer, dof_per_povm, inv_trace_tol, 
                                                                    initial_seed_mode=initial_seed_mode, evd_tol=evd_tol,
                                                                    float_type= float_type, dprobs_dict = precomputed_jacobians[germ])
            
            #print some output about the minimum eigenvalue acheived.
            if best_score is not None:
                printer.log('Score Achieved: ' + str(best_score), 2)
            printer.log('Number of fid pairs in found solution: %d'%(len(candidate_solution_list)), 2)
            
            try:
                assert(candidate_solution_list is not None)
            except AssertionError as err:
                print('Failed to find an acceptable fiducial set for germ power pair: ', germ)
                print(err)

            pairListDict[germ] = candidate_solution_list  # add to final list of per-germ pairs

    return pairListDict
    
def get_per_germ_fid_pairs_global(prep_fiducials, meas_fiducials, prep_povm_tuples,
                                 target_model, germ, germ_vector_list, mem_limit, printer, dof_per_povm, 
                                 inv_trace_tol=10, initial_seed_mode= 'greedy', evd_tol=1e-10, 
                                 float_type = _np.cdouble, dprobs_dict = None):                      
    #Get dP-matrix for full set of fiducials, where
    # P_ij = <E_i|germ^exp|rho_j>, i = composite EVec & fiducial index,
    #   j is similar, and derivs are wrt the "eigenvalues" of the germ
    #  (i.e. the parameters of target_model).
    
    #Let's assume here that the spam parameters have been set to static,
    #so the only parameters are the gate parameters. This matches what is
    #produced by the spanning germ set vector code.
    
    #Before doing anything check whether the germ_vector_list is empty, in which case
    #we can skip it and return an empty list. (empty=False)
    if not germ_vector_list:
        return [], None
    
    #but, let's just do this again anyhow to be safe.
    reduced_model = _make_spam_static(target_model)
    
    printer.log('Entered helper function _get_per_germ_fidpairs_global', 3)

    nEStrs = len(meas_fiducials)
    nPossiblePairs = len(prep_fiducials) * len(meas_fiducials)
    
    allPairIndices = list(range(nPossiblePairs))

    printer.log(f'Current Germ: {germ}', 2)
    
    printer.log('Number of possible pairs: %d'%(nPossiblePairs), 3)
    
    num_germ_vecs= len(germ_vector_list)
    
    printer.log('Number of amplified parameter directions needed for this germ: %d'%(num_germ_vecs), 2)

    #Determine which fiducial-pair indices to iterate over
    goodPairList = None
    
    #loops over a number of pairs between min_pairs_needed and up to and not including the number of possible pairs
    min_pairs_needed= ceil((num_germ_vecs/(nPossiblePairs*dof_per_povm))*nPossiblePairs)
    printer.log('Minimum Number of Pairs Needed for this Germ: %d'%(min_pairs_needed), 2)

    lst = _gsc.create_circuits(
        "pp[0]+f0+germ*power+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
        germ=germ, pp=prep_povm_tuples, power=1,
        order=('f0', 'f1', 'pp'))
    
    printer.log('Constructing Directional Derivatives for Full Fiducial Set' , 2)
    
    #concatenate the list of column vectors into a single matrix. 
    germ_vector_mat = _np.concatenate(germ_vector_list, axis=1)
    compact_directional_DDD_list = _compute_bulk_directional_ddd_compact(reduced_model, lst, germ_vector_mat, eps=evd_tol,
                             comm=None, evd_tol=evd_tol,  float_type=float_type,
                             printer=printer, return_eigs=False, dprobs_dict = dprobs_dict)
    
    #It can happen that there are fiducial pairs which are entirely insensitive to any
    #of the selected amplified parameters, which show up as empty matrices in the list of directional derivatives.
    #We should identify which fiducial pairs those are and remove them from the 
    #search space.
    
    #TODO: Add back this functionality for the parameter vector based FPR.
    printer.log('Removing useless fiducial pairs with trivial sensitivity to the vector of parameters.', 3)
    printer.log('Number of fiducial pairs before filtering trivial ones: %d' %(len(allPairIndices)), 3)
    
    #Directional derivative matrices with minimum eigenvalues below the evd_tol end up getting
    #returned as empty matrices. Filter on that to strip those out.
    cleaned_pair_indices_list= [idx for idx in allPairIndices if compact_directional_DDD_list[idx].any()]
    
    printer.log('Number of fiducial pairs after filtering trivial ones: %d' %(len(cleaned_pair_indices_list)), 3)
    #Change the value of nPossiblePairs to reflect the filtered list
    nPossiblePairs= len(cleaned_pair_indices_list)
    
    printer.log('Calculating Inverse Trace of Full Fiducial Set', 2)
    
    directional_DDD_complete = _np.sum(_np.stack([compact_directional_DDD_list[i]@compact_directional_DDD_list[i].T 
                                                  for i in cleaned_pair_indices_list], axis=2), axis=2)
    
    inv_trace_complete= _np.trace(_np.linalg.pinv(directional_DDD_complete))
    
    printer.log('Full Fiducial Set Inverse Trace: %f' % (inv_trace_complete), 2)
    
    #Need to choose how to initially seed a set of candidate fiducials.
    #I can think of two ways atm, one is to seed it with a random set of
    #fiducial pairs of size min_pairs_needed, the other is to run
    #the greedy search all the way from zero.
    
    #If random, pick a random set of fiducial pairs to start the search with:
    #if initial_seed_mode=='random':
        #set the seed for the prng:
    #    _random.seed(seed)
    #    initial_fiducial_set= list(_random_combination(cleaned_pair_indices_list, min_pairs_needed))
    #    initial_pair_count= min_pairs_needed
    #if starting from zero then we'll change min_pairs_needed to zero and set the initial_fiducial_set
    #to None (may not be needed, will remove the placeholder value later if that is the case). May also
    #need some special logic to handle the very first step of the search.
    if initial_seed_mode=='greedy':
        initial_fiducial_set=None
        initial_pair_count=0
    
    printer.log('Initial fiducial set: ' + str(initial_fiducial_set), 3)
    
    current_best_fiducial_list= initial_fiducial_set
    
    for nNeededPairs in range(initial_pair_count, nPossiblePairs):
        if nNeededPairs < min_pairs_needed:
            printer.log("Searching for the best initial set of %d pairs. Need %d pairs for the initial set." %
                    (nNeededPairs+1, min_pairs_needed), 3)
        else:
            printer.log("Beginning search for a good set of %d pairs" %
                        (nNeededPairs), 3)
            printer.log("    Inverse trace must be <= %g times more than the value for the full fiducial set" %
                        (inv_trace_tol),3)
            
        #The pairs we want to iterate over in the upcoming greedy search are those
        #in the set difference of allPairIndices-current_best_fiducial_list
        if current_best_fiducial_list is None:
            pairIndicesToIterateOver = cleaned_pair_indices_list
            current_best_jacobian= None
        else:
            pairIndicesToIterateOver = list(set(cleaned_pair_indices_list)-set(current_best_fiducial_list))
            current_best_jacobian= _np.sum(_np.stack([compact_directional_DDD_list[i]@compact_directional_DDD_list[i].T for i in current_best_fiducial_list],axis=2), axis=2)
            current_best_inv_trace = _np.trace(_np.linalg.pinv(current_best_jacobian))
         
        #Construct the update cache for the current_best_fiducial_list's jacobian.
        if current_best_jacobian is not None:
            update_cache= construct_update_cache(current_best_jacobian, evd_tol=evd_tol)
            current_best_rank= len(update_cache[0])

            #make a CompositeScore object for the current best fiducial set.
            current_best_score= _scoring.CompositeScore(-current_best_rank, current_best_inv_trace, current_best_rank)
            
        else:
            update_cache= None  
            current_best_score= None

        printer.log('Initial Score: ' + str(current_best_score), 3)
        
        #Check if the initial set is good. This is needed for the edge case where we are at the first iteration
        #after constructing a set of size min_pairs_needed to avoid having to wait till the end of the first
        #iteration of this loop.
        if (nNeededPairs>=min_pairs_needed) and (current_best_score.minor < inv_trace_tol*inv_trace_complete)\
            and (current_best_score.major <= -num_germ_vecs):
            break
        
        for pairIndexToTest in pairIndicesToIterateOver:    
            printer.log('Tested Index: %d' %(pairIndexToTest), 4)
            
            #pull out the corresponding update from the directional derivative list.
            update_to_test = compact_directional_DDD_list[pairIndexToTest]

            #We need some logic to handle the case where we are starting from scratch
            #and don't have an initial current_best_jacobian.
            if (initial_pair_count==0) and (nNeededPairs==0) and (current_best_jacobian is None):
                updated_pinv, updated_rank= _sla.pinv(update_to_test@update_to_test.T, return_rank=True)
                updated_inv_trace= _np.trace(updated_pinv)
                current_best_score =  _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                idx_current_best_update = pairIndexToTest
            elif (initial_pair_count==0) and (nNeededPairs==0) and (current_best_jacobian is not None):
                updated_pinv, updated_rank= _sla.pinv(update_to_test@update_to_test.T, return_rank=True)
                updated_inv_trace= _np.trace(updated_pinv)
                updated_score =  _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                if updated_score < current_best_score:
                    current_best_score= updated_score
                    idx_current_best_update = pairIndexToTest
            #otherwise we already have an initial Jacobian and should use the standard update logic.
            else:
                updated_inv_trace, updated_rank, _ = minamide_style_inverse_trace(update_to_test, 
                                                                 update_cache[0], update_cache[1], 
                                                                 update_cache[2], False)
                #Construct a composite score object where the major score is
                #the rank of the updated jacobian and the minor score is
                #the inverse trace.
                updated_score= _scoring.CompositeScore(-updated_rank, updated_inv_trace, updated_rank)
                
                printer.log('Updated Score: '+ str(updated_score), 4)
                
                if updated_score < current_best_score:
                    current_best_score=updated_score
                    idx_current_best_update= pairIndexToTest
        
        #Add the best index found to the list for current_best_fiducial_list (or initialize it if this list isn't set yet).
        if current_best_fiducial_list is None:
            current_best_fiducial_list= [idx_current_best_update]
        else:
            current_best_fiducial_list.append(idx_current_best_update)
            
        printer.log('Index of best update fiducial : %d'%(idx_current_best_update), 3)
        
        #check whether we have found an acceptable reduced set of fiducials.
        if (nNeededPairs>=min_pairs_needed) and (current_best_score.minor < inv_trace_tol*inv_trace_complete)\
            and (current_best_score.major <= -num_germ_vecs):
            break

    #Get list of pairs as tuples for printing & returning
    goodPairList = []
    for i in current_best_fiducial_list:
        prepfid_index = i // nEStrs; iEStr = i - prepfid_index * nEStrs
        goodPairList.append((prepfid_index, iEStr))        
                    
    if goodPairList is None:
        printer.log('Failed to find a sufficient fiducial set.',1)
    printer.log('Exiting _get_per_germ_fidpairs_global', 4)
    return goodPairList, current_best_score
    
#new function that computes the J^T J matrices but then returns the result in the form of the 
#compact EVD in order to save on memory.    
def _compute_bulk_directional_ddd_compact(model, circuits, vec_mat, eps,
                             comm=None, evd_tol=1e-10,  float_type=_np.cdouble,
                             printer=None, return_eigs=False, dprobs_dict= None):

    """
    Calculate the positive squares of the fiducial pair Jacobians.

    DerivDaggerDeriv == array (JV).H@(JV) contributions from each fiducial pair.
    V is a matrix with column vectors corresponding to vectors in model
    parameter space that the germ needs to provide sensitivity to, and thus
    the fiducial pairs need sensistivity to these as well.
    
    JV is a matrix of shape (num_probs, num_directions) that corresponds
    to the directional derivative matrix for the probalities with respect
    to the subset of amplified parameters selected for a germ.

    Parameters
    ----------
    model : Model
        The model defining the parameters to differentiate with respect to.
        
    circuits : list
        A list of the circuit objects corresponding to the prep, germ, measurement
        triplets under consideration.

    vec_mat : ndarray
        A matrix of shape (num_params, num_directions) where the column vectors
        are directions in 

    eps : float, optional
        Tolerance used for testing whether two eigenvectors are degenerate
        (i.e. abs(eval1 - eval2) < eps ? )
        
    evd_tol : float, optional
        Tolerance used for determining if a singular value has zero magnitude when constructing the
        compact SVD.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.
        
    float_type : numpy dtype object, optional
        Numpy data type to use in floating point arrays.
        
    return_eigs : bool, optional, default False
       If True then additionally return a list of the arrays of eigenvalues for each
       fiducial pair's derivative gramian.
       
    dprobs_dict : dictionary, optional (default None).
        Optionally can pass in a precomputed dictionary of jacobians (in the format output
        by bulk_dprobs) in which case they are not re-computed here. This can be useful
        if you want to use MPI for calculating the probabilities while using the native
        multi-threading in BLAS apis for other calculations. 

    Returns
    -------
    sqrteU_list : list of numpy ndarrays
        list of the non-trivial left eigenvectors for the compact EVD for each germ
        where each left eigenvector is multiplied by the sqrt of the corresponding eigenvalue.
    e_list : ndarray
        list of non-zero eigenvalue arrays for each germ.
    """
       
    #TODO: Figure out how to pipe in a comm object to parallelize some of this with MPI.
       
    #model is assumed to have the spam parameters set to static, but let's enforce that
    #again here for safety.
    static_spam_model = _make_spam_static(model)
        
    sqrteU_list=[]
    e_list=[]
    
    if printer is not None:
        printer.log('Generating compact EVD Cache',1)
        
        #calculate the jacobians in bulk.
        if dprobs_dict is None:
            printer.log('Calculating Jacobians of Probabilities.', 2)
            dprobs_dict = static_spam_model.sim.bulk_dprobs(circuits)
        
        printer.log('Constructing compact EVDs.', 2)
        with printer.progress_logging(3):

            for i, ckt in enumerate(circuits):                
                printer.show_progress(iteration=i, total=len(circuits), bar_length=25)
                
                #repackage the outcome label indexed dictionary into an ndarray
                #need to reshape the vectors along the way.
                jac = _np.concatenate([_np.reshape(vec, (1,len(vec))) for vec in dprobs_dict[ckt].values()], axis= 0)
                
                #now calculate the direction derivative matrix.
                directional_deriv = jac@vec_mat
                direc_deriv_gram = directional_deriv.T@directional_deriv                                        
                #now take twirledDerivDerivDagger and construct its compact EVD.
                e, U= compact_EVD(direc_deriv_gram, evd_tol)
                e_list.append(e)
                
                #by doing this I am assuming that the matrix is PSD, but since these are all
                #gramians that should be alright.
                
                #I want to use a rank-decomposition, so split the eigenvalues into a pair of diagonal
                #matrices with the square roots of the eigenvalues on the diagonal and fold those into
                #the matrix of eigenvectors by left multiplying.
                
                sqrteU_list.append( U@_np.diag(_np.sqrt(e)) )       
    else: 
        #calculate the jacobians in bulk.
        if dprobs_dict is None:
            dprobs_dict = static_spam_model.sim.bulk_dprobs(circuits)

        for i, ckt in enumerate(circuits):                

            #repackage the outcome label indexed dictionary into an ndarray
            #need to reshape the vectors along the way.
            jac = _np.concatenate([_np.reshape(vec, (1,len(vec))) for vec in dprobs_dict[ckt].values()], axis= 0)

            #now calculate the direction derivative matrix.
            directional_deriv = jac@vec_mat
            direc_deriv_gram = directional_deriv.T@directional_deriv
    
            #now take twirledDerivDerivDagger and construct its compact EVD.
            e, U= compact_EVD(direc_deriv_gram, evd_tol)
            e_list.append(e)

            sqrteU_list.append( U@_np.diag(_np.sqrt(e)) )         
           
    if return_eigs:
        return sqrteU_list, e_list
    else:
        return sqrteU_list
        
def _make_spam_static(model):
    #return a modified copy of the model (not in-place) where the spam elements are static.
    #I could do this more elegantly, but the easiest thing is just to make a copy, set all
    #of the parameterizations to static, then swap out the elements of original model.
    #implicitly assuming we have an explicitopmodel here.
    
    model_copy = model.copy()
    static_model_copy = model.copy()
    static_model_copy.set_all_parameterizations('static')
    
    #swap out the elements of the model we need to.
    model_copy.preps = static_model_copy.preps
    model_copy.povms = static_model_copy.povms
    
    #I messed with the parameterizations, so i'll need to rebuild the model vector.
    model_copy._rebuild_paramvec()
    
    return model_copy
    
#write a helper function for precomputing the jacobian dictionaries from bulk_dprobs
#which can then be passed into the construction of the compactEVD caches.

def compute_jacobian_dicts(model, germs, prep_fiducials, meas_fiducials, prep_povm_tuples = 'first', comm=None, 
                           mem_limit=None, verbosity = 0):
    """
    Function for precomputing the jacobian dictionaries from bulk_dprobs
    for a model with its SPAM parameters frozen, as needed for certain
    FPR applications. Can take a comm object for parallelization of the
    jacobian calculation using MPI.
    
    model : Model
        The model used for calculating the jacobian with respect to.
        
    germs :list of Circuits
        Germs to construct the jacobian dictionaries for.
    
    prep_fiducials : list of Circuits
        A list of preparation fiducials circuits.
    
    meas_fiducials : list of Circuits
        A list of measurement fiducial circuits.
        
    prep_povm_tuples : str or list of tuples (default 'first')
        Either a string or list of tuples. When a list of tuples
        these correspond to native state prep and native POVM pairs.
        When the special keyword argument 'first' is passed in the first
        native prep and first native POVM are used.
    
    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors is used for the jacobian calculation.
        
    mem_limit : int, optional (default None)
        An optional per-worker memory limit, in bytes.
        
    Returns
    --------
    jacobian_dicts : dictionary of dictionaries
        A dictionary whose keys correspond to germs and whose values correspond
        the jacobian dictionaries for the list of circuits induced by that germ
        together with the set of fiducials pairs. Format of these correspond
        to the format returned by bulk_dprobs.
    
    """
    resource_alloc = _baseobjs.ResourceAllocation(comm= comm, mem_limit = mem_limit)
    
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm= comm)

    #construct the list of circuits
    prep_povm_tuples = _set_up_prep_POVM_tuples(model, prep_povm_tuples, return_meas_dofs=False)
        
    #freeze the SPAM model parameters:
    static_spam_model = _make_spam_static(model)
        
    jacobian_dicts = {}
        
    for germ in germs:       
        printer.log(f'Current germ: {germ}', 1)
         
        lst = _gsc.create_circuits(
            "pp[0]+f0+germ*power+f1+pp[1]", f0=prep_fiducials, f1=meas_fiducials,
            germ=germ, pp=prep_povm_tuples, power=1,
            order=('f0', 'f1', 'pp'))
        
        #calculate the dprobs dictionary in bulk.
        dprobs_dict = static_spam_model.sim.bulk_dprobs(lst, resource_alloc)        
        jacobian_dicts[germ] = dprobs_dict
        
    return jacobian_dicts

#helper function for configuring the list of circuit tuples needed for prep-POVM pairs used in FPR.
def _set_up_prep_POVM_tuples(target_model, prep_povm_tuples, return_meas_dofs= False):

    if prep_povm_tuples == "first":
        firstRho = list(target_model.preps.keys())[0]
        prep_ssl = [target_model.preps[firstRho].state_space.state_space_labels]
        firstPOVM = list(target_model.povms.keys())[0]
        POVM_ssl = [target_model.povms[firstPOVM].state_space.state_space_labels]
        prep_povm_tuples = [(firstRho, firstPOVM)]
        #I think using the state space labels for firstRho and firstPOVM as the
        #circuit labels should work most of the time (new stricter line_label enforcement means
        # we need to enforce compatibility here), but this might break for
        #ImplicitModels? Not sure how those handle the state space labels for preps and povms
        #Time will tell...
    #if not we still need to extract state space labels for all of these to meet new circuit
    #label handling requirements.
    else:
        prep_ssl = [target_model.preps[lbl_tup[0]].state_space.state_space_labels for lbl_tup in prep_povm_tuples]
        POVM_ssl = [target_model.povms[lbl_tup[1]].state_space.state_space_labels for lbl_tup in prep_povm_tuples]

    #brief intercession to calculate the number of degrees of freedom for the povm.
    num_effects= len(list(target_model.povms[prep_povm_tuples[0][1]].keys()))
    dof_per_povm= num_effects-1

    prep_povm_tuples = [(_circuits.Circuit([prepLbl], line_labels=prep_ssl[i]), 
                        _circuits.Circuit([povmLbl], line_labels=POVM_ssl[i]))
                       for i, (prepLbl, povmLbl) in enumerate(prep_povm_tuples)]
    
    if return_meas_dofs:
        return prep_povm_tuples, dof_per_povm
    else:
        return prep_povm_tuples