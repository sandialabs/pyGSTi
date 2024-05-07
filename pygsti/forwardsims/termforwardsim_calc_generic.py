"""Defines generic Python-version of map forward simuator calculations"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools
import itertools as _itertools
import math as _math

import numpy as _np

from pygsti.tools import listtools as _lt

SMALL = 1e-5
LOGSMALL = -5
# a number which is used in place of zero within the
# product of term magnitudes to keep a running path
# magnitude from being zero (and losing memory of terms).


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def prs_as_polynomials(fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
                       comm=None, mem_limit=None, fastmode=True):
    """
    Computes polynomials of the probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    polynomial_vindices_per_int : int
        The number of variable indices that can fit into a single platform-width integer
        (can be computed from number of model params, but passed in for performance).

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    Returns
    -------
    list
        A list of PolynomialRep objects, one per element of `elabels`.
    """
    #print("PRS_AS_POLY circuit = ",circuit)
    #print("DB: prs_as_polys(",spamTuple,circuit,fwdsim.max_order,")")

    #NOTE for FUTURE: to adapt this to work with numerical rather than polynomial coeffs:
    # use get_direct_order_terms(order, order_base) w/order_base=0.1(?) instead of taylor_order_terms??
    # below: replace prps with: prs = _np.zeros(len(elabels),complex)  # an array in "bulk" mode
    #  use *= or * instead of .mult( and .scale(
    #  e.g. res = _np.prod([f.coeff for f in factors])
    #       res *= (pLeft * pRight)
    # - add assert(_np.linalg.norm(_np.imag(prs)) < 1e-6) at end and return _np.real(prs)

    mpv = fwdsim.model.num_params  # max_polynomial_vars

    # Construct dict of gate term reps
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {glbl:
                    [
                        [t.torep()
                         for t in fwdsim.model._circuit_layer_operator(glbl, 'op').taylor_order_terms(order, mpv)]
                        for order in range(fwdsim.max_order + 1)
                    ] for glbl in distinct_gateLabels}

    #Similar with rho_terms and E_terms, but lists
    rho_term_reps = [[t.torep()
                      for t in fwdsim.model._circuit_layer_operator(rholabel, 'prep').taylor_order_terms(order, mpv)]
                     for order in range(fwdsim.max_order + 1)]

    E_term_reps = []
    E_indices = []
    for order in range(fwdsim.max_order + 1):
        cur_term_reps = []  # the term reps for *all* the effect vectors
        cur_indices = []  # the Evec-index corresponding to each term rep
        for i, elbl in enumerate(elabels):
            term_reps = [t.torep()
                         for t in fwdsim.model._circuit_layer_operator(elbl, 'povm').taylor_order_terms(order, mpv)]
            cur_term_reps.extend(term_reps)
            cur_indices.extend([i] * len(term_reps))
        E_term_reps.append(cur_term_reps)
        E_indices.append(cur_indices)

    ##DEBUG!!!
    #print("DB NEW operation terms = ")
    #for glbl,order_terms in op_term_reps.items():
    #    print("GATE ",glbl)
    #    for i,termlist in enumerate(order_terms):
    #        print("ORDER %d" % i)
    #        for term in termlist:
    #            print("Coeff: ",str(term.coeff))

    global DEBUG_FCOUNT  # DEBUG!!!
    # db_part_cnt = 0
    # db_factor_cnt = 0
    #print("DB: pr_as_poly for ",str(tuple(map(str,circuit))), " max_order=",fwdsim.max_order)

    prps = [None] * len(elabels)  # an array in "bulk" mode? or Polynomial in "symbolic" mode?
    for order in range(fwdsim.max_order + 1):
        #print("DB: pr_as_poly order=",order)
        # db_npartitions = 0
        for p in _lt.partition_into(order, len(circuit) + 2):  # +2 for SPAM bookends
            #factor_lists = [ fwdsim.sos.operation(glbl).get_order_terms(pi) for glbl,pi in zip(circuit,p) ]
            factor_lists = [rho_term_reps[p[0]]] + \
                           [op_term_reps[glbl][pi] for glbl, pi in zip(circuit, p[1:-1])] + \
                           [E_term_reps[p[-1]]]
            factor_list_lens = list(map(len, factor_lists))
            Einds = E_indices[p[-1]]  # specifies which E-vec index each of E_term_reps[p[-1]] corresponds to

            if any([len(fl) == 0 for fl in factor_lists]): continue

            #print("DB partition = ",p, "listlens = ",[len(fl) for fl in factor_lists])
            if fastmode:  # filter factor_lists to matrix-compose all length-1 lists
                leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
                rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
                coeffSaved = [None] * (len(factor_lists) - 1)
                last_index = len(factor_lists) - 1

                for incd, fi in _lt.incd_product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(fi)]

                    if incd == 0:  # need to re-evaluate rho vector
                        rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                        for f in factors[0].pre_ops:
                            rhoVecL = f.acton(rhoVecL)
                        leftSaved[0] = rhoVecL

                        rhoVecR = factors[0].post_state
                        for f in factors[0].post_ops:
                            rhoVecR = f.acton(rhoVecR)
                        rightSaved[0] = rhoVecR

                        coeff = factors[0].coeff
                        coeffSaved[0] = coeff
                        incd += 1
                    else:
                        rhoVecL = leftSaved[incd - 1]
                        rhoVecR = rightSaved[incd - 1]
                        coeff = coeffSaved[incd - 1]

                    # propagate left and right states, saving as we go
                    for i in range(incd, last_index):
                        for f in factors[i].pre_ops:
                            rhoVecL = f.acton(rhoVecL)
                        leftSaved[i] = rhoVecL

                        for f in factors[i].post_ops:
                            rhoVecR = f.acton(rhoVecR)
                        rightSaved[i] = rhoVecR

                        coeff = coeff.mult(factors[i].coeff)
                        coeffSaved[i] = coeff

                    # for the last index, no need to save, and need to construct
                    # and apply effect vector

                    #HERE - add something like:
                    #  if factors[-1].opname == cur_effect_opname: (or opint in C-case)
                    #      <skip application of post_ops & preops - just load from (new) saved slot get pLeft & pRight>

                    for f in factors[-1].pre_ops:
                        rhoVecL = f.acton(rhoVecL)
                    E = factors[-1].post_effect  # effect representation
                    pLeft = E.amplitude(rhoVecL)

                    #Same for post_ops and rhoVecR
                    for f in factors[-1].post_ops:
                        rhoVecR = f.acton(rhoVecR)
                    E = factors[-1].pre_effect
                    pRight = _np.conjugate(E.amplitude(rhoVecR))

                    #print("DB PYTHON: final block: pLeft=",pLeft," pRight=",pRight)
                    res = coeff.mult(factors[-1].coeff)
                    res.scale((pLeft * pRight))
                    #print("DB PYTHON: result = ",res)
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx]  # final "factor" index == E-vector index
                    if prps[Ei] is None: prps[Ei] = res
                    else: prps[Ei].add_inplace(res)
                    #print("DB PYTHON: prps[%d] = " % Ei, prps[Ei])

            else:  # non-fast mode
                last_index = len(factor_lists) - 1
                #print("DB: factor lengths = ", factor_list_lens)
                for fi in _itertools.product(*[range(l) for l in factor_list_lens]):
                    factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(fi)]
                    res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
                    pLeft = _unitary_sim_pre(factors, comm, mem_limit)
                    pRight = _unitary_sim_post(factors, comm, mem_limit)
                    # if not self.unitary_evolution else 1.0
                    res.scale((pLeft * pRight))
                    final_factor_indx = fi[-1]
                    Ei = Einds[final_factor_indx]  # final "factor" index == E-vector index
                    # print("DB: pr_as_poly    ", fi, " coeffs=", [f.coeff for f in factors],
                    #       " pLeft=", pLeft, " pRight=", pRight, "res=", res)
                    if prps[Ei] is None: prps[Ei] = res
                    else: prps[Ei].add_inplace(res)

                    #if Ei == 0:
                    #    from pygsti.baseobjs.polynomial import Polynomial
                    #    print("DB pr_as_poly ",fi," running prps[",Ei,"] =",Polynomial.from_rep(prps[Ei]))

            # #DEBUG!!!
            # db_nfactors = [len(l) for l in factor_lists]
            # db_totfactors = _np.prod(db_nfactors)
            # db_factor_cnt += db_totfactors
            # DEBUG_FCOUNT += db_totfactors
            # db_part_cnt += 1
            # print("DB: pr_as_poly   partition=",p,
            #       "(cnt ",db_part_cnt," with ",db_nfactors," factors (cnt=",db_factor_cnt,")")

    #print("DONE -> FCOUNT=",DEBUG_FCOUNT)
    return prps  # can be a list of polys


def prs_directly(fwdsim, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True, wt_tol=0.0,
                 reset_term_weights=True, debug=None):
    #return _prs_directly(fwdsim, rholabel, elabels, circuit, comm, mem_limit, fastmode)
    raise NotImplementedError("No direct mode yet")


def refresh_magnitudes_in_repcache(repcache, paramvec):
    from pygsti.baseobjs.opcalc import bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex
    for repcel in repcache.values():
        for termrep in repcel[0]:  # first element of tuple contains list of term-reps
            v, c = termrep.coeff.compact_complex()
            coeff_array = _bulk_eval_compact_polynomials_complex(v, c, paramvec, (1,))
            termrep.set_magnitude_only(abs(coeff_array[0]))


def circuit_achieved_and_max_sopm(fwdsim, rholabel, elabels, circuit, repcache, threshold, min_term_mag):
    """
    Compute the achieved and maximum sum-of-path-magnitudes (SOPM) for a given circuit and model.

    This is a helper function for a TermForwardSimulator, and not typically called independently.

    A path-integral forward simulator specifies a model and path criteria (e.g. the max Taylor order).  The
    model's operations can construct Taylor terms with coefficients based on the current model parameters.
    The magnitudes of these coefficients are used to estimate the error incurred by a given path truncation
    as follows:  term coefficient magnitudes are multiplied together to get path magnitudes, and these are added
    to get an "achieved" sum-of-path-magnitudes.  This can be compared with a second quantity, the "maximum" sum-
    of-path-magnitudes based on estimates (ideally upper bounds) of the magnitudes for *all* paths.

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The forward simulator.  Contains the model that is used.

    rholabel : Label
        The preparation label, which precedes the layers in `circuit`.  Note that `circuit` should not contain
        any preparation or POVM labels - only the operation labels.

    elabels : list or tuple
        A list of POVM effect labels, which follow the layers in `circuit`.  Note that `circuit` should not contain
        any preparation or POVM labels - only the operation labels.

    circuit : Circuit
        The non-SPAM operations that make up the circuit that values are computed for.

    repcache : dict
        A dictionary of already-build preparation, operation, and POVM effect representations.  Keys are
        labels and values are the representation objects.  Use of a representation cache can significantly
        speed up multiple calls to this function.

    threshold : float
        A threshold giving the minimum path magnitude that should be included in the "achieved" sum of
        path magnitudes.  As this number gets smaller, more paths are included.

    min_term_mag : float
        The minimum magnitude a single term can have and still be considered in paths.  This essentially
        specifies a pre-path-magnitude threshold that lessens computational overhead by ignoring terms
        that have a very small magnitude.

    Returns
    -------
    achieved_sopm : float
        The achieved sum-of-path-magnitudes.
    max_sopm : float
        The approximate maximum sum-of-path-magnitudes.
    """
    mpv = fwdsim.model.num_params  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model._circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)

    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]

    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model._circuit_layer_operator(rholabel, 'prep')] + \
        [fwdsim.model._circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.prod([op.total_term_magnitude for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model._circuit_layer_operator(elbl, 'povm').total_term_magnitude
         for elbl in elabels], 'd')

    mag = _np.zeros(len(elabels), 'd')
    nPaths = _np.zeros(len(elabels), _np.int64)

    def count_path(b, mg, incd):
        mag[E_indices[b[-1]]] += mg
        nPaths[E_indices[b[-1]]] += 1

    traverse_paths_upto_threshold(factor_lists, threshold, len(elabels),
                                  foat_indices_per_op, count_path)  # sets mag and nPaths
    return mag, max_sum_of_pathmags

    #threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
    #    factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
    #    initial_threshold=current_threshold, min_threshold=pathmagnitude_gap / 1000.0, max_npaths=max_paths)


#global_cnt = 0
#Base case which works for both SV and SB evolution types thanks to Python's duck typing


def find_best_pathmagnitude_threshold(fwdsim, rholabel, elabels, circuit, polynomial_vindices_per_int,
                                      repcache, circuitsetup_cache, comm, mem_limit, pathmagnitude_gap,
                                      min_term_mag, max_paths, threshold_guess):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    circuitsetup_cache : dict, optional
        Dictionary used to cache preparation specific to this function, to
        speed up repeated calls using the same circuit and set of parameters,
        including the same repcache.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    pathmagnitude_gap : float, optional
        The amount less than the perfect sum-of-path-magnitudes that
        is desired.  This sets the target sum-of-path-magnitudes for each
        circuit -- the threshold that determines how many paths are added.

    min_term_mag : float, optional
        A technical parameter to the path pruning algorithm; this value
        sets a threshold for how small a term magnitude (one factor in
        a path magnitude) must be before it is removed from consideration
        entirely (to limit the number of even *potential* paths).  Terms
        with a magnitude lower than this values are neglected.

    max_paths : int, optional
        The maximum number of paths allowed per circuit outcome.

    threshold_guess : float, optional
        In the search for a good pathmagnitude threshold, this value is
        used as the starting point.  If 0.0 is given, a default value is used.

    Returns
    -------
    npaths : int
        the number of paths that were included.
    threshold : float
        the path-magnitude threshold used.
    target_sopm : float
        The desired sum-of-path-magnitudes.  This is `pathmagnitude_gap`
        less than the perfect "all-paths" sum.  This sums together the
        contributions of different effects.
    achieved_sopm : float
        The achieved sum-of-path-magnitudes.  Ideally this would equal
        `target_sopm`. (This also sums together the contributions of
        different effects.)
    """
    if circuitsetup_cache is None: circuitsetup_cache = {}

    if circuit not in circuitsetup_cache:
        circuitsetup_cache[circuit] = create_circuitsetup_cacheel(
            fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, fwdsim.model.num_params)
    rho_term_reps, op_term_reps, E_term_reps, \
        rho_foat_indices, op_foat_indices, E_foat_indices, E_indices = circuitsetup_cache[circuit]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model._circuit_layer_operator(rholabel, 'prep')] + \
        [fwdsim.model._circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.prod([op.total_term_magnitude for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model._circuit_layer_operator(elbl, 'povm').total_term_magnitude
         for elbl in elabels], 'd')
    target_sum_of_pathmags = max_sum_of_pathmags - pathmagnitude_gap  # absolute gap
    #target_sum_of_pathmags = max_sum_of_pathmags * (1.0 - pathmagnitude_gap)  # relative gap
    threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
        factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
        initial_threshold=threshold_guess, min_threshold=pathmagnitude_gap / (3.0 * max_paths),  # 3.0 is just heuristic
        max_npaths=max_paths)
    # above takes an array of target pathmags and gives a single threshold that works for all of them (all E-indices)

    # DEBUG PRINT
    #print("Threshold = ", threshold, " Paths=", npaths)
    #global global_cnt
    # print("Threshold = ", threshold, " Paths=", npaths, " tgt=", target_sum_of_pathmags,
    #       "cnt = ", global_cnt)  # , " time=%.3fs" % (_time.time()-t0))
    #global_cnt += 1

    # DEBUG PRINT
    # print("---------------------------")
    # print("Path threshold = ",threshold, " max=",max_sum_of_pathmags,
    #       " target=",target_sum_of_pathmags, " achieved=",achieved_sum_of_pathmags)
    # print("nPaths = ",npaths)
    # print("Num high-magnitude (|coeff|>%g, taylor<=%d) terms: %s" \
    #       % (min_term_mag, fwdsim.max_order, str([len(factors) for factors in factor_lists])))
    # print("Num FOAT: ",[len(inds) for inds in foat_indices_per_op])
    # print("---------------------------")

    target_miss = sum(achieved_sum_of_pathmags) - sum(target_sum_of_pathmags + pathmagnitude_gap)
    if target_miss > 1e-5:
        print("Warning: Achieved sum(path mags) exceeds max by ", target_miss, "!!!")

    return sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)


def compute_pruned_path_polynomials_given_threshold(threshold, fwdsim, rholabel, elabels, circuit,
                                                    polynomial_vindices_per_int, repcache, circuitsetup_cache,
                                                    comm, mem_limit, fastmode):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    circuitsetup_cache : dict, optional
        Dictionary used to cache preparation specific to this function, to
        speed up repeated calls using the same circuit and set of parameters,
        including the same repcache.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    Returns
    -------
    prps : list of PolynomialRep objects
        the polynomials for the requested circuit probabilities, computed by
        selectively summing up high-magnitude paths.
    """
    if circuitsetup_cache is None: circuitsetup_cache = {}

    if circuit not in circuitsetup_cache:
        circuitsetup_cache[circuit] = create_circuitsetup_cacheel(
            fwdsim, rholabel, elabels, circuit, repcache, fwdsim.min_term_mag, fwdsim.model.num_params)
    rho_term_reps, op_term_reps, E_term_reps, \
        rho_foat_indices, op_foat_indices, E_foat_indices, E_indices = circuitsetup_cache[circuit]

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    prps = [None] * len(elabels)
    last_index = len(factor_lists) - 1

    #print("T1 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    if fastmode == 1:  # fastmode
        leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
        rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
        coeffSaved = [None] * (len(factor_lists) - 1)

        def add_path(b, mag, incd):
            """ Relies on the fact that paths are iterated over in lexographic order, and `incd`
                tells us which index was just incremented (all indices less than this one are
                the *same* as the last call). """
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no
            # ability to cache?)
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]

            if incd == 0:  # need to re-evaluate rho vector
                rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                for f in factors[0].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factors[0].post_state
                for f in factors[0].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[0] = rhoVecR

                coeff = factors[0].coeff
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd - 1]
                rhoVecR = rightSaved[incd - 1]
                coeff = coeffSaved[incd - 1]

            # propagate left and right states, saving as we go
            for i in range(incd, last_index):
                for f in factors[i].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[i] = rhoVecL

                for f in factors[i].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = coeff.mult(factors[i].coeff)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            for f in factors[-1].pre_ops:
                rhoVecL = f.acton(rhoVecL)
            E = factors[-1].post_effect  # effect representation
            pLeft = E.amplitude(rhoVecL)

            #Same for post_ops and rhoVecR
            for f in factors[-1].post_ops:
                rhoVecR = f.acton(rhoVecR)
            E = factors[-1].pre_effect
            pRight = _np.conjugate(E.amplitude(rhoVecR))

            res = coeff.mult(factors[-1].coeff)
            res.scale((pLeft * pRight))
            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index

            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    elif fastmode == 2:  # achieved-SOPM mode
        def add_path(b, mag, incd):
            """Adds in |pathmag| = |prod(factor_coeffs)| for computing achieved SOPM"""
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff.abs() for f in factors])

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    else:
        def add_path(b, mag, incd):
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
            pLeft = _unitary_sim_pre(factors, comm, mem_limit)
            pRight = _unitary_sim_post(factors, comm, mem_limit)
            res.scale((pLeft * pRight))

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

            #print("DB running prps[",Ei,"] =",prps[Ei])

    traverse_paths_upto_threshold(factor_lists, threshold, len(
        elabels), foat_indices_per_op, add_path)  # sets mag and nPaths

    #print("T2 = %.2fs" % (_time.time()-t0)); t0 = _time.time()
    return prps  # Note: prps are PolynomialReps and not Polynomials


def create_circuitsetup_cacheel(fwdsim, rholabel, elabels, circuit, repcache, min_term_mag, mpv):
    # Construct dict of gate term reps
    mpv = fwdsim.model.num_params  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model._circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)
    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    return (rho_term_reps, op_term_reps, E_term_reps,
            rho_foat_indices, op_foat_indices, E_foat_indices,
            E_indices)


#Base case which works for both SV and SB evolution types thanks to Python's duck typing
def _prs_as_pruned_polys(fwdsim, rholabel, elabels, circuit, repcache, comm=None, mem_limit=None, fastmode=True,
                         pathmagnitude_gap=0.0, min_term_mag=0.01, max_paths=500, current_threshold=None,
                         compute_polyreps=True):
    """
    Computes probabilities for multiple spam-tuples of `circuit`

    Parameters
    ----------
    fwdsim : TermForwardSimulator
        The calculator object holding vital information for the computation.

    rholabel : Label
        Prep label for *all* the probabilities to compute.

    elabels : list
        List of effect labels, one per probability to compute.  The ordering
        of `elabels` determines the ordering of the returned probability
        polynomials.

    circuit : Circuit
        The gate sequence to sandwich between the prep and effect labels.

    repcache : dict, optional
        Dictionary used to cache operator representations to speed up future
        calls to this function that would use the same set of operations.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes.

    fastmode : bool, optional
        A switch between a faster, slighty more memory hungry mode of
        computation (`fastmode=True`)and a simpler slower one (`=False`).

    pathmagnitude_gap : float, optional
        The amount less than the perfect sum-of-path-magnitudes that
        is desired.  This sets the target sum-of-path-magnitudes for each
        circuit -- the threshold that determines how many paths are added.

    min_term_mag : float, optional
        A technical parameter to the path pruning algorithm; this value
        sets a threshold for how small a term magnitude (one factor in
        a path magnitude) must be before it is removed from consideration
        entirely (to limit the number of even *potential* paths).  Terms
        with a magnitude lower than this values are neglected.

    current_threshold : float, optional
        If the threshold needed to achieve the desired `pathmagnitude_gap`
        is greater than this value (i.e. if using current_threshold would
        result in *more* paths being computed) then this function will not
        compute any paths and exit early, returning `None` in place of the
        usual list of polynomial representations.

    max_paths : int, optional
        The maximum number of paths that will be summed to compute the polynomials
        for this circuit.

    compute_polyreps: bool, optional
        If `False`, then the polynomials are not actually constructed -- only the
        sum-of-path-magnitudes are computed.  This is useful when testing a given
        threshold to see if the paths are sufficient, before committing to building
        all of the polynomials (which can be time consuming).

    Returns
    -------
    prps : list of PolynomialRep objects
        the polynomials for the requested circuit probabilities, computed by
        selectively summing up high-magnitude paths.  If `compute_polyreps == False`,
        then an empty list is returned.
    npaths : int
        the number of paths that were included.
    threshold : float
        the path-magnitude threshold used.
    target_sopm : float
        The desired sum-of-path-magnitudes.  This is `pathmagnitude_gap`
        less than the perfect "all-paths" sum.  This sums together the
        contributions of different effects.
    achieved_sopm : float
        The achieved sum-of-path-magnitudes.  Ideally this would equal
        `target_sopm`. (This also sums together the contributions of
        different effects.)
    """
    #t0 = _time.time()
    # Construct dict of gate term reps
    mpv = fwdsim.model.num_params  # max_polynomial_vars
    distinct_gateLabels = sorted(set(circuit))

    op_term_reps = {}
    op_foat_indices = {}
    for glbl in distinct_gateLabels:
        if glbl not in repcache:
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(glbl, 'op').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            repcache[glbl] = ([t.torep() for t in hmterms], foat_indices)
        op_term_reps[glbl], op_foat_indices[glbl] = repcache[glbl]

    if rholabel not in repcache:
        hmterms, foat_indices = fwdsim.model._circuit_layer_operator(rholabel, 'prep').highmagnitude_terms(
            min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
        repcache[rholabel] = ([t.torep() for t in hmterms], foat_indices)
    rho_term_reps, rho_foat_indices = repcache[rholabel]

    elabels = tuple(elabels)  # so hashable
    if elabels not in repcache:
        E_term_indices_and_reps = []
        for i, elbl in enumerate(elabels):
            hmterms, foat_indices = fwdsim.model._circuit_layer_operator(elbl, 'povm').highmagnitude_terms(
                min_term_mag, max_taylor_order=fwdsim.max_order, max_polynomial_vars=mpv)
            E_term_indices_and_reps.extend(
                [(i, t.torep(), t.magnitude, bool(j in foat_indices)) for j, t in enumerate(hmterms)])

        #Sort all terms by magnitude
        E_term_indices_and_reps.sort(key=lambda x: x[2], reverse=True)
        E_term_reps = [x[1] for x in E_term_indices_and_reps]
        E_indices = [x[0] for x in E_term_indices_and_reps]
        E_foat_indices = [j for j, x in enumerate(E_term_indices_and_reps) if x[3] is True]
        repcache[elabels] = (E_term_reps, E_indices, E_foat_indices)

    E_term_reps, E_indices, E_foat_indices = repcache[elabels]

    prps = [None] * len(elabels)

    factor_lists = [rho_term_reps] + \
        [op_term_reps[glbl] for glbl in circuit] + \
        [E_term_reps]
    last_index = len(factor_lists) - 1

    foat_indices_per_op = [rho_foat_indices] + [op_foat_indices[glbl] for glbl in circuit] + [E_foat_indices]

    ops = [fwdsim.model._circuit_layer_operator(rholabel, 'prep')] + \
        [fwdsim.model._circuit_layer_operator(glbl, 'op') for glbl in circuit]
    max_sum_of_pathmags = _np.prod([op.total_term_magnitude for op in ops])
    max_sum_of_pathmags = _np.array(
        [max_sum_of_pathmags * fwdsim.model._circuit_layer_operator(elbl, 'povm').total_term_magnitude
         for elbl in elabels], 'd')
    target_sum_of_pathmags = max_sum_of_pathmags - pathmagnitude_gap  # absolute gap
    #target_sum_of_pathmags = max_sum_of_pathmags * (1.0 - pathmagnitude_gap)  # relative gap
    threshold, npaths, achieved_sum_of_pathmags = pathmagnitude_threshold(
        factor_lists, E_indices, len(elabels), target_sum_of_pathmags, foat_indices_per_op,
        initial_threshold=current_threshold,
        min_threshold=pathmagnitude_gap / (3.0 * max_paths),  # 3.0 is just heuristic
        max_npaths=max_paths)
    # above takes an array of target pathmags and gives a single threshold that works for all of them (all E-indices)

    # DEBUG PRINT (and global_cnt definition above)
    #print("Threshold = ", threshold, " Paths=", npaths)
    #global global_cnt
    # print("Threshold = ", threshold, " Paths=", npaths, " tgt=", target_sum_of_pathmags,
    #       "cnt = ", global_cnt)  # , " time=%.3fs" % (_time.time()-t0))
    #global_cnt += 1

    # no polyreps needed, e.g. just keep existing (cached) polys
    if not compute_polyreps or (current_threshold >= 0 and threshold >= current_threshold):
        return [], sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)

    #print("T1 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    if fastmode:
        leftSaved = [None] * (len(factor_lists) - 1)  # saved[i] is state after i-th
        rightSaved = [None] * (len(factor_lists) - 1)  # factor has been applied
        coeffSaved = [None] * (len(factor_lists) - 1)

        def add_path(b, mag, incd):
            """ Relies on the fact that paths are iterated over in lexographic order, and `incd`
                tells us which index was just incremented (all indices less than this one are
                the *same* as the last call). """
            # "non-fast" mode is the only way we know to do this, since we don't know what path will come next (no
            # ability to cache?)
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]

            if incd == 0:  # need to re-evaluate rho vector
                rhoVecL = factors[0].pre_state  # Note: `factor` is a rep & so are it's ops
                for f in factors[0].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[0] = rhoVecL

                rhoVecR = factors[0].post_state
                for f in factors[0].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[0] = rhoVecR

                coeff = factors[0].coeff
                coeffSaved[0] = coeff
                incd += 1
            else:
                rhoVecL = leftSaved[incd - 1]
                rhoVecR = rightSaved[incd - 1]
                coeff = coeffSaved[incd - 1]

            # propagate left and right states, saving as we go
            for i in range(incd, last_index):
                for f in factors[i].pre_ops:
                    rhoVecL = f.acton(rhoVecL)
                leftSaved[i] = rhoVecL

                for f in factors[i].post_ops:
                    rhoVecR = f.acton(rhoVecR)
                rightSaved[i] = rhoVecR

                coeff = coeff.mult(factors[i].coeff)
                coeffSaved[i] = coeff

            # for the last index, no need to save, and need to construct
            # and apply effect vector
            for f in factors[-1].pre_ops:
                rhoVecL = f.acton(rhoVecL)
            E = factors[-1].post_effect  # effect representation
            pLeft = E.amplitude(rhoVecL)

            #Same for post_ops and rhoVecR
            for f in factors[-1].post_ops:
                rhoVecR = f.acton(rhoVecR)
            E = factors[-1].pre_effect
            pRight = _np.conjugate(E.amplitude(rhoVecR))

            res = coeff.mult(factors[-1].coeff)
            res.scale((pLeft * pRight))
            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index

            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res

    else:
        def add_path(b, mag, incd):
            factors = [factor_lists[i][factorInd] for i, factorInd in enumerate(b)]
            res = _functools.reduce(lambda x, y: x.mult(y), [f.coeff for f in factors])
            pLeft = _unitary_sim_pre(factors, comm, mem_limit)
            pRight = _unitary_sim_post(factors, comm, mem_limit)
            res.scale((pLeft * pRight))

            final_factor_indx = b[-1]
            Ei = E_indices[final_factor_indx]  # final "factor" index == E-vector index
            #print("DB: pr_as_poly     factor coeff=",coeff," pLeft=",pLeft," pRight=",pRight, "res=",res)
            if prps[Ei] is None: prps[Ei] = res
            else: prps[Ei].add_inplace(res)  # prps[Ei] += res
            #print("DB running prps[",Ei,"] =",prps[Ei])

    traverse_paths_upto_threshold(factor_lists, threshold, len(
        elabels), foat_indices_per_op, add_path)  # sets mag and nPaths

    #print("T2 = %.2fs" % (_time.time()-t0)); t0 = _time.time()

    # #DEBUG PRINT
    # print("---------------------------")
    # print("Path threshold = ",threshold, " max=",max_sum_of_pathmags,
    #       " target=",target_sum_of_pathmags, " achieved=",achieved_sum_of_pathmags)
    # print("nPaths = ",npaths)
    # print("Num high-magnitude (|coeff|>%g, taylor<=%d) terms: %s" \
    #       % (min_term_mag, fwdsim.max_order, str([len(factors) for factors in factor_lists])))
    # print("Num FOAT: ",[len(inds) for inds in foat_indices_per_op])
    # print("---------------------------")

    target_miss = sum(achieved_sum_of_pathmags) - sum(target_sum_of_pathmags + pathmagnitude_gap)
    if target_miss > 1e-5:
        print("Warning: Achieved sum(path mags) exceeds max by ", target_miss, "!!!")

    # NOTE: prps are PolynomialReps and not Polynomials
    return prps, sum(npaths), threshold, sum(target_sum_of_pathmags), sum(achieved_sum_of_pathmags)


# foat = first-order always-traversed
def traverse_paths_upto_threshold(oprep_lists, pathmag_threshold, num_elabels, foat_indices_per_op,
                                  fn_visitpath, debug=False):
    """
    Traverse all the paths up to some path-magnitude threshold, calling
    `fn_visitpath` for each one.

    Parameters
    ----------
    oprep_lists : list of lists
        representations for the terms of each layer of the circuit whose
        outcome probability we're computing, including prep and POVM layers.
        `oprep_lists[i]` is a list of the terms available to choose from
        for the i-th circuit layer, ordered by increasing term-magnitude.

    pathmag_threshold : float
        the path-magnitude threshold to use.

    num_elabels : int
        The number of effect labels corresponding whose terms are all
        amassed in the in final `oprep_lists[-1]` list (knowing which
        elements of `oprep_lists[-1]` correspond to which effect isn't
        necessary for this function).

    foat_indices_per_op : list
        A list of lists of integers, such that `foat_indices_per_op[i]`
        is a list of indices into `oprep_lists[-1]` that marks out which
        terms are first-order (Taylor) terms that should therefore always
        be traversed regardless of their term-magnitude (foat = first-order-
        always-traverse).

    fn_visitpath : function
        A function called for each path that is traversed.  Arguments
        are `(term_indices, magnitude, incd)` where `term_indices` is
        an array of integers giving the index into each `oprep_lists[i]`
        list, `magnitude` is the path magnitude, and `incd` is the index
        of the circuit layer that was just incremented (all elements of
        `term_indices` less than this index are guaranteed to be the same
        as they were in the last call to `fn_visitpath`, and this can be
        used for faster path evaluation.

    max_npaths : int, optional
        The maximum number of paths to traverse.  If this is 0, then there
        is no limit.  Otherwise this function will return as soon as
        `max_npaths` paths are traversed.

    debug : bool, optional
        Whether to print additional debug info.

    Returns
    -------
    None
    """  # zot = zero-order-terms
    n = len(oprep_lists)
    nops = [len(oprep_list) for oprep_list in oprep_lists]
    b = [0] * n  # root
    log_thres = _math.log10(pathmag_threshold)

    def traverse_tree(root, incd, log_thres, current_mag, current_logmag, order, current_nzeros):
        """ first_order means only one b[i] is incremented, e.g. b == [0 1 0] or [4 0 0] """
        b = root
        #print("BEGIN: ",root)
        for i in reversed(range(incd, n)):
            if b[i] + 1 == nops[i]: continue
            b[i] += 1

            if order == 0:  # then incd doesn't matter b/c can inc anything to become 1st order
                sub_order = 1 if (i != n - 1 or b[i] >= num_elabels) else 0
            elif order == 1:
                # we started with a first order term where incd was incremented, and now
                # we're incrementing something else
                sub_order = 1 if i == incd else 2  # signifies anything over 1st order where >1 column has be inc'd
            else:
                sub_order = order

            logmag = current_logmag + (oprep_lists[i][b[i]].logmagnitude - oprep_lists[i][b[i] - 1].logmagnitude)
            #print("Trying: ",b)
            if logmag >= log_thres:  # or sub_order == 0:
                numerator = oprep_lists[i][b[i]].magnitude
                denom = oprep_lists[i][b[i] - 1].magnitude
                nzeros = current_nzeros

                if denom == 0:
                    # Note: adjust logmag because when term's mag == 0, it's logmag == 0 also (convention)
                    denom = SMALL; nzeros -= 1; logmag -= LOGSMALL
                if numerator == 0:
                    numerator = SMALL; nzeros += 1; logmag += LOGSMALL

                mag = current_mag * (numerator / denom)
                actual_mag = mag if (nzeros == 0) else 0.0  # magnitude is actually zero if nzeros > 0

                if fn_visitpath(b, actual_mag, i): return True  # fn_visitpath can signal early return
                if traverse_tree(b, i, log_thres, mag, logmag, sub_order, nzeros):
                    # add any allowed paths beneath this one
                    return True
            elif sub_order <= 1:
                #We've rejected term-index b[i] (in column i) because it's too small - the only reason
                # to accept b[i] or term indices higher than it is to include "foat" terms, so we now
                # iterate through any remaining foat indices for this column (we've accepted all lower
                # values of b[i], or we wouldn't be here).  Note that we just need to visit the path,
                # we don't need to traverse down, since we know the path magnitude is already too low.
                orig_bi = b[i]
                for j in foat_indices_per_op[i]:
                    if j >= orig_bi:
                        b[i] = j
                        nzeros = current_nzeros
                        numerator = oprep_lists[i][b[i]].magnitude
                        denom = oprep_lists[i][orig_bi - 1].magnitude
                        if denom == 0: denom = SMALL

                        #if numerator == 0: nzeros += 1  # not needed b/c we just leave numerator = 0
                        # OK if mag == 0 as it's not passed to any recursive calls
                        mag = current_mag * (numerator / denom)
                        actual_mag = mag if (nzeros == 0) else 0.0  # magnitude is actually zero if nzeros > 0

                        if fn_visitpath(b, actual_mag, i): return True

                        if i != n - 1:
                            # if we're not incrementing (from a zero-order term) the final index, then we
                            # need to to increment it until we hit num_elabels (*all* zero-th order paths)
                            orig_bn = b[n - 1]
                            for k in range(1, num_elabels):
                                b[n - 1] = k
                                numerator = oprep_lists[n - 1][b[n - 1]].magnitude
                                denom = oprep_lists[i][orig_bn].magnitude
                                if denom == 0: denom = SMALL
                                # zero if either numerator == 0 or mag == 0 from above.
                                mag2 = mag * (numerator / denom)
                                if fn_visitpath(b, mag2 if (nzeros == 0) else 0.0, n - 1): return True

                            b[n - 1] = orig_bn

                b[i] = orig_bi

            b[i] -= 1  # so we don't have to copy b
        #print("END: ",root)
        return False  # return value == "do we need to terminate traversal immediately?"

    current_mag = 1.0; current_logmag = 0.0
    fn_visitpath(b, current_mag, 0)  # visit root (all 0s) path
    traverse_tree(b, 0, log_thres, current_mag, current_logmag, 0, 0)

    return


def pathmagnitude_threshold(oprep_lists, e_indices, num_elabels, target_sum_of_pathmags,
                            foat_indices_per_op=None, initial_threshold=0.1,
                            min_threshold=1e-10, max_npaths=1000000):
    """
    Find the pathmagnitude-threshold needed to achieve some target sum-of-path-magnitudes:
    so that the sum of all the path-magnitudes greater than this threshold achieve the
    target (or get as close as we can).

    Parameters
    ----------
    oprep_lists : list of lists
        representations for the terms of each layer of the circuit whose
        outcome probability we're computing, including prep and POVM layers.
        `oprep_lists[i]` is a list of the terms available to choose from
        for the i-th circuit layer, ordered by increasing term-magnitude.

    e_indices : numpy array
        The effect-vector index for each element of `oprep_lists[-1]`
        (representations for *all* effect vectors exist all together
        in `oprep_lists[-1]`).

    num_elabels : int
        The total number of different effects whose reps appear in
        `oprep_lists[-1]` (also one more than the largest index in
        `e_indices`.

    target_sum_of_pathmags : array
        An array of floats of length `num_elabels` giving the target sum of path
        magnitudes desired for each effect (separately).

    foat_indices_per_op : list
        A list of lists of integers, such that `foat_indices_per_op[i]`
        is a list of indices into `oprep_lists[-1]` that marks out which
        terms are first-order (Taylor) terms that should therefore always
        be traversed regardless of their term-magnitude (foat = first-order-
        always-traverse).

    initial_threshold : float
        The starting pathmagnitude threshold to try (this function uses
        an iterative procedure to find a threshold).

    min_threshold : float
        The smallest threshold allowed.  If this amount is reached, it
        is just returned and searching stops.

    max_npaths : int, optional
        The maximum number of paths allowed per effect.

    Returns
    -------
    threshold : float
        The obtained pathmagnitude threshold.
    npaths : numpy array
        An array of length `num_elabels` giving the number of paths selected
        for each of the effect vectors.
    achieved_sopm : numpy array
        An array of length `num_elabels` giving the achieved sum-of-path-
        magnitudes for each of the effect vectors.
    """
    nIters = 0
    threshold = initial_threshold if (initial_threshold >= 0) else 0.1  # default value
    target_mag = target_sum_of_pathmags
    #print("Target magnitude: ",target_mag)
    threshold_upper_bound = 1.0
    threshold_lower_bound = None
    #mag = 0; nPaths = 0

    if foat_indices_per_op is None:
        foat_indices_per_op = [()] * len(oprep_lists)

    def count_path(b, mg, incd):
        mag[e_indices[b[-1]]] += mg
        nPaths[e_indices[b[-1]]] += 1

        return (nPaths[e_indices[b[-1]]] == max_npaths)  # trigger immediate return if hit max_npaths

    while nIters < 100:  # TODO: allow setting max_nIters as an arg?
        mag = _np.zeros(num_elabels, 'd')
        nPaths = _np.zeros(num_elabels, _np.int64)

        traverse_paths_upto_threshold(oprep_lists, threshold, num_elabels,
                                      foat_indices_per_op, count_path)  # sets mag and nPaths
        assert(max_npaths == 0 or _np.all(nPaths <= max_npaths)), "MAX PATHS EXCEEDED! (%s)" % nPaths

        if _np.all(mag >= target_mag) or _np.any(nPaths >= max_npaths):  # try larger threshold
            threshold_lower_bound = threshold
            if threshold_upper_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold *= 2
        else:  # try smaller threshold
            threshold_upper_bound = threshold
            if threshold_lower_bound is not None:
                threshold = (threshold_upper_bound + threshold_lower_bound) / 2
            else: threshold /= 2

        if threshold_upper_bound is not None and threshold_lower_bound is not None and \
           (threshold_upper_bound - threshold_lower_bound) / threshold_upper_bound < 1e-3:
            #print("Converged after %d iters!" % nIters)
            break
        if threshold_upper_bound < min_threshold:  # could also just set min_threshold to be the lower bound initially?
            threshold_upper_bound = threshold_lower_bound = min_threshold
            break

        nIters += 1

    #Run path traversal once more to count final number of paths

    def count_path_nomax(b, mg, incd):
        # never returns True - we want to check *threshold* alone selects correct # of paths
        mag[e_indices[b[-1]]] += mg
        nPaths[e_indices[b[-1]]] += 1

    mag = _np.zeros(num_elabels, 'd')
    nPaths = _np.zeros(num_elabels, _np.int64)
    traverse_paths_upto_threshold(oprep_lists, threshold_lower_bound, num_elabels,
                                  foat_indices_per_op, count_path_nomax)  # sets mag and nPaths

    return threshold_lower_bound, nPaths, mag


def _unitary_sim_pre(complete_factors, comm, mem_limit):
    rhoVec = complete_factors[0].pre_state  # a prep representation
    for f in complete_factors[0].pre_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.pre_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].pre_ops:
        rhoVec = f.acton(rhoVec)

    EVec = complete_factors[-1].post_effect
    return EVec.amplitude(rhoVec)


def _unitary_sim_post(complete_factors, comm, mem_limit):
    rhoVec = complete_factors[0].post_state  # a prep representation
    for f in complete_factors[0].post_ops:
        rhoVec = f.acton(rhoVec)
    for f in _itertools.chain(*[f.post_ops for f in complete_factors[1:-1]]):
        rhoVec = f.acton(rhoVec)  # LEXICOGRAPHICAL VS MATRIX ORDER

    for f in complete_factors[-1].post_ops:
        rhoVec = f.acton(rhoVec)
    EVec = complete_factors[-1].pre_effect
    return _np.conjugate(EVec.amplitude(rhoVec))  # conjugate for same reason as above
