"""
Functions for selecting a complete set of fiducials for a GST analysis.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy
import random
import itertools
from math import floor
from pygsti.algorithms import grasp as _grasp
from pygsti.algorithms import scoring as _scoring
from pygsti import circuits as _circuits
from pygsti import baseobjs as _baseobjs
from pygsti.modelmembers.povms import ComplementPOVMEffect as _ComplementPOVMEffect
from pygsti.tools import frobeniusdist_squared

from pygsti.algorithms.germselection import construct_update_cache, minamide_style_inverse_trace, compact_EVD, compact_EVD_via_SVD


def find_fiducials(target_model, omit_identity=True, eq_thresh=1e-6,
                   ops_to_omit=None, force_empty=True, candidate_fid_counts=2,
                   algorithm='grasp', algorithm_kwargs=None, verbosity=1,
                   prep_fids=True, meas_fids=True, candidate_list=None,
                   return_candidate_list=False, final_test= False, 
                   assume_clifford=False, candidate_seed=None):
    """
    Generate prep and measurement fiducials for a given target model.

    Parameters
    ----------
    target_model : Model
        The model you are aiming to implement.

    omit_identity : bool, optional
        Whether to remove the identity gate from the set of gates with which
        fiducials are constructed. Identity gates do nothing to alter
        fiducials, and so should almost always be left out.

    eq_thresh : float, optional
        Threshold for determining if a gate is the identity gate. If the square
        Frobenius distance between a given gate and the identity gate is less
        than this threshold, the gate is considered to be an identity gate and
        will be removed from the list of gates from which to construct
        fiducials if `omit_identity` is ``True``.

    ops_to_omit : list of string, optional
        List of strings identifying gates in the model that should not be
        used in fiducials. Oftentimes this will include the identity gate, and
        may also include entangling gates if their fidelity is anticipated to
        be much worse than that of single-system gates.

    force_empty : bool, optional (default is True)
        Whether or not to force all fiducial sets to contain the empty gate
        string as a fiducial.

    candidate_fid_counts : int or dic, optional
        A dictionary of *fid_length* : *count* key-value pairs, specifying
        the fiducial "candidate list" - a list of potential fiducials to draw from.
        *count* is either an integer specifying the number of random fiducials
        considered at the given *fid_length* or the special values `"all upto"`
        that considers all of the of all the fiducials of length up to
        the corresponding *fid_length*. If the keyword 'all' is used for the
        count value then all circuits at that particular length are added. 
        If and integer, all germs of up to length
        that length are used, the equivalent of `{specified_int: 'all upto'}`.

    algorithm : {'slack', 'grasp', 'greedy'}, optional
        Specifies the algorithm to use to generate the fiducials. Current
        options are:

        'slack'
            See :func:`_find_fiducials_integer_slack` for more details.
        'grasp'
            Use GRASP to generate random greedy fiducial sets and then locally
            optimize them. See :func:`_find_fiducials_grasp` for more
            details.
        'greedy'
            Use a greedy algorithm accelerated using low-rank update techniques.
            See :func:`_find_fiducials_greedy` for more
            details.

    algorithm_kwargs : dict
        Dictionary of ``{'keyword': keyword_arg}`` pairs providing keyword
        arguments for the specified `algorithm` function. See the documentation
        for functions referred to in the `algorithm` keyword documentation for
        what options are available for each algorithm.

    verbosity : int, optional
        How much detail to send to stdout.

    candidate_list : list of circuits, optional
        A user specified manually selected list of candidate fiducial circuits.
        Can speed up testing multiple objective function options, for example.
    
    return_candidate_list: bool, optional (default False)
        When True we return the full list of deduped candidate fiducials considered
        along with the final fiducial lists.
        
    final_test : bool, optional (default False)
        When true a final check is performed on the returned solution for the candidate
        prep and measurement lists using the function test_fiducial_list to verify
        we have an acceptable candidate set (this uses a different code path in some cases so
        can be used to detect errors). 
        
    assume_clifford : bool, optional (default False)
        When true then we assume that all of the circuits are clifford circuits,
        which allows us to use a faster deduping routine exploiting the properties
        of clifford circuits.
        

    Returns
    -------
    prepFidList : list of Circuits
        A list containing the circuits for the prep fiducials.
    measFidList : list of Circuits
        A list containing the circuits for the measurement fiducials.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)
    
    #If the user hasn't specified a candidate list manually then generate one:
    if candidate_list is None:
        if ops_to_omit is None:
            ops_to_omit = []
        
        fidOps = [gate for gate in target_model.operations if gate not in ops_to_omit]
        
        if omit_identity:
            # we assume identity gate is always the identity mx regardless of basis
            Identity = _np.identity(target_model.dim, 'd')

            for gate in fidOps:
                if frobeniusdist_squared(target_model.operations[gate], Identity) < eq_thresh:
                    fidOps.remove(gate)
        
        availableFidList = []
        if isinstance(candidate_fid_counts, int): 
            candidate_fid_counts = {candidate_fid_counts: 'all upto'}
        
        for fidLength, count in candidate_fid_counts.items():
            if count == "all upto":
                availableFidList.extend(_circuits.list_all_circuits(fidOps, 0, fidLength))
            #if "all" then include all circuits at that particular length.
            elif count =="all":
                availableFidList.extend(_circuits.list_all_circuits_one_len(fidOps, fidLength))
            else:
                availableFidList.extend(_circuits.list_random_circuits_onelen(
                    fidOps, fidLength, count, seed=candidate_seed))
        
        printer.log('Initial Length Available Fiducial List: '+ str(len(availableFidList)), 1)
        
        
        printer.log('Creating cache of fiducial process matrices.', 3)
        circuit_cache= create_circuit_cache(target_model,availableFidList)
        printer.log('Completed cache of fiducial process matrices.', 3)
        
        #print('Initial Length Available Fiducial List: ', len(availableFidList))
        
        #Now that we have a cache of PTMs as numpy arrays for the initial list of available fiducials
        #we can clean this list up to remove any effective identities and circuits with duplicate effects.
        #Use a flag to check if we can assume these are clifford circuits in which case
        #we can use a more efficient deduping routine.
        if assume_clifford:
            cleaned_availableFidList, cleaned_circuit_cache = clean_fid_list_clifford(target_model, circuit_cache, availableFidList,
                                                                            drop_identities=True, drop_duplicates=True,
                                                                            eq_thresh=eq_thresh)
        else:
            cleaned_availableFidList, cleaned_circuit_cache = clean_fid_list(target_model, circuit_cache, availableFidList,
                                                                            drop_identities=True, drop_duplicates=True,
                                                                            eq_thresh=eq_thresh)
        
        printer.log('Length Available Fiducial List Dropped Identities and Duplicates: ' + str(len(cleaned_availableFidList)), 1)
        #print('Length Available Fiducial List Dropped Identities and Duplicates: ', len(cleaned_availableFidList))
        
        #TODO: I can speed this up a bit more by looking through the available fiducial list for
        #circuits that are effective identities. Reducing the search space should be a big time-space
        #saver.
    #otherwise if the user has manually specified a list of fiducials then set cleaned_availableFidList to that and
    #create the circuit cache.
    else:
        cleaned_availableFidList = candidate_list
        cleaned_circuit_cache= create_circuit_cache(target_model,cleaned_availableFidList)
    
    #generate a cache for the allowed preps and effects based on availableFidList
    if prep_fids:
        prep_cache= create_prep_cache(target_model, cleaned_availableFidList, cleaned_circuit_cache)
    #TODO: I can technically speed things up even more if we're using the same
    #set of available fidcuials for state prep and measurement since we only
    #would need to do generate the transfer matrices for each circuit once.
    #probably not the most impactful change for the short-term though, performance
    #wise.
    if meas_fids:
        meas_cache= create_meas_cache(target_model, cleaned_availableFidList, cleaned_circuit_cache)
    

    if algorithm_kwargs is None:
        # Avoid danger of using empty dict for default value.
        algorithm_kwargs = {}

    if algorithm == 'slack':
        printer.log('Using slack algorithm.', 1)
        default_kwargs = {
            'fid_list': cleaned_availableFidList,
            'verbosity': max(0, verbosity - 1),
            'force_empty': force_empty,
            'score_func': 'all',
        }

        if ('slack_frac' not in algorithm_kwargs
                and 'fixed_slack' not in algorithm_kwargs):
            algorithm_kwargs['slack_frac'] = 1.0
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]

        prepFidList = _find_fiducials_integer_slack(model=target_model,
                                                    prep_or_meas='prep',
                                                    **algorithm_kwargs)
        if prepFidList is not None:
            prepScore = compute_composite_fiducial_score(
                target_model, prepFidList, 'prep',
                score_func=algorithm_kwargs['score_func'])
            printer.log('Preparation fiducials:', 1)
            printer.log(str([fid.str for fid in prepFidList]), 1)
            printer.log('Score: {}'.format(prepScore.minor), 1)
            #if requested do a final check with test_fiducial_list
            #to verify the algorithm succeeds
            if final_test:
                final_test_fiducial_list = test_fiducial_list(target_model, prepFidList, 'prep',
                                                    score_func=algorithm_kwargs['score_func'], return_all=False,
                                                    threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=prep_cache)
                if final_test_fiducial_list:
                    printer.log('Final test of the candidate prep fiducial lists passed.', 1)
                else:
                    printer.log('Final test of the candidate prep fiducial lists failed.', 1)

        measFidList = _find_fiducials_integer_slack(model=target_model,
                                                    prep_or_meas='meas',
                                                    **algorithm_kwargs)
        if measFidList is not None:
            measScore = compute_composite_fiducial_score(
                target_model, measFidList, 'meas',
                score_func=algorithm_kwargs['score_func'])
            printer.log('Measurement fiducials:', 1)
            printer.log(str([fid.str for fid in measFidList]), 1)
            printer.log('Score: {}'.format(measScore.minor), 1)
            #if requested do a final check with test_fiducial_list
            #to verify the algorithm succeeds
            if final_test:
                final_test_fiducial_list = test_fiducial_list(target_model, measFidList, 'meas',
                                                    score_func=algorithm_kwargs['score_func'], return_all=False,
                                                    threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=meas_cache)
                if final_test_fiducial_list:
                    printer.log('Final test of the candidate prep fiducial lists passed.', 1)
                else:
                    printer.log('Final test of the candidate prep fiducial lists failed.', 1)

    elif algorithm == 'grasp':
        printer.log('Using GRASP algorithm.', 1)
        default_kwargs = {
            'fids_list': cleaned_availableFidList,
            'alpha': 0.1,   # No real reason for setting this value of alpha.
            'op_penalty': 0.1,
            'verbosity': max(0, verbosity - 1),
            'force_empty': force_empty,
            'score_func': 'all',
            'return_all': False,
        }
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
                
        #initialize the prep and measurement fid lists to None
        #so that None gets returned if we aren't running that part
        #of the fiducial search.
        prepFidList=None
        measFidList=None
                
        if prep_fids:
            prepFidList = _find_fiducials_grasp(model=target_model,
                                            prep_or_meas='prep', 
                                            fid_cache= prep_cache,
                                            **algorithm_kwargs)

            if algorithm_kwargs['return_all'] and prepFidList[0] is not None:
                prepScore = compute_composite_fiducial_score(
                    target_model, prepFidList[0], 'prep',
                    score_func=algorithm_kwargs['score_func'])
                printer.log('Preparation fiducials:', 1)
                printer.log(str([fid.str for fid in prepFidList[0]]), 1)
                printer.log('Score: {}'.format(prepScore.minor), 1)
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, prepFidList[0], 'prep',
                                                        score_func=algorithm_kwargs['score_func'], return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=prep_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate prep fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate prep fiducial lists failed.', 1)
            elif not algorithm_kwargs['return_all'] and prepFidList is not None:
                prepScore = compute_composite_fiducial_score(
                    target_model, prepFidList, 'prep',
                    score_func=algorithm_kwargs['score_func'])
                printer.log('Preparation fiducials:', 1)
                printer.log(str([fid.str for fid in prepFidList]), 1)
                printer.log('Score: {}'.format(prepScore.minor), 1)
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, prepFidList, 'prep',
                                                        score_func=algorithm_kwargs['score_func'], return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=prep_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate prep fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate prep fiducial lists failed.', 1)
        if meas_fids:
            measFidList = _find_fiducials_grasp(model=target_model,
                                            prep_or_meas='meas',
                                            fid_cache=meas_cache,
                                            **algorithm_kwargs)

            if algorithm_kwargs['return_all'] and measFidList[0] is not None:
                measScore = compute_composite_fiducial_score(
                    target_model, measFidList[0], 'meas',
                    score_func=algorithm_kwargs['score_func'])
                printer.log('Measurement fiducials:', 1)
                printer.log(str([fid.str for fid in measFidList[0]]), 1)
                printer.log('Score: {}'.format(measScore.minor), 1)
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, measFidList[0], 'meas',
                                                        score_func=algorithm_kwargs['score_func'], return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=meas_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate meas fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate meas fiducial lists failed.', 1)
            elif not algorithm_kwargs['return_all'] and measFidList is not None:
                measScore = compute_composite_fiducial_score(
                    target_model, measFidList, 'meas',
                    score_func=algorithm_kwargs['score_func'])
                printer.log('Measurement fiducials:', 1)
                printer.log(str([fid.str for fid in measFidList]), 1)
                printer.log('Score: {}'.format(measScore.minor), 1)
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, measFidList, 'meas',
                                                        score_func=algorithm_kwargs['score_func'], return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=meas_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate meas fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate meas fiducial lists failed.', 1)
    
    elif algorithm == 'greedy':
        printer.log('Using greedy algorithm.', 1)
        default_kwargs = {
            'fids_list': cleaned_availableFidList,
            'op_penalty': 0.1,
            'verbosity': verbosity,
            'force_empty': force_empty
        }
        
        #We only support 'all' for the score function with the
        #greedy algorithm, so raise an error for other values.
        if (algorithm_kwargs.get('score_func', None) is not None) and \
           (algorithm_kwargs.get('score_func', None) != 'all'):
            raise ValueError('The greedy fiducial search algorithm currently only support the score function \'all\'.')
        
        for key in default_kwargs:
            if key not in algorithm_kwargs:
                algorithm_kwargs[key] = default_kwargs[key]
                
        #initialize the prep and measurement fid lists to None
        #so that None gets returned if we aren't running that part
        #of the fiducial search.
        prepFidList=None
        measFidList=None
                
        if prep_fids:
            prepFidList, prepScore = _find_fiducials_greedy(model=target_model,
                                            prep_or_meas='prep', 
                                            fid_cache= prep_cache,
                                            **algorithm_kwargs)
            if prepFidList is not None:
                printer.log('Preparation fiducials:', 1)
                printer.log(str([fid.str for fid in prepFidList]), 1)
                printer.log('Score: {}'.format(prepScore.minor), 1)
                
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, prepFidList, 'prep',
                                                        score_func='all', return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), fid_cache=prep_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate prep fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate prep fiducial lists failed.', 1)
        if meas_fids:
            measFidList, measScore = _find_fiducials_greedy(model=target_model,
                                            prep_or_meas='meas',
                                            fid_cache=meas_cache,
                                            **algorithm_kwargs)
            if measFidList is not None:
                printer.log('Measurement fiducials:', 1)
                printer.log(str([fid.str for fid in measFidList]), 1)
                printer.log('Score: {}'.format(measScore.minor), 1)
                
                #if requested do a final check with test_fiducial_list
                #to verify the algorithm succeeds
                if final_test:
                    final_test_fiducial_list = test_fiducial_list(target_model, measFidList, 'meas',
                                                        score_func='all', return_all=False,
                                                        threshold=algorithm_kwargs.get('threshold', 1e-6), 
                                                        fid_cache=meas_cache)
                    if final_test_fiducial_list:
                        printer.log('Final test of the candidate meas fiducial lists passed.', 1)
                    else:
                        printer.log('Final test of the candidate meas fiducial lists failed.', 1)
                
    else:
        raise ValueError("'{}' is not a valid algorithm "
                         "identifier.".format(algorithm))
    
    
    if return_candidate_list:
        return prepFidList, measFidList, cleaned_availableFidList
    else:
        return prepFidList, measFidList    


#def bool_list_to_ind_list(boolList):
#    output = _np.array([])
#    for i, boolVal in boolList:
#        if boolVal == 1:
#            output = _np.append(i)
#    return output

def xor(*args):
    """
    Implements logical xor function for arbitrary number of inputs.

    Parameters
    ----------
    args : bool-likes
        All the boolean (or boolean-like) objects to be checked for xor
        satisfaction.

    Returns
    -------
    output : bool
        True if and only if one and only one element of args is True and the
        rest are False.  False otherwise.
    """

    output = sum(bool(x) for x in args) == 1
    return output
    
#function for cleaning up the available fiducial list to drop identities and circuits with duplicate effects
def clean_fid_list(model, circuit_cache, available_fid_list,drop_identities=True, drop_duplicates=True, eq_thresh= 1e-6):
    #initialize an identity matrix of the appropriate dimension
    
    cleaned_circuit_cache= circuit_cache.copy()
    
    
    if drop_identities:        
        Identity = _np.identity(model.dim, 'd')
        
        #remove identities
        for ckt_key, PTM in circuit_cache.items():
            #Don't remove the empty circuit if it is in the list.
            if ckt_key=='{}' or ckt_key==():
                continue
            #the default tolerance for allclose is probably fine.
            if _np.linalg.norm(PTM- Identity)<eq_thresh:
                #then delete that circuit from the cleaned dictionary
                del cleaned_circuit_cache[ckt_key]
                
    cleaned_circuit_cache_1= cleaned_circuit_cache.copy()            
                
    if drop_duplicates:
        #remove circuits with duplicate PTMs
        #The list of available fidcuials is typically
        #generated in such a way to be listed in increasing order
        #of depth, so if we search for dups in that order this should
        #generally favor the shorted of a pair of duplicate PTMs.
        #cleaned_cache_keys= list(cleaned_circuit_cache.keys())
        #cleaned_cache_PTMs= list(cleaned_circuit_cache.values())
        #len_cache= len(cleaned_cache_keys)
        
        #reverse the list so that the longer circuits are at the start and shorter
        #at the end for better pop behavior.
        
        #TODO: add an option to partition the list into smaller chunks to dedupe
        #separately before regrouping and deduping as a whole. Should be a good deal faster. 
        
        unseen_circs  = list(cleaned_circuit_cache.keys())
        unseen_circs.reverse()
        unique_circs  = []
        
        #While unseen_circs is not empty
        while unseen_circs:
            current_ckt = unseen_circs.pop()
            current_ckt_PTM = cleaned_circuit_cache_1[current_ckt]
            unique_circs.append(current_ckt)            
            #now iterate through the remaining elements of the set of unseen circuits and remove any duplicates.
            is_not_duplicate=[True]*len(unseen_circs)
            for i, ckt in enumerate(unseen_circs):
                #the default tolerance for allclose is probably fine.
                if _np.linalg.norm(cleaned_circuit_cache_1[ckt]-current_ckt_PTM)<eq_thresh: #use same threshold as defined in the base find_fiducials function
                    is_not_duplicate[i]=False
            #reset the set of unseen circuits.
            unseen_circs=list(itertools.compress(unseen_circs, is_not_duplicate))
        
        #rebuild the circuit cache now that it has been de-duped:
        cleaned_circuit_cache_2= {ckt_key: cleaned_circuit_cache_1[ckt_key] for ckt_key in unique_circs}
        
    #now that we've de-duped the circuit_cache, we can pull out the keys of cleaned_circuit_cache_1 to get the
    #new list of available fiducials.
    
    available_fid_list_strings= [ckt.str for ckt in available_fid_list]
    
    cleaned_availableFidList=[]
    for i, fid_string in enumerate(available_fid_list_strings):
        if fid_string in cleaned_circuit_cache_2:
            cleaned_availableFidList.append(available_fid_list[i])
    
    return cleaned_availableFidList, cleaned_circuit_cache_2
    
#function for cleaning up the available fiducial list to drop identities and circuits with duplicate effects
#Specialized for the case where our fiducials are promised to correspond to clifford operations
#in which case we know that the matrices must correspond to permutation matrices which
#gives us some additional structure to exploit.
def clean_fid_list_clifford(model, circuit_cache, available_fid_list,drop_identities=True, drop_duplicates=True, eq_thresh= 1e-6):
    #initialize an identity matrix of the appropriate dimension
    
    cleaned_circuit_cache= circuit_cache.copy()
    
    if drop_identities:        
        Identity = _np.identity(model.dim, 'd')
        
        #remove identities
        for ckt_key, PTM in circuit_cache.items():
            #Don't remove the empty circuit if it is in the list.
            if ckt_key=='{}' or ckt_key==():
                continue
            #the default tolerance for allclose is probably fine.
            if _np.linalg.norm(PTM- Identity)<eq_thresh:
                #then delete that circuit from the cleaned dictionary
                del cleaned_circuit_cache[ckt_key]
                
    cleaned_circuit_cache_1= cleaned_circuit_cache.copy()            
    
    if drop_duplicates:
        #remove circuits with duplicate PTMs
        
        #reverse the list so that the longer circuits are at the start and shorter
        #at the end so that overwritten values in the dict are overwritten with the smallest
        #length duplicate.
        
        #Leverage the fact that we know that the PTMs for clifford circuits
        #should correspond to signed permutation matrices. Take each of the
        #permuation matrices, flatten them, then get a list of the non-zero
        #indices. Then for these non-zero indices construct a second list of
        #the sign of these entries. This should uniquely identify each of the
        #signed permutations.
        reversed_ckt_list = list(cleaned_circuit_cache_1.keys())
        reversed_ckt_list.reverse()
        unique_vec_perm_reps= {}
        for ckt in reversed_ckt_list:
            flattened_PTM= _np.ravel(cleaned_circuit_cache_1[ckt])
            #round to the nearest integer.
            rounded_flattened_PTM= _np.round(flattened_PTM, decimals=0)
            nonzero_indices= _np.nonzero(rounded_flattened_PTM)[0]
            #get the signs of the nonzero elements.
            signs= _np.sign(rounded_flattened_PTM[nonzero_indices])
            #concatenate these two arrays
            nonzero_indices_and_signs = _np.concatenate((nonzero_indices,signs))
            #cast this array to a tuple and then add it to the dictionary
            #I think technically I can hash the ndarrays directly, but just
            #to be safe cast these as tuples first.
            #Actually, I don't think I need to do the deduping in 2 stages.
            #I should be able to directly use the tuples as keys and the
            #ckts as the values in a python dictionary (which is implemented
            #using hash tables on the back end).
            unique_vec_perm_reps[tuple(nonzero_indices_and_signs)]= ckt
        
        #Now get the values of the unique_vec_perm_reps dictionary
        #These should be the depuped circuits.
        
        deduped_ckt_list= list(unique_vec_perm_reps.values())
        
        #rebuild the circuit cache now that it has been de-duped:
        cleaned_circuit_cache_2= {ckt_key: cleaned_circuit_cache_1[ckt_key] for ckt_key in deduped_ckt_list}
        
    #now that we've de-duped the circuit_cache, we can pull out the keys of cleaned_circuit_cache_2 to get the
    #new list of available fiducials.
    available_fid_list_strings= [ckt.str for ckt in available_fid_list]
    
    cleaned_availableFidList=[]
    for i, fid_string in enumerate(available_fid_list_strings):
        if fid_string in cleaned_circuit_cache_2:
            cleaned_availableFidList.append(available_fid_list[i])
    
    return cleaned_availableFidList, cleaned_circuit_cache_2
    

#new function for taking a list of available fiducials and generating a cache of the PTMs
#this will also be useful trimming the list of effective identities and fiducials with
#duplicated effects.

def create_circuit_cache(model, circuit_list):
    """
    Function for generating a cache of PTMs for the available fiducials.
    
    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    ckt_list : list of Circuits
        Full list of all fiducial circuits avalable for constructing an informationally complete state preparation.
    
    Returns
    -------
    dictionary
        A dictionary with keys given by circuits with corresponding
        entries being the PTMs for that circuit.
    
    """
    
    circuit_cache= {}
    for circuit in circuit_list:
        circuit_cache[circuit.str] = model.sim.product(circuit)
    
    return circuit_cache


#new function for generating a cache for the elements of the prep matrices and measurement matrices
#produced by create_prep_mxs and create_meas_mxs. Will also update those two functions to take a cache as
#an argument and generate the list returned by them more efficiently.

def create_prep_cache(model, available_prep_fid_list, circuit_cache=None):
    """
    Make a dictionary structure mapping native state preps and circuits to numpy
    column vectors for the corresponding effective state prep.
    
    This can then be passed into 'create_prep_mxs' to more efficiently generate the
    matrices for score function evaluation.
    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    available_prep_fid_list : list of Circuits
        Full list of all fiducial circuits avalable for constructing an informationally complete state preparation.

    circuit_cache : dict
        dictionary of PTMs for the circuits in the available_prep_fid_list
    
    Returns
    -------
    dictionary
        A dictionary with keys given be tuples of the form (native_prep, ckt) with corresponding
        entries being the numpy vectors for that state prep.
    """
    
    prep_cache = {}
    keylist=[]
    
    if circuit_cache is not None:
        for rho in model.preps.values():
            new_key= rho.to_vector().tobytes()
            keylist.append(new_key)
            for prepFid in available_prep_fid_list:
                prep_cache[(new_key,prepFid.str)] = _np.dot(circuit_cache[prepFid.str], rho.to_dense())
    
    else:
        for rho in model.preps.values():
            new_key= rho.to_vector().tobytes()
            keylist.append(new_key)
            for prepFid in available_prep_fid_list:
                prep_cache[(new_key,prepFid.str)] = _np.dot(model.sim.product(prepFid), rho.to_dense())
    return prep_cache, keylist
    

def create_meas_cache(model, available_meas_fid_list, circuit_cache=None):
    """
    Make a dictionary structure mapping native measurements and circuits to numpy
    column vectors corresponding to the transpose of the effective measurement effects.
    
    This can then be passed into 'create_meas_mxs' to more efficiently generate the
    matrices for score function evaluation.
    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    available_meas_fid_list : list of Circuits
        Full list of all fiducial circuits avalable for constructing an informationally complete measurements.
        
    circuit_cache : dict
        dictionary of PTMs for the circuits in the available_meas_fid_list

    Returns
    -------
    tuple with dictionary and lists of POVM and Effect Key pairs.
        A dictionary with keys given be tuples of the form (native_povm, native_povm_effect, ckt) with corresponding
        entries being the numpy vectors for the transpose of that effective measurement effect.
    """
    
    meas_cache = {}
    keypairlist=[]
    
    if circuit_cache is not None:
        for povm in model.povms.values():
            for E in povm.values():
                if isinstance(E, _ComplementPOVMEffect): continue  # complement is dependent on others
                new_povm_effect_key_pair= (povm.to_vector().tobytes(), E.to_vector().tobytes())
                keypairlist.append(new_povm_effect_key_pair)
                for measFid in available_meas_fid_list:
                    meas_cache[(new_povm_effect_key_pair[0],new_povm_effect_key_pair[1],measFid.str)] = _np.dot(E.to_dense(), circuit_cache[measFid.str])    
    
    else:
        for povm in model.povms.values():
            for E in povm.values():
                if isinstance(E, _ComplementPOVMEffect): continue  # complement is dependent on others
                new_povm_effect_key_pair= (povm.to_vector().tobytes(), E.to_vector().tobytes())
                keypairlist.append(new_povm_effect_key_pair)
                for measFid in available_meas_fid_list:
                    meas_cache[(new_povm_effect_key_pair[0],new_povm_effect_key_pair[1],measFid.str)] = _np.dot(E.to_dense(), model.sim.product(measFid))
                    
    return meas_cache, keypairlist
  

def create_prep_mxs(model, prep_fid_list, prep_cache=None):
    """
    Make a list of matrices for the model preparation operations.

    Makes a list of matrices, where each matrix corresponds to a single
    preparation operation in the model, and the column of each matrix is a
    fiducial acting on that state preparation.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    prep_fid_list : list of Circuits
        List of fiducial circuits for constructing an informationally complete state preparation.
        
    prep_cache : dictionary of effective state preps
        Dictionary of effective state preps cache used to accelerate the generation of the matrices
        used for score function evaluation. Default value is None.
    

    Returns
    -------
    list
        A list of matrices, each of shape `(dim, len(prep_fid_list))` where
        `dim` is the dimension of `model` (4 for a single qubit).  The length
        of this list is equal to the number of state preparations in `model`.
    """

    dimRho = model.dim
    #numRho = len(model.preps)
    numFid = len(prep_fid_list)
    outputMatList = []
    #print(prep_fid_list)
    
    if prep_cache is not None:
        #print('Using Prep Cache')
        for rho_key in prep_cache[1]:
            outputMat = _np.zeros([dimRho, numFid], float)
            for i, prepFid in enumerate(prep_fid_list):
                #if the key doesn't exist in the cache for some reason then we'll revert back to
                #doing the matrix multiplication again.
                #Actually, this is slowing things down a good amount, let's just print a
                #descriptive error message if the key is missing 
                try:
                    outputMat[:, i] = prep_cache[0][(rho_key,prepFid.str)]
                except KeyError as err:
                    print('A (Rho, Circuit) pair is missing from the cache, all such pairs should be available is using the caching option.')
                    raise err                
                    #outputMat[:, i] = _np.dot(model.sim.product(prepFid), rho.to_dense())
            outputMatList.append(outputMat)
    
    else:
        for rho in model.preps.values():
            outputMat = _np.zeros([dimRho, numFid], float)
            for i, prepFid in enumerate(prep_fid_list):
                outputMat[:, i] = _np.dot(model.sim.product(prepFid), rho.to_dense())
            outputMatList.append(outputMat)    
    
    return outputMatList


def create_meas_mxs(model, meas_fid_list, meas_cache=None):
    """
    Make a list of matrices for the model measurement operations.

    Makes a list of matrices, where each matrix corresponds to a single
    measurement effect in the model, and the column of each matrix is the
    transpose of the measurement effect acting on a fiducial.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    meas_fid_list : list of Circuits
        List of fiducial circuits for constructing an informationally complete measurement.
       
    meas_cache : dictionary of effective measurement effects
        Dictionary of effective measurement effects cache used to accelerate the generation of the matrices
        used for score function evaluation. Entries are columns of the transpose of the effects. Default value is None.

    Returns
    -------
    list
        A list of matrices, each of shape `(dim, len(meas_fid_list))` where
        `dim` is the dimension of `model` (4 for a single qubit).  The length
        of this list is equal to the number of POVM effects in `model`.
    """

    dimE = model.dim
    numFid = len(meas_fid_list)
    outputMatList = []
     
    if meas_cache is not None:
        
        for povm_key, E_key in meas_cache[1]:
            outputMat = _np.zeros([dimE, numFid], float)
            for i, measFid in enumerate(meas_fid_list):
                #if the key doesn't exist in the cache for some reason then we'll revert back to
                #doing the matrix multiplication again.
                #Actually, this is slowing things down a good amount, let's just print a
                #descriptive error message if the key is missing 
                try:
                    outputMat[:, i] = meas_cache[0][(povm_key, E_key,measFid.str)] 
                except KeyError as err:
                    print('A (POVM, Effect, Circuit) pair is missing from the cache, all such pairs should be available is using the caching option.')
                    raise err
                    #outputMat[:, i] = _np.dot(E.to_dense(), model.sim.product(measFid))
            outputMatList.append(outputMat)
    
    else:
        for povm in model.povms.values():
            for E in povm.values():
                if isinstance(E, _ComplementPOVMEffect): continue  # complement is dependent on others
                outputMat = _np.zeros([dimE, numFid], float)
                for i, measFid in enumerate(meas_fid_list):
                    outputMat[:, i] = _np.dot(E.to_dense(), model.sim.product(measFid))
                outputMatList.append(outputMat)
            
    return outputMatList


def compute_composite_fiducial_score(model, fid_list, prep_or_meas, score_func='all',
                                     threshold=1e6, return_all=False, op_penalty=0.0,
                                     l1_penalty=0.0, gate_penalty=None, fid_cache= None):
    """
    Compute a composite score for a fiducial list.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    fid_list : list of Circuits
        List of fiducial circuits to test.

    prep_or_meas : string ("prep" or "meas")
        Are we testing preparation or measurement fiducials?

    score_func : str ('all' or 'worst'), optional (default is 'all')
        Sets the objective function for scoring a fiducial set.  If 'all',
        score is (number of fiducials) * sum(1/Eigenvalues of score matrix).
        If 'worst', score is (number of fiducials) * 1/min(Eigenvalues of score
        matrix).  Note:  Choosing 'worst' corresponds to trying to make the
        optimizer make the "worst" direction (the one we are least sensitive to
        in Hilbert-Schmidt space) as minimally bad as possible.  Choosing 'all'
        corresponds to trying to make the optimizer make us as sensitive as
        possible to all directions in Hilbert-Schmidt space.  (Also note-
        because we are using a simple integer program to choose fiducials, it
        is possible to get stuck in a local minimum, and choosing one or the
        other objective function can help avoid such minima in different
        circumstances.)

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the
        fiducial set is rejected as informationally incomplete.

    return_all : bool, optional (default is False)
        Whether the spectrum should be returned along with the score.

    op_penalty : float, optional (default is 0.0)
        Coefficient of a penalty linear in the total number of gates in all
        fiducials that is added to ``score.minor``.

    l1_penalty : float, optional (default is 0.0)
        Coefficient of a penalty linear in the number of fiducials that is
        added to ``score.minor``.
        
    gate_penalty : dict, optional
        A dictionary with keys given by individual gates and values corresponding
        to the penalty to add for each instance of that gate in the fiducial set.
        
    fid_cache : dict, optional (default is None)
        A dictionary of either effective state preparations or measurement effects
        used to accelerate the generation of the matrix used for scoring.
        It's assumed that the user will pass in the correct cache based on the type
        of fiducial set being created (if wrong a fall back will revert to redoing all the
        matrix multiplication again).

    Returns
    -------
    score : CompositeScore
        The score of the fiducials.
    spectrum : numpy.array, optional
        The eigenvalues of the square of the absolute value of the score
        matrix.
    """
    # dimRho = model.dim
    if prep_or_meas == 'prep':
        fidArrayList = create_prep_mxs(model, fid_list, fid_cache)
    elif prep_or_meas == 'meas':
        fidArrayList = create_meas_mxs(model, fid_list, fid_cache)
    else:
        raise ValueError('Invalid value "{}" for prep_or_meas (must be "prep" '
                         'or "meas")!'.format(prep_or_meas))

    numFids = len(fid_list)
    scoreMx = _np.concatenate(fidArrayList, axis=1)  # shape = (dimRho, nFiducials*nPrepsOrEffects)
    scoreSqMx = _np.dot(scoreMx, scoreMx.T)  # shape = (dimRho, dimRho)
    spectrum = _np.sort(_np.abs(_np.linalg.eigvalsh(scoreSqMx)))
    
    specLen = len(spectrum)
    N_nonzero = specLen- _np.count_nonzero(spectrum<10**-10) #HARDCODED Spectrum Threshold
    if N_nonzero==0:
        nonzero_score = _np.inf
    else:
        #The scoring function in list_score is meant to be generic, but for 
        #performance reasons I want to take advantage of the fact that I know
        #certain things have already been done to the spectrum, so I'm going to
        #inline the scoring here and leave list_score alone.
        
        #don't need to check for zeros since I already counted the number
        #of nonzero eigenvalues above and handled that case there
        if score_func == 'all':
            #no need to the absolute value since I did that above
            #Non-np sum and min are faster for small arrays/lists but slower for
            #large ones.
            nonzero_score = numFids*_np.sum(1. /spectrum[-N_nonzero:])
        elif score_func == 'worst':
            nonzero_score = numFids*(1. / _np.min(spectrum[-N_nonzero:]))
        else:
            raise ValueError("'%s' is not a valid value for score_func.  "
                             "Either 'all' or 'worst' must be specified!"
                             % score_func)
    
        #nonzero_score = numFids * _scoring.list_score(spectrum[-N_nonzero:], score_func)
        
#    nonzero_score = _np.inf
#    for N in range(1, specLen + 1):
#        print(spectrum[-N:])
#        score = numFids * _scoring.list_score(spectrum[-N:], score_func)
#        if score <= 0 or _np.isinf(score) or score > threshold:
#            break   # We've found a zero eigenvalue.
#        else:
#            nonzero_score = score
#            N_nonzero = N

#the implementation of the above scoring loop can be made much faster

    nonzero_score += l1_penalty * len(fid_list)

    nonzero_score += op_penalty * sum([len(fiducial) for fiducial in fid_list])
    
    
    #print('nonzero_score before gate penalties: ', nonzero_score)
    
    #add the gate penalties.
    if gate_penalty is not None:
        for gate, penalty_value in gate_penalty.items():
            #loop through each ckt in the fiducial list.
            for fiducial in fid_list:
                #alternative approach using the string 
                #representation of the ckt.
                num_gate_instances= fiducial.str.count(gate)
                nonzero_score+= num_gate_instances*penalty_value
                        
    #print('nonzero_score after gate penalties: ', nonzero_score)
        

    score = _scoring.CompositeScore(-N_nonzero, nonzero_score, N_nonzero)

    return (score, spectrum) if return_all else score


def test_fiducial_list(model, fid_list, prep_or_meas, score_func='all',
                       return_all=False, threshold=1e6, l1_penalty=0.0,
                       op_penalty=0.0, fid_cache=None):
    """
    Tests a prep or measure fiducial list for informational completeness.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    fid_list : list of Circuits
        List of fiducial circuits to test.

    prep_or_meas : string ("prep" or "meas")
        Are we testing preparation or measurement fiducials?

    score_func : str ('all' or 'worst'), optional (default is 'all')
        Sets the objective function for scoring a fiducial set.  If 'all',
        score is (number of fiducials) * sum(1/Eigenvalues of score matrix).
        If 'worst', score is (number of fiducials) * 1/min(Eigenvalues of score
        matrix).  Note:  Choosing 'worst' corresponds to trying to make the
        optimizer make the "worst" direction (the one we are least sensitive to
        in Hilbert-Schmidt space) as minimally bad as possible.  Choosing 'all'
        corresponds to trying to make the optimizer make us as sensitive as
        possible to all directions in Hilbert-Schmidt space.  (Also note-
        because we are using a simple integer program to choose fiducials, it
        is possible to get stuck in a local minimum, and choosing one or the
        other objective function can help avoid such minima in different
        circumstances.)

    return_all : bool, optional (default is False)
        If true, function returns reciprocals of eigenvalues of fiducial score
        matrix, and the score of the fiducial set as specified by score_func, in
        addition to a boolean specifying whether or not the fiducial set is
        informationally complete

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the
        fiducial set is rejected as informationally incomplete.

    l1_penalty : float, optional (default is 0.0)
        Coefficient of a penalty linear in the number of fiducials that is
        added to ``score.minor``.

    op_penalty : float, optional (default is 0.0)
        Coefficient of a penalty linear in the total number of gates in all
        fiducials that is added to ``score.minor``.
        
    fid_cache : dict, optional (default is None)    
        A dictionary of either effective state preparations or measurement effects
        used to accelerate the generation of the matrix used for scoring.
        It's assumed that the user will pass in the correct cache based on the type
        of fiducial set being created (if wrong a fall back will revert to redoing all the
        matrix multiplication again).

    Returns
    -------
    testResult : bool
        Whether or not the specified fiducial list is informationally complete
        for the provided model, to within the tolerance specified by
        threshold.
    spectrum : array, optional
        The number of fiducials times the reciprocal of the spectrum of the
        score matrix.  Only returned if return_all == True.
    score : float, optional
        The score for the fiducial set; only returned if return_all == True.
    """

    score, spectrum = compute_composite_fiducial_score(
        model, fid_list, prep_or_meas, score_func=score_func,
        threshold=threshold, return_all=True, l1_penalty=l1_penalty,
        op_penalty=op_penalty, fid_cache=fid_cache)

    if score.N < len(spectrum):
        testResult = False
    else:
        testResult = True

    return (testResult, spectrum, score) if return_all else testResult


def build_bitvec_mx(n, k):
    """
    Create an array of all length-`n` and Hamming weight `k` binary vectors.

    Parameters
    ----------
    n : int
        The length of each bit string.

    k : int
        The hamming weight of each bit string.

    Returns
    -------
    numpy.ndarray
        An array of shape `(binom(n,k), n)` whose rows are the sought binary vectors.
    """
    bitVecMx = _np.zeros([int(scipy.special.binom(n, k)), n])
    diff = n - k

    # Recursive function for populating a matrix of arbitrary size
    def build_mx(previous_bit_locs, i, counter):
        """Allows arbitrary nesting of for loops

        Parameters
        ----------
        previous_bit_locs : tuple
            current loop contents, ex:

            >>> for i in range(10):
            >>>    for j in range(10):
            >>>        (i, j)

        i : int
            Loop depth

        counter : int
            tracks which fields of mx have been already set

        Returns
        ----------
        counter : int
            for updating the counter one loop above the current one

        """
        if i == 0:
            bitVecMx[counter][list(previous_bit_locs)] = 1
            counter += 1
        else:
            subK = k - i
            # Recursive definition allowing arbitrary size
            last_bit_loc = previous_bit_locs[-1]  # More explicit?
            for bit_loc in range(1 + last_bit_loc, diff + subK + 1):
                current_bit_locs = previous_bit_locs + (bit_loc,)

                counter = build_mx(current_bit_locs, i - 1, counter)

        # An alternative to shared state
        return counter

    counter = 0
    for bit_loc_0 in range(diff + 1):
        counter = build_mx((bit_loc_0,), k - 1, counter)  # Do subK additional iterations

    return bitVecMx


def _find_fiducials_integer_slack(model, fid_list, prep_or_meas=None,
                                  initial_weights=None, score_func='all',
                                  max_iter=100, fixed_slack=None,
                                  slack_frac=None, return_all=False,
                                  force_empty=True, force_empty_score=1e100,
                                  fixed_num=None, threshold=1e6,
                                  # forceMinScore=1e100,
                                  verbosity=1):
    """
    Find a locally optimal subset of the fiducials in fid_list.

    Locally optimal here means that no single fiducial can be excluded without
    increasing the sum of the reciprocals of the singular values of the "score
    matrix" (the matrix whose columns are the fiducials acting on the
    preparation, or the transpose of the measurement acting on the fiducials),
    by more than a fixed or variable amount of "slack", as specified by
    fixed_slack or slack_frac.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    fid_list : list of Circuits
        List of all fiducials circuits to consider.

    prep_or_meas : {'prep', 'meas'}
        Whether preparation or measturement fiducials are being selected.

    initial_weights : list-like
        List or array of either booleans or (0 or 1) integers specifying which
        fiducials in fid_list comprise the initial fiduial set.  If None, then
        starting point includes all fiducials.

    score_func : str ('all' or 'worst'), optional (default is 'all')
        Sets the objective function for scoring a fiducial set.  If 'all',
        score is (number of fiducials) * sum(1/Eigenvalues of score matrix).
        If 'worst', score is (number of fiducials) * 1/min(Eigenvalues of score
        matrix).  Note:  Choosing 'worst' corresponds to trying to make the
        optimizer make the "worst" direction (the one we are least sensitive to
        in Hilbert-Schmidt space) as minimally bad as possible.  Choosing 'all'
        corresponds to trying to make the optimizer make us as sensitive as
        possible to all directions in Hilbert-Schmidt space.  (Also note-
        because we are using a simple integer program to choose fiducials, it
        is possible to get stuck in a local minimum, and choosing one or the
        other objective function can help avoid such minima in different
        circumstances.)

    max_iter : int, optional
        The maximum number of iterations before stopping.

    fixed_slack : float, optional
        If not None, a floating point number which specifies that excluding a
        fiducial is allowed to increase the fiducial set score additively by
        fixed_slack.  You must specify *either* fixed_slack or slack_frac.

    slack_frac : float, optional
        If not None, a floating point number which specifies that excluding a
        fiducial is allowed to increase the fiducial set score multiplicatively
        by (1+slack_frac).  You must specify *either* fixed_slack or slack_frac.

    return_all : bool, optional
        If True, return the final "weights" vector and score dictionary in
        addition to the optimal fiducial list (see below).

    force_empty : bool, optional (default is True)
        Whether or not to force all fiducial sets to contain the empty gate
        string as a fiducial.

        IMPORTANT:  This only works if the first element of fid_list is the
        empty circuit.

    force_empty_score : float, optional (default is 1e100)
        When force_empty is True, what score to assign any fiducial set that
        does not contain the empty circuit as a fiducial.

    fixed_num : int, optional
        Require the output list of fiducials to contain exactly `fixed_num` elements.

    threshold : float, optional (default is 1e6)
        Entire fiducial list is first scored before attempting to select
        fiducials; if score is above threshold, then fiducial selection will
        auto-fail.  If final fiducial set selected is above threshold, then
        fiducial selection will print a warning, but return selected set.

    verbosity : int, optional
        Integer >= 0 indicating the amount of detail to print.

    Returns
    -------
    fiducial_list : list
        A list of the selected (optimized) fiducial circuits.

    weights : list
        Only returned if `return_all=True`.  The internal weights
        for each candidate germ.

    score : dict
        Only returned if `return_all=True`.  The internal dictionary
        mapping weights (as a tuple) to scores.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    if not xor(fixed_slack, slack_frac):
        raise ValueError("One and only one of fixed_slack or slack_frac should "
                         "be specified!")

    initial_test = test_fiducial_list(model, fid_list, prep_or_meas,
                                      score_func=score_func, return_all=True,
                                      threshold=threshold)
    if initial_test[0]:
        printer.log("Complete initial fiducial set succeeds.", 1)
        printer.log("Now searching for best fiducial set.", 1)
    else:
        printer.warning("Complete initial fiducial set FAILS.")
        printer.warning("Aborting search.")
        return None

    #Initially allow adding to weight. -- maybe make this an argument??
    lessWeightOnly = False

    nFids = len(fid_list)

    dimRho = model.dim

    printer.log("Starting fiducial set optimization. Lower score is better.",
                1)

    scoreD = {}

    #fidLengths = _np.array( list(map(len,fid_list)), _np.int64)
    if prep_or_meas == 'prep':
        fidArrayList = create_prep_mxs(model, fid_list)
    elif prep_or_meas == 'meas':
        fidArrayList = create_meas_mxs(model, fid_list)
    else:
        raise ValueError('prep_or_meas must be specified!')  # pragma: no cover
        # unreachable given check within test_fiducial_list above
    numMxs = len(fidArrayList)

    def compute_score(wts, cache_score=True):
        """ objective function for optimization """
        score = None
        if force_empty and _np.count_nonzero(wts[:1]) != 1:
            score = force_empty_score
#        if forceMinNum and _np.count_nonzero(wts) < forceMinNum:
#            score = forceMinScore
        if score is None:
            numFids = _np.sum(wts)
            scoreMx = _np.zeros([dimRho, int(numFids) * int(numMxs)], float)
            colInd = 0
            wts = _np.array(wts)
            wtsLoc = _np.where(wts)[0]
            for fidArray in fidArrayList:
                scoreMx[:, colInd:colInd + int(numFids)] = fidArray[:, wtsLoc]
                colInd += int(numFids)
            scoreSqMx = _np.dot(scoreMx, scoreMx.T)
#            score = numFids * _np.sum(1./_np.linalg.eigvalsh(scoreSqMx))
            score = numFids * _scoring.list_score(
                _np.linalg.eigvalsh(scoreSqMx), score_func)
            if score <= 0 or _np.isinf(score):
                score = 1e10
        if cache_score:
            scoreD[tuple(wts)] = score
        return score

    if fixed_num is not None:
        if force_empty:
            hammingWeight = fixed_num - 1
            numBits = len(fid_list) - 1
        else:
            hammingWeight = fixed_num
            numBits = len(fid_list)
        numFidLists = scipy.special.binom(numBits, hammingWeight)
        printer.log("Output set is required to be of size%s" % fixed_num, 1)
        printer.log("Total number of fiducial sets to be checked is%s"
                    % numFidLists, 1)
        printer.warning("If this is very large, you may wish to abort.")
#        print "Num bits:", numBits
#        print "Num Fid Options:", hammingWeight
        # Now a non auxillary function:
        bitVecMat = build_bitvec_mx(numBits, hammingWeight)

        if force_empty:
            bitVecMat = _np.concatenate((_np.array([[1] * int(numFidLists)]).T,
                                         bitVecMat), axis=1)
        best_score = _np.inf
        # Explicitly declare best_weights, even if it will soon be replaced
        best_weights = []
        for weights in bitVecMat:
            temp_score = compute_score(weights, cache_score=True)
            # If scores are within machine precision, we want the fiducial set
            # that requires fewer total button operations.
            if abs(temp_score - best_score) < 1e-8:
                #                print "Within machine precision!"
                bestFidList = []
                for index, val in enumerate(best_weights):
                    if val == 1:
                        bestFidList.append(fid_list[index])
                tempFidList = []
                for index, val in enumerate(weights):
                    if val == 1:
                        tempFidList.append(fid_list[index])
                tempLen = sum(len(i) for i in tempFidList)
                bestLen = sum(len(i) for i in bestFidList)
#                print tempLen, bestLen
#                print temp_score, best_score
                if tempLen < bestLen:
                    best_score = temp_score
                    best_weights = weights
                    printer.log("Switching!", 1)
            elif temp_score < best_score:
                best_score = temp_score
                best_weights = weights

        goodFidList = []
        weights = best_weights
        for index, val in enumerate(weights):
            if val == 1:
                goodFidList.append(fid_list[index])

        if return_all:
            return goodFidList, weights, scoreD
        else:
            return goodFidList

    def _get_neighbors(bool_vec):
        """ Iterate over neighbors of `bool_vec` """
        for i in range(nFids):
            v = bool_vec.copy()
            v[i] = (v[i] + 1) % 2  # toggle v[i] btwn 0 and 1
            yield v

    if initial_weights is not None:
        weights = _np.array([1 if x else 0 for x in initial_weights])
    else:
        weights = _np.ones(nFids, _np.int64)  # default: start with all germs
        lessWeightOnly = True  # we're starting at the max-weight vector

    score = compute_score(weights)
    L1 = sum(weights)  # ~ L1 norm of weights

    with printer.progress_logging(1):

        for iIter in range(max_iter):
            scoreD_keys = scoreD.keys()  # list of weight tuples already computed

            printer.show_progress(iIter, max_iter,
                                  suffix="score=%g, nFids=%d" % (score, L1))

            bFoundBetterNeighbor = False
            for neighbor in _get_neighbors(weights):
                if tuple(neighbor) not in scoreD_keys:
                    neighborL1 = sum(neighbor)
                    neighborScore = compute_score(neighbor)
                else:
                    neighborL1 = sum(neighbor)
                    neighborScore = scoreD[tuple(neighbor)]

                # Move if we've found better position; if we've relaxed, we
                # only move when L1 is improved.
                if neighborScore <= score and (neighborL1 < L1
                                               or not lessWeightOnly):
                    weights, score, L1 = neighbor, neighborScore, neighborL1
                    bFoundBetterNeighbor = True
                    printer.log("Found better neighbor: nFids = %d score = %g"
                                % (L1, score), 3)

            if not bFoundBetterNeighbor:  # Time to relax our search.
                # from now on, don't allow increasing weight L1
                lessWeightOnly = True

                if fixed_slack:
                    # Note score is positive (for sum of 1/lambda)
                    slack = score + fixed_slack
                elif slack_frac:
                    slack = score * slack_frac
                assert slack > 0

                printer.log("No better neighbor. "
                            "Relaxing score w/slack: %g => %g"
                            % (score, score + slack), 2)
                # artificially increase score and see if any neighbor is better
                # now...
                score += slack

                for neighbor in _get_neighbors(weights):
                    if sum(neighbor) < L1 and scoreD[tuple(neighbor)] < score:
                        weights, score, L1 = (neighbor,
                                              scoreD[tuple(neighbor)],
                                              sum(neighbor))
                        bFoundBetterNeighbor = True
                        printer.log("Found better neighbor: nFids = %d "
                                    "score = %g" % (L1, score), 3)

                if not bFoundBetterNeighbor:  # Relaxing didn't help!
                    printer.log("Stationary point found!", 2)
                    break  # end main for loop

            printer.log("Moving to better neighbor", 2)
        else:
            printer.log("Hit max. iterations", 2)

    printer.log("score = %s" % score, 1)
    printer.log("weights = %s" % weights, 1)
    printer.log("L1(weights) = %s" % sum(weights), 1)

    goodFidList = []
    for index, val in enumerate(weights):
        if val == 1:
            goodFidList.append(fid_list[index])

    # final_test = test_fiducial_list(model, goodFidList, prep_or_meas,
    #                                 score_func=score_func, return_all=True,
    #                                 threshold=threshold)
    if initial_test[0]:
        printer.log("Final fiducial set succeeds.", 1)
    else:
        printer.log("WARNING: Final fiducial set FAILS.", 1)

    if return_all:
        return goodFidList, weights, scoreD
    else:
        return goodFidList


def _find_fiducials_grasp(model, fids_list, prep_or_meas, alpha,
                          iterations=5, score_func='all', op_penalty=0.0,
                          l1_penalty=0.0, gate_penalty=None, return_all=False,
                          force_empty=True, threshold=1e6, seed=None,
                          verbosity=0, fid_cache= None):
    """
    Use GRASP to find a high-performing set of fiducials.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    fids_list : list of Circuits
        List of fiducial circuits to test.

    prep_or_meas : string ("prep" or "meas")
        Are we testing preparation or measurement fiducials?

    alpha : float
        A number between 0 and 1 that roughly specifies a score threshold
        relative to the spread of scores that a germ must score better than in
        order to be included in the RCL. A value of 0 for `alpha` corresponds
        to a purely greedy algorithm (only the best-scoring element is
        included in the RCL), while a value of 1 for `alpha` will include all
        elements in the RCL.

    iterations : int, optional
        Number of GRASP iterations.

    score_func : str ('all' or 'worst'), optional (default is 'all')
        Sets the objective function for scoring a fiducial set.  If 'all',
        score is (number of fiducials) * sum(1/Eigenvalues of score matrix).
        If 'worst', score is (number of fiducials) * 1/min(Eigenvalues of score
        matrix).  Note:  Choosing 'worst' corresponds to trying to make the
        optimizer make the "worst" direction (the one we are least sensitive to
        in Hilbert-Schmidt space) as minimally bad as possible.  Choosing 'all'
        corresponds to trying to make the optimizer make us as sensitive as
        possible to all directions in Hilbert-Schmidt space.  (Also note-
        because we are using a simple integer program to choose fiducials, it
        is possible to get stuck in a local minimum, and choosing one or the
        other objective function can help avoid such minima in different
        circumstances.)

    op_penalty : float, optional (defailt is 0.0)
        Coefficient of a penalty linear in the total number of gates in all
        fiducials that is added to ``score.minor``.

    l1_penalty : float, optional (defailt is 0.0)
        Coefficient of a penalty linear in the number of fiducials that is
        added to ``score.minor``.
        
    gate_penalty : dict, optional
        A dictionary with keys given by individual gates and values corresponding
        to the penalty to add for each instance of that gate in the fiducial set.

    return_all : bool, optional (default is False)
        If true, function returns reciprocals of eigenvalues of fiducial score
        matrix, and the score of the fiducial set as specified by score_func, in
        addition to a boolean specifying whether or not the fiducial set is
        informationally complete

    force_empty : bool, optional
        When `True`, the empty circuit must be a member of the chosen set.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the
        fiducial set is rejected as informationally incomplete.

    seed : int, optional
        The seed value used for the random number generator.

    verbosity : int, optional
        How much detail to send to stdout.
        
    fid_cache : dict, optional (default is None)
        A dictionary of either effective state preparations or measurement effects
        used to accelerate the generation of the matrix used for scoring.
        It's assumed that the user will pass in the correct cache based on the type
        of fiducial set being created (if wrong a fall back will revert to redoing all the
        matrix multiplication again).
        

    Returns
    -------
    best_fiducials : list
        The best-scoring list of fiducial circuits.

    initial_fiducials : list of lists
        Only returned if `return_all=True`.  A list of the initial solution
        (a solution is a list of fiducial circuits) for each grasp iteration.

    local_solutions : list of lists
        Only returned if `return_all=True`.  A list of the best solution
        (a solution is a list of fiducial circuits) for each grasp iteration.
    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    rng = random.Random(seed)

    if prep_or_meas not in ['prep', 'meas']:
        raise ValueError("'{}' is an invalid value for prep_or_meas (must be "
                         "'prep' or 'meas')!".format(prep_or_meas))

    initial_test = test_fiducial_list(model, fids_list, prep_or_meas,
                                      score_func=score_func, return_all=False,
                                      threshold=threshold, fid_cache=fid_cache)
    if initial_test:
        printer.log("Complete initial fiducial set succeeds.", 1)
        printer.log("Now searching for best fiducial set.", 1)
    else:
        printer.warning("Complete initial fiducial set FAILS.")
        printer.warning("Aborting search.")
        return (None, None, None) if return_all else None

    initialWeights = _np.zeros(len(fids_list), dtype=_np.int64)
    if force_empty:
        fidsLens = [len(fiducial) for fiducial in fids_list]
        initialWeights[fidsLens.index(0)] = 1

    def _get_neighbors_fn(weights): return _grasp.neighboring_weight_vectors(
        weights, forced_weights=initialWeights)

    printer.log("Starting fiducial list optimization. Lower score is better.",
                1)

    # Dict of keyword arguments passed to compute_score_non_AC that don't
    # change from call to call
    compute_kwargs = {
        'model': model,
        'prep_or_meas': prep_or_meas,
        'score_func': score_func,
        'threshold': threshold,
        'op_penalty': op_penalty,
        'return_all': False,
        'l1_penalty': 0.0,
        'fid_cache': fid_cache,
        'gate_penalty' : gate_penalty
    }

    final_compute_kwargs = compute_kwargs.copy()
    final_compute_kwargs['l1_penalty'] = l1_penalty

    def score_fn(fid_list): return compute_composite_fiducial_score(
        fid_list=fid_list, **compute_kwargs)

    def final_score_fn(fid_list): return compute_composite_fiducial_score(
        fid_list=fid_list, **final_compute_kwargs)

    dimRho = model.dim
    feasibleThreshold = _scoring.CompositeScore(-dimRho, threshold, dimRho)

    def rcl_fn(x): return _scoring.filter_composite_rcl(x, alpha)

    initialSolns = []
    localSolns = []

    for iteration in range(iterations):
        # This loop is parallelizable (each iteration is independent of all
        # other iterations).
        printer.log('Starting iteration {} of {}.'.format(iteration + 1,
                                                          iterations), 1)
        success = False
        failCount = 0
        while not success and failCount < 10:
            try:
                iterSolns = _grasp.run_grasp_iteration(
                    elements=fids_list, greedy_score_fn=score_fn, rcl_fn=rcl_fn,
                    local_score_fn=score_fn,
                    get_neighbors_fn=_get_neighbors_fn,
                    feasible_threshold=feasibleThreshold,
                    initial_elements=initialWeights, rng=rng,
                    verbosity=verbosity)

                initialSolns.append(iterSolns[0])
                localSolns.append(iterSolns[1])

                success = True
                printer.log('Finished iteration {} of {}.'.format(
                    iteration + 1, iterations), 1)
            except Exception as e:
                failCount += 1
                if failCount == 10:
                    raise e
                else:
                    printer.warning(e)

    finalScores = _np.array([final_score_fn(localSoln)
                             for localSoln in localSolns])
    bestSoln = localSolns[_np.argmin(finalScores)]

    return (bestSoln, initialSolns, localSolns) if return_all else bestSoln
    
def _find_fiducials_greedy(model, fids_list, prep_or_meas, op_penalty=0.0,
                          l1_penalty=0.0, gate_penalty=None,
                          force_empty=True, threshold=1e6,
                          verbosity=0, fid_cache= None, evd_tol=1e-10):
    """
    Use greedy search to find a high-performing set of fiducials.

    Parameters
    ----------
    model : Model
        The model (associates operation matrices with operation labels).

    fids_list : list of Circuits
        List of fiducial circuits to test.

    prep_or_meas : string ("prep" or "meas")
        Are we testing preparation or measurement fiducials?

    op_penalty : float, optional (defailt is 0.0)
        Coefficient of a penalty linear in the total number of gates in all
        fiducials that is added to ``score.minor``.

    l1_penalty : float, optional (defailt is 0.0)
        Coefficient of a penalty linear in the number of fiducials that is
        added to ``score.minor``.
        
    gate_penalty : dict, optional
        A dictionary with keys given by individual gates and values corresponding
        to the penalty to add for each instance of that gate in the fiducial set.

    force_empty : bool, optional
        When `True`, the empty circuit must be a member of the chosen set.

    threshold : float, optional (default is 1e6)
        Specifies a maximum score for the score matrix, above which the
        fiducial set is rejected as informationally incomplete.

    verbosity : int, optional
        How much detail to send to stdout.
        
    fid_cache : dict, optional (default is None)
        A dictionary of either effective state preparations or measurement effects
        used to accelerate the generation of the matrix used for scoring.
        It's assumed that the user will pass in the correct cache based on the type
        of fiducial set being created (if wrong a fall back will revert to redoing all the
        matrix multiplication again).
        
    evd_tol : float, optional (default 1e-10)
        Cutoff value for truncating small eigenvalues when building low-rank updates
        caches.

    Returns
    -------
    best_fiducials : list
        The best-scoring list of fiducial circuits.

    """
    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity)

    if prep_or_meas not in ['prep', 'meas']:
        raise ValueError("'{}' is an invalid value for prep_or_meas (must be "
                         "'prep' or 'meas')!".format(prep_or_meas))

    initial_test = test_fiducial_list(model, fids_list, prep_or_meas,
                                      score_func='all', return_all=False,
                                      threshold=threshold, fid_cache=fid_cache)
    if initial_test:
        printer.log("Complete initial fiducial set succeeds.", 1)
        printer.log("Now searching for best fiducial set.", 1)
    else:
        printer.warning("Complete initial fiducial set FAILS.")
        printer.warning("Aborting search.")
        return None, None

    printer.log("Starting fiducial list optimization. Lower score is better.",
                1)
    
    dimRho = model.dim
    feasibleThreshold = _scoring.CompositeScore(-dimRho, threshold, dimRho)
    
    #Build a compact EVD cache for the fiducials
    fiducial_compact_EVD_cache= construct_compact_evd_cache(model, fids_list, prep_or_meas, fid_cache, eigenvalue_tolerance=evd_tol)
    
    #initialize a container for the indices of the candidate fiducial set
    #along with a bunch of other variables we'll need.
    best_fiducial_set= None
    acceptable_candidate_found= False
    current_best_score_mx= None
    current_best_score_gramian= None
    current_update_cache=None
    current_best_inv_trace= None
    current_best_composite_score= None
    
    #Continue iterating until acceptable solution is found.

    while not acceptable_candidate_found:
        #if the best fiducial set is not yet initialized
        #and we have force_empty==True, then we will
        #initialize it with the index of the empty fiducial.
        if (best_fiducial_set is None) and force_empty:
            fidsLens = [len(fiducial) for fiducial in fids_list]
            best_fiducial_set=[fids_list[fidsLens.index(0)]]
            #calculate the score matrix
            if prep_or_meas == 'prep':
                fidArrayList = create_prep_mxs(model, best_fiducial_set, fid_cache)
            elif prep_or_meas == 'meas':
                fidArrayList = create_meas_mxs(model, best_fiducial_set, fid_cache)
            current_best_score_mx= _np.concatenate(fidArrayList, axis=1)
            current_best_score_gramian = current_best_score_mx@current_best_score_mx.conj().T
            current_best_inv_trace = _np.trace(_np.linalg.pinv(current_best_score_gramian))
            #Don't add penalties for first fiducial when forcing it to be the empty germ.
            initial_rank= _np.linalg.matrix_rank(current_best_score_mx)
            current_best_composite_score= _scoring.CompositeScore(-initial_rank, current_best_inv_trace, initial_rank)
        #separate logic if force_empty is not true. In this case we'll loop through the list
        #of fiducial indices and add the one with the best score.
        elif best_fiducial_set is None:
            for fiducial in fids_list:
                #calculate the score matrix
                #if prep_or_meas == 'prep':
                #    fidArrayList = create_prep_mxs(model, [fiducial], fid_cache)
                #elif prep_or_meas == 'meas':
                #    fidArrayList = create_meas_mxs(model, [fiducial], fid_cache)
                #current_score_mx= _np.concatenate(fidArrayList, axis=1)
                #current_score_gramian = current_score_mx@current_score_mx.conj().T
                current_score_gramian= fiducial_compact_EVD_cache[fiducial]@fiducial_compact_EVD_cache[fiducial].T
                current_inv_trace = _np.trace(_np.linalg.pinv(current_score_gramian))
                #Don't add penalties for first fiducial when forcing it to be the empty germ.
                current_rank= _np.linalg.matrix_rank(current_score_mx)
                penalized_inv_trace= add_penalties_greedy(current_inv_trace, [fiducial], l1_penalty, op_penalty, gate_penalty)
                current_composite_score= _scoring.CompositeScore(-current_rank, penalized_inv_trace, current_rank)
                
                #if current_best_composite_score hasn't been initialized then we'll
                #make the first thing we calculate the best.
                if current_best_composite_score is None:
                    current_best_composite_score= current_composite_score
                    best_fiducial_set= [fiducial]
                    #also set the current best values for the other variables
                    current_best_score_mx= current_score_mx
                    current_best_score_gramian= current_score_gramian
                    current_best_inv_trace= current_inv_trace
                #otherwise compare the current composite score to the current
                #best and replace the best if the current one is better
                else:    
                    if current_composite_score < current_best_composite_score:
                        current_best_composite_score= current_composite_score
                        #swap out the best fiducial set:
                        best_fiducial_set= [fiducial]
                        #also set the current best values for the other variables
                        current_best_score_mx= current_score_mx
                        current_best_score_gramian= current_score_gramian
                        current_best_inv_trace= current_inv_trace
                    else:
                        continue
        #If best_fiducial_set has been initialized then we must not be on the first iteration.
        #construct and update cache and loop through the 
        else:
            #reinitialize current_best_update to None.
            current_best_update=None
            #Start by building an update cache:
            current_update_cache= construct_update_cache(current_best_score_gramian, evd_tol)
            #now loop through the fiducials and keep track of the best update.
            for fiducial in fids_list:
                printer.log('Testing fiducial: '+ str(fiducial), 4)
                #if the fiducial is already in best_fiducial_set then
                #skip this one:
                if fiducial in best_fiducial_set:
                    printer.log('Already in set, skipping.', 4)
                    continue
                #otherwise we will compute the updated score.
                current_inv_trace, updated_rank, _= minamide_style_inverse_trace(fiducial_compact_EVD_cache[fiducial], current_update_cache[0], current_update_cache[1],                                                                 current_update_cache[2], force_rank_increase=False)
                #Add penalty terms to the current_inv_trace before constructing the composite score
                #for this update:
                penalized_inv_trace = add_penalties_greedy(current_inv_trace, best_fiducial_set+[fiducial],
                                                           l1_penalty, op_penalty, gate_penalty)
                
                #check if the new composite score is less than the best seen and if so replace
                #the best one with the current one.
                current_composite_score= _scoring.CompositeScore(-updated_rank, penalized_inv_trace, updated_rank)
                printer.log('Composite score for this update: '+ str(current_composite_score), 4)
                if current_composite_score < current_best_composite_score:
                    current_best_composite_score= current_composite_score
                    current_best_update= fiducial
            #append the best update that was found to the best_fiducial_set list
            printer.log('Best fiducial update found: '+ str(current_best_update), 2)
            printer.log(str(current_best_composite_score), 2)
            best_fiducial_set.append(current_best_update)
            #also update the current best score gramian for the next update cache calculation
            current_best_score_gramian= current_best_score_gramian +\
                                        fiducial_compact_EVD_cache[current_best_update]@fiducial_compact_EVD_cache[current_best_update].conj().T
        #now we'll check to see if we have found and acceptable candidate solution
        #compare to the threshold score we set above.
        if current_best_composite_score < feasibleThreshold:
            acceptable_candidate_found= True
            printer.log('Acceptable candidate solution found.', 1)
            printer.log(str(current_best_composite_score),1)
            printer.log('Exiting greedy search.')
            
    #return the final list of fiducials as well as the final score
    final_score= current_best_composite_score
    
    return best_fiducial_set, final_score    
    
#helper function for building a compact evd cache:
def construct_compact_evd_cache(model, fids_list, prep_or_meas, fid_cache, eigenvalue_tolerance=1e-10):
    sqrteU_dict = {}
    
    for fiducial in fids_list:
        if prep_or_meas == 'prep':
            fidArrayList = create_prep_mxs(model, [fiducial], fid_cache)
        elif prep_or_meas == 'meas':
            fidArrayList = create_meas_mxs(model, [fiducial], fid_cache)
        
        fid_mat= _np.concatenate(fidArrayList, axis=1)
        fid_mat_gramian = fid_mat@fid_mat.conj().T
        
        e, U = compact_EVD(fid_mat_gramian, eigenvalue_tolerance)
        #if not _np.any(e) or not _np.any(U):
        #    import ipdb
        #    ipdb.set_trace()
        #print('eigenvalues: ', e)
        #print('U: ', U)
        sqrteU_dict[fiducial]= U@_np.diag(_np.sqrt(e))  
        
        #check reconstruction:
        #print('Norm to reconstruction: ', _np.linalg.norm(fid_mat_gramian- (U@_np.diag(_np.sqrt(e)))@(U@_np.diag(_np.sqrt(e))).conj().T))
    return sqrteU_dict        
    
def add_penalties_greedy(unpenalized_score, fid_list, l1_penalty=0, op_penalty=0, gate_penalty=None):
    """
    Parameters
    ----------
    unpenalized_score : float 
        The score function value to add penalty terms to.
    
    l1_penalty : float, optional (default 0.0)
        A penalty term associated with the number of fiducials in a candidate list.
    
    op_penalty : float, optional (default 0.0)
        A penalty term associated with the total number of gate operations
        used in aggregate across all of the fiducials in a candidate set.
    
    gate_penalty : dict, optional (default None)
        A dictionary with keys given by gate labels
        and values corresponding to a penalty term for that gate.
        This penalty is added for each instance of a gate in the dictionary
        across all fiducials in a candidate list of fiducials.
    """
    penalized_score=unpenalized_score
    
    #add l1_penalty
    penalized_score += l1_penalty * len(fid_list)

    #add op_penalty
    penalized_score += op_penalty * sum([len(fiducial) for fiducial in fid_list])
   
    #add the gate penalties.
    if gate_penalty is not None:
        for gate, penalty_value in gate_penalty.items():
            #loop through each ckt in the fiducial list.
            for fiducial in fid_list:
                #alternative approach using the string 
                #representation of the ckt.
                num_gate_instances= fiducial.str.count(gate)
                penalized_score+= num_gate_instances*penalty_value
    
    return penalized_score