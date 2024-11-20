"""
Tools for the propagation of error generators through circuits.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import stim
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.modelmembers.operations import LindbladErrorgen as _LinbladErrorgen
from numpy import conjugate
from functools import reduce

def errgen_coeff_label_to_stim_pauli_strs(err_gen_coeff_label, num_qubits):
    """
    Converts an input `GlobalElementaryErrorgenLabel` to a tuple of stim.PauliString
    objects, padded with an appropriate number of identities.

    Parameters
    ----------
    err_gen_coeff_label : `GlobalElementaryErrorgenLabel` or `LocalElementaryErrorgenLabel`
        The error generator coefficient label to construct the tuple of pauli
        strings for.

    num_qubits : int
        Number of total qubits to use for the Pauli strings. Used to determine
        the number of identities added when padding.

    Returns
    -------
    tuple of stim.PauliString
        A tuple of either length 1 (for H and S) or length 2 (for C and A)
        whose entries are stim.PauliString representations of the indices for the
        input error generator label, padded with an appropriate number of identities
        given the support of the error generator label.

    """

    if isinstance(err_gen_coeff_label, _GEEL):
        #the coefficient label is a tuple with 3 elements. 
        #The first element is the error generator type.
        #the second element is a tuple of paulis either of length 1 or 2 depending on the error gen type.
        #the third element is a tuple of subsystem labels.
        errorgen_typ = err_gen_coeff_label.errorgen_type
        pauli_lbls = err_gen_coeff_label.basis_element_labels
        sslbls = err_gen_coeff_label.support

        #double check that the number of qubits specified is greater than or equal to the length of the
        #basis element labels.
        #assert len(pauli_lbls) >= num_qubits, 'Specified `num_qubits` is less than the length of the basis element labels.'

        if errorgen_typ == 'H' or errorgen_typ == 'S':
            pauli_string = num_qubits*['I']
            pauli_lbl = pauli_lbls[0]
            for i, sslbl in enumerate(sslbls):
                pauli_string[sslbl] = pauli_lbl[i]
            pauli_string = stim.PauliString(''.join(pauli_string))
            return (pauli_string,)
        elif errorgen_typ == 'C' or errorgen_typ == 'A':
            pauli_strings = []
            for pauli_lbl in pauli_lbls: #iterate through both pauli labels
                pauli_string = num_qubits*['I']
                for i, sslbl in enumerate(sslbls):
                    pauli_string[sslbl] = pauli_lbl[i]
                pauli_strings.append(stim.PauliString(''.join(pauli_string)))
            return tuple(pauli_strings)
        else:
            raise ValueError(f'Unsupported error generator type {errorgen_typ}')
    elif isinstance(err_gen_coeff_label, _LEEL):
        return tuple([stim.PauliString(bel) for bel in  err_gen_coeff_label.basis_element_labels])

    else:
        raise ValueError('Only `GlobalElementaryErrorgenLabel and LocalElementaryErrorgenLabel is currently supported.')

def bch_approximation(errgen_layer_1, errgen_layer_2, bch_order=1, truncation_threshold=1e-14):
    """
    Apply the BCH approximation at the given order to combine the input dictionaries
    of  error generator rates.

    Parameters
    ----------
    errgen_layer_1 : dict
        Dictionary of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    errgen_layer_2 : list of dicts
        See errgen_layer_1.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.

    Returns
    -------
    combined_errgen_layer : dict
        A dictionary with the same general structure as `errgen_layer_1` and `errgen_layer_2`, but with the
        rates combined according to the selected order of the BCH approximation.

    """
    new_errorgen_layer=[]
    for curr_order in range(0, bch_order):
        #add first order terms into new layer
        if curr_order == 0:
            #Get a combined set of error generator coefficient labels for these two
            #dictionaries.
            current_combined_coeff_lbls = set(errgen_layer_1.keys()) | set(errgen_layer_2.keys())
            
            first_order_dict = dict()
            #loop through the combined set of coefficient labels and add them to the new dictionary for the current BCH
            #approximation order. If present in both we sum the rates.
            for coeff_lbl in current_combined_coeff_lbls:
                #only add to the first order dictionary if the coefficient exceeds the truncation threshold.
                first_order_rate = errgen_layer_1.get(coeff_lbl, 0) + errgen_layer_2.get(coeff_lbl, 0)
                if abs(first_order_rate) > truncation_threshold:
                    first_order_dict[coeff_lbl] = first_order_rate
            
            #allow short circuiting to avoid an expensive bunch of recombination logic when only using first order BCH
            #which will likely be a common use case.
            if bch_order==1:
                return first_order_dict
            new_errorgen_layer.append(first_order_dict)
        
        #second order BCH terms.
        # (1/2)*[X,Y]
        elif curr_order == 1:
            #calculate the pairwise commutators between each of the error generators in current_errgen_dict_1 and
            #current_errgen_dict_2.
            commuted_errgen_list = []
            for error1 in errgen_layer_1.keys():
                for error2 in errgen_layer_2.keys():
                    #get the list of error generator labels
                    weight = .5*errgen_layer_1[error1]*errgen_layer_2[error2]
                    #I *think* you can pick up at most around a factor of 8 from the commutator
                    #itself. Someone should validate that. Set this conservatively, but also
                    #avoid computing commutators which will be effectively zero.
                    if abs(weight) < 10*truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight= weight)
                    commuted_errgen_list.extend(commuted_errgen_sublist)
            #print(f'{commuted_errgen_list=}')   
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            second_order_comm_dict = {error_tuple[0]: 0 for error_tuple in commuted_errgen_list}

            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list:
                second_order_comm_dict[error_tuple[0]] += error_tuple[1]
            
            #truncate any terms which are below the truncation threshold following
            #aggregation.
            second_order_comm_dict = {key: val for key, val in second_order_comm_dict.items() if abs(val)>truncation_threshold}

            new_errorgen_layer.append(second_order_comm_dict)

        #third order BCH terms
        # (1/12)*([X,[X,Y]] - [Y,[X,Y]])
        #TODO: Can make this more efficient by using linearity of commutators
        elif curr_order == 2:
            #we've already calculated (1/2)*[X,Y] in the previous order, so reuse this result.
            #two different lists for the two different commutators so that we can more easily reuse
            #this at higher order if needed.
            commuted_errgen_list_1 = []
            commuted_errgen_list_2 = []
            for error1a, error1b in zip(errgen_layer_1.keys(), errgen_layer_2.keys()):
                for error2 in second_order_comm_dict:
                    second_order_comm_rate = second_order_comm_dict[error2]
                    #I *think* you can pick up at most around a factor of 8 from the commutator
                    #itself. Someone should validate that. Set this conservatively, but also
                    #avoid computing commutators which will be effectively zero.
                    #only need a factor of 1/6 because new_errorgen_layer[1] is 1/2 the commutator 
                    weighta = (1/6)*errgen_layer_1[error1a]*second_order_comm_rate

                    if not abs(weighta) < truncation_threshold:
                        commuted_errgen_sublist = error_generator_commutator(error1a, error2, 
                                                                             weight=weighta)
                        commuted_errgen_list_1.extend(commuted_errgen_sublist)
                    
                    #only need a factor of -1/6 because new_errorgen_layer[1] is 1/2 the commutator 
                    weightb = -(1/6)*errgen_layer_2[error1b]*second_order_comm_rate
                    if not abs(weightb) < truncation_threshold:                    
                        commuted_errgen_sublist = error_generator_commutator(error1b, error2, 
                                                                             weight=weightb)
                        commuted_errgen_list_2.extend(commuted_errgen_sublist)   

            #turn the two new commuted error generator lists into dictionaries.
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            third_order_comm_dict_1 = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_1}
            third_order_comm_dict_2 = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_2}
            
            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list_1:
                third_order_comm_dict_1[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_2:
                third_order_comm_dict_2[error_tuple[0]] += error_tuple[1]
            
            #finally sum these two dictionaries, keeping only terms which are greater than the threshold.
            third_order_comm_dict = dict()
            for lbl in set(third_order_comm_dict_1) | set(third_order_comm_dict_2):
                third_order_rate = third_order_comm_dict_1.get(lbl, 0) + third_order_comm_dict_2.get(lbl, 0)
                if abs(third_order_rate) > truncation_threshold:
                    third_order_comm_dict[lbl] = third_order_rate
            #print(f'{third_order_comm_dict=}')
            new_errorgen_layer.append(third_order_comm_dict)
                         
        #fourth order BCH terms
        # -(1/24)*[Y,[X,[X,Y]]]
        elif curr_order == 3:
            #we've already calculated (1/12)*[X,[X,Y]] so reuse this result.
            #this is stored in third_order_comm_dict_1
            commuted_errgen_list = []
            for error1 in errgen_layer_2.keys():
                for error2 in third_order_comm_dict_1.keys():
                    #I *think* you can pick up at most around a factor of 8 from the commutator
                    #itself. Someone should validate that. Set this conservatively, but also
                    #avoid computing commutators which will be effectively zero.
                    #only need a factor of -1/2 because third_order_comm_dict_1 is 1/12 the nested commutator
                    weight = -.5*errgen_layer_2[error1]*third_order_comm_dict_1[error2]
                    if abs(weight) < truncation_threshold:
                        #print('continuing')
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight)
                    commuted_errgen_list.extend(commuted_errgen_sublist)
            
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            fourth_order_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list}

            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list:
                fourth_order_comm_dict[error_tuple[0]] += error_tuple[1]

            #drop any terms below the truncation threshold after aggregation
            #print(f'{fourth_order_comm_dict=}')
            fourth_order_comm_dict = {key: val for key, val in fourth_order_comm_dict.items() if abs(val)>truncation_threshold}
            new_errorgen_layer.append(fourth_order_comm_dict)
            #print(f'{fourth_order_comm_dict=}')
     
        else:
            raise NotImplementedError("Higher orders beyond fourth order are not implemented yet.")

    #Finally accumulate all of the dictionaries in new_errorgen_layer into a single one, summing overlapping terms.   
    errorgen_labels_by_order = [set(order_dict) for order_dict in new_errorgen_layer]
    complete_errorgen_labels = reduce(lambda a, b: a|b, errorgen_labels_by_order)

    #initialize a dictionary with requisite keys
    new_errorgen_layer_dict = {lbl: 0 for lbl in complete_errorgen_labels}

    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] += rate

    #Future: Possibly do one last truncation pass in case any of the different order cancel out when aggregated?

    return new_errorgen_layer_dict


def error_generator_commutator(errorgen_1, errorgen_2, flip_weight=False, weight=1.0):
    """
    Returns the commutator of two error generators. I.e. [errorgen_1, errorgen_2].
    
    Parameters
    ----------
    errorgen1 : `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalStimErrorgenLabel`
        Second error generator

    flip_weight : bool, optional (default False)
        If True flip the sign of the input value of weight kwarg.
    
    weight : float, optional (default 1.0)
        An optional weighting value to apply to the value of the commutator.

    Returns
    -------
    list of `LocalStimErrorgenLabel`s corresponding to the commutator of the two input error generators,
    weighted by the specified value of `weight`.
    """
    
    errorGens=[]
    
    if flip_weight:
        w= -weight
    else:
        w = weight

    errorgen_1_type = errorgen_1.errorgen_type
    errorgen_2_type = errorgen_2.errorgen_type

    #The first basis element label is always well defined, 
    #the second we'll define only of the error generator is C or A type.
    errorgen_1_bel_0 = errorgen_1.basis_element_labels[0] 
    errorgen_2_bel_0 = errorgen_2.basis_element_labels[0] 
    
    if errorgen_1_type == 'C' or errorgen_1_type == 'A':
        errorgen_1_bel_1 = errorgen_1.basis_element_labels[1]
    if errorgen_2_type == 'C' or errorgen_2_type == 'A':
        errorgen_2_bel_1 = errorgen_2.basis_element_labels[1]

    #create the identity stim.PauliString for later comparisons.
    identity = stim.PauliString('I'*len(errorgen_1_bel_0))
        
    if errorgen_1_type=='H' and errorgen_2_type=='H':
        ptup = com(errorgen_1_bel_0 , errorgen_2_bel_0)
        if ptup is not None:
            errorGens.append((_LSE('H', [ptup[1]]), -1j*w *ptup[0]))
        
    elif errorgen_1_type=='H' and errorgen_2_type=='S':
        ptup = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        if ptup is not None:
            if errorgen_2_bel_0 == ptup[1]:
                errorGens.append(( _LSE('S', [errorgen_2_bel_0]), 2*1j*w*ptup[0]))
            else:
                new_bels =  [errorgen_2_bel_0, ptup[1]] if stim_pauli_string_less_than(errorgen_2_bel_0, ptup[1])\
                            else [ptup[1], errorgen_2_bel_0]
                errorGens.append(( _LSE('C', new_bels), 1j*w*ptup[0]))

    elif errorgen_1_type=='S' and errorgen_2_type=='H':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
          
    elif errorgen_1_type=='H' and errorgen_2_type=='C':
        ptup1 = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        ptup2 = com(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1 is not None:
            if ptup1[1] == errorgen_2_bel_1:
                errorGens.append((_LSE('S', [errorgen_2_bel_1]), 2*1j*w*ptup1[0]))
            else:
                new_bels =  [ptup1[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup1[1], errorgen_2_bel_1)\
                            else [errorgen_2_bel_1, ptup1[1]]
                errorGens.append((_LSE('C', new_bels), 1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] == errorgen_2_bel_0:
                errorGens.append(( _LSE('S', [errorgen_2_bel_0]), 2*1j*w*ptup2[0]))
            else:
                new_bels =  [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0)\
                            else [errorgen_2_bel_0, ptup2[1]]
                errorGens.append((_LSE('C', new_bels), 1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='C' and errorgen_2_type=='H':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
    
    elif errorgen_1_type=='H' and errorgen_2_type=='A':
        ptup1 = com(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = com(errorgen_1_bel_0 , errorgen_2_bel_1)
        if ptup1 is not None:
            if ptup1[1] != errorgen_2_bel_1:
                if stim_pauli_string_less_than(ptup1[1], errorgen_2_bel_1):
                    errorGens.append((_LSE('A', [ptup1[1], errorgen_2_bel_1]), -1j*w*ptup1[0]))
                else:
                    errorGens.append((_LSE('A', [errorgen_2_bel_1, ptup1[1]]), 1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] != errorgen_2_bel_0:
                if stim_pauli_string_less_than(errorgen_2_bel_0, ptup2[1]):
                    errorGens.append((_LSE('A', [errorgen_2_bel_0, ptup2[1]]), -1j*w*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_2_bel_0]), 1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='A' and errorgen_2_type=='H':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type=='S' and errorgen_2_type=='S':
        #Commutator of S with S is zero.
        pass
                         
    elif errorgen_1_type=='S' and errorgen_2_type=='C':
        ptup1 = product(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = product(ptup1[1], errorgen_1_bel_0)
            #it shouldn't be possible for ptup2[1] to equal errorgen_1_bel_0,
            #as that would imply that errorgen_1_bel_0 was the identity.
            if ptup2[1] == identity:
                errorGens.append((_LSE('H', [errorgen_1_bel_0]), -1j*.5*w*ptup1[0]*ptup2[0]))
            else:
                if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]) , -1j*.5*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]) , 1j*.5*w*ptup1[0]*ptup2[0]))

            #ptup3 is just the product from ptup2 in reverse, so this can be done
            #more efficiently, but I'm not going to do that at present...
            ptup3 = product(errorgen_1_bel_0, ptup1[1])
            if ptup3[1] == identity:
                errorGens.append((_LSE('H', [errorgen_1_bel_0]), 1j*.5*w*ptup1[0]*ptup3[0]) )
            else:
                if stim_pauli_string_less_than(errorgen_1_bel_0, ptup3[1]):
                    errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup3[1]]) , -1j*.5*w*ptup1[0]*ptup3[0]))
                else:
                    errorGens.append((_LSE('A', [ptup3[1], errorgen_1_bel_0]) , 1j*.5*w*ptup1[0]*ptup3[0]))
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'S':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type == 'S' and errorgen_2_type == 'A':
        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                #it shouldn't be possible for errorgen_1_bel_0 to be equal to ptup2,
                #since that would imply 
                #com(errorgen_1_bel_0,com(errorgen_2_bel_0, errorgen_2_bel_1)) == errorgen_1_bel_0
                #Which I don't think is possible when these come from valid error genator indices.
                #errorgen_1_bel_0 can't be the identity,
                #And com(errorgen_1_bel_0,com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be by the same
                #argument that it can't be errorgen_1_bel_0
                if stim_pauli_string_less_than(errorgen_1_bel_0, ptup2[1]):
                    errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))
                            
    elif errorgen_1_type == 'A' and errorgen_2_type == 'S':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'C':
        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity: 
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1,errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1,errorgen_1_bel_0)                 
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))        
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
        
        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity,
                    #And com(errorgen_2_bel_0, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_2_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_2_bel_1, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #And com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_2_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_2_bel_0, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_0)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_0) can't be either
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_1_bel_1, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_1)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_1) can't be either
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
            if ptup2 is not None:
                ptup3 = com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #It shouldn't be possible for ptup3 to be the identity given valid error generator indices.
                    errorGens.append((_LSE('H', [ptup3[1]]), .25*1j*w*ptup1[0]*ptup2[0]*ptup3[0]))

    elif errorgen_1_type == 'C' and errorgen_2_type == 'A':
        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorGens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_1]), .5*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_1_bel_1, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                        errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))
                    else:
                        errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    new_bels = [ptup2[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1) else [errorgen_2_bel_1, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), .5*1j*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_1, don't need to check that errorgen_2_bel_1 isn't identity.
                    errorGens.append((_LSE('S', [errorgen_2_bel_1]), 1j*w*ptup1[0]*ptup2[0]))


        ptup1 = acom(errorgen_1_bel_0,errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    new_bels = [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0) else [errorgen_2_bel_0, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), -.5*1j*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_0, don't need to check that errorgen_2_bel_0 isn't identity.
                    errorGens.append((_LSE('S', [errorgen_2_bel_0]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
            if ptup2 is not None:
                ptup3= com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #it shouldn't be possible for ptup3 to be identity given valid error generator
                    #indices.
                    errorGens.append((_LSE('H', [ptup3[1]]), -.25*w*ptup1[0]*ptup2[0]*ptup3[0]))
    
    elif errorgen_1_type == 'A' and errorgen_2_type == 'C':
        errorGens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'A' and errorgen_2_type == 'A':
        ptup1 = product(errorgen_2_bel_1, errorgen_1_bel_1)
        ptup2 = product(errorgen_1_bel_0, errorgen_2_bel_0)

        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorGens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_1_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0) else [errorgen_1_bel_0, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), .5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_1_bel_0
                    errorGens.append((_LSE('S', [errorgen_1_bel_0]), w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_1_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1) else [errorgen_1_bel_1, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), -.5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_1_bel_1
                    errorGens.append((_LSE('S', [errorgen_1_bel_1]), -1*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_0, com(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1) else [errorgen_2_bel_1, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), .5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_1
                    errorGens.append((_LSE('S', [errorgen_2_bel_1]), w*ptup1[0]*ptup2[0]))


        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, com(errorgen_1_bel_0,errorgen_1_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0) else [errorgen_2_bel_0, ptup2[1]]
                    errorGens.append((_LSE('C', new_bels), -.5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_0
                    errorGens.append((_LSE('S', [errorgen_2_bel_0]), -1*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, errorgen_1_bel_1)
            if ptup2 is not None:
                ptup3 = com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #it shouldn't be possible for ptup3 to be identity given valid error generator
                    #indices.
                    errorGens.append((_LSE('H', [ptup3[1]]), .25*1j*w*ptup1[0]*ptup2[0]*ptup3[0]))
           
    return errorGens

def com(P1, P2):
    #P1 and P2 either commute or anticommute.
    if P1.commutes(P2):
        P3 = 0
        return None
    else:
        P3 = P1*P2
        return (P3.sign*2, P3 / P3.sign)
    #return (sign(P3) * 2 if P1 and P2 anticommute, 0 o.w.,
    #        unsigned P3)
             
def acom(P1, P2):
    #P1 and P2 either commute or anticommute.
    if P1.commutes(P2):
        P3 = P1*P2
        return (P3.sign*2, P3 / P3.sign)
    else:
        return  None
    
    #return (sign(P3) * 2 if P1 and P2 commute, 0 o.w.,
    #        unsigned P3)

def product(P1, P2):
    P3 = P1*P2
    return (P3.sign, P3 / P3.sign)
    #return (sign(P3),
    #        unsigned P3)

def stim_pauli_string_less_than(pauli1, pauli2):
    """
    Returns true if pauli1 is less than pauli lexicographically.

    Parameters
    ----------
    pauli1, pauli2 : stim.PauliString
        Paulis to compare.
    """

    #remove the signs.
    unsigned_pauli1 = pauli1/pauli1.sign
    unsigned_pauli2 = pauli2/pauli2.sign

    unsigned_pauli1_str = str(unsigned_pauli1)[1:].replace('_', 'I')
    unsigned_pauli2_str = str(unsigned_pauli2)[1:].replace('_', 'I')
    
    return unsigned_pauli1_str < unsigned_pauli2_str

