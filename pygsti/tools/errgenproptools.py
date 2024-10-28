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
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.modelmembers.operations import LindbladErrorgen as _LinbladErrorgen
from numpy import conjugate

def errgen_coeff_label_to_stim_pauli_strs(err_gen_coeff_label, num_qubits):
    """
    Converts an input `GlobalElementaryErrorgenLabel` to a tuple of stim.PauliString
    objects, padded with an appropriate number of identities.

    Parameters
    ----------
    err_gen_coeff_label : `GlobalElementaryErrorgenLabel`
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
    assert isinstance(err_gen_coeff_label, _GEEL), 'Only `GlobalElementaryErrorgenLabel is currently supported.'

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
    

def bch_approximation(errgen_layer_1, errgen_layer_2, bch_order=1):
    """
    Apply the BCH approximation at the given order to combine the input dictionaries
    of  error generator rates.

    Parameters
    ----------
    errgen_layer_1 : list of dicts
        Each lists contains dictionaries of the error generator coefficients and rates for a circuit layer. 
        Each dictionary corresponds to a different order of the BCH approximation.
        The error generator coefficients are represented using LocalStimErrorgenLabel.
    
    errgen_layer_2 : list of dicts
        See errgen_layer_1.

    Returns
    -------
    combined_errgen_layer : list of dicts?
        A list with the same general structure as `errgen_layer_1` and `errgen_layer_2`, but with the
        rates combined according to the selected order of the BCH approximation.

    """
    if bch_order != 1:
        msg = 'The implementation of the 2nd order BCH approx is still under development. For now only 1st order is supported.'
        raise NotImplementedError(msg)

    new_errorgen_layer=[]
    for curr_order in range(0,bch_order):
        working_order_dict = dict()
        #add first order terms into new layer
        if curr_order == 0:
            #get the dictionaries of error generator coefficient labels and rates
            #for the current working BCH order.
            current_errgen_dict_1 = errgen_layer_1[curr_order]
            current_errgen_dict_2 = errgen_layer_2[curr_order]
            #Get a combined set of error generator coefficient labels for these two
            #dictionaries.
            current_combined_coeff_lbls = set(current_errgen_dict_1.keys()) | set(current_errgen_dict_2.keys())
            
            #loop through the combined set of coefficient labels and add them to the new dictionary for the current BCH
            #approximation order. If present in both we sum the rates.
            for coeff_lbl in current_combined_coeff_lbls:
                working_order_dict[coeff_lbl] = current_errgen_dict_1.get(coeff_lbl, 0) + current_errgen_dict_2.get(coeff_lbl, 0) 
            new_errorgen_layer.append(working_order_dict)
        #second order BCH terms.
        elif curr_order == 1:
            current_errgen_dict_1 = errgen_layer_1[curr_order-1]
            current_errgen_dict_2 = errgen_layer_2[curr_order-1]
            #calculate the pairwise commutators between each of the error generators in current_errgen_dict_1 and
            #current_errgen_dict_2.
            for error1 in current_errgen_dict_1.keys():
                for error2 in current_errgen_dict_2.keys():
                    #get the list of error generator labels 
                    commuted_errgen_list = commute_error_generators(error1, error2, 
                                                                    weight=1/2*current_errgen_dict_1[error1]*current_errgen_dict_1[error2])
                    print(commuted_errgen_list)
                    #Add all of these error generators to the working dictionary of updated error generators and weights.
                    #There may be duplicates, which should be summed together.
                    for error_tuple in commuted_errgen_list:
                        working_order_dict[error_tuple[0]]=error_tuple[1]
            
            
            if len(errgen_layer_1)==2:
                for error_key in errgen_layer_1[1]:
                    working_order_dict[error_key]=errgen_layer_1[1][error_key]
            if len(errgen_layer_2)==2:
                for error_key in errgen_layer_2[1]:
                    working_order_dict[error_key]=errgen_layer_2[1][error_key]
            new_errorgen_layer.append(working_order_dict)

        else:
            raise ValueError("Higher Orders are not Implemented Yet")
    return new_errorgen_layer


def commute_error_generators(errorgen_1, errorgen_2, flip_weight=False, weight=1.0):
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
                errorGens.append(( _LSE('S', [errorgen_2_bel_0]), 1j*w*ptup[0]))
            else:
                errorGens.append(( _LSE('C', [errorgen_2_bel_0, ptup[1]]), 1j*w*ptup[0]))
         
    elif errorgen_1_type=='S' and errorgen_2_type=='H':
        ptup = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        if ptup is not None:
            if errorgen_2_bel_0 == ptup[1]:
                errorGens.append(( _LSE('S', [errorgen_2_bel_0]), -1j*w*ptup[0]))
            else:
                errorGens.append(( _LSE('C', [errorgen_2_bel_0, ptup[1]]), -1j*w*ptup[0]))

          
    elif errorgen_1_type=='H' and errorgen_2_type=='C':
        ptup1 = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        ptup2 = com(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1 is not None:
            if ptup1[1] == errorgen_2_bel_1:
                errorGens.append((_LSE('S', [errorgen_2_bel_1]), 1j*w*ptup1[0]))
            else:
                errorGens.append((_LSE('C', [ptup1[1], errorgen_2_bel_1]), 1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] == errorgen_2_bel_0:
                errorGens.append(( _LSE('S', [errorgen_2_bel_0]), 1j*w*ptup2[0]))
            else:
                errorGens.append((_LSE('C', [ptup2[1], errorgen_2_bel_0]), 1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='C' and errorgen_2_type=='H':
        errorGens = commute_error_generators(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
    
    elif errorgen_1_type=='H' and errorgen_2_type=='A':
        ptup1 = com(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = com(errorgen_1_bel_0 , errorgen_2_bel_1)
        if ptup1 is not None:
            if ptup1[1] != errorgen_2_bel_1:
                errorGens.append((_LSE('A', [ptup1[1], errorgen_2_bel_1]), -1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] != errorgen_2_bel_0:
                errorGens.append((_LSE('A', [errorgen_2_bel_0, ptup2[1]]), -1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='A' and errorgen_2_type=='H':
        errorGens = commute_error_generators(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type=='S' and errorgen_2_type=='S':
        #Commutator of S with S is zero.
        pass
                         
    elif errorgen_1_type=='S' and errorgen_2_type=='C':
        ptup1 = product(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append(( _LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]) )
            elif ptup1[1] == identity:
                errorGens.append(( _LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]) )
            else: #ptup2[1] == identity
                errorGens.append(( _LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]) )

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]) )
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]) )
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]) )

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = product(ptup1[1], errorgen_1_bel_0)
            #it shouldn't be possible for ptup2[1] to equal errorgen_1_bel_0,
            #as that would imply that errorgen_1_bel_0 was the identity.
            if ptup2[1] == identity:
                errorGens.append((_LSE('H', [errorgen_1_bel_0]), -1j*.5*w*ptup1[0]*ptup2[0]))
            else:
                errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]) , -1j*.5*w*ptup1[0]*ptup2[0]))

            #ptup3 is just the product from ptup2 in reverse, so this can be done
            #more efficiently, but I'm not going to do that at present...
            ptup3 = product(errorgen_1_bel_0, ptup1[1])
            if ptup3[1] == identity:
                errorGens.append((_LSE('H', [errorgen_1_bel_0]), -1j*.5*w*ptup1[0]*ptup3[0]) )
            else:
                errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup3[1]]) , -1j*.5*w*ptup1[0]*ptup3[0]))
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'S':
        errorGens = commute_error_generators(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type == 'S' and errorgen_2_type == 'A':
        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), 1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))
        
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
                errorGens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), -1j*.5*w*ptup1[0]*ptup2[0]))
                            
    elif errorgen_1_type == 'A' and errorgen_1_type == 'S':
        errorGens = commute_error_generators(errorgen_2,errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'C':
        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1,errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1,errorgen_1_bel_0)                 
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))
        
        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity,
                    #And com(errorgen_2_bel_0, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_2_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #And com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_2_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_0)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_0) can't be either
                    errorGens.append((_LSE('A', [ptup2[1] , errorgen_1_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_1)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_1) can't be either
                    errorGens.append((_LSE('A', [ptup2[1] , errorgen_1_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))

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
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                errorGens.append((_LSE('C', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorGens.append((_LSE('S', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))


        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_1]), .5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    errorGens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_2_bel_1]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0,errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_2_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))

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
        errorGens = commute_error_generators(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'A' and errorgen_2_type == 'A':
        ptup1 = product(errorgen_2_bel_1, errorgen_1_bel_1)
        ptup2 = product(errorgen_1_bel_0, errorgen_2_bel_0)

        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                errorGens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorGens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorGens.append((_LSE('H', [ptup1[1]]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_1_bel_1]), -.5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_0, com(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_2_bel_1]), .5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, com(errorgen_1_bel_0,errorgen_1_bel_1)) can't be either.
                    errorGens.append((_LSE('C', [ptup2[1], errorgen_2_bel_0]), -.5*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, errorgen_1_bel_1)
            if ptup2 is not None:
                ptup3 = com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #it shouldn't be possible for ptup3 to be identity given valid error generator
                    #indices.
                    errorGens.append((_LSE('H', [ptup3[1]]), .25*w*ptup1[0]*ptup2[0]*ptup3[0]))
           
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