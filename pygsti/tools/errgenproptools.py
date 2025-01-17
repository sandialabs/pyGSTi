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
import numpy as _np
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.modelmembers.operations import LindbladErrorgen as _LinbladErrorgen
from pygsti.circuits import Circuit as _Circuit
from functools import reduce
from itertools import chain, product

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

    if isinstance(err_gen_coeff_label, _LEEL):
        return tuple([stim.PauliString(bel) for bel in err_gen_coeff_label.basis_element_labels])

    elif isinstance(err_gen_coeff_label, _GEEL):
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
    else:
        raise ValueError('Only `GlobalElementaryErrorgenLabel and LocalElementaryErrorgenLabel is currently supported.')

#------- Error Generator Math -------------#

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

    bch_order : int, optional (default 1)
        Order of the BCH approximation to use. Currently support for up to fifth order.
    
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
            current_combined_coeff_lbls = {key: None for key in chain(errgen_layer_1, errgen_layer_2)}            

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
            #precompute an identity string for comparisons in commutator calculations.
            if errgen_layer_1:
                identity = stim.PauliString('I'*len(next(iter(errgen_layer_1)).basis_element_labels[0]))
            commuted_errgen_list = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in errgen_layer_2.items():
                    #get the list of error generator labels
                    weight = .5*error1_val*error2_val
                    #avoid computing commutators which will be effectively zero.
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight= weight, identity=identity)
                    commuted_errgen_list.extend(commuted_errgen_sublist)
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
            for (error1a, error1a_val), (error1b, error1b_val) in zip(errgen_layer_1.items(), errgen_layer_2.items()):
                for error2, error2_val in second_order_comm_dict.items():
                    
                    #only need a factor of 1/6 because new_errorgen_layer[1] is 1/2 the commutator 
                    weighta = (1/6)*error1a_val*error2_val

                    #avoid computing commutators which will be effectively zero.
                    if not abs(weighta) < truncation_threshold:
                        commuted_errgen_sublist = error_generator_commutator(error1a, error2, 
                                                                             weight=weighta, identity=identity)
                        commuted_errgen_list_1.extend(commuted_errgen_sublist)
                    
                    #only need a factor of -1/6 because new_errorgen_layer[1] is 1/2 the commutator 
                    weightb = -(1/6)*error1b_val*error2_val
                    if not abs(weightb) < truncation_threshold:                    
                        commuted_errgen_sublist = error_generator_commutator(error1b, error2, 
                                                                             weight=weightb, identity=identity)
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
            current_combined_coeff_lbls = {key: None for key in chain(third_order_comm_dict_1, third_order_comm_dict_2)}
            for lbl in current_combined_coeff_lbls:
                third_order_rate = third_order_comm_dict_1.get(lbl, 0) + third_order_comm_dict_2.get(lbl, 0)
                if abs(third_order_rate) > truncation_threshold:
                    third_order_comm_dict[lbl] = third_order_rate
            new_errorgen_layer.append(third_order_comm_dict)
                         
        #fourth order BCH terms
        # -(1/24)*[Y,[X,[X,Y]]]
        elif curr_order == 3:
            #we've already calculated (1/12)*[X,[X,Y]] so reuse this result.
            #this is stored in third_order_comm_dict_1
            commuted_errgen_list = []
            for error1, error1_val in errgen_layer_2.items():
                for error2, error2_val in third_order_comm_dict_1.items():
                    #I *think* you can pick up at most around a factor of 8 from the commutator
                    #itself. Someone should validate that. Set this conservatively, but also
                    #avoid computing commutators which will be effectively zero.
                    #only need a factor of -1/2 because third_order_comm_dict_1 is 1/12 the nested commutator
                    weight = -.5*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list.extend(commuted_errgen_sublist)
            
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            fourth_order_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list}

            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list:
                fourth_order_comm_dict[error_tuple[0]] += error_tuple[1]

            #drop any terms below the truncation threshold after aggregation
            fourth_order_comm_dict = {key: val for key, val in fourth_order_comm_dict.items() if abs(val)>truncation_threshold}
            new_errorgen_layer.append(fourth_order_comm_dict)

        #Note for fifth order and beyond we can save a bunch of commutators
        #by using the results of https://doi.org/10.1016/j.laa.2003.09.010
        #Revisit this if going up to high-order ever becomes a regular computation.
        #fifth-order BCH terms:
        #-(1/720)*([X,F] - [Y, E]) + (1/360)*([Y,F] - [X,E]) + (1/120)*([Y,G] - [X,D])
        # Where: E = [Y,C]; F = [X,B]; G=[X,C]
        # B = [X,[X,Y]]; C = [Y,[X,Y]]; D = [Y,[X,[X,Y]]]
        # B, C and D have all been previously calculated (up to the leading constant). 
        # B is proportional to third_order_comm_dict_1, C is proportional to third_order_comm_dict_2
        # D is proportional to fourth_order_comm_dict
        # This gives 9 new commutators to calculate (7 if you used linearity, and even fewer would be needed
        # using the result from the paper above, but we won't here atm).
        elif curr_order == 4:
            B = third_order_comm_dict_1
            C = third_order_comm_dict_2
            D = fourth_order_comm_dict
            #Compute the new commutators E, F and G as defined above.
            #Start with E:
            commuted_errgen_list_E = []
            for error1, error1_val in errgen_layer_2.items():
                for error2, error2_val in C.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_E.extend(commuted_errgen_sublist)
            #Next F:
            commuted_errgen_list_F = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in B.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_F.extend(commuted_errgen_sublist)
            #Then G:
            commuted_errgen_list_G = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in C.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_G.extend(commuted_errgen_sublist)

            #Turn the commutator lists into dictionaries:
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            E_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_E}
            F_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_F}
            G_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_G}
            
            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list_E:
                E_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_F:
                F_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_G:
                G_comm_dict[error_tuple[0]] += error_tuple[1]

            #drop any terms below the truncation threshold after aggregation
            E_comm_dict = {key: val for key, val in E_comm_dict.items() if abs(val)>truncation_threshold}
            F_comm_dict = {key: val for key, val in F_comm_dict.items() if abs(val)>truncation_threshold}
            G_comm_dict = {key: val for key, val in G_comm_dict.items() if abs(val)>truncation_threshold}
            #-(1/720)*([X,F] - [Y, E]) + (1/360)*([Y,F] - [X,E]) + (1/120)*([Y,G] - [X,D])
            #Now do the next round of 6 commutators: [X,F], [Y,E], [Y,F], [X,E], [Y,G] and [X,D]
            #We also need the following weight factors. F has a leading factor of (1/12)
            #E and G have a leading factor of (-1/12). D has a leading factor of (-1/24) 
            #This gives the following additional weight multipliers:
            #[X,F] = (-1/60); [Y,E] = (1/60); [Y,F]= (1/30); [X,E]= (1/30); [Y,G] = (-1/10); [X,D] = (1/5)

            #[X,F]:
            commuted_errgen_list_XF = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in F_comm_dict.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = -(1/60)*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_XF.extend(commuted_errgen_sublist)
            #[Y,E]:
            commuted_errgen_list_YE = []
            for error1, error1_val in errgen_layer_2.items():
                for error2, error2_val in E_comm_dict.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = (1/60)*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_YE.extend(commuted_errgen_sublist)
            #[Y,F]:
            commuted_errgen_list_YF = []
            for error1, error1_val in errgen_layer_2.items():
                for error2, error2_val in F_comm_dict.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = (1/30)*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_YF.extend(commuted_errgen_sublist)
            #[X,E]:
            commuted_errgen_list_XE = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in E_comm_dict.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = (1/30)*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_XE.extend(commuted_errgen_sublist)
            #[Y,G]:
            commuted_errgen_list_YG = []
            for error1, error1_val in errgen_layer_2.items():
                for error2, error2_val in G_comm_dict.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = -.1*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_YG.extend(commuted_errgen_sublist)
            #[X,D]:
            commuted_errgen_list_XD = []
            for error1, error1_val in errgen_layer_1.items():
                for error2, error2_val in D.items():
                    #Won't add any weight adjustments at this stage, will do that for next commutator.
                    weight = .2*error1_val*error2_val
                    if abs(weight) < truncation_threshold:
                        continue
                    commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                         weight=weight, identity=identity)
                    commuted_errgen_list_XD.extend(commuted_errgen_sublist)

            #Turn the commutator lists into dictionaries:
            #loop through all of the elements of commuted_errorgen_list and instantiate a dictionary with the requisite keys.
            XF_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_XF}
            YE_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_YE}
            YF_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_YF}
            XE_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_XE}
            YG_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_YG}
            XD_comm_dict = {error_tuple[0]:0 for error_tuple in commuted_errgen_list_XD}

            #Add all of these error generators to the working dictionary of updated error generators and weights.
            #There may be duplicates, which should be summed together.
            for error_tuple in commuted_errgen_list_XF:
                XF_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_YE:
                YE_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_YF:
                YF_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_XE:
                XE_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_YG:
                YG_comm_dict[error_tuple[0]] += error_tuple[1]
            for error_tuple in commuted_errgen_list_XD:
                XD_comm_dict[error_tuple[0]] += error_tuple[1]

            #finally sum these six dictionaries, keeping only terms which are greater than the threshold.
            fifth_order_comm_dict = dict()
            fifth_order_dicts = [XF_comm_dict, YE_comm_dict, YF_comm_dict, XE_comm_dict, YG_comm_dict, XD_comm_dict]
            current_combined_coeff_lbls = {key: None for key in chain(*fifth_order_dicts)}
            for lbl in current_combined_coeff_lbls:
                fifth_order_rate = sum([comm_dict.get(lbl, 0) for comm_dict in fifth_order_dicts])
                if abs(fifth_order_rate) > truncation_threshold:
                    fifth_order_comm_dict[lbl] = fifth_order_rate
            new_errorgen_layer.append(fifth_order_comm_dict)

        else:
            raise NotImplementedError("Higher orders beyond fifth order are not implemented yet.")

    #Finally accumulate all of the dictionaries in new_errorgen_layer into a single one, summing overlapping terms.   
    errorgen_labels_by_order = [{key: None for key in order_dict} for order_dict in new_errorgen_layer]
    complete_errorgen_labels = reduce(lambda a, b: a|b, errorgen_labels_by_order)

    #initialize a dictionary with requisite keys
    new_errorgen_layer_dict = {lbl: 0 for lbl in complete_errorgen_labels}

    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] += rate

    #Future: Possibly do one last truncation pass in case any of the different order cancel out when aggregated?

    return new_errorgen_layer_dict

def error_generator_commutator(errorgen_1, errorgen_2, flip_weight=False, weight=1.0, identity=None):
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
    
    identity : stim.PauliString, optional (default None)
        An optional stim.PauliString to use for comparisons to the identity.
        Passing in this kwarg isn't necessary, but can allow for reduced 
        stim.PauliString creation when calling this function many times for
        improved efficiency.

    Returns
    -------
    list of `LocalStimErrorgenLabel`s corresponding to the commutator of the two input error generators,
    weighted by the specified value of `weight`.
    """
    
    errorgens=[]
    
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
    if identity is None:
        identity = stim.PauliString('I'*len(errorgen_1_bel_0))
        
    if errorgen_1_type=='H' and errorgen_2_type=='H':
        ptup = com(errorgen_1_bel_0 , errorgen_2_bel_0)
        if ptup is not None:
            errorgens.append((_LSE('H', [ptup[1]]), -1j*w *ptup[0]))
        
    elif errorgen_1_type=='H' and errorgen_2_type=='S':
        ptup = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        if ptup is not None:
            if errorgen_2_bel_0 == ptup[1]:
                errorgens.append(( _LSE('S', [errorgen_2_bel_0]), 2*1j*w*ptup[0]))
            else:
                new_bels =  [errorgen_2_bel_0, ptup[1]] if stim_pauli_string_less_than(errorgen_2_bel_0, ptup[1])\
                            else [ptup[1], errorgen_2_bel_0]
                errorgens.append(( _LSE('C', new_bels), 1j*w*ptup[0]))

    elif errorgen_1_type=='S' and errorgen_2_type=='H':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
          
    elif errorgen_1_type=='H' and errorgen_2_type=='C':
        ptup1 = com(errorgen_2_bel_0 , errorgen_1_bel_0)
        ptup2 = com(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1 is not None:
            if ptup1[1] == errorgen_2_bel_1:
                errorgens.append((_LSE('S', [errorgen_2_bel_1]), 2*1j*w*ptup1[0]))
            else:
                new_bels =  [ptup1[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup1[1], errorgen_2_bel_1)\
                            else [errorgen_2_bel_1, ptup1[1]]
                errorgens.append((_LSE('C', new_bels), 1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] == errorgen_2_bel_0:
                errorgens.append(( _LSE('S', [errorgen_2_bel_0]), 2*1j*w*ptup2[0]))
            else:
                new_bels =  [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0)\
                            else [errorgen_2_bel_0, ptup2[1]]
                errorgens.append((_LSE('C', new_bels), 1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='C' and errorgen_2_type=='H':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
    
    elif errorgen_1_type=='H' and errorgen_2_type=='A':
        ptup1 = com(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = com(errorgen_1_bel_0 , errorgen_2_bel_1)
        if ptup1 is not None:
            if ptup1[1] != errorgen_2_bel_1:
                if stim_pauli_string_less_than(ptup1[1], errorgen_2_bel_1):
                    errorgens.append((_LSE('A', [ptup1[1], errorgen_2_bel_1]), -1j*w*ptup1[0]))
                else:
                    errorgens.append((_LSE('A', [errorgen_2_bel_1, ptup1[1]]), 1j*w*ptup1[0]))
        if ptup2 is not None:
            if ptup2[1] != errorgen_2_bel_0:
                if stim_pauli_string_less_than(errorgen_2_bel_0, ptup2[1]):
                    errorgens.append((_LSE('A', [errorgen_2_bel_0, ptup2[1]]), -1j*w*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], errorgen_2_bel_0]), 1j*w*ptup2[0]))
                          
    elif errorgen_1_type=='A' and errorgen_2_type=='H':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type=='S' and errorgen_2_type=='S':
        #Commutator of S with S is zero.
        pass
                         
    elif errorgen_1_type=='S' and errorgen_2_type=='C':
        ptup1 = pauli_product(errorgen_1_bel_0 , errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1 , errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = pauli_product(ptup1[1], errorgen_1_bel_0)
            #it shouldn't be possible for ptup2[1] to equal errorgen_1_bel_0,
            #as that would imply that errorgen_1_bel_0 was the identity.
            if ptup2[1] == identity:
                errorgens.append((_LSE('H', [errorgen_1_bel_0]), -1j*.5*w*ptup1[0]*ptup2[0]))
            else:
                if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                    errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]) , -1j*.5*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]) , 1j*.5*w*ptup1[0]*ptup2[0]))

            #ptup3 is just the product from ptup2 in reverse, so this can be done
            #more efficiently, but I'm not going to do that at present...
            ptup3 = pauli_product(errorgen_1_bel_0, ptup1[1])
            if ptup3[1] == identity:
                errorgens.append((_LSE('H', [errorgen_1_bel_0]), 1j*.5*w*ptup1[0]*ptup3[0]) )
            else:
                if stim_pauli_string_less_than(errorgen_1_bel_0, ptup3[1]):
                    errorgens.append((_LSE('A', [errorgen_1_bel_0, ptup3[1]]) , -1j*.5*w*ptup1[0]*ptup3[0]))
                else:
                    errorgens.append((_LSE('A', [ptup3[1], errorgen_1_bel_0]) , 1j*.5*w*ptup1[0]*ptup3[0]))
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'S':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)

    elif errorgen_1_type == 'S' and errorgen_2_type == 'A':
        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else:
            if ptup[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))
        
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
                    errorgens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))
                            
    elif errorgen_1_type == 'A' and errorgen_2_type == 'S':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'C' and errorgen_2_type == 'C':
        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity: 
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_1,errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1,errorgen_1_bel_0)                 
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))        
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_1, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
        
        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity,
                    #And com(errorgen_2_bel_0, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_2_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_2_bel_1, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #And com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_2_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_2_bel_0, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_0)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_0) can't be either
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_1]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_1_bel_1, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(ptup1[1], errorgen_1_bel_1)
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #And com(acom(errorgen_2_bel_0, errorgen_2_bel_1), errorgen_2_bel_1) can't be either
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), -.5*1j*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), .5*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = acom(errorgen_2_bel_0, errorgen_2_bel_1)
            if ptup2 is not None:
                ptup3 = com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #It shouldn't be possible for ptup3 to be the identity given valid error generator indices.
                    errorgens.append((_LSE('H', [ptup3[1]]), .25*1j*w*ptup1[0]*ptup2[0]*ptup3[0]))

    elif errorgen_1_type == 'C' and errorgen_2_type == 'A':
        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), 1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), 2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = pauli_product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if ptup1[1] != identity and ptup2[1] != identity:
                new_bels = [ptup1[1], ptup2[1]] if stim_pauli_string_less_than(ptup1[1], ptup2[1]) else [ptup2[1], ptup1[1]]
                errorgens.append((_LSE('C', new_bels), -1j*w*ptup1[0]*ptup2[0]))
        else: #ptup[1] == ptup[2]
            if ptup1[1] != identity:
                errorgens.append((_LSE('S', [ptup1[1]]), -2*1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_1]), .5*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_1_bel_1, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0):
                        errorgens.append((_LSE('A', [ptup2[1], errorgen_1_bel_0]), .5*w*ptup1[0]*ptup2[0]))
                    else:
                        errorgens.append((_LSE('A', [errorgen_1_bel_0, ptup2[1]]), -.5*w*ptup1[0]*ptup2[0]))

        ptup1 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    new_bels = [ptup2[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1) else [errorgen_2_bel_1, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), .5*1j*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_1, don't need to check that errorgen_2_bel_1 isn't identity.
                    errorgens.append((_LSE('S', [errorgen_2_bel_1]), 1j*w*ptup1[0]*ptup2[0]))


        ptup1 = acom(errorgen_1_bel_0,errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, acom(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either
                    new_bels = [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0) else [errorgen_2_bel_0, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), -.5*1j*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_0, don't need to check that errorgen_2_bel_0 isn't identity.
                    errorgens.append((_LSE('S', [errorgen_2_bel_0]), -1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = acom(errorgen_1_bel_0, errorgen_1_bel_1)
            if ptup2 is not None:
                ptup3= com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #it shouldn't be possible for ptup3 to be identity given valid error generator
                    #indices.
                    errorgens.append((_LSE('H', [ptup3[1]]), -.25*w*ptup1[0]*ptup2[0]*ptup3[0]))
    
    elif errorgen_1_type == 'A' and errorgen_2_type == 'C':
        errorgens = error_generator_commutator(errorgen_2, errorgen_1, flip_weight=True, weight=weight)
                         
    elif errorgen_1_type == 'A' and errorgen_2_type == 'A':
        ptup1 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_1)
        ptup2 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_0)

        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_0)
        ptup2 = pauli_product(errorgen_1_bel_1, errorgen_2_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_1, errorgen_2_bel_0)
        ptup2 = pauli_product(errorgen_2_bel_1, errorgen_1_bel_0)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = pauli_product(errorgen_1_bel_0, errorgen_2_bel_1)
        ptup2 = pauli_product(errorgen_2_bel_0, errorgen_1_bel_1)
        if ptup1[1] != ptup2[1]:
            if (ptup1[1] != identity) and (ptup2[1] != identity):
                if stim_pauli_string_less_than(ptup1[1], ptup2[1]):
                    errorgens.append((_LSE('A', [ptup1[1], ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
                else:
                    errorgens.append((_LSE('A', [ptup2[1], ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))
            elif ptup1[1] == identity:
                errorgens.append((_LSE('H', [ptup2[1]]), -1j*w*ptup1[0]*ptup2[0]))
            else: #ptup2[1] == identity
                errorgens.append((_LSE('H', [ptup1[1]]), 1j*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_0:
                    #errorgen_1_bel_0 can't be the identity.
                    #com(errorgen_1_bel_1, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_1_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_0) else [errorgen_1_bel_0, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), .5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_1_bel_0
                    errorgens.append((_LSE('S', [errorgen_1_bel_0]), w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_1_bel_1:
                    #errorgen_1_bel_1 can't be the identity.
                    #com(errorgen_1_bel_0, com(errorgen_2_bel_0, errorgen_2_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_1_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_1_bel_1) else [errorgen_1_bel_1, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), -.5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_1_bel_1
                    errorgens.append((_LSE('S', [errorgen_1_bel_1]), -1*w*ptup1[0]*ptup2[0]))
        
        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_0, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_1:
                    #errorgen_2_bel_1 can't be the identity.
                    #com(errorgen_2_bel_0, com(errorgen_1_bel_0, errorgen_1_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_2_bel_1] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_1) else [errorgen_2_bel_1, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), .5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_1
                    errorgens.append((_LSE('S', [errorgen_2_bel_1]), w*ptup1[0]*ptup2[0]))


        ptup1 = com(errorgen_1_bel_0, errorgen_1_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_2_bel_1, ptup1[1])
            if ptup2 is not None:
                if ptup2[1] != errorgen_2_bel_0:
                    #errorgen_2_bel_0 can't be the identity.
                    #com(errorgen_2_bel_1, com(errorgen_1_bel_0,errorgen_1_bel_1)) can't be either.
                    new_bels = [ptup2[1], errorgen_2_bel_0] if stim_pauli_string_less_than(ptup2[1], errorgen_2_bel_0) else [errorgen_2_bel_0, ptup2[1]]
                    errorgens.append((_LSE('C', new_bels), -.5*w*ptup1[0]*ptup2[0]))
                else: #ptup2[1] == errorgen_2_bel_0
                    errorgens.append((_LSE('S', [errorgen_2_bel_0]), -1*w*ptup1[0]*ptup2[0]))

        ptup1 = com(errorgen_2_bel_0, errorgen_2_bel_1)
        if ptup1 is not None:
            ptup2 = com(errorgen_1_bel_0, errorgen_1_bel_1)
            if ptup2 is not None:
                ptup3 = com(ptup1[1], ptup2[1])
                if ptup3 is not None:
                    #it shouldn't be possible for ptup3 to be identity given valid error generator
                    #indices.
                    errorgens.append((_LSE('H', [ptup3[1]]), .25*1j*w*ptup1[0]*ptup2[0]*ptup3[0]))
           
    return errorgens

def error_generator_composition(errorgen_1, errorgen_2, weight=1.0, identity=None):
    """
    Returns the composition of two error generators. I.e. errorgen_1[errorgen_2[\cdot]].
    
    Parameters
    ----------
    errorgen1 : `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `LocalStimErrorgenLabel`
        Second error generator
    
    weight : float, optional (default 1.0)
        An optional weighting value to apply to the value of the composition.
    
    identity : stim.PauliString, optional (default None)
        An optional stim.PauliString to use for comparisons to the identity.
        Passing in this kwarg isn't necessary, but can allow for reduced 
        stim.PauliString creation when calling this function many times for
        improved efficiency.

    Returns
    -------
    list of `LocalStimErrorgenLabel`s corresponding to the composition of the two input error generators,
    weighted by the specified value of `weight`.
    """

    composed_errorgens = []

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
    if identity is None:
        identity = stim.PauliString('I'*len(errorgen_1_bel_0))

    if errorgen_1_type == 'H' and errorgen_2_type == 'H':
        #H_P[H_Q] P->errorgen_1_bel_0, Q -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_2_bel_0
        P_eq_Q = (P==Q)
        if P.commutes(Q):
            new_eg_type, new_bels, addl_scale = _ordered_new_bels_C(P, Q, False, False, P_eq_Q)
            composed_errorgens.append((_LSE(new_eg_type, new_bels), addl_scale*w))
        else:
            PQ = pauli_product(P, Q)
            composed_errorgens.append((_LSE('H', [PQ[1]]), -1j*w*PQ[0]))
            new_eg_type, new_bels, addl_scale = _ordered_new_bels_C(P, Q, False, False, P_eq_Q)
            composed_errorgens.append((_LSE(new_eg_type, new_bels), addl_scale*w))

    elif errorgen_1_type == 'H' and errorgen_2_type == 'S':
        #H_P[S_Q] P->errorgen_1_bel_0, Q -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_2_bel_0
        PQ = pauli_product(P, Q)
        PQ_ident = (PQ[1] == identity)
        PQ_eq_Q = (PQ[1]==Q)
        if P.commutes(Q):
            new_eg_type, new_bels, addl_sign = _ordered_new_bels_A(PQ[1], Q, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -PQ[0]*addl_sign*w))
            composed_errorgens.append((_LSE('H', [P]), -w))   
        else: #if errorgen_1_bel_0 and errorgen_2_bel_0 only multiply to identity they are equal (in which case they commute).
            new_eg_type, new_bels, addl_scale = _ordered_new_bels_C(PQ[1], Q, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -1j*PQ[0]*addl_scale*w))
            composed_errorgens.append((_LSE('H', [P]), -w))

    elif errorgen_1_type == 'H' and errorgen_2_type == 'C':
        #H_A[C_{P,Q}] A->errorgen_1_bel_0, P,Q -> errorgen_2_bel_0, errorgen_2_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        A = errorgen_1_bel_0 
        #also precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)

        #Case 1: [P,Q]=0
        if P.commutes(Q):
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])

            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            PQ_ident = (PQ[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            PQ_eq_A = (PQ[1]==A)
            
            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
        else: #Case 2: {P,Q}=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_scale_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_scale_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))

    elif errorgen_1_type == 'H' and errorgen_2_type == 'A':
        #H_A[A_{P,Q}] A->errorgen_1_bel_0, P,Q -> errorgen_2_bel_0, errorgen_2_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        A = errorgen_1_bel_0
        #precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)
        #Case 1: P and Q commute.
        if P.commutes(Q):
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_scale_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_sign_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_sign_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_scale_1*w))
        else: #Case 2: {P,Q}=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])
            #also also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            PQ_ident = (PQ[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            PQ_eq_A = (PQ[1]==A)
            
            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_sign_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_sign_2*w))

    #Note: This could be done by leveraging the commutator code, but that adds
    #additional overhead which I am opting to avoid.
    elif errorgen_1_type == 'S' and errorgen_2_type == 'H':
        #S_P[H_Q] P->errorgen_1_bel_0, Q -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_2_bel_0
        PQ = pauli_product(P, Q)
        PQ_ident = (PQ[1] == identity)
        PQ_eq_Q = (PQ[1]==Q)
        if P.commutes(Q):
            new_eg_type, new_bels, addl_sign = _ordered_new_bels_A(PQ[1], P, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -PQ[0]*addl_sign*w))
            composed_errorgens.append((_LSE('H', [Q]), -w))   
        else: #if errorgen_1_bel_0 and errorgen_2_bel_0 only multiply to identity they are equal (in which case they commute).
            new_eg_type, new_bels, addl_scale = _ordered_new_bels_C(PQ[1], P, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -1j*PQ[0]*addl_scale*w))
            composed_errorgens.append((_LSE('H', [Q]), -w))

    elif errorgen_1_type == 'S' and errorgen_2_type == 'S':
        #S_P[S_Q] P->errorgen_1_bel_0, Q -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_2_bel_0
        PQ = pauli_product(P, Q)
        PQ_ident = (PQ[1] == identity)
        if not PQ_ident:
            composed_errorgens.append((_LSE('S', [PQ[1]]), w))
        composed_errorgens.append((_LSE('S', [P]), -w))
        composed_errorgens.append((_LSE('S', [Q]),- w))

    elif errorgen_1_type == 'S' and errorgen_2_type == 'C':
        #S_A[C_P,Q] A-> errorgen_1_bel_0, P->errorgen_2_bel_0, Q -> errorgen_2_bel_1
        A = errorgen_1_bel_0
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1

        #also precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)

        if P.commutes(Q): #Case 1: [P,Q] = 0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            #APQ can't equal A since that implies P==Q, which would be an invalid C term input.

            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #TODO: Cases (1a,1b) and (1c,1d) only differ by the leading sign, can compress this code a bit.
        else: #Case 2: {P,Q}=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            assert not PA_eq_QA #(I'm almost positive this should be true)

            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #TODO: Cases (2a,2b) and (2c,2d) only differ by the leading sign, can compress this code a bit.

    elif errorgen_1_type == 'S' and errorgen_2_type == 'A':
        #S_A[A_P,Q] A-> errorgen_1_bel_0, P->errorgen_2_bel_0, Q -> errorgen_2_bel_1
        A = errorgen_1_bel_0
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1

        #precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)

        if P.commutes(Q): #Case 1: [P,Q]=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)

            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            assert not PA_eq_QA #(I'm almost positive this should be true)

            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_sign_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_sign_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0  = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_sign_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0  = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_sign_1*w))
            #TODO: Cases (1a,1b) and (1c,1d) only differ by the leading sign, can compress this code a bit.
        else:
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            #APQ can't equal A since that implies P==Q, which would be an invalid C term input.

            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_sign_2*w))

            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_sign_2*w))

            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_sign_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_sign_2*w))
            #TODO: Cases (2a,2b) and (2c,2d) only differ by the leading sign, can compress this code a bit.
    
    elif errorgen_1_type == 'C' and errorgen_2_type == 'H':
        #C_P,Q[H_A]: P -> errorgen_1_bel_0, Q-> errorgen_1_bel_1, A -> errorgen_2_bel_0
        #TODO: This only differs from H-C by a few signs, should be able to combine the two implementations to save space.
        P = errorgen_1_bel_0
        Q = errorgen_1_bel_1
        A = errorgen_2_bel_0
        #precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)

        if P.commutes(Q): #[P,Q]=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])
            #also precompute whether any of these products are the identity (PQ can't be the identity if this is a valid C term).
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            PQ_ident = (PQ[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            PQ_eq_A = (PQ[1]==A)
            
            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_sign_2  = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_sign_2*w))
        else: #Case 2: {P,Q}=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_Q = (PA[1]==Q)
            QA_eq_P = (QA[1]==P)
            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_scale_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_scale_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_sign_1  = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_sign_1*w))

    elif errorgen_1_type == 'C' and errorgen_2_type == 'S': #This differs from S-C by just a few signs. Should be able to combine and significantly compress code.
        #C_P,Q[S_A] P-> errorgen_1_bel_0, Q -> errorgen_1_bel_1, A->errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_1_bel_1
        A = errorgen_2_bel_0
        #also precompute whether pairs commute or anticommute
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)

        if P.commutes(Q): #Case 1: [P,Q] = 0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            PQ = pauli_product(P, Q)
            APQ = pauli_product(A, PQ[0]*PQ[1])
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            APQ_ident = (APQ[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            #APQ can't equal A since that implies P==Q, which would be an invalid C term input.

            #Case 1a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_scale_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_sign_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_scale_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*APQ[0]*addl_sign_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_scale_2*w))
            #TODO: Cases (1a,1b) and (1c,1d) only differ by the leading sign, can compress this code a bit.
        else: #Case 2: {P,Q}=0
            #precompute some products we'll need.
            PA = pauli_product(P, A)
            QA = pauli_product(Q, A)
            #also precompute whether any of these products are the identity
            PA_ident = (PA[1] == identity)
            QA_ident = (QA[1] == identity)
            #also also precompute whether certain relevant pauli pairs are equal.
            PA_eq_QA = (PA[1]==QA[1])
            assert not PA_eq_QA #(I'm almost positive this should be true)

            #Case 2a: [A,P]=0, [A,Q]=0
            if com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_scale_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_scale_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_sign_0  = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_scale_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_sign_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_scale_1*w))
            #TODO: Cases (2a,2b) and (2c,2d) only differ by the leading sign, can compress this code a bit.

    return composed_errorgens

#helper function for getting the new (properly ordered) basis element labels, error generator type (A can turn into H with certain index combinations), and additional signs.
#reduces code repetition in composition code.
def _ordered_new_bels_A(pauli1, pauli2, first_pauli_ident, second_pauli_ident, pauli_eq):
    """
    Helper function for managing new basis element labels, error generator types and proper basis element label ordering. Returns None
    if both pauli identity flags are True, which signals that the error generator is zero (i.e. should be skipped). Same for is pauli_eq is True.
    """
    if pauli_eq:
        return (None,None,None)
    if first_pauli_ident:
        if second_pauli_ident:
            return (None,None,None)
        else:
            new_eg_type = 'H'
            new_bels = [pauli2] 
            addl_sign = 1
    else:
        if second_pauli_ident:
            new_eg_type = 'H'
            new_bels = [pauli1]
            addl_sign = -1
        else:
            new_eg_type = 'A'
            new_bels, addl_sign = ([pauli1, pauli2], 1) if stim_pauli_string_less_than(pauli1, pauli2) else ([pauli2, pauli1], -1)
    return new_eg_type, new_bels, addl_sign

def _ordered_new_bels_C(pauli1, pauli2, first_pauli_ident, second_pauli_ident, pauli_eq):
    """
    Helper function for managing new basis element labels, error generator types and proper basis element label ordering. Returns None
    if both pauli identity flags are True, which signals that the error generator is zero (i.e. should be skipped). Same for is pauli_eq is True.
    """
    if first_pauli_ident or second_pauli_ident:
        return (None,None,None)

    if pauli_eq:
        new_eg_type = 'S'
        new_bels = [pauli1]
        addl_scale_fac = 2
    else:
        new_eg_type = 'C'
        addl_scale_fac = 1
        new_bels = [pauli1, pauli2] if stim_pauli_string_less_than(pauli1, pauli2) else [pauli2, pauli1]
    return new_eg_type, new_bels, addl_scale_fac

def com(P1, P2):
    #P1 and P2 either commute or anticommute.
    if P1.commutes(P2):
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

def pauli_product(P1, P2):
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

def errorgen_layer_to_matrix(errorgen_layer, num_qubits, errorgen_matrix_dict=None, sslbls=None):
    """
    Converts an iterable over error generator coefficients and rates into the corresponding
    dense numpy array representation.

    Parameters
    ----------
    errorgen_layer : list, tuple or dict
        An iterable over error generator coefficient and rates. If a list or a tuple the
        elements should correspond to two-element tuples, the first value being an `ElementaryErrorgenLabel`
        and the second value the rate. If a dictionary the keys should be `ElementaryErrorgenLabel` and the
        values the rates.

    num_qubits : int
        Number of qubits for the error generator matrix being constructed.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.
    
    sslbls : list or tuple, optional (default None)
        A tuple or list of state space labels corresponding to the qubits upon which the error generators
        can supported. Only required when passing in a value of `errorgen_matrix_dict` with
        `GlobalElementaryErrogenLabel` keys in conjunction with an `errorgen_layer` with labels
        which are `LocalElementaryErrorgenLabel` (or vice-versa).
    
    Returns
    -------
    errorgen_mat : ndarray 
        ndarray for the dense representation of the specified error generator in the standard basis.
    """

    #if the list is empty return all zeros
    #initialize empty array for accumulation.
    mat = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    if not errorgen_layer:
        return mat
    
    if errorgen_matrix_dict is None:
        #create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    #infer the correct label type.
    if errorgen_matrix_dict:
        first_label = next(iter(errorgen_matrix_dict))
        if isinstance(first_label, _LEEL):
            label_type = 'local'
        elif isinstance(first_label, _GEEL):
            label_type = 'global'
        else:
            msg = f'Label type {type(first_label)} is not supported as a key for errorgen_matrix_dict.'\
                  + 'Please use either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel.'
            raise ValueError()
    else:
        raise ValueError('Non-empty errorgen_layer, but errorgen_matrix_dict is empty. Cannot convert.')
        
    #loop through errorgen_layer and accumulate the weighted error generators prescribed.
    if isinstance(errorgen_layer, (list, tuple)):
        first_coefficient_lbl = errorgen_layer[0][0]
        errorgen_layer_iter = errorgen_layer
    elif isinstance(errorgen_layer, dict):
        first_coefficient_lbl = next(iter(errorgen_layer))
        errorgen_layer_iter = errorgen_layer.items()
    else:
        raise ValueError(f'errorgen_layer should be either a list, tuple or dict. {type(errorgen_layer)=}')

    if ((isinstance(first_coefficient_lbl, _LEEL) and label_type == 'global') \
        or (isinstance(first_coefficient_lbl, _GEEL) and label_type == 'local')) and sslbls is None:
        msg = "You have passed in an `errogen_layer` with `LocalElementaryErrorgenLabel` coefficients, and " \
              +"an `errorgen_matrix_dict` with keys which are `GlobalElementaryErrorgenLabel` (or vice-versa). When using this "\
              +"combination you must also specify the state space labels with `sslbls`."
        raise ValueError(msg)

    if isinstance(first_coefficient_lbl, _LSE):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_local_eel()]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl.to_global_eel()]
    elif isinstance(first_coefficient_lbl, _LEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_GEEL.cast(lbl, sslbls=sslbls)]
    elif isinstance(first_coefficient_lbl, _GEEL):
        if label_type == 'local':
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[_LEEL.cast(lbl, sslbls=sslbls)]
        else:
            for lbl, rate in errorgen_layer_iter:
                mat +=  rate*errorgen_matrix_dict[lbl]
    else:
        raise ValueError('The coefficient labels in `errorgen_layer` should be either `LocalStimErrorgenLabel`, `LocalElementaryErrorgenLabel` or `GlobalElementaryErrorgenLabel`.')
    
    return mat


#Helper function for doing numeric commutators and compositions.

def error_generator_commutator_numerical(errorgen1, errorgen2, errorgen_matrix_dict=None, num_qubits=None):
    """
    Numerically compute the commutator of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `ElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the commutator of the input error generators in the standard basis.
    """

    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        #create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]\
                  - errorgen_matrix_dict[errorgen2.to_local_eel()]@errorgen_matrix_dict[errorgen1.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comm = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2] - errorgen_matrix_dict[errorgen2]@errorgen_matrix_dict[errorgen1]
        else:
            comm = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]\
                  - errorgen_matrix_dict[_LSE.cast(errorgen2)]@errorgen_matrix_dict[_LSE.cast(errorgen1)]
    return comm

def error_generator_composition_numerical(errorgen1, errorgen2, errorgen_matrix_dict=None, num_qubits=None):
    """
    Numerically compute the composition of the two specified elementary error generators.

    Parameters
    ----------
    errorgen1 : `LocalElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        First error generator.

    errorgen2 : `ElementaryErrorgenLabel` or `LocalStimErrorgenLabel`
        Second error generator.

    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.
    
    Returns
    -------
    ndarray
        Numpy array corresponding to the dense representation of the composition of the input error generators in the standard basis.
        
    """
    assert isinstance(errorgen1, (_LEEL, _LSE)) and isinstance(errorgen2, (_LEEL, _LSE))
    assert type(errorgen1) == type(errorgen2), "The elementary error generator labels have mismatched types."
    
    if errorgen_matrix_dict is None:
        #create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    first_label = next(iter(errorgen_matrix_dict))
    
    if isinstance(first_label, _LEEL):
        if isinstance(errorgen1, _LEEL):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[errorgen1.to_local_eel()]@errorgen_matrix_dict[errorgen2.to_local_eel()]
    else:
        if isinstance(errorgen1, _LSE):
            comp = errorgen_matrix_dict[errorgen1]@errorgen_matrix_dict[errorgen2]
        else:
            comp = errorgen_matrix_dict[_LSE.cast(errorgen1)]@errorgen_matrix_dict[_LSE.cast(errorgen2)]
    return comp


#-----------First-Order Approximate Error Generator Probabilities---------------#

def random_support(tableau, return_support=False):
    """ 
    Compute the number of bits over which the stabilizer state corresponding to this stim tableau
    would have measurement outcomes which are random.
    
    Parameters
    ----------
    tableau : stim.Tableau
        stim.Tableau corresponding to the stabilizer state we want the random support
        for.
    
    return_support : bool, optional (default False)
        If True also returns a list of qubit indices over which the distribution of outcome
        bit strings is random.
    """
    #TODO Test for correctness on support
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    num_random = 0
    support = []
    for i in range(len(tableau)):
        z = sim.peek_z(i)
        if z == 0:
            num_random+=1
            support.append(i)
            # For a phase reference, use the smallest state with non-zero amplitude.
        forced_bit = z == -1
        sim.postselect_z(i, desired_value=forced_bit)
    return (num_random, support) if return_support else num_random

#Courtesy of Gidney 
#https://quantumcomputing.stackexchange.com/questions/38826/how-do-i-efficiently-compute-the-fidelity-between-two-stabilizer-tableau-states
def tableau_fidelity(tableau1, tableau2):
    """
    Calculate the fidelity between the stabilizer states corresponding to the given stim
    tableaus. This returns a result in units of probability (so this may be squared
    fidelity depending on your convention).
    
    Parameters
    ----------
    tableau1 : stim.Tableau
        Stim tableau for first stabilizer state.
    tableau2 : stim.Tableau
        Stim tableau for second stabilizer state.
    """
    t3 = tableau2**-1 * tableau1
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(t3)
    p = 1
    #note to future selves: stim uses little endian convention by default, and we typically use
    #big endian. That doesn't make a difference in this case, but does elsewhere to be mindful to
    #save on grief.
    for q in range(len(t3)):
        e = sim.peek_z(q)
        if e == -1:
            return 0
        if e == 0:
            p *= 0.5
            sim.postselect_z(q, desired_value=False)
    return p

def bitstring_to_tableau(bitstring):
    """
    Map a computational basis bit string into a corresponding Tableau which maps the all zero
    state into that state.
    
    Parameters
    ----------
    bitstring : str
        String of 0's and 1's corresponding to the computational basis state to prepare the Tableau for.
    
    Returns
    -------
    stim.Tableau
        Tableau which maps the all zero string to this computational basis state
    """
    pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in bitstring]))
    #convert this to a stim.Tableau
    pauli_tableau = pauli_string.to_tableau()
    return pauli_tableau


#Modified from Gidney 
#https://quantumcomputing.stackexchange.com/questions/34610/get-the-amplitude-of-a-computational-basis-in-stim
def amplitude_of_state(tableau, desired_state):
    """
    Get the amplitude of a particular computational basis state for given
    stabilizer state.

    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau corresponding to the stabilizer state we wish to extract
        the amplitude from.
    
    desired_state : str
        String of 0's and 1's corresponding to the computational basis state to extract the amplitude for.
    """

    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    n = sim.num_qubits
    
    #convert desired state into a list of bools
    desired_state = [desired_state[i] == '1' for i in range(n)]
    
    # Determine the magnitude of the target state.
    copy = sim.copy()
    num_random = 0
    for q in range(n):
        desired_bit = desired_state[q]
        z = copy.peek_z(q)
        forced_bit = z == -1
        if z == 0:
            num_random += 1
        elif desired_bit != forced_bit: #forced bit is true if the state is |1>, so this is checking whether the bits match.
            return 0
        copy.postselect_z(q, desired_value=desired_bit)
    magnitude = 2**-(num_random / 2)
    # For a phase reference, use the smallest state with non-zero amplitude.
    copy = sim.copy()
    ref_state = [False]*n
    for q in range(n):
        z = copy.peek_z(q)
        forced_bit = z == -1
        ref_state[q] = forced_bit
        copy.postselect_z(q, desired_value=forced_bit)
    if ref_state == desired_state:
        return magnitude

    # Postselect away states that aren't the desired or reference states.
    # Also move the ref state to |00..00> and the desired state to |00..01>.
    copy = sim.copy()
    found_difference = False
    for q in range(n):
        desired_bit =  desired_state[q]
        ref_bit = ref_state[q]
        if desired_bit == ref_bit:
            copy.postselect_z(q, desired_value=ref_bit)
            if desired_bit:
                copy.x(q)
        elif not found_difference:
            found_difference = True
            if q:
                copy.swap(0, q)
            if ref_bit:
                copy.x(0)
        else:
            # Remove difference between target state and ref state at this bit.
            copy.cnot(0, q)
            copy.postselect_z(q, desired_value=ref_bit)

    # The phase difference between |00..00> and |00..01> is what we want.
    # Since other states are gone, this is the bloch vector phase of qubit 0.
    assert found_difference
    s = str(copy.peek_bloch(0))
    
    if s == "+X":
        phase_factor = 1
    if s == "-X":
        phase_factor = -1
    if s == "+Y":
        phase_factor = 1j
    if s == "-Y":
        phase_factor = -1j
    
    return phase_factor*magnitude

def pauli_phase_update(pauli, bitstring):
    """
    Takes as input a pauli and a bit string and computes the output bitstring
    and the overall phase that bit string accumulates.
    
    Parameters
    ----------
    pauli : str or stim.PauliString
        Pauli to apply
    
    bitstring : str
        String of 0's and 1's representing the bit string to apply the pauli to.
    
    Returns
    -------
    Tuple whose first element is the phase accumulated, and whose second element
    is a string corresponding to the updated bit string.
    """
    
    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)
    
    bitstring = [False if bit=='0' else True for bit in bitstring]
    #list of phase correction for each pauli (conditional on 0)
    #Read [I, X, Y, Z]
    pauli_phases_0 = [1, 1, -1j, 1]
    
    #list of the phase correction for each pauli (conditional on 1)
    #Read [I, X, Y, Z]
    pauli_phases_1 = [1, 1, 1j, -1]
    
    #list of bools corresponding to whether each pauli flips the target bit
    pauli_flips = [False, True, True, False]
    
    overall_phase = 1
    indices_to_flip = []
    for i, (elem, bit) in enumerate(zip(pauli, bitstring)):
        if bit:
            overall_phase*=pauli_phases_1[elem]
        else:
            overall_phase*=pauli_phases_0[elem]
        if pauli_flips[elem]:
            indices_to_flip.append(i)
    #if the input pauli had any overall phase associated with it add that back
    #in too.
    overall_phase*=pauli.sign
    #apply the flips to get the output bit string.
    for idx in indices_to_flip:
        bitstring[idx] = not bitstring[idx]
    #turn this back into a string
    output_bitstring = ''.join(['1' if bit else '0' for bit in bitstring])
    
    return overall_phase, output_bitstring

#TODO: This function needs a more evocative name
def phi(tableau, desired_bitstring, P, Q):
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    
    Parameters
    ----------
    tableau : stim.Tableau
        A stim Tableau corresponding to the input stabilizer state.
        
    desired_bitstring : str
        A string of zeros and ones corresponding to the bit string being measured.
        
    P : str or stim.PauliString
        The first pauli string index.
    Q : str or stim.PauliString
        The second pauli string index.
        
    Returns
    -------
    A complex number corresponding to the value of the phi function.
    """
    
    #start by getting the pauli string which maps the all-zeros string to the target bitstring.
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring]))
    
    #map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    #combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string*P
    eff_Q = Q*initial_pauli_string
    
    #now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    #the corresponding phase corrections.
    all_zeros = '0'*len(eff_P)
    phase1, bitstring1 = pauli_phase_update(eff_P, all_zeros)
    phase2, bitstring2 = pauli_phase_update(eff_Q, all_zeros)
    
    #get the amplitude of these two bitstrings in the stabilizer state.
    amp1 = amplitude_of_state(tableau, bitstring1)
    amp2 = amplitude_of_state(tableau, bitstring2) 
    
    #now apply the phase corrections. 
    amp1*=phase1
    amp2*=phase2
      
    #calculate phi.
    #The second amplitude also needs a complex conjugate applied
    phi = amp1*amp2.conjugate()
    
    #phi should ultimately be either 0, +/-1 or +/-i, scaling might overflow
    #so avoid scaling and just identify which of these it should be. For really
    #tiny phi this may still have an issue...
    if abs(phi)>1e-14:
        if abs(phi.real) > 1e-14:
            if phi.real > 0:
                return complex(1)
            else:
                return complex(-1)
        else:
            if phi.imag > 0:
                return 1j
            else:
                return -1j 
    else:
        return complex(0)

def alpha(errorgen, tableau, desired_bitstring):
    """
    First-order error generator sensitivity function for probability.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    errgen_type = errorgen.errorgen_type
    basis_element_labels = errorgen.basis_element_labels
    
    if not isinstance(basis_element_labels[0], stim.PauliString):
        basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])
    
    identity_pauli = stim.PauliString('I'*len(basis_element_labels[0]))
    
    if errgen_type == 'H':
        sensitivity = 2*phi(tableau, desired_bitstring, basis_element_labels[0], identity_pauli).imag
        
    elif errgen_type == 'S':
        sensitivity = phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[0]) \
                    - phi(tableau, desired_bitstring, identity_pauli, identity_pauli)
    elif errgen_type == 'C': #TODO simplify this logic
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[1])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli) \
                    + phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli)
        sensitivity =  first_term.real - second_term.real
    else: #A
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[1], basis_element_labels[0])
        second_term = phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli) \
                    - phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli)
        sensitivity =  first_term.imag + second_term.imag
    return sensitivity

def first_order_probability_correction(errorgen_dict, tableau, desired_bitstring):
    """
    Compute the first-order correction to the probability of the specified bit string.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.
    
    Returns
    -------
    correction : float
        float corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to first order).
    """
    
    num_random = random_support(tableau)
    scale = 1/2**(num_random) #TODO: This might overflow
    
    #now get the sum over the alphas and the error generator rate products needed.
    alpha_errgen_prods = [0]*len(errorgen_dict)
    
    for i, (lbl, rate) in enumerate(errorgen_dict.items()):
        alpha_errgen_prods[i] = alpha(lbl, tableau, desired_bitstring)*rate
    
    correction = scale*sum(alpha_errgen_prods)
    return correction

def stabilizer_probability(tableau, desired_bitstring):
    """
    Calculate the output probability for the specifed output bitstring.
    
    TODO: Should be able to do this more efficiently for many bitstrings
    by looking at the structure of the random support.
    
    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.
    
    Returns
    -------
    p : float
        probability of desired bitstring.
    """
    #compute what Gidney calls the tableau fidelity (which in this case gives the probability).
    return tableau_fidelity(tableau, bitstring_to_tableau(desired_bitstring))

def approximate_stabilizer_probability(errorgen_dict, circuit, desired_bitstring):
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.
    
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertable to a stim.Tableau.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.
    
    Returns
    -------
    p : float
        Approximate output probability for desired bitstring.
    """
    
    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    #recast keys to local stim ones if needed.
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = {_LSE.cast(lbl):val for lbl,val in errorgen_dict.items()}

    ideal_prob = stabilizer_probability(tableau, desired_bitstring)
    first_order_correction = first_order_probability_correction(errorgen_dict, tableau, desired_bitstring)
    return ideal_prob + first_order_correction

def approximate_stabilizer_probabilities(errorgen_dict, circuit):
    """
    Calculate the approximate probability distribution over all bitstrings using a first-order approximation.
    Note the size of this distribtion scales exponentially in the qubit count, so this is very inefficient for
    any more than a few qubits.

    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.
    
    circuit : `Circuit` or `stim.Tableau`
        A pygsti `Circuit` or a stim.Tableau to compute the output probability for. In either
        case this should be a Clifford circuit and convertable to a stim.Tableau.
    
    Returns
    -------
    p : float
        Approximate output probability for desired bitstring.
    """
    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    #get set of all bit strings
    num_qubits = len(tableau)
    bitstrings = ["".join(bitstring) for bitstring in product(['0','1'], repeat=num_qubits)]

    #initialize an array for the probabilities
    probs = _np.zeros(2**num_qubits)

    for i, bitstring in enumerate(bitstrings):
        probs[i] = approximate_stabilizer_probability(errorgen_dict, tableau, bitstring)

    return probs