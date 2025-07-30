"""
Tools for the propagation of error generators through circuits.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings
try:
    import stim
except ImportError:
    msg = "Stim is required for use of the error generator propagation tools module, " \
          "and it does not appear to be installed. If you intend to use this module please update" \
          " your environment."
    warnings.warn(msg)

import numpy as _np
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GEEL, LocalElementaryErrorgenLabel as _LEEL
from pygsti.baseobjs import QubitSpace as _QubitSpace
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis, ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.modelmembers.operations import LindbladErrorgen as _LinbladErrorgen
from pygsti.circuits import Circuit as _Circuit
from pygsti.tools.optools import create_elementary_errorgen_nqudit, state_to_dmvec
from functools import reduce
from itertools import chain, product
from math import factorial

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
            for error1a, error1a_val in errgen_layer_1.items():
                for error2, error2_val in second_order_comm_dict.items():
                    #only need a factor of 1/6 because new_errorgen_layer[1] is 1/2 the commutator 
                    weighta = (1/6)*error1a_val*error2_val

                    #avoid computing commutators which will be effectively zero.
                    if not abs(weighta) < truncation_threshold:
                        commuted_errgen_sublist = error_generator_commutator(error1a, error2, 
                                                                             weight=weighta, identity=identity)
                        commuted_errgen_list_1.extend(commuted_errgen_sublist)

            for error1b, error1b_val in errgen_layer_2.items():
                for error2, error2_val in second_order_comm_dict.items():
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
            B = third_order_comm_dict_1 #has a factor of 1/12 folded in already.
            C = third_order_comm_dict_2 #has a factor of -1/12 folded in already.
            D = fourth_order_comm_dict  #has a factor of -1/24 folded in already.
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
            #[X,F] = (-1/60); [Y,E] = (-1/60); [Y,F]= (1/30); [X,E]= (1/30); [Y,G] = (-1/10); [X,D] = (1/5)

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
                    weight = -(1/60)*error1_val*error2_val
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
    complete_errorgen_labels = errorgen_labels_by_order[0]
    for order_dict in errorgen_labels_by_order[1:]:
        complete_errorgen_labels.update(order_dict)

    #initialize a dictionary with requisite keys
    new_errorgen_layer_dict = {lbl: 0 for lbl in complete_errorgen_labels}

    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] += rate.real

    #Future: Possibly do one last truncation pass in case any of the different order cancel out when aggregated?

    return new_errorgen_layer_dict

def magnus_expansion(errorgen_layers, magnus_order=1, truncation_threshold=1e-14):

    """
    Function for computing the nth-order magnus expansion for a set of error generator layers.

    Parameters
    ----------
    errorgen_layers : list of dicts
        List of dictionaries of the error generator coefficients and rates for a circuit layer. 
        The error generator coefficients are represented using LocalStimErrorgenLabel.

    errorgen_transform_maps : dict
        Map giving the relationship between input error generators and their final
        value following propagation through the circuit. Needed to track any sign updates
        for terms with zero mean but nontrivial covariance.

    cov_func : 
        A function which maps tuples of elementary error generator labels at multiple times to
        a scalar quantity corresponding to the value of the covariance for that pair.
    
    magnus_order : int, optional (default 1)
        Order of the magnus expansion to apply. Currently supports up to third order.
    
    truncation_threshold : float, optional (default 1e-14)
        Threshold for which any error generators with magnitudes below this value
        are truncated.
        
    Returns
    -------
    magnus_expansion_dict : dict
        A dictionary with the same general structure as those in `errorgen_layers`, but with the
        rates combined according to the selected order of the magnus expansion.
    """

    new_errorgen_layer = []

    for curr_order in range(magnus_order):
        #first-order magnus terms:
        #\sum_{t1} A_{t1}
        if curr_order == 0:
            #Get a combined set of error generator coefficient labels for the list of dictionaries.
            current_combined_coeff_lbls = {key: None for key in chain(*errorgen_layers)}            

            first_order_dict = dict()
            #loop through the combined set of coefficient labels and add them to the new dictionary for the current BCH
            #approximation order. If present in both we sum the rates.
            for coeff_lbl in current_combined_coeff_lbls:
                #only add to the first order dictionary if the coefficient exceeds the truncation threshold.
                first_order_rate = sum([errgen_layer.get(coeff_lbl, 0) for errgen_layer in errorgen_layers])  
                if abs(first_order_rate) > truncation_threshold:
                    first_order_dict[coeff_lbl] = first_order_rate
            
            #allow short circuiting to avoid an expensive bunch of recombination logic when only using first order BCH
            #which will likely be a common use case.
            if magnus_order==1:
                return first_order_dict
            new_errorgen_layer.append(first_order_dict)
        
        #second-order magnus terms:
        #(1/2)\sum_{t1=1}^n \sum_{t2=1}^{t1-1} [A(t1), A(t2)]
        elif curr_order == 1:
            #construct a list of all of the pairs of error generator layers we need the
            #commutators for.
            errorgen_pairs = []
            for i in range(len(errorgen_layers)):
                for j in range(i):
                    errorgen_pairs.append((errorgen_layers[i], errorgen_layers[j]))
            
            #precompute an identity string for comparisons in commutator calculations.
            if errorgen_layers:
                for layer in errorgen_layers:
                    if layer:
                        identity = stim.PauliString('I'*len(next(iter(layer)).basis_element_labels[0]))
                        break
            
            #compute second-order BCH correction for each pair of error generators in the
            #errorgen_pairs list.
            commuted_errgen_list = []
            for errorgen_pair in errorgen_pairs:
                for error1, error1_val in errorgen_pair[0].items():
                    for error2, error2_val in errorgen_pair[1].items():
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

        #third order magnus terms
        #(1/6)*\sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} ( [A(t1), [A(t2), A(t3)]] - [A(t3), [A(t1), A(t2)]] )
        # -> (1/6)*\sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} [A(t1), [A(t2), A(t3)]]  
        #   -(1/6)*\sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} [A(t3), [A(t1), A(t2)]]
        #First term is zero when t2=t3, so last sum upper bound can be set to t2-1
        #Second term is zero when t1=t2, so second sum upperbound can be set to t1-1.
        #We've already computed the commutator [A(t1), A(t2)] in the second term (up to a factor of 1/2) and can reuse that here. 
        elif curr_order == 2:
            commuted_errgen_list_1 = []
            commuted_errgen_list_2 = []

            #(1/6) \sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} [A(t1), [A(t2), A(t3)]] #use linearity
            #-> (1/6) \sum_{t1=1}^{n} [A(t1), \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} [A(t2), A(t3)]]
            #when t1=t2 we pick up an extra factor of 1/2 from boundary effect in the discretization of the time-ordered integral.

            #this is a version of the running sum without the extra 1/2 from boundaries, in the time-ordered integral which is what will get propagated
            #forward through the computation.
            running_23_commutator_sum = {} 
            for i in range(len(errorgen_layers)): #t1
                new_23_commutator_terms = []
                j=i #new t2 value, can remove this and just replace j with i, keeping temporatily for clarity.
                for k in range(j): #t3
                    new_23_commutator_terms.extend(_error_generator_layer_pairwise_commutator(errorgen_layers[j], errorgen_layers[k], 
                                                                                              addl_weight=(1/12), 
                                                                                              identity=identity, 
                                                                                              truncation_threshold=truncation_threshold))
                #with the way terms are being accumulated it is always the case at this point that j=i, so we need the extra
                #factor of 1/2 on the new terms for the computation of the outer commutator with A(t1) with running_23_sum, 
                #but for future iterations we want to adjust the weights we added to undo this factor of 1/2 for later iterations.
                
                #loop through all of the elements of new_23_commutator_terms and instantiate any new keys in running_23_commutator_sum
                for error_tuple in new_23_commutator_terms:
                    if error_tuple[0] not in running_23_commutator_sum:
                        running_23_commutator_sum[error_tuple[0]] = 0

                #Now that keys are instantiated add all of these error generators to the working dictionary of updated error generators and weights.
                #There may be duplicates, which should be summed together.
                for error_tuple in new_23_commutator_terms:
                    running_23_commutator_sum[error_tuple[0]] += error_tuple[1]
                #truncate any terms which are below the truncation threshold following aggregation. 
                curr_iter_23_commutator_sum = {key: val for key, val in running_23_commutator_sum.items() if abs(val)>truncation_threshold}
                
                #and finally compute the commutator of the running sum with the t1 error generator layer
                commuted_errgen_list_1.extend(_error_generator_layer_pairwise_commutator(errorgen_layers[i], curr_iter_23_commutator_sum, 
                                                                                         identity=identity, 
                                                                                         truncation_threshold=truncation_threshold))
                #adjust the weights in running_23_commutator_sum to double to contribution added earlier bringing the weight from the Magnus expansion up to 1/6 for
                #future iterations.
                for error_tuple in new_23_commutator_terms:
                    running_23_commutator_sum[error_tuple[0]] += error_tuple[1]
                #truncate any terms which are below the truncation threshold following aggregation. 
                running_23_commutator_sum = {key: val for key, val in running_23_commutator_sum.items() if abs(val)>truncation_threshold}

            #TODO: Cache intermediate values for [A(t1), A(t2)] when doing the second-order computation to reuse here.            
            #-(1/6) \sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} [A(t3), [A(t1), A(t2)]] 
            #This sum can be reordered as follows (this was nonobvious to me until I confirmed explicitly)
            #-(1/6) \sum_{t3=1}^{n-1} \sum_{t2=t3}^{n-1} \sum_{t1=t2+1}^{n} [A(t3), [A(t1), A(t2)]]
            #-(1/6) \sum_{t3=1}^{n-1} \sum_{t1=t2+1}^{n} [A(t3), \sum_{t2=t3}^{n-1} [A(t1), A(t2)]] #applying linearity
            #when t3=t2 we pick up an extra factor of 1/2 from the discretization of the time-ordered integral. (see computation of previous term for implementation details).
            #The inner commutator sum can be accumulated in a running fashion, and this is easiest done if we run over the outer sum index in reverse.            
            running_12_commutator_sum = {}
            for k in range(len(errorgen_layers)-2, -1, -1): #t3
                new_12_commutator_terms = []
                j=k #new t2 value, can remove this and just replace j with k, keeping temporarily for clarity.
                for i in range(j+1, len(errorgen_layers)): #t1
                    new_12_commutator_terms.extend(_error_generator_layer_pairwise_commutator(errorgen_layers[i], errorgen_layers[j], 
                                                                                              addl_weight=(-1/12), identity=identity, 
                                                                                              truncation_threshold=truncation_threshold))
                #loop through all of the elements of new_12_commutator_terms and instantiate any new keys in running_12_commutator_sum
                for error_tuple in new_12_commutator_terms:
                    if error_tuple[0] not in running_12_commutator_sum:
                        running_12_commutator_sum[error_tuple[0]] = 0

                #Now that keys are instantiated add all of these error generators to the working dictionary of updated error generators and weights.
                #There may be duplicates, which should be summed together.
                for error_tuple in new_12_commutator_terms:
                    running_12_commutator_sum[error_tuple[0]] += error_tuple[1]
                #truncate any terms which are below the truncation threshold following
                #aggregation.
                curr_iter_12_commutator_sum = {key: val for key, val in running_12_commutator_sum.items() if abs(val)>truncation_threshold}

                #and finally compute the commutator of the running sum with the t3 error generator layer
                commuted_errgen_list_2.extend(_error_generator_layer_pairwise_commutator(errorgen_layers[k], curr_iter_12_commutator_sum, 
                                                                                         identity=identity, 
                                                                                         truncation_threshold=truncation_threshold))
                for error_tuple in new_12_commutator_terms:
                    running_12_commutator_sum[error_tuple[0]] += error_tuple[1]
                #truncate any terms which are below the truncation threshold following
                #aggregation.
                running_12_commutator_sum = {key: val for key, val in running_12_commutator_sum.items() if abs(val)>truncation_threshold}

            #finally combine the contents of commuted_errgen_list_1 and commuted_errgen_list_2 
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

        else: 
            raise NotImplementedError("Magnus expansions beyond third order are not implemented yet.")

    #Finally accumulate all of the dictionaries in new_errorgen_layer into a single one, summing overlapping terms.   
    errorgen_labels_by_order = [{key: None for key in order_dict} for order_dict in new_errorgen_layer]
    complete_errorgen_labels = errorgen_labels_by_order[0]
    for order_dict in errorgen_labels_by_order[1:]:
        complete_errorgen_labels.update(order_dict)

    #initialize a dictionary with requisite keys
    new_errorgen_layer_dict = {lbl: 0 for lbl in complete_errorgen_labels}

    for order_dict in new_errorgen_layer:
        for lbl, rate in order_dict.items():
            new_errorgen_layer_dict[lbl] += rate.real

    #Future: Possibly do one last truncation pass in case any of the different orders cancel out when aggregated?
    return new_errorgen_layer_dict

#TODO: Refactor a bunch of the code in this module to use this helper function.
#define a helper function to do a layerwise commutator accumulating all of the pairwise terms into a single list.
def _error_generator_layer_pairwise_commutator(errorgen_layer_1, errorgen_layer_2, addl_weight=1.0, identity=None, truncation_threshold=1e-14):
    commuted_errgen_list = []
    for error1, error1_val in errorgen_layer_1.items():
        for error2, error2_val in errorgen_layer_2.items():
            #get the list of error generator labels
            weight = addl_weight*error1_val*error2_val
            #avoid computing commutators which will be effectively zero.
            if abs(weight) < truncation_threshold:
                continue
            commuted_errgen_sublist = error_generator_commutator(error1, error2, 
                                                                weight= weight, identity=identity)
            commuted_errgen_list.extend(commuted_errgen_sublist)
    return commuted_errgen_list



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
    list of tuples. The first element of each tuple is a `LocalStimErrorgenLabel`s 
    corresponding to a component of the composition of the two input error generators.
    The second element is the weight of that term, additionally weighted by the specified
    value of `weight`.
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
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_C(P, Q, False, False, P_eq_Q)
            composed_errorgens.append((_LSE(new_eg_type, new_bels), addl_factor*w))
        else:
            PQ = pauli_product(P, Q)
            composed_errorgens.append((_LSE('H', [PQ[1]]), -1j*w*PQ[0]))
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_C(P, Q, False, False, P_eq_Q)
            composed_errorgens.append((_LSE(new_eg_type, new_bels), addl_factor*w))

    elif errorgen_1_type == 'H' and errorgen_2_type == 'S':
        #H_P[S_Q] P->errorgen_1_bel_0, Q -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_2_bel_0
        PQ = pauli_product(P, Q)
        PQ_ident = (PQ[1] == identity)
        PQ_eq_Q = (PQ[1]==Q)
        if P.commutes(Q):
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_A(PQ[1], Q, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -PQ[0]*addl_factor*w))
            composed_errorgens.append((_LSE('H', [P]), -w))   
        else: #if errorgen_1_bel_0 and errorgen_2_bel_0 only multiply to identity they are equal (in which case they commute).
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_C(PQ[1], Q, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -1j*PQ[0]*addl_factor*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))

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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))

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
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_A(PQ[1], P, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -PQ[0]*addl_factor*w))
            composed_errorgens.append((_LSE('H', [Q]), -w))   
        else: #if errorgen_1_bel_0 and errorgen_2_bel_0 only multiply to identity they are equal (in which case they commute).
            new_eg_type, new_bels, addl_factor = _ordered_new_bels_C(PQ[1], P, PQ_ident, False, PQ_eq_Q)
            if new_eg_type is not None:
                composed_errorgens.append((_LSE(new_eg_type, new_bels), -1j*PQ[0]*addl_factor*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))

            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))

            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), -1*APQ[0]*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*PQ[0]*addl_factor_2*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*addl_factor_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))

    elif errorgen_1_type == 'C' and errorgen_2_type == 'S': #TODO: This differs from S-C by just a few signs. Should be able to combine and significantly compress code.
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #TODO: Cases (2a,2b) and (2c,2d) only differ by the leading sign, can compress this code a bit.

    elif errorgen_1_type == 'C' and errorgen_2_type == 'C':
        #C_A,B[C_P,Q]: A -> errorgen_1_bel_0, B -> errorgen_1_bel_1, P -> errorgen_2_bel_0, Q -> errorgen_2_bel_1 
        A = errorgen_1_bel_0
        B = errorgen_1_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        #precompute commutation relations we'll need.
        com_PQ = P.commutes(Q)
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)
        com_BP = B.commutes(P)
        com_BQ = B.commutes(Q)

        #There are 64 separate cases, so this is gonna suck...
        if A.commutes(B):
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                AB = pauli_product(A, B)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                ABPQ = pauli_product(AB[0]*AB[1], PQ[0]*PQ[1])

                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                ABPQ_ident= (ABPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PQ_eq_AB = (PQ[1] == AB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(P, QAB[1], False, QAB_ident, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(PB[1], QA[1], PB_ident, QA_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(A, BPQ[1], False, BPQ_ident, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*PB[0]*QA[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), 1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                
            else: #[P,Q] !=0
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                AB = pauli_product(A, B)
                ABP = pauli_product(AB[0]*AB[1], P)
                ABQ = pauli_product(AB[0]*AB[1], Q)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                ABP_ident = (ABP[1] == identity)
                ABQ_ident = (ABQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                ABP_eq_Q = (ABP[1] == Q)
                ABQ_eq_P = (ABQ[1] == P) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*ABQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(ABP[1], Q, ABP_ident, False, ABP_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(ABQ[1], P, ABQ_ident, False, ABQ_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -ABP[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -ABQ[0]*addl_factor_3*w))
        else: #[A,B] != 0
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                PQB = pauli_product(PQ[0]*PQ[1], B)
                PQA = pauli_product(PQ[0]*PQ[1], A)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                PQB_ident = (PQB[1] == identity)
                PQA_ident = (PQA[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PQB_eq_A = (PQB[1] == A)
                PQA_eq_B = (PQA[1] == B) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*PQA[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQB[1], A, PQB_ident, False, PQB_eq_A)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(PQA[1], B, PQA_ident, False, PQA_eq_B)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -PQA[0]*addl_factor_3*w))
            else: #[P,Q]!=0
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0),-1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))

    elif errorgen_1_type == 'C' and errorgen_2_type == 'A':
        #C_A,B[A_P,Q]: A -> errorgen_1_bel_0, B -> errorgen_1_bel_1, P -> errorgen_2_bel_0, Q -> errorgen_2_bel_1 
        A = errorgen_1_bel_0
        B = errorgen_1_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        #precompute commutation relations we'll need.
        com_PQ = P.commutes(Q)
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)
        com_BP = B.commutes(P)
        com_BQ = B.commutes(Q)

        if A.commutes(B):
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                AB = pauli_product(A, B)
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), QAB[0]*addl_factor_3*w))
            else: #[P,Q]!=0
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                AB = pauli_product(A, B)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                ABPQ = pauli_product(AB[0]*AB[1], PQ[0]*PQ[1])

                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                ABPQ_ident= (ABPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PQ_eq_AB = (PQ[1] == AB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(PB[1], QA[1], PB_ident, QA_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*PB[0]*QA[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), QAB[0]*addl_factor_6*w))
        else: #[A,B] != 0
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
            else:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))

    elif errorgen_1_type == 'A' and errorgen_2_type == 'H':
        #A_{P,Q}[H_A] P->errorgen_1_bel_0, Q->errorgen_1_bel_1 A -> errorgen_2_bel_0
        A = errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_1_bel_1
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
                if not APQ_ident:
                    composed_errorgens.append((_LSE('H', [APQ[1]]), 1j*APQ[0]*w))
            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], Q, PA_ident, False, PA_eq_Q)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], P, QA_ident, False, QA_eq_P)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PQ[1], A, PQ_ident, False, PQ_eq_A)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*QA[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PQ[0]*addl_factor_2*w))

    elif errorgen_1_type == 'A' and errorgen_2_type == 'S':
        #A_P,Q[S_A] P->errorgen_1_bel_0, Q->errorgen_1_bel_1, A -> errorgen_2_bel_0
        P = errorgen_1_bel_0
        Q = errorgen_1_bel_1
        A = errorgen_2_bel_0

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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
            #Case 1d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1*addl_factor_1*w))
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
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))

            #Case 2b: {A,P}=0, {A,Q}=0
            elif not com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))

            #Case 2c: [A,P]=0, {A,Q}=0
            elif com_AP and not com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #Case 2d: {A,P}=0, [A,Q]=0
            elif not com_AP and com_AQ:
                new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QA[1], PA_ident, QA_ident, PA_eq_QA)
                new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(APQ[1], A, APQ_ident, False, False)
                new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(P, Q, False, False, False)
                if new_eg_type_0 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QA[0]*addl_factor_0*w))
                if new_eg_type_1 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -APQ[0]*addl_factor_1*w))
                if new_eg_type_2 is not None:
                    composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1*addl_factor_2*w))
            #TODO: Cases (2a,2b) and (2c,2d) only differ by the leading sign, can compress this code a bit.

    elif errorgen_1_type == 'A' and errorgen_2_type == 'C':
        #A_A,B[C_P,Q]: A -> errorgen_1_bel_0, B -> errorgen_1_bel_1, P -> errorgen_2_bel_0, Q -> errorgen_2_bel_1 
        A = errorgen_1_bel_0
        B = errorgen_1_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        #precompute commutation relations we'll need.
        com_PQ = P.commutes(Q)
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)
        com_BP = B.commutes(P)
        com_BQ = B.commutes(Q)

        if A.commutes(B):
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
            else: #[P,Q]!=0
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
        else: #[A,B] != 0
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                AB = pauli_product(A, B)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                ABPQ = pauli_product(AB[0]*AB[1], PQ[0]*PQ[1])

                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                ABPQ_ident= (ABPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PQ_eq_AB = (PQ[1] == AB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(PB[1], QA[1], PB_ident, QA_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*PB[0]*QA[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), -PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), 1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -1j*PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), 1j*QAB[0]*addl_factor_6*w))
            else:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                AB = pauli_product(A, B)
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*QAB[0]*addl_factor_3*w))

    elif errorgen_1_type == 'A' and errorgen_2_type == 'A':
        #A_A,B[A_P,Q]: A -> errorgen_1_bel_0, B -> errorgen_1_bel_1, P -> errorgen_2_bel_0, Q -> errorgen_2_bel_1 
        A = errorgen_1_bel_0
        B = errorgen_1_bel_1
        P = errorgen_2_bel_0
        Q = errorgen_2_bel_1
        #precompute commutation relations we'll need.
        com_PQ = P.commutes(Q)
        com_AP = A.commutes(P)
        com_AQ = A.commutes(Q)
        com_BP = B.commutes(P)
        com_BQ = B.commutes(Q)
        if A.commutes(B):
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
            else:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), BPQ[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*APQ[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*BPQ[0]*addl_factor_3*w))
        else:
            if com_PQ:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                AB = pauli_product(A, B)
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P)

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -QAB[0]*addl_factor_3*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), 1j*PAB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -1j*QAB[0]*addl_factor_3*w))
            else:
                #precompute some products we'll need.
                PA = pauli_product(P, A)
                QA = pauli_product(Q, A)
                PB = pauli_product(P, B)
                QB = pauli_product(Q, B)
                PQ = pauli_product(P, Q)
                AB = pauli_product(A, B)
                APQ = pauli_product(A, PQ[0]*PQ[1])
                BPQ = pauli_product(B, PQ[0]*PQ[1])
                PAB = pauli_product(P, AB[0]*AB[1])
                QAB = pauli_product(Q, AB[0]*AB[1])
                ABPQ = pauli_product(AB[0]*AB[1], PQ[0]*PQ[1])

                #precompute whether any of these products are identities.
                PA_ident  = (PA[1] == identity) 
                QA_ident  = (QA[1] == identity) 
                PB_ident  = (PB[1] == identity) 
                QB_ident  = (QB[1] == identity)
                APQ_ident = (APQ[1] == identity)
                BPQ_ident = (BPQ[1] == identity)
                PAB_ident = (PAB[1] == identity)
                QAB_ident = (QAB[1] == identity)
                ABPQ_ident= (ABPQ[1] == identity)
                #precompute which of the pairs of products might be equal
                PA_eq_QB = (PA[1] == QB[1])
                QA_eq_PB = (QA[1] == PB[1])
                PQ_eq_AB = (PQ[1] == AB[1])
                APQ_eq_B = (APQ[1] == B)
                BPQ_eq_A = (BPQ[1] == A)
                PAB_eq_Q = (PAB[1] == Q)
                QAB_eq_P = (QAB[1] == P) 

                if com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(PA[1], QB[1], PA_ident, QB_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -PA[0]*QB[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6),- 1j*QAB[0]*addl_factor_6*w))
                elif com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(PB[1], QA[1], PB_ident, QA_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*PB[0]*QA[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), -1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif not com_AP and com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                elif not com_AP and com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_C(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), -APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                elif not com_AP and not com_AQ and com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_A(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_C(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), 1j*QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_A(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_C(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_C(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), 1j*QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -QAB[0]*addl_factor_6*w))
                    if not ABPQ_ident:
                        composed_errorgens.append((_LSE('H', [ABPQ[1]]), -1j*ABPQ[0]*w))
                elif not com_AP and not com_AQ and not com_BP and not com_BQ:
                    new_eg_type_0, new_bels_0, addl_factor_0 = _ordered_new_bels_C(QB[1], PA[1], QB_ident, PA_ident, PA_eq_QB)
                    new_eg_type_1, new_bels_1, addl_factor_1 = _ordered_new_bels_C(QA[1], PB[1], QA_ident, PB_ident, QA_eq_PB)
                    new_eg_type_2, new_bels_2, addl_factor_2 = _ordered_new_bels_C(PQ[1], AB[1], False, False, PQ_eq_AB)
                    new_eg_type_3, new_bels_3, addl_factor_3 = _ordered_new_bels_A(APQ[1], B, APQ_ident, False, APQ_eq_B)
                    new_eg_type_4, new_bels_4, addl_factor_4 = _ordered_new_bels_A(BPQ[1], A, BPQ_ident, False, BPQ_eq_A)
                    new_eg_type_5, new_bels_5, addl_factor_5 = _ordered_new_bels_A(PAB[1], Q, PAB_ident, False, PAB_eq_Q)
                    new_eg_type_6, new_bels_6, addl_factor_6 = _ordered_new_bels_A(QAB[1], P, QAB_ident, False, QAB_eq_P)
                    if new_eg_type_0 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_0, new_bels_0), QB[0]*PA[0]*addl_factor_0*w))
                    if new_eg_type_1 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_1, new_bels_1), -QA[0]*PB[0]*addl_factor_1*w))
                    if new_eg_type_2 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_2, new_bels_2), -PQ[0]*AB[0]*addl_factor_2*w))
                    if new_eg_type_3 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_3, new_bels_3), 1j*APQ[0]*addl_factor_3*w))
                    if new_eg_type_4 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_4, new_bels_4), -1j*BPQ[0]*addl_factor_4*w))
                    if new_eg_type_5 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_5, new_bels_5), 1j*PAB[0]*addl_factor_5*w))
                    if new_eg_type_6 is not None:
                        composed_errorgens.append((_LSE(new_eg_type_6, new_bels_6), -1j*QAB[0]*addl_factor_6*w))

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
            addl_factor = 1
    else:
        if second_pauli_ident:
            new_eg_type = 'H'
            new_bels = [pauli1]
            addl_factor = -1
        else:
            new_eg_type = 'A'
            new_bels, addl_factor = ([pauli1, pauli2], 1) if stim_pauli_string_less_than(pauli1, pauli2) else ([pauli2, pauli1], -1)
    return new_eg_type, new_bels, addl_factor

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
        addl_factor = 2
    else:
        new_eg_type = 'C'
        addl_factor = 1
        new_bels = [pauli1, pauli2] if stim_pauli_string_less_than(pauli1, pauli2) else [pauli2, pauli1]
    return new_eg_type, new_bels, addl_factor

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

def iterative_error_generator_composition(errorgen_labels, rates):
    """
    Iteratively compute error generator compositions. Each error generator
    composition in general returns a list of multiple new error generators,
    so this function manages the distribution and recursive application
    of the compositions for two-or-more error generator labels.
    
    Parameters
    ----------
    errorgen_labels : tuple of `LocalStimErrorgenLabel`
        A tuple of the elementary error generator labels to be composed.
    
    rates : tuple of float
        A tuple of corresponding error generator rates of the same length as the tuple
        of error generator labels.

    Returns
    -------
    List of tuples, the first element of each tuple is a `LocalStimErrorgenLabel`.
    The second element of each tuple is the final rate for that term.
    """

    if len(errorgen_labels) == 1:
        return [(errorgen_labels[0], rates[0])]
    else:
        label_tuples_to_process = [errorgen_labels]
        rate_tuples_to_process = [rates]
    
    fully_processed_label_rate_tuples = []    
    while label_tuples_to_process:
        new_label_tuples_to_process = []
        new_rate_tuples_to_process = []

        for label_tup, rate_tup in zip(label_tuples_to_process, rate_tuples_to_process):
            #grab the last two elements of each of these and do the composition.
            new_labels_and_rates = error_generator_composition(label_tup[-2], label_tup[-1], rate_tup[-2]*rate_tup[-1])

            #if the new labels and rates sum to zero overall then we can kill this branch of the tree.
            aggregated_labels_and_rates_dict = dict()
            for lbl, rate in new_labels_and_rates:
                if aggregated_labels_and_rates_dict.get(lbl, None) is None:
                    aggregated_labels_and_rates_dict[lbl] = rate
                else:
                    aggregated_labels_and_rates_dict[lbl] += rate
            if all([abs(val)<1e-15 for val in aggregated_labels_and_rates_dict.values()]):
                continue

            label_tup_remainder = label_tup[:-2]
            rate_tup_remainder = rate_tup[:-2]
            if label_tup_remainder:
                for new_label, new_rate in aggregated_labels_and_rates_dict.items():
                    new_label_tup = label_tup_remainder + (new_label,)
                    new_rate_tup = rate_tup_remainder + (new_rate,)
                    new_label_tuples_to_process.append(new_label_tup)
                    new_rate_tuples_to_process.append(new_rate_tup)
            else:
                for new_label_rate_tup in aggregated_labels_and_rates_dict.items():
                    fully_processed_label_rate_tuples.append(new_label_rate_tup)
        label_tuples_to_process = new_label_tuples_to_process
        rate_tuples_to_process = new_rate_tuples_to_process  
    
    return fully_processed_label_rate_tuples

#Helper functions for doing numeric commutators, compositions and BCH.

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

def bch_numerical(propagated_errorgen_layers, error_propagator, bch_order=1):
    """
    Iteratively compute effective error generator layer produced by applying the BCH approximation
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of numpy.ndarrays
        List of the error generator layers to combine using the BCH approximation (in circuit ordering)

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the BCH calculation.

    bch_order : int, optional (default 1)
        Order of the BCH approximation to apply (up to 5 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of the iterative application of the BCH
        approximation.
    """
    #Need to build an appropriate basis for getting the error generator matrices.
    #accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    #only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    errorgen_lbl_matrix_dict = {lbl:mat for lbl,mat in zip(errorgen_basis.labels, errorgen_basis.elemgen_matrices)}
    
    #iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    #initialize a matrix for storing the result of doing BCH.
    bch_result = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    if len(errorgen_layer_mats)==1:
        return errorgen_layer_mats[0]
        
    #otherwise iterate through in reverse order (the propagated layers are
    #in circuit ordering and not matrix multiplication ordering at the moment)
    #and combine the terms pairwise
    combined_err_layer = errorgen_layer_mats[-1]
    for i in range(len(errorgen_layer_mats)-2, -1, -1):
        combined_err_layer = pairwise_bch_numerical(combined_err_layer, errorgen_layer_mats[i], order=bch_order)
        
    return combined_err_layer  

def pairwise_bch_numerical(mat1, mat2, order=1):
    """
    Helper function for doing the numerical BCH in a pairwise fashion. Note this function is primarily intended
    for numerical validations as part of testing infrastructure.
    """
    bch_result = _np.zeros(mat1.shape, dtype=_np.complex128)
    if order >= 1:
        bch_result += mat1 + mat2
    if order >= 2:
        commutator12 = _matrix_commutator(mat1, mat2)
        bch_result += .5*commutator12
    if order >= 3:
        commutator112 = _matrix_commutator(mat1, commutator12)
        commutator212 = _matrix_commutator(mat2, commutator12)
        bch_result += (1/12)*(commutator112-commutator212)
    if order >= 4:
        commutator2112 = _matrix_commutator(mat2, commutator112)
        bch_result += (-1/24)*commutator2112
    if order >= 5:
        commutator1112 = _matrix_commutator(mat1, commutator112)
        commutator2212 = _matrix_commutator(mat2, commutator212)
        
        commutator22212 = _matrix_commutator(mat2, commutator2212)
        commutator11112 = _matrix_commutator(mat1, commutator1112)
        commutator12212 = _matrix_commutator(mat1, commutator2212)
        commutator21112 = _matrix_commutator(mat2, commutator1112)
        commutator21212 = _matrix_commutator(mat2, _matrix_commutator(mat1, commutator212))
        commutator12112 = _matrix_commutator(mat1, commutator2112)
        
        bch_result += (-1/720)*(commutator11112 - commutator22212)
        bch_result += (1/360)*(commutator21112 - commutator12212)
        bch_result += (1/120)*(commutator21212 - commutator12112)
    return bch_result

def magnus_numerical(propagated_errorgen_layers, error_propagator, magnus_order=1):
    """
    Compute effective error generator layer produced by applying the magnus expansions
    to the list of input error generator matrices. Note this is primarily intended
    as part of testing and validation infrastructure.

    Parameters
    ----------
    propagated_errorgen_layers : list of dictionaries
        List of the error generator layers (in circuit ordering) in the form of dictionaries
        whose keys are elementary error generator labels and whose values are their corresponding
        rates. These dictionaries are in the format produced by the `ErrorGeneratorPropagator` class's
        `propagate_errorgens` method.

    error_propagator : `ErrorGeneratorPropagator`
        An `ErrorGeneratorPropagator` instance to use as part of the Magnus calculation.

    magnus_order : int, optional (default 1)
        Order of the Magnus expansion to apply (up to 3 is supported currently).

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the result of Magnus expansion.
    """

    #Need to build an appropriate basis for getting the error generator matrices.
    #accumulate the error generator coefficients needed.
    collected_coeffs = []
    for layer in propagated_errorgen_layers:
        for coeff in layer.keys():
            collected_coeffs.append(coeff.to_local_eel())
    #only want the unique ones.
    unique_coeffs = list(set(collected_coeffs))
    
    num_qubits = len(error_propagator.model.state_space.qubit_labels)
    
    errorgen_basis = _ExplicitElementaryErrorgenBasis(_QubitSpace(num_qubits), unique_coeffs, basis_1q=_BuiltinBasis('PP', 4))
    
    #iterate through each of the propagated error generator layers and turn these into dense numpy arrays
    errorgen_layer_mats = []
    for layer in propagated_errorgen_layers:
        errorgen_layer_mats.append(error_propagator.errorgen_layer_dict_to_errorgen(layer, mx_basis='pp'))
    
    #initialize a matrix for storing the result of doing magnus.
    magnus = _np.zeros((4**num_qubits, 4**num_qubits), dtype=_np.complex128)
    
    for curr_order in range(magnus_order):
        #first-order magnus terms:
        #\sum_{t1} A_{t1}
        if curr_order == 0:
            for mat in errorgen_layer_mats:
                magnus += mat
        
        #second-order magnus terms:
        #(1/2) \sum_{t1=1}^n \sum_{t2=1}^{t2} [A(t1), A(t2)]
        elif curr_order == 1:
            errorgen_pairs = []
            for i in range(len(errorgen_layer_mats)):
                for j in range(i):
                    errorgen_pairs.append((errorgen_layer_mats[i], errorgen_layer_mats[j]))
            for errorgen_pair in errorgen_pairs:
                magnus += .5*_matrix_commutator(errorgen_pair[0], errorgen_pair[1])
        
        #third-order magnus terms:
        #(1/6) \sum_{t1=1}^{n} \sum_{t2=1}^{t1} \sum_{t3=1}^{t2} ([A(t1), [A(t2),A(t3)]] + [A(t3), [A(t2), A(t1)]])
        elif curr_order == 2:
            for i in range(len(errorgen_layer_mats)):
                for j in range(i+1):
                    for k in range(j+1):
                        if i==j:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[i], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[k]))
                        if j==k:
                            magnus += (1/12)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
                        else:
                            magnus += (1/6)*_matrix_commutator(errorgen_layer_mats[k], _matrix_commutator(errorgen_layer_mats[j], errorgen_layer_mats[i]))
        else:
            raise NotImplementedError('Magnus beyond third order is not currently implemented.')
        
    return magnus  


def _matrix_commutator(mat1, mat2):
    return mat1@mat2 - mat2@mat1

def iterative_error_generator_composition_numerical(errorgen_labels, rates, errorgen_matrix_dict=None, num_qubits=None):
    """
    Iteratively compute error generator compositions. The function computes a dense representation of this composition
    numerically and is primarily intended as part of testing infrastructure.
    
    Parameters
    ----------
    errorgen_labels : tuple of `LocalStimErrorgenLabel`
        A tuple of the elementary error generator labels to be composed.
    
    rates : tuple of float
        A tuple of corresponding error generator rates of the same length as the tuple
        of error generator labels.
        
    errorgen_matrix_dict : dict, optional (default None)
        An optional dictionary mapping `ElementaryErrorgenLabel`s to numpy arrays for their dense representation.
        If not specified this will be constructed from scratch each call, so specifying this can provide a performance
        benefit.

    num_qubits : int, optional (default None)
        Number of qubits for the error generator commutator being computed. Only required if `errorgen_matrix_dict` is None.

    Returns
    -------
    numpy.ndarray
        Dense numpy array representation of the super operator corresponding to the iterated composition written in 
        the standard basis.
    """
    
    if errorgen_matrix_dict is None:
        #create an error generator basis.
        errorgen_basis = _CompleteElementaryErrorgenBasis('PP', _QubitSpace(num_qubits), default_label_type='local')
        
        #use this basis to construct a dictionary from error generator labels to their
        #matrices.
        errorgen_lbls = errorgen_basis.labels
        errorgen_matrix_dict = {lbl: mat for lbl, mat in zip(errorgen_lbls, errorgen_basis.elemgen_matrices)}

    composition = errorgen_matrix_dict[errorgen_labels[0]]
    for lbl in errorgen_labels[1:]:
        composition = composition@errorgen_matrix_dict[lbl]
    composition *= _np.prod(rates)
    return composition

#-----------First-Order Approximate Error Generator Probabilities and Expectation Values---------------#

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

def pauli_phase_update(pauli, bitstring, dual=False):
    """
    Takes as input a pauli and a bit string and computes the output bitstring
    and the overall phase that bit string accumulates.
    
    Parameters
    ----------
    pauli : str or stim.PauliString
        Pauli to apply
    
    bitstring : str
        String of 0's and 1's representing the bit string to apply the pauli to.
    
    dual : bool, optional (default False)
        If True then then the pauli is acting to the left on a row vector.
    Returns
    -------
    Tuple whose first element is the phase accumulated, and whose second element
    is a string corresponding to the updated bit string.
    """
    
    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)
    
    bitstring = [False if bit=='0' else True for bit in bitstring]
    if not dual:
        #list of phase correction for each pauli (conditional on 0)
        #Read [I, X, Y, Z]
        pauli_phases_0 = [1, 1, 1j, 1]
        
        #list of the phase correction for each pauli (conditional on 1)
        #Read [I, X, Y, Z]
        pauli_phases_1 = [1, 1, -1j, -1]
    else:
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
    phase1, bitstring1 = pauli_phase_update(eff_P, all_zeros, dual=True)
    phase2, bitstring2 = pauli_phase_update(eff_Q, all_zeros)

    #get the amplitude of these two bitstrings in the stabilizer state.
    amp1 = amplitude_of_state(tableau, bitstring1)
    amp2 = amplitude_of_state(tableau, bitstring2).conjugate()  #The second amplitude also needs a complex conjugate applied
        
    #now apply the phase corrections. 
    amp1*=phase1
    amp2*=phase2
     
    #calculate phi.
    phi = amp1*amp2
    
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

#helper function for numerically computing phi, primarily used for testing.
def phi_numerical(tableau, desired_bitstring, P, Q):
    """
    This function computes a quantity whose value is used in expression for the sensitivity of probabilities to error generators.
    (This version does this calculation numerically and is primarily intended for testing infrastructure.)
    
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
    initial_pauli_string = stim.PauliString(''.join(['I' if bit=='0' else 'X' for bit in desired_bitstring])).to_unitary_matrix(endian = 'big')
    

    #map P and Q to stim.PauliString if needed.
    if isinstance(P, str):
        P = stim.PauliString(P)
    if isinstance(Q, str):
        Q = stim.PauliString(Q)
    
    stabilizer_state = tableau.to_state_vector(endian = 'big')
    stabilizer_state.reshape((len(stabilizer_state),1))
    #combine this initial pauli string with the two input paulis
    eff_P = initial_pauli_string@P.to_unitary_matrix(endian = 'big')
    eff_Q = Q.to_unitary_matrix(endian = 'big')@initial_pauli_string
    
    #now get the bit strings which need their amplitudes extracted from the input stabilizer state and get
    #the corresponding phase corrections.
    #all_zeros = '0'*len(eff_P)
    all_zeros = _np.zeros((2**len(desired_bitstring),1))
    all_zeros[0] = 1  
    #calculate phi.
    #The second amplitude also needs a complex conjugate applied
    phi = (all_zeros.T@eff_P@stabilizer_state) * (stabilizer_state.conj().T@eff_Q@all_zeros)
    
    num_random = random_support(tableau)
    scale = 2**(num_random)

    return phi*scale

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
        sensitivity = (phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[0]) \
                    - phi(tableau, desired_bitstring, identity_pauli, identity_pauli)).real
    elif errgen_type == 'C': 
        first_term = 2*phi(tableau, desired_bitstring, basis_element_labels[0], basis_element_labels[1])
        sensitivity = first_term.real
        if basis_element_labels[0].commutes(basis_element_labels[1]):
            second_term = 2*phi(tableau, desired_bitstring, basis_element_labels[0]*basis_element_labels[1], identity_pauli)
            sensitivity -= second_term.real
    else: #A
        first_term = phi(tableau, desired_bitstring, basis_element_labels[1], basis_element_labels[0])
        if not basis_element_labels[0].commutes(basis_element_labels[1]):
            second_term = phi(tableau, desired_bitstring, basis_element_labels[1]*basis_element_labels[0], identity_pauli)
            sensitivity = 2*((first_term + second_term).imag)
        else:
            sensitivity = 2*first_term.imag
    return sensitivity

def alpha_numerical(errorgen, tableau, desired_bitstring):
    """
    First-order error generator sensitivity function for probability. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    desired_bitstring : str
        Bit string to calculate the sensitivity for.
    """
    
    #get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    #also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    #also need a superbra for the desired bitstring.
    desired_bitstring_vec = _np.zeros(2**len(desired_bitstring))
    desired_bitstring_vec[_bitstring_to_int(desired_bitstring)] = 1
    desired_bitstring_dmvec = state_to_dmvec(desired_bitstring_vec)
    desired_bitstring_dmvec.reshape((1, len(desired_bitstring_dmvec)))
    num_random = random_support(tableau)
    scale = 2**(num_random)
    
    #compute the needed trace inner product.
    alpha = _np.real_if_close(scale*(desired_bitstring_dmvec.conj().T@errorgen_superop@stabilizer_state_dmvec))
    
    return alpha

def alpha_pauli(errorgen, tableau, pauli):
    """
    First-order error generator sensitivity function for pauli expectations.
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.
    """
    
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    
    errgen_type = errorgen.errorgen_type
    basis_element_labels = errorgen.basis_element_labels
    
    if not isinstance(basis_element_labels[0], stim.PauliString):
        basis_element_labels = tuple([stim.PauliString(lbl) for lbl in basis_element_labels])
    
    identity_pauli = stim.PauliString('I'*len(basis_element_labels[0]))
    
    if errgen_type == 'H':
        pauli_bel_0_comm = com(pauli, basis_element_labels[0])
        if pauli_bel_0_comm is not None:
            sign = -1j*pauli_bel_0_comm[0]
            expectation  = sim.peek_observable_expectation(pauli_bel_0_comm[1])
            return _np.real_if_close(sign*expectation)
        else: 
            return 0 
    elif errgen_type == 'S':
        if pauli.commutes(basis_element_labels[0]):
            return 0
        else:
            expectation  = sim.peek_observable_expectation(pauli)
            return _np.real_if_close(-2*expectation)
    elif errgen_type == 'C': 
        A = basis_element_labels[0]
        B = basis_element_labels[1]
        com_AP = A.commutes(pauli)
        com_BP = B.commutes(pauli) #TODO: can skip computing this in some cases for minor performance boost.
        if A.commutes(B):
            if com_AP:
                return 0
            else:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(-4*expectation)
        else: #{A,B} = 0
            if com_AP:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(-2*expectation)
            else:
                if com_BP:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(2*expectation)
                else:
                    return 0
    else: #A
        A = basis_element_labels[0]
        B = basis_element_labels[1]
        com_AP = A.commutes(pauli)
        com_BP = B.commutes(pauli) #TODO: can skip computing this in some cases for minor performance boost.
        if A.commutes(B):
            if com_AP:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(1j*2*expectation)
            else:
                if com_BP:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(-1j*2*expectation)
                else:
                    return 0
        else: #{A,B} = 0
            if com_AP:
                return 0
            else:
                if com_BP:
                    return 0
                else:
                    ABP = pauli_product(A*B, pauli)
                    expectation = ABP[0]*sim.peek_observable_expectation(ABP[1])
                    return _np.real_if_close(1j*4*expectation)

def alpha_pauli_numerical(errorgen, tableau, pauli):
    """
    First-order error generator sensitivity function for pauli expectatons. This implementation calculates
    this quantity numerically, and as such is primarily intended for used as parting of testing
    infrastructure. 
    
    Parameters
    ----------
    errorgen : `ElementaryErrorgenLabel`
        Error generator label for which to calculate sensitivity.
    
    tableau : stim.Tableau
        Stim Tableau corresponding to the stabilizer state to calculate the sensitivity for.
        
    pauli : stim.PauliString
        Pauli to calculate the sensitivity for.
    """
    
    #get the stabilizer state corresponding to the tableau.
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    #also get the superoperator (in the standard basis) corresponding to the elementary error generator
    if isinstance(errorgen, _LSE):
        local_eel = errorgen.to_local_eel()
    elif isinstance(errorgen, _GEEL):
        local_eel = _LEEL.cast(errorgen)
    else:
        local_eel = errorgen
    
    errgen_type = local_eel.errorgen_type
    basis_element_labels = local_eel.basis_element_labels
    basis_1q = _BuiltinBasis('PP', 4)
    errorgen_superop = create_elementary_errorgen_nqudit(errgen_type, basis_element_labels, basis_1q, normalize=False, sparse=False,
                                                         tensorprod_basis=False)
    
    #finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    #flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    #compute the needed trace inner product.
    alpha = _np.real_if_close(pauli_vec.conj().T@errorgen_superop@stabilizer_state_dmvec).item()
    
    return alpha

def _bitstring_to_int(bitstring) -> int:
    if isinstance(bitstring, str):
        # If the input is a string, convert it directly
        return int(bitstring, 2)
    elif isinstance(bitstring, tuple):
        # If the input is a tuple, join the elements to form a string
        return int(''.join(bitstring), 2)
    else:
        raise ValueError("Input must be either a string or a tuple of '0's and '1's")

def stabilizer_probability_correction(errorgen_dict, tableau, desired_bitstring, order = 1, truncation_threshold = 1e-14):
    """
    Compute the kth-order correction to the probability of the specified bit string.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    desired_bitstring : str
        String of 0's and 1's corresponding to the output bitstring being measured.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    Returns
    -------
    correction : float
        float corresponding to the correction to the output probability for the
        desired bitstring induced by the error generator (to specified order).
    """
    
    num_random = random_support(tableau)
    scale = 1/2**(num_random) #TODO: This might overflow
    
    #do the first order correction separately since it doesn't require composition logic:
    #now get the sum over the alphas and the error generator rate products needed.
    alpha_errgen_prods = _np.zeros(len(errorgen_dict))
    

    for i, (lbl, rate) in enumerate(errorgen_dict.items()):
        if abs(rate) > truncation_threshold:
            alpha_errgen_prods[i] = alpha(lbl, tableau, desired_bitstring)*rate
    correction = scale*_np.sum(alpha_errgen_prods)
    if order > 1:
        #The order of the approximation determines the combinations of error generators
        #which need to be composed. (given by cartesian products of labels in errorgen_dict).
        labels_by_order = [list(product(errorgen_dict.keys(), repeat = i+1)) for i in range(1,order)]
        #Get a similar structure for the corresponding rates
        rates_by_order = [list(product(errorgen_dict.values(), repeat = i+1)) for i in range(1,order)]
        for current_order, (current_order_labels, current_order_rates) in enumerate(zip(labels_by_order, rates_by_order), start=2):
            current_order_scale = 1/factorial(current_order)
            composition_results = []
            for label_tup, rate_tup in zip(current_order_labels, current_order_rates):
                composition_results.extend(iterative_error_generator_composition(label_tup, rate_tup))
            #aggregate together any overlapping terms in composition_results
            composition_results_dict = dict()
            for lbl, rate in composition_results:
                if composition_results_dict.get(lbl,None) is None:
                    composition_results_dict[lbl] = rate
                else:
                    composition_results_dict[lbl] += rate
            alpha_errgen_prods = _np.zeros(len(composition_results_dict))
            for i, (lbl, rate) in enumerate(composition_results_dict.items()):
                if current_order_scale*abs(rate) > truncation_threshold:
                    sensitivity = alpha(lbl, tableau, desired_bitstring)
                    alpha_errgen_prods[i] = _np.real_if_close(sensitivity*rate)
            correction += current_order_scale*scale*_np.sum(alpha_errgen_prods)

    return correction

#TODO: The implementations for the pauli expectation value correction and probability correction
#are basically identical modulo some additional scale factors and the alpha function used. Should be able to combine
#the implementations into one function.
def stabilizer_pauli_expectation_correction(errorgen_dict, tableau, pauli, order = 1, truncation_threshold = 1e-14):
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    tableau : stim.Tableau
        Stim tableau corresponding to a particular stabilizer state being measured.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    
    #do the first order correction separately since it doesn't require composition logic:
    #now get the sum over the alphas and the error generator rate products needed.
    alpha_errgen_prods = _np.zeros(len(errorgen_dict))
    
    for i, (lbl, rate) in enumerate(errorgen_dict.items()):
        if abs(rate) > truncation_threshold:
            alpha_errgen_prods[i] = alpha_pauli(lbl, tableau, pauli)*rate
    correction = _np.sum(alpha_errgen_prods)
    if order > 1:
        #The order of the approximation determines the combinations of error generators
        #which need to be composed. (given by cartesian products of labels in errorgen_dict).
        labels_by_order = [list(product(errorgen_dict.keys(), repeat = i+1)) for i in range(1,order)]
        #Get a similar structure for the corresponding rates
        rates_by_order = [list(product(errorgen_dict.values(), repeat = i+1)) for i in range(1,order)]
        for current_order, (current_order_labels, current_order_rates) in enumerate(zip(labels_by_order, rates_by_order), start=2):
            current_order_scale = 1/factorial(current_order)
            composition_results = []
            for label_tup, rate_tup in zip(current_order_labels, current_order_rates):
                composition_results.extend(iterative_error_generator_composition(label_tup, rate_tup))
            #aggregate together any overlapping terms in composition_results
            composition_results_dict = dict()
            for lbl, rate in composition_results:
                if composition_results_dict.get(lbl,None) is None:
                    composition_results_dict[lbl] = rate
                else:
                    composition_results_dict[lbl] += rate
            alpha_errgen_prods = _np.zeros(len(composition_results_dict))
            for i, (lbl, rate) in enumerate(composition_results_dict.items()):
                if current_order_scale*abs(rate) > truncation_threshold:
                    sensitivity = alpha_pauli(lbl, tableau, pauli)
                    alpha_errgen_prods[i] = _np.real_if_close(sensitivity*rate)
            correction += current_order_scale*_np.sum(alpha_errgen_prods)

    return correction

def stabilizer_pauli_expectation_correction_numerical(errorgen_dict, errorgen_propagator, circuit, pauli, order = 1):
    """
    Compute the kth-order correction to the expectation value of the specified pauli.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        Circuit the expectation value is being measured against.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value correction for.

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    Returns
    -------
    correction : float
        float corresponding to the correction to the expectation value for the
        selected pauli operator induced by the error generator (to specified order).
    """
    tableau = circuit.convert_to_stim_tableau()
    
    stabilizer_state = tableau.to_state_vector(endian='big')
    stabilizer_state_dmvec = state_to_dmvec(stabilizer_state)
    stabilizer_state_dmvec.reshape((len(stabilizer_state_dmvec),1))
    
    #also get the superoperator (in the standard basis) corresponding to the taylor series
    #expansion of the specified error generator dictionary.
    taylor_expanded_errorgen = error_generator_taylor_expansion_numerical(errorgen_dict, errorgen_propagator, order=order, mx_basis='std')
    
    #finally need the superoperator for the selected pauli.
    pauli_unitary = pauli.to_unitary_matrix(endian='big')
    #flatten this row-wise
    pauli_vec = _np.ravel(pauli_unitary)
    pauli_vec.reshape((len(pauli_vec),1))
    
    expectation_correction = _np.linalg.multi_dot([pauli_vec.conj().T, taylor_expanded_errorgen,stabilizer_state_dmvec]).item()
    return expectation_correction

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

def stabilizer_pauli_expectation(tableau, pauli):
    """
    Calculate the output probability for the specifed output bitstring.
      
    Parameters
    ----------
    tableau : stim.Tableau
        Stim tableau for the stabilizer state being measured.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value for.
    
    Returns
    -------
    expected_value : float
        Expectation value of specified pauli
    """
    if pauli.sign != 1:
        pauli_sign = pauli.sign
        unsigned_pauli = pauli/pauli_sign  
    else:
        pauli_sign = 1
        unsigned_pauli = pauli
        
    sim = stim.TableauSimulator()
    sim.set_inverse_tableau(tableau**-1)
    expectation  = pauli_sign*sim.peek_observable_expectation(unsigned_pauli)
    return expectation

def approximate_stabilizer_probability(errorgen_dict, circuit, desired_bitstring, order=1, truncation_threshold=1e-14):
    """
    Calculate the approximate probability of a desired bit string using an nth-order taylor series approximation.
    
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
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
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
    correction = stabilizer_probability_correction(errorgen_dict, tableau, desired_bitstring, order, truncation_threshold)
    return ideal_prob + correction

def approximate_stabilizer_pauli_expectation(errorgen_dict, circuit, pauli, order=1, truncation_threshold=1e-14):
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
        
    pauli : str or stim.PauliString
        Pauli operator to compute expectation value for.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
    Returns
    -------
    expectation_value : float
        Approximate expectation value for desired pauli.
    """
    
    if isinstance(circuit, _Circuit):
        tableau = circuit.convert_to_stim_tableau()
    elif isinstance(circuit, stim.Tableau):
        tableau = circuit
    else:
        raise ValueError('`circuit` should either be a pygsti `Circuit` or a stim.Tableau.')

    if isinstance(pauli, str):
        pauli = stim.PauliString(pauli)

    #recast keys to local stim ones if needed.
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = {_LSE.cast(lbl):val for lbl,val in errorgen_dict.items()}

    ideal_expectation = stabilizer_pauli_expectation(tableau, pauli)
    correction = stabilizer_pauli_expectation_correction(errorgen_dict, tableau, pauli, order, truncation_threshold)
    return ideal_expectation + correction

def approximate_stabilizer_pauli_expectation_numerical(errorgen_dict, errorgen_propagator, circuit, pauli, order=1):
    """
    Calculate the approximate probability of a desired bit string using a first-order approximation.
    This function performs the corrections numerically and so it primarily intended for testing
    infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `ElementaryErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    circuit : `Circuit`
        A pygsti `Circuit` or a stim.Tableau to compute the output pauli expectation value for.
        
    pauli : stim.PauliString
        Pauli operator to compute expectation value for.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
    Returns
    -------
    expectation_value : float
        Approximate expectation value for desired pauli.
    """
    
    tableau = circuit.convert_to_stim_tableau()

    #recast keys to local stim ones if needed.
    first_lbl = next(iter(errorgen_dict))
    if isinstance(first_lbl, (_GEEL, _LEEL)):
        errorgen_dict = {_LSE.cast(lbl):val for lbl,val in errorgen_dict.items()}

    ideal_expectation = stabilizer_pauli_expectation(tableau, pauli)
    correction = stabilizer_pauli_expectation_correction_numerical(errorgen_dict, errorgen_propagator, circuit, pauli, order)
    return ideal_expectation + correction

def approximate_stabilizer_probabilities(errorgen_dict, circuit, order=1, truncation_threshold=1e-14):
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

    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding error generator rates
        are below this value. (Used internally in computation of probability corrections)
    
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
        probs[i] = approximate_stabilizer_probability(errorgen_dict, tableau, bitstring, order, truncation_threshold)

    return probs

def error_generator_taylor_expansion(errorgen_dict, order = 1, truncation_threshold = 1e-14):
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity).
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.
    
    truncation_threshold : float, optional (default 1e-14)
        Optional threshold used to truncate corrections whose corresponding rates
        are below this value.

    Returns
    -------
    list of dictionaries
        List of dictionaries whose keys are error generator labels and whose values are rates (including
        whatever scaling comes from order of taylor expansion). Each list corresponds to an order
        of the taylor expansion.
    """
       
 
    taylor_order_terms = [dict() for _ in range(order)]

    for lbl, rate in errorgen_dict.items():
        if abs(rate) > truncation_threshold:
            taylor_order_terms[0][lbl] = rate

    if order > 1:
        #The order of the approximation determines the combinations of error generators
        #which need to be composed. (given by cartesian products of labels in errorgen_dict).
        labels_by_order = [list(product(errorgen_dict.keys(), repeat = i+1)) for i in range(1,order)]
        #Get a similar structure for the corresponding rates
        rates_by_order = [list(product(errorgen_dict.values(), repeat = i+1)) for i in range(1,order)]
        for current_order, (current_order_labels, current_order_rates) in enumerate(zip(labels_by_order, rates_by_order), start=2):
            order_scale = 1/factorial(current_order)
            composition_results = []
            for label_tup, rate_tup in zip(current_order_labels, current_order_rates):
                composition_results.extend(iterative_error_generator_composition(label_tup, rate_tup))
            #aggregate together any overlapping terms in composition_results
            composition_results_dict = dict()
            for lbl, rate in composition_results:
                if composition_results_dict.get(lbl,None) is None:
                    composition_results_dict[lbl] = rate
                else:
                    composition_results_dict[lbl] += rate
            for lbl, rate in composition_results_dict.items():
                if order_scale*abs(rate) > truncation_threshold:
                    taylor_order_terms[current_order-1][lbl] = order_scale*rate

    return taylor_order_terms

def error_generator_taylor_expansion_numerical(errorgen_dict, errorgen_propagator, order = 1, mx_basis = 'pp'):
    """
    Compute the nth-order taylor expansion for the exponentiation of the error generator described by the input
    error generator dictionary. (Excluding the zeroth-order identity). This function computes a dense representation
    of this taylor expansion as a numpy array and is primarily intended for testing infrastructure.
    
    Parameters
    ----------
    errorgen_dict : dict
        Dictionary whose keys are `LocalStimErrorgenLabel` and whose values are corresponding
        rates.

    errorgen_propagator : `ErrorGeneratorPropagator`
        Error generator propagator used for constructing dense representation of the error generator dictionary.
    
    order : int, optional (default 1)
        Order of the correction (i.e. order of the taylor series expansion for
        the exponentiated error generator) to compute.

    mx_basis : `Basis` or str, optional (default 'pp')
        Basis in which to return the matrix.

    Returns
    -------
    numpy.ndarray
        A dense numpy array corresponding to the nth order taylor expansion of the specified error generator.
    """
       
    errorgen_mat = errorgen_propagator.errorgen_layer_dict_to_errorgen(errorgen_dict, mx_basis)
    taylor_expansion = _np.zeros(errorgen_mat.shape, dtype=_np.complex128)
    for i in range(1, order+1):
        taylor_expansion += 1/factorial(i)*_np.linalg.matrix_power(errorgen_mat, i)

    return taylor_expansion