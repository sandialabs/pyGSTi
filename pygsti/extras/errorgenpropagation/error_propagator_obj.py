import copy as _copy
import numpy as _np
from pygsti.extras.errorgenpropagation.localstimerrorgen import localstimerrorgen
import stim

class error_propagator:
    def __init__(self,circ,potential_errors,label_layers=False,labels=None,gate_lbled=False):
        self.circ=circ
        if not gate_lbled:
            self.__prop_dict=self.__build_error_dict(potential_errors,label_layers,labels)
        else:
            self.__prop_dict=self.__build_gate_labeled_dict(potential_errors)

    def __build_gate_labeled_dict(self,errors):
        #converts pygsti circuit to stim tableau
        stim_layers=self.circ.convert_to_stim_tableau_layers()
        stim_layers.pop(0)

        # list to hold the propagation
        propagation_layers=[]
        #go through the list of sub circuits until there are none left
        while len(stim_layers)>0:
            #get first stim layer
            top_layer=stim_layers.pop(0)
            #find stim tableu until the end
            for layer in stim_layers:
                top_layer = layer*top_layer
            propagation_layers.append(top_layer)

        #create a bunch of error layers including every layer in our model
        errorLayers=[]
        for layer in self.circ:
            error_layer=[]
            for gate_lbl in layer:
                for errs in errors[tuple(gate_lbl)]:
                    errType=errs[0]
                    paulis=[]
                    p1='I'*len(self.circ.line_labels)


                    qbt1=gate_lbl[1]
                    if gate_lbl[0] =='Gcphase':
                        qbt2=gate_lbl[2]
                    if len(errs[1]) != 2:
                        sub_pauli=errs[1].split(':')[0]
                        qbt1=int(errs[1].split(':')[1].split(',')[0])
                        if len(errs[1])==6:
                            qbt2=int(errs[1].split(':')[1].split(',')[1])
                        else:
                            qbt2=None
                    else:
                        sub_pauli=errs[1]
                    p1=p1[:qbt1]+sub_pauli[0]+p1[(qbt1+1):]
                    if qbt2 is not None:
                        p1=p1[:qbt2]+sub_pauli[1]+p1[(qbt2+1):]

                    
                    paulis.append(stim.PauliString(p1))
                    error_layer.append(localstimerrorgen(errType,paulis,label=tuple(gate_lbl)))
            errorLayers.append(error_layer)
        fully_propagated_layers=[]
        #get a subcircuit starting at layer idx
        for (idx,layer) in enumerate(errorLayers):
            #create a dictionary
            new_error_dict=dict()
            if idx <len(errorLayers)-1:
                #get an error in the layer
                for error in layer:
                    #propagate error through to the end    
                    propagated_error_gen=error.propagate_error_gen_tableau(propagation_layers[idx],1.)
                    new_error_dict[error]=propagated_error_gen   
            else:
                for error in layer:
                    labeled_error=_copy.copy(error)
                    new_error_dict[labeled_error]=(error,1)
            fully_propagated_layers.append(new_error_dict)
        return fully_propagated_layers
        




    def __build_error_dict(self,error_model,label_layers,labels):
        #converts pygsti circuit to stim tableau
        stim_layers=self.circ.convert_to_stim_tableau_layers()
        stim_layers.pop(0)

        # list to hold the propagation
        propagation_layers=[]
        #go through the list of sub circuits until there are none left
        while len(stim_layers)>0:
            #get first stim layer
            top_layer=stim_layers.pop(0)
            #find stim tableu until the end
            for layer in stim_layers:
                top_layer = layer*top_layer
            propagation_layers.append(top_layer)

        #create a bunch of error layers including every layer in our model
        errorLayers=[[_copy.deepcopy(eg) for eg in error_model] for i in range(self.circ.depth)]

        fully_propagated_layers=[]
        #get a subcircuit starting at layer idx
        for (idx,layer) in enumerate(errorLayers):
            #create a dictionary
            new_error_dict=dict()
            if idx <len(errorLayers)-1:
                #get an error in the layer
                for error in layer:
                    #propagate error through to the end    
                    propagated_error_gen=error.propagate_error_gen_tableau(propagation_layers[idx],1.)
                    if label_layers is True and labels is None:
                        error.label=layer
                    elif not (labels is None):
                        error.label=labels[idx]
                    new_error_dict[error]=propagated_error_gen   
            else:
                for error in layer:
                    labeled_error=_copy.copy(error)
                    if label_layers is True and labels is None:
                        labeled_error.label=layer
                    elif not (labels is None):
                        labeled_error.label=labels[idx]
                    new_error_dict[labeled_error]=(error,1)
            fully_propagated_layers.append(new_error_dict)


        return fully_propagated_layers
    
    def __create_merge_dict(self, starting_errors):
        stim_circ=self.circ.convert_to_stim_tableau()
        starting_error_dict=dict()
        for error in starting_errors:
            propagated_error_gen=error.propagate_error_gen_tableau(stim_circ,1.)
            starting_error_dict[error]=propagated_error_gen
        return starting_error_dict
        
    def give_propagation_dict(self):
        return _copy.deepcopy(self.__prop_dict)
    
    def merge_error_propagator(self,err_prop):

        prepended_dict_list=err_prop.give_propagation_dict()
        errors=[]
        for dictionary in prepended_dict_list:
            for key in dictionary:
                if dictionary[key][0] in errors:
                    continue
                else:
                    errors.append(dictionary[key][0])
        link_dict=self.__create_merge_dict(errors)

        new_errors=[]
        for layer in prepended_dict_list:
            
            new_layer=dict()
            for key in layer:
                if layer[key][0] in link_dict:
                    new_error=link_dict[layer[key][0]]
                    new_layer[key]=(new_error[0],new_error[1]*layer[key][1])
                else:
                    continue
                
            new_errors.append(new_layer)
        for layer in self.__prop_dict:
            new_errors.append(layer)
        
        self.__prop_dict=new_errors

    def return_prop_matrix(self,max_weight=None):       
        propagated_errors=self.__prop_dict
        indices, signs = [], []
        for l in range(len(self.__prop_dict)):
            indices.append([error_gen_to_index(propagated_errors[l][key][0].errorgen_type, propagated_errors[l][key][0].labels_to_strings(),max_weight=max_weight)
                            for key in propagated_errors[l]])
            signs.append([_np.sign(propagated_errors[l][key][1]) for key in propagated_errors[l]])

        indices = _np.array(indices)
        signs = _np.array(signs)

        return indices, signs
    
    def Organize_error_layers(self,Gate_Err_Dict,Crosstalk_model=False,labels=None):
        # get layers by sorting throug the length of a circuit
        qbts=len(self.circ.line_labels) 
        for layer_no in range(self.circ.depth):
            layer=self.circ.layer(layer_no) #looking at a circuit layer
            existant_errors=[]
            #get a gate in a layer
            for sub_lbl in layer:#look at a gate in the circuit
                
                 #create an identity pauli spanning the entire circuit
                #for an error in the predifined error model where an error is given as tuple(type, pauli)
                for errs in Gate_Err_Dict[tuple(sub_lbl)]:
                    #get error type
                    errType=errs[0]
                    paulis=[]
                    p1='I'*len(self.circ.line_labels)

                    qbt1=sub_lbl[1]
                    if sub_lbl[0]=='Gcphase':
                        qbt2=sub_lbl[2]
                    if len(errs[1]) != 2:
                        sub_pauli=errs[1].split(':')[0]
                        qbt1=int(errs[1].split(':')[1].split(',')[0])
                        if len(errs[1])==6:
                            qbt2=int(errs[1].split(':')[1].split(',')[1])
                        else:
                            qbt2=None
                    else:
                        sub_pauli=errs[1]
                    p1=p1[:qbt1]+sub_pauli[0]+p1[(qbt1+1):]
                    if qbt2 is not None:
                        p1=p1[:qbt2]+sub_pauli[1]+p1[(qbt2+1):]

                    
                    paulis.append(stim.PauliString(p1))
                    if  labels == 'layer': 
                        existant_errors.append(localstimerrorgen(errType,paulis,label=labels[layer_no]))
                    elif labels == 'gate':
                        existant_errors.append(localstimerrorgen(errType,paulis,label=tuple(sub_lbl)))
                    else:
                        existant_errors.append(localstimerrorgen(errType,paulis))
                    


            key_list=list(self.__prop_dict[layer_no].keys())
            for key in key_list:
                if not key in existant_errors:
                    self.__prop_dict[layer_no].pop(key,None)
                
                    

    
    def InverseErrorMap(self):
        InvertedMap=dict()
        for layer_no,layer in enumerate(self.__prop_dict):
            for key in layer:
                if layer[key][0] in InvertedMap:
                    errgen=_copy.copy(key)
                    InvertedMap[layer[key][0]].append(tuple([errgen,layer[key][1]**(-1)]))
                else:
                    errgen=_copy.copy(key)
                    InvertedMap[layer[key][0]]=[tuple([errgen,layer[key][1]**(-1)])]
        return InvertedMap
    
    def return_EOC_error_values(self,errorValues):
        numeric_map=dict()
        for layer_no,layer in enumerate(self.__prop_dict):
            for key in layer:
                if layer[key][0] in numeric_map and key in errorValues[layer_no]:
                    numeric_map[layer[key][0]]+=errorValues[layer_no][key]*layer[key][1]**(-1)
                elif key in errorValues[layer_no]:
                    numeric_map[layer[key][0]]=errorValues[layer_no][key]*layer[key][1]**(-1)
                else:
                    continue
        return numeric_map

            
#These are here until we can find a better place for them        
def error_gen_to_index(typ : str, paulis : tuple, max_weight=None):
    """
    A function that *defines* an indexing of the primitive error generators. Currently
    specifies indexing for all 'H' and 'S' errors. In future, will add 'C' and 'A' 
    error generators, but will maintain current indexing for 'H' and 'S'.
    
    typ: 'H' or 'S', specifying the tuype of primitive error generator
    
    paulis: tuple, single element tuple, containing a string specifying the Pauli
        the labels the 'H' or 'S' error. The string's length implicitly 
        defines the number of qubit that the error gen acts on
    """
    assert isinstance(paulis,tuple)
    p1 = paulis[0]
    if max_weight is None:
        n = len(p1)
    else:
        n=max_weight
    if typ == 'H':
        base = -1
    elif typ == 'S':
        base = 4**n -2  
    else:
        raise ValueError('Invalid error generator specification! Note "C" and "A" errors are not implemented yet.') 
    # Future to do: C and A errors
  

    return base + paulistring_to_index(p1, n)

def paulistring_to_index(ps, num_qubits):
    """
    Maps an n-qubit Pauli operator (represented as a string, list or tuple of elements from
    {'I', 'X', 'Y', 'Z'}) to an integer.  It uses the most conventional mapping, whereby, e.g.,
    if `num_qubits` is 2, then 'II' -> 0, and 'IX' -> 1, and 'ZZ' -> 15.

    ps: str, list, or tuple.

    num_qubits: int

    Returns
    int
    """
    idx = 0
    p_to_i = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    for i in range(num_qubits):
        idx += p_to_i[ps[num_qubits - 1 - i]] * 4**i
    return idx