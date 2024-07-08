import stim
from pygsti.extras.errorgenpropagation.localstimerrorgen import *
from numpy import abs,zeros, complex128
from numpy.linalg import multi_dot
from scipy.linalg import expm
from pygsti.tools.internalgates import standard_gatenames_stim_conversions
from pygsti.extras.errorgenpropagation.utilserrorgenpropagation import *
import copy as _copy

def ErrorPropagatorAnalytic(circ,errorModel,ErrorLayerDef=False,startingErrors=None):
    stim_layers=circ.convert_to_stim_tableau_layers()
    
    if startingErrors is None:
        stim_layers.pop(0)

    propagation_layers=[]
    while len(stim_layers)>0:
        top_layer=stim_layers.pop(0)
        for layer in stim_layers:
            top_layer = layer*top_layer
        propagation_layers.append(top_layer)
    
    if not ErrorLayerDef:
        errorLayers=buildErrorlayers(circ,errorModel,len(circ.line_labels))
    else:
        errorLayers=[[_copy.deepcopy(eg) for eg in errorModel] for i in range(circ.depth)]
    
    if not startingErrors is None:
        errorLayers.insert(0,startingErrors)
    
    fully_propagated_layers=[]
    for (idx,layer) in enumerate(errorLayers):
        new_error_dict=dict()
        if idx <len(errorLayers)-1:

            for error in layer:    
                propagated_error_gen=error.propagate_error_gen_tableau(propagation_layers[idx],1.)
                new_error_dict[error]=propagated_error_gen   
        else:
            for error in layer:
                new_error_dict[error]=(error,1)
        fully_propagated_layers.append(new_error_dict)

    #print(len(fully_propagated_layers))
    return fully_propagated_layers
    
def InverseErrorMap(errorMap):
    InvertedMap=dict()
    for layer_no,layer in enumerate(errorMap):
        for key in layer:
            if layer[key][0] in InvertedMap:
                errgen=_copy.copy(key)
                errgen.label=layer_no
                InvertedMap[layer[key][0]].append(tuple([errgen,layer[key][1]**(-1)]))
            else:
                errgen=_copy.copy(key)
                errgen.label=layer_no
                InvertedMap[layer[key][0]]=[tuple([errgen,layer[key][1]**(-1)])]
    return InvertedMap

def InvertedNumericMap(errorMap,errorValues):
    numeric_map=dict()
    for layer_no,layer in enumerate(errorMap):
        for key in layer:
            if layer[key][0] in numeric_map and key in errorValues[layer_no]:
                numeric_map[layer[key][0]]+=errorValues[layer_no][key]*layer[key][1]**(-1)
            elif key in errorValues[layer_no]:
                numeric_map[layer[key][0]]=errorValues[layer_no][key]*layer[key][1]**(-1)
            else:
                continue
    return numeric_map


def ErrorPropagator(circ,errorModel,MultiGateDict=None,BCHOrder=1,BCHLayerwise=False,NonMarkovian=False,MultiGate=False,ErrorLayerDef=False):
    if MultiGate and MultiGateDict is None:
        MultiGateDict=dict()
    stim_dict=standard_gatenames_stim_conversions()
    if MultiGate:
        for key in MultiGateDict:
            stim_dict[key]=stim_dict[MultiGateDict[key]]
    stim_layers=circ.convert_to_stim_tableau_layers(gate_name_conversions=stim_dict)
    stim_layers.pop(0)  #Immediatly toss the first layer because it is not important,

    propagation_layers=[]
    if not BCHLayerwise or NonMarkovian:
        while len(stim_layers) != 0:
            top_layer=stim_layers.pop(0)
            for layer in stim_layers:
                top_layer = layer*top_layer
            propagation_layers.append(top_layer)
    else:
        propagation_layers = stim_layers

    if not ErrorLayerDef:
        errorLayers=buildErrorlayers(circ,errorModel,len(circ.line_labels))
    else:
        errorLayers=[[[_copy.deepcopy(eg) for eg in errorModel]] for i in range(circ.depth)]

    num_error_layers=len(errorLayers)
    
    fully_propagated_layers=[]
    for _ in range(0,num_error_layers-1):
        err_layer=errorLayers.pop(0)
        layer=propagation_layers.pop(0)
        new_error_layer=[]
        for err_order in err_layer:
            new_error_dict=dict()
            for key in err_order:
                propagated_error_gen=key.propagate_error_gen_tableau(layer,err_order[key])
                new_error_dict[propagated_error_gen[0]]=propagated_error_gen[1]
            new_error_layer.append(new_error_dict)
        if BCHLayerwise and not NonMarkovian:
            following_layer = errorLayers.pop(0)
            new_errors=BCH_Handler(err_layer,following_layer,BCHOrder)
            errorLayers.insert(new_errors,0)
        else:
            fully_propagated_layers.append(new_error_layer)

    fully_propagated_layers.append(errorLayers.pop(0))
    if BCHLayerwise and not NonMarkovian:
        final_error=dict()
        for order in errorLayers[0]:
            for error in order:
                if error in final_error:
                    final_error[error]=final_error[error]+order[error]
                else:
                    final_error[error]=order[error]
        return final_error
    
    elif not BCHLayerwise and not NonMarkovian:
        simplified_EOC_errors=dict()
        if BCHOrder == 1:
            for layer in fully_propagated_layers:
                for order in layer:
                    for error in order:
                        if error in simplified_EOC_errors:
                            simplified_EOC_errors[error]=simplified_EOC_errors[error]+order[error]
                        else:
                            simplified_EOC_errors[error]=order[error]

        else:
            Exception("Higher propagated through Errors are not Implemented Yet")
        return simplified_EOC_errors
    
    else:
        return fully_propagated_layers



def buildErrorlayers(circ,errorDict,qubits):
    ErrorGens=[]
    #For the jth layer of each circuit
    for j in range(circ.depth):
        l = circ.layer(j) # get the layer
        errorLayer=dict()
        for _, g in enumerate(l): # for gate in layer l
            gErrorDict = errorDict[g.name] #get the errors for the gate
            p1=qubits*'I' # make some paulis why?
            p2=qubits*'I'
            for errs in gErrorDict: #for an error in the accompanying error dictionary 
                errType=errs[0]
                paulis=[]
                for ind,el in enumerate(g): #enumerate the gate ind =0 is name ind = 1 is first qubit ind = 2 is second qubit
                    if ind !=0:  #if the gate element of concern is not the name
                        p1=p1[:el] + errs[1][ind-1] +p1[(el+1):]
                
                paulis.append(stim.PauliString(p1))
                if errType in "CA":
                    for ind,el in enumerate(g):
                        if ind !=0:
                            p2=p2[:el] + errs[2][ind-1] +p2[(el+1):]
                    paulis.append(stim.PauliString(p2))     
                errorLayer[localstimerrorgen(errType,paulis)]=gErrorDict[errs]
        ErrorGens.append([errorLayer])
    return ErrorGens
'''

Inputs:
_______
err_layer (list of dictionaries)
following_layer (list of dictionaries)
BCHOrder:

'''
def BCH_Handler(err_layer,following_layer,BCHOrder):         
    new_errors=[]
    for curr_order in range(0,BCHOrder):
        working_order=dict()
        #add first order terms into new layer
        if curr_order == 0:
            for error_key in err_layer[curr_order]:
                working_order[error_key]=err_layer[curr_order][error_key]
            for error_key in following_layer[curr_order]:
                working_order[error_key]=following_layer[curr_order[error_key]] 
            new_errors.append(working_order)

        elif curr_order ==1:
            working_order={}
            for error1 in err_layer[curr_order-1]:
                for error2 in following_layer[curr_order-1]:
                    errorlist = commute_errors(error1,error2,BCHweight=1/2*err_layer[error1]*following_layer[error2])
                    for error_tuple in errorlist:
                        working_order[error_tuple[0]]=error_tuple[1]
            if len(err_layer)==2:
                for error_key in err_layer[1]:
                    working_order[error_key]=err_layer[1][error_key]
            if len(following_layer)==2:
                for error_key in following_layer[1]:
                    working_order[error_key]=following_layer[1][error_key]
            new_errors.append(working_order)

        else:
            Exception("Higher Orders are not Implemented Yet")
    return new_errors

# There's a factor of a half missing in here. 
def nm_propagators(corr, Elist,qubits):
    Kms = []
    for idm in range(len(Elist)):
        Am=zeros([4**qubits,4**qubits],dtype=complex128)
        for key in Elist[idm][0]:
            Am += key.toWeightedErrorBasisMatrix()
            # This assumes that Elist is in reverse chronological order
        partials = []
        for idn in range(idm, len(Elist)):
            An=zeros([4**qubits,4**qubits],dtype=complex128)
            for key2 in Elist[idn][0]:
                An = key2.toWeightedErrorBasisMatrix()
            partials += [corr[idm,idn] * Am @ An]
        partials[0] = partials[0]/2
        Kms += [sum(partials,0)]
    return Kms

def averaged_evolution(corr, Elist,qubits):
    Kms = nm_propagators(corr, Elist,qubits)
    return multi_dot([expm(Km) for Km in Kms])

def error_stitcher(first_error,second_error):
    link_dict=second_error.pop(0)
    new_errors=[]
    for layer in first_error:
        #print(len(layer))
        new_layer=dict()
        for key in layer:
            if layer[key][0] in link_dict:
                new_error=link_dict[layer[key][0]]
                new_layer[key]=(new_error[0],new_error[1]*layer[key][1])
            elif layer[key][0].errorgen_type =='Z':
                new_layer[key]=layer[key]
            else:
                continue
            
        new_errors.append(new_layer)
    for layer in second_error:
        new_errors.append(layer)
    return new_errors