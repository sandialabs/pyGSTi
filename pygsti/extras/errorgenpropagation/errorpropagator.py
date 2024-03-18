import stim
from pygsti.extras.errorgenpropagation.propagatableerrorgen import *
from pygsti.extras.errorgenpropagation.utilspygstistimtranslator import *
from numpy import abs
from numpy.linalg import multi_dot
from scipy.linalg import expm
from pygsti.tools.internalgates import standard_gatenames_stim_conversions


'''
takes a pygsti circuit where each gate has a defined error model and returns the errorgenerators necessary to create an
end of circuit error generator under a variety of scenarios

circ: pygsti circuit
errorModel: Dictionary defined the small markovian error generators and their rates for each gate
BCHOrder: in cases where the BCH approximation is used, carries it out to the desired order (can currently only handle order 1 or 2)
BCHLayerWise: If true will push the errors through one layer of gatesand then combines them using the bch approximation at each layer
If false will simply push all errors to the end
NonMarkovian: Pushes the error generators to the end and then formats them to work with the cumulant expansion code
MultiGateDict: Containts the translation between a numbered gate Gxpi22 and the PyGSTi standard gate used when a singular gate has
multiple error iterations
MultiGate: lets the code know 
returns: list of propagatableerrorgens
'''
def ErrorPropagator(circ,errorModel,MultiGateDict={},BCHOrder=1,BCHLayerwise=False,NonMarkovian=False,MultiGate=False):
    stim_dict=standard_gatenames_stim_conversions()
    if MultiGate:
        for key in MultiGateDict:
            stim_dict[key]=stim_dict[MultiGateDict[key]]
    stim_layers=circ.convert_to_stim_tableau_layers(gate_name_conversions=stim_dict)
    stim_layers.pop(0)  #Immeditielty toss the first layer because it is not important,

    propagation_layers=[]
    if not BCHLayerwise or NonMarkovian:
        while len(stim_layers) != 0:
            top_layer=stim_layers.pop(0)
            for layer in stim_layers:
                top_layer = layer*top_layer
            propagation_layers.append(top_layer)
    else:
        propagation_layers = stim_layers

    errorLayers=buildErrorlayers(circ,errorModel,len(circ.line_labels))

    num_error_layers=len(errorLayers)
    fully_propagated_layers=[]
    for _ in range(0,num_error_layers-1):
        err_layer=errorLayers.pop(0)
        layer=propagation_layers.pop(0)
        for err_order in err_layer:
            for errorGenerator in err_order:
                errorGenerator.propagate_error_gen_inplace_tableau(layer)
        if BCHLayerwise and not NonMarkovian:
            following_layer = errorLayers.pop(0)
            new_errors=BCH_Handler(err_layer,following_layer,BCHOrder)
            errorLayers.insert(new_errors,0)
        else:
            fully_propagated_layers.append(err_layer)

    fully_propagated_layers.append(errorLayers.pop(0))
    if BCHLayerwise and not NonMarkovian:
        for order in errorLayers:
            for error in order:
                if len(fully_propagated_layers)==0:
                    fully_propagated_layers.append(error)
                elif error in fully_propagated_layers:
                    idy=fully_propagated_layers.index(error)
                    new_error=error+fully_propagated_layers[idy]
                    fully_propagated_layers.pop(idy)
                    fully_propagated_layers.append(new_error)
                else:
                    fully_propagated_layers.append(error)
        return fully_propagated_layers
        
    elif not BCHLayerwise and not NonMarkovian:
        simplified_EOC_errors=[]
        if BCHOrder == 1:
            for layer in fully_propagated_layers:
                for order in layer:
                    for error in order:
                        if len(simplified_EOC_errors)==0:
                            simplified_EOC_errors.append(error)
                        elif error in simplified_EOC_errors:
                            idy=simplified_EOC_errors.index(error)
                            new_error=error+simplified_EOC_errors[idy]
                            simplified_EOC_errors.pop(idy)
                            if not (abs(new_error.get_Error_Rate()) <.000001):
                                simplified_EOC_errors.append(new_error)
                        else:
                            if not (abs(error.get_Error_Rate())<.000001):
                                simplified_EOC_errors.append(error)
        else:
            Exception("Higher propagated through Errors are not Implemented Yet")
        return simplified_EOC_errors
    
    else:
        return fully_propagated_layers


'''
takes two error layers (list of propagatableerrorgens) and find the bch combination of the two
err_layer: list lists of propagatableerrorgens
following_layer: list of propagatableerrorgens
BCHOrder: Order to carry the bch expansion out to, can currently be set to one or two
returns list of lists of propagatableerrorgens. The outer list contains each of them individual list denote order
'''
def BCH_Handler(err_layer,following_layer,BCHOrder):         
    new_errors=[]
    for curr_order in range(0,BCHOrder):
        working_order=[]
        if curr_order == 0:
            used_indexes=[]
            for error in err_layer[curr_order]:
                try:
                    idy=following_layer[curr_order].index(error)
                    working_order.append(error+following_layer[curr_order][idy])
                    used_indexes.append(idy)
                except:
                    working_order.append(error)
            for idy,error in enumerate(following_layer[curr_order]):
                if idy in used_indexes:
                    continue
                else:
                    working_order.append(error)
            
            new_errors.append(working_order)
        elif curr_order ==1:
            working_order=[]
            for error1 in err_layer[curr_order-1]:
                for error2 in following_layer[curr_order-1]:
                    errorlist = commute_errors(error1,error2,BCHweight=1/2)
                    for error3 in errorlist:
                        if len(working_order)==0:
                            working_order.append(error3)
                        elif error3 in working_order:
                            idy=working_order.index(error3)
                            new_error=error3+working_order[idy]
                            working_order.pop(idy)
                            working_order.append(new_error)
                        else:
                            working_order.append(error3)
            if len(err_layer)==2:
                for error3 in err_layer[1]:
                    if len(working_order)==0:
                        working_order.append(error3)
                    elif error3 in working_order:
                        idy=working_order.index(error3)
                        new_error=error3+working_order[idy]
                        working_order.pop(idy)
                        working_order.append(new_error)
                    else:
                        working_order.append(error3)
            if len(following_layer)==2:
                for error3 in following_layer[1]:
                    if len(working_order)==0:
                        working_order.append(error3)
                    elif error3 in working_order:
                        idy=working_order.index(error3)
                        new_error=error3+working_order[idy]
                        working_order.pop(idy)
                        if new_error.get_Error_Rate() != 0j:
                            working_order.append(new_error)
                    else:
                        working_order.append(error3)
            new_errors.append(working_order)

        else:
            Exception("Higher Orders are not Implemented Yet")


'''
takes a pygst circuit object and error Dictionary and creates error layers

inputs
circ: pygsti circuit
errorDict: Dictionary defined the small markovian error generators and their rates for each gate
qubits: number of qubits in the circuit

output
ErrorGens, a list of error gen layers (which are list of propagatable errorgens)

'''
def buildErrorlayers(circ,errorDict,qubits):
    ErrorGens=[]
    #For the jth layer of each circuit
    for j in range(circ.depth):
        l = circ.layer(j) # get the layer
        errorLayer=[]
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
                
                paulis.append(p1)
                if errType in "CA":
                    for ind,el in enumerate(g):
                        if ind !=0:
                            p2=p2[:el] + errs[2][ind-1] +p2[(el+1):]
                    paulis.append(p2)     
                errorLayer.append(propagatableerrorgen(errType,paulis,gErrorDict[errs]))
        ErrorGens.append([errorLayer])
    return ErrorGens



# There's a factor of a half missing in here. 
def nm_propagators(corr, Elist):
    Kms = []
    for idm in range(len(Elist)):
        Am = Elist[idm].toWeightedErrorBasisMatrix
        # This assumes that Elist is in reverse chronological order
        partials = []
        for idn in range(idm, len(Elist)):
            An = Elist[idn].toWeightedErrorBasisMatrix()
            partials += [corr[idm,idn] * Am @ An]
        partials[0] = partials[0]/2
        Kms += [sum(partials,0)]
    return Kms

def averaged_evolution(corr, Elist):
    Kms = nm_propagators(corr, Elist)
    return multi_dot([expm(Km) for Km in Kms])