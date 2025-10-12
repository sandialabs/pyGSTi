import numpy as np
from ...tools import lindbladtools as _lt
from ...tools import matrixtools
from ...tools import change_basis

import scipy
import itertools
import copy
from ... import processors
from ... import models
from ... import circuits
from ... import baseobjs


num_qubits = 4
gate_names = ['Gcnot', 'Gxpi2', 'Gypi2']
geometry = 'ring'
pspec = processors.QubitProcessorSpec(num_qubits, gate_names, geometry=geometry)
# target_model = models.create_explicit_model(pspec, evotype='densitymx')
# target_model.set_all_parameterizations('full')

# Encoding 
num_oneQ_gates = 2 # Hard coding the gate set here!
max_degree_directed_connectivity_graph = 4 # Hard coding the connectivity here!
num_channels = num_oneQ_gates + max_degree_directed_connectivity_graph


def laplace(dim=4):

    dim = 4
    adj = np.zeros((dim,dim))
    for i in range(0,dim):
        adj[i, (i-1) % dim] = 1
        adj[i, (i+1) % dim] = 1
    deg = 2*np.identity(dim)
    laplace = deg - adj
    return laplace
#####

def compute_linear_infidelities(circs, indices, signs, model_params):
    
    linear_infidelities = []
    total_evecs = []
    
    for c_idx, c in enumerate(circs):
        
        total_evec = np.zeros(256, float)
        
        for i in range(c.depth):
            
            l = c[i]
            evec = layer_to_error_vector(l, model_params, 'H')
            permutation_vector = indices[c_idx,:,i]
            signs_vector = signs[c_idx,:,i] 
            total_evec += np.array([signs_vector[idx]*evec[int(perm_idx)] for idx, perm_idx in enumerate(permutation_vector)])

        total_evecs.append(total_evec)
        linear_infidelities.append(np.sum(total_evec**2))

    return linear_infidelities, total_evecs

def error_propogation_information(c, depth, target_model, vec_dim=256):
    
    if depth is None: depth = c.depth
    permutation = np.zeros((vec_dim, depth),float)
    for i in range(c.depth):
        permutation[:, i] = np.arange(1, vec_dim + 1)
        
    for i in range(0, c.depth-1):
        pm = target_model[c[i+1]].to_dense()
        permutation[:, 0:i+1] = np.dot(pm, permutation[:, 0:i+1])
    
    indices = np.zeros((vec_dim, depth),float)
    signs = np.zeros((vec_dim, depth),float)
    
    signs[:, 0:c.depth] = np.sign(permutation[:, 0:c.depth])
    indices[:, 0:c.depth] = np.abs(permutation[:, 0:c.depth]) - 1

    return indices, signs


def create_input_data(circs, fidelities, target_model, max_depth=None):
    
    if max_depth is None: max_depth = np.max([c.depth for c in circs])
    print(max_depth)
    
    numchannels = num_channels
    numqubits = num_qubits
    numcircs = len(circs)
    
    x_circs = np.zeros((numcircs, numqubits, max_depth, numchannels), float)
    x_signs = np.zeros((numcircs, 256, max_depth), int)
    x_indices = np.zeros((numcircs, 256, max_depth), int)
    y = np.array(fidelities)
                    
    for i, c in enumerate(circs):
        if i % 25 == 0:
            print(i,end=',')
        x_circs[i, :, :, :] = circuit_to_tensor(c, max_depth)              
        c_indices, c_signs = error_propogation_information(c, max_depth, target_model, vec_dim=256)
        x_indices[i, :, :] = np.rint(c_indices)
        x_signs[i, :, :] = np.rint(c_signs)
        
    return x_circs, x_signs, x_indices, y

def split_data(x_circs, x_signs, x_indices, y, split=[0.9,0.05,0.05]):

    numtrain = int(numcircs * split[0])
    numval = int(numcircs * split[1])
    numtest = numcircs - (numval + numtrain)
    print('No. training circuits:', numtrain)
    print('No. validation circuits:', numval)
    print('No. test circuits:', numtest)

    x_circs_dict = {'train': np.zeros((numtrain, numqubits, max_depth, numchannels), float),
                    'test' : np.zeros((numtest, numqubits, max_depth, numchannels), float),
                    'validate' : np.zeros((numval, numqubits, max_depth, numchannels), float)}
    x_signs_dict = {'train': np.zeros((numtrain, numqubits, max_depth, 256), float),
                    'test' : np.zeros((numtest, numqubits, max_depth, 256), float),
                    'validate' : np.zeros((numval, numqubits, max_depth, 256), float)}
    x_indices_dict = {'train': np.zeros((numtrain, numqubits, max_depth, 256), float),
                     'test' : np.zeros((numtest, numqubits, max_depth, 256), float),
                     'validate' : np.zeros((numval, numqubits, max_depth, 256), float)}
    
    return x_circs_dict, x_signs_dict, x_indices_dict, y_dict


#### Error model generation
def sample_h_error_rate(max_rate=0.01, typ='H'):
    return  max_rate * 2*(np.random.rand()-0.5)

def sample_s_error_rate(max_rate=0.01):
    return  max_rate * np.random.rand()

def sample_ctf_orot_emodel(max_rates, cnot_scaling=2, typ='H+S'):
    
    gxs = pspec.available_gatelabels('Gxpi2',[0,1,2,3])
    gys = pspec.available_gatelabels('Gypi2',[0,1,2,3])
    gcnots = pspec.available_gatelabels('Gcnot',[0,1,2,3])
    
    model_params_sparse_rep = {}   
    model_params_sparse_rep.update({frozenset((gate,)):{} for gate in gxs})
    model_params_sparse_rep.update({frozenset((gate,)):{} for gate in gys})
    model_params_sparse_rep.update({frozenset((gate,)):{} for gate in gcnots})
    
    if 'H' in typ:
        max_h_rate = max_rates['H']
        for gate in gxs:
            er = sample_h_error_rate(max_h_rate)
            model_params_sparse_rep[frozenset((gate,))].update({('H', embed_pauli('X', gate.qubits)):er})
        for gate in gys:
            er = sample_h_error_rate(max_h_rate)
            model_params_sparse_rep[frozenset((gate,))].update({('H', embed_pauli('Y', gate.qubits)):er})
        for gate in gcnots:
            # Would be an 'over-rotation' on the CNOT if all three error rates are the same, but last one neg.
            er_ix = cnot_scaling*sample_h_error_rate(max_h_rate)/3
            er_zi = cnot_scaling*sample_h_error_rate(max_h_rate)/3
            er_zx = cnot_scaling*sample_h_error_rate(max_h_rate)/3
            model_params_sparse_rep[frozenset((gate,))].update({('H', embed_pauli('IX', gate.qubits)):er_ix, 
                                                           ('H', embed_pauli('ZI', gate.qubits)):er_zi,
                                                           ('H', embed_pauli('ZX', gate.qubits)):er_zx})
    if 'S' in typ:
        max_s_rate = max_rates['S']
        for gate in gxs:
            er = sample_s_error_rate(max_s_rate)
            model_params_sparse_rep[frozenset((gate,))].update({('S', embed_pauli('X', gate.qubits)):er})
        for gate in gys:
            er = sample_s_error_rate(max_s_rate)
            model_params_sparse_rep[frozenset((gate,))].update({('S', embed_pauli('Y', gate.qubits)):er})
        for gate in gcnots:
            er_ix = cnot_scaling*sample_s_error_rate(max_s_rate)/3
            er_zi = cnot_scaling*sample_s_error_rate(max_s_rate)/3
            er_zx = cnot_scaling*sample_s_error_rate(max_s_rate)/3
            model_params_sparse_rep[frozenset((gate,))].update({('S', embed_pauli('IX', gate.qubits)):er_ix, 
                                                                ('S', embed_pauli('ZI', gate.qubits)):er_zi,
                                                                ('S', embed_pauli('ZX', gate.qubits)):er_zx})        
        
    return model_params_sparse_rep
        
#### 4Q TARGET MODEL GENERATION

def int_to_oneQlabel(i, q):
    if i == 0: return ''
    if i == 1: return 'Gxpi2:'+str(q)
    if i == 2: return 'Gypi2:'+str(q)
    
def get_all_layers():
    
    cnots = pspec.available_gatelabels('Gcnot', [0,1,2,3]) 
    Gxs =  pspec.available_gatelabels('Gxpi2', [0,1,2,3])
    Gys =  pspec.available_gatelabels('Gypi2', [0,1,2,3])

    all_layers = []

    # no cnots
    for l in itertools.product([0,1,2], repeat=4):
        all_layers.append(''.join([int_to_oneQlabel(i, q) for q, i in enumerate(l)]))

    # 1 cnot
    for cnot in cnots:
        qs = [0,1,2,3]
        for i in cnot.qubits:
            qs.remove(i)
        for l in itertools.product([0,1,2], repeat=2):
            all_layers.append(cnot.__str__() + ''.join([int_to_oneQlabel(i, qs[q]) for q, i in enumerate(l)]))

    # 2 cnots
    for i, cnot1 in enumerate(cnots):
        for cnot2 in cnots[i:]:
            if cnot1 != cnot2:
                qs = set(list(cnot1.qubits) + list(cnot2.qubits))
                if len(qs) < 4:
                    continue
                all_layers.append(cnot1.__str__() + cnot2.__str__())
            
    return all_layers

def populate_target_model(verbose=0):
    
    all_layers = get_all_layers()
    failures = 0
    for lidx, l in enumerate(all_layers):
        if verbose>0:
            print(lidx,end=' ',flush=True)
    #    for glabel in target_model.operations:
    #        print(glabel, target_model.operations[glabel].__repr__())
        #print(f'Adding layer number {lidx}')
        try:
            pm = np.identity(4**num_qubits, float)
            label = []
        #    print(l)
        #    print(pygsti.circuits.Circuit(l))
            for glabel in circuits.Circuit(l):
                #print(glabel)
                label.append(glabel)
                pm = np.dot(pm, target_model[glabel])
            target_model.operations[tuple(label)] = pm
    #        target_model.set_all_parameterizations('full')
        except:
            #print(f'Failed on {l}!')
            failures+=1
            #print(f'Failure number {failures}.')
            #for glabel in target_model.operations:
            #    print(glabel, target_model.operations[glabel].__repr__())
            target_model.set_all_parameterizations('full')
            pm = np.identity(4**num_qubits, float)
            label = []
        #    print(l)
            #print(pygsti.circuits.Circuit(l))
            for glabel in circuits.Circuit(l):
            #    print(glabel)
                label.append(glabel)
                pm = np.dot(pm, target_model[glabel])
            target_model.operations[tuple(label)] = pm
            
    return target_model


###### Functions that encode a circuit into a tensor ###

# TO DO: Make more general

qubit_to_index = {0:0, 1:1, 2:2, 3:3}
           
def clockwise_cnot(g):
    return (g.qubits[0] - g.qubits[1]) % num_qubits == num_qubits - 1

def gate_to_index(g, q):
    assert(q in g.qubits)
    if g.name == 'Gxpi2':
        return 0
    elif g.name == 'Gypi2':
        return 1
    elif g.name == 'Gcnot':
        qs = g.qubits
        if q == g.qubits[0] and clockwise_cnot(g):
            return 2
        if q == g.qubits[1] and clockwise_cnot(g):
            return 3
        if q == g.qubits[0] and not clockwise_cnot(g):
            return 4
        if q == g.qubits[1] and not clockwise_cnot(g):
            return 5
    else:
        raise ValueError('Invalid gate name for this encoding!')
        
def layer_to_matrix(layer):
    mat = np.zeros((num_qubits, num_channels), float)
    for g in layer:
        for q in g.qubits:
            mat[q, gate_to_index(g, q)] = 1
    return mat

def circuit_to_tensor(circ, depth=None):
    
    if depth is None: depth = circ.depth
    ctensor = np.zeros((num_qubits, depth, num_channels), float)
    for i in range(circ.depth):
        ctensor[:, i, :] = layer_to_matrix(circ.layer(i))
    return ctensor

###### Functions that map layers to error vectors ######

def sparse_rep_to_dense_rep_params_dict(sparse_rep_params_dict, typ='H+S'):
    
    if typ == 'H+S':
        len_evec = 2 * 4**num_qubits
    else:
        len_evec = 4**num_qubits
    
    dense_rep_params_dict = {}    
    for key, error_dict in sparse_rep_params_dict.items():
        evec =  np.zeros(len_evec, float)
        for (etyp, paulistring), erate in error_dict.items():
             evec[error_vector_index(etyp, paulistring, typ)] = erate
        dense_rep_params_dict[key] = evec
            
    return dense_rep_params_dict

def error_vector_index(etyp, paulistring, typ='H+S'):
    if typ == 'H+S':
        if etyp == 'H':
            start = 0
        elif etyp == 'S':
            start = 4**num_qubits
    else:
        start = 0
    return start + paulistring_to_index(paulistring, num_qubits)
            
def layer_to_error_vector(layer, params_dict, typ='H+S'):
    """
    params_dict:
        keys are frozensets of gate labels. 
        labels error vectors.
    """
    if typ == 'H+S':
        len_evec = 2 * 4**num_qubits
    else:
        len_evec = 4**num_qubits
    
    evec = np.zeros(len_evec, float)
    if isinstance(layer, baseobjs.label.LabelTupTup):
        lset = frozenset(layer)
    elif isinstance(layer, baseobjs.label.LabelTup):
        lset = frozenset((layer,))
    else:
        raise ValueError("Argh!")
    #print(lset)
    for gate_combination, evec_for_gate_combination in params_dict.items():
        #print(gate_combination)
        if gate_combination.issubset(lset):
            #print('here')
            evec += evec_for_gate_combination

    return evec


######

def get_indices_for_all_weight_two_or_less_errors(num_qubits):
    
    indices_dict = {}
    for i in range(num_qubits):
        indices_dict[(i,)] = []
        for p in ['X','Y','Z']:
            ps = embed_pauli(p,(i,))
            indices_dict[(i,)].append(paulistring_to_index(ps, num_qubits))

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                indices_dict[(i,j)] = []
                for p in ['X','Y','Z']:
                    for q in ['X','Y','Z']:
                        ps = embed_pauli(p+q,(i,j))
                        indices_dict[(i,j)].append(paulistring_to_index(ps, num_qubits))

    return indices_dict
###### Functions for creating error generators etc #####

def pauli_matrix(p):
    """
    p: list or string contain 'I', 'X', 'Y', and 'Z.
    """
    id2x2 = np.array([[1.0, 0.0j], [0.0j, 1.0]])
    sigmax = np.array([[0.0j, 1.0+0.0j], [1.0+0.0j, 0.0j]])
    sigmay = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sigmaz = np.array([[1.0, 0.0j], [0.0j, -1.0]])
    sigmas= {'I': id2x2, 'X': sigmax, 'Y': sigmay, 'Z': sigmaz}

    tensorproduct = sigmas[p[0]]

    for i in range(1, len(p)):
        tensorproduct = np.kron(tensorproduct, sigmas[p[i]])

    return tensorproduct

def elementary_errorgen(typ, p):
    """
    typ: 'H' or 'S'.
    
    p: string or list of I, X, Y, and Z.
    """
    pmatrix = pauli_matrix(p)
    return change_basis(_lt.create_elementary_errorgen(typ, pmatrix, None), 'std', 'pp')

#def get_rate_of_elementary_errorgens(error_generator, typ):
#    
#    log_error
#    for i in range(
#    eeg_dual = _lt.create_elementary_errorgen_dual(typ, pmatrix, None)

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def padded_numberToBase4(n, num_qubits):
    l = num_qubits
    a = numberToBase(n, 4)
    return [0]*(num_qubits-len(a)) + a

def index_to_paulistring(i, num_qubits):
    i_to_p = {0:'I', 1:'X', 2:'Y', 3:'Z'}
    assert(i < 4**num_qubits)
    return ''.join([i_to_p[i] for i in padded_numberToBase4(i, num_qubits)])
    
def paulistring_to_index(ps, num_qubits):
    idx = 0
    p_to_i = {'I':0, 'X':1, 'Y':2, 'Z':3}
    for i in range(num_qubits):
        idx += p_to_i[ps[num_qubits - 1 - i]] * 4**i
    return idx

def embed_pauli(ps, qubits, all_qubits=[0,1,2,3]):
    embedded_ps = ['I']*len(all_qubits)
    for p, q in zip(ps, qubits):
        embedded_ps[all_qubits.index(q)] = p
    return ''.join(embedded_ps)
    
def error_rates_vector_to_error_gen(evec, num_qubits, typ='H+S'):

    if typ == 'H+S':
        hterm =  np.sum([er*elementary_errorgen('H', index_to_paulistring(i, num_qubits))
                         for i, er in enumerate(evec[:4**num_qubits])], axis=0)
        sterm =  np.sum([er*elementary_errorgen('S', index_to_paulistring(i, num_qubits))
                         for i, er in enumerate(evec[4**num_qubits:])], axis=0)
        return hterm + sterm
    
    elif typ == 'H':
        return np.sum([er*elementary_errorgen('H', index_to_paulistring(i, num_qubits))
                       for i, er in enumerate(evec)], axis=0)
    
    elif typ == 'S':
        return np.sum([er*elementary_errorgen('S', index_to_paulistring(i, num_qubits))
                       for i, er in enumerate(evec)], axis=0)
    
    else:
        assert(False)

def error_rates_vector_to_error_channel(evec, num_qubits, typ='H+S'):
    return scipy.linalg.expm(error_rates_vector_to_error_gen(evec, num_qubits, typ))


def swap_locs(layer,i,j):
    layer_native = list(layer.to_native())
    new_layer_native = len(layer)*[0]
    new_layer_native[i] = layer_native[j]
    new_layer_native[j] = layer_native[i]
    for lidx, lbl in enumerate(layer):
        if lidx not in (i,j):
            new_layer_native[lidx] = lbl
    return baseobjs.Label(new_layer_native)

def push_cnots_left(layer):
    new_layer = copy.deepcopy(layer)
    for lidx, lbl in enumerate(layer[:-1]):
        if new_layer[lidx][0]!='Gcnot' and new_layer[lidx+1][0] == 'Gcnot':
            new_layer = swap_locs(new_layer,lidx,lidx+1)
            swap_applied = True
            new_layer = push_cnots_left(new_layer)
    return new_layer

def order_cnots(layer):
    new_layer = copy.deepcopy(layer)
    for lidx, lbl in enumerate(layer[:-1]):
        if new_layer[lidx][0]=='Gcnot' and new_layer[lidx+1][0] == 'Gcnot':
            if new_layer[lidx][1] > new_layer[lidx+1][1]:
                new_layer = swap_locs(new_layer,lidx,lidx+1)
                new_layer = order_cnots(new_layer)
    return new_layer

def order_locals(layer):
    new_layer = copy.deepcopy(layer)
    for lidx, lbl in enumerate(layer[:-1]):
        if new_layer[lidx][0]!='Gcnot' and new_layer[lidx+1][0] != 'Gcnot':
            if new_layer[lidx][1] > new_layer[lidx+1][1]:
                new_layer = swap_locs(new_layer,lidx,lidx+1)
                new_layer = order_locals(new_layer)
    return new_layer

def order_layer(layer):
    layer = push_cnots_left(layer)
    layer = order_cnots(layer)
    layer = order_locals(layer)
    return layer

def order_circuit(circuit):
    new_circuit = circuit.copy(editable=True)
    for lidx, layer in enumerate(circuit):
        if isinstance(layer, baseobjs.label.LabelTup):
            layer = (layer,)
        new_circuit[lidx] = order_layer(layer)
    new_circuit.done_editing()
    return new_circuit

def apply_errors_to_pygsti_model(model,errors,verbose=0):
    new_model = model.copy()
    new_model.set_all_parameterizations('full')
    for lidx, lbl in enumerate(errors.keys()):
        if verbose>=1:
            print(lidx, end=' ', flush=True)
        new_model[lbl] = errors[lbl] @ model[lbl].to_dense()
    new_model.set_all_parameterizations('full')   
    return new_model
