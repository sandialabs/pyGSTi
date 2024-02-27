import pygsti
import numpy as _np
import copy as _copy

from pygsti.algorithms import compilers as _cmpl
from pygsti.circuits import circuit as _cir
from pygsti.baseobjs import label as _lbl
from pygsti import tools as _tools
from pygsti.tools import group as _rbobjs
from pygsti.tools import symplectic as _symp
from pygsti.algorithms import randomcircuit as _rc
from pygsti.tools import compilationtools as _comp

from pygsti.protocols import protocol as _proto
from pygsti.algorithms import rbfit as _rbfit
from pygsti.algorithms import mirroring as _mirroring

from pygsti.protocols import rb as _rb
from pygsti.protocols import vb as _vb


from pygsti.processors import QubitProcessorSpec as QPS
from pygsti.processors import CliffordCompilationRules as CCR

def create_random_quintuple_layered_circuit(pspec, qubit_labels, length = 1, two_q_gate_density = .25, one_q_gate_names = 'all', pdist = 'uniform', modelname = 'clifford', rand_state = None):
    if qubit_labels is not None:
        n = len(qubit_labels)
        qubits = qubit_labels[:]  # copy this list
    else:
        n = pspec.num_qubits
        qubits = pspec.qubit_labels[:]  # copy this list
        
    circuit = _cir.Circuit([], line_labels = qubit_labels, editable = True)
    for i in range(length):
        layer = sample_quint_layer(pspec, qubit_labels, two_q_gate_density = two_q_gate_density, one_q_gate_names = one_q_gate_names, rand_state = rand_state, pdist = pdist, modelname = modelname)
        c = _cir.Circuit(layer, line_labels = qubit_labels)
        circuit.append_circuit_inplace(c)
    circuit.done_editing()
    return circuit

def sample_quint_layer(pspec, qubit_labels, two_q_gate_density = .25, gate_args_lists = None, rand_state = None, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all'):
    layer_1 = _rc.sample_circuit_layer_of_one_q_gates(pspec, qubit_labels, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all', rand_state=rand_state)
    layer_2 = [[]]
    mid_layer = _rc.sample_circuit_layer_by_edgegrab(pspec, qubit_labels, two_q_gate_density = two_q_gate_density, rand_state=rand_state)
    layer_4 = [[]]
    layer_5 = _rc.sample_circuit_layer_of_one_q_gates(pspec, qubit_labels, pdist = 'uniform', modelname = 'clifford', one_q_gate_names = 'all', rand_state=rand_state)
    return [layer_1, layer_2, mid_layer, layer_4, layer_5]
  
def stabilizer_to_all_zs(stabilizer, rand_state):
    if rand_state is None:
        rand_state = _np.random.RandomState()
        
    n = len(stabilizer)
    
    symp_reps = _symp.compute_internal_gate_symplectic_representations()
    
    s_inv_p, p_inv_p = _symp.inverse_clifford(symp_reps['P'][0],symp_reps['P'][1])
    s_h, p_h = symp_reps['H']
    s_y, p_y = symp_reps['C1']
    
    stab_layer = []
    c_str = [[]]
    
    for i in range(n):
        if stabilizer[i] == 'Y':
            stab_layer.append((s_y, p_y))
            c_str[0].append(('Gc1','Q{}'.format(i)))
        elif stabilizer[i] == 'X':
            stab_layer.append((s_h, p_h))
            c_str[0].append(('Gc12','Q{}'.format(i)))
        elif stabilizer[i] == 'I':
            rand_clifford = str(rand_state.choice(_np.arange(24)))
            s_rand, p_rand = symp_reps['C'+rand_clifford]
            stab_layer.append((s_rand, p_rand))
            c_str[0].append(('Gc'+rand_clifford,'Q{}'.format(i)))
        else:
            s_rand, p_rand = symp_reps['C0']
            stab_layer.append((s_rand, p_rand))
            c_str[0].append(('Gc0', 'Q{}'.format(i)))
            
    s_layer, p_layer = _symp.symplectic_kronecker(stab_layer)
    stab_circuit = _cir.Circuit(c_str).parallelize()
    
    return s_layer, p_layer, stab_circuit

def symplectic_to_pauli(s,p):
    # Takes in the symplectic representation of a Pauli (ie a 2n bitstring in the Hostens notation) and converts it into a list of characters
    # representing the corresponding stabilizer.
    #     - s: Length 2n bitstring.
    #     - p: The "global" phase.
    # Returns: A list of characters ('I','Y','Z','X') representing the stabilizer that corresponds to s.
    
    n = int(len(s)/2)
    pauli = []
    for i in range(n):
        x_pow = s[i]
        z_pow = s[n+i]
        if x_pow != 0 and z_pow != 0: # Have XZ in the i-th slot, ie product is a Y
            #print('need to undo a Y, apply HP^(-1)')
            pauli.append('Y')
        elif x_pow != 0 and z_pow == 0: # Have X in the i-th slot, ie product is an X
            #print('need to undo an X, so apply inverse Hadamard, ie a Hadamard')
            pauli.append('X')
        elif x_pow == 0 and z_pow != 0: # Have Z or I in the i-th slot, so nothing needs to be done
            #print('need to undo a Z or I, ie leave it be')
            pauli.append('Z')
        else:
            pauli.append('I')
            
    return pauli

def generic_pauli_sampler(n, include_identity = False, rand_state = None):
    if rand_state is None:
        rand_state = _np.random.RandomState()

    if include_identity is False:
        while True: 
            rand_ints = rand_state.randint(0,4,n)
            if sum(rand_ints) != 0: # make sure we don't get all identities
                break
    else:
        rand_ints = rand_state.randint(0, 4, n)

    return rand_ints

def mod_oneq_pauli_sampler(n, rand_state = None):
    if rand_state is None:
        rand_state = _np.random.RandomState()
    rand_ints = rand_state.randint(1, 4, n)

    return rand_ints


def sample_random_pauli(n, pspec = None, absolute_compilation = None, qubit_labels = None, circuit = False, pauli_sampler = generic_pauli_sampler, pauli_sampler_kwargs = {'include_identity': False}, rand_state = None):
    # Samples a random Pauli along with a +-1 phase. Returns the Pauli as a list or as a circuit depending 
    # upon the value of "circuit"
    #     - n: Number of qubits
    #     - pspec: Processor spec
    #     - absolute_compilation: compilation rules 
    #     - qubit_labels:
    #     - circuit: Boolean that determines if a list of single-qubit Paulis or a compiled circuit is returned.
    if rand_state is None:
        rand_state = _np.random.RandomState()
        
    if circuit is True:
        if qubit_labels is not None: qubits = qubit_labels[:]  # copy this list
        else: qubits = pspec.qubit_labels[:]
    
    pauli_list = ['I','X','Y','Z']

    rand_ints = pauli_sampler(n = n, rand_state = rand_state, **pauli_sampler_kwargs)
    
    #if include_identity is False:
    #    while True: 
    #        rand_ints = rand_state.randint(0,4,n)
    #        if sum(rand_ints) != 0: # make sure we don't get all identities
    #            break
    #else:
    #    rand_ints = rand_state.randint(0, 4, n)
            
    pauli = [pauli_list[i] for i in rand_ints]
    if set(pauli) != set('I'): sign = rand_state.choice([-1,1])
    else: sign = 1
    
    if circuit is False:
        return pauli, sign
    else:
        pauli_layer_std_lbls = [_lbl.Label(pauli_list[rand_ints[q]], (qubits[q],)) for q in range(n)]
        # Converts the layer to a circuit, and changes to the native model.
        pauli_circuit = _cir.Circuit(layer_labels=pauli_layer_std_lbls, line_labels=qubits).parallelize()
        pauli_circuit = pauli_circuit.copy(editable=True)
        pauli_circuit.change_gate_library(absolute_compilation)
        if pauli_circuit.depth == 0:
            pauli_circuit.insert_layer_inplace([_lbl.Label(())], 0)

        pauli_circuit.done_editing()
    return pauli, sign, pauli_circuit

      
def select_neg_evecs(pauli, sign, rand_state):
    # Selects the entries in an n-qubit that will be turned be given a -1 1Q eigenstates
    #     - pauli: The n-qubit Pauli
    #     - sign: Whether you want a -1 or +1 eigenvector
    # Returns: A bitstring whose 0/1 entries specify if you have a +1 or -1 1Q eigenstate
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    n = len(pauli)
    identity_bitstring = [0 if i == 'I' else 1 for i in pauli]
    nonzero_indices = _np.nonzero(identity_bitstring)[0]
    num_nid = len(nonzero_indices)
    if num_nid % 2 == 0:
        if sign == 1:
            choices = _np.arange(start = 0, stop = num_nid+1, step = 2)
        else:
            choices = _np.arange(start = 1, stop = num_nid, step = 2)
    else:
        if sign == 1:
            choices = _np.arange(start = 0, stop = num_nid, step = 2)
        else:
            choices = _np.arange(start = 1, stop = num_nid+1, step = 2)
    num_neg_evecs = rand_state.choice(choices)
    assert((-1)**num_neg_evecs == sign)
    
    neg_evecs = rand_state.choice(nonzero_indices, num_neg_evecs, replace = False)
    
    bit_evecs = _np.zeros(n)
    bit_evecs[neg_evecs] = 1
    assert('I' not in _np.array(pauli)[nonzero_indices])
    
    return bit_evecs

def compose_initial_cliffords(prep_circuit):
    composition_rules = {'Gc0': 'Gc3',
                         'Gc2': 'Gc5',
                         'Gc12': 'Gc15'} #supposed to give Gc# * X
    
    sign_layer = prep_circuit[0]
    circ_layer = prep_circuit[1]
    composed_layer = []
    
    for i in range(len(sign_layer)):
        sign_gate = sign_layer[i]
        circ_gate = circ_layer[i]
        new_gate = circ_gate
        if sign_gate == 'Gc3': # we know that what follows must prep a X, Y, or Z stablizer
            new_gate = composition_rules[circ_gate]
        composed_layer.append(new_gate)
    return composed_layer

def sample_stabilizer(pauli, sign, rand_state):
    # Samples a random stabilizer of a Pauli, s = s_1 \otimes ... \otimes s_n. For each s_i,
    # we perform the following gates:
    #     - s_i = X: H
    #     - s_i = Y: PH
    #     - s_i = Z: I
    #     - s_i = I: A random 1Q Clifford
    # Also creates the circuit layer that is needed to prepare
    # the stabilizer state. 
    #     - pauli: a list of 1Q paulis whose tensor product gives the n-qubit Pauli
    # Returns: The symplectic representation of the stabilizer state, symplectic representation of the 
    #          preparation circuit, and a pygsti circuit representation of the prep circuit
    if rand_state is None:
        rand_state = _np.random.RandomState()
    
    n = len(pauli)
    neg_evecs = select_neg_evecs(pauli, sign, rand_state = rand_state)
    assert((-1)**sum(neg_evecs) == sign)
    zvals = [0 if neg_evecs[i] == 0 else -1 for i in range(n)]
    
    #zvals = _np.random.choice([0,-1], n) 
    #print(zvals)
    
    # init_stab, init_phase = _symp.prep_stabilizer_state(n, zvals)
    init_stab, init_phase = _symp.prep_stabilizer_state(n)
    
    symp_reps = _symp.compute_internal_gate_symplectic_representations()
    
    layer_dict = {'X': symp_reps['H'],
                'Y': tuple(_symp.compose_cliffords(symp_reps['H'][0] 
                                                   ,symp_reps['H'][1]
                                                   ,symp_reps['P'][0]
                                                   ,symp_reps['P'][1])), 
                'Z': symp_reps['I']}
    circ_dict = {'X': 'Gc12',
                 'Y': 'Gc2',
                 'Z': 'Gc0'}
    
    x_layer = [symp_reps['I'] if zvals[i] == 0 else symp_reps['X'] for i in range(len(zvals))]
    
    circ_layer = [circ_dict[i] if i in circ_dict.keys() else 'Gc'+str(rand_state.randint(24)) for i in pauli]
    #init_layer = [layer_dict[pauli[i]] if pauli[i] in layer_dict.keys() else symp_reps[circ_layer[i].replace('Gc','C')] for i in range(len(pauli))]
    
    init_layer = [symp_reps[circ_layer[i].replace('Gc', 'C')] for i in range(len(pauli))]
                  
    x_layer_rep, x_layer_phase = _symp.symplectic_kronecker(x_layer)

    layer_rep, layer_phase = _symp.symplectic_kronecker(init_layer)
    
    #stab_state, stab_phase =  _symp.apply_clifford_to_stabilizer_state(layer_rep, layer_phase,
    #                                                                  stab_state, stab_phase) 
    
    s_prep, p_prep = _symp.compose_cliffords(x_layer_rep, x_layer_phase, layer_rep, layer_phase)
    
    stab_state, stab_phase = _symp.apply_clifford_to_stabilizer_state(s_prep, p_prep, init_stab, init_phase)
    
    sign_layer = ['Gc0' if zvals[i] == 0 else 'Gc3' for i in range(len(zvals))]
    
    layer = [sign_layer, circ_layer]
    
    # sign = (-1)**(-1*_np.sum(zvals))

    #return stab_state, stab_phase, layer_rep, layer_phase, circuit_rep
    return stab_state, stab_phase, s_prep, p_prep, layer

def measure(s_state, p_state):
    num_qubits = len(p_state) // 2
    outcome = []
    for i in range(num_qubits):
        p0, p1, ss0, ss1, sp0, sp1 = _symp.pauli_z_measurement(s_state, p_state, i)
        # could cache these results in a FUTURE _stabilizer_measurement_probs function?
        if p0 != 0:
            outcome.append(0)
            s_state, p_state = ss0, sp0 % 4
        else:
            outcome.append(1)
            s_state, p_state = ss1, sp1 % 4
    return outcome

def determine_sign(s_state, p_state, measurement):
    an_outcome = measure(s_state, p_state)
    sign = [-1 if bit == 1 and pauli == 'Z' else 1 for bit, pauli in zip(an_outcome, measurement)]
    return _np.prod(sign) 

def create_direct_rb_circuit_no_inversion(pspec, clifford_compilations, length, qubit_labels=None, sampler='Qelimination',
                             samplerargs=[], addlocal=False, lsargs=[],
                             citerations=20, compilerargs=[], partitioned=False, seed=None):
    
    if qubit_labels is not None: n = len(qubit_labels)
    else: 
        n = pspec.num_qubits
        qubit_labels = pspec.qubit_labels

    rand_state = _np.random.RandomState(seed)  # Ok if seed is None
    #s_start, p_start = _symp.prep_stabilizer_state(n)
    

    rand_pauli, rand_sign, pauli_circuit = sample_random_pauli(n = n, pspec = pspec, 
                                                                   absolute_compilation = clifford_compilations['absolute'],
                                                                   circuit = True)
    
    s_inputstate, p_inputstate, s_init_layer, p_init_layer, prep_circuit = sample_stabilizer(rand_pauli, rand_sign)
    prep_circuit = compose_initial_cliffords(prep_circuit)
    s_pc, p_pc = _symp.symplectic_rep_of_clifford_circuit(pauli_circuit, pspec = pspec)
    
    # build the initial layer of the blown up circuit
    initial_circuit = _cir.Circuit([[(prep_circuit[i], qubit_labels[i]) for i in range(len(qubit_labels))]])
    full_circuit = initial_circuit.copy(editable = True)

    # Sample a random circuit of "native gates".
    if sampler == 'Qelimination' or sampler == 'edgegrab':
        circuit = _rc.create_random_circuit(pspec=pspec, length=length, qubit_labels=qubit_labels, sampler=sampler,
                                    samplerargs=samplerargs, addlocal=addlocal, lsargs=lsargs, rand_state=rand_state)
    elif sampler == create_random_quintuple_layered_circuit:
        circuit = sampler(pspec, qubit_labels, length, *samplerargs, rand_state = rand_state)
    else:
        raise ValueError(f'{sampler} is not supported')
        
    # find the symplectic matrix / phase vector this "native gates" circuit implements.
    s_rc, p_rc = _symp.symplectic_rep_of_clifford_circuit(circuit, pspec=pspec)
    
    s_composite, p_composite = _symp.compose_cliffords(s1 = s_init_layer, p1 = p_init_layer, s2 = s_rc, p2 = p_rc)

    # Apply the random circuit to the initial state (either the all 0s or a random stabilizer state)
    
    full_circuit.append_circuit_inplace(circuit)
    
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_rc, p_rc, 
                                                                            s_inputstate, p_inputstate)
    

    # Figure out what stabilizer of s_outputstate, rand_pauli was mapped too
    s_rc_inv, p_rc_inv = _symp.inverse_clifford(s_rc, p_rc) # U^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_rc_inv, p_rc_inv, s_pc, p_pc) # PU^(-1)
    s_new_pauli, p_new_pauli = _symp.compose_cliffords(s_new_pauli, p_new_pauli, s_rc, p_rc) # UPaU^(-1)
        
    pauli_vector = p_new_pauli
    pauli = [i[0] for i in _symp.find_pauli_layer(pauli_vector, [j for j in range(n)])]
    measurement, phase = ['I' if i == 'I' else 'Z' for i in pauli], None #not needed
        
    # Turn the stabilizer into an all Z and I stabilizer. Append this to the circuit.
    
    s_stab, p_stab, stab_circuit = stabilizer_to_all_zs(pauli)
    
    full_circuit.append_circuit_inplace(stab_circuit)
    
    s_inv, p_inv = _symp.inverse_clifford(s_stab, p_stab)
    s_cc, p_cc = _symp.compose_cliffords(s_inv, p_inv, s_composite, p_composite)
    s_cc, p_cc = _symp.compose_cliffords(s_composite, p_composite, s_stab, p_stab) # MUPaU^(-1)M^(-1)
    
    meas = [i[0] for i in _symp.find_pauli_layer(p_cc, [j for j in range(n)])] # not needed
     
    s_outputstate, p_outputstate = _symp.apply_clifford_to_stabilizer_state(s_stab, p_stab, s_outputstate, p_outputstate)

    full_circuit.done_editing()
    sign = determine_sign(s_outputstate, p_outputstate, measurement)
    
    if not partitioned: outcircuit = full_circuit
    else: outcircuit = [initial_circuit, circuit, stab_circuit]
        
    return outcircuit, measurement, sign

class DirectRBNIDesign(_vb.BenchmarkingDesign):
    def __init__(self, pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels=None,
                 sampler='edgegrab', samplerargs=[0.25, ],
                 addlocal=False, lsargs=(),
                 citerations=20, compilerargs=(), partitioned=False, descriptor='A DRB experiment',
                 add_default_protocol=False, seed=None, verbosity=1, num_processes=1):

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        measurements = []
        signs = []

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for lnum, l in enumerate(depths):
            lseed = self.seed + lnum * circuits_per_depth
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths) with seed {}'.format(
                    circuits_per_depth, l, lnum + 1, len(depths), lseed))

            args_list = [(pspec, clifford_compilations, l)] * circuits_per_depth
            kwargs_list = [dict(qubit_labels=qubit_labels, sampler=sampler, samplerargs=samplerargs,
                                addlocal=addlocal, lsargs=lsargs,
                                citerations=citerations, compilerargs=compilerargs,
                                partitioned=partitioned,
                                seed=lseed + i) for i in range(circuits_per_depth)]
            #results = [_rc.create_direct_rb_circuit(*(args_list[0]), **(kwargs_list[0]))]  # num_processes == 1 case
            results = _tools.mptools.starmap_with_kwargs(create_direct_rb_circuit_no_inversion, circuits_per_depth,
                                                         num_processes, args_list, kwargs_list)

            circuits_at_depth = []
            measurements_at_depth = []
            signs_at_depth = []
            for c, meas, sign in results:
                circuits_at_depth.append(c)
                measurements_at_depth.append(meas)
                signs_at_depth.append(sign)

            circuit_lists.append(circuits_at_depth)
            measurements.append(measurements_at_depth)
            signs.append(signs_at_depth)

        self._init_foundation(depths, circuit_lists, measurements, signs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, addlocal, lsargs, citerations, compilerargs, partitioned, descriptor,
                              add_default_protocol)

    def _init_foundation(self, depths, circuit_lists, measurements, signs, circuits_per_depth, qubit_labels,
                         sampler, samplerargs, addlocal, lsargs, citerations, compilerargs, partitioned, descriptor,
                         add_default_protocol):
        super().__init__(depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = circuits_per_depth
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor
        if isinstance(sampler, str):
            self.sampler = sampler
        else:
            self.sampler = 'function'
        self.samplerargs = samplerargs
        self.addlocal = addlocal
        self.lsargs = lsargs
        self.partitioned = partitioned

        if add_default_protocol:
            if randomizeout:
                defaultfit = 'A-fixed'
            else:
                defaultfit = 'full'
            self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))
            
        self.auxfile_types['signs'] = 'json' # Makes sure that signs and measurements are saved seperately
        self.auxfile_types['measurements'] = 'json'

class SummaryStatistics(_proto.Protocol):
    """
    A protocol that can construct "summary" quantities from raw data.

    Parameters
    ----------
    name : str
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.

    Attributes
    ----------
    summary_statistics : tuple
        Static list of the categories of summary information this protocol can compute.

    circuit_statistics : tuple
        Static list of the categories of circuit information this protocol can compute.
    """
    summary_statistics = ('success_counts', 'total_counts', 'hamming_distance_counts',
                          'success_probabilities', 'polarization', 'adjusted_success_probabilities', 'energies')
    circuit_statistics = ('two_q_gate_count', 'depth', 'idealout', 'circuit_index', 'width')
    # dscmp_statistics = ('tvds', 'pvals', 'jsds', 'llrs', 'sstvds')

    def __init__(self, name):
        super().__init__(name)

    def _compute_summary_statistics(self, data, energy = False):
        """
        Computes all summary statistics for the given data.

        Parameters
        ----------
        data : ProtocolData
            The data to operate on.

        Returns
        -------
        NamedDict
        """
        
        def outcome_energy(outcome, measurement, sign):
            energy = 1
            for i,j in zip(outcome,measurement):
                if i == '1' and j == 'Z':
                    energy = -1*energy
            return sign*energy

        def avg_energy(dsrow, measurement, sign):
            energy = 0
            for i in dsrow.counts:
                out_eng = outcome_energy(i[0],measurement,sign)
                energy += dsrow.counts[i] * out_eng    
            return energy / dsrow.total
        
        def success_counts(dsrow, circ, idealout):
            if dsrow.total == 0: return 0  # shortcut?
            return dsrow.get(tuple(idealout), 0.)

        def hamming_distance_counts(dsrow, circ, idealout):
            nQ = len(circ.line_labels)  # number of qubits
            assert(nQ == len(idealout[-1]))
            hamming_distance_counts = _np.zeros(nQ + 1, float)
            if dsrow.total > 0:
                for outcome_lbl, counts in dsrow.counts.items():
                    outbitstring = outcome_lbl[-1]
                    hamming_distance_counts[_tools.rbtools.hamming_distance(outbitstring, idealout[-1])] += counts
            return hamming_distance_counts

        def adjusted_success_probability(hamming_distance_counts):

            """ A scaled success probability that is useful for mirror circuit benchmarks """
            if _np.sum(hamming_distance_counts) == 0.:
                return 0.
            else:
                hamming_distance_pdf = _np.array(hamming_distance_counts) / _np.sum(hamming_distance_counts)
                adjSP = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
                return adjSP
            
        def _get_energies(icirc, circ, dsrow, measurement, sign):
            eng = avg_energy(dsrow, measurement, sign)
            ret = {'energies': eng}
            return ret

        def _get_summary_values(icirc, circ, dsrow, idealout):
            sc = success_counts(dsrow, circ, idealout)
            tc = dsrow.total
            hdc = hamming_distance_counts(dsrow, circ, idealout)
            sp = _np.nan if tc == 0 else sc / tc
            nQ = len(circ.line_labels)
            pol = (sp - 1 / 2**nQ) / (1 - 1 / 2**nQ)
            ret = {'success_counts': sc,
                   'total_counts': tc,
                   'success_probabilities': sp,
                   'polarization': pol,
                   'hamming_distance_counts': hdc,
                   'adjusted_success_probabilities': adjusted_success_probability(hdc)}
            return ret
        
        if energy is False:
            return self._compute_dict(data, self.summary_statistics,
                                  _get_summary_values, for_passes='all')
        
        else:
            return self._compute_dict(data, ['energies'],
                                     _get_energies, for_passes = 'all', energy = True)
        # Double check what _compute_dict does for other cases

    def _compute_circuit_statistics(self, data):
        """
        Computes all circuit statistics for the given data.

        Parameters
        ----------
        data : ProtocolData
            The data to operate on.

        Returns
        -------
        NamedDict
        """
        def _get_circuit_values(icirc, circ, dsrow, idealout):
            ret = {'two_q_gate_count': circ.two_q_gate_count(),
                   'depth': circ.depth,
                   'idealout': idealout,
                   'circuit_index': icirc,
                   'width': len(circ.line_labels)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self._compute_dict(data, self.circuit_statistics, _get_circuit_values, for_passes="first")

    # def compute_dscmp_data(self, data, dscomparator):

    #     def get_dscmp_values(icirc, circ, dsrow, idealout):
    #         ret = {'tvds': dscomparator.tvds.get(circ, _np.nan),
    #                'pvals': dscomparator.pVals.get(circ, _np.nan),
    #                'jsds': dscomparator.jsds.get(circ, _np.nan),
    #                'llrs': dscomparator.llrs.get(circ, _np.nan)}
    #         return ret

    #     return self.compute_dict(data, "dscmpdata", self.dsmp_statistics, get_dscmp_values, for_passes="none")

    def _compute_predicted_probs(self, data, model):
        """
        Compute the predicted success probabilities of `model` given `data`.

        Parameters
        ----------
        data : ProtocolData
            The data.

        model : SuccessFailModel
            The model.

        Returns
        -------
        NamedDict
        """
        def _get_success_prob(icirc, circ, dsrow, idealout):
            #if set(circ.line_labels) != set(qubits):
            #    trimmedcirc = circ.copy(editable=True)
            #    for q in circ.line_labels:
            #        if q not in qubits:
            #            trimmedcirc.delete_lines(q)
            #else:
            #    trimmedcirc = circ
            return {'success_probabilities': model.probabilities(circ)[('success',)]}

        return self._compute_dict(data, ('success_probabilities',),
                                  _get_success_prob, for_passes="none")

    def _compute_dict(self, data, component_names, compute_fn, for_passes="all", energy = False):
        """
        Executes a computation function row-by-row on the data in `data` and packages the results.

        Parameters
        ----------
        data : ProtocolData
            The data.

        component_names : list or tuple
            A sequence of string-valued component names which must be the keys of the dictionary
            returned by `compute_fn`.

        compute_fn : function
            A function that computes values for each item in `component_names` for each row of data.
            This function should have signature:
            `compute_fn(icirc : int, circ : Circuit, dsrow : _DataSetRow, idealout : OutcomeLabel)`
            and should return a dictionary whose keys are the same as `component_names`.

        for_passes : {'all', 'none', 'first'}
            UNUSED.  What passes within `data` values are computed for.

        Returns
        -------
        NamedDict
            A nested dictionary with indices: component-name, depth, circuit-index
            (the last level is a *list*, not a dict).
        """
        design = data.edesign
        ds = data.dataset

        depths = design.depths
        qty_data = _tools.NamedDict('Datatype', 'category', None, None,
                                    {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float',
                                                            {depth: [] for depth in depths})
                                     for comp in component_names})

        #loop over all circuits
        if energy is False:
            for depth, circuits_at_depth, idealouts_at_depth in zip(depths, design.circuit_lists, design.idealout_lists):
                for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ??
                # -- this is where Tim thinks there's a bottleneck, as these loops will be called for each
                # member of a simultaneous experiment separately instead of having an inner-more iteration
                # that loops over the "structure", i.e. the simultaneous qubit sectors.
                #TODO: <print percentage>

                    for component_name, val in compute_fn(icirc, circ, dsrow, idealout).items():
                        qty_data[component_name][depth].append(val)  # maybe use a pandas dataframe here?
        else:
            for depth, circuits_at_depth, measurements_at_depth, signs_at_depth in zip(depths, design.circuit_lists, design.measurements, design.signs):
                for icirc, (circ, measurement, sign) in enumerate(zip(circuits_at_depth, measurements_at_depth, signs_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None
                    
                    for component_name, val in compute_fn(icirc, circ, dsrow, measurement, sign).items():
                        qty_data[component_name][depth].append(val)
    
        return qty_data

    def _create_depthwidth_dict(self, depths, widths, fillfn, seriestype):
        """
        Create a nested :class:`NamedDict` with depht and width indices.

        Parameters
        ----------
        depths : list or tuple
            The (integer) depths to use.

        widths : list or tuple
            The (integer) widths to use.

        fillfn : function
            A function with no arguments that is called to return a default value
            for each (depth, width).

        seriestype : {"float", "int"}
            The type of values held by this nested dict.

        Returns
        -------
        NamedDict
        """
        return _tools.NamedDict(
            'Depth', 'int', None, None, {depth: _tools.NamedDict(
                'Width', 'int', 'Value', seriestype, {width: fillfn() for width in widths}) for depth in depths})

    def _add_bootstrap_qtys(self, data_cache, num_qtys, finitecounts=True):
        """
        Adds bootstrapped "summary data".

        The bootstrap is over both the finite counts of each circuit and
        over the circuits at each length.

        Note: only adds quantities if they're needed.

        Parameters
        ----------
        data_cache : dict
            A cache of already-existing bootstraps.

        num_qtys : int, optional
            The number of bootstrapped data to construct.

        finitecounts : bool, optional
            Whether finite counts should be used, i.e. whether the bootstrap samples
            include finite sample error with the same number of counts as the sampled
            data, or whether they have no finite sample error (just probabilities).

        Returns
        -------
        None
        """
        key = 'bootstraps' if finitecounts else 'infbootstraps'
        if key in data_cache:
            num_existing = len(data_cache['bootstraps'])
        else:
            data_cache[key] = []
            num_existing = 0

        #extract "base" values from cache, to base boostrap off of
        # Wonky try statements aren't working...
        try:
            success_probabilities = data_cache['success_probabilities']
        except:
            success_probabilities = data_cache['energies']
        try:
            total_counts = data_cache['total_counts']
        except:
            pass
        try:
            hamming_distance_counts = data_cache['hamming_distance_counts']
        except:
            pass
        depths = list(success_probabilities.keys())

        for i in range(num_existing, num_qtys):

            component_names = self.summary_statistics
            bcache = _tools.NamedDict(
                'Datatype', 'category', None, None,
                {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
                 for comp in component_names})  # ~= "RB summary dataset"

            for depth, SPs in success_probabilities.items():
                numcircuits = len(SPs)
                for k in range(numcircuits):
                    ind = _np.random.randint(numcircuits)
                    sampledSP = SPs[ind]
                    totalcounts = total_counts[depth][ind] if finitecounts else None
                    bcache['success_probabilities'][depth].append(sampledSP)
                    if finitecounts:
                        if not _np.isnan(sampledSP):
                            bcache['success_counts'][depth].append(_np.random.binomial(totalcounts, sampledSP))
                        else:
                            bcache['success_probabilities'][depth].append(sampledSP)
                        bcache['total_counts'][depth].append(totalcounts)
                    else:
                        bcache['success_counts'][depth].append(sampledSP)

                    #ind = _np.random.randint(numcircuits)  # note: old code picked different random ints
                    #totalcounts = total_counts[depth][ind] if finitecounts else None  # need this if a new randint
                    sampledHDcounts = hamming_distance_counts[depth][ind]
                    sampledHDpdf = _np.array(sampledHDcounts) / _np.sum(sampledHDcounts)

                    if finitecounts:
                        if not _np.isnan(sampledSP):
                            bcache['hamming_distance_counts'][depth].append(
                                list(_np.random.multinomial(totalcounts, sampledHDpdf)))
                        else:
                            bcache['hamming_distance_counts'][depth].append(sampledHDpdf)
                    else:
                        bcache['hamming_distance_counts'][depth].append(sampledHDpdf)

                    # replicates adjusted_success_probability function above
                    adjSP = _np.sum([(-1 / 2)**n * sampledHDpdf[n] for n in range(len(sampledHDpdf))])
                    bcache['adjusted_success_probabilities'][depth].append(adjSP)

            data_cache[key].append(bcache)

class RandomizedBenchmarking(SummaryStatistics):
    """
    The randomized benchmarking protocol.

    This same analysis protocol is used for Clifford, Direct and Mirror RB.
    The standard Mirror RB analysis is obtained by setting
    `datatype` = `adjusted_success_probabilities`.

    Parameters
    ----------
    datatype: 'success_probabilities' or 'adjusted_success_probabilities', optional
        The type of summary data to extract, average, and the fit to an exponential decay. If
        'success_probabilities' then the summary data for a circuit is the frequency that
        the target bitstring is observed, i.e., the success probability of the circuit. If
        'adjusted_success_probabilties' then the summary data for a circuit is
        S = sum_{k = 0}^n (-1/2)^k h_k where h_k is the frequency at which the output bitstring is
        a Hamming distance of k from the target bitstring, and n is the number of qubits.
        This datatype is used in Mirror RB, but can also be used in Clifford and Direct RB.
        
        #added in "energies" as a datatype for DRB without Inversion

    defaultfit: 'A-fixed' or 'full'
        The summary data is fit to A + Bp^m with A fixed and with A as a fit parameter.
        If 'A-fixed' then the default results displayed are those from fitting with A
        fixed, and if 'full' then the default results displayed are those where A is a
        fit parameter.

    asymptote : 'std' or float, optional
        The summary data is fit to A + Bp^m with A fixed and with A has a fit parameter,
        with the default results returned set by `defaultfit`. This argument specifies the
        value used when 'A' is fixed. If left as 'std', then 'A' defaults to 1/2^n if
        `datatype` is `success_probabilities` and to 1/4^n if `datatype` is
        `adjusted_success_probabilities`.

    rtype : 'EI' or 'AGI', optional
        The RB error rate definition convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic Pauli errors.
        'AGI' results in RB error rates that are associated with the average gate infidelity.

    seed : list, optional
        Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

    bootstrap_samples : float, optional
        The number of samples for generating bootstrapped error bars.

    depths: list or 'all'
        If not 'all', a list of depths to use (data at other depths is discarded).

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """

    def __init__(self, datatype='success_probabilities', defaultfit='full', asymptote='std', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', square_mean_root=False, name=None):
        """
        Initialize an RB protocol for analyzing RB data.

        Parameters
        ----------
        datatype: 'success_probabilities' or 'adjusted_success_probabilities', optional
            The type of summary data to extract, average, and the fit to an exponential decay. If
            'success_probabilities' then the summary data for a circuit is the frequency that
            the target bitstring is observed, i.e., the success probability of the circuit. If
            'adjusted_success_probabilties' then the summary data for a circuit is
            S = sum_{k = 0}^n (-1/2)^k h_k where h_k is the frequency at which the output bitstring is
            a Hamming distance of k from the target bitstring, and n is the number of qubits.
            This datatype is used in Mirror RB, but can also be used in Clifford and Direct RB.
            
            # Added in "energies" to work with Direct RB without Inversion

        defaultfit: 'A-fixed' or 'full'
            The summary data is fit to A + Bp^m with A fixed and with A as a fit parameter.
            If 'A-fixed' then the default results displayed are those from fitting with A
            fixed, and if 'full' then the default results displayed are those where A is a
            fit parameter.

        asymptote : 'std' or float, optional
            The summary data is fit to A + Bp^m with A fixed and with A has a fit parameter,
            with the default results returned set by `defaultfit`. This argument specifies the
            value used when 'A' is fixed. If left as 'std', then 'A' defaults to 1/2^n if
            `datatype` is `success_probabilities` and to 1/4^n if `datatype` is
            `adjusted_success_probabilities`.

        rtype : 'EI' or 'AGI', optional
            The RB error rate definition convention. 'EI' results in RB error rates that are associated
            with the entanglement infidelity, which is the error probability with stochastic Pauli errors.
            'AGI' results in RB error rates that are associated with the average gate infidelity.

        seed : list, optional
            Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

        bootstrap_samples : float, optional
            The number of samples for generating bootstrapped error bars.

        depths: list or 'all'
            If not 'all', a list of depths to use (data at other depths is discarded).

        name : str, optional
            The name of this protocol, also used to (by default) name the
            results produced by this protocol.  If None, the class name will
            be used.
        """
        super().__init__(name)

        assert(datatype in self.summary_statistics), "Unknown data type: %s!" % str(datatype)
        assert(datatype in ('success_probabilities', 'adjusted_success_probabilities', 'energies')), \
            "Data type '%s' must be 'success_probabilities' or 'adjusted_success_probabilities'!" % str(datatype)

        self.seed = seed
        self.depths = depths
        self.bootstrap_samples = bootstrap_samples
        self.asymptote = asymptote
        self.rtype = rtype
        self.datatype = datatype
        self.defaultfit = defaultfit
        self.square_mean_root = square_mean_root
        if self.datatype == 'energies':
            self.energies = True
        else:
            self.energies = False

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        RandomizedBenchmarkingResults
        """
        design = data.edesign

        if self.datatype not in data.cache:
            summary_data_dict = self._compute_summary_statistics(data, energy = self.energies)
            data.cache.update(summary_data_dict)
        src_data = data.cache[self.datatype]
        data_per_depth = src_data

        if self.depths == 'all':
            depths = list(data_per_depth.keys())
        else:
            depths = filter(lambda d: d in data_per_depth, self.depths)

        nqubits = len(design.qubit_labels)

        if isinstance(self.asymptote, str):
            assert(self.asymptote == 'std'), "If `asymptote` is a string it must be 'std'!"
            if self.datatype == 'success_probabilities':
                asymptote = 1 / 2**nqubits
            elif self.datatype == 'adjusted_success_probabilities':
                asymptote = 1 / 4**nqubits
            elif self.datatype == 'energies':
                asymptote = 0
            else:
                raise ValueError("No 'std' asymptote for %s datatype!" % self.asymptote)

        def _get_rb_fits(circuitdata_per_depth):
            adj_sps = []
            for depth in depths:
                percircuitdata = circuitdata_per_depth[depth]
                #print(percircuitdata)
                if self.square_mean_root:
                    #print(percircuitdata)
                    adj_sps.append(_np.nanmean(_np.sqrt(percircuitdata))**2)
                    #print(adj_sps)
                else:
                    adj_sps.append(_np.nanmean(percircuitdata))  # average [adjusted] success probabilities or energies

            #print(adj_sps)
            # Don't think this needs changed
            full_fit_results, fixed_asym_fit_results = _rbfit.std_least_squares_fit(
                depths, adj_sps, nqubits, seed=self.seed, asymptote=asymptote,
                ftype='full+FA', rtype=self.rtype)

            return full_fit_results, fixed_asym_fit_results

        #do RB fit on actual data
        # Think this works just fine
        ff_results, faf_results = _get_rb_fits(data_per_depth)

        if self.bootstrap_samples > 0:

            parameters = ['a', 'b', 'p', 'r']
            bootstraps_ff = {p: [] for p in parameters}
            bootstraps_faf = {p: [] for p in parameters}
            failcount_ff = 0
            failcount_faf = 0

            #Store bootstrap "cache" dicts (containing summary keys) as a list under data.cache
            if 'bootstraps' not in data.cache or len(data.cache['bootstraps']) < self.bootstrap_samples:
                # TIM - finite counts always True here?
                self._add_bootstrap_qtys(data.cache, self.bootstrap_samples, finitecounts=True)
            bootstrap_caches = data.cache['bootstraps']  # if finitecounts else 'infbootstraps'

            for bootstrap_cache in bootstrap_caches:
                bs_ff_results, bs_faf_results = _get_rb_fits(bootstrap_cache[self.datatype])

                if bs_ff_results['success']:
                    for p in parameters:
                        bootstraps_ff[p].append(bs_ff_results['estimates'][p])
                else:
                    failcount_ff += 1
                if bs_faf_results['success']:
                    for p in parameters:
                        bootstraps_faf[p].append(bs_faf_results['estimates'][p])
                else:
                    failcount_faf += 1

            failrate_ff = failcount_ff / self.bootstrap_samples
            failrate_faf = failcount_faf / self.bootstrap_samples

            std_ff = {p: _np.std(_np.array(bootstraps_ff[p])) for p in parameters}
            std_faf = {p: _np.std(_np.array(bootstraps_faf[p])) for p in parameters}

        else:
            bootstraps_ff = None
            std_ff = None
            failrate_ff = None

            bootstraps_faf = None
            std_faf = None
            failrate_faf = None
        # we are here
        fits = _tools.NamedDict('FitType', 'category')
        fits['full'] = _rbfit.FitResults(
            'LS', ff_results['seed'], self.rtype, ff_results['success'], ff_results['estimates'],
            ff_results['variable'], stds=std_ff, bootstraps=bootstraps_ff,
            bootstraps_failrate=failrate_ff)

        fits['A-fixed'] = _rbfit.FitResults(
            'LS', faf_results['seed'], self.rtype, faf_results['success'],
            faf_results['estimates'], faf_results['variable'], stds=std_faf,
            bootstraps=bootstraps_faf, bootstraps_failrate=failrate_faf)

        return RandomizedBenchmarkingResults(data, self, fits, depths, self.defaultfit)


class RandomizedBenchmarkingResults(_proto.ProtocolResults):
    """
    The results of running randomized benchmarking.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.

    fits : dict
        A dictionary of RB fit parameters.

    depths : list or tuple
        A sequence of the depths used in the RB experiment. The x-values
        of the RB fit curve.

    defaultfit : str
        The default key within `fits` to plot when calling :method:`plot`.
    """

    def __init__(self, data, protocol_instance, fits, depths, defaultfit):
        """
        Initialize an empty RandomizedBenchmarkingResults object.
        """
        super().__init__(data, protocol_instance)

        self.depths = depths  # Note: can be different from protocol_instance.depths (which can be 'all')
        self.rtype = protocol_instance.rtype  # replicated for convenience?
        self.fits = fits
        self.defaultfit = defaultfit
        self.auxfile_types['fits'] = 'dict:serialized-object'  # b/c NamedDict don't json

    def plot(self, fitkey=None, decay=True, success_probabilities=True, size=(8, 5), ylim=None, xlim=None,
             legend=True, title=None, figpath=None):
        """
        Plots RB data and, optionally, a fitted exponential decay.

        Parameters
        ----------
        fitkey : dict key, optional
            The key of the self.fits dictionary to plot the fit for. If None, will
            look for a 'full' key (the key for a full fit to A + Bp^m if the standard
            analysis functions are used) and plot this if possible. It otherwise checks
            that there is only one key in the dict and defaults to this. If there are
            multiple keys and none of them are 'full', `fitkey` must be specified when
            `decay` is True.

        decay : bool, optional
            Whether to plot a fit, or just the data.

        success_probabilities : bool, optional
            Whether to plot the success probabilities distribution, as a violin plot. (as well
            as the *average* success probabilities at each length).

        size : tuple, optional
            The figure size

        ylim : tuple, optional
            The y-axis range.

        xlim : tuple, optional
            The x-axis range.

        legend : bool, optional
            Whether to show a legend.

        title : str, optional
            A title to put on the figure.

        figpath : str, optional
            If specified, the figure is saved with this filename.

        Returns
        -------
        None
        """

        # Future : change to a plotly plot.
        try: import matplotlib.pyplot as _plt
        except ImportError: raise ValueError("This function requires you to install matplotlib!")

        if decay and fitkey is None:
            if self.defaultfit is not None:
                fitkey = self.defaultfit
            else:
                allfitkeys = list(self.fits.keys())
                if 'full' in allfitkeys: fitkey = 'full'
                else:
                    assert(len(allfitkeys) == 1), \
                        ("There are multiple fits, there is no defaultfit and none have the key "
                         "'full'. Please specify the fit to plot!")
                    fitkey = allfitkeys[0]

        adj_sps = []
        data_per_depth = self.data.cache[self.protocol.datatype]
        for depth in self.depths:
            percircuitdata = data_per_depth[depth]
            adj_sps.append(_np.mean(percircuitdata))  # average [adjusted] success probabilities

        _plt.figure(figsize=size)
        _plt.plot(self.depths, adj_sps, 'o', label='Average success probabilities')

        if decay:
            lengths = _np.linspace(0, max(self.depths), 200)
            a = self.fits[fitkey].estimates['a']
            b = self.fits[fitkey].estimates['b']
            p = self.fits[fitkey].estimates['p']
            #_plt.plot(lengths, a + b * p**lengths,
            #          label='Fit, r = {:.2} +/- {:.1}'.format(self.fits[fitkey].estimates['r'],
            #                                                  self.fits[fitkey].stds['r']))
            _plt.plot(lengths, a + b * p**lengths,
                      label='Fit, r = {:.2}'.format(self.fits[fitkey].estimates['r']))

        if success_probabilities:
            all_success_probs_by_depth = [data_per_depth[depth] for depth in self.depths]
            _plt.violinplot(all_success_probs_by_depth, self.depths, points=10, widths=1.,
                            showmeans=False, showextrema=False, showmedians=False)  # , label='Success probabilities')

        if title is not None: _plt.title(title)
        _plt.ylabel("Success probability")
        _plt.xlabel("RB depth $(m)$")
        _plt.ylim(ylim)
        _plt.xlim(xlim)

        if legend: _plt.legend()

        if figpath is not None: _plt.savefig(figpath, dpi=1000)
        else: _plt.show()

        return
    

RB = RandomizedBenchmarking
RBResults = RandomizedBenchmarkingResults  # shorthand