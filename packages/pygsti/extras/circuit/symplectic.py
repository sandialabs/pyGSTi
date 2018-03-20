from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
import sys

if sys.version_info >= (3,):
    long = int

from . import matrixmod2 as _mtx

def symplectic_form(n,convention='standard'):
    """
    Creates the relevant symplectic form for the number of 
    qubits specified.
    
    Parameters
    ----------
    n : int
        The number of qubits the symplectic form should be constructed for. That
        is, the function creates a 2n x 2n matrix that is a sympletic form
        over the finite field of the integers modulo 2.
        
    convention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form. In the case of 'standard', the symplectic
        form is the 2n x 2n matrix of ((0,1),(1,0)), where '1' and '0' are the identity
        and all-zeros matrices of size n x n. The 'standard' symplectic form is the
        convention used throughout most of the code. In the case of 'directsum', the
        symplectic form is the direct sum of n 2x2 bit-flip matrices.

    Returns
    -------
    numpy array
        The specified symplectic form.
        
    """
    nn = 2*n
    sym_form = _np.zeros((nn,nn),int)
    
    assert(convention == 'standard' or convention == 'directsum')
    
    if convention == 'standard':
        sym_form[n:nn,0:n] = _np.identity(n,int)
        sym_form[0:n,n:nn] = _np.identity(n,int)
                
    if convention == 'directsum':
        # This current construction method is perhaps pretty stupid.
        for j in range(0,n):
            sym_form[2*j,2*j+1] = 1
            sym_form[2*j+1,2*j] = 1    

    return sym_form

def change_symplectic_form_convention(s,outconvention='standard'):
    """
    Maps the input symplectic matrix between the 'standard' and 'directsum'
    symplectic form conventions. That is, if the input is a symplectic matrix
    with respect to the 'directsum' convention and outconvention ='standard' the 
    output of this function is the equivalent symplectic matrix in the
    'standard' symplectic form convention. Similarily, if the input is a 
    symplectic matrix with respect to the 'standard' convention and 
    outconvention = 'directsum' the output of this function is the equivalent 
    symplectic matrix in the 'direcsum' symplectic form convention. 
    
    Parameters
    ----------
    n : int
        The number of qubits the symplectic form should be constructed for. That
        is, the function creates a 2n x 2n matrix that is a sympletic form
        over the finite field of the integers modulo 2.
        
    outconvention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form. This is the convention the input is
        being converted to (and so the input should be a symplectic matrix in the
        other convention).

    Returns
    -------
    numpy array
        The specified symplectic form.
        
    """
    n = _np.shape(s)[0]//2 
       
    if n == 1:
        return _np.copy(s)
        
    permutation_matrix = _np.zeros((2*n,2*n),int)        
    for i in range(0,n):
        permutation_matrix[2*i,i] = 1
        permutation_matrix[2*i+1,n+i] = 1   
    
    if outconvention == 'standard':
        sout = _np.dot(_np.dot(permutation_matrix.T,s),permutation_matrix)
    
    if outconvention == 'directsum':
        sout = _np.dot(_np.dot(permutation_matrix,s),permutation_matrix.T)
    
    return sout


def check_symplectic(m,convention='standard'):
    """
    Checks whether a matrix is symplectic.
    
    Parameters
    ----------
    m : numpy array
        The matrix to check.
        
    convention : str, optional
        Can be either 'standard' or 'directsum', Specifies the convention of
        the symplectic form with respect to which the matrix should be
        sympletic.

    Returns
    -------
    bool
        A bool specifying whether the matrix is symplectic
        
    """
    
    n = _np.shape(m)[0]//2
    s_form = symplectic_form(n,convention=convention)  
    conj = _mtx.dotmod2(_np.dot(m,s_form),_np.transpose(m))
        
    return _np.array_equal(conj,s_form)

def inverse_symplectic(s):
    """
    Returns the inverse of a symplectic matrix over the integers mod 2.
    
    Parameters
    ----------
    s : numpy array
        The matrix to invert

    Returns
    -------
    numpy array
        The inverse of s, over the field of the integers mod 2.
        
    """ 
    assert(check_symplectic(s)), "The input matrix is not symplectic!"
    
    n = _np.shape(s)[0]//2
    s_form = symplectic_form(n)
    s_inverse = _mtx.dotmod2(_np.dot(s_form,_np.transpose(s)),s_form)
    
    assert(check_symplectic(s_inverse)), "The inverse is not symplectic. Function has failed"
    assert(_np.array_equal(_mtx.dotmod2(s_inverse,s), _np.identity(2*n,int))), "The found matrix is not the inverse of the input. Function has failed"
     
    return s_inverse

def inverse_clifford(s,p):
    """
    Returns the inverse of a Clifford gate in the symplectic representation.
    
    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford
        
    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    Returns
    -------
    sinverse : numpy array
        The symplectic matrix representing the inverse of the input Clifford.
        
    pinverse : numpy array
        The 'phase vector' representing the inverse of the input Clifford.
        
    """ 
    assert(check_valid_clifford(s,p)), "The input symplectic matrix - phase vector pair does not define a valid Clifford!"
    
    sinverse = inverse_symplectic(s)
    
    n = _np.shape(s)[0]//2
    
    # The formula used below for the inverse p vector comes from Hostens 
    # and De Moor PRA 71, 042315 (2005).
    u = _np.zeros((2*n,2*n),int)
    u[n:2*n,0:n] = _np.identity(n,int)
    
    vec1 = -1*_np.dot(_np.transpose(sinverse), p)
    inner = _np.dot(_np.dot(_np.transpose(sinverse), u), sinverse)
    temp = 2*_mtx.strictly_upper_triangle(inner) + _mtx.diagonal_as_matrix(inner)
    temp = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s), temp), s))
    vec2 =  -1*_np.dot(_np.transpose(sinverse), temp)
    vec3 = _mtx.diagonal_as_vec(inner)
    
    pinverse = vec1 + vec2 + vec3   
    pinverse = pinverse % 4

    assert(check_valid_clifford(sinverse,pinverse)), "The output does not define a valid Clifford. Function has failed"
    
    s_check, p_check = compose_cliffords(s,p,sinverse,pinverse)
    assert(_np.array_equal(s_check, _np.identity(2*n,int))), "The output is not the inverse of the input. Function has failed"
    assert(_np.array_equal(p_check, _np.zeros(2*n,int))), "The output is not the inverse of the input. Function has failed"
    
    s_check, p_check = compose_cliffords(sinverse,pinverse,s,p)
    assert(_np.array_equal(s_check, _np.identity(2*n,int))), "The output is not the inverse of the input. Function has failed"
    assert(_np.array_equal(p_check, _np.zeros(2*n,int))), "The output is not the inverse of the input. Function has failed"
    
    return sinverse, pinverse

def check_valid_clifford(s,p):
    """
    Checks if a symplectic matrix - phase vector pair (s,p) is the symplectic representation of 
    a Clifford.
    
    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford
        
    p : numpy array
        The 'phase vector' over the integers mod 4 representing the Clifford

    Returns
    -------
    bool
        True if (s,p) is the symplectic representation of some Clifford.
        
    """ 
    
    # Checks if the matrix s is symplectic, which is the only constraint on s.
    is_symplectic_matrix = check_symplectic(s)
    
    # Check whether the phase vector is valid. This currently does *not* check
    # that p is a vector over [0,1,2,3]. Perhaps it should. The constraint
    # that we check is satisfied comes from Hostens and De Moor PRA 71, 042315 (2005).
    n = _np.shape(s)[0]//2
    u = _np.zeros((2*n,2*n),int)
    u[n:2*n,0:n] = _np.identity(n,int) 
    vec = p + _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s),u),s)) 
    vec = vec % 2
    
    is_valid_phase_vector = _np.array_equal(vec,_np.zeros(len(p),int))
        
    return (is_symplectic_matrix and is_valid_phase_vector)

def construct_valid_phase_vector(s,pseed):
    """
    Constructs a phase vector that, when paired with the provided symplectic matrix, defines
    a Clifford gate. Any sympletic matrix s is a representation of some Clifford when paired
    with *some* phase vector. This finds any such phase vector, starting from the provided
    seed. If the seed phase vector is -- along with s -- a representation of some Clifford
    this seed is returned.
    
    Parameters
    ----------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the Clifford
        
    pseed : numpy array
        The seed 'phase vector' over the integers mod 4.

    Returns
    -------
    numpy array
        Some p such that (s,p) is the symplectic representation of some Clifford.
        
    """    
    pout = pseed.copy()    
    n = _np.shape(s)[0]//2
    
    assert(check_symplectic(s)), "The input matrix is not symplectic!"
    
    u = _np.zeros((2*n,2*n),int)
    u[n:2*n,0:n] = _np.identity(n,int)
    
    # Each element of this vector should be 0 (mod 2) if this is a valid phase vector. 
    # This comes from the formulas in Hostens and De Moor PRA 71, 042315 (2005).
    vec = pout + _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s),u),s)) 
    vec = vec % 2
    
    # Adds 1 mod 4 to all the elements of the vector where the required constraint is
    # not satisfied. This is then always a valid phase vector.
    pout[vec != 0] += 1
    pout = pout % 4
    
    assert(check_valid_clifford(s,pout)), "The output does not define a valid Clifford. Function has failed"
       
    return pout

def compose_cliffords(s1,p1,s2,p2):
    """
    Multiplies two cliffords in the symplectic representation. The output corresponds 
    to the symplectic representation of C2 times C1 (i.e., C1 acts first) where s1 
    (s2) and p1 (p2) are the symplectic matrix and phase vector, respectively, for 
    Clifford C1 (C2).
    
    Parameters
    ----------
    s1 : numpy array
        The symplectic matrix over the integers mod 2 representing the first Clifford
    
    p1 : numpy array
        The 'phase vector' over the integers mod 4 representing the first Clifford
        
    s2 : numpy array
        The symplectic matrix over the integers mod 2 representing the second Clifford
        
    p2 : numpy array
        The 'phase vector' over the integers mod 4 representing the second Clifford

    Returns
    -------
    s : numpy array
        The symplectic matrix over the integers mod 2 representing the composite Clifford
     
    p : numpy array
        The 'phase vector' over the integers mod 4 representing the compsite Clifford
        
    """    
    assert(_np.shape(s1) == _np.shape(s2)), "Input must be Cliffords acting on the same number of qubits!"
    assert(check_valid_clifford(s1,p1)), "The first matrix-vector pair is not a valid Clifford!"
    assert(check_valid_clifford(s2,p2)), "The second matrix-vector pair is not a valid Clifford!"
    
    n = _np.shape(s1)[0] // 2
    
    # Below we calculate the s and p for the composite Clifford using the formulas from
    # Hostens and De Moor PRA 71, 042315 (2005).
    s = _mtx.dotmod2(s2,s1)    
    
    u = _np.zeros((2*n,2*n),int)
    u[n:2*n,0:n] = _np.identity(n,int)
    
    vec1 = _np.dot(_np.transpose(s1),p2)
    inner = _np.dot(_np.dot(_np.transpose(s2),u),s2)
    matrix = 2*_mtx.strictly_upper_triangle(inner)+_mtx.diagonal_as_matrix(inner)
    vec2 = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s1),matrix),s1))
    vec3 = _np.dot(_np.transpose(s1),_mtx.diagonal_as_vec(inner))
    
    p = p1 + vec1 + vec2 - vec3    
    p = p % 4
            
    assert(check_valid_clifford(s,p)), "The output is not a valid Clifford! Function has failed."
    
    return s, p

def symplectic_representation(gllist=None):
    """
    Returns dictionaries containing the symplectic matrices and phase vectors that represent
    the specified 'standard' Clifford gates, or the representations of all the standard gates
    if no list of gate labels is supplied. These 'standard' Clifford gates are those gates that
    are already known to the code gates (e.g., the label 'CNOT' has a specfic meaning in the
    code).
    
    Parameters
    ----------
    gllist : list, optional
        If not None, a list of strings corresponding to gate labels for any of the standard 
        gates that have fixed meaning for the code (e.g., 'CNOT' corresponds to
        the CNOT gate with the first qubit the target). For example, this list could be
        gllist = ['CNOT','H','P','I','X'].

    Returns
    -------
    s_dict : dictionary of numpy arrays
        The symplectic matrices representing the requested standard gates.
        
    p_dict : dictionary of numpy arrays
        The phase vectors representing the requested standard gates.

    """    
    # Full dictionaries, containing the symplectic representations of *all* gates
    # that are hard-coded, and which have a specific meaning to the code.
    complete_s_dict = {}
    complete_p_dict = {}
    
    # The Pauli gates
    complete_s_dict['I'] = _np.array([[1,0],[0,1]],int)
    complete_s_dict['X'] = _np.array([[1,0],[0,1]],int)
    complete_s_dict['Y'] = _np.array([[1,0],[0,1]],int)
    complete_s_dict['Z'] = _np.array([[1,0],[0,1]],int)
    
    complete_p_dict['I'] = _np.array([0,0],int)
    complete_p_dict['X'] = _np.array([0,2],int)
    complete_p_dict['Y'] = _np.array([2,2],int)
    complete_p_dict['Z'] = _np.array([2,0],int)
    
    # Five single qubit gates that each represent one of five classes of Cliffords 
    # that equivalent up to Pauli gates and are not equivalent to idle (that class
    # is covered by any one of the Pauli gates above). 
    complete_s_dict['H'] = _np.array([[0,1],[1,0]],int)
    complete_s_dict['P'] = _np.array([[1,0],[1,1]],int)
    complete_s_dict['PH'] = _np.array([[0,1],[1,1]],int)
    complete_s_dict['HP'] = _np.array([[1,1],[1,0]],int)
    complete_s_dict['HPH'] = _np.array([[1,1],[0,1]],int)
    
    complete_p_dict['H'] = _np.array([0,0],int)
    complete_p_dict['P'] = _np.array([1,0],int)
    complete_p_dict['PH'] = _np.array([0,1],int)
    complete_p_dict['HP'] = _np.array([3,0],int)
    complete_p_dict['HPH'] = _np.array([0,3],int)
       
    # The CNOT gate, CPHASE gate, and SWAP gate.
    complete_s_dict['CNOT'] = _np.array([[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]],int)    
    complete_s_dict['CPHASE'] = _np.array([[1,0,0,0],[0,1,0,0],[0,1,1,0],[1,0,0,1]])
    complete_s_dict['SWAP'] = _np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    
    complete_p_dict['CNOT'] = _np.array([0,0,0,0],int)
    complete_p_dict['CPHASE'] = _np.array([0,0,0,0],int)
    complete_p_dict['SWAP'] = _np.array([0,0,0,0],int)
    
    if gllist is None:
        s_dict = complete_s_dict
        p_dict = complete_p_dict
    
    else:    
        s_dict = {}
        p_dict = {}
    
        for glabel in gllist:
            s_dict[glabel] = complete_s_dict[glabel]
            p_dict[glabel] = complete_s_dict[glabel]
    
    return s_dict, p_dict

def composite_clifford_from_clifford_circuit(circuit, s_dict=None, p_dict=None):
    """
    Returns the symplectic representation of the composite Clifford implemented by 
    the specified Clifford circuit.
    
    Parameters
    ----------
    circuit : Circuit
        The Clifford circuit to calculate the global action of, input as a
        Circuit object.
        
    s_dict : dict, optional
        If not None, a dictionary providing the symplectic matrix associated with
        each gate label (i.e., the 'phase vector' component of the symplectic 
        representation of that gate). If the circuit is contains only gates from 
        the 'standard' gates which have a hard-coded symplectic representation this 
        may be None. Otherwise it must be specified.
        
    p_dict : dict, optional
        If not None, a dictionary providing the phase vector associated with
        each gate label (i.e., the 'phase vector' component of the symplectic 
        representation of that gate). If the circuit is contains only gates from the 
        'standard' gates which have a hard-coded symplectic representation this 
        may be None. Otherwise it must be specified.

    Returns
    -------
    s : numpy array
        The symplectic matrix representing the Clifford implement by the input circuit
        
    p : dictionary of numpy arrays
        The phase vector representing the Clifford implement by the input circuit
        
    """    
    n = circuit.number_of_qubits
    depth = circuit.depth()
    
    if (s_dict is None) or (p_dict is None):        
        # Get the symplectic representation for all gates that are in-built into the code
        s_dict, p_dict = symplectic_representation()
    
    # The initial action of the circuit before any layers are applied.
    s = _np.identity(2*n,int)
    p = _np.zeros(2*n,int)
    
    for i in range(0,depth):        
        layer = circuit.get_circuit_layer(i)
        layer_s, layer_p = clifford_layer_in_symplectic_rep(layer, n, s_dict,  p_dict)
        s, p = compose_cliffords(s, p, layer_s, layer_p)
    
    return s, p

def clifford_layer_in_symplectic_rep(layer, n, s_dict=None, p_dict=None):
    """
    Returns the symplectic representation of the n-qubit Clifford implemented by a
    single quantum circuit layer.
    
    Parameters
    ----------
    layer : list    
        The Clifford circuit layer to calculate the global action of, input as a
        list of Gate objects.
        
    s_dict : dict, optional
        If not None, a dictionary providing the symplectic matrix associated with
        each gate label (i.e., the 'phase vector' component of the symplectic 
        representation of that gate). If the circuit layer contains only 'standard' 
        gates which have a hard-coded symplectic representation this 
        may be None. Otherwise it must be specified.
        
    p_dict : dict, optional
        If not None, a dictionary providing the phase vector associated with
        each gate label (i.e., the 'phase vector' component of the symplectic 
        representation of that gate). If the circuit layer contains only
        'standard' gates which have a hard-coded symplectic representation this 
        may be None. Otherwise it must be specified.

    Returns
    -------
    s : numpy array
        The symplectic matrix representing the Clifford implement by specified 
        circuit layer
        
    p : dictionary of numpy arrays
        The phase vector representing the Clifford implement by specified 
        circuit layer
        
    """ 
    
    #
    # Todo: this method is currently pretty stupid. It is probably useful to keep it, but for
    # the circuit function above to not use it, and instead just perform the action of each 
    # gate.
    #
    
    if (s_dict is None) or (p_dict is None):        
        # Get the symplectic representation for all gates that are in-built into the code
        s_dict, p_dict = symplectic_representation()
            
    s = _np.identity(2*n,int)
    p = _np.zeros(2*n,int)
    
    for gate in layer:
        
        # Checks below are commented out as they are very inefficient, so it is probably
        # better to just allow a key error.
        #assert(gate.label in list(s_dict.keys())), "Symplectic matrix for some gate labels not provided!"
        #assert(gate.label in list(p_dict.keys())), "Phase vector for some gate labels not provided!"
        matrix = s_dict[gate.label]
        phase = p_dict[gate.label]
        
        assert(gate.number_of_qubits == 1 or gate.number_of_qubits == 2), "Only 1 and 2 qubit gates are allowed!"
        
        if gate.number_of_qubits == 1:
            
            q = gate.qubits[0]
            s[q,q] = matrix[0,0]
            s[q,q+n] = matrix[0,1]
            s[q+n,q] = matrix[1,0]
            s[q+n,q+n] = matrix[1,1]
            p[q] = phase[0]
            p[q+n] = phase[1]
            
        else:
            
            q1 = gate.qubits[0]
            q2 = gate.qubits[1]
            for i in [0,1]:
                for j in [0,1]:
                    s[q1+i*n,q1+j*n] = matrix[0+2*i,0+2*j]
                    s[q1+i*n,q2+j*n] = matrix[0+2*i,1+2*j]
                    s[q2+i*n,q1+j*n] = matrix[1+2*i,0+2*j]
                    s[q2+i*n,q2+j*n] = matrix[1+2*i,1+2*j]
                    
            p[q1] = phase[0]
            p[q2] = phase[1]
            p[q1+n] = phase[2]
            p[q2+n] = phase[3]
                
    return s, p

def single_qubit_clifford_symplectic_group_relations():
    
    #
    # Todo : docstring, and think about whether this function does what we want or not.
    #
    group_relations = {}
    
    group_relations['I','I'] = 'I'
    group_relations['I','H'] = 'H'
    group_relations['I','P'] = 'P'
    group_relations['I','HP'] = 'HP'
    group_relations['I','PH'] = 'PH'
    group_relations['I','HPH'] = 'HPH'
    
    group_relations['H','I'] = 'H'
    group_relations['H','H'] = 'I'
    group_relations['H','P'] = 'PH'
    group_relations['H','HP'] = 'HPH'
    group_relations['H','PH'] = 'P'
    group_relations['H','HPH'] = 'HP'
    
    group_relations['P','I'] = 'P'
    group_relations['P','H'] = 'HP'
    group_relations['P','P'] = 'I'
    group_relations['P','HP'] = 'H'
    group_relations['P','PH'] = 'HPH'
    group_relations['P','HPH'] = 'PH'
    
    group_relations['HP','I'] = 'HP'
    group_relations['HP','H'] = 'P'
    group_relations['HP','P'] = 'HPH'
    group_relations['HP','HP'] = 'PH'
    group_relations['HP','PH'] = 'I'
    group_relations['HP','HPH'] = 'H'
    
    group_relations['PH','I'] = 'PH'
    group_relations['PH','H'] = 'HPH'
    group_relations['PH','P'] = 'H'
    group_relations['PH','HP'] = 'I'
    group_relations['PH','PH'] = 'HP'
    group_relations['PH','HPH'] = 'P'
    
    group_relations['HPH','I'] = 'HPH'
    group_relations['HPH','H'] = 'PH'
    group_relations['HPH','P'] = 'HP'
    group_relations['HPH','HP'] = 'P'
    group_relations['HPH','PH'] = 'H'
    group_relations['HPH','HPH'] = 'I'
    
    return group_relations

def unitary_to_symplectic_1Q(u,flagnonclifford=True):
    """
    Returns the symplectic representation of a single qubit Clifford unitary, 
    input as a complex matrix in the standard computational basis.
    
    Parameters
    ----------
    u : numpy array    
        The unitary matrix to construct the symplectic representation for. This
        must be a single-qubit gate (so, it is a 2 x 2 matrix), and it must be
        in the standard computational basis. E.g., the unitary for the Z gate
        is matrix ((1.,0.),(0.,-1.)). It also must be a Clifford gate in the
        standard sense.

    flagnonclifford : bool, opt
        If True, an error is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    """
    assert(_np.shape(u) == (2,2)), "Input is not a single qubit unitary!"
    
    x = _np.array([[0,1.],[1.,0]])
    z = _np.array([[1.,0],[0,-1.]])
    fund_paulis = [x,z]

    s = _np.zeros((2,2),int)
    p = _np.zeros(2,int)
    
    for pauli_label in range(0,2):
        
        # Calculate the matrix that the input unitary transforms the current Pauli group
        # generator to (X or Z).
        conj = _np.dot(_np.dot(u,fund_paulis[pauli_label]),_np.linalg.inv(u))
        
        # Find which element of the Pauli group this is, and fill out the relevant
        # bits of the symplectic matrix and phase vector as this implies.
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,4):
                    
                    pauli_ijk = (1j**(k))*_np.dot(_np.linalg.matrix_power(x,i),_np.linalg.matrix_power(z,j))
                    
                    if _np.allclose(conj,pauli_ijk):
                        s[:,pauli_label] = _np.array([i,j])
                        p[pauli_label] = k
    
    valid_clifford = check_valid_clifford(s,p)
    
    if flagnonclifford:
        assert(valid_clifford), "Input unitary is not a Clifford with respect to the standard basis!"
    
    else:
        if not valid_clifford:
            s = None
            p = None
        
    return s, p

    
def unitary_to_symplectic_2Q(u,flagnonclifford=True):
    """
    Returns the symplectic representation of a two-qubit Clifford unitary, 
    input as a complex matrix in the standard computational basis.
    
    Parameters
    ----------
    u : numpy array    
        The unitary matrix to construct the symplectic representation for. This
        must be a two-qubit gate (so, it is a 4 x 4 matrix), and it must be
        in the standard computational basis. It also must be a Clifford gate in the
        standard sense.

    flagnonclifford : bool, opt
        If True, an error is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    """
    assert(_np.shape(u) == (4,4)), "Input is not a two-qubit unitary!"
    
    x = _np.array([[0,1.],[1.,0]])
    z = _np.array([[1.,0],[0,-1.]])
    i = _np.array([[1.,0.],[0.,1.]])
    
    xi = _np.kron(x,i)
    ix = _np.kron(i,x)
    zi = _np.kron(z,i)   
    iz = _np.kron(i,z)
    
    fund_paulis = [xi,ix,zi,iz]
    
    s = _np.zeros((4,4),int)
    p = _np.zeros(4,int)
   
    for pauli_label in range(0,4):
        
        # Calculate the matrix that the input unitary transforms the current Pauli group
        # generator to (xi, ix, ...).
        conj = _np.dot(_np.dot(u,fund_paulis[pauli_label]),_np.linalg.inv(u))
        
        # Find which element of the two-qubit Pauli group this is, and fill out the relevant
        # bits of the symplectic matrix and phase vector as this implies.
        for xi_l in range(0,2):
            for ix_l in range(0,2):
                for zi_l in range(0,2):
                    for iz_l in range(0,2):
                        for phase_l in range(0,4):
                    
                            tempx = _np.dot(_np.linalg.matrix_power(xi,xi_l),_np.linalg.matrix_power(ix,ix_l))
                            tempz = _np.dot(_np.linalg.matrix_power(zi,zi_l),_np.linalg.matrix_power(iz,iz_l))
                            pauli = (1j**(phase_l))*_np.dot(tempx,tempz)
                    
                            if _np.allclose(conj,pauli):
                                s[:,pauli_label] = _np.array([xi_l,ix_l,zi_l,iz_l])
                                p[pauli_label] = phase_l
    
    valid_clifford = check_valid_clifford(s,p)
    
    if flagnonclifford:
        assert(valid_clifford), "Input unitary is not a Clifford with respect to the standard basis!"
    
    else:
        if not valid_clifford:
            s = None
            p = None
        
    return s, p
    
def unitary_to_symplectic(u,flagnonclifford=True):
    """
    Returns the symplectic representation of a one-qubit or two-qubit Clifford 
    unitary, input as a complex matrix in the standard computational basis.
    
    Parameters
    ----------
    u : numpy array    
        The unitary matrix to construct the symplectic representation for. This
        must be a one-qubit or two-qubit gate (so, it is a 2 x 2 or 4 x 4 matrix), and 
        it must be provided in the standard computational basis. It also must be a 
        Clifford gate in the standard sense.

    flagnonclifford : bool, opt
        If True, an error is raised when the input unitary is not a Clifford gate.
        If False, when the unitary is not a Clifford the returned s and p are
        None.

    Returns
    -------
    s : numpy array or None
        The symplectic matrix representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    p : numpy array or None
        The phase vector representing the unitary, or None if the input unitary
        is not a Clifford and flagnonclifford is False
        
    """
    assert(_np.shape(u) == (2,2) or _np.shape(u) == (4,4)), "Input must be a one or two qubit unitary!"
    
    if _np.shape(u) == (2,2):
        s, p = unitary_to_symplectic_1Q(u,flagnonclifford)
    if _np.shape(u) == (4,4):
        s, p = unitary_to_symplectic_2Q(u,flagnonclifford)
        
    return s, p

def random_symplectic_matrix(n,convention='standard'):
    """
    Returns a symplectic matrix of dimensions 2n x 2n sampled uniformly at random
    from the symplectic group S(n). This uses the method of Robert Koenig and John
    A. Smolin, presented in "How to efficiently select an arbitrary Clifford group 
    element".
    
    Parameters
    ----------
    n : int 
        The size of the symplectic group to sample from.
    
    convention : str, optional
        Can be either 'standard' or 'directsum', which correspond to two different
        definitions for the symplectic form. In the case of 'standard', the symplectic
        form is the 2n x 2n matrix of ((0,1),(1,0)), where '1' and '0' are the identity
        and all-zeros matrices of size n x n. The 'standard' symplectic form is the
        convention used throughout most of the code. In the case of 'directsum', the
        symplectic form is the direct sum of n 2x2 bit-flip matrices.
    
    Returns
    -------
    s : numpy array
        A uniformly sampled random symplectic matrix.
               
    """  
    
    index = random_symplectic_index(n)
    s = get_symplectic_matrix(index, n)
    
    if convention == 'standard':
        s = change_symplectic_form_convention(s)
        
    return s
        
def random_clifford(n):
    """
    Returns a Clifford, in the symplectic representation, sampled uniformly at random
    from the n-qubit Clifford group. The core of this function uses the method of 
    Robert Koenig and John A. Smolin, presented in "How to efficiently select an 
    arbitrary Clifford group element", for sampling a uniformly random symplectic 
    matrix.
    
    Parameters
    ----------
    n : int 
        The number of qubits the Clifford group is over.
    
    Returns
    -------
    s : numpy array
        The symplectic matrix representating the uniformly sampled random Clifford.
        
    p : numpy array
        The phase vector representating the uniformly sampled random Clifford.    
        
    """  
    
    s = random_symplectic_matrix(n,convention='standard')
    p = _np.zeros(2*n,int)
            
    # A matrix to hold all possible phase vectors -- half of which do not, when
    # combined with the sampled symplectic matrix -- represent Cliffords.
    all_values = _np.zeros((2*n,4),int)
    for i in range(0,2*n):
        all_values[i,:] = _np.array([0,1,2,3])
    
    # We now work out which of these are valid choices for the phase vector.
    possible = _np.zeros((2*n,4),bool)
    
    u = _np.zeros((2*n,2*n),int)
    u[n:2*n,0:n] = _np.identity(n,int)    
    v = _mtx.diagonal_as_vec(_np.dot(_np.dot(_np.transpose(s),u),s))
    v_matrix = _np.zeros((2*n,4),int)
    
    for i in range(0,4):
        v_matrix[:,i] = v
    
    summed = all_values + v_matrix
    possible[summed % 2 == 0] = True
    
    # The valid choices for the phase vector, to sample uniformly from.
    allowed_values = _np.reshape(all_values[possible],(2*n,2))
    
    # Sample a uniformly random valid phase vector.
    index = _np.random.randint(2,size=2*n)
    for i in range(0,2*n):
        p[i] = allowed_values[i,index[i]]
        
    assert(check_valid_clifford(s,p))
    
    return s,p


# Below here is code taken from the appendix of "How to efficiently select an arbitrary Clifford 
# group element",
# by Robert Koenig and John A. Smolin. It is almost exactly the same as that code, and has only
# had minor edits to make it work properly. A couple of the basic utility routines from that code
# have been moved into the matrixtools file.

#
# Todo : go through the code below and find / removing anything that duplicates other parts of 
# the code. This code also needs commenting and docstrings need adding. Write tests for all of
# the functions.

def numberofcliffords(n):
    """
    The number of Clifford gates in the n-qubit Clifford group.
    
    Parameters
    ----------
    n : int 
        The number of qubits the Clifford group is over.
    
    Returns
    -------
    long integer
       The cardinality of the n-qubit Clifford group.

    """  
    return (long(4)**long(n))*numberofsymplectic(n)

def numberofsymplectic(n): 
    """
    The number of elements in the symplectic group S(n).
    
    """  
    x = long(1)
    for j in range(1,n+1): 
        x = x*numberofcosets(j)
        
    return x

def numberofcosets(n): 
    # returns the number of different cosets 
    x= long(2)**long(2*n-1) *((long(2)**long(2*n))-long(1))    
    #x= _np.power(long(2) ,long(2*n-1))*(_np.power(long(2) ,long(2*n))-long(1))
    return x

def symplectic_innerproduct(v,w): 
    """
    Returns the symplectic inner product of two vectors F_2^(2n), where
    F_2 is the finite field containing 0 and 1, and n is the 
    """
    t=0
    for i in range(0,_np.size(v)>>1):
        t+= v[2*i]* w[2*i + 1]
        t+= w[2*i] * v[2*i + 1]
    return t%2

def symplectic_transvection(k,v): 
    """
    Applies transvection Z k to v
    
    """
    return (v+symplectic_innerproduct(k,v)*k)%2

def int_to_bitstring(i ,n): 
    """
    converts integer i to an length n array of bits 
    """
    output= _np.zeros (n, dtype='int8')    
    for j in range(0,n):
        output [j] = i&1
        i >>=1
        
    return output

def bitstring_to_int(b,nn): 
    # converts an nn-bit string b to an integer between 0 and 2^nn - 1
    output = 0
    tmp = 1
    
    for j in range(0,nn):
        if b[j] == 1: 
            output = output + tmp
        tmp = tmp*2 
        
    return output



def find_symplectic_transvection(x,y): 
# finds h1,h2 such that y = Z h1 Z h2 x 
# Lemma 2 in the text
# Note that if only one transvection is required output [1] will be
# zero and applying the all-zero transvection does nothing.

    output= _np.zeros((2, _np.size(x)),dtype='int8')
    if _np.array_equal(x,y): 
        return output
    if symplectic_innerproduct(x,y) == 1:
        output[0] = (x + y)%2 
        return output
    
    # Try to find a pair where they are both not 00
    z= _np.zeros(_np.size(x))
    for i in range(0,_np.size(x)>>1):
        ii=2*i
        if ((x[ii]+x[ii+1]) != 0) and ((y[ii]+y[ii+1]) != 0): # found the pair
            z[ii] = (x[ii] + y[ii])%2
            z[ii+1]=(x[ii+1] + y[ii+1])%2
            if (z[ii]+z[ii+1]) == 0: # they were the same so they added to 00 
                z[ii+1] = 1
                if x[ii] != x[ii+1]: 
                    z[ii] = 1
            output[0]=(x+z)%2 
            output[1]=(y+z)%2 
            return output
        
    #Failed to find any such pair, so look for two places where x has 00 and y does not,
    #and vice versa. First try y==00 and x does not.
    for i in range(0,_np.size(x)>>1):
        ii=2*i
        if ((x[ii]+x[ii+1]) != 0) and ((y[ii]+y[ii+1]) == 0): # found the pair
            if x[ii]==x[ii+1]: 
                z[ii+1]=1
            else:
                z[ii+1]=x[ii]
                z[ii]=x[ii+1] 
            break
            
    # finally try x==00 and y does not
    for i in range(0,_np.size(x)>>1): 
        ii=2*i
        if ((x[ii]+x[ii+1]) == 0) and ((y[ii]+y[ii+1]) != 0): # found the pair    
            if y[ii]==y[ii+1]:
                z[ii+1] = 1 
            else:
                z[ii+1] = y[ii]
                z[ii] = y[ii+1] 
            break
        
    output[0]=(x+z)%2 
    output[1]=(y+z)%2 
    
    return output

def get_symplectic_matrix(i ,n): 
    # output symplectic canonical matrix i of size 2nX2n
    #Note, compared to the text the transpose of the symplectic matrix is returned. 
    #This is not particularly important since Transpose(g in Sp(2n)) is in Sp(2n)
    #but it means the program doesnt quite agree with the algorithm in the text.
    #In python, row ordering of matrices is convenient , so it is used internally , 
    #but for column ordering is used in the text so that matrix multiplication of 
    #symplectics will correspond to conjugation by unitaries as conventionally defined Eq. (2). 
    #We cant just return the transpose every time as this would alternate doing the incorrect 
    #thing as the algorithm recurses.
    
    nn=2*n
    
    # step 1
    s = ((1<<nn) - 1) 
    k = (i%s) + 1
    i //= s

    # step 2 
    f1=int_to_bitstring(k,nn)

    # step 3
    e1 = _np.zeros(nn,dtype='int8') # define first basis vectors 
    e1[0] = 1
    T = find_symplectic_transvection(e1,f1) # use Lemma 2 to compute T 

    # step 4
    # b[0]=b in the text, b[1]...b[2n-2] are b_3...b_2n in the text
    bits= int_to_bitstring(i%(1<<(nn-1)),nn-1)

    # step 5
    eprime= _np.copy(e1)
    for j in range(2,nn):
        eprime[j] = bits[j-1] 
        
    h0 = symplectic_transvection(T[0] ,eprime) 
    h0 = symplectic_transvection(T[1] ,h0)

    # step 6
    if bits[0] == 1: 
        f1 *= 0
        
    #T' from the text will be Z_f1 Z_h0. If f1 has been set to zero it doesnt do anything.
    #We could now compute f2 as said in the text but step 7 is slightly 
    # changed and will recompute f1, f2 for us anyway

    # step 7
    id2 = _np.identity(2,dtype='int8')

    if n != 1:
        g = _mtx.matrix_directsum(id2 ,get_symplectic_matrix(i>>(nn-1),n-1))
    else :
        g = id2
        
    for j in range(0,nn):
        g[j] = symplectic_transvection(T[0], g[j]) 
        g[j] = symplectic_transvection(T[1], g[j]) 
        g[j] = symplectic_transvection(h0, g[j]) 
        g[j] = symplectic_transvection(f1, g[j])

    return g

def get_symplectic_label(gn,n=None): 
    # produce an index associated with group element gn
    
    if n is None:
        n = _np.shape(gn)[0]//2
        
    nn=2*n 
    
    # step 1 
    v = gn[0]
    w = gn[1]
    
    # step 2
    e1 = _np.zeros(nn,dtype='int8') # define first basis vectors 
    e1[0] = 1
    T = find_symplectic_transvection(v,e1) # use Lemma 2 to compute T

    # step 3
    tw = _np.copy(w) 
    tw = symplectic_transvection(T[0], tw) 
    tw = symplectic_transvection(T[1], tw) 
    b = tw[0]    
    h0 = _np.zeros(nn, dtype='int8') 
    h0[0] = 1
    h0[1] = 0
    for j in range(2,nn): 
        h0[j] = tw[j]
        
    # step 4
    bb = _np.zeros (nn-1,dtype='int8') 
    bb[0] = b
    for j in range(2,nn): 
        bb[j-1] = tw[j]
    zv = bitstring_to_int(v,nn) - 1 
    zw = bitstring_to_int(bb,nn - 1)
    cvw = zw*((long(2)**long(2*n))-1) + zv

    #step 5
    if n == 1:
        return cvw
    
    #step 6 
    gprime = _np.copy(gn);
    if b == 0:
        for j in range(0,nn):
            gprime[j] = symplectic_transvection(T[1], symplectic_transvection(T[0], gn[j]))
            gprime[j] = symplectic_transvection(h0, gprime[j])
            gprime[j] = symplectic_transvection(e1, gprime[j])
    else:
        for j in range(0,nn):
            gprime[j] = symplectic_transvection(T[1], symplectic_transvection(T[0], gn[j])) 
            gprime[ j]=symplectic_transvection(h0, gprime[j])

    # step 7
    gnew = gprime[2:nn,2:nn] # take submatrix 
    gnidx = get_symplectic_label(gnew,n - 1) * numberofcosets(n) + cvw
    return gnidx

def random_symplectic_index(n):
                    
    cardinality = numberofsymplectic(n)       
    max_integer = 9223372036854775808 # The maximum integer of int64 type
    
    def zeros_string(k):       
        zeros_str = ''        
        for j in range(0,k):
            zeros_str += '0'
        return zeros_str
        
    if cardinality <= max_integer:
        index = _np.random.randint(cardinality)

    else:
        digits1 = len(str(cardinality))
        digits2 = len(str(max_integer))-1
        n = digits1//digits2
        m = digits1 - n*digits2
        
        index = cardinality 
        while index >= cardinality:        
            
            temp = long(0)
            for i in range(0,n):
                add = zeros_string(m)
                sample = _np.random.randint(10**digits2)
                for j in range(0,i):
                    add += zeros_string(digits2)
                add += str(sample)
                for j in range(i+1,n):
                    add += zeros_string(digits2)                
                temp += long(add)

            add = str(_np.random.randint(10**m)) + zeros_string(n*digits2)
            index = long(add) + temp 
    
    return index


def symplectic_action(m, glabel, qlist, optype='row'):
    """
    Todo: docstring
    
    """
    assert(optype == 'row' or optype == 'column'), "optype must be 'row' or 'column'."
    
    #
    # Todo: add the option of also updating a phase vector
    # Todo: add all other 'standard' gate actions here
    # Todo: add column action
    #
    
    d = _np.shape(m)[0]//2         
    out = m.copy()
       
    if glabel == 'H':
        
        i = qlist[0]
        
        if optype == 'row':
            i = qlist[0]
            out[i,:] = m[i+d,:]   
            out[i+d,:] = m[i,:]
        else:
            assert(False),"This functionality has not yet been added!"
            
    elif glabel == 'P':
        
        i = qlist[0]        
        
        if optype == 'row':
            out[i+d,:] = m[i,:] ^ m[i+d,:]
        else:
            assert(False),"This functionality has not yet been added!"
            
    elif glabel == 'CNOT':
        
        i = qlist[0]
        j = qlist[1]
        
        if optype == 'row':
            out[j,:] = m[j,:] ^ m[i,:]    
            out[i+d,:] = m[j+d,:] ^ m[i+d,:]        
        else:
            assert(False),"This functionality has not yet been added!"
                            
    elif glabel == 'SWAP':
        
        i = qlist[0]
        j = qlist[1]
        
        if optype == 'row':
            out[j,:] = m[i,:]
            out[i,:] = m[j,:] 
            out[i+d,:] = m[j+d,:] 
            out[j+d,:] = m[i+d,:]
        else:
            assert(False),"This functionality has not yet been added!"   
    
    else:
        assert(False),"Label is not valid or currently supported"
        
    return out