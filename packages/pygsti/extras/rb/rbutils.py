""" Randomized Benhmarking Utility Routines """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import warnings as _warnings
from collections import OrderedDict as _OrderedDict
from ... import construction as _cnst
from ... import algorithms as _algs
from ... import objects as _objs
from ... import tools as _tls
from scipy.linalg import sqrtm
import itertools as _ittls

# ---- Fitting functions and related ----#
def standard_fit_function(m,A,B,p):
    """
    The standard RB decay fitting function P_m = A + B * p^m. This is 
    used in standard RB, and also variants on this (e.g., interleaved RB).
    
    Parameters
    ----------
    m : integer
        Length of random RB sequence (not including the inversion gate).
    
    A,B,p : float

    Returns
    -------
    float
    """
    return A+B*p**m

def first_order_fit_function(m,A,B,C,p):
    """
    The 'first order' fitting function P_m = A + (B + m * C) * p^m, from
    "Scalable and Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).
    This is a simplified verion of the 'first order' in that paper (see Eq. 3),
    as the model therein has 1 too many parameters for fitting. The conversion is
    A = B_1
    B = A_1 - C_1(q/p^(-2) - 1)
    C = C_1(q/p^(-2) - 1)
    where the LHS (RHS) quantites in this equation are those of our (Magesan 
    et al.'s) fitting function.

    Parameters
    ----------
    m : integer
        Length of random RB sequence (not including the inversion gate).
    
    A,B,C,p : float

    Returns
    -------
    float
    """
    return A+(B+C*m)*p**m

def p_to_r(p,d=2):
    """
    Converts an RB decay rate (p) to the RB number (r), using the relation
    r = (1 - p) * (d - 1) / d, given in, e.g., Magesan et al PRA 85
    042311 2012, or arXiv:1702.01853.
    
    Parameters
    ----------
    p : float
        Fit parameter p from P_m = A + B*p**m.
    
    d : int, optional
        Number of dimensions of the Hilbert space (default is 2,
        corresponding to a single qubit).     
     
    Returns
    -------
    r : float
        The RB number      
    
    """
    r = (1 - p) * (d - 1) / d
    return r

def r_to_p(r,d=2):
    """
    Inverse of p_to_r function. 
    
    """
    p = 1 - d * r / (d - 1)
    return p

# ----- general gate and gateset tools ---- #
def average_gate_infidelity(A,B,d=None,mxBasis="gm"):
    """
    Computes the average gate infidelity (AGI) between an actual and a target
    gate. This quantity is defined in, e.g., arXiv:1702.01853 (see Eq 2).
    Calculated via the relationship between entanglement fidelity (F_p) and average 
    gate fidelity (F_g): F_g = (d * F_p + 1)/(1 + d) given in Phys. Lett. A 303 
    (2002) 249-252.

    Parameters
    ----------
    A : array or gate
        The noisy gate whose AGI is to be computed (to the target gate B).
        
    B : array or gate
        The target gate against which "actual" is being compared.
        
    d : int
        Dimension of Hilbert space.  Taken to be `sqrt(A.shape[0])` if None.

    mxBasis : {"std","gm","pp"} or Basis object, optional
        The basis of the matrices.

    Returns
    ----------
    AGI : float
        The AGI of the noisy to the target gate.
    """
    if d is None: d = int(round(_np.sqrt(A.shape[0])))
    process_fidelity = _tls.process_fidelity(A,B,mxBasis=mxBasis)
    AGF = (d*process_fidelity + 1)/(1+d)
    AGI = 1 - AGF
    return float(AGI)

def unitarity(A,mxBasis="gm",d=2):
    """
    Returns the unitarity of a channel calculated using the equation
    u(A) = Tr( A_u^{\dagger} A_u ) / (d^2  - 1), where A_u is the unital 
    submatrix of A and d is the dimension. The unital submatrix of A is the 
    matrix obtained when the top row, and left hand column is removed from A,
    when A is written in any basis for which the first element is the 
    normalized identity (e.g., the pp or gm bases). This formula for unitarity 
    is given in Prop 1 of ``Estimating the Coherence of noise'' by Wallman et al, 
    which also defines unitarity.
    
    Parameters
    ----------
    gate : array or gate
        The gate for which the unitarity is to be computed. 
                    
    mxBasis : {"std","gm","pp"} or a Basis object, optional
        The basis of the matrix.
        
    d : int, optional
        The dimension of the Hilbert space.

    Returns
    ----------
    u : float
        The unitarity of the gate.
        
    """
    d = int(round(_np.sqrt(A.shape[0])))
    basisMxs = _tls.basis_matrices(mxBasis, d)

    if _np.allclose( basisMxs[0], _np.identity(d,'d') ):
        B = A
    else:
        B = _tls.change_basis(A, mxBasis, "gm") #everything should be able to be put in the "gm" basis
    
    unital = B[1:d**2,1:d**2]
    u = _np.trace(_np.dot(_np.conj(_np.transpose(unital)),unital)) / (d**2-1)
    return u

def average_gateset_infidelity(gs_actual,gs_target,mxBasis=None,d=None):
    """
    Computes the average gateset infidelity (AGsI) between noisy gates and target gates.
    This quantity is defined in, e.g., arXiv:1702.01853 (see Eq 2 and below), and is 
    the mean of the average gate infidelities of the actual to the target gates.
    
    Parameters
    ----------
    gs_actual : GateSet
        Noisy gateset to calculate the AGsI of (to the target gateset).

    gs_target : GateSet
        Target gateset.
        
    mxBasis : {"std","gm","pp"} or Basis object, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
        
    d : int, optional
        The dimension of the Hilbert space.  If None, it is obtained
        from the gateset.

    Returns
    -------
    AGsI : float
        The AGsI of the actual gateset to the target gateset.    
    """
    if mxBasis is None: mxBasis = gs_actual.basis
    if d is None: d = int(round(_np.sqrt(gs_actual.dim)))
    
    AGI_list = []
    for gate in list(gs_target.gates.keys()):
        AGI_list.append(average_gate_infidelity(
            gs_actual.gates[gate],
            gs_target.gates[gate],d=d,
            mxBasis=mxBasis))
    AGsI = _np.mean(AGI_list)
    return AGsI

def errormaps(gs_actual, gs_target):
    """
    Computes the 'left-multiplied' error maps associated with a noisy gate 
    set, along with the average error map. This is the gate set [n_1,...] 
    such that g_i = n_it_i, where t_i is the gate which g_i is a noisy 
    implementation of. There is an additional gate in the set, that has 
    the key 'Gavg' and is the average of the error maps.
    
    Parameters
    ----------
    gs_actual : GateSet
        The imperfect gateset.
    
    gs_target : GateSet
        The target gateset.
    
    Returns
    -------
    errormaps : GateSet
        The left multplied error gates, along with the average error map,
        with the key 'Gavg'.  
        
    """    
    errormaps_gate_list = []
    errormaps = gs_actual.copy()
    for gate in list(gs_target.gates.keys()):
        errormaps.gates[gate] = _np.dot(gs_actual.gates[gate], 
                               _np.transpose(gs_target.gates[gate]))     
        errormaps_gate_list.append(errormaps.gates[gate])
        
    errormaps.gates['Gavg'] = _np.mean( _np.array([ i for i in errormaps_gate_list]), 
                                        axis=0, dtype=_np.float64)           
    return errormaps

def gatedependence_of_errormaps(gs_actual, gs_target, norm='diamond', 
                                mxBasis=None, d=2):
    """
    Computes the "gate-dependence of errors maps" parameter defined by
    delta_avg = avg_i|| E_i - avg_i(E_i) ||, where E_i are the error maps, and
    the norm is either the diamond norm or the 1-to-1 norm. This quantity
    is defined in Magesan et al PRA 85 042311 2012, and is used to calculate
    the systematic error of the zeroth or first order theories of Magesan 
    et al, and also the theory in arXiv:1702.01853.
    
    Parameters
    ----------
    gs_actual : GateSet
        The actual gateset
    
    gs_target : GateSet
        The target gateset.
        
    norm : str, optional
        The norm used in the calculation. Can be either 'diamond' for
        the diamond norm, or '1to1' for the Hermitian 1 to 1 norm.
        
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
        
    d : int, optional
        The dimension of the Hilbert space.
   
    Returns
    -------
    delta_avg : float
        The value of the parameter defined above.
        
    """
    error_gs = errormaps(gs_actual, gs_target)
    delta = []
    
    if mxBasis is None:
        mxBasis = gs_actual.get_basis_name()
    assert(mxBasis=='pp' or mxBasis=='gm' or mxBasis=='std'), "mxBasis must be 'gm', 'pp' or 'std'."
    
    for gate in list(gs_target.gates.keys()):
        if norm=='diamond':
            delta.append(_tls.diamonddist(error_gs.gates[gate],error_gs.gates['Gavg'],
                                          mxBasis=mxBasis))            
        elif norm=='1to1': 
            gate_dif = error_gs.gates[gate]-error_gs.gates['Gavg']
            delta.append(norm1to1(gate_dif,n_samples=1000, mxBasis=mxBasis,return_list=False))            
        else:
            raise ValueError("Only diamond or 1to1 norm available.")  
            
    delta_avg = _np.mean(delta)    
    return delta_avg

# ----- Predicting the RB parameters and curve from a gateset ---- #
def predicted_RB_number(gs,gs_target,weights=None,d=None):
    """
    Predicts the RB number (RB error rate) from a gateset, using the
    asymptotically exact formula from arXiv:1702.01853, which is valid
    for various types of RB, including standard Clifford RB. The gateset 
    should be trace preserving and completely positive in some representation,
    but the particular representation of the gateset used is irrelevant as
    the predicted RB error rate is a gauge-invariant quantity. The function
    is likely reliable when complete positivity is slightly violated, although
    the theory on which it is based assumes complete positivity.
    
    Parameters
    ----------
    gs : GateSet
        The gateset to calculate the RB number of (e.g., imperfect Cliffords).
        Note that this is not necessarily the set of physical primitives -- it is
        the  gateset randomly sampled over, e.g., the Cliffords, in the RB protocol.

    gs_target: GateSet
        The target gateset.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
    
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.

    Returns
    -------
    
    r_predicted : float.
        The predicted RB number. This is valid for various types of 
        RB, including standard Clifford RB.
        
    """
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    r_predicted = p_to_r(predicted_RB_decay_parameter(gs,gs_target,weights=weights),d=d)
    return r_predicted

def predicted_RB_decay_parameter(gs,gs_target,weights=None):
    """
    Computes the second largest eigenvalue of the 'L matrix' (see the L_matrix
    function), which corresponds to the RB decay rate for trace preserving and 
    completely positive gates and standard Clifford RB. This also acurately 
    predicts the RB decay parameter for a range of other variants of basic RB.
    
    Parameters
    ----------
    gs : Gateset
        The actual gateset. Note that this is not necessarily the set of physical 
        primitives -- it is the  gateset randomly sampled over, e.g., the Cliffords, 
        in the RB protocol.
  
    gs_target : Gateset
        The target gateset corresponding to gs.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.

    Returns
    -------
    
    p : float.
        The second largest eigenvalue of L. This is the RB decay parameter
        for various types of RB (see above).
    """
    L = L_matrix(gs,gs_target,weights=weights)
    E = _np.absolute(_np.linalg.eigvals(L))
    E = _np.flipud(_np.sort(E))
    if abs(E[0] - 1) > 10**(-12):
        _warnings.warn("Predicted RB decay parameter / error rate may be unreliable:\n" +
                       "Gateset is not (approximately) trace-preserving.")
    #
    # The warning commented out below is not very sensible. It will be replaced at some
    # point by a more subtle method for checking that the relevant RB theory holds.
    #
    #if abs(E[1]) - abs(E[2]) < 10**(-1):
    #    _warnings.warn("Predicted RB decay parameter / error rate may be unreliable:\n" +
    #                   "There is more than one significant exponential in RB decay.")
    if E[1].imag > 10**(-10):
        _warnings.warn("Predicted RB decay parameter / error rate may be unreliable:\n" +
                       "The decay constant has a significant imaginary component.")
    p = E[1]
    return p

def RB_gauge(gs,gs_target,weights=None,mxBasis=None,eigenvector_weighting=1.0):
    """
    Computes the gauge transformation required so that, when the gateset is transformed
    via this gauge-transformation, the RB number (as predicted by the function 
    `predicted_RB_number` is the average gateset infidelity between the (gauge-transformed)
    `gs` gateset and the target gateset `gs_target`. The transformation is defined
    in arXiv:1702.01853
    
    Parameters
    ----------
    gs : Gateset
        The gateset to calculate the RB number of (e.g., imperfect Cliffords).
        Note that this is not necessarily the set of physical primitives -- it is
        the  gateset randomly sampled over, e.g., the Cliffords, in the RB protocol.
  
    gs_target : Gateset
        The target gateset corresponding to gs.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
                
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
        
    eigenvector_weighting : float, optional
        Must be non-zero. A weighting on the eigenvector with eigenvalue that
        is the RB decay parameter, in the sum of this eigenvector and the
        eigenvector with eigenvalue of 1 that defines the returned matrix `l_operator`. 
        The value of this factor does not change whether this `l_operator` transforms into
        a gauge in which r = AGsI, but it may impact on other properties of the
        gates in that gauge (note that it is irrelevant if the gates are unital).
        
    Returns
    -------    
    l_operator: array
        The matrix defining the gauge-transformation.
        
    """                    
    gam, vecs = _np.linalg.eig(L_matrix(gs,gs_target,weights=weights))
    absgam = abs(gam)
    index_max = _np.argmax(absgam)
    gam_max = gam[index_max]
    
    if abs(gam_max - 1) > 10**(-12):
        _warnings.warn("Gateset is not (approximately) trace-preserving.\n" +
                       "RB theory may not apply")
        
    if gam_max.imag > 10**(-12):
        _warnings.warn("RB Decay constants have a significant imaginary component.\n" + 
                       "RB theory may not apply")
      
    absgam[index_max] = 0.0
    index_2ndmax = _np.argmax(absgam)
    decay_constant = gam[index_2ndmax]
    if decay_constant.imag > 10**(-12):
        _warnings.warn("Decay constants have a significant imaginary component.\n" +
                       "RB theory may not apply")
    #
    # The warning commented out below is not very sensible. It will be replaced at some
    # point by a more subtle method for checking that the relevant RB theory holds.
    #    
    #absgam[index_2ndmax] = 0.0
    #index_3rdmax = _np.argmax(absgam)
    #if abs(decay_constant) - abs(absgam[index_3rdmax]) < 10**(-1):
    #    _warnings.warn("There is more than one significant exponential in RB decay.\n" + 
    #                   "RB theory may not apply")

    vec_l_operator = vecs[:,index_max] + eigenvector_weighting*vecs[:,index_2ndmax]
    
    if mxBasis is None:
        mxBasis = gs.get_basis_name()
    assert(mxBasis=='pp' or mxBasis=='gm' or mxBasis=='std'), "mxBasis must be 'gm', 'pp' or 'std'."
    
    if mxBasis is 'pp' or 'gm':
        assert(_np.amax(vec_l_operator.imag) < 10**(-15)), "If 'gm' or 'pp' basis, RB gauge matrix should be real."
        vec_l_operator = vec_l_operator.real
        
    vec_l_operator[abs(vec_l_operator) < 10**(-15)] = 0.
    l_operator = unvec(vec_l_operator) 
    
    return l_operator

def transform_to_RB_gauge(gs,gs_target,weights=None,mxBasis=None,eigenvector_weighting=1.0):
    """
    Transforms a GateSet into the "RB gauge" (see the `RB_gauge` function), as 
    introduced in  arXiv:1702.01853. This gauge is a function of both the gateset 
    and its target (both of which may be input in any gauge, for the purposes of 
    obtaining "r = average gateset infidelity" between the output GateSet and gs_target).
    
    Parameters
    ----------
    gs : Gateset
        The gateset to calculate the RB number of (e.g., imperfect Cliffords).
        Note that this is not necessarily the set of physical primitives -- it is
        the  gateset randomly sampled over, e.g., the Cliffords, in the RB protocol.
  
    gs_target : Gateset
        The target gateset corresponding to gs.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
                                
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
        
    eigenvector_weighting : float, optional
        Must be non-zero. A weighting on the eigenvector with eigenvalue that
        is the RB decay parameter, in the sum of the this eigenvector and the
        eigenvector with eigenvalue of 1 that defines the l_operator. The value
        of this factor should not change whether this l_operator transforms into
        a gauge in which r = AGsI, but it *might* impact on other properties of the
        gates in that gauge (certainly in some cases its value is entirely irrelevant).

    Returns
    -------   
    gs_in_RB_gauge: GateSet
        The gateset `gs` transformed into the "RB gauge".
        
    """            
    l = RB_gauge(gs,gs_target,weights=weights,mxBasis=mxBasis,
                 eigenvector_weighting=eigenvector_weighting)
    gs_in_RB_gauge = gs.copy()
    S = _objs.FullGaugeGroupElement( _np.linalg.inv(l) )
    gs_in_RB_gauge.transform( S )         
    return gs_in_RB_gauge

def L_matrix(gs,gs_target,weights=None):
    """
    Constructs the 'L' linear operator on superoperators, from arXiv:1702.01853,
    represented as a matrix via the "stack" operation. This eigenvalues of
    this matrix describe the decay constant (or constants) in an RB decay
    curve for an RB protocol whereby random elements of the provided gateset
    are sampled according to the `weights` probability distribution over the
    gateset.
    
    Parameters
    ----------
    gs : Gateset
        The gateset to calculate the RB number of (e.g., imperfect Cliffords).
        Note that this is not necessarily the set of physical primitives -- it is
        the  gateset randomly sampled over, e.g., the Cliffords, in the RB protocol.
  
    gs_target : Gateset
        The target gateset corresponding to gs.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
        
    Returns
    -------
    L : float
        A weighted version of the L operator from arXiv:1702.01853, represented 
        as a matrix using the 'stacking' convention.  
        
    """
    if weights is None:
        weights = {}
        for key in gs_target.gates.keys():
            weights[key] = 1.
            
    normalizer = _np.sum(_np.array([weights[key] for key in gs_target.gates.keys()]))
            
    dim = len(gs_target.gates.keys())
    L_matrix = (1 / normalizer) * _np.sum(weights[key]*_np.kron(gs.gates[key].T,
                 _np.linalg.inv(gs_target.gates[key])) for key in gs_target.gates.keys())
    return L_matrix

def R_matrix_predicted_RB_decay_parameter(gs,group,subset_sampling=None,group_to_gateset=None,
                                          weights=None,d=None):
    """
    Constructs the second largest eigenvalue of the 'R' matrix (see below), which
    predicts the RB decay parameter for trace-preserving gates.
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset (e.g., the Cliffords). 
        If subset_sampling is None or group_to_gateset is None, the labels
        of the gates in gs should be the same as the labels of the group
        elements in group. For Clifford RB this would be the clifford gateset
        (perhaps obtained from a primitive gateset and a compilation table).
            
    group : MatrixGroup
        The group that the 'gs' gateset contains gates from (gs does not
        need to be the full group, and could be a subset of the group). For
        Clifford RB, this would be the Clifford group.
        
    subset_sampling : list, optional
        If not None, a list of gate labels from 'gs', for which random sequences of 
        this subset of gates are implemented in the RB protocol. Even if this is 
        all of the gates of gs, this list needs to be specified if gs and group are 
        either (1) not labelled the same (and so group_to_gateset is not None), 
        or (2) gs is a subset of group.
        
    group_to_gateset : dict, optional
        If not None, a dictionary that maps labels of group elements to labels
        of gs. Only used if subset_sampling is not None. If subset_sampling is 
        not None and the gs and group elements have the same labels, this dictionary
        is not required. Otherwise it is necessary.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
      
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.
    
    Returns
    -------
    p : float
        The predicted RB decay parameter. Valid for standard 2-design (e.g., Clifford)
        RB with trace-preserving gates, and in a range of other circumstances.
        
    """ 
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    R = R_matrix(gs,group,subset_sampling=subset_sampling,
                 group_to_gateset=group_to_gateset,weights=weights,d=d)
    E = _np.absolute(_np.linalg.eigvals(R))
    E = _np.flipud(_np.sort(E))
    p = E[1]
    return p

def R_matrix(gs,group,subset_sampling=None,group_to_gateset=None,
             weights=None,d=None):
    """
    Constructs the 'R' matrix of arXiv:1702.01853
    This matrix described the exact behaviour of the average surival
    probablities of RB sequences. It is exponentially large in the number 
    of qubits.
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset (e.g., the Cliffords) to calculate the R matrix of. 
        If subset_sampling is None or group_to_gateset is None, the labels
        of the gates in gs should be the same as the labels of the group
        elements in group. For Clifford RB this would be the clifford gateset
        (perhaps obtained from a primitive gateset and a compilation table).
            
    group : MatrixGroup
        The group that the 'gs' gateset contains gates from (gs does not
        need to be the full group, and could be a subset of the group). For
        Clifford RB, this would be the Clifford group.
        
    subset_sampling : list, optional
        If not None, a list of gate labels from 'gs', for which the R matrix
        corresponding to random sequences of this subset of gates is to be
        contructed. Even if this is all of the gates of gs, this list needs to
        be specified if gs and group are either (1) not labelled the same (and so 
        group_to_gateset is not None), or (2) gs is a subset of group.
        
    group_to_gateset : dict, optional
        If not None, a dictionary that maps labels of group elements to labels
        of gs. Only used if subset_sampling is not None. If subset_sampling is 
        not None and the gs and group elements have the same labels, this dictionary
        is not required. Otherwise it is necessary.
      
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs if
        subset_sampling is None, the keys are the gates in subset_sampling if
        subset_sampling is not None, and the values are the unnormalized probabilities 
        to apply each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be positive or zero, and they must not all be zero (because, 
        when divided by their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
      
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.
    
    Returns
    -------
    R : float
        A weighted version of the R matrix from arXiv:1702.01853. 
        
    """ 
    #
    # This function is currently more complicated than necessary (subset_sampling could
    # be removed, as this can be achieved via the weights dict). However, as the
    # function is currently working this has been left for now.
    #
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    group_dim = len(group)
    R_dim = group_dim * d**2
    R = _np.zeros([R_dim,R_dim],float)
    
    if subset_sampling is None:
        if weights is None:
            weights = {}
            for key in gs.gates.keys():
                weights[key] = 1.
        normalizer = _np.sum(_np.array([weights[key] for key in gs.gates.keys()]))
        for i in range(0,group_dim):
            for j in range(0,group_dim):
                label_itoj = group.product([group.get_inv(i),j])
                for k in range (0,d**2):
                    for l in range(0,d**2):
                        R[j*d**2+k,i*d**2+l] = weights[group.labels[label_itoj]]*gs.gates[group.labels[label_itoj]][k,l]
        R = R/normalizer
                
    if subset_sampling is not None:
        if group_to_gateset is None:
            for element in subset_sampling:
                assert(element in gs.gates.keys() and element in group.labels),  "The subset\
                   of gates should be elements of the gateset and group if group_to_gateset\
                   not specificed."
            if weights is None:
                weights = {}
                for key in subset_sampling:
                    weights[key] = 1.
            normalizer = _np.sum(_np.array([weights[key] for key in subset_sampling]))
                
            for i in range(0,group_dim):
                for j in range(0,group_dim):
                    label_itoj = group.product([group.get_inv(i),j])
                    if group.labels[label_itoj] in subset_sampling:
                        for k in range (0,d**2):
                            for l in range(0,d**2):
                                R[j*d**2+k,i*d**2+l] = weights[group.labels[label_itoj]]*gs.gates[group.labels[label_itoj]][k,l]
            R = R/normalizer
        
        if group_to_gateset is not None:
            for key in group_to_gateset.keys():
                assert(key in group.labels), "group_to_gateset dictionary invalid!"
                assert(group_to_gateset[key] in gs.gates.keys()), "group_to_gateset \
                dictionary invalid!"
                
            if weights is None:
                weights = {}
                for key in subset_sampling:
                    weights[key] = 1.
            normalizer = _np.sum(_np.array([weights[key] for key in subset_sampling]))
                    
            for i in range(0,group_dim):
                for j in range(0,group_dim):
                    label_itoj = group.product([group.get_inv(i),j])
                    if group.labels[label_itoj] in group_to_gateset.keys():
                        for k in range (0,d**2):
                            for l in range(0,d**2):
                                R[j*d**2+k,i*d**2+l] = weights[group_to_gateset[
                                        group.labels[label_itoj]]]*gs.gates[group_to_gateset[
                                        group.labels[label_itoj]]][k,l]
            R = R/normalizer
            
    return R

def exact_RB_ASPs(gs,group,m_max,m_min=1,m_step=1,success_outcomelabel=('0',),
                  subset_sampling=None,group_to_gateset=None,weights=None,
                  d=None, compilation=None, twirled=False):
    """
    Calculates the exact RB average surival probablilites (ASP), using some
    simple generalizations of the formula given in Eq (2) and the surrounding 
    text of arXiv:1702.01853. This formula does not scale well with group size
    and qubit number, and for the Clifford group it is likely only practical for 
    a single qubit.
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset (e.g., the Cliffords) to calculate the RB survival
        probabilities for. If subset_sampling is None or group_to_gateset 
        is None, the labels of the gates in gs should be the same as the 
        labels of the group elements in group. For Clifford RB this should 
        be the clifford gateset (perhaps obtained from a primitive gateset 
        and a compilation table).
            
    group : MatrixGroup
        The group that the 'gs' gateset contains gates from (gs does not
        need to be the full group, and could be a subset of the group). For
        Clifford RB, this would be the Clifford group.
        
    m_max : int
        maximal sequence length of the random gates (not including the
        inversion gate).
        
    m_min : int, optional
        minimal sequence length. Defaults to the smallest valid value of 1.
        
    m_step : int, optional
        step size between sequence lengths
       
    success_outcomelabel : str or tuple, optional
        Specifies the outcome label associated with surival
              
    subset_sampling : list, optional
        If not None, a list of gate labels from 'gs'. These are the gates
        that random applied in the rb sequences. Even if this is all of the 
        gates of gs, this list needs to be specified if gs and group are either 
        (1) not labelled the same (and so group_to_gateset is not None), or 
        (2) gs is a subset of group.
        
    group_to_gateset : dict, optional
        If not None, a dictionary that maps labels of group elements to labels
        of gs. Only used if subset_sampling is not None. If subset_sampling is 
        not None and the gs and group elements have the same labels, this dictionary
        is not required. Otherwise it is necessary.
                      
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs if
        subset_sampling is None, the keys are the gates in subset_sampling if
        subset_sampling is not None, and the values are the unnormalized probabilities 
        to apply each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be positive or zero, and they must not all be zero (because, 
        when divided by their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
      
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.
        
    compilation : dict, optional
        If subset_sampling is not None and fixed_length_each_m is False this
        specifies the compilation of group elements into the gates from 
        subset_sampling in order to apply the final inverse gate.
    
    twirled : bool, optional
        If True, the state preparation is followed by a single uniformly random group
        element. If subset_sampling and weights are None, there is no reason to set 
        this to True, but otherwise it can significantly change the behaviour of the 
        RB decay.
     
    Returns
    -------
    m : float
        Array of sequence length values that the ASP has been calculated for
        
    P_m : float
        Array containing ASP values for the specified sequence length values.
        
    """
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    i_max = _np.floor((m_max - m_min ) / m_step).astype('int')
    m = _np.zeros(1+i_max,int)
    P_m = _np.zeros(1+i_max,float)
    group_dim = len(group)
    R = R_matrix(gs,group,subset_sampling=subset_sampling,
                 group_to_gateset=group_to_gateset,weights=weights,d=d)
    success_prepLabel = list(gs.preps.keys())[0] #just take first prep
    success_effectLabel = success_outcomelabel[-1] if isinstance(success_outcomelabel,tuple) else success_outcomelabel
    extended_E = _np.kron(column_basis_vector(0,group_dim).T,gs.povms['Mdefault'][success_effectLabel].T)
    extended_rho = _np.kron(column_basis_vector(0,group_dim),gs.preps[success_prepLabel])
    
    if subset_sampling is None:
        extended_E = group_dim*_np.dot(extended_E, R)
        if twirled is True:  
            extended_rho = _np.dot(R,extended_rho)
    else:
        full_gateset = _cnst.build_alias_gateset(gs,compilation)
        R_fullgroup = R_matrix(full_gateset,group,d=d)
        extended_E = group_dim*_np.dot(extended_E, R_fullgroup)
        if twirled is True:        
            extended_rho = _np.dot(R_fullgroup,extended_rho)
            
    
    Rstep = _np.linalg.matrix_power(R,m_step)
    Riterate =  _np.linalg.matrix_power(R,m_min)
    for i in range (0,1+i_max):
        m[i] = m_min + i*m_step
        P_m[i] = _np.dot(extended_E,_np.dot(Riterate,extended_rho))
        Riterate = _np.dot(Rstep,Riterate)
    return m, P_m

def L_matrix_ASPs(gs,gs_target,m_max,m_min=1,m_step=1,success_outcomelabel='0',
                  compilation=None,twirled=False,weights=None,d=None,
                  gauge_optimize=True,error_bounds=False,norm='diamond'):
    """
    Computes RB average survival probablities, as predicted by the 'L' operator
    theory of arXiv:1702.01853. Within the function, the gs is gauge-optimized to
    gs_target. This is *not* optimized to the gauge specified in arXiv:1702.01853,
    but instead performs the standard pyGSTi gauge-optimization (using the frobenius
    distance). In most cases, this is likely to be a reasonable proxy for the gauge
    optimization perscribed by arXiv:1702.01853.
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset.
           
    gs_target : Gateset
        Target gateset
        
    m_max : int
        maximal sequence length of the random gates (so not including the
        inversion gate).
        
    m_min : int, optional
        minimal sequence length. Defaults to the smallest valid value of 1.
        
    m_step : int, optional
        step size between sequence lengths
        
    success_outcomelabel : str or tuple, optional
        Specifies the outcomeM label associated with surival
               
   compilation : dict, optional
       A compilation table that has as keys the elements of a larger gateset, e.g., 
       the Cliffords, and values as lists of gates from `gs`. If this is specified then
       it is assumed that this gateset is the set from which the final inversion element
       is applied, and if `twirled` is False, this is only used to calculate the final
       error map associated with this inversion. A compilation table is necessary if 
       `twirled` is True (see below).
       
   twirled : bool, optional
        If True, the state preparation is followed by a single uniformly random group
        element. How to implement this group element in terms of the elements in `gs`
        is specified by the `compilation` dictionary, which is required if `twirled`
        is True. If `gs` is the full group, there is no purpose in this twirl, and
        hence this twirl option is not supported without a compilation table.
          
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in gs if
        subset_sampling is None, the keys are the gates in subset_sampling if
        subset_sampling is not None, and the values are the unnormalized probabilities 
        to apply each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be positive or zero, and they must not all be zero (because, 
        when divided by their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
        
    gauge_optimize : bool, optional
        Default is True, and if True a gauge-optimization to the target gateset is 
        implemented before calculating all quantities. If False, no gauge optimization
        is performed. Whether or not a gauge optimization is performed does not affect
        the rate of decay but it will generally affect the exact form of the decay. E.g.,
        if a perfect gateset is given to the function -- but in the "wrong" gauge -- no
        decay will be observed in the output P_m, but the P_m can be far from 1 (even
        for perfect SPAM) for all m. The gauge optimization is optional, as it is
        not guaranteed to always improve the accuracy of the reported P_m, although when gauge
        optimization is performed this limits the possible deviations of the reported
        P_m from the true P_m.
      
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.
        
    error_bounds : bool, optional
        Sets whether or not to return error bounds for how far the true ASPs can deviate
        from the values returned by this function.
        
    norm : str, optional
        The norm used in the error bound calculation. Default is consistent with
        arXiv:1702.01853.
    
    Returns
    -------
    m : float
        Array of sequence length values that the ASP has been calculated for
        
    P_m : float
        Array containing predicted ASP values for the specified sequence length values.
    
    if error_bounds is True, also returns:
        lower_bound: float
            Array containing lower bounds on the possible ASP values

        upper_bound: float
            Array containing upper bounds on the possible ASP values
    """
    assert((compilation is None and twirled is True) is False), "twirled cannot be true is compilation is None"   
    if d is None: d = int(round(_np.sqrt(gs.dim)))
        
    if gauge_optimize:
        gs_go = _algs.gaugeopt_to_target(gs,gs_target)
    else:
        gs_go = gs.copy()
    L = L_matrix(gs_go,gs_target,weights=weights)
    success_prepLabel = list(gs.preps.keys())[0] #just take first prep
    success_effectLabel = success_outcomelabel[-1] if isinstance(success_outcomelabel,tuple) else success_outcomelabel
    identity_vec = vec(_np.identity(d**2,float))
    
    if compilation is not None:
        gs_group = _cnst.build_alias_gateset(gs_go,compilation)
        gs_target_group = _cnst.build_alias_gateset(gs_target,compilation)
        delta = gatedependence_of_errormaps(gs_group,gs_target_group,norm=norm,d=d)
        emaps = errormaps(gs_group,gs_target_group)
        E_eff = _np.dot(gs_go.povms['Mdefault'][success_effectLabel].T,emaps.gates['Gavg'])
        
        if twirled is True:
            L_group = L_matrix(gs_group,gs_target_group)
        
    if compilation is None:
        delta = gatedependence_of_errormaps(gs_go,gs_target,norm=norm,d=d)
        emaps = errormaps(gs_go,gs_target)
        E_eff = _np.dot(gs_go.povms['Mdefault'][success_effectLabel].T,emaps.gates['Gavg'])
    
    i_max = _np.floor((m_max - m_min ) / m_step).astype('int')
    m = _np.zeros(1+i_max,int)
    P_m = _np.zeros(1+i_max,float)
    upper_bound = _np.zeros(1+i_max,float)
    lower_bound = _np.zeros(1+i_max,float)
    
    Lstep = _np.linalg.matrix_power(L,m_step)
    Literate =  _np.linalg.matrix_power(L,m_min)
    for i in range (0,1+i_max):
        m[i] = m_min + i*m_step
        if twirled:
            L_m_rdd = unvec(_np.dot(L_group,_np.dot(Literate,identity_vec)))
        else:
            L_m_rdd = unvec(_np.dot(Literate,identity_vec))
        P_m[i] = _np.dot(E_eff,_np.dot(L_m_rdd,gs_go.preps[success_prepLabel]))
        Literate = _np.dot(Lstep,Literate)
        upper_bound[i] = P_m[i] + delta/2
        lower_bound[i] = P_m[i] - delta/2
        if upper_bound[i] > 1:
            upper_bound[i]=1
        if lower_bound[i] < 0:
            lower_bound[i]=0
    if error_bounds:    
        return m, P_m, upper_bound, lower_bound
    else:
        return m, P_m

def Magesan_theory_parameters(gs_actual, gs_target, success_outcomelabel=('0',), 
                              norm='1to1', d=2):                   
    """
    From a given actual and target gateset, computes the parameters
    of the 'zeroth order' and 'first order' RB theories of Magesan et al PRA 85
    042311 2012.
    
    Parameters
    ----------
    gs_actual : GateSet
        The gateset to compute the parameters for
    
    gs_target : GateSet
       Target gateset.
        
    success_outcomelabel : str or tuple, optional
        The outcome label associated with survival.    
        
    norm : str, optional
        The norm used in the calculation of the error bounds. Defaults to the
        Hermitian 1-to-1 norm as used in arxiv:1109.6887. Other option is
        'diamond' which uses the diamond norm to compute the bounds.
    
    d : int, optional
        The dimension.
    
    Returns
    -------
    Magesan_theory_params : dictionary of floats
    
        r : Predicted RB number.
        p : Predicted RB decay rate.
        
        A,B : Predicted SPAM constants in 'zeroth order theory'.
        Conversion to quanities in PRA 85 042311 2012 is
        B = A_0, A = B_0.
        
        A1, B1, C1 : Predicted SPAM constants in 'first order theory'.
        Conversion to quanities in PRA 85 042311 2012 is
        A1 = B_1, B1 = A_1 - C_1(q/p^(-2) - 1), C1 = C_1(q/p^(-2) - 1)
        where the parameter name on the LHS of each equality is that used
        herein, and the parameter name on the RHS of each equality is that
        used in PRA 85 042311 2012.
        
        delta : measure of gate-depedence of the noise, as defined in 
        PRA 85 042311 2012 (taking the case of time-independent noise therein).    
    """
    Magesan_theory_params = {}
    Magesan_theory_params['r'] = average_gateset_infidelity(gs_actual,gs_target,None,d)    
    Magesan_theory_params['p'] = r_to_p(Magesan_theory_params['r'],d)
    Magesan_theory_params['delta'] = gatedependence_of_errormaps(gs_actual, 
                                                                 gs_target, norm,d)
    error_gs = errormaps(gs_actual, gs_target)   
       
    R_list = []
    Q_list = []
    for gate in list(gs_target.gates.keys()):
        R_list.append(_np.dot(_np.dot(error_gs.gates[gate],gs_target.gates[gate]),
              _np.dot(error_gs.gates['Gavg'],_np.transpose(gs_target.gates[gate]))))
        Q_list.append(_np.dot(gs_target.gates[gate],
              _np.dot(error_gs.gates[gate],_np.transpose(gs_target.gates[gate]))))
    
    error_gs.gates['GR'] = _np.mean(_np.array([ i for i in R_list]), 
                                      axis=0, dtype=_np.float64)
    error_gs.gates['GQ'] = _np.mean(_np.array([ i for i in Q_list]), 
                                      axis=0, dtype=_np.float64)    
    error_gs.gates['GQ2'] = _np.dot(error_gs.gates['GQ'],error_gs.gates['Gavg'])
    
    error_gs.preps['rhoc_mixed'] = 1./d*error_gs['identity']

    #HERE
    #Assumes standard POVM labels
    povm = _objs.POVM( [('plus_cm', gs_target.povms['Mdefault']['0']),
                        ('minus_cm', gs_target.povms['Mdefault']['1'])] )
    gsl = [('Gavg',),('GR',),('Gavg','GQ',)]   
    ave_error_gsl = _cnst.create_gatestring_list("a", a=gsl)
    N=1
    data = _cnst.generate_fake_data(error_gs, ave_error_gsl, N, 
                                    sampleError="none", seed=1)

    if isinstance(success_outcomelabel, tuple):
        success_outcomelabel_cm = (success_outcomelabel[0] +'_cm',)
    else:
        success_outcomelabel_cm = success_outcomelabel +'_cm'
    
    pr_L_p = data[('Gavg',)][success_outcomelabel]
    pr_L_I = data[('Gavg',)][success_outcomelabel_cm]
    pr_R_p = data[('GR',)][success_outcomelabel]
    pr_R_I = data[('GR',)][success_outcomelabel_cm]
    pr_Q_p = data[('Gavg','GQ',)][success_outcomelabel]
    p = Magesan_theory_params['p']    
    B_1 = pr_R_I
    A_1 = (pr_Q_p/p) - pr_L_p + ((p -1)*pr_L_I/p) \
                            + ((pr_R_p - pr_R_I)/p)
    C_1 = pr_L_p - pr_L_I
    q = average_gate_infidelity(error_gs.gates['GQ2'],_np.identity(d**2,float),d)
    q = r_to_p(q,d)
    
    Magesan_theory_params['A'] = pr_L_I
    Magesan_theory_params['B'] = pr_L_p - pr_L_I       
    Magesan_theory_params['A1'] = B_1
    Magesan_theory_params['B1'] = A_1 - C_1*(q - 1)/p**2
    Magesan_theory_params['C1'] = C_1*(q- p**2)/p**2

    return Magesan_theory_params

# needs more testing              
def Magesan_error_bound(m,delta,order='zeroth'):
    """
    Given a 'delta' and a sequence length 'm', computes the value of the systematic
    error bound from Magesan et al PRA 85 042311 2012 between the predicted RB average 
    surival probabilities of the 'first' or 'zeroth' order models therein, and the actual 
    ASPs.
    
    Parameters
    ----------
    m : float
        Sequence length, and so it is often an int
    
    delta : float
        The size of the 'delta' parameter for the gate set in question
      
    order : str, optional
        May be 'zeroth or 'first'. The order fitting model for which the
        error bound should be calcluated.
    
    Returns
    -------
    sys_eb: float
        The systematic error bound at sequence length m.
    
    """
    sys_eb = (delta + 1)**(m+1) - 1
    
    if order=='first':
        sys_eb = sys_eb - (m+1)*delta

    return sys_eb

def seb_upper(y,m,delta,order='zeroth'):
    """
    Finds an upper bound on the RB average surival probability (ASP) from the 
    predicted RB ASP of the 'first' or 'zeroth' order models from 
    Magesan et al PRA 85 042311 2012, using the bounds given therein.
    
    Parameters
    ----------
    y : ASP predicted by the theory of Magesan et al PRA 85 042311 2012
    
    m : float
        Sequence length, and so it is often an int
    
    delta : float
        The size of the 'delta' parameter for the gate set in question
      
    order : str, optional
        May be 'zeroth or 'first'. The order fitting model for which the
        upper bound should be calcluated.
    
    Returns
    -------
    upper: float
        The upper bound on the ASP at sequence length m.
    """
    sys_eb = Magesan_error_bound(m,delta,order)
    
    upper = y + sys_eb
    upper[upper > 1]=1
     
    return upper

def seb_lower(y,m,delta,order='zeroth'):
    """
    Finds a lower bound on the surival probability from the analytic value. See
    'seb_upper' above for further details.
    """
    sys_eb = Magesan_error_bound(m,delta,order)
    
    lower = y - sys_eb
    lower[lower < 0]=0
       
    return lower

# ----- Matrix tools functions ---- #
def column_basis_vector(i,dim):
    """
    Returns the ith standard basis vector in dimension dim.
    """
    output = _np.zeros([dim,1],float)
    output[i] = 1.
    return output
 
def vec(matrix_in):
    """
    Stacks the columns of a matrix to return a vector
    """
    return [b for a in _np.transpose(matrix_in) for b in a]

def unvec(vector_in):
    """
    Slices a vector into the columns of a matrix.
    """
    dim = int(_np.sqrt(len(vector_in)))
    return _np.transpose(_np.array(list(
                zip(*[_ittls.chain(vector_in,
                            _ittls.repeat(None, dim-1))]*dim))))

def norm1(matr):
    """
    Returns the 1 norm of a matrix
    """
    return float(_np.real(_np.trace(sqrtm(_np.dot(matr.conj().T,matr)))))

def random_hermitian(dimension):
    """
    Generates a random Hermitian matrix
    """
    my_norm = 0.
    while my_norm < 0.5:
        dimension = int(dimension)
        a = _np.random.random(size=[dimension,dimension])
        b = _np.random.random(size=[dimension,dimension])
        c = a+1.j*b + (a+1.j*b).conj().T
        my_norm = norm1(c)
    return c / my_norm

def norm1to1(operator, n_samples=10000, mxBasis="gm",return_list=False):
    """
    Returns the Hermitian 1-to-1 norm of a superoperator represented in
    the standard basis, calculated via Monte-Carlo sampling. Definition
    of Hermitian 1-to-1 norm can be found in arxiv:1109.6887.
    """
    if mxBasis=='gm':
        std_operator = _tls.change_basis(operator, 'gm', 'std')
    elif mxBasis=='pp':
        std_operator = _tls.change_basis(operator, 'pp', 'std')
    elif mxBasis=='std':
        std_operator = operator
    else:
        raise ValueError("mxBasis should be 'gm', 'pp' or 'std'!")
    
    rand_dim = int(_np.sqrt(float(len(std_operator))))
    vals = [ norm1(unvec(_np.dot(std_operator,vec(random_hermitian(rand_dim)))))
             for n in range(n_samples)]
    if return_list:
        return vals
    else:
        return max(vals)
    
# ----- Wallman and Flammia error bars tools ---- #
# Not tested, as W&F error bars are not currently supported
def _H_WF(epsilon,nu):
    """
    Implements Eq. 9 from Wallman and Flammia 
    (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    return (1./(1-epsilon))**((1.-epsilon)/(nu+1.)) * \
        (float(nu)/(nu+epsilon))**((float(nu)+epsilon)/(nu+1.))


def _sigma_m_squared_base_WF(m,r):
    """
    Implements Eq. 6 (ignoring higher order terms) from Wallman and Flammia
    (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    return m**2 * r**2 + 7./4 * m * r**2


def _K_WF(epsilon,delta,m,r,sigma_m_squared_func=_sigma_m_squared_base_WF):
    """
    Implements Eq. 10 (rounding up) from Wallman and Flammia
    (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    sigma_m_squared = sigma_m_squared_func(m,r)
    return int(_np.ceil(-_np.log(2./delta) / 
                         _np.log(_H_WF(epsilon,sigma_m_squared))))

def create_K_m_sched(m_min,m_max,Delta_m,epsilon,delta,r_0,
                     sigma_m_squared_func=_sigma_m_squared_base_WF):
    """
    Computes a "K_m" schedule, that is, how many sequences of Clifford length m
    should be sampled over a range of m values, given certain precision
    specifications.
    
    For further discussion of the epsilon, delta, r, and sigma_m_squared_func
    parameters, see 
    http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032, referred
    to in this documentation as "W&F".
    
    Parameters
    ----------
    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.
        
    epsilon : float
        Specifies desired confidence interval half-width for each average
        survival probability estimate \hat{F}_m (See W&F Eq. 8).  
        E.g., epsilon = 0.01 means that the confidence interval width for 
        \hat{F}_m is 0.02.  The smaller epsilon is, the larger each value of
        K_m will be.

    delta : float
        Specifies desired confidence level for confidence interval specified
        by epsilon. delta = 1-0.6827 corresponds to a confidence level of 1
        sigma.  This value should be used if W&F-derived error bars are
        desired. (See W&F Eq. 8).  The smaller delta is, the larger each value
        of K_m will be.

    r_0 : float
        Estimate of upper bound of the RB number for the system in question.  
        The smaller r is, the smaller each value of K_m will be.  However, if
        the system's actual RB number is larger than r_0, then the W&F-derived
        error bars cannot be assumed to be valid.  Additionally, it is assumed
        that m*r_0 << 1.
    
    sigma_m_squared_func : function, optional
        Function used to serve as the rough upper bound on the variance of 
        \hat{F}_m.  Default is _sigma_m_squared_base_WF, which implements 
        Eq. 6 of W&F (ignoring higher order terms).

    Returns
    ----------
    K_m_sched : OrderedDict
        An ordered dictionary whose keys are Clifford sequence lengths m and 
        whose values are number of Clifford sequences of length m to sample 
        (determined by _K_WF(m,epsilon,delta,r_0)).
    """
    K_m_sched = _OrderedDict()
    for m in range(m_min,m_max+1,Delta_m):
        K_m_sched[m] = _K_WF(epsilon,delta,m,r_0,
                             sigma_m_squared_func=sigma_m_squared_func)
    return K_m_sched

# ----- dataset tools ---- #
def dataset_to_summary_dict(dataset,seqs,success_outcome_label,use_frequencies=False):
    """
    Maps an RB dataset to an ordered dictionary; keys are 
    sequence lengths, values are number of success counts or frequencies, where
    value of the i^th element is the total number of successes seen for the 
    i^th sequence of length m.  (Auxiliary keys map (m,'N') and (m,'K') to  
    number of repetitions per sequence and number of sequences, respectively,
    for sequences of length m.)
    """
    output = _OrderedDict({})
    if not use_frequencies:
#        N = None
        for seq in seqs:
            m = len(seq)
            N_temp = int(_np.round(dataset[seq].total()))
            try:
                output[m,'N']
            except:
                output[m,'N'] = N_temp
            if output[m,'N'] != N_temp:
                raise ValueError("Different N values used at same m!")
            try:
                output[m,'K'] += 1
            except:
                output[m,'K'] = 1
#            if N is None:
#                N = N_temp
#            elif N_temp != N:
#                raise ValueError("Different N values discovered!")
            n = dataset[seq][success_outcome_label]
            try:
                output[m].append(n)
            except:
                output[m] = [n]
        return output
    else:
#        N = None
        for seq in seqs:
            m = len(seq)
#            N_temp = int(_np.round(dataset[seq].total()))
#            if N is None:
#                N = N_temp
#            elif N_temp != N:
#                raise ValueError("Different N values discovered!")
            try:
                output[m,'K'] += 1
            except:
                output[m,'K'] = 1
            N = dataset[seq].total()
            n = dataset[seq][success_outcome_label]
            frac = float(n) / float(N)
            try:
                output[m].append(frac)
            except:
                output[m] = [frac]
        return output

def summary_dict_to_f1_hat_dict(summary_dict,use_frequencies=False):
    f1_hat_dict = _OrderedDict({})
    if not use_frequencies:
        for m in summary_dict.keys():
            if isinstance(m,int):
                K = summary_dict[m,'K']
                N = summary_dict[m,'N']
                f1_hat_dict[m] = 1. / (N*K) * _np.sum(summary_dict[m])
        return f1_hat_dict
    else:
        for m in summary_dict.keys():
            if isinstance(m,int):
                K = summary_dict[m,'K']
                f1_hat_dict[m] = 1. / (K) * _np.sum(summary_dict[m])
        return f1_hat_dict

def summary_dict_to_f_empirical_squared_dict(summary_dict,use_frequencies = False):
    """
    Maps summary dict (defined in rbutils.dataset_to_summary_dict) to 
    f_empirical_squared_dict.
    """
    f_empirical_squared_dict = _OrderedDict({})
    if not use_frequencies:
        for m in summary_dict.keys():
            if isinstance(m,int):
                K = summary_dict[m,'K']#len(summary_dict[m])
                bias_correction = K / (K + 1.)
                N = summary_dict[m,'N']
                f_empirical_squared_dict[m,'N'] = N
                f_empirical_squared_dict[m] = bias_correction * 1. / (2 * K**2 * N**2) * _np.sum([(nl - nm)**2 for nl in summary_dict[m] for nm in summary_dict[m]])
        return f_empirical_squared_dict
    else:
        for m in summary_dict.keys():
            if isinstance(m,int):
                K = summary_dict[m,'K']#len(summary_dict[m])
                bias_correction = K / (K + 1.)
                f_empirical_squared_dict[m] = bias_correction * 1. / (2 * K**2) * _np.sum([(fl - fm)**2 for fl in summary_dict[m] for fm in summary_dict[m]])
        return f_empirical_squared_dict

def summary_dict_to_delta_f1_squared_dict(summary_dict,infinite_data=True):
    """
    Maps summary dict (defined in rbutils.dataset_to_summary_dict) to 
    delta_f1_squared_dict.
    """
#    if infinite_data and not use_frequencies:
#        raise ValueError('If infinite_data is True, then use_frequencies must be True too!')
    delta_f1_squared_dict = _OrderedDict({})
    if infinite_data:
        use_frequencies = True
    else:
        use_frequencies = False
    delta_f_empirical_squared_dict = summary_dict_to_f_empirical_squared_dict(summary_dict,use_frequencies)
    f1_hat_dict = summary_dict_to_f1_hat_dict(summary_dict,use_frequencies)
    for m in summary_dict.keys():
        if isinstance(m,int):
            K = summary_dict[m,'K']
            if infinite_data:
                term_1 = 1. / (2*K)
            else:
                N = summary_dict[m,'N']
                term_1 = f1_hat_dict[m] * (1 - f1_hat_dict[m]) / float(N)
            term_2 = delta_f_empirical_squared_dict[m]
            delta_f1_squared_dict[m] = 1. / K * _np.max([term_1, term_2])
    return delta_f1_squared_dict

def checkEqual(iterator):
    """
    Checks if every element of iterator is the same.
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def summary_dict_to_one_freq_dict(summary_dict):
    """
    Maps summary dict (defined in rbutils.dataset_to_summary_dict) to one_freq_dict;
    used when at least one sequence length has only one observed frequency.
    """
    one_freq_dict = _OrderedDict({})
    one_freq_dict['m_list'] = []
    one_freq_dict['n_0_list'] = []
    one_freq_dict['N_list'] = []
    one_freq_dict['K_list'] = []
    for key in summary_dict.keys():
        if isinstance(key,int):
            if checkEqual(summary_dict[key]):
                one_freq_dict['m_list'].append(key)
                one_freq_dict['n_0_list'].append(summary_dict[key][0])
                one_freq_dict['N_list'].append(summary_dict[key,u'N'])
                one_freq_dict['K_list'].append(summary_dict[key,u'K'])
    return one_freq_dict

# def dataset_to_mkn_dict(dataset,seqs,success_spam_label):
#     """
#     Maps an RB dataset to an ordered dictionary; keys are 
#     sequence lengths, values are arrays of length N (number of experimental samples) where
#     value of the i^th element is the total number of RB sequences of length M that saw
#     i successes.
#     """
#     output = _OrderedDict({})
#     N = None
#     for seq in seqs:
#         m = len(seq)
#         N_temp = int(_np.round(dataset[seq].total()))
#         if N is None:
#             N = N_temp
#         elif N_temp != N:
#             raise ValueError("Different N values discovered!")
#         n = dataset[seq][success_spam_label]
#         try:
#             output[m][n] += 1
#         except:
#             output[m] = _np.zeros(N+1)
#             output[m][n] += 1
#     return output
# 
# def mkn_dict_to_weighted_delta_f1_hat_dict(mkn_dict):
#     """
#     Maps mkn dict (defined in rbutils.dataset_to_mkn_dict) to 
#     weighted_f1_hat_dict.
#     """
#     weighted_f1_hat_dict = _OrderedDict({})
#     for m in mkn_dict.keys():
#         K = _np.sum(mkn_dict[m])
#         N = len(mkn_dict[m]) - 1
#         kn = mkn_dict[m]
#         f1_hat = 1. / (K * N) * _np.sum([n * kn[n] for n in xrange(N+1)])
#         weighted_f1_hat_dict[m] = f1_hat * (1. - f1_hat) / N
#     return weighted_f1_hat_dict
# 
# def mkn_dict_to_f_empirical_squared_dict(mkn_dict):
#     """
#     Maps mkn dict (defined in rbutils.dataset_to_mkn_dict) to 
#     f_empirical_squared_dict.
#     """
#     f_empirical_squared_dict = _OrderedDict({})
#     for m in mkn_dict.keys():
#         K = _np.sum(mkn_dict[m])
#         N = len(mkn_dict[m]) - 1
#         kn = mkn_dict[m]
#         f_empirical_squared_dict[m] = 1. / (2 * K**2 * N**2) * _np.sum(
#                 [ (i - j) * kn[i] * kn[j] for i in xrange(N+1) for j in xrange(N+1)])
#     return f_empirical_squared_dict
#     
# def mkn_dict_to_delta_f1_squared_dict(mkn_dict):
#     """
#     Maps mkn dict (defined in rbutils.dataset_to_mkn_dict) to 
#     delta_f1_squared_dict.
#     """
#     delta_f1_squared_dict = _OrderedDict({})
#     f_empirical_squared_dict = mkn_dict_to_f_empirical_squared_dict(mkn_dict)
#     weighted_delta_f1_hat_dict = mkn_dict_to_weighted_delta_f1_hat_dict(mkn_dict)
#     for m in mkn_dict.keys():
#         K = _np.sum(mkn_dict[m])
#         delta_f1_squared_dict[m] = 1. / K * _np.max(f_empirical_squared_dict[m],weighted_delta_f1_hat_dict[m])
#     return delta_f1_squared_dict
