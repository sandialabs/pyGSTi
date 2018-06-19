""" RB-related functions of gates and gatesets """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import analysis as _analysis
from ... import tools as _tls
from ... import objects as _objs

import numpy as _np
import warnings as _warnings

def average_gate_infidelity(A ,B, mxBasis="gm"):
    # Todo : docstring
    return 1 - _tls.average_gate_fidelity(A ,B, mxBasis)

def entanglement_infidelity(A, B, mxBasis=None):
    # Todo : docstring
    return 1 - float(_tls.process_fidelity(A, B, mxBasis))

def gateset_infidelity(gs, gs_target, itype = 'EI', 
                       weights=None, mxBasis=None):
    """
    Computes the average-over-gates of the average gate infidelity between 
    gates in `gs` and the gates in `gs_target`. This quantity is sometimes
    called the "average error rate" and Proctor et al Phys. Rev. Lett. 119, 
    130502 (2017) it is called the average gateset infidelity.
    
    Parameters
    ----------
    gs : GateSet
        The gateset to calculate the average infidelity, to `gs_target`, of.

    gs_target : GateSet
        The gateset to calculate the average infidelity, to `gs`, of.
        
    itype : str, optional
        The infidelity type. Either 'EI', corresponding to entanglement
        infidelity, or 'AGI', corresponding to average gate infidelity.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are, possibly unnormalized, probabilities. 
        These probabilities corresponding to the weighting in the average,
        so if the gateset contains gates A and B and weights[A] = 2 and
        weights[B] = 1 then the output is Inf(A)*2/3  + Inf(B)/3 where
        Inf(X) is the infidelity (to the corresponding element in the other
        gateset) of X. If None, a uniform-average is taken, equivalent to
        setting all the weights to 1.

    mxBasis : {"std","gm","pp"} or Basis object, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
        
    Returns
    -------
    float
        The average infidelity between the two gatesets.
        
    """
    assert(itype == 'AGI' or itype == 'EI'), "The infidelity type must be `AGI` (average gate infidelity) or `EI` (entanglement infidelity)"
    
    if mxBasis is None: mxBasis = gs.basis
        
    sum_of_weights = 0
    I_list = []
    for gate in list(gs_target.gates.keys()):
        if itype == 'AGI':
            I = average_gate_infidelity(gs[gate],gs_target[gate], mxBasis=mxBasis)
        if itype == 'EI':
            I = entanglement_infidelity(gs[gate],gs_target[gate], mxBasis=mxBasis)
        if weights is None:
            w = 1
        else:
            w = weights[gate]
       
        I_list.append(w*I)
        sum_of_weights += w
        
    assert(sum_of_weights > 0), "The sum of the weights should be positive!"
    AI = _np.sum(I_list)/sum_of_weights
        
    return AI

def predicted_RB_number(gs, gs_target, weights=None, d=None, rtype='EI'):
    """
    Predicts the RB number (aka, the RB error rate) from a gateset, 
    using the "L-matrix" theory from Proctor et al Phys. Rev. Lett. 119, 
    130502 (2017). Note that this gives the same predictions as the 
    theory by Wallman Quantum 2, 47 (2018).
    
    This theory is valid for various types of RB, including standard 
    Clifford RB -- i.e., it will accurately predict the per-Clifford 
    error rate reported by standard Clifford RB. It is also valid for
    "direct RB" under broad circumstances.
    
    For this function to be valid the gateset should be trace preserving 
    and completely positive in some representation, but the particular 
    representation of the gateset used is irrelevant, as the predicted RB 
    error rate is a gauge-invariant quantity. The function is likely reliable 
    when complete positivity is slightly violated, although the theory on
    which it is based assumes complete positivity.
    
    Parameters
    ----------
    gs : GateSet
        The gateset to calculate the RB number of. This gateset is the 
        gateset randomly sampled over, so this is not necessarily the 
        set of physical primitives. In Clifford RB this is a set of 
        Clifford gates; in "direct RB" this normally would be the 
        physical primitives.

    gs_target: GateSet
        The target gateset, corresponding to `gs`. This function is not invariant 
        under swapping `gs` and `gs_target`: this GateSet must be the target gateset,
        and should consistent of perfect gates.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are the unnormalized probabilities to apply
        each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be non-negative, and they must not all be zero. 
        Because, when divided by their sum, they must be a valid probability
        distribution. If None, the weighting defaults to an equal weighting 
        on all gates, as this is used in many RB protocols (e.g., Clifford RB).
        But, this weighting is flexible in the "direct RB" protocol.
    
    d : int, optional
        The Hilbert space dimension.  If None, then sqrt(gs.dim) is used.
        
    rtype : str, optional
        The type of RB error rate, either "EI" or "AGI", corresponding to 
        different dimension-dependent rescalings of the RB decay constant
        p obtained from fitting to Pm = A + Bp^m. "EI" corresponds to
        an RB error rate that is associated with entanglement infidelity, which
        is the probability of error for a gate with stochastic errors. This is 
        the RB error rate defined in the "direct RB" protocol. "AGI" corresponds
        to an RB error rate that is associated with average gate infidelity. This 
        is the more standard (but perhaps less well motivated) definition of the 
        RB error rate.

    Returns
    -------
    
    r : float.
        The predicted RB number.
        
    """
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    p = predicted_RB_decay_parameter(gs, gs_target, weights=weights)
    r =  _analysis.p_to_r(p, d=d, rtype=rtype)
    return r

def predicted_RB_decay_parameter(gs, gs_target, weights=None):
    """
    Computes the second largest eigenvalue of the 'L matrix' (see the `L_matrix`
    function). For standard Clifford RB and direct RB, this corresponds to the 
    RB decay parameter p in Pm = A + Bp^m for "reasonably low error" trace 
    preserving and completely positive gates. See also the `predicted_RB_number` 
    function.
    
    Parameters
    ----------
    gs : Gateset
        The gateset to calculate the RB decay parameter of. This gateset is the 
        gateset randomly sampled over, so this is not necessarily the 
        set of physical primitives. In Clifford RB this is a set of 
        Clifford gates; in "direct RB" this normally would be the 
        physical primitives.
  
    gs_target : Gateset
        The target gateset corresponding to gs. This function is not invariant under
        swapping `gs` and `gs_target`: this GateSet must be the target gateset, and
        should consistent of perfect gates.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are the unnormalized probabilities to apply
        each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be non-negative, and they must not all be zero. 
        Because, when divided by their sum, they must be a valid probability
        distribution. If None, the weighting defaults to an equal weighting 
        on all gates, as this is used in many RB protocols (e.g., Clifford RB).
        But, this weighting is flexible in the "direct RB" protocol.

    Returns
    -------
    
    p : float.
        The second largest eigenvalue of L. This is the RB decay parameter
        for various types of RB.
    """
    L = L_matrix(gs, gs_target, weights=weights)
    E = _np.absolute(_np.linalg.eigvals(L))
    E = _np.flipud(_np.sort(E))
    if abs(E[0] - 1) > 10**(-12):
        _warnings.warn("Output may be unreliable because the gateset is not approximately trace-preserving.")

    if E[1].imag > 10**(-10):
        _warnings.warn("Output may be unreliable because the RB decay constant has a significant imaginary component.")
    p = E[1]
    return p

def rb_gauge(gs, gs_target, weights=None, mxBasis=None, eigenvector_weighting=1.0):
    """
    Computes the gauge transformation required so that, when the gateset is transformed
    via this gauge-transformation, the RB number -- as predicted by the function 
    `predicted_RB_number` -- is the average gateset infidelity between the transformed
    `gs` gateset and the target gateset `gs_target`. This transformation is defined
    Proctor et al Phys. Rev. Lett. 119, 130502 (2017), and see also Wallman Quantum 2, 
    47 (2018).
    
    Parameters
    ----------
    gs : Gateset
        The RB gateset. This is not necessarily the set of physical primitives -- it 
        is the gateset randomly sampled over in the RB protocol (e.g., the Cliffords).
  
    gs_target : Gateset
        The target gateset corresponding to gs. This function is not invariant under
        swapping `gs` and `gs_target`: this GateSet must be the target gateset, and
        should consistent of perfect gates.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are the unnormalized probabilities to apply
        each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be non-negative, and they must not all be zero. 
        Because, when divided by their sum, they must be a valid probability
        distribution. If None, the weighting defaults to an equal weighting 
        on all gates, as this is used in many RB protocols (e.g., Clifford RB).
        But, this weighting is flexible in the "direct RB" protocol.
                
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from the gateset.
        
    eigenvector_weighting : float, optional
        Must be non-zero. A weighting on the eigenvector with eigenvalue that
        is the RB decay parameter, in the sum of this eigenvector and the
        eigenvector with eigenvalue of 1 that defines the returned matrix `l_operator`. 
        The value of this factor does not change whether this `l_operator` transforms into
        a gauge in which r = AGsI, but it may impact on other properties of the
        gates in that gauge. It is irrelevant if the gates are unital.
        
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
        _warnings.warn("Output may be unreliable because the gateset is not approximately trace-preserving.")        
      
    absgam[index_max] = 0.0
    index_2ndmax = _np.argmax(absgam)
    decay_constant = gam[index_2ndmax]
    if decay_constant.imag > 10**(-12):
        _warnings.warn("Output may be unreliable because the RB decay constant has a significant imaginary component.")

    vec_l_operator = vecs[:,index_max] + eigenvector_weighting*vecs[:,index_2ndmax]
    
    if mxBasis is None:
        mxBasis = gs.basis.name
    assert(mxBasis=='pp' or mxBasis=='gm' or mxBasis=='std'), "mxBasis must be 'gm', 'pp' or 'std'."
    
    if mxBasis is 'pp' or 'gm':
        assert(_np.amax(vec_l_operator.imag) < 10**(-15)), "If 'gm' or 'pp' basis, RB gauge matrix should be real."
        vec_l_operator = vec_l_operator.real
        
    vec_l_operator[abs(vec_l_operator) < 10**(-15)] = 0.
    l_operator = _tls.unvec(vec_l_operator) 
    
    return l_operator

def transform_to_rb_gauge(gs, gs_target, weights=None, mxBasis=None, eigenvector_weighting=1.0):
    """
    Transforms a GateSet into the "RB gauge" (see the `RB_gauge` function), as 
    introduced in Proctor et al Phys. Rev. Lett. 119, 130502 (2017). This gauge 
    is a function of both the gateset and its target. These may be input in any 
    gauge, for the purposes of obtaining "r = average gateset infidelity" between 
    the output GateSet and gs_target.
    
    Parameters
    ----------
    gs : Gateset
        The RB gateset. This is not necessarily the set of physical primitives -- it 
        is the gateset randomly sampled over in the RB protocol (e.g., the Cliffords).
  
    gs_target : Gateset
        The target gateset corresponding to gs. This function is not invariant under
        swapping `gs` and `gs_target`: this GateSet must be the target gateset, and
        should consistent of perfect gates.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are the unnormalized probabilities to apply
        each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be non-negative, and they must not all be zero. 
        Because, when divided by their sum, they must be a valid probability
        distribution. If None, the weighting defaults to an equal weighting 
        on all gates, as this is used in many RB protocols (e.g., Clifford RB).
        But, this weighting is flexible in the "direct RB" protocol.
                
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from the gateset.
        
    eigenvector_weighting : float, optional
        Must be non-zero. A weighting on the eigenvector with eigenvalue that
        is the RB decay parameter, in the sum of this eigenvector and the
        eigenvector with eigenvalue of 1 that defines the returned matrix `l_operator`. 
        The value of this factor does not change whether this `l_operator` transforms into
        a gauge in which r = AGsI, but it may impact on other properties of the
        gates in that gauge. It is irrelevant if the gates are unital.
        
    Returns
    -------   
    gs_in_RB_gauge: GateSet
        The gateset `gs` transformed into the "RB gauge".
        
    """            
    l = rb_gauge(gs,gs_target,weights=weights,mxBasis=mxBasis,
                 eigenvector_weighting=eigenvector_weighting)
    gs_in_RB_gauge = gs.copy()
    S = _objs.FullGaugeGroupElement( _np.linalg.inv(l) )
    gs_in_RB_gauge.transform( S )         
    return gs_in_RB_gauge

def L_matrix(gs, gs_target, weights=None):
    """
    Constructs a generalization of the 'L-matrix' linear operator on superoperators,
    from Proctor et al Phys. Rev. Lett. 119, 130502 (2017), represented as a 
    matrix via the "stack" operation. This eigenvalues of this matrix 
    describe the decay constant (or constants) in an RB decay curve for an 
    RB protocol whereby random elements of the provided gateset are sampled 
    according to the `weights` probability distribution over the
    gateset. So, this facilitates predictions of Clifford RB and direct RB 
    decay curves.
    
    Parameters
    ----------
    gs : Gateset
        The RB gateset. This is not necessarily the set of physical primitives -- it 
        is the gateset randomly sampled over in the RB protocol (e.g., the Cliffords).
  
    gs_target : Gateset
        The target gateset corresponding to gs. This function is not invariant under
        swapping `gs` and `gs_target`: this GateSet must be the target gateset, and
        should consistent of perfect gates.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates 
        in `gs` and the values are the unnormalized probabilities to apply
        each gate at each stage of the RB protocol. If not None, the values 
        in weights must all be non-negative, and they must not all be zero. 
        Because, when divided by their sum, they must be a valid probability
        distribution. If None, the weighting defaults to an equal weighting 
        on all gates, as this is used in many RB protocols (e.g., Clifford RB).
        But, this weighting is flexible in the "direct RB" protocol.
                
    Returns
    -------
    L : float
        A weighted version of the L operator from Proctor et al Phys. Rev. Lett. 
        119, 130502 (2017), represented as a matrix using the 'stacking' convention.  
        
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

def R_matrix_predicted_RB_decay_parameter(gs, group, subset_sampling=None, 
                                          group_to_gateset=None, weights=None, d=None):
    """
    Returns the second largest eigenvalue of a generalization of the 'R-matrix' [see the 
    `R_matrix` function] introduced in Proctor et al Phys. Rev. Lett. 119, 130502 (2017).
    This number is a prediction of the RB decay parameter for trace-preserving gates and 
    a variety of forms of RB, including Clifford and direct RB.
    
    Parameters
    ----------
    gs : Gateset
        The gateset to predict the RB decay paramter for. If `subset_sampling` is None 
        or `group_to_gateset` is None, the labels of the gates in `gs` should be the 
        same as the labels of the group elements in `group`. For Clifford RB this 
        would be the clifford gateset, for direct RB it would be the primitive gates.
            
    group : MatrixGroup
        The group that the `gs` gateset contains gates from (`gs` does not
        need to be the full group, and could be a subset of `group`). For
        Clifford RB and direct RB, this would be the Clifford group.
        
    subset_sampling : list, optional
        If not None, a list of gate labels from `gs`, for which random sequences of 
        this subset of gates are implemented in the RB protocol. Even if this is 
        all of the gates of `gs`, this list needs to be specified if `gs` and 
        `group` are either (1) not labelled the same (and so `group_to_gateset` is 
        not None), or (2) `gs` is a subset of group.
        
    group_to_gateset : dict, optional
        If not None, a dictionary that maps labels of group elements to labels
        of `gs`. Only used if `subset_sampling` is not None. If `subset_sampling` is 
        not None and the `gs` and `group` elements have the same labels, this dictionary
        is not required. Otherwise it is necessary.
        
    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates in `gs` 
        and the values are the unnormalized probabilities to apply each gate at 
        each stage of the RB protocol. If not None, the values in weights must all
        be positive or zero, and they must not all be zero (because, when divided by 
        their sum, they must be a valid probability distribution). If None, the
        weighting defaults to an equal weighting on all gates, as used in most RB
        protocols.
      
    d : int, optional
        The Hilbert space dimension. If None, then sqrt(gs.dim) is used.
    
    Returns
    -------
    p : float
        The predicted RB decay parameter. Valid for standard Clifford RB or direct RB
        with trace-preserving gates, and in a range of other circumstances.
        
    """ 
    if d is None: d = int(round(_np.sqrt(gs.dim)))
    R = R_matrix(gs,group,subset_sampling=subset_sampling,
                 group_to_gateset=group_to_gateset,weights=weights,d=d)
    E = _np.absolute(_np.linalg.eigvals(R))
    E = _np.flipud(_np.sort(E))
    p = E[1]
    return p

def R_matrix(gs, group, subset_sampling=None, group_to_gateset=None,
             weights=None, d=None):
    """
    Constructs a generalization of the 'R-matrix' of Proctor et al Phys. 
    Rev. Lett. 119, 130502 (2017). This matrix described the *exact* behaviour 
    of the average surival probablities of RB sequences. This matrix is 
    super-exponentially large in the number of qubits, but can be constructed 
    for 1-qubit gatesets.
    
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
    extended_E = _np.kron(_tls.column_basis_vector(0,group_dim).T,gs.povms['Mdefault'][success_effectLabel].T)
    extended_rho = _np.kron(_tls.column_basis_vector(0,group_dim),gs.preps[success_prepLabel])
    
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
    identity_vec = _tls.vec(_np.identity(d**2,float))
    
    if compilation is not None:
        gs_group = _cnst.build_alias_gateset(gs_go,compilation)
        gs_target_group = _cnst.build_alias_gateset(gs_target,compilation)
        delta = gate_dependence_of_errormaps(gs_group,gs_target_group,norm=norm,d=d)
        emaps = errormaps(gs_group,gs_target_group)
        E_eff = _np.dot(gs_go.povms['Mdefault'][success_effectLabel].T,emaps.gates['Gavg'])
        
        if twirled is True:
            L_group = L_matrix(gs_group,gs_target_group)
        
    if compilation is None:
        delta = gate_dependence_of_errormaps(gs_go,gs_target,norm=norm,d=d)
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
            L_m_rdd = _tls.unvec(_np.dot(L_group,_np.dot(Literate,identity_vec)))
        else:
            L_m_rdd = _tls.unvec(_np.dot(Literate,identity_vec))
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
    
    
def errormaps(gs, gs_target):
    """
    Computes the 'left-multiplied' error maps associated with a noisy gate 
    set, along with the average error map. This is the gate set [E_1,...] 
    such that 
    
    G_i = E_iT_i, 
    
    where T_i is the gate which G_i is a noisy 
    implementation of. There is an additional gate in the set, that has 
    the key 'Gavg' and which is the average of the error maps.
    
    Parameters
    ----------
    gs : GateSet
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
    errormaps = gs.copy()
    for gate in list(gs_target.gates.keys()):
        errormaps.gates[gate] = _np.dot(gs.gates[gate], 
                               _np.transpose(gs_target.gates[gate]))     
        errormaps_gate_list.append(errormaps.gates[gate])
        
    errormaps.gates['Gavg'] = _np.mean( _np.array([ i for i in errormaps_gate_list]), 
                                        axis=0, dtype=_np.float64)           
    return errormaps

def gate_dependence_of_errormaps(gs, gs_target, norm='diamond', mxBasis=None):
    """
    Computes the "gate-dependence of errors maps" parameter defined by
    
    delta_avg = avg_i|| E_i - avg_i(E_i) ||, 
    
    where E_i are the error maps, and the norm is either the diamond norm 
    or the 1-to-1 norm. This quantity is defined in Magesan et al PRA 85 
    042311 2012.
    
    Parameters
    ----------
    gs : GateSet
        The actual gateset
    
    gs_target : GateSet
        The target gateset.
        
    norm : str, optional
        The norm used in the calculation. Can be either 'diamond' for
        the diamond norm, or '1to1' for the Hermitian 1 to 1 norm.
        
    mxBasis : {"std","gm","pp"}, optional
        The basis of the gatesets. If None, the basis is obtained from
        the gateset.
 
    Returns
    -------
    delta_avg : float
        The value of the parameter defined above.
        
    """
    error_gs = errormaps(gs, gs_target)
    delta = []
    
    if mxBasis is None:
        mxBasis = gs.basis.name
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

def Magesan_theory_parameters(gs, gs_target, success_outcomelabel=('0',), 
                              norm='1to1', d=2):                   
    """
    From a given actual and target gateset, computes the parameters
    of the 'zeroth order' and 'first order' RB theories of Magesan et al PRA 85
    042311 2012.
    
    Parameters
    ----------
    gs : GateSet
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
        A1 = B_1, 
        B1 = A_1 - C_1(q/p^(-2) - 1), 
        C1 = C_1(q/p^(-2) - 1)
        where the parameter name on the LHS of each equality is that used
        herein, and the parameter name on the RHS of each equality is that
        used in PRA 85 042311 2012.
        
        delta : measure of gate-depedence of the noise, as defined in 
        PRA 85 042311 2012 (taking the case of time-independent noise therein).    
    """
    Magesan_theory_params = {}
    Magesan_theory_params['r'] = average_gate_infidelity(gs,gs_target)    
    Magesan_theory_params['p'] = r_to_p(Magesan_theory_params['r'],d,itype='AGI')
    Magesan_theory_params['delta'] = gate_dependence_of_errormaps(gs, gs_target,
                                                                  norm,None,d)
    error_gs = errormaps(gs, gs_target)   
       
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
    
    error_gs.preps['rhoc_mixed'] = 1./d*_cnst.basis_build_identity_vec(error_gs.basis)

    #Assumes standard POVM labels
    povm = _objs.UnconstrainedPOVM( [('0_cm', gs_target.povms['Mdefault']['0']),
                                     ('1_cm', gs_target.povms['Mdefault']['1'])] )
    ave_error_gsl = _cnst.gatestring_list([('rho0','Gavg'),('rho0','GR'),('rho0','Gavg','GQ')])
    N=1
    data = _cnst.generate_fake_data(error_gs, ave_error_gsl, N, 
                                    sampleError="none")

    #TIM: not sure how below code is supposed to work...
    #if isinstance(success_outcomelabel, tuple):
    #     = (success_outcomelabel[0] +'_cm',)
    #else:
    #    success_outcomelabel_cm = success_outcomelabel +'_cm'
    success_outcomelabel_cm = success_outcomelabel #Eriks HACK to get code to run...
    
    pr_L_p = data[('rho0','Gavg')][success_outcomelabel]
    pr_L_I = data[('rho0','Gavg')][success_outcomelabel_cm]
    pr_R_p = data[('rho0','GR')][success_outcomelabel]
    pr_R_I = data[('rho0','GR')][success_outcomelabel_cm]
    pr_Q_p = data[('rho0','Gavg','GQ')][success_outcomelabel]
    p = Magesan_theory_params['p']    
    B_1 = pr_R_I
    A_1 = (pr_Q_p/p) - pr_L_p + ((p -1)*pr_L_I/p) \
                            + ((pr_R_p - pr_R_I)/p)
    C_1 = pr_L_p - pr_L_I
    q = average_gate_infidelity(error_gs.gates['GQ2'],_np.identity(d**2,float))
    q = r_to_p(q,d)
    
    Magesan_theory_params['A'] = pr_L_I
    Magesan_theory_params['B'] = pr_L_p - pr_L_I       
    Magesan_theory_params['A1'] = B_1
    Magesan_theory_params['B1'] = A_1 - C_1*(q - 1)/p**2
    Magesan_theory_params['C1'] = C_1*(q- p**2)/p**2

    return Magesan_theory_params
            
def Magesan_systematic_error_bounds(m,delta,order='zeroth'):
    """
    Finds an upper bound on the RB average surival probability (ASP) from the 
    predicted RB ASP of the 'first' or 'zeroth' order models from 
    Magesan et al PRA 85 042311 2012, using the bounds given therein.
    
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
    if order == 'zeroth':
        sys_eb = (delta + 1)**(m+1) - 1
    
    elif order == 'first':
        sys_eb = sys_eb - (m+1)*delta
    else:
        raise ValueError("`order` should be 'zeroth' or 'first'.")
    
    upper = y + sys_eb
    upper[upper > 1]=1

    lower = y - sys_eb
    lower[lower < 0]=0
       
    return upper, lower