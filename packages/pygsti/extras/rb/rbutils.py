from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Randomized Benhmarking Utility Routines """

import numpy as _np
from collections import OrderedDict as _OrderedDict
from ... import objects as _objs
from ... import construction as _cnst
from ... import algorithms as _algs
from ... import tools as _tls
from scipy.linalg import sqrtm
from scipy.linalg import eig
import itertools as _ittls

def standard_fit_function(m,A,B,p):
    """
    Computes the standard fitting function for RB average survival probablities 
    P_m = A + B * p^m, as provided in, e.g., Equation 1 of "Randomized benchmarking 
    with confidence" (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).

    Parameters
    ----------
    m : integer
        Length of random RB sequence (so not including the inversion gate).
    
    A,B,p : float

    Returns
    -------
    float
    """
    return A+B*p**m

def first_order_fit_function(m,A1,B1,C1,p):
    """
    Computes the 'first order' fitting function for RB average survival probablities
    P_m = A1 + (B1 + m * C1) * p^m. This is a simplified verion of the 'first order'
    function P_m = A_1*p^m + B_1 + C_1 (m-1)(q-p^2)p^(m-2), as provided in Equation 3 
    of "Scalable and Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).
    The model therein has 1 to many parameters for the fitting. The conversion is
    A1 = B_1
    B1 = A_1 - C_1(q/p^(-2) - 1)
    C1 = C_1(q/p^(-2) - 1)
    where the LHS (RHS) quantites in this equation are those of our (Magesan et al.'s)
    fitting function.

    Parameters
    ----------
    m : integer
        Length of random RB sequence (so not including the inversion gate).
    
    A1,B1,C1,f1 : float

    Returns
    -------
    float
    """
    return A1+(B1+C1*m)*p**m

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
    Inverse of p_to_r function: see above   
    """
    p = 1 - d * r / (d - 1)
    return p

def group_twirl(M,group):
    """
    Returns the matrix group twirl of a map M:  
    Twirl(M) = 1/|group| * Sum_{C in group} (C^-1 * M * C)
    
    Parameters
    ----------
    M : array or gate
        The map to be twirled.

    group : MatrixGroup
        The group to twirl over (e.g., the Clifford group).
    
    Returns
    -------
    M_twirl : array
        The twirl of M.
    """
    G = len(group)
    M_twirl = 1.0/G * _np.sum(
        _np.dot( _np.dot(group.get_matrix_inv(i),M),
                 group.get_matrix(i)) for i in range(G))
    return M_twirl

def average_gate_infidelity(gate_actual,gate_target,d=2):
    """
    Computes the average gate infidelity (AGI) between an actual and a target
    gate. This quantity is defined in, e.g., arXiv:1702.01853 (see Eq (2)).
    Calculated via the relationship between process fidelity (F_p) and average gate
    fidelity (F_g): F_g = (d * F_p + 1)/(1 + d) given in
    Phys. Lett. A 303 (2002) 249-252 (F_p is called entanglement Fidelity therein).

    Parameters
    ----------
    gate_actual : array or gate
        The noisy gate whose AGI is to be computed (to the target gate).
        
    gate_target : array or gate
        The target gate against which "actual" is being compared.

    Returns
    ----------
    AGI : float
        The AGI of the noisy to the target gate.
    """
    process_fidelity = _tls.process_fidelity(gate_actual,gate_target)
    AGF = (d*process_fidelity + 1)/(1+d)
    AGI = 1 - AGF
    return float(AGI)

def average_gateset_infidelity(gs_actual,gs_target,d=2):
    """
    Computes the average gateset infidelity (AGsI) between noisy gates and target gates.
    This quantity is defined in, e.g., arXiv:1702.01853 (see Eq (2) and below), and is 
    the mean of the average gate infidelities of the actual to the target gates.
    
    Parameters
    ----------
    gs_actual : GateSet
        Noisy gateset to calculate the AGsI of (to the target gateset).

    gs_clifford_target : GateSet
        Target gateset.

    Returns
    -------
    AGsI : float
        The AGsI of the actual gateset to the target gateset.    
    """
    AGI_list = []
    for gate in list(gs_target.gates.keys()):
        AGI_list.append(average_gate_infidelity(gs_actual.gates[gate],
                gs_target.gates[gate],d))
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
    
    gs_target : GateSet
        Target gateset.
    
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
        
    errormaps['Gavg'] = _np.mean( _np.array([ i for i in errormaps_gate_list]), 
                                      axis=0, dtype=_np.float64)           
    return errormaps

def gatedependence_of_errormaps(gs_actual, gs_target, norm='1to1', d=2):
    """
    Computes the 'delta' parameter used to calculate the systematic error
    of the zeroth or first order theories of Magesan et al PRA 85
    042311 2012, and the systematic error of the 'L matrix' RB theory of
    arXiv:1702.01853. This parameter is a measure of the gate-dependence 
    of the error maps (wrt a target gateset) of the noisy gate set.
    
    Parameters
    ----------
    gs_actual : GateSet
    
    gs_target : GateSet
        Target gateset.
        
    norm : Str, optional
        The norm used in the calculation. Can be either 'diamond' for
        the diamond norm, or '1to1' for the Hermitian 1 to 1 norm.
    
    Returns
    -------
    delta_avg : float
        The value of the delta parameter calculated for the given
        norm and gate sets.    
    """
    error_gs = errormaps(gs_actual, gs_target)
    delta = []
    
    for gate in list(gs_target.gates.keys()):
        if norm=='diamond':
            delta.append(_tls.diamonddist(error_gs.gates[gate],
                                           error_gs.gates['Gavg']))            
        elif norm=='1to1':
            gate_dif = _tls.gm_to_std(error_gs.gates[gate]-error_gs.gates['Gavg'],d)
            delta.append(norm1to1(gate_dif,n_samples=1000, return_list=False))            
        else:
            raise ValueError("Only diamond or 1to1 norm available. "
                             + "set norm='diamond' or norm='1to1'")            
    delta_avg = _np.mean(delta)    
    return delta_avg
        

def Magesan_theory_parameters(gs_actual, gs_target, success_spamlabel='plus', 
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
        
    success_spamlabel : str, optional
        The spam label associated with survival.    
        
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
    Magesan_theory_params['r'] = average_gateset_infidelity(gs_actual,gs_target,d)    
    Magesan_theory_params['p'] = r_to_p(Magesan_theory_params['r'],d)
    Magesan_theory_params['delta'] = gatedependence_of_errormaps(gs_actual, 
                                                                 gs_target, norm,d)
    error_gs = errormaps(gs_actual, gs_target)   
       
    R_list = []
    Q_list = []
    for gate in list(gs_target.gates.keys()):
        R_list.append(_np.dot(_np.dot(error_gs[gate],gs_target.gates[gate]),
              _np.dot(error_gs['Gavg'],_np.transpose(gs_target.gates[gate]))))
        Q_list.append(_np.dot(gs_target.gates[gate],
              _np.dot(error_gs[gate],_np.transpose(gs_target.gates[gate]))))
    
    error_gs['GR'] = _np.mean(_np.array([ i for i in R_list]), 
                                      axis=0, dtype=_np.float64)
    error_gs['GQ'] = _np.mean(_np.array([ i for i in Q_list]), 
                                      axis=0, dtype=_np.float64)    
    error_gs['GQ2'] = _np.dot(error_gs['GQ'],error_gs['Gavg'])
    
    error_gs['rhoc_mixed'] = 1./d*error_gs['identity']
    error_gs.spamdefs['plus_cm'] = ('rhoc_mixed','E0')
    error_gs.spamdefs['minus_cm'] = ('rhoc_mixed','remainder')
    gsl = [('Gavg',),('GR',),('Gavg','GQ',)]   
    ave_error_gsl = _cnst.create_gatestring_list("a", a=gsl)
    N=1
    data = _cnst.generate_fake_data(error_gs, ave_error_gsl, N, 
                                    sampleError="none", seed=1)
    
    success_spamlabel_cm = success_spamlabel +'_cm'
    
    pr_L_p = data[('Gavg',)][success_spamlabel]
    pr_L_I = data[('Gavg',)][success_spamlabel_cm]
    pr_R_p = data[('GR',)][success_spamlabel]
    pr_R_I = data[('GR',)][success_spamlabel_cm]
    pr_Q_p = data[('Gavg','GQ',)][success_spamlabel]
    p = Magesan_theory_params['p']    
    B_1 = pr_R_I
    A_1 = (pr_Q_p/p) - pr_L_p + ((p -1)*pr_L_I/p) \
                            + ((pr_R_p - pr_R_I)/p)
    C_1 = pr_L_p - pr_L_I
    q = average_gate_infidelity(error_gs['GQ2'],_np.identity(d**2,float),d)
    q = r_to_p(q,d)
    
    if p < 0.01:
        print("Warning: first order theory parameters are not guaranteed \
              to be reliable with a very large decay rate")        
    Magesan_theory_params['A'] = pr_L_I
    Magesan_theory_params['B'] = pr_L_p - pr_L_I       
    Magesan_theory_params['A1'] = B_1
    Magesan_theory_params['B1'] = A_1 - C_1*(q - 1)/p**2
    Magesan_theory_params['C1'] = C_1*(q- p**2)/p**2

    return Magesan_theory_params

def gateset_metrics(gs_actual, gs_target, group,d=2,output=False):
    """
    From a given actual and target gateset, computes three error rates
    that are related to the RB number: the average gateset infidelity,
    the scalar error rates derived from the R and L matrices.
    
    Parameters
    ----------
    gs_actual : GateSet
        The gateset to compute the parameters for
    
    gs_target : GateSet
        Target gateset.
       
    group: MatrixGroup
        The matrix group that the gateset is an implementation of.
    
    d : int, optional
        The dimension.
    
    output : bool, optional
        If True, then these parameters are printed to screen in addtion
        to being returned by the function.
    
    Returns
    -------
    
    gateset_metrics : dictionary of floats.
        The three theoretical error metrics.
    """
    gateset_metrics = {}
    gateset_metrics['AGsI'] = average_gateset_infidelity(gs_actual,gs_target,d)
    gateset_metrics['r_R'] = p_to_r(R_single_decay_parameter(gs_actual,group,d))
    gateset_metrics['r_L'] = p_to_r(L_single_decay_parameter(gs_actual,gs_target,d))    
    if output is True:
        print('The average gateset infidelity is:', gateset_metrics['AGsI'])
        print('The scalar error metric obtained from the R matrix is:', 
              gateset_metrics['r_R'])
        print('The scalar error metric obtained from the L matrix is:', 
              gateset_metrics['r_L'])    
    return gateset_metrics 
    
def R_single_decay_parameter(gs,group,d=2):
    """
    Computes the second largest eigenvalue of the R matrix.
    
    Parameters
    ----------
    gs_actual : GateSet

    group: MatrixGroup
        The matrix group that the gateset is an implementation of.
    
    d : int, optional
        The dimension.

    Returns
    -------
    
    E : float.
        The second largest eigenvalue of R.
    """
    R = R_matrix(gs,group,d)
    E = _np.absolute(_np.linalg.eigvals(R))
    E = _np.flipud(_np.sort(E))
    return E[1]

def L_single_decay_parameter(gs,gs_target,d=2):
    """
    As above, but for the L matrix.
    """
    L = L_matrix(gs,gs_target)
    E = _np.absolute(_np.linalg.eigvals(L))
    E = _np.flipud(_np.sort(E))
    return E[1]

def R_matrix(gs,group,d=2):
    """
    Constructs the 'R' matrix of Eq (2) in arXiv:1702.01853
    This matrix described the exact behaviour of the average surival
    probablities of RB sequences. For the Clifford group, it is 
    exponentially large in the number of qubits n.
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset for some group (e.g., the Cliffords) to 
        calculate the R matrix of. Gate elements must be labelled as
        'Gc..' where .. are the labels of the MatrixGroup 'group'.
        
    
    group : MatrixGroup
        The group that the 'gs' gateset is an implementation of. The
        group elements should be labelled to correspond to the elements
        of 'gs'.
      
    d : int, optional
        Dimension of the Hilbert space. Defaults to a single qubit.
    
    Returns
    -------
    R : float
        The R matrix from arXiv:1702.01853        
    """           
    group_dim = len(group)
    R_dim = group_dim * d**2
    R = _np.zeros([R_dim,R_dim],float)
    for i in range(0,group_dim):
        for j in range(0,group_dim):
            gate_label_itoj = group.product([group.get_inv(i),j])
            for k in range (0,d**2):
                for l in range(0,d**2):
                    R[j*d**2+k,i*d**2+l] = gs['Gc'+str(gate_label_itoj)][k,l]
    R = R/group_dim
    
    return R

def exact_RB_ASPs(gs,group,m_max,m_min=1,m_step=1,d=2,success_spamlabel='plus'):
    """
    Calculates the exact RB average surival probablilites (ASP), using the 
    formula given in Eq (2) and the surrounding text of arXiv:1702.01853
    
    Parameters
    ----------
    gs : Gateset
        The noisy gateset for some group (e.g., the Cliffords). Gate elements 
        must be labelled as 'Gc..' where .. are the labels of the MatrixGroup 
        'group'.
           
    group : MatrixGroup
        The group that the 'gs' gateset is an implementation of. The
        group elements should be labelled to correspond to the elements
        of 'gs'.
        
    m_max : int
        maximal sequence length of the random gates (so not including the
        inversion gate).
        
    m_min : int, optional
        minimal sequence length. Defaults to the smallest valid value of 1.
        
    m_step : int, optional
        step size between sequence lengths
      
    d : int, optional
        Dimension of the Hilbert space. Defaults to a single qubit.
        
    success_spamlabel : str, optional
        Specifies the SPAM label associated with surival
    
    Returns
    -------
    m : float
        Array of sequence length values that the ASP has been calculated for
        
    P_m : float
        Array containing ASP values for the specified sequence length values.
        
    """  
    i_max = _np.floor((m_max - m_min ) / m_step).astype('int')
    m = _np.zeros(1+i_max,int)
    P_m = _np.zeros(1+i_max,float)
    group_dim = len(group)
    R_dim = group_dim * d**2
    R = R_matrix(gs,group,d)
    rho_index = gs.spamdefs[success_spamlabel][0]
    E_index = gs.spamdefs[success_spamlabel][1]
    extended_E = _np.kron(column_basis_vector(0,group_dim).T,gs[E_index].T)
    extended_rho = _np.kron(column_basis_vector(0,group_dim),gs[rho_index])
    for i in range (0,1+i_max):
        m[i] = m_min + i*m_step
        P_m[i] = group_dim*_np.dot(extended_E,_np.dot(
                _np.linalg.matrix_power(R,m_min + i*m_step+1),extended_rho))
    return m, P_m

def L_matrix(gs,gs_target):
    """
    Constructs the 'L' linear operator on superoperators, from arXiv:1702.01853,
    represented as a matrix. This matrix provides a good approximation to the 
    RB decay curve for low-error gatesets.
    
    Parameters
    ----------
    gs : Gateset
  
    gs_target : Gateset
        Target gateset

    Returns
    -------
    L : float
        The L operator from arXiv:1702.01853, represented as a matrix.       
    """  
    dim = len(gs_target.gates.keys())
    L_matrix = 1 / dim * _np.sum(_np.kron(gs[key].T,
                 _np.linalg.inv(gs_target[key])) for key in gs_target.gates.keys())
    return L_matrix

def L_matrix_ASPs(gs,gs_target,m_max,m_min=1,m_step=1,d=2,success_spamlabel='plus',
                  norm='diamond'):
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
        The noisy gateset for some group (e.g., the Cliffords). Gate elements 
        must be labelled as 'Gc..' where .. are the labels of the MatrixGroup 
        'group'.
           
    gs_target : Gateset
        Target gateset
        
    m_max : int
        maximal sequence length of the random gates (so not including the
        inversion gate).
        
    m_min : int, optional
        minimal sequence length. Defaults to the smallest valid value of 1.
        
    m_step : int, optional
        step size between sequence lengths
      
    d : int, optional
        Dimension of the Hilbert space. Defaults to a single qubit.
        
    success_spamlabel : str, optional
        Specifies the SPAM label associated with surival
        
    norm : str, optional
        The norm used in the error bound calculation. Default is consistent with
        arXiv:1702.01853.
    
    Returns
    -------
    m : float
        Array of sequence length values that the ASP has been calculated for
        
    P_m : float
        Array containing predicted ASP values for the specified sequence length values.
        
    lower_seb_PM: float
        Array containing lower bounds on the possible ASP values

    upper_seb_PM: float
        Array containing upper bounds on the possible ASP values
    """      
    gs_go = _algs.gaugeopt_to_target(gs,gs_target)
    L = L_matrix(gs_go,gs_target)
    dim = len(gs_target.gates.keys())
    rho_index = gs.spamdefs[success_spamlabel][0]
    E_index = gs.spamdefs[success_spamlabel][1]
    emaps = errormaps(gs_go,gs_target)
    E_eff = _np.dot(gs_go[E_index].T,emaps['Gavg'])
    identity_vec = vec(_np.identity(d**2,float))    
    delta = gatedependence_of_errormaps(gs_go,gs_target,norm,d)
    
    i_max = _np.floor((m_max - m_min ) / m_step).astype('int')
    m = _np.zeros(1+i_max,int)
    P_m = _np.zeros(1+i_max,float)
    upper_seb_PM = _np.zeros(1+i_max,float)
    lower_seb_PM = _np.zeros(1+i_max,float)
    for i in range (0,1+i_max):
        m[i] = m_min + i*m_step
        L_m_rdd = unvec(_np.dot(_np.linalg.matrix_power(L,m_min + i*m_step),identity_vec))
        P_m[i] = _np.dot(E_eff,_np.dot(L_m_rdd,gs_go[rho_index]))
        upper_seb_PM[i] = P_m[i] + delta/2
        lower_seb_PM[i] = P_m[i] - delta/2
        if upper_seb_PM[i] > 1:
            upper_seb_PM[i]=1
        if lower_seb_PM[i] < 0:
            lower_seb_PM[i]=0
        
    return m, P_m, lower_seb_PM, upper_seb_PM
          
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
                _ittls.izip(*[_ittls.chain(vector_in,
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

def norm1to1(operator, n_samples=10000, return_list=False):
    """
    Returns the Hermitian 1-to-1 norm of a superoperator represented in
    the standard basis, calculated via Monte-Carlo sampling. Definition
    of Hermitian 1-to-1 norm can be found in arxiv:1109.6887.
    """
    rand_dim = int(_np.sqrt(float(len(operator))))
    vals = [ norm1(unvec(_np.dot(operator,vec(random_hermitian(rand_dim)))))
             for n in range(n_samples)]
    if return_list:
        return vals
    else:
        return max(vals)
    
    
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

def dataset_to_summary_dict(dataset,seqs,success_spam_label,use_frequencies=False):
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
            n = dataset[seq][success_spam_label]
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
            n = dataset[seq][success_spam_label]
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