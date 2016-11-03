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
from ... import tools as _tls
from scipy.linalg import sqrtm
import itertools as _ittls

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


def rb_decay_WF(m,A,B,f):#Taken from Wallman and Flammia- Eq. 1
    """
    Computes the survival probability function F = A + B * f^m, as provided
    in Equation 1 of "Randomized benchmarking with confidence" 
    (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).

    Parameters
    ----------
    m : integer
        RB sequence length minus one
    
    A,B,f : float

    Returns
    -------
    float
    """
    return A+B*f**m

def rb_decay_1st_order(m,A1,B1,C1,f1):
    """
    Computes an altered verion of the first order survival probability function 
    F = A_1*f1^m + B_1 + C_1 (m-1)(q-p^2)p^(m-2), as provided in Equation 3 of 
    "Scalable and Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).
    The reason for the change is that the model in this paper has 1 to many
    parameters for the fitting, and is also ill-defined for m=1 and f1=0, which
    is problematic for fitting. The conversion is
    A1 = B_1
    B1 = A_1 - C_1(q/f1^(-2) - 1)
    C1 = C_1(q/f1^(-2) - 1)

    Parameters
    ----------
    m : integer
        RB sequence length minus one
    
    A1,B1,C1,f1 : float

    Returns
    -------
    float
    """

    return A1+(B1+C1*m)*f1**m


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


def f_to_F_avg(f,d=2):
    """
    Following Wallman and Flammia Eq. 2, maps fit decay fit parameter f to
    F_avg, that is, the average gate fidelity of a noise channel \mathcal{E}
    with respect to the identity channel (see W&F Eq.3).
    
    Parameters
    ----------
    f : float
        Fit parameter f from \bar{F}_m = A + B*f**m.
    
    d : int, optional
        Number of dimensions of the Hilbert space (default is 2, corresponding
        to a single qubit).     
     
    Returns
    ----------
    F_avg : float
        Average gate fidelity F_avg(\mathcal{E}) = \int(d\psi Tr[\psi 
        \mathcal{E}(\psi)]), where d\psi is the uniform measure over all pure
        states (see W&F Eq. 3).
    
    """
    F_avg = ((d-1)*f+1.)/d
    return F_avg


def f_to_r(f,d=2):
    """
    Following Wallman and Flammia, maps fit decay fit parameter f to r, the 
    "average gate infidelity".  This quantity is what is often referred to as
    "the RB number". 
    
    Parameters
    ----------
    f : float
        Fit parameter f from \bar{F}_m = A + B*f**m.
    
    d : int, optional
        Number of dimensions of the Hilbert space (default is 2,
        corresponding to a single qubit).     
     
    Returns
    -------
    r : float
        The average gate infidelity, that is, "the RB number".      
    
    """
    r = 1 - f_to_F_avg(f,d=d)
    return r

def D1f1_to_gdep(D1,f1):
    """
    Calculates the measure of date-dependence in the noise as defined below
    Equation 4 in "Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).

    Parameters
    ----------
    D1, f1 : float
        Two of the fit parameters from first order fitting model
      
    Returns
    -------
    gdep : float
    
    """
    gdep = D1 - f1**2
    return gdep

def clifford_twirl(M,clifford_group):
    """
    Returns the Clifford twirl of a map M:  
    Twirl(M) = 1/|Clifford group| * Sum_{C in Clifford group} (C^-1 * M * C)
    
    Parameters
    ----------
    M : array or gate
        The CPTP map to be twirled.

    clifford_group : MatrixGroup
        Which Clifford group to use.
    
    Returns
    -------
    M_twirl : array
        The Clifford twirl of M.
    """
    G = len(clifford_group) #order of clifford group
    M_twirl = 1.0/G * _np.sum(
        _np.dot( _np.dot(clifford_group.get_matrix_inv(i),M),
                 clifford_group.get_matrix(i)) for i in range(G))
    return M_twirl

def depolarisation_parameter(gate, clifford_group, d=2):
    """
    Returns the depolarisation parameter of the Clifford-twirled
    version of 'gate'.
    
    Parameters
    ----------
    gate : array or gate
        The map to be twirled and dep. par. to be extracted.

    clifford_group : MatrixGroup
        Which Clifford group to use.
    
    Returns
    -------
    p : float
        The depolarisation parameter
    """
    
    twirled_channel = clifford_twirl(gate,clifford_group)
    p = 1./(d**2 -1) * (_np.trace(twirled_channel) - 1)
    
    return p
    
def analytic_rb_gate_error_rate(actual, target, clifford_group):
    """
    Computes the twirled Clifford error rate for a given gate.

    Parameters
    ----------
    actual : array or gate
        The noisy gate whose twirled Clifford error rate is to be computed.
        
    target : array or gate
        The target gate against which "actual" is being compared.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    
    Returns
    ----------
    error_rate : float
        The twirled Clifford error rate.
    """
    
    twirled_channel = clifford_twirl(_np.dot(actual,_np.linalg.inv(target)),
                                     clifford_group)
    #KENNY: is below formulat correct for arbitrary clifford groups? (or 
    #  should we assert twirled_channel.shape == (4,4) here? 
    # from docstring:  *At present only works for single-qubit gates.*
    error_rate = 0.5 * (1 - 1./3 * (_np.trace(twirled_channel) - 1))
    return error_rate


def analytic_rb_clifford_gateset_error_rate(gs_clifford_actual,
                                            gs_clifford_target,
                                            clifford_group):
    """
    Computes the average twirled Clifford error rate for a noisy Clifford 
    gate set.  This is, analytically, "the RB number".
    *At present only works for single-qubit gate sets.*    
    
    Parameters
    ----------
    gs_clifford_actual : GateSet
        A gate set of estimated Clifford gates.  If the experimental gate set
        is, as is typical, not a Clifford gate set, then said gate set should
        be converted to a Clifford gate set using a clifford-to-primitive map
        with `pygsti.construction.build_alias_gateset` .

    gs_clifford_target : GateSet
        The corresponding ideal gate set of the same (Clifford) gates as
        `gs_clifford_actual`.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    Returns
    -------
    r_analytic : float
        The average per-Clifford error rate of the noisy Clifford gate set.
        This is, analytically, "the RB number".    
    """
    error_list = []
    for gate in list(gs_clifford_target.gates.keys()):
        error_list.append(analytic_rb_gate_error_rate(
                gs_clifford_actual.gates[gate],
                gs_clifford_target.gates[gate],
                clifford_group))
    r_analytic = _np.mean(error_list)
    return r_analytic

def error_gate_set(gs_actual, gs_target):
    
    """
    Computes the 'left-multiplied' error maps associated with a noisy gate 
    set, along with the average error map. This is the gate set [n_1,...] 
    such that g_i = t_i, where t_i is the gate which g_i is a noisy 
    implementation of, the final gate in the set has the key Gavg and is the
    average of the error maps.
    
    Parameters
    ----------
    gs_actual : GateSet
    
    gs_target : GateSet
        The corresponding ideal gate set of the gates as `gs_actual`.
    
    Returns
    -------
    error_gs : GateSet
        The left multplied error gates, along with the average error map,
        with the key 'Gavg'.
    
    """
    
    error_gate_list = []
    error_gs = gs_actual.copy()
    for gate in list(gs_target.gates.keys()):
        error_gs.gates[gate] = _np.dot(gs_actual.gates[gate], 
                               _np.transpose(gs_target.gates[gate]))     
        error_gate_list.append(error_gs.gates[gate])
        
    error_gs['Gavg'] = _np.mean( _np.array([ i for i in error_gate_list]), 
                                      axis=0, dtype=_np.float64)    
       
    return error_gs

def delta_parameter(gs_actual, gs_target, norm='diamond'):
    
    """
    Computes the 'delta' parameter used in the systematic error of the
    zeroth, first (or higher order) fitting models. This delta parameter
    is with respect to how close 'gs_actual' is to 'gs_target'. This 
    parameter is a measure of the gate-dependence of the error of the 
    gate set.
    
    Parameters
    ----------
    gs_actual : GateSet
    
    gs_target : GateSet
        The corresponding ideal gate set of the gates as `gs_actual`.
        
    norm : Str
        The norm used in the calculation.
    
    Returns
    -------
    delta_avg : float
        The value of the delta parameter calculated for the given
        norm and gate sets.
    
    """
    error_gs = error_gate_set(gs_actual, gs_target)
    delta = []
    
    for gate in list(gs_target.gates.keys()):
        if norm=='diamond':
            delta.append( _tls.diamonddist(error_gs.gates[gate],
                                           error_gs.gates['Gavg']))
            
        elif norm=='1to1':
            # TIM: Here dimension has been forced to be 2, as the function is
            # not passed the dimension of the gates. This should be
            # changed at some point.
            gate_dif = _tls.gm_to_std(error_gs.gates[gate]-error_gs.gates['Gavg'],2)
            delta.append(norm1to1(gate_dif,n_samples=1000, return_list=False))
            
        else:
            raise ValueError, "Only diamond or 1to1 norm available. \
            set norm='diamond' or norm='1to1'"
            
    delta_avg = _np.mean(delta)
    
    return delta_avg
        

def analytic_rb_parameters(gs_actual, gs_target, clifford_group, 
                           success_spamlabel, norm='diamond', d=2):
    # Tim: This function has a range of issues if d !=2.                     
    """
    Computes the analytic zeroth and first order fit parameters from
    a given noisy gate set. Also calculates the delta parameter used
    in bounding the difference between the analytic decay curve and
    an actual decay curve.
    
    Parameters
    ----------
    gs_actual : GateSet
    
    gs_target : GateSet
        The corresponding ideal gate set of the gates as `gs_actual`.
      
    clifford_group : MatrixGroup
        Which Clifford group to use.
        
    success_spamlabel : str
        The spam label associated with survival.    
        
    norm : str
        The norm used in the calculation.
    
    d : int
        The dimension.
    
    Returns
    -------
    analytic_params : dictionary of floats
        The values of the analytic fit parameters for the given gate 
        set. These parameters are as defined in Magesan et al PRA 85
        042311 2012, and we are considering the case of time-indep
        gates. The parameters here are converted to those therein
        as 
        B -> A_0, 
        A -> B_0,
        A1 = B_1
        B1 = A_1 - C_1(q/f1^(-2) - 1)
        C1 = C_1(q/f1^(-2) - 1)
        f1 = p
        where the parameter name used herein is first.
    
    """
    if d != 2:
        print('Analytical decay rate is correct only for d=2')
    error_gs = error_gate_set(gs_actual, gs_target)    
           
    analytic_params = {}
    analytic_params['r'] = analytic_rb_clifford_gateset_error_rate(
                            gs_actual, gs_target, clifford_group)    
    analytic_params['f'] = depolarisation_parameter(error_gs['Gavg'], 
                                             clifford_group, d=2)
    analytic_params['delta'] = delta_parameter(gs_actual, gs_target, norm)
       
    R_list = []
    Q_list = []
    for gate in list(gs_target.gates.keys()):
        R_list.append(_np.dot(_np.dot(error_gs[gate],gs_target.gates[gate]),
              _np.dot(error_gs['Gavg'],_np.transpose(gs_target.gates[gate]))))
        Q_list.append(_np.dot(gs_target.gates[gate],
              _np.dot(error_gs[gate],_np.transpose(gs_target.gates[gate]))))
    
    error_gs['GR'] = _np.mean( _np.array([ i for i in R_list]), 
                                      axis=0, dtype=_np.float64)
    error_gs['GQ'] = _np.mean( _np.array([ i for i in Q_list]), 
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
    an_f = analytic_params['f']
    
    B_1 = pr_R_I
    A_1 = (pr_Q_p/an_f) - pr_L_p + ((an_f -1)*pr_L_I/an_f) \
                            + ((pr_R_p - pr_R_I)/an_f)
    C_1 = pr_L_p - pr_L_I
    q = depolarisation_parameter(error_gs['GQ2'], clifford_group, d=2)
    
    if an_f < 0.01:
        print("Warning: first order analytical constants are not guaranteed \
              to be reliable with very large errors")
        
    analytic_params['A'] = pr_L_I
    analytic_params['B'] = pr_L_p - pr_L_I       
    analytic_params['A1'] = B_1
    analytic_params['B1'] = A_1 - C_1*(q - 1)/an_f**2
    analytic_params['C1'] = C_1*(q- an_f**2)/an_f**2

    return analytic_params

def systematic_error_bound(m,delta,order='zeroth'):
    """
    Computes the value of the systematic error bound at a given sequence
    length.
    
    Parameters
    ----------
    m : float
        Sequence length, and so it is often an int
    
    delta : float
        The size of the 'delta' parameter for the gate set in question
      
    order : str
        May be 'zeroth or 'first'. The order fitting model for which the
        error bound should be calcluated.
    
    Returns
    -------
    sys_eb: float
        The systematic error bound
    
    """
    sys_eb = (delta + 1)**(m+1) - 1
    
    if order=='first':
        sys_eb = sys_eb - (m+1)*delta

    return sys_eb

def seb_upper(y,m,delta,order='zeroth'):
    """
    Finds an upper bound on the surival probability from the analytic value
    """
    sys_eb = systematic_error_bound(m,delta,order)
    
    upper = y + sys_eb
    upper[upper > 1]=1
     
    return upper

def seb_lower(y,m,delta,order='zeroth'):
    """
    Finds a lower bound on the surival probability from the analytic value
    """
    sys_eb = systematic_error_bound(m,delta,order)
    
    lower = y - sys_eb
    lower[lower < 0]=0
       
    return lower

def vec(matrix_in):
    """
    Stack the columns of a matrix to return a vector
    """
    return [b for a in _np.transpose(matrix_in) for b in a]

def unvec(vector_in):
    """
    Slice a vector into columns of a matrix.
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
    Generates a randmon Hermitian matrix
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
    the standard basis, calculated via Monte-Carlo sampling.
    """
    rand_dim = int(_np.sqrt(float(len(operator))))
    vals = [ norm1(unvec(_np.dot(operator,vec(random_hermitian(rand_dim)))))
             for n in range(n_samples)]
    if return_list:
        return vals
    else:
        return max(vals)
