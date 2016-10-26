from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Randomized Benhmarking Utility Routines """

import numpy as _np
from collections import OrderedDict as _OrderedDict

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

def rb_decay_1st_order(m,A1,B1,C1,D1,f1):
    """
    Computes the first order survival probability function 
    F = A1 + B1*f1^m + C1(D1-f1^2)f1^(m-2), as provided in Equation 3 of "Scalable 
    and Robust Randomized Benchmarking of Quantum Processes" 
    (http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.106.180504).

    Parameters
    ----------
    m : integer
        RB sequence length minus one
    
    A1,B1,C1,D1,f1 : float

    Returns
    -------
    float
    """

    return A1+B1*f1**m+C1*(m-1)*(D1-f1**2)*f1**(m-2)


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

