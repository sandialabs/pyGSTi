from __future__ import division, print_function, absolute_import, unicode_literals
import pygsti
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize import minimize as _minimize
from matplotlib import pyplot as _plt
from collections import OrderedDict
import pickle

from functools import reduce

def _H_WF(epsilon,nu):
    """
    Implements Eq. 9 from Wallman and Flammia (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    return (1./(1-epsilon))**((1.-epsilon)/(nu+1.)) * (float(nu)/(nu+epsilon))**((float(nu)+epsilon)/(nu+1.))

def _sigma_m_squared_base_WF(m,r):
    """
    Implements Eq. 6 (ignoring higher order terms) from Wallman and Flammia (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    return m**2 * r**2 + 7./4 * m * r**2

def _K_WF(epsilon,delta,m,r,sigma_m_squared_func=_sigma_m_squared_base_WF):
    """
    Implements Eq. 10 (rounding up) from Wallman and Flammia (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).
    """
    sigma_m_squared = sigma_m_squared_func(m,r)
    return int(_np.ceil(-_np.log(2./delta) / _np.log(_H_WF(epsilon,sigma_m_squared))))


def rb_decay_WF(m,A,B,f):#Taken from Wallman and Flammia- Eq. 1
    """
    Computes the survival probability function F = A + B * f^m, as provided in Equation 1
    of "Randomized benchmarking with confidence" 
    (http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032).

    Parameters
    ----------
    m : integer
        RB sequence length minus one
    
    A : float
    
    B : float
    
    f : float
    Returns
    ----------
    float
    """
    return A+B*f**m

#def rb_decay_MGE(m,A,B,C,...)

def make_K_m_sched(m_min,m_max,Delta_m,epsilon,delta,r_0,sigma_m_squared_func=_sigma_m_squared_base_WF):
    """
    Computes a "K_m" schedule, that is, how many sequences of Clifford length m
    should be sampled over a range of m values, given certain precision specifications.
    For further discussion of the epsilon, delta, r, and sigma_m_squared_func
    parameters, see http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032,
    referred herein as "W&F".
    
    Parameters
    ----------
    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.
        
    epsilon : float
        Specifies desired confidence interval half-width for each average survival
        probability estimate \hat{F}_m (See W&F Eq. 8).  
        E.g., epsilon = 0.01 means that the confidence interval width for \hat{F}_m  
        is 0.02.  The smaller epsilon is, the larger each value of K_m will be.

    delta : float
        Specifies desired confidence level for confidence interval specified by epsilon.
        delta = 1-0.6827 corresponds to a confidence level of 1 sigma.  This value
        should be used if W&F-derived error bars are desired.
        (See W&F Eq. 8).  The smaller delta is, the larger each value of K_m will be.

    r_0 : float
        Estimate of upper bound of the RB number for the system in question.  
        The smaller r is, the smaller each value of K_m will be.  However, if the 
        system's actual RB number is larger than r_0, then the W&F-derived error bars
        cannot be assumed to be valid.  Additionally, it is assumed that m*r_0 << 1.
    
    sigma_m_squared_func : function
        Function used to serve as the rough upper bound on the variance of \hat{F}_m.
        Default is _sigma_m_squared_base_WF, which implements Eq. 6 of W&F (ignoring 
        higher order terms).

    Returns
    ----------
    K_m_sched : OrderedDict
        An ordered dictionary whose keys are Clifford sequence lengths m and whose values
        are number of Clifford sequences of length m to sample 
        (determined by _K_WF(m,epsilon,delta,r_0)).
    """
    K_m_sched = OrderedDict()
    for m in range(m_min,m_max+1,Delta_m):
        K_m_sched[m] = _K_WF(epsilon,delta,m,r_0,sigma_m_squared_func=sigma_m_squared_func)
    return K_m_sched

def f_to_F_avg(f,d=2):
    """
    Following Wallman and Flammia Eq. 2, maps fit decay fit parameter f to F_avg, that is,
    the average gate fidelity of a noise channel \mathcal{E} with respect to the identity
    channel (see W&F Eq.3).
    
    Parameters
    ----------
    f : float
        Fit parameter f from \bar{F}_m = A + B*f**m.
    
    d : int
        Number of dimensions of the Hilbert space (default is 2, corresponding to a single
        qubit).     
     
    Returns
    ----------
    F_avg : float
        Average gate fidelity F_avg(\mathcal{E}) = \int(d\psi Tr[\psi \mathcal{E}(\psi)]),
        where d\psi is the uniform measure over all pure states (see W&F Eq. 3).
    
    """
    F_avg = ((d-1)*f+1.)/d
    return F_avg

def f_to_r(f,d=2):
    """
    Following Wallman and Flammia, maps fit decay fit parameter f to r, the "average
    gate infidelity".  This quantity is what is often referred to as "the RB number". 
    
    Parameters
    ----------
    f : float
        Fit parameter f from \bar{F}_m = A + B*f**m.
    
    d : int
        Number of dimensions of the Hilbert space (default is 2, corresponding to a single
        qubit).     
     
    Returns
    ----------
    r : float
        The average gate infidelity, that is, "the RB number".      
    
    """
    r = 1 - f_to_F_avg(f,d=d)
    return r

def cliff_twirl(M):
    """
    Returns the Clifford twirl of a map M:  
    Twirl(M) = 1/|Clifford group| * Sum_{C in Clifford group} (C^-1 * M * C)
    
    *At present only works for single-qubit Clifford group.*
    Parameters
    ----------
    M : array or gate
        The CPTP map to be twirled.
    
    Returns
    ----------
    M_twirl : array
        The Clifford twirl of M.
    """
    if M.shape == (4,4):
        M_twirl = 1./len(CliffMatD) * _np.sum(_np.dot(_np.dot(CliffMatInvD[i],M),CliffMatD[i]) for i in range(24))
        return M_twirl
    else:
        raise ValueError("Clifford twirl for non-single qubit Clifford groups not yet implemented!")

def make_real_cliffsD(gs_real,primD):
    """
    Deprecated.  See makeRealCliffs_gs instead.
    """
    CliffRealMatsD = {}
    for i in range(24):
        gatestr = []
        for gate in CliffD[i]:
            gatestr += primD[gate]
        CliffRealMatsD[i] = gs_real.product(gatestr)
    return CliffRealMatsD

def make_real_cliffs_gs(gs_real,primD):
    """
    Turns a "real" (non-perfect) gate set into a "real" (non-perfect) Clifford gate set. 
    *At present only works for single-qubit Clifford group.*

    Parameters
    ----------
    gs_real : gate set
        A "real" (non-ideal) gate set.
    
    primD : dictionary
        A primitives dictionary, mapping the "canonical gate set" {I, X(pi/2), X(-pi/2),
        X(pi), Y(pi/2), Y(-pi/2), Y(pi)} to the gate set that is the target gate set for
        gs_real.
    
    Returns
    ----------
    gs_real_cliffs : gate set
        A gate set of imperfect Cliffords; each Clifford is constructed out of the gates
        contained in gs_real.
    """
    gs_real_cliffs = pygsti.construction.build_gateset([2],[('Q0',)], [],
                                [],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["1"],
                                 spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )
    for i in range(24):
        gatestr = []
        for gate in CliffD[i]:
            gatestr += primD[gate]
        gs_real_cliffs.gates['Gc'+str(i)] = pygsti.objects.FullyParameterizedGate(gs_real.product(gatestr))
    return gs_real_cliffs

#    gatestr += subgatestr
#cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]        


def analytic_rb_gate_error_rate(actual, target):
    """
    Computes the twirled Clifford error rate for a given gate.
    *At present only works for single-qubit gates.*

    Parameters
    ----------
    actual : array or gate
        The noisy gate whose twirled Clifford error rate is to be computed.
        
    target : array or gate
        The target gate against which "actual" is being compared.
    
    Returns
    ----------
    error_rate : float
        The twirled Clifford error rate.
    """
    
    twirled_channel = cliff_twirl(_np.dot(actual,_np.linalg.matrix_power(target,-1)))
    error_rate = 0.5 * (1 - 1./3 * (_np.trace(twirled_channel) - 1))
    return error_rate

def analytic_rb_cliff_gateset_error_rate(gs_real_cliffs):
    """
    Computes the average twirled Clifford error rate for a noisy Clifford gate set.
    This is, analytically, "the RB number".
    *At present only works for single-qubit gate sets.*    
    
    Parameters
    ----------
    
    gs_real_cliffs : gate set
        A gate set of noisy Clifford gates.  If the experimental gate set is, as is typical,
        not a Clifford gate set, then said gate set should be converted to a Clifford gate set 
        using makeRealCliffs_gs.

    Returns
    ----------
    r_analytic : float
        The average per-Clifford error rate of the noisy Clifford gate set.  This is,
        analytically, "the RB number".
    
    """
    error_list = []
    for gate in list(gs_cliff_generic_1q.gates.keys()):
        error_list.append(analytic_rb_gate_error_rate(gs_real_cliffs[gate],gs_cliff_generic_1q[gate]))
    r_analytic = _np.mean(error_list)
    return r_analytic


class rb_results():
    """
    This is the RB object which takes in the RB data set along with other parameters
    and can then process the RB data in a variety of manners, including providing plots.
    As in other docstrings, "W&F" refers to Wallman and Flammia's 
    http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032.
    
    Parameters
    ----------
    dataset : dataset
        The dataset which contains all the experimental RB data.
    
    prim_seq_list : list
        The list of gate sequences used for RB, where each gate is a "primitive",
        that is, a basic physical operation.  (The primitives are frequently
        {I, X(pi/2), Y(pi/2)}).
    
    cliff_len_list : list
        A list declaring how long each sequence in prim_seq_length is, in terms of Clifford
        operations.  Without this list, it is impossible to analyze RB data correctly.
    
    d : int
        Hilbert space dimension.  Default is 2 (corresponding to a single qubit).
    
    prim_dict : dictionary
        A primitives dictionary, mapping the "canonical gate set" {I, X(pi/2), X(-pi/2),
        X(pi), Y(pi/2), Y(-pi/2), Y(pi)} to the gate set of primitives (physical operations).
    
    pre_avg : bool
        Whether or not survival probabilities for different sequences of the same length
        are to be averaged together before curve fitting is performed.  Some information
        is lost when performing pre-averaging, but it follows the literature.
    
    epsilon : float
        Specifies desired confidence interval half-width for each average survival
        probability estimate \hat{F}_m (See W&F Eq. 8).  
        E.g., epsilon = 0.01 means that the confidence interval width for \hat{F}_m  
        is 0.02.  Only to be specified if W&F error bars are desired.
        See make_K_m_sched for further details.

    delta : float
        Specifies desired confidence level for confidence interval specified by epsilon.
        delta = 1-0.6827 corresponds to a confidence level of 1 sigma.  This value
        should be used if W&F-derived error bars are desired.
        (See W&F Eq. 8).  The smaller delta is, the larger each value of K_m will be.
        Only to be specified if W&F error bars are desired. See make_K_m_sched for further details.

    r_0 : float
        Estimate of upper bound of the RB number for the system in question.  
        The smaller r is, the smaller each value of K_m will be.  However, if the 
        system's actual RB number is larger than r_0, then the W&F-derived error bars
        cannot be assumed to be valid.  Additionally, it is assumed that m_max*r_0 << 1.
        Only to be specified if W&F error bars are desired.  See make_K_m_sched for further details.
        

    """
    def __init__(self,dataset,prim_seq_list,cliff_len_list,d=2,prim_dict=None,pre_avg = True, epsilon=None,delta=None,r_0=None):
        """
        Set initial attribute values.
        """
        self.dataset = dataset
        self.prim_seq_list_orig = prim_seq_list
        self.cliff_len_list_orig = cliff_len_list
        
        self.prim_seq_list = prim_seq_list
        self.prim_len_list = list(map(len,prim_seq_list))
        self.cliff_len_list = cliff_len_list
#        self.successes = successes
        self.prim_analyzed = False
        self.cliff_analyzed = False
        self.prim_dict = prim_dict
#        self.data_parsed = False

        self.d = d

        self.epsilon = epsilon
        self.delta = delta
        self.r_0 = r_0

        self.pre_avg = pre_avg
        
        self.bootstrap = False
#        self.p0 = p0
    def parse_data(self):
        """
        Process RB data into survival probabilities along with new sequence length lists
        (if needed due to pre-averaging).
        """
        prim_len_list = []
        successes = []
        N_list = []
        for seq_num, seq in enumerate(self.prim_seq_list):
            data_line = self.dataset[seq]
            plus = data_line['plus']
            minus = data_line['minus']
            N = plus + minus
            prim_length = len(seq)
            prim_len_list.append(prim_length)
            seq_success_prob = 1 - plus/float(N)
            successes.append(seq_success_prob)
            N_list.append(N)
            if seq_success_prob < 0:
                raise ValueError('Survival probability less than 0!')

        if self.pre_avg:
            cliff_zip = list(zip(self.cliff_len_list,successes,N_list))
            cliff_zip = sorted(cliff_zip,key=lambda x: x[0])
            #cliff_zip = _np.array(cliff_zip,dtype=[('length',int),('F',float),('N',float)])
            #cliff_zip = _np.sort(cliff_zip,order='length')
            cliff_avg = []
            cliff_avg_len_list = []
            total_N_list = []
            total_N = 0
            current_len = 0
            total = 0
            total_seqs = 0
            for i in range(len(cliff_zip)):
                tup = cliff_zip[i]
                if tup[0] != current_len:
                    if current_len != 0:
                        cliff_avg_len_list.append(current_len)
                        cliff_avg.append(float(total) / total_seqs)
                        total_N_list.append(total_N)
                    current_len = tup[0]
                    total = 0
                    total_seqs = 0
                    total_N = 0
                total += tup[1]
                total_N += tup[2]
                total_seqs += 1

            self.total_N_list = _np.array(total_N_list)

            prim_avg = []
            prim_avg_len_list = []
            current_len = 0
            total = 0
            total_seqs = 0

            prim_zip = list(zip(prim_len_list,successes))

            prim_zip = list(zip(self.prim_len_list,successes,N_list))
            prim_zip = sorted(prim_zip,key=lambda x: x[0])
#            prim_zip = _np.array(prim_zip,dtype=[('length',int),('F',float),('N',float)])
#            prim_zip = _np.sort(prim_zip,order='length')

            for i in range(len(cliff_zip)):
                tup = prim_zip[i]
                if tup[0] != current_len:
                    if current_len != 0:
                        prim_avg_len_list.append(current_len)
                        prim_avg.append(float(total) / total_seqs)
                    current_len = tup[0]
                    total = 0
                    total_seqs = 0
                total += tup[1]
                total_seqs += 1

            self.cliff_len_list = cliff_avg_len_list
            self.cliff_successes = cliff_avg

            self.prim_len_list = prim_avg_len_list
            self.prim_successes = prim_avg            
        else:
            self.prim_successes = successes
            self.cliff_successes = successes

#        self.successes = successes
#        self.prim_len_list = prim_len_list
#        self.data_parsed = True
#    def parse_data_preavg(self):
#        if not self.data_parsed:
#            self.parse_data()


    def analyze_data(self,rb_decay_func = rb_decay_WF,process_prim = False,process_cliff = False, f0 = [0.98],AB0=[0.5,0.5]):#[0.5,0.5,0.98]):
        """
        Analyze RB data to compute fit parameters and in turn the RB error rate.
        """

        if process_prim:
            xdata = self.prim_len_list
            ydata = self.prim_successes
            def obj_func_full(params):
                A,B,f = params
                val = _np.sum((A+B*f**xdata-ydata)**2)
                return val
            def obj_func_1d(f):
                A = 0.5
                B = 0.5
                val = obj_func_full([A,B,f])
                return val
            self.prim_initial_soln = _minimize(obj_func_1d,f0,method='L-BFGS-B',bounds=[(0.,1.)])
            f1 = [self.prim_initial_soln.x[0]]
            p0 = AB0 + f1
            self.prim_end_soln = _minimize(obj_func_full,p0,method='L-BFGS-B',bounds=[(0.,1.),(0.,1.),(0.,1.)])
            A,B,f = self.prim_end_soln.x
#            results = _curve_fit(rb_decay_func,self.prim_len_list,self.prim_successes,p0 = p0)
#            A,B,f = results[0]
#            cov = results[1]
            self.prim_A = A
            self.prim_B = B
            self.prim_f = f
#            self.prim_cov = cov
            self.prim_F_avg = f_to_F_avg(self.prim_f)
            self.prim_r = f_to_r(self.prim_f)
            self.prim_analyzed = True
        if process_cliff:
            xdata = self.cliff_len_list
            ydata = self.cliff_successes
            def obj_func_full(params):
                A,B,f = params
                val = _np.sum((A+B*f**xdata-ydata)**2)
                return val
            def obj_func_1d(f):
                A = 0.5
                B = 0.5
                val = obj_func_full([A,B,f])
                return val
            self.cliff_initial_soln = _minimize(obj_func_1d,f0,method='L-BFGS-B',bounds=[(0.,1.)])
            f0 = self.cliff_initial_soln.x[0]
            p0 = AB0 + [f0]
            self.cliff_end_soln = _minimize(obj_func_full,p0,method='L-BFGS-B',bounds=[(0.,1.),(0.0,1.),(0.,1.)])
            A,B,f = self.cliff_end_soln.x
#            results = _curve_fit(rb_decay_func,self.cliff_len_list,self.cliff_successes,p0 = p0)
#            A,B,f = results[0]
#            cov = results[1]
            self.cliff_A = A
            self.cliff_B = B
            self.cliff_f = f
#            self.cliff_cov = cov
            self.cliff_F_avg = f_to_F_avg(self.cliff_f)
            self.cliff_r = f_to_r(self.cliff_f)
            self.cliff_analyzed = True

    def print_cliff(self):
        """
        Display per Clifford gate RB error rate.
        """
        if self.cliff_analyzed:
            print("For Cliffords:")
            print("A =", self.cliff_A)
            print("B =", self.cliff_B)
            print("f =", self.cliff_f)
            print("F_avg =", self.cliff_F_avg)
            print("r =", self.cliff_r)
        else:
            print("No Clifford analysis performed!")
    def print_prim(self):
        """
        Display per primitive gate RB error rate.  These physical interpretation of these
        numbers may not be reliable; per Clifford error rates are recommended instead. 
        """
        if self.prim_analyzed:
            print("For primitives:")
            print("A =", self.prim_A)
            print("B =", self.prim_B)
            print("f =", self.prim_f)
            print("F_avg =", self.prim_F_avg)
            print("r =", self.prim_r)
        else:
            print("No primimitives analysis performed!")
    def plot(self,prim_or_cliff,xlim=None,ylim=None,save_fig_path=None):
        """
        Plot RB decay curve, either as a function of primitive sequence length
        or Clifford sequence length.
        """
        assert prim_or_cliff in ['prim', 'cliff'], 'Please provide a valid quantity to plot.'

        newplot = _plt.figure(figsize=(8, 4))
        newplotgca = newplot.gca()
        if prim_or_cliff == 'prim':
            xdata = self.prim_len_list
            ydata = self.prim_successes
            try:
                A = self.prim_A
                B = self.prim_B
                f = self.prim_f
            except:
                print("Prim data not found.")
            xlabel = 'RB sequence length (primitives)'
        else:
            xdata = self.cliff_len_list
            ydata = self.cliff_successes
            try:
                A = self.cliff_A
                B = self.cliff_B
                f = self.cliff_f
            except:
                print("Clifford data not found.")
            xlabel = 'RB sequence length (Cliffords)'
        cmap = _plt.cm.get_cmap('Set1')
        #        try:
        newplotgca.plot(xdata,ydata,'.', markersize=15, clip_on=False, color=cmap(30))
        newplotgca.plot(_np.arange(max(xdata)),
                    rb_decay_WF(_np.arange(max(xdata)),A,B,f),'-', lw=2, color=cmap(110))
        #        except:
        #            print "Unable to plot fit.  Have you run the RB analysis?"

        newplotgca.set_xlabel(xlabel, fontsize=15)
        newplotgca.set_ylabel('Success Rate',fontsize=15)
        newplotgca.set_title('RB Success Curve', fontsize=20)

        newplotgca.set_frame_on(False)
        newplotgca.yaxis.grid(True)
        newplotgca.tick_params(axis='x', top='off', labelsize=12)
        newplotgca.tick_params(axis='y', left='off', right='off', labelsize=12)

        if xlim:
            _plt.xlim(xlim)
        if ylim:
            _plt.ylim(ylim)
        if save_fig_path:
            newplot.savefig(save_fig_path)
    
    def generate_error_bars(self,method,process_prim=False,process_cliff=False,bootstrap_resamples = 100,p0 = [0.5,0.5,0.98],seed=None):
        """
        Compute error bars on RB fit parameters, including the RB decay rate.
        Error bars can be computed either using a non-parametric bootstrap, or quasi-analytic
        methods provided in W&F.
        *At present, only the bootstrap method is fully supported.*
        """
        assert method in ['bootstrap','analytic']

        if method=='bootstrap':
            if self.bootstrap == False:
                if seed:
                    _np.random.seed(seed)
                bootstrapped_dataset_list = []
                if process_prim:
                    prim_A_list = []
                    prim_B_list = []
                    prim_f_list = []
                if process_cliff:
                    cliff_A_list = []
                    cliff_B_list = []
                    cliff_f_list = []
                for resample in range(bootstrap_resamples):
                    bootstrapped_dataset_list.append(pygsti.bootstrap.make_bootstrap_dataset(self.dataset,'nonparametric'))
                for resample in range(bootstrap_resamples):
                    temp_rb_results = process_rb_data(bootstrapped_dataset_list[resample],self.prim_seq_list_orig,self.cliff_len_list_orig,prim_dict = self.prim_dict,pre_avg=self.pre_avg,process_prim=process_prim,process_cliff=process_cliff,f0 = [0.98],AB0=[0.5,0.5])
                    if process_prim:
                        prim_A_list.append(temp_rb_results.prim_A)
                        prim_B_list.append(temp_rb_results.prim_B)
                        prim_f_list.append(temp_rb_results.prim_f)
                    if process_cliff:
                        cliff_A_list.append(temp_rb_results.cliff_A)
                        cliff_B_list.append(temp_rb_results.cliff_B)
                        cliff_f_list.append(temp_rb_results.cliff_f)
            if process_prim:
                if self.bootstrap == False:
                    self.prim_A_error_BS = _np.std(prim_A_list,ddof=1)
                    self.prim_B_error_BS = _np.std(prim_B_list,ddof=1)
                    self.prim_f_error_BS = _np.std(prim_f_list,ddof=1)

                    self.prim_F_avg_error_BS = (self.d-1.)/self.d * self.prim_f_error_BS
                    self.prim_r_error_BS = self.prim_F_avg_error_BS
                    
#                print "prim A error =", self.prim_A_error_BS
#                print "prim B error =", self.prim_B_error_BS
#                print "prim f error =", self.prim_f_error_BS
#                print "prim A =", self.prim_A, "+/-", self.prim_A_error_BS
#                print "prim B =", self.prim_B, "+/-", self.prim_B_error_BS
#                print "prim f =", self.prim_f, "+/-", self.prim_f_error_BS
#                print "prim F_avg =", self.prim_F_avg, "+/-", self.prim_F_avg_error_BS
#                print "prim r =", self.prim_r, "+/-", self.prim_r_error_BS

            if process_cliff:
                if self.bootstrap == False:
                    self.cliff_A_error_BS = _np.std(cliff_A_list,ddof=1)
                    self.cliff_B_error_BS = _np.std(cliff_B_list,ddof=1)
                    self.cliff_f_error_BS = _np.std(cliff_f_list,ddof=1)

                    self.cliff_F_avg_error_BS = (self.d-1.)/self.d * self.cliff_f_error_BS
                    self.cliff_r_error_BS = self.cliff_F_avg_error_BS

#                print "cliff A error =", self.cliff_A_error_BS
#                print "cliff B error =", self.cliff_B_error_BS
#                print "cliff f error =", self.cliff_f_error_BS
#                print "Cliff A =", self.cliff_A, "+/-", self.cliff_A_error_BS
#                print "Cliff B =", self.cliff_B, "+/-", self.cliff_B_error_BS
#                print "Cliff f =", self.cliff_f, "+/-", self.cliff_f_error_BS
#                print "Cliff F_avg =", self.cliff_F_avg, "+/-", self.cliff_F_avg_error_BS
#                print "Cliff r =", self.cliff_r, "+/-", self.cliff_r_error_BS


            self.bootstrap = True

        elif method=='analytic':
            print("WARNING: ANALYTIC BOOSTRAP ERROR BAR METHOD NOT YET GUARANTEED TO BE STABLE.")
            print('Processesing analytic bootstrap, following Wallman and Flammia.')
            print('The error bars are reliable if and only if the schedule for K_m')
            print('has been chosen appropriately, given:')
            print('delta =',self.delta)
            print('epsilon =',self.epsilon)
            print('r_0 =',self.r_0)
            
            self.sigma_list = _np.sqrt(self.epsilon**2 + 1./self.total_N_list)
            results = _curve_fit(rb_decay_WF,self.cliff_len_list,self.cliff_successes,p0 = p0,sigma = self.sigma_list)
            self.full_results = results
            self.cliff_A_error_WF = _np.sqrt(results[1][0,0])
            self.cliff_B_error_WF = _np.sqrt(results[1][1,1])
            self.cliff_f_error_WF = _np.sqrt(results[1][2,2])

            self.cliff_F_avg_error_WF = (self.d-1.)/self.d * self.cliff_f_error_WF
            self.cliff_r_error_WF = self.cliff_F_avg_error_WF

        print("Error bars computed.  Use error_bars method to access.")

    def print_error_bars(self,method,process_prim=False,process_cliff=False):
        assert method in ['bootstrap', 'analytic']
        if method == 'bootstrap':
            if not self.bootstrap:
                raise ValueError('Bootstrapped error bars requested but not yet generated; use generate_error_bars method first.')
            print("Results with boostrapped-derived error bars (1 sigma):")
            if process_prim:
                print("prim A =", self.prim_A, "+/-", self.prim_A_error_BS)
                print("prim B =", self.prim_B, "+/-", self.prim_B_error_BS)
                print("prim f =", self.prim_f, "+/-", self.prim_f_error_BS)
                print("prim F_avg =", self.prim_F_avg, "+/-", self.prim_F_avg_error_BS)
                print("prim r =", self.prim_r, "+/-", self.prim_r_error_BS)
            if process_cliff:
                print("Cliff A =", self.cliff_A, "+/-", self.cliff_A_error_BS)
                print("Cliff B =", self.cliff_B, "+/-", self.cliff_B_error_BS)
                print("Cliff f =", self.cliff_f, "+/-", self.cliff_f_error_BS)
                print("Cliff F_avg =", self.cliff_F_avg, "+/-", self.cliff_F_avg_error_BS)
                print("Cliff r =", self.cliff_r, "+/-", self.cliff_r_error_BS)
        elif method == 'analytic':
                print("Results with Wallman and Flammia-derived error bars (1 sigma):")
                print("Cliff A =", self.cliff_A, "+/-", self.cliff_A_error_WF)
                print("Cliff B =", self.cliff_B, "+/-", self.cliff_B_error_WF)
                print("Cliff f =", self.cliff_f, "+/-", self.cliff_f_error_WF)
                print("Cliff F_avg =", self.cliff_F_avg, "+/-", self.cliff_F_avg_error_WF)
                print("Cliff r =", self.cliff_r, "+/-", self.cliff_r_error_WF)
                if self.cliff_r - self.cliff_r_error_WF > self.r_0:
                    print("r is bigger than r_0 + sigma_r, so above error bars should not be trusted.")
                    
gs_cliff_canon = pygsti.construction.build_gateset([2],[('Q0',)], ['Gi','Gxp2','Gxp','Gxmp2','Gyp2','Gyp','Gymp2'], 
                                [ "I(Q0)","X(pi/2,Q0)", "X(pi,Q0)", "X(-pi/2,Q0)",
                                          "Y(pi/2,Q0)", "Y(pi,Q0)", "Y(-pi/2,Q0)"],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["1"], 
                                 spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )


CliffD = {}
#Definitions taken from arXiv:1508.06676v1
#0-indexing used instead of 1-indexing

CliffD[0] = ['Gi',]
CliffD[1] = ['Gyp2','Gxp2']
CliffD[2] = ['Gxmp2','Gymp2']
CliffD[3] = ['Gxp',]
CliffD[4] = ['Gymp2','Gxmp2']
CliffD[5] = ['Gxp2','Gymp2']
CliffD[6] = ['Gyp',]
CliffD[7] = ['Gymp2','Gxp2']
CliffD[8] = ['Gxp2','Gyp2']
CliffD[9] = ['Gxp','Gyp']
CliffD[10] = ['Gyp2','Gxmp2']
CliffD[11] = ['Gxmp2','Gyp2']
CliffD[12] = ['Gyp2','Gxp']
CliffD[13] = ['Gxmp2']
CliffD[14] = ['Gxp2','Gymp2','Gxmp2']
CliffD[15] = ['Gymp2']
CliffD[16] = ['Gxp2']
CliffD[17] = ['Gxp2','Gyp2','Gxp2']
CliffD[18] = ['Gymp2','Gxp']
CliffD[19] = ['Gxp2','Gyp']
CliffD[20] = ['Gxp2','Gymp2','Gxp2']
CliffD[21] = ['Gyp2']
CliffD[22] = ['Gxmp2','Gyp']
CliffD[23] = ['Gxp2','Gyp2','Gxmp2']

CliffMatD = {}
CliffMatInvD = {}
for i in range(24):
    CliffMatD[i] = gs_cliff_canon.product(CliffD[i])
    CliffMatInvD[i] = _np.linalg.matrix_power(CliffMatD[i],-1)

gs_cliff_generic_1q = pygsti.construction.build_gateset([2],[('Q0',)], [],
                                [],
                                 prepLabels=["rho0"], prepExpressions=["0"],
                                 effectLabels=["E0"], effectExpressions=["1"],
                                 spamdefs={'plus': ('rho0','E0'), 'minus': ('rho0','remainder') } )

for i in list(CliffMatD.keys()):
    gate_lbl = 'Gc'+str(i)
    gs_cliff_generic_1q.gates[gate_lbl] = pygsti.objects.FullyParameterizedGate(CliffMatD[i])

def makeCliffGroupTable():
    CliffGroupTable = _np.zeros([24,24], dtype=int)
    counter = 0
    for i in range(24):
        for j in range(24):
            test = _np.dot(CliffMatD[j],CliffMatD[i])#Want reverse order for multiplication here because gates are applied left to right.
            for k in range(24):
                diff = _np.linalg.norm(test-CliffMatD[k])
                if diff < 10**-10:
                    CliffGroupTable[i,j]=k
                    counter += 1
                    break
    if counter!=24**2:
        raise Exception('Error!')
    return CliffGroupTable

CliffGroupTable = makeCliffGroupTable()

CliffInvTable = {}
for i in range(24):
    for j in range(24):
        if CliffGroupTable[i,j] == 0:
            CliffInvTable[i] = j
            
def lookup_cliff_prod(i,j):
    """
    Auxiliary function for looking up the product of two Cliffords.
    """
    return CliffGroupTable[i,j]

def make_random_RB_cliff_string(m,seed=None):
    """
    Generate a random RB sequence.
    
    Parameters
    ----------
    m : int
        Sequence length is m+1 (because m Cliffords are chosen at random, then one additional
        Clifford is selected to invert the sequence).
    
    seed : int, optional
        Seed for the random number generator.
    
    Returns
    ----------
    cliff_string : list
        Random Clifford sequence of length m+1.  For ideal Cliffords, the sequence
        implements the identity operation.
    """
    if seed:
        _np.random.seed()
    cliff_string = _np.random.randint(0,24,m)
    effective_cliff = reduce(lookup_cliff_prod,cliff_string)
    cliff_inv = CliffInvTable[effective_cliff]
    cliff_string = _np.append(cliff_string,cliff_inv)
    return cliff_string

def make_random_RB_cliff_string_lists(m_min,m_max,Delta_m,K_m_sched,generic_or_canonical_or_primitive,primD=None,seed=None):
    """
    Makes a list of random RB sequences.
    
    Parameters
    ----------
    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.

    K_m_sched : int or OrderedDict
        If an integer, the fixed number of Clifford sequences to be sampled at each length m.
        If an OrderedDict, then a mapping from Clifford sequence length m to number of 
        Cliffords to be sampled at that length.
    
    generic_or_canonical_or_primitive : string
        What kind of gate set should the selected gate sequences be expressed as:
        "generic" : Clifford gates are used, with labels "Gc0" through "Gc23".
        "canonical" : The "canonical" gate set is used (so called because of its 
            abundance in the literature for describing Clifford operations).  This gate set 
            contains the gates {I, X(pi/2), X(-pi/2), X(pi), Y(pi/2), Y(-pi/2), Y(pi)}
        "primitive" : A gate set is used which is neither "generic" nor "canonical".  E.g.,
            {I, X(pi/2), Y(pi/2)}.  In this case, primD must be specified.
    
    primD : dictionary     
        A primitives dictionary, mapping the "canonical gate set" {I, X(pi/2), X(-pi/2),
        X(pi), Y(pi/2), Y(-pi/2), Y(pi)} to the gate set of primitives whose gate labels 
        are to be used in the generated RB sequences.
    
    seed : int
        Seed for random number generator; optional.
    
    Returns
    -----------
    cliff_string_list : list
        List of gate strings; each gate string is an RB experiment.
    
    cliff_len_list : list
        List of Clifford lengths for cliff_string_list.  cliff_len_list[i] is the number of
        Clifford operations selected for the creation of cliff_string_list[i].
    """
    if seed is not None:
        _np.random.seed(seed)
    cliff_string_list = []
    if not isinstance(K_m_sched,OrderedDict):
        print("K_m_sched is not an OrderedDict, so Wallman and Flammia error bars are not valid.")
        if not isinstance(K_m_sched,int):
            raise ValueError('K_m_sched must be an OrderedDict or an int!')
        K_m_sched_dict = OrderedDict()
        for m in range(m_min, m_max+1,Delta_m):
            K_m_sched_dict[m] = K_m_sched
    else:
        K_m_sched_dict = K_m_sched
    for m in range(m_min,m_max+1,Delta_m):
        temp_list = []
        K_m = K_m_sched_dict[m]
        for i in range(K_m):
            temp_list.append(tuple(make_random_RB_cliff_string(m).tolist()))
#        temp_list = pygsti.listtools.remove_duplicates(temp_list)
#        print len(temp_list)
        cliff_string_list += temp_list
    cliff_len_list = list(map(len,cliff_string_list))
    if generic_or_canonical_or_primitive == 'generic':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            cliff_string_list[cliff_tup_num] = ['Gc'+str(i) for i in cliff_tup]
    elif generic_or_canonical_or_primitive == 'canonical':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            gatestr = []
            for cliff in cliff_tup:
                gatestr += CliffD[cliff]
            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]
    elif generic_or_canonical_or_primitive == 'primitive':
        for cliff_tup_num, cliff_tup in enumerate(cliff_string_list):
            gatestr = []
            for cliff in cliff_tup:
                subgatestr = []
                for gate in CliffD[cliff]:
                    subgatestr += primD[gate]
                gatestr += subgatestr
            cliff_string_list[cliff_tup_num] = [str(i) for i in gatestr]        
    else:
        raise ValueError('generic_or_canonical_or_primitive must be "generic" or "canonical" or "primitive"!')
    cliff_string_list =  pygsti.construction.gatestring_list(cliff_string_list)
    return cliff_string_list, cliff_len_list

def write_empty_rb_files(filename,m_min,m_max,Delta_m,K_m,generic_or_canonical_or_primitive,primD=None,seed=None):
    """
    Wrapper for make_random_RB_cliff_string_lists.  Functionality is same as 
    random_RB_cliff_string_lists, except that both an empty data template file is written
    to disk as is the list recording the Clifford length of each gate sequence.
    See docstring for make_random_RB_cliff_string_lists for more details.
    """
    random_RB_cliff_string_lists, cliff_lens = make_random_RB_cliff_string_lists(m_min,m_max,Delta_m,K_m,generic_or_canonical_or_primitive,primD=primD,seed=seed)
    pygsti.io.write_empty_dataset(filename+'.txt',random_RB_cliff_string_lists)
#    seq_len_list = map(len,random_RB_cliff_string_lists)
    temp_file = open(filename+'_cliff_seq_lengths.pkl','w')
    pickle.dump(cliff_lens,temp_file) 
#    for cliff_len in cliff_lens:
#        temp_file.write(str(cliff_len)+'\n')
    temp_file.close()        
    return random_RB_cliff_string_lists, cliff_lens
#Want to keep track of both Clifford sequences and primitive sequences.

def process_rb_data(dataset,prim_seq_list,cliff_len_list,prim_dict=None,pre_avg=True,process_prim=False,process_cliff=False,f0 = [0.98],AB0 = [0.5,0.5]):
    """
    Process RB data, yielding an RB results object containing desired RB quantities.
    See docstring for rb_results for more details.
    """
    results_obj = rb_results(dataset,prim_seq_list,cliff_len_list,prim_dict=prim_dict,pre_avg=pre_avg)
    results_obj.parse_data()
    results_obj.analyze_data(process_prim = process_prim,process_cliff = process_cliff,f0 = f0, AB0 = AB0)
    return results_obj
    
# def rb_decay_WF_rate(dataset,avg_gates_per_cliff=None,showPlot=False,xlim=None,ylim=None,saveFigPath=None,printData=False,p0=[0.5,0.5,0.98]):
#     RBlengths = []
#     RBsuccesses = []
#     for key in list(dataset.keys()):
#         dataLine = dataset[key]
#         plus = dataLine['plus']
#         minus = dataLine['minus']
#         N = plus + minus
#         key_len = len(key)
#         RBlengths.append(key_len)
#         RBsuccessProb=1 - dataLine['plus']/float(N)
#         RBsuccesses.append(RBsuccessProb)
#         if dataLine['plus']/float(N) > 1:
#             print(key)
#         if printData:
#             print(key_len,RBsuccessProb)
#     results = _curve_fit(rb_decay_WF,RBlengths,RBsuccesses,p0=p0)
#     A,B,f = results[0]
#     cov = results[1]
#     if saveFigPath or showPlot:
#         newplot = _plt.figure()
#         newplotgca = newplot.gca()
#         newplotgca.plot(RBlengths,RBsuccesses,'.')
#         newplotgca.plot(range(max(RBlengths)),
#                         rb_decay_WF(_np.arange(max(RBlengths)),A,B,f),'+')
#         newplotgca.set_xlabel('RB sequence length (non-Clifford)')
#         newplotgca.set_ylabel('Success rate')
#         newplotgca.set_title('RB success')
#         if xlim:
#             _plt.xlim(xlim)
#         if ylim:
#             _plt.ylim(ylim)
#     if saveFigPath:
#         newplot.savefig(saveFigPath)
#     print "f (for gates) =",f
#     if avg_gates_per_cliff:
#         print "f (for Cliffords) = f^(avg. gates/Cliffords) =",f**avg_gates_per_cliff
#     return A,B,f,cov
