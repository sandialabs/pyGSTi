from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines Randomized Benhmarking support objects """

from ... import drivers as _drivers
from . import rbutils as _rbutils

import numpy as _np
from numpy import random as _rndm
from functools import reduce as _reduce
from matplotlib import pyplot as _plt
from scipy.optimize import curve_fit as _curve_fit

class MatrixGroup(object):
    def __init__(self, listOfMatrices, labels=None):
        self.mxs = list(listOfMatrices)
        self.labels = list(labels) if (labels is not None) else None
        assert(labels is None or len(labels) == len(listOfMatrices))
        if labels is not None:
            self.label_indices = { lbl:indx for indx,lbl in enumerate(labels)}
        else:
            self.label_indices = None

        N = len(self.mxs)
        if N > 0:
            mxDim = self.mxs[0].shape[0]
            assert(_np.isclose(0,_np.linalg.norm(
                        self.mxs[0] - _np.identity(mxDim)))), \
                        "First element must be the identity matrix!"
        
        #Construct group table
        self.product_table = -1 * _np.ones([N,N], dtype=int)
        for i in range(N):
            for j in range(N):
                ij_product = _np.dot(self.mxs[j],self.mxs[i])
                  #Dot in reverse order here for multiplication here because
                  # gates are applied left to right.

                for k in range(N):
                    if _np.isclose(_np.linalg.norm(ij_product-self.mxs[k]),0):
                        self.product_table[i,j] = k; break
        assert (-1 not in self.product_table), "Cannot construct group table"
        
        #Construct inverse table
        self.inverse_table = -1 * _np.ones(N, dtype=int)
        for i in range(N):
            for j in range(N):
                if self.product_table[i,j] == 0: # the identity
                    self.inverse_table[i] = j; break
        assert (-1 not in self.inverse_table), "Cannot construct inv table"

    def get_matrix(self, i):
        if not isinstance(i,int): i = self.label_indices[i]
        return self.mxs[i]

    def get_matrix_inv(self, i):
        if not isinstance(i,int): i = self.label_indices[i]
        return self.mxs[self.inverse_table[i]]

    def get_inv(self, i):
        if isinstance(i,int):
            return self.inverse_table[i]
        else:
            i = self.label_indices[i]
            return self.labels[ self.inverse_table[i] ]
        
    def product(self, indices):
        if len(indices) == 0: return None
        if isinstance(indices[0],int):
            return _reduce(lambda i,j: self.product_table[i,j], indices)
        else:
            indices = [ self.label_indices[i] for i in indices ]
            fi = _reduce(lambda i,j: self.product_table[i,j], indices)
            return self.labels[fi]

    def __len__(self):
        return len(self.mxs)



class RBResults(object):
    """
    Defines a class which holds an RB data set along with other parameters,
    and can process the RB data in a variety of manners, including the 
    generation of plots.

    As in other docstrings, "W&F" refers to Wallman and Flammia's 
    http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032.
    """
    
    def __init__(self, dataset, result_dicts, basename, alias_maps, 
                 success_spamlabel='plus', dim=2, pre_avg=True, f0=[0.98],
                 AB0=[0.5,0.5]):
        """
        Constructs a new RBResults object.

        Parameters
        ----------
        dataset : dataset
            The dataset which contains all the experimental RB data.
        
        prim_seq_list : list
            The list of gate sequences used for RB, where each gate is a 
            "primitive", that is, a basic physical operation.  (The primitives
            are frequently {I, X(pi/2), Y(pi/2)}).
        
        cliff_len_list : list
            A list declaring how long each sequence in prim_seq_length is, in
            terms of Clifford operations.  Without this list, it is impossible
            to analyze RB data correctly.
        
        d : int, optional
            Hilbert space dimension.  Default is 2, corresponding to a single
            qubit.
        
        prim_dict : dictionary, optional
            A primitives dictionary, mapping the "canonical gate set" 
            {I, X(pi/2), X(-pi/2), X(pi), Y(pi/2), Y(-pi/2), Y(pi)} to the
            gate set of primitives (physical operations).
        
        pre_avg : bool, optional
            Whether or not survival probabilities for different sequences of
            the same length are to be averaged together before curve fitting
            is performed.  Some information is lost when performing
            pre-averaging, but it follows the literature.
        
        epsilon : float, optional
            Specifies desired confidence interval half-width for each average
            survival probability estimate \hat{F}_m (See W&F Eq. 8).  
            E.g., epsilon = 0.01 means that the confidence interval width for
            \hat{F}_m is 0.02.  Only to be specified if W&F error bars are
            desired.  See make_K_m_sched for further details.
    
        delta : float, optional
            Specifies desired confidence level for confidence interval
            specified by epsilon.  delta = 1-0.6827 corresponds to a
            confidence level of 1 sigma.  This value should be used if 
            W&F-derived error bars are desired.  (See W&F Eq. 8).  The smaller
            delta is, the larger each value of K_m will be.  Only to be
            specified if W&F error bars are desired. See make_K_m_sched for 
            further details.
    
        r_0 : float, optional
            Estimate of upper bound of the RB number for the system in 
            question. The smaller r is, the smaller each value of K_m will be.
            However, if the system's actual RB number is larger than r_0, then
            the W&F-derived error bars cannot be assumed to be valid.  Addition
            ally, it is assumed that m_max*r_0 << 1.  Only to be specified if
            W&F error bars are desired. See make_K_m_sched for further details.
        """
        self.dataset = dataset
        self.dicts = result_dicts
        self.basename = basename
        self.alias_maps = alias_maps
        self.d = dim
        self.pre_avg = pre_avg
        self.success_spamlabel = success_spamlabel
        self.f0 = f0
        self.AB0 = AB0

    def detail_str(self, gstyp):
        """
        Display the per-"gatestring type" RB error rate.
        """
        s = ""
        key_list = ['A','B','f','F_avg','r']
        if gstyp in self.dicts and self.dicts[gstyp] is not None:            
            #print("For %ss:" % gstyp)
            s += "%s results\n" % gstyp
            if 'A_error_BS' in self.dicts[gstyp]: 
                s += "  - with boostrapped-derived error bars (1 sigma):\n"
                for key in key_list:
                    s += "%s = %s +/- %s\n" % (key,str(self.dicts[gstyp][key]),
                                    str(self.dicts[gstyp][key + "_error_BS"]))
            if 'A_error_WF' in self.dicts[gstyp]:
                s += "  - with Wallman and Flammia-derived error bars" + \
                    "(1 sigma):\n"
                for key in key_list:
                    s += "%s = %s +/- %s\n" % (key,str(self.dicts[gstyp][key]),
                                    str(self.dicts[gstyp][key + "_error_WF"]))

            if 'A_error_BS' not in self.dicts[gstyp] and \
                    'A_error_WF' not in self.dicts[gstyp]:
                for key in key_list:
                    s += "%s = %s\n" % (key, str(self.dicts[gstyp][key]))
        else:
            s += "No %s analysis performed!\n" % gstyp
        return s

    def print_detail(self, gstyp):
        """
        Display per Clifford gate RB error rate.
        """
        print(self.detail_str(gstyp))

    def print_clifford(self):
        """
        Display per Clifford gate RB error rate.
        """
        self.print_detail('clifford')

    def print_primitive(self):
        """
        Display per primitive gate RB error rate.  These physical
        interpretation of these numbers may not be reliable; per Clifford 
        error rates are recommended instead. 
        """
        self.print_detail('primitive')

    def __str__(self):
        s = ""
        for gstyp in self.dicts:
            s += self.detail_str(gstyp) + "\n"
        return s


    def plot(self,gstyp,xlim=None,ylim=None,save_fig_path=None):
        """
        Plot RB decay curve, either as a function of primitive sequence length
        or Clifford sequence length.

        TODO: docstring describing parameters
        """
        if gstyp not in self.dicts:
            raise ValueError("%s data not found!" % gstyp)

        newplot = _plt.figure(figsize=(8, 4))
        newplotgca = newplot.gca()

        xdata = self.dicts[gstyp]['lengths']
        ydata = self.dicts[gstyp]['successes']
        A = self.dicts[gstyp]['A']
        B = self.dicts[gstyp]['B']
        f = self.dicts[gstyp]['f']
        xlabel = 'RB sequence length (%ss)' % gstyp

        cmap = _plt.cm.get_cmap('Set1')
        newplotgca.plot(xdata,ydata,'.', markersize=15, clip_on=False,
                        color=cmap(30))
        newplotgca.plot(_np.arange(max(xdata)),
                        _rbutils.rb_decay_WF(_np.arange(max(xdata)),A,B,f),
                        '-', lw=2, color=cmap(110))

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
    

    def compute_bootstrap_error_bars(self, gstyp_list, resamples = 100,
                                     p0 = [0.5,0.5,0.98], seed=None,
                                     randState=None):
        """
        Compute error bars on RB fit parameters, including the RB decay rate.
        Error bars can be computed either using a non-parametric bootstrap, 
        or quasi-analytic methods provided in W&F.

        *At present, only the bootstrap method is fully supported.*

        TODO: docstring describing parameters
        """
        if randState is None:
            rndm = _rndm.RandomState(seed) # ok if seed is None
        else:
            rndm = randState

        #Setup lists to hold items to take stddev of:
        A_list = {}; B_list = {}; f_list = {}
        for gstyp in gstyp_list:
            A_list[gstyp] = []; B_list[gstyp] = []; f_list[gstyp] = []

        #Create bootstrap datasets
        bootstrapped_dataset_list = []
        for resample in range(resamples):
            bootstrapped_dataset_list.append(
                _drivers.bootstrap.make_bootstrap_dataset(
                    self.dataset,'nonparametric'))

        #Run RB analysis for each dataset
        base_gatestrings = self.dicts[self.basename]['gatestrings']
        alias_maps = { k:mp for k,mp in self.alias_maps.items()
                       if k in gstyp_list } #only alias maps of requested
        from .rbcore import do_rb_base as _do_rb_base
        for dsBootstrap in bootstrapped_dataset_list:
            resample_results = _do_rb_base(dsBootstrap, base_gatestrings,
                                           self.basename, alias_maps,
                                           self.success_spamlabel, self.d,
                                           self.pre_avg,self.f0, self.AB0)
            for gstyp in gstyp_list:
                A_list[gstyp].append(resample_results.dicts[gstyp]['A'])
                B_list[gstyp].append(resample_results.dicts[gstyp]['B'])
                f_list[gstyp].append(resample_results.dicts[gstyp]['f'])

        for gstyp in gstyp_list:
            self.dicts[gstyp]['A_error_BS'] = _np.std(A_list[gstyp],ddof=1)
            self.dicts[gstyp]['B_error_BS'] = _np.std(B_list[gstyp],ddof=1)
            self.dicts[gstyp]['f_error_BS'] = _np.std(f_list[gstyp],ddof=1)
            self.dicts[gstyp]['F_avg_error_BS'] = (self.d-1.) / self.d \
                                         * self.dicts[gstyp]['f_error_BS']
            self.dicts[gstyp]['r_error_BS'] = \
                self.dicts[gstyp]['F_avg_error_BS']

        print("Bootstrapped error bars computed.  Use print methods to access.")


    def compute_analytic_error_bars(self, epsilon, delta, r_0, 
                                    p0 = [0.5,0.5,0.98]):
        print("WARNING: ANALYTIC BOOSTRAP ERROR BAR METHOD NOT YET" +
              "GUARANTEED TO BE STABLE.")
        print('Processesing analytic bootstrap, following Wallman and' +
              'Flammia.\nThe error bars are reliable if and only if the' +
              'schedule for K_m has been chosen appropriately, given:')
        print('delta =',delta)
        print('epsilon =',epsilon)
        print('r_0 =',r_0)

        gstyp_list = ['clifford'] #KENNY: WF assume clifford-gatestring data (??)
        
        for gstyp in gstyp_list:
            Ns = _np.array(self.dicts[gstyp]['counts'])
            sigma_list = _np.sqrt(epsilon**2 + 1./Ns)
            results = _curve_fit(_rbutils.rb_decay_WF,
                             self.dicts[gstyp]['lengths'],
                             self.dicts[gstyp]['successes'],
                             p0 = p0, 
                             sigma = sigma_list)
        self.dicts[gstyp]['WF fit full results'] = results
        self.dicts[gstyp]['A_error_WF'] = _np.sqrt(results[1][0,0])
        self.dicts[gstyp]['B_error_WF'] = _np.sqrt(results[1][1,1])
        self.dicts[gstyp]['f_error_WF'] = _np.sqrt(results[1][2,2])
        self.dicts[gstyp]['F_avg_error_WF'] = (self.d-1.)/self.d  \
                                   * self.dicts[gstyp]['f_error_WF']
        self.dicts[gstyp]['r_error_WF'] = \
            self.dicts[gstyp]['F_avg_error_WF']

        print("Analytic error bars computed.  Use print methods to access.")



#    def print_error_bars(self,method,process_prim=False,process_cliff=False):
#        """
#        TODO: docstring
#        """
#        assert method in ['bootstrap', 'analytic']
#        if method == 'bootstrap':
#            if not self.bootstrap:
#                raise ValueError('Bootstrapped error bars requested but not' +
#                       'yet generated; use generate_error_bars method first.')
#            print("Results with boostrapped-derived error bars (1 sigma):")
#            if process_prim:
#                print("prim A =", self.prim_A, "+/-", self.prim_A_error_BS)
#                print("prim B =", self.prim_B, "+/-", self.prim_B_error_BS)
#                print("prim f =", self.prim_f, "+/-", self.prim_f_error_BS)
#                print("prim F_avg =", self.prim_F_avg, "+/-", self.prim_F_avg_error_BS)
#                print("prim r =", self.prim_r, "+/-", self.prim_r_error_BS)
#            if process_cliff:
#                print("Cliff A =", self.cliff_A, "+/-", self.cliff_A_error_BS)
#                print("Cliff B =", self.cliff_B, "+/-", self.cliff_B_error_BS)
#                print("Cliff f =", self.cliff_f, "+/-", self.cliff_f_error_BS)
#                print("Cliff F_avg =", self.cliff_F_avg, "+/-", self.cliff_F_avg_error_BS)
#                print("Cliff r =", self.cliff_r, "+/-", self.cliff_r_error_BS)
#        elif method == 'analytic':
#                print("Results with Wallman and Flammia-derived error bars (1 sigma):")
#                print("Cliff A =", self.cliff_A, "+/-", self.cliff_A_error_WF)
#                print("Cliff B =", self.cliff_B, "+/-", self.cliff_B_error_WF)
#                print("Cliff f =", self.cliff_f, "+/-", self.cliff_f_error_WF)
#                print("Cliff F_avg =", self.cliff_F_avg, "+/-", self.cliff_F_avg_error_WF)
#                print("Cliff r =", self.cliff_r, "+/-", self.cliff_r_error_WF)
#                if self.cliff_r - self.cliff_r_error_WF > self.r_0:
#                    print("r is bigger than r_0 + sigma_r, so above error bars should not be trusted.")
                    
