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
    """
    Encapsulates a group where each element is represented by a matrix
    """

    def __init__(self, listOfMatrices, labels=None):
        """
        Constructs a new MatrixGroup object

        Parameters
        ----------
        listOfMatrices : list
            A list of the group elements (should be 2d numpy arrays).

        labels : list, optional
            A label corresponding to each group element.
        """
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
        """
        Returns the matrix corresponding to index or label `i`

        Parameters
        ----------
        i : int or str
            If an integer, an element index.  Otherwise, an element label.
        
        Returns
        -------
        numpy array
        """
        if not isinstance(i,int): i = self.label_indices[i]
        return self.mxs[i]

    def get_matrix_inv(self, i):
        """
        Returns the inverse of the matrix corresponding to index or label `i`

        Parameters
        ----------
        i : int or str
            If an integer, an element index.  Otherwise, an element label.
        
        Returns
        -------
        numpy array
        """
        if not isinstance(i,int): i = self.label_indices[i]
        return self.mxs[self.inverse_table[i]]

    def get_inv(self, i):
        """
        Returns the index/label corresponding to the inverse of index/label `i`

        Parameters
        ----------
        i : int or str
            If an integer, an element index.  Otherwise, an element label.
        
        Returns
        -------
        int or str
            If `i` is an integer, returns the element's index.  Otherwise
            returns the element's label.
        """
        if isinstance(i,int):
            return self.inverse_table[i]
        else:
            i = self.label_indices[i]
            return self.labels[ self.inverse_table[i] ]
        
    def product(self, indices):
        """
        Returns the index/label of corresponding to the product of a list
        or tuple of indices/labels.

        Parameters
        ----------
        indices : iterable
            Specifies the sequence of group elements to include in the matrix 
            product.  If `indices` contains integers, they an interpreted as
            group element indices, and an integer is returned.  Otherwise,
            `indices` is assumed to contain group element labels, and a label
            is returned.
        
        Returns
        -------
        int or str
            If `indices` contains integers, returns the resulting element's
            index.  Otherwise returns the resulting element's label.
        """
        if len(indices) == 0: return None
        if isinstance(indices[0],int):
            return _reduce(lambda i,j: self.product_table[i,j], indices)
        else:
            indices = [ self.label_indices[i] for i in indices ]
            fi = _reduce(lambda i,j: self.product_table[i,j], indices)
            return self.labels[fi]

    def __len__(self):
        """ Returns the order of the group (number of elements) """
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
                 AB0=[0.5,0.5], ABCD0=None):
        """
        Constructs a new RBResults object.

        Parameters
        ----------
        dataset : dataset
            The dataset which contains all the experimental RB data.

        result_dicts : dict
            A dictionary of dictionaries of RB result values.  Keys are 
            gate-label-sets, e.g. "clifford", "canonical", and "primivitve".
            Values are dictionaries containing RB input and computed values.

        basename : str
            A name given to the "base" gate-label-set, usually "clifford",
            which coresponds to the gate labels used in `dataset`.

        alias_maps : dict of dicts, optional
            If not None, a dictionary whose keys name other (non-"base") 
            gate-label-sets, and whose values are "alias" dictionaries 
            which map the "base" labels to those of the named gate-label-set.
            These maps specify how to move from the "base" gate-label-set to 
            the others.
            RB values for each of these gate-label-sets will be present in the 
            returned results object.

        success_spamlabel : str, optional
            The spam label which denotes the *expected* outcome of preparing,
            doing nothing (or the identity), and measuring.  In the ideal case
            of perfect gates, the probability of seeing this outcome when just
            preparing and measuring (no intervening gates) is 100%.
            
        dim : int, optional
            Hilbert space dimension.  Default corresponds to a single qubit.
    
        pre_avg : bool, optional
            Whether or not survival probabilities for different sequences of
            the same length were averaged together before curve fitting
            was performed.
    
        f0 : list, optional
            A length-1 list specifying the starting value of the 'f' fitting
            parameter that was used in curve fitting.
            
        AB0 : list, optional
            A length-2 list, [A0, B0], of starting values for the 'A' and 'B'
            fitting parameters in the zeroth order fit. 
            
        ABCD0 : list, optional
            A length-4 list, [A10, B10, C10, D10], of starting values for the 
            'A1' and 'B1', 'C1' and 'D1' fitting parameters in the first order
            fit. 
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
        self.ABCD0 = ABCD0

    def detail_str(self, gstyp, order):
        """
        Format a string with computed RB values using the given 
        gate-label-set, `gstyp` and order.  For example, if `gstyp` == "clifford",
        and order == 'zeroth' then the per-"clifford" zeroth order fit RB error rates 
        and parameters are given.
        
        Parameters
        ----------
        gstyp : str
            The gate-label-set specifying which RB error rates and parameters
            to extract.
            
        order : str
            Allowed values are 'zeroth' or 'first'. Specifies whether the zeroth or
            first order fitting model results are formatted.
            
        Returns
        -------
        str
        """
        s = ""
        key_list = ['A','B','f','F_avg','r']
        key_list_1st_order = ['A1','B1','C1','D1','f1','F_avg1','r1','gdep']
        if gstyp in self.dicts and self.dicts[gstyp] is not None:            
            #print("For %ss:" % gstyp)
            if order=='zeroth':
                s += "%s results \n" % gstyp
                s += "  - Using %s order fitting function: A + B*f^m  \n" % order
                if 'A_error_BS' in self.dicts[gstyp]: 
                    s += "  - with boostrapped-derived error bars (1 sigma):\n"
                    for key in key_list:
                        s += "%s = %s +/- %s\n" % (key,str(self.dicts[gstyp][key]),
                                        str(self.dicts[gstyp][key + "_error_BS"]))
                if 'A_error_WF' in self.dicts[gstyp]:
                    s += "   - with Wallman and Flammia-derived error bars" + \
                        "(1 sigma):\n"
                    for key in key_list:
                        s += "%s = %s +/- %s\n" % (key,str(self.dicts[gstyp][key]),
                                        str(self.dicts[gstyp][key + "_error_WF"]))

                if 'A_error_BS' not in self.dicts[gstyp] and \
                        'A_error_WF' not in self.dicts[gstyp]:
                    for key in key_list:            
                        s += "%s = %s\n" % (key, str(self.dicts[gstyp][key]))
            if order=='first':
                s += "%s results \n" % gstyp
                s += "   - Using %s order fitting function: A1 + B1*f^m + C1*(m-1)(D1-f^2)*f^(m-2)  \n" % order
                if 'A1_error_BS' in self.dicts[gstyp]: 
                    s += "   - with boostrapped-derived error bars (1 sigma):\n"
                    for key in key_list_1st_order:
                        s += "%s = %s +/- %s\n" % (key,str(self.dicts[gstyp][key]),
                                        str(self.dicts[gstyp][key + "_error_BS"]))
                if 'A1_error_BS' not in self.dicts[gstyp]:
                    for key in key_list_1st_order:            
                        s += "%s = %s\n" % (key, str(self.dicts[gstyp][key]))

        else:
            s += "No %s analysis performed!\n" % gstyp
        return s

    def print_detail(self, gstyp, order='zeroth'):
        """
        Print computed RB values using the given gate-label-set, `gstyp`,
        and order parameter.

        For example, if `gstyp` == "clifford" and `order' == "all", then 
        the per-"clifford" RB error rates and parameters for both zeroth
        and first order fitting are printed.

        Parameters
        ----------
        gstyp : str
            The gate-label-set specifying which RB error rates and parameters
            to extract.
            
        order : str
            Allowed values are 'zeroth', 'first' or 'all'. Specifies whether 
            the zeroth or first order fitting model results are displayed,
            or both.
        """
        if order=='all':
            print(self.detail_str(gstyp, order='zeroth'))
            print(self.detail_str(gstyp, order='first'))
                       
        else:
            print(self.detail_str(gstyp, order))

    def print_clifford(self,order='zeroth'):
        """
        Display per Clifford gate RB error rate.
        """      
        self.print_detail('clifford',order)

    def print_primitive(self,order='zeroth'):
        """
        Display per primitive gate RB error rate.  The physical
        interpretation of these numbers may not be reliable; per Clifford 
        error rates are recommended instead. 
        """
        self.print_detail('primitive',order)

    def __str__(self):
        s = ""
        for gstyp in self.dicts:
            s += self.detail_str(gstyp) + "\n"
        return s


    def plot(self,gstyp,xlim=None,ylim=None,save_fig_path=None,order='zeroth'):
        """
        Plot RB decay curve, as a function of some the sequence length
        computed using the `gstyp` gate-label-set.

        Parameters
        ----------
        gstyp : str
            The gate-label-set specifying which translation (i.e. strings with
            which gate labels) to use when computing sequence lengths.

        xlim : tuple, optional
            The x-range as (xmin,xmax).

        ylim : tuple, optional
            The y-range as (ymin,ymax).

        save_fig_path : str, optional
            If not None, the filename where the resulting plot should be saved.
            
        order : str
            Optional. Allowed values are 'zeroth', 'first' or 'all'. Specifies 
            whether the zeroth or first order fitting model results are plotted,
            or both.

        Returns
        -------
        None
        """
        if gstyp not in self.dicts:
            raise ValueError("%s data not found!" % gstyp)

        newplot = _plt.figure(figsize=(8, 4))
        newplotgca = newplot.gca()

        # Note: minus one to get xdata discounts final Clifford-inverse
        xdata = _np.asarray(self.dicts[gstyp]['lengths']) - 1 
        ydata = _np.asarray(self.dicts[gstyp]['successes'])
        A = self.dicts[gstyp]['A']
        B = self.dicts[gstyp]['B']
        f = self.dicts[gstyp]['f']
        A1 = self.dicts[gstyp]['A1']
        B1 = self.dicts[gstyp]['B1']
        C1 = self.dicts[gstyp]['C1']
        D1 = self.dicts[gstyp]['D1']
        f1 = self.dicts[gstyp]['f1']
        xlabel = 'RB sequence length (%ss)' % gstyp

        cmap = _plt.cm.get_cmap('Set1')
        newplotgca.plot(xdata,ydata,'.', markersize=15, clip_on=False,
                        color=cmap(30))
        
        if order=='zeroth':
            newplotgca.plot(_np.arange(max(xdata)),
                            _rbutils.rb_decay_WF(_np.arange(max(xdata)),A,B,f),
                            '-', lw=2, color=cmap(110))

            newplotgca.set_xlabel(xlabel, fontsize=15)
            newplotgca.set_ylabel('Success Rate',fontsize=15)
            newplotgca.set_title('RB Success Curve', fontsize=20)
            
        if order=='first':
            newplotgca.plot(_np.arange(max(xdata)),
                            _rbutils.rb_decay_1st_order(_np.arange(max(xdata)),
                            A1,B1,C1,D1,f1),'-', lw=2, color=cmap(110))

            newplotgca.set_xlabel(xlabel, fontsize=15)
            newplotgca.set_ylabel('Success Rate',fontsize=15)
            newplotgca.set_title('RB First Order Fit Success Curve', fontsize=20)
            
        if order=='all':
            newplotgca.plot(_np.arange(max(xdata)),
                            _rbutils.rb_decay_WF(_np.arange(max(xdata)),A,B,f),
                            '-', lw=2, color=cmap(110), label='Zeroth order fit')
            newplotgca.plot(_np.arange(max(xdata)),
                            _rbutils.rb_decay_1st_order(_np.arange(max(xdata)),
                            A1,B1,C1,D1,f1),'--', lw=4, color=cmap(50),
                            label='First order fit')

            newplotgca.set_xlabel(xlabel, fontsize=15)
            newplotgca.set_ylabel('Success Rate',fontsize=15)
            newplotgca.set_title('RB Success Curves', fontsize=20)  
            

        newplotgca.set_frame_on(False)
        newplotgca.yaxis.grid(True)
        newplotgca.tick_params(axis='x', top='off', labelsize=12)
        newplotgca.tick_params(axis='y', left='off', right='off', labelsize=12)
        
        if order=='all':
            _plt.legend(loc='upper right')

        if xlim:
            _plt.xlim(xlim)
        if ylim:
            _plt.ylim(ylim)
        if save_fig_path:
            newplot.savefig(save_fig_path)
    

    def compute_bootstrap_error_bars(self, gstyp_list = "all", resamples = 100,
                                    seed=None, randState=None):
        """
        Compute error bars on RB fit parameters, including the RB decay rate
        using a non-parametric bootstrap method.

        Parameters
        ----------
        gstyp_list : list, optional
           A list of gate-label-set values (e.g. "clifford", "primitive")
           specifying which "per-X" RB values to compute error bars for.
           The special value "all" can be used to compute error bars for all
           existing gate-label-sets.

        resamples : int, optional
            The number of nonparametric bootstrap resamplings

        seed : int, optional
            Seed for random number generator; optional.
    
        randState : numpy.random.RandomState, optional
            A RandomState object to generate samples from. Can be useful to set
            instead of `seed` if you want reproducible distribution samples
            across multiple random function calls but you don't want to bother
            with manually incrementing seeds between those calls.

        Returns
        -------
        None
        """
        if randState is None:
            rndm = _rndm.RandomState(seed) # ok if seed is None
        else:
            rndm = randState

        if gstyp_list == "all":
            gstyp_list = list(self.dicts.keys())

        #Setup lists to hold items to take stddev of:
        A_list = {}; B_list = {}; f_list = {}; A1_list = {}
        B1_list = {}; C1_list = {}; D1_list = {}; f1_list = {};
        for gstyp in gstyp_list:
            A_list[gstyp] = []; B_list[gstyp] = []; f_list[gstyp] = []; \
            A1_list[gstyp] = []; B1_list[gstyp] = []; C1_list[gstyp] = []; \
            D1_list[gstyp] = []; f1_list[gstyp] = []

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
                                           self.pre_avg,self.f0, self.AB0,
                                           self.ABCD0)
            for gstyp in gstyp_list:
                A_list[gstyp].append(resample_results.dicts[gstyp]['A'])
                B_list[gstyp].append(resample_results.dicts[gstyp]['B'])
                f_list[gstyp].append(resample_results.dicts[gstyp]['f'])
                A1_list[gstyp].append(resample_results.dicts[gstyp]['A1'])
                B1_list[gstyp].append(resample_results.dicts[gstyp]['B1'])
                C1_list[gstyp].append(resample_results.dicts[gstyp]['C1'])
                D1_list[gstyp].append(resample_results.dicts[gstyp]['D1'])
                f1_list[gstyp].append(resample_results.dicts[gstyp]['f1'])
               
        for gstyp in gstyp_list:
            self.dicts[gstyp]['A_error_BS'] = _np.std(A_list[gstyp],ddof=1)
            self.dicts[gstyp]['B_error_BS'] = _np.std(B_list[gstyp],ddof=1)
            self.dicts[gstyp]['f_error_BS'] = _np.std(f_list[gstyp],ddof=1)
            self.dicts[gstyp]['A1_error_BS'] = _np.std(A1_list[gstyp],ddof=1)
            self.dicts[gstyp]['B1_error_BS'] = _np.std(B1_list[gstyp],ddof=1)
            self.dicts[gstyp]['C1_error_BS'] = _np.std(C1_list[gstyp],ddof=1)
            self.dicts[gstyp]['D1_error_BS'] = _np.std(D1_list[gstyp],ddof=1)
            self.dicts[gstyp]['f1_error_BS'] = _np.std(f1_list[gstyp],ddof=1)
            
            self.dicts[gstyp]['F_avg_error_BS'] = (self.d-1.) / self.d \
                                         * self.dicts[gstyp]['f_error_BS']
            self.dicts[gstyp]['F_avg1_error_BS'] = (self.d-1.) / self.d \
                                         * self.dicts[gstyp]['f1_error_BS']
            self.dicts[gstyp]['r_error_BS'] = \
                self.dicts[gstyp]['F_avg_error_BS']
            self.dicts[gstyp]['r1_error_BS'] = \
                self.dicts[gstyp]['F_avg1_error_BS']
            self.dicts[gstyp]['gdep_error_BS'] = _np.sqrt( \
                self.dicts[gstyp]['D1_error_BS']**2 + \
                4 * self.dicts[gstyp]['f1']**2 \
                *self.dicts[gstyp]['f1_error_BS']**2 )

        print("Bootstrapped error bars computed.  Use print methods to access.")


    def compute_analytic_error_bars(self, epsilon, delta, r_0, 
                                    p0 = [0.5,0.5,0.98]):
        """
        Compute error bars on RB fit parameters, including the RB decay rate
        using the quasi-analytic methods provided in W&F.

        *At present, this method is not fully supported.*

        Parameters
        ----------
        epsilon : float
            Specifies desired confidence interval half-width for each average
            survival probability estimate \hat{F}_m (See W&F Eq. 8).  
            E.g., epsilon = 0.01 means that the confidence interval width for
            \hat{F}_m is 0.02. See `create_K_m_sched` for further details.
    
        delta : float
            Specifies desired confidence level for confidence interval
            specified by epsilon.  delta = 1-0.6827 corresponds to a
            confidence level of 1 sigma.  (See W&F Eq. 8).  The smaller
            delta is, the larger each value of K_m will be. See 
            `create_K_m_sched` for further details.
    
        r_0 : float
            Estimate of upper bound of the RB number for the system in 
            question. The smaller r is, the smaller each value of K_m will be.
            However, if the system's actual RB number is larger than r_0, then
            the W&F-derived error bars cannot be assumed to be valid.  Addition
            ally, it is assumed that m_max*r_0 << 1.  See `create_K_m_sched`
            for further details.

        p0 : list, optional
            A list of [f,A,B] parameters to seed the RB curve fitting.  Usually
            the default values are fine.

        Returns
        -------
        None
        """
        print("WARNING: ANALYTIC BOOSTRAP ERROR BAR METHOD NOT YET" +
              "GUARANTEED TO BE STABLE.")
        print("ERROR BARS ONLY FOR ZEROTH ORDER FIT.")
        print('Processesing analytic bootstrap, following Wallman and' +
              'Flammia.\nThe error bars are reliable if and only if the' +
              'schedule for K_m has been chosen appropriately, given:')
        print('delta =',delta)
        print('epsilon =',epsilon)
        print('r_0 =',r_0)

        gstyp_list = ['clifford'] 
          #KENNY: does WF assume clifford-gatestring data?
        
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
