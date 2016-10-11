from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines Randomized Benhmarking results object """

from ... import drivers as _drivers



class RBResults(object):
    """
    Defines a class which holds an RB data set along with other parameters,
    and can process the RB data in a variety of manners, including the 
    generation of plots.

    As in other docstrings, "W&F" refers to Wallman and Flammia's 
    http://iopscience.iop.org/article/10.1088/1367-2630/16/10/103032.
    """
    
    def __init__(self, dataset, prim_seq_list, cliff_len_list, d=2,
                 prim_dict=None, pre_avg=True, epsilon=None,
                 delta=None,r_0=None):
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
        Process RB data into survival probabilities along with new sequence
        length lists (if needed due to pre-averaging).
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


    def analyze_data(self,rb_decay_func = rb_decay_WF,process_prim = False,
                     process_cliff = False, f0 = [0.98], AB0=[0.5,0.5]):
        """
        Analyze RB data to compute fit parameters and in turn the RB error
        rate.

        TODO: docstring describing parameters
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
            self.prim_initial_soln = _minimize(obj_func_1d,f0,
                                               method='L-BFGS-B',
                                               bounds=[(0.,1.)])
            f1 = [self.prim_initial_soln.x[0]]
            p0 = AB0 + f1
            self.prim_end_soln = _minimize(obj_func_full,p0,
                                           method='L-BFGS-B',
                                           bounds=[(0.,1.),(0.,1.),(0.,1.)])
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
            self.cliff_initial_soln = _minimize(obj_func_1d,f0,
                                                method='L-BFGS-B',
                                                bounds=[(0.,1.)])
            f0 = self.cliff_initial_soln.x[0]
            p0 = AB0 + [f0]
            self.cliff_end_soln = _minimize(obj_func_full,p0,
                                            method='L-BFGS-B',
                                            bounds=[(0.,1.),(0.0,1.),(0.,1.)])
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
        Display per primitive gate RB error rate.  These physical
        interpretation of these numbers may not be reliable; per Clifford 
        error rates are recommended instead. 
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

        TODO: docstring describing parameters
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
    

    def generate_error_bars(self, method,process_prim=False,
                            process_cliff=False,
                            bootstrap_resamples = 100,
                            p0 = [0.5,0.5,0.98], seed=None):
        """
        Compute error bars on RB fit parameters, including the RB decay rate.
        Error bars can be computed either using a non-parametric bootstrap, 
        or quasi-analytic methods provided in W&F.

        *At present, only the bootstrap method is fully supported.*

        TODO: docstring describing parameters
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
                    bootstrapped_dataset_list.append(
                        _drivers.bootstrap.make_bootstrap_dataset(
                            self.dataset,'nonparametric'))
                for resample in range(bootstrap_resamples):
                    temp_rb_results = process_rb_data(
                        bootstrapped_dataset_list[resample],
                        self.prim_seq_list_orig,
                        self.cliff_len_list_orig,
                        prim_dict = self.prim_dict,
                        pre_avg = self.pre_avg,
                        process_prim = process_prim,
                        process_cliff = process_cliff,
                        f0 = [0.98],AB0=[0.5,0.5])
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

                    self.prim_F_avg_error_BS = (self.d-1.) / self.d \
                                                * self.prim_f_error_BS
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

                    self.cliff_F_avg_error_BS = (self.d-1.)/self.d \
                                                  * self.cliff_f_error_BS
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
            results = _curve_fit(rb_decay_WF,
                                 self.cliff_len_list,
                                 self.cliff_successes,
                                 p0 = p0, 
                                 sigma = self.sigma_list)
            self.full_results = results
            self.cliff_A_error_WF = _np.sqrt(results[1][0,0])
            self.cliff_B_error_WF = _np.sqrt(results[1][1,1])
            self.cliff_f_error_WF = _np.sqrt(results[1][2,2])

            self.cliff_F_avg_error_WF = (self.d-1.)/self.d  \
                                         * self.cliff_f_error_WF
            self.cliff_r_error_WF = self.cliff_F_avg_error_WF

        print("Error bars computed.  Use error_bars method to access.")

    def print_error_bars(self,method,process_prim=False,process_cliff=False):
        """
        TODO: docstring
        """
        assert method in ['bootstrap', 'analytic']
        if method == 'bootstrap':
            if not self.bootstrap:
                raise ValueError('Bootstrapped error bars requested but not' +
                       'yet generated; use generate_error_bars method first.')
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
                    
