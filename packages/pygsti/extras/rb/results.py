#
# Todo -- go through and delete all of this?
#
class RBResults(object):
    
    def __init__(self):

        self.data = {}

    def compute_bootstrap_error_bars(self, gstyp_list = ("clifford",), resamples = 100,
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

        seed : int or numpy.random.RandomState, optional
            Seed for random number generator.  A RandomState object to generate
            samples from, which can be useful if you want reproducible
            distribution of samples across multiple random function calls but
            you don't want to bother with manually changing seeds between
            those calls.

        Returns
        -------
        None
        """
        #Setup lists to hold items to take stddev of:
        A_list = []; B_list = []; f_list = []; 
        if self.fit == 'first order':
            C_list = []

        #Create bootstrap datasets
        bootstrapped_dataset_list = []
        for _ in range(resamples):
            bootstrapped_dataset_list.append(
                _drivers.bootstrap.make_bootstrap_dataset(
                    self.dataset,'nonparametric',seed=seed))

        #Run RB analysis for each dataset
        gatestrings = self.results['gatestrings']
        #alias_maps = { k:mp for k,mp in self.alias_maps.items()
        #               if k in gstyp_list } #only alias maps of requested
        from .rbcore import do_rb_base as _do_rb_base
        for dsBootstrap in bootstrapped_dataset_list:
            resample_results = _do_rb_base(dsBootstrap, gatestrings,
                                           self.fit, 
                                           self.fit_parameters_dict,
                                           self.success_outcomelabel,
                                           self.d,
                                           self.weight_data,
                                           self.pre_avg,
                                           self.infinite_data,
                                           self.one_freq_adjust)
                
            A_list.append(resample_results.results['A'])
            B_list.append(resample_results.results['B'])
            f_list.append(resample_results.results['f'])
            if self.fit == 'first order':
                C_list.append(resample_results.results['C'])
               

        self.results['A_error_BS'] = _np.std(A_list,ddof=1)
        self.results['B_error_BS'] = _np.std(B_list,ddof=1)
        self.results['f_error_BS'] = _np.std(f_list,ddof=1)
        if self.fit == 'first order':
            self.results['C_error_BS'] = _np.std(C_list,ddof=1)

        self.results['r_error_BS'] = (self.d-1.) / self.d \
                                         * self.results['f_error_BS']

        print("Bootstrapped error bars computed.  Use print methods to access.")