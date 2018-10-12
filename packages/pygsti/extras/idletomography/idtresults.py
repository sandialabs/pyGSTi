""" Idle Tomography results object """
from __future__ import division, print_function, absolute_import, unicode_literals

class IdleTomographyResults(object):
    """ TODO: docstrings! """
    def __init__(self, dataset, maxLengths, maxErrWeight, fitOrder, 
                 error_list, intrinsic_rates, pauli_fidpairs,
                 observed_rate_infos):

        self.dataset = dataset
        self.max_lengths = maxLengths
        self.max_error_weight = maxErrWeight
        self.fit_order = fitOrder
        
        # the intrinsic error Paulis, as a list of NQPauliOp objects.  Gives the ordering of 
        #  each value of self.intrinsic_rates
        self.error_list = error_list[:]

        # the intrinsic error rates. Allowed keys are "hamiltonian", "stochastic", "affine"
        #  Values are numpy arrays whose components correspond to the error_list Paulis
        self.intrinsic_rates = intrinsic_rates.copy() 

        # the fiducial pairs ("configurations") used to specify the observable rates.
        #  Allowed keys are "hamiltonian", "stochastic", "stochastic/affine"
        #  Values are lists of (prep,meas) tuples of NQPauliState objects.
        self.pauli_fidpairs = pauli_fidpairs.copy()

        # the observed error rates and meta-information.
        #  Allowed keys are the same as those of self.pauli_fidpairs,
        #  and a key's corresponding value is a list of dictionaries
        #  where each dict describes the observed rates of the
        #  corresponding fiducial pair in self.pauli_fidpairs:
        #  self.observed_rate_infos['hamiltonian'][fidpair] is a
        #    dict of "info dicts" (those returned from get_obs_..._err_rate)
        #    whose keys are NQPauliOp *observables*.
        #  self.observed_rate_infos['stochastic'][fidpair] or
        #    self.observed_rate_infos['stochastic/affine'][fidpair] is a
        #    dict of info dicts whose keys are NQPauliState *outcomes*
        self.observed_rate_infos = observed_rate_infos.copy()

        # can be used to store true or predicted 
        self.predicted_obs_rates = None 


    def __str__(self):
        s = "Idle Tomography Results\n"
        if "stochastic" in self.intrinsic_rates:
            s += "Intrinsic stochastic rates: \n"
            s += "\n".join("  %s: %g" % (str(err),rate) for err,rate in
                           zip(self.error_list, self.intrinsic_rates['stochastic']))
            s += "\n"
                
        if "affine" in self.intrinsic_rates:
            s += "Intrinsic affine rates: \n"
            s += "\n".join("  %s: %g" % (str(err),rate) for err,rate in
                           zip(self.error_list, self.intrinsic_rates['affine']))
            s += "\n"

        if "hamiltonian" in self.intrinsic_rates:
            s += "Intrinsic hamiltonian rates:\n"
            s += "\n".join("  %s: %g" % (str(err),rate) for err,rate in
                           zip(self.error_list, self.intrinsic_rates['hamiltonian']))
            s += "\n"

        return s

    def plot_observable_rate(self, typ, fidpair):
        pass
