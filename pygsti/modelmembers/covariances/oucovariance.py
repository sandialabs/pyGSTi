"""
Covariance function specialized to the case of an Ornstein-Uhlenbeck (OU) process.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelmembers.covariances.covariancefunc import CovarianceFunction as _CovarianceFunction
from pygsti.modelmembers import ModelMember as _ModelMember
from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel as _ElementaryErrorgenLabel
import numpy as _np

class OUCovarianceFunction(_CovarianceFunction):

    """
    Covariance function based on the Ornstein-Uhlenbeck process.
    """

    @classmethod
    def from_errorgen_labels(error_generator_labels):
        pass

    def __init__(self, error_generator_labels):
        """
        Initialize an instance of an Ornstein-Uhlenbeck covariance function.
        The covariance function of an OU process is given by:

        cov(x_s, x_t) = (sigma^2 * t_c/2) * exp(-abs(s-t)/t_c)

        Where t_c is the characteristic correlation time for the process,
        and sigma^2 is the characteristic variance scale for the process.

        Parameters
        ----------
        error_generator_labels : dict, or list of tuples
            Either a dictionary or a list of tuples which specifies the pairs of
            elementary error generators with non-trivial covariances. If a dictionary
            the keys should be tuples containing either one or two error generator labels.
            If length one then this is interpreted to mean the autocorrelation between this error
            with itself at different times. If length two then this is interpreted to be a
            cross-correlation between two different elementary error generators at different times.
            The values of this dictionary are tuple corresponding to the initial values of
            t_c and sigma^2 for this covariance function, respectively.

            If a list of tuples, these tuples have the same format/interpretation as desribed above, but are
            instantiated with t_c and sigma^2 values of zero.    

            #TODO: Add casting logic so that we only need the keys to be castable to error generator labels.

        """

        _ModelMember.__init__(self, None, None)

        #assert that all of the keys/list tuples are error generator labels.
        for label_tup in error_generator_labels:
            for lbl in label_tup:
                assert isinstance(lbl, _ElementaryErrorgenLabel), 'All keys must be `ElementaryErrorgenLabel`s.'

        #need a mapping between error generator labels and parameter indices
        self._errgen_label_to_param_idx = dict()
        for i, label_tup in enumerate(error_generator_labels):
            if len(label_tup) == 1:
                self._errgen_label_to_param_idx[(label_tup[0], label_tup[0])] = i
            elif len(label_tup) == 2:
                self._errgen_label_to_param_idx[label_tup] = i
                #also make sure the other direction maps to the same index
                self._errgen_label_to_param_idx[(label_tup[1], label_tup[0])] = i
            else:
                raise ValueError('Too many error generator labels to unpack. Length of tuple of errorgen labels should be either 1 or 2.')

        #allocate arrays to internally store the correlation time and variance scale separately.
        self.correlation_times = _np.zeros(len(error_generator_labels))
        self.variances = _np.zeros(len(error_generator_labels))

        #if a dictionary initialize these values to those values.
        if isinstance(error_generator_labels, dict):
            for i, (corr_time, var) in enumerate(error_generator_labels.values()):
                self.correlation_times[i] = corr_time
                self.variances[i] = var

        #Concatenate the correlation time and variance vectors.
        self._paramvals = _np.concatenate([self.correlation_times, self.variances])
        
        #label parameters as either correlation times or variances.
        self._paramlbls = []
        for lbl_tup in self._errgen_label_to_param_idx.keys():
            self._paramlbls.append(f'Correlation Time for {lbl_tup}')
        for lbl_tup in self._errgen_label_to_param_idx.keys():
            self._paramlbls.append(f'Covariance for {lbl_tup}')
        self._paramlbls = _np.array(self._paramlbls, dtype=object)      

    def to_vector(self):
        return self._paramvals

    def from_vector(self, v, close=False, dirty_value=True):
        self._paramvals[:] = v
        n = len(self.correlation_times)
        self.correlation_times[0:n] = self._paramvals[:n]
        self.variances[n:] = self._paramvals[n:]
        self.dirty = dirty_value
    
    #TODO: Add logic for handling different error generator label types.
    def __call__(self, errorgen1, time1, errorgen2, time2):
        """
        Computes the correlation function. Takes as input two error generator labels and two times.
        Returns 0 if both error generator labels are not present in this covariance function.
        """
        idx = self._errgen_label_to_param_idx.get((errorgen1, errorgen2), None)

        if idx is None:
            return 0
        else:
            cov_val = .5*self.variances[idx]*self.correlation_times[idx]*_np.exp(-abs(time1-time2)/self.correlation_times[idx])
            return cov_val

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return 2*len(self.correlation_times)