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
from pygsti.baseobjs.errorgenlabel import ElementaryErrorgenLabel as _ElementaryErrorgenLabel

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

        cov(x_s, x_t) = (sigma^2 * t_c/2) * exp(-t_c*abs(s-t))

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

        #assert that all of the keys/list tuples are error generator labels.
        for label_tup in error_generator_labels:
            for lbl in label_tup:
                assert isinstance(lbl, _ElementaryErrorgenLabel), 'All keys must be `ElementaryErrorgenLabel`s.'

        if isinstance(error_generator_labels, list):
            msg = 'When specifying a list for `error_generator_labels` the values should be tuple with the format specified in the docstring.'
            assert isinstance(error_generator_labels[0], tuple), msg



            #need a mapping between error generator labels and parameter indices
            self._errgen_label_to_param_idx = dict()

            for label_tup in error_generator_labels:
                if len()


    def to_vector():
        pass

    def from_vector(self, v, close=False, dirty_value=True):
        return super().from_vector(v, close, dirty_value)
    
    def __call__(self):
        pass


    