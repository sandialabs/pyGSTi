"""
Defines the OplessForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .forwardsim import CacheForwardSimulator as _CacheForwardSimulator
from ..layouts.cachedlayout import CachedCOPALayout as _CachedCOPALayout


class SuccessFailForwardSimulator(_CacheForwardSimulator):

    def create_layout(self, circuits, dataset=None, resource_alloc=None,
                      array_types=('E',), derivative_dimension=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for `circuits` and `dataset`.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be computed.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types.  See :method:`ForwardSimulator.create_layout`.

        derivative_dimension : int, optional
            Optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            non-None when `array_types` contains `'ep'` or `'epp'` types.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        Returns
        -------
        CachedCOPALayout
        """
        #Note: resource_alloc not even used -- make a slightly more complex "default" strategy?
        cache = {c: self.model._circuit_cache(c) for c in circuits}
        num_params = derivative_dimension if (derivative_dimension is not None) else self.model.num_params
        return _CachedCOPALayout.create_from(circuits, self.model, dataset, (num_params, num_params),
                                             resource_alloc, cache)

    def _compute_circuit_outcome_probabilities_with_cache(self, array_to_fill, circuit, outcomes, resource_alloc, cache,
                                                          time=None):
        #REMOVE
        #if False and cache:  # TEST (disabled) - use cached polys to evaluate - REMOVE this?
        #    cpolys = cache
        #    ps = _bulk_eval_compact_polynomials(cpolys[0], cpolys[1], self.model.to_vector(), (len(outcomes),))
        #    assert(_np.linalg.norm(_np.imag(ps)) < 1e-6)
        #    array_to_fill[:] = _np.real(ps)
        #else:

        sp = self.model._success_prob(circuit, cache)
        probs = {('success',): sp, ('fail',): 1 - sp}
        for i, outcome in enumerate(outcomes):
            array_to_fill[i] = probs[outcome]

    def _compute_circuit_outcome_probability_derivatives_with_cache(self, array_to_fill, circuit, outcomes, param_slice,
                                                                    resource_alloc, cache):
        # array to fill has shape (num_outcomes, len(param_slice)) and should be filled with the "w.r.t. param_slice"
        # derivatives of each specified circuit outcome probability.

        #REMOVE
        #if False and cache:  # TEST (disabled)
        #    cpolys = cache
        #    dpolys = _compact_deriv(cpolys[0], cpolys[1], _slct.indices(param_slice))
        #    array_to_fill[:, :] = _bulk_eval_compact_polynomials(dpolys[0], dpolys[1], self.model.to_vector(),
        #                                                         (len(outcomes), _slct.length(param_slice)))
        #else:

        dsp = self.model._success_dprob(circuit, param_slice, cache)
        dprobs = {('success',): dsp, ('fail',): -dsp}
        for i, outcome in enumerate(outcomes):
            array_to_fill[i, :] = dprobs[outcome]

        # Alternate finite difference option?
        #for j in range(np):
        #    p_plus_dp = p.copy()
        #    p_plus_dp[j] += eps
        #    self.from_vector(p_plus_dp)
        #    probs1 = self.probs(c,clip_to,cache)
        #    mx_to_fill[k,j] = (probs1[outcome]-probs0[outcome]) / eps
        #self.from_vector(p)
