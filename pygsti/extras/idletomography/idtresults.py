#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Idle Tomography results object """


class IdleTomographyResults(object):
    """
    A container for idle tomography results:  intrinsic and observable errors,
    along with supporting information.
    """

    def __init__(self, dataset, max_lengths, max_error_weight, fit_order,
                 pauli_dicts, idle_str, error_list, intrinsic_rates,
                 pauli_fidpairs, observed_rate_infos):
        """
        Create a IdleTomographyResults object.

        Parameters
        ----------
        dataset : DataSet
            The dataset that was analyzed, containing the observed counts.

        max_lengths : list
            The series of maximum lengths used.

        max_error_weight : int
            The maximum error weight.

        fit_order : int
            The order of the polynomial fits used.

        pauli_dicts : tuple
            A 2-tuple of `(prepDict,measDict)` Pauli basis dictionaries.

        idle_str : Circuit
            The idle operation that was characterized.

        error_list : list
            A list of :class:`NQPauliOp` objects describing the errors
            Paulis considered for each intrinsic-error type.

        intrinsic_rates : dict
            A dictionary of the intrinsic rates.  Keys are intrinsic-rate-types,
            i.e. 'hamiltonian', 'stochastic', or 'affine'.  Values are numpy
            arrays of length `len(error_list)`.

        pauli_fidpairs : dict
            A dictionary of the pauli-state fiducial pairs.  Keys are
            observed-rate-types, i.e. 'samebasis' or 'diffbasis', and
            values are lists of `(prep,meas)` 2-tuples of
            :class:`NQPauliState` objects.

        observed_rate_infos : dict
            A dictionary of observed-rate information dictionaries.  Keys are
            observed-rate-types, i.e. 'samebasis' or 'diffbasis', and
            values are further dictionaries indexed by fiducial pair (i.e. an
            element of `pauli_fidpairs`, then either a :class:`NQOutcome` (for
            the "samebasis" case) or :class:`NQPauliOp` (for "diffbasis") case.
            After these two indexes, the value is *another* dictionary of
            information about the observeable rate so defined.  So, to get to
            an actual "info dict" you need to do something like:
            `observed_rate_infos[typ][fidpair][obsORoutcome]`
        """

        self.dataset = dataset
        self.max_lengths = max_lengths
        self.max_error_weight = max_error_weight
        self.fit_order = fit_order
        self.prep_basis_strs, self.meas_basis_strs = pauli_dicts
        self.idle_str = idle_str

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
            s += "\n".join("  %s: %g" % (str(err), rate) for err, rate in
                           zip(self.error_list, self.intrinsic_rates['stochastic']))
            s += "\n"

        if "affine" in self.intrinsic_rates:
            s += "Intrinsic affine rates: \n"
            s += "\n".join("  %s: %g" % (str(err), rate) for err, rate in
                           zip(self.error_list, self.intrinsic_rates['affine']))
            s += "\n"

        if "hamiltonian" in self.intrinsic_rates:
            s += "Intrinsic hamiltonian rates:\n"
            s += "\n".join("  %s: %g" % (str(err), rate) for err, rate in
                           zip(self.error_list, self.intrinsic_rates['hamiltonian']))
            s += "\n"

        return s
