"""
The EmbeddedErrorgen class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.baseobjs.basis import Basis as _Basis
import warnings as _warnings

from pygsti.modelmembers.operations.embeddedop import EmbeddedOp as _EmbeddedOp
from pygsti.modelmembers.operations.lindbladerrorgen import LindbladErrorgen as _LinbladErrorGen


# Idea:
# Op = exp(Errgen); Errgen is an error just on 2nd qubit (and say we have 3 qubits)
# so Op = I x (I+eps*A) x I (small eps limit); eps*A is 1-qubit error generator
# also Op ~= I+Errgen in small eps limit, so
# Errgen = I x (I+eps*A) x I - I x I x I
#        = I x I x I + eps I x A x I - I x I x I
#        = eps I x A x I = I x eps*A x I
# --> we embed error generators by tensoring with I's on non-target sectors.
#  (identical to how be embed operations)


class EmbeddedErrorgen(_EmbeddedOp):
    """
    An error generator containing a single lower (or equal) dimensional operation within it.

    An EmbeddedErrorGen acts as the null map (zero) on all of its domain except the
    subspace of its contained error generator, where it acts as the contained item does.

    Parameters
    ----------
    state_space : StateSpace
        Specifies the density matrix space upon which this operation acts.

    target_labels : list of strs
        The labels contained in `state_space` which demarcate the
        portions of the state space acted on by `errgen_to_embed` (the
        "contained" error generator).

    errgen_to_embed : LinearOperator
        The error generator object that is to be contained within this
        error generator, and that specifies the only non-trivial action
        of the EmbeddedErrorgen.
    """

    def __init__(self, state_space, target_labels, errgen_to_embed: _LinbladErrorGen):
        _EmbeddedOp.__init__(self, state_space, target_labels, errgen_to_embed)

        # set "API" error-generator members (to interface properly w/other objects)
        # FUTURE: create a base class that defines this interface (maybe w/properties?)
        #self.sparse = True # Embedded error generators are *always* sparse (pointless to
        #                   # have dense versions of these)

        embedded_matrix_basis = errgen_to_embed.matrix_basis
        if isinstance(embedded_matrix_basis, str):
            self.matrix_basis = embedded_matrix_basis
        else:  # assume a Basis object
            my_basis_dim = self.state_space.dim
            self.matrix_basis = _Basis.cast(embedded_matrix_basis.name, my_basis_dim, sparse=True)

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        _EmbeddedOp.from_vector(self, v, close, dirty_value)
        self.dirty = dirty_value

    #TODO: I don't think the return_basis flag actually works atm. Maybe remove?
    #TODO: Refactor naming to match EmbeddedOp. Only reason we can't just directly use the
    #method from the parent class is naming convention mismatches for methods on children.
    def coefficients(self, return_basis=False, logscale_nonham=False, label_type='global', identity_label='I'):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients of this operation.

        Note that these are not necessarily the parameter values, as these
        coefficients are generally functions of the parameters (so as to keep
        the coefficients positive, for instance).

        Parameters
        ----------
        return_basis : bool
            Whether to also return a :class:`Basis` containing the elements
            with which the error generator terms were constructed.

        logscale_nonham : bool, optional
            Whether or not the non-hamiltonian error generator coefficients
            should be scaled so that the returned dict contains:
            `(1 - exp(-d^2 * coeff)) / d^2` instead of `coeff`.  This
            essentially converts the coefficient into a rate that is
            the contribution this term would have within a depolarizing
            channel where all stochastic generators had this same coefficient.
            This is the value returned by :meth:`error_rates`.
        
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

        Returns
        -------
        embedded_coeffs : dict
            Keys are instances of `ElementaryErrorgenLabel`, which wrap the 
            `(termType, basisLabel1, <basisLabel2>)` information for each coefficient.
            Where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"`(Correlation)  or `"A"` (Affine).  Hamiltonian and S terms always have a
            single basis label while 'C' and 'A' terms have two.
        """
        coeffs_to_embed = self.embedded_op.coefficients(return_basis, logscale_nonham, label_type)

        if coeffs_to_embed:
            embedded_labels = self.coefficient_labels(label_type=label_type, identity_label=identity_label)
            embedded_coeffs = {lbl:val for lbl, val in zip(embedded_labels, coeffs_to_embed.values())}
        else:
            embedded_coeffs = dict()

        return embedded_coeffs

    def coefficient_labels(self, label_type='global', identity_label='I'):
        """
        The elementary error-generator labels corresponding to the elements of :meth:`coefficients_array`.

        Parameters
        ----------
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        if label_type=='global' and self._cached_embedded_errorgen_labels_global is not None:
            return self._cached_embedded_errorgen_labels_global
        elif label_type=='local' and self._cached_embedded_errorgen_labels_local is not None and self._cached_embedded_label_identity_label==identity_label:
            return self._cached_embedded_errorgen_labels_local

        labels_to_embed = self.embedded_op.coefficient_labels(label_type)
        embedded_labels = self._embed_labels(labels_to_embed, label_type, identity_label)
        
        return embedded_labels


    def coefficients_array(self):
        """
        The weighted coefficients of this error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :meth:`coefficients`,
        weighted so that different error generators can be weighted differently when a
        `errorgen_penalty_factor` is used in an objective function.

        Returns
        -------
        numpy.ndarray
            A 1D array of length equal to the number of coefficients in the linear
            combination of standard error generators that is this error generator.
        """
        return self.embedded_op.coefficients_array()

    def coefficients_array_deriv_wrt_params(self):
        """
        The jacobian of :meth:`coefficients_array` with respect to this error generator's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients in the linear combination of standard error generators that is this error
            generator, and `num_params` is this error generator's number of parameters.
        """
        return self.embedded_op.coefficients_array_deriv_wrt_params()

    def error_rates(self, label_type='global', identity_label='I'):
        """
        Constructs a dictionary of the error rates associated with this error generator.

        These error rates pertain to the *channel* formed by exponentiating this object.

        The "error rate" for an individual Hamiltonian error is the angle
        about the "axis" (generalized in the multi-qubit case)
        corresponding to a particular basis element, i.e. `theta` in
        the unitary channel `U = exp(i * theta/2 * BasisElement)`.

        The "error rate" for an individual Stochastic error is the
        contribution that basis element's term would have to the
        error rate of a depolarization channel.  For example, if
        the rate corresponding to the term ('S','X') is 0.01 this
        means that the coefficient of the rho -> X*rho*X-rho error
        generator is set such that if this coefficient were used
        for all 3 (X,Y, and Z) terms the resulting depolarizing
        channel would have error rate 3*0.01 = 0.03.

        Note that because error generator terms do not necessarily
        commute with one another, the sum of the returned error
        rates is not necessarily the error rate of the overall
        channel.

        Parameters
        ----------
        label_type : str, optional (default 'global')
            String specifying which type of `ElementaryErrorgenLabel` to use
            as the keys for the returned dictionary. Allowed options are
            'global' for `GlobalElementaryErrorgenLabel` and 'local' for
            `LocalElementaryErrorgenLabel`.

        identity_label : str, optional (default 'I')
            An optional string specifying the basis element label for the
            identity. Used when label_type is 'local' to allow for embedding
            local basis element labels into the appropriate higher dimensional
            space. Only change when using a basis for which 'I' does not denote
            the identity.

        Returns
        -------
        lindblad_term_dict : dict
            Keys are instances of `ElementaryErrorgenLabel`, which wrap the 
            `(termType, basisLabel1, <basisLabel2>)` information for each coefficient.
            Where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"`(Correlation)  or `"A"` (Affine).  Hamiltonian and S terms always have a
            single basis label while 'C' and 'A' terms have two.
        """
        return self.coefficients(return_basis=False, logscale_nonham=True, label_type=label_type, identity_label=identity_label)

    def set_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False, truncate=True):
        """
        Sets the coefficients of terms in this error generator.

        The dictionary `lindblad_term_dict` has tuple-keys describing the type
        of term and the basis elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are instances of `ElementaryErrorgenLabel`, which wrap the 
            `(termType, basisLabel1, <basisLabel2>)` information for each coefficient.
            Where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"`(Correlation)  or `"A"` (Affine).  Hamiltonian and S terms always have a
            single basis label while 'C' and 'A' terms have two. Values are corresponding rates.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :meth:`errorgen_coefficients`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :meth:`set_error_rates`.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given coefficients
            cannot be parameterized as specified.

        Returns
        -------
        None
        """
        if lindblad_term_dict:
            unembedded_coeffs = self._unembed_coeff_dict_labels(lindblad_term_dict)
            self.embedded_op.set_coefficients(unembedded_coeffs, action, logscale_nonham, truncate)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator.

        Coefficients are set so that the contributions of the resulting
        channel's error rate are given by the values in `lindblad_term_dict`.
        See :meth:`error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case, when they may be complex.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_coefficients(lindblad_term_dict, action, logscale_nonham=True)

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized derivatives of the
        flattened error generator matrix with respect to a single operator
        parameter.  Thus, each column is of length op_dim^2 and there is one
        column per operation parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension^2, num_params)
        """
        _warnings.warn("Using finite differencing to compute EmbeddedErrorGen derivative!")
        #raise NotImplementedError("deriv_wrt_params is not implemented for EmbeddedErrorGen objects")
        return super(EmbeddedErrorgen, self).deriv_wrt_params(wrt_filter)

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this error generator with respect to its parameters.

        This function returns a tensor whose first axis corresponds to the
        flattened operation matrix and whose 2nd and 3rd axes correspond to the
        parameters that are differentiated with respect to.

        Parameters
        ----------
        wrt_filter1 : list or numpy.ndarray
            List of parameter indices to take 1st derivatives with respect to.
            (None means to use all the this operation's parameters.)

        wrt_filter2 : list or numpy.ndarray
            List of parameter indices to take 2nd derivatives with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Hessian with shape (dimension^2, num_params1, num_params2)
        """
        _warnings.warn("Using finite differencing to compute EmbeddedErrorGen hessian!")
        #raise NotImplementedError("hessian_wrt_params is not implemented for EmbeddedErrorGen objects")
        return super(EmbeddedErrorgen, self).hessian_wrt_params(wrt_filter1, wrt_filter2)

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (viewed as a matrix).

        Returns
        -------
        float
        """
        return self.embedded_op.onenorm_upperbound()
        # b/c ||A x B|| == ||A|| ||B|| and ||I|| == 1.0

    def __str__(self):
        """ Return string representation """
        s = "Embedded error generator with full dimension %d and state space %s\n" % (self.dim, self.state_space)
        s += " that embeds the following %d-dimensional operation into acting on the %s space\n" \
             % (self.embedded_op.dim, str(self.target_labels))
        s += str(self.embedded_op)
        return s
