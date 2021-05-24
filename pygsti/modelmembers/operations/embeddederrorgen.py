"""
The EmbeddedErrorgen class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections
from .embeddedop import EmbeddedOp as _EmbeddedOp

from ...objects.basis import Basis as _Basis, EmbeddedBasis as _EmbeddedBasis

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

    def __init__(self, state_space, target_labels, errgen_to_embed):
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

            #OLD: constructs a subset of this errorgen's full mxbasis, but not the whole thing:
            #self.matrix_basis = _Basis(
            #    name="embedded_" + embedded_matrix_basis.name,
            #    matrices=[self._embed_basis_mx(mx) for mx in
            #              embedded_matrix_basis.get_composite_matrices()],
            #    sparse=True)

        #OLD: when a generators "rep" was self.err_gen_mx
        #if errgen_to_embed._evotype == "densitymx":
        #    self._construct_errgen_matrix()
        #else:
        #    self.err_gen_mx = None

    #TODO REMOVE (UNUSED)
    #def _construct_errgen_matrix(self):
    #    #Always construct a sparse errgen matrix, so just use
    #    # base class's .to_sparse() (which calls embedded errorgen's
    #    # .to_sparse(), which will convert a dense->sparse embedded
    #    # error generator, but this is fine).
    #    self.err_gen_mx = self.tosparse()

    #TODO REMOVE (UNUSED)
    #def _embed_basis_mx(self, mx):
    #    """ Take a dense or sparse basis matrix and embed it. """
    #    mxAsGate = StaticDenseOp(mx) if isinstance(mx, _np.ndarray) \
    #        else StaticDenseOp(mx.todense())  # assume mx is a sparse matrix
    #    return EmbeddedOp(self.state_space, self.targetLabels,
    #                      mxAsGate).tosparse()  # always convert to *sparse* basis els

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

    def coefficients(self, return_basis=False, logscale_nonham=False):
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
            This is the value returned by :method:`error_rates`.

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are integers starting at 0.  Values are complex
            coefficients.
        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `Ltermdict` to basis matrices.
        """
        embedded_coeffs = self.embedded_op.coefficients(return_basis, logscale_nonham)
        embedded_Ltermdict = _collections.OrderedDict()

        if return_basis:
            # embed basis
            Ltermdict, basis = embedded_coeffs
            embedded_basis = _EmbeddedBasis(basis, self.state_space, self.targetLabels)
            bel_map = {lbl: embedded_lbl for lbl, embedded_lbl in zip(basis.labels, embedded_basis.labels)}

            #go through and embed Ltermdict labels
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([bel_map[x] for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict, embedded_basis
        else:
            #go through and embed Ltermdict labels
            Ltermdict = embedded_coeffs
            for k, val in Ltermdict.items():
                embedded_key = (k[0],) + tuple([_EmbeddedBasis.embed_label(x, self.targetLabels) for x in k[1:]])
                embedded_Ltermdict[embedded_key] = val
            return embedded_Ltermdict

    def coefficients_array(self):
        """
        The weighted coefficients of this error generator in terms of "standard" error generators.

        Constructs a 1D array of all the coefficients returned by :method:`coefficients`,
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
        The jacobian of :method:`coefficients_array` with respect to this error generator's parameters.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(num_coeffs, num_params)` where `num_coeffs` is the number of
            coefficients in the linear combination of standard error generators that is this error
            generator, and `num_params` is this error generator's number of parameters.
        """
        return self.embedded_op.coefficients_array_deriv_wrt_params()

    def error_rates(self):
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

        Returns
        -------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are real error rates except for the 2-basis-label
            case.
        """
        return self.coefficients(return_basis=False, logscale_nonham=True)

    def set_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False):
        """
        Sets the coefficients of terms in this error generator.

        The dictionary `lindblad_term_dict` has tuple-keys describing the type
        of term and the basis elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            or `"A"` (Affine).  Hamiltonian and Affine terms always have a
            single basis label (so key is a 2-tuple) whereas Stochastic tuples
            have 1 basis label to indicate a *diagonal* term and otherwise have
            2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Values are the coefficients of these error generators,
            and should be real except for the 2-basis-label case.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error-generator coefficients.

        logscale_nonham : bool, optional
            Whether or not the values in `lindblad_term_dict` for non-hamiltonian
            error generators should be interpreted as error *rates* (of an
            "equivalent" depolarizing channel, see :method:`errorgen_coefficients`)
            instead of raw coefficients.  If True, then the non-hamiltonian
            coefficients are set to `-log(1 - d^2*rate)/d^2`, where `rate` is
            the corresponding value given in `lindblad_term_dict`.  This is what is
            performed by the function :method:`set_error_rates`.

        Returns
        -------
        None
        """
        unembedded_Ltermdict = _collections.OrderedDict()
        for k, val in lindblad_term_dict.items():
            unembedded_key = (k[0],) + tuple([_EmbeddedBasis.unembed_label(x, self.targetLabels) for x in k[1:]])
            unembedded_Ltermdict[unembedded_key] = val
        self.embedded_op.set_coefficients(unembedded_Ltermdict, action, logscale_nonham)

    def set_error_rates(self, lindblad_term_dict, action="update"):
        """
        Sets the coeffcients of terms in this error generator.

        Coefficients are set so that the contributions of the resulting
        channel's error rate are given by the values in `lindblad_term_dict`.
        See :method:`error_rates` for more details.

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
        raise NotImplementedError("deriv_wrt_params is not implemented for EmbeddedErrorGen objects")

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
        raise NotImplementedError("hessian_wrt_params is not implemented for EmbeddedErrorGen objects")

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
             % (self.embedded_op.dim, str(self.targetLabels))
        s += str(self.embedded_op)
        return s
