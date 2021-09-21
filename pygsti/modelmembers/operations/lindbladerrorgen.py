"""
The LindbladErrorgen class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings
import collections as _collections

import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

from pygsti.baseobjs.opcalc import compact_deriv as _compact_deriv, \
    bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex, \
    abs_sum_bulk_eval_compact_polynomials_complex as _abs_sum_bulk_eval_compact_polynomials_complex
from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers import term as _term
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel as _GlobalElementaryErrorgenLabel
from pygsti.tools import basistools as _bt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


class LindbladErrorgen(_LinearOperator):
    """
    An Lindblad-form error generator.

    This error generator consisting of terms that, with appropriate constraints
    ensurse that the resulting (after exponentiation) operation/layer operation
    is CPTP.  These terms can be divided into "Hamiltonian"-type terms, which
    map rho -> i[H,rho] and "non-Hamiltonian"/"other"-type terms, which map rho
    -> A rho B + 0.5*(ABrho + rhoAB).

    Parameters
    ----------
    dim : int
        The Hilbert-Schmidt (superoperator) dimension, which will be the
        dimension of the created operator.

    lindblad_term_dict : dict
        A dictionary specifying which Linblad terms are present in the
        parameteriztion.  Keys are `(termType, basisLabel1, <basisLabel2>)`
        tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
        (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
        have a single basis label (so key is a 2-tuple) whereas Stochastic
        tuples with 1 basis label indicate a *diagonal* term, and are the
        only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
        Stochastic term tuples can include 2 basis labels to specify
        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
        strings or integers.  Values are complex coefficients.

    basis : Basis, optional
        A basis mapping the labels used in the keys of `lindblad_term_dict` to
        basis matrices (e.g. numpy arrays or Scipy sparse matrices).

    param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
        Describes how the Lindblad coefficients/projections relate to the
        error generator's parameter values.  Allowed values are:
        `"unconstrained"` (coeffs are independent unconstrained parameters),
        `"cptp"` (independent parameters but constrained so map is CPTP),
        `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
        `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

    nonham_mode : {"diagonal", "diag_affine", "all"}
        Which non-Hamiltonian Lindblad projections are potentially non-zero.
        Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
        `"diag_affine"` (diagonal coefficients + affine projections), and
        `"all"` (the entire matrix of coefficients is allowed).

    truncate : bool, optional
        Whether to truncate the projections onto the Lindblad terms in
        order to meet constraints (e.g. to preserve CPTP) when necessary.
        If False, then an error is thrown when the given dictionary of
        Lindblad terms doesn't conform to the constrains.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis for this error generator's linear mapping. Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    evotype : {"densitymx","svterm","cterm"}
        The evolution type of the error generator being constructed.
        `"densitymx"` means the usual Lioville density-matrix-vector
        propagation via matrix-vector products.  `"svterm"` denotes
        state-vector term-based evolution (action of operation is obtained by
        evaluating the rank-1 terms up to some order).  `"cterm"` is similar
        but uses Clifford operation action on stabilizer states.
    """

    @classmethod
    def from_operation_matrix(cls, op_matrix, parameterization='CPTP', lindblad_basis='pp',
                              mx_basis='pp', truncate=True, evotype="default", state_space=None):
        """
        Creates a Lindblad-parameterized error generator from an operation.

        Here "operation" means the exponentiated error generator, so this method
        essentially takes the matrix log of `op_matrix` and constructs an error
        generator from this using :method:`from_error_generator`.

        Parameters
        ----------
        op_matrix : numpy array or SciPy sparse matrix
            a square 2D array that gives the raw operation matrix, assumed to
            be in the `mx_basis` basis, to parameterize.  The shape of this
            array sets the dimension of the operation. If None, then it is assumed
            equal to `unitary_postfactor` (which cannot also be None). The
            quantity `op_matrix inv(unitary_postfactor)` is parameterized via
            projection onto the Lindblad terms.

        ham_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        nonham_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the non-Hamiltonian (generalized
            Stochastic-type) lindblad error Allowed values are Matrix-unit
            (std), Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt), list of
            numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            operation's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `operation` cannot
            be realized by the specified set of Lindblad projections.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        evotype : Evotype or str, optional
            The evolution type.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        state_space : TODO docstring

        Returns
        -------
        LindbladOp
        """

        #Compute an errorgen from the given op_matrix. Works with both
        # dense and sparse matrices.

        sparseOp = _sps.issparse(op_matrix)

        #Init base from error generator: sets basis members and ultimately
        # the parameters in self.paramvals
        if sparseOp:
            #Instead of making error_generator(...) compatible with sparse matrices
            # we require sparse matrices to have trivial initial error generators
            # or we convert to dense:
            if _mt.safe_norm(op_matrix - _sps.identity(op_matrix.shape[0], 'd')) < 1e-8:
                errgenMx = _sps.csr_matrix(op_matrix.shape, dtype='d')  # all zeros
            else:
                errgenMx = _sps.csr_matrix(
                    _ot.error_generator(op_matrix.toarray(), _np.identity(op_matrix.shape[0], 'd'),
                                        mx_basis, "logGTi"), dtype='d')
        else:
            errgenMx = _ot.error_generator(op_matrix, _np.identity(op_matrix.shape[0], 'd'),
                                           mx_basis, "logGTi")
        return cls.from_error_generator(errgenMx, parameterization, lindblad_basis,
                                        mx_basis, truncate, evotype, state_space=state_space)

    @classmethod
    def from_error_generator(cls, errgen_or_dim, parameterization="CPTP", lindblad_basis='pp', mx_basis='pp',
                             truncate=True, evotype="default", state_space=None):
        """
        TODO: docstring - take from now-private version below Note: errogen_or_dim can be an integer => zero errgen
        """
        parameterization = LindbladParameterization.cast(parameterization)
        ham_basis = lindblad_basis if parameterization.ham_params_allowed else None
        nonham_basis = lindblad_basis if parameterization.nonham_params_allowed else None

        errgen = _np.zeros((errgen_or_dim, errgen_or_dim), 'd') \
            if isinstance(errgen_or_dim, (int, _np.int64)) else errgen_or_dim
        return cls._from_error_generator(errgen, parameterization, ham_basis, nonham_basis,
                                         mx_basis, truncate, evotype, state_space)

    @classmethod
    def _from_error_generator(cls, errgen, parameterization="CPTP", ham_basis="pp", nonham_basis="pp",
                              mx_basis="pp", truncate=True, evotype="default", state_space=None):
        """
        Create a Lindblad-form error generator from an error generator matrix and a basis.

        The basis specifies how to decompose (project) the error generator.

        Parameters
        ----------
        errgen : numpy array or SciPy sparse matrix
            a square 2D array that gives the full error generator. The shape of
            this array sets the dimension of the operator. The projections of
            this quantity onto the `ham_basis` and `nonham_basis` are closely
            related to the parameters of the error generator (they may not be
            exactly equal if, e.g `cptp=True`).

        ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        nonham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis is used to construct the non-Hamiltonian-type lindblad error
            Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt), list of numpy arrays, or a custom basis object.

        param_mode : {"unconstrained", "cptp", "depol", "reldepol"}
            Describes how the Lindblad coefficients/projections relate to the
            operation's parameter values.  Allowed values are:
            `"unconstrained"` (coeffs are independent unconstrained parameters),
            `"cptp"` (independent parameters but constrained so map is CPTP),
            `"reldepol"` (all non-Ham. diagonal coeffs take the *same* value),
            `"depol"` (same as `"reldepol"` but coeffs must be *positive*)

        nonham_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad projections are potentially non-zero.
            Allowed values are: `"diagonal"` (only the diagonal Lind. coeffs.),
            `"diag_affine"` (diagonal coefficients + affine projections), and
            `"all"` (the entire matrix of coefficients is allowed).

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
            The source and destination basis, respectively.  Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given `errgen` cannot
            be realized by the specified set of Lindblad projections.

        evotype : {"densitymx","svterm","cterm"}
            The evolution type of the error generator being constructed.
            `"densitymx"` means usual Lioville density-matrix-vector propagation
            via matrix-vector products.  `"svterm"` denotes state-vector term-
            based evolution (action of operation is obtained by evaluating the rank-1
            terms up to some order).  `"cterm"` is similar but uses Clifford operation
            action on stabilizer states.

        state_space : TODO docstring

        Returns
        -------
        LindbladErrorgen
        """

        dim = errgen.shape[0]
        if state_space is None:
            state_space = _statespace.default_space_for_dim(dim)

        #Maybe this is unnecessary now, but determine whether the bases
        # given to us are sparse or not and make them all consistent
        # (maybe this is needed by lindblad_errorgen_projections call below?)
        sparse = None
        if ham_basis is not None:
            if isinstance(ham_basis, _Basis): sparse = ham_basis.sparse
            elif isinstance(ham_basis, str): sparse = _sps.issparse(errgen)
            elif len(ham_basis) > 0: sparse = _sps.issparse(ham_basis[0])
        if sparse is None and nonham_basis is not None:
            if isinstance(nonham_basis, _Basis): sparse = nonham_basis.sparse
            elif isinstance(nonham_basis, str): sparse = _sps.issparse(errgen)
            elif len(nonham_basis) > 0: sparse = _sps.issparse(nonham_basis[0])
        if sparse is None: sparse = False  # the default

        #Create or convert bases to consistent sparsity
        if not isinstance(ham_basis, _Basis):
            # needed b/c ham_basis could be a Basis w/dim=0 which can't be cast as dim=dim
            ham_basis = _Basis.cast(ham_basis, dim, sparse=sparse)
        if not isinstance(nonham_basis, _Basis):
            nonham_basis = _Basis.cast(nonham_basis, dim, sparse=sparse)
        if not isinstance(mx_basis, _Basis):
            matrix_basis = _Basis.cast(mx_basis, dim, sparse=sparse)
        else: matrix_basis = mx_basis

        # errgen + bases => coeffs
        parameterization = LindbladParameterization.cast(parameterization)
        hamC, otherC = \
            _ot.lindblad_errorgen_projections(
                errgen, ham_basis, nonham_basis, matrix_basis, normalize=False,
                return_generators=False, other_mode=parameterization.nonham_mode, sparse=sparse)

        # coeffs + bases => Ltermdict, basis
        Ltermdict, basis = _ot.projections_to_lindblad_terms(
            hamC, otherC, ham_basis, nonham_basis, parameterization.nonham_mode)

        return cls(Ltermdict, parameterization, basis, matrix_basis, truncate, evotype, state_space)

    def __init__(self, lindblad_term_dict, parameterization='auto', lindblad_basis='pp',
                 mx_basis="pp", truncate=True, evotype="default", state_space=None):

        # FUTURE:
        # - maybe allow basisdict values to specify an "embedded matrix" w/a tuple like
        #  e.g. a *list* of (matrix, state_space_label) elements -- e.g. [(sigmaX,'Q1'), (sigmaY,'Q4')]
        # - maybe let keys be tuples of (basisname, state_space_label) e.g. (('X','Q1'),('Y','Q4')) -- and
        # maybe allow ('XY','Q1','Q4')? format when can assume single-letter labels.
        # - could add standard basis dict items so labels like "X", "XY", etc. are understood?
        state_space = _statespace.StateSpace.cast(state_space)

        #convert lindblad term dict to use *local* labels (ok to specify w/global labels)
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = state_space.tensor_product_block_labels(0)  # just take first TPB labels as all labels
        lindblad_term_dict = _collections.OrderedDict(
            [(_LocalElementaryErrorgenLabel.cast(lbl, sslbls, identity_label_1Q), val)
             for lbl, val in lindblad_term_dict.items()])

        self.parameterization = LindbladParameterization.from_lindblad_terms(lindblad_term_dict) \
            if parameterization == "auto" else LindbladParameterization.cast(parameterization)

        #Decide on our rep-type ahead of time so we know whether to make bases sparse
        # (a LindbladErrorgen with a sparse rep => sparse bases and similar with dense rep)
        evotype = _Evotype.cast(evotype)
        reptype_preferences = ('lindblad errorgen', 'dense superop', 'sparse superop') \
            if evotype.prefer_dense_reps else ('lindblad errorgen', 'sparse superop', 'dense superop')
        for reptype in reptype_preferences:
            if evotype.supports(reptype):
                self._rep_type = reptype; break
        else:
            raise ValueError("Evotype doesn't support any of the representations a LindbladErrorgen requires.")
        sparse_bases = bool(self._rep_type == 'sparse superop')  # we use sparse bases iff we have a sparse rep

        state_space = _statespace.StateSpace.cast(state_space)
        dim = state_space.dim  # Store superop dimension
        self.lindblad_basis = _Basis.cast(lindblad_basis, dim, sparse=sparse_bases)

        # lindblad_term_dict, basis => bases + parameter values
        # but maybe we want lindblad_term_dict, basisdict => basis + projections/coeffs,
        #  then projections/coeffs => paramvals? since the latter is what set_errgen needs
        hamC, otherC, self.ham_basis, self.other_basis = \
            _ot.lindblad_terms_to_projections(lindblad_term_dict, self.lindblad_basis,
                                              self.parameterization.nonham_mode)

        self.ham_basis_size = len(self.ham_basis)
        self.other_basis_size = len(self.other_basis)

        assert(self.parameterization.ham_params_allowed or self.ham_basis_size == 0), \
            "Hamiltonian lindblad terms are not allowed!"
        assert(self.parameterization.nonham_params_allowed or self.other_basis_size == 0), \
            "Non-Hamiltonian lindblad terms are not allowed!"

        # Check that bases have the desired sparseness (should be same as lindblad_basis)
        assert (self.ham_basis_size == 0 or self.ham_basis.sparse == sparse_bases)
        assert (self.other_basis_size == 0 or self.other_basis.sparse == sparse_bases)

        self.matrix_basis = _Basis.cast(mx_basis, dim, sparse=sparse_bases)

        self.paramvals = _ot.lindblad_projections_to_paramvals(
            hamC, otherC, self.parameterization.param_mode, self.parameterization.nonham_mode, truncate)

        #Fast CSR-matrix summing variables: N/A if not sparse or using terms
        self._CSRSumIndices = self._CSRSumData = self._CSRSumPtr = None

        # Generator matrices & cache qtys: N/A for term-based evotypes
        self.hamGens = self.otherGens = None
        self.hamGens_1norms = self.otherGens_1norms = None
        self._onenorm_upbound = None
        self.Lmx = None
        self._coefficient_weights = None

        #All representations need to track 1norms:
        self.hamGens, self.otherGens = self._init_generators(dim)

        if self.hamGens is not None:
            self.hamGens_1norms = _np.array([_mt.safe_onenorm(mx) for mx in self.hamGens], 'd')
        if self.otherGens is not None:
            if self.parameterization.nonham_mode == "diagonal":
                self.otherGens_1norms = _np.array([_mt.safe_onenorm(mx) for mx in self.otherGens], 'd')
            else:
                self.otherGens_1norms = _np.array([_mt.safe_onenorm(mx)
                                                   for oGenRow in self.otherGens for mx in oGenRow], 'd')

        #Create a representation of the type chosen above:
        if self._rep_type == 'lindblad errorgen':
            rep = evotype.create_lindblad_errorgen_rep(lindblad_term_dict, self.lindblad_basis, state_space)

        else:  # Otherwise create a sparse or dense matrix representation

            #Allocate space fo Cholesky mx (used in _construct_errgen_matrix)
            # (intermediate storage for matrix and for deriv computation)
            bsO = self.other_basis_size
            self.Lmx = _np.zeros((bsO - 1, bsO - 1), 'complex') if bsO > 0 else None

            if sparse_bases:  # then construct a sparse-matrix representation (self._rep_type == 'sparse superop')
                #Precompute for faster CSR sums in _construct_errgen
                all_csr_matrices = []
                if self.hamGens is not None:
                    all_csr_matrices.extend(self.hamGens)

                if self.otherGens is not None:
                    if self.parameterization.nonham_mode == "diagonal":
                        oList = self.otherGens
                    else:  # nonham_mode in ("diag_affine", "all")
                        oList = [mx for mxRow in self.otherGens for mx in mxRow]
                    all_csr_matrices.extend(oList)

                flat_dest_indices, flat_src_data, flat_nnzptr, indptr, indices, N = \
                    _mt.csr_sum_flat_indices(all_csr_matrices)
                self._CSRSumIndices = flat_dest_indices
                self._CSRSumData = flat_src_data
                self._CSRSumPtr = flat_nnzptr

                self._data_scratch = _np.zeros(len(indices), complex)  # *complex* scratch space for updating rep
                rep = evotype.create_sparse_rep(_np.ascontiguousarray(_np.zeros(len(indices), 'd')),
                                                _np.ascontiguousarray(indices, _np.int64),
                                                _np.ascontiguousarray(indptr, _np.int64),
                                                state_space)
            else:  # self._rep_type = 'dense superop'
                rep = evotype.create_dense_superop_rep(None, state_space)

        _LinearOperator.__init__(self, rep, evotype)  # sets self.dim
        self._update_rep()  # updates _rep whether it's a dense or sparse matrix
        self._paramlbls = _np.array(_ot.lindblad_param_labels(self.ham_basis, self.other_basis,
                                                              self.parameterization.param_mode,
                                                              self.parameterization.nonham_mode), dtype=object)
        assert(self._onenorm_upbound is not None)  # _update_rep should set this
        #Done with __init__(...)

    def _init_generators(self, dim):
        #assumes self.dim, self.ham_basis, self.other_basis, and self.matrix_basis are setup...

        d = int(round(_np.sqrt(dim)))
        assert(d * d == dim), "Errorgen dim must be a perfect square"

        # Get basis transfer matrix
        sparse_bases = bool(self._rep_type == 'sparse superop')
        mxBasisToStd = self.matrix_basis.create_transform_matrix(
            _BuiltinBasis("std", self.matrix_basis.dim, sparse_bases))
        # use BuiltinBasis("std") instead of just "std" in case matrix_basis is a TensorProdBasis
        leftTrans = _spsl.inv(mxBasisToStd.tocsc()).tocsr() if _sps.issparse(mxBasisToStd) \
            else _np.linalg.inv(mxBasisToStd)
        rightTrans = mxBasisToStd

        hamBasisMxs = self.ham_basis.elements
        otherBasisMxs = self.other_basis.elements

        hamGens, otherGens = _ot.lindblad_error_generators(
            hamBasisMxs, otherBasisMxs, normalize=False,
            other_mode=self.parameterization.nonham_mode)  # in std basis

        # Note: lindblad_error_generators will return sparse generators when
        #  given a sparse basis (or basis matrices)

        if hamGens is not None:
            bsH = len(hamGens) + 1  # projection-basis size (not nec. == dim)
            _ot._assert_shape(hamGens, (bsH - 1, dim, dim), sparse_bases)

            # apply basis change now, so we don't need to do so repeatedly later
            if sparse_bases:
                hamGens = [_mt.safe_real(_mt.safe_dot(leftTrans, _mt.safe_dot(mx, rightTrans)),
                                         inplace=True, check=True) for mx in hamGens]
                for mx in hamGens: mx.sort_indices()
                # for faster addition ops in _construct_errgen_matrix
            else:
                #hamGens = _np.einsum("ik,akl,lj->aij", leftTrans, hamGens, rightTrans)
                hamGens = _np.transpose(_np.tensordot(
                    _np.tensordot(leftTrans, hamGens, (1, 1)), rightTrans, (2, 0)), (1, 0, 2))
        else:
            bsH = 0
        assert(bsH == self.ham_basis_size)

        if otherGens is not None:

            if self.parameterization.nonham_mode == "diagonal":
                bsO = len(otherGens) + 1  # projection-basis size (not nec. == dim)
                _ot._assert_shape(otherGens, (bsO - 1, dim, dim), sparse_bases)

                # apply basis change now, so we don't need to do so repeatedly later
                if sparse_bases:
                    otherGens = [_mt.safe_real(_mt.safe_dot(leftTrans, _mt.safe_dot(mx, rightTrans)),
                                               inplace=True, check=True) for mx in otherGens]
                    for mx in otherGens: mx.sort_indices()
                    # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,akl,lj->aij", leftTrans, otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 1)), rightTrans, (2, 0)), (1, 0, 2))

            elif self.parameterization.nonham_mode == "diag_affine":
                # projection-basis size (not nec. == dim) [~shape[1] but works for lists too]
                bsO = len(otherGens[0]) + 1
                _ot._assert_shape(otherGens, (2, bsO - 1, dim, dim), sparse_bases)

                # apply basis change now, so we don't need to do so repeatedly later
                if sparse_bases:
                    otherGens = [[_mt.safe_dot(leftTrans, _mt.safe_dot(mx, rightTrans))
                                  for mx in mxRow] for mxRow in otherGens]

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                        # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                          otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 2)), rightTrans, (3, 0)), (1, 2, 0, 3))

            else:
                bsO = len(otherGens) + 1  # projection-basis size (not nec. == dim)
                _ot._assert_shape(otherGens, (bsO - 1, bsO - 1, dim, dim), sparse_bases)

                # apply basis change now, so we don't need to do so repeatedly later
                if sparse_bases:
                    otherGens = [[_mt.safe_dot(leftTrans, _mt.safe_dot(mx, rightTrans))
                                  for mx in mxRow] for mxRow in otherGens]
                    #Note: complex OK here, as only linear combos of otherGens (like (i,j) + (j,i)
                    # terms) need to be real

                    for mxRow in otherGens:
                        for mx in mxRow: mx.sort_indices()
                        # for faster addition ops in _construct_errgen_matrix
                else:
                    #otherGens = _np.einsum("ik,abkl,lj->abij", leftTrans,
                    #                            otherGens, rightTrans)
                    otherGens = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, otherGens, (1, 2)), rightTrans, (3, 0)), (1, 2, 0, 3))

        else:
            bsO = 0
        assert(bsO == self.other_basis_size)
        return hamGens, otherGens

    def _init_terms(self, lindblad_term_dict, basis, dim, max_polynomial_vars):

        # needed b/c operators produced by lindblad_error_generators have an extra 'd' scaling
        d = int(round(_np.sqrt(dim)))
        mpv = max_polynomial_vars

        # Lookup dictionaries for getting the *parameter* index associated
        # with a particlar basis label.  The -1 to compensates for the fact
        # that the identity element is the first element of each non-empty basis
        # and this does not have a correponding parameter/projection.
        hamBasisIndices = {lbl: i - 1 for i, lbl in enumerate(self.ham_basis.labels)}
        otherBasisIndices = {lbl: i - 1 for i, lbl in enumerate(self.other_basis.labels)}

        # as we expect `basis` will contain *dense* basis
        # matrices (maybe change in FUTURE?)
        numHamParams = max(len(hamBasisIndices) - 1, 0)  # compensate for first basis el,
        numOtherBasisEls = max(len(otherBasisIndices) - 1, 0)  # being the identity. (if there are any els at all)

        # Create Lindbladian terms - rank1 terms in the *exponent* with polynomial
        # coeffs (w/ *local* variable indices) that get converted to per-order
        # terms later.
        IDENT = None  # sentinel for the do-nothing identity op
        Lterms = []
        for termLbl in lindblad_term_dict:  # assumed to contain *local* elem errgen labels as keys
            termType = termLbl.errorgen_type
            bels = termLbl.basis_element_labels
            if termType == "H":  # Hamiltonian
                k = hamBasisIndices[bels[0]]  # index of parameter
                # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
                scale, U = _mt.to_unitary(basis[bels[0]])
                scale *= _np.sqrt(d) / 2  # mimics rho1's _np.sqrt(d) / 2 scaling in `hamiltonian_to_lindbladian`
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({(k,): -1j * scale}, mpv), U, IDENT, self._evotype, self.state_space))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(_Polynomial({(k,): +1j * scale}, mpv),
                                                                        IDENT, U.conjugate().T, self._evotype,
                                                                        self.state_space))

            elif termType == "S":  # Stochastic
                if self.parameterization.nonham_mode in ("diagonal", "diag_affine"):
                    if self.parameterization.param_mode in ("depol", "reldepol"):
                        # => same single param for all stochastic terms
                        k = numHamParams + 0  # index of parameter
                    else:
                        k = numHamParams + otherBasisIndices[bels[0]]  # index of parameter
                    scale, U = _mt.to_unitary(basis[bels[0]])  # ensure all Rank1Term operators are *unitary*
                    scale *= _np.sqrt(d)  # mimics "rho1 *= d" scaling in `nonham_lindbladian`
                    Lm = Ln = U
                    # power to raise parameter to in order to get coeff
                    pw = 2 if self.parameterization.param_mode in ("cptp", "depol") else 1

                    Lm_dag = Lm.conjugate().T
                    # assumes basis is dense (TODO: make sure works for sparse case too - and np.dots below!)
                    Ln_dag = Ln.conjugate().T
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        _Polynomial({(k,) * pw: 1.0 * scale**2}, mpv), Ln, Lm_dag, self._evotype, self.state_space
                    ))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        _Polynomial({(k,) * pw: -0.5 * scale**2}, mpv), IDENT, _np.dot(Ln_dag, Lm),
                        self._evotype, self.state_space
                    ))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        _Polynomial({(k,) * pw: -0.5 * scale**2}, mpv), _np.dot(Lm_dag, Ln), IDENT,
                        self._evotype, self.state_space
                    ))

                else:
                    i = otherBasisIndices[bels[0]]  # index of row in "other" coefficient matrix
                    j = otherBasisIndices[bels[1]]  # index of col in "other" coefficient matrix
                    scalem, Um = _mt.to_unitary(basis[bels[0]])  # ensure all Rank1Term operators are *unitary*
                    scalen, Un = _mt.to_unitary(basis[bels[1]])  # ensure all Rank1Term operators are *unitary*
                    Lm, Ln = Um, Un
                    scale = scalem * scalen
                    scale *= d  # mimics "rho1 *= d" scaling in `nonham_lindbladian`

                    # TODO: create these polys and place below...
                    polyTerms = {}
                    assert(self.parameterization.param_mode != "depol"), \
                        "`depol` mode not supported when nonham_mode=='all'"
                    assert(self.parameterization.param_mode != "reldepol"), \
                        "`reldepol` mode not supported when nonham_mode=='all'"
                    if self.parameterization.param_mode == "cptp":
                        # otherCoeffs = _np.dot(self.Lmx,self.Lmx.T.conjugate())
                        # coeff_ij = sum_k Lik * Ladj_kj = sum_k Lik * conjugate(L_jk)
                        #          = sum_k (Re(Lik) + 1j*Im(Lik)) * (Re(L_jk) - 1j*Im(Ljk))
                        def i_re(a, b): return numHamParams + (a * numOtherBasisEls + b)
                        def i_im(a, b): return numHamParams + (b * numOtherBasisEls + a)
                        for k in range(0, min(i, j) + 1):
                            if k <= i and k <= j:
                                polyTerms[(i_re(i, k), i_re(j, k))] = 1.0
                            if k <= i and k < j:
                                polyTerms[(i_re(i, k), i_im(j, k))] = -1.0j
                            if k < i and k <= j:
                                polyTerms[(i_im(i, k), i_re(j, k))] = 1.0j
                            if k < i and k < j:
                                polyTerms[(i_im(i, k), i_im(j, k))] = 1.0
                    else:  # param_mode == "unconstrained"
                        # coeff_ij = otherParam[i,j] + 1j*otherParam[j,i] (otherCoeffs is Hermitian)
                        ijIndx = numHamParams + (i * numOtherBasisEls + j)
                        jiIndx = numHamParams + (j * numOtherBasisEls + i)
                        polyTerms = {(ijIndx,): 1.0, (jiIndx,): 1.0j}

                    base_poly = _Polynomial(polyTerms, mpv)
                    Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(1.0 * base_poly * scale, Ln, Lm,
                                                                            self._evotype, self.state_space))
                    # adjoint(_np.dot(Lm_dag,Ln))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        -0.5 * base_poly * scale, IDENT, _np.dot(Ln_dag, Lm), self._evotype, self.state_space
                    ))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        -0.5 * base_poly * scale, _np.dot(Lm_dag, Ln), IDENT, self._evotype, self.state_space
                    ))

            elif termType == "A":  # Affine
                assert(self.parameterization.nonham_mode == "diag_affine")
                if self.parameterization.param_mode in ("depol", "reldepol"):
                    # => same single param for all stochastic terms
                    k = numHamParams + 1 + otherBasisIndices[bels[0]]  # index of parameter
                else:
                    k = numHamParams + numOtherBasisEls + otherBasisIndices[bels[0]]  # index of parameter

                # rho -> basis[bels[0]] * I = basis[bels[0]] * sum{ P_i rho P_i } where Pi's
                #  are the normalized paulis (including the identity), and rho has trace == 1
                #  (all but "I/d" component of rho are annihilated by pauli sum; for the I/d component, all
                #   d^2 of the terms in the sum is P/sqrt(d) * I/d * P/sqrt(d) == I/d^2, so the result is just "I")
                scale, U = _mt.to_unitary(basis[bels[0]])  # ensure all Rank1Term operators are *unitary*
                L = U
                # Note: only works when `d` corresponds to integral # of qubits!
                Bmxs = _bt.basis_matrices("pp", dim, sparse=False)
                # scaling to make Bmxs unitary (reverse of `scale` above, where scale * U == basis[.])
                Bscale = dim**0.25

                for B in Bmxs:  # Note: *include* identity! (see pauli scratch notebook for details)
                    UB = Bscale * B  # UB is unitary
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        _Polynomial({(k,): 1.0 * scale / Bscale**2}, mpv),
                        _np.dot(L, UB), UB, self._evotype, self.state_space))  # /(dim-1.)

                #TODO: check normalization of these terms vs those used in projections.

        #DEBUG
        #print("DB: params = ", list(enumerate(self.paramvals)))
        #print("DB: Lterms = ")
        #for i,lt in enumerate(Lterms):
        #    print("Term %d:" % i)
        #    print("  coeff: ", str(lt.coeff)) # list(lt.coeff.keys()) )
        #    print("  pre:\n", lt.pre_ops[0] if len(lt.pre_ops) else "IDENT")
        #    print("  post:\n",lt.post_ops[0] if len(lt.post_ops) else "IDENT")

        #Make compact polys that are ready to (repeatedly) evaluate (useful
        # for term-based calcs which call total_term_magnitude() a lot)
        poly_coeffs = [t.coeff for t in Lterms]
        tapes = [poly.compact(complex_coeff_tape=True) for poly in poly_coeffs]
        if len(tapes) > 0:
            vtape = _np.concatenate([t[0] for t in tapes])
            ctape = _np.concatenate([t[1] for t in tapes])
        else:
            vtape = _np.empty(0, _np.int64)
            ctape = _np.empty(0, complex)
        coeffs_as_compact_polys = (vtape, ctape)

        #DEBUG TODO REMOVE (and make into test) - check norm of rank-1 terms
        # (Note: doesn't work for Clifford terms, which have no .base):
        # rho =OP=> coeff * A rho B
        # want to bound | coeff * Tr(E Op rho) | = | coeff | * | <e|A|psi><psi|B|e> |
        # so A and B should be unitary so that | <e|A|psi><psi|B|e> | <= 1
        # but typically these are unitaries / (sqrt(2)*nqubits)
        #import bpdb; bpdb.set_trace()
        #scale = 1.0
        #for t in Lterms:
        #    for op in t._rep.pre_ops:
        #        test = _np.dot(_np.conjugate(scale * op.base.T), scale * op.base)
        #        assert(_np.allclose(test, _np.identity(test.shape[0], 'd')))
        #    for op in t._rep.post_ops:
        #        test = _np.dot(_np.conjugate(scale * op.base.T), scale * op.base)
        #        assert(_np.allclose(test, _np.identity(test.shape[0], 'd')))

        return Lterms, coeffs_as_compact_polys

    def _set_params_from_matrix(self, errgen, truncate):
        """ Sets self.paramvals based on `errgen` """
        hamC, otherC = \
            _ot.lindblad_errorgen_projections(
                errgen, self.ham_basis, self.other_basis, self.matrix_basis, normalize=False,
                return_generators=False, other_mode=self.parameterization.nonham_mode,
                sparse=bool(self._rep_type == 'sparse superop'))  # in std basis

        self.paramvals = _ot.lindblad_projections_to_paramvals(
            hamC, otherC, self.parameterization.param_mode, self.parameterization.nonham_mode, truncate)
        self._update_rep()

    def _update_rep(self):
        """
        Updates self._rep, which contains a representation of this error generator
        as either a dense or sparse matrix.  This routine essentially builds the
        error generator matrix using the current parameters and updates self._rep
        accordingly (by rewriting its data).
        """
        dim = self.dim
        hamCoeffs, otherCoeffs = _ot.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.parameterization.param_mode, self.parameterization.nonham_mode, self.Lmx)
        onenorm = 0.0

        #Finally, build operation matrix from generators and coefficients:
        if self._rep_type == 'lindblad errorgen':
            # the code below is for updating sparse or dense matrix representations.  If our
            # evotype has a native Lindblad representation, maybe in the FUTURE we should
            # call an update method of it here?

            #Still need to update onenorm - FUTURE: maybe put this logic inside rep?
            if hamCoeffs is not None:
                onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
            if otherCoeffs is not None:
                onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))

        elif self._rep_type == 'sparse superop':  # then bases & errgen are sparse
            coeffs = None
            data = self._data_scratch
            data.fill(0.0)  # data starts at zero

            if hamCoeffs is not None:
                onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
                if otherCoeffs is not None:
                    onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))
                    coeffs = _np.concatenate((hamCoeffs, otherCoeffs.flat), axis=0)
                else:
                    coeffs = hamCoeffs
            elif otherCoeffs is not None:
                onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))
                coeffs = otherCoeffs.flatten()

            if coeffs is not None:
                _mt.csr_sum_flat(data, coeffs, self._CSRSumIndices, self._CSRSumData, self._CSRSumPtr)

            #Don't perform this check as this function is called a *lot* and it
            # could adversely impact performance
            #assert(_np.isclose(_np.linalg.norm(data.imag), 0)), \
            #    "Imaginary error gen norm: %g" % _np.linalg.norm(data.imag)

            #Update the rep's sparse matrix data stored in self._rep_data (the rep already
            # has the correct sparse matrix structure, as given by indices and indptr in
            # __init__, so we just update the *data* array).
            self._rep.data[:] = data.real

        else:  # dense matrices
            if hamCoeffs is not None:
                #lnd_error_gen = _np.einsum('i,ijk', hamCoeffs, self.hamGens)
                lnd_error_gen = _np.tensordot(hamCoeffs, self.hamGens, (0, 0))
                onenorm += _np.dot(self.hamGens_1norms, _np.abs(hamCoeffs))
            else:
                lnd_error_gen = _np.zeros((dim, dim), 'complex')

            if otherCoeffs is not None:
                if self.parameterization.nonham_mode == "diagonal":
                    #lnd_error_gen += _np.einsum('i,ikl', otherCoeffs, self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, (0, 0))
                    onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs))

                else:  # nonham_mode in ("diag_affine", "all")
                    #lnd_error_gen += _np.einsum('ij,ijkl', otherCoeffs,
                    #                            self.otherGens)
                    lnd_error_gen += _np.tensordot(otherCoeffs, self.otherGens, ((0, 1), (0, 1)))
                    onenorm += _np.dot(self.otherGens_1norms, _np.abs(otherCoeffs.flat))

            assert(_np.isclose(_np.linalg.norm(lnd_error_gen.imag), 0)), \
                "Imaginary error gen norm: %g" % _np.linalg.norm(lnd_error_gen.imag)
            #print("errgen pre-real = \n"); _mt.print_mx(lnd_error_gen,width=4,prec=1)
            self._rep.base[:, :] = lnd_error_gen.real

        self._onenorm_upbound = onenorm
        #assert(self._onenorm_upbound >= _np.linalg.norm(self.to_dense(), ord=1) - 1e-6)  #DEBUG

    def to_dense(self, on_space='minimal'):
        """
        Return this error generator as a dense matrix.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        if self._rep_type == 'lindblad errorgen':
            assert(on_space in ('minimal', 'HilbertSchmidt'))
            #Then we need to do similar things to __init__ for a dense rep - maybe consolidate?
            hamCoeffs, otherCoeffs = _ot.paramvals_to_lindblad_projections(
                self.paramvals, self.ham_basis_size, self.other_basis_size,
                self.parameterization.param_mode, self.parameterization.nonham_mode)

            hamGens, otherGens = self._init_generators(self.dim)

            if hamCoeffs is not None:
                lnd_error_gen = _np.tensordot(hamCoeffs, hamGens, (0, 0))
            else:
                lnd_error_gen = _np.zeros((self.dim, self.dim), 'complex')

            if otherCoeffs is not None:
                if self.parameterization.nonham_mode == "diagonal":
                    lnd_error_gen += _np.tensordot(otherCoeffs, otherGens, (0, 0))
                else:  # nonham_mode in ("diag_affine", "all")
                    lnd_error_gen += _np.tensordot(otherCoeffs, otherGens, ((0, 1), (0, 1)))

            assert(_np.isclose(_np.linalg.norm(lnd_error_gen.imag), 0)), \
                "Imaginary error gen norm: %g" % _np.linalg.norm(lnd_error_gen.imag)
            return lnd_error_gen.real

        elif self._rep_type == 'sparse superop':
            return self.to_sparse(on_space).toarray()
        else:  # dense rep
            return self._rep.to_dense(on_space)

    def to_sparse(self, on_space='minimal'):
        """
        Return the error generator as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if self._rep_type == 'lindblad errorgen':
            assert(on_space in ('minimal', 'HilbertSchmidt'))
            #Need to do similar things to __init__ - maybe consolidate?
            hamCoeffs, otherCoeffs = _ot.paramvals_to_lindblad_projections(
                self.paramvals, self.ham_basis_size, self.other_basis_size,
                self.parameterization.param_mode, self.parameterization.nonham_mode)

            hamGens, otherGens = self._init_generators(self.dim)

            if hamCoeffs is not None:
                lnd_error_gen = sum([c * gen for c, gen in zip(hamCoeffs, hamGens)])
            else:
                lnd_error_gen = _sps.csr_matrix((self.dim, self.dim))

            if otherCoeffs is not None:
                if self.parameterization.nonham_mode == "diagonal":
                    lnd_error_gen += sum([c * gen for c, gen in zip(otherCoeffs, otherGens)])
                else:  # nonham_mode in ("diag_affine", "all")
                    lnd_error_gen += sum([c * gen for cRow, genRow in zip(otherCoeffs, otherGens)
                                          for c, gen in zip(cRow, genRow)])

            return lnd_error_gen
        elif self._rep_type == 'sparse superop':
            assert(on_space in ('minimal', 'HilbertSchmidt'))
            return _sps.csr_matrix((self._rep.data, self._rep.indices, self._rep.indptr),
                                   shape=(self.dim, self.dim))
        else:  # dense rep
            return _sps.csr_matrix(self.to_dense(on_space))

    #def torep(self):
    #    """
    #    Return a "representation" object for this error generator.
    #
    #    Such objects are primarily used internally by pyGSTi to compute
    #    things like probabilities more efficiently.
    #
    #    Returns
    #    -------
    #    OpRep
    #    """
    #    if self._evotype == "densitymx":
    #        if self._rep_type == 'sparse superop':
    #            A = self.err_gen_mx
    #            return replib.DMOpRepSparse(
    #                _np.ascontiguousarray(A.data),
    #                _np.ascontiguousarray(A.indices, _np.int64),
    #                _np.ascontiguousarray(A.indptr, _np.int64))
    #        else:
    #            return replib.DMOpRepDense(_np.ascontiguousarray(self.err_gen_mx, 'd'))
    #    else:
    #        raise NotImplementedError("torep(%s) not implemented for %s objects!" %
    #                                  (self._evotype, self.__class__.__name__))

    def taylor_order_terms(self, order, max_polynomial_vars=100, return_coeff_polys=False):
        """
        Get the `order`-th order Taylor-expansion terms of this operation.

        This function either constructs or returns a cached list of the terms at
        the given order.  Each term is "rank-1", meaning that its action on a
        density matrix `rho` can be written:

        `rho -> A rho B`

        The coefficients of these terms are typically polynomials of the operation's
        parameters, where the polynomial's variable indices index the *global*
        parameters of the operation's parent (usually a :class:`Model`), not the
        operation's local parameter array (i.e. that returned from `to_vector`).

        Parameters
        ----------
        order : int
            The order of terms to get.

        max_polynomial_vars : int, optional
            maximum number of variables the created polynomials can have.

        return_coeff_polys : bool
            Whether a parallel list of locally-indexed (using variable indices
            corresponding to *this* object's parameters rather than its parent's)
            polynomial coefficients should be returned as well.

        Returns
        -------
        terms : list
            A list of :class:`RankOneTerm` objects.
        coefficients : list
            Only present when `return_coeff_polys == True`.
            A list of *compact* polynomial objects, meaning that each element
            is a `(vtape,ctape)` 2-tuple formed by concatenating together the
            output of :method:`Polynomial.compact`.
        """
        assert(self._rep_type == 'lindblad errorgen'), \
            "Only evotypes with native Lindblad errorgen representations can utilize Taylor terms"
        assert(order == 0), \
            "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        assert(return_coeff_polys is False)
        if self._rep.Lterms is None:
            Ltermdict, basis = self._rep.LtermdictAndBasis
            self._rep.Lterms, self._rep.Lterm_coeffs = self._init_terms(Ltermdict, basis,
                                                                        self.dim, max_polynomial_vars)
        return self._rep.Lterms  # terms with local-index polynomial coefficients

    #def get_direct_order_terms(self, order): # , order_base=None - unused currently b/c order is always 0...
    #    v = self.to_vector()
    #    poly_terms = self.get_taylor_order_terms(order)
    #    return [ term.evaluate_coeff(v) for term in poly_terms ]

    @property
    def total_term_magnitude(self):
        """
        Get the total (sum) of the magnitudes of all this operator's terms.

        The magnitude of a term is the absolute value of its coefficient, so
        this function returns the number you'd get from summing up the
        absolute-coefficients of all the Taylor terms (at all orders!) you
        get from expanding this operator in a Taylor series.

        Returns
        -------
        float
        """
        # return (sum of absvals of term coeffs)
        assert(self._rep.Lterms is not None), "Must call `taylor_order_terms` before calling total_term_magnitude!"
        vtape, ctape = self._rep.Lterm_coeffs
        return _abs_sum_bulk_eval_compact_polynomials_complex(vtape, ctape, self.to_vector(), len(self._rep.Lterms))

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this operator's terms.

        Computes the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        # In general: d(|x|)/dp = d( sqrt(x.r^2 + x.im^2) )/dp = (x.r*dx.r/dp + x.im*dx.im/dp) / |x| = Re(x * conj(dx/dp))/|x|  # noqa: E501
        # The total term magnitude in this case is sum_i( |coeff_i| ) so we need to compute:
        # d( sum_i( |coeff_i| )/dp = sum_i( d(|coeff_i|)/dp ) = sum_i( Re(coeff_i * conj(d(coeff_i)/dp)) / |coeff_i| )

        wrtInds = _np.ascontiguousarray(_np.arange(self.num_params), _np.int64)  # for Cython arg mapping
        vtape, ctape = self._rep.Lterm_coeffs
        coeff_values = _bulk_eval_compact_polynomials_complex(vtape, ctape, self.to_vector(), (len(self._rep.Lterms),))
        coeff_deriv_polys = _compact_deriv(vtape, ctape, wrtInds)
        coeff_deriv_vals = _bulk_eval_compact_polynomials_complex(coeff_deriv_polys[0], coeff_deriv_polys[1],
                                                                  self.to_vector(), (len(self._rep.Lterms),
                                                                                     len(wrtInds)))
        abs_coeff_values = _np.abs(coeff_values)
        abs_coeff_values[abs_coeff_values < 1e-10] = 1.0  # so ratio is 0 in cases where coeff_value == 0
        ret = _np.sum(_np.real(coeff_values[:, None] * _np.conj(coeff_deriv_vals))
                      / abs_coeff_values[:, None], axis=0)  # row-sum
        assert(_np.linalg.norm(_np.imag(ret)) < 1e-8)
        return ret.real

        #DEBUG
        #ret2 = _np.empty(self.num_params,'d')
        #eps = 1e-8
        #orig_vec = self.to_vector().copy()
        #f0 = sum([abs(coeff) for coeff in coeff_values])
        #for i in range(self.num_params):
        #    v = orig_vec.copy()
        #    v[i] += eps
        #    new_coeff_values = _bulk_eval_compact_polynomials_complex(vtape, ctape, v, (len(self.Lterms),))
        #    ret2[i] = ( sum([abs(coeff) for coeff in new_coeff_values]) - f0 ) / eps

        #test3 = _np.linalg.norm(ret-ret2)
        #print("TEST3 = ",test3)
        #if test3 > 10.0:
        #    import bpdb; bpdb.set_trace()
        #return ret

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.paramvals)

    def to_vector(self):
        """
        Extract a vector of the underlying operation parameters from this operation.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        return self.paramvals

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
        assert(len(v) == self.num_params)
        self.paramvals[:] = v
        self._update_rep()
        self.dirty = dirty_value

    def coefficients(self, return_basis=False, logscale_nonham=False):
        """
        Constructs a dictionary of the Lindblad-error-generator coefficients of this error generator.

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
        hamC, otherC = _ot.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.parameterization.param_mode, self.parameterization.nonham_mode, self.Lmx)

        Ltermdict_and_maybe_basis = _ot.projections_to_lindblad_terms(
            hamC, otherC, self.ham_basis, self.other_basis, self.parameterization.nonham_mode, return_basis)

        #convert to *global* lindblad terms
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.tensor_product_block_labels(0)  # just take first TPB labels as all labels
        Ltermdict, basis = Ltermdict_and_maybe_basis if return_basis else (Ltermdict_and_maybe_basis, None)
        Ltermdict = _collections.OrderedDict([(_GlobalElementaryErrorgenLabel.cast(k, sslbls, identity_label_1Q), v)
                                              for k, v in Ltermdict.items()])

        if logscale_nonham:
            dim = self.dim
            for k in Ltermdict.keys():
                if k.errorgen_type == "S":  # reverse mapping: err_coeff -> err_rate
                    Ltermdict[k] = (1 - _np.exp(-dim * Ltermdict[k])) / dim  # err_rate = (1-exp(-d^2*errgen_coeff))/d^2

        return (Ltermdict, basis) if return_basis else Ltermdict

    def coefficient_labels(self):
        """
        The elementary error-generator labels corresponding to the elements of :method:`coefficients_array`.

        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        labels = []
        LEEL = _LocalElementaryErrorgenLabel  # shorthand
        if self.ham_basis is not None:
            labels.extend([LEEL('H', (basis_lbl,)) for basis_lbl in self.ham_basis.labels[1:]])
        if self.other_basis is not None:
            if self.nonham_mode == "diagonal":
                labels.extend([LEEL('S', (basis_lbl,)) for basis_lbl in self.other_basis.labels[1:]])
            elif self.nonham_mode == "diag_affine":
                labels.extend([LEEL('S', (basis_lbl,)) for basis_lbl in self.other_basis.labels[1:]])
                labels.extend([LEEL('A', (basis_lbl,)) for basis_lbl in self.other_basis.labels[1:]])
            else:  # 'all' mode
                labels.extend([LEEL('S', (basis_lbl1, basis_lbl2))
                               for basis_lbl1 in self.other_basis.labels[1:]
                               for basis_lbl2 in self.other_basis.labels[1:]])

        #convert to *global* lindblad terms
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.tensor_product_block_labels(0)  # just take first TPB labels as all labels
        labels = [_GlobalElementaryErrorgenLabel.cast(local_lbl, sslbls, identity_label_1Q) for local_lbl in labels]

        return tuple(labels)

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
        hamC, otherC = _ot.paramvals_to_lindblad_projections(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.parameterization.param_mode, self.parameterization.nonham_mode, self.Lmx)

        ret = _np.concatenate((hamC, otherC.flat))  # will be complex if otherC is
        if self._coefficient_weights is not None:
            ret *= self._coefficient_weights
        return ret

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
        hamCderiv, otherCderiv = _ot.paramvals_to_lindblad_projections_deriv(
            self.paramvals, self.ham_basis_size, self.other_basis_size,
            self.parameterization.param_mode, self.parameterization.nonham_mode, self.Lmx)

        if otherCderiv.ndim == 3:  # (coeff_dim_1, coeff_dim_2, param_dim) => (coeff_dim, param_dim)
            otherCderiv = otherCderiv.reshape((otherCderiv.shape[0] * otherCderiv.shape[1], otherCderiv.shape[2]))

        ret = _np.concatenate((hamCderiv, otherCderiv), axis=0)
        if self._coefficient_weights is not None:
            ret *= self._coefficient_weights[:, None]
        return ret

    def error_rates(self):
        """
        Constructs a dictionary of the error rates associated with this error generator.

        The error rates pertain to the *channel* formed by exponentiating this object.

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

    def set_coefficients(self, lindblad_term_dict, action="update", logscale_nonham=False, truncate=True):
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

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If False, then an error is thrown when the given coefficients
            cannot be parameterized as specified.

        Returns
        -------
        None
        """
        existing_Ltermdict, basis = self.coefficients(return_basis=True, logscale_nonham=False)

        #convert keys to global elementary errorgen labels:
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.tensor_product_block_labels(0)  # just take first TPB labels as all labels
        lindblad_term_dict = _collections.OrderedDict(
            [(_GlobalElementaryErrorgenLabel.cast(k, sslbls, identity_label_1Q), v)
             for k, v in lindblad_term_dict.items()])

        if action == "reset":
            for k in existing_Ltermdict:
                existing_Ltermdict[k] = 0.0

        for k, v in lindblad_term_dict.items():
            if logscale_nonham and k.errorgen_type == "S":
                # treat the value being set in lindblad_term_dict as the *channel* stochastic error rate, and
                # set the errgen coefficient to the value that would, in a depolarizing channel, give
                # that per-Pauli (or basis-el general?) stochastic error rate. See lindbladtools.py also.
                # errgen_coeff = -log(1-d^2*err_rate) / d^2
                dim = self.dim
                v = -_np.log(1 - dim * v) / dim

            if k not in existing_Ltermdict:
                raise KeyError("Invalid L-term descriptor (key) `%s`" % str(k))
            elif action == "update" or action == "reset":
                existing_Ltermdict[k] = v
            elif action == "add":
                existing_Ltermdict[k] += v
            else:
                raise ValueError('Invalid `action` argument: must be one of "update", "add", or "reset"')

        #convert existing_Ltermdict keys to *local* labels for use with lindblad_terms_to_projections:
        existing_Ltermdict = _collections.OrderedDict(
            [(_LocalElementaryErrorgenLabel.cast(lbl, sslbls, identity_label_1Q), val)
             for lbl, val in existing_Ltermdict.items()])

        hamC, otherC, _, _ = \
            _ot.lindblad_terms_to_projections(existing_Ltermdict, basis, self.parameterization.nonham_mode)
        pvec = _ot.lindblad_projections_to_paramvals(
            hamC, otherC, self.parameterization.param_mode,
            self.parameterization.nonham_mode, truncate=truncate)  # shouldn't need to truncate?
        self.from_vector(pvec)

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

    def coefficient_weights(self, weights):
        """
        TODO: docstring
        """
        lookup = _ot.lindblad_terms_projection_indices(self.ham_basis, self.other_basis,
                                                       self.parameterization.nonham_mode)
        rev_lookup = {i: lbl for lbl, i in lookup.items()}

        if self._coefficient_weights is None:
            return {}

        ret = {}
        for i, val in enumerate(self._coefficient_weights):
            if val != 1.0:
                ret[rev_lookup[i]] = val
        return ret

    def set_coefficient_weights(self, weights):
        """
        TODO: docstring
        """
        lookup = _ot.lindblad_terms_projection_indices(self.ham_basis, self.other_basis,
                                                       self.parameterization.nonham_mode)
        if self._coefficient_weights is None:
            self._coefficient_weights = _np.ones(len(self.coefficients_array()), 'd')
        for lbl, wt in weights.items():
            self._coefficient_weights[lookup[lbl]] = wt

    def transform_inplace(self, s):
        """
        Update error generator E with inv(s) * E * s,

        Generally, the transform function updates the *parameters* of
        the operation such that the resulting operation matrix is altered as
        described above.  If such an update cannot be done (because
        the operation parameters do not allow for it), ValueError is raised.

        Parameters
        ----------
        s : GaugeGroupElement
            A gauge group element which specifies the "s" matrix
            (and it's inverse) used in the above similarity transform.

        Returns
        -------
        None
        """
        from pygsti.models import gaugegroup as _gaugegroup
        if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
           isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
            U = s.transform_matrix
            Uinv = s.transform_matrix_inverse

            #conjugate Lindbladian exponent by U:
            err_gen_mx = self.to_sparse() if self._rep_type == 'sparse superop' else self.to_dense()
            err_gen_mx = _mt.safe_dot(Uinv, _mt.safe_dot(err_gen_mx, U))
            trunc = bool(isinstance(s, _gaugegroup.UnitaryGaugeGroupElement))
            self._set_params_from_matrix(err_gen_mx, truncate=trunc)
            self.dirty = True
            #Note: truncate=True above for unitary transformations because
            # while this trunctation should never be necessary (unitaries map CPTP -> CPTP)
            # sometimes a unitary transform can modify eigenvalues to be negative beyond
            # the tight tolerances checked when truncate == False. Maybe we should be able
            # to give a tolerance as `truncate` in the future?

        else:
            raise ValueError("Invalid transform for this LindbladErrorgen: type %s"
                             % str(type(s)))

    #I don't think this is ever needed
    #def spam_transform_inplace(self, s, typ):
    #    """
    #    Update operation matrix `O` with `inv(s) * O` OR `O * s`, depending on the value of `typ`.
    #
    #    This functions as `transform_inplace(...)` but is used when this
    #    Lindblad-parameterized operation is used as a part of a SPAM
    #    vector.  When `typ == "prep"`, the spam vector is assumed
    #    to be `rho = dot(self, <spamvec>)`, which transforms as
    #    `rho -> inv(s) * rho`, so `self -> inv(s) * self`. When
    #    `typ == "effect"`, `e.dag = dot(e.dag, self)` (not that
    #    `self` is NOT `self.dag` here), and `e.dag -> e.dag * s`
    #    so that `self -> self * s`.
    #
    #    Parameters
    #    ----------
    #    s : GaugeGroupElement
    #        A gauge group element which specifies the "s" matrix
    #        (and it's inverse) used in the above similarity transform.
    #
    #    typ : { 'prep', 'effect' }
    #        Which type of SPAM vector is being transformed (see above).
    #
    #    Returns
    #    -------
    #    None
    #    """
    #    assert(typ in ('prep', 'effect')), "Invalid `typ` argument: %s" % typ
    #
    #    if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) or \
    #       isinstance(s, _gaugegroup.TPSpamGaugeGroupElement):
    #        U = s.transform_matrix
    #        Uinv = s.transform_matrix_inverse
    #        err_gen_mx = self.to_sparse() if self._rep_type == 'sparse superop' else self.to_dense()
    #
    #        #just act on postfactor and Lindbladian exponent:
    #        if typ == "prep":
    #            err_gen_mx = _mt.safe_dot(Uinv, err_gen_mx)
    #        else:
    #            err_gen_mx = _mt.safe_dot(err_gen_mx, U)
    #
    #        self._set_params_from_matrix(err_gen_mx, truncate=True)
    #        self.dirty = True
    #        #Note: truncate=True above because some unitary transforms seem to
    #        ## modify eigenvalues to be negative beyond the tolerances
    #        ## checked when truncate == False.
    #    else:
    #        raise ValueError("Invalid transform for this LindbladDenseOp: type %s"
    #                         % str(type(s)))

    def _d_hdp(self):
        return self.hamGens.transpose((1, 2, 0))  # PRETRANS
        #return _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)

    def _d_odp(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0
        dim = self.dim

        assert(bsO > 0), "Cannot construct dOdp when other_basis_size == 0!"
        if self.parameterization.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]

            # Derivative of exponent wrt other param; shape == [dim,dim,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [dim,dim,1]
            if self.parameterization.param_mode == "depol":  # all coeffs same & == param^2
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'depol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None] * 2*otherParams[0]
                dOdp = _np.sum(_np.transpose(self.otherGens, (1, 2, 0)), axis=2)[:, :, None] * 2 * otherParams[0]
            elif self.parameterization.param_mode == "reldepol":  # all coeffs same & == param
                assert(len(otherParams) == 1), "Should only have 1 non-ham parameter in 'reldepol' case!"
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None]
                dOdp = _np.sum(_np.transpose(self.otherGens, (1, 2, 0)), axis=2)[:, :, None] * 2 * otherParams[0]
            elif self.parameterization.param_mode == "cptp":  # (coeffs = params^2)
                #dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams)
                dOdp = _np.transpose(self.otherGens, (1, 2, 0)) * 2 * otherParams  # just a broadcast
            else:  # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('alj->lja', self.otherGens)
                dOdp = _np.transpose(self.otherGens, (1, 2, 0))

        elif self.parameterization.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            # Note: otherGens has shape (2,bsO-1,dim,dim) with diag-term generators
            # in first "row" and affine generators in second row.

            # Derivative of exponent wrt other param; shape == [dim,dim,2,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [dim,dim,bs]
            if self.parameterization.param_mode == "depol":  # all coeffs same & == param^2
                diag_params = otherParams[0:1]
                dOdp = _np.empty((dim, dim, bsO), 'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2*diag_params[0] # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:, :, 0] = _np.sum(self.otherGens[0], axis=0) * 2 * diag_params[0]  # single diagonal term
                dOdp[:, :, 1:] = _np.transpose(self.otherGens[1], (1, 2, 0))  # no need for affine_params
            elif self.parameterization.param_mode == "reldepol":  # all coeffs same & == param^2
                dOdp = _np.empty((dim, dim, bsO), 'complex')
                #dOdp[:,:,0]  = _np.einsum('alj->lj', self.otherGens[0]) # single diagonal term
                #dOdp[:,:,1:] = _np.einsum('alj->lja', self.otherGens[1]) # affine part: each gen has own param
                dOdp[:, :, 0] = _np.sum(self.otherGens[0], axis=0)  # single diagonal term
                dOdp[:, :, 1:] = _np.transpose(self.otherGens[1], (1, 2, 0))  # affine part: each gen has own param
            elif self.parameterization.param_mode == "cptp":  # (coeffs = params^2)
                diag_params = otherParams[0:bsO - 1]
                dOdp = _np.empty((dim, dim, 2, bsO - 1), 'complex')
                #dOdp[:,:,0,:] = _np.einsum('alj,a->lja', self.otherGens[0], 2*diag_params)
                #dOdp[:,:,1,:] = _np.einsum('alj->lja', self.otherGens[1]) # no need for affine_params
                dOdp[:, :, 0, :] = _np.transpose(self.otherGens[0], (1, 2, 0)) * 2 * diag_params  # broadcast works
                dOdp[:, :, 1, :] = _np.transpose(self.otherGens[1], (1, 2, 0))  # no need for affine_params
            else:  # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('ablj->ljab', self.otherGens) # -> shape (dim,dim,2,bsO-1)
                dOdp = _np.transpose(self.otherGens, (2, 3, 0, 1))  # -> shape (dim,dim,2,bsO-1)

        else:  # nonham_mode == "all" ; all lindblad terms included
            assert(self.parameterization.param_mode in ("cptp", "unconstrained"))

            if self.parameterization.param_mode == "cptp":
                L, Lbar = self.Lmx, self.Lmx.conjugate()
                F1 = _np.tril(_np.ones((bsO - 1, bsO - 1), 'd'))
                F2 = _np.triu(_np.ones((bsO - 1, bsO - 1), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [dim,dim,bs-1,bs-1]
                # Note: replacing einsums here results in at least 3 numpy calls (probably slower?)
                dOdp = _np.einsum('amlj,mb,ab->ljab', self.otherGens, Lbar, F1)  # only a >= b nonzero (F1)
                dOdp += _np.einsum('malj,mb,ab->ljab', self.otherGens, L, F1)    # ditto
                dOdp += _np.einsum('bmlj,ma,ab->ljab', self.otherGens, Lbar, F2)  # only b > a nonzero (F2)
                dOdp += _np.einsum('mblj,ma,ab->ljab', self.otherGens, L, F2.conjugate())  # ditto
            else:  # "unconstrained"
                F0 = _np.identity(bsO - 1, 'd')
                F1 = _np.tril(_np.ones((bsO - 1, bsO - 1), 'd'), -1)
                F2 = _np.triu(_np.ones((bsO - 1, bsO - 1), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [dim,dim,bs-1,bs-1]
                #dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
                #dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
                #           _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
                #dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
                #           _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)
                tmp_ablj = _np.transpose(self.otherGens, (2, 3, 0, 1))  # ablj -> ljab
                tmp_balj = _np.transpose(self.otherGens, (2, 3, 1, 0))  # balj -> ljab
                dOdp = tmp_ablj * F0  # a == b case
                dOdp += tmp_ablj * F1 + tmp_balj * F1  # a > b (F1)
                dOdp += tmp_balj * F2 - tmp_ablj * F2  # a < b (F2)

        # apply basis transform
        tr = len(dOdp.shape)  # tensor rank
        assert((tr - 2) in (1, 2)), "Currently, dodp can only have 1 or 2 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
        return _np.real(dOdp)

    def _d2_odp2(self):
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0
        dim = self.dim

        assert(bsO > 0), "Cannot construct dOdp when other_basis_size == 0!"
        if self.parameterization.nonham_mode == "diagonal":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams)

            # Derivative of exponent wrt other param; shape == [dim,dim,nP,nP]
            if self.parameterization.param_mode == "depol":
                assert(nP == 1)
                #d2Odp2  = _np.einsum('alj->lj', self.otherGens)[:,:,None,None] * 2
                d2Odp2 = _np.sum(self.otherGens, axis=0)[:, :, None, None] * 2
            elif self.parameterization.param_mode == "cptp":
                assert(nP == bsO - 1)
                #d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
                d2Odp2 = _np.transpose(self.otherGens, (1, 2, 0))[:, :, :, None] * 2 * _np.identity(nP, 'd')
            else:  # param_mode == "unconstrained" or "reldepol"
                assert(nP == bsO - 1)
                d2Odp2 = _np.zeros([dim, dim, nP, nP], 'd')

        elif self.parameterization.nonham_mode == "diag_affine":
            otherParams = self.paramvals[nHam:]
            nP = len(otherParams)

            # Derivative of exponent wrt other param; shape == [dim,dim,nP,nP]
            if self.parameterization.param_mode == "depol":
                assert(nP == bsO)  # 1 diag param + (bsO-1) affine params
                d2Odp2 = _np.empty((dim, dim, nP, nP), 'complex')
                #d2Odp2[:,:,0,0]  = _np.einsum('alj->lj', self.otherGens[0]) * 2 # single diagonal term
                d2Odp2[:, :, 0, 0] = _np.sum(self.otherGens[0], axis=0) * 2  # single diagonal term
                d2Odp2[:, :, 1:, 1:] = 0  # 2nd deriv wrt. all affine params == 0
            elif self.parameterization.param_mode == "cptp":
                assert(nP == 2 * (bsO - 1)); hnP = bsO - 1  # half nP
                d2Odp2 = _np.empty((dim, dim, nP, nP), 'complex')
                #d2Odp2[:,:,0:hnP,0:hnP] = _np.einsum('alj,aq->ljaq', self.otherGens[0], 2*_np.identity(nP,'d'))
                d2Odp2[:, :, 0:hnP, 0:hnP] = _np.transpose(self.otherGens[0], (1, 2, 0))[
                    :, :, :, None] * 2 * _np.identity(nP, 'd')
                d2Odp2[:, :, hnP:, hnP:] = 0  # 2nd deriv wrt. all affine params == 0
            else:  # param_mode == "unconstrained" or "reldepol"
                assert(nP == 2 * (bsO - 1))
                d2Odp2 = _np.zeros([dim, dim, nP, nP], 'd')

        else:  # nonham_mode == "all" : all lindblad terms included
            nP = bsO - 1
            if self.parameterization.param_mode == "cptp":
                d2Odp2 = _np.zeros([dim, dim, nP, nP, nP, nP], 'complex')  # yikes! maybe make this SPARSE in future?

                #Note: correspondence w/Erik's notes: a=alpha, b=beta, q=gamma, r=delta
                # indices of d2Odp2 are [i,j,a,b,q,r]

                def iter_base_ab_qr(ab_inc_eq, qr_inc_eq):
                    """ Generates (base,ab,qr) tuples such that `base` runs over
                        all possible 'other' params and 'ab' and 'qr' run over
                        parameter indices s.t. ab > base and qr > base.  If
                        ab_inc_eq == True then the > becomes a >=, and likewise
                        for qr_inc_eq.  Used for looping over nonzero hessian els. """
                    for _base in range(nP):
                        start_ab = _base if ab_inc_eq else _base + 1
                        start_qr = _base if qr_inc_eq else _base + 1
                        for _ab in range(start_ab, nP):
                            for _qr in range(start_qr, nP):
                                yield (_base, _ab, _qr)

                for base, a, q in iter_base_ab_qr(True, True):  # Case1: base=b=r, ab=a, qr=q
                    d2Odp2[:, :, a, base, q, base] = self.otherGens[a, q] + self.otherGens[q, a]
                for base, a, r in iter_base_ab_qr(True, False):  # Case2: base=b=q, ab=a, qr=r
                    d2Odp2[:, :, a, base, base, r] = -1j * self.otherGens[a, r] + 1j * self.otherGens[r, a]
                for base, b, q in iter_base_ab_qr(False, True):  # Case3: base=a=r, ab=b, qr=q
                    d2Odp2[:, :, base, b, q, base] = 1j * self.otherGens[b, q] - 1j * self.otherGens[q, b]
                for base, b, r in iter_base_ab_qr(False, False):  # Case4: base=a=q, ab=b, qr=r
                    d2Odp2[:, :, base, b, base, r] = self.otherGens[b, r] + self.otherGens[r, b]

            else:  # param_mode == "unconstrained"
                d2Odp2 = _np.zeros([dim, dim, nP, nP, nP, nP], 'd')  # all params linear

        # apply basis transform
        tr = len(d2Odp2.shape)  # tensor rank
        assert((tr - 2) in (2, 4)), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
        return _np.real(d2Odp2)

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
        if self._rep_type == 'sparse superop':
            #raise NotImplementedError(("LindbladErrorgen.deriv_wrt_params(...) can only be called "
            #                           "when using *dense* basis elements!"))
            _warnings.warn("Using finite differencing to compute LindbladErrorGen derivative!")
            return super(LindbladErrorgen, self).deriv_wrt_params(wrt_filter)

        dim = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size

        #Deriv wrt hamiltonian params
        if bsH > 0:
            dH = self._d_hdp()
            dH = dH.reshape((dim**2, bsH - 1))  # [iFlattenedOp,iHamParam]
        else:
            dH = _np.empty((dim**2, 0), 'd')  # so concat works below

        #Deriv wrt other params
        if bsO > 0:
            dO = self._d_odp()
            dO = dO.reshape((dim**2, -1))  # [iFlattenedOp,iOtherParam]
        else:
            dO = _np.empty((dim**2, 0), 'd')  # so concat works below

        derivMx = _np.concatenate((dH, dO), axis=1)
        assert(_np.linalg.norm(_np.imag(derivMx)) < IMAG_TOL)  # allowed to be complex?
        derivMx = _np.real(derivMx)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

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
        if self._rep_type == 'sparse superop':
            raise NotImplementedError(("LindbladErrorgen.hessian_wrt_params(...) can only be called when using a"
                                       " *dense* rep!"))  # needed because the _d2_odp2 function assumes dense mxs

        dim = self.dim
        bsH = self.ham_basis_size
        bsO = self.other_basis_size
        nHam = bsH - 1 if (bsH > 0) else 0

        #Split hessian in 4 pieces:   d2H  |  dHdO
        #                             dHdO |  d2O
        # But only d2O is non-zero - and only when cptp == True

        nTotParams = self.num_params
        hessianMx = _np.zeros((dim**2, nTotParams, nTotParams), 'd')

        #Deriv wrt other params
        if bsO > 0:  # if there are any "other" params
            nP = nTotParams - nHam  # num "other" params, e.g. (bsO-1) or (bsO-1)**2
            d2Odp2 = self._d2_odp2()
            d2Odp2 = d2Odp2.reshape((dim**2, nP, nP))

            #d2Odp2 has been reshape so index as [iFlattenedOp,iDeriv1,iDeriv2]
            assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
            hessianMx[:, nHam:, nHam:] = _np.real(d2Odp2)  # d2O block of hessian

        if wrt_filter1 is None:
            if wrt_filter2 is None:
                return hessianMx
            else:
                return _np.take(hessianMx, wrt_filter2, axis=2)
        else:
            if wrt_filter2 is None:
                return _np.take(hessianMx, wrt_filter1, axis=1)
            else:
                return _np.take(_np.take(hessianMx, wrt_filter1, axis=1),
                                wrt_filter2, axis=2)

    def onenorm_upperbound(self):
        """
        Returns an upper bound on the 1-norm for this error generator (viewed as a matrix).

        Returns
        -------
        float
        """
        # computes sum of 1-norms of error generator terms multiplied by abs(coeff) values
        # because ||A + B|| <= ||A|| + ||B|| and ||cA|| == abs(c)||A||
        return self._onenorm_upbound

    def to_memoized_dict(self, mmg_memo):
        """Create a serializable dict with references to other objects in the memo.

        Parameters
        ----------
        mmg_memo: dict
            Memo dict from a ModelMemberGraph, i.e. keys are object ids and values
            are ModelMemberGraphNodes (which contain the serialize_id). This is NOT
            the same as other memos in ModelMember (e.g. copy, allocate_gpindices, etc.).

        Returns
        -------
        mm_dict: dict
            A dict representation of this ModelMember ready for serialization
            This must have at least the following fields:
                module, class, submembers, params, state_space, evotype
            Additional fields may be added by derived classes.
        """
        mm_dict = super().to_memoized_dict(mmg_memo)

        mm_dict['rep_type'] = self._rep_type
        mm_dict['parameterization'] = self.parameterization.to_nice_serialization()
        mm_dict['lindblad_basis'] = self.lindblad_basis.to_nice_serialization()
        mm_dict['matrix_basis'] = self.matrix_basis.to_nice_serialization()
        mm_dict['coefficients'] = [(k, v) for k, v in self.coefficients().items()]
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        lindblad_term_dict = {tuple(k): v for k, v in mm_dict['coefficients']}  # convert keys list->tuple
        parameterization = LindbladParameterization.from_nice_serialization(mm_dict['parameterization'])
        lindblad_basis = _Basis.from_nice_serialization(mm_dict['lindblad_basis'])
        mx_basis = _Basis.from_nice_serialization(mm_dict['matrix_basis'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        truncate = False  # shouldn't need to truncate since we're reloading a valid set of coefficients

        return cls(lindblad_term_dict, parameterization, lindblad_basis,
                   mx_basis, truncate, mm_dict['evotype'], state_space)

    def __str__(self):
        s = "Lindblad error generator with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        return s


class LindbladParameterization(_NicelySerializable):
    """
    An object encapsulating a particular way of parameterizing a LindbladErrorgen

    A `LindbladParameterization` is a high-level parameterization-type (e.g. `"H+S"`)
    that contains two "modes" - one describing the number (and structure) of the
    non-Hamiltonian Lindblad coefficients (`nonham_mode') and one describing how the
    Lindblad coefficients are converted to/from parameters (`param_mode`).

    Parameters
    ----------
    nonham_mode : str
        The "non-Hamiltonian mode" describes which non-Hamiltonian Lindblad
        coefficients are stored in a LindbladOp, and is one
        of `"diagonal"` (only the diagonal elements of the full coefficient
        matrix as a 1D array), `"diag_affine"` (a 2-by-d array of the diagonal
        coefficients on top of the affine projections), or `"all"` (the entire
        coefficient matrix).

    param_mode : str
        The "parameter mode" describes how the Lindblad coefficients/projections
        are converted into parameter values.  This can be:
        `"unconstrained"` (coefficients are independent unconstrained parameters),
        `"cptp"` (independent parameters but constrained so map is CPTP),
        `"depol"` (all non-Ham. diagonal coeffs are the *same, positive* value), or
        `"reldepol"` (same as `"depol"` but no positivity constraint).

    ham_params_allowed, nonham_params_allowed : bool
        Whether or not Hamiltonian and non-Hamiltonian parameters are allowed.
    """

    @classmethod
    def from_lindblad_terms(cls, errs):
        """
        Helper function giving minimal Lindblad parameterization needed for specified errors.

        Parameters
        ----------
        errs : dict
            Error dictionary with keys as `(termType, basisLabel)` tuples, where
            `termType` can be `"H"` (Hamiltonian), `"S"` (Stochastic), or `"A"`
            (Affine), and `basisLabel` is a string of I, X, Y, or Z to describe a
            Pauli basis element appropriate for the gate (i.e. having the same
            number of letters as there are qubits in the gate).  For example, you
            could specify a 0.01-radian Z-rotation error and 0.05 rate of Pauli-
            stochastic X errors on a 1-qubit gate by using the error dictionary:
            `{('H','Z'): 0.01, ('S','X'): 0.05}`.

        Returns
        -------
        parameterization : str
            Parameterization string for constructing Lindblad error generators.
        """
        paramtypes = []
        if any([lbl.errorgen_type == 'H' for lbl in errs]): paramtypes.append('H')
        if any([lbl.errorgen_type == 'S' for lbl in errs]): paramtypes.append('S')
        if any([lbl.errorgen_type == 'A' for lbl in errs]): paramtypes.append('A')
        if any([lbl.errorgen_type == 'S' and len(lbl.basis_element_labels) == 2 for lbl in errs]):
            # parameterization must be "CPTP" if there are any ('S',b1,b2) keys
            parameterization = "CPTP"
        else:
            parameterization = '+'.join(paramtypes)
        return cls.cast(parameterization)

    @classmethod
    def cast(cls, obj):
        """
        Converts a string into a LindbladParameterization object if necessary.

        Parameters
        ----------
        obj : str or LindbladParameterization
            The object to convert.

        Returns
        -------
        LindbladParameterization
        """
        if isinstance(obj, LindbladParameterization):
            return obj

        param_type = obj
        if param_type == "CPTP":
            nonham_mode = "all"; param_mode = "cptp"
        elif param_type == "H":
            nonham_mode = "all"; param_mode = "cptp"  # these don't matter since there's no non-ham errors
        elif param_type in ("H+S", "S"):
            nonham_mode = "diagonal"; param_mode = "cptp"
        elif param_type in ("H+s", "s"):
            nonham_mode = "diagonal"; param_mode = "unconstrained"
        elif param_type in ("H+S+A", "S+A"):
            nonham_mode = "diag_affine"; param_mode = "cptp"
        elif param_type in ("H+s+A", "s+A"):
            nonham_mode = "diag_affine"; param_mode = "unconstrained"
        elif param_type in ("H+D", "D"):
            nonham_mode = "diagonal"; param_mode = "depol"
        elif param_type in ("H+d", "d"):
            nonham_mode = "diagonal"; param_mode = "reldepol"
        elif param_type in ("H+D+A", "D+A"):
            nonham_mode = "diag_affine"; param_mode = "depol"
        elif param_type in ("H+d+A", "d+A"):
            nonham_mode = "diag_affine"; param_mode = "reldepol"
        elif param_type == "GLND":
            nonham_mode = "all"; param_mode = "unconstrained"
        else:
            raise ValueError("Unrecognized base type in `param_type`=%s" % param_type)

        ham_allowed = True if (("H" == param_type)
                               or ("H+" in param_type)
                               or param_type in ("CPTP", "GLND")) else False
        nonham_allowed = False if param_type == "H" else True
        return cls(nonham_mode, param_mode, ham_allowed, nonham_allowed, param_type)

    def __init__(self, nonham_mode, param_mode, ham_params_allowed, nonham_params_allowed, abbrev=None):
        self.nonham_mode = nonham_mode
        self.param_mode = param_mode
        self.ham_params_allowed = ham_params_allowed
        self.nonham_params_allowed = nonham_params_allowed
        self.abbrev = abbrev

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'mode': self.param_mode,
                      'non_hamiltonian_mode': self.nonham_mode,
                      'hamiltonian_parameters_allowed': self.ham_params_allowed,
                      'non_hamiltonian_parameters_allowed': self.nonham_params_allowed,
                      'abbreviation': self.abbrev})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(state['non_hamiltonian_mode'], state['mode'], state['hamiltonian_parameters_allowed'],
                   state['non_hamiltonian_parameters_allowed'], state['abbreviation'])

    def __str__(self):
        if self.abbrev is not None:
            return self.abbrev
        else:
            return "CustomLindbladParam(%s,%s,%s,%s)" % (self.nonham_mode, self.param_mode,
                                                         str(self.ham_params_allowed), str(self.nonham_params_allowed))
