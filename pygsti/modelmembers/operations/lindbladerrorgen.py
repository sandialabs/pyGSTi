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
import copy as _copy
import itertools as _itertools

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl

from pygsti.baseobjs.opcalc import compact_deriv as _compact_deriv, \
    bulk_eval_compact_polynomials_complex as _bulk_eval_compact_polynomials_complex, \
    abs_sum_bulk_eval_compact_polynomials_complex as _abs_sum_bulk_eval_compact_polynomials_complex
from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as _LindbladCoefficientBlock
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
    """

    _generators_cache = {}  # a custom cache for _init_generators method calls

    @classmethod
    def from_operation_matrix_and_blocks(cls, op_matrix, lindblad_coefficient_blocks, lindblad_basis='auto',
                                         mx_basis='pp', truncate=True, evotype="default", state_space=None):
        
        """
        Create a Lindblad-parameterized error generator from an operation matrix and coefficient blocks.

        Parameters
        ----------
        op_matrix : numpy array or SciPy sparse matrix
            A square 2D array that gives the raw operation matrix, assumed to be in the `mx_basis` basis.
            The shape of this array sets the dimension of the operation.

        lindblad_coefficient_blocks : list
            A list of Lindblad coefficient blocks to set from the error generator projections.

        lindblad_basis : {'auto', 'PP', 'std', 'gm', 'qt'}, optional
            The basis used for Lindblad terms. Default is 'auto'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in order to meet constraints.
            Default is True. (e.g. to preserve CPTP) when necessary. If False, then an error is thrown 
            when the Lindblad terms don't conform to the constrains.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

        Returns
        -------
        LindbladErrorgen
        """

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
        for blk in lindblad_coefficient_blocks:
            blk.set_from_errorgen_projections(errgenMx, mx_basis, truncate=truncate)
        return cls(lindblad_coefficient_blocks, lindblad_basis, mx_basis, evotype, state_space)

    @classmethod
    def from_operation_matrix(cls, op_matrix, parameterization='CPTP', lindblad_basis='PP',
                              mx_basis='pp', truncate=True, evotype="default", state_space=None):
        """
        Creates a Lindblad-parameterized error generator from an operation.

        Here "operation" means the exponentiated error generator, so this method
        essentially takes the matrix log of `op_matrix` and constructs an error
        generator from this using :meth:`from_error_generator`.

        Parameters
        ----------
        op_matrix : numpy array or SciPy sparse matrix
            A square 2D array that gives the raw operation matrix, assumed to be in the `mx_basis` basis.
            The shape of this array sets the dimension of the operation.

        parameterization : str, optional (default 'CPTP') 
            Describes how the Lindblad coefficients/projections relate to the error generator's parameter values.
            Default is "CPTP". Supported strings are those castable to `LindbladParameterization`. See
            `LindbladParameterization` for supported options.

        lindblad_basis : {'PP', 'std', 'gm', 'qt'}, optional
            The basis used for Lindblad terms. Default is 'PP'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'. Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in order to meet constraints.
            Default is True.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.
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
    def from_error_generator(cls, errgen_or_dim, parameterization="CPTP", lindblad_basis='PP', mx_basis='pp',
                             truncate=True, evotype="default", state_space=None):
        """
        Create a Lindblad-parameterized error generator from an error generator matrix or dimension.

        Parameters
        ----------
        errgen_or_dim : numpy array, SciPy sparse matrix, or int
            A square 2D array that gives the full error generator or an integer specifying the dimension
            of a zero error generator.

        parameterization : str, optional (default 'CPTP') 
            Describes how the Lindblad coefficients/projections relate to the error generator's parameter values.
            Default is "CPTP". Supported strings are those castable to `LindbladParameterization`. See
            `LindbladParameterization` for supported options.

        lindblad_basis : {'PP', 'std', 'gm', 'qt'}, optional
            The basis used for Lindblad terms. Default is 'PP'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'. Allowed
            values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
            and Qutrit (qt) (or a custom basis object).

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in order to meet constraints.
            Default is True.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

        Returns
        -------
        LindbladErrorgen
        """
        errgen = _np.zeros((errgen_or_dim, errgen_or_dim), 'd') \
            if isinstance(errgen_or_dim, (int, _np.int64)) else errgen_or_dim
        return cls._from_error_generator(errgen, parameterization, lindblad_basis,
                                         mx_basis, truncate, evotype, state_space)

    @classmethod
    def from_error_generator_and_blocks(cls, errgen_or_dim, lindblad_coefficient_blocks,
                                        lindblad_basis='PP', mx_basis='pp',
                                        truncate=True, evotype="default", state_space=None):
        """
        Create a Lindblad-parameterized error generator from an error generator matrix or dimension and coefficient blocks.

        Parameters
        ----------
        errgen_or_dim : numpy array, SciPy sparse matrix, or int
            A square 2D array that gives the full error generator or an integer specifying the dimension
            of a zero error generator.

        lindblad_coefficient_blocks : list
            A list of Lindblad coefficient blocks to set from the error generator projections.

        lindblad_basis : {'PP', 'std', 'gm', 'qt'}, optional
            The basis used for Lindblad terms. Default is 'PP'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in order to meet constraints.
            Default is True.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

        Returns
        -------
        LindbladErrorgen
        """
        errgenMx = _np.zeros((errgen_or_dim, errgen_or_dim), 'd') \
            if isinstance(errgen_or_dim, (int, _np.int64)) else errgen_or_dim
        for blk in lindblad_coefficient_blocks:
            blk.set_from_errorgen_projections(errgenMx, mx_basis, truncate=truncate)
        return cls(lindblad_coefficient_blocks, lindblad_basis, mx_basis, evotype, state_space)

    @classmethod
    def _from_error_generator(cls, errgen, parameterization="CPTP", lindblad_basis="PP",
                              mx_basis="pp", truncate=True, evotype="default", state_space=None):
        """
        Create a Lindblad-form error generator from an error generator matrix and a basis.

        The basis specifies how to decompose (project) the error generator.

        Parameters
        ----------
        errgen : numpy array or SciPy sparse matrix
            a square 2D array that gives the full error generator. The shape of
            this array sets the dimension of the operator. The projections of
            this quantity are closely related to the parameters of the error 
            generator (they may not be exactly equal if parameterization = 'CPTP').

        lindblad_basis : {'PP', 'std', 'gm', 'qt'}, optional
            The basis used for Lindblad terms. Default is 'PP'.

        parameterization : str, optional (default 'CPTP') 
            Describes how the Lindblad coefficients/projections relate to the error generator's parameter values.
            Default is "CPTP". Supported strings are those castable to `LindbladParameterization`. See
            `LindbladParameterization` for supported options.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'.

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

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

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
        if isinstance(lindblad_basis, _Basis):
            sparse = lindblad_basis.sparse
        else:
            if isinstance(lindblad_basis, str): sparse = _sps.issparse(errgen)
            elif len(lindblad_basis) > 0: sparse = _sps.issparse(lindblad_basis[0])
            lindblad_basis = _Basis.cast(lindblad_basis, dim, sparse=sparse)

        if sparse is None: sparse = False  # the default

        #Create or convert matrix basis to consistent sparsity
        if not isinstance(mx_basis, _Basis):
            matrix_basis = _Basis.cast(mx_basis, dim, sparse=sparse)
        else: matrix_basis = mx_basis

        # errgen + bases => coeffs
        parameterization = LindbladParameterization.cast(parameterization)

        # Create blocks based on bases along - no specific errorgen labels
        blocks = []
        for blk_type, blk_param_mode in zip(parameterization.block_types, parameterization.param_modes):
            blk = _LindbladCoefficientBlock(blk_type, lindblad_basis, param_mode=blk_param_mode)
            blk.set_from_errorgen_projections(errgen, matrix_basis, truncate=truncate)
            blocks.append(blk)

        return cls(blocks, "auto", mx_basis, evotype, state_space)

    @classmethod
    def from_elementary_errorgens(cls, elementary_errorgens, parameterization='auto', elementary_errorgen_basis='PP',
                              mx_basis="pp", truncate=True, evotype="default", state_space=None):
        """
        Create a Lindblad-parameterized error generator from elementary error generators.

        Parameters
        ----------
        elementary_errorgens : dict
            A dictionary of elementary error generators. Keys are labels specifying the type and basis
            elements of the elementary error generators, and values are the corresponding coefficients.
            Keys are `(termType, basisLabel1, <basisLabel2>)` tuples, where `termType` is 
            `"H"` (Hamiltonian), `"S"` (Stochastic), `"C"` (Correlation) or `"A"` (Active).  
            Hamiltonian and Stochastic terms always have a single basis label (so key is a 2-tuple) 
            whereas C and A tuples have 2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are pauli strings. Values are coefficients.

        parameterization : str, optional (default 'CPTP') 
            Describes how the Lindblad coefficients/projections relate to the error generator's parameter values.
            Default is "CPTP". Supported strings are those castable to `LindbladParameterization`. See
            `LindbladParameterization` for supported options.

        elementary_errorgen_basis : {'PP', 'std', 'gm', 'qt'}, optional
            The basis used for the elementary error generators. Default is 'PP'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'.

        truncate : bool, optional
            Whether to truncate the projections onto the Lindblad terms in order to meet constraints.
            Default is True.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

        Returns
        -------
        LindbladErrorgen
        """

        state_space = _statespace.StateSpace.cast(state_space)
        dim = state_space.dim  # Store superop dimension
        basis = _Basis.cast(elementary_errorgen_basis, dim)

        #convert elementary errorgen labels to *local* labels (ok to specify w/global labels)
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = state_space.sole_tensor_product_block_labels  # first TPB labels == all labels
        elementary_errorgens = _collections.OrderedDict(
            [(_LocalElementaryErrorgenLabel.cast(lbl, sslbls, identity_label_1Q), val)
             for lbl, val in elementary_errorgens.items()])

        parameterization = LindbladParameterization.minimal_from_elementary_errorgens(elementary_errorgens) \
            if parameterization == "auto" else LindbladParameterization.cast(parameterization)

        eegs_by_typ = {
            'ham': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type == 'H'},
            'other_diagonal': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type == 'S'},
            'other': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type != 'H'}
        }

        blocks = []
        for blk_type, blk_param_mode in zip(parameterization.block_types, parameterization.param_modes):
            relevant_eegs = eegs_by_typ[blk_type]  # KeyError => unrecognized block type!
            bels = sorted(set(_itertools.chain(*[lbl.basis_element_labels for lbl in relevant_eegs.keys()])))
            blk = _LindbladCoefficientBlock(blk_type, basis, bels, param_mode=blk_param_mode)
            blk.set_elementary_errorgens(relevant_eegs, truncate=truncate)
            blocks.append(blk)

        return cls(blocks, basis, mx_basis, evotype, state_space)

    def __init__(self, lindblad_coefficient_blocks, lindblad_basis='auto', mx_basis='pp',
                 evotype="default", state_space=None):
        
        """
        Initialize a LindbladErrorgen object.

        Parameters
        ----------
        lindblad_coefficient_blocks : list of LindbladCoefficientBlock
            A list of Lindblad coefficient blocks that define the error generator.

        lindblad_basis : {'auto', 'PP', 'std', 'gm', 'qt'} or Basis object, optional
            The basis used for Lindblad terms. If 'auto', the basis is inferred from the coefficient blocks.
            Default is 'auto'.

        mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object, optional
            The basis for this error generator's linear mapping. Default is 'pp'.

        evotype : {"default", "densitymx", "svterm", "cterm"}, optional
            The evolution type of the error generator being constructed. Default is "default".

        state_space : StateSpace, optional
            The state space for the error generator. Default is None.

        Raises
        ------
        ValueError
            If the provided evotype does not support any of the required representations for a LindbladErrorgen.
        """

        if isinstance(lindblad_coefficient_blocks, dict):  # backward compat warning
            _warnings.warn(("You're trying to create a LindbladErrorgen object using a dictionary.  This"
                            " constructor was recently updated to take a list of LindbladCoefficientBlock"
                            " objects (not a dict) for increased flexibility.  You probably want to call"
                            " a LindbladErrorgen.from_elementary_errorgens(...) instead."))

        state_space = _statespace.StateSpace.cast(state_space)
        dim = state_space.dim  # Store superop dimension
        #Decide on our rep-type ahead of time so we know whether to make bases sparse
        # (a LindbladErrorgen with a sparse rep => sparse bases and similar with dense rep)
        evotype = _Evotype.cast(evotype, state_space=state_space)
        reptype_preferences = ('lindblad errorgen', 'dense superop', 'sparse superop') \
            if evotype.prefer_dense_reps else ('lindblad errorgen', 'sparse superop', 'dense superop')
        for reptype in reptype_preferences:
            if evotype.supports(reptype):
                self._rep_type = reptype; break
        else:
            raise ValueError("Evotype doesn't support any of the representations a LindbladErrorgen requires.")
        sparse_bases = bool(self._rep_type == 'sparse superop')  # we use sparse bases iff we have a sparse rep        

        if lindblad_basis == "auto":
            assert(all([(blk._basis is not None) for blk in lindblad_coefficient_blocks])), \
                "When `lindblad_basis == 'auto'`, the supplied coefficient blocks must have valid bases!"
            default_lindblad_basis = None
        else:
            default_lindblad_basis = _Basis.cast(lindblad_basis, dim, sparse=sparse_bases)

        for blk in lindblad_coefficient_blocks:
            if blk._basis is None: blk._basis = default_lindblad_basis
            elif blk._basis.sparse != sparse_bases:  # update block bases to desired sparsity if needed
                blk._basis = blk._basis.with_sparsity(sparse_bases)

        self.coefficient_blocks = lindblad_coefficient_blocks
        self.matrix_basis = _Basis.cast(mx_basis, dim, sparse=sparse_bases)

        nP = sum([blk.num_params for blk in lindblad_coefficient_blocks])
        self.paramvals = _np.empty(nP, 'd'); off = 0
        for blk in lindblad_coefficient_blocks:
            self.paramvals[off:off + blk.num_params] = blk.to_vector()
            off += blk.num_params

        #Fast CSR-matrix summing variables: N/A if not sparse or using terms
        self._CSRSumIndices = self._CSRSumData = self._CSRSumPtr = None

        # Generator matrices & cache qtys: N/A for term-based evotypes
        #TODO - maybe move some/all of these to the coefficient block class:
        self._onenorm_upbound = None
        self._coefficient_weights = None

        #All representations need to track 1norms:
        self.lindblad_term_superops_and_1norms = [
            blk.create_lindblad_term_superoperators(self.matrix_basis, sparse_bases, include_1norms=True, flat=True)
            for blk in lindblad_coefficient_blocks]

        #combine all of the linblad term superoperators across the blocks to a single concatenated tensor.
        self.combined_lindblad_term_superops = _np.concatenate([Lterm_superops for (Lterm_superops, _) in self.lindblad_term_superops_and_1norms], axis=0)

        #Create a representation of the type chosen above:
        if self._rep_type == 'lindblad errorgen':
            rep = evotype.create_lindblad_errorgen_rep(lindblad_coefficient_blocks, state_space)

        else:  # Otherwise create a sparse or dense matrix representation

            if sparse_bases:  # then construct a sparse-matrix representation (self._rep_type == 'sparse superop')
                #Precompute for faster CSR sums in _construct_errgen
                all_csr_matrices = list(_itertools.chain.from_iterable(
                    [superops for superops, norms in self.lindblad_term_superops_and_1norms]))
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
                # UNSPECIFIED BASIS -- we set basis=None below, which may not work with all evotypes,
                #  and should be replaced with the basis of contained ops (if any) once we establish
                #  a common .basis or ._basis attribute of representations (which could still be None)
                # Update: fixed now (I think) - this seems like a legit matrix_basis to use... REMOVE comment?
                rep = evotype.create_dense_superop_rep(None, self.matrix_basis, state_space)

        _LinearOperator.__init__(self, rep, evotype)  # sets self.dim
        self._update_rep()  # updates _rep whether it's a dense or sparse matrix
        self._paramlbls = _np.array(list(_itertools.chain.from_iterable(
            [blk.param_labels for blk in self.coefficient_blocks])), dtype=object)
        assert(self._onenorm_upbound is not None)  # _update_rep should set this

        # Done with __init__(...)

    def _init_terms(self, coefficient_blocks, max_polynomial_vars):

        Lterms = []; off = 0
        for blk in self.coefficient_blocks:
            Lterms.extend(blk.create_lindblad_term_objects(off, max_polynomial_vars, self._evotype, self.state_space))
            off += blk.num_params

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

        # Project errgen to give coefficient block data
        remaining_errgen = errgen
        for blk in self.coefficient_blocks:
            projected_errgen = blk.set_from_errorgen_projections(remaining_errgen, self.matrix_basis,
                                                                 return_projected_errorgen=True, truncate=truncate)
            remaining_errgen = remaining_errgen - projected_errgen

        # set paramvals from coefficient block data
        off = 0
        for blk in self.coefficient_blocks:
            self.paramvals[off:off + blk.num_params] = blk.to_vector()
            off += blk.num_params

        self._update_rep()
        #assert(_np.allclose(errgen, self.to_dense()))  # DEBUG

    def _update_rep(self):
        """
        Updates self._rep, which contains a representation of this error generator
        as either a dense or sparse matrix.  This routine essentially builds the
        error generator matrix using the current coefficient block data (which from_vector
        should keep in sync with the parameters) and updates self._rep accordingly (by
        rewriting its data).
        """
        # Update 1-norm of composite errorgen
        onenorm = sum([_np.dot(_np.abs(blk.block_data.flat), one_norms) for blk, (_, one_norms)
                       in zip(self.coefficient_blocks, self.lindblad_term_superops_and_1norms)])
        assert(_np.imag(onenorm) < 1e-6)
        onenorm = _np.real(onenorm)

        # Build operation matrix from generators and coefficients:
        if self._rep_type == 'lindblad errorgen':
            # the code below is for updating sparse or dense matrix representations.  If our
            # evotype has a native Lindblad representation, maybe in the FUTURE we should
            # call an update method of it here?

            #Still need to update onenorm - FUTURE: maybe put this logic inside rep? (done above now)
            #onenorm = sum([_np.dot(blk.block_data.flat, one_norms) for blk, (_, one_norms)
            #               in zip(self.coefficient_blocks, self.lindblad_term_superops_and_1norms)])
            pass

        elif self._rep_type == 'sparse superop':  # then bases & errgen are sparse
            coeffs = None
            data = self._data_scratch
            data.fill(0.0)  # data starts at zero

            # Get coefficients in a single flat array (Note: this can be complex)
            coeffs = _np.array(list(_itertools.chain(*[blk.block_data.flat for blk in self.coefficient_blocks])))
            if len(coeffs) > 0:
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
            comb_blk_datas = _np.concatenate([blk.block_data.ravel() for blk in self.coefficient_blocks])
            lnd_error_gen = _np.einsum('i,ijk->jk', comb_blk_datas, self.combined_lindblad_term_superops)
            
            #This test has been previously commented out in the sparse case, should we do the same for this one?
            assert(_np.linalg.norm(lnd_error_gen.imag)<1e-10), \
                "Imaginary error gen norm: %g" % _np.linalg.norm(lnd_error_gen.imag)
            self._rep.base[:, :] = lnd_error_gen.real

        self._onenorm_upbound = onenorm

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
            comb_blk_datas = _np.concatenate([blk.block_data.ravel() for blk in self.coefficient_blocks])
            lnd_error_gen = _np.einsum('i,ijk->jk', comb_blk_datas, self.combined_lindblad_term_superops)

            assert(_np.linalg.norm(lnd_error_gen.imag)<1e-10), \
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
            return _sps.csr_matrix(self.to_dense(on_space))
        elif self._rep_type == 'sparse superop':
            assert(on_space in ('minimal', 'HilbertSchmidt'))
            return _sps.csr_matrix((self._rep.data, self._rep.indices, self._rep.indptr),
                                   shape=(self.dim, self.dim))
        else:  # dense rep
            return _sps.csr_matrix(self.to_dense(on_space))


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
            output of :meth:`Polynomial.compact`.
        """
        assert(self._rep_type == 'lindblad errorgen'), \
            "Only evotypes with native Lindblad errorgen representations can utilize Taylor terms"
        assert(order == 0), \
            "Error generators currently treat all terms as 0-th order; nothing else should be requested!"
        assert(return_coeff_polys is False)
        if self._rep.Lterms is None:
            Lblocks = self._rep.lindblad_coefficient_blocks
            self._rep.Lterms, self._rep.Lterm_coeffs = self._init_terms(Lblocks, max_polynomial_vars)
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

        off = 0
        for blk in self.coefficient_blocks:
            blk.from_vector(self.paramvals[off: off + blk.num_params])
            off += blk.num_params

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
            This is the value returned by :meth:`error_rates`.

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"` (Correlation) or `"A"` (Active).  Hamiltonian and Stochastic terms 
            always have a single basis label (so key is a 2-tuple) whereas C and A tuples
            have 2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are pauli strings. Values are coefficients.
        basis : Basis
            A Basis mapping the basis labels used in the
            keys of `Ltermdict` to basis matrices.
        """
        elem_errorgens = {}
        bases = set()
        for blk in self.coefficient_blocks:
            elem_errorgens.update(blk.elementary_errorgens)
            if blk._basis not in bases:
                bases.add(blk._basis)

        #convert to *global* elementary errorgen labels
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.sole_tensor_product_block_labels  # take first TPB labels as all labels
        elem_errorgens = _collections.OrderedDict(
            [(_GlobalElementaryErrorgenLabel.cast(local_eeg_lbl, sslbls, identity_label_1Q), value)
             for local_eeg_lbl, value in elem_errorgens.items()])

        if logscale_nonham:
            dim = self.dim
            for k in elem_errorgens.keys():
                if k.errorgen_type == "S":  # reverse mapping: err_coeff -> err_rate
                    elem_errorgens[k] = (1 - _np.exp(-dim * elem_errorgens[k])) / dim
                    # err_rate = (1-exp(-d^2*errgen_coeff))/d^2

        if return_basis:
            assert(len(bases) == 1), \
                "Cannot return basis from `coefficients` when different coefficient blocks have different bases!"
            return elem_errorgens, list(bases)[0]
        else:
            return elem_errorgens

    def coefficient_labels(self):
        """
        The elementary error-generator labels corresponding to the elements of :meth:`coefficients_array`.

        Returns
        -------
        tuple
            A tuple of (<type>, <basisEl1> [,<basisEl2]) elements identifying the elementary error
            generators of this gate.
        """
        labels = []
        for blk in self.coefficient_blocks:
            #labels.extend(blk.coefficent_labels)
            labels.extend(blk.elementary_errorgens.keys())

        #convert to *global* elementary errorgen labels
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.sole_tensor_product_block_labels  # take first TPB labels as all labels
        return tuple([_GlobalElementaryErrorgenLabel.cast(local_eeg_lbl, sslbls, identity_label_1Q)
                      for local_eeg_lbl in labels])

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
        # Note: ret will be complex if any block's data is
        #ret = _np.concatenate([blk.block_data.flat for blk in self.coefficient_blocks])
        ret = _np.concatenate([list(blk.elementary_errorgens.values()) for blk in self.coefficient_blocks])
        if self._coefficient_weights is not None:
            ret *= self._coefficient_weights
        return ret

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
        blk_derivs = []; off = 0
        for blk in self.coefficient_blocks:
            #bd = blk.deriv_wrt_params(self.paramvals[off:off + blk.num_params])
            #if bd.ndim == 3:  # (coeff_dim_1, coeff_dim_2, param_dim) => (coeff_dim, param_dim)
            #    bd = bd.reshape((bd.shape[0] * bd.shape[1], bd.shape[2]))
            bd = blk.elementary_errorgen_deriv_wrt_params(self.paramvals[off:off + blk.num_params])
            blk_derivs.append(bd)
            off += blk.num_params

        ret = _spl.block_diag(*blk_derivs)

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
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"` (Correlation) or `"A"` (Active).  Hamiltonian and Stochastic terms 
            always have a single basis label (so key is a 2-tuple) whereas C and A tuples
            have 2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are pauli strings. Values are coefficients. 
            Values are real error rates except for the 2-basis-label case.
        """
        return self.coefficients(return_basis=False, logscale_nonham=True)

    def set_coefficients(self, elementary_errorgens, action="update", logscale_nonham=False, truncate=True):
        """
        Sets the coefficients of elementary error generator terms in this error generator.

        The dictionary `lindblad_term_dict` has tuple-keys describing the type
        of term and the basis elements used to construct it, e.g. `('H','X')`.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"` (Correlation) or `"A"` (Active).  Hamiltonian and Stochastic terms 
            always have a single basis label (so key is a 2-tuple) whereas C and A tuples
            have 2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are pauli strings.

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
        #convert keys to local elementary errorgen labels (the same as those used by the coefficient blocks):
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = self.state_space.sole_tensor_product_block_labels  # take first TPB labels as all labels
        elem_errorgens = _collections.OrderedDict(
            [(_LocalElementaryErrorgenLabel.cast(k, sslbls, identity_label_1Q), v)
             for k, v in elementary_errorgens.items()])

        processed = set()  # keep track of which entries in elem_errorgens have been processed by a block
        for blk in self.coefficient_blocks:
            blk_elem_errorgens = blk.elementary_errorgens

            if action == "reset":
                for k in blk_elem_errorgens:
                    blk_elem_errorgens[k] = 0.0

            for k, v in elem_errorgens.items():
                if logscale_nonham and k.errorgen_type == "S":
                    # treat the value being set in lindblad_term_dict as the *channel* stochastic error rate, and
                    # set the errgen coefficient to the value that would, in a depolarizing channel, give
                    # that per-Pauli (or basis-el general?) stochastic error rate. See lindbladtools.py also.
                    # errgen_coeff = -log(1-d^2*err_rate) / d^2
                    dim = self.dim
                    v = -_np.log(1 - dim * v) / dim

                if k not in blk_elem_errorgens or k in processed:
                    continue  # ignore labels not in this block

                if action == "update" or action == "reset":
                    blk_elem_errorgens[k] = v
                elif action == "add":
                    blk_elem_errorgens[k] += v
                else:
                    raise ValueError('Invalid `action` argument: must be one of "update", "add", or "reset"')

                processed.add(k)  # mark that this label has been processed (so no other blocks process it)

            blk.set_elementary_errorgens(blk_elem_errorgens, truncate=truncate)

        # Now update paramvals using block data
        off = 0
        for blk in self.coefficient_blocks:
            self.paramvals[off:off + blk.num_params] = blk.to_vector()
            off += blk.num_params
        self._update_rep()
        self.dirty = True

    def set_error_rates(self, elementary_errorgens, action="update"):
        """
        Sets the coeffcients of elementary error generator terms in this error generator.

        Coefficients are set so that the contributions of the resulting
        channel's error rate are given by the values in `lindblad_term_dict`.
        See :meth:`error_rates` for more details.

        Parameters
        ----------
        lindblad_term_dict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic),
            `"C"` (Correlation) or `"A"` (Active).  Hamiltonian and Stochastic terms 
            always have a single basis label (so key is a 2-tuple) whereas C and A tuples
            have 2 basis labels to specify off-diagonal non-Hamiltonian Lindblad
            terms.  Basis labels are pauli strings.

        action : {"update","add","reset"}
            How the values in `lindblad_term_dict` should be combined with existing
            error rates.

        Returns
        -------
        None
        """
        self.set_coefficients(elementary_errorgens, action, logscale_nonham=True)

    def coefficient_weights(self, weights):
        """
        Get the non-default coefficient weights.

        This method returns a dictionary of coefficient weights that are not equal to the default value of 1.0.

        Parameters
        ----------
        weights : dict
            A dictionary where keys are coefficient labels and values are the corresponding weights.

        Returns
        -------
        dict
            A dictionary where keys are coefficient labels and values are the corresponding weights that are not equal to 1.0.
        """
        coeff_labels = self.coefficient_labels()
        lbl_lookup = {i: lbl for i, lbl in enumerate(coeff_labels)}

        if self._coefficient_weights is None:
            return {}

        ret = {}
        for i, val in enumerate(self._coefficient_weights):
            if val != 1.0:
                ret[lbl_lookup[i]] = val
        return ret

    def set_coefficient_weights(self, weights):
        """
        Set the coefficient weights.

        This method sets the weights for the coefficients of the error generator. If the coefficient weights
        array is not initialized, it initializes it to an array of ones.

        Parameters
        ----------
        weights : dict
            A dictionary where keys are coefficient labels and values are the corresponding weights to set.
        """
        coeff_labels = self.coefficient_labels()
        ilbl_lookup = {lbl: i for i, lbl in enumerate(coeff_labels)}
        if self._coefficient_weights is None:
            self._coefficient_weights = _np.ones(len(self.coefficients_array()), 'd')
        for lbl, wt in weights.items():
            self._coefficient_weights[ilbl_lookup[lbl]] = wt

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
            err_gen_mx = Uinv  @ (err_gen_mx @ U)
            trunc = 1e-6 if isinstance(s, _gaugegroup.UnitaryGaugeGroupElement) else False
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
            _warnings.warn("Using finite differencing to compute LindbladErrorGen derivative!")
            return super(LindbladErrorgen, self).deriv_wrt_params(wrt_filter)

        dim = self.dim
        blk_superop_derivs = []; off = 0
        for blk, (superops, _) in zip(self.coefficient_blocks, self.lindblad_term_superops_and_1norms):
            superop_deriv = blk.superop_deriv_wrt_params(superops, self.paramvals[off: off + blk.num_params], True)
            superop_deriv = superop_deriv.reshape((dim**2, -1))  # [iFlattenedOp, iParam]
            blk_superop_derivs.append(superop_deriv)
            off += blk.num_params

        derivMx = _np.concatenate(blk_superop_derivs, axis=1)

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
        nTotParams = self.num_params
        hessianMx = _np.zeros((dim**2, nTotParams, nTotParams), 'd')

        # Hessian is block diagonal since every coefficient is only
        # dependent on the parameters of (at most) a single coefficient block.
        # For example, when there is a H and O block, the Hessian can be
        # split into 4 pieces:   d2H  |  dHdO
        #                        dHdO |  d2O
        # But only d2O is non-zero (and only when the O block's param_mode == 'cholesky'
        off = 0
        for blk, (superops, _) in zip(self.coefficient_blocks, self.lindblad_term_superops_and_1norms):
            blk_Np = blk.num_params
            superop_hessian = blk.superop_hessian_wrt_params(superops, self.paramvals[off: off + blk_Np], True)
            superop_hessian = superop_hessian.reshape((dim**2, blk_Np, blk_Np))  # [iFlattenedOp, iParam1, iParam2]

            hessianMx[:, off:off + blk_Np, off:off + blk_Np] = superop_hessian  # d2(blk)/dp2 block of hessian
            off += blk_Np

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
        mm_dict['matrix_basis'] = self.matrix_basis.to_nice_serialization()
        mm_dict['coefficient_blocks'] = [blk.to_nice_serialization() for blk in self.coefficient_blocks]
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        mx_basis = _Basis.from_nice_serialization(mm_dict['matrix_basis'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        coeff_blocks = [_LindbladCoefficientBlock.from_nice_serialization(blk)
                        for blk in mm_dict['coefficient_blocks']]

        return cls(coeff_blocks, 'auto', mx_basis, mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return (all([myblk.is_similar(oblk) for myblk, oblk in zip(self.coefficient_blocks, other.coefficient_blocks)])
                and (self.matrix_basis == other.matrix_basis))

    def __str__(self):
        s = "Lindblad error generator with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        return s

    def _oneline_contents(self):
        """ Summarizes the contents of this object in a single line.  Does not summarize submembers. """
        MAXLEN = 60
        coeff_dict = self.coefficients(); s = ""
        for lbl, val in coeff_dict.items():
            if len(s) > MAXLEN:
                s += "..."; break
            s += str(lbl) + ": " + str(_np.round(val, 3)) + ", "
        else:
            s = s[:-2] if len(s) > 2 else s  # trim ", " off end
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
    def minimal_from_elementary_errorgens(cls, errs):
        """
        Helper function giving minimal Lindblad parameterization needed for specified errors.

        Parameters
        ----------
        errs : dict
            Error dictionary with keys as `(termType, basisLabel)` tuples, where
            `termType` can be `"H"` (Hamiltonian), `"S"` (Stochastic), or `"A"`
            (Active), and `basisLabel` is a string of I, X, Y, or Z, or to describe a
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
        if any([lbl.errorgen_type == 'C' for lbl in errs]): paramtypes.append('C')
        if any([lbl.errorgen_type == 'A' for lbl in errs]): paramtypes.append('A')
        #if any([lbl.errorgen_type == 'S' and len(lbl.basis_element_labels) == 2 for lbl in errs]):
        #    # parameterization must be "CPTP" if there are any ('S',b1,b2) keys
        if 'C' in paramtypes or 'A' in paramtypes:
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

        if isinstance(obj, str):
            if obj.startswith('exp(') and obj.endswith(')'):
                abbrev = obj[len('exp('):-1]
                meta = 'exp'
            elif obj.startswith('1+(') and obj.endswith(')'):
                abbrev = obj[len('1+('):-1]
                meta = '1+'
            elif obj.startswith('lindblad '):
                _warnings.warn(("Use of 'lindblad <type>' is deprecated and will be removed.  "
                                "You should use 'exp(<type>)' or '1+(<type>)' instead"))
                abbrev = obj[len('lindblad '):]
                meta = 'exp'
            else:
                abbrev = obj
                meta = None  # 'exp' by default?

            if abbrev == "CPTP":
                _warnings.warn("Using 'CPTP' as a Lindblad type is deprecated, and you should now use 'CPTPLND'")
                block_types = ['ham', 'other']; param_modes = ['elements', 'cholesky']
            elif abbrev == "CPTPLND":
                block_types = ['ham', 'other']; param_modes = ['elements', 'cholesky']
            elif abbrev == "GLND":
                block_types = ['ham', 'other']; param_modes = ['elements', 'elements']
            else:
                block_types = []; param_modes = []
                for p in abbrev.split('+'):
                    if p == 'H':
                        block_types.append('ham'); param_modes.append('elements')
                    elif p in ('S', 's'):
                        block_types.append('other_diagonal')
                        param_modes.append('cholesky' if p == 'S' else 'elements')
                    elif p in ('D', 'd'):
                        block_types.append('other_diagonal')
                        param_modes.append('depol' if p == 'D' else 'reldepol')
                    else:
                        raise ValueError("Unrecognized symbol '%s' in `%s`" % (p, abbrev))
            return cls(block_types, param_modes, abbrev, meta)
        else:
            raise ValueError("Cannot convert %s to LindbladParameterization!" % str(type(obj)))

    def __init__(self, block_types, param_modes, abbrev=None, meta=None):
        self.block_types = tuple(block_types)
        self.param_modes = tuple(param_modes)
        self.abbrev = abbrev
        self.meta = meta

    def __hash__(self):
        return hash((self.block_types, self.param_modes))

    def __eq__(self, other):
        if not isinstance(other, LindbladParameterization): return False
        return (self.param_modes == other.param_modes
                and self.block_types == other.block_types)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'block_types': self.block_types,
                      'block_parameter_modes': self.param_modes,
                      'abbreviation': self.abbrev,
                      'meta_data': self.meta})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        if 'non_hamiltonian_mode' in state:  # FOR BACKWARD COMPATIBILITY (REMOVE in FUTURE)
            if state['hamiltonian_parameters_allowed']:
                state['block_types'] = ['ham']
                state['block_parameter_modes'] = ['elements']
            else:
                state['block_types'] = []
                state['block_parameter_modes'] = []
            if state['non_hamiltonian_parameters_allowed']:
                nonham_blktyp = 'other' if state['non_hamiltonian_mode'] == 'all' else 'other_diagonal'
                nonham_mode = state['mode']
                if nonham_mode == 'cptp': nonham_mode = 'cholesky'
                elif nonham_mode == 'unconstrained': nonham_mode = 'elements'
                state['block_types'].append(nonham_blktyp)
                state['block_parameter_modes'].append(nonham_mode)

        return cls(state['block_types'], state['block_parameter_modes'], state['abbreviation'],
                   state.get('meta_data', None))

    def __str__(self):
        if self.abbrev is not None:
            if self.meta is not None:
                return self.abbrev + "[%s]" % str(self.meta)
            else:
                return self.abbrev
        else:
            return "CustomLindbladParam(%s,%s%s)" % ('+'.join(self.block_types), '+'.join(self.param_modes),
                                                     (",%s" % str(self.meta) if (self.meta is not None) else ""))
