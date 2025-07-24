import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import collections as _collections
import copy as _copy
import warnings as _warnings

from pygsti.tools import lindbladtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools import fastcalc as _fc
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.modelmembers import term as _term
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable

from functools import lru_cache

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


class LindbladCoefficientBlock(_NicelySerializable):
    """ SCRATCH:
        This routine computes the Hamiltonian and Non-Hamiltonian ("other")
        superoperator generators which correspond to the terms of the Lindblad
        expression:

        L(rho) = sum_i( h_i [A_i,rho] ) +
                 sum_ij( o_ij * (B_i rho B_j^dag -
                                 0.5( rho B_j^dag B_i + B_j^dag B_i rho) ) )

        where {A_i} and {B_i} are bases (possibly the same) for Hilbert Schmidt
        (density matrix) space with the identity element removed so that each
        A_i and B_i are traceless.  If we write L(rho) in terms of superoperators
        H_i and O_ij,

        L(rho) = sum_i( h_i H_i(rho) ) + sum_ij( o_ij O_ij(rho) )

        then this function computes the matrices for H_i and O_ij using the given
        density matrix basis.  Thus, if `dmbasis` is expressed in the standard
        basis (as it should be), the returned matrices are also in this basis.

        If these elements are used as projectors it may be usedful to normalize
        them (by setting `normalize=True`).  Note, however, that these projectors
        are not all orthogonal - in particular the O_ij's are not orthogonal to
        one another.

        Parameters
        ----------
        dmbasis_ham : list
            A list of basis matrices {B_i} *including* the identity as the first
            element, for the returned Hamiltonian-type error generators.  This
            argument is easily obtained by call to  :func:`pp_matrices` or a
            similar function.  The matrices are expected to be in the standard
            basis, and should be traceless except for the identity.  Matrices
            should be NumPy arrays or SciPy CSR sparse matrices.

        dmbasis_other : list
            A list of basis matrices {B_i} *including* the identity as the first
            element, for the returned Stochastic-type error generators.  This
            argument is easily obtained by call to  :func:`pp_matrices` or a
            similar function.  The matrices are expected to be in the standard
            basis, and should be traceless except for the identity.  Matrices
            should be NumPy arrays or SciPy CSR sparse matrices.

        normalize : bool
            Whether or not generators should be normalized so that
            numpy.linalg.norm(generator.flat) == 1.0  Note that the generators
            will still, in general, be non-orthogonal.

        other_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad error generators to construct.
            Allowed values are: `"diagonal"` (only the diagonal Stochastic
            generators are returned; that is, the generators corresponding to the
            `i==j` terms in the Lindblad expression.), `"diag_affine"` (diagonal +
            affine generators), and `"all"` (all generators).
    """

    _superops_cache = {}  # a custom cache for create_lindblad_term_superoperators method calls

    def __init__(self, block_type, basis, basis_element_labels=None, initial_block_data=None, param_mode='static',
                 truncate=False):
        super().__init__()
        self._block_type = block_type  # 'ham' or 'other' or 'other_diagonal'
        self._param_mode = param_mode  # 'static', 'elements', 'cholesky', or 'real_cholesky', 'depol', 'reldepol'
        self._basis = basis  # must be a full Basis object, not just a string, as we otherwise don't know dimension
        self._bel_labels = tuple(basis_element_labels) if (basis_element_labels is not None) \
            else tuple(basis.labels[1:])  # Note: don't include identity
        self._cache_mx = _np.zeros((len(self._bel_labels), len(self._bel_labels)), 'complex') \
            if self._block_type == 'other' else None

        self._set_block_data(initial_block_data, truncate)

    def _set_block_data(self, block_data, truncate):
        #Sets self.block_data directly, which may later be overwritten by to/from vector calls so this
        # is somewhat dangerous to call, thus a private member.  It should really only be used during
        # initialization and to set/update a param_mode=='static' block.
        # Note: block_data == None can be used to initialize all-zero parameters
        num_bels = len(self._bel_labels)
        if self._block_type in ('ham', 'other_diagonal'):
            block_shape = (num_bels,); block_dtype = 'd'
        elif self._block_type == 'other':
            block_shape = (num_bels, num_bels); block_dtype = 'complex'
        else:
            raise ValueError("Invalid `block_type`: %s!" % str(self._block_type))

        if block_data is not None:
            assert(block_data.shape == block_shape), \
                "Invalid `initial_block_data` shape: expected %s, received %s" % (
                    str(block_shape), str(block_data.shape))
            assert(block_data.dtype == _np.dtype(block_dtype)), \
                "Invalid `initial_block_data` dtype: expected %s, received %s" % (
                    str(block_dtype), str(block_data.dtype))
            self.block_data = _np.ascontiguousarray(block_data)
        else:
            self.block_data = _np.zeros(block_shape, block_dtype)

        self._truncate_block_data(truncate)

    @property
    def basis_element_labels(self):
        return self._bel_labels

    @property
    def num_params(self):
        if self._param_mode == 'static': return 0
        if self._block_type == 'ham':
            if self._param_mode == 'elements': return len(self._bel_labels)
        elif self._block_type == 'other_diagonal':
            if self._param_mode in ('depol', 'reldepol'): return 1
            elif self._param_mode in ('elements', 'cholesky'): return len(self._bel_labels)
        elif self._block_type == 'other':
            if self._param_mode in ('elements', 'cholesky'): return len(self._bel_labels)**2
        else:
            raise ValueError("Internal error: invalid block type!")
        raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                         % (self._param_mode, self._block_type))

    def create_lindblad_term_superoperators(self, mx_basis='pp', sparse="auto", include_1norms=False, flat=False):
        """
        Compute the superoperator-generators corresponding to the Lindblad coefficiens in this block.
        TODO: docstring update

        Returns
        -------
        generators : numpy.ndarray or list of SciPy CSR matrices
            If parent holds a dense basis, this is an array of shape `(n,d,d)` for 'ham'
            and 'other_diagonal' blocks or `(n,n,d,d)` for 'other' blocks, where `d` is
            the *dimension* of the relevant basis and `n` is the number of basis elements
            included in this coefficient block.  In the case of 'ham' and 'other_diagonal' type
            blocks, `generators[i]` gives the superoperator matrix corresponding to the block's
            `i`-th coefficient (and `i`-th basis element).  In the 'other' case, `generators[i,j]`
            corresponds to the `(i,j)`-th coefficient.  If the parent holds a sparse basis,
            the dimensions of size `n` are lists of CSR matrices with shape `(d,d)`.
        """
        assert(self._basis is not None), "Cannot create lindblad superoperators without a basis!"
        basis = self._basis
        sparse = basis.sparse if (sparse == "auto") else sparse
        mxs = [basis[lbl] for lbl in self._bel_labels]
        nMxs = len(mxs)
        if nMxs == 0:
            return ([], []) if include_1norms else []  # short circuit - no superops to return

        d = mxs[0].shape[0]
        d2 = d**2
        mx_basis = _Basis.cast(mx_basis, d2, sparse=sparse)

        cache_key = (self._block_type, tuple(self._bel_labels), mx_basis, basis)
        if cache_key not in self._superops_cache:

            #Create "flat" list of superop matrices in 'std' basis
            if self._block_type == 'ham':
                superops = [None] * nMxs if sparse else _np.empty((nMxs, d2, d2), 'complex')
                for i, B in enumerate(mxs):
                    superops[i] = _lt.create_lindbladian_term_errorgen('H', B, sparse=sparse)  # in std basis
            elif self._block_type == 'other_diagonal':
                superops = [None] * nMxs if sparse else _np.empty((nMxs, d2, d2), 'complex')
                for i, Lm in enumerate(mxs):
                    superops[i] = _lt.create_lindbladian_term_errorgen('O', Lm, Lm, sparse)  # in std basis
            elif self._block_type == 'other':
                superops = [None] * nMxs**2 if sparse else _np.empty((nMxs**2, d2, d2), 'complex')
                for i, Lm in enumerate(mxs):
                    for j, Ln in enumerate(mxs):
                        superops[i * nMxs + j] = _lt.create_lindbladian_term_errorgen('O', Lm, Ln, sparse)
            else:
                raise ValueError("Invalid block_type '%s'" % str(self._block_type))

            # Convert matrices to desired `mx_basis` basis
            if mx_basis != "std":
                # Get basis transfer matrices from 'std' <-> desired mx_basis
                mxBasisToStd = mx_basis.create_transform_matrix(_BuiltinBasis("std", mx_basis.dim, sparse))
                # use BuiltinBasis("std") instead of just "std" in case mx_basis is a TensorProdBasis

                rightTrans = mxBasisToStd
                leftTrans = _spsl.inv(mxBasisToStd.tocsc()).tocsr() if _sps.issparse(mxBasisToStd) \
                    else _np.linalg.inv(mxBasisToStd)

                if sparse:
                    #Note: complex OK here sometimes, as only linear combos of "other" gens
                    # (like (i,j) + (j,i) terms) need to be real.
                    superops = [leftTrans @ (mx @ rightTrans) for mx in superops]
                    for mx in superops: mx.sort_indices()
                else:
                    #superops = _np.einsum("ik,akl,lj->aij", leftTrans, superops, rightTrans)
                    superops = _np.transpose(_np.tensordot(
                        _np.tensordot(leftTrans, superops, (1, 1)), rightTrans, (2, 0)), (1, 0, 2))

            superop_1norms = _np.array([_mt.safe_onenorm(mx) for mx in superops], 'd')

            self._superops_cache[cache_key] = (superops, superop_1norms)

        cached_superops, cached_superops_1norms = self._superops_cache[cache_key]

        if flat or self._block_type in ('ham', 'other_diagonal'):
            return (_copy.deepcopy(cached_superops), cached_superops_1norms.copy()) \
                if include_1norms else _copy.deepcopy(cached_superops)

        else:  # flat == False and block type is 'other', so need to reshape:
            superops = [[cached_superops[i * nMxs + j] for j in range(nMxs)] for i in range(nMxs)] \
                if sparse else cached_superops.copy().reshape((nMxs, nMxs, d2, d2))
            return (superops, cached_superops_1norms.copy().reshape((nMxs, nMxs))) \
                if include_1norms else superops

    def create_lindblad_term_objects(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        # needed b/c operators produced by lindblad_error_generators have an extra 'd' scaling
        #d = int(round(_np.sqrt(dim)))
        mpv = max_polynomial_vars
        pio = parameter_index_offset

        IDENT = None  # sentinel for the do-nothing identity op
        Lterms = []

        if self._block_type == 'ham':
            for k, bel_label in enumerate(self._bel_labels):  # k == index of local parameter that is coefficient
                # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
                scale, U = _mt.to_unitary(self._basis[bel_label])
                #scale /= 2  # DEBUG REMOVE - this makes ham terms w/PP the same as earlier versions of pyGSTi

                if self._param_mode == 'elements':
                    cpi = (pio + k,)  # coefficient's parameter indices (with offset)
                elif self._param_mode == 'static':
                    cpi = ()  # not multiplied by any parameters
                    scale *= self.block_data[k]  # but scale factor gets multiplied by (static) coefficient
                else: raise ValueError("Internal error: invalid param mode!!")

                #Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({cpi: -1j * scale}, mpv), U, IDENT, evotype, state_space))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({cpi: +1j * scale}, mpv), IDENT, U.conjugate().T, evotype, state_space))

        elif self._block_type == 'other_diagonal':
            for k, bel_label in enumerate(self._bel_labels):  # k == index of local parameter that is coefficient
                # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
                scale, U = _mt.to_unitary(self._basis[bel_label])
                scale = scale**2  # because there are two "U"s in each overall term below

                if self._param_mode in ('depol', 'reldepol'):
                    cpi = (pio + 0,)
                elif self._param_mode in ('cholesky', 'elements'):
                    cpi = (pio + k,)  # coefficient's parameter indices (with offset)
                elif self._param_mode == 'static':
                    cpi = ()  # not multiplied by any parameters
                    scale *= self.block_data[k]  # but scale factor gets multiplied by (static) coefficient
                else: raise ValueError("Internal error: invalid param mode!!")

                pw = 2 if self._param_mode in ("cholesky", "depol") else 1
                Lm = Ln = U
                Lm_dag = Lm.conjugate().T  # assumes basis is dense (TODO: make sure works
                Ln_dag = Ln.conjugate().T  # for sparse case too - and np.dots below!)

                #Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
                # e.g. in 2nd term, _np.dot(Ln_dag, Lm) == adjoint(_np.dot(Lm_dag,Ln))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({cpi * pw: 1.0 * scale}, mpv), Ln, Lm, evotype, state_space))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({cpi * pw: -0.5 * scale}, mpv), IDENT, _np.dot(Ln_dag, Lm), evotype, state_space))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    _Polynomial({cpi * pw: -0.5 * scale}, mpv), _np.dot(Lm_dag, Ln), IDENT, evotype, state_space))

        elif self._block_type == 'other':

            num_bels = len(self._bel_labels)
            for i, bel_labeli in enumerate(self._bel_labels):
                for j, bel_labelj in enumerate(self._bel_labels):

                    scalem, Um = _mt.to_unitary(self._basis[bel_labeli])  # ensure all Rank1Term operators are *unitary*
                    scalen, Un = _mt.to_unitary(self._basis[bel_labelj])  # ensure all Rank1Term operators are *unitary*
                    Lm, Ln = Um, Un
                    Lm_dag = Lm.conjugate().T; Ln_dag = Ln.conjugate().T
                    scale = scalem * scalen

                    polyTerms = {}
                    if self._param_mode == 'cholesky':
                        # coeffs = _np.dot(self.Lmx,self.Lmx.T.conjugate())
                        # coeffs_ij = sum_k Lik * Ladj_kj = sum_k Lik * conjugate(L_jk)
                        #           = sum_k (Re(Lik) + 1j*Im(Lik)) * (Re(L_jk) - 1j*Im(Ljk))
                        def i_re(a, b): return pio + (a * num_bels + b)
                        def i_im(a, b): return pio + (b * num_bels + a)
                        for k in range(0, min(i, j) + 1):
                            if k <= i and k <= j:
                                polyTerms[(i_re(i, k), i_re(j, k))] = 1.0
                            if k <= i and k < j:
                                polyTerms[(i_re(i, k), i_im(j, k))] = -1.0j
                            if k < i and k <= j:
                                polyTerms[(i_im(i, k), i_re(j, k))] = 1.0j
                            if k < i and k < j:
                                polyTerms[(i_im(i, k), i_im(j, k))] = 1.0
                    elif self._param_mode == 'elements':  # unconstrained
                        # coeffs_ij = param[i,j] + 1j*param[j,i] (coeffs == block_data is Hermitian)
                        ijIndx = pio + (i * num_bels + j)
                        jiIndx = pio + (j * num_bels + i)
                        polyTerms = {(ijIndx,): 1.0, (jiIndx,): 1.0j}
                    elif self._param_mode == 'static':
                        polyTerms = {(): self.block_data[i, j]}
                    else: raise ValueError("Internal error: invalid param mode!!")

                    #Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
                    base_poly = _Polynomial(polyTerms, mpv) * scale
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        1.0 * base_poly, Ln, Lm, evotype, state_space))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        -0.5 * base_poly, IDENT, _np.dot(Ln_dag, Lm), evotype, state_space))  # adjoint(dot(Lm_dag,Ln))
                    Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                        -0.5 * base_poly, _np.dot(Lm_dag, Ln), IDENT, evotype, state_space))
        else:
            raise ValueError("Invalid block_type '%s'" % str(self._block_type))

        #DEBUG
        #print("DB: params = ", list(enumerate(self.paramvals)))
        #print("DB: Lterms = ")
        #for i,lt in enumerate(Lterms):
        #    print("Term %d:" % i)
        #    print("  coeff: ", str(lt.coeff)) # list(lt.coeff.keys()) )
        #    print("  pre:\n", lt.pre_ops[0] if len(lt.pre_ops) else "IDENT")
        #    print("  post:\n",lt.post_ops[0] if len(lt.post_ops) else "IDENT")

        return Lterms

    #TODO: could cache this and update only when needed (would need to add dirty flag logic)
    @property
    def elementary_errorgen_indices(self):
        """
        TODO docstring - rewrite this docstring - especially return value!
        Constructs a  dictionary mapping Lindblad term labels to projection coefficients.

        This method is used for finding the index of a particular error generator coefficient
        in the 1D array formed by concatenating the Hamiltonian and flattened stochastic
        projection arrays.

        Parameters
        ----------
        ham_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct `ham_projs`.  Allowed values are Matrix-unit
            (std), Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt), list of
            numpy arrays, or a custom basis object.

        other_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
            The basis used to construct `other_projs`.  Allowed values are
            Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt),
            list of numpy arrays, or a custom basis object.

        other_mode : {"diagonal", "diag_affine", "all"}
            Which non-Hamiltonian Lindblad error projections `other_projs` includes.
            Allowed values are: `"diagonal"` (only the diagonal Stochastic),
            `"diag_affine"` (diagonal + affine generators), and `"all"`
            (all generators).

        Returns
        -------
        Ltermdict : dict
            Keys are `(termType, basisLabel1, <basisLabel2>)`
            tuples, where `termType` is `"H"` (Hamiltonian), `"S"` (Stochastic), or
            `"A"` (Affine).  Hamiltonian and Affine terms always have a single basis
            label (so key is a 2-tuple) whereas Stochastic tuples have 1 basis label
            to indicate a *diagonal* term and otherwise have 2 basis labels to
            specify off-diagonal non-Hamiltonian Lindblad terms.  Basis labels
            are taken from `ham_basis` and `other_basis`.  Values are integer indices.
        """

        # Note: returned dictionary's value specify a linear combination of
        # this coefficient block's coefficients that product the given (by the key)
        # elementary error generator.  Values are lists of (c_i, index_i) pairs,
        # such that the given elementary generator == sum_i c_i * coefficients_in_flattened_block[index_i]
        from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL

        elem_errgen_indices = _collections.OrderedDict()
        if self._block_type == 'ham':  # easy case, since these are elementary error generators
            for i, lbl in enumerate(self._bel_labels):
                elem_errgen_indices[_LEEL('H', (lbl,))] = [(1.0, i)]

        elif self._block_type == 'other_diagonal':  # easy case, since these are elementary error generators
            for i, lbl in enumerate(self._bel_labels):
                elem_errgen_indices[_LEEL('S', (lbl,))] = [(1.0, i)]

        elif self._block_type == 'other':
            # Difficult case, as coefficients do not correspond to elementary errorgens, so
            # there's no single index for, e.g. ('C', lbl1, lbl2) - rather this elementary
            # errorgen is a linear combination of two coefficients.
            stride = len(self._bel_labels)
            for i, lbl1 in enumerate(self._bel_labels):
                ii = i * stride + i
                elem_errgen_indices[_LEEL('S', (lbl1,))] = [(1.0, ii)]

                for j, lbl2 in enumerate(self._bel_labels[i + 1:], start=i + 1):
                    ij = i * stride + j
                    ji = j * stride + i

                    #Contributions from C_PQ and A_PQ coeffs NH_PQ and NH_QP coeffs:
                    # NH_PQ = (C_PQ + i A_PQ)/2
                    # NH_QP = (C_PQ - i A_PQ)/2
                    elem_errgen_indices[_LEEL('C', (lbl1, lbl2))] = [(0.5, ij), (0.5, ji)]  # C_PQ contributions
                    elem_errgen_indices[_LEEL('A', (lbl1, lbl2))] = [(0.5j, ij), (-0.5j, ji)]  # A_PQ contributions
        else:
            raise ValueError("Internal error: invalid block type!")

        return elem_errgen_indices

    @property
    def _block_data_indices(self):
        """
        Effectively the inverse of elementary_errorgen_indices.

        The keys of the returned dict are (flattened) block_data indices and the
        values specify a linear combination of elementary errorgens via their labels.
        """
        from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL

        block_data_indices = _collections.OrderedDict()
        if self._block_type == 'ham':  # easy case, since these are elementary error generators
            for i, lbl in enumerate(self._bel_labels):
                block_data_indices[i] = [(1.0, _LEEL('H', (lbl,)))]

        elif self._block_type == 'other_diagonal':  # easy case, since these are elementary error generators
            for i, lbl in enumerate(self._bel_labels):
                block_data_indices[i] = [(1.0, _LEEL('S', (lbl,)))]

        elif self._block_type == 'other':
            # Difficult case, as coefficients do not correspond to elementary errorgens, so
            # there's no single index for, e.g. ('C', lbl1, lbl2) - rather this elementary
            # errorgen is a linear combination of two coefficients.
            stride = len(self._bel_labels)
            for i, lbl1 in enumerate(self._bel_labels):
                ii = i * stride + i
                block_data_indices[ii] = [(1.0, _LEEL('S', (lbl1,)))]
                for j, lbl2 in enumerate(self._bel_labels[i + 1:], start=i + 1):
                    ij = i * stride + j
                    ji = j * stride + i

                    #Contributions from NH_PQ and NH_QP coeffs to C_PQ and A_PQ coeffs:
                    # C_PQ = NH_PQ + NH_QP
                    # A_PQ = i(NH_QP - NH_PQ)
                    block_data_indices[ij] = [(1.0, _LEEL('C', (lbl1, lbl2))), (-1.0j, _LEEL('A', (lbl1, lbl2)))]
                    # NH_PQ = (C_PQ + i A_PQ)/2, but here we care that NH_PQ appears w/1.0 in C_PQ and -1j in A_QP
                    block_data_indices[ji] = [(1.0, _LEEL('C', (lbl1, lbl2))), (+1.0j, _LEEL('A', (lbl1, lbl2)))]
                    # NH_QP = (C_PQ - i A_PQ)/2, but here we care that NH_QP appears w/1.0 in C_PQ and +1j in A_QP
        else:
            raise ValueError("Internal error: invalid block type!")

        return block_data_indices

    #TODO: could cache this and update only when needed (would need to add dirty flag logic)
    @property
    def elementary_errorgens(self):
        """
        Converts a set of coefficients for this block into a linear combination of elementary error generators.

        This linear combination is given by a dictionary with keys equal to elementary
        error generator labels and values equal to their coefficients in the linear combination.

        Parameters
        ----------
        block_data : numpy.ndarray
            A 1- or 2-dimensional array with each dimension of size `len(self.basis_element_labels)`,
            specifying the coefficients of this block.  Array is 1-dimensional when this block is
            of type `'ham'` or `'other_diagonal'` and is 2-dimensional for type `'other'`.

        Returns
        -------
        elementary_errorgens : dict
            Specifies `block_data` as a linear combination of elementary error generators.
            Keys are :class:`LocalElementaryErrorgenLabel` objects and values are floats.
        """
        elementary_errorgens = _collections.OrderedDict()
        eeg_indices = self.elementary_errorgen_indices
        flat_data = self.block_data.ravel()

        for eeg_lbl, linear_combo in eeg_indices.items():
            val = _np.sum([coeff * flat_data[index] for coeff, index in linear_combo])
            elementary_errorgens[eeg_lbl] = _np.real_if_close(val).item()  # item() -> scalar
            #set_basis_el(lbl, basis[lbl])  # REMOVE

        return elementary_errorgens

    def set_elementary_errorgens(self, elementary_errorgens, on_missing='ignore', truncate=False):
        # Note: could return a "stripped" version of elementary_errorgens
        # expects a dict with keys == local elem errgen labels
        flat_data = self.block_data.ravel().copy()
        unused_elementary_errorgens = elementary_errorgens.copy()  # to return
        for i, linear_combo in self._block_data_indices.items():
            val = 0
            for coeff, eeg_lbl in linear_combo:
                if eeg_lbl in elementary_errorgens:
                    val += coeff * elementary_errorgens[eeg_lbl]
                    if eeg_lbl in unused_elementary_errorgens:
                        del unused_elementary_errorgens[eeg_lbl]
                elif on_missing == 'warn':
                    _warnings.warn("Missing entry for %s in dictionary of elementary errorgens.  Assuming 0."
                                   % str(eeg_lbl))
                elif on_missing == 'raise':
                    raise ValueError("Missing entry for %s in dictionary of elementary errorgens." % str(eeg_lbl))
            flat_data[i] = val

        self.block_data[(slice(None, None),) * self.block_data.ndim] = flat_data.reshape(self.block_data.shape)
        self._truncate_block_data(truncate)

        return unused_elementary_errorgens

    def set_from_errorgen_projections(self, errorgen, errorgen_basis='pp', return_projected_errorgen=False,
                                      truncate=False):
        elementary_errorgen_lbls = list(self.elementary_errorgens.keys())
        out = _ot.extract_elementary_errorgen_coefficients(errorgen, elementary_errorgen_lbls,
                                                           self._basis, errorgen_basis,
                                                           return_projected_errorgen)
        elementary_errorgens = out[0] if return_projected_errorgen else out
        unused = self.set_elementary_errorgens(elementary_errorgens, on_missing='raise', truncate=truncate)
        assert(len(unused) == 0)
        return out[1] if return_projected_errorgen else None

    @property
    def coefficient_labels(self):
        """Labels for the elements of `self.block_data` (flattened if relevant)"""
        if self._block_type == 'ham':
            labels = ["%s Hamiltonian error coefficient" % lbl for lbl in self._bel_labels]
        elif self._block_type == 'other_diagonal':
            labels = ["(%s,%s) other error coefficient" % (lbl, lbl) for lbl in self._bel_labels]
        elif self._block_type == 'other':
            labels = []
            for i, ilbl in enumerate(self._bel_labels):
                for j, jlbl in enumerate(self._bel_labels):
                    labels.append("(%s,%s) other error coefficient" % (ilbl, jlbl))
        else:
            raise ValueError("Internal error: invalid block type!")
        return labels

    @property
    def param_labels(self):
        """
        Generate human-readable labels for the parameters of this block.

        Returns
        -------
        param_labels : list
            A list of strings that describe each parameter.
        """
        if self._param_mode == 'static':
            return []  # static parameterization has zero parameters thus no labels

        if self._block_type == 'ham':
            if self._param_mode == 'elements':
                labels = ["%s Hamiltonian error coefficient" % lbl for lbl in self._bel_labels]
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            if self._param_mode in ("depol", "reldepol"):
                if self._param_mode == "depol":
                    labels = ["sqrt(common stochastic error coefficient for depolarization)"]
                else:  # "reldepol" -- no sqrt since not necessarily positive
                    labels = ["common stochastic error coefficient for depolarization"]

            elif self._param_mode == "cholesky":
                labels = ["sqrt(%s stochastic coefficient)" % lbl for lbl in self._bel_labels]
            elif self._param_mode == 'elements':
                labels = ["%s stochastic coefficient" % lbl for lbl in self._bel_labels]
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other':
            labels = []
            if self._param_mode == "cholesky":  # params mx stores Cholesky decomp
                for i, ilbl in enumerate(self._bel_labels):
                    for j, jlbl in enumerate(self._bel_labels):
                        if i == j: labels.append("%s diagonal element of non-H coeff Cholesky decomp" % ilbl)
                        elif j < i: labels.append("Re[(%s,%s) element of non-H coeff Cholesky decomp]" % (ilbl, jlbl))
                        else: labels.append("Im[(%s,%s) element of non-H coeff Cholesky decomp]" % (ilbl, jlbl))

            elif self._param_mode == "elements":  # params mx stores (hermitian) coefficient matirx directly
                for i, ilbl in enumerate(self._bel_labels):
                    for j, jlbl in enumerate(self._bel_labels):
                        if i == j: labels.append("%s diagonal element of non-H coeff matrix" % ilbl)
                        elif j < i: labels.append("Re[(%s,%s) element of non-H coeff matrix]" % (ilbl, jlbl))
                        else: labels.append("Im[(%s,%s) element of non-H coeff matrix]" % (ilbl, jlbl))
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
        else:
            raise ValueError("Internal error: invalid block type!")

        return labels

    def _block_data_to_params(self, truncate=False):
        """
        Compute parameter values from coefficient values.

        Constructs an array of paramter values from an arrays of
        coefficient values for this block.

        Parameters
        ----------
        truncate : bool or float, optional
            Whether to truncate the coefficients given by `block_data` in
            order to meet constraints (e.g. to preserve CPTP) when necessary.
            If >= 0 or False, then an error is thrown when the given projections
            cannot be parameterized as specified, using the value given as the
            the maximum negative eigenvalue that is tolerated (`False` is equivalent
            to 1e-12).  True tolerates *any* negative eigenvalues.

        Returns
        -------
        param_vals : numpy.ndarray
            A 1D array of real parameter values. Length is `len(self.basis_element_labels)`
            in the case of `'ham'` or `'other_diagonal'` blocks, and `len(self.basis_element_labels)**2`
            in the case of `'other'` blocks.
        """
        if truncate is False:
            ttol = -1e-15  # (was -1e-12) # truncation tolerance
        elif truncate is True:
            ttol = -_np.inf
        else:
            ttol = -truncate

        if self._param_mode == 'static':
            return _np.empty(0, 'd')

        if self._block_type == 'ham':
            if self._param_mode == "elements":
                params = self.block_data.real
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            if self._param_mode in ("depol", "reldepol"):
                # params is a *single-element* 1D vector of the sqrt of each diagonal el
                assert(self._param_mode == "reldepol" or all([v >= ttol for v in self.block_data])), \
                    "Lindblad stochastic coefficients are not positive (truncate == %s)!" % str(truncate)
                assert(all([_np.isclose(v, self.block_data[0], atol=1e-6) for v in self.block_data])), \
                    "Diagonal lindblad coefficients are not equal (truncate == %s)!" % str(truncate)
                if self._param_mode == "depol":
                    avg_coeff = _np.mean(self.block_data.clip(0, 1e100))  # was 1e-16
                    params = _np.array([_np.sqrt(_np.real(avg_coeff))], 'd')  # shape (1,)
                else:  # "reldepol" -- no sqrt since not necessarily positive
                    avg_coeff = _np.mean(self.block_data)
                    params = _np.array([_np.real(avg_coeff)], 'd')  # shape (1,)

            elif self._param_mode == "cholesky":  # params is a 1D vector of the sqrts of diagonal els
                assert(all([v >= ttol for v in self.block_data])), \
                    "Lindblad stochastic coefficients are not positive (truncate == %s)!" % str(truncate)
                coeffs = self.block_data.clip(0, 1e100)  # was 1e-16
                params = _np.sqrt(coeffs.real)  # shape (len(self._bel_labels),)
            elif self._param_mode == "elements":  # "unconstrained":
                # params is a 1D vector of the real diagonal els of coefficient mx
                params = self.block_data.real  # shape (len(self._bel_labels),)
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other':
            assert(_np.isclose(_np.linalg.norm(self.block_data - self.block_data.T.conjugate()), 0)
                   ), "Lindblad 'other' coefficient mx is not Hermitian!"

            num_bels = len(self._bel_labels)
            params = _np.empty((num_bels, num_bels), 'd')

            if self._param_mode == "cholesky":  # params mx stores Cholesky decomp
                #assert(_np.allclose(block_data, block_data.T.conjugate()))

                #Identify any all-zero row&col indices, i.e. when the i-th row and i-th
                # column are all zero.  When this happens, remove them from the cholesky decomp,
                # algorithm and perofrm this decomp manually: the corresponding Lmx row & col
                # are just 0:
                zero_inds = set([i for i in range(self.block_data.shape[0])
                                 if (_np.linalg.norm(self.block_data[i, :])
                                     + _np.linalg.norm(self.block_data[:, i])) < 1e-12 * self.block_data.shape[0]])
                num_nonzero = self.block_data.shape[0] - len(zero_inds)

                next_nonzero = 0; next_zero = num_nonzero
                perm = _np.zeros(self.block_data.shape, 'd')  # permute all zero rows/cols to end
                for i in range(self.block_data.shape[0]):
                    if i in zero_inds:
                        perm[next_zero, i] = 1.0; next_zero += 1
                    else:
                        perm[next_nonzero, i] = 1.0; next_nonzero += 1

                perm_block_data = perm @ self.block_data @ perm.T
                nonzero_block_data = perm_block_data[0:num_nonzero, 0:num_nonzero]
                assert(_np.isclose(_np.linalg.norm(self.block_data), _np.linalg.norm(nonzero_block_data)))

                #evals, U = _np.linalg.eigh(nonzero_block_data)  # works too (assert hermiticity above)
                evals, U = _np.linalg.eig(nonzero_block_data)

                assert(all([ev > ttol for ev in evals])), \
                    ("Lindblad coefficients are not CPTP (truncate == %s)! (largest neg = %g)"
                     % (str(truncate), min(evals.real)))

                if ttol < 0:  # if we're truncating and assert above allows *negative* eigenvalues
                    #push any slightly negative evals of other_projs positive so that
                    # the Cholesky decomp will work.
                    Ui = _np.linalg.inv(U)
                    pos_evals = evals.clip(1e-16, None)
                    nonzero_block_data = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))
                    try:
                        nonzero_Lmx = _np.linalg.cholesky(nonzero_block_data)
                        # if Lmx not postitive definite, try again with 1e-12 (same lines as above)
                    except _np.linalg.LinAlgError:                         # pragma: no cover
                        pos_evals = evals.clip(1e-12, 1e100)                # pragma: no cover
                        nonzero_block_data = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))  # pragma: no cover
                        nonzero_Lmx = _np.linalg.cholesky(nonzero_block_data)
                else:  # truncate == False or == 0 case
                    nonzero_Lmx = _np.linalg.cholesky(nonzero_block_data)

                perm_Lmx = _np.zeros(self.block_data.shape, 'complex')
                perm_Lmx[0:num_nonzero, 0:num_nonzero] = nonzero_Lmx
                Lmx = perm.T @ perm_Lmx @ perm

                for i in range(num_bels):
                    assert(_np.linalg.norm(_np.imag(Lmx[i, i])) < IMAG_TOL)
                    params[i, i] = Lmx[i, i].real
                    for j in range(i):
                        params[i, j] = Lmx[i, j].real
                        params[j, i] = Lmx[i, j].imag

            elif self._param_mode == "elements":  # params mx stores block_data (hermitian) directly
                for i in range(num_bels):
                    assert(_np.linalg.norm(_np.imag(self.block_data[i, i])) < IMAG_TOL)
                    params[i, i] = self.block_data[i, i].real
                    for j in range(i):
                        params[i, j] = self.block_data[i, j].real
                        params[j, i] = self.block_data[i, j].imag
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
            params = params.ravel()  # make into 1D parameter array

        else:
            raise ValueError("Internal error: invalid block type!")

        assert(not _np.iscomplexobj(params))   # params should always be *real*
        assert(len(params) == self.num_params)
        return params

    def _truncate_block_data(self, truncate=False):
        if truncate is False: return
        v = self._block_data_to_params(truncate)
        self.from_vector(v)

    def to_vector(self):
        """
        Compute parameter values for this coefficient block.

        Returns
        -------
        param_vals : numpy.ndarray
            A 1D array of real parameter values. Length is `len(self.basis_element_labels)`
            in the case of `'ham'` or `'other_diagonal'` blocks, and `len(self.basis_element_labels)**2`
            in the case of `'other'` blocks.
        """
        return self._block_data_to_params(truncate=False)

    def from_vector(self, v):
        """
        Construct Lindblad coefficients (for this block) from a set of parameter values.

        This function essentially performs the inverse of
        :meth:`coefficients_to_paramvals`.

        Parameters
        ----------
        v : numpy.ndarray
            A 1D array of real parameter values.
        """
        if self._param_mode == 'static':
            assert(len(v) == 0), "'static' paramterized blocks should have zero parameters!"
            return  # self.block_data remains the same - no update

        if self._block_type == 'ham':
            if self._param_mode == 'elements':
                self.block_data[:] = v
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            num_bels = len(self._bel_labels)
            expected_shape = (1,) if (self._param_mode in ("depol", "reldepol")) else (num_bels,)
            assert(v.shape == expected_shape)

            # compute intermediate `p` that inflates 1-param depol cases to be full length (num_bels)
            p = v[0] * _np.ones(num_bels, 'd') \
                if self._param_mode in ("depol", "reldepol") else v

            if self._param_mode in ("cholesky", "depol"):  # constrained-to-positive param modes
                self.block_data[:] = p**2
            elif self._param_mode in ("reldepol", "elements"):
                self.block_data[:] = p
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other':
            num_bels = len(self._bel_labels)
            params = v.reshape((num_bels, num_bels))

            if self._param_mode == "cholesky":
                #  params is an array of length num_bels*num_bels that
                #  encodes a lower-triangular matrix "cache_mx" via:
                #  cache_mx[i,i] = params[i,i]
                #  cache_mx[i,j] = params[i,j] + 1j*params[j,i] (i > j)

                cache_mx = self._cache_mx

                params_upper_indices = _fc.fast_triu_indices(num_bels) 
                params_upper = 1j*params[params_upper_indices]
                params_lower = (params.T)[params_upper_indices]

                cache_mx_trans = cache_mx.T
                cache_mx_trans[params_upper_indices] = params_lower + params_upper
                        
                diag_indices = cached_diag_indices(num_bels)
                cache_mx[diag_indices] = params[diag_indices]
                
                #The matrix of (complex) "other"-coefficients is build by assuming
                # cache_mx is its Cholesky decomp; means otherCoeffs is pos-def.

                # NOTE that the Cholesky decomp with all positive real diagonal
                # elements is *unique* for a given positive-definite block_data
                # matrix, but we don't care about this uniqueness criteria and so
                # the diagonal els of cache_mx can be negative and that's fine -
                # block_data will still be posdef.
                self.block_data[:, :] = cache_mx@cache_mx.T.conj()


            elif self._param_mode == "elements":  # params mx stores block_data (hermitian) directly
                #params holds block_data real and imaginary parts directly
                params_upper_indices = _fc.fast_triu_indices(num_bels) 
                params_upper = -1j*params[params_upper_indices]
                params_lower = (params.T)[params_upper_indices]

                block_data_trans = self.block_data.T
                self.block_data[params_upper_indices] = params_lower + params_upper
                block_data_trans[params_upper_indices] = params_lower - params_upper

                diag_indices = cached_diag_indices(num_bels)
                self.block_data[diag_indices] = params[diag_indices]

            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
        else:
            raise ValueError("Internal error: invalid block type!")

    #def paramvals_to_coefficients_deriv(self, parameter_values, cache_mx=None):
    def deriv_wrt_params(self, v=None):
        """
        Construct derivative of Lindblad coefficients (for this block) from a set of parameter values.

        This function gives the Jacobian of what is returned by
        :func:`paramvals_to_coefficients` (as a function of the parameters).

        Parameters
        ----------
        v : numpy.ndarray, optional
            A 1D array of real parameter values.  If not specified, then self.to_vector() is used.

        Returns
        -------
        block_data_deriv : numpy.ndarray
            A real array of shape `(nBEL,nP)` or `(nBEL,nBEL,nP)`, depending on the block type,
            where `nBEL` is this block's number of basis elements (see `self.basis_element_labels`)
            and `nP` is the number of parameters (the length of `parameter_values`).
        """
        num_bels = len(self._bel_labels)
        v = self.to_vector() if (v is None) else v
        nP = len(v)
        assert(nP == self.num_params)

        if self._param_mode == 'static':
            if self._block_type in ('ham', 'other_diagonal'):
                return _np.zeros((num_bels, 0), 'd')
            elif self._block_type == 'other':
                return _np.zeros((num_bels, num_bels, 0), 'd')
            else: raise ValueError("Internal error: invalid block type!")

        if self._block_type == 'ham':
            if self._param_mode == 'elements':
                assert(nP == num_bels), "expected number of parameters == %d, not %d!" % (num_bels, nP)
                block_data_deriv = _np.identity(num_bels, 'd')
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            block_data_deriv = _np.zeros((num_bels, nP), 'd')
            if self._param_mode in ("depol", "reldepol"):
                assert(nP == 1), "expected number of parameters == 1, not %d!" % nP
                if self._param_mode == "depol":
                    block_data_deriv[:, 0] = 2.0 * v[0]
                else:  # param_mode == "reldepol"
                    block_data_deriv[:, 0] = 1.0

            elif self._param_mode == "cholesky":
                assert(nP == num_bels), "expected number of parameters == %d, not %d!" % (num_bels, nP)
                block_data_deriv[:, :] = 2.0 * _np.diag(v)
            elif self._param_mode == "elements":  # "unconstrained"
                assert(nP == num_bels), "expected number of parameters == %d, not %d!" % (num_bels, nP)
                block_data_deriv[:, :] = _np.identity(num_bels, 'd')
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other':
            params = v.reshape((num_bels, num_bels))
            cache_mx = self._cache_mx
            dcache_mx = _np.zeros((nP, num_bels, num_bels), 'complex')
            stride = num_bels

            if self._param_mode == "cholesky":
                #  params is an array of length (num_bels)*(num_bels) that
                #  encodes a lower-triangular matrix "cache_mx" via:
                #  cache_mx[i,i] = params[i,i]
                #  cache_mx[i,j] = params[i,j] + 1j * params[j,i] (i > j)
                for i in range(num_bels):
                    cache_mx[i, i] = params[i, i]
                    dcache_mx[i * stride + i, i, i] = 1.0
                    for j in range(i):
                        cache_mx[i, j] = params[i, j] + 1j * params[j, i]
                        dcache_mx[i * stride + j, i, j] = 1.0
                        dcache_mx[j * stride + i, i, j] = 1.0j

                #The matrix of (complex) "other"-coefficients is build by
                # assuming cache_mx is its Cholesky decomp; means otherCoeffs
                # is pos-def.

                # NOTE that the Cholesky decomp with all positive real diagonal
                # elements is *unique* for a given positive-definite block_data
                # matrix, but we don't care about this uniqueness criteria and so
                # the diagonal els of cache_mx can be negative and that's fine -
                # block_data will still be posdef.
                #block_data = _np.dot(cache_mx, cache_mx.T.conjugate())  # C * C^T
                block_data_deriv = _np.dot(dcache_mx, cache_mx.T.conjugate()) \
                    + _np.dot(cache_mx, dcache_mx.conjugate().transpose((0, 2, 1))).transpose((1, 0, 2))
                # deriv = dC * C^T + C * dC^T

                block_data_deriv = _np.rollaxis(block_data_deriv, 0, 3)  # => shape = (num_bels, num_bels, nP)

            elif self._param_mode == "elements":  # params mx stores block_data (hermitian) directly
                # parameter_values holds block_data real and imaginary parts directly
                block_data_deriv = _np.zeros((num_bels, num_bels, nP), 'complex')

                for i in range(num_bels):
                    block_data_deriv[i, i, i * stride + i] = 1.0
                    for j in range(i):
                        block_data_deriv[i, j, i * stride + j] = 1.0
                        block_data_deriv[i, j, j * stride + i] = 1.0j
                        block_data_deriv[j, i, i * stride + j] = 1.0
                        block_data_deriv[j, i, j * stride + i] = -1.0j

            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
        else:
            raise ValueError("Internal error: invalid block type!")

        return block_data_deriv

    def elementary_errorgen_deriv_wrt_params(self, v=None):
        eeg_indices = self.elementary_errorgen_indices
        blkdata_deriv = self.deriv_wrt_params(v)
        if blkdata_deriv.ndim == 3:  # (coeff_dim_1, coeff_dim_2, param_dim) => (coeff_dim, param_dim)
            blkdata_deriv = blkdata_deriv.reshape((blkdata_deriv.shape[0] * blkdata_deriv.shape[1],
                                                   blkdata_deriv.shape[2]))  # blkdata_deriv rows <=> flat_data indices

        eeg_deriv = _np.zeros((len(eeg_indices), self.num_params), 'd')  # may need to be complex?

        # Note: ordering in eeg_indices matches that of self.elementary_errorgens (as it must for this to be correct)
        for i, (eeg_lbl, linear_combo) in enumerate(eeg_indices.items()):
            deriv = _np.sum([coeff * blkdata_deriv[index, :] for coeff, index in linear_combo], axis=0)
            eeg_deriv[i, :] = _np.real_if_close(deriv)
        return eeg_deriv

    def superop_deriv_wrt_params(self, superops, v=None, superops_are_flat=False):
        """
        TODO: docstring

        superops : numpy.ndarray
            Output of create_lindblad_term_superoperators (with `flat=True` if
            `superops_are_flat==True`), so that this is a 3- or 4-dimensional array
            indexed by `(iSuperop, superop_row, superop_col)` or
            `(iSuperop1, iSuperop2, superop_row, superop_col)`.

        Returns
        -------
        numpy.ndarray
            per-superop-element derivative, indexed by `(superop_row, superop_col, parameter_index)`
            or `(superop_row, superop_col, parameter_index1, parameter_index2)` where there are two
            parameter indices because parameters are indexed by an (i,j) pair rather than a single index.
        """
        if self._param_mode == 'static':
            if superops_are_flat or self._block_type != 'other':
                return _np.zeros((superops.shape[1], superops.shape[2], 0), 'd')
            else:
                return _np.zeros((superops.shape[2], superops.shape[3], 0), 'd')

        if self._block_type == 'ham':
            if self._param_mode == 'elements':
                dOdp = superops.transpose((1, 2, 0))  # PRETRANS, this was:
                # _np.einsum("ik,akl,lj->ija", self.leftTrans, self.hamGens, self.rightTrans)
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            if v is None: v = self.to_vector()
            assert(len(v) == self.num_params)

            # Derivative of exponent wrt other param; shape == [dim,dim,bs-1]
            #  except "depol" & "reldepol" cases, when shape == [dim,dim,1]
            if self._param_mode == "depol":  # all coeffs same & == param^2
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None] * 2*otherParams[0]
                dOdp = _np.sum(_np.transpose(superops, (1, 2, 0)), axis=2)[:, :, None] * 2 * v[0]
            elif self._param_mode == "reldepol":  # all coeffs same & == param
                #dOdp  = _np.einsum('alj->lj', self.otherGens)[:,:,None]
                dOdp = _np.sum(_np.transpose(superops, (1, 2, 0)), axis=2)[:, :, None] * 2 * v[0]
            elif self._param_mode == "cholesky":  # (coeffs = params^2)
                #dOdp  = _np.einsum('alj,a->lja', self.otherGens, 2*otherParams)
                dOdp = _np.transpose(superops, (1, 2, 0)) * 2 * v  # just a broadcast
            elif self._param_mode == "elements":  # "unconstrained" (coeff == params)
                #dOdp  = _np.einsum('alj->lja', self.otherGens)
                dOdp = _np.transpose(superops, (1, 2, 0))
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other':
            num_bels = len(self._bel_labels)

            if self._param_mode == "cholesky":
                if superops_are_flat:  # then un-flatten
                    superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
                L, Lbar = self._cache_mx, self._cache_mx.conjugate()
                F1 = _np.tril(_np.ones((num_bels, num_bels), 'd'))
                F2 = _np.triu(_np.ones((num_bels, num_bels), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [dim,dim,bel_index,bel_index]
                # Note: replacing einsums here results in at least 3 numpy calls (probably slower?)
                dOdp = _np.einsum('amlj,mb,ab->ljab', superops, Lbar, F1)  # only a >= b nonzero (F1)
                dOdp += _np.einsum('malj,mb,ab->ljab', superops, L, F1)    # ditto
                dOdp += _np.einsum('bmlj,ma,ab->ljab', superops, Lbar, F2)  # only b > a nonzero (F2)
                dOdp += _np.einsum('mblj,ma,ab->ljab', superops, L, F2.conjugate())  # ditto
            elif self._param_mode == "elements":  # "unconstrained"
                if superops_are_flat:  # then un-flatten
                    superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
                F0 = _np.identity(num_bels, 'd')
                F1 = _np.tril(_np.ones((num_bels, num_bels), 'd'), -1)
                F2 = _np.triu(_np.ones((num_bels, num_bels), 'd'), 1) * 1j

                # Derivative of exponent wrt other param; shape == [dim,dim,bs-1,bs-1]
                #dOdp  = _np.einsum('ablj,ab->ljab', self.otherGens, F0)  # a == b case
                #dOdp += _np.einsum('ablj,ab->ljab', self.otherGens, F1) + \
                #           _np.einsum('balj,ab->ljab', self.otherGens, F1) # a > b (F1)
                #dOdp += _np.einsum('balj,ab->ljab', self.otherGens, F2) - \
                #           _np.einsum('ablj,ab->ljab', self.otherGens, F2) # a < b (F2)
                tmp_ablj = _np.transpose(superops, (2, 3, 0, 1))  # ablj -> ljab
                tmp_balj = _np.transpose(superops, (2, 3, 1, 0))  # balj -> ljab
                dOdp = tmp_ablj * F0  # a == b case
                dOdp += tmp_ablj * F1 + tmp_balj * F1  # a > b (F1)
                dOdp += tmp_balj * F2 - tmp_ablj * F2  # a < b (F2)
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
        else:
            raise ValueError("Internal error: invalid block type!")

        # apply basis transform
        tr = len(dOdp.shape)  # tensor rank
        assert((tr - 2) in (1, 2)), "Currently, dodp can only have 1 or 2 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(dOdp)) < IMAG_TOL)
        return _np.real(dOdp)

    def superop_hessian_wrt_params(self, superops, v=None, superops_are_flat=False):
        """
        TODO: docstring

        Returns
        -------
        numpy.ndarray
            Indexed by (superop_row, superop_col, param1, param2).
        """
        if self._param_mode == 'static':
            if superops_are_flat or self._block_type != 'other':
                return _np.zeros((superops.shape[1], superops.shape[2], 0, 0), 'd')
            else:
                return _np.zeros((superops.shape[2], superops.shape[3], 0, 0), 'd')

        num_bels = len(self._bel_labels)
        nP = self.num_params

        if self._block_type == 'ham':
            if self._param_mode == 'elements':
                d2Odp2 = _np.zeros((superops.shape[1], superops.shape[2], nP, nP), 'd')
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))

        elif self._block_type == 'other_diagonal':
            if v is None: v = self.to_vector()
            assert(len(v) == nP)

            # Derivative of exponent wrt other param; shape == [dim,dim,nP,nP]
            if self._param_mode == "depol":
                #d2Odp2  = _np.einsum('alj->lj', self.otherGens)[:,:,None,None] * 2
                d2Odp2 = _np.sum(superops, axis=0)[:, :, None, None] * 2
            elif self.parameterization.param_mode == "cptp":
                assert(nP == num_bels)
                #d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
                d2Odp2 = _np.transpose(superops, (1, 2, 0))[:, :, :, None] * 2 * _np.identity(nP, 'd')
            else:  # param_mode == "unconstrained" or "reldepol"
                assert(nP == num_bels)
            d2Odp2 = _np.zeros((superops.shape[1], superops.shape[2], nP, nP), 'd')

        elif self._block_type == 'other':
            if self._param_mode == "cholesky":
                if superops_are_flat:  # then un-flatten
                    superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
                d2Odp2 = _np.zeros([superops.shape[2], superops.shape[3], nP, nP, nP, nP], 'complex')
                # yikes! maybe make this SPARSE in future?

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
                    d2Odp2[:, :, a, base, q, base] = superops[a, q] + superops[q, a]
                for base, a, r in iter_base_ab_qr(True, False):  # Case2: base=b=q, ab=a, qr=r
                    d2Odp2[:, :, a, base, base, r] = -1j * superops[a, r] + 1j * superops[r, a]
                for base, b, q in iter_base_ab_qr(False, True):  # Case3: base=a=r, ab=b, qr=q
                    d2Odp2[:, :, base, b, q, base] = 1j * superops[b, q] - 1j * superops[q, b]
                for base, b, r in iter_base_ab_qr(False, False):  # Case4: base=a=q, ab=b, qr=r
                    d2Odp2[:, :, base, b, base, r] = superops[b, r] + superops[r, b]

            elif self._param_mode == 'elements':  # unconstrained
                d2Odp2 = _np.zeros([superops.shape[2], superops.shape[3], nP, nP, nP, nP], 'd')  # all params linear
            else:
                raise ValueError("Internal error: invalid parameter mode (%s) for block type %s!"
                                 % (self._param_mode, self._block_type))
        else:
            raise ValueError("Internal error: invalid block type!")

        # apply basis transform
        tr = len(d2Odp2.shape)  # tensor rank
        assert((tr - 2) in (2, 4)), "Currently, d2Odp2 can only have 2 or 4 derivative dimensions"

        assert(_np.linalg.norm(_np.imag(d2Odp2)) < IMAG_TOL)
        return _np.real(d2Odp2)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'block_type': self._block_type,
                      'parameterization_mode': self._param_mode,
                      'basis_element_labels': list(self._bel_labels),
                      'basis': self._basis.to_nice_serialization(),
                      'block_data': self._encodemx(self.block_data)})
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        block_data = cls._decodemx(state['block_data'])
        basis = _Basis.from_nice_serialization(state['basis'])
        return cls(state['block_type'], basis, state['basis_element_labels'], block_data,
                   state['parameterization_mode'])

    def is_similar(self, other_coeff_block):
        """ TODO: docstring """
        if not isinstance(other_coeff_block, LindbladCoefficientBlock): return False
        return ((self._block_type == other_coeff_block._block_type)
                and (self._param_mode == other_coeff_block._param_mode)
                and (self._bel_labels == other_coeff_block._bel_labels)
                and (self._basis == other_coeff_block._basis))

    def convert(self, param_mode):
        """
        TODO: docstring  - return a *new* LindbladCoefficientBlock with the same block type and data,
        but with the given parameterization mode.
        """
        return LindbladCoefficientBlock(self._block_type, self._basis, self._bel_labels, self.block_data, param_mode)

    def __str__(self):
        s = '%s-type lindblad coefficient block with param-mode %s and basis labels %s.' % (
            self._block_type, self._param_mode, self._bel_labels)
        if len(self._bel_labels) < 10:
            s += " Coefficients are:\n" + str(_np.round(self.block_data, 4))
        return s

@lru_cache(maxsize=16)
def cached_diag_indices(n):
    return _np.diag_indices(n)