import numpy as _np
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import collections as _collections
import copy as _copy
import warnings as _warnings
from pygsti.tools import lindbladtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.modelmembers import term as _term
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from functools import lru_cache
try:
    from pygsti.tools import fastcalc as _fc
    triu_indices = _fc.fast_triu_indices
except ImportError:
    msg = 'Could not import cython module `fastcalc`. This may indicate that your cython extensions for pyGSTi failed to.'\
          +'properly build. Lack of cython extensions can result in significant performance degredation so we recommend trying to rebuild them.'\
           'Falling back to numpy implementation for triu_indices.'
    _warnings.warn(msg)
    triu_indices = _np.triu_indices

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero
from typing import Union, Literal
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LEEL


def _custom_superops_stdbasis_conversion(mx_basis, sparse, superops):
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
    return superops


def _unflatten_cached_other_lindblad_term_superoperators(cached_superops: Union[list, _np.ndarray], cached_superops_1norms: _np.ndarray, include_1norms: bool):
        # flat == False and block type is 'other', so need to reshape:
        nMxs = round(cached_superops_1norms.size ** 0.5)
        if isinstance(cached_superops, list):
            superops = [ [cached_superops[i * nMxs + j] for j in range(nMxs)] for i in range(nMxs) ]
        else:
            d2 = round(_np.sqrt(cached_superops.size / nMxs**2))
            superops = cached_superops.copy().reshape((nMxs, nMxs, d2, d2))
            
        if include_1norms:
            return superops, cached_superops_1norms.copy().reshape((nMxs, nMxs))
        else:
            return superops


class InvalidBlockTypeError(ValueError):

    msg = "Internal error: invalid block type!"

    def __init__(self) -> None:
        super().__init__(msg)


class InvalidParamModeError(ValueError):

    msg_template = "Internal error: invalid parameter mode (%s) for block type %s!"

    def __init__(self, *args: object) -> None:
        msg = InvalidParamModeError.msg_template % (args[0], args[1])
        super().__init__(msg)


class LindbladCoefficientBlock(_NicelySerializable):
    """ 
    Class for storing and managing the parameters associated with particular subblocks of error-generator
    parameters. Responsible for management of different internal representations utilized when employing
    various error generator constraints.
    """

    _superops_cache = {}  # a custom cache for create_lindblad_term_superoperators method calls

    def __init__(self, block_type, basis, basis_element_labels=None, initial_block_data=None, param_mode='static',
                 truncate=False):
        """
        Parameters
        ----------
        block_type : str
            String specifying the type of error generator parameters contained within this block. Allowed
            values are 'ham' (for Hamiltonian error generators), 'other_diagonal' (for Pauli stochastic error generators),
            and 'other' (for Pauli stochastic, Pauli correlation and active error generators).
        
        basis : `Basis`
            `Basis` object to be used by this coefficient block. Not this must be an actual `Basis` object, and not
            a string (as the coefficient block doesn't have the requisite dimensionality information needed for casting).
        
        basis_element_labels : list or tuple of str
            Iterable of strings corresponding to the basis element subscripts used by the error generators managed by
            this coefficient block.
        
        initial_block_data : _np.ndarray, optional (default None)
            Numpy array with initial parameter values to use in setting initial state of this coefficient block.
        
        param_mode : str, optional (default 'static')
            String specifying the type of internal parameterization used by this coefficient block. Allowed options are:
            
            - For all block types: 'static'
            - For 'ham': 'elements'
            - For 'other_diagonal': 'elements', 'cholesky', 'depol', 'reldepol'
            - For 'other': 'elements', 'cholesky'

            Note that the most commonly encounted settings in practice are 'elements' and 'cholesky',
            which when used in the right combination are utilized in the construction of GLND and CPTPLND
            parameterized models. For both GLND and CPTPLND the 'ham' block used the 'elements' `param_mode`.
            GLND the 'other' block uses 'elements', and for CPTPLND it uses 'cholesky'. 
            
            'depol' and 'reldepol' are special modes used only for Pauli stochastic only coefficient blocks
            (i.e. 'other_diagonal'), and correspond to special reduced parameterizations applicable to depolarizing
            channels. (TODO: Add better explanation of the difference between depol and reldepol).
        
        truncate : bool, optional (default False)
            Flag specifying whether to truncate the parameters given by `initial_block_data` in order to meet
            constraints (e.g. to preserve CPTP) when necessary. If False, then an error is thrown when the 
            given intial data cannot be parameterized as specified. 
        """

        super().__init__()
        self._block_type = block_type  # 'ham' or 'other' or 'other_diagonal'
        self._param_mode = param_mode  # 'static', 'elements', 'cholesky', 'depol', 'reldepol'
        self._basis = basis  # must be a full Basis object, not just a string, as we otherwise don't know dimension
        self._bel_labels = tuple(basis_element_labels) if (basis_element_labels is not None) \
            else tuple(basis.labels[1:])  # Note: don't include identity
        self._cache_mx = _np.zeros((len(self._bel_labels), len(self._bel_labels)), 'complex') \
            if self._block_type == 'other' else None

        #this would get set to True in the very next method call anyway
        self._coefficients_need_update = True
        self._cached_elementary_errorgens = None
        self._cached_elementary_errorgen_indices = None

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

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

    @property
    def basis_element_labels(self):
        return self._bel_labels

    @property
    def num_params(self) -> int:
        if self._block_type == 'ham':
            return self._num_params_ham()
        elif self._block_type == 'other_diagonal':
            return self._num_params_otherdiag()
        elif self._block_type == 'other':
            return self._num_params_other()
        else:
            raise InvalidBlockTypeError()

    def _num_params_ham(self):
        if self._param_mode == 'static':
            return 0
        elif self._param_mode ==  'elements':
            return len(self._bel_labels)
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)
    
    def _num_params_otherdiag(self):
        if self._param_mode == 'static':
            return 0
        elif self._param_mode in ('depol', 'reldepol'):
            return 1
        elif self._param_mode in ('elements', 'cholesky'):
            return len(self._bel_labels)
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _num_params_other(self):
        if self._param_mode == 'static':
            return 0
        elif self._param_mode in ('elements', 'cholesky'):
            return len(self._bel_labels)**2
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _update_superops_cache(self, mxs, mx_basis, cache_key):
        d2 = mx_basis.dim
        sparse = mx_basis.sparse
        nMxs = len(mxs)
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

        # Convert matrices to desired `mx_basis` basis
        if mx_basis != "std":
            superops = _custom_superops_stdbasis_conversion(mx_basis, sparse, superops)
        
        superop_1norms = _np.array([_mt.safe_onenorm(mx) for mx in superops], 'd')

        self._superops_cache[cache_key] = (superops, superop_1norms)
        return 

    def create_lindblad_term_superoperators(self, mx_basis='pp', sparse: Union[Literal['auto'], bool]="auto", include_1norms=False, flat=False):
        assert(self._basis is not None), "Cannot create lindblad superoperators without a basis!"
        basis = self._basis
        sparse = basis.sparse if (sparse == "auto") else sparse
        mxs = [basis[lbl] for lbl in self._bel_labels]
        if len(mxs) == 0:
            return ([], []) if include_1norms else []  # short circuit - no superops to return

        mx_basis = _Basis.cast(mx_basis, mxs[0].size, sparse=sparse)

        cache_key = (self._block_type, tuple(self._bel_labels), mx_basis, basis)
        if cache_key not in self._superops_cache:
            self._update_superops_cache(mxs, mx_basis, cache_key)
        cached_superops, cached_superops_1norms = self._superops_cache[cache_key]

        if flat or self._block_type in ('ham', 'other_diagonal'):
            if include_1norms:
                return _copy.deepcopy(cached_superops), cached_superops_1norms.copy()
            else:
                return _copy.deepcopy(cached_superops)

        out = _unflatten_cached_other_lindblad_term_superoperators(cached_superops, cached_superops_1norms, include_1norms)
        return out

    def create_lindblad_term_objects(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        # needed b/c operators produced by lindblad_error_generators have an extra 'd' scaling
        mpv = max_polynomial_vars
        pio = parameter_index_offset

        if self._block_type == 'ham':
            return self._create_lindblad_term_objects_ham(pio, mpv, evotype, state_space)

        if self._block_type == 'other_diagonal':
            return self._create_lindblad_term_objects_otherdiag(pio, mpv, evotype, state_space)
        
        if self._block_type == 'other':
            return self._create_lindblad_term_objects_other(pio, mpv, evotype, state_space)
        
        raise ValueError("Invalid block_type '%s'" % str(self._block_type))

    def _create_lindblad_term_objects_ham(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        assert self._block_type == 'ham'
        mpv = max_polynomial_vars
        pio = parameter_index_offset
        Lterms = []
        for k, bel_label in enumerate(self._bel_labels):  # k == index of local parameter that is coefficient
            # ensure all Rank1Term operators are *unitary*, so we don't need to track their "magnitude"
            scale, U = _mt.to_unitary(self._basis[bel_label])

            if self._param_mode == 'elements':
                cpi = (pio + k,)  # coefficient's parameter indices (with offset)
            elif self._param_mode == 'static':
                cpi = ()  # not multiplied by any parameters
                scale *= self.block_data[k]  # but scale factor gets multiplied by (static) coefficient
            else: raise ValueError("Internal error: invalid param mode!!")

            #Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
            Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                _Polynomial({cpi: -1j * scale}, mpv), U, None, evotype, state_space))
            Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                _Polynomial({cpi: +1j * scale}, mpv), None, U.conjugate().T, evotype, state_space))
        return Lterms
    
    def _create_lindblad_term_objects_otherdiag(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        assert self._block_type == 'other_diagonal'
        mpv = max_polynomial_vars
        pio = parameter_index_offset
        Lterms = []
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
            Ln_dag = Ln.conjugate().T  # for sparse case too - and _np.dots below!)

            #Note: 2nd op to create_from must be the *adjoint* of the op you'd normally write down
            # e.g. in 2nd term, _np.dot(Ln_dag, Lm) == adjoint(_np.dot(Lm_dag,Ln))
            Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                _Polynomial({cpi * pw: 1.0 * scale}, mpv), Ln, Lm, evotype, state_space))
            Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                _Polynomial({cpi * pw: -0.5 * scale}, mpv), None, _np.dot(Ln_dag, Lm), evotype, state_space))
            Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                _Polynomial({cpi * pw: -0.5 * scale}, mpv), _np.dot(Lm_dag, Ln), None, evotype, state_space))
        return Lterms
    
    def _create_lindblad_term_objects_other(self, parameter_index_offset, max_polynomial_vars, evotype, state_space):
        assert self._block_type == 'other'
        num_bels = len(self._bel_labels)
        mpv = max_polynomial_vars
        pio = parameter_index_offset
        Lterms = []
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
                    -0.5 * base_poly, None, _np.dot(Ln_dag, Lm), evotype, state_space))  # adjoint(dot(Lm_dag,Ln))
                Lterms.append(_term.RankOnePolynomialOpTerm.create_from(
                    -0.5 * base_poly, _np.dot(Lm_dag, Ln), None, evotype, state_space))
        return Lterms

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
        if not self._coefficients_need_update and self._cached_elementary_errorgen_indices is not None:
            return self._cached_elementary_errorgen_indices

        if   self._block_type == 'ham':
            idx = self._elementary_errorgen_indices_ham()
        elif self._block_type == 'other_diagonal':
            idx = self._elementary_errorgen_indices_otherdiag()
        elif self._block_type == 'other':
            idx = self._elementary_errorgen_indices_other()
        else:
            raise InvalidBlockTypeError()

        self._cached_elementary_errorgen_indices = idx
        return idx

    def _elementary_errorgen_indices_ham(self):
        elem_errgen_indices = _collections.OrderedDict()
        for i, lbl in enumerate(self._bel_labels):
            elem_errgen_indices[_LEEL('H', (lbl,))] = [(1.0, i)]
        return elem_errgen_indices

    def _elementary_errorgen_indices_otherdiag(self):
        elem_errgen_indices = _collections.OrderedDict()
        for i, lbl in enumerate(self._bel_labels):
            elem_errgen_indices[_LEEL('S', (lbl,))] = [(1.0, i)]
        return elem_errgen_indices

    def _elementary_errorgen_indices_other(self):
        elem_errgen_indices = _collections.OrderedDict()
        stride = len(self._bel_labels)
        for i, lbl1 in enumerate(self._bel_labels):
            ii = i * stride + i
            # the diagonal (stochastic) term
            elem_errgen_indices[_LEEL('S', (lbl1,))] = [(1.0, ii)]
            # the off-diagonal C and A combinations
            for j, lbl2 in enumerate(self._bel_labels[i+1:], start=i+1):
                ij = i * stride + j
                ji = j * stride + i
                # C_{P,Q} = ( NH_{P,Q} + NH_{Q,P} )
                elem_errgen_indices[_LEEL('C', (lbl1, lbl2))] = [(0.5, ij), (0.5, ji)]
                # A_{P,Q} = i ( NH_{Q,P} – NH_{P,Q} )
                elem_errgen_indices[_LEEL('A', (lbl1, lbl2))] = [(0.5j, ij), (-0.5j, ji)]
        return elem_errgen_indices

    @property
    def _block_data_indices(self):
        """
        Effectively the inverse of elementary_errorgen_indices.

        The keys of the returned dict are (flattened) block_data indices and the
        values specify a linear combination of elementary errorgens via their labels.
        """
        if self._block_type == 'ham':
            return self._block_data_indices_ham()
        elif self._block_type == 'other_diagonal':
            return self._block_data_indices_otherdiag()
        elif self._block_type == 'other':
            return self._block_data_indices_other()
        else:
            raise InvalidBlockTypeError()
    
    def _block_data_indices_ham(self):
        block_data_indices = dict()
        for i, lbl in enumerate(self._bel_labels):
            block_data_indices[i] = [(1.0, _LEEL('H', (lbl,)))]
        return block_data_indices
    
    def _block_data_indices_otherdiag(self):
        block_data_indices = dict()
        for i, lbl in enumerate(self._bel_labels):
            block_data_indices[i] = [(1.0, _LEEL('S', (lbl,)))]
            return block_data_indices
        
    def _block_data_indices_other(self):
        # Difficult case, as coefficients do not correspond to elementary errorgens, so
        # there's no single index for, e.g. ('C', lbl1, lbl2) - rather this elementary
        # errorgen is a linear combination of two coefficients.
        block_data_indices = dict()
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
        return block_data_indices

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
        if not self._coefficients_need_update and self._cached_elementary_errorgens is not None:
            return self._cached_elementary_errorgens

        elementary_errorgens = dict()
        eeg_indices = self.elementary_errorgen_indices
        flat_data = self.block_data.ravel()

        for eeg_lbl, linear_combo in eeg_indices.items():
            val = _np.sum([coeff * flat_data[index] for coeff, index in linear_combo])
            elementary_errorgens[eeg_lbl] = _np.real_if_close(val).item()  # item() -> scalar
            #set_basis_el(lbl, basis[lbl])  # REMOVE
        #cache the error generator dictionary for future use
        self._cached_elementary_errorgens = elementary_errorgens
        self._coefficients_need_update = False

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

        self.block_data[:] = flat_data.reshape(self.block_data.shape)
        self._truncate_block_data(truncate)

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

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

        #set a flag to indicate that the coefficients (as returned by elementary_errorgens)
        #need to be updated.
        self._coefficients_need_update = True

        return out[1] if return_projected_errorgen else None

    @property
    def coefficient_labels(self):
        """Labels for the elements of `self.block_data` (flattened if relevant)"""
        if self._block_type == 'ham':
            return self._coefficient_labels_ham()
        elif self._block_type == 'other_diagonal':
            return self._coefficient_labels_otherdiag()
        elif self._block_type == 'other':
            return self._coefficient_labels_other()
        else:
            raise InvalidBlockTypeError()
    
    def _coefficient_labels_ham(self):
        return ["%s Hamiltonian error coefficient" % lbl for lbl in self._bel_labels]
    
    def _coefficient_labels_otherdiag(self):
        return ["(%s,%s) other error coefficient" % (lbl, lbl) for lbl in self._bel_labels]
    
    def _coefficient_labels_other(self):
        labels = []
        for i, ilbl in enumerate(self._bel_labels):
            for j, jlbl in enumerate(self._bel_labels):
                labels.append("(%s,%s) other error coefficient" % (ilbl, jlbl))
        return labels

    @property
    def param_labels(self) -> list:
        """
        Generate human-readable labels for the parameters of this block.
        """
        if   self._block_type == 'ham':
            return self._param_labels_ham()
        elif self._block_type == 'other_diagonal':
            return self._param_labels_otherdiag()
        elif self._block_type == 'other':
            return self._param_labels_other()
        else:
            raise InvalidBlockTypeError()

    def _param_labels_ham(self) -> list:
        if   self._param_mode == 'static':
            return []
        elif self._param_mode == 'elements':
            return [f"{lbl} Hamiltonian error coefficient" for lbl in self._bel_labels]
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _param_labels_otherdiag(self) -> list:
        if   self._param_mode == 'static':
            return []
        elif self._param_mode == "depol":
            return ["sqrt(common stochastic error coefficient for depolarization)"]
        elif self._param_mode == 'reldepol':
            return ["common stochastic error coefficient for depolarization"]
        elif self._param_mode == "cholesky":
            return [f"sqrt({lbl} stochastic coefficient)" for lbl in self._bel_labels]
        elif self._param_mode == "elements":
            return [f"{lbl} stochastic coefficient" for lbl in self._bel_labels]
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _param_labels_other(self) -> list:
        labels = []
        if   self._param_mode == 'static':
            return labels
        elif self._param_mode == "cholesky":
            # Cholesky‐decomp parameters: real diag, real lower, imag lower
            for i, ilbl in enumerate(self._bel_labels):
                for j, jlbl in enumerate(self._bel_labels):
                    if  i == j:
                        label = f"{ilbl} diagonal element of non-H coeff Cholesky decomp"
                    elif j < i:
                        label = f"Re[({ilbl},{jlbl}) element of non-H coeff Cholesky decomp]"
                    else:
                        label = f"Im[({ilbl},{jlbl}) element of non-H coeff Cholesky decomp]"
                    labels.append(label)
        elif self._param_mode == "elements":
            # Unconstrained Hermitian‐matrix parameters
            for i, ilbl in enumerate(self._bel_labels):
                for j, jlbl in enumerate(self._bel_labels):
                    if  i == j:
                        label = f"{ilbl} diagonal element of non-H coeff matrix"
                    elif j < i:
                        label = f"Re[({ilbl},{jlbl}) element of non-H coeff matrix]"
                    else:
                        label = f"Im[({ilbl},{jlbl}) element of non-H coeff matrix]"
                    labels.append(label)
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)
        return labels

    def _block_data_to_params(self, truncate: Union[bool, float]=False):
        """
        Compute parameter values implied by self.block_data.
        (The public method now simply dispatches on block type.)
        """
        if truncate is False:
            ttol = -1e-14
        elif truncate is True:
            ttol = -_np.inf
        else:
            ttol = -truncate

        if self._param_mode == 'static':
            return _np.empty(0, 'd')

        if   self._block_type == 'ham':
            params = self._block_data_to_params_ham()
        elif self._block_type == 'other_diagonal':
            params = self._block_data_to_params_otherdiag(ttol)
        elif self._block_type == 'other':
            params = self._block_data_to_params_other(ttol)
        else:
            raise InvalidBlockTypeError()

        assert(not _np.iscomplexobj(params))
        assert(len(params) == self.num_params)
        return params

    def _block_data_to_params_ham(self) -> _np.ndarray:
        if self._param_mode == "elements":
            return self.block_data.real # type: ignore
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _block_data_to_params_otherdiag(self, ttol) -> _np.ndarray:
        # depol / reldepol share a one-parameter constraint
        if self._param_mode in ("depol", "reldepol"):
            if self._param_mode == "depol":
                nonneg = [v >= ttol for v in self.block_data]
                assert all(nonneg), "Lindblad stochastic coefficients are not positive!"
            # check all diagonal entries are equal
            first = self.block_data[0]
            all_equal = [_np.isclose(v, first, atol=1e-6) for v in self.block_data]
            assert all(all_equal), "Diagonal lindblad coefficients are not equal!"
            # build the single parameter
            if self._param_mode == "depol":
                avg = _np.mean(self.block_data.clip(0, 1e100))
                return _np.array([_np.sqrt(_np.real(avg))], 'd')
            else:  # 'reldepol'
                avg = _np.mean(self.block_data)
                return _np.array([_np.real(avg)], 'd')

        # cholesky: each block_data[i] = param[i]**2
        elif self._param_mode == "cholesky":
            nonneg = [v >= ttol for v in self.block_data]
            assert all(nonneg), "Lindblad stochastic coefficients are not positive!"
            clipped = self.block_data.clip(0, 1e100)
            return _np.sqrt(clipped.real)

        # elements: unconstrained real diag
        elif self._param_mode == "elements":
            return self.block_data.real

        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _block_data_to_params_other(self, ttol):
        """Handle the 'other' block (full Hermitian matrix)."""
        numpy_isclose_abstol_default = 1e-8
        is_hermitian =  _mt.is_hermitian(self.block_data, numpy_isclose_abstol_default)
        assert is_hermitian, "Lindblad 'other' coefficient matrix is not Hermitian!"

        num_bels = len(self._bel_labels)
        params_mat = _np.empty((num_bels, num_bels), 'd')

        if self._param_mode == "cholesky":
            # build the Cholesky‐style parameters out of block_data = C C†
            col_norms   = _np.linalg.norm(self.block_data, axis=0)
            row_norms   = _np.linalg.norm(self.block_data, axis=1)
            ind_weights = col_norms + row_norms
            zero_inds   = set(_np.where(ind_weights < 1e-12 * num_bels)[0])
            num_nonzero = num_bels - len(zero_inds)

            # permute zero rows/cols to end
            perm = _np.zeros((num_bels, num_bels), 'd')
            ni = nz = 0
            for i in range(num_bels):
                if i in zero_inds:
                    perm[num_nonzero + nz, i] = 1.0; nz += 1
                else:
                    perm[ni, i] = 1.0; ni += 1

            # extract nonzero block, compute eigen‐decomp & cholesky
            pdata = perm @ self.block_data @ perm.T
            small = pdata[0:num_nonzero, 0:num_nonzero]
            evals, U = _np.linalg.eigh(small)
            assert(all([ev > ttol for ev in evals])), "Lindblad coefficients are not CPTP!"
            if ttol < 0:
                # truncate small negatives up
                Ui = U.T.conj()
                pev = evals.clip(1e-16, None)
                small = U @ (pev[:,None] * Ui)
                try:
                    Lsmall = _np.linalg.cholesky(small)
                except _np.linalg.LinAlgError:
                    pev = evals.clip(1e-12, 1e100)
                    small = U @ (pev[:,None] * Ui)
                    Lsmall = _np.linalg.cholesky(small)
            else:
                Lsmall = _np.linalg.cholesky(small)

            # re‐embed into full L
            Lfull = _np.zeros((num_bels, num_bels), 'complex')
            Lfull[:num_nonzero, :num_nonzero] = Lsmall
            Lmx = perm.T @ Lfull @ perm

            # now extract the real diag & real/imag lower‐triangle
            for i in range(num_bels):
                assert(_np.abs(Lmx[i,i].imag) < IMAG_TOL)
                params_mat[i,i] = Lmx[i,i].real
                for j in range(i):
                    params_mat[i,j] = Lmx[i,j].real
                    params_mat[j,i] = Lmx[i,j].imag

        elif self._param_mode == "elements":
            # store directly the Hermitian matrix's real diag + real/imag lower‐triangle
            for i in range(num_bels):
                assert(_np.abs(self.block_data[i,i].imag) < IMAG_TOL)
                params_mat[i,i] = self.block_data[i,i].real
                for j in range(i):
                    params_mat[i,j] = self.block_data[i,j].real
                    params_mat[j,i] = self.block_data[i,j].imag

        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

        # flatten to 1D vector
        return params_mat.ravel()

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

    def from_vector(self, v: _np.ndarray):
        """
        Construct Lindblad coefficients (for this block) from a set of parameter values.
        This is now just a dispatcher into the three block-type specific implementations.
        """
        # static blocks carry no parameters
        if self._param_mode == 'static':
            assert len(v) == 0, "'static' parameterized blocks should have zero parameters!"
            return

        # dispatch on block type
        if   self._block_type == 'ham':
            self._from_vector_ham(v)
        elif self._block_type == 'other_diagonal':
            self._from_vector_otherdiag(v)
        elif self._block_type == 'other':
            self._from_vector_other(v)
        else:
            raise InvalidBlockTypeError()

        # mark cache stale
        self._coefficients_need_update = True
        return

    def _from_vector_ham(self, v: _np.ndarray):
        """Inverse of _block_data_to_params_ham: load block_data from v for 'ham'."""
        if self._param_mode == 'elements':
            self.block_data[:] = v
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _from_vector_otherdiag(self, v: _np.ndarray):
        """Inverse of _block_data_to_params_otherdiag: load block_data from v for 'other_diagonal'."""
        num_bels = len(self._bel_labels)

        if self._param_mode in ('depol', 'reldepol'):
            v = v.item() * _np.ones(num_bels, 'd')
        else:
            assert v.shape == (num_bels,)

        if self._param_mode in ('cholesky', 'depol'):
            self.block_data[:] = v**2
        elif self._param_mode in ('reldepol', 'elements'):
            self.block_data[:] = v
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _from_vector_other(self, v: _np.ndarray):
        """Inverse of _block_data_to_params_other: load block_data from v for full 'other' block."""
        num_bels = len(self._bel_labels)
        params = v.reshape((num_bels, num_bels))

        if self._param_mode == 'cholesky':
            # rebuild the Hermitian block_data = L · L† from the Cholesky‐style params
            cache_mx : _np.ndarray = self._cache_mx # type: ignore
            # fill cache_mx from params:
            #   diag entries:
            diag_idxs = cached_diag_indices(num_bels)
            cache_mx[diag_idxs] = params[diag_idxs]
            # lower‐triangle real, upper‐triangle imaginary
            upper = triu_indices(num_bels)
            cache_mx_T = cache_mx.T
            cache_mx_T[upper] = params.T[upper] + 1j * params[upper]
            cache_mx[:] = cache_mx_T.T
            # now form block_data = L L†
            self.block_data[:, :] = cache_mx @ cache_mx.conjugate().T

        elif self._param_mode == 'elements':
            # direct Hermitian storage: real diag + real/imag lower‐triangle
            upper = triu_indices(num_bels)
            # real(diag) and real(lower)
            self.block_data[upper] = params[upper].real
            # imag(upper)
            self.block_data.T[upper] = -1j * params[upper]
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def deriv_wrt_params(self, v=None):
        """
        Jacobian of block_data w.r.t. the real parameters v.
        Dispatches to _deriv_wrt_params_<block_type>.
        """
        num_bels = len(self._bel_labels)
        v = self.to_vector() if (v is None) else v
        nP = len(v)
        assert(nP == self.num_params), f"Expected {self.num_params} parameters, got {nP}!"

        # static blocks have zero‐dimensional parameter space
        if self._param_mode == 'static':
            if self._block_type in ('ham', 'other_diagonal'):
                return _np.zeros((num_bels, 0), 'd')
            elif self._block_type == 'other':
                return _np.zeros((num_bels, num_bels, 0), 'd')
            else:
                raise InvalidBlockTypeError()

        # dispatch by block type
        if   self._block_type == 'ham':
            return self._deriv_wrt_params_ham()
        elif self._block_type == 'other_diagonal':
            return self._deriv_wrt_params_otherdiag(v)
        elif self._block_type == 'other':
            return self._deriv_wrt_params_other(v)
        else:
            raise InvalidBlockTypeError()

    def _deriv_wrt_params_ham(self) -> _np.ndarray:
        """deriv_wrt_params for the 'ham' block."""
        num_bels = len(self._bel_labels)
        if self._param_mode == 'elements':
            # d(block_data[i])/d v[j] = δ_{i,j}
            return _np.identity(num_bels, 'd')
        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

    def _deriv_wrt_params_otherdiag(self, v: _np.ndarray) -> _np.ndarray:
        """deriv_wrt_params for the 'other_diagonal' block."""
        num_bels = len(self._bel_labels)
        nP = len(v)
        mat = _np.zeros((num_bels, nP), 'd')

        if self._param_mode in ('depol', 'reldepol'):
            assert nP == 1, f"Expected 1 parameter, got {nP}!"
            if self._param_mode == 'depol':
                mat[:, 0] = 2.0 * v[0]
            else:  # 'reldepol'
                mat[:, 0] = 1.0

        elif self._param_mode == 'cholesky':
            assert nP == num_bels, f"Expected {num_bels} parameters, got {nP}!"
            # block_data[i] = v[i]**2  =>  d(block_data[i])/d v[j] = 2 v[i] δ_{i,j}
            mat[:, :] = 2.0 * _np.diag(v)

        elif self._param_mode == 'elements':
            assert nP == num_bels, f"Expected {num_bels} parameters, got {nP}!"
            # block_data[i] = v[i]  =>  derivative is identity
            mat[:, :] = _np.identity(num_bels, 'd')

        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

        return mat

    def _deriv_wrt_params_other(self, v: _np.ndarray) -> _np.ndarray:
        """deriv_wrt_params for the 'other' block (full Hermitian matrix)."""
        num_bels = len(self._bel_labels)
        nP = len(v)
        params = v.reshape((num_bels, num_bels))

        if self._param_mode == 'cholesky':
            # We have block_data = C C†, where C = cache_mx depends on params.
            # Build C and dC/dp, then d(block_data)/dp = dC·C† + C·(dC)†
            cache_mx : _np.ndarray = self._cache_mx  # type: ignore
            dC = _np.zeros((nP, num_bels, num_bels), 'complex')
            stride = num_bels

            # fill C and dC
            for i in range(num_bels):
                cache_mx[i, i] = params[i, i]
                dC[i*stride + i, i, i] = 1.0
                for j in range(i):
                    cache_mx[i, j] = params[i, j] + 1j * params[j, i]
                    dC[i*stride + j, i, j] = 1.0
                    dC[j*stride + i, i, j] = 1.0j

            # form the Jacobian: shape = (num_bels, num_bels, nP)
            term1 = _np.tensordot(dC, cache_mx.T.conj(), (1, 0))  # (nP,i,j)
            term2 = _np.tensordot(cache_mx, dC.conjugate().transpose((0,2,1)), (1,0))  # (i,j,nP)
            # combine and roll axis 0 → last
            dB = term1 + term2.transpose((2,0,1))
            return _np.rollaxis(dB, 0, 3)

        elif self._param_mode == 'elements':
            # direct Hermitian: block_data[i,j] = v[i,j]_real + i v[j,i]_real
            # so we only get linear contributions
            mat = _np.zeros((num_bels, num_bels, nP), 'complex')
            stride = num_bels
            for i in range(num_bels):
                # diag
                mat[i, i, i*stride + i] = 1.0
                # off‐diag
                for j in range(i):
                    mat[i, j, i*stride + j] = 1.0
                    mat[i, j, j*stride + i] = 1.0j
                    mat[j, i, i*stride + j] = 1.0
                    mat[j, i, j*stride + i] = -1.0j
            return mat

        else:
            raise InvalidParamModeError(self._param_mode, self._block_type)

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
            elif self._param_mode == "cholesky":
                assert(nP == num_bels)
                #d2Odp2  = _np.einsum('alj,aq->ljaq', self.otherGens, 2*_np.identity(nP,'d'))
                d2Odp2 = _np.transpose(superops, (1, 2, 0))[:, :, :, None] * 2 * _np.identity(nP, 'd')
            else:  # param_mode == "elements" or "reldepol"
                assert(nP == num_bels)
            d2Odp2 = _np.zeros((superops.shape[1], superops.shape[2], nP, nP), 'd')

        elif self._block_type == 'other':
            if self._param_mode == "cholesky":
                if superops_are_flat:  # then un-flatten
                    superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
                sqrt_nP = _np.sqrt(nP)
                snP = int(sqrt_nP)
                assert snP == sqrt_nP == num_bels
                d2Odp2 = _np.zeros([superops.shape[2], superops.shape[3], snP, snP, snP, snP], 'complex')
                # yikes! maybe make this SPARSE in future?

                #Note: correspondence w/Erik's notes: a=alpha, b=beta, q=gamma, r=delta
                # indices of d2Odp2 are [i,j,a,b,q,r]

                def iter_base_ab_qr(ab_inc_eq, qr_inc_eq):
                    """ Generates (base,ab,qr) tuples such that `base` runs over
                        all possible 'other' params and 'ab' and 'qr' run over
                        parameter indices s.t. ab > base and qr > base.  If
                        ab_inc_eq == True then the > becomes a >=, and likewise
                        for qr_inc_eq.  Used for looping over nonzero hessian els. """
                    for _base in range(snP):
                        start_ab = _base if ab_inc_eq else _base + 1
                        start_qr = _base if qr_inc_eq else _base + 1
                        for _ab in range(start_ab, snP):
                            for _qr in range(start_qr, snP):
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
                if superops_are_flat:  # then un-flatten
                    superops = superops.reshape((num_bels, num_bels, superops.shape[1], superops.shape[2]))
                sqrt_nP = _np.sqrt(nP)
                snP = int(sqrt_nP)
                assert snP == sqrt_nP == num_bels
                d2Odp2 = _np.zeros([superops.shape[2], superops.shape[3], snP, snP, snP, snP], 'd')  # all params linear
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
