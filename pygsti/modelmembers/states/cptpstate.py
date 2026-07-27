"""
The CPTPState class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.modelmembers.states.densestate import DenseState as _DenseState
from pygsti.modelmembers.states.state import State as _State
from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.tools import matrixtools as _mt


IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


class CPTPState(_DenseState):
    """
    TODO: update docstring
    A state vector constrained to correspond ot a positive density matrix.

    This state vector that is parameterized through the Cholesky decomposition of
    it's standard-basis representation as a density matrix (not a Liouville
    vector).  The resulting state vector thus represents a positive density
    matrix, and additional constraints on the parameters also guarantee that the
    trace == 1.  This state vector is meant for use with CPTP processes, hence
    the name.

    Parameters
    ----------
    vec : array_like or State
        a 1D numpy array representing the state operation.  The
        shape of this array sets the dimension of the state.

    basis : {"std", "gm", "pp", "qt"} or Basis
        The basis `vec` is in.  Needed because this parameterization
        requires we construct the density matrix corresponding to
        the Lioville vector `vec`.

    trunctate : bool, optional
        Whether or not a non-positive, trace=1 `vec` should
        be truncated to force a successful construction.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, vec, basis, truncate=False, evotype="default", state_space=None):
        vector = _State._to_vector(vec)
        basis = _Basis.cast(basis, len(vector))

        self.basis = basis
        self.basis_mxs = basis.elements  # shape (len(vec), dmDim, dmDim)
        self.basis_mxs = _np.rollaxis(self.basis_mxs, 0, 3)  # shape (dmDim, dmDim, len(vec))
        assert(self.basis_mxs.shape[-1] == len(vector))

        # set self.params and self.dmDim
        self._set_params_from_vector(vector, truncate)

        #parameter labels (parameter encode the Cholesky Lmx)
        labels = {}; dmDim = self.dmDim
        for i in range(dmDim):
            labels[i * dmDim + i] = "(%d, %d) element of density matrix Cholesky deomp" % (i, i)
            for j in range(i):
                labels[i * dmDim + j] = "Re[(%d, %d) element of density matrix Cholesky deomp]" % (i, j)
                labels[j * dmDim + i] = "Im[(%d, %d) element of density matrix Cholesky deomp]" % (i, j)
        labels = [lbl for indx, lbl in sorted(list(labels.items()), key=lambda x: x[0])]

        #scratch space
        self.Lmx = _np.zeros((self.dmDim, self.dmDim), 'complex')

        state_space = _statespace.default_space_for_dim(len(vector)) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)

        evotype = _Evotype.cast(evotype, state_space=state_space)
        _DenseState.__init__(self, vector, basis, evotype, state_space)
        self._paramlbls = _np.array(labels, dtype=object)

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
        mm_dict = super().to_memoized_dict(mmg_memo)  # contains 'dense_state_vector' via DenseState base class
        mm_dict['basis'] = self.basis.to_nice_serialization()

        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        vec = _np.array(mm_dict['dense_state_vector'])
        state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        basis = _Basis.from_nice_serialization(mm_dict['basis'])
        truncate = False  # shouldn't need to since we're loading a valid object
        return cls(vec, basis, truncate, mm_dict['evotype'], state_space)

    def _set_params_from_vector(self, vector, truncate):
        density_mx = _np.dot(self.basis_mxs, vector)
        density_mx = density_mx.squeeze()
        dmDim = density_mx.shape[0]
        assert(dmDim == density_mx.shape[1]), "Density matrix must be square!"

        trc = _np.trace(density_mx)
        assert(truncate or _np.isclose(trc, 1.0)), \
            "`vec` must correspond to a trace-1 density matrix (truncate == False)!"

        if not _np.isclose(trc, 1.0):  # truncate to trace == 1
            density_mx -= _np.identity(dmDim, 'd') / dmDim * (trc - 1.0)

        # push any slightly negative evals of density_mx positive
        # so that the Cholesky decomp will work.
        U, evals, Ui = _mt.eigendecomposition(density_mx, assume_hermitian=True)

        assert(truncate or all([ev >= -1e-12 for ev in evals])), \
            "`vec` must correspond to a positive density matrix (truncate == False)!"

        pos_evals = evals.clip(1e-16, 1e100)
        density_mx = U @ _np.diag(pos_evals) @ Ui
        try:
            Lmx = _np.linalg.cholesky(density_mx)
        except _np.linalg.LinAlgError:  # Lmx not postitive definite?
            pos_evals = evals.clip(1e-12, 1e100)  # try again with 1e-12
            density_mx =  U @ _np.diag(pos_evals) @ Ui
            Lmx = _np.linalg.cholesky(density_mx)

        #check TP condition: that diagonal els of Lmx squared add to 1.0
        Lmx_norm = _np.linalg.norm(Lmx)  # = sqrt(tr(Lmx' Lmx))
        assert(_np.isclose(Lmx_norm, 1.0)), \
            "Cholesky decomp didn't preserve trace=1!"

        self.dmDim = dmDim
        self.params = _np.empty(dmDim**2, 'd')
        for i in range(dmDim):
            assert(_np.linalg.norm(_np.imag(Lmx[i, i])) < IMAG_TOL)
            self.params[i * dmDim + i] = Lmx[i, i].real  # / paramNorm == 1 as asserted above
            for j in range(i):
                self.params[i * dmDim + j] = Lmx[i, j].real
                self.params[j * dmDim + i] = Lmx[i, j].imag

    def _construct_vector(self):
        dmDim = self.dmDim

        #  params is an array of length dmDim^2 that
        #  encodes a lower-triangular matrix "Lmx" via:
        #  Lmx[i,i] = params[i*dmDim + i] / param-norm  # i = 0...dmDim-2
        #     *last diagonal el is given by sqrt(1.0 - sum(L[i,j]**2))
        #  Lmx[i,j] = params[i*dmDim + j] + 1j*params[j*dmDim+i] (i > j)

        param2Sum = _np.vdot(self.params, self.params)  # or "dot" would work, since params are real
        paramNorm = _np.sqrt(param2Sum)  # also the norm of *all* Lmx els

        for i in range(dmDim):
            self.Lmx[i, i] = self.params[i * dmDim + i] / paramNorm
            for j in range(i):
                self.Lmx[i, j] = (self.params[i * dmDim + j] + 1j * self.params[j * dmDim + i]) / paramNorm

        Lmx_norm = _np.linalg.norm(self.Lmx)  # = sqrt(tr(Lmx' Lmx))
        assert(_np.isclose(Lmx_norm, 1.0)), "Violated trace=1 condition!"

        #The (complex, Hermitian) density matrix is build by
        # assuming Lmx is its Cholesky decomp, which makes
        # the density matrix is pos-def.
        density_mx = _np.dot(self.Lmx, self.Lmx.T.conjugate())
        assert(_np.isclose(_np.trace(density_mx), 1.0)), "density matrix must be trace == 1"

        # write density matrix in given basis: = sum_i alpha_i B_i
        # ASSUME that basis is orthogonal, i.e. Tr(Bi^dag*Bj) = delta_ij
        basis_mxs = _np.rollaxis(self.basis_mxs, 2)  # shape (dmDim, dmDim, len(vec))
        vec = _np.array([_np.vdot(M, density_mx) for M in basis_mxs])

        #for now, assume Liouville vector should always be real (TODO: add 'real' flag later?)
        assert(_np.linalg.norm(_np.imag(vec)) < IMAG_TOL)
        vec = _np.real(vec)

        self._ptr.flags.writeable = True
        self._ptr[:] = vec[:]  # so shape is (dim,1) - the convention for spam vectors
        self._ptr.flags.writeable = False

    def set_dense(self, vec):
        """
        Set the dense-vector value of this state vector.

        Attempts to modify this state vector's parameters so that the raw
        state vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or State
            A numpy array representing a state vector, or a State object.

        Returns
        -------
        None
        """
        try:
            self._set_params_from_vector(vec, truncate=False)
            self.dirty = True
        except AssertionError as e:
            raise ValueError("Error initializing the parameters of this "
                             "CPTPState object: " + str(e))

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        assert(self.dmDim**2 == self.dim)  # should at least be true without composite bases...
        return self.dmDim**2

    def to_vector(self):
        """
        Get the state vector parameters as an array of values.

        Returns
        -------
        numpy array
            The parameters as a 1D array with length num_params().
        """
        return self.params

    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize the state vector using a 1D array of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of state vector parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this state vector's current
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
        self.params[:] = v[:]
        self._construct_vector()
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this state vector.

        Construct a matrix whose columns are the derivatives of the state vector
        with respect to a single param.  Thus, each column is of length
        dimension and there is one column per state vector parameter.

        Parameters
        ----------
        wrt_filter : list or numpy.ndarray
            List of parameter indices to take derivative with respect to.
            (None means to use all the this operation's parameters.)

        Returns
        -------
        numpy array
            Array of derivatives, shape == (dimension, num_params)
        """
        dmDim = self.dmDim
        nP = len(self.params)
        assert(nP == dmDim**2)  # number of parameters

        # v_i = trace( B_i^dag * Lmx * Lmx^dag )
        # d(v_i) = trace( B_i^dag * (dLmx * Lmx^dag + Lmx * (dLmx)^dag) )  #trace = linear so commutes w/deriv
        #               /
        # where dLmx/d[ab] = {
        #               \
        L, Lbar = self.Lmx, self.Lmx.conjugate()
        F1 = _np.tril(_np.ones((dmDim, dmDim), 'd'))
        F2 = _np.triu(_np.ones((dmDim, dmDim), 'd'), 1) * 1j
        conj_basis_mxs = self.basis_mxs.conjugate()

        # Derivative of vector wrt params; shape == [vecLen,dmDim,dmDim] *not dealing with TP condition yet*
        # (first get derivative assuming last diagonal el of Lmx *is* a parameter, then use chain rule)
        dVdp = _np.einsum('aml,mb,ab->lab', conj_basis_mxs, Lbar, F1)  # only a >= b nonzero (F1)
        dVdp += _np.einsum('mal,mb,ab->lab', conj_basis_mxs, L, F1)    # ditto
        dVdp += _np.einsum('bml,ma,ab->lab', conj_basis_mxs, Lbar, F2)  # only b > a nonzero (F2)
        dVdp += _np.einsum('mbl,ma,ab->lab', conj_basis_mxs, L, F2.conjugate())  # ditto

        dVdp.shape = [dVdp.shape[0], nP]  # jacobian with respect to "p" params,
        # which don't include normalization for TP-constraint

        #Now get jacobian of actual params wrt the params used above. Denote the actual
        # params "P" in variable names, so p_ij = P_ij / sqrt(sum(P_xy**2))
        param2Sum = _np.vdot(self.params, self.params)
        paramNorm = _np.sqrt(param2Sum)  # norm of *all* Lmx els (note lastDiagEl
        dpdP = _np.identity(nP, 'd')

        # all p_ij params ==  P_ij / paramNorm = P_ij / sqrt(sum(P_xy**2))
        # and so have derivs wrt *all* Pxy elements.
        for ij in range(nP):
            for kl in range(nP):
                if ij == kl:
                    # dp_ij / dP_ij = 1.0 / (sum(P_xy**2))^(1/2) - 0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_ij
                    #               = 1.0 / (sum(P_xy**2))^(1/2) - P_ij^2 / (sum(P_xy**2))^(3/2)
                    dpdP[ij, ij] = 1.0 / paramNorm - self.params[ij]**2 / paramNorm**3
                else:
                    # dp_ij / dP_kl = -0.5 * P_ij / (sum(P_xy**2))^(3/2) * 2*P_kl
                    #               = - P_ij * P_kl / (sum(P_xy**2))^(3/2)
                    dpdP[ij, kl] = - self.params[ij] * self.params[kl] / paramNorm**3

        #Apply the chain rule to get dVdP:
        dVdP = _np.dot(dVdp, dpdP)  # shape (vecLen, nP) - the jacobian!
        dVdp = dpdP = None  # free memory!

        assert(_np.linalg.norm(_np.imag(dVdP)) < IMAG_TOL)
        derivMx = _np.real(dVdP)

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this state vector has a non-zero Hessian with respect to its parameters.

        Returns
        -------
        bool
        """
        return True

    def hessian_wrt_params(self, wrt_filter1=None, wrt_filter2=None):
        """
        Construct the Hessian of this state vector with respect to its parameters.

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
            Hessian with shape (dimension, num_params1, num_params2)
        """
        raise NotImplementedError("TODO: add hessian computation for CPTPState")
