"""
The FullCPTPOp class and supporting functionality.
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

from pygsti.modelmembers.operations.linearop import LinearOperator as _LinearOperator
from pygsti.modelmembers.operations.krausop import KrausOperatorInterface as _KrausOperatorInterface

from pygsti.evotypes import Evotype as _Evotype
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.tools import jamiolkowski as _jt
from pygsti.tools import basistools as _bt

IMAG_TOL = 1e-7


class FullCPTPOp(_KrausOperatorInterface, _LinearOperator):
    """
    TODO: update docstring
    An operator that is constrained to be CPTP.

    This operation is parameterized by (normalized) elements of the Cholesky decomposition
    of the quantum channel's Choi matrix.
    """

    @classmethod
    def from_superop_matrix(cls, superop_mx, basis, evotype, state_space=None, truncate=False):
        choi_mx = _jt.jamiolkowski_iso(superop_mx, basis, basis)  # normalized (trace == 1) Choi matrix
        return cls(choi_mx, basis, evotype, state_space, truncate)

    def __init__(self, choi_mx, basis, evotype, state_space=None, truncate=False):
        choi_mx = _LinearOperator.convert_to_matrix(choi_mx)
        state_space = _statespace.default_space_for_dim(choi_mx.shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)
        evotype = _Evotype.cast(evotype)
        self._basis = _Basis.cast(basis, state_space.dim) if (basis is not None) else None  # for Hilbert-Schmidt space

        #scratch space
        self.Lmx = _np.zeros((state_space.dim, state_space.dim), complex)

        #Currently just use dense rep - maybe something more specific in future?
        try:
            udim = state_space.udim
            kraus_rank = state_space.dim
            kraus_reps = [evotype.create_dense_unitary_rep(_np.zeros((udim, udim), complex), self._basis, state_space)
                          for i in range(kraus_rank)]
            rep = evotype.create_kraus_rep(self._basis, kraus_reps, state_space)
            self._reptype = 'kraus'
        except Exception:
            rep = evotype.create_dense_superop_rep(_np.identity(state_space.dim, 'd'), self._basis, state_space)
            self._reptype = 'dense'

        _LinearOperator.__init__(self, rep, evotype)
        self._set_params_from_choi_mx(choi_mx, truncate)
        self._update_rep()

    def _set_params_from_choi_mx(self, choi_mx, truncate):
        #given choi_mx (assumed to be in self._basis), compute and set parameters -> self.params[:]
        trc = _np.trace(choi_mx)
        assert(choi_mx.shape[0] == self.state_space.dim)
        assert(truncate or _np.isclose(trc, 1.0)), \
            "`choi_mx` must have trace=1 (truncate == False)!"

        dim = self.state_space.dim
        if not _np.isclose(trc, 1.0):  # truncate to trace == 1
            choi_mx -= _np.identity(dim, 'd') / dim * (trc - 1.0)

        #push any slightly negative evals of density_mx positive
        # so that the Cholesky decomp will work.
        evals, U = _np.linalg.eig(choi_mx)
        Ui = _np.linalg.inv(U)

        assert(truncate or all([ev >= -1e-12 for ev in evals])), \
            "`choi_mx` must be positive (truncate == False)!"

        pos_evals = evals.clip(1e-16, 1e100)
        choi_mx = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))
        try:
            Lmx = _np.linalg.cholesky(choi_mx)
        except _np.linalg.LinAlgError:  # Lmx not postitive definite?
            pos_evals = evals.clip(1e-12, 1e100)  # try again with 1e-12
            choi_mx = _np.dot(U, _np.dot(_np.diag(pos_evals), Ui))
            Lmx = _np.linalg.cholesky(choi_mx)

        #check TP condition: that diagonal els of Lmx squared add to 1.0
        Lmx_norm = _np.trace(_np.dot(Lmx.T.conjugate(), Lmx))  # sum of magnitude^2 of all els
        assert(_np.isclose(Lmx_norm, 1.0)), "Cholesky decomp didn't preserve trace=1!"

        self.params = _np.empty(dim**2, 'd')
        for i in range(dim):
            assert(_np.linalg.norm(_np.imag(Lmx[i, i])) < IMAG_TOL)
            self.params[i * dim + i] = Lmx[i, i].real  # / paramNorm == 1 as asserted above
            for j in range(i):
                self.params[i * dim + j] = Lmx[i, j].real
                self.params[j * dim + i] = Lmx[i, j].imag

    def _get_choi_mx_from_params(self):
        dim = self.state_space.dim

        #  params is an array of length dim^2 that
        #  encodes a lower-triangular matrix "Lmx" via:
        #  Lmx[i,i] = params[i*dim + i] / param-norm  # i = 0...dim-2
        #     *last diagonal el is given by sqrt(1.0 - sum(L[i,j]**2))
        #  Lmx[i,j] = params[i*dim + j] + 1j*params[j*dim+i] (i > j)

        param2Sum = _np.vdot(self.params, self.params)  # or "dot" would work, since params are real
        paramNorm = _np.sqrt(param2Sum)  # also the norm of *all* Lmx els

        for i in range(dim):
            self.Lmx[i, i] = self.params[i * dim + i] / paramNorm
            for j in range(i):
                self.Lmx[i, j] = (self.params[i * dim + j] + 1j * self.params[j * dim + i]) / paramNorm

        choi_mx = _np.dot(self.Lmx, self.Lmx.T.conjugate())
        assert(_np.isclose(_np.trace(choi_mx), 1.0)), "Choi matrix trace == 1 condition violated!"
        return choi_mx

    def _update_dense_rep(self):
        choi_mx = self._get_choi_mx_from_params()
        self._rep.base[:, :] = _jt.jamiolkowski_iso_inv(choi_mx, self._basis, self._basis)
        # Note: we assume superop transfer mx and choi mx are in same self._basis

    def _update_kraus_rep(self):
        choi_mx = self._get_choi_mx_from_params(); d = self.state_space.udim
        choi_mx = _bt.change_basis(choi_mx, self._basis, _Basis.cast('std', choi_mx.shape[0]))
        evals, evecs = _np.linalg.eig(choi_mx * d)  # 'un-normalize' choi_mx so that:
        # op(rho) = sum_IJ choi_IJ BI rho BJ_dag is true (assumed for kraus op construction below)

        assert(all([ev > -1e-7 for ev in evals])), "Failed Kraus decomp - Choi mx must not be positive!"
        for i, (kraus_rep, ev) in enumerate(zip(self._rep.kraus_reps, evals)):
            # We know kraus_rep is a unitary op rep, with .base == dense mx that is Kraus op
            kraus_rep.base[:, :] = evecs[:, i].reshape(d, d) * _np.sqrt(ev)  # Note: ev can be ~0 and this is OK
            kraus_rep.base_has_changed()

    def _update_rep(self):
        if self._reptype == 'kraus':
            self._update_kraus_rep()
        else:  # self._reptype == 'dense':
            self._update_dense_rep()

    def to_dense(self, on_space='minimal'):
        """
        Return the dense array used to represent this operation within its evolution type.

        Note: for efficiency, this doesn't copy the underlying data, so
        the caller should copy this data before modifying it.

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
        return self._rep.to_dense(on_space)  # both types of possible reps implement 'to_dense'

    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this state vector.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return self.state_space.dim**2

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
        self._update_rep()
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
        raise NotImplementedError("TODO: add deriv_wrt_params computation for FullCPTPOp!")

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
        raise NotImplementedError("TODO: add hessian computation for FullCPTPOp")
