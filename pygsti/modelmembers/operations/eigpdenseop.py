"""
The EigenvalueParamDenseOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import functools as _functools

import numpy as _np

from pygsti.modelmembers.operations.denseop import DenseOperator as _DenseOperator
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.tools import matrixtools as _mt

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


class EigenvalueParamDenseOp(_DenseOperator):
    """
    A real operation matrix parameterized only by its eigenvalues.

    These eigenvalues are assumed to be either real or to occur in
    conjugate pairs.  Thus, the number of parameters is equal to the
    number of eigenvalues.

    Parameters
    ----------
    matrix : numpy array
        a square 2D numpy array that gives the raw operation matrix to
        paramterize.  The shape of this array sets the dimension
        of the operation.

    include_off_diags_in_degen_blocks : bool or int
        If True, include as parameters the (initially zero)
        off-diagonal elements in degenerate blocks of the
        the diagonalized operation matrix.  If an integer, no
        off-diagonals are included in blocks larger than n x n, where
        `n == include_off_diags_in_degen_blocks`.  This is an option
        specifically used in the per-germ-power fiducial pair
        reduction (FPR) algorithm.

    tp_constrained_and_unital : bool
        If True, assume the top row of the operation matrix is fixed
        to `[1, 0, ... 0]` and should not be parameterized, and verify
        that the matrix is unital.  In this case, "1" is always a
        fixed (not-paramterized) eigenvalue with eigenvector
        `[1,...0]` and if include_off_diags_in_degen_blocks is True
        any off diagonal elements lying on the top row are *not*
        parameterized as implied by the TP constraint.

    basis : Basis or {'pp','gm','std'} or None
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.  If None, certain functionality,
        such as access to Kraus operators, will be unavailable.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, matrix, include_off_diags_in_degen_blocks=False,
                 tp_constrained_and_unital=False, basis=None, evotype="default", state_space=None):

        def cmplx_compare(ia, ib):
            return _mt.complex_compare(evals[ia], evals[ib])
        cmplx_compare_key = _functools.cmp_to_key(cmplx_compare)

        def isreal(a):
            """ b/c numpy's isreal tests for strict equality w/0 """
            return _np.isclose(_np.imag(a), 0.0)

        # Since matrix is real, eigenvalues must either be real or occur in
        #  conjugate pairs.  Find and sort by conjugate pairs.

        assert(_np.linalg.norm(_np.imag(matrix)) < IMAG_TOL)  # matrix should be real
        evals, B = _np.linalg.eig(matrix)  # matrix == B * diag(evals) * Bi
        dim = len(evals)

        #Sort eigenvalues & eigenvectors by:
        # 1) unit eigenvalues first (with TP eigenvalue first of all)
        # 2) non-unit real eigenvalues in order of magnitude
        # 3) complex eigenvalues in order of real then imaginary part

        unitInds = []; realInds = []; complexInds = []
        for i, ev in enumerate(evals):
            if _np.isclose(ev, 1.0): unitInds.append(i)
            elif isreal(ev): realInds.append(i)
            else: complexInds.append(i)

        if tp_constrained_and_unital:
            #check matrix is TP and unital
            unitRow = _np.zeros((len(evals)), 'd'); unitRow[0] = 1.0
            assert(_np.allclose(matrix[0, :], unitRow))
            assert(_np.allclose(matrix[:, 0], unitRow))

            #find the eigenvector with largest first element and make sure
            # this is the first index in unitInds
            k = _np.argmax([B[0, i] for i in unitInds])
            if k != 0:  # swap indices 0 <-> k in unitInds
                t = unitInds[0]; unitInds[0] = unitInds[k]; unitInds[k] = t

            #Assume we can recombine unit-eval eigenvectors so that the first
            # one (actually the closest-to-unit-row one) == unitRow and the
            # rest do not have any 0th component.
            iClose = _np.argmax([abs(B[0, ui]) for ui in unitInds])
            B[:, unitInds[iClose]] = unitRow
            for i, ui in enumerate(unitInds):
                if i == iClose: continue
                B[0, ui] = 0.0; B[:, ui] /= _np.linalg.norm(B[:, ui])
            assert(_np.allclose(matrix, B @ _np.diag(evals) @ _np.linalg.inv(B)))

        realInds = sorted(realInds, key=lambda i: -abs(evals[i]))
        complexInds = sorted(complexInds, key=cmplx_compare_key)
        new_ordering = unitInds + realInds + complexInds

        #Re-order the eigenvalues & vectors
        sorted_evals = _np.zeros(evals.shape, 'complex')
        sorted_B = _np.zeros(B.shape, 'complex')
        for i, indx in enumerate(new_ordering):
            sorted_evals[i] = evals[indx]
            sorted_B[:, i] = B[:, indx]

        #Save the final list of (sorted) eigenvalues & eigenvectors
        self.evals = sorted_evals
        self.B = sorted_B
        self.Bi = _np.linalg.inv(sorted_B)

        self.options = {'includeOffDiags': include_off_diags_in_degen_blocks,
                        'TPandUnital': tp_constrained_and_unital}

        #Check that nothing has gone horribly wrong
        assert(_np.allclose(_np.dot(
            self.B, _np.dot(_np.diag(self.evals), self.Bi)), matrix))

        #Build a list of parameter descriptors.  Each element of self.params
        # is a list of (prefactor, (i,j)) tuples.
        self.params = []
        paramlbls = []
        i = 0; N = len(self.evals); processed = [False] * N
        while i < N:
            if processed[i]:
                i += 1; continue

            # Find block (i -> j) of degenerate eigenvalues
            j = i + 1
            while j < N and _np.isclose(self.evals[i], self.evals[j]): j += 1
            blkSize = j - i

            #Add eigenvalues as parameters
            ev = self.evals[i]  # current eigenvalue being processed
            if isreal(ev):

                # Side task: for a *real* block of degenerate evals, we want
                # to ensure the eigenvectors are real, which numpy doesn't
                # always guarantee (could be conj. pairs for instance).

                # Solve or Cmx: [v1,v2,v3,v4]Cmx = [v1',v2',v3',v4'] ,
                # where ' qtys == real, so Im([v1,v2,v3,v4]Cmx) = 0
                # Let Cmx = Cr + i*Ci, v1 = v1.r + i*v1.i, etc.,
                #  then solve [v1.r, ...]Ci + [v1.i, ...]Cr = 0
                #  which can be cast as [Vr,Vi]*[Ci] = 0
                #                               [Cr]      (nullspace of V)
                # Note: only involve complex evecs (don't disturb TP evec!)
                evecIndsToMakeReal = []
                for k in range(i, j):
                    if _np.linalg.norm(self.B[:, k].imag) >= IMAG_TOL:
                        evecIndsToMakeReal.append(k)

                nToReal = len(evecIndsToMakeReal)
                if nToReal > 0:
                    vecs = _np.empty((dim, nToReal), 'complex')
                    for ik, k in enumerate(evecIndsToMakeReal):
                        vecs[:, ik] = self.B[:, k]
                    V = _np.concatenate((vecs.real, vecs.imag), axis=1)
                    nullsp = _mt.nullspace(V)
                    # if nullsp.shape[1] < nToReal: # DEBUG
                    #    raise ValueError("Nullspace only has dimension %d when %d was expected! "
                    #                     "(i=%d, j=%d, blkSize=%d)\nevals = %s" \
                    #                     % (nullsp.shape[1],nToReal, i,j,blkSize,str(self.evals)) )
                    assert(nullsp.shape[1] >= nToReal), "Cannot find enough real linear combos!"
                    nullsp = nullsp[:, 0:nToReal]  # truncate #cols if there are more than we need

                    Cmx = nullsp[nToReal:, :] + 1j * nullsp[0:nToReal, :]  # Cr + i*Ci
                    new_vecs = vecs @ Cmx
                    assert(_np.linalg.norm(new_vecs.imag) < IMAG_TOL), \
                        "Imaginary mag = %g!" % _np.linalg.norm(new_vecs.imag)
                    for ik, k in enumerate(evecIndsToMakeReal):
                        self.B[:, k] = new_vecs[:, ik]
                    self.Bi = _np.linalg.inv(self.B)

                #Now, back to constructing parameter descriptors...
                for k in range(i, j):
                    if tp_constrained_and_unital and k == 0: continue
                    prefactor = 1.0; mx_indx = (k, k)
                    self.params.append([(prefactor, mx_indx)])
                    paramlbls.append("Real eigenvalue %d" % k)
                    processed[k] = True
            else:
                iConjugate = {}
                for k in range(i, j):
                    #Find conjugate eigenvalue to eval[k]
                    conj = _np.conj(self.evals[k])  # == conj(ev), indep of k
                    conjB = _np.conj(self.B[:, k])
                    for l in range(j, N):
                        # numpy normalizes but doesn't fix "phase" of evecs
                        if _np.isclose(conj, self.evals[l]) \
                           and (_np.allclose(conjB, self.B[:, l])
                                or _np.allclose(conjB, 1j * self.B[:, l])
                                or _np.allclose(conjB, -1j * self.B[:, l])
                                or _np.allclose(conjB, -1 * self.B[:, l])):
                            self.params.append([  # real-part param
                                (1.0, (k, k)),  # (prefactor, index)
                                (1.0, (l, l))])
                            self.params.append([  # imag-part param
                                (1j, (k, k)),  # (prefactor, index)
                                (-1j, (l, l))])
                            paramlbls.append("Eigenvalue-pair (%d,%d) Re-part" % (k, l))
                            paramlbls.append("Eigenvalue-pair (%d,%d) Im-part" % (k, l))
                            processed[k] = processed[l] = True
                            iConjugate[k] = l  # save conj. pair index for below
                            break
                    else:
                        # should be unreachable, since we ensure mx is real above - but
                        # this may fail when there are multiple degenerate complex evals
                        # since the evecs can get mixed (and we check for evec "match" above)
                        raise ValueError("Could not find conjugate pair "
                                         + " for %s" % self.evals[k])  # pragma: no cover

            if include_off_diags_in_degen_blocks is True \
               or (isinstance(include_off_diags_in_degen_blocks, int)
                   and 1 < blkSize < include_off_diags_in_degen_blocks):
                #Note: we removed " and blkSize == 2" part of above condition, as
                # the purpose was just to avoid adding lots of off-diag elements
                # in accidentally-degenerate cases, BUT we need to handle blkSize>2
                # appropriately to do FPR on idle gates.  There may be a better
                # heuristic for avoiding accidental degeneracies (FUTURE work).
                for k1 in range(i, j - 1):
                    for k2 in range(k1 + 1, j):
                        if isreal(ev):
                            # k1,k2 element
                            if not tp_constrained_and_unital or k1 != 0:  # (k2 can never be 0)
                                self.params.append([(1.0, (k1, k2))])
                                paramlbls.append("Off-diag (%d,%d) of real eigval block" % (k1, k2))

                            # k2,k1 element
                            if not tp_constrained_and_unital or k1 != 0:  # (k2 can never be 0)
                                self.params.append([(1.0, (k2, k1))])
                                paramlbls.append("Off-diag (%d,%d) of real eigval block" % (k2, k1))
                        else:
                            k1c, k2c = iConjugate[k1], iConjugate[k2]

                            # k1,k2 element
                            self.params.append([  # real-part param
                                (1.0, (k1, k2)),
                                (1.0, (k1c, k2c))])
                            self.params.append([  # imag-part param
                                (1j, (k1, k2)),
                                (-1j, (k1c, k2c))])
                            paramlbls.append("Off-diags (%d,%d), (%d,%d) Re-part for eigval-pair blocks" % (
                                k1, k2, k1c, k2c))
                            paramlbls.append("Off-diags (%d,%d), (%d,%d) Im-part for eigval-pair blocks" % (
                                k1, k2, k1c, k2c))

                            # k2,k1 element
                            self.params.append([  # real-part param
                                (1.0, (k2, k1)),
                                (1.0, (k2c, k1c))])
                            self.params.append([  # imag-part param
                                (1j, (k2, k1)),
                                (-1j, (k2c, k1c))])
                            paramlbls.append("Off-diags (%d,%d), (%d,%d) Re-part for eigval-pair blocks" % (
                                k2, k1, k2c, k1c))
                            paramlbls.append("Off-diags (%d,%d), (%d,%d) Im-part for eigval-pair blocks" % (
                                k2, k1, k2c, k1c))

            i = j  # advance to next block

        #Allocate array of parameter values (all zero initially)
        self.paramvals = _np.zeros(len(self.params), 'd')

        #Finish LinearOperator construction
        mx = _np.empty(matrix.shape, "d")
        _DenseOperator.__init__(self, mx, basis, evotype, state_space)
        self._ptr.flags.writeable = False  # only _construct_matrix can change array
        self._construct_matrix()  # construct base from the parameters

        #Set parameter labels
        self._paramlbls = _np.array(paramlbls, dtype=object)

    def _construct_matrix(self):
        """
        Build the internal operation matrix using the current parameters.
        """
        base_diag = _np.diag(self.evals)
        for pdesc, pval in zip(self.params, self.paramvals):
            for prefactor, (i, j) in pdesc:
                base_diag[i, j] += prefactor * pval
        matrix = _np.dot(self.B, _np.dot(base_diag, self.Bi))
        assert(_np.linalg.norm(matrix.imag) < IMAG_TOL)
        assert(matrix.shape == (self.dim, self.dim))
        self._ptr.flags.writeable = True
        self._ptr[:, :] = matrix.real
        self._ptr.flags.writeable = False

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
        mm_dict = super().to_memoized_dict(mmg_memo)  # includes 'dense_matrix' from DenseOperator
        mm_dict['include_off_diags_in_degen_blocks'] = self.options['includeOffDiags']
        mm_dict['tp_constrained_and_unital'] = self.options['TPandUnital']
        return mm_dict

    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        matrix = cls._decodemx(mm_dict['dense_matrix'])
        state_space = _StateSpace.from_nice_serialization(mm_dict['state_space'])
        return cls(matrix, mm_dict['include_off_diags_in_degen_blocks'],
                   mm_dict['tp_constrained_and_unital'], mm_dict['evotype'], state_space)

    def _is_similar(self, other, rtol, atol):
        """ Returns True if `other` model member (which it guaranteed to be the same type as self) has
            the same local structure, i.e., not considering parameter values or submembers """
        return all([self.options[k] == other.options[k] for k in self.options])

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
        self.paramvals = v
        self._construct_matrix()
        self.dirty = dirty_value

    def deriv_wrt_params(self, wrt_filter=None):
        """
        The element-wise derivative this operation.

        Construct a matrix whose columns are the vectorized
        derivatives of the flattened operation matrix with respect to a
        single operation parameter.  Thus, each column is of length
        op_dim^2 and there is one column per operation parameter.

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

        # matrix = B * diag * Bi, and only diag depends on parameters
        #  (and only linearly), so:
        # d(matrix)/d(param) = B * d(diag)/d(param) * Bi

        # EigenvalueParameterizedGates are assumed to be real
        derivMx = _np.zeros((self.dim**2, self.num_params), 'd')

        # Compute d(diag)/d(param) for each params, then apply B & Bi
        for k, pdesc in enumerate(self.params):
            dMx = _np.zeros((self.dim, self.dim), 'complex')
            for prefactor, (i, j) in pdesc:
                dMx[i, j] = prefactor
            tmp = _np.dot(self.B, _np.dot(dMx, self.Bi))
            if _np.linalg.norm(tmp.imag) >= IMAG_TOL:  # just a warning until we figure this out.
                print("EigenvalueParamDenseOp deriv_wrt_params WARNING:"
                      " Imag part = ", _np.linalg.norm(tmp.imag), " pdesc = ", pdesc)  # pragma: no cover
            #assert(_np.linalg.norm(tmp.imag) < IMAG_TOL), \
            #       "Imaginary mag = %g!" % _np.linalg.norm(tmp.imag)
            derivMx[:, k] = tmp.real.flatten()

        if wrt_filter is None:
            return derivMx
        else:
            return _np.take(derivMx, wrt_filter, axis=1)

    def has_nonzero_hessian(self):
        """
        Whether this operation has a non-zero Hessian with respect to its parameters.

        (i.e. whether it only depends linearly on its parameters or not)

        Returns
        -------
        bool
        """
        return False
