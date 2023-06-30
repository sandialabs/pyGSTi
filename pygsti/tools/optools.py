"""
Utility functions operating on operation matrices
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import warnings as _warnings

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import functools as _functools

from pygsti.tools import basistools as _bt
from pygsti.tools import jamiolkowski as _jam
from pygsti.tools import lindbladtools as _lt
from pygsti.tools import matrixtools as _mt
from pygsti.baseobjs.basis import Basis as _Basis, ExplicitBasis as _ExplicitBasis, DirectSumBasis as _DirectSumBasis
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel
from pygsti.tools.legacytools import deprecate as _deprecated_fn

IMAG_TOL = 1e-7  # tolerance for imaginary part being considered zero


def _flat_mut_blks(i, j, block_dims):
    # like _mut(i,j,dim).flatten() but works with basis *blocks*
    N = sum(block_dims)
    mx = _np.zeros((N, N), 'd'); mx[i, j] = 1.0
    ret = _np.zeros(sum([d**2 for d in block_dims]), 'd')
    i = 0; off = 0
    for d in block_dims:
        ret[i:i + d**2] = mx[off:off + d, off:off + d].flatten()
        i += d**2; off += d
    return ret


def _hack_sqrtm(a):
    sqrt, _ = _spl.sqrtm(a, disp=False)  # Travis found this scipy function
    # to be incorrect in certain cases (we need a workaround)
    if _np.any(_np.isnan(sqrt)):  # this is sometimes a good fallback when sqrtm doesn't work.
        ev, U = _np.linalg.eig(a)
        sqrt = _np.dot(U, _np.dot(_np.diag(_np.sqrt(ev)), _np.linalg.inv(U)))
    # Scipy 1.10 fix for PR 16294 (which doubles precision of complex to complex)
    if _np.iscomplexobj(a):
        sqrt = sqrt.astype(a.dtype, copy=False)

    return sqrt


def fidelity(a, b):
    """
    Returns the quantum state fidelity between density matrices.

    This given by:

      `F = Tr( sqrt{ sqrt(a) * b * sqrt(a) } )^2`

    To compute process fidelity, pass this function the
    Choi matrices of the two processes, or just call
    :func:`entanglement_fidelity` with the operation matrices.

    Parameters
    ----------
    a : numpy array
        First density matrix.

    b : numpy array
        Second density matrix.

    Returns
    -------
    float
        The resulting fidelity.
    """
    evals, U = _np.linalg.eig(a)
    if len([ev for ev in evals if abs(ev) > 1e-8]) == 1:
        # special case when a is rank 1, a = vec * vec^T and sqrt(a) = a
        ivec = _np.argmax(evals)
        vec = U[:, ivec:(ivec + 1)]
        F = evals[ivec].real * _np.dot(_np.conjugate(_np.transpose(vec)), _np.dot(b, vec)).real  # vec^T * b * vec
        return float(F)

    evals, U = _np.linalg.eig(b)
    if len([ev for ev in evals if abs(ev) > 1e-8]) == 1:
        # special case when b is rank 1 (recally fidelity is sym in args)
        ivec = _np.argmax(evals)
        vec = U[:, ivec:(ivec + 1)]
        F = evals[ivec].real * _np.dot(_np.conjugate(_np.transpose(vec)), _np.dot(a, vec)).real  # vec^T * a * vec
        return float(F)

    #if _np.array_equal(a, b): return 1.0  # HACK - some cases when a and b are perfecty equal sqrtm(a) fails...
    sqrtA = _hack_sqrtm(a)  # _spl.sqrtm(a)
    # test the scipy sqrtm function - sometimes fails when rank defficient
    #assert(_np.linalg.norm(_np.dot(sqrtA, sqrtA) - a) < 1e-8)
    if _np.linalg.norm(_np.dot(sqrtA, sqrtA) - a) > 1e-8:
        evals = _np.linalg.eigvals(a)
        _warnings.warn(("sqrtm(a) failure when computing fidelity - beware result. "
                        "Maybe due to rank defficiency - eigenvalues of a are: %s") % evals)
    F = (_mt.trace(_hack_sqrtm(_np.dot(sqrtA, _np.dot(b, sqrtA)))).real)**2  # Tr( sqrt{ sqrt(a) * b * sqrt(a) } )^2
    return float(F)


def frobeniusdist(a, b):
    """
    Returns the frobenius distance between gate or density matrices.

    This is given by :

      `sqrt( sum( (a_ij-b_ij)^2 ) )`

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    Returns
    -------
    float
        The resulting frobenius distance.
    """
    return _mt.frobeniusnorm(a - b)


def frobeniusdist_squared(a, b):
    """
    Returns the square of the frobenius distance between gate or density matrices.

    This is given by :

      `sum( (A_ij-B_ij)^2 )`

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    Returns
    -------
    float
        The resulting frobenius distance.
    """
    return _mt.frobeniusnorm_squared(a - b)


def residuals(a, b):
    """
    Calculate residuals between the elements of two matrices

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    Returns
    -------
    np.array
        residuals
    """
    return (a - b).flatten()


def tracenorm(a):
    """
    Compute the trace norm of matrix `a` given by:

    `Tr( sqrt{ a^dagger * a } )`

    Parameters
    ----------
    a : numpy array
        The matrix to compute the trace norm of.

    Returns
    -------
    float
    """
    if _np.linalg.norm(a - _np.conjugate(a.T)) < 1e-8:
        #Hermitian, so just sum eigenvalue magnitudes
        return _np.sum(_np.abs(_np.linalg.eigvals(a)))
    else:
        #Sum of singular values (positive by construction)
        return _np.sum(_np.linalg.svd(a, compute_uv=False))


def tracedist(a, b):
    """
    Compute the trace distance between matrices.

    This is given by:

      `D = 0.5 * Tr( sqrt{ (a-b)^dagger * (a-b) } )`

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    Returns
    -------
    float
    """
    return 0.5 * tracenorm(a - b)


def diamonddist(a, b, mx_basis='pp', return_x=False):
    """
    Returns the approximate diamond norm describing the difference between gate matrices.

    This is given by :

      `D = ||a - b ||_diamond = sup_rho || AxI(rho) - BxI(rho) ||_1`

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    mx_basis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    return_x : bool, optional
        Whether to return a numpy array encoding the state (rho) at
        which the maximal trace distance occurs.

    Returns
    -------
    dm : float
        Diamond norm
    W : numpy array
        Only returned if `return_x = True`.  Encodes the state rho, such that
        `dm = trace( |(J(a)-J(b)).T * W| )`.
    """
    mx_basis = _bt.create_basis_for_matrix(a, mx_basis)

    #currently cvxpy is only needed for this function, so don't import until here

    import cvxpy as _cvxpy

    #Check if using version < 1.0
    old_cvxpy = bool(tuple(map(int, _cvxpy.__version__.split('.'))) < (1, 0))

    # This SDP implementation is a modified version of Kevin's code

    #Compute the diamond norm

    #Uses the primal SDP from arXiv:1207.5726v2, Sec 3.2

    #Maximize 1/2 ( < J(phi), X > + < J(phi).dag, X.dag > )
    #Subject to  [[ I otimes rho0, X],
    #            [X.dag, I otimes rho1]] >> 0
    #              rho0, rho1 are density matrices
    #              X is linear operator

    #Jamiolkowski representation of the process
    #  J(phi) = sum_ij Phi(Eij) otimes Eij

    #< a, b > = Tr(a.dag b)

    #def vec(matrix_in):
    #    # Stack the columns of a matrix to return a vector
    #    return _np.transpose(matrix_in).flatten()
    #
    #def unvec(vector_in):
    #    # Slice a vector into columns of a matrix
    #    d = int(_np.sqrt(vector_in.size))
    #    return _np.transpose(vector_in.reshape( (d,d) ))

    #Code below assumes *un-normalized* Jamiol-isomorphism, so multiply by
    # density mx dimension (`smallDim`) below
    JAstd = _jam.fast_jamiolkowski_iso_std(a, mx_basis)
    JBstd = _jam.fast_jamiolkowski_iso_std(b, mx_basis)

    #Do this *after* the fast_jamiolkowski_iso calls above because these will convert
    # a & b to a "single-block" basis representation when mx_basis has multiple blocks.
    dim = JAstd.shape[0]
    smallDim = int(_np.sqrt(dim))
    JAstd *= smallDim  # see above comment
    JBstd *= smallDim  # see above comment
    assert(dim == JAstd.shape[1] == JBstd.shape[0] == JBstd.shape[1])

    #CHECK: Kevin's jamiolowski, which implements the un-normalized isomorphism:
    #  smallDim * _jam.jamiolkowski_iso(M, "std", "std")
    #def kevins_jamiolkowski(process, representation = 'superoperator'):
    #    # Return the Choi-Jamiolkowski representation of a quantum process
    #    # Add methods as necessary to accept different representations
    #    process = _np.array(process)
    #    if representation == 'superoperator':
    #        # Superoperator is the linear operator acting on vec(rho)
    #        dimension = int(_np.sqrt(process.shape[0]))
    #        print "dim = ",dimension
    #        jamiolkowski_matrix = _np.zeros([dimension**2, dimension**2], dtype='complex')
    #        for i in range(dimension**2):
    #            Ei_vec= _np.zeros(dimension**2)
    #            Ei_vec[i] = 1
    #            output = unvec(_np.dot(process,Ei_vec))
    #            tmp = _np.kron(output, unvec(Ei_vec))
    #            print "E%d = \n" % i,unvec(Ei_vec)
    #            #print "contrib =",_np.kron(output, unvec(Ei_vec))
    #            jamiolkowski_matrix += tmp
    #        return jamiolkowski_matrix
    #JAstd_kev = jamiolkowski(a)
    #JBstd_kev = jamiolkowski(b)
    #print "diff a = ",_np.linalg.norm(JAstd_kev/2.0-JAstd)
    #print "diff b = ",_np.linalg.norm(JBstd_kev/2.0-JBstd)

    #Kevin's function: def diamondnorm( jamiolkowski_matrix ):
    jamiolkowski_matrix = JBstd - JAstd

    # Here we define a bunch of auxiliary matrices because CVXPY doesn't use complex numbers

    K = jamiolkowski_matrix.real  # J.real
    L = jamiolkowski_matrix.imag  # J.imag

    if old_cvxpy:
        Y = _cvxpy.Variable(dim, dim)  # X.real
        Z = _cvxpy.Variable(dim, dim)  # X.imag

        sig0 = _cvxpy.Variable(smallDim, smallDim)  # rho0.real
        sig1 = _cvxpy.Variable(smallDim, smallDim)  # rho1.real
        tau0 = _cvxpy.Variable(smallDim, smallDim)  # rho1.imag
        tau1 = _cvxpy.Variable(smallDim, smallDim)  # rho1.imag

    else:
        Y = _cvxpy.Variable(shape=(dim, dim))  # X.real
        Z = _cvxpy.Variable(shape=(dim, dim))  # X.imag

        sig0 = _cvxpy.Variable(shape=(smallDim, smallDim))  # rho0.real
        sig1 = _cvxpy.Variable(shape=(smallDim, smallDim))  # rho1.real
        tau0 = _cvxpy.Variable(shape=(smallDim, smallDim))  # rho1.imag
        tau1 = _cvxpy.Variable(shape=(smallDim, smallDim))  # rho1.imag

    ident = _np.identity(smallDim, 'd')

    objective = _cvxpy.Maximize(_cvxpy.trace(K.T @ Y + L.T @ Z))
    constraints = [_cvxpy.bmat([
        [_cvxpy.kron(ident, sig0), Y, -_cvxpy.kron(ident, tau0), -Z],
        [Y.T, _cvxpy.kron(ident, sig1), Z.T, -_cvxpy.kron(ident, tau1)],
        [_cvxpy.kron(ident, tau0), Z, _cvxpy.kron(ident, sig0), Y],
        [-Z.T, _cvxpy.kron(ident, tau1), Y.T, _cvxpy.kron(ident, sig1)]]) >> 0,
        _cvxpy.bmat([[sig0, -tau0],
                     [tau0, sig0]]) >> 0,
        _cvxpy.bmat([[sig1, -tau1],
                     [tau1, sig1]]) >> 0,
        sig0 == sig0.T,
        sig1 == sig1.T,
        tau0 == -tau0.T,
        tau1 == -tau1.T,
        _cvxpy.trace(sig0) == 1.,
        _cvxpy.trace(sig1) == 1.]

    prob = _cvxpy.Problem(objective, constraints)
    try:
        prob.solve(solver="CVXOPT")
#       prob.solve(solver="ECOS")
#       prob.solve(solver="SCS")#This always fails
    except _cvxpy.error.SolverError as e:
        _warnings.warn("CVXPY failed: %s - diamonddist returning -2!" % str(e))
        return (-2, _np.zeros((dim, dim))) if return_x else -2
    except:
        _warnings.warn("CVXOPT failed (uknown err) - diamonddist returning -2!")
        return (-2, _np.zeros((dim, dim))) if return_x else -2

    #Validate result
    #assert( abs(_np.trace(_np.dot(K.T,Y.value) + _np.dot(L.T,Z.value))-prob.value) < 1e-6 ), \
    #    "Diamondnorm mismatch"

    if return_x:
        X = Y.value + 1j * Z.value  # encodes state at which maximum trace-distance occurs
        return prob.value, X
    else:
        return prob.value


def jtracedist(a, b, mx_basis='pp'):  # Jamiolkowski trace distance:  Tr(|J(a)-J(b)|)
    """
    Compute the Jamiolkowski trace distance between operation matrices.

    This is given by:

      D = 0.5 * Tr( sqrt{ (J(a)-J(b))^2 } )

    where J(.) is the Jamiolkowski isomorphism map that maps a operation matrix
    to it's corresponding Choi Matrix.

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    float
    """
    JA = _jam.fast_jamiolkowski_iso_std(a, mx_basis)
    JB = _jam.fast_jamiolkowski_iso_std(b, mx_basis)
    return tracedist(JA, JB)


def entanglement_fidelity(a, b, mx_basis='pp', is_tp=None, is_unitary=None):
    """
    Returns the "entanglement" process fidelity between gate  matrices.

    This is given by:

      `F = Tr( sqrt{ sqrt(J(a)) * J(b) * sqrt(J(a)) } )^2`

    where J(.) is the Jamiolkowski isomorphism map that maps a operation matrix
    to it's corresponding Choi Matrix.
    
    When the both of the input matrices a and b are TP, and
    the target matrix b is unitary then we can use a more efficient
    formula:
    
      `F= Tr(a @ b.conjugate().T)/d^2
        
    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis of the matrices.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).
        
    is_tp : bool, optional (default None)
        Flag indicating both matrices are TP. If None (the default), 
        an explicit check is performed. If True/False, the check is 
        skipped and the provided value is used (faster, but should only 
        be used when the user is certain this is true apriori).

    is_unitary : bool, optional (default None)
        Flag indicating that the second matrix, b, is
        unitary. If None (the default) an explicit check is performed.
        If True/False, the check is skipped and the provided value is used
        (faster, but should only be used when the user is certain 
        this is true apriori).

    Returns
    -------
    float
    """
    d2 = a.shape[0]
    
    #if the tp flag isn't set we'll calculate whether it is true here
    if is_tp is None:
        def is_tp_fn(x): return _np.isclose(x[0, 0], 1.0) and all(
        [_np.isclose(x[0, i], 0) for i in range(1,d2)])
        
        is_tp= (is_tp_fn(a) and is_tp_fn(b))
   
    #if the unitary flag isn't set we'll calculate whether it is true here 
    if is_unitary is None:
        is_unitary= _np.allclose(_np.identity(d2, 'd'), _np.dot(b, b.conjugate().T))
    
    if is_tp and is_unitary:  # then assume TP-like gates & use simpler formula
        #old version, slower than einsum
        #TrLambda = _np.trace(_np.dot(a, b.conjugate().T))  # same as using _np.linalg.inv(b)
        
        #Use einsum black magic to only calculate the diagonal elements
        #if the basis is either pp or gm we know the elements are real-valued, so we
        #don't need to take the conjugate
        if mx_basis=='pp' or mx_basis=='gm':
            TrLambda = _np.einsum('ij,ji->',a, b.T)
        else:
            TrLambda = _np.einsum('ij,ji->',a, b.conjugate().T)
        return TrLambda / d2

    JA = _jam.jamiolkowski_iso(a, mx_basis, mx_basis)
    JB = _jam.jamiolkowski_iso(b, mx_basis, mx_basis)
    return fidelity(JA, JB)


def average_gate_fidelity(a, b, mx_basis='pp', is_tp=None, is_unitary=None):
    """
    Computes the average gate fidelity (AGF) between two gates.

    Average gate fidelity (`F_g`) is related to entanglement fidelity
    (`F_p`), via:

      `F_g = (d * F_p + 1)/(1 + d)`,

    where d is the Hilbert space dimension. This formula, and the
    definition of AGF, can be found in Phys. Lett. A 303 249-252 (2002).

    Parameters
    ----------
    a : array or gate
        The gate to compute the AGI to b of. E.g., an imperfect
        implementation of b.

    b : array or gate
        The gate to compute the AGI to a of. E.g., the target gate
        corresponding to a.

    mx_basis : {"std","gm","pp"} or Basis object, optional
        The basis of the matrices.
        
    is_tp : bool, optional (default None)
        Flag indicating both matrices are TP. If None (the default), 
        an explicit check is performed. If True/False, the check is 
        skipped and the provided value is used (faster, but should only 
        be used when the user is certain this is true apriori).

    is_unitary : bool, optional (default None)
        Flag indicating that the second matrix, b, is
        unitary. If None (the default) an explicit check is performed.
        If True/False, the check is skipped and the provided value is used
        (faster, but should only be used when the user is certain 
        this is true apriori).

    Returns
    -------
    AGI : float
        The AGI of a to b.
    """
    d = int(round(_np.sqrt(a.shape[0])))
    PF = entanglement_fidelity(a, b, mx_basis, is_tp, is_unitary)
    AGF = (d * PF + 1) / (1 + d)
    return float(AGF)


def average_gate_infidelity(a, b, mx_basis='pp', is_tp=None, is_unitary=None):
    """
    Computes the average gate infidelity (`AGI`) between two gates.

    Average gate infidelity is related to entanglement infidelity
    (`EI`) via:

      `AGI = (d * (1-EI) + 1)/(1 + d)`,

    where d is the Hilbert space dimension. This formula, and the
    definition of AGI, can be found in Phys. Lett. A 303 249-252 (2002).

    Parameters
    ----------
    a : array or gate
        The gate to compute the AGI to b of. E.g., an imperfect
        implementation of b.

    b : array or gate
        The gate to compute the AGI to a of. E.g., the target gate
        corresponding to a.

    mx_basis : {"std","gm","pp"} or Basis object, optional
        The basis of the matrices.
        
    is_tp : bool, optional (default None)
        Flag indicating both matrices are TP. If None (the default), 
        an explicit check is performed. If True/False, the check is 
        skipped and the provided value is used (faster, but should only 
        be used when the user is certain this is true apriori).

    is_unitary : bool, optional (default None)
        Flag indicating that the second matrix, b, is
        unitary. If None (the default) an explicit check is performed.
        If True/False, the check is skipped and the provided value is used
        (faster, but should only be used when the user is certain 
        this is true apriori).
       
    Returns
    -------
    float
    """
    return 1 - average_gate_fidelity(a, b, mx_basis, is_tp, is_unitary)


def entanglement_infidelity(a, b, mx_basis='pp', is_tp=None, is_unitary=None):
    """
    Returns the entanglement infidelity (EI) between gate matrices.

    This i given by:

      `EI = 1 - Tr( sqrt{ sqrt(J(a)) * J(b) * sqrt(J(a)) } )^2`

    where J(.) is the Jamiolkowski isomorphism map that maps a operation matrix
    to it's corresponding Choi Matrix.

    Parameters
    ----------
    a : numpy array
        First matrix.

    b : numpy array
        Second matrix.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The basis of the matrices.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt)
        (or a custom basis object).
        
    is_tp : bool, optional (default None)
        Flag indicating both matrices are TP. If None (the default), 
        an explicit check is performed. If True/False, the check is 
        skipped and the provided value is used (faster, but should only 
        be used when the user is certain this is true apriori).

    is_unitary : bool, optional (default None)
        Flag indicating that the second matrix, b, is
        unitary. If None (the default) an explicit check is performed.
        If True/False, the check is skipped and the provided value is used
        (faster, but should only be used when the user is certain 
        this is true apriori).

    Returns
    -------
    EI : float
        The EI of a to b.
    """
    return 1 - float(entanglement_fidelity(a, b, mx_basis, is_tp, is_unitary))


def gateset_infidelity(model, target_model, itype='EI',
                       weights=None, mx_basis=None, is_tp=None, is_unitary=None):
    """
    Computes the average-over-gates of the infidelity between gates in `model` and the gates in `target_model`.

    If `itype` is 'EI' then the "infidelity" is the entanglement infidelity; if
    `itype` is 'AGI' then the "infidelity" is the average gate infidelity (AGI
    and EI are related by a dimension dependent constant).

    This is the quantity that RB error rates are sometimes claimed to be
    related to directly related.

    Parameters
    ----------
    model : Model
        The model to calculate the average infidelity, to `target_model`, of.

    target_model : Model
        The model to calculate the average infidelity, to `model`, of.

    itype : str, optional
        The infidelity type. Either 'EI', corresponding to entanglement
        infidelity, or 'AGI', corresponding to average gate infidelity.

    weights : dict, optional
        If not None, a dictionary of floats, whereby the keys are the gates
        in `model` and the values are, possibly unnormalized, probabilities.
        These probabilities corresponding to the weighting in the average,
        so if the model contains gates A and B and weights[A] = 2 and
        weights[B] = 1 then the output is Inf(A)*2/3  + Inf(B)/3 where
        Inf(X) is the infidelity (to the corresponding element in the other
        model) of X. If None, a uniform-average is taken, equivalent to
        setting all the weights to 1.

    mx_basis : {"std","gm","pp"} or Basis object, optional
        The basis of the models. If None, the basis is obtained from
        the model.
        
    is_tp : bool, optional (default None)
        Flag indicating both matrices are TP. If None (the default), 
        an explicit check is performed. If True/False, the check is 
        skipped and the provided value is used (faster, but should only 
        be used when the user is certain this is true apriori).

    is_unitary : bool, optional (default None)
        Flag indicating that the second matrix, b, is
        unitary. If None (the default) an explicit check is performed.
        If True/False, the check is skipped and the provided value is used
        (faster, but should only be used when the user is certain 
        this is true apriori).

    Returns
    -------
    float
        The weighted average-over-gates infidelity between the two models.
    """
    assert(itype == 'AGI' or itype == 'EI'), \
        "The infidelity type must be `AGI` (average gate infidelity) or `EI` (entanglement infidelity)"

    if mx_basis is None: mx_basis = model.basis

    sum_of_weights = 0
    I_list = []
    for gate in list(target_model.operations.keys()):
        if itype == 'AGI':
            I = average_gate_infidelity(model.operations[gate], target_model.operations[gate], mx_basis, is_tp)
        if itype == 'EI':
            I = entanglement_infidelity(model.operations[gate], target_model.operations[gate], mx_basis, is_tp)
        if weights is None:
            w = 1
        else:
            w = weights[gate]

        I_list.append(w * I)
        sum_of_weights += w

    assert(sum_of_weights > 0), "The sum of the weights should be positive!"
    AI = _np.sum(I_list) / sum_of_weights

    return AI


def unitarity(a, mx_basis="gm"):
    """
    Returns the "unitarity" of a channel.

    Unitarity is defined as in Wallman et al, ``Estimating the Coherence of
    noise'' NJP 17 113020 (2015). The unitarity is given by (Prop 1 in Wallman
    et al):

    `u(a) = Tr( A_u^{\dagger} A_u ) / (d^2  - 1)`,

    where A_u is the unital submatrix of a, and d is the dimension of
    the Hilbert space. When a is written in any basis for which the
    first element is the  normalized identity (e.g., the pp or gm
    bases), The unital submatrix of a is the matrix obtained when the
    top row and left hand column is removed from a.

    Parameters
    ----------
    a : array or gate
        The gate for which the unitarity is to be computed.

    mx_basis : {"std","gm","pp"} or a Basis object, optional
        The basis of the matrix.

    Returns
    -------
    float
    """
    d = int(round(_np.sqrt(a.shape[0])))
    basisMxs = _bt.basis_matrices(mx_basis, a.shape[0])

    if _np.allclose(basisMxs[0], _np.identity(d, 'd')):
        B = a
    else:
        B = _bt.change_basis(a, mx_basis, "gm")  # everything should be able to be put in the "gm" basis

    unital = B[1:d**2, 1:d**2]
    #old version
    #u = _np.trace(_np.dot(_np.conj(_np.transpose(unital)), unital)) / (d**2 - 1)
    #new version
    u= _np.einsum('ij,ji->', unital.conjugate().T, unital ) / (d**2 - 1)
    return u


def fidelity_upper_bound(operation_mx):
    """
    Get an upper bound on the fidelity of the given operation matrix with any unitary operation matrix.

    The closeness of the result to one tells
     how "unitary" the action of operation_mx is.

    Parameters
    ----------
    operation_mx : numpy array
        The operation matrix to act on.

    Returns
    -------
    float
        The resulting upper bound on fidelity(operation_mx, anyUnitaryGateMx)
    """
    choi = _jam.jamiolkowski_iso(operation_mx, choi_mx_basis="std")
    choi_evals, choi_evecs = _np.linalg.eig(choi)
    maxF_direct = max([_np.sqrt(max(ev.real, 0.0)) for ev in choi_evals]) ** 2

    iMax = _np.argmax([ev.real for ev in choi_evals])  # index of maximum eigenval
    closestVec = choi_evecs[:, iMax:(iMax + 1)]

    # #print "DEBUG: closest evec = ", closestUnitaryVec
    # new_evals = _np.zeros( len(closestUnitaryVec) ); new_evals[iClosestU] = 1.0
    # # gives same result:
    # closestUnitaryJmx = _np.dot(choi_evecs, _np.dot( _np.diag(new_evals), _np.linalg.inv(choi_evecs) ) )
    closestJmx = _np.kron(closestVec, _np.transpose(_np.conjugate(closestVec)))  # closest rank-1 Jmx
    closestJmx /= _mt.trace(closestJmx)  # normalize so trace of Jmx == 1.0

    maxF = fidelity(choi, closestJmx)

    if not _np.isnan(maxF):

        #Uncomment for debugging
        #if abs(maxF - maxF_direct) >= 1e-6:
        #    print "DEBUG: operation_mx:\n",operation_mx
        #    print "DEBUG: choi_mx:\n",choi
        #    print "DEBUG choi_evals = ",choi_evals, " iMax = ",iMax
        #    #print "DEBUG: J = \n", closestUnitaryJmx
        #    print "DEBUG: eigvals(J) = ", _np.linalg.eigvals(closestJmx)
        #    print "DEBUG: trace(J) = ", _mt.trace(closestJmx)
        #    print "DEBUG: maxF = %f,  maxF_direct = %f" % (maxF, maxF_direct)
        #    raise ValueError("ERROR: maxF - maxF_direct = %f" % (maxF -maxF_direct))
        assert(abs(maxF - maxF_direct) < 1e-6)
    else:
        maxF = maxF_direct  # case when maxF is nan, due to scipy sqrtm function being buggy - just use direct F

    closestOpMx = _jam.jamiolkowski_iso_inv(closestJmx, choi_mx_basis="std")
    return maxF, closestOpMx

    #closestU_evals, closestU_evecs = _np.linalg.eig(closestUnitaryGateMx)
    #print "DEBUG: U = \n", closestUnitaryGateMx
    #print "DEBUG: closest U evals = ",closestU_evals
    #print "DEBUG:  evecs = \n",closestU_evecs


def compute_povm_map(model, povmlbl):
    """
    Constructs a gate-like quantity for the POVM within `model`.

    This is done by embedding the `k`-outcome classical output space of the POVM
    in the Hilbert-Schmidt space of `k` by `k` density matrices by placing the
    classical probability distribution along the diagonal of the density matrix.
    Currently, this is only implemented for the case when `k` equals `d`, the
    dimension of the POVM's Hilbert space.

    Parameters
    ----------
    model : Model
        The model supplying the POVM effect vectors and the basis those
        vectors are in.

    povmlbl : str
        The POVM label

    Returns
    -------
    numpy.ndarray
        The matrix of the "POVM map" in the `model.basis` basis.
    """
    povmVectors = [v.to_dense()[:, None] for v in model.povms[povmlbl].values()]
    if isinstance(model.basis, _DirectSumBasis):  # HACK - need to get this to work with general bases
        blkDims = [int(_np.sqrt(comp.dim)) for comp in model.basis.component_bases]
    else:
        blkDims = [int(round(_np.sqrt(model.dim)))]  # [d] where density matrix is dxd

    nV = len(povmVectors)
    #assert(d**2 == model.dim), "Model dimension (%d) is not a perfect square!" % model.dim
    #assert( nV**2 == d ), "Can only compute POVM metrics when num of effects == H space dimension"
    #   I don't think above assert is needed - should work in general (Robin?)
    povm_mx = _np.concatenate(povmVectors, axis=1).T  # "povm map" ( B(H) -> S_k ) (shape= nV,model.dim)

    Sk_embedding_in_std = _np.zeros((model.dim, nV))
    for i in range(nV):
        Sk_embedding_in_std[:, i] = _flat_mut_blks(i, i, blkDims)

    std_basis = _Basis.cast('std', model.dim)  # make sure std basis is just straight-up d-dimension
    std_to_basis = model.basis.reverse_transform_matrix(std_basis)
    # OLD: _bt.create_transform_matrix("std", model.basis, blkDims)
    assert(std_to_basis.shape == (model.dim, model.dim))

    return _np.dot(std_to_basis, _np.dot(Sk_embedding_in_std, povm_mx))


def povm_fidelity(model, target_model, povmlbl):
    """
    Computes the process (entanglement) fidelity between POVM maps.

    Parameters
    ----------
    model : Model
        The model the POVM belongs to.

    target_model : Model
        The target model (which also has `povmlbl`).

    povmlbl : Label
        Label of the POVM to get the fidelity of.

    Returns
    -------
    float
    """
    povm_mx = compute_povm_map(model, povmlbl)
    target_povm_mx = compute_povm_map(target_model, povmlbl)
    return entanglement_fidelity(povm_mx, target_povm_mx, target_model.basis)


def povm_jtracedist(model, target_model, povmlbl):
    """
    Computes the Jamiolkowski trace distance between POVM maps using :func:`jtracedist`.

    Parameters
    ----------
    model : Model
        The model the POVM belongs to.

    target_model : Model
        The target model (which also has `povmlbl`).

    povmlbl : Label
        Label of the POVM to get the trace distance of.

    Returns
    -------
    float
    """
    povm_mx = compute_povm_map(model, povmlbl)
    target_povm_mx = compute_povm_map(target_model, povmlbl)
    return jtracedist(povm_mx, target_povm_mx, target_model.basis)


def povm_diamonddist(model, target_model, povmlbl):
    """
    Computes the diamond distance between POVM maps using :func:`diamonddist`.

    Parameters
    ----------
    model : Model
        The model the POVM belongs to.

    target_model : Model
        The target model (which also has `povmlbl`).

    povmlbl : Label
        Label of the POVM to get the diamond distance of.

    Returns
    -------
    float
    """
    povm_mx = compute_povm_map(model, povmlbl)
    target_povm_mx = compute_povm_map(target_model, povmlbl)
    return diamonddist(povm_mx, target_povm_mx, target_model.basis)


#decompose operation matrix into axis of rotation, etc
def decompose_gate_matrix(operation_mx):
    """
    Decompse a gate matrix into fixed points, axes of rotation, angles of rotation, and decay rates.

    This funtion computes how the action of a operation matrix can be is
    decomposed into fixed points, axes of rotation, angles of rotation, and
    decays.  Also determines whether a gate appears to be valid and/or unitary.

    Parameters
    ----------
    operation_mx : numpy array
        The operation matrix to act on.

    Returns
    -------
    dict
        A dictionary describing the decomposed action. Keys are:

          'isValid' : bool
              whether decomposition succeeded
          'isUnitary' : bool
              whether operation_mx describes unitary action
          'fixed point' : numpy array
              the fixed point of the action
          'axis of rotation' : numpy array or nan
              the axis of rotation
          'decay of diagonal rotation terms' : float
              decay of diagonal terms
          'rotating axis 1' : numpy array or nan
              1st axis orthogonal to axis of rotation
          'rotating axis 2' : numpy array or nan
              2nd axis orthogonal to axis of rotation
          'decay of off diagonal rotation terms' : float
              decay of off-diagonal terms
          'pi rotations' : float
              angle of rotation in units of pi radians
    """

    op_evals, op_evecs = _np.linalg.eig(_np.asarray(operation_mx))
    # fp_eigenvec = None
    # aor_eval = None; aor_eigenvec = None
    # ra_eval  = None; ra1_eigenvec = None; ra2_eigenvec = None

    TOL = 1e-4  # 1e-7

    unit_eval_indices = [i for (i, ev) in enumerate(op_evals) if abs(ev - 1.0) < TOL]
    #unit_eval_indices = [ i for (i,ev) in enumerate(op_evals) if ev > (1.0-TOL) ]

    conjpair_eval_indices = []
    for (i, ev) in enumerate(op_evals):
        if i in unit_eval_indices: continue  # don't include the unit eigenvalues in the conjugate pair count
        # don't include existing conjugate pairs
        if any([(i in conjpair) for conjpair in conjpair_eval_indices]): continue
        for (j, ev2) in enumerate(op_evals[i + 1:]):
            if abs(ev - _np.conjugate(ev2)) < TOL:
                conjpair_eval_indices.append((i, j + (i + 1)))
                break  # don't pair i-th eigenvalue with any other (pairs should be disjoint)

    real_eval_indices = []  # indices of real eigenvalues that are not units or a part of any conjugate pair
    complex_eval_indices = []  # indices of complex eigenvalues that are not units or a part of any conjugate pair
    for (i, ev) in enumerate(op_evals):
        if i in unit_eval_indices: continue  # don't include the unit eigenvalues
        if any([(i in conjpair) for conjpair in conjpair_eval_indices]): continue  # don't include the conjugate pairs
        if abs(ev.imag) < TOL: real_eval_indices.append(i)
        else: complex_eval_indices.append(i)

    #if len(real_eval_indices + unit_eval_indices) > 0:
    #    max_real_eval = max([ op_evals[i] for i in real_eval_indices + unit_eval_indices])
    #    min_real_eval = min([ op_evals[i] for i in real_eval_indices + unit_eval_indices])
    #else:
    #    max_real_eval = _np.nan
    #    min_real_eval = _np.nan
    #
    #fixed_points = [ op_evecs[:,i] for i in unit_eval_indices ]
    #real_eval_axes = [ op_evecs[:,i] for i in real_eval_indices ]
    #conjpair_eval_axes = [ (op_evecs[:,i],op_evecs[:,j]) for (i,j) in conjpair_eval_indices ]
    #
    #ret = { }

    nQubits = _np.log2(operation_mx.shape[0]) / 2
    if nQubits == 1:
        #print "DEBUG: 1 qubit decomp --------------------------"
        #print "   --> evals = ", op_evals
        #print "   --> unit eval indices = ", unit_eval_indices
        #print "   --> conj eval indices = ", conjpair_eval_indices
        #print "   --> unpaired real eval indices = ", real_eval_indices

        #Special case: if have two conjugate pairs, check if one (or both) are real
        #  and break the one with the largest (real) value into two unpaired real evals.
        if len(conjpair_eval_indices) == 2:
            iToBreak = None
            if abs(_np.imag(op_evals[conjpair_eval_indices[0][0]])) < TOL and \
               abs(_np.imag(op_evals[conjpair_eval_indices[1][0]])) < TOL:
                iToBreak = _np.argmax([_np.real(conjpair_eval_indices[0][0]), _np.real(conjpair_eval_indices[1][0])])
            elif abs(_np.imag(op_evals[conjpair_eval_indices[0][0]])) < TOL: iToBreak = 0
            elif abs(_np.imag(op_evals[conjpair_eval_indices[1][0]])) < TOL: iToBreak = 1

            if iToBreak is not None:
                real_eval_indices.append(conjpair_eval_indices[iToBreak][0])
                real_eval_indices.append(conjpair_eval_indices[iToBreak][1])
                del conjpair_eval_indices[iToBreak]

        #Find eigenvector corresponding to fixed point (or closest we can get).   This
        # should be a unit eigenvalue with identity eigenvector.
        if len(unit_eval_indices) > 0:
            #Find linear least squares solution within possibly degenerate unit-eigenvalue eigenspace
            # of eigenvector closest to identity density mx (the desired fixed point), then orthogonalize
            # the remaining eigenvectors w.r.t this one.
            A = _np.take(op_evecs, unit_eval_indices, axis=1)
            b = _np.array([[1], [0], [0], [0]], 'd')  # identity density mx
            x = _np.dot(_np.linalg.pinv(_np.dot(A.T, A)), _np.dot(A.T, b))
            fixedPtVec = _np.dot(A, x)  # fixedPtVec / _np.linalg.norm(fixedPtVec)
            fixedPtVec = fixedPtVec[:, 0]

            iLargestContrib = _np.argmax(_np.abs(x))  # index of gate eigenvector which contributed the most
            for ii, i in enumerate(unit_eval_indices):
                if ii == iLargestContrib:
                    op_evecs[:, i] = fixedPtVec
                    iFixedPt = i
                else:
                    op_evecs[:, i] = op_evecs[:, i] - _np.vdot(fixedPtVec, op_evecs[:, i]) * fixedPtVec
                    for jj, j in enumerate(unit_eval_indices[:ii]):
                        if jj == iLargestContrib: continue
                        op_evecs[:, i] = op_evecs[:, i] - _np.vdot(op_evecs[:, j], op_evecs[:, i]) * op_evecs[:, j]
                    op_evecs[:, i] /= _np.linalg.norm(op_evecs[:, i])

        elif len(real_eval_indices) > 0:
            # just take eigenvector corresponding to the largest real eigenvalue?
            #iFixedPt = real_eval_indices[ _np.argmax( [ op_evals[i] for i in real_eval_indices ] ) ]

            # ...OR take eigenvector corresponding to a real unpaired eigenvalue closest to identity:
            idmx = _np.array([[1], [0], [0], [0]], 'd')  # identity density mx
            iFixedPt = real_eval_indices[_np.argmin([_np.linalg.norm(op_evecs[i] - idmx) for i in real_eval_indices])]

        else:
            #No unit or real eigenvalues => two complex conjugate pairs or unpaired complex evals --> bail out
            return {'isValid': False, 'isUnitary': False, 'msg': "All evals are complex."}

        #Find eigenvector corresponding to axis of rotation: find the *largest* unpaired real/unit eval
        indsToConsider = (unit_eval_indices + real_eval_indices)[:]
        del indsToConsider[indsToConsider.index(iFixedPt)]  # don't consider fixed pt evec

        if len(indsToConsider) > 0:
            iRotAxis = indsToConsider[_np.argmax([op_evals[i] for i in indsToConsider])]
        else:
            #No unit or real eigenvalues => an unpaired complex eval --> bail out
            return {'isValid': False, 'isUnitary': False, 'msg': "Unpaired complex eval."}

        #There are only 2 eigenvalues left -- hopefully a conjugate pair giving rotation
        inds = list(range(4))
        del inds[inds.index(iFixedPt)]
        del inds[inds.index(iRotAxis)]
        if abs(op_evals[inds[0]] - _np.conjugate(op_evals[inds[1]])) < TOL:
            iConjPair1, iConjPair2 = inds
        else:
            return {'isValid': False, 'isUnitary': False, 'msg': "No conjugate pair for rotn."}

        return {'isValid': True,
                'isUnitary': bool(len(unit_eval_indices) >= 2),
                'fixed point': op_evecs[:, iFixedPt],
                'axis of rotation': op_evecs[:, iRotAxis],
                'rotating axis 1': op_evecs[:, iConjPair1],
                'rotating axis 2': op_evecs[:, iConjPair2],
                'decay of diagonal rotation terms': 1.0 - abs(op_evals[iRotAxis]),
                'decay of off diagonal rotation terms': 1.0 - abs(op_evals[iConjPair1]),
                'pi rotations': _np.angle(op_evals[iConjPair1]) / _np.pi,
                'msg': "Success"}

    else:
        return {'isValid': False,
                'isUnitary': False,
                'msg': "Unsupported number of qubits: %d" % nQubits}


def state_to_dmvec(psi):
    """
    Compute the vectorized density matrix which acts as the state `psi`.

    This is just the outer product map |psi> => |psi><psi| with the
    output flattened, i.e. `dot(psi, conjugate(psi).T)`.

    Parameters
    ----------
    psi : numpy array
        The state vector.

    Returns
    -------
    numpy array
        The vectorized density matrix.
    """
    psi = psi.reshape((psi.size, 1))  # convert to (N,1) shape if necessary
    dm = _np.dot(psi, _np.conjugate(psi.T))
    return dm.flatten()


def dmvec_to_state(dmvec, tol=1e-6):
    """
    Compute the pure state describing the action of density matrix vector `dmvec`.

    If `dmvec` represents a mixed state, ValueError is raised.

    Parameters
    ----------
    dmvec : numpy array
        The vectorized density matrix, assumed to be in the standard (matrix
        unit) basis.

    tol : float, optional
        tolerance for determining whether an eigenvalue is zero.

    Returns
    -------
    numpy array
        The pure state, as a column vector of shape = (N,1)
    """
    d2 = dmvec.size; d = int(round(_np.sqrt(d2)))
    dm = dmvec.reshape((d, d))
    evals, evecs = _np.linalg.eig(dm)

    k = None
    for i, ev in enumerate(evals):
        if abs(ev) > tol:
            if k is None: k = i
            else: raise ValueError("Cannot convert mixed dmvec to pure state!")
    if k is None: raise ValueError("Cannot convert zero dmvec to pure state!")
    psi = evecs[:, k] * _np.sqrt(evals[k])
    psi.shape = (d, 1)
    return psi


def unitary_to_superop(u, superop_mx_basis='pp'):
    """ TODO: docstring """
    return _bt.change_basis(unitary_to_std_process_mx(u), 'std', superop_mx_basis)


@_deprecated_fn('pygsti.tools.unitary_to_std_process_mx(...) or unitary_to_superop(...)')
def unitary_to_process_mx(u):
    return unitary_to_std_process_mx(u)


def unitary_to_std_process_mx(u):
    """
    Compute the superoperator corresponding to unitary matrix `u`.

    Computes a super-operator (that acts on (row)-vectorized density matrices)
    from a unitary operator (matrix) `u` which acts on state vectors.  This
    super-operator is given by the tensor product of `u` and `conjugate(u)`,
    i.e. `kron(u,u.conj)`.

    Parameters
    ----------
    u : numpy array
        The unitary matrix which acts on state vectors.

    Returns
    -------
    numpy array
        The super-operator process matrix.
    """
    # u -> kron(u,Uc) since u rho U_dag -> kron(u,Uc)
    #  since AXB --row-vectorize--> kron(A,B.T)*vec(X)
    return _np.kron(u, _np.conjugate(u))


def superop_is_unitary(superop_mx, mx_basis='pp', rank_tol=1e-6):
    """ TODO: docstring """
    J = _jam.fast_jamiolkowski_iso_std(superop_mx, op_mx_basis=mx_basis)  # (Choi mx basis doesn't matter)
    return bool(_np.linalg.matrix_rank(J, rank_tol) == 1)


def superop_to_unitary(superop_mx, mx_basis='pp', check_superop_is_unitary=True):
    """ TODO: docstring"""
    if check_superop_is_unitary and not superop_is_unitary(superop_mx, mx_basis):
        raise ValueError("Superoperator matrix does not perform a unitary action!")
    return std_process_mx_to_unitary(_bt.change_basis(superop_mx, mx_basis, 'std'))


@_deprecated_fn('pygsti.tools.std_process_mx_to_unitary(...) or superop_to_unitary(...)')
def process_mx_to_unitary(superop):
    return std_process_mx_to_unitary(superop)


def std_process_mx_to_unitary(superop_mx):
    """
    Compute the unitary corresponding to the (unitary-action!) super-operator `superop`.

    This function assumes `superop` acts on (row)-vectorized
    density matrices, and that the super-operator is of the form
    `kron(U,U.conj)`.

    Parameters
    ----------
    superop : numpy array
        The superoperator matrix which acts on vectorized
        density matrices (in the 'std' matrix-unit basis).

    Returns
    -------
    numpy array
        The unitary matrix which acts on state vectors.
    """
    d2 = superop_mx.shape[0]; d = int(round(_np.sqrt(d2)))
    U = _np.empty((d, d), 'complex')

    for i in range(d):
        densitymx_i = _np.zeros((d, d), 'd'); densitymx_i[i, i] = 1.0  # |i><i|
        UiiU = _np.dot(superop_mx, densitymx_i.flat).reshape((d, d))  # U|i><i|U^dag

        if i > 0:
            j = 0
            densitymx_ij = _np.zeros((d, d), 'd'); densitymx_ij[i, j] = 1.0  # |i><j|
            UijU = _np.dot(superop_mx, densitymx_ij.flat).reshape((d, d))  # U|i><j|U^dag
            Uj = U[:, j]
            Ui = _np.dot(UijU, Uj)
        else:
            ##method1: use random state projection
            #rand_state = _np.random.rand(d)
            #projected_rand_state = _np.dot(UiiU, rand_state)
            #assert(_np.linalg.norm(projected_rand_state) > 1e-8)
            #projected_rand_state /= _np.linalg.norm(projected_rand_state)
            #Ui = projected_rand_state

            #method2: get eigenvector corresponding to largest eigenvalue (more robust)
            evals, evecs = _np.linalg.eig(UiiU)
            imaxeval = _np.argmax(_np.abs(evals))
            #Check that other eigenvalue are small? (not sure if this is sufficient condition though...)
            #assert(all([abs(ev / evals[imaxeval]) < 1e-6 for i, ev in enumerate(evals) if i != imaxeval])), \
            #    "Superoperator matrix does not perform a unitary action!"
            Ui = evecs[:, imaxeval]
            Ui /= _np.linalg.norm(Ui)
        U[:, i] = Ui

    return U


def spam_error_generator(spamvec, target_spamvec, mx_basis, typ="logGTi"):
    """
    Construct an error generator from a SPAM vector and it's target.

    Computes the value of the error generator given by
    `errgen = log( diag(spamvec / target_spamvec) )`, where division is
    element-wise.  This results in a (non-unique) error generator matrix
    `E` such that `spamvec = exp(E) * target_spamvec`.

    Note: This is currently of very limited use, as the above algorithm fails
    whenever `target_spamvec` has zero elements where `spamvec` doesn't.

    Parameters
    ----------
    spamvec : ndarray
        The SPAM vector.

    target_spamvec : ndarray
        The target SPAM vector.

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    typ : {"logGTi"}
        The type of error generator to compute.  Allowed values are:

        - "logGTi" : errgen = log( diag(spamvec / target_spamvec) )

    Returns
    -------
    errgen : ndarray
        The error generator.
    """
    # Compute error generator for rho:   rho = exp(E)rho0 => rho = A*rho0 => A = diag(rho/rho0)
    assert(typ == "logGTi"), "Only logGTi type is supported so far"

    d2 = len(spamvec)
    errgen = _np.zeros((d2, d2), 'd')  # type assumes this is density-mx evolution
    diags = []
    for a, b in zip(spamvec, target_spamvec):
        if _np.isclose(b, 0.0):
            if _np.isclose(a, b): d = 1
            else: raise ValueError("Cannot take spam_error_generator")
        else:
            d = a / b
        diags.append(d)
    errgen[_np.diag_indices(d2)] = diags
    return _spl.logm(errgen)


def error_generator(gate, target_op, mx_basis, typ="logG-logT", logG_weight=None):
    """
    Construct the error generator from a gate and its target.

    Computes the value of the error generator given by
    errgen = log( inv(target_op) * gate ), so that
    gate = target_op * exp(errgen).

    Parameters
    ----------
    gate : ndarray
        The operation matrix

    target_op : ndarray
        The target operation matrix

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    typ : {"logG-logT", "logTiG", "logGTi"}
        The type of error generator to compute.  Allowed values are:

        - "logG-logT" : errgen = log(gate) - log(target_op)
        - "logTiG" : errgen = log( dot(inv(target_op), gate) )
        - "logGTi" : errgen = log( dot(gate,inv(target_op)) )

    logG_weight: float or None (default)
        Regularization weight for logG-logT penalty of approximate logG.
        If None, the default weight in :func:`approximate_matrix_log` is used.
        Note that this will result in a logG close to logT, but G may not exactly equal exp(logG).
        If self-consistency with func:`operation_from_error_generator` is desired,
        consider testing lower (or zero) regularization weight.

    Returns
    -------
    errgen : ndarray
        The error generator.
    """
    TOL = 1e-8

    if typ == "logG-logT":
        try:
            logT = _mt.unitary_superoperator_matrix_log(target_op, mx_basis)
        except AssertionError:  # if not unitary, fall back to just taking the real log
            logT = _mt.real_matrix_log(target_op, "raise", TOL)  # make a fuss if this can't be done

        if logG_weight is not None:
            logG = _mt.approximate_matrix_log(gate, logT, target_weight=logG_weight)
        else:
            logG = _mt.approximate_matrix_log(gate, logT)

        # Both logG and logT *should* be real, so we just take the difference.
        if _np.linalg.norm(_np.imag(logG)) < TOL and \
           _np.linalg.norm(_np.imag(logT)) < TOL:
            return _np.real(logG - logT)

        #Otherwise, there could be branch cut issues or worse, so just
        # raise an error for now (maybe return a dummy if needed elsewhere?)
        raise ValueError("Could not construct a real logarithms for the "
                         "'logG-logT' generator.  Perhaps you should use "
                         "the 'logTiG' or 'logGTi' generator instead?")

    elif typ == "logTiG":
        target_op_inv = _spl.inv(target_op)
        try:
            errgen = _mt.near_identity_matrix_log(_np.dot(target_op_inv, gate), TOL)
        except AssertionError:  # not near the identity, fall back to the real log
            _warnings.warn(("Near-identity matrix log failed; falling back "
                            "to real matrix log for logTiG error generator"))
            errgen = _mt.real_matrix_log(_np.dot(target_op_inv, gate), "warn", TOL)

        if _np.linalg.norm(errgen.imag) > TOL:
            _warnings.warn("Falling back to approximate log for logTiG error generator")
            errgen = _mt.approximate_matrix_log(_np.dot(target_op_inv, gate),
                                                _np.zeros(gate.shape, 'd'), tol=TOL)

    elif typ == "logGTi":
        target_op_inv = _spl.inv(target_op)
        try:
            errgen = _mt.near_identity_matrix_log(_np.dot(gate, target_op_inv), TOL)
        except AssertionError as e:  # not near the identity, fall back to the real log
            _warnings.warn(("Near-identity matrix log failed; falling back "
                            "to real matrix log for logGTi error generator:\n%s") % str(e))
            errgen = _mt.real_matrix_log(_np.dot(gate, target_op_inv), "warn", TOL)

        if _np.linalg.norm(errgen.imag) > TOL:
            _warnings.warn("Falling back to approximate log for logGTi error generator")
            errgen = _mt.approximate_matrix_log(_np.dot(gate, target_op_inv),
                                                _np.zeros(gate.shape, 'd'), tol=TOL)

    elif typ == "logGTi-quick":
        #errgen = _spl.logm(_np.dot(gate, _spl.inv(target_op)))
        return _np.real(_spl.logm(_np.dot(gate, _spl.inv(target_op))))

    else:
        raise ValueError("Invalid error-generator type: %s" % typ)

    if _np.linalg.norm(_np.imag(errgen)) > TOL:
        raise ValueError("Could not construct a real generator!")
        #maybe this is actually ok, but a complex error generator will
        # need to be plotted differently, etc -- TODO
    return _np.real(errgen)


def operation_from_error_generator(error_gen, target_op, mx_basis, typ="logG-logT"):
    """
    Construct a gate from an error generator and a target gate.

    Inverts the computation done in :func:`error_generator` and
    returns the value of the gate given by
    gate = target_op * exp(error_gen).

    Parameters
    ----------
    error_gen : ndarray
        The error generator matrix

    target_op : ndarray
        The target operation matrix

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    typ : {"logG-logT", "logG-logT-quick", "logTiG", "logGTi"}
        The type of error generator to invert.  Allowed values are:

        - "logG-logT" : gate = exp( errgen + log(target_op) ) using internal logm
        - "logG-logT-quick" : gate = exp( errgen + log(target_op) ) using SciPy logm
        - "logTiG" : gate = dot( target_op, exp(errgen) )
        - "logGTi" : gate = dot( exp(errgen), target_op )

    Returns
    -------
    ndarray
        The operation matrix.
    """
    TOL = 1e-8

    if typ == "logG-logT":
        try:
            logT = _mt.unitary_superoperator_matrix_log(target_op, mx_basis)
        except AssertionError:
            logT = _mt.real_matrix_log(target_op, "raise", TOL)

        return _spl.expm(error_gen + logT)
    elif typ == "logG-logT-quick":
        return _spl.expm(error_gen + _spl.logm(target_op))
    elif typ == "logTiG":
        return _np.dot(target_op, _spl.expm(error_gen))
    elif typ == "logGTi":
        return _np.dot(_spl.expm(error_gen), target_op)
    else:
        raise ValueError("Invalid error-generator type: %s" % typ)


def elementary_errorgens(dim, typ, basis):
    """
    Compute the elementary error generators of a certain type.

    Parameters
    ----------
    dim : int
        The dimension of the error generators to be returned.  This is also the
        associated gate dimension, and must be a perfect square, as `sqrt(dim)`
        is the dimension of density matrices. For a single qubit, dim == 4.

    typ : {'H', 'S', 'C', 'A'}
        The type of error generators to construct.

    basis : Basis or str
        Which basis is used to construct the error generators.  Note that this
        is not the basis *of* the returned error generators (which is always
        the `'std'` matrix-unit basis) but that used to define the different
        elementary generator operations themselves.

    Returns
    -------
    generators : numpy.ndarray
        An array of shape (#basis-elements,dim,dim).  `generators[i]` is the
        generator corresponding to the ith basis matrix in the
        *std* (matrix unit) basis.  (Note that in most cases #basis-elements
        == dim, so the size of `generators` is (dim,dim,dim) ).  Each
        generator is normalized so that as a vector it has unit Frobenius norm.
    """
    d2 = dim
    d = int(_np.sqrt(d2))
    assert(_np.isclose(d * d, d2))  # d2 must be a perfect square
    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"

    #Get a list of the basis matrices
    basis = _Basis.cast(basis, d2)
    basis_lbls = basis.labels[1:]  # skip identity
    basis_mxs = basis.elements[1:]  # skip identity
    assert(basis.first_element_is_identity), "First element of basis must be the identity!"
    assert(len(basis_mxs) < d2)  # OK if there are fewer basis matrices (e.g. for bases w/multiple blocks)

    elem_errgens = {}
    if typ in 'HS':
        for lbl, mx in zip(basis_lbls, basis_mxs):
            key = _LocalElementaryErrorgenLabel(typ, (lbl,))
            elem_errgens[key] = _lt.create_elementary_errorgen(typ, mx)
    else:  # typ in 'CA'
        for i, (lblA, mxA) in enumerate(zip(basis_lbls, basis_mxs)):
            for lblB, mxB in zip(basis_lbls[i + 1:], basis_mxs[i + 1:]):
                key = _LocalElementaryErrorgenLabel(typ, (lblA, lblB))
                elem_errgens[key] = _lt.create_elementary_errorgen(typ, mxA, mxB)

    return elem_errgens


def elementary_errorgens_dual(dim, typ, basis):
    """
    Compute the set of dual-to-elementary error generators of a given type.

    These error generators are dual to the elementary error generators
    constructed by :func:`elementary_errorgens`.

    Parameters
    ----------
    dim : int
        The dimension of the error generators to be returned.  This is also the
        associated gate dimension, and must be a perfect square, as `sqrt(dim)`
        is the dimension of density matrices. For a single qubit, dim == 4.

    typ : {'H', 'S', 'C', 'A'}
        The type of error generators to construct.

    basis : Basis or str
        Which basis is used to construct the error generators.  Note that this
        is not the basis *of* the returned error generators (which is always
        the `'std'` matrix-unit basis) but that used to define the different
        elementary generator operations themselves.

    Returns
    -------
    generators : numpy.ndarray
        An array of shape (#basis-elements,dim,dim).  `generators[i]` is the
        generator corresponding to the ith basis matrix in the
        *std* (matrix unit) basis.  (Note that in most cases #basis-elements
        == dim, so the size of `generators` is (dim,dim,dim) ).  Each
        generator is normalized so that as a vector it has unit Frobenius norm.
    """
    d2 = dim
    d = int(_np.sqrt(d2))
    assert(_np.isclose(d * d, d2))  # d2 must be a perfect square
    assert(typ in ('H', 'S', 'C', 'A')), "`typ` must be one of 'H', 'S', 'C', or 'A'"

    #Get a list of the basis matrices
    basis = _Basis.cast(basis, d2)
    basis_lbls = basis.labels[1:]  # skip identity
    basis_mxs = basis.elements[1:]  # skip identity
    assert(basis.first_element_is_identity), "First element of basis must be the identity!"
    assert(len(basis_mxs) < d2)  # OK if there are fewer basis matrices (e.g. for bases w/multiple blocks)

    elem_errgens = {}
    if typ in 'HS':
        for lbl, mx in zip(basis_lbls, basis_mxs):
            key = _LocalElementaryErrorgenLabel(typ, (lbl,))
            elem_errgens[key] = _lt.create_elementary_errorgen_dual(typ, mx)
    else:  # typ in 'CA'
        for i, (lblA, mxA) in enumerate(zip(basis_lbls, basis_mxs)):
            for lblB, mxB in zip(basis_lbls[i + 1:], basis_mxs[i + 1:]):
                key = _LocalElementaryErrorgenLabel(typ, (lblA, lblB))
                elem_errgens[key] = _lt.create_elementary_errorgen_dual(typ, mxA, mxB)

    return elem_errgens


def extract_elementary_errorgen_coefficients(errorgen, elementary_errorgen_labels, elementary_errorgen_basis='pp',
                                             errorgen_basis='pp', return_projected_errorgen=False):
    """ TODO: docstring """
    # the same as decompose_errorgen but given a dict/list of elementary errorgens directly instead of a basis and type
    if isinstance(errorgen_basis, _Basis):
        errorgen_std = _bt.change_basis(errorgen, errorgen_basis, errorgen_basis.create_equivalent('std'))

        #expand operation matrix so it acts on entire space of dmDim x dmDim density matrices
        errorgen_std = _bt.resize_std_mx(errorgen_std, 'expand', errorgen_basis.create_equivalent('std'),
                                         errorgen_basis.create_simple_equivalent('std'))
    else:
        errorgen_std = _bt.change_basis(errorgen, errorgen_basis, "std")
    flat_errorgen_std = errorgen_std.toarray().flatten() if _sps.issparse(errorgen_std) else errorgen_std.flatten()

    d2 = errorgen_std.shape[0]
    d = int(_np.sqrt(d2))
    assert(_np.isclose(d * d, d2))  # d2 must be a perfect square

    elementary_errorgen_basis = _Basis.cast(elementary_errorgen_basis, d2)
    projections = {}
    if return_projected_errorgen:
        space_projector = _np.empty((d2 * d2, len(elementary_errorgen_labels)), complex)

    for i, eeg_lbl in enumerate(elementary_errorgen_labels):
        key = _LocalElementaryErrorgenLabel.cast(eeg_lbl)
        bel_lbls = key.basis_element_labels
        bmx0 = elementary_errorgen_basis[bel_lbls[0]]
        bmx1 = elementary_errorgen_basis[bel_lbls[1]] if (len(bel_lbls) > 1) else None
        flat_projector = _lt.create_elementary_errorgen_dual(key.errorgen_type, bmx0, bmx1, sparse=False).flatten()
        projections[key] = _np.real_if_close(_np.vdot(flat_projector, flat_errorgen_std), tol=1000)
        if return_projected_errorgen:
            space_projector[:, i] = flat_projector

        if not _np.isreal(projections[key]):
            _warnings.warn("Taking abs() of non-real projection for %s: %s" % (str(eeg_lbl), str(projections[key])))
            projections[key] = abs(projections[key])

    if return_projected_errorgen:
        flat_projected_errorgen_std = (space_projector @ _np.linalg.pinv(space_projector)) @ flat_errorgen_std
        projected_errorgen = _bt.change_basis(flat_projected_errorgen_std.reshape((d2, d2)), "std", errorgen_basis)
        return projections, projected_errorgen
    else:
        return projections


def project_errorgen(errorgen, elementary_errorgen_type, elementary_errorgen_basis,
                     errorgen_basis="pp", return_dual_elementary_errorgens=False,
                     return_projected_errorgen=False):
    """
    Compute the projections of a gate error generator onto a set of elementary error generators.
    TODO: docstring update

    This standard set of errors is given by `projection_type`, and is constructed
    from the elements of the `projection_basis` basis.

    Parameters
    ----------
    errorgen : : ndarray
        The error generator matrix to project.

    projection_type : {"hamiltonian", "stochastic", "affine"}
        The type of error generators to project the gate error generator onto.
        If "hamiltonian", then use the Hamiltonian generators which take a density
        matrix rho -> -i*[ H, rho ] for Pauli-product matrix H.  If "stochastic",
        then use the Stochastic error generators which take rho -> P*rho*P for
        Pauli-product matrix P (recall P is self adjoint).  If "affine", then
        use the affine error generators which take rho -> P (superop is |P>><<1|).

    projection_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    return_generators : bool, optional
        If True, return the error generators projected against along with the
        projection values themseves.

    return_scale_fctr : bool, optional
        If True, also return the scaling factor that was used to multply the
        projections onto *normalized* error generators to get the returned
        values.

    Returns
    -------
    projections : numpy.ndarray
        An array of length equal to the number of elements in the
        basis used to construct the projectors.  Typically this is
        is also the dimension of the gate (e.g. 4 for a single qubit).
    generators : numpy.ndarray
        Only returned when `return_generators == True`.  An array of shape
        (#basis-els,op_dim,op_dim) such that  `generators[i]` is the
        generator corresponding to the i-th basis element.  Note
        that these matricies are in the *std* (matrix unit) basis.
    """

    if isinstance(errorgen_basis, _Basis):
        errorgen_std = _bt.change_basis(errorgen, errorgen_basis, errorgen_basis.create_equivalent('std'))

        #expand operation matrix so it acts on entire space of dmDim x dmDim density matrices
        errorgen_std = _bt.resize_std_mx(errorgen_std, 'expand', errorgen_basis.create_equivalent('std'),
                                         errorgen_basis.create_simple_equivalent('std'))
    else:
        errorgen_std = _bt.change_basis(errorgen, errorgen_basis, "std")
    flat_errorgen_std = errorgen_std.toarray().flatten() if _sps.issparse(errorgen_std) else errorgen_std.flatten()

    d2 = errorgen_std.shape[0]
    d = int(_np.sqrt(d2))
    # nQubits = _np.log2(d)
    assert(_np.isclose(d * d, d2))  # d2 must be a perfect square

    projectors = elementary_errorgens_dual(d2, elementary_errorgen_type, elementary_errorgen_basis)
    projections = {}

    if return_projected_errorgen:
        space_projector = _np.empty((d2 * d2, len(projectors)), complex)

    for i, (lbl, projector) in enumerate(projectors.items()):
        flat_projector = projector.flatten()
        proj = _np.real_if_close(_np.vdot(flat_projector, flat_errorgen_std), tol=1000)
        if return_projected_errorgen:
            space_projector[:, i] = flat_projector

        # # DEBUG - for checking why perfect gates gave weird projections --> log ambiguity
        # print("DB: rawproj(%d) = " % i, proj)
        # errorgen_pp = errorgen.copy() #_bt.change_basis(errorgen_std,"std","pp")
        # lindbladMx_pp = _bt.change_basis(lindbladMx,"std","pp")
        # if proj > 1.0:
        #    for k in range(errorgen_std.shape[0]):
        #        for j in range(errorgen_std.shape[1]):
        #            if abs(errorgen_pp[k,j].conjugate() * lindbladMx_pp[k,j]) > 1e-2:
        #                print(" [%d,%d]: + " % (k,j), errorgen_pp[k,j].conjugate(),
        #                      "*", lindbladMx_pp[k,j],
        #                      "=", (errorgen_pp[k,j].conjugate() * lindbladMx_pp[i,j]))

        #assert(_np.isreal(proj)), "non-real projection: %s" % str(proj) #just a warning now
        if not _np.isreal(proj):
            _warnings.warn("Taking abs() of non-real projection for %s: %s" % (str(lbl), str(proj)))
            proj = abs(proj)
        projections[lbl] = proj

    if return_projected_errorgen:
        flat_projected_errorgen_std = (space_projector @ _np.linalg.pinv(space_projector)) @ flat_errorgen_std
        projected_errorgen = _bt.change_basis(flat_projected_errorgen_std.reshape((d2, d2)), "std", errorgen_basis)

        if return_dual_elementary_errorgens:
            return projections, projectors, projected_errorgen
        else:
            return projections, projected_errorgen
    else:
        if return_dual_elementary_errorgens:
            return projections, projectors
        else:
            return projections


def _assert_shape(ar, shape, sparse=False):
    """ Asserts ar.shape == shape ; works with sparse matrices too """
    if not sparse or len(shape) == 2:
        assert(ar.shape == shape), \
            "Shape mismatch: %s != %s!" % (str(ar.shape), str(shape))
    else:
        if len(shape) == 3:  # first "dim" is a list
            assert(len(ar) == shape[0]), \
                "Leading dim mismatch: %d != %d!" % (len(ar), shape[0])
            assert(shape[0] == 0 or ar[0].shape == (shape[1], shape[2])), \
                "Shape mismatch: %s != %s!" % (str(ar[0].shape), str(shape[1:]))
        elif len(shape) == 4:  # first 2 dims are lists
            assert(len(ar) == shape[0]), \
                "Leading dim mismatch: %d != %d!" % (len(ar), shape[0])
            assert(shape[0] == 0 or len(ar[0]) == shape[1]), \
                "Second dim mismatch: %d != %d!" % (len(ar[0]), shape[1])
            assert(shape[0] == 0 or shape[1] == 0 or ar[0][0].shape == (shape[2], shape[3])), \
                "Shape mismatch: %s != %s!" % (str(ar[0][0].shape), str(shape[2:]))
        else:
            raise NotImplementedError("Number of dimensions must be <= 4!")


#Note: first two ards are essentially a LocalElementaryErrorgenLabel -- maybe take one of those instead?
def create_elementary_errorgen_nqudit(typ, basis_element_labels, basis_1q, normalize=False,
                                      sparse=False, tensorprod_basis=False):
    """
    TODO: docstring  - labels can be, e.g. ('H', 'XX') and basis should be a 1-qubit basis w/single-char labels
    """
    return _create_elementary_errorgen_nqudit(typ, basis_element_labels, basis_1q,
                                              normalize, sparse, tensorprod_basis, create_dual=False)


def create_elementary_errorgen_nqudit_dual(typ, basis_element_labels, basis_1q, normalize=False,
                                           sparse=False, tensorprod_basis=False):
    """
    TODO: docstring  - labels can be, e.g. ('H', 'XX') and basis should be a 1-qubit basis w/single-char labels
    """
    return _create_elementary_errorgen_nqudit(typ, basis_element_labels, basis_1q,
                                              normalize, sparse, tensorprod_basis, create_dual=True)


def _create_elementary_errorgen_nqudit(typ, basis_element_labels, basis_1q, normalize=False,
                                       sparse=False, tensorprod_basis=False, create_dual=False):
    create_fn = _lt.create_elementary_errorgen_dual if create_dual else _lt.create_elementary_errorgen
    if typ in 'HS':
        B = _functools.reduce(_np.kron, [basis_1q[bel] for bel in basis_element_labels[0]])
        ret = create_fn(typ, B, sparse=sparse)  # in std basis
    elif typ in 'CA':
        B = _functools.reduce(_np.kron, [basis_1q[bel] for bel in basis_element_labels[0]])
        C = _functools.reduce(_np.kron, [basis_1q[bel] for bel in basis_element_labels[1]])
        ret = create_fn(typ, B, C, sparse=sparse)  # in std basis
    else:
        raise ValueError("Invalid elementary error generator type: %s" % str(typ))

    if normalize:
        normfn = _spsl.norm if sparse else _np.linalg.norm
        norm = normfn(ret)  # same as norm(term.flat)
        if not _np.isclose(norm, 0):
            ret /= norm  # normalize projector
            assert(_np.isclose(normfn(ret), 1.0))

    if tensorprod_basis:
        # convert from "flat" std basis to tensorprod of std bases (same elements but in
        # a different order).  Important if want to also construct ops by kroneckering the
        # returned maps with, e.g., identities
        nQubits = int(round(_np.log(ret.shape[0]) / _np.log(4))); assert(ret.shape[0] == 4**nQubits)
        current_basis = _Basis.cast('std', ret.shape[0])
        tensorprod_basis = _Basis.cast('std', [(4,) * nQubits])
        ret = _bt.change_basis(ret, current_basis, tensorprod_basis)

    return ret


#TODO: replace two_qubit_gate, one_qubit_gate, unitary_to_pauligate_* with
# calls to this one and unitary_to_std_processmx
def rotation_gate_mx(r, mx_basis="gm"):
    """
    Construct a rotation operation matrix.

    Build the operation matrix corresponding to the unitary

    `exp(-i * (r[0]/2*PP[0]*sqrt(d) + r[1]/2*PP[1]*sqrt(d) + ...) )`

    where `PP' is the array of Pauli-product matrices
    obtained via `pp_matrices(d)`, where `d = sqrt(len(r)+1)`.
    The division by 2 is for convention, and the `sqrt(d)` is to
    essentially un-normalise the matrices returned by
    :func:`pp_matrices` to they are equal to products of the
    *standard* Pauli matrices.

    Parameters
    ----------
    r : tuple
        A tuple of coeffiecients, one per non-identity
        Pauli-product basis element

    mx_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        a d^2 x d^2 operation matrix in the specified basis.
    """
    d = int(round(_np.sqrt(len(r) + 1)))
    assert(d**2 == len(r) + 1), "Invalid number of rotation angles"

    #get Pauli-product matrices (in std basis)
    pp = _bt.basis_matrices('pp', d**2)
    assert(len(r) == len(pp[1:]))

    #build unitary (in std basis)
    ex = _np.zeros((d, d), 'complex')
    for rot, pp_mx in zip(r, pp[1:]):
        ex += rot / 2.0 * pp_mx * _np.sqrt(d)
    U = _spl.expm(-1j * ex)

    return unitary_to_superop(U, mx_basis)


def project_model(model, target_model,
                  projectiontypes=('H', 'S', 'H+S', 'LND'),
                  gen_type="logG-logT",
                  logG_weight=None):
    """
    Construct a new model(s) by projecting the error generator of `model` onto some sub-space then reconstructing.

    Parameters
    ----------
    model : Model
        The model whose error generator should be projected.

    target_model : Model
        The set of target (ideal) gates.

    projectiontypes : tuple of {'H','S','H+S','LND','LNDF'}
        Which projections to use.  The length of this tuple gives the
        number of `Model` objects returned.  Allowed values are:

        - 'H' = Hamiltonian errors
        - 'S' = Stochastic Pauli-channel errors
        - 'H+S' = both of the above error types
        - 'LND' = errgen projected to a normal (CPTP) Lindbladian
        - 'LNDF' = errgen projected to an unrestricted (full) Lindbladian

    gen_type : {"logG-logT", "logTiG", "logGTi"}
        The type of error generator to compute.
        For more details, see func:`error_generator`.

    logG_weight: float or None (default)
        Regularization weight for approximate logG in logG-logT generator.
        For more details, see func:`error_generator`.

    Returns
    -------
    projected_models : list of Models
        Elements are projected versions of `model` corresponding to
        the elements of `projectiontypes`.
    Nps : list of parameter counts
        Integer parameter counts for each model in `projected_models`.
        Useful for computing the expected log-likelihood or chi2.
    """

    from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock
    opLabels = list(model.operations.keys())  # operation labels
    basis = model.basis
    proj_basis = basis  # just use the same basis here (could make an arg later?)

    #OLD REMOVE
    ##The projection basis needs to be a basis for density matrices
    ## (i.e. 2x2 mxs in 1Q case) rather than superoperators (4x4 mxs
    ## in 1Q case) - whcih is what model.basis is.  So, we just extract
    ## a builtin basis name for the projection basis.
    #if basis.name in ('pp', 'gm', 'std', 'qt'):
    #    proj_basis_name = basis.name
    #else:
    #    proj_basis_name = 'pp'  # model.basis is weird so just use paulis as projection basis

    if basis.name != target_model.basis.name:
        raise ValueError("Basis mismatch between model (%s) and target (%s)!"
                         % (model.basis.name, target_model.basis.name))

    # Note: set to "full" parameterization so we can set the gates below
    #  regardless of what parameterization the original model had.
    gsDict = {}; NpDict = {}
    for p in projectiontypes:
        gsDict[p] = model.copy()
        gsDict[p].set_all_parameterizations("full")
        NpDict[p] = 0

    errgens = [error_generator(model.operations[gl],
                               target_model.operations[gl],
                               target_model.basis, gen_type, logG_weight)
               for gl in opLabels]

    for gl, errgen in zip(opLabels, errgens):
        if ('H' in projectiontypes) or ('H+S' in projectiontypes):
            hamBlk = LindbladCoefficientBlock('ham', proj_basis)
            hamBlk.set_from_errorgen_projections(errgen, errorgen_basis=basis)
            hamGens = hamBlk.create_lindblad_term_superoperators(mx_basis=basis)
            ham_error_gen = _np.tensordot(hamBlk.block_data, hamGens, (0, 0))

        if ('S' in projectiontypes) or ('H+S' in projectiontypes):
            stoBlk = LindbladCoefficientBlock('other_diagonal', proj_basis)
            stoBlk.set_from_errorgen_projections(errgen, errorgen_basis=basis)
            stoGens = stoBlk.create_lindblad_term_superoperators(mx_basis=basis)
            sto_error_gen = _np.tensordot(stoBlk.block_data, stoGens, (0, 0))

        if ('LND' in projectiontypes) or ('LNDF' in projectiontypes):
            HBlk = LindbladCoefficientBlock('ham', proj_basis)
            otherBlk = LindbladCoefficientBlock('other', proj_basis)
            HBlk.set_from_errorgen_projections(errgen, errorgen_basis=basis)
            otherBlk.set_from_errorgen_projections(errgen, errorgen_basis=basis)

            HGens = HBlk.create_lindblad_term_superoperators(mx_basis=basis)
            otherGens = otherBlk.create_lindblad_term_superoperators(mx_basis=basis)

            #Note: return values *can* be None if an empty/None basis is given
            #lnd_error_gen = _np.einsum('i,ijk', HProj, HGens) + \
            #                _np.einsum('ij,ijkl', OProj, OGens)
            lnd_error_gen = _np.tensordot(HBlk.block_data, HGens, (0, 0)) + \
                _np.tensordot(otherBlk.block_data, otherGens, ((0, 1), (0, 1)))

        targetOp = target_model.operations[gl]

        if 'H' in projectiontypes:
            gsDict['H'].operations[gl] = operation_from_error_generator(
                ham_error_gen, targetOp, basis, gen_type)
            NpDict['H'] += len(hamBlk.block_data)

        if 'S' in projectiontypes:
            gsDict['S'].operations[gl] = operation_from_error_generator(
                sto_error_gen, targetOp, basis, gen_type)
            NpDict['S'] += len(stoBlk.block_data)

        if 'H+S' in projectiontypes:
            gsDict['H+S'].operations[gl] = operation_from_error_generator(
                ham_error_gen + sto_error_gen, targetOp, basis, gen_type)
            NpDict['H+S'] += len(hamBlk.block_data) + len(stoBlk.block_data)

        if 'LNDF' in projectiontypes:
            gsDict['LNDF'].operations[gl] = operation_from_error_generator(
                lnd_error_gen, targetOp, basis, gen_type)
            NpDict['LNDF'] += HBlk.block_data.size + otherBlk.block_data.size

        if 'LND' in projectiontypes:
            evals, U = _np.linalg.eig(otherBlk.block_data)
            pos_evals = evals.clip(0, 1e100)  # clip negative eigenvalues to 0
            OProj_cp = _np.dot(U, _np.dot(_np.diag(pos_evals), _np.linalg.inv(U)))
            #OProj_cp is now a pos-def matrix
            #lnd_error_gen_cp = _np.einsum('i,ijk', HProj, HGens) + \
            #                   _np.einsum('ij,ijkl', OProj_cp, OGens)
            lnd_error_gen_cp = _np.tensordot(HBlk.block_data, HGens, (0, 0)) + \
                _np.tensordot(OProj_cp, otherGens, ((0, 1), (0, 1)))
            #lnd_error_gen_cp = _bt.change_basis(lnd_error_gen_cp, "std", basis)

            gsDict['LND'].operations[gl] = operation_from_error_generator(
                lnd_error_gen_cp, targetOp, basis, gen_type)
            NpDict['LND'] += HBlk.block_data.size + otherBlk.block_data.size

        #Removed attempt to contract H+S to CPTP by removing positive stochastic projections,
        # but this doesn't always return the gate to being CPTP (maybe b/c of normalization)...
        #sto_error_gen_cp = _np.einsum('i,ijk', stoProj.clip(None,0), stoGens)
        #  # (only negative stochastic projections OK)
        #sto_error_gen_cp = _tools.std_to_pp(sto_error_gen_cp)
        #gsHSCP.operations[gl] = _tools.operation_from_error_generator(
        #    ham_error_gen, targetOp, gen_type) #+sto_error_gen_cp

    #DEBUG!!!
    #print("DEBUG: BEST sum neg evals = ",_tools.sum_of_negative_choi_eigenvalues(model))
    #print("DEBUG: LNDCP sum neg evals = ",_tools.sum_of_negative_choi_eigenvalues(gsDict['LND']))

    #Check for CPTP where expected
    #assert(_tools.sum_of_negative_choi_eigenvalues(gsHSCP) < 1e-6)
    #assert(_tools.sum_of_negative_choi_eigenvalues(gsDict['LND']) < 1e-6)

    #Collect and return requrested results:
    ret_gs = [gsDict[p] for p in projectiontypes]
    ret_Nps = [NpDict[p] for p in projectiontypes]
    return ret_gs, ret_Nps


def compute_best_case_gauge_transform(gate_mx, target_gate_mx, return_all=False):
    """
    Returns a gauge transformation that maps `gate_mx` into a matrix that is co-diagonal with `target_gate_mx`.

    (Co-diagonal means that they share a common set of eigenvectors.)

    Gauge transformations effectively change the basis of all the gates in a model.
    From the perspective of a single gate a gauge transformation leaves it's
    eigenvalues the same and changes its eigenvectors.  This function finds a *real*
    transformation that transforms the eigenspaces of `gate_mx` so that there exists
    a set of eigenvectors which diagonalize both `gate_mx` and `target_gate_mx`.

    Parameters
    ----------
    gate_mx : numpy.ndarray
        Gate matrix to transform.

    target_gate_mx : numpy.ndarray
        Target gate matrix.

    return_all : bool, optional
        If true, also return the matrices of eigenvectors
        for `Ugate` for gate_mx and `Utgt` for target_gate_mx such
        that `U = dot(Utgt, inv(Ugate))` is real.

    Returns
    -------
    U : numpy.ndarray
        A gauge transformation such that if `epgate = U * gate_mx * U_inv`,
        then `epgate` (which has the same eigenalues as `gate_mx`), can be
        diagonalized with a set of eigenvectors that also diagonalize
        `target_gate_mx`.  Furthermore, `U` is real.
    Ugate, Utgt : numpy.ndarray
        only if `return_all == True`.  See above.
    """

    # A complication that must be dealt with is that
    # the eigenvalues of `target_gate_mx` can be degenerate,
    # and so matching up eigenvalues can't be done *just* based on value.
    # Our algorithm consists of two steps:
    # 1) match gate & target eigenvalues based on value, ensuring conjugacy
    #    relationships between eigenvalues are preserved.
    # 2) for each eigenvalue/vector of `gate`, project the eigenvector onto
    #    the eigenspace of `tgt_gate` corresponding to the matched eigenvalue.
    #    (treat conj-pair eigenvalues of `gate` together).

    # we want a matrix that gauge-transforms gate_mx into a matrix as
    # close to target_gate_mx as possible, i.e. that puts gate_mx's
    # eigenvalues in the eigenspaces of target_gate_mx.  This is done
    # by Ubest = _np.dot(Utgt, inv(Uop)), but there are often degrees
    # of freedom in Uop because of its degeneracies.  Also, we want Ubest
    # to be *real*, so we need to ensure the conjugacy structure of Utgt
    # and Uop match...

    assert(_np.linalg.norm(gate_mx.imag) < 1e-8)
    assert(_np.linalg.norm(target_gate_mx.imag) < 1e-8)

    if True:  # NEW approach that gives sorted eigenvectors
        def _get_eigenspace_pairs(mx, tol=1e-6):
            evals, U = _np.linalg.eig(mx)  # so mx = U * evals * u_inv
            espace_pairs = {}; conj_pair_indices = []

            #Pass 1: real evals and positive-imaginary-element-of-conjugate pair evals
            #  (these are the representatives of "eigenspace pairs")
            for i, ev in enumerate(evals):
                if ev.imag < -tol:
                    conj_pair_indices.append(i); continue  # save for pass2

                #see if ev is already in espace_pairs
                for k, v in espace_pairs.items():
                    if abs(k - ev) < tol:
                        espace_pairs[k]['indices'].append(i)
                        espace_pairs[k]['conj_pair_indices'].append(None)
                        #espace_pairs[k]['evecs'].append(U[:,i])
                        break
                else:
                    espace_pairs[ev] = {'indices': [i], 'conj_pair_indices': [None]}

            #Pass 2: negative-imaginary-part elements of evals that occur in conjugate pairs
            for i in conj_pair_indices:
                ev_pos = _np.conjugate(evals[i])
                for k, v in espace_pairs.items():  # ev_pos *should* be in espace_pairs
                    if abs(k - ev_pos) < tol:
                        #found the correct eigenspace-pair to add this eval & evec to,
                        # now figure our where to put this index based on conjugacy relationships,
                        # i.e. U[:,esp['indices'][i]] is always conjugate to U[:,esp['conj_pair_indices'][i]]
                        for jj, j in enumerate(espace_pairs[k]['indices']):
                            if espace_pairs[k]['conj_pair_indices'][jj] is None:  # an empty slot
                                espace_pairs[k]['conj_pair_indices'][jj] = i
                                U[:, i] = U[:, j].conj()
                                break
                        else:
                            raise ValueError("Nowhere to place a conjugate eigenvector %d-dim eigenbasis for %s!"
                                             % (len(espace_pairs[k]['indices']), str(k)))

                        break
                else:
                    raise ValueError("Expected to find %s as an espace-pair representative in %s"
                                     % (str(ev_pos), str(espace_pairs.keys())))

            #if not (_np.allclose(mx, _np.dot(U, _np.dot(_np.diag(evals), _np.linalg.inv(U))))):
            #    import bpdb; bpdb.set_trace()
            return evals, U, espace_pairs

        def standard_diag(mx, tol=1e-6):
            evals, U, espairs = _get_eigenspace_pairs(mx)
            std_evals = []
            std_evecs = []
            sorted_rep_evals = sorted(list(espairs.keys()), key=lambda x: (x.real, x.imag))
            for ev in sorted_rep_evals:  # iterate in sorted order just for definitiveness
                info = espairs[ev]
                dim = len(info['indices'])  # dimension of this eigenspace (and it's pair, if there is one)

                #Ensure real eigenvalue blocks should have real eigenvectors
                if abs(ev.imag) < tol:
                    #find linear combinations of the eigenvectors that are real
                    Usub = U[:, info['indices']]
                    if _np.linalg.norm(Usub.imag) > tol:
                        # Im part of Usub * combo = Usub.real*combo.imag + Usub.imag*combo.real
                        combo_real_imag = _mt.nullspace(_np.concatenate((Usub.imag, Usub.real), axis=1))
                        combos = combo_real_imag[0:dim, :] + 1j * combo_real_imag[dim:, :]
                        if combos.shape[1] > dim:  # if Usub is (actually or near) rank defficient, and we get more
                            combos = combos[:, 0:dim]  # combos than we need, just discard the last ones
                        if combos.shape[1] != dim:
                            raise ValueError(("Can only find %d (< %d) *real* linear combinations of"
                                              " vectors in eigenspace for %s!") % (combos.shape[1], dim, str(ev)))
                        U[:, info['indices']] = _np.dot(Usub, combos)
                        assert(_np.linalg.norm(U[:, info['indices']].imag) < tol)

                    #Add real eigenvalues and vectors
                    std_evals.extend([ev] * dim)
                    std_evecs.extend([U[:, i] for i in info['indices']])

                else:  # complex eigenvalue case - should have conjugate pair info
                    #Ensure blocks for conjugate-pairs of eigenvalues follow one after another and
                    # corresponding eigenvectors (e.g. the first of each block) are conjugate pairs
                    # (this is already done in the eigenspace construction)
                    assert(len(info['conj_pair_indices']) == dim)
                    std_evals.extend([ev] * dim)
                    std_evals.extend([_np.conjugate(ev)] * dim)
                    std_evecs.extend([U[:, i] for i in info['indices']])
                    std_evecs.extend([U[:, i] for i in info['conj_pair_indices']])

            return _np.array(std_evals), _np.array(std_evecs).T

        #Create "gate_tilde" which has the eigenvectors of gate_mx around the matched eigenvalues of target_gate_mx
        # Doing this essentially decouples the problem of eigenvalue matching from the rest of the task -
        # after gate_tilde is created, it and target_gate_mx have exactly the *same* eigenvalues.
        evals_tgt, Utgt = _np.linalg.eig(target_gate_mx)
        evals_gate, Uop = _np.linalg.eig(gate_mx)
        pairs = _mt.minweight_match_realmxeigs(evals_gate, evals_tgt)
        replace_evals = _np.array([evals_tgt[j] for _, j in pairs])
        gate_tilde = _np.dot(Uop, _np.dot(_np.diag(replace_evals), _np.linalg.inv(Uop)))

        #Create "standard diagonalizations" of gate_tilde and target_gate_mx, which give
        # sort the eigenvalues and ensure eigenvectors occur in *corresponding* conjugate pairs
        # (e.g. even when evals +1j and -1j have multiplicity 4, the first 4-D eigenspace, the
        evals_tgt, Utgt = standard_diag(target_gate_mx)
        evals_tilde, Uop = standard_diag(gate_tilde)
        assert(_np.allclose(evals_tgt, evals_tilde))

        #Update Utgt so that Utgt * inv_Uop is close to the identity
        kite = _mt.compute_kite(evals_tgt)  # evals are grouped by standard_diag, so this works
        D_prior_to_proj = _np.dot(_np.linalg.inv(Utgt), Uop)
        #print("D prior to projection to ",kite," kite:"); _mt.print_mx(D_prior_to_proj)
        D = _mt.project_onto_kite(D_prior_to_proj, kite)
        start = 0
        for i, k in enumerate(kite):
            slc = slice(start, start + k)
            dstart = start + k
            for kk in kite[i + 1:]:
                if k == kk and _np.isclose(evals_tgt[start], evals_tgt[dstart].conj()):  # conjugate block!
                    dslc = slice(dstart, dstart + kk)
                    # enforce block conjugacy needed to retain Uproj conjugacy structure
                    D[dslc, dslc] = D[slc, slc].conj()
                    break
                dstart += kk
            start += k
        Utgt = _np.dot(Utgt, D)  # update Utgt

        Utrans = _np.dot(Utgt, _np.linalg.inv(Uop))
        assert(_np.linalg.norm(_np.imag(Utrans)) < 1e-7)
        Utrans = Utrans.real  # _np.real_if_close(Utrans, tol=1000)

        if return_all:
            return Utrans, Uop, Utgt, evals_tgt
        else:
            return Utrans

    evals_tgt, Utgt = _np.linalg.eig(target_gate_mx)
    evals_gate, Uop = _np.linalg.eig(gate_mx)

    #_, pairs = _mt.minweight_match(evals_tgt, evals_gate, return_pairs=True)
    pairs = _mt.minweight_match_realmxeigs(evals_tgt, evals_gate)

    #Form eigenspaces of Utgt
    eigenspace = {}  # key = index of target eigenval, val = assoc. eigenspace
    for i, ev in enumerate(evals_tgt):
        for j in eigenspace:
            if _np.isclose(ev, evals_tgt[j]):  # then add evector[i] to this eigenspace
                eigenspace[j].append(Utgt[:, i])
                eigenspace[i] = eigenspace[j]  # reference!
                break
        else:
            eigenspace[i] = [Utgt[:, i]]  # new list = new eigenspace

    #Project each eigenvector (col of Uop) onto space of cols
    evectors = {}  # key = index of gate eigenval, val = assoc. (projected) eigenvec
    for ipair, (i, j) in enumerate(pairs):
        #print("processing pair (i,j) = ",i,j)
        if j in evectors: continue  # we already processed this one!

        # non-orthog projection:
        # v = E * coeffs s.t. |E*coeffs-v|^2 is minimal  (E is not square so can't invert)
        # --> E.dag * v = E.dag * E * coeffs
        # --> inv(E.dag * E) * E.dag * v = coeffs
        # E*coeffs = E * inv(E.dag * E) * E.dag * v
        E = _np.array(eigenspace[i]).T; Edag = E.T.conjugate()
        coeffs = _np.dot(_np.dot(_np.linalg.inv(_np.dot(Edag, E)), Edag), Uop[:, j])
        evectors[j] = _np.dot(E, coeffs)

        #check for conjugate pair
        #DB: print("Looking for conjugate:")
        for i2, j2 in pairs[ipair + 1:]:
            if abs(evals_gate[j].imag) > 1e-6 and _np.isclose(evals_gate[j], _np.conjugate(evals_gate[j2])) \
               and _np.allclose(Uop[:, j], Uop[:, j2].conj()):
                #DB: print("Found conjugate at j = ",j2)
                evectors[j2] = _np.conjugate(evectors[j])
                # x = _np.linalg.solve(_np.dot(Edag, E), _np.dot(Edag, evectors[j2]))
                #assert(_np.isclose(_np.linalg.norm(x),_np.linalg.norm(coeffs))) ??
                #check that this vector is in the span of eigenspace[i2]?

    #build new "Utgt" using specially chosen linear combos of degenerate-eigenvecs
    Uproj = _np.array([evectors[i] for i in range(Utgt.shape[1])]).T
    assert(_np.allclose(_np.dot(Uproj, _np.dot(_np.diag(evals_tgt), _np.linalg.inv(Uproj))), target_gate_mx))

    #This is how you get the eigenspace-projected gate:
    #  epgate = _np.dot(Uproj, _np.dot(_np.diag(evals_gate), Uproj_inv))
    #  epgate = _np.real_if_close(epgate, tol=1000)

    # G = Uop * evals_gate * Uop_inv  => eval_gate = Uop_inv * G * Uop
    # epgate = Uproj * evals_gate * Uproj_inv  (eigenspace-projected gate)
    # so  epgate = (Uproj Uop_inv) G (Uproj Uop_inv)_inv => (Uproj Uop_inv) is
    # a "best_gauge_transform" for G, i.e. it makes G codiagonal with G_tgt
    Ubest = _np.dot(Uproj, _np.linalg.inv(Uop))
    assert(_np.linalg.norm(_np.imag(Ubest)) < 1e-7)
    # this should never happen & indicates an uncaught failure in
    # minweight_match_realmxeigs(...)

    Ubest = Ubest.real

    if return_all:
        return Ubest, Uop, Uproj, evals_tgt
    else:
        return Ubest


def project_to_target_eigenspace(model, target_model, eps=1e-6):
    """
    Project each gate of `model` onto the eigenspace of the corresponding gate within `target_model`.

    Returns the resulting `Model`.

    Parameters
    ----------
    model : Model
        Model to act on.

    target_model : Model
        The target model, whose gates define the target eigenspaces being projected onto.

    eps : float, optional
        Small magnitude specifying how much to "nudge" the target gates
        before eigen-decomposing them, so that their spectra will have the
        same conjugacy structure as the gates of `model`.

    Returns
    -------
    Model
    """
    ret = target_model.copy()
    ret.set_all_parameterizations("full")  # so we can freely assign gates new values

    for gl, gate in model.operations.items():
        tgt_gate = target_model.operations[gl]

        #Essentially, we want to replace the eigenvalues of `tgt_gate`
        # (and *only* the eigenvalues) with those of `gate`.  This is what
        # a "best gate gauge transform does" (by definition)
        gate_mx = gate.to_dense(on_space='minimal')
        Ugauge = compute_best_case_gauge_transform(gate_mx, tgt_gate.to_dense(on_space='minimal'))
        Ugauge_inv = _np.linalg.inv(Ugauge)

        epgate = _np.dot(Ugauge, _np.dot(gate_mx, Ugauge_inv))
        ret.operations[gl] = epgate

    return ret


def unitary_to_pauligate(u):
    """
    Get the linear operator on (vectorized) density matrices corresponding to a n-qubit unitary operator on states.

    Parameters
    ----------
    u : numpy array
        A dxd array giving the action of the unitary
        on a state in the sigma-z basis.
        where d = 2 ** n-qubits

    Returns
    -------
    numpy array
        The operator on density matrices that have been
        vectorized as d**2 vectors in the Pauli basis.
    """
    assert u.shape[0] == u.shape[1], '"Unitary" matrix is not square'
    return unitary_to_superop(u, 'pp')


def is_valid_lindblad_paramtype(typ):
    """
    Whether `typ` is a recognized Lindblad-gate parameterization type.

    A *Lindblad type* is comprised of a parameter specification followed
    optionally by an evolution-type suffix.  The parameter spec can be
    "GLND" (general unconstrained Lindbladian), "CPTP" (cptp-constrained),
    or any/all of the letters "H" (Hamiltonian), "S" (Stochastic, CPTP),
    "s" (Stochastic), "A" (Affine), "D" (Depolarization, CPTP),
    "d" (Depolarization) joined with plus (+) signs.  Note that "A" cannot
    appear without one of {"S","s","D","d"}. The suffix can be non-existent
    (density-matrix), "terms" (state-vector terms) or "clifford terms"
    (stabilizer-state terms).  For example, valid Lindblad types are "H+S",
    "H+d+A", "CPTP clifford terms", or "S+A terms".

    Parameters
    ----------
    typ : str
        A paramterization type.

    Returns
    -------
    bool
    """
    from pygsti.modelmembers.operations.lindbladerrorgen import LindbladParameterization as _LP
    try:
        _LP.cast(typ)
        return True
    except ValueError:
        return False

    #OLD: return typ in ("CPTP", "H+S", "S", "H+S+A", "S+A", "H+D", "D", "H+D+A", "D+A",
    #OLD:                "GLND", "H+s", "s", "H+s+A", "s+A", "H+d", "d", "H+d+A", "d+A", "H")


def effect_label_to_outcome(povm_and_effect_lbl):
    """
    Extract the outcome label from a "simplified" effect label.

    Simplified effect labels are not themselves so simple.  They
    combine POVM and effect labels so that accessing any given effect
    vector is simpler.

    If `povm_and_effect_lbl` is `None` then `"NONE"` is returned.

    Parameters
    ----------
    povm_and_effect_lbl : Label
        Simplified effect vector.

    Returns
    -------
    str
    """
    # Helper fn: POVM_ELbl:sslbls -> Elbl mapping
    if povm_and_effect_lbl is None:
        return "NONE"  # Dummy label for placeholding
    else:
        if isinstance(povm_and_effect_lbl, _Label):
            last_underscore = povm_and_effect_lbl.name.rindex('_')
            effect_lbl = povm_and_effect_lbl.name[last_underscore + 1:]
        else:
            last_underscore = povm_and_effect_lbl.rindex('_')
            effect_lbl = povm_and_effect_lbl[last_underscore + 1:]
        return effect_lbl  # effect label alone *is* the outcome


def effect_label_to_povm(povm_and_effect_lbl):
    """
    Extract the POVM label from a "simplified" effect label.

    Simplified effect labels are not themselves so simple.  They
    combine POVM and effect labels so that accessing any given effect
    vector is simpler.

    If `povm_and_effect_lbl` is `None` then `"NONE"` is returned.

    Parameters
    ----------
    povm_and_effect_lbl : Label
        Simplified effect vector.

    Returns
    -------
    str
    """
    # Helper fn: POVM_ELbl:sslbls -> POVM mapping
    if povm_and_effect_lbl is None:
        return "NONE"  # Dummy label for placeholding
    else:
        if isinstance(povm_and_effect_lbl, _Label):
            last_underscore = povm_and_effect_lbl.name.rindex('_')
            povm_name = povm_and_effect_lbl.name[:last_underscore]
        else:
            last_underscore = povm_and_effect_lbl.rindex('_')
            povm_name = povm_and_effect_lbl[:last_underscore]
        return povm_name
