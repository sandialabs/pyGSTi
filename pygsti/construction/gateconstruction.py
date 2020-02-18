""" Functions for creating gates """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.linalg as _spl

from ..tools.optools import unitary_to_pauligate

## Pauli basis matrices
from ..tools.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz

sigmaii = _np.kron(id2x2, id2x2)
sigmaix = _np.kron(id2x2, sigmax)
sigmaiy = _np.kron(id2x2, sigmay)
sigmaiz = _np.kron(id2x2, sigmaz)
sigmaxi = _np.kron(sigmax, id2x2)
sigmaxx = _np.kron(sigmax, sigmax)
sigmaxy = _np.kron(sigmax, sigmay)
sigmaxz = _np.kron(sigmax, sigmaz)
sigmayi = _np.kron(sigmay, id2x2)
sigmayx = _np.kron(sigmay, sigmax)
sigmayy = _np.kron(sigmay, sigmay)
sigmayz = _np.kron(sigmay, sigmaz)
sigmazi = _np.kron(sigmaz, id2x2)
sigmazx = _np.kron(sigmaz, sigmax)
sigmazy = _np.kron(sigmaz, sigmay)
sigmazz = _np.kron(sigmaz, sigmaz)


def single_qubit_gate(hx, hy, hz, noise=0):
    """
    Construct the single-qubit operation matrix.

    Build the operation matrix given by exponentiating -i * (hx*X + hy*Y + hz*Z),
    where X, Y, and Z are the sigma matrices.  Thus, hx, hy, and hz
    correspond to rotation angles divided by 2.  Additionally, a uniform
    depolarization noise can be applied to the gate.

    Parameters
    ----------
    hx : float
        Coefficient of sigma-X matrix in exponent.

    hy : float
        Coefficient of sigma-Y matrix in exponent.

    hz : float
        Coefficient of sigma-Z matrix in exponent.

    noise: float, optional
        The amount of uniform depolarizing noise.

    Returns
    -------
    numpy array
        4x4 operation matrix which operates on a 1-qubit
        density matrix expressed as a vector in the
        Pauli basis ( {I,X,Y,Z}/sqrt(2) ).
    """
    ex = -1j * (hx * sigmax + hy * sigmay + hz * sigmaz)
    D = _np.diag([1] + [1 - noise] * (4 - 1))
    return _np.dot(D, unitary_to_pauligate(_spl.expm(ex)))


def two_qubit_gate(ix=0, iy=0, iz=0, xi=0, xx=0, xy=0, xz=0, yi=0, yx=0, yy=0, yz=0, zi=0, zx=0, zy=0, zz=0, ii=0):
    """
    Construct the single-qubit operation matrix.

    Build the operation matrix given by exponentiating -i * (xx*XX + xy*XY + ...)
    where terms in the exponent are tensor products of two Pauli matrices.

    Parameters
    ----------
    ix : float, optional
        Coefficient of IX matrix in exponent.
    iy : float, optional
        Coefficient of IY matrix in exponent.
    iy : float, optional
        Coefficient of IY matrix in exponent.
    iz : float, optional
        Coefficient of IZ matrix in exponent.
    xi : float, optional
        Coefficient of XI matrix in exponent.
    xx : float, optional
        Coefficient of XX matrix in exponent.
    xy : float, optional
        Coefficient of XY matrix in exponent.
    xz : float, optional
        Coefficient of XZ matrix in exponent.
    yi : float, optional
        Coefficient of YI matrix in exponent.
    yx : float, optional
        Coefficient of YX matrix in exponent.
    yy : float, optional
        Coefficient of YY matrix in exponent.
    yz : float, optional
        Coefficient of YZ matrix in exponent.
    zi : float, optional
        Coefficient of ZI matrix in exponent.
    zx : float, optional
        Coefficient of ZX matrix in exponent.
    zy : float, optional
        Coefficient of ZY matrix in exponent.
    zz : float, optional
        Coefficient of ZZ matrix in exponent.
    ii : float, optional
        Coefficient of II matrix in exponent.
    Returns
    -------
    numpy array
        16x16 operation matrix which operates on a 2-qubit
        density matrix expressed as a vector in the
        Pauli-Product basis.
    """
    ex = ii * _np.identity(4, 'complex')
    ex += ix * sigmaix
    ex += iy * sigmaiy
    ex += iz * sigmaiz
    ex += xi * sigmaxi
    ex += xx * sigmaxx
    ex += xy * sigmaxy
    ex += xz * sigmaxz
    ex += yi * sigmayi
    ex += yx * sigmayx
    ex += yy * sigmayy
    ex += yz * sigmayz
    ex += zi * sigmazi
    ex += zx * sigmazx
    ex += zy * sigmazy
    ex += zz * sigmazz
    return unitary_to_pauligate(_spl.expm(-1j * ex))
    #TODO: fix noise op to depolarizing
