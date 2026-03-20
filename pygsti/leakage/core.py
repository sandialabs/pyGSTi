#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg as la

from pygsti.baseobjs.basis import Basis
from pygsti.tools import basistools as pgbt
from pygsti.tools import matrixtools as pgmt
from pygsti.tools.metaprogramming import set_docstring


NOTATION = \
"""
Definitions for Hilbert--Schmidt space
--------------------------------------
We use H to denote a fixed complex Hilbert space equipped with a standard
basis, and we use U to denote various subspaces of H (possibly H itself).

The Hilbert--Schmidt space of linear operators from U to U is denoted M[U],
and the set of linear transformations (or superoperators) from M[U] to M[U]
is denoted S[U].

The orthogonal complement of U in H is denoted U^⟂. There are multiple ways
to extend a linear operator Φ ∈ M[U] to all of H = U ⨁ U^⟂.

    We call an extension ...

      * direct if U and U^⟂ are invariant subspaces for both Φ and Φ^†,
      * annhilating if ker(Φ) and ker(Φ^†) contain U^⟂, and
      * unitary if it is direct and A is unitary on U^⟂.

    We further call a unitary extension ...

      * unintrusive if it acts as a multiple of the identity on U^⟂, and
      * demure if it acts as the identity on U^⟂.

Leakage modeling tends to use annhilating extension for Hermitian operators
and demure extension for unitary operators.


Basic definitions for leakage modeling
--------------------------------------
Let B denote a Hermitian basis for the Hilbert--Schmidt space M[H]. We say that
B supports leakage modeling if

    (1) it as a unique element whose label consists of (and only of) one
        or more copies of the character 'I', and
    (2) that element is proportional to a real orthogonal projector on H.

Assuming B is such a basis, its element that satisfies (1) and (2) is called
its computational basis matrix. (Not be confused with "the computational
basis" of a Hilbert space, which is synonymous with the standard basis.)

The computational basis matrix defines a few related objects.

  * The computational subspace, C, is the range of the computational basis matrix.

  * The computational effect is the Hermitian operator in M[H] that orthogonally
    projects onto the computational subspace.

  * The computational projector is the Hermiticty-preserving superoperator in
    S[H] that orthogonally projects from M[H] to M[C].

"""


def computational_effect(basis: Basis) -> np.ndarray:
    assert hasattr(basis, 'labels')
    candidates = [ell for ell in basis.labels if len(ell.strip('I')) == 0]  # type: ignore
    candidates = sorted(candidates, key=lambda ell: -len(ell))
    assert len(candidates) > 0
    label = candidates[0]
    E = basis.ellookup[label].copy()  # type: ignore
    k = np.linalg.matrix_rank(E)
    E *= (k/np.trace(E))
    return E


@set_docstring(NOTATION)
# TODO: document me
def computational_superkets(basis: Basis, E: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Our desired subspace is
        { rho : ker(rho) contains ker(E) } .
    Since E is (proportional to) a projector, this is equivalent to
        { rho : E rho E = rho }.
    """
    if not basis.implies_leakage_modeling:
        return np.eye(basis.dim)
    if not basis.is_hermitian():
        raise ValueError()
    if E is None:
        E = computational_effect(basis)
        k = np.linalg.matrix_rank(E)
    else:
        assert isinstance(E, np.ndarray)
        E = E.copy()
        k = np.linalg.matrix_rank(E)
        E *= (k/np.trace(E))
    if not pgmt.is_projector(E):
        raise ValueError()
    proj_elements  = [ E @ B @ E for B in basis.elements ]
    subspace_frame = np.column_stack([ pgbt.stdmx_to_vec(pB, basis) for pB in proj_elements ])
    # ^ a "frame" is an overcomplete basis.
    subspace_frame = subspace_frame.real
    # ^ Since basis.elements are Hermitian and E is Hermitian, we have that proj_elements are
    #   Hermitian and that their vectorizations are real. We cast to real here just to
    #   eliminate possible rounding errors.
    U_full : np.ndarray = la.qr(subspace_frame, pivoting=True)[0] # type: ignore
    U = U_full[:, :k**2]
    return U


@set_docstring(NOTATION)
def computational_projector(basis: Basis) -> np.ndarray:
    dim = basis.dim
    if basis.first_element_is_identity:
        return np.eye(dim)
    E = computational_effect(basis)
    U = computational_superkets(basis, E)
    P = U @ U.T
    return P
