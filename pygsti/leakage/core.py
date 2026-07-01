#***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from pygsti.baseobjs.basis import Basis, ExplicitBasis, _eye_label
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
Let B denote a Hermitian basis for the Hilbert--Schmidt space M[H].

We say that B *supports* leakage modeling if there's at least one element whose
label is a string of the form 'I'*k for some integer k, and the element with
the longest such label is proportional to a real orthogonal projector on H.

Assuming B is such a basis, its element satisfying the conditions above is
called the *computational basis matrix*. (Not be confused with "the computational
basis" of a Hilbert space, which is synonymous with the standard basis.)

The computational basis matrix defines a few related objects.

  * The computational subspace, C, is the range of the computational basis matrix.

  * The computational effect is the Hermitian operator in M[H] that orthogonally
    projects onto the computational subspace.

  * The computational projector is the Hermicity-preserving superoperator in
    S[H] that orthogonally projects from M[H] to M[C].

We say that B *implies* leakage modeling if C is a proper subspace of H.
"""


@set_docstring(
"""
Return the computational effect of `basis`: the Hermitian operator E ∈ M[H] that
orthogonally projects H onto the computational subspace C.

E is obtained from the computational basis matrix of `basis` (the element whose label is
the longest string of the form 'I'*k) by cleaning it into a genuine real orthogonal
projector. Its range is C and its rank is dim(C).

Raises a ValueError if `basis` does not support leakage modeling, i.e. if it has no
element whose label is a string of the form 'I'*k that is proportional to a real
orthogonal projector on H.
""" + NOTATION)
def computational_effect(basis: Basis) -> np.ndarray:
    label = _eye_label(basis)
    E = basis.ellookup[label].copy()  # type: ignore
    E = pgmt.induced_projector(E, tol=1e-10, require_real=True)
    if E.size == 0:
        raise ValueError(f'basis {basis} does not support leakage modeling.')
    return E


@set_docstring(
"""
Return a matrix U whose columns form an orthonormal basis (in the superket sense of
`basis`) for M[C], the space of operators supported on the computational subspace C.

M[C] is the set of ρ ∈ M[H] whose kernel contains that of the computational effect E.
Since E is (proportional to) a projector, this is the same as

    { ρ ∈ M[H] : E ρ E = ρ } .

We build U by projecting every element of `basis` onto this set (ρ ↦ E ρ E), vectorizing
the results in `basis`, and orthonormalizing the resulting (overcomplete) frame with a
pivoted QR factorization. If C has dimension k, then U has k² columns.

If `basis` does not imply leakage modeling then C = H, M[C] = M[H], and this function
returns the identity of order basis.dim.
""" + NOTATION)
def computational_superkets(basis: Basis) -> np.ndarray:
    if not basis.implies_leakage_modeling:
        return np.eye(basis.dim)
    if not basis.is_hermitian():
        raise ValueError()
    E = computational_effect(basis)
    k = np.linalg.matrix_rank(E)
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


@set_docstring(
"""
Return the computational projector of `basis`: the Hermiticity-preserving superoperator
P ∈ S[H] that orthogonally projects M[H] onto M[C].

P is assembled from the computational superkets U as P = U Uᵀ. Applied to a superket, P
annihilates every component outside M[C] while leaving components within M[C] untouched.

If `basis` does not imply leakage modeling then M[C] = M[H] and P is the identity of
S[H] (an order-basis.dim identity matrix).
""" + NOTATION)
def computational_projector(basis: Basis) -> np.ndarray:
    dim = basis.dim
    if basis.first_element_is_identity:
        return np.eye(dim)
    U = computational_superkets(basis)
    P = U @ U.T
    return P


def augment_for_leakage_modeling(basis: Basis, E: np.ndarray) -> Basis:
    """
    Returns a Basis `b_out` that's spiritially similar to `basis`, while
    implying leakage modeling with C = range(E) as the computational subspace.

    The elements of `b_out` have the following properties.

      * The first element is proportional to E. Its label consists only of 'I's,
        and is the longest such label in the basis.

      * The first rank(E)^2 elements span M[C]. Labels of these elements (other
        than the very first) match those of their corresponding elements in `basis`. 

      * All subsequent elements span M[H] \\ M[C]. Their labels are of the form
        'L[ell]', where 'ell' is the label of their corresponding element in `basis`.

      * The final element is proportional to the projector onto C^⟂
        and is labeled 'L'.

    Notes
    -----
    Let k=rank(E). Assume that E has already been replaced by the orthogonal
    projector onto its range, and index the basis elements starting from zero.

    We construct basis elements 1, ..., k^2-1 by first projecting the original
    basis elements onto M[C]. Then we use pivoted QR to identify the k^2-1
    matrices that are "most supported" on M[C] after projecting out E.

    We construct the remaining elements by projecting the original basis
    elements onto M[C]^⟂. Then we use pivoted QR to identify the
    dim(basis) - k^2 - 1 matrices that are "most supported" on M[C]^⟂
    after projecting out I - E.
    """
    # Step 0: argument validation
    if la.norm(np.imag(E)) > 1e-10:
        raise ValueError("E must be real")
    pgmt.assert_hermitian(E, tol=1e-10)
    E  = np.real(E)
    E  = (E + E.T)/2  # ensure symmetry in exact arithmetic
    k  = np.linalg.matrix_rank(E)
    E *= (k/np.trace(E))
    if not pgmt.is_projector(E):
        raise ValueError("E must be (proportional to) a projector")
    try:
        I_lbl = _eye_label(basis)
    except ValueError:
        I_lbl = 'I'

    # Step 1: build computational subspace (cs) basis elements and labels.
    cs_elements = [ E @ B @ E            for B in basis.elements ]      # type: ignore
    cs_elements = [ (B + B.T.conj()) / 2 for B in cs_elements ]
    mat1 = E.ravel().reshape(-1, 1)
    mat2 = np.column_stack([ B.ravel() for B in cs_elements ])
    p = pgmt.pivot_indices_after_deflation(mat1, mat2)
    p = p[:k**2-1]
    cs_elements = [ E     ] + [ cs_elements[i]  for i in p ]
    cs_labels   = [ I_lbl ] + [ basis.labels[i] for i in p ]            # type: ignore

    # Step 2: build orthogonal complement (oc) basis elements and labels.
    E_comp = np.eye(E.shape[0]) - E
    oc_elements = [ B - E @ B @ E        for B in basis.elements ]      # type: ignore
    oc_elements = [ (B + B.T.conj()) / 2 for B in oc_elements ]
    mat1 = E_comp.ravel().reshape(-1, 1)
    mat2 = np.column_stack([ B.ravel() for B in oc_elements ])
    p    = pgmt.pivot_indices_after_deflation(mat1, mat2)
    p    = p[:basis.dim - k**2 - 1]
    oc_elements = [ oc_elements[i]           for i in p ] + [ E_comp ]
    oc_labels   = [ f'L[{basis.labels[i]}]'  for i in p ] + [ 'L'    ]  # type: ignore

    # Step 3: stitch together and normalize.
    labels   = cs_labels   + oc_labels
    elements = np.array(cs_elements + oc_elements)
    for element in elements:
        element /= la.norm(element)
        element[:] = element.round(decimals=16)
    new_name  = 'Leakage augmented ' + basis.name
    new_basis = ExplicitBasis(elements, labels, name=new_name)
    assert new_basis.implies_leakage_modeling

    return new_basis
