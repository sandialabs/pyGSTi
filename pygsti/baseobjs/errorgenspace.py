"""
Defines the ErrorgenSpace class and supporting functionality.
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

from pygsti.tools import matrixtools as _mt


class ErrorgenSpace(object):
    """
    A vector space of error generators, spanned by some basis.

    This object collects the information needed to specify a space
    within the space of all error generators.
    """

    def __init__(self, vectors, basis):
        self.vectors = vectors
        self.elemgen_basis = basis
        #Question: have multiple bases or a single one?
        #self._vectors = [] if (items is None) else items  # list of (basis, vectors_mx) pairs
        # map sslbls => (vectors, basis) where basis.sslbls == sslbls
        # or basis => vectors if bases can hash well(?)

    def intersection(self, other_space, free_on_unspecified_space=False):
        """
        TODO: docstring
        """
        #Note: currently we assume self.vectors is a *dense* numpy array, but this may/should be expanded to
        # work with or solely utilize SPARSE matrices in the future.
        dtype = self.vectors.dtype

        if free_on_unspecified_space:
            common_basis = self.elemgen_basis.union(other_space.elemgen_basis)
            diff_self = common_basis.difference(self.elemgen_basis)
            diff_other = common_basis.difference(other_space.elemgen_basis)
            Vl, Vli, Wl, Wli = (self.vectors.shape[1], len(diff_self), other_space.vectors.shape[1], len(diff_other))

            #Fill in matrix to take nullspace of: [ V I | W I ] where V and W are self's and other_space's vectors
            # in the common basis and I's stand for filling in the identity on rows corresponding to missing elements
            # in each spaces basis, respectively.
            i = 0  # column offset
            VIWI = _np.zeros((len(common_basis), Vl + Vli + Wl + Wli), dtype)  # SPARSE in future?
            VIWI[common_basis.label_indices(self.elemgen_basis.labels), 0:Vl] = self.vectors[:, :]; i += Vl
            VIWI[common_basis.label_indices(diff_self.labels), i:i + Vli] = _np.identity(Vli, dtype); i += Vli
            VIWI[common_basis.label_indices(other_space.elemgen_basis.labels), i:i + Wl] = other_space.vectors[:, :]
            i += Wl
            VIWI[common_basis.label_indices(diff_other.labels), i:i + Wli] = _np.identity(Wli, dtype)

            ns = _mt.nullspace(VIWI)
            intersection_vecs = _np.dot(VIWI[:, 0:(Vl + Vli)], ns[0:(Vl + Vli), :])  # on common_basis

        else:
            common_basis = self.elemgen_basis.intersection(other_space.elemgen_basis)
            Vl, Wl = (self.vectors.shape[1], other_space.vectors.shape[1])

            #Fill in matrix to take nullspace of: [ V | W ] restricted to rows corresponding to shared elementary
            # error generators (if one space has a elemgen in its basis that the other doesn't, then any intersection
            # vector cannot contain this elemgen).
            VW = _np.zeros((len(common_basis), Vl + Wl), dtype)  # SPARSE in future?
            VW[:, 0:Vl] = self.vectors[self.elemgen_basis.label_indices(common_basis.labels), :]
            VW[:, Vl:] = other_space.vectors[other_space.elemgen_basis.label_indices(common_basis.labels), :]

            ns = _mt.nullspace(VW)
            intersection_vecs = _np.dot(VW[:, 0:Vl], ns[0:Vl, :])  # on common_basis

        return ErrorgenSpace(intersection_vecs, common_basis)

    def union(self, other_space):
        """
        TODO: docstring
        """
        raise NotImplementedError("TODO in FUTURE")


#class LowWeightErrorgenSpace(ErrorgenSpace):
#    """
#    Like a SimpleErrorgenSpace but spanned by only the elementary error generators corresponding to
#    low-weight (up to some maximum weight) basis elements
#    (so far, only Pauli-product bases work for this, since `Basis` objects don't keep track of each
#    element's weight (?)).
#    """
#    pass
