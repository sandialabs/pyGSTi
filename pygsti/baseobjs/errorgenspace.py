"""
Defines the ErrorgenSpace class and supporting functionality.
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
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.tools import matrixtools as _mt
from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis

class ErrorgenSpace(_NicelySerializable):


    def __init__(self, vectors, basis):
        """
        A vector space of error generators, spanned by some basis.

        This object collects the information needed to specify a space
        within the space of all error generators.

        Parameters
        ----------
        vectors : numpy array
            List of vectors that span the space
        
            elemgen_basis : ElementaryErrorgenBasis
                The elementary error generator basis that define the entries of self.vectors 
        """
        super().__init__()
        self.vectors = vectors
        self.elemgen_basis = basis
        #Question: have multiple bases or a single one?
        #self._vectors = [] if (items is None) else items  # list of (basis, vectors_mx) pairs
        # map sslbls => (vectors, basis) where basis.sslbls == sslbls
        # or basis => vectors if bases can hash well(?)
    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'vectors' : self._encodemx(self.vectors),
                      'basis': self.elemgen_basis._to_nice_serialization()
        })
        return state
    @classmethod
    def from_nice_serialization(cls, state):
        return cls(cls._decodemx(state['vectors']), ExplicitElementaryErrorgenBasis.from_nice_serialization(state['basis']))
    def intersection(self, other_space, free_on_unspecified_space=False, use_nice_nullspace=False):
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

            ns = _mt.nice_nullspace(VIWI) if use_nice_nullspace else _mt.nullspace(VIWI)
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

    def __eq__(self, other):

        return _np.allclose(self.vectors, other.vectors) and self.elemgen_basis.__eq__(other.elemgen_basis)
    def union(self, other_space):
        """
        TODO: docstring
        """
        raise NotImplementedError("TODO in FUTURE")

    def normalize(self, norm_order=2):
        """
        Normalize the vectors defining this space according to a given norm.

        Parameters
        ----------
        norm_order : int, optional
            The order of the norm to use.

        Returns
        -------
        None
        """
        for j in range(self.vectors.shape[1]):
            sign = +1 if max(self.vectors[:, j]) >= -min(self.vectors[:, j]) else -1
            self.vectors[:, j] /= sign * _np.linalg.norm(self.vectors[:, j], ord=norm_order)