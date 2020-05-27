"""
Encapsulates a group in terms of matrices and relations
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .. import drivers as _drivers
from ..modelpacks.legacy import std1Q_Cliffords

import numpy as _np
from functools import reduce as _reduce


def is_integer(x):
    """
    Check if `x` is an integer type.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
    """
    #TODO: combine with compattools.isint(x) ??
    return bool(isinstance(x, int) or isinstance(x, _np.integer))


def construct_1q_clifford_group():
    """
    Returns the 1 qubit Clifford group as a MatrixGroup object

    Returns
    -------
    MatrixGroup
    """
    mdl = std1Q_Cliffords.target_model()
    return MatrixGroup(mdl.operations.values(), mdl.operations.keys())


class MatrixGroup(object):
    """
    Encapsulates a group where each element is represented by a matrix.

    Parameters
    ----------
    list_of_matrices : list
        A list of the group elements (should be 2d numpy arrays), and
        can be mdl.gate.values() for some Model `mdl` that forms a group.

    labels : list, optional
        A label corresponding to each group element.
    """

    def __init__(self, list_of_matrices, labels=None):
        """
        Constructs a new MatrixGroup object

        Parameters
        ----------
        list_of_matrices : list
            A list of the group elements (should be 2d numpy arrays), and
            can be mdl.gate.values() for some Model `mdl` that forms a group.

        labels : list, optional
            A label corresponding to each group element.
        """
        self.mxs = list(list_of_matrices)
        self.labels = list(labels) if (labels is not None) else None
        assert(labels is None or len(labels) == len(list_of_matrices))
        if labels is not None:
            self.label_indices = {lbl: indx for indx, lbl in enumerate(labels)}
        else:
            self.label_indices = None

        N = len(self.mxs)
        if N > 0:
            mxDim = self.mxs[0].shape[0]
            assert(_np.isclose(0, _np.linalg.norm(
                self.mxs[0] - _np.identity(mxDim)))), \
                "First element must be the identity matrix!"

        #Construct group table
        self.product_table = -1 * _np.ones([N, N], dtype=int)
        for i in range(N):
            for j in range(N):
                ij_product = _np.dot(self.mxs[j], self.mxs[i])
                #Dot in reverse order here for multiplication here because
                #gates are applied left to right.

                for k in range(N):
                    if _np.isclose(_np.linalg.norm(ij_product - self.mxs[k]), 0):
                        self.product_table[i, j] = k; break
        assert (-1 not in self.product_table), "Cannot construct group table"

        #Construct inverse table
        self.inverse_table = -1 * _np.ones(N, dtype=int)
        for i in range(N):
            for j in range(N):
                if self.product_table[i, j] == 0:  # the identity
                    self.inverse_table[i] = j; break
        assert (-1 not in self.inverse_table), "Cannot construct inv table"

    def matrix(self, i):
        """
        Returns the matrix corresponding to index or label `i`

        Parameters
        ----------
        i : int or other
            If an integer, an element index.  Otherwise, an element label.

        Returns
        -------
        numpy array
        """
        if not is_integer(i): i = self.label_indices[i]
        return self.mxs[i]

    def inverse_matrix(self, i):
        """
        Returns the inverse of the matrix corresponding to index or label `i`

        Parameters
        ----------
        i : int or other
            If an integer, an element index.  Otherwise, an element label.

        Returns
        -------
        numpy array
        """
        if not is_integer(i): i = self.label_indices[i]
        return self.mxs[self.inverse_table[i]]

    def inverse_index(self, i):
        """
        Returns the index/label corresponding to the inverse of index/label `i`

        Parameters
        ----------
        i : int or str
            If an integer, an element index.  Otherwise, an element label.

        Returns
        -------
        int or str
            If `i` is an integer, returns the element's index.  Otherwise
            returns the element's label.
        """
        if is_integer(i):
            return self.inverse_table[i]
        else:
            i = self.label_indices[i]
            return self.labels[self.inverse_table[i]]

    def product(self, indices):
        """
        Returns the index/label of corresponding to the product of a list or tuple of indices/labels.

        Parameters
        ----------
        indices : iterable
            Specifies the sequence of group elements to include in the matrix
            product.  If `indices` contains integers, they an interpreted as
            group element indices, and an integer is returned.  Otherwise,
            `indices` is assumed to contain group element labels, and a label
            is returned.

        Returns
        -------
        int or str
            If `indices` contains integers, returns the resulting element's
            index.  Otherwise returns the resulting element's label.
        """
        if len(indices) == 0: return None
        if is_integer(indices[0]):
            return _reduce(lambda i, j: self.product_table[i, j], indices)
        else:
            indices = [self.label_indices[i] for i in indices]
            fi = _reduce(lambda i, j: self.product_table[i, j], indices)
            return self.labels[fi]

    def __len__(self):
        """
        Returns the order of the group (the number of elements)
        """
        return len(self.mxs)
