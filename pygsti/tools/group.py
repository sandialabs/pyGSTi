"""
Encapsulates a group in terms of matrices and relations
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from functools import reduce as _reduce
import numbers as _numbers

import numpy as _np

from pygsti.tools.legacytools import deprecate as _deprecated_fn


def _isint(x):
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
    return isinstance(x, _numbers.Integral)


@_deprecated_fn('isinstance(x, numbers.Integral)')
def is_integer(x):
    """
    Check if `x` is an integer type.

    .. deprecated:: 0.10.2
        Use `isinstance(x, numbers.Integral)` instead.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
    """
    return _isint(x)


def construct_1q_clifford_group():
    """
    Returns the 1 qubit Clifford group as a MatrixGroup object

    Returns
    -------
    MatrixGroup
    """
    from pygsti.modelpacks.legacy import std1Q_Cliffords
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
        self.mxs = [m.to_dense() if hasattr(m, 'to_dense') else _np.asarray(m)
                    for m in list_of_matrices]
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
        if N > 0:
            A = _np.asarray(self.mxs)

            def make_key(m):
                # Round to 9 decimals and add 0.0 to normalize -0.0
                return (_np.round(m, 9) + 0.0).tobytes()

            lookup = {make_key(A[k]): k for k in range(N)}

            for i in range(N):
                # Batched product of all matrices in A with matrix i: (N, d, d)
                # Gates applied left-to-right means product sequence g_j * g_i
                row_products = _np.matmul(A, A[i])

                for j in range(N):
                    product_key = make_key(row_products[j])
                    k = lookup.get(product_key, -1)

                    if k < 0:
                        # Fallback to tolerant linear scan on hash boundary miss
                        diffs = _np.abs(A - row_products[j]).reshape(N, -1).max(axis=1)
                        best_k = int(_np.argmin(diffs))
                        if _np.isclose(_np.linalg.norm(row_products[j] - self.mxs[best_k]), 0):
                            k = best_k

                    self.product_table[i, j] = k
        assert (-1 not in self.product_table), "Cannot construct group table"

        #Construct inverse table
        self.inverse_table = -1 * _np.ones(N, dtype=int)
        if N > 0:
            rows, cols = _np.nonzero(self.product_table == 0)
            self.inverse_table[rows] = cols
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
        if not _isint(i): i = self.label_indices[i]
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
        if not _isint(i): i = self.label_indices[i]
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
        if _isint(i):
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
        if _isint(indices[0]):
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
