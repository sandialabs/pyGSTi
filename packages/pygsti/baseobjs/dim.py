""" Defines the Dim class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import collections as _collections
import numbers     as _numbers

class Dim(object):
    '''
    Encapsulates the dimOrBlockDims object, as well as block dim processing

    dmDim : int
        The (matrix) dimension of the overall density matrix
        within which the block-diagonal density matrix described by
        dimOrBlockDims is embedded, equal to the sum of the
        individual block dimensions. (The overall density matrix
        is a dmDim x dmDim matrix, and is contained in a space
        of dimension dmDim**2).
    gateDim : int
        The (matrix) dimension of the "gate-space" corresponding
        to the density matrix space, equal to the dimension
        of the density matrix space, sum( ith-block_dimension^2 ).
        Gate matrices are thus gateDim x gateDim dimensions.
    blockDims : list of ints
        Dimensions of the individual matrix-blocks.  The direct sum
        of the matrix spaces (of dim matrix-block-dim^2) forms the
        density matrix space.  Equals:
        [ dimOrBlockDims ] : if dimOrBlockDims is a single int
          dimOrBlockDims   : otherwise
    '''
    def __init__(self, dimOrBlockDims):
        """
        Performs basic processing on the dimensions
          of a direct-sum space.

        Parameters
        ----------
        dimOrBlockDims : int or list of ints
            Structure of the density-matrix space.
            A list of integers designates the space is
              the direct sum of spaces with the square of the given
              matrix-block dimensions.  Matrices in this space are
              represented in the standard basis by a block-diagonal
              matrix with blocks of the given dimensions.
            A single integer is equivalent to a list with a single
              element, and so designates the space of matrices with
              the given dimension, and thus a space of the dimension^2.

        Returns
        -------
        """
        assert dimOrBlockDims is not None, 'Dim object requires non-None dim'
        if isinstance(dimOrBlockDims, Dim):
            self.dmDim     = dimOrBlockDims.dmDim
            self.gateDim   = dimOrBlockDims.gateDim
            self.blockDims = dimOrBlockDims.blockDims
        elif isinstance(dimOrBlockDims, _collections.Container):
            # *full* density matrix is dmDim x dmDim
            self.dmDim = sum([blockDim for blockDim in dimOrBlockDims])

            # gate matrices will be vecDim x vecDim
            self.gateDim = sum([blockDim**2 for blockDim in dimOrBlockDims])

            self.blockDims = dimOrBlockDims
        elif isinstance(dimOrBlockDims, _numbers.Integral):
            self.dmDim = dimOrBlockDims
            self.gateDim = dimOrBlockDims**2
            self.blockDims = [dimOrBlockDims]
        else:
            raise TypeError("Invalid dimOrBlockDims = %s" % str(dimOrBlockDims))
        self.embedDim = self.dmDim ** 2

    def __str__(self):
        return 'Dim: dmDim {} gateDim {} blockDims {} embedDim {}'.format(self.dmDim, self.gateDim, self.blockDims, self.embedDim)

    def __repr__(self):
        return str(self)

    def __getitem__(self, index):
        if index == 0:
            return self.dmDim
        elif index == 1:
            return self.gateDim
        elif index == 2:
            return self.blockDims
        else:
            raise IndexError

    def __hash__(self):
        return hash(tuple(self.blockDims))

    def __eq__(self, other):
        return bool((self.dmDim == other.dmDim) and (self.gateDim == other.gateDim)
                    and (self.blockDims == other.blockDims))

    def __ne__(self, other):
        return not self.__eq__(other)
